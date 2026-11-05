"""Public eager PyTorch S2-Pro inference and reference-audio ICL API."""

from __future__ import annotations

import re
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch

from ._vendor.fish_speech.content_sequence import TextPart, VQPart
from ._vendor.fish_speech.conversation import Conversation, Message
from ._vendor.fish_speech.models.text2semantic.inference import (
    RAS_WIN_SIZE,
    decode_to_audio,
)
from ._vendor.fish_speech.models.text2semantic.llama import DualARTransformer
from ._vendor.fish_speech.tokenizer import IM_END_TOKEN
from .ar import DecodeEvent, decode_one_token_ar
from .hub import MODEL_ID, MODEL_REVISION
from .types import SpeechOutput, SpeechReference


class S2Pro:
    """Batch size 1; requests share the official KV caches and run serially.

    slow_model/fast_model are views of a single DualARTransformer's real modules.
    token_fusion is its actual codebook_embeddings submodule. The complete masked
    sum/scale operation lives in the official BaseTransformer.forward_generate.
    """

    def __init__(self, model: DualARTransformer, codec, checkpoint: Path):
        self.model = model
        self.codec = codec
        self.checkpoint = checkpoint
        self.tokenizer = model.tokenizer  # FishTokenizer wraps Transformers AutoTokenizer
        self.slow_model = SimpleNamespace(
            embeddings=model.embeddings,
            layers=model.layers,
            norm=model.norm,
            output_weight=model.embeddings.weight,  # tied LM head in S2-Pro
            forward_generate=model.forward_generate,
        )
        self.fast_model = SimpleNamespace(
            embeddings=model.fast_embeddings,
            project_in=model.fast_project_in,
            layers=model.fast_layers,
            norm=model.fast_norm,
            output=model.fast_output,
            forward_generate=model.forward_generate_fast,
        )
        self.token_fusion = model.codebook_embeddings
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path = MODEL_ID,
        *,
        revision: str = MODEL_REVISION,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        max_seq_len: int = 4096,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> S2Pro:
        """Automatically download/cache pinned weights, or load a local directory."""
        from ._loading import load_components

        return cls(
            *load_components(
                model_id_or_path,
                revision=revision,
                device=device,
                dtype=dtype,
                max_seq_len=max_seq_len,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )

    @torch.inference_mode()
    def encode_reference(self, audio_path: str | Path, transcript: str) -> SpeechReference:
        """Reference WAV + its transcript -> reusable DAC codes; no training/ASR.

        SoundFile reads the WAV without requiring TorchCodec/FFmpeg. Resampling
        and codec.encode use the same components as official encode_audio().
        Preserve the supplied amplitude: no extra gain or loudness normalization.
        """
        if not transcript.strip():
            raise ValueError("Reference transcript must not be empty.")
        path = Path(audio_path)
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[0] == 0:
            raise ValueError("Reference audio is empty.")
        from torchaudio.functional import resample

        with self._lock:
            parameter = next(self.codec.parameters())
            # [samples, channels] -> mono [1, samples], resample in float32.
            waveform = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0).to(parameter.device)
            waveform = resample(waveform, sample_rate, self.codec.sample_rate)
            lengths = torch.tensor([waveform.shape[-1]], device=parameter.device, dtype=torch.long)
            # Official DAC API: audio [batch=1, channels=1, samples].
            indices, feature_lengths = self.codec.encode(
                waveform.unsqueeze(0).to(parameter.dtype),
                lengths,
            )
            codes = indices[0, :, : int(feature_lengths[0].item())].long().cpu().clone()
        return SpeechReference(transcript.strip(), codes, path, len(audio) / sample_rate)

    def pack_prompt(self, text: str, *, reference: SpeechReference | None = None) -> torch.Tensor:
        """Pack the ICL example and the new text into the official [11, length] prompt.

        With a reference, the sequence is (newlines kept by the official Conversation):

        <|im_start|>system\n
        convert the provided text to speech reference to the following:\n\n
        Text:\n<|speaker:0|>{reference.transcript}\n\nSpeech:\n
        {VQPart: reference.codes, a matrix of audio tokens, not a text placeholder}
        <|im_end|>\n
        <|im_start|>user\n<|speaker:0|>{new text}<|im_end|>\n
        <|im_start|>assistant\n<|voice|>

        The assistant turn has no im_end; generation of new audio starts there.
        The reference stays in the prompt only: it is not prepended to the output
        WAV and does not go through audio_parts/audio_projector.
        """

        if not re.search(r"<\|speaker:\d+\|>", text):
            text = "<|speaker:0|>" + text
        if reference is None:
            system_parts = [TextPart(text="convert the provided text to speech")]
        else:
            codes = reference.codes.detach().cpu()
            if codes.ndim != 2 or codes.shape[0] != self.model.config.num_codebooks or not codes.numel():
                raise ValueError("Reference codes must have shape [10, nonzero_frames].")
            transcript = reference.transcript
            if not re.search(r"<\|speaker:\d+\|>", transcript):
                transcript = "<|speaker:0|>" + transcript
            # Same system_parts as official generate_long() with a single reference.
            # VQPart embeds the audio codes in the system turn; no file path is sent as text.
            system_parts = [
                TextPart(
                    text="convert the provided text to speech reference to the following:\n\nText:\n"
                ),
                TextPart(text=transcript),
                TextPart(text="\n\nSpeech:\n"),
                VQPart(codes=codes),
            ]
        conversation = Conversation(
            [
                Message(role="system", parts=system_parts),
                Message(role="user", parts=[TextPart(text=text)]),
                Message(role="assistant", modality="voice", add_im_end=False),
            ]
        )
        prompt, _, _ = conversation.encode_for_inference(
            self.tokenizer,
            num_codebooks=self.model.config.num_codebooks,
        )
        # The official ContentSequence.encode_for_inference() does the packing:
        #   text positions:  prompt[0] = tokenizer IDs; prompt[1:11] = 0.
        #   audio positions: prompt[0] = semantic_begin_id + codes[0];
        #                    prompt[1:11] = codes[0:10], one column per frame.
        # All 11 rows share one time axis; the 10 codebooks are not flattened.
        return prompt.to(device=next(self.model.parameters()).device)

    def encode_text(self, text: str) -> torch.Tensor:
        """Text-only compatibility helper; use pack_prompt(..., reference=...) for ICL."""
        return self.pack_prompt(text)

    @torch.inference_mode()
    def generate_codes(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 30,
        on_event: Callable[[DecodeEvent], None] | None = None,
    ) -> tuple[torch.Tensor, str]:
        """Time AR outside; the inspectable upstream-equivalent decoder runs depth AR.

        The fast cache restarts at position 0 for every frame; the slow cache
        advances through the text prompt and previously completed audio frames.
        Call via infer() to serialize access to the shared caches.
        on_event receives detached CPU snapshots from the single-frame core;
        event.frame_index identifies the outer time step, including a closing EOS.
        """
        model, cfg = self.model, self.model.config
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype

        prompt_length = prompt.shape[1]
        if prompt_length + max_new_tokens > cfg.max_seq_len:
            raise ValueError(
                "Prompt + max_new_tokens exceeds max_seq_len; increase the context or shorten text."
            )

        with torch.device(device):
            model.setup_caches(1, prompt_length + max_new_tokens, dtype=dtype)

        eos = self.tokenizer.get_token_id(IM_END_TOKEN)
        bias = torch.full((1, 1, cfg.vocab_size), -torch.inf, device=device, dtype=dtype)
        bias[..., cfg.semantic_begin_id : cfg.semantic_end_id + 1] = 0
        bias[..., eos] = 0
        temp = torch.tensor(temperature, device=device, dtype=dtype)
        nucleus = torch.tensor(top_p, device=device, dtype=dtype)
        history = torch.zeros(
            (cfg.num_codebooks + 1, RAS_WIN_SIZE), device=device, dtype=torch.long
        )
        current = prompt.unsqueeze(0)
        positions = torch.arange(prompt_length, device=device)
        frames, reason = [], "length"
        for step in range(max_new_tokens):
            frame = decode_one_token_ar(
                model,
                current,
                positions,
                temp,
                nucleus,
                top_k,
                bias,
                audio_masks=None,
                audio_parts=None,
                previous_tokens=history if step else None,
                kv_len=prompt_length + step,
                on_event=on_event,
                frame_index=step,
            )
            # Stop even if the very first prediction is EOS; never decode an EOS frame.
            if frame[0, 0].item() == eos:
                reason = "eos"
                break
            frames.append(frame[1:].clone())
            history = history.roll(-1, dims=1)
            history[:, -1] = frame[:, 0]
            current = frame.unsqueeze(0)
            positions = torch.tensor([prompt_length + step], device=device)
        codes = torch.cat(frames, dim=1) if frames else prompt.new_empty((cfg.num_codebooks, 0))
        return codes, reason

    @torch.inference_mode()
    def infer(
        self,
        text: str,
        output_path: str | Path | None = "output.wav",
        *,
        reference: SpeechReference | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 30,
        on_event: Callable[[DecodeEvent], None] | None = None,
    ) -> SpeechOutput:
        """Text (+ optional transcript/audio ICL example) -> new mono PCM16 WAV.

        Pass on_event to inspect the same generation without changing its sampler.
        """
        with self._lock:
            prompt = self.pack_prompt(text, reference=reference)

            codes, reason = self.generate_codes(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                on_event=on_event,
            )
            if codes.shape[1] == 0:
                raise RuntimeError(
                    "The model emitted EOS before any audio frame; no WAV was written."
                )
            if reason == "length":
                warnings.warn(
                    "Reached max_new_tokens before EOS; speech may be truncated.", stacklevel=2
                )

            # The upstream RVQ decoder clamps indices in place; retain raw AR codes.
            audio = decode_to_audio(codes.clone(), self.codec).float().cpu().numpy()
            path = Path(output_path) if output_path is not None else None
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, self.codec.sample_rate, format="WAV", subtype="PCM_16")
            return SpeechOutput(audio, self.codec.sample_rate, codes.cpu(), reason, path)
