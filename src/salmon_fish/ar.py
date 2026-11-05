"""One audio frame, two AR axes: the small, inspectable teaching core.

Adapted from Fish Speech's decode_one_token_ar at commit
0f8a12e8a651062fdee9c33a894bdccd3a8513f1; Fish Audio Community License.
See Notice and third_party/FISH_SPEECH_LICENSE. Models/cache/sampling are upstream.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._vendor.fish_speech.models.text2semantic.inference import (
    RAS_HIGH_TEMP,
    RAS_HIGH_TOP_P,
    sample,
)
from ._vendor.fish_speech.models.text2semantic.llama import DualARTransformer


@dataclass(frozen=True)
class DecodeEvent:
    """One stage's detached CPU tensor snapshots; no live model/cache references.

    frame_index is the outer time step (None for standalone calls).
    codebook_index is the fast/depth step; None except for the codebook stage.
    kv_len mirrors the explicit slow-prefix argument, not cache capacity;
    None leaves the upstream default in effect.
    """

    stage: str
    tensors: dict[str, torch.Tensor]
    frame_index: int | None = None
    codebook_index: int | None = None
    kv_len: int | None = None


@torch.inference_mode()
def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor | None,
    audio_parts: torch.Tensor | None,
    previous_tokens: torch.Tensor | None = None,
    kv_len: int | None = None,
    *,
    on_event: Callable[[DecodeEvent], None] | None = None,
    frame_index: int | None = None,
) -> torch.Tensor:
    """Match the official single-frame decoder, optionally exposing its stages.

    Positional inputs and [1 + num_codebooks, 1] int32 output match upstream.
    Caller sets up the official KV caches first. No cache reset, EOS shortcut,
    sampling rewrite, or extra random draw is introduced here. Fast position 0
    consumes the slow condition; positions 1..9 predict residual codes 1..9.
    on_event=None makes no CPU snapshots. Callbacks run synchronously and should
    not consume randomness or mutate the model when comparing seeded results.
    """

    def emit(stage: str, *, codebook_index: int | None = None, **tensors):
        if on_event is not None:
            snapshots = {
                name: value.detach().to("cpu", copy=True) for name, value in tensors.items()
            }
            on_event(DecodeEvent(stage, snapshots, frame_index, codebook_index, kv_len))

    emit("input", tokens=x, input_pos=input_pos)

    # 1. Observe the first slow layer's input: the actual upstream fusion result.
    # A temporary read-only hook avoids duplicating embedding sum/mask/scale math.
    hook = None
    if on_event is not None:
        hook = model.layers[0].register_forward_pre_hook(
            lambda module, args: emit("fusion", embeddings=args[0])
        )
    try:
        slow = model.forward_generate(
            x,
            input_pos,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            kv_len=kv_len,
        )
    finally:
        if hook is not None:
            hook.remove()
    emit("slow", logits=slow.logits, fast_condition=slow.hidden_states)

    # 2. Semantic token: keep both upstream draws, even when RAS will not be used.
    # Skipping the high-temperature draw would change every later seeded sample.
    biased_logits = slow.logits + semantic_logit_bias
    normal, normal_probs = sample(biased_logits, temperature, top_p, top_k)
    high_temp = torch.tensor(RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype)
    high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
    high, high_probs = sample(biased_logits, high_temp, high_top_p, top_k)
    use_high = torch.zeros((), device=normal.device, dtype=torch.bool)
    main_token = normal
    if previous_tokens is not None:
        in_window = (previous_tokens[0] == normal).any()
        is_semantic = (normal >= model.config.semantic_begin_id) & (
            normal <= model.config.semantic_end_id
        )
        use_high = in_window & is_semantic
        main_token = torch.where(use_high, high, normal)
    emit(
        "semantic",
        logits=biased_logits,
        normal_probs=normal_probs,
        high_probs=high_probs,
        normal_candidate=normal,
        high_candidate=high,
        used_high=use_high,
        token=main_token,
    )

    # 3. Fast position 0 seeds its KV cache with the slow condition. Its output
    # logits are intentionally unused: codebook 0 is already the semantic code.
    position = torch.tensor([0], device=slow.hidden_states.device, dtype=torch.long)
    prefill_logits = model.forward_generate_fast(slow.hidden_states, position)
    emit("fast_prefill", input_pos=position, condition=slow.hidden_states, logits=prefill_logits)
    code = torch.clamp(
        main_token - model.config.semantic_begin_id,
        min=0,
        max=model.config.codebook_size - 1,
    )
    codebooks = [main_token, code]
    hidden = model.fast_embeddings(code)
    emit("codebook", codebook_index=0, token=code, embedding=hidden)

    # 4. Depth AR: the previous code's embedding predicts the next residual code.
    for index in range(1, model.config.num_codebooks):
        position = torch.tensor([index], device=hidden.device, dtype=torch.long)
        logits = model.forward_generate_fast(hidden, position)
        code, probs = sample(logits, temperature, top_p, top_k)
        emit(
            "codebook",
            codebook_index=index,
            input_pos=position,
            input_embedding=hidden,
            logits=logits,
            probs=probs,
            token=code,
        )
        hidden = model.fast_embeddings(code)
        codebooks.append(code)

    # 5. Keep upstream behavior even for EOS: the caller discards its audio row.
    frame = torch.stack(codebooks, dim=1).T
    emit("frame", tokens=frame)
    return frame
