"""Load the real S2-Pro checkpoint using official components."""

import json
from collections import OrderedDict
from pathlib import Path

import torch

from ._vendor.fish_speech.models.text2semantic.inference import load_codec_model
from ._vendor.fish_speech.models.text2semantic.llama import (
    DualARTransformer,
    _remap_fish_qwen3_omni_keys,
)
from .hub import MODEL_ID, MODEL_REVISION, resolve_checkpoint


def load_components(
    model_id_or_path: str | Path = MODEL_ID,
    *,
    revision: str = MODEL_REVISION,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    max_seq_len: int = 4096,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
):
    """Download pinned weights if needed; local directories also work offline."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass device='cpu' for slow CPU inference.")
    dtype = dtype or (torch.bfloat16 if target.type == "cuda" else torch.float32)
    checkpoint = resolve_checkpoint(
        model_id_or_path, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only
    )
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "fish_qwen3_omni":
        raise ValueError("This API requires an S2-Pro fish_qwen3_omni checkpoint.")
    if not 16 <= max_seq_len <= config["text_config"]["max_seq_len"]:
        raise ValueError("max_seq_len must be between 16 and the checkpoint context limit.")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())

    # load_weights=True is essential: the official loader defaults to False.
    # It converts fish_qwen3_omni config/keys to DualARTransformer, including QKV.
    model = (
        DualARTransformer.from_pretrained(
            str(checkpoint),
            load_weights=True,
            max_length=max_seq_len,
        )
        .to(device=target, dtype=dtype)
        .eval()
    )
    # Upstream loads with strict=False. Reject missing/unexpected parameters so
    # an incomplete checkpoint cannot silently leave randomly initialized layers.
    supplied = set(_remap_fish_qwen3_omni_keys(OrderedDict.fromkeys(index["weight_map"])))
    required = set(model.state_dict())
    if supplied != required:
        raise ValueError(
            f"Checkpoint key mismatch: missing={required - supplied}, unexpected={supplied - required}"
        )
    # Official loader folds weight normalization in FP32 before casting.
    codec = load_codec_model(checkpoint / "codec.pth", target, dtype)
    return model, codec, checkpoint
