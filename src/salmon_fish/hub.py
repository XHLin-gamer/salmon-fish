"""Download model assets on demand using Hugging Face's shared snapshot cache."""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "fishaudio/s2-pro"
MODEL_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"
MODEL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "codec.pth",
    "LICENSE.md",
)


def download_model(
    model_id: str = MODEL_ID,
    *,
    revision: str = MODEL_REVISION,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Fetch weights/tokenizer/codec once; return the cached checkpoint directory.

    The default location follows HF_HOME/HF_HUB_CACHE. cache_dir overrides it;
    local_dir optionally materializes a standalone checkpoint directory. Offline
    calls use local_files_only=True. Authentication follows the standard HF_TOKEN
    environment/login configuration; no credentials are stored by this package.
    """
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_files_only=local_files_only,
            library_name="salmon-fish",
        )
    )


def resolve_checkpoint(
    model_id_or_path: str | Path,
    *,
    revision: str = MODEL_REVISION,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    candidate = Path(model_id_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    # An explicitly local path should fail locally rather than become a Hub ID.
    if isinstance(model_id_or_path, Path) or str(model_id_or_path).startswith((".", "/", "~")):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {candidate}")
    return download_model(
        str(model_id_or_path),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
