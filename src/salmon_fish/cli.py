"""Console entry point; --help/download do not import Torch or initialize CUDA."""

import argparse
from pathlib import Path

from ._version import __version__
from .hub import MODEL_ID, MODEL_REVISION, download_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="salmon-fish", description="S2-Pro inference without SGLang"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download/cache weights, tokenizer and codec")
    infer = commands.add_parser(
        "infer", help="Generate a WAV, optionally using reference-audio ICL"
    )
    for command in (download, infer):
        command.add_argument(
            "--model", default=MODEL_ID, help="Hugging Face ID or local checkpoint directory"
        )
        command.add_argument("--revision", default=MODEL_REVISION)
        command.add_argument("--cache-dir", type=Path)
        command.add_argument("--offline", action="store_true", help="Use only already cached files")
    download.add_argument(
        "--local-dir", type=Path, help="Optional standalone checkpoint destination"
    )
    infer.add_argument("text", help="New text to synthesize")
    infer.add_argument("--output", default="output.wav")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--reference-audio", type=Path)
    infer.add_argument("--reference-text", help="Exact transcript of the reference audio")
    infer.add_argument("--max-seq-len", type=int, default=4096)
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--top-p", type=float, default=0.7)
    infer.add_argument("--top-k", type=int, default=30)
    infer.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "download":
        print(
            download_model(
                args.model,
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_dir=args.local_dir,
                local_files_only=args.offline,
            )
        )
        return
    if bool(args.reference_audio) != bool(args.reference_text):
        parser.error("--reference-audio and --reference-text must be supplied together")

    import torch

    from .pipeline import S2Pro

    tts = S2Pro.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        max_seq_len=args.max_seq_len,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    reference = (
        tts.encode_reference(args.reference_audio, args.reference_text)
        if args.reference_audio
        else None
    )
    torch.manual_seed(args.seed)
    result = tts.infer(
        args.text,
        args.output,
        reference=reference,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(
        f"{result.path}: {len(result.audio) / result.sample_rate:.2f}s, "
        f"{result.sample_rate} Hz, {result.codes.shape[1]} frames, {result.finish_reason}"
    )
