"""Salmon Fish: official S2-Pro components through a small Python inference API.

Importing this package never downloads weights or initializes CUDA.
"""

from ._version import __version__
from .hub import MODEL_ID, MODEL_REVISION, download_model

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "S2Pro",
    "DecodeEvent",
    "decode_one_token_ar",
    "SpeechOutput",
    "SpeechReference",
    "__version__",
    "download_model",
]


def __getattr__(name: str):
    if name in {"DecodeEvent", "decode_one_token_ar"}:
        from . import ar

        return getattr(ar, name)
    if name == "S2Pro":
        from .pipeline import S2Pro

        return S2Pro
    if name in {"SpeechOutput", "SpeechReference"}:
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
