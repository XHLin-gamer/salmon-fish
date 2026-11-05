"""Public inference result and reusable ICL reference types."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SpeechOutput:
    audio: np.ndarray  # mono float32, on CPU
    sample_rate: int
    codes: torch.Tensor  # [10, frames], on CPU; excludes EOS
    finish_reason: str  # "eos" or "length"
    path: Path | None


@dataclass(frozen=True)
class SpeechReference:
    """Reusable ICL example: an exact transcript paired with codec audio codes."""

    transcript: str
    codes: torch.Tensor  # CPU long [10, reference_frames]; no BOS/EOS
    audio_path: Path
    duration_seconds: float
