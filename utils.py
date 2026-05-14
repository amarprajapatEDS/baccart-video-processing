"""Generic utilities used across the Baccarat Vision AI pipeline."""
from __future__ import annotations

import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(preferred: str = "cuda") -> str:
    try:
        import torch
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def make_round_sequence(now: Optional[datetime] = None, counter: int = 1) -> str:
    """Returns timestamp-based sequence like 20260508_231500_001."""
    now = now or datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{counter:03d}"


def now_ms() -> int:
    return int(time.time() * 1000)


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


class RingTimer:
    """Tracks rolling timestamps for FPS / latency calculations."""

    def __init__(self, capacity: int = 240):
        self.capacity = capacity
        self._ts: Deque[float] = deque(maxlen=capacity)

    def tick(self, t: Optional[float] = None) -> None:
        self._ts.append(t if t is not None else time.monotonic())

    def fps(self, window_s: float = 2.0) -> float:
        if len(self._ts) < 2:
            return 0.0
        now = self._ts[-1]
        cutoff = now - window_s
        recent = [t for t in self._ts if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        if span <= 0:
            return 0.0
        return (len(recent) - 1) / span

    def __len__(self) -> int:
        return len(self._ts)


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
