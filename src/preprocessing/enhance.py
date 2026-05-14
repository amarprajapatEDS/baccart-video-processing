"""Lightweight GPU-friendly enhancement (contrast + sharpening + optional CLAHE)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _unsharp_mask(img: np.ndarray, strength: float) -> np.ndarray:
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


def _apply_clahe(img: np.ndarray, clip: float, grid: Tuple[int, int]) -> np.ndarray:
    import cv2
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance_frame(
    frame: np.ndarray,
    contrast: float = 1.15,
    sharpen: float = 0.35,
    apply_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Boost contrast, then sharpen — keeps additive latency tiny on a GPU.

    All ops are CPU-implemented here for portability; in production these would
    be CUDA kernels or torchvision.transforms.v2 GPU transforms.
    """
    out = frame
    if apply_clahe:
        out = _apply_clahe(out, clahe_clip, clahe_grid)
    if contrast != 1.0:
        out = np.clip(out.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
    if sharpen > 0:
        out = _unsharp_mask(out, sharpen)
    return out
