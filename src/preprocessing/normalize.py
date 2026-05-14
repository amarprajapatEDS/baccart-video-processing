"""Resolution normalization to the reference space (default 1280x720)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_resolution(
    frame: np.ndarray,
    reference: Tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """Resize a frame to the reference (W, H) using bilinear interpolation.

    Returns the same array if already at the reference size — no copy.
    """
    import cv2

    h, w = frame.shape[:2]
    ref_w, ref_h = reference
    if w == ref_w and h == ref_h:
        return frame
    return cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
