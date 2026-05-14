"""Fixed-coordinate ROI masking and cropping for the Baccarat table."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from config import ROI


def crop_roi(frame: np.ndarray, roi: ROI) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi.to_pixels(w, h)
    return frame[y1:y2, x1:x2]


def apply_roi(frame: np.ndarray, rois: Dict[str, ROI], names: Iterable[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name in rois:
            out[name] = crop_roi(frame, rois[name])
    return out


def roi_center(roi: ROI, frame_w: int, frame_h: int) -> Tuple[float, float]:
    x1, y1, x2, y2 = roi.to_pixels(frame_w, frame_h)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def mask_outside_rois(frame: np.ndarray, rois: Dict[str, ROI], names: Iterable[str]) -> np.ndarray:
    """Zero out pixels outside the union of given ROIs — useful for noise control."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for name in names:
        if name not in rois:
            continue
        x1, y1, x2, y2 = rois[name].to_pixels(w, h)
        mask[y1:y2, x1:x2] = 255
    if frame.ndim == 3:
        mask3 = np.repeat(mask[:, :, None], frame.shape[2], axis=2)
        return np.where(mask3 > 0, frame, 0).astype(frame.dtype)
    return np.where(mask > 0, frame, 0).astype(frame.dtype)


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
