"""Lightweight motion detection in the Shoe and Cleanup ROIs.

Uses frame-differencing on a downsampled grayscale crop. Returns a normalized
motion magnitude in [0, 1]; the caller compares against a threshold.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from config import ROI
from src.preprocessing.roi import crop_roi


class MotionDetector:
    def __init__(self, threshold: float = 0.025, downsample: int = 4):
        self.threshold = float(threshold)
        self.downsample = int(downsample)
        self._prev: Dict[str, np.ndarray] = {}

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        import cv2
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.downsample > 1:
            img = cv2.resize(
                img,
                (max(1, img.shape[1] // self.downsample), max(1, img.shape[0] // self.downsample)),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def detect(self, name: str, roi_crop: np.ndarray) -> float:
        gray = self._to_gray(roi_crop)
        prev = self._prev.get(name)
        self._prev[name] = gray
        if prev is None or prev.shape != gray.shape:
            return 0.0
        diff = np.abs(gray.astype(np.int16) - prev.astype(np.int16))
        return float((diff > 25).mean())

    def has_motion(self, name: str, roi_crop: np.ndarray) -> bool:
        return self.detect(name, roi_crop) >= self.threshold

    def reset(self) -> None:
        self._prev.clear()
