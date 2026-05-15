"""Classical OpenCV card detector — runs when YOLO weights are missing.

Strategy: bright/light rectangles on a darker felt are extremely likely to be
playing cards. We isolate the dealing zone (a wide horizontal band below the
dealer), threshold for bright pixels, find contours that pass a
size+aspect-ratio gate, and emit them as Detection objects so the rest of
the pipeline (slot mapper, stability tracker, FSM) works unchanged.

This won't recognize rank/suit — that needs the trained classifier — but it
DOES give the user real-time bounding boxes plus enough signal for the FSM
to advance from IDLE → DEALING → RESULT_STABLE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import ROI
from src.preprocessing.roi import crop_roi

from .yolo_detector import Detection


log = logging.getLogger(__name__)


@dataclass
class ClassicalDetectorParams:
    """All tunables in one place — adjust if your stream has unusual lighting."""
    brightness_threshold: int = 175      # pixels above this are "bright"
    min_card_w: int = 18
    min_card_h: int = 24
    max_card_w: int = 180
    max_card_h: int = 180
    min_aspect_ratio: float = 0.45       # h/w lower bound (rotated cards)
    max_aspect_ratio: float = 2.6        # h/w upper bound (portrait cards)
    min_fill_ratio: float = 0.55         # contour area / bbox area
    max_detections: int = 12
    morph_kernel: int = 3
    morph_iterations: int = 2


class ClassicalCardDetector:
    """Bright-rectangle finder restricted to a "dealing zone" ROI."""

    def __init__(
        self,
        dealing_zone: ROI,
        params: Optional[ClassicalDetectorParams] = None,
    ):
        self.dealing_zone = dealing_zone
        self.params = params or ClassicalDetectorParams()
        self._mode = "classical"
        log.info("classical card detector active (dealing zone: %s)", dealing_zone)

    @property
    def ready(self) -> bool:
        return True

    def predict(self, frame: np.ndarray) -> List[Detection]:
        import cv2

        h, w = frame.shape[:2]
        zone_x1, zone_y1, zone_x2, zone_y2 = self.dealing_zone.to_pixels(w, h)
        crop = frame[zone_y1:zone_y2, zone_x1:zone_x2]
        if crop.size == 0:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.params.brightness_threshold, 255, cv2.THRESH_BINARY)
        k = max(1, self.params.morph_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.params.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[float, Detection]] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < self.params.min_card_w or ch < self.params.min_card_h:
                continue
            if cw > self.params.max_card_w or ch > self.params.max_card_h:
                continue
            aspect = ch / max(1, cw)
            if aspect < self.params.min_aspect_ratio or aspect > self.params.max_aspect_ratio:
                continue
            bbox_area = float(cw * ch)
            cnt_area = float(cv2.contourArea(c))
            if bbox_area <= 0:
                continue
            fill = cnt_area / bbox_area
            if fill < self.params.min_fill_ratio:
                continue

            fx1 = zone_x1 + x
            fy1 = zone_y1 + y
            fx2 = fx1 + cw
            fy2 = fy1 + ch
            score = float(min(0.99, 0.5 + 0.5 * fill))
            det = Detection(
                bbox=(float(fx1), float(fy1), float(fx2), float(fy2)),
                conf=score,
                class_id=0,
                class_name="card",
            )
            candidates.append((score, det))

        candidates.sort(key=lambda t: t[0], reverse=True)
        return [det for _, det in candidates[: self.params.max_detections]]
