"""YOLOv8/v10-Nano card detector with TensorRT FP16 path and PyTorch fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


log = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple              # (x1, y1, x2, y2) in pixels of the input frame
    conf: float
    class_id: int = 0        # card class is resolved by the rank/suit classifier
    class_name: str = "card"

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class CardDetector:
    """Wraps an ultralytics YOLO model. Loads .engine (TensorRT) if present,
    falls back to .pt weights, and to random output for smoke-testing.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        max_detections: int = 12,
        device: str = "cuda",
        precision: str = "fp16",
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.precision = precision
        self._model = None
        self._mode = "stub"
        self._try_load(warmup)

    def _try_load(self, warmup: bool) -> None:
        if not Path(self.model_path).exists():
            log.warning("yolo weights not found at %s — running in stub mode", self.model_path)
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._mode = "ultralytics"
            if warmup:
                self._warmup()
            log.info("yolo loaded: %s (mode=%s, precision=%s, device=%s)",
                     self.model_path, self._mode, self.precision, self.device)
        except Exception as e:
            log.warning("failed to load yolo model %s: %s — falling back to stub", self.model_path, e)
            self._model = None
            self._mode = "stub"

    def _warmup(self) -> None:
        if self._model is None:
            return
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(2):
            try:
                self._model.predict(dummy, conf=self.conf_threshold, iou=self.iou_threshold,
                                    device=self.device, half=(self.precision == "fp16"), verbose=False)
            except Exception:
                return

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if self._mode != "ultralytics" or self._model is None:
            return []
        results = self._model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            half=(self.precision == "fp16"),
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        conf = r.boxes.conf.detach().cpu().numpy()
        cls = r.boxes.cls.detach().cpu().numpy().astype(int) if r.boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        for i in range(len(xyxy)):
            out.append(Detection(
                bbox=tuple(map(float, xyxy[i].tolist())),
                conf=float(conf[i]),
                class_id=int(cls[i]),
                class_name="card",
            ))
        return out

    @property
    def ready(self) -> bool:
        return self._mode == "ultralytics"
