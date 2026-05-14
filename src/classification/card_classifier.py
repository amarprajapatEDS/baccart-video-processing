"""High-speed [Rank][Suit] classifier for cropped cards.

MobileNetV3-Small backbone, FP16-optimized, target latency <2ms per card on
RTX 3060 Tensor Cores. Batched across all detected cards in a frame for
optimal throughput.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .card_classes import index_to_card


log = logging.getLogger(__name__)


@dataclass
class CardPrediction:
    label: str
    conf: float
    rank: str
    suit: str


def build_classifier(num_classes: int = 52, backbone: str = "mobilenet_v3_small"):
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    if backbone == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    return model


class CardClassifier:
    def __init__(
        self,
        weights_path: str,
        num_classes: int = 52,
        backbone: str = "mobilenet_v3_small",
        input_size: Tuple[int, int] = (96, 128),
        precision: str = "fp16",
        device: str = "cuda",
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        import torch
        self.torch = torch
        self.input_size = input_size
        self.precision = precision
        self.device_str = self._resolve_device(device)
        self.device = torch.device(self.device_str)
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.pixel_std = np.array(pixel_std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.num_classes = num_classes

        self.model = build_classifier(num_classes=num_classes, backbone=backbone).to(self.device)
        self._mode = "init"
        if Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
            self._mode = "loaded"
            log.info("classifier loaded: %s on %s (%s)", weights_path, self.device_str, self.precision)
        else:
            log.warning("classifier weights not found at %s — using untrained model (stub mode)", weights_path)
            self._mode = "stub"
        self.model.eval()
        if self.precision == "fp16" and self.device.type == "cuda":
            self.model = self.model.half()

    @staticmethod
    def _resolve_device(preferred: str) -> str:
        import torch
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _preprocess(self, crops: List[np.ndarray]) -> "torch.Tensor":
        import cv2
        h, w = self.input_size
        batch = np.empty((len(crops), 3, h, w), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop.size == 0:
                batch[i] = 0.0
                continue
            resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            if resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x = resized.astype(np.float32) / 255.0
            x = x.transpose(2, 0, 1)  # CHW
            batch[i] = x
        batch = (batch - self.pixel_mean) / self.pixel_std
        t = self.torch.from_numpy(batch).to(self.device, non_blocking=True)
        if self.precision == "fp16" and self.device.type == "cuda":
            t = t.half()
        return t

    def predict(self, crops: List[np.ndarray]) -> List[CardPrediction]:
        if not crops:
            return []
        with self.torch.no_grad():
            x = self._preprocess(crops)
            logits = self.model(x).float()
            probs = self.torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        out: List[CardPrediction] = []
        for c, i in zip(conf.cpu().tolist(), idx.cpu().tolist()):
            label = index_to_card(int(i))
            out.append(CardPrediction(label=label, conf=float(c), rank=label[:-1], suit=label[-1]))
        return out

    @property
    def ready(self) -> bool:
        return self._mode == "loaded"
