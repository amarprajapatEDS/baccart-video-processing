"""Maps raw card detections to fixed slots p1-p3 / b1-b3 based on ROI overlap."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from config import ROI

from .yolo_detector import Detection
from src.preprocessing.roi import iou


@dataclass
class SlotAssignment:
    slot: str
    detection: Detection
    iou: float


class SlotMapper:
    """Pure geometry mapper — assigns each detection to the slot with which it has
    the highest IoU overlap, subject to a minimum IoU threshold.
    """

    def __init__(self, rois: Dict[str, ROI], slot_names: Iterable[str], min_iou: float = 0.15):
        self.rois = rois
        self.slot_names = list(slot_names)
        self.min_iou = min_iou

    def _roi_box_px(self, name: str, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.rois[name].to_pixels(frame_w, frame_h)
        return (float(x1), float(y1), float(x2), float(y2))

    def assign(
        self,
        detections: List[Detection],
        frame_w: int,
        frame_h: int,
    ) -> Dict[str, Optional[SlotAssignment]]:
        slot_boxes = {n: self._roi_box_px(n, frame_w, frame_h) for n in self.slot_names if n in self.rois}
        assignments: Dict[str, Optional[SlotAssignment]] = {n: None for n in slot_boxes}

        scored: List[Tuple[float, str, Detection]] = []
        for det in detections:
            for name, box in slot_boxes.items():
                score = iou(det.bbox, box)
                if score >= self.min_iou:
                    scored.append((score, name, det))

        scored.sort(key=lambda t: t[0], reverse=True)
        used_detections = set()
        for score, name, det in scored:
            if assignments[name] is not None:
                continue
            det_id = id(det)
            if det_id in used_detections:
                continue
            assignments[name] = SlotAssignment(slot=name, detection=det, iou=score)
            used_detections.add(det_id)
        return assignments
