"""Save per-slot card crops to disk during pipeline runs.

Lets the user collect real labeled training data from their actual casino
stream. Each detected card is cropped and written under
    <output_dir>/unlabeled/<timestamp>_<seq>_<slot>.png

The accompanying tools/label_card_crops.py reorganizes these into the
ImageFolder layout that train_classifier.py expects:
    <output_dir>/<RankSuit>/*.png        (e.g. AH/, 10S/, KD/)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np


log = logging.getLogger(__name__)


class CardCropCapturer:
    """Buffered crop writer. Drops crops on a sampling cadence so the disk
    isn't flooded — one crop per slot every `every_n_frames` is plenty for
    building a training set."""

    def __init__(
        self,
        output_dir: Path,
        every_n_frames: int = 10,
        min_w: int = 24,
        min_h: int = 32,
    ):
        self.output_dir = Path(output_dir).expanduser() / "unlabeled"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_frames = max(1, int(every_n_frames))
        self.min_w = int(min_w)
        self.min_h = int(min_h)
        self._frame_count = 0
        self._capture_count = 0
        log.info(
            "card crop capture ENABLED — writing every %d frames to %s",
            self.every_n_frames, self.output_dir,
        )

    @property
    def captured(self) -> int:
        return self._capture_count

    def capture(
        self,
        slot_assignments: Dict[str, Optional[object]],
        frame: np.ndarray,
        frame_seq: int,
    ) -> int:
        self._frame_count += 1
        if self._frame_count % self.every_n_frames != 0:
            return 0
        try:
            import cv2
        except ImportError:
            return 0

        written = 0
        h, w = frame.shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for slot, assignment in slot_assignments.items():
            if assignment is None:
                continue
            detection = assignment.detection
            x1, y1, x2, y2 = map(int, detection.bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if (x2 - x1) < self.min_w or (y2 - y1) < self.min_h:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = (
                f"{timestamp}_seq{frame_seq:06d}_{slot}_"
                f"conf{int(detection.conf * 100):03d}_{self._capture_count:05d}.png"
            )
            try:
                cv2.imwrite(str(self.output_dir / fname), crop)
                self._capture_count += 1
                written += 1
            except Exception as e:
                log.warning("failed to write crop %s: %s", fname, e)
        return written
