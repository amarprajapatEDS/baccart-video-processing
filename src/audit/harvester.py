"""Visual failure-case harvester.

Maintains a rolling circular buffer of recent frames in RAM. When a trigger
condition fires (low confidence, safety state, timeout, excessive drift), the
buffer plus the next N seconds of frames are archived for offline review.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)


class HarvestReason(str, Enum):
    LOW_CONF = "LOW_CONF"
    STATE_UNCERTAIN = "STATE_UNCERTAIN"
    STREAM_UNSTABLE = "STREAM_UNSTABLE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    TIMEOUT = "TIMEOUT"
    EXCESSIVE_DRIFT = "EXCESSIVE_DRIFT"


@dataclass
class _ActiveClip:
    reason: HarvestReason
    started_at: float
    target_end_at: float
    writer: object
    path: Path


class FailureHarvester:
    def __init__(
        self,
        archive_dir: Path,
        clip_seconds: int = 10,
        ring_seconds: float = 5.0,
        target_fps: int = 30,
    ):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.clip_seconds = int(clip_seconds)
        self.ring_seconds = float(ring_seconds)
        self.target_fps = int(target_fps)
        self._ring: Deque[Tuple[float, np.ndarray]] = deque(maxlen=int(ring_seconds * target_fps))
        self._active: Optional[_ActiveClip] = None

    def push_frame(self, frame: np.ndarray, t: Optional[float] = None) -> None:
        ts = t if t is not None else time.monotonic()
        self._ring.append((ts, frame.copy()))
        if self._active is not None:
            try:
                self._active.writer.write(frame)
            except Exception as e:
                log.warning("harvester writer failed: %s", e)
                self._close_active()
                return
            if ts >= self._active.target_end_at:
                self._close_active()

    def trigger(self, reason: HarvestReason, frame_size: Tuple[int, int]) -> Optional[Path]:
        if self._active is not None:
            self._active.target_end_at = max(self._active.target_end_at,
                                              time.monotonic() + self.clip_seconds)
            return self._active.path

        try:
            import cv2
        except ImportError:
            log.warning("opencv not available — cannot write harvest clip")
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.archive_dir / f"{ts}_{reason.value}.mp4"
        w, h = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(self.target_fps), (w, h))
        if not writer.isOpened():
            log.warning("harvester could not open writer at %s", path)
            return None
        for _, f in list(self._ring):
            writer.write(f)
        self._active = _ActiveClip(
            reason=reason,
            started_at=time.monotonic(),
            target_end_at=time.monotonic() + self.clip_seconds,
            writer=writer,
            path=path,
        )
        log.info("harvest started: %s reason=%s", path, reason.value)
        return path

    def _close_active(self) -> None:
        if self._active is None:
            return
        try:
            self._active.writer.release()
        except Exception:
            pass
        log.info("harvest closed: %s", self._active.path)
        self._active = None

    def close(self) -> None:
        self._close_active()
