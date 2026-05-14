"""Reads frames from a (possibly animated) WebP file as if it were a live stream.

Useful for offline pipeline validation before a real RTSP/HLS feed is wired up.
Conforms to the same .read() / .is_alive() / .release() protocol as StreamReader
so the pipeline orchestrator is source-agnostic.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .stream import StreamFrame, StreamHealth


log = logging.getLogger(__name__)


class WebPFrameReader:
    """Decodes animated WebP frames via PIL and paces them at a target FPS.

    - Static (1-frame) WebP: emits the same frame repeatedly at `target_fps`.
    - Animated WebP: emits frames in order; if `use_native_durations` is True,
      respects the per-frame durations stored in the file. Loops if `loop=True`,
      otherwise reports `finished=True` after one pass.
    """

    is_finite = True  # a webp file has a definite end (unless looped)

    def __init__(
        self,
        path: str,
        target_fps: int = 30,
        loop: bool = True,
        use_native_durations: bool = True,
    ):
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Pillow is required to read WebP files. `pip install pillow`"
            ) from e

        self.path = str(Path(path).resolve())
        self.target_fps = max(1, int(target_fps))
        self.loop = bool(loop)
        self.use_native_durations = bool(use_native_durations)

        self._img = Image.open(self.path)
        self._img.load()
        self._n_frames = int(getattr(self._img, "n_frames", 1))
        self._frame_idx = 0
        self._exhausted = False
        self._seq = 0
        self._next_due: Optional[float] = None
        self._last_ok_ts: Optional[float] = None
        self._default_frame_ms = 1000.0 / self.target_fps
        self._durations_ms = self._scan_durations()

        log.info(
            "webp reader opened: %s frames=%d size=%dx%d target_fps=%d loop=%s",
            self.path, self._n_frames, self._img.width, self._img.height,
            self.target_fps, self.loop,
        )

    def _scan_durations(self) -> List[float]:
        out: List[float] = []
        if self._n_frames <= 1:
            return [self._default_frame_ms]
        for i in range(self._n_frames):
            self._img.seek(i)
            d = self._img.info.get("duration")
            if not self.use_native_durations or not isinstance(d, (int, float)) or d <= 0:
                d = self._default_frame_ms
            out.append(float(d))
        self._img.seek(0)
        return out

    def _decode_current(self) -> np.ndarray:
        if self._n_frames > 1:
            self._img.seek(self._frame_idx)
        rgb = self._img.convert("RGB")
        arr = np.array(rgb)
        return arr[:, :, ::-1].copy()  # RGB → BGR for cv2 compatibility

    def read(self) -> Optional[StreamFrame]:
        if self._exhausted:
            return None
        now = time.monotonic()
        if self._next_due is None:
            self._next_due = now
        elif now < self._next_due:
            time.sleep(self._next_due - now)
            now = time.monotonic()

        try:
            frame = self._decode_current()
        except Exception as e:
            log.warning("webp decode failed at frame %d: %s", self._frame_idx, e)
            self._exhausted = True
            return None

        dur_ms = self._durations_ms[self._frame_idx] if self._frame_idx < len(self._durations_ms) else self._default_frame_ms
        self._next_due = now + dur_ms / 1000.0
        self._seq += 1
        self._last_ok_ts = now

        if self._n_frames > 1:
            self._frame_idx += 1
            if self._frame_idx >= self._n_frames:
                if self.loop:
                    self._frame_idx = 0
                else:
                    self._exhausted = True

        return StreamFrame(
            frame=frame,
            pts_ms=now * 1000.0,
            seq=self._seq,
            captured_at_monotonic=now,
            health=StreamHealth.HEALTHY,
        )

    def is_alive(self, max_silence_s: float = 2.0) -> bool:
        if self._exhausted:
            return False
        if self._last_ok_ts is None:
            return True
        return (time.monotonic() - self._last_ok_ts) <= max_silence_s

    @property
    def finished(self) -> bool:
        return self._exhausted

    def release(self) -> None:
        try:
            self._img.close()
        except Exception:
            pass
