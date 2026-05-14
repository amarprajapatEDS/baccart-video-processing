"""GPU-accelerated RTSP/HLS stream reader with NVDEC hint and CPU fallback."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import numpy as np


class StreamHealth(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNSTABLE = "STREAM_UNSTABLE"
    DISCONNECTED = "DISCONNECTED"


@dataclass
class StreamFrame:
    frame: np.ndarray
    pts_ms: float
    seq: int
    captured_at_monotonic: float
    health: StreamHealth = StreamHealth.HEALTHY


class StreamReader:
    """Reads frames from an RTSP/HLS/HTTP/file source via OpenCV+FFmpeg.

    Attempts NVDEC via FFmpeg backend hint, falls back to default decoding.
    Designed to be polled in a tight loop; the watchdog wraps reconnect logic.
    """

    is_finite: bool = False  # remote streams are treated as infinite

    def __init__(
        self,
        source: str,
        nvdec: bool = True,
        socket_timeout_s: float = 5.0,
        target_fps: int = 30,
        is_finite: bool = False,
        loop: bool = False,
    ):
        self.source = source
        self.nvdec = nvdec
        self.socket_timeout_s = socket_timeout_s
        self.target_fps = target_fps
        self.is_finite = is_finite
        self.loop = loop
        self._cap = None
        self._seq = 0
        self._last_ok_ts: Optional[float] = None
        self._exhausted = False
        self._open()

    def _open(self) -> None:
        import cv2

        backend = cv2.CAP_FFMPEG
        if self.nvdec:
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;tcp|stimeout;{int(self.socket_timeout_s * 1_000_000)}"
                "|hwaccel;cuda|hwaccel_output_format;cuda",
            )
        cap = cv2.VideoCapture(self.source, backend)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._last_ok_ts = time.monotonic()

    def read(self) -> Optional[StreamFrame]:
        if self._cap is None or self._exhausted:
            return None
        ok, frame = self._cap.read()
        now = time.monotonic()
        if not ok or frame is None:
            if self.is_finite and self.loop:
                import cv2
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._exhausted = True
                    return None
            elif self.is_finite:
                self._exhausted = True
                return None
            else:
                return None
        self._seq += 1
        self._last_ok_ts = now
        return StreamFrame(
            frame=frame,
            pts_ms=now * 1000.0,
            seq=self._seq,
            captured_at_monotonic=now,
            health=StreamHealth.HEALTHY,
        )

    def is_alive(self, max_silence_s: float = 2.0) -> bool:
        if self._cap is None or self._last_ok_ts is None:
            return False
        if self._exhausted:
            return False
        return (time.monotonic() - self._last_ok_ts) <= max_silence_s

    @property
    def finished(self) -> bool:
        return self._exhausted

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[StreamFrame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f
