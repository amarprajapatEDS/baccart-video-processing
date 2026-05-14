"""Stream health watchdog: monitors fps + socket health and triggers reconnects."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .stream import StreamFrame, StreamHealth, StreamReader


log = logging.getLogger(__name__)


@dataclass
class WatchdogPolicy:
    min_fps: float = 15.0
    fps_window_s: float = 2.0
    max_silence_s: float = 2.0
    reconnect_backoff_s: float = 1.0
    reconnect_max_backoff_s: float = 8.0


class StreamWatchdog:
    """Wraps a frame source, restarts it on degradation.

    For infinite sources (RTSP/HLS) keeps trying to reconnect indefinitely.
    For finite sources (file, animated webp) terminates cleanly when the
    source signals `finished=True` — the pipeline checks `watchdog.finished`
    after a None to decide whether to exit.
    """

    def __init__(
        self,
        factory: Callable[[], object],
        policy: Optional[WatchdogPolicy] = None,
    ):
        self.factory = factory
        self.policy = policy or WatchdogPolicy()
        self.reader: Optional[object] = None
        self._frame_times = []
        self._last_health = StreamHealth.DISCONNECTED
        self._backoff = self.policy.reconnect_backoff_s
        self._finished = False
        self._connect()

    def _connect(self) -> None:
        while True:
            try:
                self.reader = self.factory()
                self._last_health = StreamHealth.HEALTHY
                self._backoff = self.policy.reconnect_backoff_s
                log.info("stream connected")
                return
            except Exception as e:
                log.warning("stream connect failed (%s); backing off %.1fs", e, self._backoff)
                time.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self.policy.reconnect_max_backoff_s)

    def _record_frame_time(self, t: float) -> None:
        cutoff = t - self.policy.fps_window_s
        self._frame_times.append(t)
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.pop(0)

    def _current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def read(self) -> Optional[StreamFrame]:
        if self._finished:
            return None
        if self.reader is None:
            self._connect()

        assert self.reader is not None
        frame = self.reader.read()
        now = time.monotonic()

        source_finished = bool(getattr(self.reader, "finished", False))
        source_is_finite = bool(getattr(self.reader, "is_finite", False))

        if frame is None or not self.reader.is_alive(self.policy.max_silence_s):
            if source_is_finite and source_finished:
                log.info("finite source exhausted — stopping watchdog")
                self._finished = True
                self._last_health = StreamHealth.DISCONNECTED
                self.reader.release()
                self.reader = None
                return None
            log.warning("stream stalled or returned None — reconnecting")
            self._last_health = StreamHealth.UNSTABLE
            try:
                self.reader.release()
            except Exception:
                pass
            self.reader = None
            self._connect()
            return None

        self._record_frame_time(now)
        fps = self._current_fps()
        if fps and fps < self.policy.min_fps:
            self._last_health = StreamHealth.DEGRADED
            frame.health = StreamHealth.DEGRADED
        else:
            self._last_health = StreamHealth.HEALTHY
            frame.health = StreamHealth.HEALTHY
        return frame

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def health(self) -> StreamHealth:
        return self._last_health

    def release(self) -> None:
        if self.reader is not None:
            self.reader.release()
            self.reader = None
