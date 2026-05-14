"""FPS / latency / stream-health telemetry — the pipeline self-watchdog."""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

from utils import RingTimer


log = logging.getLogger(__name__)


class PipelineHealth(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass
class Telemetry:
    min_fps: float = 15.0
    fps_window_s: float = 2.0
    target_latency_min_ms: float = 200.0
    target_latency_max_ms: float = 500.0
    sample_window: int = 240

    _ring: RingTimer = field(default_factory=lambda: RingTimer(240))
    _latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=240))
    _below_min_since: Optional[float] = None
    _restart_callbacks: list = field(default_factory=list)

    def on_restart(self, fn) -> None:
        self._restart_callbacks.append(fn)

    def tick_frame(self) -> None:
        self._ring.tick()
        fps = self._ring.fps(self.fps_window_s)
        now = time.monotonic()
        if fps and fps < self.min_fps:
            if self._below_min_since is None:
                self._below_min_since = now
            elif (now - self._below_min_since) >= self.fps_window_s:
                self._below_min_since = None
                for cb in self._restart_callbacks:
                    try:
                        cb()
                    except Exception as e:
                        log.warning("restart callback failed: %s", e)
        else:
            self._below_min_since = None

    def record_latency(self, e2e_ms: float) -> None:
        self._latencies_ms.append(float(e2e_ms))

    def current_fps(self) -> float:
        return self._ring.fps(self.fps_window_s)

    def latency_stats(self):
        if not self._latencies_ms:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        s = sorted(self._latencies_ms)
        n = len(s)
        def pct(p):
            k = max(0, min(n - 1, int(round(p * (n - 1)))))
            return s[k]
        return {"avg": sum(s) / n, "p50": pct(0.5), "p95": pct(0.95), "max": s[-1]}

    def health(self) -> PipelineHealth:
        fps = self.current_fps()
        if fps == 0.0:
            return PipelineHealth.HEALTHY
        if fps < self.min_fps:
            return PipelineHealth.UNHEALTHY
        stats = self.latency_stats()
        if stats["p95"] > self.target_latency_max_ms * 1.5:
            return PipelineHealth.DEGRADED
        return PipelineHealth.HEALTHY
