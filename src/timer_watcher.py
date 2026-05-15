"""Timer-based round-start detector.

Watches a small ROI where the betting countdown is displayed. A live countdown
renders changing digits, which produces a steady stream of small pixel
differences frame-to-frame. When the countdown ends (digits disappear or
freeze on "0"), the motion in that ROI drops sharply. The transition from
ACTIVE to IDLE is the moment the round starts.

This is a more deterministic round-start signal than dealer-hand motion in
the shoe ROI, which can mis-fire from any nearby movement.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from enum import Enum
from typing import Deque, Optional, Tuple

import numpy as np


log = logging.getLogger(__name__)


class TimerPhase(str, Enum):
    UNKNOWN = "UNKNOWN"
    ACTIVE = "ACTIVE"      # countdown running (regular small motion in ROI)
    IDLE = "IDLE"          # countdown not running (static or absent)


class TimerEvent(str, Enum):
    TIMER_STARTED = "TIMER_STARTED"
    TIMER_ENDED = "TIMER_ENDED"


class TimerWatcher:
    """Motion-on-ROI detector that classifies an area as ACTIVE or IDLE.

    Parameters:
        motion_threshold: fraction of changed pixels above which a frame
            counts as "active". Tune this if your stream has heavy
            compression noise (raise it) or very subtle digit changes
            (lower it).
        active_dwell_s: must observe consistent activity for this long
            before flipping to ACTIVE — debounces single noisy frames.
        idle_dwell_s: must observe consistent inactivity for this long
            before flipping to IDLE.
        history_s: rolling motion-history window length in seconds.
        downsample: factor to shrink the ROI before motion math (speed).
    """

    def __init__(
        self,
        motion_threshold: float = 0.015,
        active_dwell_s: float = 1.5,
        idle_dwell_s: float = 0.5,
        history_s: float = 2.0,
        downsample: int = 2,
        fps_estimate: int = 30,
    ):
        self.motion_threshold = float(motion_threshold)
        self.active_dwell_s = float(active_dwell_s)
        self.idle_dwell_s = float(idle_dwell_s)
        self.downsample = max(1, int(downsample))
        self._prev: Optional[np.ndarray] = None
        history_size = max(4, int(history_s * fps_estimate))
        self._motion: Deque[float] = deque(maxlen=history_size)
        self._phase = TimerPhase.UNKNOWN
        self._candidate_phase: Optional[TimerPhase] = None
        self._candidate_since: float = 0.0

    def reset(self) -> None:
        self._prev = None
        self._motion.clear()
        self._phase = TimerPhase.UNKNOWN
        self._candidate_phase = None
        self._candidate_since = 0.0

    def _classify(self, avg_motion: float) -> TimerPhase:
        if avg_motion >= self.motion_threshold:
            return TimerPhase.ACTIVE
        return TimerPhase.IDLE

    def _required_dwell_s(self, candidate: TimerPhase) -> float:
        if candidate == TimerPhase.ACTIVE:
            return self.active_dwell_s
        return self.idle_dwell_s

    def observe(self, crop: np.ndarray, now: Optional[float] = None) -> Tuple[TimerPhase, Optional[TimerEvent], float]:
        """Update with a new timer-ROI crop.

        Returns: (current_phase, transition_event_or_None, avg_motion_fraction).
        """
        import cv2

        if now is None:
            now = time.monotonic()

        if crop is None or crop.size == 0:
            return self._phase, None, 0.0

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        if self.downsample > 1:
            gray = cv2.resize(
                gray,
                (max(1, gray.shape[1] // self.downsample), max(1, gray.shape[0] // self.downsample)),
                interpolation=cv2.INTER_AREA,
            )

        motion_fraction = 0.0
        if self._prev is not None and self._prev.shape == gray.shape:
            diff = np.abs(gray.astype(np.int16) - self._prev.astype(np.int16))
            motion_fraction = float((diff > 22).mean())
        self._prev = gray
        self._motion.append(motion_fraction)

        avg = float(np.mean(self._motion)) if self._motion else 0.0
        candidate = self._classify(avg)

        if candidate != self._candidate_phase:
            self._candidate_phase = candidate
            self._candidate_since = now

        event: Optional[TimerEvent] = None
        dwell = now - self._candidate_since
        if candidate != self._phase and dwell >= self._required_dwell_s(candidate):
            prev_phase = self._phase
            self._phase = candidate
            if candidate == TimerPhase.ACTIVE and prev_phase in (TimerPhase.IDLE, TimerPhase.UNKNOWN):
                event = TimerEvent.TIMER_STARTED
                log.info("timer started (motion=%.4f)", avg)
            elif candidate == TimerPhase.IDLE and prev_phase == TimerPhase.ACTIVE:
                event = TimerEvent.TIMER_ENDED
                log.info("timer ended → round start signal (motion=%.4f)", avg)

        return self._phase, event, avg

    @property
    def phase(self) -> TimerPhase:
        return self._phase
