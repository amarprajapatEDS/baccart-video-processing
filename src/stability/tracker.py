"""Per-slot center-point stability tracker — the P0 mathematical guarantee.

A slot is "stable" only when its bbox center (x, y) has drifted < ±drift_px
for `stable_frames` consecutive frames. Any movement above the threshold
resets the timer (per the v2.2 spec).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple


@dataclass
class SlotStability:
    slot: str
    stable_count: int = 0
    last_center: Optional[Tuple[float, float]] = None
    missing_count: int = 0
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=60))

    @property
    def is_stable_required(self) -> bool:
        return self.stable_count >= 0  # placeholder; the tracker computes the real flag

    def reset(self) -> None:
        self.stable_count = 0
        self.last_center = None
        self.missing_count = 0
        self.history.clear()


class StabilityTracker:
    def __init__(
        self,
        drift_px: float,
        stable_frames: int,
        jitter_filter_frames: int,
        slot_names,
    ):
        self.drift_px = float(drift_px)
        self.stable_frames = int(stable_frames)
        self.jitter_filter_frames = int(jitter_filter_frames)
        self.slots: Dict[str, SlotStability] = {n: SlotStability(slot=n) for n in slot_names}
        self._last_drifts: Dict[str, float] = {n: 0.0 for n in slot_names}

    def reset(self) -> None:
        for s in self.slots.values():
            s.reset()
        for k in self._last_drifts:
            self._last_drifts[k] = 0.0

    def update(self, observations: Dict[str, Optional[Tuple[float, float]]]) -> Dict[str, bool]:
        """Update with per-slot center observations and return per-slot stability flag.

        `observations[slot]` is the (x, y) center or None if the slot is empty
        this frame. Single-frame absences are tolerated up to
        `jitter_filter_frames` before the slot is reset.
        """
        out: Dict[str, bool] = {}
        for name, slot in self.slots.items():
            obs = observations.get(name)
            if obs is None:
                slot.missing_count += 1
                if slot.missing_count > self.jitter_filter_frames:
                    slot.reset()
                out[name] = False
                continue
            slot.missing_count = 0
            if slot.last_center is None:
                slot.last_center = obs
                slot.stable_count = 1
                slot.history.append(obs)
                self._last_drifts[name] = 0.0
                out[name] = False
                continue
            dx = obs[0] - slot.last_center[0]
            dy = obs[1] - slot.last_center[1]
            drift = math.hypot(dx, dy)
            self._last_drifts[name] = drift
            if drift <= self.drift_px:
                slot.stable_count += 1
                ax = (slot.last_center[0] * (slot.stable_count - 1) + obs[0]) / slot.stable_count
                ay = (slot.last_center[1] * (slot.stable_count - 1) + obs[1]) / slot.stable_count
                slot.last_center = (ax, ay)
            else:
                slot.stable_count = 1
                slot.last_center = obs
            slot.history.append(obs)
            out[name] = slot.stable_count >= self.stable_frames
        return out

    def avg_drift_px(self) -> float:
        active = [d for s, d in self._last_drifts.items() if self.slots[s].last_center is not None]
        if not active:
            return 0.0
        return sum(active) / len(active)

    def active_slots(self) -> Dict[str, bool]:
        return {n: (s.last_center is not None) for n, s in self.slots.items()}
