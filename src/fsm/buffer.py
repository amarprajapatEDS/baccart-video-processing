"""1.5s 'Wait-and-See' Vision Buffer.

After the initial 4 cards (p1, p2, b1, b2) are detected the AI MUST wait
`vision_buffer_s` before finalizing extraction, in case 3rd cards (p3 or b3)
appear. If a new card surfaces during the buffer, the timer resets. This is
a vision-only buffer — no rule logic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class VisionBuffer:
    duration_s: float
    started_at: Optional[float] = None
    last_known_active_slots: Set[str] = None

    def __post_init__(self) -> None:
        if self.last_known_active_slots is None:
            self.last_known_active_slots = set()

    @property
    def active(self) -> bool:
        return self.started_at is not None

    def start(self, active_slots: Set[str]) -> None:
        self.started_at = time.monotonic()
        self.last_known_active_slots = set(active_slots)

    def reset_with(self, active_slots: Set[str]) -> None:
        self.started_at = time.monotonic()
        self.last_known_active_slots = set(active_slots)

    def cancel(self) -> None:
        self.started_at = None
        self.last_known_active_slots = set()

    def observe(self, active_slots: Set[str]) -> None:
        """If a NEW slot becomes active during the buffer, restart the timer."""
        if not self.active:
            return
        added = active_slots - self.last_known_active_slots
        if added:
            self.started_at = time.monotonic()
            self.last_known_active_slots = set(active_slots)

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.monotonic() - self.started_at

    def is_done(self) -> bool:
        return self.active and self.elapsed() >= self.duration_s
