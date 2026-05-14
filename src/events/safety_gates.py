"""Safety Gates: suppress publication when data integrity is not met."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class GateDecision(str, Enum):
    PUBLISH = "PUBLISH"
    SUPPRESS_LOW_CONF = "SUPPRESS_LOW_CONF"
    SUPPRESS_NO_CARDS = "SUPPRESS_NO_CARDS"
    SUPPRESS_UNSTABLE = "SUPPRESS_UNSTABLE"


@dataclass
class GateResult:
    decision: GateDecision
    reason: str


class SafetyGates:
    def __init__(self, min_card_conf_publish: float):
        self.min_card_conf_publish = float(min_card_conf_publish)

    def evaluate_stable(
        self,
        active_slots: set,
        min_card_conf: float,
        all_stable: bool,
    ) -> GateResult:
        if not active_slots:
            return GateResult(GateDecision.SUPPRESS_NO_CARDS, "no active cards in active slots")
        if not all_stable:
            return GateResult(GateDecision.SUPPRESS_UNSTABLE, "not all active slots are stable")
        if min_card_conf < self.min_card_conf_publish:
            return GateResult(
                GateDecision.SUPPRESS_LOW_CONF,
                f"min_card_conf={min_card_conf:.3f} < {self.min_card_conf_publish:.3f}",
            )
        return GateResult(GateDecision.PUBLISH, "all gates passed")
