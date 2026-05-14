"""JSON event payload schemas — match the v2.2 spec exactly."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CardPayload:
    val: Optional[str]
    conf: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"val": self.val, "conf": self.conf}


@dataclass
class CardsBlock:
    player: Dict[str, CardPayload]
    banker: Dict[str, CardPayload]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player": {k: v.to_dict() for k, v in self.player.items()},
            "banker": {k: v.to_dict() for k, v in self.banker.items()},
        }


@dataclass
class MetricsBlock:
    phase_conf: float
    min_card_conf: float
    avg_drift_px: float
    stream_health: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StableEvent:
    event: str
    round_sequence: str
    cards: CardsBlock
    metrics: MetricsBlock
    timestamp_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "round_sequence": self.round_sequence,
            "cards": self.cards.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class SimpleEvent:
    event: str
    round_sequence: Optional[str]
    timestamp_ms: int
    metrics: Optional[MetricsBlock] = None
    reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "event": self.event,
            "round_sequence": self.round_sequence,
            "timestamp_ms": self.timestamp_ms,
        }
        if self.metrics is not None:
            out["metrics"] = self.metrics.to_dict()
        if self.reason is not None:
            out["reason"] = self.reason
        if self.extra:
            out["extra"] = self.extra
        return out
