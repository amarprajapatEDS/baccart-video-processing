"""Multi-frame label consensus — the rank/suit must be identical for N frames."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


@dataclass
class ConsensusResult:
    label: Optional[str]
    conf: float
    agreement: int   # number of consecutive frames with the same label
    converged: bool  # agreement >= required window


class LabelConsensus:
    """Tracks the last N label predictions per slot, requires unanimous agreement."""

    def __init__(self, window: int, slot_names):
        self.window = int(window)
        self.history: Dict[str, Deque[Tuple[str, float]]] = {
            n: deque(maxlen=self.window) for n in slot_names
        }

    def reset(self) -> None:
        for q in self.history.values():
            q.clear()

    def reset_slot(self, slot: str) -> None:
        if slot in self.history:
            self.history[slot].clear()

    def update(self, slot: str, label: Optional[str], conf: float) -> ConsensusResult:
        if label is None:
            self.history[slot].clear()
            return ConsensusResult(label=None, conf=0.0, agreement=0, converged=False)

        q = self.history[slot]
        if q and q[-1][0] != label:
            q.clear()
        q.append((label, conf))

        agreement = len(q)
        converged = agreement >= self.window
        avg_conf = sum(c for _, c in q) / agreement if agreement else 0.0
        return ConsensusResult(label=label, conf=avg_conf, agreement=agreement, converged=converged)

    def snapshot(self) -> Dict[str, ConsensusResult]:
        out: Dict[str, ConsensusResult] = {}
        for slot, q in self.history.items():
            if not q:
                out[slot] = ConsensusResult(label=None, conf=0.0, agreement=0, converged=False)
                continue
            labels = [l for l, _ in q]
            most_common, count = Counter(labels).most_common(1)[0]
            avg_conf = sum(c for l, c in q if l == most_common) / max(1, count)
            out[slot] = ConsensusResult(
                label=most_common,
                conf=avg_conf,
                agreement=count,
                converged=count >= self.window,
            )
        return out
