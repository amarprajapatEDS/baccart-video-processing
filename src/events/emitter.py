"""Event emitter — formats JSON payloads and dispatches them to one or more sinks."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

from utils import make_round_sequence, now_ms

from .schemas import CardPayload, CardsBlock, MetricsBlock, SimpleEvent, StableEvent


class EventSink(Protocol):
    def emit(self, payload: Dict) -> None: ...


class StdoutSink:
    def emit(self, payload: Dict) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


class FileSink:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, payload: Dict) -> None:
        line = json.dumps(payload) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)


class EventEmitter:
    def __init__(
        self,
        sinks: Iterable[EventSink],
        player_slots: Iterable[str] = ("p1", "p2", "p3"),
        banker_slots: Iterable[str] = ("b1", "b2", "b3"),
    ):
        self.sinks: List[EventSink] = list(sinks)
        self.player_slots = tuple(player_slots)
        self.banker_slots = tuple(banker_slots)
        self._round_counter = 0
        self._current_round_sequence: Optional[str] = None

    def begin_round(self) -> str:
        self._round_counter += 1
        self._current_round_sequence = make_round_sequence(counter=self._round_counter)
        return self._current_round_sequence

    @property
    def current_round_sequence(self) -> Optional[str]:
        return self._current_round_sequence

    def end_round(self) -> None:
        self._current_round_sequence = None

    def _dispatch(self, payload: Dict) -> None:
        for sink in self.sinks:
            try:
                sink.emit(payload)
            except Exception:
                pass

    def emit_simple(self, event: str, reason: str = "", metrics: Optional[MetricsBlock] = None,
                    extra: Optional[Dict] = None) -> Dict:
        payload = SimpleEvent(
            event=event,
            round_sequence=self._current_round_sequence,
            timestamp_ms=now_ms(),
            metrics=metrics,
            reason=reason or None,
            extra=extra or {},
        ).to_dict()
        self._dispatch(payload)
        return payload

    def emit_round_start(self, metrics: Optional[MetricsBlock] = None) -> Dict:
        seq = self.begin_round()
        return self.emit_simple("ROUND_START_DETECTED", reason="dealing began",
                                metrics=metrics, extra={"round_sequence": seq})

    def emit_round_end(self, metrics: Optional[MetricsBlock] = None) -> Dict:
        payload = self.emit_simple("ROUND_END_DETECTED", reason="cleanup completed",
                                   metrics=metrics)
        self.end_round()
        return payload

    def emit_result_stable(
        self,
        slot_labels: Dict[str, Optional[str]],
        slot_confs: Dict[str, Optional[float]],
        metrics: MetricsBlock,
    ) -> Dict:
        def build_block(slots: Iterable[str]) -> Dict[str, CardPayload]:
            return {
                s: CardPayload(val=slot_labels.get(s), conf=slot_confs.get(s))
                for s in slots
            }
        if self._current_round_sequence is None:
            self.begin_round()
        evt = StableEvent(
            event="RESULT_STABLE_DETECTED",
            round_sequence=self._current_round_sequence,  # type: ignore[arg-type]
            cards=CardsBlock(
                player=build_block(self.player_slots),
                banker=build_block(self.banker_slots),
            ),
            metrics=metrics,
            timestamp_ms=now_ms(),
        ).to_dict()
        self._dispatch(evt)
        return evt
