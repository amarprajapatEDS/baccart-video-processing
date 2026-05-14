import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.events import EventEmitter, FileSink, SafetyGates
from src.events.schemas import MetricsBlock


class _CaptureSink:
    def __init__(self):
        self.events = []

    def emit(self, payload):
        self.events.append(payload)


def test_stable_event_matches_spec_schema():
    sink = _CaptureSink()
    em = EventEmitter([sink])
    em.begin_round()
    metrics = MetricsBlock(
        phase_conf=0.992, min_card_conf=0.978, avg_drift_px=1.2, stream_health="HEALTHY"
    )
    em.emit_result_stable(
        slot_labels={"p1": "8H", "p2": "3D", "p3": None, "b1": "9S", "b2": "2C", "b3": None},
        slot_confs={"p1": 0.994, "p2": 0.981, "p3": None, "b1": 0.978, "b2": 0.996, "b3": None},
        metrics=metrics,
    )
    assert len(sink.events) == 1
    e = sink.events[0]
    assert e["event"] == "RESULT_STABLE_DETECTED"
    assert "round_sequence" in e and e["round_sequence"]
    assert set(e["cards"]["player"].keys()) == {"p1", "p2", "p3"}
    assert set(e["cards"]["banker"].keys()) == {"b1", "b2", "b3"}
    assert e["cards"]["player"]["p1"] == {"val": "8H", "conf": 0.994}
    assert e["cards"]["player"]["p3"] == {"val": None, "conf": None}
    assert set(e["metrics"].keys()) == {"phase_conf", "min_card_conf", "avg_drift_px", "stream_health"}
    assert "timestamp_ms" in e


def test_safety_gate_blocks_low_confidence():
    g = SafetyGates(min_card_conf_publish=0.95)
    r = g.evaluate_stable({"p1", "p2"}, min_card_conf=0.80, all_stable=True)
    assert r.decision.value == "SUPPRESS_LOW_CONF"
    r = g.evaluate_stable(set(), min_card_conf=0.99, all_stable=True)
    assert r.decision.value == "SUPPRESS_NO_CARDS"
    r = g.evaluate_stable({"p1"}, min_card_conf=0.99, all_stable=False)
    assert r.decision.value == "SUPPRESS_UNSTABLE"
    r = g.evaluate_stable({"p1"}, min_card_conf=0.99, all_stable=True)
    assert r.decision.value == "PUBLISH"


def test_file_sink_writes_one_line_per_event(tmp_path: Path = None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "events.jsonl"
        sink = FileSink(path)
        em = EventEmitter([sink])
        em.begin_round()
        metrics = MetricsBlock(
            phase_conf=0.99, min_card_conf=0.96, avg_drift_px=1.0, stream_health="HEALTHY"
        )
        em.emit_simple("ROUND_START_DETECTED", reason="test")
        em.emit_result_stable(
            slot_labels={"p1": "AH", "p2": None, "p3": None, "b1": None, "b2": None, "b3": None},
            slot_confs={"p1": 0.96, "p2": None, "p3": None, "b1": None, "b2": None, "b3": None},
            metrics=metrics,
        )
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)


def test_round_sequence_format():
    sink = _CaptureSink()
    em = EventEmitter([sink])
    seq = em.begin_round()
    parts = seq.split("_")
    assert len(parts) == 3
    assert len(parts[0]) == 8 and parts[0].isdigit()
    assert len(parts[1]) == 6 and parts[1].isdigit()
    assert len(parts[2]) == 3 and parts[2].isdigit()


if __name__ == "__main__":
    test_stable_event_matches_spec_schema()
    test_safety_gate_blocks_low_confidence()
    test_file_sink_writes_one_line_per_event()
    test_round_sequence_format()
    print("OK")
