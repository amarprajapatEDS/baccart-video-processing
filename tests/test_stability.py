import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stability import LabelConsensus, StabilityTracker


def test_stable_after_n_consecutive_frames_within_drift():
    t = StabilityTracker(drift_px=3.0, stable_frames=10, jitter_filter_frames=3, slot_names=["p1"])
    for i in range(9):
        flags = t.update({"p1": (100.0 + 0.5, 200.0)})
        assert flags["p1"] is False, f"frame {i}: prematurely stable"
    flags = t.update({"p1": (100.0, 200.0)})
    assert flags["p1"] is True


def test_drift_above_threshold_resets_streak():
    t = StabilityTracker(drift_px=3.0, stable_frames=5, jitter_filter_frames=3, slot_names=["p1"])
    for _ in range(4):
        t.update({"p1": (100.0, 200.0)})
    flags = t.update({"p1": (200.0, 200.0)})
    assert flags["p1"] is False, "drift>threshold must reset"
    assert t.slots["p1"].stable_count == 1


def test_jitter_filter_tolerates_brief_absence():
    t = StabilityTracker(drift_px=3.0, stable_frames=4, jitter_filter_frames=3, slot_names=["p1"])
    for _ in range(3):
        t.update({"p1": (100.0, 200.0)})
    for _ in range(2):
        flags = t.update({"p1": None})
        assert flags["p1"] is False
        assert t.slots["p1"].last_center is not None, "single-frame absence must not reset"
    flags = t.update({"p1": (100.0, 200.0)})
    assert t.slots["p1"].stable_count >= 4


def test_jitter_filter_resets_after_extended_absence():
    t = StabilityTracker(drift_px=3.0, stable_frames=4, jitter_filter_frames=3, slot_names=["p1"])
    for _ in range(3):
        t.update({"p1": (100.0, 200.0)})
    for _ in range(4):
        t.update({"p1": None})
    assert t.slots["p1"].last_center is None, "extended absence must reset"


def test_label_consensus_requires_unanimous_window():
    c = LabelConsensus(window=5, slot_names=["p1"])
    for _ in range(4):
        r = c.update("p1", "8H", 0.99)
        assert r.converged is False
    r = c.update("p1", "8H", 0.99)
    assert r.converged is True


def test_label_consensus_resets_on_disagreement():
    c = LabelConsensus(window=5, slot_names=["p1"])
    for _ in range(4):
        c.update("p1", "8H", 0.99)
    r = c.update("p1", "9H", 0.99)
    assert r.agreement == 1
    assert r.converged is False


if __name__ == "__main__":
    test_stable_after_n_consecutive_frames_within_drift()
    test_drift_above_threshold_resets_streak()
    test_jitter_filter_tolerates_brief_absence()
    test_jitter_filter_resets_after_extended_absence()
    test_label_consensus_requires_unanimous_window()
    test_label_consensus_resets_on_disagreement()
    print("OK")
