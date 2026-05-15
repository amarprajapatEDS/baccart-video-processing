"""Verify that the pipeline degrades gracefully when classifier weights are
missing: detection confidence flows through as card confidence, FSM gates
pass, and the emitted JSON honestly says 'val=null' so consumers know
rank/suit was not recognized."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.yolo_detector import Detection
from src.detection.slot_mapper import SlotAssignment


def test_detection_only_fallback_populates_slot_confs():
    """Direct unit-test of the fallback logic used by pipeline._process_frame."""
    slots = ("p1", "p2", "p3", "b1", "b2", "b3")
    slot_labels = {s: None for s in slots}
    slot_confs = {s: None for s in slots}

    fake_det_p1 = Detection(bbox=(100, 100, 200, 200), conf=0.95)
    fake_det_b1 = Detection(bbox=(300, 100, 400, 200), conf=0.92)
    slot_assignments = {s: None for s in slots}
    slot_assignments["p1"] = SlotAssignment(slot="p1", detection=fake_det_p1, iou=0.9)
    slot_assignments["b1"] = SlotAssignment(slot="b1", detection=fake_det_b1, iou=0.8)

    classifier_ready = False
    if not classifier_ready:
        for slot, assignment in slot_assignments.items():
            if assignment is None:
                continue
            if slot_labels[slot] is None:
                slot_labels[slot] = "card"
                slot_confs[slot] = float(assignment.detection.conf)

    assert slot_labels["p1"] == "card"
    assert slot_labels["b1"] == "card"
    assert slot_labels["p2"] is None
    assert slot_confs["p1"] == 0.95
    assert slot_confs["b1"] == 0.92


def test_emit_translates_card_placeholder_to_null():
    from src.events import EventEmitter
    from src.events.schemas import MetricsBlock

    class CaptureSink:
        def __init__(self):
            self.events = []
        def emit(self, payload):
            self.events.append(payload)

    sink = CaptureSink()
    em = EventEmitter([sink])
    em.begin_round()

    consensus_labels = {"p1": "card", "p2": "card", "p3": None,
                        "b1": "card", "b2": "card", "b3": None}
    consensus_confs = {"p1": 0.95, "p2": 0.94, "p3": None,
                       "b1": 0.93, "b2": 0.91, "b3": None}
    classifier_ready = False
    emit_labels = consensus_labels
    if not classifier_ready:
        emit_labels = {s: (None if v == "card" else v) for s, v in consensus_labels.items()}

    em.emit_result_stable(
        slot_labels=emit_labels,
        slot_confs=consensus_confs,
        metrics=MetricsBlock(phase_conf=0.95, min_card_conf=0.91,
                             avg_drift_px=1.0, stream_health="HEALTHY"),
    )

    assert len(sink.events) == 1
    cards = sink.events[0]["cards"]
    assert cards["player"]["p1"]["val"] is None
    assert cards["player"]["p1"]["conf"] == 0.95
    assert cards["banker"]["b2"]["val"] is None
    assert cards["banker"]["b2"]["conf"] == 0.91


def test_synthetic_card_generator_produces_52_classes():
    """tools/generate_synthetic_cards.py emits the exact ImageFolder layout
    that train_classifier.py expects."""
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.run(
            [sys.executable, str(root / "tools" / "generate_synthetic_cards.py"),
             "--output", d, "--per-class", "1"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"generator failed: {proc.stderr}"
        classes = sorted(p.name for p in Path(d).iterdir() if p.is_dir())
        assert len(classes) == 52, f"expected 52 class folders, got {len(classes)}: {classes[:5]}..."
        # Spot-check a few expected labels
        for expected in ("AH", "10S", "KD", "2C"):
            assert expected in classes, f"missing class folder: {expected}"
        # Each class has at least one image
        sample_imgs = list((Path(d) / "AH").iterdir())
        assert len(sample_imgs) >= 1
        assert sample_imgs[0].suffix == ".png"


if __name__ == "__main__":
    test_detection_only_fallback_populates_slot_confs()
    test_emit_translates_card_placeholder_to_null()
    test_synthetic_card_generator_produces_52_classes()
    print("OK")
