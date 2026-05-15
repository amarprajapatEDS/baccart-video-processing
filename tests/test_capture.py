"""Tests for the card-crop capture pipeline and the headless labeling helper."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.capture import CardCropCapturer
from src.detection.yolo_detector import Detection
from src.detection.slot_mapper import SlotAssignment


def _fake_frame(h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _assignment(slot: str, x1: int, y1: int, x2: int, y2: int, conf: float = 0.95):
    det = Detection(bbox=(float(x1), float(y1), float(x2), float(y2)), conf=conf)
    return SlotAssignment(slot=slot, detection=det, iou=0.9)


def test_capturer_writes_one_crop_per_slot():
    with tempfile.TemporaryDirectory() as d:
        cap = CardCropCapturer(Path(d), every_n_frames=1, min_w=10, min_h=10)
        assignments = {
            "p1": _assignment("p1", 10, 10, 80, 100),
            "p2": _assignment("p2", 90, 10, 160, 100),
            "p3": None,
            "b1": None,
            "b2": None,
            "b3": None,
        }
        written = cap.capture(assignments, _fake_frame(), frame_seq=1)
        assert written == 2
        files = sorted(p.name for p in (Path(d) / "unlabeled").iterdir())
        assert len(files) == 2
        assert any("_p1_" in n for n in files)
        assert any("_p2_" in n for n in files)


def test_capturer_respects_every_n_frames():
    with tempfile.TemporaryDirectory() as d:
        cap = CardCropCapturer(Path(d), every_n_frames=5, min_w=10, min_h=10)
        assignments = {"p1": _assignment("p1", 10, 10, 80, 100),
                       "p2": None, "p3": None, "b1": None, "b2": None, "b3": None}
        total = 0
        for i in range(12):
            total += cap.capture(assignments, _fake_frame(), frame_seq=i)
        # Should write on frames 5 and 10 → 2 crops.
        assert total == 2


def test_capturer_skips_too_small_crops():
    with tempfile.TemporaryDirectory() as d:
        cap = CardCropCapturer(Path(d), every_n_frames=1, min_w=50, min_h=50)
        tiny = {"p1": _assignment("p1", 0, 0, 5, 5),
                "p2": None, "p3": None, "b1": None, "b2": None, "b3": None}
        assert cap.capture(tiny, _fake_frame(), frame_seq=1) == 0


def test_label_card_crops_batch_mode_moves_files_correctly():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        unlabeled = root / "unlabeled"
        unlabeled.mkdir(parents=True)
        # Make 3 fake crops
        for name in ("a.png", "b.png", "c.png"):
            (unlabeled / name).write_bytes(b"\x89PNG\r\n\x1a\n")
        mapping = {"a.png": "AH", "b.png": "10S", "c.png": "ZZ"}  # ZZ invalid
        mapping_file = root / "labels.json"
        mapping_file.write_text(json.dumps(mapping))

        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "label_card_crops.py"),
             "--input", str(root), "--batch", str(mapping_file)],
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, f"label tool failed: {proc.stderr}"
        assert (root / "AH" / "a.png").exists()
        assert (root / "10S" / "b.png").exists()
        # invalid label leaves the file in unlabeled
        assert (unlabeled / "c.png").exists()
        assert not (root / "ZZ").exists()


def test_capture_cli_flag_accepted():
    """run.py --capture-crops should not be rejected by argparse."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
    assert "--capture-crops" in proc.stdout
    assert "--capture-every" in proc.stdout


if __name__ == "__main__":
    test_capturer_writes_one_crop_per_slot()
    test_capturer_respects_every_n_frames()
    test_capturer_skips_too_small_crops()
    test_label_card_crops_batch_mode_moves_files_correctly()
    test_capture_cli_flag_accepted()
    print("OK")
