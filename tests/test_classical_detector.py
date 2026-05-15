"""Tests for the OpenCV-based classical card detector."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROI
from src.detection import ClassicalCardDetector, ClassicalDetectorParams


def _synth_frame_with_card(card_box=(640, 540, 720, 640), frame_size=(720, 1280)) -> np.ndarray:
    """Make a frame with a dark felt and one bright card-shaped rectangle."""
    h, w = frame_size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    x1, y1, x2, y2 = card_box
    frame[y1:y2, x1:x2] = (230, 230, 230)
    return frame


def test_detects_single_bright_card_in_dealing_zone():
    frame = _synth_frame_with_card(card_box=(580, 480, 660, 580))
    det = ClassicalCardDetector(
        dealing_zone=ROI("dz", 0.20, 0.55, 0.85, 0.90),
        params=ClassicalDetectorParams(min_card_w=40, min_card_h=60, brightness_threshold=150),
    )
    dets = det.predict(frame)
    assert len(dets) >= 1, "should detect the synthetic card"
    d = dets[0]
    x1, y1, x2, y2 = d.bbox
    assert abs(x1 - 580) < 8 and abs(y1 - 480) < 8
    assert abs(x2 - 660) < 8 and abs(y2 - 580) < 8
    assert 0.5 <= d.conf <= 0.99


def test_ignores_cards_outside_dealing_zone():
    frame = _synth_frame_with_card(card_box=(50, 50, 130, 150))  # top-left, outside zone
    det = ClassicalCardDetector(
        dealing_zone=ROI("dz", 0.30, 0.60, 0.80, 0.85),
        params=ClassicalDetectorParams(min_card_w=40, min_card_h=60),
    )
    dets = det.predict(frame)
    assert len(dets) == 0


def test_filters_objects_with_wrong_aspect_ratio():
    h, w = 720, 1280
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    frame[500:530, 400:1000] = (240, 240, 240)
    det = ClassicalCardDetector(
        dealing_zone=ROI("dz", 0.15, 0.55, 0.90, 0.90),
        params=ClassicalDetectorParams(
            min_card_w=30, min_card_h=40,
            max_aspect_ratio=2.5, min_aspect_ratio=0.5,
        ),
    )
    dets = det.predict(frame)
    assert len(dets) == 0, "long horizontal bar should not be classified as a card"


def test_detects_multiple_cards():
    h, w = 720, 1280
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    for x in (520, 620, 720):
        frame[480:580, x:x + 80] = (230, 230, 230)
    det = ClassicalCardDetector(
        dealing_zone=ROI("dz", 0.20, 0.55, 0.85, 0.90),
        params=ClassicalDetectorParams(min_card_w=40, min_card_h=60),
    )
    dets = det.predict(frame)
    assert len(dets) >= 3, f"should detect 3 cards, got {len(dets)}"


def test_max_detections_caps_output():
    h, w = 720, 1280
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    for i in range(6):
        x = 300 + i * 100
        frame[500:580, x:x + 70] = (235, 235, 235)
    det = ClassicalCardDetector(
        dealing_zone=ROI("dz", 0.15, 0.55, 0.90, 0.90),
        params=ClassicalDetectorParams(min_card_w=40, min_card_h=60, max_detections=3),
    )
    dets = det.predict(frame)
    assert len(dets) <= 3


def test_detector_marks_ready_unconditionally():
    det = ClassicalCardDetector(dealing_zone=ROI("dz", 0.0, 0.0, 1.0, 1.0))
    assert det.ready is True


if __name__ == "__main__":
    test_detects_single_bright_card_in_dealing_zone()
    test_ignores_cards_outside_dealing_zone()
    test_filters_objects_with_wrong_aspect_ratio()
    test_detects_multiple_cards()
    test_max_detections_caps_output()
    test_detector_marks_ready_unconditionally()
    print("OK")
