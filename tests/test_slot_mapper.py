import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import default_config
from src.detection.slot_mapper import SlotMapper
from src.detection.yolo_detector import Detection


def test_slot_mapper_assigns_to_overlapping_roi():
    cfg = default_config()
    mapper = SlotMapper(cfg.rois, slot_names=list(cfg.player_slots) + list(cfg.banker_slots), min_iou=0.10)

    w, h = 1280, 720
    p1_box = cfg.rois["p1"].to_pixels(w, h)
    det = Detection(bbox=tuple(map(float, p1_box)), conf=0.97)
    out = mapper.assign([det], w, h)
    assert out["p1"] is not None and out["p1"].detection is det


def test_slot_mapper_picks_highest_iou_when_overlap_multiple():
    cfg = default_config()
    mapper = SlotMapper(cfg.rois, slot_names=list(cfg.player_slots) + list(cfg.banker_slots), min_iou=0.05)
    w, h = 1280, 720
    p2 = cfg.rois["p2"].to_pixels(w, h)
    det = Detection(bbox=tuple(map(float, p2)), conf=0.95)
    out = mapper.assign([det], w, h)
    assert out["p2"] is not None
    assert out["p2"].detection is det


def test_no_assignment_below_min_iou():
    cfg = default_config()
    mapper = SlotMapper(cfg.rois, slot_names=list(cfg.player_slots) + list(cfg.banker_slots), min_iou=0.5)
    w, h = 1280, 720
    det = Detection(bbox=(0.0, 0.0, 5.0, 5.0), conf=0.99)
    out = mapper.assign([det], w, h)
    assert all(v is None for v in out.values())


if __name__ == "__main__":
    test_slot_mapper_assigns_to_overlapping_roi()
    test_slot_mapper_picks_highest_iou_when_overlap_multiple()
    test_no_assignment_below_min_iou()
    print("OK")
