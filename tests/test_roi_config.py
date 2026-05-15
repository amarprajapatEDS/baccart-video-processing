"""Tests for the ROI override loader (config.load_roi_config)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    apply_roi_overrides,
    default_config,
    load_roi_config,
)


def test_load_yaml_flat_form():
    yaml_text = (
        "shoe: [0.78, 0.45, 0.95, 0.78]\n"
        "p1:   [0.36, 0.62, 0.46, 0.82]\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        rois = load_roi_config(path)
        assert set(rois.keys()) == {"shoe", "p1"}
        assert (rois["shoe"].x1, rois["shoe"].y1, rois["shoe"].x2, rois["shoe"].y2) == (0.78, 0.45, 0.95, 0.78)
        assert rois["p1"].name == "p1"
    finally:
        Path(path).unlink()


def test_load_yaml_nested_form():
    yaml_text = (
        "rois:\n"
        "  shoe: [0.78, 0.45, 0.95, 0.78]\n"
        "  p1: { x1: 0.36, y1: 0.62, x2: 0.46, y2: 0.82 }\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        rois = load_roi_config(path)
        assert set(rois.keys()) == {"shoe", "p1"}
        assert rois["p1"].x1 == 0.36
    finally:
        Path(path).unlink()


def test_load_json():
    payload = {"shoe": [0.78, 0.45, 0.95, 0.78], "p1": [0.36, 0.62, 0.46, 0.82]}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        rois = load_roi_config(path)
        assert set(rois.keys()) == {"shoe", "p1"}
    finally:
        Path(path).unlink()


def test_invalid_coordinates_rejected():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("shoe: [0.5, 0.5, 0.3, 0.4]\n")  # x2 < x1
        path = f.name
    try:
        try:
            load_roi_config(path)
            assert False, "should have rejected zero/negative area"
        except ValueError as e:
            assert "area" in str(e)
    finally:
        Path(path).unlink()


def test_out_of_range_coordinates_rejected():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("shoe: [0.5, 0.5, 1.2, 0.9]\n")  # 1.2 > 1.0
        path = f.name
    try:
        try:
            load_roi_config(path)
            assert False, "should have rejected coord > 1.0"
        except ValueError as e:
            assert "0, 1" in str(e) or "[0, 1]" in str(e)
    finally:
        Path(path).unlink()


def test_missing_file_raises_file_not_found():
    try:
        load_roi_config("/tmp/__roi_does_not_exist__.yaml")
        assert False
    except FileNotFoundError:
        pass


def test_apply_overrides_merges_into_config():
    cfg = default_config()
    overrides = load_roi_config("configs/pragmatic_speed_baccarat.yaml")
    apply_roi_overrides(cfg, overrides)
    new_shoe = cfg.rois["shoe"]
    assert "shoe" in overrides
    assert (new_shoe.x1, new_shoe.y1, new_shoe.x2, new_shoe.y2) == (
        overrides["shoe"].x1, overrides["shoe"].y1,
        overrides["shoe"].x2, overrides["shoe"].y2,
    )
    for slot in ("shoe", "cleanup", "p1", "p2", "p3", "b1", "b2", "b3"):
        assert slot in cfg.rois


def test_default_card_slots_align_with_dealing_area():
    """Cards in Pragmatic Play / Evolution layouts are dealt face-up across
    the upper-middle of the frame (y≈0.40-0.58), NOT over the lower betting
    UI. Card-slot ROIs must sit in that band."""
    cfg = default_config()
    for slot in cfg.player_slots + cfg.banker_slots:
        roi = cfg.rois[slot]
        assert 0.30 <= roi.y1 <= 0.45, f"{slot} y1={roi.y1} should sit in the dealing band"
        assert 0.50 <= roi.y2 <= 0.65, f"{slot} y2={roi.y2} should sit in the dealing band"


if __name__ == "__main__":
    test_load_yaml_flat_form()
    test_load_yaml_nested_form()
    test_load_json()
    test_invalid_coordinates_rejected()
    test_out_of_range_coordinates_rejected()
    test_missing_file_raises_file_not_found()
    test_apply_overrides_merges_into_config()
    test_default_card_slots_align_with_dealing_area()
    print("OK")
