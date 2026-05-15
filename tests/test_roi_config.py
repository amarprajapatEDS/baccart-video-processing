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
    original_shoe = cfg.rois["shoe"]
    overrides = load_roi_config("configs/pragmatic_speed_baccarat.yaml")
    apply_roi_overrides(cfg, overrides)
    new_shoe = cfg.rois["shoe"]
    assert (new_shoe.x1, new_shoe.y1, new_shoe.x2, new_shoe.y2) == (0.78, 0.45, 0.95, 0.78)
    assert (new_shoe.x1, new_shoe.y1) != (original_shoe.x1, original_shoe.y1) or \
           (new_shoe.x2, new_shoe.y2) == (original_shoe.x2, original_shoe.y2)
    # all required slots present
    for slot in ("shoe", "cleanup", "p1", "p2", "p3", "b1", "b2", "b3"):
        assert slot in cfg.rois


def test_default_rois_are_in_lower_half_of_frame():
    """The dealing-area ROIs should be below center, not over the dealer's torso."""
    cfg = default_config()
    for slot in cfg.player_slots + cfg.banker_slots:
        roi = cfg.rois[slot]
        assert roi.y1 >= 0.55, f"{slot} y1={roi.y1} is too high (should be below dealer)"
        assert roi.y2 <= 0.90


if __name__ == "__main__":
    test_load_yaml_flat_form()
    test_load_yaml_nested_form()
    test_load_json()
    test_invalid_coordinates_rejected()
    test_out_of_range_coordinates_rejected()
    test_missing_file_raises_file_not_found()
    test_apply_overrides_merges_into_config()
    test_default_rois_are_in_lower_half_of_frame()
    print("OK")
