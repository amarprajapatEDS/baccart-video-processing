import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROI, all_card_labels, default_config
from src.preprocessing.roi import iou


def test_card_labels_count_is_52():
    labels = all_card_labels()
    assert len(labels) == 52
    assert len(set(labels)) == 52


def test_roi_to_pixels_clamps_to_frame():
    r = ROI("test", -0.1, -0.1, 1.5, 1.5)
    x1, y1, x2, y2 = r.to_pixels(1280, 720)
    assert (x1, y1) == (0, 0)
    assert (x2, y2) == (1280, 720)


def test_iou_known_overlap():
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25.0 / 175.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_default_config_has_all_required_rois():
    cfg = default_config()
    required = {"shoe", "cleanup", "p1", "p2", "p3", "b1", "b2", "b3"}
    assert required.issubset(set(cfg.rois.keys()))


def test_default_config_thresholds_match_spec():
    cfg = default_config()
    assert cfg.stability.drift_threshold_px == 3.0
    assert cfg.stability.stable_frames == 10
    assert cfg.stability.jitter_filter_frames == 3
    assert cfg.fsm.round_start_n == 3
    assert cfg.fsm.result_stable_n == 10
    assert cfg.fsm.round_end_n == 5
    assert cfg.fsm.vision_buffer_s == 1.5
    assert cfg.fsm.dealing_timeout_s == 60.0
    assert cfg.gating.min_card_conf_publish == 0.95
    assert cfg.gating.uncertain_card_conf == 0.85
    assert cfg.gating.harvest_card_conf == 0.90
    assert cfg.telemetry.min_inference_fps == 15.0


def test_drift_threshold_scales_with_resolution():
    cfg = default_config()
    assert cfg.scaled_drift_threshold(720) == 3.0
    assert abs(cfg.scaled_drift_threshold(360) - 1.5) < 1e-9
    assert abs(cfg.scaled_drift_threshold(1080) - 4.5) < 1e-9


if __name__ == "__main__":
    test_card_labels_count_is_52()
    test_roi_to_pixels_clamps_to_frame()
    test_iou_known_overlap()
    test_default_config_has_all_required_rois()
    test_default_config_thresholds_match_spec()
    test_drift_threshold_scales_with_resolution()
    print("OK")
