"""Tests for the real-time visualization layer."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import default_config
from src.detection import Detection
from src.detection.slot_mapper import SlotAssignment
from src.fsm import GameState
from src.visualization import (
    MultiDisplay,
    NullDisplay,
    OverlayRenderer,
    RenderContext,
    build_display,
)


def _empty_ctx(state=GameState.IDLE, slots=("p1", "p2", "p3", "b1", "b2", "b3")):
    return RenderContext(
        state=state,
        detections=[],
        slot_assignments={s: None for s in slots},
        consensus_labels={s: None for s in slots},
        consensus_confs={s: None for s in slots},
        stable_slots=set(),
        active_slots=set(),
        converged_slots=set(),
        min_card_conf=0.0,
        avg_drift_px=0.0,
        fps=30.0,
        e2e_latency_ms=120.0,
        stream_health="HEALTHY",
        vision_buffer_progress=0.0,
        vision_buffer_active=False,
    )


def test_overlay_draw_modifies_frame():
    cfg = default_config()
    r = OverlayRenderer(cfg.rois, cfg.player_slots, cfg.banker_slots, banner_duration_s=1.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = r.draw(frame, _empty_ctx())
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert not np.array_equal(out, frame), "draw should annotate the frame"


def test_overlay_paints_timer_roi_when_phase_known():
    """The timer ROI should be drawn in gold when ACTIVE, dim when IDLE."""
    cfg = default_config()
    assert "timer" in cfg.rois, "default config must include a timer ROI"
    r = OverlayRenderer(cfg.rois, cfg.player_slots, cfg.banker_slots)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ctx_active = _empty_ctx()
    ctx_active.timer_phase = "ACTIVE"
    out_active = r.draw(frame, ctx_active)

    ctx_unknown = _empty_ctx()
    ctx_unknown.timer_phase = "UNKNOWN"
    out_unknown = r.draw(frame, ctx_unknown)

    tx1, ty1, tx2, ty2 = cfg.rois["timer"].to_pixels(1280, 720)
    active_pixels = out_active[ty1:ty2, tx1:tx2]
    unknown_pixels = out_unknown[ty1:ty2, tx1:tx2]
    assert not np.array_equal(active_pixels, unknown_pixels), (
        "timer ROI should be drawn differently in ACTIVE vs UNKNOWN phase"
    )


def test_overlay_draws_detection_boxes_and_labels():
    cfg = default_config()
    r = OverlayRenderer(cfg.rois, cfg.player_slots, cfg.banker_slots)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    p1_box = cfg.rois["p1"].to_pixels(1280, 720)
    det = Detection(bbox=tuple(map(float, p1_box)), conf=0.95)
    ctx = _empty_ctx(state=GameState.DEALING)
    ctx.detections = [det]
    ctx.slot_assignments["p1"] = SlotAssignment(slot="p1", detection=det, iou=0.9)
    ctx.consensus_labels["p1"] = "8H"
    ctx.consensus_confs["p1"] = 0.97
    ctx.active_slots = {"p1"}
    ctx.stable_slots = {"p1"}
    out = r.draw(frame, ctx)
    coloured_pixels = np.sum(out.any(axis=-1))
    assert coloured_pixels > 0, "expected non-zero pixels after drawing detection"


def test_banner_lifecycle():
    cfg = default_config()
    r = OverlayRenderer(cfg.rois, cfg.player_slots, cfg.banker_slots, banner_duration_s=0.15)
    assert r.active_banner is None
    r.push_event("MATCH START DETECTED", color=(0, 200, 0), subtitle="round 1")
    assert r.active_banner is not None
    assert r.active_banner.text == "MATCH START DETECTED"
    time.sleep(0.20)
    assert r.active_banner is None, "banner should expire after duration"


def test_banner_overwrites_when_pushed_quickly():
    cfg = default_config()
    r = OverlayRenderer(cfg.rois, cfg.player_slots, cfg.banker_slots, banner_duration_s=1.0)
    r.push_event("MATCH START DETECTED", color=(0, 200, 0))
    r.push_event("RESULT STABLE", color=(0, 215, 255), subtitle="P=8H/3D B=9S/2C")
    r.push_event("MATCH END DETECTED", color=(255, 130, 50))
    assert r.active_banner is not None
    assert r.active_banner.text == "MATCH END DETECTED"


def test_multi_display_fans_out():
    class Probe:
        def __init__(self):
            self.frames = 0
            self.closed = False

        def show(self, f):
            self.frames += 1

        def close(self):
            self.closed = True

    a, b = Probe(), Probe()
    md = MultiDisplay([a, b])
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    md.show(frame)
    md.show(frame)
    assert a.frames == 2 and b.frames == 2
    md.close()
    assert a.closed and b.closed


def test_build_display_disabled():
    cfg = default_config()
    cfg.visualization.enabled = False
    d = build_display(cfg)
    assert isinstance(d, NullDisplay)
    d.show(np.zeros((10, 10, 3), dtype=np.uint8))
    d.close()


def test_build_display_none_backend():
    cfg = default_config()
    cfg.visualization.enabled = True
    cfg.visualization.backends = ("none",)
    d = build_display(cfg)
    assert isinstance(d, NullDisplay)


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_mjpeg_server_serves_index_and_streams_frame():
    import urllib.request

    cfg = default_config()
    cfg.visualization.enabled = True
    cfg.visualization.backends = ("web",)
    cfg.visualization.web_host = "127.0.0.1"
    cfg.visualization.web_port = _free_port()
    d = build_display(cfg)
    try:
        d.show(np.full((120, 160, 3), 200, dtype=np.uint8))
        url = f"http://127.0.0.1:{cfg.visualization.web_port}/"
        with urllib.request.urlopen(url, timeout=2.0) as r:
            assert r.status == 200
            body = r.read()
        assert b"Baccarat Vision AI" in body
        assert b"/stream.mjpg" in body

        with urllib.request.urlopen(f"http://127.0.0.1:{cfg.visualization.web_port}/healthz", timeout=2.0) as r:
            assert r.read() == b"ok"
    finally:
        d.close()


if __name__ == "__main__":
    test_overlay_draw_modifies_frame()
    test_overlay_paints_timer_roi_when_phase_known()
    test_overlay_draws_detection_boxes_and_labels()
    test_banner_lifecycle()
    test_banner_overwrites_when_pushed_quickly()
    test_multi_display_fans_out()
    test_build_display_disabled()
    test_build_display_none_backend()
    test_mjpeg_server_serves_index_and_streams_frame()
    print("OK")
