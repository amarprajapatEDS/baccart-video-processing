"""Tests for the TimerWatcher (timer-based round-start detector)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.timer_watcher import TimerEvent, TimerPhase, TimerWatcher


H, W = 24, 80


def _changing_digits_frame(seed: int) -> np.ndarray:
    """Simulate a countdown frame — most pixels stable, ~20% change vs prev."""
    rng = np.random.default_rng(seed)
    frame = np.full((H, W, 3), 30, dtype=np.uint8)
    mask = rng.random((H, W)) < 0.20
    frame[mask] = (240, 240, 240)
    return frame


def _static_frame() -> np.ndarray:
    return np.full((H, W, 3), 30, dtype=np.uint8)


def test_transitions_idle_to_active_when_digits_start_changing():
    w = TimerWatcher(motion_threshold=0.05, active_dwell_s=0.05, idle_dwell_s=0.05,
                     history_s=0.25, fps_estimate=30)
    t = 0.0
    for _ in range(5):
        w.observe(_static_frame(), now=t)
        t += 0.05
    assert w.phase == TimerPhase.IDLE

    events = []
    for i in range(20):
        _, ev, _ = w.observe(_changing_digits_frame(i), now=t)
        if ev is not None:
            events.append(ev)
        t += 0.05
    assert TimerEvent.TIMER_STARTED in events
    assert w.phase == TimerPhase.ACTIVE


def test_emits_timer_ended_when_motion_drops():
    w = TimerWatcher(motion_threshold=0.05, active_dwell_s=0.05, idle_dwell_s=0.05,
                     history_s=0.25, fps_estimate=30)
    t = 0.0
    for i in range(15):
        w.observe(_changing_digits_frame(i), now=t)
        t += 0.05
    assert w.phase == TimerPhase.ACTIVE

    events = []
    for _ in range(20):
        _, ev, _ = w.observe(_static_frame(), now=t)
        if ev is not None:
            events.append(ev)
        t += 0.05
    assert TimerEvent.TIMER_ENDED in events
    assert w.phase == TimerPhase.IDLE


def test_brief_noise_spike_does_not_flip_phase():
    """A single noisy frame shouldn't drop ACTIVE -> IDLE before dwell elapses."""
    w = TimerWatcher(motion_threshold=0.05, active_dwell_s=0.05, idle_dwell_s=0.5, fps_estimate=30)
    t = 0.0
    for i in range(15):
        w.observe(_changing_digits_frame(i), now=t)
        t += 0.05
    assert w.phase == TimerPhase.ACTIVE

    w.observe(_static_frame(), now=t)
    t += 0.05
    assert w.phase == TimerPhase.ACTIVE, "single quiet frame must not flip phase before dwell"


def test_reset_clears_history():
    w = TimerWatcher(motion_threshold=0.05, active_dwell_s=0.05, idle_dwell_s=0.05,
                     history_s=0.25, fps_estimate=30)
    for i in range(20):
        w.observe(_changing_digits_frame(i), now=i * 0.05)
    assert w.phase == TimerPhase.ACTIVE
    w.reset()
    assert w.phase == TimerPhase.UNKNOWN


def test_empty_crop_is_safe():
    w = TimerWatcher()
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    phase, ev, motion = w.observe(empty)
    assert phase == TimerPhase.UNKNOWN
    assert ev is None
    assert motion == 0.0


if __name__ == "__main__":
    test_transitions_idle_to_active_when_digits_start_changing()
    test_emits_timer_ended_when_motion_drops()
    test_brief_noise_spike_does_not_flip_phase()
    test_reset_clears_history()
    test_empty_crop_is_safe()
    print("OK")
