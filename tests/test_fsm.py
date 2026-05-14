import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import default_config
from src.fsm import BaccaratFSM, FrameObservation, GameState, PhaseEvent


def _obs(**kwargs):
    defaults = dict(
        shoe_motion=False,
        cleanup_motion=False,
        active_slots=set(),
        stable_slots=set(),
        converged_slots=set(),
        min_card_conf=0.0,
        avg_drift_px=0.0,
        stream_unstable=False,
        timestamp=time.monotonic(),
    )
    defaults.update(kwargs)
    return FrameObservation(**defaults)


def test_idle_to_dealing_requires_n_frames():
    cfg = default_config()
    fsm = BaccaratFSM(cfg.fsm, cfg.gating)
    for _ in range(cfg.fsm.round_start_n - 1):
        t = fsm.step(_obs(shoe_motion=True, active_slots={"p1"}))
        assert t is None
    t = fsm.step(_obs(shoe_motion=True, active_slots={"p1"}))
    assert t is not None
    assert t.to_state == GameState.DEALING
    assert t.event == PhaseEvent.ROUND_START_DETECTED


def test_dealing_blocks_publish_until_vision_buffer_done():
    cfg = default_config()
    cfg.fsm.vision_buffer_s = 0.05
    fsm = BaccaratFSM(cfg.fsm, cfg.gating)
    for _ in range(cfg.fsm.round_start_n):
        fsm.step(_obs(shoe_motion=True, active_slots={"p1"}))
    assert fsm.state == GameState.DEALING

    full_initial = {"p1", "p2", "b1", "b2"}
    obs = _obs(
        shoe_motion=False,
        active_slots=full_initial,
        stable_slots=full_initial,
        converged_slots=full_initial,
        min_card_conf=0.99,
    )
    t = fsm.step(obs)
    assert t is None, "must wait for vision buffer"

    time.sleep(cfg.fsm.vision_buffer_s + 0.02)
    t = fsm.step(_obs(
        shoe_motion=False,
        active_slots=full_initial,
        stable_slots=full_initial,
        converged_slots=full_initial,
        min_card_conf=0.99,
    ))
    assert t is not None and t.event == PhaseEvent.RESULT_STABLE_DETECTED


def test_low_confidence_suppresses_result_stable():
    cfg = default_config()
    cfg.fsm.vision_buffer_s = 0.05
    fsm = BaccaratFSM(cfg.fsm, cfg.gating)
    for _ in range(cfg.fsm.round_start_n):
        fsm.step(_obs(shoe_motion=True, active_slots={"p1"}))
    assert fsm.state == GameState.DEALING
    time.sleep(0.07)

    full = {"p1", "p2", "b1", "b2"}
    t = fsm.step(_obs(
        active_slots=full,
        stable_slots=full,
        converged_slots=full,
        min_card_conf=0.80,
    ))
    assert t is None or t.event != PhaseEvent.RESULT_STABLE_DETECTED


def test_stream_unstable_takes_precedence():
    cfg = default_config()
    fsm = BaccaratFSM(cfg.fsm, cfg.gating)
    t = fsm.step(_obs(stream_unstable=True))
    assert t is not None
    assert t.event == PhaseEvent.STREAM_UNSTABLE
    assert fsm.state == GameState.STREAM_UNSTABLE


def test_dealing_timeout_triggers_system_error():
    cfg = default_config()
    cfg.fsm.dealing_timeout_s = 0.02
    fsm = BaccaratFSM(cfg.fsm, cfg.gating)
    for _ in range(cfg.fsm.round_start_n):
        fsm.step(_obs(shoe_motion=True, active_slots={"p1"}))
    assert fsm.state == GameState.DEALING
    time.sleep(0.05)
    t = fsm.step(_obs(active_slots={"p1"}))
    assert t is not None
    assert t.event == PhaseEvent.SYSTEM_ERROR


def test_round_end_requires_n_frames_of_cleanup_and_empty():
    cfg = default_config()
    cfg.fsm.vision_buffer_s = 0.02
    fsm = BaccaratFSM(cfg.fsm, cfg.gating)
    for _ in range(cfg.fsm.round_start_n):
        fsm.step(_obs(shoe_motion=True, active_slots={"p1"}))

    full = {"p1", "p2", "b1", "b2"}
    fsm.step(_obs(
        active_slots=full, stable_slots=full, converged_slots=full, min_card_conf=0.99
    ))
    time.sleep(cfg.fsm.vision_buffer_s + 0.02)
    t = fsm.step(_obs(
        active_slots=full, stable_slots=full, converged_slots=full, min_card_conf=0.99
    ))
    assert t is not None and t.event == PhaseEvent.RESULT_STABLE_DETECTED
    assert fsm.state == GameState.RESULT_STABLE

    for _ in range(cfg.fsm.round_end_n - 1):
        t = fsm.step(_obs(cleanup_motion=True, active_slots=set()))
        assert t is None
    t = fsm.step(_obs(cleanup_motion=True, active_slots=set()))
    assert t is not None
    assert t.event == PhaseEvent.ROUND_END_DETECTED


if __name__ == "__main__":
    test_idle_to_dealing_requires_n_frames()
    test_dealing_blocks_publish_until_vision_buffer_done()
    test_low_confidence_suppresses_result_stable()
    test_stream_unstable_takes_precedence()
    test_dealing_timeout_triggers_system_error()
    test_round_end_requires_n_frames_of_cleanup_and_empty()
    print("OK")
