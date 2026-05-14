import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fsm.buffer import VisionBuffer


def test_buffer_starts_and_completes():
    b = VisionBuffer(duration_s=0.20)
    assert b.active is False
    b.start({"p1", "p2", "b1", "b2"})
    assert b.active is True
    assert b.is_done() is False
    time.sleep(0.25)
    assert b.is_done() is True


def test_new_slot_resets_buffer():
    b = VisionBuffer(duration_s=0.20)
    b.start({"p1", "p2", "b1", "b2"})
    time.sleep(0.10)
    b.observe({"p1", "p2", "b1", "b2", "p3"})
    time.sleep(0.10)
    assert b.is_done() is False, "new slot during buffer should reset timer"
    time.sleep(0.15)
    assert b.is_done() is True


def test_no_new_slot_does_not_reset():
    b = VisionBuffer(duration_s=0.10)
    b.start({"p1", "p2", "b1", "b2"})
    time.sleep(0.08)
    b.observe({"p1", "p2", "b1", "b2"})
    time.sleep(0.06)
    assert b.is_done() is True, "stable slot set should not delay buffer"


def test_cancel_clears_state():
    b = VisionBuffer(duration_s=0.20)
    b.start({"p1"})
    b.cancel()
    assert b.active is False
    assert b.is_done() is False


if __name__ == "__main__":
    test_buffer_starts_and_completes()
    test_new_slot_resets_buffer()
    test_no_new_slot_does_not_reset()
    test_cancel_clears_state()
    print("OK")
