"""Tests for the animated-WebP reader and source factory routing."""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import (
    UnrecoverableSourceError,
    WebPFrameReader,
    build_source,
    describe_source,
)
from src.ingestion.watchdog import StreamWatchdog, WatchdogPolicy


def _make_animated_webp(path: Path, n_frames: int = 6, size=(160, 120), duration_ms: int = 50) -> None:
    from PIL import Image
    imgs = [Image.new("RGB", size, (i * 30 % 255, 100, 200)) for i in range(n_frames)]
    imgs[0].save(
        str(path),
        format="WEBP",
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )


def _make_static_webp(path: Path, size=(160, 120)) -> None:
    from PIL import Image
    Image.new("RGB", size, (123, 45, 67)).save(str(path), format="WEBP")


def test_webp_reader_decodes_all_frames_in_order():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "clip.webp"
        _make_animated_webp(p, n_frames=5, duration_ms=10)
        reader = WebPFrameReader(str(p), target_fps=120, loop=False, use_native_durations=False)
        frames = []
        for _ in range(5):
            f = reader.read()
            assert f is not None, "expected frame, got None mid-sequence"
            frames.append(f)
        assert reader.read() is None, "non-loop reader must end after n_frames"
        assert reader.finished is True
        assert all(fr.frame.shape == (120, 160, 3) for fr in frames)
        seqs = [fr.seq for fr in frames]
        assert seqs == sorted(seqs)


def test_webp_reader_loops_when_loop_true():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "loop.webp"
        _make_animated_webp(p, n_frames=3, duration_ms=10)
        reader = WebPFrameReader(str(p), target_fps=120, loop=True, use_native_durations=False)
        for _ in range(9):
            f = reader.read()
            assert f is not None
        assert reader.finished is False


def test_webp_reader_handles_static_image():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "static.webp"
        _make_static_webp(p)
        reader = WebPFrameReader(str(p), target_fps=120, loop=True, use_native_durations=False)
        f1 = reader.read()
        f2 = reader.read()
        assert f1 is not None and f2 is not None
        assert f1.frame.shape == f2.frame.shape


def test_webp_reader_paces_at_target_fps():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "paced.webp"
        _make_animated_webp(p, n_frames=4, duration_ms=50)
        reader = WebPFrameReader(str(p), target_fps=20, loop=False, use_native_durations=True)
        t0 = time.monotonic()
        for _ in range(4):
            assert reader.read() is not None
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.10, f"expected ≥0.10s elapsed, got {elapsed:.3f}s"


def test_source_factory_routes_webp_to_webp_reader():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.webp"
        _make_static_webp(p)
        src = build_source(str(p), target_fps=30, loop=False)
        try:
            assert isinstance(src, WebPFrameReader)
            assert src.is_finite is True
            assert src.loop is False
        finally:
            src.release()


def test_describe_source_classifies_inputs():
    assert describe_source("rtsp://h/s") == "rtsp"
    assert describe_source("https://x/y.m3u8") == "hls"
    assert describe_source("https://x/y.mp4") == "http"
    assert describe_source("/tmp/clip.webp") == "webp (local)"
    assert describe_source("https://x/y.webp") == "webp (remote)"
    assert describe_source("/tmp/clip.mp4") == "file"


def test_build_source_missing_local_file_raises_unrecoverable():
    raised = False
    try:
        build_source("/tmp/__definitely_does_not_exist__.webp")
    except UnrecoverableSourceError as e:
        raised = True
        msg = str(e)
        assert "not found" in msg
        assert "hint" in msg
    assert raised, "expected UnrecoverableSourceError for missing local file"


def test_build_source_directory_as_source_raises_unrecoverable():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        raised = False
        try:
            build_source(d)
        except UnrecoverableSourceError as e:
            raised = True
            assert "directory" in str(e)
        assert raised, "expected UnrecoverableSourceError for directory source"


def test_watchdog_stops_immediately_on_unrecoverable_source():
    def factory():
        raise UnrecoverableSourceError("nope")
    wd = StreamWatchdog(factory=factory, policy=WatchdogPolicy())
    assert wd.finished is True
    assert wd.read() is None
    wd.release()


def test_watchdog_stays_healthy_under_slow_pipeline_loop():
    """A slow pipeline iteration must NOT make the watchdog flag the source as
    degraded. Stream health is about the source, not pipeline throughput."""
    import numpy as np
    from src.ingestion.stream import StreamFrame, StreamHealth

    class FastFileLikeReader:
        is_finite = True
        finished = False

        def __init__(self):
            self._seq = 0
            self._last_ok = time.monotonic()

        def read(self):
            self._seq += 1
            self._last_ok = time.monotonic()
            return StreamFrame(
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
                pts_ms=self._last_ok * 1000.0,
                seq=self._seq,
                captured_at_monotonic=self._last_ok,
                health=StreamHealth.HEALTHY,
            )

        def is_alive(self, max_silence_s: float = 2.0) -> bool:
            return (time.monotonic() - self._last_ok) <= max_silence_s

        def release(self) -> None:
            pass

    wd = StreamWatchdog(factory=FastFileLikeReader, policy=WatchdogPolicy(min_fps=15.0))
    try:
        for _ in range(4):
            f = wd.read()
            assert f is not None, "frame should not be None on a healthy mock source"
            assert f.health == StreamHealth.HEALTHY, (
                f"slow pipeline loop incorrectly marked source as {f.health}"
            )
            time.sleep(0.25)  # ~4 FPS — would have tripped the old DEGRADED gate
        assert wd.health == StreamHealth.HEALTHY
    finally:
        wd.release()


if __name__ == "__main__":
    test_webp_reader_decodes_all_frames_in_order()
    test_webp_reader_loops_when_loop_true()
    test_webp_reader_handles_static_image()
    test_webp_reader_paces_at_target_fps()
    test_source_factory_routes_webp_to_webp_reader()
    test_describe_source_classifies_inputs()
    test_build_source_missing_local_file_raises_unrecoverable()
    test_build_source_directory_as_source_raises_unrecoverable()
    test_watchdog_stops_immediately_on_unrecoverable_source()
    test_watchdog_stays_healthy_under_slow_pipeline_loop()
    print("OK")
