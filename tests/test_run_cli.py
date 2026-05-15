"""Tests for run.py CLI argument validation."""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, extra_env=None):
    env = {"PYTHONPATH": str(ROOT)}
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        capture_output=True,
        text=True,
        timeout=20,
        env={**__import__("os").environ, **env},
    )
    return proc


def test_missing_yolo_weights_fails_with_exit_2():
    proc = _run(["--yolo-weights", "/tmp/__nope_yolo__.pt", "--max-frames", "1"])
    assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}: {proc.stderr}"
    assert "--yolo-weights" in proc.stderr
    assert "not found" in proc.stderr


def test_missing_classifier_weights_fails_with_exit_2():
    proc = _run(["--classifier-weights", "/tmp/__nope_clf__.pt"])
    assert proc.returncode == 2
    assert "--classifier-weights" in proc.stderr
    assert "not found" in proc.stderr


def test_missing_roi_config_fails_with_exit_2():
    proc = _run(["--roi-config", "/tmp/__nope_roi__.yaml"])
    assert proc.returncode == 2
    assert "--roi-config" in proc.stderr
    assert "not found" in proc.stderr


def test_existing_paths_pass_validation():
    """When all explicitly-passed paths exist, validation succeeds. The pipeline
    will still fail later (no real model behind the weight file, --source not
    given) — we only care that the validation gate doesn't trip."""
    with tempfile.TemporaryDirectory() as d:
        fake_yolo = Path(d) / "y.pt"
        fake_clf = Path(d) / "c.pt"
        fake_roi = Path(d) / "r.yaml"
        fake_yolo.write_bytes(b"\x00")
        fake_clf.write_bytes(b"\x00")
        fake_roi.write_text("rois:\n  shoe: [0.0, 0.0, 0.1, 0.1]\n")
        proc = _run([
            "--yolo-weights", str(fake_yolo),
            "--classifier-weights", str(fake_clf),
            "--roi-config", str(fake_roi),
            "--source", "/tmp/__missing_source__.webp",
            "--no-display",
            "--max-frames", "1",
        ])
        # exit code != 2 — validation passed (source-not-found is handled later
        # by UnrecoverableSourceError and exits the pipeline cleanly, not 2)
        assert proc.returncode != 2, (
            f"validation incorrectly rejected real paths: {proc.stderr}"
        )


if __name__ == "__main__":
    test_missing_yolo_weights_fails_with_exit_2()
    test_missing_classifier_weights_fails_with_exit_2()
    test_missing_roi_config_fails_with_exit_2()
    test_existing_paths_pass_validation()
    print("OK")
