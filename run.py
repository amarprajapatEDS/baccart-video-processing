"""Entry point for the Baccarat Vision AI Watcher.

Usage:
    python run.py --source rtsp://host:8554/stream
    python run.py --source path/to/recording.mp4 --max-frames 1000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import apply_roi_overrides, default_config, load_roi_config
from src.pipeline import BaccaratPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baccarat Vision AI Watcher — Casino Live AI v5.8",
    )
    p.add_argument("--source", type=str, default=None,
                   help="RTSP/HLS URL, HTTP(S) video URL, local video file, or animated .webp")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames (for offline testing)")
    p.add_argument("--loop", dest="loop", action="store_true",
                   help="Loop the source when it ends (default: on for finite sources)")
    p.add_argument("--no-loop", dest="loop", action="store_false",
                   help="Exit when a finite source is exhausted")
    p.set_defaults(loop=None)
    p.add_argument("--target-fps", type=int, default=None,
                   help="Target ingestion FPS (controls webp playback rate)")
    p.add_argument("--webp-real-time", dest="webp_native", action="store_true",
                   help="Use the WebP file's per-frame durations (default)")
    p.add_argument("--webp-fixed-fps", dest="webp_native", action="store_false",
                   help="Ignore native durations, pace at --target-fps")
    p.set_defaults(webp_native=None)
    p.add_argument("--yolo-weights", type=str, default=None)
    p.add_argument("--classifier-weights", type=str, default=None)
    p.add_argument("--roi-config", type=str, default=None,
                   help="YAML/JSON file with ROI coordinate overrides "
                        "(see configs/pragmatic_speed_baccarat.yaml)")
    p.add_argument("--use-timer", action="store_true",
                   help="Enable timer-based round-start detection. Watches the "
                        "'timer' ROI for the moment the betting countdown ends "
                        "(motion in that ROI transitions ACTIVE -> IDLE), and "
                        "uses that as the round-start trigger in addition to "
                        "the existing shoe-motion check.")
    p.add_argument("--timer-threshold", type=float, default=None,
                   help="Motion-fraction threshold for the timer ROI "
                        "(default 0.015). Raise if the timer area has "
                        "compression noise; lower if digits are tiny.")
    p.add_argument("--display", type=str, default=None,
                   help="comma-separated: web,window,file,none  (default: web)")
    p.add_argument("--web-host", type=str, default=None,
                   help="MJPEG server bind address (default 0.0.0.0)")
    p.add_argument("--web-port", type=int, default=None,
                   help="MJPEG server port (default 8089) — open http://<host>:<port>/")
    p.add_argument("--record", type=str, default=None,
                   help="Path for annotated MP4 output (implies file display)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable all visualization backends (for benchmarks / batch)")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _validate_explicit_path(label: str, path: Optional[str]) -> None:
    if path is None:
        return
    p = Path(path).expanduser()
    if not p.exists():
        siblings_hint = ""
        parent = p.parent if str(p.parent) else Path(".")
        if parent.exists() and parent.is_dir():
            try:
                names = sorted(c.name for c in parent.iterdir() if c.is_file())[:8]
                if names:
                    siblings_hint = f"\n    available in {parent}/: {', '.join(names)}"
            except OSError:
                pass
        sys.stderr.write(
            f"error: {label} file not found: {p}\n"
            f"    current dir:  {Path.cwd()}\n"
            f"    resolved to:  {p.resolve() if p.parent.exists() else p}"
            f"{siblings_hint}\n"
        )
        sys.exit(2)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _validate_explicit_path("--yolo-weights", args.yolo_weights)
    _validate_explicit_path("--classifier-weights", args.classifier_weights)
    _validate_explicit_path("--roi-config", args.roi_config)

    cfg = default_config()
    if args.source:
        cfg.ingestion.source = args.source
    if args.target_fps is not None:
        cfg.ingestion.target_fps = args.target_fps
    if args.loop is not None:
        cfg.ingestion.loop = args.loop
    if args.webp_native is not None:
        cfg.ingestion.webp_use_native_durations = args.webp_native
    if args.yolo_weights:
        cfg.detection.model_path = args.yolo_weights
    if args.classifier_weights:
        cfg.classification.model_path = args.classifier_weights
    if args.roi_config:
        overrides = load_roi_config(args.roi_config)
        apply_roi_overrides(cfg, overrides)
        logging.getLogger(__name__).info(
            "loaded %d ROI overrides from %s", len(overrides), args.roi_config
        )
    if args.use_timer:
        cfg.fsm.use_timer = True
    if args.timer_threshold is not None:
        cfg.fsm.timer_motion_threshold = args.timer_threshold

    if args.no_display:
        cfg.visualization.enabled = False
    if args.display:
        cfg.visualization.backends = tuple(b.strip() for b in args.display.split(",") if b.strip())
        cfg.visualization.enabled = bool(cfg.visualization.backends) and cfg.visualization.backends != ("none",)
    if args.web_host:
        cfg.visualization.web_host = args.web_host
    if args.web_port is not None:
        cfg.visualization.web_port = args.web_port
    if args.record:
        cfg.visualization.file_path = args.record
        backends = list(cfg.visualization.backends)
        if "file" not in backends:
            backends.append("file")
        cfg.visualization.backends = tuple(backends)
        cfg.visualization.enabled = True

    pipeline = BaccaratPipeline(cfg)
    pipeline.run(max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
