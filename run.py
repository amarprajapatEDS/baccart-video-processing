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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import default_config
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


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
