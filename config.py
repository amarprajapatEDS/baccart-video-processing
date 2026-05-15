"""Configuration for Baccarat Vision AI (Watcher).

All thresholds, ROI coordinates, and FSM parameters defined here trace
directly to the Casino Live AI v5.8 spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


CARD_RANKS: Tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
CARD_SUITS: Tuple[str, ...] = ("H", "D", "C", "S")


def all_card_labels() -> List[str]:
    return [f"{r}{s}" for r in CARD_RANKS for s in CARD_SUITS]


@dataclass
class ROI:
    """Normalized ROI rectangle in [0, 1] coords against reference resolution."""
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        return (
            max(0, int(round(self.x1 * width))),
            max(0, int(round(self.y1 * height))),
            min(width, int(round(self.x2 * width))),
            min(height, int(round(self.y2 * height))),
        )


@dataclass
class IngestionConfig:
    source: str = "rtsp://localhost:8554/baccarat"
    protocol: str = "rtsp"
    nvdec: bool = True
    reconnect_timeout_s: float = 3.0
    target_fps: int = 30
    socket_timeout_s: float = 5.0
    read_buffer_frames: int = 4
    loop: bool = True
    webp_use_native_durations: bool = True


@dataclass
class PreprocessingConfig:
    reference_size: Tuple[int, int] = (1280, 720)
    enhance_contrast: float = 1.15
    sharpen_strength: float = 0.35
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)


@dataclass
class DetectionConfig:
    model_path: str = "weights/yolov8n_cards.engine"
    framework: str = "ultralytics"
    conf_threshold: float = 0.45
    iou_threshold: float = 0.50
    max_detections: int = 12
    use_tensorrt: bool = True
    precision: str = "fp16"
    device: str = "cuda"


@dataclass
class ClassificationConfig:
    model_path: str = "weights/mobilenetv3_cards_fp16.pt"
    backbone: str = "mobilenet_v3_small"
    num_classes: int = 52
    input_size: Tuple[int, int] = (96, 128)
    precision: str = "fp16"
    target_latency_ms_per_card: float = 2.0
    device: str = "cuda"
    pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class StabilityConfig:
    drift_threshold_px: float = 3.0
    stable_frames: int = 10
    jitter_filter_frames: int = 3
    reference_height: int = 720


@dataclass
class FSMConfig:
    round_start_n: int = 3
    result_stable_n: int = 10
    round_end_n: int = 5
    vision_buffer_s: float = 1.5
    dealing_timeout_s: float = 60.0
    uncertainty_dwell_s: float = 3.0
    # Timer-based round detection (optional). When True, the pipeline runs
    # a TimerWatcher over the `timer` ROI and the FSM accepts EITHER motion
    # detection OR a timer-ended event as the round-start trigger.
    use_timer: bool = False
    timer_motion_threshold: float = 0.015
    timer_active_dwell_s: float = 1.5
    timer_idle_dwell_s: float = 0.5


@dataclass
class GatingConfig:
    min_card_conf_publish: float = 0.95
    uncertain_card_conf: float = 0.85
    harvest_card_conf: float = 0.90
    harvest_drift_px: float = 4.0
    harvest_clip_seconds: int = 10


@dataclass
class TelemetryConfig:
    min_inference_fps: float = 15.0
    fps_window_s: float = 2.0
    target_e2e_latency_ms_min: float = 200.0
    target_e2e_latency_ms_max: float = 500.0
    sample_window: int = 240


@dataclass
class VisualizationConfig:
    enabled: bool = True
    backends: Tuple[str, ...] = ("web",)
    web_host: str = "0.0.0.0"
    web_port: int = 8089
    window_title: str = "Baccarat Vision AI"
    file_path: str = "annotated_output.mp4"
    file_fps: int = 30
    banner_duration_s: float = 1.8
    show_rois: bool = True
    show_metrics: bool = True
    jpeg_quality: int = 80


@dataclass
class StorageConfig:
    log_dir: Path = field(default=Path("logs"))
    archive_dir: Path = field(default=Path("archive"))
    snapshot_dir: Path = field(default=Path("snapshots"))
    weights_dir: Path = field(default=Path("weights"))
    rolling_retention_minutes: int = 60
    async_queue_size: int = 1024


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.resolve())
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    rois: Dict[str, ROI] = field(default_factory=lambda: {
        # Tuned for typical Pragmatic Play / Evolution Live baccarat layouts:
        # cards are dealt FACE-UP across the upper-middle of the frame
        # (y=0.40-0.58), NOT on the betting UI strip near the bottom.
        # The shoe is on the dealer's right at the table level.
        "shoe":     ROI("shoe",     0.85, 0.40, 0.99, 0.65),
        "cleanup":  ROI("cleanup",  0.25, 0.40, 0.75, 0.60),
        "p1":       ROI("p1",       0.43, 0.40, 0.51, 0.58),
        "p2":       ROI("p2",       0.36, 0.40, 0.44, 0.58),
        "p3":       ROI("p3",       0.29, 0.40, 0.37, 0.58),
        "b1":       ROI("b1",       0.51, 0.40, 0.59, 0.58),
        "b2":       ROI("b2",       0.59, 0.40, 0.67, 0.58),
        "b3":       ROI("b3",       0.66, 0.40, 0.74, 0.58),
        # Optional countdown-timer ROI for TimerWatcher (--use-timer).
        "timer":    ROI("timer",    0.42, 0.02, 0.58, 0.10),
    })
    card_labels: List[str] = field(default_factory=all_card_labels)
    player_slots: Tuple[str, ...] = ("p1", "p2", "p3")
    banker_slots: Tuple[str, ...] = ("b1", "b2", "b3")
    seed: int = 42

    def __post_init__(self) -> None:
        self.storage.log_dir = self._resolve(self.storage.log_dir)
        self.storage.archive_dir = self._resolve(self.storage.archive_dir)
        self.storage.snapshot_dir = self._resolve(self.storage.snapshot_dir)
        self.storage.weights_dir = self._resolve(self.storage.weights_dir)
        for d in (
            self.storage.log_dir,
            self.storage.archive_dir,
            self.storage.snapshot_dir,
            self.storage.weights_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        if len(self.card_labels) != self.classification.num_classes:
            raise ValueError(
                f"num_classes={self.classification.num_classes} != card_labels={len(self.card_labels)}"
            )

    def _resolve(self, p: Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.project_root / p)

    def scaled_drift_threshold(self, frame_height: int) -> float:
        return self.stability.drift_threshold_px * (frame_height / self.stability.reference_height)


def default_config() -> Config:
    return Config()


def load_roi_config(path: str) -> Dict[str, ROI]:
    """Load ROI overrides from a YAML or JSON file.

    Accepted shapes:

        # flat dict form
        shoe: [0.78, 0.45, 0.95, 0.78]
        p1:   [0.36, 0.62, 0.46, 0.82]

        # nested form (also accepted)
        rois:
          shoe: { x1: 0.78, y1: 0.45, x2: 0.95, y2: 0.78 }
          p1:   { x1: 0.36, y1: 0.62, x2: 0.46, y2: 0.82 }
    """
    import json
    from pathlib import Path as _P

    p = _P(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"ROI config not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to load .yaml/.yml ROI configs") from e
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        raise ValueError(f"unsupported ROI config extension: {suffix}")

    if isinstance(data, dict) and "rois" in data and isinstance(data["rois"], dict):
        data = data["rois"]
    if not isinstance(data, dict):
        raise ValueError("ROI config must be a mapping of name → coords")

    out: Dict[str, ROI] = {}
    for name, value in data.items():
        if isinstance(value, (list, tuple)) and len(value) == 4:
            x1, y1, x2, y2 = value
        elif isinstance(value, dict) and {"x1", "y1", "x2", "y2"} <= set(value):
            x1, y1, x2, y2 = value["x1"], value["y1"], value["x2"], value["y2"]
        else:
            raise ValueError(
                f"ROI '{name}' must be [x1,y1,x2,y2] or {{x1,y1,x2,y2}}, got: {value!r}"
            )
        for coord_name, c in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
                raise ValueError(f"ROI '{name}'.{coord_name}={c!r} must be a float in [0, 1]")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI '{name}' has zero/negative area: x2-x1={x2-x1}, y2-y1={y2-y1}")
        out[name] = ROI(name=name, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

    return out


def apply_roi_overrides(cfg: Config, overrides: Dict[str, ROI]) -> Config:
    """Merge ROI overrides into a Config in-place and return it."""
    cfg.rois.update(overrides)
    return cfg
