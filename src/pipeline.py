"""Main orchestration loop for the Baccarat Vision AI Watcher.

Wires together: stream → preprocess → detect → classify → stabilize →
state-machine → safety-gate → event-emit → audit.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from config import Config

from src.audit import AsyncLogger, FailureHarvester, HarvestReason, Telemetry
from src.classification import CardClassifier
from src.detection import CardDetector, ClassicalCardDetector, SlotMapper
from config import ROI
from src.events import EventEmitter, FileSink, SafetyGates, StdoutSink
from src.events.schemas import MetricsBlock
from src.fsm import BaccaratFSM, FrameObservation, GameState, PhaseEvent
from src.ingestion import StreamWatchdog, build_source, describe_source
from src.ingestion.watchdog import WatchdogPolicy
from src.ingestion.stream import StreamHealth
from src.motion import MotionDetector
from src.preprocessing import crop_roi, enhance_frame, normalize_resolution
from src.stability import LabelConsensus, StabilityTracker
from src.timer_watcher import TimerEvent, TimerPhase, TimerWatcher
from src.visualization import OverlayRenderer, RenderContext, build_display
from src.visualization.overlay import (
    COLOR_BAD, COLOR_GOLD, COLOR_OK, COLOR_PLAYER, COLOR_WARN,
)
from utils import monotonic_ms, now_ms


log = logging.getLogger(__name__)


@dataclass
class FrameContext:
    frame: np.ndarray
    ingest_monotonic_ms: float
    stream_health: StreamHealth


class BaccaratPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.slots = list(cfg.player_slots) + list(cfg.banker_slots)

        watchdog_policy = WatchdogPolicy(
            min_fps=cfg.telemetry.min_inference_fps,
            fps_window_s=cfg.telemetry.fps_window_s,
        )
        log.info("source: %s (%s)", cfg.ingestion.source, describe_source(cfg.ingestion.source))
        self.watchdog = StreamWatchdog(
            factory=lambda: build_source(
                source=cfg.ingestion.source,
                target_fps=cfg.ingestion.target_fps,
                nvdec=cfg.ingestion.nvdec,
                socket_timeout_s=cfg.ingestion.socket_timeout_s,
                loop=cfg.ingestion.loop,
                use_native_durations=cfg.ingestion.webp_use_native_durations,
            ),
            policy=watchdog_policy,
        )
        self.detector = CardDetector(
            model_path=str(cfg.storage.weights_dir / "yolov8n_cards.pt")
            if not cfg.detection.model_path else cfg.detection.model_path,
            conf_threshold=cfg.detection.conf_threshold,
            iou_threshold=cfg.detection.iou_threshold,
            max_detections=cfg.detection.max_detections,
            device=cfg.detection.device,
            precision=cfg.detection.precision,
        )
        self.classical_detector: Optional[ClassicalCardDetector] = None
        if not self.detector.ready:
            log.warning(
                "YOLO weights missing — falling back to classical OpenCV card detector. "
                "Provide trained weights via --yolo-weights for rank/suit recognition."
            )
            dealing_zone = self._compute_dealing_zone(cfg.rois, cfg.player_slots, cfg.banker_slots)
            self.classical_detector = ClassicalCardDetector(dealing_zone=dealing_zone)
        self.classifier = CardClassifier(
            weights_path=cfg.classification.model_path,
            num_classes=cfg.classification.num_classes,
            backbone=cfg.classification.backbone,
            input_size=cfg.classification.input_size,
            precision=cfg.classification.precision,
            device=cfg.classification.device,
            pixel_mean=cfg.classification.pixel_mean,
            pixel_std=cfg.classification.pixel_std,
        )
        self.slot_mapper = SlotMapper(rois=cfg.rois, slot_names=self.slots, min_iou=0.10)
        self.motion = MotionDetector()
        self.timer_watcher: Optional[TimerWatcher] = None
        if cfg.fsm.use_timer:
            if "timer" not in cfg.rois:
                log.warning("--use-timer requested but no 'timer' ROI is configured; ignoring")
            else:
                self.timer_watcher = TimerWatcher(
                    motion_threshold=cfg.fsm.timer_motion_threshold,
                    active_dwell_s=cfg.fsm.timer_active_dwell_s,
                    idle_dwell_s=cfg.fsm.timer_idle_dwell_s,
                    fps_estimate=cfg.ingestion.target_fps,
                )
                log.info(
                    "timer-based round detection ENABLED (ROI: %s, threshold=%.4f)",
                    cfg.rois["timer"], cfg.fsm.timer_motion_threshold,
                )
        self.stability = StabilityTracker(
            drift_px=cfg.stability.drift_threshold_px,
            stable_frames=cfg.stability.stable_frames,
            jitter_filter_frames=cfg.stability.jitter_filter_frames,
            slot_names=self.slots,
        )
        self.consensus = LabelConsensus(window=cfg.stability.stable_frames, slot_names=self.slots)
        self.fsm = BaccaratFSM(fsm_cfg=cfg.fsm, gating_cfg=cfg.gating)
        self.gates = SafetyGates(min_card_conf_publish=cfg.gating.min_card_conf_publish)
        self.async_log = AsyncLogger(
            log_dir=cfg.storage.log_dir,
            snapshot_dir=cfg.storage.snapshot_dir,
            queue_size=cfg.storage.async_queue_size,
        )
        self.emitter = EventEmitter(
            sinks=[StdoutSink(), FileSink(cfg.storage.log_dir / "events.jsonl")],
            player_slots=cfg.player_slots,
            banker_slots=cfg.banker_slots,
        )
        self.telemetry = Telemetry(
            min_fps=cfg.telemetry.min_inference_fps,
            fps_window_s=cfg.telemetry.fps_window_s,
            target_latency_min_ms=cfg.telemetry.target_e2e_latency_ms_min,
            target_latency_max_ms=cfg.telemetry.target_e2e_latency_ms_max,
            sample_window=cfg.telemetry.sample_window,
        )
        self.telemetry.on_restart(self._on_pipeline_restart)
        self.harvester = FailureHarvester(
            archive_dir=cfg.storage.archive_dir,
            clip_seconds=cfg.gating.harvest_clip_seconds,
            target_fps=cfg.ingestion.target_fps,
        )
        self.overlay = OverlayRenderer(
            rois=cfg.rois,
            player_slots=cfg.player_slots,
            banker_slots=cfg.banker_slots,
            banner_duration_s=cfg.visualization.banner_duration_s,
            show_rois=cfg.visualization.show_rois,
            show_metrics=cfg.visualization.show_metrics,
        )
        self.display = build_display(cfg)

    def _on_pipeline_restart(self) -> None:
        log.warning("inference fps below threshold — resetting pipeline state")
        self.fsm.reset()
        self.stability.reset()
        self.consensus.reset()
        self.motion.reset()

    @staticmethod
    def _compute_dealing_zone(rois, player_slots, banker_slots) -> ROI:
        """Bounding rect that covers every slot ROI — used by the classical detector."""
        all_slots = list(player_slots) + list(banker_slots)
        xs = [c for s in all_slots if s in rois for c in (rois[s].x1, rois[s].x2)]
        ys = [c for s in all_slots if s in rois for c in (rois[s].y1, rois[s].y2)]
        if not xs or not ys:
            return ROI("dealing_zone", 0.15, 0.55, 0.85, 0.85)
        pad = 0.02
        return ROI(
            "dealing_zone",
            x1=max(0.0, min(xs) - pad),
            y1=max(0.0, min(ys) - pad),
            x2=min(1.0, max(xs) + pad),
            y2=min(1.0, max(ys) + pad),
        )

    def run(self, max_frames: Optional[int] = None) -> None:
        log.info("baccarat vision AI starting")
        processed = 0
        try:
            while True:
                if max_frames is not None and processed >= max_frames:
                    break
                stream_frame = self.watchdog.read()
                if stream_frame is None:
                    if self.watchdog.finished:
                        log.info("source exhausted after %d frames", processed)
                        break
                    continue
                self._process_frame(stream_frame.frame, stream_frame.captured_at_monotonic,
                                    stream_frame.health)
                processed += 1
        except KeyboardInterrupt:
            log.info("interrupted; shutting down")
        finally:
            self.shutdown()

    def _process_frame(self, frame: np.ndarray, ingest_ts: float, health: StreamHealth) -> None:
        norm = normalize_resolution(frame, self.cfg.preprocessing.reference_size)
        enhanced = enhance_frame(
            norm,
            contrast=self.cfg.preprocessing.enhance_contrast,
            sharpen=self.cfg.preprocessing.sharpen_strength,
            apply_clahe=self.cfg.preprocessing.apply_clahe,
            clahe_clip=self.cfg.preprocessing.clahe_clip_limit,
            clahe_grid=self.cfg.preprocessing.clahe_tile_grid,
        )

        self.harvester.push_frame(enhanced)

        shoe_crop = crop_roi(enhanced, self.cfg.rois["shoe"])
        cleanup_crop = crop_roi(enhanced, self.cfg.rois["cleanup"])
        shoe_motion = self.motion.has_motion("shoe", shoe_crop)
        cleanup_motion = self.motion.has_motion("cleanup", cleanup_crop)

        timer_phase = TimerPhase.UNKNOWN
        timer_motion_value = 0.0
        timer_event: Optional[TimerEvent] = None
        if self.timer_watcher is not None and "timer" in self.cfg.rois:
            timer_crop = crop_roi(enhanced, self.cfg.rois["timer"])
            timer_phase, timer_event, timer_motion_value = self.timer_watcher.observe(timer_crop)

        if self.detector.ready:
            detections = self.detector.predict(enhanced)
        elif self.classical_detector is not None:
            detections = self.classical_detector.predict(enhanced)
        else:
            detections = []
        h, w = enhanced.shape[:2]
        slot_assignments = self.slot_mapper.assign(detections, w, h)

        crops: List[np.ndarray] = []
        crop_slots: List[str] = []
        slot_centers: Dict[str, Optional[tuple]] = {s: None for s in self.slots}

        for slot, assignment in slot_assignments.items():
            if assignment is None:
                continue
            x1, y1, x2, y2 = map(int, assignment.detection.bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(enhanced[y1:y2, x1:x2])
            crop_slots.append(slot)
            slot_centers[slot] = assignment.detection.center

        predictions = self.classifier.predict(crops) if (crops and self.classifier.ready) else []

        slot_labels: Dict[str, Optional[str]] = {s: None for s in self.slots}
        slot_confs: Dict[str, Optional[float]] = {s: None for s in self.slots}
        for slot, pred in zip(crop_slots, predictions):
            slot_labels[slot] = pred.label
            slot_confs[slot] = pred.conf

        stable_flags = self.stability.update(slot_centers)
        active_slots: Set[str] = {s for s, c in slot_centers.items() if c is not None}
        stable_slots: Set[str] = {s for s, ok in stable_flags.items() if ok}

        converged_slots: Set[str] = set()
        consensus_labels: Dict[str, Optional[str]] = {s: None for s in self.slots}
        consensus_confs: Dict[str, Optional[float]] = {s: None for s in self.slots}
        for slot in self.slots:
            result = self.consensus.update(slot, slot_labels[slot], slot_confs[slot] or 0.0)
            consensus_labels[slot] = result.label
            consensus_confs[slot] = result.conf
            if result.converged:
                converged_slots.add(slot)

        active_confs = [c for s, c in slot_confs.items() if s in active_slots and c is not None]
        min_card_conf = min(active_confs) if active_confs else 0.0
        avg_drift_px = self.stability.avg_drift_px()

        stream_unstable = health in (StreamHealth.UNSTABLE, StreamHealth.DISCONNECTED, StreamHealth.DEGRADED)
        obs = FrameObservation(
            shoe_motion=shoe_motion,
            cleanup_motion=cleanup_motion,
            active_slots=active_slots,
            stable_slots=stable_slots,
            converged_slots=converged_slots,
            min_card_conf=min_card_conf,
            avg_drift_px=avg_drift_px,
            stream_unstable=stream_unstable,
            timer_phase_ended=(timer_event == TimerEvent.TIMER_ENDED),
            timestamp=time.monotonic(),
        )

        transition = self.fsm.step(obs)
        if transition is not None:
            self._handle_transition(transition, consensus_labels, consensus_confs, min_card_conf,
                                    avg_drift_px, health, enhanced.shape)

        self._maybe_harvest(min_card_conf, avg_drift_px, enhanced.shape)

        e2e_ms = monotonic_ms() - (ingest_ts * 1000.0)
        self.telemetry.record_latency(e2e_ms)
        self.telemetry.tick_frame()

        vb = self.fsm.vision_buffer
        vb_active = vb.active and not vb.is_done()
        vb_progress = min(1.0, vb.elapsed() / vb.duration_s) if vb.active else 0.0

        ctx = RenderContext(
            state=self.fsm.state,
            detections=detections,
            slot_assignments=slot_assignments,
            consensus_labels=consensus_labels,
            consensus_confs=consensus_confs,
            stable_slots=stable_slots,
            active_slots=active_slots,
            converged_slots=converged_slots,
            min_card_conf=min_card_conf,
            avg_drift_px=avg_drift_px,
            fps=self.telemetry.current_fps(),
            e2e_latency_ms=e2e_ms,
            stream_health=health.value,
            vision_buffer_progress=vb_progress,
            vision_buffer_active=vb_active,
            timer_phase=timer_phase.value,
            timer_motion=timer_motion_value,
        )
        annotated = self.overlay.draw(enhanced, ctx)
        self.display.show(annotated)

    def _handle_transition(
        self,
        t,
        consensus_labels: Dict[str, Optional[str]],
        consensus_confs: Dict[str, Optional[float]],
        min_card_conf: float,
        avg_drift_px: float,
        stream_health: StreamHealth,
        frame_shape,
    ) -> None:
        metrics = MetricsBlock(
            phase_conf=min_card_conf,
            min_card_conf=min_card_conf,
            avg_drift_px=avg_drift_px,
            stream_health=stream_health.value,
        )

        if t.event == PhaseEvent.ROUND_START_DETECTED:
            payload = self.emitter.emit_round_start(metrics=metrics)
            self.async_log.write_event(payload)
            self.overlay.push_event("MATCH START DETECTED", color=COLOR_OK,
                                    subtitle=f"round {self.emitter.current_round_sequence or '-'}")

        elif t.event == PhaseEvent.RESULT_STABLE_DETECTED:
            active = t.observation.active_slots
            gate = self.gates.evaluate_stable(
                active_slots=active,
                min_card_conf=min_card_conf,
                all_stable=bool(active and active.issubset(t.observation.stable_slots)),
            )
            if gate.decision.value == "PUBLISH":
                payload = self.emitter.emit_result_stable(
                    slot_labels=consensus_labels,
                    slot_confs=consensus_confs,
                    metrics=metrics,
                )
                self.async_log.write_event(payload)
                self.overlay.push_event(
                    "RESULT STABLE", color=COLOR_GOLD,
                    subtitle=self._fmt_cards_summary(consensus_labels),
                )
            else:
                payload = self.emitter.emit_simple(
                    "STATE_UNCERTAIN",
                    reason=f"safety gate: {gate.decision.value} — {gate.reason}",
                    metrics=metrics,
                )
                self.async_log.write_event(payload)
                self.overlay.push_event("STATE UNCERTAIN", color=COLOR_WARN,
                                        subtitle=gate.reason)

        elif t.event == PhaseEvent.ROUND_END_DETECTED:
            payload = self.emitter.emit_round_end(metrics=metrics)
            self.async_log.write_event(payload)
            self.overlay.push_event("MATCH END DETECTED", color=(255, 130, 50),
                                    subtitle="table cleared")
            self.consensus.reset()
            self.stability.reset()

        elif t.event == PhaseEvent.STATE_UNCERTAIN:
            payload = self.emitter.emit_simple("STATE_UNCERTAIN", reason=t.reason, metrics=metrics)
            self.async_log.write_event(payload)
            self.harvester.trigger(HarvestReason.STATE_UNCERTAIN, (frame_shape[1], frame_shape[0]))
            self.overlay.push_event("STATE UNCERTAIN", color=COLOR_WARN, subtitle=t.reason)

        elif t.event == PhaseEvent.STREAM_UNSTABLE:
            payload = self.emitter.emit_simple("STREAM_UNSTABLE", reason=t.reason, metrics=metrics)
            self.async_log.write_event(payload)
            self.harvester.trigger(HarvestReason.STREAM_UNSTABLE, (frame_shape[1], frame_shape[0]))
            self.overlay.push_event("STREAM UNSTABLE", color=COLOR_BAD, subtitle=t.reason)

        elif t.event == PhaseEvent.SYSTEM_ERROR:
            payload = self.emitter.emit_simple("SYSTEM_ERROR", reason=t.reason, metrics=metrics)
            self.async_log.write_event(payload)
            self.harvester.trigger(HarvestReason.SYSTEM_ERROR, (frame_shape[1], frame_shape[0]))
            self.overlay.push_event("SYSTEM ERROR", color=COLOR_BAD, subtitle=t.reason)
            self.fsm.reset()
            self.stability.reset()
            self.consensus.reset()

    @staticmethod
    def _fmt_cards_summary(labels: Dict[str, Optional[str]]) -> str:
        parts = []
        for slot in ("p1", "p2", "p3", "b1", "b2", "b3"):
            v = labels.get(slot)
            if v:
                parts.append(f"{slot.upper()}={v}")
        return "  ".join(parts) if parts else "no cards"

    def _maybe_harvest(self, min_card_conf: float, avg_drift_px: float, frame_shape) -> None:
        if 0.0 < min_card_conf < self.cfg.gating.harvest_card_conf:
            self.harvester.trigger(HarvestReason.LOW_CONF, (frame_shape[1], frame_shape[0]))
        elif avg_drift_px > self.cfg.gating.harvest_drift_px:
            self.harvester.trigger(HarvestReason.EXCESSIVE_DRIFT, (frame_shape[1], frame_shape[0]))

    def shutdown(self) -> None:
        log.info("shutting down baccarat vision AI")
        try:
            self.display.close()
        except Exception:
            pass
        try:
            self.watchdog.release()
        except Exception:
            pass
        try:
            self.harvester.close()
        except Exception:
            pass
        try:
            self.async_log.close()
        except Exception:
            pass
