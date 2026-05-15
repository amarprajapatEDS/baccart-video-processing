"""Real-time overlay renderer: draws FSM state, ROIs, card boxes, banners, metrics."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from config import ROI
from src.detection import Detection
from src.fsm import GameState


COLOR_PLAYER = (255, 180, 0)      # BGR — cyan-ish blue
COLOR_BANKER = (0, 80, 255)       # red
COLOR_SHOE = (200, 200, 0)        # teal
COLOR_CLEANUP = (180, 0, 180)     # magenta
COLOR_TEXT_FG = (255, 255, 255)
COLOR_TEXT_BG = (20, 20, 20)
COLOR_OK = (60, 200, 60)
COLOR_WARN = (0, 165, 255)
COLOR_BAD = (0, 0, 255)
COLOR_GOLD = (0, 215, 255)


STATE_COLORS: Dict[GameState, Tuple[int, int, int]] = {
    GameState.IDLE: (110, 110, 110),
    GameState.DEALING: (0, 200, 255),
    GameState.RESULT_STABLE: COLOR_OK,
    GameState.ROUND_END_DETECTED: (255, 130, 50),
    GameState.STATE_UNCERTAIN: COLOR_WARN,
    GameState.STREAM_UNSTABLE: COLOR_BAD,
    GameState.SYSTEM_ERROR: COLOR_BAD,
}


@dataclass
class BannerEvent:
    text: str
    color: Tuple[int, int, int]
    subtitle: str = ""
    started_at: float = field(default_factory=time.monotonic)
    duration_s: float = 1.8

    def alpha(self, now: float) -> float:
        elapsed = now - self.started_at
        if elapsed >= self.duration_s:
            return 0.0
        # Fade-out in the last 25% of duration.
        fade_at = self.duration_s * 0.75
        if elapsed < fade_at:
            return 1.0
        return max(0.0, 1.0 - (elapsed - fade_at) / (self.duration_s - fade_at))


@dataclass
class RenderContext:
    state: GameState
    detections: List[Detection]
    slot_assignments: Dict[str, Optional[object]]   # SlotAssignment or None
    consensus_labels: Dict[str, Optional[str]]
    consensus_confs: Dict[str, Optional[float]]
    stable_slots: set
    active_slots: set
    converged_slots: set
    min_card_conf: float
    avg_drift_px: float
    fps: float
    e2e_latency_ms: float
    stream_health: str
    vision_buffer_progress: float = 0.0  # 0..1 (1=done)
    vision_buffer_active: bool = False
    timer_phase: str = "UNKNOWN"   # ACTIVE / IDLE / UNKNOWN
    timer_motion: float = 0.0


class OverlayRenderer:
    def __init__(
        self,
        rois: Dict[str, ROI],
        player_slots: Tuple[str, ...],
        banker_slots: Tuple[str, ...],
        banner_duration_s: float = 1.8,
        show_rois: bool = True,
        show_metrics: bool = True,
    ):
        self.rois = rois
        self.player_slots = tuple(player_slots)
        self.banker_slots = tuple(banker_slots)
        self.banner_duration_s = float(banner_duration_s)
        self.show_rois = bool(show_rois)
        self.show_metrics = bool(show_metrics)
        self._banners: Deque[BannerEvent] = deque(maxlen=4)

    def push_event(self, text: str, color: Tuple[int, int, int], subtitle: str = "") -> None:
        self._banners.append(
            BannerEvent(text=text, color=color, subtitle=subtitle, duration_s=self.banner_duration_s)
        )

    def _put_text(
        self,
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float = 0.55,
        thickness: int = 1,
        color: Tuple[int, int, int] = COLOR_TEXT_FG,
        bg: bool = True,
    ) -> None:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        if bg:
            (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = org
            cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + baseline), COLOR_TEXT_BG, -1)
        cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

    def _draw_rois(self, img: np.ndarray, active_slots: set, stable_slots: set,
                   timer_phase: str = "UNKNOWN") -> None:
        import cv2
        h, w = img.shape[:2]
        for name, roi in self.rois.items():
            x1, y1, x2, y2 = roi.to_pixels(w, h)
            if name == "shoe":
                color, thickness = COLOR_SHOE, 1
            elif name == "cleanup":
                color, thickness = COLOR_CLEANUP, 1
            elif name == "timer":
                if timer_phase == "ACTIVE":
                    color = COLOR_GOLD
                elif timer_phase == "IDLE":
                    color = (110, 110, 110)
                else:
                    color = (180, 180, 180)
                thickness = 2 if timer_phase == "ACTIVE" else 1
            elif name in self.player_slots:
                color = COLOR_PLAYER
                thickness = 2 if name in active_slots else 1
            elif name in self.banker_slots:
                color = COLOR_BANKER
                thickness = 2 if name in active_slots else 1
            else:
                color, thickness = (180, 180, 180), 1

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            label = name.upper()
            if name == "timer" and timer_phase != "UNKNOWN":
                label = f"TIMER {timer_phase}"
            if name in active_slots and name in stable_slots:
                label += " OK"
            self._put_text(img, label, (x1 + 4, y1 + 16), scale=0.45, color=color)

    def _draw_detections(
        self,
        img: np.ndarray,
        ctx: RenderContext,
    ) -> None:
        import cv2
        slot_for_det: Dict[int, str] = {}
        for slot, assignment in ctx.slot_assignments.items():
            if assignment is None:
                continue
            slot_for_det[id(assignment.detection)] = slot

        for det in ctx.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            slot = slot_for_det.get(id(det))
            if slot is None:
                color = (200, 200, 200)
            elif slot in self.player_slots:
                color = COLOR_PLAYER
            elif slot in self.banker_slots:
                color = COLOR_BANKER
            else:
                color = COLOR_OK
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label_parts = []
            if slot:
                label_parts.append(slot.upper())
            consensus_label = ctx.consensus_labels.get(slot) if slot else None
            consensus_conf = ctx.consensus_confs.get(slot) if slot else None
            if consensus_label:
                label_parts.append(consensus_label)
                if consensus_conf is not None:
                    label_parts.append(f"{consensus_conf*100:.0f}%")
            else:
                label_parts.append(f"{det.conf*100:.0f}%")

            text = " ".join(label_parts)
            self._put_text(img, text, (x1, max(14, y1 - 6)), scale=0.55, thickness=1, color=color)

    def _draw_header(self, img: np.ndarray, ctx: RenderContext) -> None:
        import cv2
        h, w = img.shape[:2]
        bar_h = 38
        cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(img[:bar_h], 0.6, np.full_like(img[:bar_h], 0), 0.4, 0, img[:bar_h])

        state_color = STATE_COLORS.get(ctx.state, (200, 200, 200))
        self._put_text(img, f"STATE: {ctx.state.value}", (10, 26), scale=0.7, thickness=2, color=state_color, bg=False)

        right_text = (
            f"FPS {ctx.fps:5.1f}   "
            f"E2E {ctx.e2e_latency_ms:5.0f}ms   "
            f"min_conf {ctx.min_card_conf:.2f}   "
            f"drift {ctx.avg_drift_px:.1f}px   "
            f"stream {ctx.stream_health}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(right_text, font, 0.5, 1)
        self._put_text(img, right_text, (w - tw - 10, 26), scale=0.5, color=COLOR_TEXT_FG, bg=False)

    def _draw_vision_buffer(self, img: np.ndarray, progress: float) -> None:
        import cv2
        h, w = img.shape[:2]
        bar_y = 40
        margin = 80
        x1, x2 = margin, w - margin
        bar_w = x2 - x1
        cv2.rectangle(img, (x1, bar_y), (x2, bar_y + 8), (50, 50, 50), -1)
        fill = int(x1 + bar_w * min(1.0, max(0.0, progress)))
        cv2.rectangle(img, (x1, bar_y), (fill, bar_y + 8), COLOR_GOLD, -1)
        self._put_text(
            img, "1.5s vision buffer",
            (x1, bar_y + 22), scale=0.45, color=COLOR_GOLD, bg=False,
        )

    def _draw_card_readout(self, img: np.ndarray, ctx: RenderContext) -> None:
        import cv2
        h, w = img.shape[:2]
        panel_h = 56
        cv2.rectangle(img, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(img[h - panel_h:], 0.65, np.full_like(img[h - panel_h:], 0), 0.35, 0, img[h - panel_h:])

        def slot_str(slot: str) -> str:
            lbl = ctx.consensus_labels.get(slot)
            conf = ctx.consensus_confs.get(slot)
            if not lbl:
                return f"{slot.upper()}=---"
            return f"{slot.upper()}={lbl}({(conf or 0)*100:.0f}%)"

        player_text = "PLAYER  " + "  ".join(slot_str(s) for s in self.player_slots)
        banker_text = "BANKER  " + "  ".join(slot_str(s) for s in self.banker_slots)
        self._put_text(img, player_text, (12, h - panel_h + 22), scale=0.55, color=COLOR_PLAYER, bg=False)
        self._put_text(img, banker_text, (12, h - 12), scale=0.55, color=COLOR_BANKER, bg=False)

    def _draw_banner(self, img: np.ndarray) -> None:
        import cv2
        now = time.monotonic()
        active = None
        while self._banners and self._banners[0].alpha(now) <= 0:
            self._banners.popleft()
        if not self._banners:
            return
        active = self._banners[-1]

        h, w = img.shape[:2]
        bar_h = 70
        y1 = h // 2 - bar_h // 2
        y2 = y1 + bar_h
        overlay = img.copy()
        cv2.rectangle(overlay, (0, y1), (w, y2), active.color, -1)
        alpha = 0.55 * active.alpha(now)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        text_alpha = active.alpha(now)
        text_color = (
            int(min(255, COLOR_TEXT_FG[0] * text_alpha + (1 - text_alpha) * active.color[0])),
            int(min(255, COLOR_TEXT_FG[1] * text_alpha + (1 - text_alpha) * active.color[1])),
            int(min(255, COLOR_TEXT_FG[2] * text_alpha + (1 - text_alpha) * active.color[2])),
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(active.text, font, 1.3, 3)
        cv2.putText(img, active.text, ((w - tw) // 2, y1 + bar_h // 2 + th // 2 - 6),
                    font, 1.3, text_color, 3, cv2.LINE_AA)
        if active.subtitle:
            (sw, _), _ = cv2.getTextSize(active.subtitle, font, 0.6, 1)
            cv2.putText(img, active.subtitle, ((w - sw) // 2, y2 - 8),
                        font, 0.6, COLOR_TEXT_FG, 1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, ctx: RenderContext) -> np.ndarray:
        canvas = frame.copy()
        if self.show_rois:
            self._draw_rois(canvas, ctx.active_slots, ctx.stable_slots,
                            timer_phase=ctx.timer_phase)
        self._draw_detections(canvas, ctx)
        if self.show_metrics:
            self._draw_header(canvas, ctx)
        if ctx.vision_buffer_active:
            self._draw_vision_buffer(canvas, ctx.vision_buffer_progress)
        self._draw_card_readout(canvas, ctx)
        self._draw_banner(canvas)
        return canvas

    @property
    def active_banner(self) -> Optional[BannerEvent]:
        now = time.monotonic()
        while self._banners and self._banners[0].alpha(now) <= 0:
            self._banners.popleft()
        return self._banners[-1] if self._banners else None
