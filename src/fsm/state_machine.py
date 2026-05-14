"""The Baccarat Vision AI Finite State Machine.

Phases: IDLE → DEALING → RESULT_STABLE → ROUND_END_DETECTED
Safety: STATE_UNCERTAIN, STREAM_UNSTABLE, SYSTEM_ERROR

Transitions are gated by N-frame verification + the 1.5s vision buffer.
The FSM emits public events but does NOT compute game scores — fact extraction
only, per the responsibility firewall.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from config import FSMConfig, GatingConfig

from .buffer import VisionBuffer


class GameState(str, Enum):
    IDLE = "IDLE"
    DEALING = "DEALING"
    RESULT_STABLE = "RESULT_STABLE"
    ROUND_END_DETECTED = "ROUND_END_DETECTED"
    STATE_UNCERTAIN = "STATE_UNCERTAIN"
    STREAM_UNSTABLE = "STREAM_UNSTABLE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class PhaseEvent(str, Enum):
    ROUND_START_DETECTED = "ROUND_START_DETECTED"
    RESULT_STABLE_DETECTED = "RESULT_STABLE_DETECTED"
    ROUND_END_DETECTED = "ROUND_END_DETECTED"
    STATE_UNCERTAIN = "STATE_UNCERTAIN"
    STREAM_UNSTABLE = "STREAM_UNSTABLE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class FrameObservation:
    shoe_motion: bool
    cleanup_motion: bool
    active_slots: Set[str] = field(default_factory=set)
    stable_slots: Set[str] = field(default_factory=set)
    converged_slots: Set[str] = field(default_factory=set)
    min_card_conf: float = 0.0
    avg_drift_px: float = 0.0
    stream_unstable: bool = False
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class StateTransition:
    from_state: GameState
    to_state: GameState
    event: Optional[PhaseEvent]
    reason: str
    observation: FrameObservation


class BaccaratFSM:
    INITIAL_CARDS: Set[str] = {"p1", "p2", "b1", "b2"}

    def __init__(self, fsm_cfg: FSMConfig, gating_cfg: GatingConfig):
        self.cfg = fsm_cfg
        self.gating = gating_cfg
        self.state = GameState.IDLE
        self.vision_buffer = VisionBuffer(duration_s=fsm_cfg.vision_buffer_s)

        self._round_start_streak = 0
        self._round_end_streak = 0
        self._dealing_started_at: Optional[float] = None
        self._uncertain_started_at: Optional[float] = None

    def reset(self) -> None:
        self.state = GameState.IDLE
        self.vision_buffer.cancel()
        self._round_start_streak = 0
        self._round_end_streak = 0
        self._dealing_started_at = None
        self._uncertain_started_at = None

    def step(self, obs: FrameObservation) -> Optional[StateTransition]:
        if obs.stream_unstable:
            return self._maybe_transition(
                GameState.STREAM_UNSTABLE,
                PhaseEvent.STREAM_UNSTABLE,
                reason="stream watchdog flagged degraded/disconnected",
                obs=obs,
            )

        if self.state == GameState.STREAM_UNSTABLE:
            if not obs.stream_unstable:
                return self._maybe_transition(
                    GameState.IDLE, None, reason="stream recovered", obs=obs
                )
            return None

        if self.state == GameState.IDLE:
            return self._step_idle(obs)
        if self.state == GameState.DEALING:
            return self._step_dealing(obs)
        if self.state == GameState.RESULT_STABLE:
            return self._step_result(obs)
        if self.state == GameState.ROUND_END_DETECTED:
            return self._step_round_end(obs)
        if self.state == GameState.STATE_UNCERTAIN:
            return self._step_uncertain(obs)
        if self.state == GameState.SYSTEM_ERROR:
            return self._maybe_transition(GameState.IDLE, None, "operator reset", obs)
        return None

    def _step_idle(self, obs: FrameObservation) -> Optional[StateTransition]:
        if obs.shoe_motion and obs.active_slots:
            self._round_start_streak += 1
        else:
            self._round_start_streak = 0
        if self._round_start_streak >= self.cfg.round_start_n:
            self._dealing_started_at = obs.timestamp
            self._round_start_streak = 0
            return self._maybe_transition(
                GameState.DEALING,
                PhaseEvent.ROUND_START_DETECTED,
                reason=f"shoe motion + first card for {self.cfg.round_start_n} frames",
                obs=obs,
            )
        return None

    def _step_dealing(self, obs: FrameObservation) -> Optional[StateTransition]:
        if self._dealing_started_at is not None:
            if (obs.timestamp - self._dealing_started_at) >= self.cfg.dealing_timeout_s:
                self._dealing_started_at = None
                return self._maybe_transition(
                    GameState.SYSTEM_ERROR,
                    PhaseEvent.SYSTEM_ERROR,
                    reason=f"DEALING exceeded {self.cfg.dealing_timeout_s}s",
                    obs=obs,
                )

        if (
            obs.min_card_conf > 0
            and obs.min_card_conf < self.gating.uncertain_card_conf
            and obs.converged_slots
        ):
            if self._uncertain_started_at is None:
                self._uncertain_started_at = obs.timestamp
            elif (obs.timestamp - self._uncertain_started_at) >= self.cfg.uncertainty_dwell_s:
                self._uncertain_started_at = None
                return self._maybe_transition(
                    GameState.STATE_UNCERTAIN,
                    PhaseEvent.STATE_UNCERTAIN,
                    reason="card conf below uncertain threshold for dwell time",
                    obs=obs,
                )
        else:
            self._uncertain_started_at = None

        initial_present = self.INITIAL_CARDS.issubset(obs.active_slots)
        if initial_present and not self.vision_buffer.active:
            self.vision_buffer.start(obs.active_slots)
        elif self.vision_buffer.active:
            self.vision_buffer.observe(obs.active_slots)

        if not self.vision_buffer.is_done():
            return None

        all_active_stable = obs.active_slots and obs.active_slots.issubset(obs.stable_slots)
        all_active_converged = obs.active_slots and obs.active_slots.issubset(obs.converged_slots)
        conf_ok = obs.min_card_conf >= self.gating.min_card_conf_publish

        if all_active_stable and all_active_converged and conf_ok:
            return self._maybe_transition(
                GameState.RESULT_STABLE,
                PhaseEvent.RESULT_STABLE_DETECTED,
                reason="all active cards stable+converged with min_conf>=publish threshold",
                obs=obs,
            )
        return None

    def _step_result(self, obs: FrameObservation) -> Optional[StateTransition]:
        if obs.cleanup_motion and not obs.active_slots:
            self._round_end_streak += 1
        else:
            self._round_end_streak = 0
        if self._round_end_streak >= self.cfg.round_end_n:
            self._round_end_streak = 0
            self.vision_buffer.cancel()
            return self._maybe_transition(
                GameState.ROUND_END_DETECTED,
                PhaseEvent.ROUND_END_DETECTED,
                reason=f"cleanup + empty slots for {self.cfg.round_end_n} frames",
                obs=obs,
            )
        return None

    def _step_round_end(self, obs: FrameObservation) -> Optional[StateTransition]:
        return self._maybe_transition(GameState.IDLE, None, "round end → idle", obs)

    def _step_uncertain(self, obs: FrameObservation) -> Optional[StateTransition]:
        if obs.min_card_conf >= self.gating.uncertain_card_conf and obs.converged_slots:
            return self._maybe_transition(GameState.DEALING, None, "uncertainty cleared", obs)
        return None

    def _maybe_transition(
        self,
        to_state: GameState,
        event: Optional[PhaseEvent],
        reason: str,
        obs: FrameObservation,
    ) -> Optional[StateTransition]:
        if to_state == self.state and event is None:
            return None
        prev = self.state
        self.state = to_state
        if to_state == GameState.IDLE:
            self.vision_buffer.cancel()
            self._dealing_started_at = None
            self._round_start_streak = 0
            self._round_end_streak = 0
            self._uncertain_started_at = None
        return StateTransition(from_state=prev, to_state=to_state, event=event, reason=reason, observation=obs)
