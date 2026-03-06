from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyzlc

from .irl_wrapper import (
    IRL_HardwareDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    PandaGripperDataWrapper,
)
from .utils import NonBlockingKeyPress, UIConsole, VoidEvent
from ..control_pair.rollout_single_franka_control_pair import (
    RolloutSingleFrankaControlPair,
)
from ..policy.policy import RemotePolicy


class RolloutState(str, Enum):
    WAITING = "waiting"
    ROLLING = "rolling"
    STOPPED = "stopped"
    EXITING = "exiting"


class RolloutEvent(str, Enum):
    NEW_ROLLOUT = "new_rollout"
    END = "end"
    QUIT = "quit"


Transition = Tuple[RolloutState, Optional[Callable[[], None]]]
StateEventPair = Tuple[RolloutState, RolloutEvent]


class RolloutStateMachine:
    def __init__(
        self,
        initial_state: RolloutState,
        on_enter: Optional[Callable[[RolloutState], None]] = None,
    ) -> None:
        self._state = initial_state
        self._transitions: Dict[StateEventPair, Transition] = {}
        self._on_enter = on_enter

    @property
    def state(self) -> RolloutState:
        return self._state

    def register_transition(
        self,
        from_state: RolloutState,
        event: RolloutEvent,
        to_state: RolloutState,
        action: Optional[Callable[[], None]] = None,
    ) -> None:
        self._transitions[(from_state, event)] = (to_state, action)

    def trigger(self, event: RolloutEvent) -> bool:
        transition = self._transitions.get((self._state, event))
        if transition is None:
            return False

        next_state, action = transition
        if action is not None:
            action()
        self._state = next_state
        if self._on_enter is not None:
            self._on_enter(next_state)
        return True


@dataclass
class FrankaBesoRolloutConfig:
    policy_name: str
    task: str
    fps: int = 30
    obs_topic: Optional[str] = None
    action_topic: Optional[str] = None


class FrankaBesoRolloutManager:
    def __init__(
        self,
        obs_sources: List[IRL_HardwareDataWrapper],
        control_pair: RolloutSingleFrankaControlPair,
        cfg: FrankaBesoRolloutConfig,
    ) -> None:
        self.obs_sources = obs_sources
        self.control_pair = control_pair
        self.cfg = cfg
        self.fps = int(cfg.fps)
        self.last_timestamp: Optional[float] = None

        self.policy = RemotePolicy(
            cfg.policy_name, obs_topic=cfg.obs_topic, action_topic=cfg.action_topic
        )

        self._ui_console = UIConsole()
        self._start_rollout_event = VoidEvent()
        self._stop_rollout_event = VoidEvent()
        self._state_machine = RolloutStateMachine(
            initial_state=RolloutState.WAITING,
            on_enter=self._on_state_enter,
        )
        self._state_machine.register_transition(
            RolloutState.WAITING,
            RolloutEvent.NEW_ROLLOUT,
            RolloutState.ROLLING,
            action=self._start_rollout,
        )
        self._state_machine.register_transition(
            RolloutState.ROLLING,
            RolloutEvent.END,
            RolloutState.STOPPED,
            action=self._end_rollout,
        )
        self._state_machine.register_transition(
            RolloutState.STOPPED,
            RolloutEvent.QUIT,
            RolloutState.EXITING,
            action=self._close,
        )

        self.cameras: List[ImageDataWrapper] = []
        self.arm_wrapper: Optional[PandaArmDataWrapper] = None
        self.gripper_wrapper: Optional[PandaGripperDataWrapper] = None
        for hw in obs_sources:
            if isinstance(hw, ImageDataWrapper) or hw.hw_type == "camera":
                self.cameras.append(hw)  # type: ignore[arg-type]
            elif isinstance(hw, PandaArmDataWrapper) or hw.hw_type == "follower_arm":
                self.arm_wrapper = hw  # type: ignore[assignment]
            elif isinstance(hw, PandaGripperDataWrapper) or hw.hw_type == "follower_gripper":
                self.gripper_wrapper = hw

        if self.arm_wrapper is None:
            raise ValueError("Missing PandaArmDataWrapper for rollout.")
        if self.gripper_wrapper is None:
            raise ValueError("Missing PandaGripperDataWrapper for rollout.")

        self._need_policy_reset = False
        self._rollout_start_wall_ts = 0.0
        self._last_action_ts = 0.0
        self._closed = False

    def register_start_rollout_event(self, handler: Callable[[], None]) -> None:
        self._start_rollout_event.subscribe(handler)

    def register_stop_rollout_event(self, handler: Callable[[], None]) -> None:
        self._stop_rollout_event.subscribe(handler)

    def run(self) -> None:
        self._on_state_enter(self._state_machine.state)
        try:
            with NonBlockingKeyPress() as kp:
                while self._state_machine.state != RolloutState.EXITING:
                    key = kp.get_data()
                    if key:
                        self._handle_keypress(key)
                    if self._state_machine.state == RolloutState.ROLLING:
                        try:
                            self._rollout_step()
                        except Exception as exc:
                            pyzlc.error(f"Error during rollout step: {exc}")
        finally:
            self._close()

    def _handle_keypress(self, key: str) -> None:
        if key == "n" and self._state_machine.state == RolloutState.WAITING:
            self._state_machine.trigger(RolloutEvent.NEW_ROLLOUT)
        elif key == "e" and self._state_machine.state == RolloutState.ROLLING:
            self._state_machine.trigger(RolloutEvent.END)
        elif key == "q" and self._state_machine.state == RolloutState.STOPPED:
            self._state_machine.trigger(RolloutEvent.QUIT)

    def _on_state_enter(self, state: RolloutState) -> None:
        if state == RolloutState.WAITING:
            self._ui_console.update_hint("Press 'n' to start rollout")
        elif state == RolloutState.ROLLING:
            self._ui_console.update_hint("Rolling... Press 'e' to end rollout")
        elif state == RolloutState.STOPPED:
            self._ui_console.update_hint("Rollout ended. Press 'q' to quit")
        elif state == RolloutState.EXITING:
            self._ui_console.update_hint("Exiting rollout")

    def _start_rollout(self) -> None:
        self._ui_console.update_hint("Starting rollout...")
        self.last_timestamp = None
        self._need_policy_reset = True
        self._rollout_start_wall_ts = time.time()
        self._last_action_ts = self._rollout_start_wall_ts
        self._start_rollout_event.emit()

    def _end_rollout(self) -> None:
        self._ui_console.update_hint("Stopping rollout...")
        self._stop_rollout_event.emit()
        self._ui_console.log("Rollout ended.")

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_rollout_event.emit()
        self._ui_console.update_hint("Rollout manager closed.")

    def _rollout_step(self) -> None:
        if self.last_timestamp is None:
            self.last_timestamp = time.perf_counter()

        obs = self._build_observation(reset_policy=self._need_policy_reset)
        self._need_policy_reset = False
        self.policy.send_observation(obs)

        action_msg = self.policy.current_action
        if action_msg is not None:
            action_ts = float(action_msg["timestamp"])
            if action_ts >= self._rollout_start_wall_ts and action_ts >= self._last_action_ts:
                action = np.asarray(action_msg["action"], dtype=np.float64).reshape(-1)
                self.control_pair.update_action(action)
                self._last_action_ts = action_ts

        elapsed = time.perf_counter() - self.last_timestamp
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.0:
            pyzlc.sleep(sleep_time)
        self.last_timestamp = time.perf_counter()

    def _build_observation(self, reset_policy: bool) -> Dict[str, Any]:
        return {
            "state": self._build_state_vector().tolist(),
            "images": self._build_images(),
            "task": self.cfg.task,
            "reset_policy": reset_policy,
        }

    def _build_state_vector(self) -> np.ndarray:
        arm_state = self.arm_wrapper.capture_step()
        q = np.asarray(arm_state["q"], dtype=np.float32).reshape(-1)

        grip_state = self.gripper_wrapper.capture_step()
        gripper_width = float(grip_state["width"])

        return np.concatenate([q, np.asarray([gripper_width], dtype=np.float32)])

    def _build_images(self) -> Dict[str, Any]:
        images: Dict[str, Any] = {}
        for cam in self.cameras:
            frame = cam.capture_step()
            if frame is None:
                continue
            h, w, c = frame.shape
            images[cam.hw_name] = {
                "height": int(h),
                "width": int(w),
                "channels": int(c),
                "rgb_data": frame.tobytes(),
            }
        return images
