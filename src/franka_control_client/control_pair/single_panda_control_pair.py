import pyzlc
import numpy as np
from typing import Tuple

from .control_pair import ControlPair
from ..franka_robot.franka_panda import FrankaPanda
from ..franka_robot.panda_arm import ControlMode


GRIPPER_SPEED = 0.5
GRIPPER_THRESHOLD = 0.05
FOLLOWER_GRIPPER_CLOSE = 0.02
FOLLOWER_GRIPPER_OPEN = 0.07
CONTROL_DT_S = 0.01  # 100 Hz


class SinglePandaKTControlPair(ControlPair):
    def __init__(
        self,
        leader: FrankaPanda,
        follower: FrankaPanda,
        align_q: Tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(align_q) != 7:
            raise ValueError(f"align_q must have 7 joints, got {len(align_q)}")

        self.leader = leader
        self.follower = follower
        self._align_q = tuple(float(v) for v in align_q)

    def control_rest(self) -> None:
        # Arm reset: IDLE -> move align_q
        align_q_np = np.asarray(self._align_q, dtype=np.float64)
        self.follower.panda_arm.send_joint_position_command(align_q_np)
        self.leader.panda_arm.set_franka_arm_control_mode(ControlMode.GRAVITYCOMP)
        self.follower.panda_arm.set_franka_arm_control_mode(
            ControlMode.HybridJointImpedance
        )

    def control_step(self) -> None:
        leader_arm_state = self.leader.panda_arm.current_state
        if leader_arm_state is not None:
            q_leader = np.asarray(leader_arm_state["q"], dtype=np.float64).reshape(-1)
            if q_leader.size == 7:
                self.follower.panda_arm.send_joint_position_command(q_leader)

        leader_gripper_state = self.leader.panda_gripper.current_state
        if leader_gripper_state is not None:
            leader_width = float(leader_gripper_state["width"])
            target_width = (
                FOLLOWER_GRIPPER_CLOSE
                if leader_width < GRIPPER_THRESHOLD
                else FOLLOWER_GRIPPER_OPEN
            )
            self.follower.panda_gripper.send_gripper_command(
                width=target_width,
                speed=GRIPPER_SPEED,
            )

        pyzlc.sleep(CONTROL_DT_S)

    def control_end(self) -> None:
        pyzlc.info("SinglePandaKTControlPair: control_end")
        self.leader.panda_arm.set_franka_arm_control_mode(ControlMode.IDLE)
        self.follower.panda_arm.set_franka_arm_control_mode(ControlMode.IDLE)

        self.leader.panda_arm.move_franka_arm_to_joint_position(self._align_q)
        self.follower.panda_arm.move_franka_arm_to_joint_position(self._align_q)

        self.follower.panda_gripper.send_gripper_command(
                width=0.07,
                speed=GRIPPER_SPEED,
            )


