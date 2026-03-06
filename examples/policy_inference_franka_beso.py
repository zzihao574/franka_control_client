from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

import pyzlc
import yaml

from franka_control_client.camera.camera import CameraDevice
from franka_control_client.control_pair.rollout_single_franka_control_pair import (
    RolloutSingleFrankaControlPair,
)
from franka_control_client.franka_robot.franka_panda import FrankaPanda
from franka_control_client.franka_robot.panda_arm import RemotePandaArm
from franka_control_client.franka_robot.panda_gripper import RemotePandaGripper
from franka_control_client.policy_inference.franka_beso_rollout_manager import (
    FrankaBesoRolloutConfig,
    FrankaBesoRolloutManager,
)
from franka_control_client.policy_inference.irl_wrapper import (
    IRL_HardwareDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    PandaGripperDataWrapper,
)


def _parse_args() -> argparse.Namespace:
    default_cfg = (
        Path(__file__).resolve().parents[1] / "configs" / "beso_rollout_franka.yaml"
    )
    parser = argparse.ArgumentParser("Franka BESO rollout (GPU machine)")
    parser.add_argument("--config", type=str, default=str(default_cfg))
    return parser.parse_args()


def _load_cfg(path: str) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["rollout_node"]


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config)

    pyzlc.init(
        cfg["pyzlc_name"],
        cfg["pyzlc_host"],
        group=cfg["pyzlc_group"],
        group_name=cfg["pyzlc_group_name"],
        group_port=int(cfg["pyzlc_group_port"]),
    )

    follower = FrankaPanda(
        "franka_panda_follower",
        RemotePandaArm(cfg["follower_arm"]),
        RemotePandaGripper(cfg["follower_gripper"]),
    )
    control_pair = RolloutSingleFrankaControlPair(
        follower=follower,
        control_dt_s=float(cfg["control_dt_s"]),
        gripper_speed=float(cfg["gripper_speed"]),
        align_q=tuple(cfg["align_q"]),
    )

    obs_sources: List[IRL_HardwareDataWrapper] = []
    for camera_name in cfg["camera_names"]:
        obs_sources.append(
            ImageDataWrapper(
                CameraDevice(camera_name, preview=bool(cfg["camera_preview"])),
                hw_name=camera_name,
            )
        )
    obs_sources.append(PandaArmDataWrapper(follower.panda_arm, hw_name=cfg["follower_arm"]))
    obs_sources.append(
        PandaGripperDataWrapper(
            follower.panda_gripper,
            hw_name=cfg["follower_gripper"],
        )
    )

    rollout_cfg = FrankaBesoRolloutConfig(
        policy_name=cfg["policy_name"],
        task=str(cfg.get("task", "")),
        fps=int(cfg["fps"]),
        obs_topic=cfg["obs_topic"],
        action_topic=cfg["action_topic"],
    )
    manager = FrankaBesoRolloutManager(
        obs_sources=obs_sources,
        control_pair=control_pair,
        cfg=rollout_cfg,
    )
    manager.register_start_rollout_event(control_pair.start_control_pair)
    manager.register_stop_rollout_event(control_pair.stop_control_pair)

    try:
        manager.run()
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
