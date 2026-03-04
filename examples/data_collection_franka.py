import time
from typing import List

import pyzlc

from franka_control_client.camera.camera import CameraDevice
from franka_control_client.data_collection.lerobot_data_collection import (
    LeRobotDataCollection,
)
from franka_control_client.data_collection.wrapper import (
    HardwareDataWrapper,
    ImageDataWrapper,
)
from franka_control_client.data_collection.wrapper import (
    PandaArmDataWrapper,
    PandaGripperDataWrapper,
)
from franka_control_client.franka_robot.franka_panda import (
    FrankaPanda,
    RemotePandaArm,
    RemotePandaGripper,
)
from franka_control_client.control_pair.single_panda_control_pair import (
    SinglePandaKTControlPair,
)
from franka_control_client.franka_robot.panda_arm import ControlMode

ZLC_IP = "141.3.53.63"
GROUP = "224.0.0.1"
GROUP_PORT = 7729
GROUP_NAME = "robot_lab_201_202"

LEADER_ARM = "Panda201"
FOLLOWER_ARM = "Panda202"

LEADER_GRIPPER = "PandaGripper201"
FOLLOWER_GRIPPER = "PandaGripper202"

ALIGN_Q = (
    -0.838027175526648,
    0.23739903886903796,
    0.998479903707504,
    -1.9307362981093554,
    -0.1399801990323587,
    2.0812839427259235,
    1.0100087088974428,
)

if __name__ == "__main__":
    pyzlc.init(
        "data_collection",
        ZLC_IP,
        group=GROUP,
        group_port=GROUP_PORT,
        group_name=GROUP_NAME,
    )

    leader = FrankaPanda(
        "franka_panda",
        RemotePandaArm(LEADER_ARM),
        RemotePandaGripper(LEADER_GRIPPER),
    )
    follower = FrankaPanda(
        "franka_panda_follower",
        RemotePandaArm(FOLLOWER_ARM),
        RemotePandaGripper(FOLLOWER_GRIPPER),
    )

    control_pair = SinglePandaKTControlPair(leader, follower, align_q=ALIGN_Q)

    camera_centric = ImageDataWrapper(CameraDevice("centric_cam", preview=True))
    camera_wrist = ImageDataWrapper(CameraDevice("wrist_cam", preview=True))

    data_collectors: List[HardwareDataWrapper] = []
    data_collectors.append(camera_centric)
    data_collectors.append(camera_wrist)
    data_collectors.append(PandaArmDataWrapper(leader.panda_arm))
    data_collectors.append(PandaGripperDataWrapper(leader.panda_gripper))
    data_collectors.append(PandaArmDataWrapper(follower.panda_arm, include_action=True))
    data_collectors.append(PandaGripperDataWrapper(follower.panda_gripper, include_action=True))

    name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    data_collection_manager = LeRobotDataCollection(
        data_collectors, f"/home/irl-admin/zihao_zhang/datasets/{name}", task="pick_and_place"
    )

    data_collection_manager.register_start_collecting_event(
        control_pair.start_control_pair
    )
    data_collection_manager.register_stop_collecting_event(
        control_pair.stop_control_pair
    )

    leader.panda_arm.set_franka_arm_control_mode(ControlMode.IDLE)
    follower.panda_arm.set_franka_arm_control_mode(ControlMode.IDLE)

    leader.panda_arm.move_franka_arm_to_joint_position(ALIGN_Q)
    follower.panda_arm.move_franka_arm_to_joint_position(ALIGN_Q)
    
    follower.panda_gripper.start_control()
    leader.panda_gripper.stop_control()
    follower.panda_gripper.send_gripper_command(
                width=0.07,
                speed=0.5,
            )

    data_collection_manager.run()
    pyzlc.shutdown()
