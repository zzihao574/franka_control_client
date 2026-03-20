#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pyzlc
import torch
from omegaconf import DictConfig


log = logging.getLogger(__name__)


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class BeastPolicyNodeConfig:
    checkpoint_path: str
    dataset_root: str
    device: str
    enforce_init_pos: bool
    obs_topic: str
    action_topic: str
    pyzlc_name: str
    pyzlc_host: str
    pyzlc_group: str
    pyzlc_group_name: str
    pyzlc_group_port: int
    seed: int
    debug_print_control_points: bool = False


class BeastPolicyNode:
    def __init__(self, cfg: BeastPolicyNodeConfig) -> None:
        self.cfg = cfg
        self._running = False
        self._obs_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._obs_seq_lock = threading.Lock()
        self._active_rollout_id: int | None = None
        self._last_enqueued_obs_seq: int | None = None
        self._state_mean: torch.Tensor | None = None
        self._state_std: torch.Tensor | None = None

        pyzlc.init(
            self.cfg.pyzlc_name,
            self.cfg.pyzlc_host,
            group=self.cfg.pyzlc_group,
            group_name=self.cfg.pyzlc_group_name,
            group_port=self.cfg.pyzlc_group_port,
        )
        pyzlc.register_subscriber_handler(self.cfg.obs_topic, self._on_observation)
        self._action_pub = pyzlc.Publisher(self.cfg.action_topic)

        self.policy, self.device, self.task = self._load_policy()
        self.policy.reset()

    def _clear_pending_observations(self) -> None:
        saw_stop_sentinel = False
        while True:
            try:
                item = self._obs_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                saw_stop_sentinel = True
        if saw_stop_sentinel:
            self._obs_queue.put(None)

    def _on_observation(self, msg: dict[str, Any]) -> None:
        rollout_id = int(msg["rollout_id"])
        obs_seq = int(msg["obs_seq"])

        with self._obs_seq_lock:
            if self._active_rollout_id is None or rollout_id > self._active_rollout_id:
                self._active_rollout_id = rollout_id
                self._last_enqueued_obs_seq = None
                self._clear_pending_observations()
            elif rollout_id < self._active_rollout_id:
                return

            if self._last_enqueued_obs_seq is not None and obs_seq <= self._last_enqueued_obs_seq:
                return

            self._last_enqueued_obs_seq = obs_seq

        self._obs_queue.put(msg)

    def _load_stats(self) -> dict[str, dict[str, Any]]:
        dataset_root = Path(self.cfg.dataset_root).expanduser().resolve()
        stats_path = dataset_root / "meta" / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"stats.json not found: {stats_path}")

        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        if "action" not in stats:
            raise KeyError(f"'action' stats missing in: {stats_path}")

        return stats

    def _resolve_task(self) -> str:
        import pandas as pd

        dataset_root = Path(self.cfg.dataset_root).expanduser().resolve()
        tasks_path = dataset_root / "meta" / "tasks.parquet"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"tasks.parquet not found: {tasks_path}")

        tasks = pd.read_parquet(tasks_path)
        task_list = [str(task).strip() for task in tasks.index.tolist() if str(task).strip()]
        unique_tasks = list(dict.fromkeys(task_list))
        if len(unique_tasks) != 1:
            raise ValueError(
                f"Expected exactly one task in {tasks_path}, got {len(unique_tasks)}: {unique_tasks}"
            )
        return unique_tasks[0]

    def _load_policy(self):
        from policies.beastf.beastf_config import BeastVLAConfig
        from policies.beastf.modeling_beastf import BeastVLAPolicy

        task = self._resolve_task()

        ckpt = Path(self.cfg.checkpoint_path).expanduser().resolve()
        if not ckpt.is_dir():
            raise FileNotFoundError(
                f"checkpoint_path must be a pretrained_model directory, got: {ckpt}"
            )

        stats = self._load_stats()
        device = torch.device(self.cfg.device)
        state_stats = stats.get("observation.state")
        if state_stats is not None and "mean" in state_stats and "std" in state_stats:
            self._state_mean = torch.as_tensor(state_stats["mean"], dtype=torch.float32, device=device)
            self._state_std = torch.as_tensor(state_stats["std"], dtype=torch.float32, device=device)

        model_cfg = BeastVLAConfig.from_pretrained(ckpt)
        model_cfg.device = str(device)
        model_cfg.enforce_init_pos = self.cfg.enforce_init_pos

        if getattr(model_cfg, "return_act_chunk", False):
            raise ValueError(
                "Beast rollout requires return_act_chunk=false, "
                "because the rollout manager expects one action per control step."
            )

        image_keys = [
            key for key in model_cfg.input_features.keys()
            if str(key).startswith("observation.image.")
        ]
        if not image_keys:
            raise ValueError("Beast config does not define any observation.image.* inputs")

        policy = BeastVLAPolicy.from_pretrained(
            ckpt,
            config=model_cfg,
            dataset_stats=stats,
            task=task,
            strict=False,
        )
        policy = policy.to(device)
        policy.eval()
        policy.model.debug_print_control_points = self.cfg.debug_print_control_points

        return policy, device, task

    def _decode_image(self, img: Any) -> np.ndarray:
        if isinstance(img, np.ndarray):
            if img.ndim != 3:
                raise ValueError(f"Expected HWC image array, got shape {img.shape}")
            return np.ascontiguousarray(img)

        if not isinstance(img, dict):
            raise TypeError(f"Unsupported image payload type: {type(img)}")

        required_keys = {"height", "width", "rgb_data"}
        missing = required_keys.difference(img.keys())
        if missing:
            raise KeyError(f"Image payload missing keys: {sorted(missing)}")

        h = int(img["height"])
        w = int(img["width"])
        c = int(img.get("channels", 3))

        arr = np.frombuffer(img["rgb_data"], dtype=np.uint8)
        expected = h * w * c
        if arr.size != expected:
            raise ValueError(
                f"Decoded image has {arr.size} values, expected {expected} for shape ({h}, {w}, {c})"
            )

        return arr.reshape((h, w, c)).copy()

    def _image_to_tensor(
        self,
        image: np.ndarray,
        *,
        expected_channels: int,
    ) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image array, got shape {image.shape}")
        if image.shape[2] != expected_channels:
            raise ValueError(
                f"Image channel mismatch: got {image.shape[2]}, expected {expected_channels}"
            )

        chw = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)).copy()
        tensor = torch.from_numpy(chw).unsqueeze(0)  # [1, C, H, W]
        return tensor.to(self.device)

    def _build_policy_input(self, obs_msg: dict[str, Any]) -> dict[str, Any]:
        if "images" not in obs_msg:
            raise KeyError("Observation is missing 'images'")

        images = obs_msg["images"]
        if not isinstance(images, dict):
            raise TypeError("'images' must be a dict keyed by camera name")

        batch: dict[str, Any] = {}

        for feat_key, feat in self.policy.config.input_features.items():
            key = str(feat_key)
            if not key.startswith("observation.image."):
                continue

            cam_name = key.replace("observation.image.", "", 1)
            if cam_name not in images:
                raise KeyError(f"Camera '{cam_name}' missing in observation images")

            image = self._decode_image(images[cam_name])
            shape = tuple(feat.shape)
            if len(shape) != 3:
                raise ValueError(f"Invalid visual feature shape for {key}: {shape}")

            batch[key] = self._image_to_tensor(image, expected_channels=shape[0])

        state = obs_msg.get("state")
        if state is not None:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if self._state_mean is not None and self._state_std is not None:
                state_tensor = (state_tensor - self._state_mean) / (self._state_std + 1e-8)
            batch["observation.state"] = state_tensor

        batch["task"] = self.task
        return batch

    def _process_observation(self, obs_msg: dict[str, Any]) -> None:
        rollout_id = int(obs_msg["rollout_id"])
        obs_seq = int(obs_msg["obs_seq"])

        if bool(obs_msg.get("reset_policy", False)):
            self.policy.reset()

        policy_input = self._build_policy_input(obs_msg)

        with torch.inference_mode():
            action = self.policy.select_action(policy_input)

        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError(
                f"Expected select_action() to return shape [1, action_dim], got {tuple(action.shape)}"
            )

        action_vec = action[0]
        payload = {
            "rollout_id": rollout_id,
            "source_obs_seq": obs_seq,
            "timestamp": time.time(),
            "action": action_vec.detach().cpu().tolist(),
            "shape": list(action_vec.shape),
        }
        self._action_pub.publish(payload)

    def run(self) -> None:
        self._running = True

        while self._running:
            try:
                obs_msg = self._obs_queue.get()
                if obs_msg is None:
                    break
                self._process_observation(obs_msg)
            except Exception as exc:
                pyzlc.error(f"beast_policy_node step error: {exc}")

    def stop(self) -> None:
        self._running = False
        self._obs_queue.put(None)


def _build_node_cfg(cfg: DictConfig) -> BeastPolicyNodeConfig:
    return BeastPolicyNodeConfig(
        checkpoint_path=str(cfg.checkpoint_path),
        dataset_root=str(cfg.dataset_root),
        device=str(cfg.device),
        enforce_init_pos=bool(cfg.get("enforce_init_pos", True)),
        obs_topic=str(cfg.obs_topic),
        action_topic=str(cfg.action_topic),
        pyzlc_name=str(cfg.pyzlc_name),
        pyzlc_host=str(cfg.pyzlc_host),
        pyzlc_group=str(cfg.pyzlc_group),
        pyzlc_group_name=str(cfg.pyzlc_group_name),
        pyzlc_group_port=int(cfg.pyzlc_group_port),
        seed=int(cfg.seed),
        debug_print_control_points=bool(cfg.get("debug_print_control_points", False)),
    )


@hydra.main(
    config_path="../../../configs",
    config_name="beast_rollout_franka.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    policy_cfg = cfg.policy_node
    lerobot_repo_root = Path(str(policy_cfg.lerobot_repo_root)).expanduser().resolve()
    src_root = lerobot_repo_root / "src"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Beast src directory not found: {src_root}")

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    node_cfg = _build_node_cfg(policy_cfg)
    set_seed_everywhere(node_cfg.seed)

    node = BeastPolicyNode(node_cfg)
    log.info(
        "Beast policy node started successfully. checkpoint=%s device=%s task=%s obs_topic=%s action_topic=%s",
        node_cfg.checkpoint_path,
        node_cfg.device,
        node.task,
        node_cfg.obs_topic,
        node_cfg.action_topic,
    )
    try:
        node.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, exiting.")
    finally:
        node.stop()
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
