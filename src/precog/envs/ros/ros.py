#!/usr/bin/env python3
"""
Generic, config-driven ROS 2 <-> ML environment bridge.

Sensors and actuators are entirely described by env_config.yaml. This
file reads that config, dynamically resolves the ROS message classes,
wires up subscriptions/publishers, and exposes a Gym-like reset/step
API. Adding or removing a topic never requires touching this file.
"""

import threading
from dataclasses import dataclass
from typing import Any, Optional

import yaml
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosidl_runtime_py.utilities import get_message

from .codecs import OBS_CODECS, ACTION_CODECS, TRANSFORMS


@dataclass
class ObsSpec:
    key: str
    topic: str
    msg_type: str
    codec: str
    shape: tuple
    max_age_sec: float = 1.0
    required: bool = True


@dataclass
class ActionSpec:
    key: str
    topic: str
    msg_type: str
    codec: str


def _build_qos(cfg: dict) -> QoSProfile:
    reliability = (
        ReliabilityPolicy.BEST_EFFORT
        if cfg.get("reliability") == "best_effort"
        else ReliabilityPolicy.RELIABLE
    )
    return QoSProfile(
        reliability=reliability,
        history=HistoryPolicy.KEEP_LAST,
        depth=cfg.get("history_depth", 10),
    )


class ROSEnvironment(Node):
    """Config-driven, Gym-like wrapper around a set of ROS 2 topics."""

    def __init__(self, config_path: str, device: str = "cpu"):
        super().__init__("ros_environment_wrapper")

        self.device = torch.device(device)
        self.state_lock = threading.Lock()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        qos = _build_qos(cfg.get("qos_default", {}))

        obs_cfgs = list(cfg.get("observations", []))
        obs_cfgs.extend(cfg.get("expert_action_observations", []))
        self.obs_specs = [ObsSpec(**o) for o in obs_cfgs]
        self.action_specs = [ActionSpec(**a) for a in cfg.get("actions", [])]

        self._latest_obs: dict[str, Optional[np.ndarray]] = {
            s.key: None for s in self.obs_specs
        }
        self._latest_obs_time: dict[str, float] = {s.key: 0.0 for s in self.obs_specs}
        self._ready_events = {s.key: threading.Event() for s in self.obs_specs}

        # --- Dynamically wire up subscriptions ---
        for spec in self.obs_specs:
            if spec.codec not in OBS_CODECS:
                raise ValueError(
                    f"No codec registered for '{spec.codec}' (obs key '{spec.key}')"
                )
            msg_cls = get_message(spec.msg_type)
            self.create_subscription(
                msg_cls, spec.topic, self._make_obs_callback(spec), qos
            )
            self.get_logger().info(
                f"Subscribed: {spec.topic} ({spec.msg_type}) -> '{spec.key}'"
            )

        # --- Dynamically wire up publishers ---
        self._action_publishers: dict[str, Any] = {}
        self._action_msg_cls: dict[str, Any] = {}
        for spec in self.action_specs:
            if spec.codec not in ACTION_CODECS:
                raise ValueError(
                    f"No codec registered for '{spec.codec}' (action key '{spec.key}')"
                )
            msg_cls = get_message(spec.msg_type)
            self._action_msg_cls[spec.key] = msg_cls
            self._action_publishers[spec.key] = self.create_publisher(
                msg_cls, spec.topic, qos
            )
            self.get_logger().info(
                f"Publishing: '{spec.key}' -> {spec.topic} ({spec.msg_type})"
            )

        # --- Resolve transforms ---
        transform_names = cfg.get("transforms", [])
        self._transforms = []
        for name in transform_names:
            if name not in TRANSFORMS:
                raise ValueError(f"No transform registered for '{name}'")
            self._transforms.append(TRANSFORMS[name])
            self.get_logger().info(f"Transform: '{name}'")

        self._spin_executor = SingleThreadedExecutor()
        self._spin_executor.add_node(self)
        self._bg_thread = threading.Thread(target=self._spin_executor.spin, daemon=True)
        self._bg_thread.start()

    # -----------------------------------------------------------
    def _make_obs_callback(self, spec: ObsSpec):
        codec = OBS_CODECS[spec.codec]

        def _cb(msg):
            array = codec(msg)
            with self.state_lock:
                self._latest_obs[spec.key] = array
                self._latest_obs_time[spec.key] = (
                    self.get_clock().now().nanoseconds / 1e9
                )
            self._ready_events[spec.key].set()

        return _cb

    # -----------------------------------------------------------
    def reset(self, timeout_sec: float = 30.0) -> dict[str, Optional[torch.Tensor]]:
        """Blocks until every *required* observation has fired at least once."""
        required = [s.key for s in self.obs_specs if s.required]
        self.get_logger().info(f"Waiting for: {required}")
        for key in required:
            if not self._ready_events[key].wait(timeout_sec):
                raise TimeoutError(
                    f"No message received on '{key}' within {timeout_sec}s"
                )
        return self._snapshot()

    def _snapshot(self) -> dict[str, Optional[torch.Tensor]]:
        """Latest tensor per key, or None if missing/stale - the encoder decides
        how to handle a gap (e.g. learned mask token) rather than this layer
        silently feeding it stale data."""
        now = self.get_clock().now().nanoseconds / 1e9
        out = {}
        with self.state_lock:
            for spec in self.obs_specs:
                value = self._latest_obs[spec.key]
                age = now - self._latest_obs_time[spec.key]
                if value is None or age > spec.max_age_sec:
                    out[spec.key] = None
                else:
                    out[spec.key] = torch.as_tensor(
                        value, device=self.device
                    ).unsqueeze(0)
        for fn in self._transforms:
            out = fn(out)
        return out

    def step(self, actions: dict[str, np.ndarray]) -> dict[str, Optional[torch.Tensor]]:
        """Publish a dict of {action_key: array} and return the latest observations."""
        for spec in self.action_specs:
            if spec.key not in actions:
                continue
            encode = ACTION_CODECS[spec.codec]
            msg = encode(actions[spec.key], self._action_msg_cls[spec.key])
            self._action_publishers[spec.key].publish(msg)
        return self._snapshot()

    def close(self):
        self._spin_executor.shutdown()
        self.destroy_node()
        self._bg_thread.join()


if __name__ == "__main__":
    import sys

    rclpy.init()
    env = ROSEnvironment(
        config_path=sys.argv[1] if len(sys.argv) > 1 else "env_config.yaml"
    )
    try:
        obs = env.reset()
        print("Initial obs keys:", list(obs.keys()))
    finally:
        env.close()
        rclpy.shutdown()
