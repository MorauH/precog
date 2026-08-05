"""
World process: ROS bridge + input formatting + control head → action.
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import signal
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from precog.envs import ROSEnvironment
from precog.envs.ros.source_selector import SourceSelector
from precog.messaging import ShmTensorSlot
from precog.model import DEFAULT_CONFIG
from precog.model.config import _env_shapes_from_yaml, resolve_config_dims
from precog.model.control_head import ControlHead
from precog.model.hierarchical_clock import LevelClock
from precog.processes.action_utils import action_from_obs, action_to_dict


def _set_realtime(priority: int, core: Optional[int]) -> None:
    """Set SCHED_FIFO scheduling and optionally pin to a CPU core."""
    try:
        import os

        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)

        if core is not None:
            os.sched_setaffinity(0, {core})

        print(
            f"[L0] Real-time scheduling: SCHED_FIFO prio={priority}"
            + (f" core={core}" if core is not None else "")
        )
    except PermissionError:
        print("[L0] WARNING: Cannot set SCHED_FIFO -- run with CAP_SYS_NICE or as root")
    except Exception as e:
        print(f"[L0] WARNING: Scheduling config failed: {e}")


def _format_inputs(
    obs_dict: Dict[str, torch.Tensor],
    prev_action: Optional[torch.Tensor],
    observation_keys: List[str],
    observation_shapes: Dict[str, tuple],
    action_scales: torch.Tensor,
    control_dim: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Concatenate observation values and optional prev_action into a flat level-0 input."""
    parts = []
    for key in observation_keys:
        value = obs_dict.get(key)
        if value is None:
            shape = observation_shapes.get(key, (1, 0))
            value = torch.zeros(batch_size, shape[-1], device=device)
        parts.append(value)
    if prev_action is not None:
        parts.append(prev_action / action_scales)
    else:
        parts.append(torch.zeros(batch_size, control_dim, device=device))
    return torch.cat(parts, dim=-1)


def run_world(
    stop_event: multiprocessing.Event,
    *,
    clock: LevelClock,
    upward_writer: ShmTensorSlot,
    downward_reader: ShmTensorSlot,
    env_config_path: str = "./src/precog/envs/ros/env_config.yaml",
    num_levels: int = 0,
    control_repr_reader: Optional[ShmTensorSlot] = None,
    headless: bool = True,
    dashboard_port: int = 8080,
    rt_priority: int = 50,
    rt_core: Optional[int] = None,
):
    if rt_priority > 0:
        _set_realtime(rt_priority, rt_core)

    import rclpy, yaml

    rclpy.init()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    device = "cpu"
    with open(env_config_path) as f:
        env_cfg = yaml.safe_load(f)
    env_shapes = _env_shapes_from_yaml(env_cfg)
    config = resolve_config_dims(DEFAULT_CONFIG, env_shapes)

    env = ROSEnvironment(config_path=env_config_path, device=device)
    action_keys = [s.key for s in env.action_specs]

    source_selector = SourceSelector()
    source_selector.blend_ratio_steer = 0.0
    source_selector.blend_ratio_acc = 0.0

    ctrl_cfg = config.control_head
    control_level_idx = config.control_level_idx
    control_input_dim = config.level_configs[control_level_idx].d_representation
    control_head = ControlHead(
        input_dim=control_input_dim,
        hidden_dims=ctrl_cfg.hidden_dims,
        output_dim=ctrl_cfg.output_dim,
        output_scales=ctrl_cfg.output_scales or None,
    ).to(device)
    control_head.train()

    ctrl_optimizer = torch.optim.AdamW(control_head.parameters(), lr=1e-3)

    if control_head.output_scales is not None:
        action_scales = control_head.output_scales.to(device)
    else:
        action_scales = torch.ones(config.control_dim, device=device)

    dashboard = None
    if not headless:
        try:
            from precog.dashboard import DashboardServer

            dashboard = DashboardServer(port=dashboard_port)
            dashboard.start()
            if num_levels > 0:
                dashboard.watch_level_logs(num_levels)
        except Exception:
            pass

    obs = env.reset()
    prev_action = torch.zeros(1, config.control_dim, device=device)

    tick_count = 0
    blend_steer = 0.0
    blend_acc = 0.0

    _last_tick_time = time.monotonic()
    _hz = 0.0
    _last_gc_collect = 0

    gc.disable()

    try:
        while rclpy.ok() and not stop_event.is_set():
            if dashboard is not None:
                ctrl = dashboard.controls
                new_steer = ctrl.blend_steer_safe
                new_acc = ctrl.blend_acc_safe
                if abs(new_steer - blend_steer) > 1e-6:
                    blend_steer = new_steer
                    source_selector.blend_ratio_steer = blend_steer
                if abs(new_acc - blend_acc) > 1e-6:
                    blend_acc = new_acc
                    source_selector.blend_ratio_acc = blend_acc

            expert_action = action_from_obs(obs, action_keys, device)

            if source_selector.last_output_valid:
                prev_action = torch.tensor(
                    [[source_selector.last_steer, source_selector.last_acc]],
                    dtype=torch.float32,
                    device=device,
                )
            elif expert_action is not None:
                prev_action = expert_action
            else:
                prev_action = torch.zeros(
                    1, config.control_dim, dtype=torch.float32, device=device
                )

            if downward_reader is not None:
                raw = downward_reader.read(timeout_us=5000)
                if raw is not None:
                    signal_down = raw.to(torch.device(device))

            clock.tick()
            current_tick = clock.tick_count
            sim_time = clock.sim_time

            formatted_input = _format_inputs(
                obs,
                prev_action,
                config.observation_keys,
                config.observation_shapes,
                action_scales,
                config.control_dim,
                1,
                device,
            )

            if not torch.isnan(formatted_input).any():
                upward_writer.write(formatted_input)

            raw_top = downward_reader.read(timeout_us=5000)
            if raw_top is not None:
                z_ctrl = raw_top.to(device)
                action = control_head(z_ctrl.unsqueeze(1))[:, -1]

                if expert_action is not None and (
                    blend_steer < 0.999 or blend_acc < 0.999
                ):
                    blend_weight = 1.0 - (blend_steer + blend_acc) / 2.0
                    loss_ctrl = blend_weight * torch.nn.functional.mse_loss(
                        action, expert_action
                    )
                    ctrl_optimizer.zero_grad()
                    loss_ctrl.backward()
                    ctrl_optimizer.step()
            else:
                action = torch.zeros(1, config.control_dim, device=device)

            act_dict = action_to_dict(action, action_keys)

            obs = env.step(act_dict)

            tick_count += 1

            now = time.monotonic()
            _hz = 0.99 * _hz + 0.01 / max(now - _last_tick_time, 1e-6)
            _last_tick_time = now

            if tick_count - _last_gc_collect > 20_000:
                gc.collect()
                _last_gc_collect = tick_count

            if dashboard is not None and tick_count % 10 == 0:
                ma = action[0].tolist()
                pa = prev_action[0].tolist()
                ea = (
                    expert_action[0].tolist()
                    if expert_action is not None
                    else [None] * len(action_keys)
                )
                dashboard.update(
                    {
                        "tick": tick_count,
                        "tick_rate": _hz,
                        "blend_ratio_steer": dashboard.controls.blend_steer_safe,
                        "blend_ratio_acc": dashboard.controls.blend_acc_safe,
                        "action": ma,
                        "prev_action": pa,
                        "expert_action": ea,
                        "source_selector": source_selector.snapshot(),
                        "surprise": [],
                    }
                )

            if tick_count % 200 == 0:
                ma = action[0].tolist()
                print(f"[World] tick={tick_count}  action=[{ma[0]:.4f}, {ma[1]:.4f}]")

            clock.sleep_until_next_tick()

    except KeyboardInterrupt:
        pass
    finally:
        print("[World] Shutting down")
        if dashboard is not None:
            dashboard.stop()
        source_selector.close()
        env.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass
