"""
Level 0 process: ROS bridge + Level 0 SSM forward + control head → action.

Runs on CPU at ~200Hz. Never calls backward(). Communicates with the
Learner process via ring buffers (experience) and weight sync channels
(updated model parameters).

This is the only hard-real-time process — it must never be preempted
or blocked by Learner/upper-level processes.
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
from precog.model import (
    DEFAULT_CONFIG,
    ForwardOutput,
    HierarchicalPCWorldModel,
    PerLevelSnapshot,
)
from precog.model.config import _env_shapes_from_yaml, resolve_config_dims
from precog.messaging import ShmRingBuffer, WeightSync

ENV_CONFIG_PATH = "./src/precog/envs/ros/env_config.yaml"


def _action_to_dict(tensor: torch.Tensor, keys: list[str]) -> dict[str, np.ndarray]:
    values = tensor[0].detach().cpu().numpy()
    if values.ndim == 0:
        values = np.array([values.item()])
    return {k: values[i].item() for i, k in enumerate(keys)}


def _action_from_obs(
    obs: dict, action_keys: list[str], device: torch.device
) -> Optional[torch.Tensor]:
    values = []
    for k in action_keys:
        v = obs.get(f"expert_{k}")
        if v is None:
            return None
        val = v.squeeze()
        if val.ndim == 0:
            val = val.unsqueeze(-1)
        values.append(val)
    return torch.stack(values, dim=-1).to(device)


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
        print("[L0] WARNING: Cannot set SCHED_FIFO — run with CAP_SYS_NICE or as root")
    except Exception as e:
        print(f"[L0] WARNING: Scheduling config failed: {e}")


def run_level0(
    stop_event: multiprocessing.Event,
    experience_rings: List[ShmRingBuffer],
    weight_syncs: List[WeightSync],
    *,
    headless: bool = True,
    dashboard_port: int = 8080,
    level_frequencies: Optional[List[float]] = None,
    upward_writer_proxy: Optional[tuple] = None,
    downward_reader_proxy: Optional[tuple] = None,
    rt_priority: int = 50,
    rt_core: Optional[int] = None,
):
    """Entry point for the Level 0 process.

    Args:
        stop_event: Set by launcher to request clean shutdown.
        experience_rings: One ShmRingBuffer per level for experience → Learner.
        weight_syncs: One WeightSync per level for Learner → params.
        headless: If False, start the web dashboard in a thread.
        dashboard_port: Port for the dashboard server.
        level_frequencies: Hz for each level (default: [200, 100]).
        upward_writer_proxy: (name, shape, dtype_str) tuple for sending z0 to L1.
        downward_reader_proxy: (name, shape, dtype_str) tuple for reading
            pred_from_above from L1.
        rt_priority: SCHED_FIFO priority (1–99). Higher = higher priority.
            Set to 0 to skip scheduling change.
        rt_core: CPU core to pin L0 to (0-indexed). None = no pinning.
    """
    if rt_priority > 0:
        _set_realtime(rt_priority, rt_core)

    import rclpy
    import yaml

    rclpy.init()

    device = "cpu"

    with open(ENV_CONFIG_PATH) as f:
        env_cfg = yaml.safe_load(f)
    env_shapes = _env_shapes_from_yaml(env_cfg)
    config = resolve_config_dims(DEFAULT_CONFIG, env_shapes)

    env = ROSEnvironment(config_path=ENV_CONFIG_PATH, device=device)
    action_keys = [s.key for s in env.action_specs]

    source_selector = SourceSelector()
    source_selector.blend_ratio_steer = 0.0
    source_selector.blend_ratio_acc = 0.0

    model = HierarchicalPCWorldModel(config).to(device)

    if level_frequencies is None:
        level_frequencies = [200, 100]

    runner = model.build_runner(
        level_frequencies=level_frequencies,
        time_scale=1.0,
        batch_size=1,
        device=device,
        online_learning=False,
    )

    dashboard = None
    if not headless:
        try:
            from precog.dashboard import DashboardServer

            dashboard = DashboardServer(port=dashboard_port)
            dashboard.start()
        except Exception:
            pass

    obs = env.reset()

    upward_writer = None
    if upward_writer_proxy is not None:
        from precog.messaging import ShmTensorSlot

        name, shape, dtype_str = upward_writer_proxy
        dtype = np.dtype(dtype_str)
        upward_writer = ShmTensorSlot.attach(name, shape, dtype)

    downward_reader = None
    if downward_reader_proxy is not None:
        from precog.messaging import ShmTensorSlot

        name, shape, dtype_str = downward_reader_proxy
        dtype = np.dtype(dtype_str)
        downward_reader = ShmTensorSlot.attach(name, shape, dtype)

    print("[L0] Starting with initial weights; weight sync is async")

    tick_count = 0
    blend_steer = 0.0
    blend_acc = 0.0

    _last_tick_time = time.monotonic()
    _hz = 0.0

    weight_versions = {ws.level_idx: 0 for ws in weight_syncs}
    _last_gc_collect = 0
    _exp_write_interval = 2  # write experience every 2 ticks (~100Hz at 200Hz)

    runner.clock.reset()

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

            expert_action = _action_from_obs(obs, action_keys, device)

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

            # Read a_t (top-down prediction) from L1 via downward slot (strictly non-blocking)
            a_t = None
            if downward_reader is not None:
                raw = downward_reader.read(timeout_us=0)
                if raw is not None:
                    a_t = raw.to(torch.device(device))

            # Level 0 forward pass only (no upper levels)
            fwd: ForwardOutput = runner.forward_level0(
                obs,
                prev_action,
                a_t=a_t,
            )

            # Send z0 upward to L1
            if upward_writer is not None and fwd.level_states:
                z0 = fwd.level_states[0].last_z
                if z0 is not None:
                    upward_writer.write(z0)

            # Batch experience writes — not every tick, write every N ticks
            if fwd.level_snapshots and tick_count % _exp_write_interval == 0:
                experience_rings[0].write(fwd.level_snapshots[0])

            # Version-guarded weight sync — L0 only syncs its own level
            # (L1 weights are synced by the L1 process)
            if tick_count % 4 == 0:
                ws = weight_syncs[0]
                new_ver = ws.read_latest_if_new(
                    model.levels[0],
                    weight_versions[0],
                    device=torch.device(device),
                )
                if new_ver != weight_versions[0]:
                    weight_versions[0] = new_ver
                    ws.read_latest(
                        model.control_head,
                        device=torch.device(device),
                    )
                    print(f"[L0] Updated weights (v{new_ver})")

            now = time.monotonic()
            _hz = 0.9 * _hz + 0.1 / max(now - _last_tick_time, 1e-6)
            _last_tick_time = now

            act_dict = _action_to_dict(fwd.action, action_keys)
            obs = env.step(act_dict)
            tick_count += 1

            if tick_count - _last_gc_collect > 20000:
                gc.collect()
                _last_gc_collect = tick_count

            if dashboard is not None and tick_count % 10 == 0:
                ma = fwd.action[0].tolist()
                pa = prev_action[0].tolist()
                ea = (
                    expert_action[0].tolist()
                    if expert_action is not None
                    else [None] * len(action_keys)
                )
                surprise = []
                for ls in fwd.level_states:
                    if ls.last_pred_error is not None:
                        surprise.append(ls.last_pred_error.pow(2).mean().item())
                    else:
                        surprise.append(0.0)
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
                        "surprise": surprise,
                    }
                )

            if tick_count % 200 == 0:
                ma = fwd.action[0].tolist()
                print(f"[L0] tick={tick_count}  action=[{ma[0]:.4f}, {ma[1]:.4f}]")

            runner.clock.sleep_until_next_tick()

    except KeyboardInterrupt:
        pass
    finally:
        print("[L0] Shutting down")
        if dashboard is not None:
            dashboard.stop()
        source_selector.close()
        env.close()
        rclpy.shutdown()
