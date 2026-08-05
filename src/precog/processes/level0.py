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
    PerLevelSnapshot,
)
from precog.model.config import _env_shapes_from_yaml, resolve_config_dims
from precog.model.control_head import ControlHead
from precog.model.hierarchical_clock import ClockConfig, HierarchicalClock
from precog.model.level_state import LevelState
from precog.model.pc_level_jepa import PCLevel
from precog.messaging import ShmRingBuffer, WeightSync
from precog.processes.action_utils import action_from_obs, action_to_dict

ENV_CONFIG_PATH = "./src/precog/envs/ros/env_config.yaml"


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


def _encode_level0(
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
        rt_priority: SCHED_FIFO priority (1-99). Higher = higher priority.
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

    # ------------------------------------------------------------------ #
    # Direct component construction (no HierarchicalPCWorldModel)
    # ------------------------------------------------------------------ #

    level = PCLevel(
        d_below=config.d_input,
        d_above=config.level_configs[0].d_representation
        if len(config.level_configs) > 1
        else None,
        config=config.level_configs[0],
    ).to(device)

    ctrl_cfg = config.control_head
    control_level_idx = config.control_level_idx
    control_input_dim = config.level_configs[control_level_idx].d_representation
    control_head = ControlHead(
        input_dim=control_input_dim,
        hidden_dims=ctrl_cfg.hidden_dims,
        output_dim=ctrl_cfg.output_dim,
        output_scales=ctrl_cfg.output_scales or None,
    ).to(device)

    if level_frequencies is None:
        level_frequencies = [200, 100]

    clock = HierarchicalClock(
        ClockConfig(level_frequencies=level_frequencies, time_scale=1.0)
    )

    d_state = level.ssm.d_state
    d_repr = level.d_repr
    level_state = LevelState.init(
        level_idx=0, d_state=d_state, d_repr=d_repr, batch_size=1, device=device
    )

    if control_head.output_scales is not None:
        action_scales = control_head.output_scales.to(device)
    else:
        action_scales = torch.ones(config.control_dim, device=device)

    # ------------------------------------------------------------------ #

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

    clock.reset()

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

            # Read a_t (top-down prediction) from L1 via downward slot (strictly non-blocking)
            a_t = None
            if downward_reader is not None:
                raw = downward_reader.read(timeout_us=0)
                if raw is not None:
                    a_t = raw.to(torch.device(device))

            clock.tick()
            current_tick = clock._tick_count
            sim_time = clock.sim_time()

            # Encode observations
            z_level0 = _encode_level0(
                obs,
                prev_action,
                config.observation_keys,
                config.observation_shapes,
                action_scales,
                config.control_dim,
                1,
                device,
            )

            # NaN guard
            if torch.isnan(z_level0).any():
                fwd = ForwardOutput(
                    action=torch.zeros(1, config.control_dim, device=device),
                    level_snapshots=[],
                    updated_levels=[],
                    level_states=[level_state],
                    sim_time=sim_time,
                    total_surprise=None,
                    tick_count=current_tick,
                )
            else:
                h_in = level_state.hidden.detach()
                z_t_pred = (
                    level_state.last_z_pred.detach()
                    if level_state.last_z_pred is not None
                    else None
                )

                a_t_single = a_t.detach().to(device) if a_t is not None else None

                z_t_out, z_next_pred, h_new, z_delta, x_star = level.step(
                    z_level0, h_in, a_t_single, z_t_pred, None
                )

                surprise_i = z_delta.pow(2).mean()
                sigreg_mean = level.sigreg._mean.detach().clone()
                sigreg_outer = level.sigreg._outer.detach().clone()

                snapshot = PerLevelSnapshot(
                    level_idx=0,
                    signal_from_below=z_level0.detach(),
                    h_init=h_in.detach(),
                    a_t=a_t_single.detach() if a_t_single is not None else None,
                    z_t_pred=z_t_pred.detach() if z_t_pred is not None else None,
                    task_target=None,
                    x_actual=None,
                    prev_action=prev_action.detach(),
                    sigreg_mean=sigreg_mean,
                    sigreg_outer=sigreg_outer,
                    is_single_step=True,
                )

                level_state.update(
                    hidden=h_new.detach(),
                    z=z_t_out.detach(),
                    z_pred=z_next_pred.detach(),
                    pred_error=z_delta.detach(),
                    tick=current_tick,
                    sim_time=sim_time,
                )

                action = control_head(z_t_out.unsqueeze(1))[:, -1]

                fwd = ForwardOutput(
                    action=action,
                    level_snapshots=[snapshot],
                    updated_levels=[0],
                    level_states=[level_state],
                    sim_time=sim_time,
                    total_surprise=surprise_i,
                    tick_count=current_tick,
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
            if tick_count % 4 == 0:
                ws = weight_syncs[0]
                new_ver = ws.read_latest_if_new(
                    level,
                    weight_versions[0],
                    device=torch.device(device),
                )
                if new_ver != weight_versions[0]:
                    weight_versions[0] = new_ver
                    ws.read_latest(
                        control_head,
                        device=torch.device(device),
                    )
                    print(f"[L0] Updated weights (v{new_ver})")

            now = time.monotonic()
            _hz = 0.9 * _hz + 0.1 / max(now - _last_tick_time, 1e-6)
            _last_tick_time = now

            act_dict = action_to_dict(fwd.action, action_keys)
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

            clock.sleep_until_next_tick()

    except KeyboardInterrupt:
        pass
    finally:
        print("[L0] Shutting down")
        if dashboard is not None:
            dashboard.stop()
        source_selector.close()
        env.close()
        rclpy.shutdown()
