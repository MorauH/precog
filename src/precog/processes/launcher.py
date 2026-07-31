"""
Launcher: orchestrates spawning of all processes, monitors staleness,
handles Learner crash recovery, and coordinates startup/shutdown.

Spawns:
  - Level 0 process (ROS bridge + forward + control head)
  - Level 1+ processes (one per upper level, with inter-level channels)
  - Learner process (replay + backward + step)
  - Dashboard (thread in L0 or standalone)

Channel inventory (total: 4N-2):
  - Upward slots (N-1): Level L → Level L+1  (z_L)
  - Downward slots (N-1): Level L+1 → Level L (pred_from_above)
  - Experience queues (N): Each level → Learner
  - Weight sync channels (N): Learner → Each level

The launcher acts as supervisor:
  - If Learner crashes, L0 continues driving on last-known-good weights.
  - Launcher restarts Learner transparently.
  - Stopping L0 signal-cascades to all child processes.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from precog.messaging import ShmTensorSlot, WeightSync
from precog.model import DEFAULT_CONFIG
from precog.model.config import _env_shapes_from_yaml, resolve_config_dims

ENV_CONFIG_PATH = "./src/precog/envs/ros/env_config.yaml"


def _worker_l0(
    stop_event, experience_queues, weight_syncs,
    headless, dashboard_port, level_frequencies,
    upward_writer_proxy, downward_reader_proxy,
    rt_priority, rt_core,
):
    """Wrapper that catches exceptions in the L0 worker."""
    from precog.processes.level0 import run_level0
    try:
        run_level0(
            stop_event,
            experience_queues,
            weight_syncs,
            headless=headless,
            dashboard_port=dashboard_port,
            level_frequencies=level_frequencies,
            upward_writer_proxy=upward_writer_proxy,
            downward_reader_proxy=downward_reader_proxy,
            rt_priority=rt_priority,
            rt_core=rt_core,
        )
    except Exception as e:
        print(f"[Launcher] L0 process crashed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


def _worker_learner(stop_event, experience_queues, weight_syncs, device, learning_rate,
                    ctrl_weight, staleness_max, sync_interval):
    """Wrapper that catches exceptions in the Learner worker."""
    from precog.processes.learner import run_learner
    try:
        run_learner(
            stop_event,
            experience_queues,
            weight_syncs,
            device=device,
            learning_rate=learning_rate,
            ctrl_weight=ctrl_weight,
            staleness_max=staleness_max,
            sync_interval_seconds=sync_interval,
        )
    except Exception as e:
        print(f"[Launcher] Learner process crashed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


def _worker_level_n(
    stop_event, level_idx, config_dict, d_below, d_above, frequency,
    experience_queue, weight_sync, upward_reader, upward_writer,
    downward_reader, downward_writer, device,
):
    """Wrapper that catches exceptions in an upper-level worker."""
    from precog.processes.level_n import run_level_n
    from precog.model.config import PCLevelConfig

    cfg = PCLevelConfig(**config_dict)

    kwargs: Dict[str, Any] = {
        "stop_event": stop_event,
        "level_idx": level_idx,
        "config": cfg,
        "d_below": d_below,
        "d_above": d_above,
        "frequency": frequency,
        "experience_queue": experience_queue,
        "weight_sync": weight_sync,
        "device": device,
        "obj_key": cfg.objective_observable_key,
        "obj_target": cfg.objective_target_value,
    }

    if upward_reader is not None:
        kwargs["upward_reader"] = _ShmReaderHandle(*upward_reader)
    if upward_writer is not None:
        kwargs["upward_writer"] = _ShmWriterHandle(*upward_writer)
    if downward_reader is not None:
        kwargs["downward_reader"] = _ShmReaderHandle(*downward_reader)
    if downward_writer is not None:
        kwargs["downward_writer"] = _ShmWriterHandle(*downward_writer)

    try:
        run_level_n(**kwargs)
    except Exception as e:
        print(f"[Launcher] Level {level_idx} process crashed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


class _ShmReaderHandle:
    """Serializable handle for attaching to a ShmTensorSlot reader in a child process."""
    def __init__(self, name: str, shape: tuple, dtype_str: str):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype_str)

    def reader(self) -> ShmTensorSlot:
        return ShmTensorSlot.attach(self.name, self.shape, self.dtype)


class _ShmWriterHandle:
    """Serializable handle for attaching to a ShmTensorSlot writer in a child process."""
    def __init__(self, name: str, shape: tuple, dtype_str: str):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype_str)

    def writer(self) -> ShmTensorSlot:
        return ShmTensorSlot.attach(self.name, self.shape, self.dtype)


class Launcher:
    """Orchestrator for the pipelined actor architecture."""

    def __init__(
        self,
        num_levels: int = 2,
        ctrl_level_idx: int = 0,
        *,
        device: str = "cpu",
        headless: bool = True,
        dashboard_port: int = 8080,
        level_frequencies: Optional[List[float]] = None,
        learning_rate: float = 1e-3,
        ctrl_weight: float = 1.0,
        staleness_max: int = 500,
        sync_interval: float = 0.1,
        rt_priority: int = 0,
        rt_core: Optional[int] = None,
    ):
        self._num_levels = num_levels
        self._ctrl_level_idx = ctrl_level_idx
        self._device = device
        self._headless = headless
        self._dashboard_port = dashboard_port
        self._level_frequencies = level_frequencies
        self._learning_rate = learning_rate
        self._ctrl_weight = ctrl_weight
        self._staleness_max = staleness_max
        self._sync_interval = sync_interval
        self._rt_priority = rt_priority
        self._rt_core = rt_core

        self._stop_event: Optional[multiprocessing.Event] = None
        self._l0_process: Optional[multiprocessing.Process] = None
        self._learner_process: Optional[multiprocessing.Process] = None
        self._upper_processes: List[multiprocessing.Process] = []

        self._experience_queues: List[multiprocessing.Queue] = []
        self._weight_syncs: List[WeightSync] = []

        self._upward_slots: List[ShmTensorSlot] = []
        self._downward_slots: List[ShmTensorSlot] = []

        self._config = self._resolve_config()

    def _resolve_config(self):
        with open(ENV_CONFIG_PATH) as f:
            env_cfg = yaml.safe_load(f)
        env_shapes = _env_shapes_from_yaml(env_cfg)
        return resolve_config_dims(DEFAULT_CONFIG, env_shapes)

    def _auto_frequencies(self) -> List[float]:
        if self._level_frequencies:
            return self._level_frequencies
        base = max(200.0, 200.0)
        return [base / (2 ** i) for i in range(self._num_levels)]

    def start(self) -> None:
        """Spawn all processes and begin supervision."""
        ctx = multiprocessing.get_context("spawn")

        self._stop_event = ctx.Event()
        freqs = self._auto_frequencies()

        self._create_channels(ctx)
        self._spawn_learner(ctx)
        self._spawn_upper_levels(ctx, freqs)
        self._spawn_l0(ctx, freqs)

        self._supervise()

    def _create_channels(self, ctx) -> None:
        """Create all inter-process communication channels."""
        config = self._config

        for i in range(self._num_levels):
            self._experience_queues.append(ctx.Queue(maxsize=1024))
            self._weight_syncs.append(WeightSync(f"sync_L{i}", i))

        for i in range(self._num_levels - 1):
            lvl_cfg = config.level_configs[i]
            d_repr = lvl_cfg.d_representation

            slot_up = ShmTensorSlot(f"up_L{i}_to_L{i+1}", (1, d_repr))
            self._upward_slots.append(slot_up)

            slot_down = ShmTensorSlot(f"down_L{i+1}_to_L{i}", (1, d_repr))
            self._downward_slots.append(slot_down)

    def _spawn_learner(self, ctx) -> None:
        learner_kwargs = {
            "stop_event": self._stop_event,
            "experience_queues": self._experience_queues,
            "weight_syncs": self._weight_syncs,
            "device": self._device,
            "learning_rate": self._learning_rate,
            "ctrl_weight": self._ctrl_weight,
            "staleness_max": self._staleness_max,
            "sync_interval": self._sync_interval,
        }

        self._learner_process = ctx.Process(
            target=_worker_learner,
            kwargs=learner_kwargs,
            name="learner",
            daemon=True,
        )
        self._learner_process.start()
        print(f"[Launcher] Learner process started (pid={self._learner_process.pid})")

    def _spawn_upper_levels(self, ctx, freqs: List[float]) -> None:
        config = self._config

        for i in range(1, self._num_levels):
            lvl_cfg = config.level_configs[i]

            d_below = config.level_configs[i - 1].d_representation
            d_above = (
                config.level_configs[i + 1].d_representation
                if i + 1 < self._num_levels
                else None
            )

            upward_reader = None
            upward_writer = None
            downward_reader = None
            downward_writer = None

            if i > 0:
                slot_idx = i - 1
                upward_reader = (
                    self._upward_slots[slot_idx].name,
                    self._upward_slots[slot_idx].shape,
                    str(self._upward_slots[slot_idx]._dtype),
                )

            if i + 1 < self._num_levels:
                slot_idx = i
                upward_writer = (
                    self._upward_slots[slot_idx].name,
                    self._upward_slots[slot_idx].shape,
                    str(self._upward_slots[slot_idx]._dtype),
                )

            if i + 1 < self._num_levels:
                slot_idx = i
                downward_reader = (
                    self._downward_slots[slot_idx].name,
                    self._downward_slots[slot_idx].shape,
                    str(self._downward_slots[slot_idx]._dtype),
                )

            if i == 0:
                slot_idx = 0
            else:
                slot_idx = i - 1

            if i < self._num_levels:
                if slot_idx < len(self._downward_slots):
                    downward_writer = (
                        self._downward_slots[slot_idx].name,
                        self._downward_slots[slot_idx].shape,
                        str(self._downward_slots[slot_idx]._dtype),
                    )

            cfg_dict = {
                "d_representation": lvl_cfg.d_representation,
                "ssm": {
                    "d_state": lvl_cfg.ssm.d_state,
                    "dt_min": lvl_cfg.ssm.dt_min,
                    "dt_max": lvl_cfg.ssm.dt_max,
                },
                "prediction_head_hidden": lvl_cfg.prediction_head_hidden,
                "sigreg_tau": lvl_cfg.sigreg_tau,
                "sigreg_var_threshold": lvl_cfg.sigreg_var_threshold,
                "objective_enabled": lvl_cfg.objective_enabled,
                "objective_observable_key": lvl_cfg.objective_observable_key,
                "objective_target_value": lvl_cfg.objective_target_value,
                "objective_ae_weight": lvl_cfg.objective_ae_weight,
                "objective_task_weight": lvl_cfg.objective_task_weight,
                "translator_hidden": lvl_cfg.translator_hidden,
            }

            kwargs = {
                "stop_event": self._stop_event,
                "level_idx": i,
                "config_dict": cfg_dict,
                "d_below": d_below,
                "d_above": d_above,
                "frequency": freqs[i],
                "experience_queue": self._experience_queues[i],
                "weight_sync": self._weight_syncs[i],
                "upward_reader": upward_reader,
                "upward_writer": upward_writer,
                "downward_reader": downward_reader,
                "downward_writer": downward_writer,
                "device": self._device,
            }

            proc = ctx.Process(
                target=_worker_level_n,
                kwargs=kwargs,
                name=f"level{i}",
                daemon=True,
            )
            proc.start()
            self._upper_processes.append(proc)
            print(f"[Launcher] L{i} process started (pid={proc.pid})")

    def _spawn_l0(self, ctx, freqs: List[float]) -> None:
        config = self._config

        upward_writer_proxy = None
        downward_reader_proxy = None

        if self._num_levels > 1:
            slot_up = self._upward_slots[0]
            upward_writer_proxy = (
                slot_up.name,
                slot_up.shape,
                str(slot_up._dtype),
            )
            slot_down = self._downward_slots[0]
            downward_reader_proxy = (
                slot_down.name,
                slot_down.shape,
                str(slot_down._dtype),
            )

        l0_kwargs = {
            "stop_event": self._stop_event,
            "experience_queues": self._experience_queues,
            "weight_syncs": self._weight_syncs,
            "headless": self._headless,
            "dashboard_port": self._dashboard_port,
            "level_frequencies": freqs,
            "upward_writer_proxy": upward_writer_proxy,
            "downward_reader_proxy": downward_reader_proxy,
            "rt_priority": self._rt_priority,
            "rt_core": self._rt_core,
        }

        self._l0_process = ctx.Process(
            target=_worker_l0,
            kwargs=l0_kwargs,
            name="level0",
        )
        self._l0_process.start()
        print(f"[Launcher] L0 process started (pid={self._l0_process.pid})")

    def _supervise(self) -> None:
        """Monitor child processes and handle crashes."""

        def _signal_handler(signum, frame):
            print(f"\n[Launcher] Received signal {signum}, initiating shutdown")
            self.shutdown()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            while True:
                if self._l0_process is not None and not self._l0_process.is_alive():
                    print("[Launcher] L0 process exited, shutting down")
                    self.shutdown()
                    break

                if (self._learner_process is not None
                        and not self._learner_process.is_alive()):
                    print("[Launcher] Learner process crashed!")
                    if not self._stop_event.is_set():
                        print("[Launcher] Restarting Learner...")
                        self._start_learner()

                for i, proc in enumerate(self._upper_processes):
                    if proc is not None and not proc.is_alive():
                        print(f"[Launcher] L{i+1} process crashed!")
                        if not self._stop_event.is_set():
                            print(f"[Launcher] Restarting L{i+1}...")
                            self._restart_upper_level(i)

                time.sleep(0.5)

        except KeyboardInterrupt:
            self.shutdown()

    def _start_learner(self) -> None:
        """Start or restart the Learner process."""
        ctx = multiprocessing.get_context("spawn")

        learner_kwargs = {
            "stop_event": self._stop_event,
            "experience_queues": self._experience_queues,
            "weight_syncs": self._weight_syncs,
            "device": self._device,
            "learning_rate": self._learning_rate,
            "ctrl_weight": self._ctrl_weight,
            "staleness_max": self._staleness_max,
            "sync_interval": self._sync_interval,
        }

        self._learner_process = ctx.Process(
            target=_worker_learner,
            kwargs=learner_kwargs,
            name="learner",
            daemon=True,
        )
        self._learner_process.start()
        print(f"[Launcher] Learner process restarted (pid={self._learner_process.pid})")

    def _restart_upper_level(self, idx: int) -> None:
        print(f"[Launcher] Upper level restart not yet implemented for idx={idx}")

    def shutdown(self) -> None:
        """Stop all child processes gracefully."""
        if self._stop_event is not None:
            self._stop_event.set()

        all_procs = [self._learner_process, self._l0_process] + self._upper_processes
        for proc in all_procs:
            if proc is not None and proc.is_alive():
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)

        for q in self._experience_queues:
            q.close()

        for slot in self._upward_slots:
            slot.close()
        for slot in self._downward_slots:
            slot.close()

        print("[Launcher] All processes stopped")
