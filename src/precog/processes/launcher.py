"""
Launcher: orchestrates spawning of all processes and coordinates shutdown.

Spawns:
  - World process (interface with env & calc control outputs)
  - Level processes (one per level, with inter-level channels)
  - Dashboard (thread in L0 or standalone)

Channel inventory (total: 2N-2):
  - Upward slots (N-1): Level L → Level L+1  (z_L)
  - Downward slots (N-1): Level L+1 → Level L (pred_from_above)
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from precog.messaging import ShmBeat, ShmTensorSlot
from precog.model import DEFAULT_CONFIG
from precog.model.config import _env_shapes_from_yaml, resolve_config_dims

ENV_CONFIG_PATH = "./src/precog/envs/ros/env_config.yaml"


def _worker_world(
    stop_event,
    headless,
    dashboard_port,
    beat_name,
    num_levels,
    base_frequency,
    upward_writer_proxy,
    downward_reader_proxy,
    rt_priority,
    rt_core,
    env_config_path="",
):
    """Wrapper that catches exceptions in the world worker."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    from precog.processes.world import run_world
    from precog.messaging.clock import ShmBeat
    from precog.model.hierarchical_clock import LevelClock

    beat = ShmBeat(beat_name, num_levels, create=False, base_frequency=base_frequency)
    clock = LevelClock.writer(beat)

    upward_writer = (
        _ShmWriterHandle(*upward_writer_proxy).writer()
        if upward_writer_proxy is not None
        else None
    )
    downward_reader = (
        _ShmReaderHandle(*downward_reader_proxy).reader()
        if downward_reader_proxy is not None
        else None
    )

    try:
        run_world(
            stop_event,
            clock=clock,
            upward_writer=upward_writer,
            downward_reader=downward_reader,
            env_config_path=env_config_path,
            num_levels=num_levels,
            headless=headless,
            dashboard_port=dashboard_port,
            rt_priority=rt_priority,
            rt_core=rt_core,
        )
    except Exception as e:
        print(f"[Launcher] World process crashed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
    finally:
        if upward_writer is not None:
            upward_writer.close()
        if downward_reader is not None:
            downward_reader.close()
        beat.close()


def _worker_level_n(
    stop_event,
    level_idx,
    config_dict,
    d_below,
    d_above,
    frequency,
    beat_name,
    num_levels,
    base_frequency,
    upward_reader,
    upward_writer,
    downward_reader,
    downward_writer,
    device,
):
    """Wrapper that catches exceptions in an level worker."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    from precog.processes.level_n import run_level_n
    from precog.model.config import PCLevelConfig, SSMConfig
    from precog.messaging.clock import ShmBeat
    from precog.model.hierarchical_clock import LevelClock

    beat = ShmBeat(beat_name, num_levels, create=False, base_frequency=base_frequency)
    divisor = round(base_frequency / frequency)
    clock = LevelClock.reader(beat, divisor=divisor, level_idx=level_idx)

    cfg_dict_clean = config_dict.copy()
    ssm_dict = cfg_dict_clean.pop("ssm", {})
    cfg = PCLevelConfig(ssm=SSMConfig(**ssm_dict), **cfg_dict_clean)

    kwargs: Dict[str, Any] = {
        "stop_event": stop_event,
        "level_idx": level_idx,
        "config": cfg,
        "d_below": d_below,
        "d_above": d_above,
        "clock": clock,
        "frequency": frequency,
        "device": device,
        "obj_key": cfg.objective_observable_key,
        "obj_target": cfg.objective_target_value,
        "lr": 1e-3,
    }

    _slots: list[ShmTensorSlot] = []
    if upward_reader is not None:
        s = _ShmReaderHandle(*upward_reader).reader()
        kwargs["upward_reader"] = s
        _slots.append(s)
    if upward_writer is not None:
        s = _ShmWriterHandle(*upward_writer).writer()
        kwargs["upward_writer"] = s
        _slots.append(s)
    if downward_reader is not None:
        s = _ShmReaderHandle(*downward_reader).reader()
        kwargs["downward_reader"] = s
        _slots.append(s)
    if downward_writer is not None:
        s = _ShmWriterHandle(*downward_writer).writer()
        kwargs["downward_writer"] = s
        _slots.append(s)

    try:
        run_level_n(**kwargs)
    except Exception as e:
        print(f"[Launcher] Level {level_idx} process crashed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
    finally:
        for s in _slots:
            s.close()
        beat.close()


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
        num_levels: int = 3,
        *,
        device: Optional[str] = None,
        headless: bool = False,
        dashboard_port: int = 8080,
        level_frequencies: Optional[List[float]] = None,
        rt_priority: int = 0,
        rt_core: Optional[int] = None,
    ):
        self._num_levels = num_levels
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._headless = headless
        self._dashboard_port = dashboard_port
        self._level_frequencies = level_frequencies
        self._rt_priority = rt_priority
        self._rt_core = rt_core

        self._stop_event: Optional[multiprocessing.Event] = None
        self._world_process: Optional[multiprocessing.Process] = None
        self._level_processes: List[multiprocessing.Process] = []
        self._level_kwargs: List[Dict[str, Any]] = []
        self._shutting_down = False

        self._upward_slots: List[ShmTensorSlot] = []
        self._downward_slots: List[ShmTensorSlot] = []

        self._config = self._resolve_config()

    def _resolve_config(self):
        with open(ENV_CONFIG_PATH) as f:
            env_cfg = yaml.safe_load(f)
        env_shapes = _env_shapes_from_yaml(env_cfg)

        raw_level_devices = env_cfg.get("level_devices", {})
        self._level_devices: Dict[int, str] = {
            int(k): str(v) for k, v in raw_level_devices.items()
        }

        return resolve_config_dims(DEFAULT_CONFIG, env_shapes)

    def _auto_frequencies(self) -> List[float]:
        if self._level_frequencies:
            return self._level_frequencies
        base = 200.0
        return [base / (2**i) for i in range(self._num_levels)]

    def start(self) -> None:
        """Spawn all processes and begin supervision."""
        ctx = multiprocessing.get_context("spawn")

        self._stop_event = ctx.Event()
        freqs = self._auto_frequencies()

        self._create_channels(ctx)
        base_freq = max(freqs)
        self._beat = ShmBeat(
            "main", self._num_levels, create=True, base_frequency=base_freq
        )
        self._spawn_world(ctx, base_freq)
        self._spawn_levels(ctx, freqs)

        self._supervise()

    def _create_channels(self, ctx) -> None:
        """Create all inter-process communication channels."""
        config = self._config

        # Slot 0: world ↔ L0 (sized by raw input dim, and L0's repr dim)
        up0 = ShmTensorSlot("up_world_to_L0", (1, config.d_input))
        self._upward_slots.append(up0)
        down0 = ShmTensorSlot(
            "down_L0_to_world", (1, config.level_configs[0].d_representation)
        )
        self._downward_slots.append(down0)

        # Slots 1..n-1: between PCLevels
        for i in range(self._num_levels - 1):
            lower_cfg = config.level_configs[i]
            upper_cfg = config.level_configs[i + 1]
            d_repr_lower = lower_cfg.d_representation
            d_repr_upper = upper_cfg.d_representation

            slot_up = ShmTensorSlot(f"up_L{i}_to_L{i + 1}", (1, d_repr_lower))
            self._upward_slots.append(slot_up)

            slot_down = ShmTensorSlot(f"down_L{i + 1}_to_L{i}", (1, d_repr_upper))
            self._downward_slots.append(slot_down)

    def _spawn_levels(self, ctx, freqs: List[float]) -> None:
        config = self._config

        for i in range(0, self._num_levels):
            lvl_cfg = config.level_configs[i]

            d_below = (
                config.d_input
                if i == 0
                else config.level_configs[i - 1].d_representation
            )
            d_above = (
                config.level_configs[i + 1].d_representation
                if i + 1 < self._num_levels
                else None
            )

            upward_reader = (
                self._upward_slots[i].name,
                self._upward_slots[i].shape,
                str(self._upward_slots[i]._dtype),
            )

            upward_writer = (
                (
                    self._upward_slots[i + 1].name,
                    self._upward_slots[i + 1].shape,
                    str(self._upward_slots[i + 1]._dtype),
                )
                if i + 1 < self._num_levels
                else None
            )

            downward_reader = (
                (
                    self._downward_slots[i + 1].name,
                    self._downward_slots[i + 1].shape,
                    str(self._downward_slots[i + 1]._dtype),
                )
                if i + 1 < self._num_levels
                else None
            )

            downward_writer = (
                self._downward_slots[i].name,
                self._downward_slots[i].shape,
                str(self._downward_slots[i]._dtype),
            )

            cfg_dict = {
                "d_representation": lvl_cfg.d_representation,
                "ssm": {
                    "d_state": lvl_cfg.ssm.d_state,
                    "dt_min": lvl_cfg.ssm.dt_min,
                    "dt_max": lvl_cfg.ssm.dt_max,
                },
                "encoder_hidden": lvl_cfg.encoder_hidden,
                "sigreg_tau": lvl_cfg.sigreg_tau,
                "sigreg_var_threshold": lvl_cfg.sigreg_var_threshold,
                "objective_enabled": lvl_cfg.objective_enabled,
                "objective_observable_key": lvl_cfg.objective_observable_key,
                "objective_target_value": lvl_cfg.objective_target_value,
                "objective_ae_weight": lvl_cfg.objective_ae_weight,
                "objective_task_weight": lvl_cfg.objective_task_weight,
                "translator_hidden": lvl_cfg.translator_hidden,
            }

            base_freq = max(freqs)
            kwargs = {
                "stop_event": self._stop_event,
                "level_idx": i,
                "config_dict": cfg_dict,
                "d_below": d_below,
                "d_above": d_above,
                "frequency": freqs[i],
                "beat_name": "main",
                "num_levels": self._num_levels,
                "base_frequency": base_freq,
                "upward_reader": upward_reader,
                "upward_writer": upward_writer,
                "downward_reader": downward_reader,
                "downward_writer": downward_writer,
                "device": self._level_devices.get(i, self._device),
            }

            self._level_kwargs.append(kwargs)

            proc = ctx.Process(
                target=_worker_level_n,
                kwargs=kwargs,
                name=f"level{i}",
                daemon=True,
            )
            proc.start()
            self._level_processes.append(proc)
            print(f"[Launcher] L{i} process started (pid={proc.pid})")

    def _spawn_world(self, ctx, freq: float) -> None:
        config = self._config

        upward_writer_proxy = None
        downward_reader_proxy = None

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

        base_freq = max(self._level_frequencies) if self._level_frequencies else freq
        world_kwargs = {
            "stop_event": self._stop_event,
            "headless": self._headless,
            "dashboard_port": self._dashboard_port,
            "beat_name": "main",
            "num_levels": self._num_levels,
            "base_frequency": base_freq,
            "upward_writer_proxy": upward_writer_proxy,
            "downward_reader_proxy": downward_reader_proxy,
            "rt_priority": self._rt_priority,
            "rt_core": self._rt_core,
            "env_config_path": ENV_CONFIG_PATH,
        }

        self._world_process = ctx.Process(
            target=_worker_world,
            kwargs=world_kwargs,
            name="world",
        )
        self._world_process.start()
        print(f"[Launcher] World process started (pid={self._world_process.pid})")

    def _supervise(self) -> None:
        """Monitor child processes and handle crashes."""

        def _signal_handler(signum, frame):
            print(f"\n[Launcher] Received signal {signum}, initiating shutdown")
            self.shutdown()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            while True:
                if self._shutting_down:
                    break

                if (
                    self._world_process is not None
                    and not self._world_process.is_alive()
                ):
                    print("[Launcher] World process exited, shutting down")
                    self.shutdown()
                    break

                for i, proc in enumerate(self._level_processes):
                    if proc is not None and not proc.is_alive():
                        print(f"[Launcher] L{i} process crashed!")
                        if not self._stop_event.is_set():
                            print(f"[Launcher] Restarting L{i}...")
                            self._restart_level(i)

                time.sleep(0.5)

        except KeyboardInterrupt:
            self.shutdown()

    def _restart_level(self, idx: int) -> None:
        if idx >= len(self._level_kwargs):
            print(f"[Launcher] No saved kwargs for level idx={idx}")
            return
        ctx = multiprocessing.get_context("spawn")
        kwargs = self._level_kwargs[idx]
        level_idx = kwargs["level_idx"]
        proc = ctx.Process(
            target=_worker_level_n,
            kwargs=kwargs,
            name=f"level{level_idx}",
            daemon=True,
        )
        proc.start()
        if idx < len(self._level_processes):
            self._level_processes[idx] = proc
        else:
            self._level_processes.append(proc)
        print(f"[Launcher] L{level_idx} process restarted (pid={proc.pid})")

    def shutdown(self) -> None:
        """Stop all child processes gracefully."""
        if self._shutting_down:
            return
        self._shutting_down = True

        if self._stop_event is not None:
            self._stop_event.set()

        all_procs = [self._world_process] + self._level_processes
        for proc in all_procs:
            if proc is not None and proc.is_alive():
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
                if proc.is_alive() and proc.pid is not None:
                    os.kill(proc.pid, signal.SIGKILL)
                    proc.join(timeout=1.0)

        for slot in self._upward_slots:
            slot.close()
        for slot in self._downward_slots:
            slot.close()

        if hasattr(self, "_beat"):
            self._beat.close()
            self._beat.unlink()

        print("[Launcher] All processes stopped")
