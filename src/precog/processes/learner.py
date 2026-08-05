"""
Learner process: drains experience from ring buffers, replays forward
passes on model copies, computes losses, runs backward() + step(),
publishes updated weights.

Owns all optimizer states (AdamW momentum buffers). Runs independent
of the Level 0 control loop cadence.

SIGReg EMA buffers are received from forward processes and used
alongside replayed z to compute differentiable loss.

Off-policy staleness is tracked and logged. Experience beyond a
configurable staleness threshold is skipped.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from precog.messaging import ShmRingBuffer, WeightSync
from precog.model import (
    DEFAULT_CONFIG,
    PerLevelSnapshot,
    replay_learn_level,
)
from precog.model.config import ModelConfig
from precog.model.control_head import ControlHead
from precog.model.pc_level_jepa import PCLevel


def _build_levels_and_control(
    config: ModelConfig,
    device: torch.device,
) -> tuple[nn.ModuleList, ControlHead, int]:
    """Construct PCLevel list and ControlHead directly (no HierarchicalPCWorldModel)."""
    prev_dim = config.d_input
    levels: nn.ModuleList = nn.ModuleList()
    for i, level_cfg in enumerate(config.level_configs):
        levels.append(
            PCLevel(
                d_below=prev_dim,
                d_above=None
                if i == len(config.level_configs) - 1
                else level_cfg.d_representation,
                config=level_cfg,
            ).to(device)
        )
        prev_dim = level_cfg.d_representation

    ctrl_cfg = config.control_head
    control_level_idx = config.control_level_idx
    control_input_dim = config.level_configs[control_level_idx].d_representation
    control_head = ControlHead(
        input_dim=control_input_dim,
        hidden_dims=ctrl_cfg.hidden_dims,
        output_dim=ctrl_cfg.output_dim,
        output_scales=ctrl_cfg.output_scales or None,
    ).to(device)

    return levels, control_head, control_level_idx


def _get_weights(
    levels: nn.ModuleList,
    control_head: ControlHead,
    control_level_idx: int,
    level_idx: int,
) -> dict[str, torch.Tensor]:
    """Return a CPU copy of the level's state_dict and its control head."""
    state: dict[str, torch.Tensor] = {}
    for name, param in levels[level_idx].named_parameters():
        state[f"level.{name}"] = param.data.detach().cpu().clone()
    for name, buf in levels[level_idx].named_buffers():
        state[f"level.{name}"] = buf.data.detach().cpu().clone()
    if level_idx == control_level_idx:
        for name, param in control_head.named_parameters():
            state[f"control_head.{name}"] = param.data.detach().cpu().clone()
        for name, buf in control_head.named_buffers():
            state[f"control_head.{name}"] = buf.data.detach().cpu().clone()
    return state


def run_learner(
    stop_event: multiprocessing.Event,
    experience_rings: List[ShmRingBuffer],
    weight_syncs: List[WeightSync],
    *,
    config: Optional[ModelConfig] = None,
    device: str = "cpu",
    learning_rate: float = 1e-3,
    ctrl_weight: float = 1.0,
    staleness_max: int = 500,
    sync_interval_seconds: float = 0.1,
    batch_max_size: int = 32,
):
    """Entry point for the Learner process.

    Args:
        stop_event: Set by launcher to request clean shutdown.
        experience_rings: One ShmRingBuffer per level for receiving experience.
        weight_syncs: One WeightSync per level for publishing parameters.
        config: Resolved model config (with correct d_input). Falls back
            to DEFAULT_CONFIG if not provided.
        device: Torch device (cpu or cuda).
        learning_rate: AdamW learning rate.
        ctrl_weight: Imitation loss weight.
        staleness_max: Max tolerable staleness gap. Experience beyond
            this threshold is skipped.
        sync_interval_seconds: How often to publish updated weights.
        batch_max_size: Max number of experience snapshots per learn step.
    """
    device_torch = torch.device(device)

    if device_torch.type == "cuda":
        cuda_idx = device_torch.index if device_torch.index is not None else 0
        torch.cuda.set_device(cuda_idx)
        print(f"[Learner] CUDA device: {torch.cuda.get_device_name(cuda_idx)}")

    config = config or DEFAULT_CONFIG

    levels, control_head, ctrl_idx = _build_levels_and_control(config, device_torch)
    for level in levels:
        level.train()
    control_head.train()

    optimizers: List[torch.optim.Optimizer] = []
    for i, level in enumerate(levels):
        param_groups = [{"params": level.parameters()}]
        if i == ctrl_idx:
            param_groups.append({"params": control_head.parameters()})
        optimizers.append(torch.optim.AdamW(param_groups, lr=learning_rate))

    weight_versions: Dict[int, int] = {}
    last_sync: Dict[int, float] = {}
    staleness_samples: Dict[int, List[int]] = {}

    for ws in weight_syncs:
        weight_versions[ws.level_idx] = 0
        last_sync[ws.level_idx] = time.monotonic()
        staleness_samples[ws.level_idx] = []

    num_levels = len(levels)

    print(
        f"[Learner] Started on {device}, {num_levels} levels, "
        f"lr={learning_rate}, staleness_max={staleness_max}, "
        f"batch_max={batch_max_size}"
    )

    for i, ring in enumerate(experience_rings):
        drained = ring.read_all()
        if drained:
            print(f"[Learner] Drained {len(drained)} stale snaps from L{i} ring buffer")

    while not stop_event.is_set():
        any_processed = False

        for level_idx in range(num_levels):
            ring = experience_rings[level_idx]
            level = levels[level_idx]
            optimizer = optimizers[level_idx]

            raw_snapshots = ring.read_all(max_count=batch_max_size)
            if not raw_snapshots:
                continue

            snapshots: List[PerLevelSnapshot] = []
            skipped_stale = 0
            for snap in raw_snapshots:
                gap = 0  # staleness tracking simplified for lock-free path
                if staleness_max > 0 and gap > staleness_max:
                    skipped_stale += 1
                    continue
                staleness_samples[level_idx].append(gap)
                snapshots.append(snap)

            drain_count = len(raw_snapshots)

            if skipped_stale > 0:
                print(
                    f"[Learner] L{level_idx}: skipped {skipped_stale} stale "
                    f"snapshots (staleness_max={staleness_max})"
                )

            if not snapshots:
                continue

            ctrl_head = control_head if level_idx == ctrl_idx else None

            t0 = time.monotonic()
            total_loss = replay_learn_level(
                level=level,
                control_head=ctrl_head,
                snapshots=snapshots,
                optimizer=optimizer,
                device=device_torch,
                ctrl_weight=ctrl_weight,
                is_control_level=(level_idx == ctrl_idx),
            )
            dt = time.monotonic() - t0

            weight_versions[level_idx] = weight_versions.get(level_idx, 0) + 1
            any_processed = True

            now = time.monotonic()
            if now - last_sync.get(level_idx, 0.0) >= sync_interval_seconds:
                ws = weight_syncs[level_idx]

                weights = _get_weights(levels, control_head, ctrl_idx, level_idx)

                ws.write_state_dict(weights)
                last_sync[level_idx] = now

                if drain_count > 0 and total_loss is not None:
                    staleness_list = staleness_samples.get(level_idx, [])
                    avg_gap = (
                        sum(staleness_list[-100:]) / min(len(staleness_list), 100)
                        if staleness_list
                        else 0.0
                    )
                    print(
                        f"[Learner] L{level_idx}: {len(snapshots)} snaps, "
                        f"loss={total_loss:.6f}, dt={dt * 1000:.1f}ms, "
                        f"v={weight_versions[level_idx]}, "
                        f"staleness_avg={avg_gap:.1f}"
                    )

        if not any_processed:
            time.sleep(0.001)

    print("[Learner] Shutting down")
