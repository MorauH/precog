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

from precog.model import (
    DEFAULT_CONFIG,
    ForwardOutput,
    HierarchicalPCWorldModel,
    PerLevelSnapshot,
    replay_learn_level,
)
from precog.model.config import ModelConfig
from precog.messaging import WeightSync


def run_learner(
    stop_event: multiprocessing.Event,
    experience_queues: List[multiprocessing.Queue],
    weight_syncs: List[WeightSync],
    *,
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
        experience_queues: One Queue per level for receiving experience.
        weight_syncs: One WeightSync per level for publishing parameters.
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
        torch.cuda.set_device(device_torch)
        print(f"[Learner] CUDA device: {torch.cuda.get_device_name(device_torch)}")

    config = DEFAULT_CONFIG

    learner_model = HierarchicalPCWorldModel(config).to(device_torch)
    learner_model.train()

    ctrl_idx = config.control_level_idx

    optimizers: List[torch.optim.Optimizer] = []
    for i, level in enumerate(learner_model.levels):
        param_groups = [{"params": level.parameters()}]
        if i == ctrl_idx:
            param_groups.append({"params": learner_model.control_head.parameters()})
        optimizers.append(torch.optim.AdamW(param_groups, lr=learning_rate))

    weight_versions: Dict[int, int] = {}
    last_sync: Dict[int, float] = {}
    last_log: Dict[int, float] = {}
    staleness_samples: Dict[int, List[int]] = {}

    for ws in weight_syncs:
        weight_versions[ws.level_idx] = 0
        last_sync[ws.level_idx] = time.monotonic()
        last_log[ws.level_idx] = 0.0
        staleness_samples[ws.level_idx] = []

    num_levels = len(learner_model.levels)

    print(f"[Learner] Started on {device}, {num_levels} levels, "
          f"lr={learning_rate}, staleness_max={staleness_max}, "
          f"batch_max={batch_max_size}")

    while not stop_event.is_set():
        any_processed = False

        for level_idx in range(num_levels):
            queue = experience_queues[level_idx]
            level = learner_model.levels[level_idx]
            optimizer = optimizers[level_idx]

            snapshots: List[PerLevelSnapshot] = []
            drain_count = 0
            skipped_stale = 0

            while not queue.empty() and len(snapshots) < batch_max_size:
                try:
                    snap = queue.get_nowait()
                    drain_count += 1

                    exp_version = getattr(snap, '_weight_version', 0)
                    gap = weight_versions.get(level_idx, 0) - exp_version

                    if staleness_max > 0 and gap > staleness_max:
                        skipped_stale += 1
                        continue

                    staleness_samples[level_idx].append(gap)
                    snapshots.append(snap)
                except Exception:
                    break

            if skipped_stale > 0:
                print(
                    f"[Learner] L{level_idx}: skipped {skipped_stale} stale "
                    f"snapshots (staleness_max={staleness_max})"
                )

            if not snapshots:
                continue

            ctrl_head = (
                learner_model.control_head
                if level_idx == ctrl_idx
                else None
            )

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

                if level_idx == ctrl_idx:
                    weights = learner_model.get_weights(level_idx)
                else:
                    weights = learner_model.get_weights(level_idx)

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
                        f"loss={total_loss:.6f}, dt={dt*1000:.1f}ms, "
                        f"v={weight_versions[level_idx]}, "
                        f"staleness_avg={avg_gap:.1f}"
                    )

        if not any_processed:
            time.sleep(0.001)

    print("[Learner] Shutting down")
