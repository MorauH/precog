"""
Generic upper-level process (L1, L2, ... Ln).

Each upper-level process:
  1. Reads the level below's representation (z_below) from an upward ShmTensorSlot.
  2. Receives top-down predictions from the level above via a downward ShmTensorSlot.
  3. Accumulates z_below values into sequences.
  4. On its own tick cadence, runs the SSM forward pass.
  5. Sends its representation upward and its prediction downward.
  6. Pushes experience to the Learner's ring buffer.
  7. Reads updated weights from the Learner via WeightSync.

Never calls backward().
"""

from __future__ import annotations

import multiprocessing
import time
from typing import List, Optional

import torch

from precog.messaging import (
    PerLevelSnapshot,
    ShmTensorSlot,
    WeightSync,
)
from precog.model import (
    ClockConfig,
    HierarchicalClock,
    PCLevel,
)
from precog.model.config import PCLevelConfig


def _snapshot_from_step(
    level_idx: int,
    z_below: torch.Tensor,
    h_init: torch.Tensor,
    pred_from_above: Optional[torch.Tensor],
    task_target: Optional[torch.Tensor],
    x_actual: Optional[torch.Tensor],
    sigreg_mean: torch.Tensor,
    sigreg_outer: torch.Tensor,
    prev_action: Optional[torch.Tensor],
    weight_version: int,
) -> PerLevelSnapshot:
    return PerLevelSnapshot(
        level_idx=level_idx,
        z_below=z_below.detach().cpu(),
        h_init=h_init.detach().cpu(),
        pred_from_above=pred_from_above.detach().cpu() if pred_from_above is not None else None,
        task_target=task_target.detach().cpu() if task_target is not None else None,
        x_actual=x_actual.detach().cpu() if x_actual is not None else None,
        prev_action=prev_action.detach().cpu() if prev_action is not None else None,
        sigreg_mean=sigreg_mean.detach().cpu().clone(),
        sigreg_outer=sigreg_outer.detach().cpu().clone(),
        is_single_step=True,
    )


def run_level_n(
    stop_event: multiprocessing.Event,
    level_idx: int,
    config: PCLevelConfig,
    d_below: int,
    d_above: Optional[int],
    *,
    frequency: float,
    experience_queue: multiprocessing.Queue,
    weight_sync: WeightSync,
    upward_reader: Optional[ShmTensorSlot] = None,
    upward_writer: Optional[ShmTensorSlot] = None,
    downward_reader: Optional[ShmTensorSlot] = None,
    downward_writer: Optional[ShmTensorSlot] = None,
    device: str = "cpu",
    accumulate_for_forward: bool = True,
    obj_key: str = "",
    obj_target: float = 0.0,
):
    """Entry point for an upper-level forward process.

    Args:
        stop_event: Set by launcher to request clean shutdown.
        level_idx: Index of this level in the hierarchy.
        config: PCLevelConfig for this level.
        d_below: Dimension of the level below's representation.
        d_above: Dimension of the level above's representation (None if top).
        frequency: Tick rate in Hz for this level.
        experience_queue: Queue for sending experience to the Learner.
        weight_sync: WeightSync for receiving updated parameters.
        upward_reader: Reads z_below from the level below.
        upward_writer: Writes z_t upward to the level above.
        downward_reader: Reads pred_from_above from the level above.
        downward_writer: Writes pred_below downward to the level below.
        device: Torch device for computation.
        accumulate_for_forward: If True, accumulate z_below into sequences
            before forward pass. If False, use single-step every tick.
        obj_key: Observable key for objective (empty if disabled).
        obj_target: Target value for objective.
    """
    dev = torch.device(device)

    level = PCLevel(
        d_below=d_below,
        d_above=d_above,
        config=config,
    ).to(dev)

    clock_cfg = ClockConfig(
        level_frequencies=[frequency],
        time_scale=1.0,
    )
    clock = HierarchicalClock(clock_cfg)

    batch_size = 1
    h_state = level.init_hidden(batch_size, dev)

    accumulated: List[torch.Tensor] = []
    pfa_accumulated: List[torch.Tensor] = []

    weight_version = 0

    print(f"[L{level_idx}] Started on {device}, "
          f"d_below={d_below}, d_repr={config.d_representation}, "
          f"d_above={d_above}, freq={frequency}Hz")

    while not stop_event.is_set():
        z_below: Optional[torch.Tensor] = None
        if upward_reader is not None:
            raw = upward_reader.read(timeout_us=10000)
            if raw is not None:
                z_below = raw.to(dev)

        pred_from_above: Optional[torch.Tensor] = None
        if downward_reader is not None:
            raw = downward_reader.read(timeout_us=5000)
            if raw is not None:
                pred_from_above = raw.to(dev)

        if z_below is None:
            time.sleep(0.0001)
            continue

        if accumulate_for_forward:
            accumulated.append(z_below)
            if pred_from_above is not None:
                pfa_accumulated.append(pred_from_above)
        else:
            accumulated = [z_below]
            pfa_accumulated = [pred_from_above] if pred_from_above is not None else []

        clock.tick()

        if not clock.should_update(0):
            continue

        if not accumulated:
            continue

        seq = torch.stack(accumulated, dim=1)
        pfa_seq = None
        if pfa_accumulated and pfa_accumulated[0] is not None:
            pfa_seq = torch.stack(pfa_accumulated, dim=1)

        task_target = None
        x_actual = None
        if config.objective_enabled and obj_key:
            task_target = torch.full(
                (batch_size, 1), obj_target, device=dev
            )

        h_in = h_state.detach()

        if seq.shape[1] == 1:
            pfa_single = pfa_seq[:, 0] if pfa_seq is not None else None
            z_t, pred_below, h_new, eps_t, x_star = level.step(
                seq[:, 0], h_in, pfa_single, task_target
            )
        else:
            z_seq, pred_seq, h_new, eps_seq, _, x_star_seq = level.forward(
                seq, h_in, pfa_seq, task_target
            )
            z_t = z_seq[:, -1]
            pred_below = pred_seq[:, -1]
            eps_t = eps_seq[:, -1]
            x_star = x_star_seq[:, -1, :] if x_star_seq is not None else None

        sigreg_mean = level.sigreg._mean.detach().clone()
        sigreg_outer = level.sigreg._outer.detach().clone()

        if upward_writer is not None:
            upward_writer.write(z_t)

        if downward_writer is not None:
            downward_writer.write(pred_below)

        snap = _snapshot_from_step(
            level_idx=level_idx,
            z_below=seq[:, 0],
            h_init=h_in,
            pred_from_above=(
                pfa_seq[:, 0]
                if pfa_seq is not None
                else None
            ),
            task_target=task_target,
            x_actual=x_actual,
            sigreg_mean=sigreg_mean,
            sigreg_outer=sigreg_outer,
            prev_action=None,
            weight_version=weight_version,
        )
        experience_queue.put(snap)

        h_state = h_new.detach()

        ws_data = weight_sync.read_latest_bytes()
        if ws_data is not None:
            from precog.messaging.weight_sync import deserialize_to_model
            deserialize_to_model(level, ws_data, dev)
            weight_version += 1

        accumulated = []
        pfa_accumulated = []

        clock.sleep_until_next_tick()

    print(f"[L{level_idx}] Shutting down")
