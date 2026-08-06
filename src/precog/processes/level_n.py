"""
Generic upper-level process (L1, L2, ... Ln).

Each upper-level process:
  1. Reads the level below's prediction error (z_delta) from an upward ShmTensorSlot.
  2. Receives top-down predictions from the level above via a downward ShmTensorSlot.
  3. Accumulates signal_below values into sequences.
  4. On its own tick cadence, runs the SSM forward pass.
  5. Sends its prediction error (z_delta) upward and its prediction (z_next_pred) downward.
  6. Does online learning (forward-only, backward on its own tick).

Never calls backward() from a separate learner.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import List, Optional

from precog.model.pc_level_jepa import PCLevelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

from precog.messaging import LevelLog, ShmTensorSlot
from precog.model import LevelClock, PCLevel
from precog.model.config import PCLevelConfig
from precog.model.sigreg import SIGReg


def run_level_n(
    stop_event: multiprocessing.Event,
    level_idx: int,
    config: PCLevelConfig,
    d_below: int,
    d_above: Optional[int],
    *,
    clock: LevelClock,
    frequency: float = 0.0,
    upward_reader: Optional[ShmTensorSlot] = None,
    upward_writer: Optional[ShmTensorSlot] = None,
    downward_reader: Optional[ShmTensorSlot] = None,
    downward_writer: Optional[ShmTensorSlot] = None,
    task_actual_reader: Optional[ShmTensorSlot] = None,
    device: str = "cpu",
    accumulate_for_forward: bool = True,
    obj_key: str = "",
    obj_target: float = 0.0,
    lr: float = 0e-3,
):
    """Entry point for an upper-level forward process.

    Args:
        stop_event: Set by launcher to request clean shutdown.
        level_idx: Index of this level in the hierarchy.
        config: PCLevelConfig for this level.
        d_below: Dimension of the level below's representation.
        d_above: Dimension of the level above's representation (None if top).
        frequency: Tick rate in Hz for this level.
        upward_reader: Reads z_below from the level below.
        upward_writer: Writes z_t upward to the level above.
        downward_reader: Reads pred_from_above from the level above.
        downward_writer: Writes pred_below downward to the level below.
        task_actual_reader: Reads task reference value.
        device: Torch device for computation.
        accumulate_for_forward: If True, accumulate z_below into sequences
            before forward pass. If False, use single-step every tick.
        obj_key: Observable key for objective (empty if disabled).
        obj_target: Target value for objective.
        lr: Learning rate for online weight updates.
    """
    dev = torch.device(device)

    level = PCLevel(
        d_below=d_below,
        d_above=d_above,
        config=config,
    ).to(dev)
    level.train()

    optimizer = torch.optim.AdamW(level.parameters(), lr=lr, weight_decay=1e-4)

    loss_translate_weight = (
        config.objective_ae_weight if config.objective_enabled else 0.0
    )
    loss_task_weight = config.objective_task_weight if config.objective_enabled else 0.0

    sigreg_loss_fn = SIGReg(
        d_repr=config.d_representation,
        online_tau=config.sigreg_tau,
        var_threshold=config.sigreg_var_threshold,
    ).to(dev)

    batch_size = 1
    h_state = level.init_hidden(batch_size, dev)
    prev_z_pred: Optional[torch.Tensor] = None
    last_signal_above: Optional[torch.Tensor] = (
        None  # Hold value as this updates slower
    )
    task_actual: Optional[torch.Tensor] = None  # Hold value as this updates slower

    accumulated_below: List[torch.Tensor] = []
    accumulated_above: List[torch.Tensor] = []

    level_log = LevelLog(level_idx)
    _idle_start = time.perf_counter()

    print(
        f"[L{level_idx}] Started Online Learning on {device}, "
        f"d_below={d_below}, d_repr={config.d_representation}, "
        f"d_above={d_above}, freq={frequency}Hz, lr={lr}"
    )

    while not stop_event.is_set():
        # TODO: clock.wait_tick()

        #  ----- Read bottom up signal
        signal_below: Optional[torch.Tensor] = None
        if upward_reader is not None:
            raw_below = upward_reader.read(timeout_us=5000)
            if raw_below is not None:
                signal_below = raw_below.to(dev)

        # ----- Read (or hold) top-down signal
        if downward_reader is not None:
            raw_above = downward_reader.read(timeout_us=5000)
            if raw_above is not None:
                last_signal_above = raw_above.to(dev)

        # ----- Read task actual reference value
        if task_actual_reader is not None:
            raw_task = task_actual_reader.read(timeout_us=5000)
            if raw_task is not None:
                task_actual = raw_task.to(dev)

        if signal_below is None:
            clock.mark_done()
            time.sleep(0.01)
            continue

        if d_above is not None:
            # Forward-fill last seen signal from above; fallback to zeros if none received yet
            cur_above = (
                last_signal_above
                if last_signal_above is not None
                else torch.zeros(batch_size, d_above, device=dev)
            )
        else:
            cur_above = None

        if accumulate_for_forward:
            accumulated_below.append(signal_below)
            if cur_above is not None:
                accumulated_above.append(cur_above)
        else:
            accumulated_below = [signal_below]
            accumulated_above = [cur_above] if cur_above is not None else []

        # ----- Format sequence tensors
        seq_below = torch.stack(accumulated_below, dim=1)
        seq_above = torch.stack(accumulated_above, dim=1) if accumulated_above else None

        task_target = None
        x_actual = None
        if config.objective_enabled and obj_key:
            task_target = torch.full((batch_size, 1), obj_target, device=dev)

        h_in = h_state.detach()

        optimizer.zero_grad()

        # ----- Forward pass
        if seq_below.shape[1] == 1:
            # Single step
            out: PCLevelOutput = level.step(
                signal_below=seq_below[:, 0],
                h_prev=h_in,
                signal_above=seq_above[:, 0] if seq_above is not None else None,
                z_t_pred=prev_z_pred.detach() if prev_z_pred is not None else None,
            )
            z_t = out.z_t
            z_next_pred = out.z_next_pred
            h_new = out.ssm_h
            x_pred = out.x_pred
            x_curr = out.x_curr
            z_delta_up = out.z_delta

            # Prediction loss — SSM output vs current encoder output.
            # Both in the same forward graph → no stale-parameter issue.
            # z_t is detached as target so the encoder doesn't get a trivial
            # "match yourself" gradient; the encoder signal comes from the SSM
            # path (z_t is attached in ssm_input) + loss_translate + sigreg.
            loss_pred = F.mse_loss(z_next_pred, z_t.detach())

        else:
            # Multi-step sequence mode
            out: PCLevelOutput = level.forward(
                signal_below_seq=seq_below,
                h0=h_in,
                signal_above_seq=seq_above,
                z_t_pred_init=prev_z_pred.detach() if prev_z_pred is not None else None,
            )
            z_t_seq = out.z_t  # (1, T, d_repr)
            z_next_pred_seq = out.z_next_pred  # (1, T, d_repr)

            z_t = z_t_seq[:, -1]
            z_next_pred = z_next_pred_seq[:, -1]
            h_new = out.ssm_h
            x_pred = out.x_pred[:, -1] if out.x_pred else None
            x_curr = out.x_curr[:, -1] if out.x_curr else None
            z_delta_up = out.z_delta[:, -1] if out.z_delta is not None else None

            # Prediction loss — all within a single forward graph, so gradients flow
            loss_pred = torch.tensor(0.0, device=dev)
            if seq_below.shape[1] > 1:
                preds = z_next_pred_seq[:, :-1]
                targets = z_t_seq[:, 1:].detach()
                loss_pred = F.mse_loss(preds, targets)

        # ----- Loss calc

        with torch.no_grad():
            sigreg_loss_fn.update_online(z_t.detach())
        loss_sigreg = sigreg_loss_fn.compute_loss_online()

        # Optional task losses
        #   loss_task      = MSE(x_pred, task_target)  — SSM's predicted x vs target → flows through SSM
        #   loss_translate = MSE(x_curr, task_actual)   — encoder readout vs actual   → flows through translator only
        loss_task = torch.tensor(0.0, device=dev)
        loss_translate = torch.tensor(0.0, device=dev)

        if (
            config.objective_enabled
            and x_pred is not None
            and x_curr is not None
            and task_target is not None
            and task_actual is not None
        ):
            loss_task = F.mse_loss(x_pred, task_target)
            loss_translate = F.mse_loss(x_curr, task_actual)

        total_loss = (
            loss_pred
            + (1e-2 * loss_sigreg)
            + (loss_task_weight * loss_task)
            + (loss_translate_weight * loss_translate)
        )

        # ----- Backward and optimize
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(level.parameters(), max_norm=1.0)
            optimizer.step()

        # ----- Publish outputs
        if upward_writer is not None:
            upward_signal = z_delta_up if z_delta_up is not None else z_t
            upward_writer.write(upward_signal)

        if downward_writer is not None:
            downward_writer.write(z_next_pred)

        # ----- State update
        h_state = h_new.detach()
        prev_z_pred = z_next_pred.detach()

        accumulated_below.clear()
        accumulated_above.clear()

        # ----- Diagnostics
        _now = time.perf_counter()
        idle_us = (_now - _idle_start) * 1_000_000
        _idle_start = _now
        with torch.no_grad():
            surprise = (
                F.mse_loss(prev_z_pred, z_t).item()
                if prev_z_pred is not None and z_t is not None
                else 0.0
            )
        level_log.write(
            tick_count=clock.tick_count,
            sim_time=clock.sim_time,
            idle_time_us=idle_us,
            surprise=surprise,
            loss_pred=loss_pred.item(),
            loss_sigreg=loss_sigreg.item(),
            loss_task=loss_task.item()
            if isinstance(loss_task, torch.Tensor)
            else loss_task,
            loss_translate=loss_translate.item()
            if isinstance(loss_translate, torch.Tensor)
            else loss_translate,
            loss_total=total_loss.item(),
        )

        clock.mark_done()

    level_log.close()
    level_log.unlink()
    print(f"[L{level_idx}] Shutting down")
