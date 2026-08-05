from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class PerLevelSnapshot:
    """All data needed to replay one level's forward pass and compute loss."""

    level_idx: int

    signal_from_below: torch.Tensor
    h_init: torch.Tensor
    a_t: Optional[torch.Tensor]
    z_t_pred: Optional[torch.Tensor]
    task_target: Optional[torch.Tensor]
    x_actual: Optional[torch.Tensor]
    prev_action: Optional[torch.Tensor]

    sigreg_mean: Optional[torch.Tensor]
    sigreg_outer: Optional[torch.Tensor]

    is_single_step: bool = True


@dataclass
class ForwardOutput:
    action: torch.Tensor
    level_snapshots: List[PerLevelSnapshot]
    updated_levels: List[int]
    level_states: List
    sim_time: float
    total_surprise: Optional[torch.Tensor]
    tick_count: int = 0


# ------------------------------------------------------------------ #
# Standalone replay for decoupled learner process
# ------------------------------------------------------------------ #


def replay_learn_level(
    level: torch.nn.Module,
    control_head: Optional[torch.nn.Module],
    snapshots: List[PerLevelSnapshot],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    ctrl_weight: float = 1.0,
    is_control_level: bool = False,
    micro_batch: int = 0,
) -> Optional[float]:
    """Replay forward passes from snapshots and run backward + step.

    Uses per-snapshot backward+step to keep autograd graphs small --
    critical for CPU performance. A single backward through N stacked
    graphs on CPU scales poorly (the autograd engine holds N×
    intermediate tensors). Processing each snapshot independently
    keeps each graph tiny.

    Args:
        level: The PC level to train (learner's copy).
        control_head: Control head if this is the control level.
        snapshots: Batched experience snapshots for this level.
        optimizer: Optimizer for this level's parameters.
        device: Torch device for computation.
        ctrl_weight: Imitation loss weight.
        is_control_level: Whether this level drives the control head.
        micro_batch: Accumulate gradients over this many snapshots before
            stepping (0 = step per snapshot). Helps when batch size is
            small and gradient noise is high.

    Returns:
        Average loss across all snapshots, or None if no valid snapshots.
    """
    if not snapshots:
        return None

    sigreg_mean_saved = level.sigreg._mean.clone()
    sigreg_outer_saved = level.sigreg._outer.clone()

    if snapshots[0].sigreg_mean is not None:
        level.sigreg._mean.copy_(snapshots[-1].sigreg_mean)
        level.sigreg._outer.copy_(snapshots[-1].sigreg_outer)

    total_loss = 0.0
    count = 0
    micro_count = 0

    def _apply_gradients():
        nonlocal micro_count
        if micro_count == 0:
            return
        grad_nan = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        if grad_nan:
            optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
        micro_count = 0

    for snap in snapshots:
        if snap.is_single_step:
            signal = snap.signal_from_below.to(device, non_blocking=True)
            h_in = snap.h_init.to(device, non_blocking=True)
            a_t_rl: Optional[torch.Tensor] = (
                snap.a_t.to(device, non_blocking=True) if snap.a_t is not None else None
            )
            z_t_pred_rl: Optional[torch.Tensor] = (
                snap.z_t_pred.to(device, non_blocking=True)
                if snap.z_t_pred is not None
                else None
            )
            tt: Optional[torch.Tensor] = (
                snap.task_target.to(device, non_blocking=True)
                if snap.task_target is not None
                else None
            )

            z_t_out, _, _, z_delta, x_star = level.step(
                signal, h_in, a_t_rl, z_t_pred_rl, tt
            )
            z_delta_seq = z_delta.unsqueeze(0)
            z_seq = z_t_out.unsqueeze(0)
        else:
            signal = snap.signal_from_below.to(device)
            h_in = snap.h_init.to(device)
            a_seq_rl: Optional[torch.Tensor] = (
                snap.a_t.to(device) if snap.a_t is not None else None
            )
            z_t_pred_rl: Optional[torch.Tensor] = (
                snap.z_t_pred.to(device) if snap.z_t_pred is not None else None
            )
            tt: Optional[torch.Tensor] = (
                snap.task_target.to(device) if snap.task_target is not None else None
            )

            z_seq, _, _, z_delta_seq, _, x_star_seq = level.forward(
                signal, h_in, a_seq_rl, z_t_pred_rl, tt
            )
            x_star = x_star_seq[:, -1, :] if x_star_seq is not None else None

        surprise_i = z_delta_seq.pow(2).mean()
        sigreg_loss = level.sigreg.compute_loss_online()
        loss = surprise_i + sigreg_loss

        if x_star is not None:
            xa = (
                snap.x_actual.to(device, non_blocking=True)
                if snap.x_actual is not None
                else None
            )
            tgt = (
                snap.task_target.to(device, non_blocking=True)
                if snap.task_target is not None
                else None
            )
            if xa is not None and tgt is not None:
                loss_ae = F.mse_loss(x_star, xa)
                loss_task = F.mse_loss(x_star, tgt)
                loss = loss + level.ae_weight * loss_ae + level.task_weight * loss_task

        if (
            is_control_level
            and snap.prev_action is not None
            and control_head is not None
        ):
            prev_act = snap.prev_action.to(device, non_blocking=True)
            z_ctrl = z_seq[:, -1].unsqueeze(1)
            action_pred_ctrl = control_head(z_ctrl)[:, -1]
            ctrl_loss = F.mse_loss(action_pred_ctrl, prev_act)
            loss = loss + ctrl_weight * ctrl_loss

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            micro_count = 0
            continue

        loss.backward()
        micro_count += 1
        total_loss += loss.detach().item()
        count += 1

        if micro_batch <= 0 or micro_count >= micro_batch:
            _apply_gradients()

    _apply_gradients()

    level.sigreg._mean.copy_(sigreg_mean_saved)
    level.sigreg._outer.copy_(sigreg_outer_saved)

    return total_loss / max(count, 1)
