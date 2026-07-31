from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .hierarchical_clock import ClockConfig, HierarchicalClock
from .level_state import LevelState
from .diagnostics import DiagnosticsCollector

if TYPE_CHECKING:
    from .model import HierarchicalPCWorldModel


@dataclass
class RunnerConfig:
    level_frequencies: List[float]
    time_scale: float = 1.0
    batch_size: int = 1
    device: str = "cpu"
    accumulate_for_upper: bool = True
    online_learning: bool = True
    learning_rate: float = 1e-3
    imitation_loss_weight: float = 1.0


@dataclass
class TickResult:
    action: torch.Tensor
    level_states: List[LevelState]
    updated_levels: List[int]
    sim_time: float
    total_surprise: Optional[torch.Tensor]


@dataclass
class PerLevelSnapshot:
    """All data needed to replay one level's forward pass and compute loss."""

    level_idx: int

    z_below: torch.Tensor
    h_init: torch.Tensor
    pred_from_above: Optional[torch.Tensor]
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
    level_states: List[LevelState]
    sim_time: float
    total_surprise: Optional[torch.Tensor]
    tick_count: int = 0


class MultiRateRunner:
    def __init__(self, model, runner_cfg: RunnerConfig):
        self.model = model
        self.cfg = runner_cfg
        self.device = torch.device(runner_cfg.device)

        clock_cfg = ClockConfig(
            level_frequencies=runner_cfg.level_frequencies,
            time_scale=runner_cfg.time_scale,
        )
        self.clock = HierarchicalClock(clock_cfg)

        self._level_states: List[LevelState] = self._init_level_states()

        self._accum_buffers: List[List[torch.Tensor]] = [
            [] for _ in runner_cfg.level_frequencies
        ]

        self._optimizers: List[torch.optim.Optimizer] = []
        self._lr = runner_cfg.learning_rate
        self._online_learning = runner_cfg.online_learning
        self._ctrl_weight = runner_cfg.imitation_loss_weight

        ch = self.model.control_head
        if ch.output_scales is not None:
            self._action_scales = ch.output_scales.to(self.device)
        else:
            self._action_scales = torch.ones(
                self.model.config.control_dim, device=self.device
            )

        ctrl_idx = self.model.control_level_idx
        for i, level in enumerate(self.model.levels):
            param_groups = [{"params": level.parameters()}]
            if i == ctrl_idx:
                param_groups.append({"params": self.model.control_head.parameters()})
            self._optimizers.append(torch.optim.AdamW(param_groups, lr=self._lr))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def tick(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
        *,
        diagnostics: Optional[DiagnosticsCollector] = None,
    ) -> TickResult:
        t_start = time.monotonic() if diagnostics is not None else 0.0

        fwd = self.forward(obs_dict, prev_action)

        self.learn_step([fwd])

        result = TickResult(
            action=fwd.action,
            level_states=fwd.level_states,
            updated_levels=fwd.updated_levels,
            sim_time=fwd.sim_time,
            total_surprise=fwd.total_surprise,
        )

        if diagnostics is not None and diagnostics.cfg.enabled:
            diagnostics.record_tick(
                result,
                self,
                process_time=time.monotonic() - t_start,
            )

        return result

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        """Forward-only pass. Never calls backward()."""
        if prev_action is not None:
            prev_action = prev_action.detach()

        self.clock.tick()
        current_tick = self.clock._tick_count
        sim_time = self.clock.sim_time()

        updated_levels: List[int] = []
        total_surprise: Optional[torch.Tensor] = None
        level_snapshots: List[PerLevelSnapshot] = []

        z_level0 = self._encode_level0(obs_dict, prev_action)
        if torch.isnan(z_level0).any():
            return ForwardOutput(
                action=torch.zeros(self.cfg.batch_size, self.model.config.control_dim, device=self.device),
                level_snapshots=[],
                updated_levels=[],
                level_states=list(self._level_states),
                sim_time=sim_time,
                total_surprise=None,
                tick_count=current_tick,
            )

        z_input = z_level0.unsqueeze(1)
        surprise_terms: List[torch.Tensor] = []

        for i, level in enumerate(self.model.levels):
            state = self._level_states[i]
            lvl_cfg = self.model.config.level_configs[i]

            self._accum_buffers[i].append(z_input[:, -1].detach())

            if not self.clock.should_update(i):
                if state.last_z is not None:
                    z_input = state.last_z.unsqueeze(1)
                continue

            updated_levels.append(i)

            if i == 0:
                seq = z_input
                self._accum_buffers[0] = []
            else:
                seq = self._pop_sequence_for_level(i, z_input)

            pred_from_above: Optional[torch.Tensor] = None
            if i + 1 < len(self._level_states):
                above_state = self._level_states[i + 1]
                if above_state.last_z is not None:
                    above_level = self.model.levels[i + 1]
                    with torch.no_grad():
                        pfa = above_level.predict_downward(above_state.last_z)
                    T = seq.shape[1]
                    pred_from_above = pfa.unsqueeze(1).expand(-1, T, -1)

            h_in = state.hidden.detach()

            task_target = None
            x_actual = None
            if lvl_cfg.objective_enabled:
                b = seq.shape[0]
                task_target = torch.full(
                    (b, 1),
                    lvl_cfg.objective_target_value,
                    device=self.device,
                )
                raw = obs_dict.get(lvl_cfg.objective_observable_key)
                if raw is not None:
                    x_actual = raw.reshape(b, -1)[:, :1]

            is_single = seq.shape[1] == 1

            if is_single:
                pfa_single = pred_from_above[:, 0] if pred_from_above is not None else None
                z_t, _, h_new, eps_t, x_star = level.step(
                    seq[:, 0], h_in, pfa_single, task_target
                )
                z_seq = z_t.unsqueeze(1)
                epsilon = eps_t.unsqueeze(1)
            else:
                z_seq, _, h_new, epsilon, _, x_star_seq = level.forward(
                    seq, h_in, pred_from_above, task_target
                )
                x_star = x_star_seq[:, -1, :] if x_star_seq is not None else None

            surprise_i = epsilon.pow(2).mean()
            surprise_terms.append(surprise_i.detach())

            sigreg_mean = level.sigreg._mean.detach().clone()
            sigreg_outer = level.sigreg._outer.detach().clone()

            pfa_for_replay: Optional[torch.Tensor] = None
            if is_single:
                pfa_for_replay = (
                    pred_from_above[:, 0].detach()
                    if pred_from_above is not None
                    else None
                )
            else:
                pfa_for_replay = (
                    pred_from_above.detach()
                    if pred_from_above is not None
                    else None
                )

            z_below_for_replay = seq[:, 0].detach() if is_single else seq.detach()

            snapshot = PerLevelSnapshot(
                level_idx=i,
                z_below=z_below_for_replay,
                h_init=h_in.detach(),
                pred_from_above=pfa_for_replay,
                task_target=task_target.detach() if task_target is not None else None,
                x_actual=x_actual.detach() if x_actual is not None else None,
                prev_action=(
                    prev_action.detach()
                    if prev_action is not None and i == self.model.control_level_idx
                    else None
                ),
                sigreg_mean=sigreg_mean,
                sigreg_outer=sigreg_outer,
                is_single_step=is_single,
            )
            level_snapshots.append(snapshot)

            state.update(
                hidden=h_new.detach(),
                z=z_seq[:, -1].detach(),
                pred_error=epsilon.detach(),
                tick=current_tick,
                sim_time=sim_time,
            )

            z_input = z_seq.detach()

        if surprise_terms:
            total_surprise = torch.stack(surprise_terms).sum()

        ctrl_idx = self.model.control_level_idx
        ctrl_state = self._level_states[ctrl_idx]

        z_ctrl = ctrl_state.last_z.unsqueeze(1)
        action = self.model.control_head(z_ctrl)[:, -1]

        return ForwardOutput(
            action=action,
            level_snapshots=level_snapshots,
            updated_levels=updated_levels,
            level_states=list(self._level_states),
            sim_time=sim_time,
            total_surprise=total_surprise,
            tick_count=current_tick,
        )

    def learn_step(self, forward_outputs: List[ForwardOutput]) -> None:
        """Replay forward passes and run backward + step for all levels.

        Args:
            forward_outputs: One or more ForwardOutput from forward().
        """
        if not self._online_learning or self._lr <= 0:
            return

        ctrl_idx = self.model.control_level_idx

        per_level_batches: Dict[int, List[PerLevelSnapshot]] = {}
        for fwd in forward_outputs:
            for snap in fwd.level_snapshots:
                per_level_batches.setdefault(snap.level_idx, []).append(snap)

        for level_idx, snapshots in per_level_batches.items():
            if not snapshots:
                continue

            level = self.model.levels[level_idx]
            optimizer = self._optimizers[level_idx]

            sigreg_mean_saved = level.sigreg._mean.clone()
            sigreg_outer_saved = level.sigreg._outer.clone()

            if snapshots[0].sigreg_mean is not None:
                level.sigreg._mean.copy_(snapshots[-1].sigreg_mean)
                level.sigreg._outer.copy_(snapshots[-1].sigreg_outer)

            loss_terms: List[torch.Tensor] = []

            for snap in snapshots:
                if snap.is_single_step:
                    z_below = snap.z_below.to(self.device).unsqueeze(0)
                    h_in = snap.h_init.to(self.device)
                    pfa: Optional[torch.Tensor] = snap.pred_from_above.to(self.device) if snap.pred_from_above is not None else None
                    tt: Optional[torch.Tensor] = snap.task_target.to(self.device) if snap.task_target is not None else None

                    z_t, _, _, eps_t, x_star = level.step(z_below[0], h_in, pfa, tt)
                    epsilon = eps_t.unsqueeze(0)
                    z_seq = z_t.unsqueeze(0)
                else:
                    z_below = snap.z_below.to(self.device)
                    h_in = snap.h_init.to(self.device)
                    pfa_seq: Optional[torch.Tensor] = snap.pred_from_above.to(self.device) if snap.pred_from_above is not None else None
                    tt: Optional[torch.Tensor] = snap.task_target.to(self.device) if snap.task_target is not None else None

                    z_seq, _, _, epsilon, _, x_star_seq = level.forward(
                        z_below, h_in, pfa_seq, tt
                    )
                    x_star = x_star_seq[:, -1, :] if x_star_seq is not None else None

                surprise_i = epsilon.pow(2).mean()
                sigreg_loss = level.sigreg.compute_loss_online()
                loss = surprise_i + sigreg_loss
                loss_terms.append(loss)

                if x_star is not None:
                    xa = snap.x_actual.to(self.device) if snap.x_actual is not None else None
                    tgt = snap.task_target.to(self.device) if snap.task_target is not None else None
                    if xa is not None and tgt is not None:
                        loss_ae = F.mse_loss(x_star, xa)
                        loss_task = F.mse_loss(x_star, tgt)
                        loss = loss + level.ae_weight * loss_ae + level.task_weight * loss_task
                        loss_terms[-1] = loss

                if level_idx == ctrl_idx and snap.prev_action is not None:
                    prev_act = snap.prev_action.to(self.device)
                    z_ctrl = z_seq[:, -1].unsqueeze(1)
                    action_pred_ctrl = self.model.control_head(z_ctrl)[:, -1]
                    ctrl_loss = F.mse_loss(action_pred_ctrl, prev_act)
                    loss = loss + self._ctrl_weight * ctrl_loss
                    loss_terms[-1] = loss

            total_loss = torch.stack(loss_terms).sum()

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad()
            else:
                total_loss.backward()
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

            level.sigreg._mean.copy_(sigreg_mean_saved)
            level.sigreg._outer.copy_(sigreg_outer_saved)

    def reset(self, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.cfg.batch_size = batch_size
        self.clock.reset()
        self._level_states = self._init_level_states()
        self._accum_buffers = [[] for _ in self.cfg.level_frequencies]

    def detach_states(self):
        for state in self._level_states:
            state.detach_()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _init_level_states(self) -> List[LevelState]:
        states = []
        for i, level in enumerate(self.model.levels):
            d_state = level.ssm.d_state
            d_repr = level.d_repr
            states.append(
                LevelState.init(
                    level_idx=i,
                    d_state=d_state,
                    d_repr=d_repr,
                    batch_size=self.cfg.batch_size,
                    device=self.device,
                )
            )
        return states

    def _encode_level0(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shapes = self.model.config.observation_shapes
        parts = []
        for key in self.model.config.observation_keys:
            value = obs_dict.get(key)
            if value is None:
                shape = shapes.get(key, (1, 0))
                value = torch.zeros(self.cfg.batch_size, shape[-1], device=self.device)
            parts.append(value)
        if prev_action is not None:
            parts.append(prev_action / self._action_scales)
        else:
            parts.append(
                torch.zeros(
                    self.cfg.batch_size,
                    self.model.config.control_dim,
                    device=self.device,
                )
            )
        return torch.cat(parts, dim=-1)

    def _pop_sequence_for_level(
        self, level_idx: int, fallback_z: torch.Tensor
    ) -> torch.Tensor:
        buf = self._accum_buffers[level_idx]

        if self.cfg.accumulate_for_upper and buf:
            seq = torch.stack(buf, dim=1)
            self._accum_buffers[level_idx] = []
            return seq

        self._accum_buffers[level_idx] = []
        return fallback_z


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
) -> Optional[float]:
    """Replay forward passes from snapshots and run backward + step.

    Used by the decoupled Learner process, which replays experience on
    its own model copies without sharing an autograd graph.

    Args:
        level: The PC level to train (learner's copy).
        control_head: Control head if this is the control level.
        snapshots: Batched experience snapshots for this level.
        optimizer: Optimizer for this level's parameters.
        device: Torch device for computation.
        ctrl_weight: Imitation loss weight.
        is_control_level: Whether this level drives the control head.

    Returns:
        Total loss scalar, or None if no valid snapshots.
    """
    if not snapshots:
        return None

    sigreg_mean_saved = level.sigreg._mean.clone()
    sigreg_outer_saved = level.sigreg._outer.clone()

    if snapshots[0].sigreg_mean is not None:
        level.sigreg._mean.copy_(snapshots[-1].sigreg_mean)
        level.sigreg._outer.copy_(snapshots[-1].sigreg_outer)

    loss_terms: List[torch.Tensor] = []

    for snap in snapshots:
        if snap.is_single_step:
            z_below = snap.z_below.to(device).unsqueeze(0)
            h_in = snap.h_init.to(device)
            pfa: Optional[torch.Tensor] = (
                snap.pred_from_above.to(device)
                if snap.pred_from_above is not None
                else None
            )
            tt: Optional[torch.Tensor] = (
                snap.task_target.to(device)
                if snap.task_target is not None
                else None
            )

            z_t, _, _, eps_t, x_star = level.step(z_below[0], h_in, pfa, tt)
            epsilon = eps_t.unsqueeze(0)
            z_seq = z_t.unsqueeze(0)
        else:
            z_below = snap.z_below.to(device)
            h_in = snap.h_init.to(device)
            pfa_seq: Optional[torch.Tensor] = (
                snap.pred_from_above.to(device)
                if snap.pred_from_above is not None
                else None
            )
            tt: Optional[torch.Tensor] = (
                snap.task_target.to(device)
                if snap.task_target is not None
                else None
            )

            z_seq, _, _, epsilon, _, x_star_seq = level.forward(
                z_below, h_in, pfa_seq, tt
            )
            x_star = x_star_seq[:, -1, :] if x_star_seq is not None else None

        surprise_i = epsilon.pow(2).mean()
        sigreg_loss = level.sigreg.compute_loss_online()
        loss = surprise_i + sigreg_loss
        loss_terms.append(loss)

        if x_star is not None:
            xa = (
                snap.x_actual.to(device)
                if snap.x_actual is not None
                else None
            )
            tgt = (
                snap.task_target.to(device)
                if snap.task_target is not None
                else None
            )
            if xa is not None and tgt is not None:
                loss_ae = F.mse_loss(x_star, xa)
                loss_task = F.mse_loss(x_star, tgt)
                loss = loss + level.ae_weight * loss_ae + level.task_weight * loss_task
                loss_terms[-1] = loss

        if is_control_level and snap.prev_action is not None and control_head is not None:
            prev_act = snap.prev_action.to(device)
            z_ctrl = z_seq[:, -1].unsqueeze(1)
            action_pred_ctrl = control_head(z_ctrl)[:, -1]
            ctrl_loss = F.mse_loss(action_pred_ctrl, prev_act)
            loss = loss + ctrl_weight * ctrl_loss
            loss_terms[-1] = loss

    total_loss = torch.stack(loss_terms).sum()

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        optimizer.zero_grad()
    else:
        total_loss.backward()
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

    level.sigreg._mean.copy_(sigreg_mean_saved)
    level.sigreg._outer.copy_(sigreg_outer_saved)

    return total_loss.item()
