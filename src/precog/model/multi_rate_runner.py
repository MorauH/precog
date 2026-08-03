from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .hierarchical_clock import ClockConfig, HierarchicalClock
from .level_state import LevelState

if TYPE_CHECKING:
    from .model import HierarchicalPCWorldModel


@dataclass
class RunnerConfig:
    level_frequencies: List[float]
    time_scale: float = 1.0
    batch_size: int = 1
    device: str = "cpu"
    accumulate_for_upper: bool = True


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

        ch = self.model.control_head
        if ch.output_scales is not None:
            self._action_scales = ch.output_scales.to(self.device)
        else:
            self._action_scales = torch.ones(
                self.model.config.control_dim, device=self.device
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        """Full multi-level forward pass. Never calls backward()."""
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
                action=torch.zeros(
                    self.cfg.batch_size,
                    self.model.config.control_dim,
                    device=self.device,
                ),
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

            a_t: Optional[torch.Tensor] = None
            if i + 1 < len(self._level_states):
                above_state = self._level_states[i + 1]
                if above_state.last_z_pred is not None:
                    T = seq.shape[1]
                    a_t = above_state.last_z_pred.unsqueeze(1).expand(-1, T, -1)

            h_in = state.hidden.detach()
            z_t_pred = (
                state.last_z_pred.detach() if state.last_z_pred is not None else None
            )

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
                a_t_single = a_t[:, 0] if a_t is not None else None
                z_t_out, z_next_pred, h_new, z_delta, x_star = level.step(
                    seq[:, 0], h_in, a_t_single, z_t_pred, task_target
                )
                z_seq = z_t_out.unsqueeze(1)
                z_delta_seq = z_delta.unsqueeze(1)
                z_next_pred_single = z_next_pred
            else:
                z_seq, z_next_pred_seq, h_new, z_delta_seq, _, x_star_seq = (
                    level.forward(seq, h_in, a_t, z_t_pred, task_target)
                )
                z_next_pred_single = z_next_pred_seq[:, -1]
                x_star = x_star_seq[:, -1, :] if x_star_seq is not None else None

            surprise_i = z_delta_seq.pow(2).mean()
            surprise_terms.append(surprise_i.detach())

            sigreg_mean = level.sigreg._mean.detach().clone()
            sigreg_outer = level.sigreg._outer.detach().clone()

            a_t_for_replay: Optional[torch.Tensor] = None
            if is_single:
                a_t_for_replay = a_t[:, 0].detach() if a_t is not None else None
            else:
                a_t_for_replay = a_t.detach() if a_t is not None else None

            signal_for_replay = seq[:, 0].detach() if is_single else seq.detach()
            z_t_pred_for_replay = z_t_pred.detach() if z_t_pred is not None else None

            snapshot = PerLevelSnapshot(
                level_idx=i,
                signal_from_below=signal_for_replay,
                h_init=h_in.detach(),
                a_t=a_t_for_replay,
                z_t_pred=z_t_pred_for_replay,
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
                z_pred=z_next_pred_single.detach(),
                pred_error=z_delta_seq[:, -1].detach(),
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

    def forward_level0(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
        a_t: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        """Lightweight forward pass for L0 process — level 0 only.

        Runs encoding, Level 0 encoder + SSM step, and control head. Does NOT
        iterate upper levels (those run in separate level_n processes).

        Args:
            obs_dict: Raw observation dict.
            prev_action: Previous action for imitation loss replay.
            a_t: Top-down signal from L1 (SSM output z_t+1_pred). If None,
                 no top-down signal.

        Returns:
            ForwardOutput with only level 0's snapshot and states.
        """
        if prev_action is not None:
            prev_action = prev_action.detach()

        self.clock.tick()
        current_tick = self.clock._tick_count
        sim_time = self.clock.sim_time()

        z_level0 = self._encode_level0(obs_dict, prev_action)
        if torch.isnan(z_level0).any():
            return ForwardOutput(
                action=torch.zeros(
                    self.cfg.batch_size,
                    self.model.config.control_dim,
                    device=self.device,
                ),
                level_snapshots=[],
                updated_levels=[],
                level_states=list(self._level_states),
                sim_time=sim_time,
                total_surprise=None,
                tick_count=current_tick,
            )

        level = self.model.levels[0]
        state = self._level_states[0]
        lvl_cfg = self.model.config.level_configs[0]
        h_in = state.hidden.detach()
        z_t_pred = state.last_z_pred.detach() if state.last_z_pred is not None else None

        task_target = None
        x_actual = None
        if lvl_cfg.objective_enabled:
            task_target = torch.full(
                (1, 1),
                lvl_cfg.objective_target_value,
                device=self.device,
            )
            raw = obs_dict.get(lvl_cfg.objective_observable_key)
            if raw is not None:
                x_actual = raw.reshape(1, -1)[:, :1]

        a_t_single = a_t.detach().to(self.device) if a_t is not None else None
        z_t_out, z_next_pred, h_new, z_delta, x_star = level.step(
            z_level0, h_in, a_t_single, z_t_pred, task_target
        )

        surprise_i = z_delta.pow(2).mean()
        sigreg_mean = level.sigreg._mean.detach().clone()
        sigreg_outer = level.sigreg._outer.detach().clone()

        z_t_pred_for_replay = z_t_pred.detach() if z_t_pred is not None else None

        snapshot = PerLevelSnapshot(
            level_idx=0,
            signal_from_below=z_level0.detach(),
            h_init=h_in.detach(),
            a_t=a_t_single.detach() if a_t_single is not None else None,
            z_t_pred=z_t_pred_for_replay,
            task_target=task_target.detach() if task_target is not None else None,
            x_actual=x_actual.detach() if x_actual is not None else None,
            prev_action=prev_action.detach() if prev_action is not None else None,
            sigreg_mean=sigreg_mean,
            sigreg_outer=sigreg_outer,
            is_single_step=True,
        )

        state.update(
            hidden=h_new.detach(),
            z=z_t_out.detach(),
            z_pred=z_next_pred.detach(),
            pred_error=z_delta.detach(),
            tick=current_tick,
            sim_time=sim_time,
        )

        action = self.model.control_head(z_t_out.unsqueeze(1))[:, -1]

        return ForwardOutput(
            action=action,
            level_snapshots=[snapshot],
            updated_levels=[0],
            level_states=list(self._level_states),
            sim_time=sim_time,
            total_surprise=surprise_i,
            tick_count=current_tick,
        )

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
    micro_batch: int = 0,
) -> Optional[float]:
    """Replay forward passes from snapshots and run backward + step.

    Uses per-snapshot backward+step to keep autograd graphs small —
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
