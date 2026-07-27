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

        if prev_action is not None:
            prev_action = prev_action.detach()

        self.clock.tick()
        current_tick = self.clock._tick_count
        sim_time = self.clock.sim_time()

        updated_levels: List[int] = []
        total_surprise: Optional[torch.Tensor] = None
        action: Optional[torch.Tensor] = None

        z_level0 = self._encode_level0(obs_dict, prev_action)
        z_input = z_level0.unsqueeze(1)

        surprise_terms: List[torch.Tensor] = []

        for i, level in enumerate(self.model.levels):
            state = self._level_states[i]

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

            if seq.shape[1] == 1:
                pfa = pred_from_above[:, 0] if pred_from_above is not None else None
                z_t, _, h_new, eps_t = level.step(seq[:, 0], h_in, pfa)
                z_seq = z_t.unsqueeze(1)
                epsilon = eps_t.unsqueeze(1)
                sigreg_loss = level.sigreg.compute_loss_online()
            else:
                z_seq, _, h_new, epsilon, sigreg_loss = level.forward(
                    seq, h_in, pred_from_above
                )

            surprise_i = epsilon.pow(2).mean()
            surprise_terms.append(surprise_i.detach())

            loss = surprise_i + sigreg_loss

            if i == self.model.control_level_idx and prev_action is not None:
                z_ctrl = z_seq[:, -1].unsqueeze(1)
                action_pred_ctrl = self.model.control_head(z_ctrl)[:, -1]
                ctrl_loss = F.mse_loss(action_pred_ctrl, prev_action)
                loss = loss + self._ctrl_weight * ctrl_loss

            if self._online_learning and self._lr > 0:
                loss.backward()
                self._optimizers[i].step()
                self._optimizers[i].zero_grad()

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

        if diagnostics is not None and diagnostics.cfg.enabled:
            diagnostics.record_tick(
                TickResult(
                    action=action,
                    level_states=list(self._level_states),
                    updated_levels=updated_levels,
                    sim_time=sim_time,
                    total_surprise=total_surprise,
                ),
                self,
                process_time=time.monotonic() - t_start,
            )

        return TickResult(
            action=action,
            level_states=list(self._level_states),
            updated_levels=updated_levels,
            sim_time=sim_time,
            total_surprise=total_surprise,
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
        parts = [obs_dict[key] for key in self.model.config.observation_keys]
        if prev_action is not None:
            parts.append(prev_action)
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
