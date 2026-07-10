from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .fnn import FNN
from .ssm import SelectiveSSM
from .pc_level_jepa import PCLevel
from .control_head import ControlHead
from .config import ModelConfig
from .multi_rate_runner import MultiRateRunner, RunnerConfig


def _probe_compile_backends() -> list[str]:
    """Return a list of working torch.compile backends, best first."""
    working: list[str] = []

    @torch.compile(backend="inductor")
    def _probe(x: torch.Tensor) -> torch.Tensor:
        return x.matmul(x.t())

    try:
        _probe(torch.randn(4, 4))
        working.append("inductor")
    except Exception:
        pass

    return working


class HierarchicalPCWorldModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        prev_dim = config.d_input

        self.levels: nn.ModuleList = nn.ModuleList()
        for i, level_cfg in enumerate(config.level_configs):
            self.levels.append(
                PCLevel(
                    d_below=prev_dim,
                    d_above=None
                    if i == len(self.config.level_configs) - 1
                    else level_cfg.d_representation,
                    config=level_cfg,
                )
            )
            prev_dim = level_cfg.d_representation

        ctrl_cfg = config.control_head
        control_input_dim = (
            config.level_configs[config.control_level_idx].d_representation * 2
        )

        self.control_head = ControlHead(
            input_dim=control_input_dim,
            hidden_dims=ctrl_cfg.hidden_dims,
            output_dim=ctrl_cfg.output_dim,
        )

        self.control_level_idx = config.control_level_idx

    # ------------------------------------------------------------------ #
    # Training forward (full sequences, no rate scheduling)
    # ------------------------------------------------------------------ #

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
        hidden_states: Optional[List[torch.Tensor]] = None,
        hidden_states_target: Optional[List[torch.Tensor]] = None,
        return_all: bool = False,
    ):
        batch = next(iter(obs_dict.values())).shape[0]
        obs_device = next(iter(obs_dict.values())).device

        if hidden_states is None:
            hidden_states = [
                level.init_hidden(batch, device=obs_device) for level in self.levels
            ]
        if hidden_states_target is None:
            hidden_states_target = [
                level.init_hidden_target(batch, device=obs_device)
                for level in self.levels
            ]

        parts = [obs_dict[key] for key in self.config.observation_keys]

        if prev_action is not None:
            parts.append(
                prev_action if prev_action.dim() == 3 else prev_action.unsqueeze(1)
            )
        else:
            parts.append(
                torch.zeros(
                    batch,
                    obs_dict[self.config.observation_keys[0]].shape[1],
                    self.config.control_dim,
                    device=obs_device,
                )
            )

        z_level = torch.cat(parts, dim=-1)

        # ---------------------------------------------------------------- #
        # Phase 1 — Bottom-up: raw representations (no top-down)
        # ---------------------------------------------------------------- #
        z_below_inputs: List[torch.Tensor] = []
        z_levels_raw: List[torch.Tensor] = []
        z_hat_next_list: List[torch.Tensor] = []
        pred_below_list: List[torch.Tensor] = []
        prediction_errors: List[torch.Tensor] = []
        h_new_list: List[torch.Tensor] = []
        h_target_new_list: List[torch.Tensor] = []

        curr = z_level
        for i, level in enumerate(self.levels):
            z_below_inputs.append(curr)
            z_seq, _, z_hat_next_seq, pred_below_seq, h_new, h_target_new, epsilon = (
                level.forward(curr, hidden_states[i], hidden_states_target[i], None)
            )
            h_new_list.append(h_new)
            h_target_new_list.append(h_target_new)
            z_levels_raw.append(z_seq)
            z_hat_next_list.append(z_hat_next_seq)
            pred_below_list.append(pred_below_seq)
            prediction_errors.append(epsilon)
            curr = z_seq

        # ---------------------------------------------------------------- #
        # Phase 2 — Top-down: correct lower levels from above
        # ---------------------------------------------------------------- #
        level_outputs = list(z_levels_raw)
        for i in range(len(self.levels) - 2, -1, -1):
            pred_from_above_seq = pred_below_list[i + 1]
            z_corr, z_hat_corr, pred_below_corr, eps_corr = self.levels[
                i
            ].apply_top_down_correction(
                z_levels_raw[i], z_below_inputs[i], pred_from_above_seq
            )
            level_outputs[i] = z_corr
            z_hat_next_list[i] = z_hat_corr
            pred_below_list[i] = pred_below_corr
            prediction_errors[i] = eps_corr

        ctrl_idx = self.control_level_idx
        z_ctrl = level_outputs[ctrl_idx]
        z_hat_ctrl = z_hat_next_list[ctrl_idx]
        action_pred = self.control_head(torch.cat([z_ctrl, z_hat_ctrl], dim=-1))

        if not return_all:
            return action_pred

        return {
            "action_pred": action_pred,
            "z_levels": level_outputs,
            "z_hat_next": z_hat_next_list,
            "prediction_errors": prediction_errors,
            "total_surprise": sum(e.pow(2).mean() for e in prediction_errors),
            "hidden_states": h_new_list,
            "hidden_states_target": h_target_new_list,
        }

    # ------------------------------------------------------------------ #
    # TorchScript / torch.compile acceleration
    # ------------------------------------------------------------------ #

    def compile(self, mode: str = "reduce-overhead"):
        """torch.compile all hot paths for online execution.

        Call once after construction, before any ticks.
        Tries inductor first, falls back to cudagraphs on CUDA, silently skips
        if neither works.
        Set ``mode="max-autotune"`` for highest throughput (one-time pause).
        """
        backends = _probe_compile_backends()
        if not backends:
            return

        backend = backends[0]
        for m in [*self.levels, self.control_head]:
            for attr in ("step", "forward"):
                fn = getattr(m, attr, None)
                if fn is None:
                    continue
                try:
                    setattr(m, attr, torch.compile(fn, dynamic=False, backend=backend))
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # EMA update
    # ------------------------------------------------------------------ #

    def update_ema(self, tau: float = 0.997):
        for level in self.levels:
            level.update_target_ema(tau)

    # ------------------------------------------------------------------ #
    # Multi-rate runner factory
    # ------------------------------------------------------------------ #

    def build_runner(
        self,
        level_frequencies: List[float],
        time_scale: float = 1.0,
        batch_size: int = 1,
        device: str = "cpu",
        accumulate_for_upper: bool = True,
        online_learning: bool = True,
        grad_accumulation_steps: int = 1,
    ) -> MultiRateRunner:
        if len(level_frequencies) != len(self.levels):
            raise ValueError(
                f"level_frequencies must have one entry per level "
                f"({len(self.levels)}), got {len(level_frequencies)}."
            )
        cfg = RunnerConfig(
            level_frequencies=level_frequencies,
            time_scale=time_scale,
            batch_size=batch_size,
            device=device,
            accumulate_for_upper=accumulate_for_upper,
            online_learning=online_learning,
            grad_accumulation_steps=grad_accumulation_steps,
            imitation_loss_weight=self.config.imitation_loss_weight,
        )
        return MultiRateRunner(self, cfg)
