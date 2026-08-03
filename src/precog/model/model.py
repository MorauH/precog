from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .pc_level_jepa import PCLevel
from .control_head import ControlHead
from .config import ModelConfig
from .multi_rate_runner import MultiRateRunner, RunnerConfig


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
        control_input_dim = config.level_configs[
            config.control_level_idx
        ].d_representation

        self.control_head = ControlHead(
            input_dim=control_input_dim,
            hidden_dims=ctrl_cfg.hidden_dims,
            output_dim=ctrl_cfg.output_dim,
            output_scales=ctrl_cfg.output_scales or None,
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
        return_all: bool = False,
    ):
        batch = next(iter(obs_dict.values())).shape[0]
        obs_device = next(iter(obs_dict.values())).device

        if hidden_states is None:
            hidden_states = [
                level.init_hidden(batch, device=obs_device) for level in self.levels
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
        z_levels: List[torch.Tensor] = []
        prediction_errors: List[torch.Tensor] = []
        h_new_list: List[torch.Tensor] = []
        sigreg_losses: List[torch.Tensor] = []

        curr = z_level
        for i, level in enumerate(self.levels):
            z_seq, _, h_new, z_delta_seq, sigreg_loss, _ = level.forward(
                curr, hidden_states[i], None, None, None
            )
            h_new_list.append(h_new)
            z_levels.append(z_seq)
            prediction_errors.append(z_delta_seq)
            sigreg_losses.append(sigreg_loss)
            curr = z_seq

        ctrl_idx = self.control_level_idx
        z_ctrl = z_levels[ctrl_idx]
        action_pred = self.control_head(z_ctrl)

        if not return_all:
            return action_pred

        return {
            "action_pred": action_pred,
            "z_levels": z_levels,
            "prediction_errors": prediction_errors,
            "total_surprise": sum(e.pow(2).mean() for e in prediction_errors),
            "hidden_states": h_new_list,
            "sigreg_losses": sigreg_losses,
        }

    # ------------------------------------------------------------------ #
    # Weight accessors for inter-process sync
    # ------------------------------------------------------------------ #

    def get_weights(self, level_idx: int) -> dict[str, torch.Tensor]:
        """Return a CPU copy of the level's state_dict and its control head."""
        if level_idx < 0 or level_idx >= len(self.levels):
            raise ValueError(f"Invalid level_idx {level_idx}")
        state = {}
        for name, param in self.levels[level_idx].named_parameters():
            state[f"level.{name}"] = param.data.detach().cpu().clone()
        for name, buf in self.levels[level_idx].named_buffers():
            state[f"level.{name}"] = buf.data.detach().cpu().clone()
        if level_idx == self.control_level_idx:
            for name, param in self.control_head.named_parameters():
                state[f"control_head.{name}"] = param.data.detach().cpu().clone()
            for name, buf in self.control_head.named_buffers():
                state[f"control_head.{name}"] = buf.data.detach().cpu().clone()
        return state

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
        online_learning: bool = False,
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
            imitation_loss_weight=self.config.imitation_loss_weight,
        )
        return MultiRateRunner(self, cfg)
