"""
model.py — HierarchicalPCWorldModel (multi-rate compatible)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .fnn import FNN
from .ssm import SelectiveSSM
from .pc_level import PCLevel
from .modality_encoder import ModalityEncoder
from .control_head import ControlHead
from .config import ModelConfig
from .multi_rate_runner import MultiRateRunner, RunnerConfig


class HierarchicalPCWorldModel(nn.Module):
    """
    Hierarchical Predictive Coding World Model with JEPA stabilisation.

    Architecture
    ~~~~~~~~~~~~
    Level 0  : Per-modality encoders (preserve sensor identity).
    Level 1+ : PCLevels (each contains SelectiveSSM + prediction heads).
    Control  : Head on top of the chosen mid-level (z_mid ‖ z_hat_mid).

    Multi-rate execution
    ~~~~~~~~~~~~~~~~~~~~
    For multi-frequency control, wrap the model with a MultiRateRunner::

        runner = model.build_runner(
            level_frequencies=[100, 10, 1],
            time_scale=1.0,
        )
        while True:
            result = runner.tick(obs_dict, prev_action)
            robot.send(result.action)
            runner.clock.sleep_until_next_tick()

    The model's `forward()` still works for training (full sequences, no
    rate scheduling).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ---------------------------------------------------------------- #
        # Level 0: Modality-specific encoders
        # ---------------------------------------------------------------- #
        self.modality_encoders: nn.ModuleDict = nn.ModuleDict()
        for mod in config.modalities:
            self.modality_encoders[mod.name] = ModalityEncoder(
                input_dim=mod.input_dim,
                output_dim=mod.output_dim,
                hidden_dim=getattr(mod, "hidden_dims", None),
            )

        # ---------------------------------------------------------------- #
        # PC Levels
        # ---------------------------------------------------------------- #
        self.levels: nn.ModuleList = nn.ModuleList()
        prev_dim = config.d_level0

        for i, level_cfg in enumerate(config.level_configs):
            self.levels.append(
                PCLevel(
                    d_below=prev_dim,
                    d_above= None if i == len(self.config.level_configs) else level_cfg.d_representation,
                    config=level_cfg
                )
            )
            prev_dim = level_cfg.d_representation

        # ---------------------------------------------------------------- #
        # Control Head
        # ---------------------------------------------------------------- #
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
        """Full-sequence forward pass for training.

        Parameters
        ----------
        obs_dict:
            Dict of (B, T, sensor_dim) tensors.
        prev_action:
            (B, T, action_dim) or (B, action_dim).
        hidden_states / hidden_states_target:
            Optional initial hidden states per level.  If None, zeros are
            used.  Pass the returned ``h_new`` tensors to chain sequences.
        return_all:
            If True, return a dict with all intermediate outputs.

        Returns
        -------
        action_pred (B, T, action_dim)  if return_all is False, else a dict.
        """
        batch = next(iter(obs_dict.values())).shape[0]

        # Initialise hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                level.init_hidden(batch)[0] for level in self.levels
            ]
        if hidden_states_target is None:
            hidden_states_target = [
                level.init_hidden(batch)[1] for level in self.levels
            ]

        # Level 0 encoding
        level0_list = []
        for name, encoder in self.modality_encoders.items():
            x = obs_dict[name]
            if name == "control" and prev_action is not None:
                x = (
                    prev_action.unsqueeze(1)
                    if prev_action.dim() == 2
                    else prev_action
                )
            level0_list.append(encoder(x))

        z_level = torch.cat(level0_list, dim=-1)  # (B, T, d_level0)

        # Hierarchical forward
        level_outputs: List[torch.Tensor] = []
        z_hat_next_list: List[torch.Tensor] = []
        prediction_errors: List[torch.Tensor] = []
        h_new_list: List[torch.Tensor] = []
        h_target_new_list: List[torch.Tensor] = []

        for i, level in enumerate(self.levels):
            pred_from_above = level_outputs[-1] if i > 0 else None

            z_seq, z_hat_next_seq, pred_below_seq, h_new, h_target_new, epsilon = (
                level.forward(
                    z_level,
                    hidden_states[i],
                    hidden_states_target[i],
                    pred_from_above,
                )
            )

            h_new_list.append(h_new[:, -1])
            h_target_new_list.append(h_target_new[:, -1])
            level_outputs.append(z_seq)
            z_hat_next_list.append(z_hat_next_seq)
            prediction_errors.append(epsilon)
            z_level = z_seq

        # Control head
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
            # Updated hidden states (for chaining across sequence chunks)
            "hidden_states": h_new_list,
            "hidden_states_target": h_target_new_list,
        }

    # ------------------------------------------------------------------ #
    # EMA update
    # ------------------------------------------------------------------ #

    def update_ema(self, tau: float = 0.997):
        """Update all target encoders (call after optimizer.step())."""
        for level in self.levels:
            level.update_ema(tau)

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
    ) -> MultiRateRunner:
        """Create a MultiRateRunner bound to this model.

        Parameters
        ----------
        level_frequencies:
            Hz for each level, index 0 = lowest/fastest.
            Must have one entry per level (len == len(self.levels)).
            E.g. [100, 10, 1] for a 3-level model.
        time_scale:
            Simulation speed multiplier.  1.0 = real-time.
        batch_size, device:
            Inference batch size and device for state tensors.
        accumulate_for_upper:
            See RunnerConfig.

        Returns
        -------
        MultiRateRunner ready for online stepping.

        Example
        -------
        >>> runner = model.build_runner([100, 10, 1], time_scale=5.0)
        >>> for obs, act in env:
        ...     result = runner.tick(obs, act)
        ...     robot.send(result.action)
        ...     runner.clock.sleep_until_next_tick()
        """
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
        )
        return MultiRateRunner(self, cfg)
