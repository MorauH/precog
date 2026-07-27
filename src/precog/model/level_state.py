"""
LevelState
==========
Lightweight container for the live state of a single PCLevel during
multi-rate execution.

Separating state from the nn.Module means:
 * The model itself stays stateless between calls (easier to batch / save).
 * Stale outputs from slow levels are trivially available to fast levels
   without any special casing inside forward().
 * Episode resets, checkpointing, and parallel rollouts are clean operations
   on plain Python dataclasses + tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LevelState:
    """All mutable state for one PCLevel during online execution."""

    level_idx: int

    hidden: torch.Tensor                      # (B, d_state)

    last_z: Optional[torch.Tensor] = None          # (B, d_repr)
    last_pred_error: Optional[torch.Tensor] = None

    last_update_tick: int = 0
    last_update_sim_time: float = 0.0

    def update(
        self,
        hidden: torch.Tensor,
        z: torch.Tensor,
        pred_error: torch.Tensor,
        tick: int,
        sim_time: float,
    ):
        self.hidden = hidden
        self.last_z = z
        self.last_pred_error = pred_error
        self.last_update_tick = tick
        self.last_update_sim_time = sim_time

    def detach_(self):
        self.hidden = self.hidden.detach()
        if self.last_z is not None:
            self.last_z = self.last_z.detach()

    @classmethod
    def init(
        cls,
        level_idx: int,
        d_state: int,
        d_repr: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "LevelState":
        return cls(
            level_idx=level_idx,
            hidden=torch.zeros(batch_size, d_state, device=device, dtype=dtype),
            last_z=torch.zeros(batch_size, d_repr, device=device, dtype=dtype),
        )

    def __repr__(self) -> str:
        shape = tuple(self.last_z.shape) if self.last_z is not None else None
        return (
            f"LevelState(idx={self.level_idx}, "
            f"z_shape={shape}, "
            f"tick={self.last_update_tick})"
        )
