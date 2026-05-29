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

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LevelState:
    """All mutable state for one PCLevel during online execution.

    Parameters
    ----------
    level_idx:
        Which level this belongs to (0 = lowest / fastest).
    hidden:
        SSM hidden state, shape (B, d_state).
    hidden_target:
        EMA target network hidden state, shape (B, d_state).
    last_z:
        Most-recently-produced representation, shape (B, d_repr).
        Held constant between updates (zero-order hold).
    last_z_hat_next:
        Most-recently-predicted next representation, shape (B, d_repr).
        Also held constant.
    last_pred_error:
        Scalar prediction-error tensor from the last update (for logging).
    last_update_tick:
        Base-clock tick index when this level last ran.
    last_update_sim_time:
        Simulated time (seconds) when this level last ran.
    """
    level_idx: int

    # Hidden states
    hidden: torch.Tensor                      # (B, d_state)
    hidden_target: torch.Tensor               # (B, d_state)

    # Cached outputs (zero-order hold between updates)
    last_z: Optional[torch.Tensor] = None          # (B, d_repr)
    last_z_hat_next: Optional[torch.Tensor] = None  # (B, d_repr)
    last_pred_error: Optional[torch.Tensor] = None

    # Bookkeeping
    last_update_tick: int = 0
    last_update_sim_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def update(
        self,
        hidden: torch.Tensor,
        hidden_target: torch.Tensor,
        z: torch.Tensor,
        z_hat_next: torch.Tensor,
        pred_error: torch.Tensor,
        tick: int,
        sim_time: float,
    ):
        """Atomically update all fields after a forward pass."""
        self.hidden = hidden
        self.hidden_target = hidden_target
        self.last_z = z
        self.last_z_hat_next = z_hat_next
        self.last_pred_error = pred_error
        self.last_update_tick = tick
        self.last_update_sim_time = sim_time

    def detach_(self):
        """Detach all tensors in-place (call between episodes or gradient steps)."""
        self.hidden = self.hidden.detach()
        self.hidden_target = self.hidden_target.detach()
        if self.last_z is not None:
            self.last_z = self.last_z.detach()
        if self.last_z_hat_next is not None:
            self.last_z_hat_next = self.last_z_hat_next.detach()

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
        """Create a zero-initialised LevelState."""
        return cls(
            level_idx=level_idx,
            hidden=torch.zeros(batch_size, d_state, device=device, dtype=dtype),
            hidden_target=torch.zeros(batch_size, d_state, device=device, dtype=dtype),
            last_z=torch.zeros(batch_size, d_repr, device=device, dtype=dtype),
            last_z_hat_next=torch.zeros(batch_size, d_repr, device=device, dtype=dtype),
        )

    def __repr__(self) -> str:
        shape = tuple(self.last_z.shape) if self.last_z is not None else None
        return (
            f"LevelState(idx={self.level_idx}, "
            f"z_shape={shape}, "
            f"tick={self.last_update_tick})"
        )
