"""
Selective State Space Model (Mamba-inspired).

Implements the core recurrence:
    h_t = Ā(x_t) ⊙ h_{t-1}  +  B̄(x_t) ⊙ x_proj(x_t)
    y_t = C(x_t) · h_t

Where Ā and B̄ are derived from an input-dependent timescale Δ and
a diagonal, stable state matrix A:

    A   = -exp(A_log)                 (always negative → stable)
    Δ   = softplus(δ_proj(x))         (always positive timescale)
    Ā   = exp(Δ ⊙ A)                  (ZOH discretization)
    B̄   = Δ ⊙ B_proj(x)              (simplified ZOH)
    C   = C_proj(x)                   (input-dependent read-out)

The "selective" property: Δ, B, C are all functions of the input.
This means the model learns *what to remember and what to ignore*
based on content, rather than using a fixed recurrence like an RNN.

This is implemented for step-by-step (streaming) inference, which is
the mode used during online learning on the car. A batched forward()
method is also provided for offline pre-training on log sequences.

Biological plausibility note:
    The PC framework (hierarchy + error propagation) provides the
    biologically-motivated structure. The SSM is a pragmatic choice
    for the prediction function within each PC level — efficient,
    stable, and well-suited to temporal sensor data.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SSMConfig


class SelectiveSSM(nn.Module):
    """
    Args:
        d_input:  Dimension of input at this level (Level 0 concatenation
                  for Level 1; level representation dim for Levels 2-3).
        d_output: Dimension of the output representation for this level.
        config:   SSMConfig carrying d_state and timescale bounds.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        config: SSMConfig,
    ) -> None:
        super().__init__()

        self.d_input = d_input
        self.d_state = config.d_state
        self.d_output = d_output

        # ------------------------------------------------------------------
        # State matrix A — diagonal, log-parameterized
        # Initialized so |A| spans [5, 5*N] giving A_bar in [0.5, 0.995]
        # across the delta range.  Five-fold upscale prevents unbounded
        # state growth seen with the standard [1, N] range.
        # ------------------------------------------------------------------
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32) * 5)
        )

        # ------------------------------------------------------------------
        # Selective projections — all input-dependent
        # ------------------------------------------------------------------
        # B: how strongly the current input drives state updates
        self.B_proj = nn.Linear(d_input, config.d_state, bias=False)

        # C: how to read the current state into the output
        self.C_proj = nn.Linear(d_input, config.d_state, bias=False)

        # Δ (delta): input-dependent timescale, one per state dimension
        # Bias initialized so that softplus(bias) ≈ uniform in [dt_min, dt_max]
        self.delta_proj = nn.Linear(d_input, config.d_state, bias=True)
        nn.init.uniform_(
            self.delta_proj.bias,
            math.log(config.dt_min),
            math.log(config.dt_max),
        )

        # ------------------------------------------------------------------
        # Input pre-processing and output projection
        # ------------------------------------------------------------------
        # Expand input before selective projections (as in Mamba)
        self.in_proj = nn.Linear(d_input, d_input)

        # Project full state to output representation dim
        self.out_proj = nn.Linear(config.d_state, d_output)

        # Output normalisation: keeps representations well-scaled across
        # levels, important for stable PC error signals
        self.norm = nn.LayerNorm(d_output)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero-initialised hidden state. (batch, d_state)"""
        return torch.zeros(batch_size, self.d_state, device=device)

    @property
    def A(self) -> torch.Tensor:
        """Diagonal stable state matrix (negative real)."""
        return -torch.exp(self.A_log)

    def step(
        self,
        x: torch.Tensor,  # (batch, d_input)
        h: torch.Tensor,  # (batch, d_state)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single timestep. Used during online inference and
        step-by-step online learning on the car.

        Returns:
            y:     Output representation  (batch, d_output)
            h_new: Updated hidden state   (batch, d_state)
        """
        x = self.in_proj(x)  # (batch, d_input)

        delta = F.softplus(self.delta_proj(x))  # (batch, d_state)
        A_bar = torch.exp(delta * self.A).clamp(max=0.99)  # (batch, d_state)
        B_bar = delta * self.B_proj(x)  # (batch, d_state)

        h_new = A_bar * h + B_bar  # (batch, d_state)

        C = self.C_proj(x)  # (batch, d_state)
        y = self.out_proj(C * h_new)  # (batch, d_output)
        y = self.norm(y)

        return y, h_new

    def forward(
        self,
        x_seq: torch.Tensor,  # (batch, seq_len, d_input)
        h0: Optional[torch.Tensor] = None,  # (batch, d_state)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full sequence. Used during offline pre-training on
        Foxglove log data.

        Returns:
            outputs:  (batch, seq_len, d_output)
            h_final:  (batch, d_state)  — can be used to warm-start
                      online inference from the end of a log segment.
        """
        batch_size, seq_len, _ = x_seq.shape

        h = h0 if h0 is not None else self.init_hidden(batch_size, x_seq.device)

        outputs = []
        for t in range(seq_len):
            y, h = self.step(x_seq[:, t], h)
            outputs.append(y)

        return torch.stack(outputs, dim=1), h
