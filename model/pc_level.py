"""
Predictive Coding Level.

Each level in the PC hierarchy is responsible for four things:

  1. REPRESENT — run the input through the SSM to produce a representation z_t.
                 The input is the level below's representation, not raw sensors
                 (except for Level 1, which receives concatenated Level 0 outputs).

  2. PREDICT DOWNWARD — generate a top-down prediction of what the level below
                        should look like, from the current representation z_t.
                        This prediction is sent down and compared against the
                        actual Level below representation to form the PC error.

  3. PREDICT FORWARD — generate a one-step-ahead prediction ẑ_{t+1} of this
                       level's own representation. This is what the control
                       head reads from — the system acts from anticipated future
                       state rather than just current state.

  4. RECEIVE AND PROPAGATE ERROR — accept the prediction error from the level
                                   below (ε_below), and optionally the
                                   top-down prediction from the level above
                                   (pred_from_above), to modulate its own
                                   representation update via error_modulate().

──────────────────────────────────────────────
PC information flow (two adjacent levels)
──────────────────────────────────────────────

Level N+1 (higher, slower)
  │
  │  pred_from_above ──────────────────────────┐
  │  (top-down prediction of z_N)              │
  │                                            ▼
Level N ◄──── z_below (from Level N-1) ──► SSM ──► z_N
  │                                        ▲    │
  │  ε_below ──────────────────────────────┘    │
  │  (prediction error from Level N-1)          │
  │                                             ▼
  ├── predict_downward(z_N) ─────────────► pred_of_below
  │   sent to Level N-1 as pred_from_above      │
  │                                             ▼
  └── predict_forward(z_N)  ─────────────► ẑ_{N, t+1}
      read by control head if this is the       │
      designated control level                  ▼
                                           to Level N+1

──────────────────────────────────────────────
Error modulation
──────────────────────────────────────────────

The PC learning signal has two parts:

  ε_below = z_below - pred_of_below
      "How wrong was my prediction of the level below?"
      This is the upward-propagating PC error — the standard Rao & Ballard
      prediction error. It gates learning at this level.

  ε_above = z_N - pred_from_above
      "How wrong was the level above's prediction of me?"
      This modulates the representation: when the level above is surprised
      by this level, it nudges z_N toward resolving that surprise.

In the online setting, the magnitude of ε_below at runtime is used as an
uncertainty signal: high error → unfamiliar situation → increase learning
rate, reduce trust in control output.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import PCLevelConfig
from .fnn import FNN
from .ssm import SelectiveSSM


class PCLevel(nn.Module):
    """
    One level of the predictive coding hierarchy.

    Args:
        d_below:  Dimension of the level below's representation.
                  For Level 1 this is sum(modality output dims).
                  For Levels 2+ this is the previous level's d_representation.
        d_above:  Dimension of the level above's representation.
                  Used to size the top-down error correction projection.
                  Pass None for the highest level (no level above).
        config:   PCLevelConfig for this level.
    """

    def __init__(
        self,
        d_below: int,
        d_above: Optional[int],
        config: PCLevelConfig,
    ) -> None:
        super().__init__()

        self.d_below = d_below
        self.d_repr = config.d_representation
        self.d_above = d_above

        # ------------------------------------------------------------------
        # 1. REPRESENT — SSM processes the level below's representation
        # ------------------------------------------------------------------
        self.ssm = SelectiveSSM(
            d_input=d_below,
            d_output=config.d_representation,
            config=config.ssm,
        )

        # ------------------------------------------------------------------
        # 2. PREDICT DOWNWARD — top-down generative prediction
        #    z_t (this level) → predicted representation of level below
        #    Small FNN: the PC generative model
        # ------------------------------------------------------------------
        self.predict_downward = FNN(
            input_dim=config.d_representation,
            output_dim=d_below,
            hidden_dims=config.prediction_head_hidden,
        )

        # ------------------------------------------------------------------
        # 3. PREDICT FORWARD — anticipatory one-step-ahead prediction
        #    z_t (this level) → ẑ_{t+1} (this level, next step)
        #    Used by the control head: act from predicted future, not just now
        # ------------------------------------------------------------------
        self.predict_forward = FNN(
            input_dim=config.d_representation,
            output_dim=config.d_representation,
            hidden_dims=config.forward_head_hidden,
        )

        # ------------------------------------------------------------------
        # 4a. ERROR CORRECTION — top-down modulation
        #     When the level above predicts this level and is wrong, its
        #     error (ε_above = z_t - pred_from_above) is projected and
        #     added to this level's representation.
        #     This implements the bidirectional PC update:
        #       z_t ← z_t + α · proj(ε_above)
        # ------------------------------------------------------------------
        if d_above is not None:
            self.error_correction = FNN(
                input_dim=config.d_representation,   # ε_above has same dim as z_t
                output_dim=config.d_representation,
                hidden_dims=[],                       # linear correction by default
            )
        else:
            self.error_correction = None

        # ------------------------------------------------------------------
        # 4b. UNCERTAINTY SCALING — scalar gain on error correction
        #     Learned per-level sensitivity to top-down error.
        #     Initialised near zero so early training is bottom-up dominant.
        # ------------------------------------------------------------------
        self.alpha = nn.Parameter(torch.zeros(1))

    # -----------------------------------------------------------------------
    # Hidden state management — delegates to SSM
    # -----------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised SSM hidden state. (batch, d_state)"""
        return self.ssm.init_hidden(batch_size, device)

    # -----------------------------------------------------------------------
    # Core forward interfaces
    # -----------------------------------------------------------------------

    def step(
        self,
        z_below: torch.Tensor,                      # (batch, d_below)
        h: torch.Tensor,                             # (batch, d_state)
        pred_from_above: Optional[torch.Tensor],     # (batch, d_repr) | None
    ) -> Tuple[
        torch.Tensor,   # z_t          representation      (batch, d_repr)
        torch.Tensor,   # z_hat_next   forward prediction  (batch, d_repr)
        torch.Tensor,   # pred_below   downward prediction (batch, d_below)
        torch.Tensor,   # h_new        updated hidden state
        torch.Tensor,   # epsilon_below prediction error   (batch, d_below)
    ]:
        """
        Process a single timestep — used during online inference and
        step-by-step continuous learning on the car.

        The level:
          (a) Runs the SSM to produce raw representation z_raw.
          (b) Optionally modulates z_raw with top-down error correction.
          (c) Predicts the level below (downward).
          (d) Predicts its own next state (forward).
          (e) Returns the prediction error for the level below,
              which becomes the PC learning signal at this level
              and propagates upward as input to error_correction
              at the level above.

        Args:
            z_below:         Representation from the level below at time t.
            h:               SSM hidden state from time t-1.
            pred_from_above: Top-down prediction of this level's representation
                             from the level above. None if this is the top level.

        Returns:
            z_t:          This level's representation after error modulation.
            z_hat_next:   One-step-ahead prediction of this level's repr.
            pred_below:   Top-down prediction of what z_below should be.
            h_new:        Updated SSM hidden state.
            epsilon_below: Prediction error for the level below.
                           Shape (batch, d_below).
                           Magnitude used as uncertainty signal at runtime.
        """
        # (a) Bottom-up: run SSM on level below's representation
        z_raw, h_new = self.ssm.step(z_below, h)               # (batch, d_repr)

        # (b) Top-down error correction from level above
        #     ε_above = z_raw - pred_from_above  (surprise at this level)
        #     z_t = z_raw + α · correction(ε_above)
        if pred_from_above is not None and self.error_correction is not None:
            epsilon_above = z_raw - pred_from_above             # (batch, d_repr)
            correction = self.error_correction(epsilon_above)   # (batch, d_repr)
            z_t = z_raw + torch.tanh(self.alpha) * correction
        else:
            z_t = z_raw

        # (c) Predict downward: generative prediction of level below
        pred_below = self.predict_downward(z_t)                 # (batch, d_below)

        # (d) Predict forward: anticipatory prediction of own next state
        z_hat_next = self.predict_forward(z_t)                  # (batch, d_repr)

        # (e) Prediction error for the level below
        #     Detach z_below from the graph here: the gradient for the SSM
        #     flows through pred_below (generative path), not through z_below
        #     directly. This mirrors the PC local update rule.
        epsilon_below = z_below.detach() - pred_below           # (batch, d_below)

        return z_t, z_hat_next, pred_below, h_new, epsilon_below

    def forward(
        self,
        z_below_seq: torch.Tensor,                          # (batch, seq, d_below)
        h0: Optional[torch.Tensor] = None,                  # (batch, d_state)
        pred_from_above_seq: Optional[torch.Tensor] = None, # (batch, seq, d_repr)
    ) -> Tuple[
        torch.Tensor,   # z_seq          (batch, seq, d_repr)
        torch.Tensor,   # z_hat_next_seq (batch, seq, d_repr)
        torch.Tensor,   # pred_below_seq (batch, seq, d_below)
        torch.Tensor,   # h_final        (batch, d_state)
        torch.Tensor,   # epsilon_below_seq (batch, seq, d_below)
    ]:
        """
        Process a full sequence — used during offline pre-training on logs.

        Iterates step() across time, accumulating all outputs.
        h_final can warm-start online inference from the end of a log segment.
        """
        batch_size, seq_len, _ = z_below_seq.shape
        device = z_below_seq.device

        h = h0 if h0 is not None else self.init_hidden(batch_size, device)

        z_list, z_hat_next_list, pred_below_list, eps_list = [], [], [], []

        for t in range(seq_len):
            pfa = pred_from_above_seq[:, t] if pred_from_above_seq is not None else None

            z_t, z_hat_next, pred_below, h, eps = self.step(
                z_below_seq[:, t], h, pfa
            )

            z_list.append(z_t)
            z_hat_next_list.append(z_hat_next)
            pred_below_list.append(pred_below)
            eps_list.append(eps)

        return (
            torch.stack(z_list, dim=1),
            torch.stack(z_hat_next_list, dim=1),
            torch.stack(pred_below_list, dim=1),
            h,
            torch.stack(eps_list, dim=1),
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @property
    def uncertainty(self) -> float:
        """
        Convenience accessor: the learned top-down correction gain α.
        torch.tanh(alpha) ∈ (-1, 1); magnitude indicates how strongly
        this level is influenced by top-down error signals.
        Useful for inspecting whether the hierarchy is genuinely hierarchical
        after training.
        """
        return float(torch.tanh(self.alpha).item())
