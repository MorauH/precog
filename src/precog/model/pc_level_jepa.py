"""
Predictive Coding Level.

Each level in the PC hierarchy is responsible for three things:

  1. REPRESENT — run the input through the SSM to produce a representation z_t.
                 The input is the level below's representation, not raw sensors
                 (except for Level 1, which receives concatenated Level 0 outputs).

  2. PREDICT DOWNWARD — generate a top-down prediction of what the level below
                        should look like, from the current representation z_t.
                        This prediction is sent down and compared against the
                        actual Level below representation to form the PC error.

  3. RECEIVE AND PROPAGATE ERROR — accept the prediction error from the level
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
  └── predict_downward(z_N) ─────────────► pred_of_below
      sent to Level N-1 as pred_from_above

The SSM recurrence inherently models temporal dynamics.
No separate forward-prediction head is needed — the hidden
state h → h' already captures "where things are going."

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

──────────────────────────────────────────────
SIGReg
──────────────────────────────────────────────

SIGReg (Sigma Regularization) prevents representational collapse by
regularizing the covariance matrix of the online representations.
It replaces the previous EMA target network approach.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import PCLevelConfig
from .fnn import FNN
from .sigreg import SIGReg
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
        # SIGReg — prevents representational collapse
        # ------------------------------------------------------------------
        self.sigreg = SIGReg(
            d_repr=config.d_representation,
            online_tau=config.sigreg_tau,
            var_threshold=config.sigreg_var_threshold,
        )

        # ------------------------------------------------------------------
        # 2. PREDICT DOWNWARD — top-down generative prediction
        #    z_t (this level) → predicted representation of level below
        # ------------------------------------------------------------------
        self.predict_downward = FNN(
            input_dim=config.d_representation,
            output_dim=d_below,
            hidden_dims=config.prediction_head_hidden,
        )

        # ------------------------------------------------------------------
        # 3a. ERROR CORRECTION — top-down modulation
        #     When the level above predicts this level and is wrong, its
        #     error (ε_above = z_t - pred_from_above) is projected and
        #     added to this level's representation.
        # ------------------------------------------------------------------
        if d_above is not None:
            self.error_correction = FNN(
                input_dim=config.d_representation,
                output_dim=config.d_representation,
                hidden_dims=[],
            )
        else:
            self.error_correction = None

        # ------------------------------------------------------------------
        # 3b. UNCERTAINTY SCALING — scalar gain on error correction
        #     Learned per-level sensitivity to top-down error.
        #     Initialised near zero so early training is bottom-up dominant.
        # ------------------------------------------------------------------
        self.alpha = nn.Parameter(torch.zeros(1))

    # -----------------------------------------------------------------------
    # Hidden state management
    # -----------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised SSM hidden state. (batch, d_state)"""
        return self.ssm.init_hidden(batch_size, device)

    # -----------------------------------------------------------------------
    # Top-down correction (refinement pass, no SSM re-run)
    # -----------------------------------------------------------------------

    def apply_top_down_correction(
        self,
        z_raw_seq: torch.Tensor,
        z_below_seq: torch.Tensor,
        pred_from_above_seq: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply top-down PC correction without re-running the SSM.

        Called during the top-down refinement pass after all levels have
        computed their raw representations.  Re-applies:
            z_corr = z_raw + α·tanh · correction(z_raw - pred_from_above)
        and recomputes predict_downward + epsilon_below.

        If error_correction is None (top level), z_corr == z_raw.
        """
        if self.error_correction is None:
            return (
                z_raw_seq,
                self.predict_downward(z_raw_seq.reshape(-1, self.d_repr)).reshape(
                    *z_raw_seq.shape[:-1], self.d_below
                ),
                torch.zeros_like(z_below_seq),
            )

        seq_len = z_raw_seq.shape[1]
        z_corr_list, pred_below_list, eps_list = [], [], []

        for t in range(seq_len):
            z_raw = z_raw_seq[:, t]
            pfa = pred_from_above_seq[:, t]
            z_below = z_below_seq[:, t]

            epsilon_above = z_raw - pfa
            correction = self.error_correction(epsilon_above)
            z_corr = z_raw + torch.tanh(self.alpha) * correction

            z_corr_list.append(z_corr)
            pred_below_list.append(self.predict_downward(z_corr))
            eps_list.append(z_below.detach() - pred_below_list[-1])

        return (
            torch.stack(z_corr_list, dim=1),
            torch.stack(pred_below_list, dim=1),
            torch.stack(eps_list, dim=1),
        )

    # -----------------------------------------------------------------------
    # Core forward interfaces
    # -----------------------------------------------------------------------

    def step(
        self,
        z_below: torch.Tensor,
        h: torch.Tensor,
        pred_from_above: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Process a single timestep — used during online inference.

        SIGReg updates the online covariance estimate to prevent collapse.

        Args:
            z_below:         Representation from the level below at time t.
            h:               SSM hidden state from t-1.
            pred_from_above: Top-down prediction from level above (None if top).

        Returns:
            z_t:           Representation (used for PC + control).
            pred_below:    Top-down prediction of what z_below should be.
            h_new:         Updated hidden state.
            epsilon_below: PC prediction error for the level below.
        """
        z_raw, h_new = self.ssm.step(z_below, h)

        self.sigreg.update_online(z_raw.detach().squeeze(0))

        if pred_from_above is not None and self.error_correction is not None:
            epsilon_above = z_raw - pred_from_above
            correction = self.error_correction(epsilon_above)
            z_t = z_raw + torch.tanh(self.alpha) * correction
        else:
            z_t = z_raw

        pred_below = self.predict_downward(z_t)
        epsilon_below = z_below.detach() - pred_below

        return (z_t, pred_below, h_new, epsilon_below)

    def forward(
        self,
        z_below_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        pred_from_above_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Process a full sequence — used for upper levels that accumulate
        slower-rate inputs or for batch processing.

        SIGReg loss is computed from the sequence batch and returned for
        addition to the training loss.

        h_final can warm-start the next segment.
        """
        batch_size, seq_len, _ = z_below_seq.shape
        device = z_below_seq.device

        h = h0 if h0 is not None else self.init_hidden(batch_size, device)

        z_list, pred_below_list, eps_list = [], [], []

        for t in range(seq_len):
            pfa = pred_from_above_seq[:, t] if pred_from_above_seq is not None else None

            z_t, pred_below, h, eps = self.step(z_below_seq[:, t], h, pfa)

            z_list.append(z_t)
            pred_below_list.append(pred_below)
            eps_list.append(eps)

        z_seq = torch.stack(z_list, dim=1)

        return (
            z_seq,
            torch.stack(pred_below_list, dim=1),
            h,
            torch.stack(eps_list, dim=1),
            self.sigreg.batch_loss(z_seq),
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @property
    def uncertainty(self) -> float:
        return float(torch.tanh(self.alpha).item())
