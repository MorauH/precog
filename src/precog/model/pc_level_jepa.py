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

──────────────────────────────────────────────
JEPA + SIGReg
──────────────────────────────────────────────

This level uses SIGReg (Sigma Regularization) instead of a separate EMA
target network. SIGReg prevents representational collapse by regularizing
the covariance matrix of the online representations.

For JEPA, the prediction target is simply the online encoder's output from
the next timestep, detached:  z[t+1].detach().

  jepa_loss_t = MSE(ẑ_{t+1},  z_{t+1}.detach())
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
        # SIGReg — prevents representational collapse (replaces EMA target)
        # ------------------------------------------------------------------
        self.sigreg = SIGReg(
            d_repr=config.d_representation,
            online_tau=config.sigreg_tau,
            var_threshold=config.sigreg_var_threshold,
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
                input_dim=config.d_representation,  # ε_above has same dim as z_t
                output_dim=config.d_representation,
                hidden_dims=[],  # linear correction by default
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
    # Top-down correction (refinement pass, no SSM re-run)
    # -----------------------------------------------------------------------

    def apply_top_down_correction(
        self,
        z_raw_seq: torch.Tensor,  # (B, T, d_repr) from pass 1
        z_below_seq: torch.Tensor,  # (B, T, d_below) original input
        pred_from_above_seq: torch.Tensor,  # (B, T, d_repr) from level above
    ) -> Tuple[
        torch.Tensor,  # z_corr_seq         (B, T, d_repr)
        torch.Tensor,  # z_hat_next_seq     (B, T, d_repr)
        torch.Tensor,  # pred_below_seq     (B, T, d_below)
        torch.Tensor,  # epsilon_below_seq  (B, T, d_below)
    ]:
        """Apply top-down PC correction without re-running the SSM.

        Called during the top-down refinement pass after all levels have
        computed their raw representations.  Re-applies:
            z_corr = z_raw + α·tanh · correction(z_raw - pred_from_above)
        and recomputes downstream heads (predict_forward, predict_downward,
        epsilon_below).

        If error_correction is None (top level, unused), z_corr == z_raw.
        """
        if self.error_correction is None:
            return (
                z_raw_seq,
                self.predict_forward(z_raw_seq.reshape(-1, self.d_repr)).reshape(
                    z_raw_seq.shape
                ),
                self.predict_downward(z_raw_seq.reshape(-1, self.d_repr)).reshape(
                    *z_raw_seq.shape[:-1], self.d_below
                ),
                torch.zeros_like(z_below_seq),
            )

        seq_len = z_raw_seq.shape[1]
        z_corr_list, z_hat_next_list, pred_below_list, eps_list = [], [], [], []

        for t in range(seq_len):
            z_raw = z_raw_seq[:, t]  # (B, d_repr)
            pfa = pred_from_above_seq[:, t]  # (B, d_repr)
            z_below = z_below_seq[:, t]  # (B, d_below)

            epsilon_above = z_raw - pfa
            correction = self.error_correction(epsilon_above)
            z_corr = z_raw + torch.tanh(self.alpha) * correction

            z_corr_list.append(z_corr)
            z_hat_next_list.append(self.predict_forward(z_corr))
            pred_below_list.append(self.predict_downward(z_corr))
            eps_list.append(z_below.detach() - pred_below_list[-1])

        return (
            torch.stack(z_corr_list, dim=1),
            torch.stack(z_hat_next_list, dim=1),
            torch.stack(pred_below_list, dim=1),
            torch.stack(eps_list, dim=1),
        )

    # -----------------------------------------------------------------------
    # Core forward interfaces
    # -----------------------------------------------------------------------

    def step(
        self,
        z_below: torch.Tensor,  # (batch, d_below)
        h: torch.Tensor,  # (batch, d_state)
        pred_from_above: Optional[torch.Tensor],  # (batch, d_repr) | None
    ) -> Tuple[
        torch.Tensor,  # z_t          representation           (batch, d_repr)
        torch.Tensor,  # z_hat_next   forward prediction       (batch, d_repr)
        torch.Tensor,  # pred_below   downward prediction      (batch, d_below)
        torch.Tensor,  # h_new        updated hidden state     (batch, d_state)
        torch.Tensor,  # epsilon_below prediction error        (batch, d_below)
    ]:
        """
        Process a single timestep — used during online inference.

        SIGReg integration
        ─────────────────
        The step runs a single SSM path. SIGReg replaces the EMA target
        network by maintaining an online covariance estimate that prevents
        representational collapse.  The estimate is updated here; the loss
        is computed periodically by the runner.

        predict_forward reads from z_t and produces ẑ_{t+1}.
        For JEPA the training loop pairs:

            jepa_loss_t = MSE(ẑ_{t+1},  z_{t+1}.detach())

        No separate target network is needed — SIGReg keeps z_t healthy.

        Args:
            z_below:         Representation from the level below at time t.
            h:               SSM hidden state from t-1.
            pred_from_above: Top-down prediction of this level from level above.
                             None if this is the top level.

        Returns:
            z_t:           Representation (used for PC + control).
            z_hat_next:    One-step-ahead prediction.
            pred_below:    Top-down prediction of what z_below should be.
            h_new:         Updated hidden state.
            epsilon_below: PC prediction error for the level below.
                           Magnitude is the runtime uncertainty signal.
        """
        z_raw, h_new = self.ssm.step(z_below, h)  # (batch, d_repr)

        # ── SIGReg — update online covariance estimate (detached) ─────────
        self.sigreg.update_online(z_raw.detach().squeeze(0))

        # ── Top-down PC error correction ──────────────────────────────────
        if pred_from_above is not None and self.error_correction is not None:
            epsilon_above = z_raw - pred_from_above  # (batch, d_repr)
            correction = self.error_correction(epsilon_above)  # (batch, d_repr)
            z_t = z_raw + torch.tanh(self.alpha) * correction
        else:
            z_t = z_raw

        # ── Predict downward — PC generative path ─────────────────────────
        pred_below = self.predict_downward(z_t)  # (batch, d_below)

        # ── Predict forward — JEPA anticipatory path ──────────────────────
        z_hat_next = self.predict_forward(z_t)  # (batch, d_repr)

        # ── PC prediction error for the level below ───────────────────────
        epsilon_below = z_below.detach() - pred_below  # (batch, d_below)

        return (
            z_t,
            z_hat_next,
            pred_below,
            h_new,
            epsilon_below,
        )

    def forward(
        self,
        z_below_seq: torch.Tensor,  # (batch, seq, d_below)
        h0: Optional[torch.Tensor] = None,  # (batch, d_state)
        pred_from_above_seq: Optional[torch.Tensor] = None,  # (batch, seq, d_repr)
    ) -> Tuple[
        torch.Tensor,  # z_seq             (batch, seq, d_repr)
        torch.Tensor,  # z_hat_next_seq    (batch, seq, d_repr)
        torch.Tensor,  # pred_below_seq    (batch, seq, d_below)
        torch.Tensor,  # h_final           (batch, d_state)
        torch.Tensor,  # epsilon_below_seq (batch, seq, d_below)
        torch.Tensor,  # sigreg_loss       scalar
    ]:
        """
        Process a full sequence — used for upper levels that accumulate
        slower-rate inputs or for batch processing.

        Iterates step() across time, accumulating all outputs.

        SIGReg loss is computed from the sequence batch and returned for
        addition to the training loss.

        h_final can warm-start the next segment, preserving continuity
        across log file boundaries or lap transitions.
        """
        batch_size, seq_len, _ = z_below_seq.shape
        device = z_below_seq.device

        h = h0 if h0 is not None else self.init_hidden(batch_size, device)

        z_list, z_hat_next_list, pred_below_list, eps_list = (
            [],
            [],
            [],
            [],
        )

        for t in range(seq_len):
            pfa = pred_from_above_seq[:, t] if pred_from_above_seq is not None else None

            z_t, z_hat_next, pred_below, h, eps = self.step(
                z_below_seq[:, t], h, pfa
            )

            z_list.append(z_t)
            z_hat_next_list.append(z_hat_next)
            pred_below_list.append(pred_below)
            eps_list.append(eps)

        z_seq = torch.stack(z_list, dim=1)

        return (
            z_seq,
            torch.stack(z_hat_next_list, dim=1),
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
        """
        Convenience accessor: the learned top-down correction gain α.
        torch.tanh(alpha) ∈ (-1, 1); magnitude indicates how strongly
        this level is influenced by top-down error signals.
        Useful for inspecting whether the hierarchy is genuinely hierarchical
        after training.
        """
        return float(torch.tanh(self.alpha).item())
