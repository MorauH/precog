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
import copy

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

        # Target SSM - updated by EMA only
        self.ssm_target = copy.deepcopy(self.ssm)
        for p in self.ssm_target.parameters():
            p.requires_grad_(False)

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

    def init_hidden_target(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised target SSM hidden state. (batch, d_state)"""
        return self.ssm_target.init_hidden(batch_size, device)

    # -----------------------------------------------------------------------
    # Target update
    # -----------------------------------------------------------------------
    
    @torch.no_grad()
    def update_target_ema(self, tau: float = 0.997) -> None:
        """Call after every optimizer step."""
        for p_online, p_target in zip(
            self.ssm.parameters(),
            self.ssm_target.parameters()
        ):
            p_target.data = tau * p_target.data + (1 - tau) * p_online.data

    # -----------------------------------------------------------------------
    # Core forward interfaces
    # -----------------------------------------------------------------------
    
    def step(
        self,
        z_below: torch.Tensor,                      # (batch, d_below)
        h: torch.Tensor,                            # (batch, d_state) - online
        h_target: torch.Tensor,                     # (batch, d_state) - target
        pred_from_above: Optional[torch.Tensor],    # (batch, d_repr) | None
    ) -> Tuple[
        torch.Tensor,   # z_t          online representation  (batch, d_repr)
        torch.Tensor,   # z_t_target   target representation  (batch, d_repr)
        torch.Tensor,   # z_hat_next   forward prediction     (batch, d_repr)
        torch.Tensor,   # pred_below   downward prediction    (batch, d_below)
        torch.Tensor,   # h_new        updated hidden state   (batch, d_state)
        torch.Tensor,   # h_target_new updated hidden state   (batch, d_state)
        torch.Tensor,   # epsilon_below prediction error      (batch, d_below)
    ]:
        """
        Process a single timestep — used during online inference and
        step-by-step continuous learning on the car.
 
        JEPA integration
        ────────────────
        The step now runs two SSM paths in parallel:
 
          Online path  (self.ssm)        — receives gradients, drives learning.
          Target path  (self.ssm_target) — no gradients, EMA-updated only.
 
        predict_forward reads from z_t (online) and produces ẑ_{t+1}.
        The JEPA loss is computed *outside* this function by the training loop:
 
            jepa_loss_t = MSE(ẑ_{t+1},  z_target_{t+1}.detach())
                                ↑ from this step    ↑ from next step's z_t_target
 
        Gradients flow:  jepa_loss → predict_forward → online SSM
        No gradients to: target SSM (detach enforced by requires_grad=False + no_grad)
 
        PC integration
        ──────────────
        predict_downward and error propagation are unchanged. They operate on
        z_t (online) so they remain part of the gradient graph as before.
        The PC loss and JEPA loss are separate terms summed in the training loop.
 
        Args:
            z_below:         Representation from the level below at time t.
            h:               Online SSM hidden state from t-1.
            h_target:        Target SSM hidden state from t-1.
            pred_from_above: Top-down prediction of this level from level above.
                             None if this is the top level.
 
        Returns:
            z_t:           Online representation (used for PC + control).
            z_t_target:    Target representation (used as JEPA target at t+1).
            z_hat_next:    One-step-ahead prediction (JEPA source at t, loss at t+1).
            pred_below:    Top-down prediction of what z_below should be.
            h_new:         Updated online hidden state.
            h_target_new:  Updated target hidden state.
            epsilon_below: PC prediction error for the level below.
                           Magnitude is the runtime uncertainty signal.
        """ 
        # ── (a) Online path — gradient-enabled ────────────────────────────
        z_raw, h_new = self.ssm.step(z_below, h)               # (batch, d_repr)
 
        # ── (b) Target path — no gradients ────────────────────────────────
        #    z_t_target is the JEPA prediction target for the *previous*
        #    timestep's z_hat_next. The training loop pairs them as:
        #        loss += MSE(z_hat_next[t-1], z_t_target[t].detach())
        with torch.no_grad():
            z_t_target, h_target_new = self.ssm_target.step(
                z_below, h_target
            )                                                   # (batch, d_repr)
 
        # ── (c) Top-down PC error correction — online path only ───────────
        #    ε_above = z_raw - pred_from_above  (surprise at this level)
        #    z_t = z_raw + α · correction(ε_above)
        #    The target path is not modulated: it must remain a clean,
        #    stable encoding of the input for use as a JEPA target.
        if pred_from_above is not None and self.error_correction is not None:
            epsilon_above = z_raw - pred_from_above             # (batch, d_repr)
            correction = self.error_correction(epsilon_above)   # (batch, d_repr)
            z_t = z_raw + torch.tanh(self.alpha) * correction
        else:
            z_t = z_raw
 
        # ── (d) Predict downward — PC generative path ─────────────────────
        pred_below = self.predict_downward(z_t)                 # (batch, d_below)
 
        # ── (e) Predict forward — JEPA anticipatory path ──────────────────
        #    Gradients flow through predict_forward and back into the online
        #    SSM. z_t_target (next step) is the target — detached in the
        #    training loop, not here, so we can accumulate it freely.
        z_hat_next = self.predict_forward(z_t)                  # (batch, d_repr)
 
        # ── (f) PC prediction error for the level below ───────────────────
        #    z_below is detached so gradients flow through pred_below only,
        #    preserving the PC local learning rule.
        epsilon_below = z_below.detach() - pred_below           # (batch, d_below)
 
        return z_t, z_t_target, z_hat_next, pred_below, h_new, h_target_new, epsilon_below
    
    def forward(
        self,
        z_below_seq: torch.Tensor,                          # (batch, seq, d_below)
        h0: Optional[torch.Tensor] = None,                  # (batch, d_state)
        h0_target: Optional[torch.Tensor] = None,           # (batch, d_state)
        pred_from_above_seq: Optional[torch.Tensor] = None, # (batch, seq, d_repr)
    ) -> Tuple[
        torch.Tensor,   # z_seq             (batch, seq, d_repr)
        torch.Tensor,   # z_target_seq      (batch, seq, d_repr)
        torch.Tensor,   # z_hat_next_seq    (batch, seq, d_repr)
        torch.Tensor,   # pred_below_seq    (batch, seq, d_below)
        torch.Tensor,   # h_final           (batch, d_state)
        torch.Tensor,   # h_target_final    (batch, d_state)
        torch.Tensor,   # epsilon_below_seq (batch, seq, d_below)
    ]:
        """
        Process a full sequence — used during offline pre-training on logs.
 
        Iterates step() across time, accumulating all outputs.
 
        JEPA loss is computed by the training loop after this call:
 
            # Align: prediction at t should match target at t+1
            jepa_loss = MSE(
                z_hat_next_seq[:, :-1],       # predictions  t=0..T-2
                z_target_seq [:, 1: ].detach() # targets      t=1..T-1
            )
 
        h_final and h_target_final can warm-start the next segment,
        preserving continuity across log file boundaries or lap transitions.
        """
        batch_size, seq_len, _ = z_below_seq.shape
        device = z_below_seq.device
 
        h        = h0        if h0        is not None else self.init_hidden(batch_size, device)
        h_target = h0_target if h0_target is not None else self.init_hidden_target(batch_size, device)
 
        z_list, z_target_list, z_hat_next_list, pred_below_list, eps_list = [], [], [], [], []
 
        for t in range(seq_len):
            pfa = pred_from_above_seq[:, t] if pred_from_above_seq is not None else None
 
            z_t, z_t_target, z_hat_next, pred_below, h, h_target, eps = self.step(
                z_below_seq[:, t], h, h_target, pfa
            )
 
            z_list.append(z_t)
            z_target_list.append(z_t_target)
            z_hat_next_list.append(z_hat_next)
            pred_below_list.append(pred_below)
            eps_list.append(eps)
 
        return (
            torch.stack(z_list,          dim=1),
            torch.stack(z_target_list,   dim=1),
            torch.stack(z_hat_next_list, dim=1),
            torch.stack(pred_below_list, dim=1),
            h,
            h_target,
            torch.stack(eps_list,        dim=1),
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
