"""
Predictive Coding Level.

Each level in the PC hierarchy:

  1. REPRESENT — SSM encodes the level below's representation → z_t.

  2. PREDICT DOWNWARD — top-down generative prediction of what the level
     below should look like.  Compared against actual z_below to form the
     PC prediction error ε_below.

  3. RECEIVE AND PROPAGATE ERROR — top-down error correction from the
     level above, plus (optionally) a task objective injected as a
     top-down signal.

──────────────────────────────────────────────
Objective injection (top-down task signal)
──────────────────────────────────────────────

When objective_enabled is True the level accepts an external task_target.
Two mechanisms work together:

  A. Top-down injection (forward — immediate behavioural shift)
       target_encoder(y) → z_obj  (scalar target → d_repr vector)
       ε_obj = z_raw - z_obj
       z_raw += α_obj·tanh · obj_correction(ε_obj)

  B. Translator + task loss (backward — permanent SSM learning)
       x* = translator(z_t)                (z_t → observable)
       loss_ae  = MSE(x*, x_actual)        (keep decoder faithful)
       loss_task = MSE(x*, task_target)    (shape SSM toward target)

Gradients from loss_task flow into z_t and the SSM.  The translator
receives both autoencoder and task gradients — the autoencoder weight
dominates to keep the decoder mapping accurate.

──────────────────────────────────────────────
PC information flow
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

──────────────────────────────────────────────
SIGReg
──────────────────────────────────────────────

SIGReg prevents representational collapse by regularizing the covariance
matrix of the online representations.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import PCLevelConfig
from .fnn import FNN
from .sigreg import SIGReg
from .ssm import SelectiveSSM


class PCLevel(nn.Module):
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
        self.objective_enabled = config.objective_enabled

        self.ssm = SelectiveSSM(
            d_input=d_below,
            d_output=config.d_representation,
            config=config.ssm,
        )

        self.sigreg = SIGReg(
            d_repr=config.d_representation,
            online_tau=config.sigreg_tau,
            var_threshold=config.sigreg_var_threshold,
        )

        self.predict_downward = FNN(
            input_dim=config.d_representation,
            output_dim=d_below,
            hidden_dims=config.prediction_head_hidden,
        )

        if d_above is not None:
            self.error_correction = FNN(
                input_dim=config.d_representation,
                output_dim=config.d_representation,
                hidden_dims=[],
            )
        else:
            self.error_correction = None

        self.alpha = nn.Parameter(torch.zeros(1))

        # ------------------------------------------------------------------
        # Objective injection — top-down task signal
        # ------------------------------------------------------------------
        if self.objective_enabled:
            self.target_encoder = FNN(1, config.d_representation, [])
            self.obj_correction = FNN(
                config.d_representation, config.d_representation, []
            )
            self.alpha_obj = nn.Parameter(torch.zeros(1))
            self.translator = FNN(
                config.d_representation, 1, config.translator_hidden
            )
            self._ae_weight = config.objective_ae_weight
            self._task_weight = config.objective_task_weight
        else:
            self.target_encoder = None
            self.obj_correction = None
            self.alpha_obj = None
            self.translator = None
            self._ae_weight = 0.0
            self._task_weight = 0.0

    # -----------------------------------------------------------------------
    # Hidden state
    # -----------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.ssm.init_hidden(batch_size, device)

    # -----------------------------------------------------------------------
    # Top-down correction (refinement pass, no SSM re-run)
    # -----------------------------------------------------------------------

    def apply_top_down_correction(
        self,
        z_raw_seq: torch.Tensor,
        z_below_seq: torch.Tensor,
        pred_from_above_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.error_correction is None:
            return (
                z_raw_seq,
                self.predict_downward(
                    z_raw_seq.reshape(-1, self.d_repr)
                ).reshape(*z_raw_seq.shape[:-1], self.d_below),
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
        task_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Process a single timestep.

        Args:
            z_below:         Representation from the level below at time t.
            h:               SSM hidden state from t-1.
            pred_from_above: Top-down prediction from level above (None if top).
            task_target:     Target value for objective (B, 1) or None.
                             Only used when objective_enabled=True.

        Returns:
            z_t:           Representation (used for PC + control).
            pred_below:    Top-down prediction of what z_below should be.
            h_new:         Updated hidden state.
            epsilon_below: PC prediction error.
            x_star:        Translator output (B, 1) or None if not enabled.
        """
        z_raw, h_new = self.ssm.step(z_below, h)

        self.sigreg.update_online(z_raw.detach().squeeze(0))

        # ── Objective top-down injection ──────────────────────────────────
        if task_target is not None and self.objective_enabled:
            z_obj = self.target_encoder(task_target)
            epsilon_obj = z_raw - z_obj
            correction_obj = self.obj_correction(epsilon_obj)
            z_raw = z_raw + torch.tanh(self.alpha_obj) * correction_obj

        # ── PC top-down correction ────────────────────────────────────────
        if pred_from_above is not None and self.error_correction is not None:
            epsilon_above = z_raw - pred_from_above
            correction = self.error_correction(epsilon_above)
            z_t = z_raw + torch.tanh(self.alpha) * correction
        else:
            z_t = z_raw

        pred_below = self.predict_downward(z_t)
        epsilon_below = z_below.detach() - pred_below

        # ── Translator output ─────────────────────────────────────────────
        x_star = self.translator(z_t) if self.objective_enabled else None

        return (z_t, pred_below, h_new, epsilon_below, x_star)

    def forward(
        self,
        z_below_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        pred_from_above_seq: Optional[torch.Tensor] = None,
        task_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Process a full sequence — used for upper-level accumulation
        or batch processing.

        Returns:
            z_seq, pred_below_seq, h_final, epsilon_below_seq,
            sigreg_loss, x_star_seq
        """
        batch_size, seq_len, _ = z_below_seq.shape
        device = z_below_seq.device

        h = h0 if h0 is not None else self.init_hidden(batch_size, device)

        z_list, pred_below_list, eps_list, x_list = [], [], [], []

        for t in range(seq_len):
            pfa = (
                pred_from_above_seq[:, t]
                if pred_from_above_seq is not None
                else None
            )

            z_t, pred_below, h, eps, x_star = self.step(
                z_below_seq[:, t], h, pfa, task_target
            )

            z_list.append(z_t)
            pred_below_list.append(pred_below)
            eps_list.append(eps)
            if x_star is not None:
                x_list.append(x_star)

        z_seq = torch.stack(z_list, dim=1)

        x_star_seq = (
            torch.stack(x_list, dim=1) if x_list else None
        )

        return (
            z_seq,
            torch.stack(pred_below_list, dim=1),
            h,
            torch.stack(eps_list, dim=1),
            self.sigreg.batch_loss(z_seq),
            x_star_seq,
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @property
    def uncertainty(self) -> float:
        return float(torch.tanh(self.alpha).item())

    @property
    def ae_weight(self) -> float:
        return self._ae_weight

    @property
    def task_weight(self) -> float:
        return self._task_weight
