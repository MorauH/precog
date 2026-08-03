"""
JEPA Predictive Coding Level.

Each level in the JEPA-PC hierarchy:

  1. ENCODE — signal from below passes through the encoder to produce
     the representation z_t.

  2. PREDICT — SSM takes (z_t, a_t) and produces z_t+1_pred: the
     prediction of what z will be at the next timestep.

  3. COMPUTE ERROR — z_delta = z_t - z_t_pred (where z_t_pred is the
     SSM output saved from the previous timestep) is the prediction
     error used for training.

──────────────────────────────────────────────
JEPA information flow
──────────────────────────────────────────────

Level N+1 (higher, slower)
  │
  │  a_t (SSM prediction from Level N+1) ───────────┐
  │                                                  ▼
Level N ◄── signal_from_below ──► encoder ──► z_t ──► SSM ──► z_t+1_pred
  │  (from Level N-1)                       │       ▲              │
  │                                         │       │              │
  │  z_delta ───────────────────────────────┘       a_t            │
  │  (prediction error, sent upward)         (from above)          │
  │                                                                ▼
  └── z_t+1_pred ──────────────────────────────────► sent downward
      saved as z_t_pred for next step

──────────────────────────────────────────────
Objective injection (top-down task signal)
──────────────────────────────────────────────

When objective_enabled is True the level accepts an external task_target.
Two mechanisms work together:

  A. Top-down injection (forward — immediate behavioural shift)
       target_encoder(y) → z_obj  (scalar target → d_repr vector)
       epsilon_obj = z_t - z_obj
       z_t += alpha_obj * tanh · obj_correction(epsilon_obj)

  B. Translator + task loss (backward — permanent SSM learning)
       x* = translator(z_t)                (z_t → observable)
       loss_ae  = MSE(x*, x_actual)        (keep decoder faithful)
       loss_task = MSE(x*, task_target)    (shape SSM toward target)

──────────────────────────────────────────────
SIGReg
──────────────────────────────────────────────

SIGReg prevents representational collapse of the encoder output z_t
by regularizing the covariance matrix of the online representations.
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

        # Encoder: signal from below → representation z_t
        self.encoder = FNN(
            input_dim=d_below,
            output_dim=config.d_representation,
            hidden_dims=config.encoder_hidden,
        )

        # SSM: predicts next z_t from (z_t, a_t)
        ssm_input_dim = config.d_representation + (d_above or 0)
        self.ssm = SelectiveSSM(
            d_input=ssm_input_dim,
            d_output=config.d_representation,
            config=config.ssm,
        )

        # SIGReg: regularize encoder output z_t
        self.sigreg = SIGReg(
            d_repr=config.d_representation,
            online_tau=config.sigreg_tau,
            var_threshold=config.sigreg_var_threshold,
        )

        # ------------------------------------------------------------------
        # Objective injection — top-down task signal
        # ------------------------------------------------------------------
        if self.objective_enabled:
            self.target_encoder = FNN(1, config.d_representation, [])
            self.obj_correction = FNN(
                config.d_representation, config.d_representation, []
            )
            self.alpha_obj = nn.Parameter(torch.zeros(1))
            self.translator = FNN(config.d_representation, 1, config.translator_hidden)
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
    # Core forward interfaces
    # -----------------------------------------------------------------------

    def step(
        self,
        signal_from_below: torch.Tensor,
        h: torch.Tensor,
        a_t: Optional[torch.Tensor] = None,
        z_t_pred: Optional[torch.Tensor] = None,
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
            signal_from_below: Raw signal from the level below (or encoded obs
                               for Level 0) at time t.
            h:                 SSM hidden state from t-1.
            a_t:               Top-down prediction from level above
                               (SSM output z_t+1_pred of above level).
                               None if this is the top level.
            z_t_pred:          SSM prediction from the PREVIOUS timestep
                               (saved z_t+1_pred from t-1). Used to compute
                               the prediction error z_delta.
                               None on the very first step (treated as zeros).
            task_target:       Target value for objective (B, 1) or None.
                               Only used when objective_enabled=True.

        Returns:
            z_t:           Encoder output (B, d_repr) — upward signal.
            z_next_pred:   SSM prediction (B, d_repr) — downward signal,
                           saved as z_t_pred for the next step.
            h_new:         Updated SSM hidden state (B, d_state).
            z_delta:       Prediction error (B, d_repr):
                           z_t - z_t_pred. Used for training.
            x_star:        Translator output (B, 1) or None if not enabled.
        """
        # 1. Encode signal from below
        z_t = self.encoder(signal_from_below)

        self.sigreg.update_online(z_t.detach().squeeze(0))

        # 2. Objective top-down injection (modifies z_t)
        if task_target is not None and self.objective_enabled:
            target_encoder = self.target_encoder
            obj_correction = self.obj_correction
            alpha_obj = self.alpha_obj
            assert target_encoder is not None
            assert obj_correction is not None
            assert alpha_obj is not None
            z_obj = target_encoder(task_target)
            epsilon_obj = z_t - z_obj
            correction_obj = obj_correction(epsilon_obj)
            z_t = z_t + torch.tanh(alpha_obj) * correction_obj

        # 3. Build SSM input: concat(z_t, a_t)
        if self.d_above is not None:
            if a_t is not None:
                ssm_input = torch.cat([z_t, a_t], dim=-1)
            else:
                a_t_zeros = torch.zeros(
                    z_t.shape[0], self.d_above, device=z_t.device, dtype=z_t.dtype
                )
                ssm_input = torch.cat([z_t, a_t_zeros], dim=-1)
        else:
            ssm_input = z_t

        # 4. SSM predicts next representation
        z_next_pred, h_new = self.ssm.step(ssm_input, h)

        # 5. Compute prediction error
        if z_t_pred is not None:
            z_delta = z_t - z_t_pred
        else:
            z_delta = torch.zeros_like(z_t)

        # 6. Translator output
        if self.objective_enabled:
            translator = self.translator
            assert translator is not None
            x_star = translator(z_t)
        else:
            x_star = None

        return (z_t, z_next_pred, h_new, z_delta, x_star)

    def forward(
        self,
        signal_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        a_seq: Optional[torch.Tensor] = None,
        z_t_pred_init: Optional[torch.Tensor] = None,
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
            z_seq:           (B, T, d_repr) encoder outputs.
            z_next_pred_seq: (B, T, d_repr) SSM predictions.
            h_final:         (B, d_state) final SSM hidden state.
            z_delta_seq:     (B, T, d_repr) prediction errors.
            sigreg_loss:     Scalar SIGReg regularization loss.
            x_star_seq:      (B, T, 1) or None, translator outputs.
        """
        batch_size, seq_len, _ = signal_seq.shape
        device = signal_seq.device

        h = h0 if h0 is not None else self.init_hidden(batch_size, device)

        z_t_pred = (
            z_t_pred_init
            if z_t_pred_init is not None
            else torch.zeros(batch_size, self.d_repr, device=device)
        )

        z_list, z_next_pred_list, z_delta_list, x_list = [], [], [], []

        for t in range(seq_len):
            a_t = (
                a_seq[:, t] if a_seq is not None and self.d_above is not None else None
            )

            z_t, z_next_pred, h, z_delta, x_star = self.step(
                signal_seq[:, t], h, a_t, z_t_pred, task_target
            )

            z_list.append(z_t)
            z_next_pred_list.append(z_next_pred)
            z_delta_list.append(z_delta)
            if x_star is not None:
                x_list.append(x_star)

            z_t_pred = z_next_pred.detach()

        z_seq = torch.stack(z_list, dim=1)

        x_star_seq = torch.stack(x_list, dim=1) if x_list else None

        return (
            z_seq,
            torch.stack(z_next_pred_list, dim=1),
            h,
            torch.stack(z_delta_list, dim=1),
            self.sigreg.batch_loss(z_seq),
            x_star_seq,
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @property
    def ae_weight(self) -> float:
        return self._ae_weight

    @property
    def task_weight(self) -> float:
        return self._task_weight
