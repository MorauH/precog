"""
JEPA Predictive Coding Level.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from .config import PCLevelConfig
from .fnn import FNN
from .sigreg import SIGReg
from .ssm import SelectiveSSM


@dataclass
class PCLevelOutput:
    """Structured container for single-step and sequence outputs."""
    z_t: torch.Tensor                       # Encoder output: representation at current step (B, d_repr) or (B, T, d_repr)
    z_next_pred: torch.Tensor               # SSM output: predicted representation for next step
    ssm_h: torch.Tensor                     # Updated SSM hidden state (B, d_state)
    z_delta: Optional[torch.Tensor] = None  # Prediction error: z_t - z_t_pred
    x_pred: Optional[torch.Tensor] = None   # Translator/Task prediction output

class PCLevel(nn.Module):
    """
    Predictive Coding (PC) Level module with FNN Encoder, 
    Selective SSM dynamics predictor, and optional task head.
    """
    def __init__(
        self,
        d_below: int,
        d_above: Optional[int],
        config: PCLevelConfig,
    ) -> None:
        super().__init__()

        self.d_below = d_below
        self.d_above = d_above
        self.d_repr = config.d_representation
        self.objective_enabled = config.objective_enabled

        # Encoder: Maps input from level below to representation space
        self.encoder = FNN(
            input_dim=d_below,
            output_dim=self.d_repr,
            hidden_dims=config.encoder_hidden,
        )

        # Predictive SSM: Maps (z_t, signal_above) to z_{t+1} prediction
        ssm_input_dim = self.d_repr + (d_above if d_above is not None else 0)
        self.ssm = SelectiveSSM(
            d_input=ssm_input_dim,
            d_output=self.d_repr,
            config=config.ssm,
        )

        # Optional Objective / Task Translator Head
        if self.objective_enabled:
            self.translator = FNN(
                input_dim=self.d_repr,
                output_dim=1,
                hidden_dims=config.translator_hidden
            )
        else:
            self.translator = None

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Helper to initialize zero-state for SSM."""
        return self.ssm.init_hidden(batch_size, device)

    def _prepare_ssm_input(
        self, 
        z_t: torch.Tensor, 
        signal_above: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenates current latent state with top-down feedback if present."""
        if self.d_above is None:
            return z_t

        if signal_above is None:
            signal_above = torch.zeros(
                z_t.shape[0], self.d_above, device=z_t.device, dtype=z_t.dtype
            )
        return torch.cat([z_t, signal_above], dim=-1)

    def step(
        self,
        signal_below: torch.Tensor,
        h_prev: torch.Tensor,
        signal_above: Optional[torch.Tensor] = None,
        z_t_pred: Optional[torch.Tensor] = None,
    ) -> PCLevelOutput:
        """
        Processes a single timestep (Online/Recurrent Execution Mode).
        """
        # 1. Encode bottom-up signal
        z_t = self.encoder(signal_below)

        # 2. Compute prediction error if previous prediction is supplied
        z_delta = (z_t - z_t_pred) if z_t_pred is not None else None

        # 3. Step the SSM forward dynamics
        ssm_input = self._prepare_ssm_input(z_t, signal_above)
        z_next_pred, h_new = self.ssm.step(ssm_input, h_prev)

        # 4. Optional task prediction output
        x_pred = self.translator(z_t) if self.translator is not None else None

        return PCLevelOutput(
            z_t=z_t,
            z_next_pred=z_next_pred,
            ssm_h=h_new,
            z_delta=z_delta,
            x_pred=x_pred,
        )

    def forward(
        self,
        signal_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        signal_above_seq: Optional[torch.Tensor] = None,
        z_t_pred_init: Optional[torch.Tensor] = None,
    ) -> PCLevelOutput:
        """
        Processes a sequence of timesteps (Batch / Sequence Mode).
        Inputs shape: (Batch, Seq_Len, Dim)
        """
        batch_size, seq_len, _ = signal_seq.shape
        device = signal_seq.device

        h = h0 if h0 is not None else self.init_hidden(batch_size, device)
        z_t_pred = z_t_pred_init

        z_list, z_next_pred_list, z_delta_list, x_list = [], [], [], []

        for t in range(seq_len):
            sig_below_t = signal_seq[:, t]
            sig_above_t = signal_above_seq[:, t] if signal_above_seq is not None else None

            out = self.step(
                signal_below=sig_below_t,
                h_prev=h,
                signal_above=sig_above_t,
                z_t_pred=z_t_pred,
            )

            z_list.append(out.z_t)
            z_next_pred_list.append(out.z_next_pred)
            if out.z_delta is not None:
                z_delta_list.append(out.z_delta)
            if out.x_pred is not None:
                x_list.append(out.x_pred)

            # Update loop variables
            h = out.ssm_h
            z_t_pred = out.z_next_pred.detach()  # Detach prediction to isolate graphs

        return PCLevelOutput(
            z_t=torch.stack(z_list, dim=1),
            z_next_pred=torch.stack(z_next_pred_list, dim=1),
            ssm_h=h,
            z_delta=torch.stack(z_delta_list, dim=1) if z_delta_list else None,
            x_pred=torch.stack(x_list, dim=1) if x_list else None,
        )


# NOT USED. FOR REFERENCE ONLY
class PCLevelCriterion(nn.Module):
    """External Loss / Objective Evaluator for PCLevel."""
    def __init__(self, config: PCLevelConfig):
        super().__init__()
        self.sigreg = SIGReg(
            d_repr=config.d_representation,
            online_tau=config.sigreg_tau,
            var_threshold=config.sigreg_var_threshold,
        )
        self.translate_weight = config.objective_ae_weight if config.objective_enabled else 0.0
        self.task_weight = config.objective_task_weight if config.objective_enabled else 0.0

    def forward(
        self, 
        outputs: PCLevelOutput, 
        task_target: Optional[torch.Tensor] = None,
        task_actual: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Update SIGReg statistics during training steps
        if self.training:
            self.sigreg.update_online(outputs.z_t.detach())

        # Representation loss (SIGReg)
        loss_sigreg = self.sigreg(outputs.z_t)

        # Optional task loss
        loss_task = torch.tensor(0.0, device=outputs.z_t.device)
        loss_translate = torch.tensor(0.0, device=outputs.z_t.device)
        if outputs.x_pred is not None and task_target is not None and task_actual is not None:
            loss_task = nn.functional.mse_loss(outputs.x_pred, task_target)
            loss_translate = nn.functional.mse_loss(outputs.x_pred, task_actual)

        total_loss = loss_sigreg + (self.task_weight * loss_task) + (self.translate_weight * loss_translate)

        return {
            "loss_total": total_loss,
            "loss_sigreg": loss_sigreg,
            "loss_task": loss_task,
            "loss_translate": loss_translate,
        }
