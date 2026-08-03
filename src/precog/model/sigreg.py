"""
Sigma Regularization (SIGReg) for JEPA — prevents representational collapse
without a target network or EMA, replacing the EMA-based JEPA target solution.

Regularizes the covariance matrix of latent representations across two terms:

  - **Variance term**: each dimension must have std dev ≥ var_threshold
  - **Covariance term**: off-diagonal elements → 0 (decorrelation)

With SIGReg, the JEPA loss uses the online representation directly as its target
(z[t+1].detach()) instead of a separate EMA network — collapse is prevented by
the explicit regularization rather than by artificially slowing the target.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sigma Regularization layer for a single PC level.

    Maintains an online EMA estimate of the representation mean and outer
    product so that the covariance loss can be computed periodically even
    when the runner feeds single-timestep samples (batch_size=1).

    For multi-timestep batches (offline / accumulator path) the static
    ``batch_loss()`` method computes the loss directly from the batch.
    """

    def __init__(
        self,
        d_repr: int,
        online_tau: float = 0.999,
        var_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_repr = d_repr
        self.online_tau = online_tau
        self.var_threshold = var_threshold

        # Running EMA estimates for online (single-step) path
        self.register_buffer("_mean", torch.zeros(d_repr))
        self.register_buffer("_outer", torch.eye(d_repr) * 1e-6)

    # ------------------------------------------------------------------ #
    # Online path — maintains running EMA, loss queried periodically
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update_online(self, z: torch.Tensor) -> None:
        """Update running covariance estimate from a *detached* sample.

        Args:
            z: shape (D,) or (B, D).  Must already be detached.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        for i in range(z.shape[0]):
            zi = z[i]
            self._mean.lerp_(zi, 1 - self.online_tau)
            self._outer.lerp_(torch.outer(zi, zi), 1 - self.online_tau)

    def compute_loss_online(self) -> torch.Tensor:
        """Compute SIGReg loss from the running online covariance estimate.

        Returns a scalar loss tensor.  Returns 0.0 if the covariance is
        not yet meaningful (e.g. all-zeros).
        """
        cov = self._outer - torch.outer(self._mean, self._mean)
        var = cov.diag().clamp(min=0.0)
        var_loss = F.relu(self.var_threshold - var.sqrt()).mean()

        D = self.d_repr
        idx = torch.arange(D, device=cov.device)
        mask = idx.unsqueeze(0) != idx.unsqueeze(1)
        off = cov[mask]
        cov_loss = off.pow(2).sum() / max(1, int(mask.sum().item()))

        return var_loss + cov_loss

    # ------------------------------------------------------------------ #
    # Batch path — computes loss directly from a (B, T, D) tensor
    # ------------------------------------------------------------------ #

    @staticmethod
    def batch_loss(
        z_seq: torch.Tensor,
        var_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Compute SIGReg covariance loss from a multi-timestep batch.

        Args:
            z_seq: shape (B, T, D).
            var_threshold: minimum per-dimension standard deviation.

        Returns:
            Scalar loss tensor (0 if N < 2).
        """
        z = z_seq.reshape(-1, z_seq.shape[-1])  # (N, D)  ,  N = B * T
        N, D = z.shape
        if N < 2:
            return torch.tensor(0.0, device=z.device)

        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = z_centered.T @ z_centered / (N - 1 + 1e-8)
        var = cov.diag().clamp(min=0.0)
        var_loss = F.relu(var_threshold - var.sqrt()).mean()

        idx = torch.arange(D, device=z.device)
        mask = idx.unsqueeze(0) != idx.unsqueeze(1)
        off = cov[mask]
        cov_loss = off.pow(2).sum() / max(1, int(mask.sum().item()))

        return var_loss + cov_loss
