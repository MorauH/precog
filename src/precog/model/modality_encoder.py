from typing import List, Optional

import torch
import torch.nn as nn

from .fnn import FNN


class ModalityEncoder(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: Optional[List[int]] = None
    ):
        super().__init__()
        self.net = FNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dim or [],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        y = self.net(x.reshape(-1, x.shape[-1]))
        return y.reshape(*leading_shape, -1)
