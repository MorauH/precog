from typing import List

import torch
import torch.nn as nn

from .fnn import FNN


class ControlHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.net = FNN(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        y = self.net(x.reshape(-1, x.shape[-1]))
        return y.reshape(*leading_shape, -1)
