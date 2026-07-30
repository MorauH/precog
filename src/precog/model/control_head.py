from typing import List, Optional

import torch
import torch.nn as nn

from .fnn import FNN


class ControlHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        output_scales: Optional[List[float]] = None,
    ):
        super().__init__()
        self.net = FNN(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
        )
        if output_scales and len(output_scales) == output_dim:
            self.register_buffer(
                "output_scales", torch.tensor(output_scales, dtype=torch.float32)
            )
        else:
            self.output_scales = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        y = self.net(x.reshape(-1, x.shape[-1]))
        y = y.reshape(*leading_shape, -1)
        if self.output_scales is not None:
            y = self.output_scales * torch.tanh(y)
        return y
