from typing import List

import torch
import torch.nn as nn


class FNN(nn.Module):
    """
    Feedforward network used as:
      - prediction heads between PC levels  (top-down generative)
      - forward prediction heads            (anticipatory, one step ahead)
      - internal projections in the control head

    Uses LayerNorm + GELU activations for training stability, which matters
    particularly in the online learning setting where gradient updates are
    frequent and potentially noisy.

    Args:
        input_dim:   Input feature dimension.
        output_dim:  Output feature dimension.
        hidden_dims: List of hidden layer widths. Empty list = single
                     linear layer with no activation (pure projection).
        dropout:     Dropout probability on hidden activations.
                     Set to 0 for online inference to avoid stochasticity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim)
        Returns:
            (..., output_dim)
        """
        return self.net(x)
