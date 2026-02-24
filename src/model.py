"""
MLP network for the lid-driven cavity PINN.

Architecture: 2 inputs (x, y) → 6 × 64 hidden (tanh) → 3 outputs (u, v, p)
All parameters in float64 for numerical stability.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron used as the PINN surrogate.

    Input  : (x, y) coordinates — shape (N, 2)
    Output : (u, v, p) fields  — shape (N, 3)
    """

    def __init__(self) -> None:
        super().__init__()

        layer_dims = [2, 64, 64, 64, 64, 64, 64, 3]
        self.layers = nn.ModuleList(
            [nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)]
        )

        self._init_weights()

        # Convert all parameters and buffers to float64
        self.double()

    def _init_weights(self) -> None:
        gain = nn.init.calculate_gain("tanh")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (N, 2), columns are [x_coord, y_coord], dtype float64

        Returns:
            shape (N, 3), columns are [u, v, p], dtype float64
        """
        out = x
        for layer in self.layers[:-1]:
            out = torch.tanh(layer(out))
        out = self.layers[-1](out)
        return out
