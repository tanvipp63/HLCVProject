from __future__ import annotations

import torch
import torch.nn as nn


class MLPProbe(nn.Module):
    """
    MLP probe for reconstructing image patches from foundation model features.
    """

    def __init__(
        self,
        input_dim: int = 768,
        patch_size: int = 14,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.output_dim = 3 * patch_size * patch_size

        layers = []

        in_features = input_dim

        # Hidden layers
        for _ in range(num_layers - 1):

            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])

            in_features = hidden_dim

        # Output layer
        layers.append(
            nn.Linear(in_features, self.output_dim)
        )

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Shape: (B, input_dim)

        Returns
        -------
        Tensor
            Shape: (B, 3, patch_size, patch_size)
        """

        if x.ndim == 3:
            batch_size, num_tokens, _ = x.shape
            x = x.reshape(batch_size * num_tokens, self.input_dim)
            x = self.mlp(x)
            x = x.view(
                batch_size,
                num_tokens,
                3,
                self.patch_size,
                self.patch_size,
            )
        else:
            x = self.mlp(x)
            x = x.view(-1, 3, self.patch_size, self.patch_size)

        return x


    @property
    def num_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )