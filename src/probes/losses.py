from __future__ import annotations

import torch
import torch.nn as nn


class RGBLoss(nn.Module):
    """
    Mean Squared Error loss for RGB patch reconstruction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return self.loss(pred, target)


class EdgeLoss(nn.Module):
    """
    Placeholder for edge reconstruction loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return self.loss(pred, target)


class BoundaryLoss(nn.Module):
    """
    Placeholder for boundary reconstruction loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return self.loss(pred, target)