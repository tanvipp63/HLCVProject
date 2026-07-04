from __future__ import annotations

import torch
import torch.nn as nn

class MSELoss(nn.Module):
    """
    Mean Squared Error loss.
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
    
class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        target = target.float()

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