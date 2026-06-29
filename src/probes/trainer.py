from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm


class ProbeTrainer:
    """
    Generic trainer for probe models.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(
        self,
        dataloader,
    ) -> float:

        self.model.train()

        running_loss = 0.0

        for inputs, targets in tqdm(
            dataloader,
            desc="Training",
        ):

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(inputs)

            loss = self.criterion(
                predictions,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ) -> float:

        self.model.eval()

        running_loss = 0.0

        for inputs, targets in tqdm(
            dataloader,
            desc="Validation",
        ):

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(inputs)

            loss = self.criterion(
                predictions,
                targets,
            )

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
    ) -> dict:

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"| Train Loss: {train_loss:.6f} "
                f"| Val Loss: {val_loss:.6f}"
            )

        return history

    def save_checkpoint(
        self,
        path: str,
    ) -> None:

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str,
    ) -> None:

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )