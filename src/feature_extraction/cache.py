from pathlib import Path
from typing import Union

import torch


class FeatureCache:
    """
    Handles saving and loading feature batches.

    Each split contains:

    labels.pt

    Each layer directory contains:

    batch_00000.pt
    batch_00001.pt
    ...
    """

    def __init__(self, cache_dir: Union[str, Path]):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def get_layer_dir(
        self,
        encoder_name: str,
        split: str,
        layer: int = -1,
    ) -> Path:

        encoder_dir = self.cache_dir / encoder_name / split

        if layer == -1:
            layer_dir = "final"
        else:
            layer_dir = f"layer{layer}"

        path = encoder_dir / layer_dir
        path.mkdir(parents=True, exist_ok=True)

        return path

    def get_batch_path(
        self,
        encoder_name: str,
        split: str,
        layer: int,
        batch_idx: int,
    ) -> Path:

        return (
            self.get_layer_dir(
                encoder_name,
                split,
                layer,
            )
            / f"batch_{batch_idx:05d}.pt"
        )

    def get_labels_path(
        self,
        encoder_name: str,
        split: str,
    ) -> Path:

        path = self.cache_dir / encoder_name / split
        path.mkdir(parents=True, exist_ok=True)

        return path / "labels.pt"

    # ------------------------------------------------------------------
    # Feature batches
    # ------------------------------------------------------------------

    def save_batch(
        self,
        features: torch.Tensor,
        encoder_name: str,
        split: str,
        layer: int,
        batch_idx: int,
    ) -> None:

        torch.save(
            features,
            self.get_batch_path(
                encoder_name,
                split,
                layer,
                batch_idx,
            ),
        )

    def load_batch(
        self,
        encoder_name: str,
        split: str,
        layer: int,
        batch_idx: int,
    ) -> torch.Tensor:

        return torch.load(
            self.get_batch_path(
                encoder_name,
                split,
                layer,
                batch_idx,
            )
        )

    def list_batches(
        self,
        encoder_name: str,
        split: str,
        layer: int,
    ) -> list[Path]:

        return sorted(
            self.get_layer_dir(
                encoder_name,
                split,
                layer,
            ).glob("batch_*.pt")
        )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def save_labels(
        self,
        labels: torch.Tensor,
        encoder_name: str,
        split: str,
    ) -> None:

        torch.save(
            labels,
            self.get_labels_path(
                encoder_name,
                split,
            ),
        )

    def load_labels(
        self,
        encoder_name: str,
        split: str,
    ) -> torch.Tensor:

        return torch.load(
            self.get_labels_path(
                encoder_name,
                split,
            )
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def exists(
        self,
        encoder_name: str,
        split: str,
        layer: int = -1,
    ) -> bool:

        return len(
            self.list_batches(
                encoder_name,
                split,
                layer,
            )
        ) > 0