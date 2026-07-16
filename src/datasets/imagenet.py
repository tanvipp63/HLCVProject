from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


class ImageNetDataset:
    """
    Lightweight wrapper around torchvision.datasets.ImageFolder.

    Supports train/validation splits, optional random subsampling, convenient DataLoader creation
    """

    def __init__(
        self,
        root: Union[str , Path],
        split: str,
        transform=None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            root: Root directory containing ImageNet.
                  Expected structure:
                      root/
                          train/
                          val/

            split: Either "train" or "val".

            transform: torchvision transform pipeline.

            max_samples: If specified, randomly samples this many images.

            seed: Random seed for reproducible subsampling.
        """

        if split not in ("train", "val"):
            raise ValueError("split must be either 'train' or 'val'")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.seed = seed

        self.dataset = self._load_dataset()

    def _load_dataset(self):
        """Load the ImageNet split and optionally create a random subset."""

        dataset_path = self.root / self.split

        dataset = ImageFolder(
            root=dataset_path,
            transform=self.transform,
        )

        if self.max_samples is not None:
            if self.max_samples > len(dataset):
                raise ValueError(
                    f"Requested {self.max_samples} samples but dataset only contains {len(dataset)} images."
                )

            generator = torch.Generator().manual_seed(self.seed)

            indices = torch.randperm(
                len(dataset),
                generator=generator,
            )[: self.max_samples]

            dataset = Subset(dataset, indices.tolist())

        return dataset

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader.
        """

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def classes(self):
        """Return ImageNet class names."""

        if isinstance(self.dataset, Subset):
            return self.dataset.dataset.classes

        return self.dataset.classes

    @property
    def class_to_idx(self):
        """Return ImageNet class-to-index mapping."""

        if isinstance(self.dataset, Subset):
            return self.dataset.dataset.class_to_idx

        return self.dataset.class_to_idx