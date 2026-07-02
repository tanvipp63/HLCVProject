from pathlib import Path

import torch


class FeatureCache:
    """
    Handles saving and loading extracted feature caches.
    """

    def __init__(self, cache_dir: str | Path):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(
        self,
        encoder_name: str,
        split: str,
        layer: int = -1,
    ) -> Path:
        """
        Returns the cache file path.

        Example:
            cache/dino/val_final.pt
            cache/dino/val_layer3.pt

            Now changed to: cache/dino/val/layer3/features.pt
        """

        encoder_dir = self.cache_dir / encoder_name / split / layer
        encoder_dir.mkdir(parents=True, exist_ok=True)

        # if layer == -1:
        #     filename = f"{split}_final.pt"
        # else:
        #     filename = f"{split}_layer{layer}.pt"

        filename = f"features.pt"

        return encoder_dir / filename

    def exists(
        self,
        encoder_name: str,
        split: str,
        layer: int = -1,
    ) -> bool:

        return self.get_cache_path(
            encoder_name,
            split,
            layer,
        ).exists()

    def save(
        self,
        data: dict,
        encoder_name: str,
        split: str,
        layer: int = -1,
    ) -> None:

        path = self.get_cache_path(
            encoder_name,
            split,
            layer,
        )

        torch.save(data, path)

    def load(
        self,
        encoder_name: str,
        split: str,
        layer: int = -1,
    ) -> dict:

        path = self.get_cache_path(
            encoder_name,
            split,
            layer,
        )

        return torch.load(path)