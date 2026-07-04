from __future__ import annotations

from typing import Optional

import torch
import torchvision.transforms.functional as TF
from pathlib import Path
import kornia
from .extractor import PatchExtractor


class PatchTargets:
    """
    Generates supervision targets for probe training.

    Currently supports:
        - RGB patches

    Future:
        - Edge maps
        - Boundary maps
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        cache_dir: str
    ) -> None:

        self.extractor = PatchExtractor(
            image_size=image_size,
            patch_size=patch_size,
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)        

    def extract_rgb_target(
        self,
        image: torch.Tensor,
        token_indices: Optional[int | list[int]] = None,
    ) -> torch.Tensor:
        """
        Extract RGB patch targets.

        Returns
        -------
        Single patch:
            (3, patch_size, patch_size)

        Multiple patches:
            (N, 3, patch_size, patch_size)
        """

        return self.extractor.extract_patch(
            image=image,
            token_indices=token_indices,
        )

    def extract_edge_target(
        self,
        image: torch.Tensor,
        token_indices: Optional[int | list[int]] = None,
    ) -> torch.Tensor:
        """
        Extract edge map patches corresponding to the specified tokens.

        Args:
            image: RGB image tensor of shape (3, H, W).
            token_indices: Token(s) to extract. If None, extracts all patches.

        Returns:
            Tensor of edge patches.
        """

        # Convert RGB to grayscale
        gray = TF.rgb_to_grayscale(image)

        # Kornia expects a batch dimension
        gray = gray.unsqueeze(0)

        # Canny edge detector
        _, edges = kornia.filters.canny(gray)

        # Remove batch dimension
        edges = edges.squeeze(0)

        # Extract patches
        return self.extractor.extract_patch(
            image=edges,
            token_indices=token_indices,
        )

    def extract_boundary_target(
        self,
        image: torch.Tensor,
        token_indices: Optional[int | list[int]] = None,
    ) -> torch.Tensor:
        """
        Placeholder for boundary target extraction.
        """

        raise NotImplementedError(
            "Boundary target extraction not implemented yet."
        )
    
    def get_cache_path(
        self,
        split: str,
        target_type: str,
        encoder_name: str
    ) -> Path:

        target_dir = self.cache_dir / "targets" / split / encoder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        return target_dir / f"{target_type}.pt"

    def save(
        self,
        data: torch.Tensor,
        split: str,
        target_type: str,
        encoder_name: str
    ) -> None:

        path = self.get_cache_path(split, target_type, encoder_name)
        torch.save(data, path)

    def load(
        self,
        split: str,
        target_type: str,
        encoder_name: str
    ) -> torch.Tensor:

        path = self.get_cache_path(split, target_type, encoder_name)
        return torch.load(path)    