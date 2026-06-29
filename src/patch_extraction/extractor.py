from __future__ import annotations

from typing import Optional

import torch


class PatchExtractor:
    """
    Extracts image patches corresponding to foundation model patch tokens.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
    ) -> None:

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size."
            )

        self.image_size = image_size
        self.patch_size = patch_size

        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size ** 2

    def token_to_coords(
        self,
        token_idx: int,
    ) -> tuple[int, int]:
        """
        Converts a token index into the top-left pixel coordinate
        of its corresponding image patch.

        Returns
        -------
        (top, left)
        """

        if token_idx < 0 or token_idx >= self.num_tokens:
            raise ValueError(
                f"token_idx must be in [0, {self.num_tokens - 1}]"
            )

        row = token_idx // self.grid_size
        col = token_idx % self.grid_size

        top = row * self.patch_size
        left = col * self.patch_size

        return top, left

    def extract_patch(
        self,
        image: torch.Tensor,
        token_indices: Optional[int | list[int]] = None,
    ) -> torch.Tensor:
        """
        Extract one or more image patches.

        Parameters
        ----------
        image
            Tensor of shape (3, H, W)

        token_indices
            None      -> all patches
            int       -> one patch
            list[int] -> selected patches

        Returns
        -------
        If one patch:
            (3, patch_size, patch_size)

        Otherwise:
            (num_patches, 3, patch_size, patch_size)
        """

        if image.ndim != 3:
            raise ValueError(
                "Expected image tensor of shape (3, H, W)."
            )

        single_patch = False

        if token_indices is None:
            token_indices = list(range(self.num_tokens))

        elif isinstance(token_indices, int):
            token_indices = [token_indices]
            single_patch = True

        patches = []

        for token_idx in token_indices:

            top, left = self.token_to_coords(token_idx)

            patch = image[
                :,
                top : top + self.patch_size,
                left : left + self.patch_size,
            ]

            patches.append(patch)

        patches = torch.stack(patches)

        if single_patch:
            return patches[0]

        return patches