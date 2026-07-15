from __future__ import annotations

from typing import Optional, Union

import open_clip
import torch
from torchvision import transforms

from .base_encoder import BaseEncoder


class CLIPEncoder(BaseEncoder):
    """
    Wrapper around the CLIP ViT-B/16 visual encoder.
    """

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
    ) -> None:

        self.device = torch.device(device)

        self.model_name = model_name
        self.pretrained = pretrained

        # Model metadata
        self.patch_size = 16
        self.embedding_dim = 768
        self.num_layers = 12

        self.model = None
        self.transform = None

    def load_model(self):
        """
        Load the pretrained CLIP visual encoder.
        """

        if self.model is None:

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
            )

            self.model = model.visual.eval().to(self.device)
            self.transform = preprocess

        return self.model

    def get_transform(self):
        """
        Return the preprocessing transform expected by CLIP.
        """

        if self.transform is None:
            self.load_model()

        return self.transform
    
    def get_target_transform(self):
        """
        Return the preprocessing transform for target extraction.

        Applies the same spatial preprocessing as CLIP, but leaves the
        image in raw RGB tensor format [0,1] (no normalization).
        """

        if self.transform is None:
            self.load_model()

        return transforms.Compose(self.transform.transforms[:-1])

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        layers: Optional[list[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract patch-token features.

        Args:
            images:
                Tensor of shape (B, 3, H, W).

            layers:
                None -> return final layer only.
                Otherwise return requested intermediate layers.

        Returns:
            dict[int, torch.Tensor]

            Final:
                {-1 : (B, N, C)}

            Intermediate:
                {layer_idx : (B, N, C)}
        """

        model = self.load_model()

        images = images.to(self.device)

        # --------------------------------------------------
        # Final layer
        # --------------------------------------------------

        if layers is None:
            final_features = None

            def final_hook(module, inputs, outputs):
                nonlocal final_features

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                if outputs.ndim == 2:
                    outputs = outputs.unsqueeze(0)

                if outputs.ndim >= 3:
                    final_features = outputs[:, 1:, :].detach().cpu()

            hook = model.transformer.resblocks[-1].register_forward_hook(final_hook)

            with torch.no_grad():
                _ = model(images)

            hook.remove()

            if final_features is None:
                raise RuntimeError("Failed to capture final CLIP patch-token features.")

            return {-1: final_features}

        # --------------------------------------------------
        # Intermediate layers
        # --------------------------------------------------

        intermediates = {}
        hooks = []

        for layer_idx in layers:

            def make_hook(idx):

                def hook(module, inputs, outputs):
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    if outputs.ndim == 2:
                        outputs = outputs.unsqueeze(0)

                    if outputs.ndim >= 3:
                        outputs = outputs[:, 1:, :]

                    intermediates[idx] = outputs.detach().cpu()

                return hook

            hooks.append(
                model.transformer.resblocks[layer_idx].register_forward_hook(
                    make_hook(layer_idx)
                )
            )

        # Run one forward pass
        with torch.no_grad():
            _ = model(images)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return intermediates