from __future__ import annotations

from typing import Optional

import open_clip
import torch

from .base_encoder import BaseEncoder


class CLIPEncoder(BaseEncoder):
    """
    Wrapper around the CLIP ViT-B/16 visual encoder.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
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
            with torch.no_grad():
                features = model(images)

            if isinstance(features, tuple):
                features = features[0]

            if features.ndim == 2:
                features = features.unsqueeze(0)

            return {
                -1: features.cpu()
            }

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