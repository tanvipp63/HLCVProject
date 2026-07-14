from __future__ import annotations

import os
import ssl
from typing import Optional, Union

import certifi
import torch
from torchvision import transforms

from .base_encoder import BaseEncoder


class DINOEncoder(BaseEncoder):
    """
    Wrapper around the DINOv2 ViT-B/14 encoder.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        model_name: str = "dinov2_vitb14",
    ) -> None:

        self.device = torch.device(device)
        self.model_name = model_name

        self.patch_size = 14
        self.embedding_dim = 768
        self.num_layers = 12

        self.model = None

    def load_model(self):
        """
        Load the pretrained DINOv2 model.
        """

        if self.model is None:
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl._create_default_https_context = lambda: ssl_context

            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                self.model_name,
                pretrained=True,
            )

            self.model.eval()
            self.model.to(self.device)

        return self.model

    def get_transform(self, image_size: int = 224):
        """
        Return the preprocessing transform expected by DINOv2.
        """

        resize_size = int(image_size * 256 / 224)

        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD,
            ),
        ])

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
                Otherwise return the requested intermediate layers.

        Returns
        -------
        dict

        Final layer:
            {-1 : (B, N, C)}

        Intermediate:
            {layer_idx : (B, N, C)}
        """

        model = self.load_model()

        images = images.to(self.device)

        # Final layer only
        if layers is None:
            output = model.forward_features(images)

            return {
                -1: output["x_norm_patchtokens"].cpu()
            }

        # Intermediate layers
        intermediates = {}
        hooks = []

        for layer_idx in layers:

            def make_hook(idx):

                def hook(module, inputs, outputs):
                    # outputs = (B, N+1, C)
                    # Remove CLS token
                    intermediates[idx] = outputs[:, 1:, :].detach().cpu()

                return hook

            hooks.append(
                model.blocks[layer_idx].register_forward_hook(
                    make_hook(layer_idx)
                )
            )

        model.forward_features(images)

        for hook in hooks:
            hook.remove()

        return intermediates