from __future__ import annotations

from typing import Optional, Union

import torch
from transformers import AutoImageProcessor, AutoModel

from .base_encoder import BaseEncoder


class SigLIPEncoder(BaseEncoder):
    """
    Wrapper around a Hugging Face SIGLIP vision encoder.
    """

    def __init__(
        self,
        device: Union[torch.device, str] = "cpu",
        model_name: str = "google/siglip-base-patch16-224",
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name

        self.patch_size = 16
        self.embedding_dim = 768
        self.num_layers = 12

        self.model = None
        self.processor = None

    def _preprocess_image(self, image):
        """Apply the SIGLIP image processor to a single image."""
        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)

        return pixel_values

    def load_model(self):
        """
        Load the pretrained SIGLIP model and image processor.
        """
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        
        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        return self.model

    def get_transform(self):
        """
        Return a preprocessing callable for SIGLIP images.
        """
        return self._preprocess_image
    
    def get_target_transform(self):
        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        def transform(image):
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                do_normalize=False,
            )

            pixel_values = inputs["pixel_values"]

            if pixel_values.ndim == 4:
                pixel_values = pixel_values.squeeze(0)

            return pixel_values

        return transform

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        layers: Optional[list[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract patch-token features from the SIGLIP vision backbone.

        Returns:
            dict[int, torch.Tensor]

            Final:
                {-1: (B, N, C)}

            Intermediate:
                {layer_idx: (B, N, C)}
        """
        model = self.load_model()
        images = images.to(self.device)

        vision_model = getattr(model, "vision_model", model)

        if layers is None:
            outputs = vision_model(
                pixel_values=images,
                output_hidden_states=True,
                return_dict=True,
            )

            patch_tokens = outputs.last_hidden_state
            return {-1: patch_tokens.cpu()}

        outputs = vision_model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )

        intermediates = {}
        hidden_states = getattr(outputs, "hidden_states", None)

        if hidden_states is None:
            fallback = outputs.last_hidden_state
            for layer_idx in layers:
                if layer_idx < 0:
                    continue
                intermediates[layer_idx] = fallback.detach().cpu()
            return intermediates

        for layer_idx in layers:
            if layer_idx < 0:
                continue

            if layer_idx + 1 < len(hidden_states):
                hidden_state = hidden_states[layer_idx + 1]
                intermediates[layer_idx] = hidden_state.detach().cpu()
            else:
                intermediates[layer_idx] = torch.empty(0)

        return intermediates
