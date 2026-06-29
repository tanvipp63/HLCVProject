from typing import Optional

import torch
from tqdm import tqdm


class FeatureExtractor:
    """
    Extracts features from a dataset using a foundation model encoder.
    """

    def __init__(self, encoder):

        self.encoder = encoder

    @torch.no_grad()
    def extract(
        self,
        dataloader,
        layers: Optional[list[int]] = None,
    ) -> dict:
        """
        Extract features for an entire dataset.

        Returns
        -------
        dict

        {
            layer_idx: {
                "features": Tensor,
                "labels": Tensor,
            }
        }
        """

        features = {}
        labels = []

        for images, batch_labels in tqdm(
            dataloader,
            desc="Extracting features",
        ):

            batch_features = self.encoder.extract_features(
                images,
                layers=layers,
            )

            for layer, feat in batch_features.items():

                if layer not in features:
                    features[layer] = []

                features[layer].append(feat)

            labels.append(batch_labels)

        labels = torch.cat(labels, dim=0)

        output = {}

        for layer, feats in features.items():

            output[layer] = {
                "features": torch.cat(feats, dim=0),
                "labels": labels,
            }

        return output