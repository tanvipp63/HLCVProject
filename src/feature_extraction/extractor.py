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
        cache,
        encoder_name: str,
        split: str,
        layers: Optional[list[int]] = None,
    ):
        """
        Extract features and stream them to the feature cache one batch at a time.

        Avoids accumulating the entire dataset in RAM.
        """

        for batch_idx, (images, batch_labels) in enumerate(
            tqdm(
                dataloader,
                desc="Extracting features",
            )
        ):

            batch_features = self.encoder.extract_features(
                images,
                layers=layers,
            )

            for layer, feat in batch_features.items():

                cache.save_batch(
                    features=feat,
                    encoder_name=encoder_name,
                    split=split,
                    layer=layer,
                    batch_idx=batch_idx,
                )

        print(
            f"Saved {batch_idx + 1} feature batches."
        )