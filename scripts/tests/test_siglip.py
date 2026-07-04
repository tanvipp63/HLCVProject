import sys
from pathlib import Path

import torch

from src.datasets import ImageNetDataset
from src.encoders import SigLIPEncoder
from src.utils.config import load_config


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    paths = load_config("configs/paths.yaml")

    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------

    encoder = SigLIPEncoder(device=device)

    print(f"Loaded {encoder.model_name}")
    print(f"Patch size      : {encoder.patch_size}")
    print(f"Embedding dim   : {encoder.embedding_dim}")
    print(f"Number of layers: {encoder.num_layers}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = ImageNetDataset(
        root=Path(paths["imagenet_root"]),
        split="val",
        transform=encoder.get_transform(),
        max_samples=1,
    )

    dataloader = dataset.get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # Load one image
    # --------------------------------------------------

    images, labels = next(iter(dataloader))

    print(f"\nImage shape : {images.shape}")
    print(f"Label       : {labels.item()}")

    # --------------------------------------------------
    # Final layer
    # --------------------------------------------------

    features = encoder.extract_features(images)
    patch_tokens = features[-1]

    print("\n===== Final Layer =====")
    print(f"Patch token tensor shape : {patch_tokens.shape}")
    print(f"Batch size          : {patch_tokens.shape[0]}")
    print(f"Number of patches   : {patch_tokens.shape[1]}")
    print(f"Embedding dimension : {patch_tokens.shape[2]}")

    # --------------------------------------------------
    # Intermediate layers
    # --------------------------------------------------

    print("\n===== Intermediate Layers =====")

    intermediate = encoder.extract_features(images, layers=[3, 6, 9, 11])
    for layer, feat in intermediate.items():
        print(f"Layer {layer}: {feat.shape}")

    print("\n✓ SIGLIP test passed.")


if __name__ == "__main__":
    main()
