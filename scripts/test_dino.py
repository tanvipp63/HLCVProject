from pathlib import Path

import torch

from src.datasets import ImageNetDataset
from src.encoders import DINOEncoder
from src.utils.config import load_config


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------

    encoder = DINOEncoder(device=device)

    print(f"Loaded {encoder.model_name}")
    print(f"Patch size      : {encoder.patch_size}")
    print(f"Embedding dim   : {encoder.embedding_dim}")
    print(f"Number of layers: {encoder.num_layers}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    paths = load_config("configs/paths.yaml")

    dataset = ImageNetDataset(
        root=Path(paths["imagenet_root"]),
        split="val",
        transform=encoder.get_transform(),
        max_samples=1,
    )

    dataloader = dataset.get_dataloader(
        batch_size=1,
        shuffle=False,
    )

    # --------------------------------------------------
    # Load one image
    # --------------------------------------------------

    images, labels = next(iter(dataloader))

    print(f"\nImage shape : {images.shape}")
    print(f"Label       : {labels.item()}")

    # --------------------------------------------------
    # Feature extraction
    # --------------------------------------------------

    features = encoder.extract_features(images)

    patch_tokens = features[-1]

    print(f"\nPatch token tensor shape : {patch_tokens.shape}")

    B, N, C = patch_tokens.shape

    print(f"Batch size          : {B}")
    print(f"Number of patches   : {N}")
    print(f"Embedding dimension : {C}")

    patch_grid = int(N ** 0.5)

    print(f"Patch grid: {patch_grid} x {patch_grid}")
    print(f"Patch size: {encoder.patch_size}")
    print(f"Embedding dimension: {encoder.embedding_dim}")

    print("\n✓ DINO test passed.")


if __name__ == "__main__":
    main()