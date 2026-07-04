from pathlib import Path

import torch

from src.datasets import ImageNetDataset
from src.encoders import DINOEncoder
from src.probes import MLPProbe
from src.utils.config import load_config


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    paths = load_config("configs/paths.yaml")

    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------

    encoder = DINOEncoder(device=device)

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
    )

    images, _ = next(iter(dataloader))

    print(f"Image shape : {images.shape}")

    # --------------------------------------------------
    # Extract Features
    # --------------------------------------------------

    with torch.no_grad():

        features = encoder.extract_features(images)[-1]

    print(f"Feature shape : {features.shape}")

    # --------------------------------------------------
    # Probe
    # --------------------------------------------------

    probe = MLPProbe(
        input_dim=encoder.embedding_dim,
        patch_size=encoder.patch_size,
    ).to(device)

    print(f"Probe parameters : {probe.num_parameters:,}")

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------

    token_features = features.to(device)

    predictions = probe(token_features)

    print("\n===== Probe Output =====")
    print(f"Prediction shape : {predictions.shape}")

    print("\n✓ Probe test passed.")


if __name__ == "__main__":
    main()