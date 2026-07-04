from pathlib import Path

import torch

from src.datasets import ImageNetDataset
from src.encoders import DINOEncoder
from src.patch_extraction import PatchExtractor
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

    images, labels = next(iter(dataloader))

    # Remove batch dimension
    image = images[0]

    print(f"Image shape: {image.shape}")

    # --------------------------------------------------
    # Patch Extractor
    # --------------------------------------------------

    extractor = PatchExtractor(
        image_size=224,
        patch_size=encoder.patch_size,
    )

    print(f"Grid size : {extractor.grid_size}")
    print(f"Num tokens: {extractor.num_tokens}")

    # --------------------------------------------------
    # Test token -> coordinates
    # --------------------------------------------------

    print("\n===== Token Coordinates =====")

    for token in [0, 15, 16, 57, 255]:
        top, left = extractor.token_to_coords(token)
        print(f"Token {token:3d} -> ({top}, {left})")

    # --------------------------------------------------
    # Single patch
    # --------------------------------------------------

    print("\n===== Single Patch =====")

    patch = extractor.extract_patch(
        image,
        token_indices=57,
    )

    print(patch.shape)

    # --------------------------------------------------
    # Multiple patches
    # --------------------------------------------------

    print("\n===== Multiple Patches =====")

    patches = extractor.extract_patch(
        image,
        token_indices=[0, 57, 255],
    )

    print(patches.shape)

    # --------------------------------------------------
    # All patches
    # --------------------------------------------------

    print("\n===== All Patches =====")

    patches = extractor.extract_patch(image)

    print(patches.shape)

    print("\n✓ Patch extraction test passed.")


if __name__ == "__main__":
    main()