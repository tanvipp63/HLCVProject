from pathlib import Path
import argparse
import torch

from src.datasets import ImageNetDataset
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.feature_extraction import FeatureExtractor, FeatureCache
from src.utils.config import load_config


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoder",
        choices=["dino", "clip", "siglip"],
        required=True,
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Intermediate layers to extract (e.g. 3 6 9 11). Defaults to final layer only.",
    )

    parser.add_argument(
        "--split",
        default="val",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = load_config("configs/paths.yaml")

    # ------------------------------
    # Encoder
    # ------------------------------

    if args.encoder == "dino":
        encoder = DINOEncoder(device)

    elif args.encoder == "clip":
        encoder = CLIPEncoder(device)

    else:
        encoder = SigLIPEncoder(device)

    # ------------------------------
    # Dataset
    # ------------------------------

    dataset = ImageNetDataset(
        root=Path(config["imagenet_root"]),
        split=args.split,
        transform=encoder.get_transform(),
        max_samples=args.max_samples,
    )

    dataloader = dataset.get_dataloader(
        batch_size=32,
        shuffle=False,
    )

    # ------------------------------
    # Cache
    # ------------------------------

    cache = FeatureCache(config["cache_dir"])

    # ------------------------------
    # Extract
    # ------------------------------

    extractor = FeatureExtractor(encoder)

    extractor.extract(
        dataloader=dataloader,
        cache=cache,
        encoder_name=args.encoder,
        split=args.split,
        layers=args.layers,
    )

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()