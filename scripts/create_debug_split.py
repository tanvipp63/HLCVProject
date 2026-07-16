# This script creates a small train/val split from the cached imagenet val set for smoke testing the full pipeline on the HPC cluster.
from pathlib import Path
import argparse

import torch

from src.feature_extraction import FeatureCache
from src.patch_extraction.targets import PatchTargets
from src.utils.config import load_config
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder


def main():

    parser = argparse.ArgumentParser(
        description="Create a small train/val split from the cached validation set for smoke testing."
    )

    parser.add_argument(
        "--encoder",
        choices=["dino", "clip", "siglip"],
        required=True,
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    config = load_config("configs/paths.yaml")

    cache = FeatureCache(config["cache_dir"])

    if args.encoder == "dino":
        encoder = DINOEncoder()

    elif args.encoder == "clip":
        encoder = CLIPEncoder()

    else:
        encoder = SigLIPEncoder()

    target_generator = PatchTargets(
        image_size=224,
        patch_size=encoder.patch_size,
        cache_dir=config["cache_dir"],
    )

    # --------------------------------------------------
    # Features
    # --------------------------------------------------

    features = cache.load(
        args.encoder,
        "val",
        layer=args.layer,
    )

    total_samples = features["features"].shape[0]

    if args.train_size >= total_samples:
        raise ValueError(
            f"train_size ({args.train_size}) must be smaller than "
            f"the number of samples ({total_samples})."
        )

    train_features = {}
    val_features = {}

    for key, value in features.items():

        if isinstance(value, torch.Tensor):

            train_features[key] = value[: args.train_size]
            val_features[key] = value[args.train_size :]

        else:

            train_features[key] = value
            val_features[key] = value

    cache.save(
        train_features,
        args.encoder,
        "train",
        layer=args.layer,
    )

    cache.save(
        val_features,
        args.encoder,
        "val",
        layer=args.layer,
    )

    print(
        f"Train features: {train_features['features'].shape}"
    )
    print(
        f"Val features:   {val_features['features'].shape}"
    )

    # --------------------------------------------------
    # Targets
    # --------------------------------------------------

    for target_type in ["rgb", "edges"]:

        targets = target_generator.load(
            "val",
            target_type,
            args.encoder,
        )

        train_targets = targets[: args.train_size]
        val_targets = targets[args.train_size :]

        target_generator.save(
            train_targets,
            "train",
            target_type,
            args.encoder,
        )

        target_generator.save(
            val_targets,
            "val",
            target_type,
            args.encoder,
        )

        print(
            f"{target_type.capitalize()} train: {train_targets.shape}"
        )
        print(
            f"{target_type.capitalize()} val:   {val_targets.shape}"
        )

    print("\nDebug split created successfully.")


if __name__ == "__main__":
    main()