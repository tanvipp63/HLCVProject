from pathlib import Path
import argparse

import torch

from src.patch_extraction.targets import PatchTargets
from src.utils.config import load_config
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.datasets import ImageNetDataset


# Paths
config = load_config("configs/paths.yaml")
cache_dir = Path(config["cache_dir"])


if __name__ == "__main__":

    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoder",
        choices=["dino", "clip", "siglip"],
        required=True,
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

    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------

    if args.encoder == "dino":
        encoder = DINOEncoder(device)

    elif args.encoder == "clip":
        encoder = CLIPEncoder(device)

    else:
        encoder = SigLIPEncoder(device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = ImageNetDataset(
        root=Path(config["imagenet_root"]),
        split=args.split,
        transform=encoder.get_target_transform(),
        max_samples=args.max_samples,
    )

    dataloader = dataset.get_dataloader(
        batch_size=32,
        shuffle=False,
    )

    # --------------------------------------------------
    # Target generator
    # --------------------------------------------------

    target_generator = PatchTargets(
        image_size=224,
        patch_size=encoder.patch_size,
        cache_dir=cache_dir,
    )

    # --------------------------------------------------
    # Extract targets
    # --------------------------------------------------

    for batch_idx, (images, _) in enumerate(dataloader):

        batch_rgb = []
        batch_edges = []
        # batch_boundaries = []

        for image in images:

            rgb = target_generator.extract_rgb_target(image)
            edges = target_generator.extract_edge_target(image)
            # boundaries = target_generator.extract_boundary_target(image)

            batch_rgb.append(rgb)
            batch_edges.append(edges)
            # batch_boundaries.append(boundaries)

        batch_rgb = torch.stack(batch_rgb)
        batch_edges = torch.stack(batch_edges)
        # batch_boundaries = torch.stack(batch_boundaries)

        target_generator.save_batch(
            data=batch_rgb,
            split=args.split,
            target_type="rgb",
            encoder_name=args.encoder,
            batch_idx=batch_idx,
        )

        target_generator.save_batch(
            data=batch_edges,
            split=args.split,
            target_type="edges",
            encoder_name=args.encoder,
            batch_idx=batch_idx,
        )

        # target_generator.save_batch(
        #     data=batch_boundaries,
        #     split=args.split,
        #     target_type="boundaries",
        #     encoder_name=args.encoder,
        #     batch_idx=batch_idx,
        # )

    print("Target extraction complete.")
