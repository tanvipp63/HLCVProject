from pathlib import Path
import argparse
from src.patch_extraction.targets import PatchTargets
import torch
import os
from src.utils.config import load_config
from src.probes import MLPProbe, ProbeTrainer
from src.probes.losses import *
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.feature_extraction import FeatureExtractor, FeatureCache
from src.patch_extraction import PatchExtractor
from src.datasets import ImageNetDataset

#Paths
config = load_config("configs/paths.yaml")
cache_dir = Path(config["cache_dir"])
cache = FeatureCache(config["cache_dir"])


if __name__ == "__main__":
    #Arg parser
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

    if args.encoder == "dino":
        encoder = DINOEncoder(device)
    elif args.encoder == "clip":
        encoder = CLIPEncoder(device)
    else:
        encoder = SigLIPEncoder(device)


    dataset = ImageNetDataset(
        root=Path(config["imagenet_root"]),
        split=args.split,
        transform=encoder.get_target_transform(),
        max_samples=args.max_samples,
    )

    dataloader = dataset.get_dataloader(
        batch_size=1,
        shuffle=False,
    )    

    target_generator = PatchTargets(
        image_size=224,
        patch_size=encoder.patch_size,
        cache_dir=cache_dir,
    )

    all_rgb = []
    all_edges = []
    all_boundaries = []
        
    for image_idx, (image, _) in enumerate(dataloader):
        image = image[0]

        #Extract targets
        rgb = target_generator.extract_rgb_target(image)
        edges = target_generator.extract_edge_target(image)
        # boundaries = target_generator.extract_boundary_target(image)
        all_rgb.append(rgb)
        all_edges.append(edges)
        # all_boundaries.append(boundaries)
    
    all_rgb = torch.stack(all_rgb)
    all_edges = torch.stack(all_edges)
    # all_boundaries = torch.stack(all_boundaries)

    #Save targets
    target_generator.save(
        all_rgb,
        split=args.split,
        target_type="rgb",
        encoder_name=args.encoder
    )
    target_generator.save(
        all_edges,
        split=args.split,
        target_type="edges",
        encoder_name=args.encoder
    )
    # target_generator.save(
    #     all_boundaries,
    #     split=args.split,
    #     target_type="boundaries",
    #     encoder_name=args.encoder
    # )
