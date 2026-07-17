from pathlib import Path
import argparse

import torch
import torch.nn as nn

from src.datasets import ImageNetDataset
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.feature_extraction import FeatureCache
from src.utils.config import load_config


def process_split(
    split: str,
    dataloader,
    encoder_cache: FeatureCache,
    anyup_cache: FeatureCache,
    upsampler,
    encoder_name: str,
    max_cached_batches: int | None,
    device: torch.device,
):

    print(f"\nProcessing {split} split...", flush=True)

    avg_pool = None

    for batch_idx, batch in enumerate(dataloader):

        if (
            max_cached_batches is not None
            and batch_idx >= max_cached_batches
        ):
            break

        #
        # Depending on ImageNetDataset, this will either be:
        #
        # images, labels = batch
        # or
        # images = batch
        #
        images = batch[0]

        lr_features = encoder_cache.load_batch(
            encoder_name=encoder_name,
            split=split,
            layer=-1,
            batch_idx=batch_idx,
        )

        images = images.to(device, non_blocking=True)
        lr_features = lr_features.to(device, non_blocking=True)

        with torch.no_grad():

            hr_features = upsampler(
                images,
                lr_features,
            )

            _, _, H, W = hr_features.shape
            _, _, h, w = lr_features.shape

            assert H % h == 0, f"HR height {H} not divisible by LR height {h}"
            assert W % w == 0, f"HR width {W} not divisible by LR width {w}"

            kernel_h = H // h
            kernel_w = W // w

            if avg_pool is None:
                avg_pool = nn.AvgPool2d(
                    kernel_size=(kernel_h, kernel_w),
                    stride=(kernel_h, kernel_w),
                )

            pooled_features = avg_pool(hr_features)

        assert pooled_features.shape == lr_features.shape

        anyup_cache.save_batch(
            features=pooled_features.cpu(),
            encoder_name=encoder_name,
            split=split,
            layer=-1,
            batch_idx=batch_idx,
        )

        if batch_idx % 10 == 0:
            print(
                f"[{split}] Batch {batch_idx} | "
                f"LR: {tuple(lr_features.shape)} | "
                f"HR: {tuple(hr_features.shape)} | "
                f"Pooled: {tuple(pooled_features.shape)}",
                flush=True,
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoder",
        choices=["dino", "clip", "siglip"],
        required=True,
    )

    parser.add_argument(
        "--max_train_cached_batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_val_cached_batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = load_config("configs/paths.yaml")

    #
    # Encoder
    #

    if args.encoder == "dino":
        encoder = DINOEncoder(device)

    elif args.encoder == "clip":
        encoder = CLIPEncoder(device)

    else:
        encoder = SigLIPEncoder(device)

    print(f"\nEncoder: {args.encoder}", flush=True)

    #
    # Datasets
    #

    train_dataset = ImageNetDataset(
        root=Path(config["imagenet_root"]),
        split="train",
        transform=encoder.get_transform(),
    )

    val_dataset = ImageNetDataset(
        root=Path(config["imagenet_root"]),
        split="val",
        transform=encoder.get_transform(),
    )

    train_loader = train_dataset.get_dataloader(
        batch_size=32,
        shuffle=False,
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=32,
        shuffle=False,
    )

    #
    # Feature caches
    #

    encoder_cache = FeatureCache(
        config["cache_dir"],
    )

    anyup_cache = FeatureCache(
        Path(config["cache_dir"]) / "anyup",
    )

    print(
        f"\nSaving AnyUp caches to {anyup_cache.cache_dir}",
        flush=True,
    )

    #
    # Load AnyUp
    #

    upsampler = torch.hub.load(
        "wimmerth/anyup",
        "anyup_multi_backbone",
        use_natten=True,
    ).to(device)

    upsampler.eval()

    #
    # Process train
    #

    process_split(
        split="train",
        dataloader=train_loader,
        encoder_cache=encoder_cache,
        anyup_cache=anyup_cache,
        upsampler=upsampler,
        encoder_name=args.encoder,
        max_cached_batches=args.max_train_cached_batches,
        device=device,
    )

    #
    # Process val
    #

    process_split(
        split="val",
        dataloader=val_loader,
        encoder_cache=encoder_cache,
        anyup_cache=anyup_cache,
        upsampler=upsampler,
        encoder_name=args.encoder,
        max_cached_batches=args.max_val_cached_batches,
        device=device,
    )

    print("\nAnyUp feature extraction complete.")


if __name__ == "__main__":
    main()