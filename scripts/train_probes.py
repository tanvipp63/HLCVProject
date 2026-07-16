#imports
from pathlib import Path
import argparse
import time
from itertools import islice

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.utils.config import load_config
from src.probes import MLPProbe, ProbeTrainer
from src.probes.losses import *
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.feature_extraction import FeatureCache, CachedFeatureDataset
from src.patch_extraction import PatchTargets, CachedTargetDataset


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
        "--layers",
        type=int,
        nargs="+",
        default=[-1],
        help="Intermediate layers to extract (e.g. 3 6 9 11). Defaults to final layer only.",
    )
    parser.add_argument(
        "--target_type",
        choices=["rgb", "edges", "boundaries"],
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_train_cached_batches",
        type=int,
        default=None,
        help="Maximum number of cached training batches to use. Default: use all cached batches.",
    )
    parser.add_argument(
        "--max_val_cached_batches",
        type=int,
        default=None,
        help="Maximum number of cached validation batches to use. Default: use all cached batches.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
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

    print("Starting train_probes.py", flush=True)

    for layer in args.layers:

        # --------------------------------------------------
        # Cached datasets
        # --------------------------------------------------

        train_feature_dataset = CachedFeatureDataset(
            cache=cache,
            encoder_name=args.encoder,
            split="train",
            layer=layer,
        )

        val_feature_dataset = CachedFeatureDataset(
            cache=cache,
            encoder_name=args.encoder,
            split="val",
            layer=layer,
        )

        print(f"Feature datasets loaded for layer={layer}.", flush=True)

        target_generator = PatchTargets(
            image_size=224,
            patch_size=encoder.patch_size,
            cache_dir=cache_dir,
        )

        train_target_dataset = CachedTargetDataset(
            target_generator=target_generator,
            split="train",
            target_type=args.target_type,
            encoder_name=args.encoder,
        )

        val_target_dataset = CachedTargetDataset(
            target_generator=target_generator,
            split="val",
            target_type=args.target_type,
            encoder_name=args.encoder,
        )

        print(f"Target datasets loaded for layer={layer}.", flush=True)

        assert len(train_feature_dataset) == len(train_target_dataset)
        assert len(val_feature_dataset) == len(val_target_dataset)

        print(f"Train cached batches: {len(train_feature_dataset)}")
        print(f"Val cached batches:   {len(val_feature_dataset)}")

        # --------------------------------------------------
        # Probe
        # --------------------------------------------------

        probe = MLPProbe(
            input_dim=encoder.embedding_dim,
            patch_size=encoder.patch_size,
        ).to(device)

        print(f"Probe parameters: {probe.num_parameters:,}")

        # --------------------------------------------------
        # Trainer
        # --------------------------------------------------

        if args.target_type == "rgb":
            criterion = MSELoss()
        else:
            criterion = BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )

        trainer = ProbeTrainer(
            model=probe,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(f"Trainer initialized for layer={layer}. Beginning training.", flush=True)

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        checkpoint_dir = Path(config["checkpoints_dir"])

        if layer == -1:
            layer_dir = "final"
        else:
            layer_dir = f"layer{layer}"

        checkpoint_dir = (
            checkpoint_dir
            / args.target_type
            / args.encoder
            / layer_dir
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if args.max_train_cached_batches is None:
            effective_train_batches = len(train_feature_dataset)
        else:
            effective_train_batches = min(
                args.max_train_cached_batches,
                len(train_feature_dataset),
            )

        print(
            f"Training using {effective_train_batches} / "
            f"{len(train_feature_dataset)} cached batches.",
            flush=True,
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        patience = 5
        min_delta = 1e-3

        for epoch in range(args.num_epochs):

            train_start = time.perf_counter()

            print(
                f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========",
                flush=True,
            )

            # ------------------------------
            # Train
            # ------------------------------

            train_running_loss = 0.0
            train_num_batches = 0

            train_iterator = zip(
                train_feature_dataset,
                train_target_dataset,
            )

            if args.max_train_cached_batches is not None:
                train_iterator = islice(
                    train_iterator,
                    args.max_train_cached_batches,
                )

            for batch_idx, (features, targets) in enumerate(train_iterator):

                batch_start = time.time()

                train_features = features.reshape(
                    -1,
                    encoder.embedding_dim,
                )

                train_targets = targets.reshape(
                    -1,
                    targets.shape[2],
                    encoder.patch_size,
                    encoder.patch_size,
                )

                train_dataset = TensorDataset(
                    train_features,
                    train_targets,
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                )

                running_loss, num_batches = trainer.train_epoch(
                    train_loader
                )

                batch_time = time.time() - batch_start

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch + 1} | Cached batch {batch_idx} / {effective_train_batches} | "
                        f"Batch Time: {batch_time:.2f} s", flush=True
                    )

                train_running_loss += running_loss
                train_num_batches += num_batches

            train_time = time.perf_counter() - train_start

            # ------------------------------
            # Validation
            # ------------------------------

            val_running_loss = 0.0
            val_num_batches = 0

            val_start = time.perf_counter()

            val_iterator = zip(
                val_feature_dataset,
                val_target_dataset,
            )

            if args.max_val_cached_batches is not None:
                val_iterator = islice(
                    val_iterator,
                    args.max_val_cached_batches,
                )

            for features, targets in val_iterator:

                val_features = features.reshape(
                    -1,
                    encoder.embedding_dim,
                )

                val_targets = targets.reshape(
                    -1,
                    targets.shape[2],
                    encoder.patch_size,
                    encoder.patch_size,
                )

                val_dataset = TensorDataset(
                    val_features,
                    val_targets,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                )

                running_loss, num_batches = trainer.validate(
                    val_loader
                )

                val_running_loss += running_loss
                val_num_batches += num_batches

            val_time = time.perf_counter() - val_start
            epoch_time = train_time + val_time

            train_loss = train_running_loss / train_num_batches
            val_loss = val_running_loss / val_num_batches

            # ------------------------------
            # Save best checkpoint
            # ------------------------------

            if val_loss < best_val_loss - min_delta:

                best_val_loss = val_loss
                epochs_without_improvement = 0

                trainer.save_checkpoint(
                    path=checkpoint_dir / "best_model.pt",
                    epoch=epoch + 1,
                    val_loss=val_loss,
                )

                print(
                    f"Saved best model at epoch {epoch + 1} "
                    f"(val loss = {val_loss:.6f})"
                )

            else:

                epochs_without_improvement += 1

            # ------------------------------
            # Scheduler
            # ------------------------------

            if trainer.scheduler is not None:
                trainer.scheduler.step()

            # ------------------------------
            # Logging
            # ------------------------------

            print(
                f"Train time: {train_time:.2f} s | "
                f"Val time: {val_time:.2f} s | "
                f"Epoch time: {epoch_time:.2f} s",
                flush=True,
            )

            print(
                f"Epoch [{epoch + 1}/{args.num_epochs}] "
                f"| Train Loss: {train_loss:.6f} "
                f"| Val Loss: {val_loss:.6f}"
            )

            if epochs_without_improvement >= patience:

                print(
                    f"Early stopping triggered after "
                    f"{epoch + 1} epochs."
                )

                break

        print(f"Training complete for layer={layer}.")
