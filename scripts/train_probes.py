#imports
from pathlib import Path
import argparse

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
            elif args.target_type == "edges":
                criterion = BCEWithLogitsLoss()
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

        best_val_loss = float("inf")

        for epoch in range(args.num_epochs):

            # ------------------------------
            # Train
            # ------------------------------

            train_running_loss = 0.0
            train_num_batches = 0
        
        for features, targets in zip(
            train_feature_dataset,
            train_target_dataset,
        ):

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

            train_running_loss += running_loss
            train_num_batches += num_batches
        
        # ------------------------------
        # Validation
        # ------------------------------

        val_running_loss = 0.0
        val_num_batches = 0

        for features, targets in zip(
            val_feature_dataset,
            val_target_dataset,
        ):

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
        
        train_loss = train_running_loss / train_num_batches
        val_loss = val_running_loss / val_num_batches

        # ------------------------------
        # Save best checkpoint
        # ------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            trainer.save_checkpoint(
                path=checkpoint_dir / "best_model.pt",
                epoch=epoch + 1,
                val_loss=val_loss,
            )

            print(
                f"Saved best model at epoch {epoch + 1} "
                f"(val loss = {val_loss:.6f})"
            )

        # ------------------------------
        # Scheduler
        # ------------------------------

        if trainer.scheduler is not None:
            trainer.scheduler.step()

        # ------------------------------
        # Logging
        # ------------------------------

        print(
            f"Epoch [{epoch + 1}/{args.num_epochs}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}"
        )

    print("Training complete.")