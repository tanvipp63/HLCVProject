#imports
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
from torch.utils.data import TensorDataset, DataLoader

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
        default=256,
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
        features_train = cache.load(args.encoder, "train", layer=layer)
        features_val = cache.load(args.encoder, "val", layer=layer)

        target_generator = PatchTargets(
            image_size=224,
            patch_size=encoder.patch_size,
            cache_dir=cache_dir,
        )
        targets_train = target_generator.load("train", args.target_type, args.encoder)
        targets_val = target_generator.load("val", args.target_type, args.encoder)

        #Init probe
        probe = MLPProbe(
            input_dim=encoder.embedding_dim,
            patch_size=encoder.patch_size,
        ).to(device)
        print(f"Probe parameters : {probe.num_parameters:,}")

        #Configure trainer
        if args.target_type == "rgb":
            criterion = MSELoss()
        elif args.target_type == "edges":
            criterion = BCEWithLogitsLoss()
        elif args.target_type == "boundaries":
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

        #Train
        checkpoint_dir = Path(f"{config['checkpoints_dir']}")
        train_features = features_train["features"].reshape(-1, encoder.embedding_dim)
        train_targets = targets_train.reshape(-1, targets_train.shape[2], encoder.patch_size, encoder.patch_size)

        val_features = features_val["features"].reshape(-1, encoder.embedding_dim)
        val_targets = targets_val.reshape(-1, targets_val.shape[2], encoder.patch_size, encoder.patch_size)
        
        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        print(f"Train features: {features_train['features'].shape}")
        print(f"Val features:   {features_val['features'].shape}")
        print(f"Train targets:  {targets_train.shape}")
        print(f"Val targets:    {targets_val.shape}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        inputs, targets = next(iter(train_loader))
        print(f"Input batch:  {inputs.shape}")
        print(f"Target batch: {targets.shape}")

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.num_epochs,
            checkpoint_dir=checkpoint_dir,
            encoder_name=args.encoder,
            layer=layer,
            target_type=args.target_type
        )        
        print("Training complete.")
