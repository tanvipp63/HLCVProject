#imports
from pathlib import Path
import argparse
import torch
import os
from src.utils.config import load_config
from src.probes import MLPProbe, ProbeTrainer
from src.probes.losses import *
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.feature_extraction import FeatureExtractor, FeatureCache

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
        default=None,
        help="Intermediate layers to extract (e.g. 3 6 9 11). Defaults to final layer only.",
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

        #Init probe
        probe = MLPProbe(
            input_dim=encoder.embedding_dim,
            patch_size=encoder.patch_size,
        ).to(device)
        print(f"Probe parameters : {probe.num_parameters:,}")

        #Configure trainer
        criterion = RGBLoss()
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
        checkpoint_dir = Path("checkpoints/dino/layer3")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        #Need to load train and val targets TODO first is have a script that generates patch targets first

        # history = trainer.fit(
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     epochs=50,
        #     best_checkpoint_path=checkpoint_dir / "best.pt",
        # )        

        #Validate

