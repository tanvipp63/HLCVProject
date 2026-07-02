#imports
from pathlib import Path
import argparse
import torch
import os
from src.utils.config import load_config

#Paths
config = load_config("configs/paths.yaml")
cache_dir = Path(config["cache_dir"])

encoder_names = [
    p.name
    for p in cache_dir.iterdir()
    if p.is_dir()
]

for encoder in encoder_names:
    encoder_path = cache_dir / encoder
    train_dir = encoder_path / "train"
    val_dir = encoder_path / "val"
    print(f"Train dir: {train_dir}")

    train_layers = [
        p.name
        for p in train_dir.iterdir()
        if p.is_dir()
    ]
    print(f"Layers: {train_layers}")

    for layer in train_layers:
        features = Path(train_dir / layer / "features.pt")