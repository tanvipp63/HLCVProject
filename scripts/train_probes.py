#imports
from pathlib import Path
import argparse
import torch
import os
from src.utils.config import load_config

#Paths
config = load_config("configs/paths.yaml")
cache_dir = config["cache_dir"]

#Extract different layers at which token was extracted, including final layer
encoder_names = [
    name for name in os.listdir(cache_dir)
    if os.path.isdir(os.path.join(cache_dir, name))
]
print(f"Encoders: {encoder_names}")

#For each encoder, train probe for every cached layer and final layer
for encoder in encoder_names:
    encoder_path = os.path.join(cache_dir, encoder)
    encoder_files = [f for f in os.listdir(encoder_path) if os.path.isfile(os.path.join(encoder_path, f))]

