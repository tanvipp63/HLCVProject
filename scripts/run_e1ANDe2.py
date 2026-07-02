#imports
from pathlib import Path
import argparse
import torch
import os
from src.utils.config import load_config

#Paths
config = load_config("configs/paths.yaml")
cache_dir = config["cache_dir"]

#Set up csv for results
results_dir = config["results_dir"]
filename = "out.csv"
results_file = os.path.join(results_dir, "E1_and_E2", filename)
results_row = {"Encoder":None,
               "Image_Idx":None,
               "Patch_Idx":None,
               "Layer":None,
               "PSNR":None,
               "SSIM":None,
               "MSE":None,
               "F1":None,
               "IoU":None
                } #tokens by encoder identified by image and patch

#Extract different layers at which token was extracted, including final layer
encoder_names = [
    name for name in os.listdir(cache_dir)
    if os.path.isdir(os.path.join(cache_dir, name))
]
print(f"Encoders: {encoder_names}")

#For each encoder
for encoder in encoder_names:
    encoder_path = os.path.join(cache_dir, encoder)
    encoder_files = [f for f in os.listdir(encoder_path) if os.path.isfile(os.path.join(encoder_path, f))]
    encoder_val = [f for f in encoder_files if "val" in f]
    encoder_train = [f for f in encoder_files if "train" in f]

    #Load trained MLP model for the probe
    encoder_best_probe = ""