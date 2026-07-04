#imports
from pathlib import Path
import argparse
import torch
import os
from src.utils.config import load_config
from src.feature_extraction import FeatureExtractor, FeatureCache
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.probes import MLPProbe
from src.patch_extraction.targets import PatchTargets
from src.metrics import rgb, binary, segmentation
import csv

#Paths
config = load_config("configs/paths.yaml")
cache_dir = config["cache_dir"]
cache = FeatureCache(config["cache_dir"])
checkpoint_dir = Path(f"{config['checkpoints_dir']}")

#Set up csv for results
results_dir = config["results_dir"]
filename = "out.csv"
results_file = os.path.join(results_dir, "E1_and_E2", filename)
results = []

#Function to reconstruct image from patches
def reconstruct_image(
    patches: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Reconstruct an image from predicted patches.

    Args:
        patches: (num_patches, C, patch_size, patch_size)
        image_size: e.g. 224
        patch_size: e.g. 14

    Returns:
        (C, image_size, image_size)
    """

    channels = patches.shape[1]
    patches_per_side = image_size // patch_size

    image = torch.zeros(
        channels,
        image_size,
        image_size,
        device=patches.device,
    )

    idx = 0

    for row in range(patches_per_side):
        for col in range(patches_per_side):

            y0 = row * patch_size
            y1 = y0 + patch_size

            x0 = col * patch_size
            x1 = x0 + patch_size

            image[:, y0:y1, x0:x1] = patches[idx]
            idx += 1

    return image

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
    parser.add_argument(
        "--target_type",
        choices=["rgb", "edges", "boundaries"],
        required=True
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

    target_generator = PatchTargets(
        image_size=224,
        patch_size=encoder.patch_size,
        cache_dir=cache_dir,
    )

    for layer in args.layers:
        features = cache.load(args.encoder, "val", layer=layer)["features"]

        #Load trained MLP model for the probe
        if layer == -1:
            layer_dir = "final"
        else:
            layer_dir = f"layer{layer}"
        checkpoint_path = checkpoint_dir / args.target_type / args.encoder / layer_dir / "best_model.pt"
        
        probe = MLPProbe(
            input_dim=encoder.embedding_dim,
            patch_size=encoder.patch_size,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        probe.load_state_dict(
            checkpoint["model_state_dict"]
        )

        probe.to(device)
        probe.eval()

        predictions = []

        with torch.no_grad():
            for image_features in features:

                image_features = image_features.to(device)

                pred = probe(image_features)

                predictions.append(pred)
        
        predictions = torch.stack(predictions)

        targets = target_generator.load("val", args.target_type, args.encoder)
        assert predictions.shape == targets.shape

        # Evaluate
        for image_idx, (pred, target) in enumerate(zip(predictions, targets)):

            target = target.to(device)

            if args.target_type == "rgb":

                pred_image = reconstruct_image(
                    pred,
                    image_size=224,
                    patch_size=encoder.patch_size,
                )

                target_image = reconstruct_image(
                    target,
                    image_size=224,
                    patch_size=encoder.patch_size,
                )

                psnr = rgb.psnr(pred_image, target_image)
                ssim = rgb.ssim(pred_image, target_image)
                mse = rgb.mse(pred_image, target_image)

                results.append({
                    "encoder": args.encoder,
                    "layer": layer,
                    "target": args.target_type,
                    "image": image_idx+1, #imagenet indexes start from 1
                    "psnr": psnr,
                    "ssim": ssim,
                    "mse": mse,
                    "f1": None,
                    "iou": None,
                })

            else:

                f1 = binary.dice_score(pred, target)
                iou = binary.iou_score(pred, target)

                results.append({
                    "encoder": args.encoder,
                    "layer": layer,
                    "target": args.target_type,
                    "image": image_idx,
                    "psnr": None,
                    "ssim": None,
                    "mse": None,
                    "f1": f1.item(),
                    "iou": iou.item(),
                })
    


    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "encoder",
                "layer",
                "target",
                "image",
                "psnr",
                "ssim",
                "mse",
                "f1",
                "iou",
            ],
        )

        if write_header:
            writer.writeheader()

        writer.writerows(results)