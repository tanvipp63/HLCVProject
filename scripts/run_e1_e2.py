#imports
from pathlib import Path
import argparse
import time
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from src.utils.config import load_config
from src.feature_extraction import FeatureCache, CachedFeatureDataset
from src.encoders import DINOEncoder, CLIPEncoder, SigLIPEncoder
from src.probes import MLPProbe
from src.patch_extraction import PatchTargets, CachedTargetDataset
from src.metrics import rgb, binary
import csv

#Paths
config = load_config("configs/paths.yaml")
cache_dir = config["cache_dir"]
cache = FeatureCache(config["cache_dir"])
checkpoint_dir = Path(f"{config['checkpoints_dir']}")

#Set up csv for results
results_dir = config["results_dir"]
filename = "out.csv"
results_file = os.path.join(results_dir, "E1_E2", filename)
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

    print("Starting run_e1_e2.py", flush=True)

    for layer in args.layers:
        print(f"\n========== Evaluating layer={layer} ==========")

        feature_dataset = CachedFeatureDataset(
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

        target_dataset = CachedTargetDataset(
            target_generator=target_generator,
            split="val",
            target_type=args.target_type,
            encoder_name=args.encoder,
        )

        assert len(feature_dataset) == len(target_dataset)
        print(f"Val cached batches: {len(feature_dataset)}", flush=True)

        #Load trained MLP model for the probe
        if layer == -1:
            layer_dir = "final"
        else:
            layer_dir = f"layer{layer}"
        checkpoint_path = checkpoint_dir / args.target_type / args.encoder / layer_dir / "best_model.pt"

        if args.target_type == "rgb":
            output_channels = 3
        else:
            output_channels = 1

        probe = MLPProbe(
            input_dim=encoder.embedding_dim,
            patch_size=encoder.patch_size,
            output_channels=output_channels,
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

        global_image_idx = 1

        print(
            f"Loading checkpoint: {checkpoint_path}",
            flush=True,
        )

        print(
            f"Running evaluation on {len(feature_dataset)} cached validation batches.",
            flush=True,
        )

        with torch.no_grad():
            for batch_idx, (feature_batch, target_batch) in enumerate(
                zip(
                    feature_dataset,
                    target_dataset,
                )
            ):
                batch_start = time.perf_counter()

                feature_batch = feature_batch.to(device)
                target_batch = target_batch.to(device)

                image_loader = DataLoader(
                    TensorDataset(feature_batch, target_batch),
                    batch_size=args.batch_size,
                    shuffle=False,
                )

                if batch_idx % 10 == 0:
                    print(
                        f"Evaluating cached batch {batch_idx} / {len(feature_dataset)}",
                        flush=True,
                    )

                for batch_features, batch_targets in image_loader:
                    batch_features = batch_features.to(device)
                    prediction_batch = probe(batch_features)
                    assert prediction_batch.shape == batch_targets.shape

                    for pred, target in zip(prediction_batch, batch_targets):
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
                                "image": global_image_idx,
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
                                "image": global_image_idx,
                                "psnr": None,
                                "ssim": None,
                                "mse": None,
                                "f1": f1,
                                "iou": iou,
                            })

                        global_image_idx += 1

                batch_time = time.perf_counter() - batch_start
                print(
                    f"Layer {layer} | Cached batch {batch_idx} / {len(feature_dataset)} | Batch Time: {batch_time:.2f} s",
                    flush=True,
                )
    


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
    
    print(f"Wrote {len(results)} rows to {results_path}")