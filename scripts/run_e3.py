#imports
from pathlib import Path
import argparse
import time
import torch
import os
from itertools import islice
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
cache_original = FeatureCache(config["cache_dir"])
cache_anyup = FeatureCache(Path(config["cache_dir"]) / "anyup")
checkpoint_dir = Path(f"{config['checkpoints_dir']}")

#Set up csv for results
results_dir = config["results_dir"]
filename = "out.csv"
results_file = os.path.join(results_dir, "E3", filename)
all_results = []

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


def evaluate_probe(
    probe,
    feature_dataset,
    target_dataset,
    encoder,
    device,
    layer,
    batch_size,
    target_type,
    probe_type,
    encoder_name,
    max_val_cached_batches=None,
):
    """
    Evaluate a single probe on validation data.
    
    Args:
        probe: MLPProbe model in eval mode
        feature_dataset: CachedFeatureDataset
        target_dataset: CachedTargetDataset
        encoder: Encoder (for patch_size)
        device: torch.device
        layer: layer index
        batch_size: evaluation batch size
        target_type: "rgb", "edges", or "boundaries"
        probe_type: "original" or "anyup"
        encoder_name: "dino", "clip", or "siglip"
        max_val_cached_batches: max cached batches to evaluate (None = all)
    
    Returns:
        list of result dicts
    """
    
    # Compute effective number of cached batches to evaluate
    effective_val_batches = len(feature_dataset)
    if max_val_cached_batches is not None:
        effective_val_batches = min(effective_val_batches, max_val_cached_batches)
    
    global_image_idx = 1
    batch_results = []
    
    with torch.no_grad():
        iterator = zip(
            feature_dataset,
            target_dataset,
        )
        
        iterator = islice(
            iterator,
            effective_val_batches,
        )
        
        for batch_idx, (feature_batch, target_batch) in enumerate(iterator):
            batch_start = time.perf_counter()

            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)

            image_loader = DataLoader(
                TensorDataset(feature_batch, target_batch),
                batch_size=batch_size,
                shuffle=False,
            )

            if batch_idx % 10 == 0:
                print(
                    f"[{probe_type}] Evaluating cached batch {batch_idx} / {effective_val_batches}",
                    flush=True,
                )

            for batch_features, batch_targets in image_loader:
                prediction_batch = probe(batch_features)
                assert prediction_batch.shape == batch_targets.shape

                for pred, target in zip(prediction_batch, batch_targets):
                    if target_type == "rgb":

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

                        batch_results.append({
                            "encoder": encoder_name,
                            "layer": layer,
                            "target": target_type,
                            "probe_type": probe_type,
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

                        batch_results.append({
                            "encoder": encoder_name,
                            "layer": layer,
                            "target": target_type,
                            "probe_type": probe_type,
                            "image": global_image_idx,
                            "psnr": None,
                            "ssim": None,
                            "mse": None,
                            "f1": f1.item(),
                            "iou": iou.item(),
                        })

                    global_image_idx += 1

            batch_time = time.perf_counter() - batch_start
            print(
                f"[{probe_type}] Layer {layer} | Cached batch {batch_idx} / {effective_val_batches} | "
                f"Batch Time: {batch_time:.2f} s",
                flush=True,
            )
    
    return batch_results


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
        "--max_val_cached_batches",
        type=int,
        default=None,
        help="Maximum number of cached validation batches to use.",
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

    print("Starting run_e3.py (Original vs AnyUp Comparison)", flush=True)
    print(f"Encoder: {args.encoder}", flush=True)
    print(f"Target type: {args.target_type}", flush=True)

    for layer in args.layers:
        print(f"\n========== Evaluating layer={layer} ==========")

        # Load target dataset (shared)
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

        # Determine layer directory
        if layer == -1:
            layer_dir = "final"
        else:
            layer_dir = f"layer{layer}"

        # Determine output channels
        if args.target_type == "rgb":
            output_channels = 3
        else:
            output_channels = 1

        # ============================================================
        # ORIGINAL PROBE EVALUATION
        # ============================================================

        print(f"\n--- Original Probe ---", flush=True)

        feature_dataset_original = CachedFeatureDataset(
            cache=cache_original,
            encoder_name=args.encoder,
            split="val",
            layer=layer,
        )

        assert len(feature_dataset_original) == len(target_dataset)
        
        # Compute effective batch count for original
        effective_val_batches_original = len(feature_dataset_original)
        if args.max_val_cached_batches is not None:
            effective_val_batches_original = min(effective_val_batches_original, args.max_val_cached_batches)
        
        print(f"Original val cached batches: {len(feature_dataset_original)} (using: {effective_val_batches_original})", flush=True)

        checkpoint_path_original = (
            checkpoint_dir / args.target_type / args.encoder / layer_dir / "best_model.pt"
        )

        probe_original = MLPProbe(
            input_dim=encoder.embedding_dim,
            patch_size=encoder.patch_size,
            output_channels=output_channels,
        )

        checkpoint_original = torch.load(
            checkpoint_path_original,
            map_location=device,
        )

        probe_original.load_state_dict(
            checkpoint_original["model_state_dict"]
        )

        probe_original.to(device)
        probe_original.eval()

        print(
            f"Loaded checkpoint: {checkpoint_path_original}",
            flush=True,
        )

        results_original = evaluate_probe(
            probe=probe_original,
            feature_dataset=feature_dataset_original,
            target_dataset=target_dataset,
            encoder=encoder,
            device=device,
            layer=layer,
            batch_size=args.batch_size,
            target_type=args.target_type,
            probe_type="original",
            encoder_name=args.encoder,
            max_val_cached_batches=args.max_val_cached_batches,
        )

        all_results.extend(results_original)

        # ============================================================
        # ANYUP PROBE EVALUATION
        # ============================================================

        print(f"\n--- AnyUp Probe ---", flush=True)

        feature_dataset_anyup = CachedFeatureDataset(
            cache=cache_anyup,
            encoder_name=args.encoder,
            split="val",
            layer=layer,
        )

        assert len(feature_dataset_anyup) == len(target_dataset)
        
        # Compute effective batch count for AnyUp
        effective_val_batches_anyup = len(feature_dataset_anyup)
        if args.max_val_cached_batches is not None:
            effective_val_batches_anyup = min(effective_val_batches_anyup, args.max_val_cached_batches)
        
        print(f"AnyUp val cached batches: {len(feature_dataset_anyup)} (using: {effective_val_batches_anyup})", flush=True)

        checkpoint_path_anyup = (
            checkpoint_dir / args.target_type / f"{args.encoder}_anyup" / layer_dir / "best_model.pt"
        )

        probe_anyup = MLPProbe(
            input_dim=encoder.embedding_dim,
            patch_size=encoder.patch_size,
            output_channels=output_channels,
        )

        checkpoint_anyup = torch.load(
            checkpoint_path_anyup,
            map_location=device,
        )

        probe_anyup.load_state_dict(
            checkpoint_anyup["model_state_dict"]
        )

        probe_anyup.to(device)
        probe_anyup.eval()

        print(
            f"Loaded checkpoint: {checkpoint_path_anyup}",
            flush=True,
        )

        results_anyup = evaluate_probe(
            probe=probe_anyup,
            feature_dataset=feature_dataset_anyup,
            target_dataset=target_dataset,
            encoder=encoder,
            device=device,
            layer=layer,
            batch_size=args.batch_size,
            target_type=args.target_type,
            probe_type="anyup",
            encoder_name=args.encoder,
            max_val_cached_batches=args.max_val_cached_batches,
        )

        all_results.extend(results_anyup)

        # ============================================================
        # COMPARISON
        # ============================================================

        print(f"\n========== COMPARISON: layer={layer} ==========", flush=True)

        if args.target_type == "rgb":
            metrics = ["psnr", "ssim", "mse"]
        else:
            metrics = ["f1", "iou"]

        for metric in metrics:
            original_vals = [r[metric] for r in results_original if r[metric] is not None]
            anyup_vals = [r[metric] for r in results_anyup if r[metric] is not None]

            if original_vals:
                original_mean = sum(original_vals) / len(original_vals)
                anyup_mean = sum(anyup_vals) / len(anyup_vals)
                diff = anyup_mean - original_mean

                print(
                    f"{metric.upper():8} | Original: {original_mean:.6f} | "
                    f"AnyUp: {anyup_mean:.6f} | Diff: {diff:+.6f}",
                    flush=True,
                )

    # ================================================================
    # SAVE RESULTS TO CSV
    # ================================================================

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
                "probe_type",
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

        writer.writerows(all_results)
    
    print(f"\nWrote {len(all_results)} rows to {results_path}", flush=True)
