#imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
#generate e1 and e2 plots with:
#python plots.py --target_type rgb --experiment e1_e2 --palette Set2 --point_size 7 --font_size 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for data, by experiment")
    parser.add_argument(
        "--target_type",
        choices=["rgb", "edges", "boundaries"],
        required=True
    )
    parser.add_argument(
        "--experiment",
        choices=["e1_e2", "e3"],
        required=True,
    )
    parser.add_argument(
        "--palette",
        default="tab10",
        help="Seaborn/matplotlib color palette to use for the per-encoder lines (e.g. 'tab10', 'viridis', 'Set2').",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=8,
        help="Marker size (in points) for the line plot markers.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=2,
        help="Line width for the plotted lines.",
    )
    parser.add_argument(
        "--style",
        default="whitegrid",
        help="Seaborn style theme (e.g. 'whitegrid', 'darkgrid', 'ticks').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI to use when saving raster info inside the SVG figure (affects figure sizing crispness).",
    )
    parser.add_argument(
        "--font_size",
        type=float,
        default=12,
        help="Base font size (in points) for axis labels, tick labels, and legend text. The title is scaled up from this.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help=(
            "Total number of transformer blocks in the backbone. DINOv2, CLIP, and SigLIP are all "
            "used here in their ViT-B configuration, which has 12 blocks, so layer == -1 (the final "
            "output layer) is remapped to this value for plotting."
        ),
    )
    args = parser.parse_args()

    sns.set_theme(style=args.style, font_scale=args.font_size / 10)

    #Get data to plot
    results_dir = f"data/{args.experiment}"
    csv_path = f"{results_dir}/{args.target_type}.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    # Single CSV holding all encoders, distinguished by the 'encoder' column
    df = pd.read_csv(csv_path)

    # layer == -1 is Python-style shorthand for "the final layer", i.e. the output after the
    # last transformer block. DINOv2, CLIP, and SigLIP are all run here in their ViT-B config
    # (12 transformer blocks each), so remap -1 -> num_layers for every encoder before plotting,
    # otherwise it would be sorted/plotted as if it were "before" layer 0.
    final_layer_mask = df["layer"] == -1
    if final_layer_mask.any():
        df.loc[final_layer_mask, "layer"] = args.num_layers

    # Pretty-name lookups for titles/axis labels
    metric_names = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "f1": "F1 Score",
        "iou": "IoU",
    }
    processed_names = {
        "mean": "Mean",
        "std": "Std. Dev.",
    }
    target_names = {
        "rgb": "RGB",
        "edges": "Edges",
        "boundaries": "Boundaries",
    }

    if args.experiment == "e1_e2":
        metric_processed = ['mean', 'std']
        if args.target_type == "rgb":
            metrics = ["psnr", "ssim"]
        else:
            metrics = ["f1", "iou"]
        for metric in metrics:
            # Aggregate across images: one value per (encoder, layer) pair
            agg = (
                df.groupby(["encoder", "layer"])[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values(["encoder", "layer"])
            )
            for processed in metric_processed:
                fig, ax = plt.subplots(figsize=(8, 6))

                sns.lineplot(
                    data=agg,
                    x="layer",
                    y=processed,
                    hue="encoder",
                    palette=args.palette,
                    marker="o",
                    markersize=args.point_size,
                    linewidth=args.line_width,
                    ax=ax,
                )

                metric_label = metric_names.get(metric, metric)
                processed_label = processed_names.get(processed, processed)
                target_label = target_names.get(args.target_type, args.target_type)

                ax.set_title(
                    f"{target_label}: {processed_label} {metric_label} by Layer",
                    fontsize=args.font_size * 1.2,
                )
                ax.set_xlabel("Layer", fontsize=args.font_size)
                ax.set_ylabel(f"{processed_label} {metric_label}", fontsize=args.font_size)
                ax.tick_params(axis="both", labelsize=args.font_size * 0.9)
                legend = ax.legend(title="Encoder", fontsize=args.font_size * 0.9)
                legend.get_title().set_fontsize(args.font_size * 0.9)

                # Mark the remapped final-layer tick so it reads as "final", not just a number
                layer_ticks = sorted(agg["layer"].unique())
                tick_labels = [
                    f"{lt} (final)" if lt == args.num_layers else str(lt) for lt in layer_ticks
                ]
                ax.set_xticks(layer_ticks)
                ax.set_xticklabels(tick_labels)

                fig.tight_layout()

                #Save plot
                output = f"{args.experiment}_{args.target_type}_{metric}_{processed}.svg"
                fig.savefig(output, dpi=args.dpi)
                plt.close(fig)
                print(f"Saved {output}")
    else:
        pass #dont have e3 data yet