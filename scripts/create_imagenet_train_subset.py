from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import argparse
import random


def main():

    parser = argparse.ArgumentParser(
        description="Create a balanced ImageNet subset of training data directly from the Kaggle ZIP."
    )

    parser.add_argument(
        "--zip_path",
        type=Path,
        required=True,
        help="Path to imagenet-object-localization-challenge.zip",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the subset will be extracted.",
    )

    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    train_prefix = "ILSVRC/Data/CLS-LOC/train/"

    class_to_files = defaultdict(list)

    print("Scanning ZIP...")

    with ZipFile(args.zip_path, "r") as z:

        # --------------------------------------------------
        # Build mapping: class -> list of JPEG members
        # --------------------------------------------------
        for member in z.namelist():

            if (
                member.startswith(train_prefix)
                and member.endswith(".JPEG")
            ):

                rel = member[len(train_prefix):]

                cls = rel.split("/")[0]

                class_to_files[cls].append(member)

        print(f"Found {len(class_to_files)} classes.")

        total_images = 0

        # --------------------------------------------------
        # Sample and extract
        # --------------------------------------------------
        for idx, cls in enumerate(sorted(class_to_files)):

            files = class_to_files[cls]

            if len(files) < args.samples_per_class:
                raise RuntimeError(
                    f"{cls} only contains {len(files)} images."
                )

            chosen = random.sample(
                files,
                args.samples_per_class,
            )

            out_dir = args.output_dir / "train" / cls
            out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for member in chosen:

                filename = Path(member).name

                with z.open(member) as src, open(out_dir / filename, "wb") as dst:
                    dst.write(src.read())

            total_images += len(chosen)

            if (idx + 1) % 50 == 0 or idx + 1 == len(class_to_files):
                print(
                    f"[{idx + 1:4d}/{len(class_to_files)}] "
                    f"{total_images:,} images extracted..."
                )

    print()
    print("Done!")
    print(f"Classes: {len(class_to_files)}")
    print(f"Images : {total_images:,}")


if __name__ == "__main__":
    main()