# Characterizing Sub-Patch Spatial Information in Vision Foundation Model Representations

High-Level Computer Vision Project (Summer 2026)

## Project Overview

This repository explores whether transformer patch tokens from vision foundation models retain recoverable sub-patch spatial information. It includes tools to:

- extract patch-token features from pretrained visual encoders
- cache extracted features for later experiment use
- map patch token indices back to image patch coordinates
- define and train probe models for patch reconstruction

The code is structured to support multiple backbones, including CLIP, DINOv2, and SIGLIP.

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
   - or create the Conda environment from `environment.yml`

2. Prepare config:
   - Copy `configs/paths_example.yml` to `configs/paths.yml`
   - Set `imagenet_root` to your ImageNet root containing `train/` and `val/`
   - Ensure `cache_dir`, `results_dir`, and `checkpoints_dir` are valid paths

3. Prepare data:
   - Place ImageNet data under the configured `imagenet_root`
   - Expected structure:
     - `imagenet_root/train/`
     - `imagenet_root/val/`

4. Run feature extraction:
   - `python scripts/extract_features.py --encoder clip --split val --layers 3 6 9 11`
   - or use the equivalent CLI in `src/anyup/anyup.py`

5. Run patch/target extraction:
   - `python scripts/extract_targets.py --encoder clip --split val --layers 3 6 9 11`
   - or use the equivalent CLI in `src/anyup/anyup.py`

6. Train probes per encoder and per layer of interest:
   - `python -m scripts.train_probes --encoder dino --layers 3 --target_type rgb --num_epochs 20`
   - train by encoder, layer, target and num epochs.

7. Run Experiments:
   - `python -m scripts.run_e1ANDe2 --encoder dino --layers 3 --target_type rgb`

## Repository Layout

- `configs/`
  - `paths.yml` / `paths_example.yml` — dataset and cache paths
  - `default.yml`, `clip.yml`, `dino.yml`, `siglip.yml` — additional experiment configuration examples

- `scripts/`
  - `extract_features.py` — main feature extraction script
  - `test_clip.py`, `test_dino.py`, `test_siglip.py`, `test_probe.py`, `test_patch_extractor.py` — lightweight script checks for key functionality
  - `run_e1.py`, `run_e2.py`, `run_e3.py`, `run_e4.py` — placeholder files for experiment pipelines

- `src/`
  - `anyup/` — AnyUp-related CLI and helper code
  - `datasets/` — dataset wrappers
  - `encoders/` — pretrained model encoder wrappers
  - `feature_extraction/` — feature extraction and caching utilities
  - `patch_extraction/` — patch coordinate utilities
  - `probes/` — probe model and training utilities
  - `utils/` — common utilities like config loading

- `cache/` — saved feature caches created by extraction scripts
- `results/` — output figures and tables
- `cluster/` — HPC job templates
- `notebooks/` — analysis notebooks

## Core Components

### `src/datasets/imagenet.py`

- `ImageNetDataset` wraps `torchvision.datasets.ImageFolder`
- supports `train` / `val` splits and optional random subsampling via `max_samples`
- exposes `get_dataloader()` for PyTorch training/evaluation loops

### `src/encoders/`

- `CLIPEncoder` — CLIP ViT-B/16 via `open_clip`
- `DINOEncoder` — DINOv2 ViT-B/14 via `torch.hub`
- `SigLIPEncoder` — SIGLIP model via Hugging Face `transformers`

Each encoder provides:
- `get_transform()` for image preprocessing
- `extract_features(images, layers=None)` to return patch-token features
- intermediate layer extraction support when `layers` is provided

### `src/feature_extraction/`

- `FeatureExtractor` runs a model over a dataset and aggregates features and labels
- `FeatureCache` saves and loads results in `cache/<encoder>/`
- cache naming convention:
  - final layer: `<split>_final.pt`
  - intermediate layer: `<split>_layer<layer>.pt`

### `src/patch_extraction/extractor.py`

- `PatchExtractor` converts patch token indices into pixel patch coordinates
- supports extraction of individual or multiple patches from an image tensor

### `src/probes/`

- `MLPProbe` reconstructs image patches from token features
- `ProbeTrainer` provides generic training, validation, checkpoint save/load methods

### `src/utils/config.py`

- `load_config(config_path)` loads YAML config files
- resolves relative paths and supports both `.yaml` and `.yml` extensions

## Running the Code

### Feature extraction

Use the extraction script to build cached feature files for a given encoder and ImageNet split.

Example:

```bash
python scripts/extract_features.py --encoder dino --split val --layers 3 6 9 11
```

This will:
- load the DINO encoder
- create an `ImageNetDataset` using `configs/paths.yml`
- extract patch-token features for the requested layers
- save cache files under `cache/dino/`

### Running tests / sanity checks

Use the test scripts to verify key components:

```bash
python scripts/test_clip.py
python scripts/test_dino.py
python scripts/test_siglip.py
python scripts/test_patch_extractor.py
python scripts/test_probe.py
```

Each test script performs a minimal forward pass and prints feature shapes or probe outputs.

## Configuration

The primary configuration entrypoint is `configs/paths.yml`.

Example:

```yaml
imagenet_root: /path/to/imagenet
cache_dir: cache/
results_dir: results/
checkpoints_dir: checkpoints/
```

Additional config files like `configs/default.yml`, `configs/clip.yml`, `configs/dino.yml`, and `configs/siglip.yml` can be used as templates for experiment settings.

## Notes

- `ImageNetDataset` expects a standard ImageNet directory layout with `train/` and `val/` subfolders.
- The code is written for PyTorch and uses GPU when available.
- The repo includes both `requirements.txt` and `environment.yml` for reproducible setup.
