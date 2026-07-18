# Characterizing Sub-Patch Spatial Information in Vision Foundation Model Representations

High-Level Computer Vision Project (Summer 2026)

## Overview

This repository studies whether patch-token embeddings from vision foundation models retain recoverable sub-patch spatial information.

The current codebase supports:

- Baseline feature extraction from DINO, CLIP, and SigLIP
- Patch-level target extraction (RGB and edges)
- Probe training on cached features
- AnyUp-based feature enhancement and re-training of probes
- Side-by-side evaluation of baseline vs AnyUp probes

Current status highlights:

- Supported encoders: dino, clip, siglip
- Implemented targets for extraction: rgb and edges
- Boundary extraction is not implemented yet (see src/patch_extraction/targets.py)

Main experiment scripts:

- E1/E2 evaluation: scripts/run_e1_e2.py
- E3 comparison (Original vs AnyUp): scripts/run_e3.py

## End-to-End Pipeline

1. Extract baseline features for train and val
2. Extract patch targets for train and val
3. Train baseline probes
4. Extract AnyUp-enhanced features (train and val in one run)
5. Train AnyUp probes
6. Compare baseline vs AnyUp using E3

## Setup

### 1) Environment

Option A (recommended): conda

```bash
conda env create -f environment.yml
conda activate hlcv
```

Option B: pip

```bash
pip install -r requirements.txt
```

### 2) Paths configuration

Create configs/paths.yaml from configs/paths_example.yml and fill in real paths.

Example:

```yaml
imagenet_root: /path/to/imagenet
cache_dir: cache/
results_dir: results/
checkpoints_dir: checkpoints/
```

Expected ImageNet layout:

- /path/to/imagenet/train
- /path/to/imagenet/val

## Core Scripts

### Feature and target extraction

- scripts/extract_features.py
   - Extracts cached encoder features for one split at a time
   - Key args: --encoder, --split, --layers, --max_samples

- scripts/extract_targets.py
   - Extracts cached probe targets for one split at a time
   - Currently writes RGB and edges targets
   - Key args: --encoder, --split, --max_samples

- scripts/extract_anyup_features.py
   - Loads cached baseline features and applies AnyUp
   - Processes both train and val in one run (no --split argument)
   - Uses cached baseline final-layer features (layer = -1) as input
   - Key args: --encoder, --max_train_cached_batches, --max_val_cached_batches, --micro_batch_size

### Probe training

- scripts/train_probes.py
   - Trains probes on baseline cached features
   - Key args: --encoder, --layers, --target_type, --max_train_cached_batches, --max_val_cached_batches, --num_epochs

- scripts/train_probes_anyup.py
   - Same training flow, but uses cache_dir/anyup as input cache
   - Checkpoints are stored under <encoder>_anyup
   - Key args: same as scripts/train_probes.py

### Evaluation

- scripts/run_e1_e2.py
   - Evaluates a single probe family (baseline)
   - Does not expose a max_val_cached_batches argument
   - Writes per-image metrics to results/E1_E2/out.csv

- scripts/run_e3.py
   - Loads baseline and AnyUp probes for the same encoder/layer
   - Evaluates both on the same validation targets
   - Supports --max_val_cached_batches to limit validation cached batches
   - Writes per-image metrics to results/E3/out.csv with probe_type column

## Preconditions Per Stage

Run stages in this order to avoid missing-cache or missing-checkpoint errors.

1. Before scripts/extract_anyup_features.py:
   - Baseline feature cache must already exist for the same encoder and split.
   - Required input path pattern:
     - cache/<encoder>/<split>/final/batch_*.pt
2. Before scripts/train_probes.py:
   - Baseline feature caches and matching target caches must exist.
3. Before scripts/train_probes_anyup.py:
   - AnyUp feature caches and matching target caches must exist.
4. Before scripts/run_e1_e2.py:
   - Baseline checkpoints must exist for requested encoder/layer/target.
5. Before scripts/run_e3.py:
   - Both baseline and AnyUp checkpoints must exist for requested encoder/layer/target.

## Typical Commands

### A) Baseline feature extraction (train + val)

```bash
python scripts/extract_features.py --encoder dino --split train --layers -1 3 6 9
python scripts/extract_features.py --encoder dino --split val   --layers -1 3 6 9
```

Repeat for clip and siglip.

If you only need final-layer features, omit --layers and defaults will be used by the script.

### B) Target extraction (train + val)

```bash
python scripts/extract_targets.py --encoder dino --split train
python scripts/extract_targets.py --encoder dino --split val
```

Repeat for clip and siglip.

If you plan to train with --target_type boundaries, note that boundary extraction is not currently implemented.

### C) Baseline probe training

```bash
python -m scripts.train_probes --encoder dino --target_type rgb --layers -1 3 6 9 --max_train_cached_batches 500 --max_val_cached_batches 100
```

### D) AnyUp feature extraction

```bash
python scripts/extract_anyup_features.py --encoder dino --max_train_cached_batches 500 --max_val_cached_batches 100 --micro_batch_size 8
```

### E) AnyUp probe training

```bash
python -m scripts.train_probes_anyup --encoder dino --target_type rgb --layers -1 3 6 9 --max_train_cached_batches 500 --max_val_cached_batches 100
```

### F) E3 comparison (Original vs AnyUp)

```bash
python -m scripts.run_e3 --encoder dino --target_type rgb --layers -1 3 6 9 --max_val_cached_batches 100
```

Note: results/E3/out.csv is opened in append mode. Remove or archive old CSV files if you need a clean run-level output.

## Cache and Checkpoint Layout

### Feature cache

Baseline features:

- cache/<encoder>/<split>/final/batch_00000.pt
- cache/<encoder>/<split>/layer3/batch_00000.pt
- cache/<encoder>/<split>/layer6/batch_00000.pt
- cache/<encoder>/<split>/layer9/batch_00000.pt

AnyUp features:

- cache/anyup/<encoder>/<split>/final/batch_00000.pt

Note: AnyUp extraction currently writes final layer features (layer = -1 cache path semantics still map to final).

### Targets cache

- cache/targets/<split>/<encoder>/rgb/batch_00000.pt
- cache/targets/<split>/<encoder>/edges/batch_00000.pt

### Probe checkpoints

Baseline:

- checkpoints/<target_type>/<encoder>/<layer_dir>/best_model.pt

AnyUp:

- checkpoints/<target_type>/<encoder>_anyup/<layer_dir>/best_model.pt

Where layer_dir is final for layer -1, otherwise layer<index>.

## Cluster Usage (HTCondor)

The cluster folder contains ready-to-submit job files and a shared launcher:

- cluster/execute.sh
   - cd to project root on cluster
   - run configured Python binary with passed arguments

Recent AnyUp/E3 submit files support 4 layers in one file using queue Layer from:

- cluster/train_probes_anyup_dino.sub
- cluster/train_probes_anyup_clip.sub
- cluster/train_probes_anyup_siglip.sub
- cluster/run_e3_dino.sub
- cluster/run_e3_clip.sub
- cluster/run_e3_siglip.sub

These files currently target rgb and are configured with batch limits used for cluster-scale runs.

Example submit:

```bash
condor_submit cluster/train_probes_anyup_dino.sub
condor_submit cluster/run_e3_dino.sub
```

## Repository Structure

- configs/
   - paths_example.yml: template for paths.yaml
   - paths.yaml: local machine or cluster-specific absolute/relative paths

- scripts/
   - extract_features.py
   - extract_targets.py
   - extract_anyup_features.py
   - train_probes.py
   - train_probes_anyup.py
   - run_e1_e2.py
   - run_e3.py

- src/
   - datasets/: ImageNet data wrapper
   - encoders/: DINO, CLIP, SigLIP wrappers
   - feature_extraction/: extractor and batch cache API
   - patch_extraction/: patch utilities and target caching
   - metrics/: RGB and binary metrics
   - probes/: MLP probe and trainer
   - utils/: config loading

- cluster/: HTCondor submit files and execute.sh
- notebooks/: experiment notebooks
- results/: CSV outputs and analysis artifacts

## Current Notes

- scripts/run_e4.py exists but is currently empty.
- Boundary target type is present as an interface option in some scripts but extraction is not implemented in PatchTargets.
- Batch-cache datasets are intentionally lazy: each dataset item is one cached batch tensor.
- Feature/target dataset lengths can differ in partial-cache scenarios; training and E3 scripts cap usage to effective min counts or explicit max_*_cached_batches limits.
