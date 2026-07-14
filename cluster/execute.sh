#!/usr/bin/env bash

# Path to the `.py` file you want to run
PROJECT_ROOT="/home/hlcv_team015/HLCVProject/"
# Path to the Python binary of the conda environment
PYTHON="/home/hlcv_team015/miniconda3/envs/hlcv/bin/python"

cd $PROJECT_ROOT
$PYTHON "$@"
