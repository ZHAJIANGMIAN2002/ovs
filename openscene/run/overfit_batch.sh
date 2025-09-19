#!/bin/bash

# This script runs the overfitting test for the PTV3 model on a single batch of ScanNet data.

# Define an experiment directory for the overfitting test.
exp_dir=out/overfit_scannet_ptv3
config=config/scannet/train_ptv3.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Create the experiment directory if it doesn't exist.
mkdir -p ${exp_dir}

# Set the PYTHONPATH to include the project root and the Pointcept library.
export PYTHONPATH=.:../Pointcept

# Execute the overfitting Python script, passing the config file and a specific save path.
python -u run/overfit_batch.py \
    --config=${config} \
    save_path ${exp_dir} 