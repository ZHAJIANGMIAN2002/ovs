#!/bin/sh
set -x

exp_dir=$1
config=$2

mkdir -p ${exp_dir}

export PYTHONPATH=.:../Pointcept

# The python script will handle the multi-gpu spawning internally via mp.spawn
python -u run/distill.py \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/distill-$(date +"%Y%m%d_%H%M").log