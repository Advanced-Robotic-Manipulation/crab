#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crab_vla
cd ~/crab
exec ~/bin/watch-proc --run "python train.py --config configs/train_6dof_manipulation.yaml --resume outputs/crab_smolvla_6dof_tactile/step_12500"
