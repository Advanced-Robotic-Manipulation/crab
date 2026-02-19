#!/bin/bash
# Crab Robot Web Control Panel launcher
# Run on Aibek: ./start_webapp.sh
# Then open http://aibek:8080 from any browser

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mobile-robot

cd ~/AnywhereVLA/lerobot/examples/crab/webapp
python main.py
