#!/bin/bash
# SmolVLA Quick Start for Crab Robot
#
# Usage:
#   ./start_smolvla.sh sync /path/to/model "pick up the block"
#   ./start_smolvla.sh server                   # Start async server
#   ./start_smolvla.sh client /path/to/model "task"  # Run async client

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="$(dirname $(dirname $SCRIPT_DIR))"

# Activate conda environment
source /home/orin/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_vla

cd "$SCRIPT_DIR"

MODE="${1:-help}"

case "$MODE" in
    sync)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 sync <model_path> <task>"
            echo "Example: $0 sync ./outputs/smolvla_crab \"pick up the red block\""
            exit 1
        fi
        MODEL_PATH="$2"
        TASK="$3"
        echo "Running SmolVLA in synchronous mode..."
        echo "Model: $MODEL_PATH"
        echo "Task: $TASK"
        python smolvla_inference.py --model "$MODEL_PATH" --task "$TASK"
        ;;
        
    server)
        PORT="${2:-8080}"
        echo "Starting SmolVLA async policy server on port $PORT..."
        python smolvla_async_server.py --port "$PORT"
        ;;
        
    client)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 client <model_path> <task> [server_address]"
            echo "Example: $0 client ./outputs/smolvla_crab \"pick up block\" localhost:8080"
            exit 1
        fi
        MODEL_PATH="$2"
        TASK="$3"
        SERVER="${4:-localhost:8080}"
        echo "Running SmolVLA async client..."
        echo "Model: $MODEL_PATH"
        echo "Task: $TASK"
        echo "Server: $SERVER"
        python smolvla_async_client.py --model "$MODEL_PATH" --task "$TASK" --server "$SERVER"
        ;;
        
    test)
        echo "Testing SmolVLA model loading..."
        python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('SmolVLA import: OK')

# Test config
from config_smolvla_crab import get_crab_smolvla_config
config = get_crab_smolvla_config()
print(f'Crab config loaded: OK')
print(f'  Input features: {len(config.input_features)}')
print(f'  Output features: {len(config.output_features)}')
print(f'  Action chunk size: {config.chunk_size}')
"
        ;;
        
    *)
        echo "SmolVLA Quick Start for Crab Robot"
        echo ""
        echo "Usage:"
        echo "  $0 sync <model> <task>         Run synchronous inference"
        echo "  $0 server [port]               Start async policy server"
        echo "  $0 client <model> <task> [srv] Run async client"
        echo "  $0 test                        Test model loading"
        echo ""
        echo "Examples:"
        echo "  $0 test"
        echo "  $0 sync ./outputs/smolvla_crab \"pick up the block\""
        echo "  $0 server 8080"
        echo "  $0 client ./outputs/smolvla_crab \"task\" localhost:8080"
        ;;
esac
