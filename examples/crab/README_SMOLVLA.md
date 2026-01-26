# SmolVLA Setup for Crab Robot

## Environment
- **Conda env**: lerobot_vla
- **PyTorch**: 2.5.1 (CUDA enabled)
- **TorchVision**: 0.20.1
- **Device**: Jetson Orin NX 16GB

## Input/Output Configuration

### Inputs (3 cameras + state)
- observation.images.main_camera: (3, 480, 640)
- observation.images.left_arm_camera: (3, 480, 640)
- observation.images.right_arm_camera: (3, 480, 640)
- observation.state: (14,) - joint positions + base velocity

### Outputs (14 action dimensions)
| Index | Joint | Description |
|-------|-------|-------------|
| 0-5 | Left arm | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| 6-11 | Right arm | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| 12 | base_x.vel | Linear velocity |
| 13 | base_theta.vel | Angular velocity |

## Files

Location: /home/orin/anywhereVLA/workspaces/miilv_ws/lerobot/examples/crab/

- start_smolvla.sh          # Quick launcher script
- smolvla_inference.py      # Synchronous inference (~10 Hz)
- smolvla_async_server.py   # Async policy server
- smolvla_async_client.py   # Async robot client  
- config_smolvla_crab.py    # Crab-specific SmolVLA config
- README_SMOLVLA.md         # This file

## Usage

### Test Setup
    cd /home/orin/anywhereVLA/workspaces/miilv_ws/lerobot/examples/crab
    ./start_smolvla.sh test

### Synchronous Inference
    ./start_smolvla.sh sync /path/to/your/trained/model "pick up the red block"

### Asynchronous Inference (30% faster)
    # Terminal 1: Start policy server
    ./start_smolvla.sh server 8080

    # Terminal 2: Run robot client
    ./start_smolvla.sh client /path/to/model "task description" localhost:8080

## Architecture

    Qwen 3 8B (Ollama)     SmolVLA (trained)
       Reasoner    --------->   Executor
       - Task decomposition    - Motor commands
       - Haptic reasoning      - 14-dim actions
       - Scene understanding   - 50-action chunks

---
*Last updated: January 2025*
