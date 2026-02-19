#!/usr/bin/env python3
"""
Inference script for the trained CrabXVLA model.

Usage:
    python inference.py \
        --checkpoint outputs/crab_xvla_distill/best \
        --config configs/train_crab_xvla.yaml

Can also serve as a FastAPI server for the robot:
    python inference.py \
        --checkpoint outputs/crab_xvla_distill/best \
        --config configs/train_crab_xvla.yaml \
        --serve --port 8000
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load trained CrabXVLA model from checkpoint."""
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(Path(config_path).parent.parent)

    # Build model
    from training.crab_xvla_wrapper import build_model
    model = build_model(cfg)

    # Load checkpoint weights
    ckpt_dir = Path(checkpoint_path)
    model_path = ckpt_dir / "model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = model.to(device)
    model.eval()

    return model, cfg


def predict(model, processor, images, state, tactile_left, tactile_right,
            task_description: str, domain_id: int = 0, device: str = "cuda",
            num_views: int = 3, max_action_dim: int = 20, real_action_dim: int = 14,
            steps: int = 10):
    """
    Run inference on a single observation.

    Args:
        model: CrabXVLAWrapper model
        processor: XVLAProcessor
        images: list of PIL Images (up to num_views)
        state: np.array [14] robot state
        tactile_left: np.array [100] left tactile
        tactile_right: np.array [100] right tactile
        task_description: str
        domain_id: int
        steps: int denoising steps

    Returns:
        np.array [num_actions, 14] predicted actions
    """
    with torch.no_grad():
        # Encode language
        lang = processor.encode_language(task_description)
        input_ids = lang["input_ids"].to(device)  # [1, L]

        # Encode images
        img_data = processor.encode_image(images)
        image_input = img_data["image_input"].to(device)  # [1, V, C, H, W]
        image_mask = img_data["image_mask"].to(device)  # [1, V]

        # Proprio (pad to max_action_dim)
        proprio = np.zeros(max_action_dim, dtype=np.float32)
        proprio[:real_action_dim] = state[:real_action_dim]
        proprio = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(device)

        # Tactile
        tl = torch.tensor(tactile_left, dtype=torch.float32).unsqueeze(0).to(device)
        tr = torch.tensor(tactile_right, dtype=torch.float32).unsqueeze(0).to(device)

        # Domain ID
        did = torch.tensor([domain_id], dtype=torch.long).to(device)

        # Generate actions
        actions = model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=did,
            proprio=proprio,
            tactile_left=tl,
            tactile_right=tr,
            steps=steps,
        )  # [1, num_actions, 14]

        return actions[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="CrabXVLA Inference")
    parser.add_argument("--config", "-c", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=10, help="Denoising steps")
    parser.add_argument("--serve", action="store_true", help="Run as FastAPI server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model, cfg = load_model(args.config, args.checkpoint, args.device)
    processor = model.processor
    model_cfg = cfg["model"]

    logger.info(f"Model loaded. Device: {args.device}")
    logger.info(f"Action dim: {model_cfg['real_action_dim']} real, {model_cfg['max_action_dim']} padded")
    logger.info(f"Num actions: {model_cfg['num_actions']}")

    if args.serve:
        logger.info(f"Starting FastAPI server on port {args.port}...")
        # Could use X-VLA's built-in FastAPI server or a custom one
        # For now, just print usage
        logger.info("Server mode not yet implemented. Use predict() directly.")
    else:
        logger.info("Interactive mode. Use predict() function for inference.")
        logger.info("Example:")
        logger.info("  from inference import load_model, predict")
        logger.info("  model, cfg = load_model('configs/train_crab_xvla.yaml', 'outputs/crab_xvla_distill/best')")
        logger.info("  actions = predict(model, model.processor, images, state, tl, tr, 'pick and place')")


if __name__ == "__main__":
    main()
