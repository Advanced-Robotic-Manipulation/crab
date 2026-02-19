#!/usr/bin/env python3
"""
CrabXVLA Inference — run trained X-VLA model on the Crab robot.

X-VLA: 0.9B params, Florence-2 backbone, flow matching, 10 denoising steps.
14-DOF output (padded to 20 internally), 30-step action chunks, 224x224 images.

Usage:
  python run_inference_xvla.py --task "Pick and place the can, medium hardness"
  python run_inference_xvla.py --task "Drive up to the box" --steps 200
  python run_inference_xvla.py --test  # test model loading only, no robot
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import base64
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
XVLA_REPO = Path.home() / "X-VLA"
XVLA_MODEL_DIR = Path.home() / "crab_xvla_distill"
DEFAULT_CHECKPOINT = XVLA_MODEL_DIR / "best"
DEFAULT_CONFIG = XVLA_MODEL_DIR / "config.yaml"
CRAB_REMOTE_IP = "192.168.50.239"

ACTION_KEYS = [
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
    "base_x.vel", "base_theta.vel",
]

STATE_KEYS = ACTION_KEYS  # same 14 dims

CAMERA_KEYS = ["main_camera", "left_arm_camera", "right_arm_camera"]

# Safety clamps
MAX_LINEAR_VEL = 0.20
MAX_ANGULAR_VEL = 0.25


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model(cfg: dict, checkpoint_path: str, device: torch.device):
    """Build CrabXVLAWrapper and load trained weights."""
    # Add X-VLA repo and model dir to path
    sys.path.insert(0, str(XVLA_REPO))
    sys.path.insert(0, str(XVLA_MODEL_DIR))

    from training.crab_xvla_wrapper import build_model

    # Use local HF model (avoids HuggingFace download on Orin)
    local_hf_path = Path(checkpoint_path) / "xvla_hf"
    if local_hf_path.exists():
        logger.info(f"Using local HF model at {local_hf_path}")
        cfg["model"]["pretrained_path"] = str(local_hf_path)
    else:
        logger.info("No local HF model found, will download from HuggingFace...")

    cfg["_config_dir"] = str(XVLA_MODEL_DIR)
    model = build_model(cfg)

    ckpt_dir = Path(checkpoint_path)
    model_path = ckpt_dir / "model.pt"
    logger.info(f"Loading checkpoint from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    model.to(device).half().eval()
    logger.info("X-VLA model ready (FP16)")
    return model


def obs_to_xvla_input(obs: dict, task: str, cfg: dict, device: torch.device):
    """Convert raw robot observation to X-VLA model inputs."""
    model_cfg = cfg["model"]
    real_action_dim = model_cfg.get("real_action_dim", 14)
    max_action_dim = model_cfg.get("max_action_dim", 20)
    domain_id = model_cfg.get("domain_id", 19)
    image_size = tuple(cfg["dataset"]["image_size"])  # [224, 224]

    # State — pad to max_action_dim
    state = np.zeros(max_action_dim, dtype=np.float32)
    for i, k in enumerate(STATE_KEYS):
        state[i] = obs.get(k, 0.0)

    # Cameras — convert to PIL Images for X-VLA processor
    pil_images = []
    for cam_key in CAMERA_KEYS:
        img = obs.get(cam_key)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        pil_images.append(Image.fromarray(img))

    # Tactile
    tactile_left = obs.get("tactile_left")
    if tactile_left is None:
        tactile_left = np.zeros(100, dtype=np.float32)
    elif isinstance(tactile_left, str):
        raw_bytes = base64.b64decode(tactile_left)
        tactile_left = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32)
    else:
        tactile_left = np.asarray(tactile_left, dtype=np.float32).flatten()

    tactile_right = obs.get("tactile_right")
    if tactile_right is None:
        tactile_right = np.zeros(100, dtype=np.float32)
    elif isinstance(tactile_right, str):
        raw_bytes = base64.b64decode(tactile_right)
        tactile_right = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32)
    else:
        tactile_right = np.asarray(tactile_right, dtype=np.float32).flatten()

    return {
        "task": task,
        "state": state,
        "images": pil_images,
        "tactile_left": tactile_left,
        "tactile_right": tactile_right,
        "domain_id": domain_id,
        "max_action_dim": max_action_dim,
        "real_action_dim": real_action_dim,
    }


def predict_action_chunk(model, processor, obs_data: dict, device: torch.device, steps: int = 10):
    """Run X-VLA inference on a single observation. Returns [num_actions, 14] action chunk."""
    # Encode language
    lang = processor.encode_language(obs_data["task"])
    input_ids = lang["input_ids"].to(device)

    # Encode images
    img_data = processor.encode_image(obs_data["images"])
    image_input = img_data["image_input"].to(device)
    image_mask = img_data["image_mask"].to(device)

    # Proprio
    proprio = torch.tensor(obs_data["state"], dtype=torch.float32).unsqueeze(0).to(device)

    # Tactile
    tl = torch.tensor(obs_data["tactile_left"], dtype=torch.float32).unsqueeze(0).to(device)
    tr = torch.tensor(obs_data["tactile_right"], dtype=torch.float32).unsqueeze(0).to(device)

    # Domain ID
    did = torch.tensor([obs_data["domain_id"]], dtype=torch.long).to(device)

    # Generate actions [1, num_actions, 14]
    with torch.no_grad(), torch.cuda.amp.autocast():
        actions = model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=did,
            proprio=proprio,
            tactile_left=tl,
            tactile_right=tr,
            steps=steps,
        )

    # Return full action chunk [num_actions, 14]
    return actions[0].float().cpu().numpy()


def build_action_dict(action_np: np.ndarray) -> dict:
    """Build action dict from 14-DOF model output."""
    action_dict = {k: float(action_np[i]) for i, k in enumerate(ACTION_KEYS)}

    # Safety clamp base velocities
    bx = action_dict["base_x.vel"]
    bt = action_dict["base_theta.vel"]
    action_dict["base_x.vel"] = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, bx))
    action_dict["base_theta.vel"] = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, bt))
    return action_dict


def run_test(cfg, checkpoint_path, device, steps=10):
    """Test model loading and forward pass with dummy data."""
    model = load_model(cfg, checkpoint_path, device)
    processor = model.processor
    model_cfg = cfg["model"]

    logger.info("Running test inference with dummy data...")

    # Create dummy PIL images
    dummy_images = [Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(3)]

    obs_data = {
        "task": "Pick and place the can, medium hardness",
        "state": np.zeros(model_cfg.get("max_action_dim", 20), dtype=np.float32),
        "images": dummy_images,
        "tactile_left": np.zeros(100, dtype=np.float32),
        "tactile_right": np.zeros(100, dtype=np.float32),
        "domain_id": model_cfg.get("domain_id", 19),
        "max_action_dim": model_cfg.get("max_action_dim", 20),
        "real_action_dim": model_cfg.get("real_action_dim", 14),
    }

    t0 = time.perf_counter()
    action_chunk = predict_action_chunk(model, processor, obs_data, device, steps=steps)
    dt = time.perf_counter() - t0

    logger.info(f"Output shape: {action_chunk.shape}")
    logger.info(f"Inference time: {dt*1000:.0f}ms")
    logger.info(f"Action range: [{action_chunk.min():.4f}, {action_chunk.max():.4f}]")
    logger.info(f"First action: {action_chunk[0]}")
    logger.info(f"Chunk size: {action_chunk.shape[0]} actions")

    # Warmup benchmark
    logger.info("Benchmarking (3 warmup + 5 timed)...")
    for _ in range(3):
        predict_action_chunk(model, processor, obs_data, device, steps=steps)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        predict_action_chunk(model, processor, obs_data, device, steps=steps)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    avg = np.mean(times)
    logger.info(f"Avg inference: {avg:.0f}ms ({1000/avg:.1f} raw FPS)")
    chunk_time = action_chunk.shape[0] / 10.0  # at 10fps execution
    logger.info(f"Effective FPS with chunking: ~{action_chunk.shape[0] / (chunk_time + avg/1000):.1f}")
    logger.info("TEST PASSED")


def run_inference(cfg, checkpoint_path, task, fps, max_steps, max_episodes,
                  device, robot_ip, denoising_steps):
    from lerobot.robots.crab import CrabClient, CrabClientConfig
    from lerobot.utils.robot_utils import precise_sleep

    model = load_model(cfg, checkpoint_path, device)
    processor = model.processor
    model_cfg = cfg["model"]

    # Warmup
    logger.info("Warming up model...")
    dummy_images = [Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(3)]
    dummy_data = {
        "task": task,
        "state": np.zeros(model_cfg.get("max_action_dim", 20), dtype=np.float32),
        "images": dummy_images,
        "tactile_left": np.zeros(100, dtype=np.float32),
        "tactile_right": np.zeros(100, dtype=np.float32),
        "domain_id": model_cfg.get("domain_id", 19),
        "max_action_dim": model_cfg.get("max_action_dim", 20),
        "real_action_dim": model_cfg.get("real_action_dim", 14),
    }
    for _ in range(3):
        predict_action_chunk(model, processor, dummy_data, device, steps=denoising_steps)
    logger.info("Warmup done")

    # Connect to robot
    robot_config = CrabClientConfig(remote_ip=robot_ip)
    robot = CrabClient(robot_config)
    logger.info(f"Connecting to robot at {robot_ip}...")
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot")
    logger.info("Connected to robot")

    target_dt = 1.0 / fps
    total_steps = 0
    inference_times = []
    num_actions = model_cfg.get("num_actions", 30)

    try:
        for episode in range(max_episodes):
            logger.info(f"\n{'='*50}")
            logger.info(f"Episode {episode + 1}/{max_episodes} | Task: {task}")
            logger.info(f"{'='*50}")

            steps_this_episode = 0

            while steps_this_episode < max_steps:
                # Get fresh observation and predict action chunk
                obs = robot.get_observation()
                obs_data = obs_to_xvla_input(obs, task, cfg, device)

                t0 = time.perf_counter()
                action_chunk = predict_action_chunk(model, processor, obs_data, device, steps=denoising_steps)
                inf_time = (time.perf_counter() - t0) * 1000
                inference_times.append(inf_time)

                # Execute all actions in the chunk at target FPS
                for i in range(min(num_actions, max_steps - steps_this_episode)):
                    step_start = time.perf_counter()

                    action_dict = build_action_dict(action_chunk[i])
                    robot.send_action(action_dict)

                    steps_this_episode += 1
                    total_steps += 1

                    elapsed = time.perf_counter() - step_start
                    if elapsed < target_dt:
                        precise_sleep(target_dt - elapsed)

                logger.info(
                    f"  Step {steps_this_episode}/{max_steps} | "
                    f"inference={inf_time:.0f}ms | chunk={num_actions} | "
                    f"base=({action_dict['base_x.vel']:.3f}, {action_dict['base_theta.vel']:.3f})"
                )

            logger.info(f"Episode {episode + 1} complete ({steps_this_episode} steps)")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    finally:
        logger.info("Disconnecting robot...")
        robot.disconnect()

        if inference_times:
            avg = np.mean(inference_times)
            std = np.std(inference_times)
            chunk_exec_time = num_actions / fps
            effective_fps = num_actions / (avg / 1000 + chunk_exec_time)
            logger.info(f"Inference: {avg:.0f} +/- {std:.0f} ms ({len(inference_times)} chunks)")
            logger.info(f"Effective FPS: ~{effective_fps:.1f} (chunk={num_actions}, exec={chunk_exec_time:.1f}s)")
        logger.info(f"Total steps executed: {total_steps}")


def main():
    parser = argparse.ArgumentParser(description="CrabXVLA Inference")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Path to checkpoint directory")
    parser.add_argument("--config", "-c", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to YAML config")
    parser.add_argument("--task", "-t", type=str, default="Pick and place the can, medium hardness",
                        help="Task description")
    parser.add_argument("--fps", type=int, default=10, help="Control frequency (default 10, X-VLA is slower)")
    parser.add_argument("--steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--denoising-steps", type=int, default=10, help="Flow matching denoising steps")
    parser.add_argument("--robot-ip", type=str, default=CRAB_REMOTE_IP, help="Robot IP")
    parser.add_argument("--test", action="store_true", help="Test model loading only (no robot)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"X-VLA repo: {XVLA_REPO}")

    cfg = load_config(args.config)

    if args.test:
        run_test(cfg, args.checkpoint, device, steps=args.denoising_steps)
    else:
        run_inference(cfg, args.checkpoint, args.task, args.fps, args.steps,
                      args.episodes, device, args.robot_ip, args.denoising_steps)


if __name__ == "__main__":
    main()
