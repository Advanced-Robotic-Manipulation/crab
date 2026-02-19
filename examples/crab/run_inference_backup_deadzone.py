#!/usr/bin/env python3
"""
CrabSmolVLA Inference — run trained model on the Crab robot.

Supports both full 14-DOF and right-arm-only 6-DOF models via config.
When action_indices is set in config, only those joints are predicted by
the model; other joints hold their current position, base velocity = 0.

Usage:
  python run_inference.py --task "Pick and place object"
  python run_inference.py --task "Pick and place object" --steps 100 --fps 15
  python run_inference.py --test  # test model loading only, no robot
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import base64
import numpy as np
import torch
from torchvision.transforms.functional import resize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Defaults (right-arm model) ----
DEFAULT_MODEL = Path.home() / "crab_smolvla_distill/outputs/crab_smolvla_distill/best/model.pt"
DEFAULT_CONFIG = Path.home() / "crab_smolvla_distill/configs/train_crab_smolvla_distill.yaml"
TRAINING_PKG = Path.home() / "crab_smolvla_distill"
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
CAMERA_DS_KEYS = [f"observation.images.{k}" for k in CAMERA_KEYS]

# Safety clamps (match training data range)
MAX_LINEAR_VEL = 0.20   # m/s
MAX_ANGULAR_VEL = 0.25  # rad/s

# Deadzone — snap small base velocities to 0 for clean stopping
# Training data: base_x in {0, 0.135}, base_theta in {-0.15, 0, 0.15}
BASE_LINEAR_DEADZONE = 0.04    # |base_x| < 0.04 → stop
BASE_ANGULAR_DEADZONE = 0.02   # |base_theta| < 0.02 → straight


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model(cfg: dict, checkpoint_path: str, device: torch.device, use_fp16: bool = True):
    """Build CrabSmolVLAWrapper and load trained weights."""
    sys.path.insert(0, str(TRAINING_PKG))
    from training.crab_smolvla_wrapper import build_model

    logger.info("Building model (this will download SmolVLA base if not cached)...")
    model = build_model(cfg)

    # Patch SmolVLA config to match our action/state dims
    max_action_dim = cfg["model"].get("max_action_dim", 32)
    max_state_dim = cfg["model"].get("max_state_dim", 32)
    smolvla_cfg = model.smolvla.config
    smolvla_cfg.output_features["action"].shape = (max_action_dim,)
    smolvla_cfg.input_features["observation.state"].shape = (max_state_dim,)
    logger.info(f"Patched SmolVLA config: action shape=({max_action_dim},), state shape=({max_state_dim},)")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    logger.info("Model ready")
    return model


def obs_to_batch(obs: dict, task: str, image_size: tuple, chunk_size: int,
                 device: torch.device, state_indices: list = None,
                 action_dim: int = 14):
    """Convert raw robot observation to model batch dict."""

    # State — select subset if configured
    if state_indices:
        state = torch.tensor([obs.get(STATE_KEYS[i], 0.0) for i in state_indices],
                             dtype=torch.float32)
    else:
        state = torch.tensor([obs.get(k, 0.0) for k in STATE_KEYS], dtype=torch.float32)

    # Cameras [3, H, W] normalized to [0, 1]
    images = {}
    for cam_key, ds_key in zip(CAMERA_KEYS, CAMERA_DS_KEYS):
        img = obs.get(cam_key)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = resize(img_t, list(image_size), antialias=True)
        images[ds_key] = img_t.unsqueeze(0)

    # Tactile [100] each
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

    # Dummy action
    dummy_action = torch.zeros(1, chunk_size, action_dim, dtype=torch.float32)

    batch = {
        "images": {k: v.to(device) for k, v in images.items()},
        "state": state.unsqueeze(0).to(device),
        "tactile_left": torch.from_numpy(tactile_left).unsqueeze(0).to(device),
        "tactile_right": torch.from_numpy(tactile_right).unsqueeze(0).to(device),
        "action": dummy_action.to(device),
        "task": [task],
    }

    return batch


def build_action_dict(action_np: np.ndarray, obs: dict, action_indices: list = None) -> dict:
    """
    Build action dict from model output.

    If action_indices is set (e.g. [12,13] for driving, [6-11] for right arm):
    - ONLY predicted joints are included in the dict
    - Non-predicted joints are OMITTED (not echo-commanded) to avoid servo oscillation
    - Base velocity defaults to 0 if not predicted
    """
    if action_indices is None:
        # Full 14-DOF model — include all joints
        action_dict = {k: float(action_np[i]) for i, k in enumerate(ACTION_KEYS)}
    else:
        # Partial model — only include predicted joints
        action_dict = {}
        idx_set = set(action_indices)
        model_idx = 0
        for i, k in enumerate(ACTION_KEYS):
            if i in idx_set:
                action_dict[k] = float(action_np[model_idx])
                model_idx += 1
            # Non-predicted joints: OMIT entirely (don't echo obs back to servos)
            # This prevents servo oscillation / gripper shaking

        # Ensure base velocity keys exist (default 0 if not predicted)
        if "base_x.vel" not in action_dict:
            action_dict["base_x.vel"] = 0.0
        if "base_theta.vel" not in action_dict:
            action_dict["base_theta.vel"] = 0.0

    # Deadzone + safety clamp for base velocities
    # Training data is discrete: base_x in {0, 0.135}, base_theta in {-0.15, 0, 0.15}
    # Model outputs continuous values — apply deadzone to snap small values to 0 (clean stop)
    bx = action_dict["base_x.vel"]
    bt = action_dict["base_theta.vel"]
    if abs(bx) < BASE_LINEAR_DEADZONE:
        bx = 0.0
    if abs(bt) < BASE_ANGULAR_DEADZONE:
        bt = 0.0
    action_dict["base_x.vel"] = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, bx))
    action_dict["base_theta.vel"] = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, bt))
    return action_dict


def run_test(cfg, checkpoint_path, device):
    """Test model loading and forward pass with dummy data."""
    model = load_model(cfg, checkpoint_path, device, use_fp16=True)
    action_dim = cfg["model"].get("action_dim", 14)

    logger.info(f"Running test inference with dummy data (action_dim={action_dim})...")
    dummy_batch = {
        "images": {k: torch.randn(1, 3, 256, 256, device=device) for k in CAMERA_DS_KEYS},
        "state": torch.randn(1, action_dim, device=device),
        "tactile_left": torch.randn(1, 100, device=device),
        "tactile_right": torch.randn(1, 100, device=device),
        "action": torch.zeros(1, 50, action_dim, device=device),
        "task": ["Pick and place object"],
    }

    with torch.no_grad(), torch.cuda.amp.autocast():
        t0 = time.perf_counter()
        actions = model.predict_action(dummy_batch)
        dt = time.perf_counter() - t0

    logger.info(f"Output shape: {actions.shape}")
    logger.info(f"Inference time: {dt*1000:.0f}ms")
    logger.info(f"Action range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    logger.info("TEST PASSED")


def run_inference(
    cfg: dict,
    checkpoint_path: str,
    task: str,
    fps: int,
    max_steps: int,
    max_episodes: int,
    device: torch.device,
    use_fp16: bool = True,
    robot_ip: str = CRAB_REMOTE_IP,
):
    from lerobot.robots.crab import CrabClient, CrabClientConfig
    from lerobot.utils.robot_utils import precise_sleep

    model = load_model(cfg, checkpoint_path, device, use_fp16=use_fp16)

    image_size = tuple(cfg["dataset"]["image_size"])
    chunk_size = cfg["model"]["chunk_size"]
    action_dim = cfg["model"].get("action_dim", 14)
    action_indices = cfg["dataset"].get("action_indices", None)
    state_indices = cfg["dataset"].get("state_indices", None)

    if action_indices:
        controlled_keys = [ACTION_KEYS[i] for i in action_indices]
        logger.info(f"Partial control mode: predicting {len(action_indices)} joints: {controlled_keys}")
        logger.info(f"Other joints will hold current position, base vel = 0")
    else:
        logger.info(f"Full control mode: predicting all {action_dim} dims")

    # Warmup
    logger.info("Warming up model...")
    state_dim = len(state_indices) if state_indices else 14
    dummy_batch = {
        "images": {k: torch.randn(1, 3, *image_size, device=device) for k in CAMERA_DS_KEYS},
        "state": torch.randn(1, state_dim, device=device),
        "tactile_left": torch.randn(1, 100, device=device),
        "tactile_right": torch.randn(1, 100, device=device),
        "action": torch.zeros(1, chunk_size, action_dim, device=device),
        "task": [task],
    }

    for i in range(3):
        with torch.no_grad(), torch.cuda.amp.autocast():
            model.predict_action(dummy_batch)
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

    try:
        for episode in range(max_episodes):
            logger.info(f"\n{'='*50}")
            logger.info(f"Episode {episode + 1}/{max_episodes}")
            logger.info(f"{'='*50}")

            steps_this_episode = 0
            model.smolvla.reset()

            while steps_this_episode < max_steps:
                step_start = time.perf_counter()

                # Get fresh observation
                obs = robot.get_observation()

                # Build batch
                batch = obs_to_batch(obs, task, image_size, chunk_size, device,
                                     state_indices=state_indices, action_dim=action_dim)

                # Predict
                t0 = time.perf_counter()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    action = model.predict_action(batch)
                inf_time = (time.perf_counter() - t0) * 1000
                inference_times.append(inf_time)

                action_np = action[0].float().cpu().numpy()  # [action_dim]

                # Build full 14-dim action dict (handles partial control)
                action_dict = build_action_dict(action_np, obs, action_indices)

                robot.send_action(action_dict)

                steps_this_episode += 1
                total_steps += 1

                # Maintain FPS
                elapsed = time.perf_counter() - step_start
                if elapsed < target_dt:
                    precise_sleep(target_dt - elapsed)

                if steps_this_episode % 30 == 0:
                    # Show raw model output vs deadzoned output
                    if action_indices and (12 in action_indices or 13 in action_indices):
                        raw_bx = float(action_np[action_indices.index(12)]) if 12 in action_indices else 0
                        raw_bt = float(action_np[action_indices.index(13)]) if 13 in action_indices else 0
                        logger.info(
                            f"  Step {steps_this_episode}/{max_steps} | "
                            f"inference={inf_time:.0f}ms | "
                            f"raw=({raw_bx:.4f},{raw_bt:.4f}) → "
                            f"out=({action_dict['base_x.vel']:.3f},{action_dict['base_theta.vel']:.3f})"
                        )
                    else:
                        logger.info(
                            f"  Step {steps_this_episode}/{max_steps} | "
                            f"inference={inf_time:.0f}ms | "
                            f"base_vel=({action_dict['base_x.vel']:.3f}, {action_dict['base_theta.vel']:.3f})"
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
            logger.info(f"Inference: {avg:.0f} +/- {std:.0f} ms ({len(inference_times)} chunks)")
        logger.info(f"Total steps executed: {total_steps}")


def main():
    parser = argparse.ArgumentParser(description="CrabSmolVLA Inference")
    parser.add_argument("--model", "-m", type=str, default=str(DEFAULT_MODEL),
                        help="Path to model.pt checkpoint")
    parser.add_argument("--config", "-c", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to YAML config")
    parser.add_argument("--task", "-t", type=str, default="Pick and place object",
                        help="Task description")
    parser.add_argument("--fps", type=int, default=15, help="Control frequency")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument("--robot-ip", type=str, default=CRAB_REMOTE_IP, help="Robot IP")
    parser.add_argument("--forward-deadzone", type=float, default=None,
                        help="Forward velocity deadzone (default: 0.04). |base_x| below this → 0.")
    parser.add_argument("--turn-deadzone", type=float, default=None,
                        help="Turn velocity deadzone (default: 0.02). |base_theta| below this → 0.")
    parser.add_argument("--test", action="store_true", help="Test model loading only (no robot)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = load_config(args.config)

    # Override deadzones if specified
    if args.forward_deadzone is not None or args.turn_deadzone is not None:
        global BASE_LINEAR_DEADZONE, BASE_ANGULAR_DEADZONE
        if args.forward_deadzone is not None:
            BASE_LINEAR_DEADZONE = args.forward_deadzone
        if args.turn_deadzone is not None:
            BASE_ANGULAR_DEADZONE = args.turn_deadzone
        logger.info(f"Deadzones: forward={BASE_LINEAR_DEADZONE}, turn={BASE_ANGULAR_DEADZONE}")

    if args.test:
        run_test(cfg, args.model, device)
    else:
        run_inference(
            cfg=cfg,
            checkpoint_path=args.model,
            task=args.task,
            fps=args.fps,
            max_steps=args.steps,
            max_episodes=args.episodes,
            device=device,
            use_fp16=not args.fp32,
            robot_ip=args.robot_ip,
        )


if __name__ == "__main__":
    main()
