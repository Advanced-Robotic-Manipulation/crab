#!/usr/bin/env python3
"""
SmolVLA Inference for Crab Robot

This script provides optimized SmolVLA inference for the Crab bimanual mobile robot.
Supports both synchronous and asynchronous inference modes.

Input: 3 cameras (main, left_arm, right_arm) - 640x480
Output: 14 actions (6 left arm + 6 right arm + 2 wheels)

Usage:
  # Synchronous inference (simpler, ~10 Hz)
  python smolvla_inference.py --model /path/to/model --task "pick up the block"

  # Asynchronous inference (faster, ~30 Hz) - requires policy server running
  python smolvla_inference.py --model /path/to/model --task "pick up the block" --async --server localhost:8080
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.crab import CrabClient, CrabClientConfig
from lerobot.utils.constants import ACTION, OBS_STR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Crab Robot Configuration ====
CRAB_REMOTE_IP = "192.168.50.239"  # Orin IP

# Action dimensions for Crab robot:
# - Left arm: 6 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
# - Right arm: 6 (same as left)
# - Mobile base: 2 (base_x.vel, base_theta.vel)
# Total: 14 action dimensions
CRAB_ACTION_DIM = 14
CRAB_STATE_DIM = 14  # Matching observation state

# Camera configuration
# Expected by SmolVLA: 640x480 RGB images
CAMERA_NAMES = ["main_camera", "left_arm_camera", "right_arm_camera"]


class CrabSmolVLAInference:
    """SmolVLA inference wrapper optimized for Crab robot on Jetson Orin NX."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_fp16: bool = True,
        compile_model: bool = False,  # torch.compile - experimental on Orin
    ):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        
        logger.info(f"Loading SmolVLA model from: {model_path}")
        logger.info(f"Device: {self.device}, FP16: {use_fp16}")
        
        # Load model
        self.model = SmolVLAPolicy.from_pretrained(model_path)
        self.model.to(self.device)
        
        if use_fp16 and device == "cuda":
            self.model = self.model.half()
            logger.info("Converted model to FP16 for faster inference")
        
        # Optional: torch.compile for potential speedup (experimental on Orin)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        self.model.eval()
        
        # Create pre/post processors
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            model_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}}
        )
        
        # Warmup
        self._warmup()

    def _warmup(self, n_warmup: int = 3):
        """Warmup the model for stable inference times."""
        logger.info(f"Warming up model with {n_warmup} forward passes...")
        
        # Create dummy input matching Crab robot observations
        dummy_obs = {
            "observation.state": torch.randn(1, CRAB_STATE_DIM, device=self.device),
        }
        for cam_name in CAMERA_NAMES:
            dummy_obs[f"observation.images.{cam_name}"] = torch.randn(
                1, 3, 480, 640, device=self.device
            )
        
        if self.use_fp16:
            for k, v in dummy_obs.items():
                if v.dtype == torch.float32:
                    dummy_obs[k] = v.half()
        
        with torch.no_grad():
            for i in range(n_warmup):
                start = time.perf_counter()
                try:
                    _ = self.model.select_action(dummy_obs)
                except Exception as e:
                    logger.debug(f"Warmup {i+1} error (expected for first runs): {e}")
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(f"  Warmup {i+1}: {elapsed:.1f}ms")
        
        logger.info("Warmup complete")

    @torch.no_grad()
    def infer(self, observation: dict, task: str, dataset_features: dict, robot_type: str = "crab") -> dict:
        """
        Run inference on a single observation.
        
        Args:
            observation: Raw observation from robot.get_observation()
            task: Natural language task description
            dataset_features: Features dict from hw_to_dataset_features
            robot_type: Robot type identifier
            
        Returns:
            Action dict ready for robot.send_action()
        """
        # Build inference frame
        obs_frame = build_inference_frame(
            observation=observation,
            ds_features=dataset_features,
            device=self.device,
            task=task,
            robot_type=robot_type
        )
        
        # Preprocess
        obs_processed = self.preprocess(obs_frame)
        
        # Convert to FP16 if needed
        if self.use_fp16:
            for k, v in obs_processed.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    obs_processed[k] = v.half()
        
        # Run model
        start = time.perf_counter()
        action = self.model.select_action(obs_processed)
        inference_time = (time.perf_counter() - start) * 1000
        
        # Postprocess
        action = self.postprocess(action)
        action_dict = make_robot_action(action, dataset_features)
        
        return action_dict, inference_time


def run_sync_inference(
    model_path: str,
    task: str,
    max_episodes: int = 1,
    max_steps: int = 500,
    fps: int = 30,
    use_fp16: bool = True,
):
    """Run synchronous SmolVLA inference loop."""
    
    # Initialize inference
    inference = CrabSmolVLAInference(
        model_path=model_path,
        device="cuda",
        use_fp16=use_fp16,
    )
    
    # Connect to robot
    robot_config = CrabClientConfig(remote_ip=CRAB_REMOTE_IP)
    robot = CrabClient(robot_config)
    
    logger.info(f"Connecting to Crab robot at {CRAB_REMOTE_IP}...")
    robot.connect()
    
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to robot")
    
    logger.info("Connected to Crab robot")
    
    # Get dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}
    
    logger.info(f"Action features: {list(action_features.keys())}")
    logger.info(f"Task: {task}")
    
    target_dt = 1.0 / fps
    inference_times = []
    
    try:
        for episode in range(max_episodes):
            logger.info(f"\n=== Episode {episode + 1}/{max_episodes} ===")
            
            for step in range(max_steps):
                step_start = time.perf_counter()
                
                # Get observation
                obs = robot.get_observation()
                
                # Run inference
                action, inf_time = inference.infer(
                    observation=obs,
                    task=task,
                    dataset_features=dataset_features,
                    robot_type="crab"
                )
                inference_times.append(inf_time)
                
                # Send action to robot
                robot.send_action(action)
                
                # Maintain target FPS
                elapsed = time.perf_counter() - step_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                
                actual_dt = time.perf_counter() - step_start
                if step % 30 == 0:
                    logger.info(
                        f"Step {step}: inference={inf_time:.1f}ms, "
                        f"loop={actual_dt*1000:.1f}ms ({1/actual_dt:.1f} Hz)"
                    )
            
            logger.info(f"Episode {episode + 1} complete")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    
    finally:
        robot.disconnect()
        
        if inference_times:
            avg_inf = np.mean(inference_times)
            std_inf = np.std(inference_times)
            logger.info(f"\nInference stats: {avg_inf:.1f} +/- {std_inf:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="SmolVLA inference for Crab robot")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained SmolVLA model (local path or HuggingFace repo)"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Natural language task description"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target control frequency"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16 (slower but potentially more accurate)"
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=CRAB_REMOTE_IP,
        help="Crab robot IP address"
    )
    
    args = parser.parse_args()
    
    # Update robot IP if provided
    global CRAB_REMOTE_IP
    CRAB_REMOTE_IP = args.robot_ip
    
    run_sync_inference(
        model_path=args.model,
        task=args.task,
        max_episodes=args.episodes,
        max_steps=args.steps,
        fps=args.fps,
        use_fp16=not args.fp32,
    )


if __name__ == "__main__":
    main()
