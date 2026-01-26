#!/usr/bin/env python3
"""
SmolVLA Async Client for Crab Robot

This runs the robot control loop and streams observations to the policy server.
Provides ~30% faster task completion compared to synchronous inference.

Usage:
  # First start the server:
  python smolvla_async_server.py --port 8080
  
  # Then run this client:
  python smolvla_async_client.py --model /path/to/model --task "pick up the block" --server localhost:8080
"""

import argparse
import logging
import threading

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.crab import CrabClientConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crab robot settings
CRAB_REMOTE_IP = "192.168.50.239"


def main():
    parser = argparse.ArgumentParser(description="SmolVLA async client for Crab robot")
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
        "--server",
        type=str,
        default="localhost:8080",
        help="Policy server address (host:port)"
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=CRAB_REMOTE_IP,
        help="Crab robot IP address"
    )
    parser.add_argument(
        "--chunk-threshold",
        type=float,
        default=0.5,
        help="Trigger new inference when action queue drops below this fraction"
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=50,
        help="Number of actions per inference chunk"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show action queue visualization after run"
    )
    
    args = parser.parse_args()
    
    # Camera configuration matching Crab robot
    # These must match the cameras expected by the trained model
    camera_cfg = {
        "main_camera": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "left_arm_camera": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        "right_arm_camera": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }
    
    # Robot configuration (client mode - connects to host running on Orin)
    robot_cfg = CrabClientConfig(
        remote_ip=args.robot_ip,
        cameras=camera_cfg,
    )
    
    # Async client configuration
    client_cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address=args.server,
        policy_device="cuda",
        client_device="cpu",
        policy_type="smolvla",
        pretrained_name_or_path=args.model,
        chunk_size_threshold=args.chunk_threshold,
        actions_per_chunk=args.actions_per_chunk,
    )
    
    logger.info(f"Connecting to policy server at {args.server}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    
    # Create and start client
    client = RobotClient(client_cfg)
    
    if client.start():
        # Start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.receive_actions,
            daemon=True
        )
        action_receiver_thread.start()
        
        try:
            logger.info("Starting control loop...")
            client.control_loop(args.task)
        except KeyboardInterrupt:
            logger.info("\nStopping...")
        finally:
            client.stop()
            action_receiver_thread.join(timeout=2.0)
            
            if args.visualize:
                visualize_action_queue_size(client.action_queue_size)
    else:
        logger.error("Failed to start client")


if __name__ == "__main__":
    main()
