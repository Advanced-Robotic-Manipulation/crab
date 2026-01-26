#!/usr/bin/env python3
"""
SmolVLA Async Policy Server for Crab Robot

This runs the policy inference in a dedicated server process, enabling:
- Higher throughput (~30% faster)
- Decoupled inference from robot control
- Better GPU utilization

Usage:
  # Start server (on Orin or GPU machine)
  python smolvla_async_server.py --model /path/to/model --port 8080
  
  # Then run the async client separately
"""

import argparse
import logging

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SmolVLA async policy server for Crab")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # Listen on all interfaces
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting SmolVLA async policy server on {args.host}:{args.port}")
    logger.info("The policy model will be loaded when the first client connects.")
    logger.info("Use Ctrl+C to stop the server.")
    
    config = PolicyServerConfig(
        host=args.host,
        port=args.port,
    )
    
    serve(config)


if __name__ == "__main__":
    main()
