# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import logging
import os
import signal
import sys
from dataclasses import dataclass

# Ensure pygame can initialize joystick without a display (important for headless/SSH sessions)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import draccus

from lerobot.robots.crab import CrabClient, CrabClientConfig
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TeleopConfig:
    # Robot client configuration
    robot: CrabClientConfig = CrabClientConfig(
        remote_ip="192.168.50.239",
    )

    # Leader arm configurations (using persistent symlinks)
    left_leader: SO101LeaderConfig = SO101LeaderConfig(
        port="/dev/manipulator_left",
        id="left_leader_arm"
    )
    right_leader: SO101LeaderConfig = SO101LeaderConfig(
        port="/dev/manipulator_right",
        id="right_leader_arm"
    )

    # Gamepad configuration
    gamepad: GamepadTeleopConfig = GamepadTeleopConfig(use_gripper=False)

    # Control settings
    fps: int = 30

    # Base speed settings - NORMAL MODE (slower turning for precision)
    max_linear_velocity: float = 0.27      # Forward/backward speed
    max_angular_velocity: float = 0.6     # Turning speed (reduced from 1.0 for precision)

    # FAST MODE multipliers (activated by holding LB)
    fast_mode_linear_multiplier: float = 1.0    # 2x forward speed in fast mode
    fast_mode_angular_multiplier: float = 1.5   # 1.5x turning speed in fast mode

    # Connection settings
    connect_timeout_s: int = 10  # Increased from default 5s
    max_connection_retries: int = 3
    retry_delay_s: float = 2.0


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True


def connect_with_retry(device, name: str, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
    """Attempt to connect a device with retries."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to {name} (attempt {attempt + 1}/{max_retries})...")
            device.connect()
            if device.is_connected:
                logger.info(f"Successfully connected to {name}")
                return True
            else:
                logger.warning(f"{name} connect() returned but is_connected is False")
        except Exception as e:
            logger.warning(f"Failed to connect to {name}: {e}")

        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    return False


def check_gamepad_connected(gamepad: GamepadTeleop) -> bool:
    """Properly check if gamepad is connected and has a valid joystick."""
    if not gamepad.is_connected:
        return False

    # Check if the internal gamepad controller has a valid joystick
    if gamepad.gamepad is None:
        return False

    # For pygame-based controller, check if joystick is initialized
    if hasattr(gamepad.gamepad, 'joystick'):
        if gamepad.gamepad.joystick is None:
            return False

    # Check if the controller is still running
    if hasattr(gamepad.gamepad, 'running'):
        if not gamepad.gamepad.running:
            return False

    return True


def is_fast_mode_pressed(gamepad: GamepadTeleop) -> bool:
    """
    Check if LB (Left Bumper) is pressed for fast mode.
    LB is typically button 4 on Xbox-style controllers.
    """
    if gamepad.gamepad is None:
        return False

    try:
        # For pygame-based controller
        if hasattr(gamepad.gamepad, 'joystick') and gamepad.gamepad.joystick is not None:
            import pygame
            pygame.event.pump()  # Process events to get fresh button state
            return gamepad.gamepad.joystick.get_button(4)  # LB is button 4

        # For HID-based controller (macOS)
        if hasattr(gamepad.gamepad, 'device'):
            # HID button mapping varies, but LB is usually in byte 6
            # This would need testing on the specific controller
            return False

    except Exception as e:
        logger.debug(f"Error checking fast mode button: {e}")

    return False


def disconnect_all(robot, left_leader, right_leader, gamepad):
    """Safely disconnect all devices."""
    devices = [
        (robot, "robot"),
        (left_leader, "left_leader"),
        (right_leader, "right_leader"),
        (gamepad, "gamepad")
    ]

    for device, name in devices:
        try:
            if device is not None and hasattr(device, 'is_connected'):
                if device.is_connected:
                    logger.info(f"Disconnecting {name}...")
                    device.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting {name}: {e}")


@draccus.wrap()
def main(cfg: TeleopConfig):
    global _shutdown_requested

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    robot = None
    left_leader = None
    right_leader = None
    gamepad = None

    # Track fast mode state for logging
    last_fast_mode = False

    try:
        # --- Initialize robot and teleop devices ---
        logger.info("Initializing devices...")

        # Override connect timeout if specified in config
        if hasattr(cfg, 'connect_timeout_s'):
            cfg.robot.connect_timeout_s = cfg.connect_timeout_s

        robot = CrabClient(cfg.robot)
        left_leader = SO101Leader(cfg.left_leader)
        right_leader = SO101Leader(cfg.right_leader)
        gamepad = GamepadTeleop(cfg.gamepad)

        # --- Connect to devices with retry logic ---
        max_retries = cfg.max_connection_retries
        retry_delay = cfg.retry_delay_s

        # Connect robot first (most critical)
        if not connect_with_retry(robot, "robot", max_retries, retry_delay):
            raise RuntimeError("Failed to connect to robot after multiple attempts. "
                             "Make sure start_host.sh is running on the robot.")

        # Connect leader arms
        if not connect_with_retry(left_leader, "left_leader", max_retries, retry_delay):
            raise RuntimeError("Failed to connect to left leader arm. "
                             "Check if /dev/manipulator_left exists.")

        if not connect_with_retry(right_leader, "right_leader", max_retries, retry_delay):
            raise RuntimeError("Failed to connect to right leader arm. "
                             "Check if /dev/manipulator_right exists.")

        # Connect gamepad (optional - can work without it)
        if not connect_with_retry(gamepad, "gamepad", 2, 1.0):
            logger.warning("Gamepad not connected. Base control will be disabled.")

        # Init rerun viewer
        try:
            init_rerun(session_name="crab_teleop")
        except Exception as e:
            logger.warning(f"Failed to initialize rerun viewer: {e}")

        # Validate connections
        connection_status = {
            "robot": robot.is_connected,
            "left_leader": left_leader.is_connected,
            "right_leader": right_leader.is_connected,
            "gamepad": check_gamepad_connected(gamepad),
        }

        logger.info(f"Connection status: {connection_status}")

        # Robot and arms are required
        if not all([connection_status["robot"],
                   connection_status["left_leader"],
                   connection_status["right_leader"]]):
            raise RuntimeError(f"Critical devices not connected: {connection_status}")

        # Gamepad is optional - warn but continue
        gamepad_available = connection_status["gamepad"]
        if not gamepad_available:
            logger.warning("Gamepad not available. Teleop will run without base control.")

        # Print speed settings
        logger.info("=" * 50)
        logger.info("SPEED SETTINGS:")
        logger.info(f"  Normal mode - Linear: {cfg.max_linear_velocity:.2f} m/s, Angular: {cfg.max_angular_velocity:.2f} rad/s")
        fast_linear = cfg.max_linear_velocity * cfg.fast_mode_linear_multiplier
        fast_angular = cfg.max_angular_velocity * cfg.fast_mode_angular_multiplier
        logger.info(f"  Fast mode (hold LB) - Linear: {fast_linear:.2f} m/s, Angular: {fast_angular:.2f} rad/s")
        logger.info("=" * 50)

        logger.info("Starting teleop loop... Press Ctrl+C to stop.")

        loop_count = 0
        error_count = 0
        max_consecutive_errors = 10

        while not _shutdown_requested:
            t0 = time.perf_counter()
            loop_count += 1

            try:
                # --- Check fast mode (LB button) ---
                fast_mode = False
                if gamepad_available and check_gamepad_connected(gamepad):
                    fast_mode = is_fast_mode_pressed(gamepad)

                # Log mode changes
                if fast_mode != last_fast_mode:
                    if fast_mode:
                        logger.info("FAST MODE ACTIVATED (LB held)")
                    else:
                        logger.info("Normal mode")
                    last_fast_mode = fast_mode

                # Calculate current speed limits based on mode
                if fast_mode:
                    current_linear_vel_max = cfg.max_linear_velocity * cfg.fast_mode_linear_multiplier
                    current_angular_vel_max = cfg.max_angular_velocity * cfg.fast_mode_angular_multiplier
                else:
                    current_linear_vel_max = cfg.max_linear_velocity
                    current_angular_vel_max = cfg.max_angular_velocity

                # --- Get actions from teleop devices ---

                # 1. Get gamepad action for base (if available)
                if gamepad_available and check_gamepad_connected(gamepad):
                    gamepad_action = gamepad.get_action()
                    gamepad_linear_vel = gamepad_action.get("delta_x", 0.0) * current_linear_vel_max
                    gamepad_angular_vel = -gamepad_action.get("delta_y", 0.0) * current_angular_vel_max
                else:
                    gamepad_linear_vel = 0.0
                    gamepad_angular_vel = 0.0

                # 2. Get leader arm actions
                left_arm_action = left_leader.get_action()
                right_arm_action = right_leader.get_action()

                # Prefix arm actions
                prefixed_left = {f"left_{k}": v for k, v in left_arm_action.items()}
                prefixed_right = {f"right_{k}": v for k, v in right_arm_action.items()}

                # Debug: Log velocities periodically
                if loop_count % 30 == 0:  # Every ~1 second at 30fps
                    mode_str = "FAST" if fast_mode else "NORMAL"
                    logger.info(f"Mode: {mode_str} | Linear: {gamepad_linear_vel:.3f} | Angular: {gamepad_angular_vel:.3f}")

                # 3. Combine base action
                base_action = {
                    "base_x.vel": float(gamepad_linear_vel),
                    "base_theta.vel": float(gamepad_angular_vel),
                }

                # 4. Combine all actions
                action = {**prefixed_left, **prefixed_right, **base_action}

                # --- Send action to robot and visualize ---
                robot.send_action(action)
                observation = robot.get_observation()

                try:
                    log_rerun_data(observation=observation, action=action)
                except Exception as e:
                    # Rerun visualization is non-critical
                    if loop_count % 100 == 0:
                        logger.debug(f"Rerun log error (non-critical): {e}")

                # Reset error count on success
                error_count = 0

            except Exception as e:
                error_count += 1
                logger.warning(f"Error in teleop loop (#{error_count}): {e}")

                if error_count >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({error_count}). Stopping.")
                    break

            precise_sleep(max(0.0, 1.0 / cfg.fps - (time.perf_counter() - t0)))

        logger.info("Teleop loop ended.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Always cleanup, even on error
        logger.info("Cleaning up...")
        disconnect_all(robot, left_leader, right_leader, gamepad)
        logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
