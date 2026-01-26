# !/usr/bin/env python

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

import sys
import logging
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig

logger = logging.getLogger(__name__)


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None
        self._is_connected = False

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        """Connect to the gamepad."""
        self._is_connected = False
        
        try:
            # use HidApi for macos
            if sys.platform == "darwin":
                # NOTE: On macOS, pygame doesn't reliably detect input from some controllers so we fall back to hidapi
                from .gamepad_utils import GamepadControllerHID as Gamepad
            else:
                from .gamepad_utils import GamepadController as Gamepad

            self.gamepad = Gamepad()
            self.gamepad.start()
            
            # Check if gamepad actually found a controller
            if self.gamepad is not None:
                # For pygame-based controller, check if joystick is initialized
                if hasattr(self.gamepad, 'joystick'):
                    if self.gamepad.joystick is not None:
                        self._is_connected = True
                        logger.info(f"Gamepad connected: {self.gamepad.joystick.get_name()}")
                    else:
                        logger.warning("Gamepad controller created but no joystick found")
                        self._is_connected = False
                # For HID-based controller, check if device is connected
                elif hasattr(self.gamepad, 'device'):
                    if self.gamepad.device is not None:
                        self._is_connected = True
                        logger.info("Gamepad connected (HID mode)")
                    else:
                        logger.warning("Gamepad controller created but no HID device found")
                        self._is_connected = False
                # Check running flag as fallback
                elif hasattr(self.gamepad, 'running'):
                    self._is_connected = self.gamepad.running
                    if self._is_connected:
                        logger.info("Gamepad connected")
                    else:
                        logger.warning("Gamepad failed to initialize")
                else:
                    # Assume connected if gamepad object exists
                    self._is_connected = True
                    logger.info("Gamepad connected (assumed)")
            else:
                logger.warning("Failed to create gamepad controller")
                self._is_connected = False
                
        except Exception as e:
            logger.error(f"Failed to connect gamepad: {e}")
            self.gamepad = None
            self._is_connected = False

    def get_action(self) -> dict[str, Any]:
        """Get the current action from the gamepad."""
        if not self._is_connected or self.gamepad is None:
            # Return zero action if not connected
            action_dict = {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
            }
            if self.config.use_gripper:
                action_dict["gripper"] = GripperAction.STAY.value
            return action_dict
        
        try:
            # Update the controller to get fresh inputs
            self.gamepad.update()

            # Get movement deltas from the controller
            delta_x, delta_y, delta_z = self.gamepad.get_deltas()

            # Create action from gamepad input
            gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

            action_dict = {
                "delta_x": gamepad_action[0],
                "delta_y": gamepad_action[1],
                "delta_z": gamepad_action[2],
            }

            # Default gripper action is to stay
            gripper_action = GripperAction.STAY.value
            if self.config.use_gripper:
                gripper_command = self.gamepad.gripper_command()
                gripper_action = gripper_action_map[gripper_command]
                action_dict["gripper"] = gripper_action

            return action_dict
            
        except Exception as e:
            logger.warning(f"Error getting gamepad action: {e}")
            # Return zero action on error
            action_dict = {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
            }
            if self.config.use_gripper:
                action_dict["gripper"] = GripperAction.STAY.value
            return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None or not self._is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        try:
            # Update gamepad state to get fresh inputs
            self.gamepad.update()

            # Check if intervention is active
            is_intervention = self.gamepad.should_intervene()

            # Get episode end status
            episode_end_status = self.gamepad.get_episode_end_status()
            terminate_episode = episode_end_status in [
                TeleopEvents.RERECORD_EPISODE,
                TeleopEvents.FAILURE,
            ]
            success = episode_end_status == TeleopEvents.SUCCESS
            rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

            return {
                TeleopEvents.IS_INTERVENTION: is_intervention,
                TeleopEvents.TERMINATE_EPISODE: terminate_episode,
                TeleopEvents.SUCCESS: success,
                TeleopEvents.RERECORD_EPISODE: rerecord_episode,
            }
            
        except Exception as e:
            logger.warning(f"Error getting teleop events: {e}")
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            try:
                self.gamepad.stop()
            except Exception as e:
                logger.warning(f"Error stopping gamepad: {e}")
            self.gamepad = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if gamepad is connected and has a valid controller."""
        if not self._is_connected or self.gamepad is None:
            return False
        
        # Additional check for pygame-based controller
        if hasattr(self.gamepad, 'joystick'):
            if self.gamepad.joystick is None:
                return False
        
        # Check running flag
        if hasattr(self.gamepad, 'running'):
            if not self.gamepad.running:
                return False
        
        return True

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    @property
    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
