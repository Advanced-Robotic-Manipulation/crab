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

import logging
import time
import numpy as np
from dataclasses import asdict, fields
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from ..mobile_base.mobile_base import MobileBase
from ..mobile_base.config_mobile_base import MobileBaseConfig
from .config_crab import CrabConfig
from .haptic_sensor import HapticSensor

logger = logging.getLogger(__name__)

# Motor names for telemetry
MOTOR_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


class Crab(Robot):
    """
    The Crab robot is a bimanual mobile manipulator with a 2-wheel differential drive base.
    """

    config_class = CrabConfig
    name = "crab"

    def __init__(self, config: CrabConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO101FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            max_relative_target=config.max_relative_target,
            use_degrees=config.use_degrees,
            cameras={},
        )

        right_arm_config = SO101FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            max_relative_target=config.max_relative_target,
            use_degrees=config.use_degrees,
            cameras={},
        )

        self.left_arm = SO101Follower(left_arm_config)
        self.right_arm = SO101Follower(right_arm_config)
        camera_configs_dict = {
            "main_camera": config.cameras.main_camera,
            "left_arm_camera": config.cameras.left_arm_camera,
            "right_arm_camera": config.cameras.right_arm_camera,
        }
        self.cameras = make_cameras_from_configs(camera_configs_dict)
        self.haptic_enabled = config.haptic_enabled
        self.haptic_sensor: HapticSensor | None = HapticSensor(config.haptic) if self.haptic_enabled else None

        # Construct MobileBaseConfig from settings, preserving nested dataclasses
        mb_settings_dict = {f.name: getattr(config.mobile_base, f.name) for f in fields(config.mobile_base)}
        mb_config = MobileBaseConfig(
            id=f"{config.id}_mobile_base" if config.id else "mobile_base",
            calibration_dir=config.calibration_dir,
            **mb_settings_dict
        )
        self.mobile_base = MobileBase(mb_config)

        # Track which optional components are actually connected
        self._mobile_base_connected = False
        self._cameras_connected = {}  # cam_name -> bool
        self._haptic_connected = False

        # Motor telemetry setting
        self._motor_telemetry_enabled = getattr(config, 'motor_telemetry_enabled', True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Bimanual arms
        motors = {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors}
        motors.update({f"right_{motor}.pos": float for motor in self.right_arm.bus.motors})
        # Mobile base
        motors.update({"base_x.vel": float, "base_theta.vel": float})
        return motors

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "main_camera": (self.config.cameras.main_camera.height, self.config.cameras.main_camera.width, 3),
            "left_arm_camera": (
                self.config.cameras.left_arm_camera.height,
                self.config.cameras.left_arm_camera.width,
                3,
            ),
            "right_arm_camera": (
                self.config.cameras.right_arm_camera.height,
                self.config.cameras.right_arm_camera.width,
                3,
            ),
        }

    @property
    def _haptic_ft(self) -> dict[str, type]:
        return HapticSensor.get_feature_types() if self.haptic_enabled else {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._haptic_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Returns True if the required components (arms) are connected.
        Optional components (base, cameras, haptic) are not required."""
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_fully_connected(self) -> bool:
        """Returns True if ALL components are connected."""
        base = self.left_arm.is_connected and self.right_arm.is_connected
        base = base and self._mobile_base_connected
        base = base and all(self._cameras_connected.values())
        if self.haptic_enabled:
            base = base and self._haptic_connected
        return base

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot hardware. Arms are required, other components are optional."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # === REQUIRED: Arms must connect ===
        logger.info("Connecting left arm...")
        try:
            self.left_arm.connect(calibrate)
            logger.info(f"  Left arm connected: {self.left_arm.is_connected}")
        except Exception as e:
            logger.error(f"  FAILED to connect left arm: {e}")
            raise  # Arms are required, so re-raise

        logger.info("Connecting right arm...")
        try:
            self.right_arm.connect(calibrate)
            logger.info(f"  Right arm connected: {self.right_arm.is_connected}")
        except Exception as e:
            logger.error(f"  FAILED to connect right arm: {e}")
            raise  # Arms are required, so re-raise

        # === OPTIONAL: Mobile base ===
        logger.info("Connecting mobile base...")
        try:
            self.mobile_base.connect(calibrate)
            self._mobile_base_connected = self.mobile_base.is_connected
            logger.info(f"  Mobile base connected: {self._mobile_base_connected}")
        except Exception as e:
            logger.warning(f"  FAILED to connect mobile base: {e}")
            logger.warning("  Continuing without mobile base - base control will be disabled")
            self._mobile_base_connected = False

        # === OPTIONAL: Cameras ===
        for cam_name, cam in self.cameras.items():
            logger.info(f"Connecting {cam_name}...")
            try:
                cam.connect()
                self._cameras_connected[cam_name] = cam.is_connected
                logger.info(f"  {cam_name} connected: {cam.is_connected}")
            except Exception as e:
                logger.warning(f"  FAILED to connect {cam_name}: {e}")
                logger.warning(f"  Continuing without {cam_name}")
                self._cameras_connected[cam_name] = False

        # === OPTIONAL: Haptic sensor ===
        if self.haptic_enabled and self.haptic_sensor:
            logger.info("Connecting haptic sensor...")
            try:
                self.haptic_sensor.connect()
                self._haptic_connected = self.haptic_sensor.is_connected
                logger.info(f"  Haptic sensor connected: {self._haptic_connected}")
            except Exception as e:
                logger.warning(f"  FAILED to connect haptic sensor: {e}")
                logger.warning("  Continuing without haptic sensor")
                self._haptic_connected = False
                self.haptic_enabled = False

        # Summary
        logger.info("=" * 40)
        logger.info("CONNECTION SUMMARY:")
        logger.info(f"  Left arm:     {'OK' if self.left_arm.is_connected else 'FAILED'}")
        logger.info(f"  Right arm:    {'OK' if self.right_arm.is_connected else 'FAILED'}")
        logger.info(f"  Mobile base:  {'OK' if self._mobile_base_connected else 'DISABLED'}")
        for cam_name, connected in self._cameras_connected.items():
            logger.info(f"  {cam_name}: {'OK' if connected else 'DISABLED'}")
        if self.haptic_sensor:
            logger.info(f"  Haptic:       {'OK' if self._haptic_connected else 'DISABLED'}")
        logger.info(f"  Motor telemetry: {'ENABLED' if self._motor_telemetry_enabled else 'DISABLED'}")
        logger.info("=" * 40)

        logger.info(f"{self} connected (is_fully_connected={self.is_fully_connected})")

    @property
    def is_calibrated(self) -> bool:
        calibrated = self.left_arm.is_calibrated and self.right_arm.is_calibrated
        if self._mobile_base_connected:
            calibrated = calibrated and self.mobile_base.is_calibrated
        return calibrated

    def calibrate(self) -> None:
        logger.info("Calibrating left arm...")
        self.left_arm.calibrate()
        logger.info("Calibrating right arm...")
        self.right_arm.calibrate()
        if self._mobile_base_connected:
            logger.info("Calibrating mobile base...")
            self.mobile_base.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()
        if self._mobile_base_connected:
            self.mobile_base.configure()

    def _read_motor_telemetry(self) -> dict[str, float]:
        """
        Read motor telemetry (current, temperature, voltage, load) from both arms.
        Returns a dict with keys like 'left_shoulder_pan.current', 'right_gripper.temp', etc.

        This data is logged to Rerun for visualization during teleop.
        """
        telemetry = {}

        # Read from left arm
        try:
            for motor_id in range(1, 7):
                motor_name = MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")

                # Read Present_Current (address 69, 2 bytes)
                current_raw, result, _ = self.left_arm.bus.packet_handler.read2ByteTxRx(
                    self.left_arm.bus.port_handler, motor_id, 69
                )
                if result == 0:  # COMM_SUCCESS
                    telemetry[f"left_{motor_name}.current"] = float(current_raw * 6.5)  # Convert to mA

                # Read Present_Temperature (address 63, 1 byte)
                temp_raw, result, _ = self.left_arm.bus.packet_handler.read1ByteTxRx(
                    self.left_arm.bus.port_handler, motor_id, 63
                )
                if result == 0:
                    telemetry[f"left_{motor_name}.temp"] = float(temp_raw)

                # Read Present_Voltage (address 62, 1 byte)
                volt_raw, result, _ = self.left_arm.bus.packet_handler.read1ByteTxRx(
                    self.left_arm.bus.port_handler, motor_id, 62
                )
                if result == 0:
                    telemetry[f"left_{motor_name}.voltage"] = float(volt_raw / 10.0)

                # Read Present_Load (address 60, 2 bytes)
                load_raw, result, _ = self.left_arm.bus.packet_handler.read2ByteTxRx(
                    self.left_arm.bus.port_handler, motor_id, 60
                )
                if result == 0:
                    # Sign-magnitude encoding, bit 10 is sign
                    if load_raw > 1023:
                        load = -(load_raw - 1024) / 10.0
                    else:
                        load = load_raw / 10.0
                    telemetry[f"left_{motor_name}.load"] = load
        except Exception as e:
            logger.debug(f"Failed to read left arm telemetry: {e}")

        # Read from right arm
        try:
            for motor_id in range(1, 7):
                motor_name = MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")

                # Read Present_Current
                current_raw, result, _ = self.right_arm.bus.packet_handler.read2ByteTxRx(
                    self.right_arm.bus.port_handler, motor_id, 69
                )
                if result == 0:
                    telemetry[f"right_{motor_name}.current"] = float(current_raw * 6.5)

                # Read Present_Temperature
                temp_raw, result, _ = self.right_arm.bus.packet_handler.read1ByteTxRx(
                    self.right_arm.bus.port_handler, motor_id, 63
                )
                if result == 0:
                    telemetry[f"right_{motor_name}.temp"] = float(temp_raw)

                # Read Present_Voltage
                volt_raw, result, _ = self.right_arm.bus.packet_handler.read1ByteTxRx(
                    self.right_arm.bus.port_handler, motor_id, 62
                )
                if result == 0:
                    telemetry[f"right_{motor_name}.voltage"] = float(volt_raw / 10.0)

                # Read Present_Load
                load_raw, result, _ = self.right_arm.bus.packet_handler.read2ByteTxRx(
                    self.right_arm.bus.port_handler, motor_id, 60
                )
                if result == 0:
                    if load_raw > 1023:
                        load = -(load_raw - 1024) / 10.0
                    else:
                        load = load_raw / 10.0
                    telemetry[f"right_{motor_name}.load"] = load
        except Exception as e:
            logger.debug(f"Failed to read right arm telemetry: {e}")

        return telemetry

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Get arm observations and add prefixes
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # Get base observations (or zeros if not connected)
        if self._mobile_base_connected:
            try:
                base_obs = self.mobile_base.get_observation()
                obs_dict["base_x.vel"] = base_obs.get("linear_velocity", 0.0)
                obs_dict["base_theta.vel"] = base_obs.get("angular_velocity", 0.0)
            except Exception as e:
                logger.warning(f"Failed to get mobile base observation: {e}")
                obs_dict["base_x.vel"] = 0.0
                obs_dict["base_theta.vel"] = 0.0
        else:
            obs_dict["base_x.vel"] = 0.0
            obs_dict["base_theta.vel"] = 0.0

        # Haptic sensor (or zeros if not connected)
        if self.haptic_enabled and self._haptic_connected and self.haptic_sensor:
            try:
                obs_dict.update(self.haptic_sensor.get_observation())
            except Exception as e:
                logger.warning(f"Failed to get haptic observation: {e}")
                for key in self._haptic_ft:
                    obs_dict[key] = 0.0
        else:
            for key in self._haptic_ft:
                obs_dict[key] = 0.0

        # Motor telemetry (current, temperature, voltage, load)
        if self._motor_telemetry_enabled:
            try:
                telemetry = self._read_motor_telemetry()
                obs_dict.update(telemetry)
            except Exception as e:
                logger.debug(f"Failed to read motor telemetry: {e}")

        # Capture images from cameras (or black frames if not connected)
        for cam_key, cam in self.cameras.items():
            if self._cameras_connected.get(cam_key, False):
                try:
                    start = time.perf_counter()
                    obs_dict[cam_key] = cam.async_read()
                    dt_ms = (time.perf_counter() - start) * 1e3
                    logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
                except Exception as e:
                    logger.warning(f"Failed to read {cam_key}: {e}")
                    # Return black frame
                    shape = self._cameras_ft[cam_key]
                    obs_dict[cam_key] = np.zeros(shape, dtype=np.uint8)
            else:
                # Return black frame for disconnected cameras
                shape = self._cameras_ft[cam_key]
                obs_dict[cam_key] = np.zeros(shape, dtype=np.uint8)

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Separate actions for each component
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        base_action_x = action.get("base_x.vel", 0.0)
        base_action_theta = action.get("base_theta.vel", 0.0)

        # Send actions to arms
        self.left_arm.send_action(left_action)
        self.right_arm.send_action(right_action)

        # Send action to base (if connected)
        if self._mobile_base_connected:
            try:
                base_action = {
                    "linear_velocity": float(base_action_x),
                    "angular_velocity": float(base_action_theta),
                }
                self.mobile_base.send_action(base_action)
            except Exception as e:
                logger.warning(f"Failed to send base action: {e}")

        return action

    def stop_base(self):
        if self._mobile_base_connected:
            logger.info("Stopping base.")
            try:
                self.mobile_base.stop()
            except Exception as e:
                logger.warning(f"Failed to stop base: {e}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stop base first (if connected)
        if self._mobile_base_connected:
            self.stop_base()
            try:
                self.mobile_base.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting mobile base: {e}")
            self._mobile_base_connected = False

        # Disconnect arms
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        # Disconnect cameras
        for cam_name, cam in self.cameras.items():
            if self._cameras_connected.get(cam_name, False):
                try:
                    cam.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting {cam_name}: {e}")
            self._cameras_connected[cam_name] = False

        # Disconnect haptic sensor
        if self.haptic_sensor and self._haptic_connected:
            try:
                self.haptic_sensor.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting haptic sensor: {e}")
            self._haptic_connected = False

        logger.info(f"{self} disconnected.")
