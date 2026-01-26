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

import base64
import json
import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_crab import CrabClientConfig


class CrabClient(Robot):
    config_class = CrabClientConfig
    name = "crab_client"

    def __init__(self, config: CrabClientConfig):
        import zmq

        self._zmq = zmq
        super().__init__(config)
        self.config = config

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations
        self.teleop_keys = config.teleop_keys
        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames = {}
        self.last_remote_state = {}

        self._is_connected = False
        self.logs = {}

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        # Based on SO101Follower motors
        arm_joints = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        motors = {}
        for arm_prefix in ["left", "right"]:
            for joint in arm_joints:
                motors[f"{arm_prefix}_{joint}.pos"] = float
        # Mobile base
        motors.update({"base_x.vel": float, "base_theta.vel": float})
        return motors

    @cached_property
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

    @cached_property
    def _haptic_ft(self) -> dict[str, type]:
        return {
            'haptic_sensor1.force': float,
            'haptic_sensor2.force': float,
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._haptic_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError("CrabClient is already connected.")

        zmq = self._zmq
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_cmd_socket.connect(f"tcp://{self.remote_ip}:{self.port_zmq_cmd}")
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_observation_socket.connect(f"tcp://{self.remote_ip}:{self.port_zmq_observations}")
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks:
            raise DeviceNotConnectedError("Timeout waiting for Crab Host to connect.")

        self._is_connected = True
        logging.info(f"Connected to Crab Host at {self.remote_ip}")

    @property
    def is_calibrated(self) -> bool:
        # Calibration is handled host-side
        return True

    def calibrate(self) -> None:
        # Calibration is handled host-side
        pass

    def configure(self) -> None:
        # Configuration is handled host-side
        pass

    def _poll_and_get_latest_message(self) -> str | None:
        # ... (implementation identical to lekiwi_client)
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

        if self.zmq_observation_socket not in socks:
            return None

        last_msg = None
        while True:
            try:
                msg = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break
        return last_msg

    def _parse_and_decode_observation(self, obs_string: str) -> dict[str, Any] | None:
        try:
            observation = json.loads(obs_string)
            obs_dict = {k: v for k, v in observation.items() if k not in self._cameras_ft}

            for cam_name, image_b64 in observation.items():
                if cam_name not in self._cameras_ft:
                    continue
                if not image_b64:
                    obs_dict[cam_name] = None
                    continue
                jpg_data = base64.b64decode(image_b64)
                np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                obs_dict[cam_name] = frame
            return obs_dict
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logging.error(f"Error processing observation: {e}")
            return None

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("CrabClient is not connected.")

        latest_message_str = self._poll_and_get_latest_message()

        if latest_message_str is not None:
            new_obs = self._parse_and_decode_observation(latest_message_str)
            if new_obs is not None:
                self.last_remote_state = new_obs

        # Fill missing camera frames with black images if needed
        for cam_name, cam_shape in self._cameras_ft.items():
            if self.last_remote_state.get(cam_name) is None:
                self.last_remote_state[cam_name] = np.zeros(cam_shape, dtype=np.uint8)

        for haptic_key in self._haptic_ft:
            if haptic_key not in self.last_remote_state:
                self.last_remote_state[haptic_key] = 0.0

        return self.last_remote_state.copy()

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray) -> dict[str, float]:
        # Simple differential drive control
        x_cmd = 0.0  # m/s forward/backward
        theta_cmd = 0.0  # rad/s rotation

        # This can be adjusted for finer control
        forward_speed = 0.10
        rotation_speed = 45.0  # degrees/s

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd = forward_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd = -forward_speed
        if self.teleop_keys["left"] in pressed_keys:
            theta_cmd = rotation_speed
        if self.teleop_keys["right"] in pressed_keys:
            theta_cmd = -rotation_speed

        return {"base_x.vel": x_cmd, "base_theta.vel": theta_cmd}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("CrabClient is not connected.")

        self.zmq_cmd_socket.send_string(json.dumps(action))
        return action

    def disconnect(self):
        if not self._is_connected:
            return
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
        logging.info("Disconnected from Crab Host.")
