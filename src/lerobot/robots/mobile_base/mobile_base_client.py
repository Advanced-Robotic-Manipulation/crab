import base64
import json
import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.robot import Robot
from .config_mobile_base import MobileBaseClientConfig


class MobileBaseClient(Robot):
    config_class = MobileBaseClientConfig
    name = "mobile_base_client"

    def __init__(self, config: MobileBaseClientConfig):
        import zmq
        self._zmq = zmq
        super().__init__(config)
        self.config = config

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations
        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_remote_state = {}
        self._is_connected = False

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        return self.action_features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {"linear_velocity": float, "angular_velocity": float}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError("MobileBaseClient is already connected.")

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
            raise DeviceNotConnectedError("Timeout waiting for MobileBase Host to connect.")

        self._is_connected = True
        logging.info(f"Connected to MobileBase Host at {self.remote_ip}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _poll_and_get_latest_message(self) -> str | None:
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

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("MobileBaseClient is not connected.")
        latest_message_str = self._poll_and_get_latest_message()
        if latest_message_str is not None:
            try:
                self.last_remote_state = json.loads(latest_message_str)
            except json.JSONDecodeError:
                pass
        return self.last_remote_state.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("MobileBaseClient is not connected.")
        self.zmq_cmd_socket.send_string(json.dumps(action))
        return action

    def disconnect(self):
        if not self._is_connected:
            return
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
        logging.info("Disconnected from MobileBase Host.")

    def stop(self):
        self.send_action({"linear_velocity": 0.0, "angular_velocity": 0.0})
