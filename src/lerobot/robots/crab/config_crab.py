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

from dataclasses import dataclass, field, fields

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig
from .haptic_sensor import HapticSensorConfig
from ..mobile_base.config_mobile_base import MobileBaseSettings, MobileBaseClientSettings


@dataclass
class CrabCamerasConfig:
    main_camera: OpenCVCameraConfig = field(
        default_factory=lambda: OpenCVCameraConfig(index_or_path=-1, width=640, height=480, fps=30)
    )
    left_arm_camera: OpenCVCameraConfig = field(
        default_factory=lambda: OpenCVCameraConfig(index_or_path=-1, width=640, height=480, fps=30)
    )
    right_arm_camera: OpenCVCameraConfig = field(
        default_factory=lambda: OpenCVCameraConfig(index_or_path=-1, width=640, height=480, fps=30)
    )

    def items(self):
        for f in fields(self):
            yield f.name, getattr(self, f.name)


@RobotConfig.register_subclass("crab")
@dataclass
class CrabConfig(RobotConfig):
    """Configuration for the Crab robot, which runs on the host machine."""
    left_arm_port: str
    right_arm_port: str

    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False
    cameras: CrabCamerasConfig = field(default_factory=CrabCamerasConfig)
    haptic_enabled: bool = False
    haptic: HapticSensorConfig = field(default_factory=HapticSensorConfig)
    mobile_base: MobileBaseSettings = field(default_factory=MobileBaseSettings)

    # Motor telemetry settings
    motor_telemetry_enabled: bool = True  # Enable motor smart data (current, temp, voltage, load)


@dataclass
class CrabHostConfig:
    """Configuration for the server hosting the Crab robot."""
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 86400
    watchdog_timeout_ms: int = 300000
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("crab_client")
@dataclass
class CrabClientConfig(RobotConfig):
    """Configuration for the client to control the Crab robot remotely."""
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "quit": "q",
        }
    )

    cameras: CrabCamerasConfig = field(default_factory=CrabCamerasConfig)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
    mobile_base: MobileBaseClientSettings = field(default_factory=lambda: MobileBaseClientSettings(remote_ip="0.0.0.0"))
