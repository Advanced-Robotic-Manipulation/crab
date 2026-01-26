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

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import serial

logger = logging.getLogger(__name__)


@dataclass
class HapticSensorConfig:
    port: str = "/dev/ttyACM2"
    baudrate: int = 115200
    timeout: float = 1.0
    use_for_inference: bool = True
    sensor1_max_force: float = 10000.0  # A0 large round, max 10kg
    sensor2_max_force: float = 1000.0   # A1 small square, max 1kg


@dataclass
class HapticReading:
    force: float = 0.0
    force_smoothed: float = 0.0
    resistance: float = 0.0
    has_pressure: bool = False
    max_force_reached: bool = False
    sensor_type: str = "unknown"


class HapticSensor:
    def __init__(self, config: HapticSensorConfig):
        self.config = config
        self._serial: serial.Serial | None = None
        self._is_connected = False
        self._sensor1 = HapticReading()
        self._sensor2 = HapticReading()
        self._read_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        if self._is_connected:
            return
        try:
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout
            )
            time.sleep(2.0)
            self._serial.flushInput()
            self._serial.flushOutput()
            self._is_connected = True
            self._stop_event.clear()
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()
            logger.info(f"Haptic sensor connected on {self.config.port}")
        except serial.SerialException as e:
            logger.error(f"Failed to connect haptic sensor: {e}")
            raise

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._stop_event.set()
        if self._read_thread:
            self._read_thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None
        self._is_connected = False
        logger.info("Haptic sensor disconnected")

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._serial and self._serial.in_waiting:
                    line = self._serial.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('{'):
                        self._parse_json(line)
            except Exception:
                time.sleep(0.01)

    def _parse_json(self, line: str) -> None:
        try:
            data = json.loads(line)
            with self._lock:
                if 'sensor1' in data:
                    s1 = data['sensor1']
                    self._sensor1 = HapticReading(
                        force=s1.get('force', 0.0),
                        force_smoothed=s1.get('forceSmoothed', s1.get('force', 0.0)),
                        resistance=s1.get('resistance', 0.0),
                        has_pressure=s1.get('hasPressure', False),
                        max_force_reached=s1.get('maxForceReached', False),
                        sensor_type=s1.get('type', 'LARGE_ROUND')
                    )
                if 'sensor2' in data:
                    s2 = data['sensor2']
                    self._sensor2 = HapticReading(
                        force=s2.get('force', 0.0),
                        force_smoothed=s2.get('forceSmoothed', s2.get('force', 0.0)),
                        resistance=s2.get('resistance', 0.0),
                        has_pressure=s2.get('hasPressure', False),
                        max_force_reached=s2.get('maxForceReached', False),
                        sensor_type=s2.get('type', 'SMALL_SQUARE')
                    )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def get_readings(self) -> dict[str, HapticReading]:
        with self._lock:
            return {
                'sensor1': HapticReading(**self._sensor1.__dict__),
                'sensor2': HapticReading(**self._sensor2.__dict__)
            }

    def _normalize(self, value: float, max_val: float) -> float:
        normalized = min(max(value / max_val, 0.0), 1.0)
        return round(normalized, 4)

    def get_observation(self) -> dict[str, float]:
        readings = self.get_readings()
        f1 = self._normalize(readings['sensor1'].force_smoothed, self.config.sensor1_max_force)
        f2 = self._normalize(readings['sensor2'].force_smoothed, self.config.sensor2_max_force)
        return {
            'haptic_sensor1.force': f1,
            'haptic_sensor2.force': f2,
        }

    @staticmethod
    def get_feature_types() -> dict[str, type]:
        return {
            'haptic_sensor1.force': float,
            'haptic_sensor2.force': float,
        }

