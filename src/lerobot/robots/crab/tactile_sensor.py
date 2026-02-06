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

"""
Dual Tactile Sensor - 2x 10x10 Force Sensor Matrices

Hardware: ESP32 + CP2102N USB-UART + 2x 10x10 tactile sensors
Protocol: 921600 baud, send 0xFE to request, receive packed 12-bit data

Sensor mapping (discovered through testing):
  - Finger 0: Not connected (always 0)
  - Finger 1: RIGHT sensor (after calibration)
  - Finger 2: LEFT sensor (after calibration)

Orientation corrections applied:
  - LEFT:  [::-1, ::-1] (flip both X and Y)
  - RIGHT: [:, ::-1]    (flip Y only)
"""

import logging
import serial
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TactileSensorConfig:
    """Configuration for the dual tactile sensor."""
    port: str = "/dev/tactile_sensor"
    baudrate: int = 921600
    timeout: float = 1.0
    # Sensor finger indices (protocol index -> physical sensor)
    left_finger_idx: int = 2
    right_finger_idx: int = 1
    # Normalization: max sum value for full pressure
    max_force_sum: float = 150000.0


class TactileSensor:
    """
    Reader for dual 10x10 tactile force sensor matrices.
    
    Reads asynchronously in a background thread to avoid blocking the main loop.
    Returns corrected orientation matrices for left and right sensors.
    """
    
    REQUEST_CMD = 0xFE
    MATRIX_SIZE = 10
    
    def __init__(self, config: TactileSensorConfig):
        self.config = config
        self._serial: Optional[serial.Serial] = None
        self._is_connected = False
        
        # Latest sensor data (protected by lock)
        self._lock = threading.Lock()
        self._left_matrix: np.ndarray = np.zeros((10, 10), dtype=np.uint16)
        self._right_matrix: np.ndarray = np.zeros((10, 10), dtype=np.uint16)
        self._last_read_time: float = 0.0
        
        # Background thread
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def connect(self) -> None:
        """Open serial connection and start background reader."""
        if self._is_connected:
            return
        
        try:
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout
            )
            time.sleep(0.1)
            self._serial.reset_input_buffer()
            
            # Initial sync request
            self._serial.write(bytes([self.REQUEST_CMD]))
            time.sleep(0.3)
            self._serial.reset_input_buffer()
            
            self._is_connected = True
            
            # Start background reader
            self._stop_event.clear()
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()
            
            logger.info(f"Tactile sensor connected on {self.config.port}")
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect tactile sensor: {e}")
            raise
    
    def disconnect(self) -> None:
        """Stop background reader and close serial connection."""
        if not self._is_connected:
            return
        
        self._stop_event.set()
        if self._read_thread:
            self._read_thread.join(timeout=2.0)
        
        if self._serial:
            self._serial.close()
            self._serial = None
        
        self._is_connected = False
        logger.info("Tactile sensor disconnected")
    
    def _read_raw(self) -> Optional[np.ndarray]:
        """Read raw 3x10x10 data from sensor (blocking)."""
        if not self._serial or not self._serial.is_open:
            return None
        
        try:
            self._serial.write(bytes([self.REQUEST_CMD]))
            time.sleep(0.008)  # ~8ms for data to arrive
            
            all_pressure = np.zeros((3, 10, 10), dtype=np.uint16)
            packing = 0
            rcv = 0
            
            for finger in range(3):
                for x in range(10):
                    for y in range(10):
                        if packing == 0:
                            b1 = self._serial.read(1)
                            b2 = self._serial.read(1)
                            if not b1 or not b2:
                                return None
                            b1 = struct.unpack("B", b1)[0]
                            b2 = struct.unpack("B", b2)[0]
                            all_pressure[finger][x][y] = (b1 << 4) | (b2 >> 4)
                            rcv = b2
                        else:
                            b3 = self._serial.read(1)
                            if not b3:
                                return None
                            b3 = struct.unpack("B", b3)[0]
                            all_pressure[finger][x][y] = ((rcv & 0x0F) << 8) | b3
                        packing = (packing + 1) % 2
                
                # Skip thermal data (2 values per finger)
                for _ in range(2):
                    if packing == 0:
                        self._serial.read(2)
                    else:
                        self._serial.read(1)
                    packing = (packing + 1) % 2
            
            self._serial.reset_input_buffer()
            return all_pressure
            
        except Exception as e:
            logger.debug(f"Tactile read error: {e}")
            return None
    
    def _read_loop(self) -> None:
        """Background thread that continuously reads sensor data."""
        while not self._stop_event.is_set():
            try:
                raw = self._read_raw()
                if raw is not None:
                    # Extract and apply orientation corrections
                    # LEFT sensor: flip both X and Y
                    left = raw[self.config.left_finger_idx][::-1, ::-1]
                    # RIGHT sensor: flip Y only
                    right = raw[self.config.right_finger_idx][:, ::-1]
                    
                    with self._lock:
                        self._left_matrix = left
                        self._right_matrix = right
                        self._last_read_time = time.time()
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.debug(f"Tactile read loop error: {e}")
                time.sleep(0.05)
    
    def get_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the latest sensor matrices.
        
        Returns:
            Tuple of (left_matrix, right_matrix), each shape (10, 10), dtype uint16
        """
        with self._lock:
            return self._left_matrix.copy(), self._right_matrix.copy()
    
    def get_observation(self) -> dict:
        """
        Get observation dict for robot integration.
        
        Returns dict with:
          - tactile_left: (10, 10) uint16 array
          - tactile_right: (10, 10) uint16 array  
          - tactile_left.sum: float (normalized 0-1)
          - tactile_right.sum: float (normalized 0-1)
        """
        left, right = self.get_matrices()
        
        # Normalize sums to 0-1 range for scalar features
        left_sum = min(float(np.sum(left)) / self.config.max_force_sum, 1.0)
        right_sum = min(float(np.sum(right)) / self.config.max_force_sum, 1.0)
        
        return {
            "tactile_left": left,
            "tactile_right": right,
            "tactile_left.sum": left_sum,
            "tactile_right.sum": right_sum,
        }
    
    @staticmethod
    def get_feature_types() -> dict:
        """
        Return feature type definitions for lerobot observation schema.
        
        Matrices are stored as (10, 10) uint16 arrays.
        Sums are normalized floats for quick scalar access.
        """
        return {
            "tactile_left": (10, 10),      # 10x10 matrix
            "tactile_right": (10, 10),     # 10x10 matrix
            "tactile_left.sum": float,     # Normalized sum
            "tactile_right.sum": float,    # Normalized sum
        }
