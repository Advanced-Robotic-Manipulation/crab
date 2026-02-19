# Copyright 2025 Crab Robot Team. All rights reserved.

"""
Pedals Teleoperator for Crab Robot base control.

Pedal mapping:
- turn_left (D2): Turn robot left
- turn_right (D3): Turn robot right  
- go_back (D4): Move robot backward
- go_forward (D5): Move robot forward

This implementation uses a background thread for non-blocking serial reads,
similar to how the gamepad controller works.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

import serial

from .configuration_pedals import PedalsTeleopConfig

logger = logging.getLogger(__name__)


class PedalsTeleop:
    """
    Teleoperator class for foot pedals controlling the mobile base.
    Uses async background thread for serial reads (non-blocking).
    """

    config_class = PedalsTeleopConfig
    name = "pedals"

    PEDAL_NAMES = ['turn_left', 'turn_right', 'go_back', 'go_forward']

    def __init__(self, config: PedalsTeleopConfig):
        self.config = config
        self.serial: Optional[serial.Serial] = None
        self._is_connected = False
        
        # Thread-safe state storage
        self._lock = threading.Lock()
        self._state: Dict[str, bool] = {name: False for name in self.PEDAL_NAMES}
        
        # Background reader thread
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self) -> None:
        """Connect to the Arduino pedal controller."""
        self._is_connected = False
        self._stop_event.clear()
        
        try:
            logger.info(f"Connecting to pedals at {self.config.port}...")
            self.serial = serial.Serial(
                self.config.port, 
                self.config.baud_rate, 
                timeout=0.05  # Short timeout for responsive reading
            )
            
            time.sleep(2.0)  # Wait for Arduino reset
            self.serial.reset_input_buffer()
            
            # Read startup message if any
            startup = self.serial.readline().decode(errors='ignore').strip()
            if startup:
                logger.info(f"Pedals Arduino: {startup}")
            
            # Verify connection with ping
            self.serial.write(b'ping\n')
            time.sleep(0.1)
            response = self.serial.readline().decode(errors='ignore').strip()
            
            if response == "pong":
                logger.info("Pedals connected and verified")
            else:
                logger.warning(f"Pedals ping got: '{response}', continuing anyway")
            
            self._is_connected = True
            
            # Start background reader thread
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()
            logger.info("Pedals background reader started")
                
        except serial.SerialException as e:
            logger.error(f"Failed to connect to pedals: {e}")
            self.serial = None
            self._is_connected = False
        except Exception as e:
            logger.error(f"Unexpected error connecting to pedals: {e}")
            self.serial = None
            self._is_connected = False

    def _read_loop(self) -> None:
        """Background thread: continuously read pedal states."""
        logger.debug("Pedals reader thread started")
        
        while not self._stop_event.is_set():
            if self.serial is None:
                time.sleep(0.1)
                continue
                
            try:
                # Request state
                self.serial.write(b'r\n')
                
                # Read response (non-blocking due to short timeout)
                response = self.serial.readline().decode(errors='ignore').strip()
                
                if response and ':' in response:
                    new_state = {name: False for name in self.PEDAL_NAMES}
                    for part in response.split(','):
                        if ':' in part:
                            name, val = part.split(':', 1)
                            name = name.strip()
                            if name in new_state:
                                new_state[name] = val.strip() == '1'
                    
                    # Update state atomically
                    with self._lock:
                        self._state = new_state
                
                # Small sleep to not overwhelm serial
                time.sleep(0.02)  # ~50Hz polling
                
            except serial.SerialException as e:
                logger.warning(f"Pedals serial error: {e}")
                time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Pedals read error: {e}")
                time.sleep(0.05)
        
        logger.debug("Pedals reader thread stopped")

    def get_action(self) -> Dict[str, Any]:
        """Get current action from pedals (non-blocking, returns cached state)."""
        with self._lock:
            state = self._state.copy()
        
        # Convert to delta velocities
        delta_x = 0.0
        if state['go_forward']:
            delta_x = 1.0
        elif state['go_back']:
            delta_x = -1.0
        
        delta_y = 0.0
        if state['turn_left']:
            delta_y = 1.0
        elif state['turn_right']:
            delta_y = -1.0
        
        return {"delta_x": delta_x, "delta_y": delta_y}

    def get_pedal_state(self) -> Dict[str, bool]:
        """Get raw pedal states for logging (non-blocking)."""
        with self._lock:
            return self._state.copy()

    def disconnect(self) -> None:
        """Disconnect from the Arduino."""
        # Stop reader thread
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        
        # Close serial
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception as e:
                logger.warning(f"Error closing pedals serial: {e}")
            self.serial = None
        
        self._is_connected = False
        logger.info("Pedals disconnected")

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.serial is not None
