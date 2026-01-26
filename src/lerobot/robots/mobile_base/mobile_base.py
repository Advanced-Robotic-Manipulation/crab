import logging
import time
from functools import cached_property
from threading import Thread, Lock, Event
from typing import Any

import numpy as np

from pymodbus.client import ModbusSerialClient
try:
    from pymodbus.binary import Endian
except ImportError:
    from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_mobile_base import MobileBaseConfig, FortuneX4MotorConfig

logger = logging.getLogger(__name__)


class FortuneX4Motor:
    def __init__(self, client: ModbusSerialClient, motor_config: FortuneX4MotorConfig, hardware_config):
        self.client = client
        self.config = motor_config
        self.hw_config = hardware_config
        self._initialize_controller()

    def _initialize_controller(self):
        try:
            self.client.write_register(self.hw_config.vmin_reg, int(self.hw_config.v_min * 1000), slave=self.config.id)
            self.client.write_register(self.hw_config.ilimit_reg, int(self.hw_config.i_limit * 1000), slave=self.config.id)
            self.client.write_register(self.hw_config.temp_reg, self.hw_config.temp_shutdown, slave=self.config.id)
            self.client.write_register(self.hw_config.timeout_reg, int(self.hw_config.timeout_ms / 40), slave=self.config.id)
            self.client.write_register(self.hw_config.pwm_limit_reg, self.hw_config.pwm_limit, slave=self.config.id)
            self.client.write_register(self.hw_config.pwm_inc_reg, self.hw_config.pwm_inc_limit, slave=self.config.id)
            self.client.write_register(self.hw_config.spd_pid_p_reg, int(self.hw_config.speed_pid_p), slave=self.config.id)
            self.client.write_register(self.hw_config.spd_pid_i_reg, int(self.hw_config.speed_pid_i), slave=self.config.id)
            self.reset_mode()
        except Exception as e:
            logger.error(f"Motor {self.config.id} initialization failed: {e}")
            raise

    def reset_mode(self, stop: bool = True):
        try:
            self.client.write_register(self.hw_config.error_reg, 0, slave=self.config.id)
            self.client.write_register(self.hw_config.mode_reg, self.hw_config.mode_speed, slave=self.config.id)
            if stop:
                self.set_speed(0)
        except Exception as e:
            logger.error(f"Motor {self.config.id} reset error: {e}")

    def set_speed(self, rad_s: float):
        modified_rad_s = rad_s / self.hw_config.gear_ratio
        ticks_per_s = (self.hw_config.ticks_per_rev / (2 * np.pi)) * modified_rad_s
        if self.config.reverse:
            ticks_per_s = -ticks_per_s

        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        builder.add_16bit_int(int(round(ticks_per_s)))
        payload = builder.to_registers()

        try:
            if not self.client.is_socket_open():
                self.client.connect()
            self.client.write_registers(self.hw_config.setpoint_reg, payload, slave=self.config.id)
        except Exception as e:
            logger.error(f"Motor {self.config.id} set_speed failed: {e}")
            try:
                self.client.close()
                self.client.connect()
            except Exception:
                pass

    def read_speed(self) -> float:
        try:
            rr = self.client.read_holding_registers(self.hw_config.feedback_reg, 1, slave=self.config.id)
            if not hasattr(rr, 'registers'):
                return 0.0
            decoder = BinaryPayloadDecoder.fromRegisters(rr.registers, byteorder=Endian.BIG, wordorder=Endian.LITTLE)
            raw = decoder.decode_16bit_int()
            if self.config.reverse:
                raw = -raw
            modified_raw = self.hw_config.gear_ratio * raw
            return (2 * np.pi) / self.hw_config.ticks_per_rev * modified_raw
        except Exception:
            return 0.0


class MobileBase(Robot):
    """
    Mobile base with async Modbus communication.

    All Modbus operations run in a background thread to avoid blocking the main loop.
    get_observation() and send_action() are non-blocking - they read/write shared state.
    """

    config_class = MobileBaseConfig
    name = "mobile_base"

    def __init__(self, config: MobileBaseConfig):
        super().__init__(config)
        self.config = config

        self.client = ModbusSerialClient(
            port=config.port,
            baudrate=config.baudrate,
            parity=config.parity,
            stopbits=config.stopbits,
            bytesize=config.bytesize,
            timeout=config.timeout,
        )

        self._is_connected = False
        self.left_motors: list[FortuneX4Motor] = []
        self.right_motors: list[FortuneX4Motor] = []
        self.last_recovery_time = 0.0
        self.recovery_motor_index = 0

        # Threading for async Modbus communication
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._lock = Lock()

        # Shared state (protected by _lock)
        self._latest_observation: dict[str, float] = {
            "linear_velocity": 0.0,
            "angular_velocity": 0.0
        }
        self._pending_action: dict[str, float] = {
            "linear_velocity": 0.0,
            "angular_velocity": 0.0
        }
        self._has_new_action = False

        # Stats for debugging
        self._loop_count = 0
        self._last_stats_time = 0.0

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        return self.action_features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {"linear_velocity": float, "angular_velocity": float}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True):
        if self._is_connected:
            raise DeviceAlreadyConnectedError("MobileBase is already connected.")
        if not self.client.connect():
            raise DeviceNotConnectedError(f"Failed to connect to serial port {self.config.port}")

        self.left_motors = [FortuneX4Motor(self.client, motor_cfg, self.config.motor_hardware) for motor_cfg in self.config.left_motors]
        self.right_motors = [FortuneX4Motor(self.client, motor_cfg, self.config.motor_hardware) for motor_cfg in self.config.right_motors]

        self._is_connected = True

        # Start background communication thread
        self._start_communication_thread()

        logger.info("MobileBase connected (async mode).")

    def _start_communication_thread(self):
        """Start the background thread for Modbus communication."""
        self._stop_event = Event()
        self._thread = Thread(target=self._communication_loop, name="MobileBase-Modbus", daemon=True)
        self._thread.start()
        logger.info("MobileBase communication thread started.")

    def _stop_communication_thread(self):
        """Stop the background communication thread."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("MobileBase communication thread did not stop cleanly.")

        self._thread = None
        self._stop_event = None

    def _communication_loop(self):
        """
        Background thread that handles all Modbus communication.

        Optimized to:
        - Always write commands immediately for responsive control
        - Read feedback every N cycles (configurable) to reduce latency
        
        With timeout=0.005s and read_every_n_cycles=5:
        - Write-only cycles: ~52ms (19 Hz)
        - Read cycles: ~104ms
        - Average: ~63ms (16 Hz)
        """
        logger.info("MobileBase communication loop started.")
        
        # Get read frequency from config (default: read every 5 cycles)
        read_every_n = getattr(self.config, "read_every_n_cycles", 5)
        cycle_count = 0

        while not self._stop_event.is_set():
            try:
                # === READ: Only every N cycles to reduce latency ===
                if cycle_count % read_every_n == 0:
                    left_speeds = [m.read_speed() for m in self.left_motors]
                    right_speeds = [m.read_speed() for m in self.right_motors]

                    avg_left = sum(left_speeds) / len(left_speeds) if left_speeds else 0.0
                    avg_right = sum(right_speeds) / len(right_speeds) if right_speeds else 0.0

                    linear = (avg_right + avg_left) * self.config.kinematics.wheel_radius / 2.0
                    angular = (avg_right - avg_left) * self.config.kinematics.wheel_radius / self.config.kinematics.wheel_separation

                    with self._lock:
                        self._latest_observation = {
                            "linear_velocity": linear,
                            "angular_velocity": angular
                        }

                # Get pending action (always check)
                with self._lock:
                    if self._has_new_action:
                        action = self._pending_action.copy()
                        self._has_new_action = False
                    else:
                        action = None

                # === WRITE: Always send commands immediately ===
                if action is not None:
                    linear_vel = action.get("linear_velocity", 0.0)
                    angular_vel = action.get("angular_velocity", 0.0)

                    wl_raw, wr_raw = self._inverse_kinematics(linear_vel, angular_vel)
                    wl = np.clip(wl_raw, -self.config.kinematics.max_wheel_speed, self.config.kinematics.max_wheel_speed)
                    wr = np.clip(wr_raw, -self.config.kinematics.max_wheel_speed, self.config.kinematics.max_wheel_speed)

                    for m in self.left_motors:
                        m.set_speed(wl)
                    for m in self.right_motors:
                        m.set_speed(wr)

                # === MAINTENANCE: Periodic recovery ===
                self._periodic_recovery()

                # === STATS ===
                cycle_count += 1
                self._loop_count += 1
                now = time.time()
                if now - self._last_stats_time >= 10.0:
                    elapsed = now - self._last_stats_time if self._last_stats_time > 0 else 10.0
                    hz = self._loop_count / elapsed
                    logger.debug(f"MobileBase async loop: {hz:.1f} Hz (read every {read_every_n})")
                    self._loop_count = 0
                    self._last_stats_time = now

            except Exception as e:
                logger.warning(f"MobileBase communication error: {e}")
                time.sleep(0.1)  # Back off on error

        logger.info("MobileBase communication loop stopped.")


    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Returns the latest cached observation (NON-BLOCKING).

        The actual Modbus reads happen in the background thread.
        This just returns the most recent values.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("MobileBase is not connected.")

        with self._lock:
            return self._latest_observation.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Queues an action to be sent (NON-BLOCKING).

        The actual Modbus writes happen in the background thread.
        This just updates the pending action.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("MobileBase is not connected.")

        with self._lock:
            self._pending_action = {
                "linear_velocity": float(action.get("linear_velocity", 0.0)),
                "angular_velocity": float(action.get("angular_velocity", 0.0))
            }
            self._has_new_action = True

        return action

    def _inverse_kinematics(self, linear_vel: float, angular_vel: float) -> tuple[float, float]:
        # Invert angular when going backward (car-like steering)
        if linear_vel < 0:
            angular_vel = -angular_vel
        vel_diff = (angular_vel * self.config.kinematics.wheel_separation) / 2.0
        left_vel = (linear_vel - vel_diff) / self.config.kinematics.wheel_radius
        right_vel = (linear_vel + vel_diff) / self.config.kinematics.wheel_radius
        return left_vel, right_vel

    def _periodic_recovery(self):
        current_time = time.time()
        if current_time - self.last_recovery_time >= 5.0:
            all_motors = self.left_motors + self.right_motors
            if all_motors:
                motor_idx = self.recovery_motor_index % len(all_motors)
                all_motors[motor_idx].reset_mode(stop=False)
                self.recovery_motor_index += 1
            self.last_recovery_time = current_time

    def stop(self):
        """Send stop command (queued, non-blocking)."""
        self.send_action({"linear_velocity": 0.0, "angular_velocity": 0.0})
        # Give the thread a moment to process the stop command
        time.sleep(0.05)

    def disconnect(self):
        if not self.is_connected:
            return

        # Send stop command first
        try:
            self.stop()
        except Exception as e:
            logger.warning(f"Error stopping mobile base: {e}")

        # Stop the communication thread
        self._stop_communication_thread()

        # Close Modbus connection
        self.client.close()
        self._is_connected = False
        logger.info("MobileBase disconnected.")
