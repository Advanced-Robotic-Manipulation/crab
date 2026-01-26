from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig

@dataclass
class FortuneX4MotorConfig:
    """Configuration for a single Fortune X4 motor controller."""
    id: int
    reverse: bool = False

@dataclass
class MotorHardwareConfig:
    """Hardware constants and register addresses for Fortune X4 controllers."""
    gear_ratio: float = 22.0
    ticks_per_rev: float = 3840.0
    setpoint_reg: int = 3
    feedback_reg: int = 69
    angle_reg: int = 67
    error_reg: int = 29
    mode_reg: int = 0
    vmin_reg: int = 9
    ilimit_reg: int = 10
    temp_reg: int = 11
    timeout_reg: int = 18
    pwm_limit_reg: int = 21
    pwm_inc_reg: int = 22
    spd_pid_p_reg: int = 13
    spd_pid_i_reg: int = 12
    pwm_limit: int = 800
    pwm_inc_limit: int = 20
    i_limit: float = 20.0
    v_min: float = 12.0
    speed_pid_p: float = 60.0
    speed_pid_i: float = 2.8
    timeout_ms: int = 2000
    temp_shutdown: int = 75
    mode_speed: int = 2

@dataclass
class KinematicsConfig:
    """Kinematic parameters for the differential drive mobile base."""
    wheel_separation: float = 0.4
    wheel_radius: float = 0.05
    max_wheel_speed: float = 16.0

@dataclass
class MobileBaseSettings:
    """Settings for the MobileBase robot (host-side)."""
    port: str = "/dev/ttyCH340"
    baudrate: int = 115200
    parity: str = 'N'
    stopbits: int = 1
    bytesize: int = 8
    timeout: float = 0.005
    read_every_n_cycles: int = 5  # Read feedback every N cycles (1 = every cycle, higher = faster writes)
    
    left_motors: list[FortuneX4MotorConfig] = field(default_factory=lambda: [FortuneX4MotorConfig(id=1, reverse=True)])
    right_motors: list[FortuneX4MotorConfig] = field(default_factory=lambda: [FortuneX4MotorConfig(id=2, reverse=False)])
    
    motor_hardware: MotorHardwareConfig = field(default_factory=MotorHardwareConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)

@RobotConfig.register_subclass("mobile_base")
@dataclass
class MobileBaseConfig(RobotConfig, MobileBaseSettings):
    """Configuration for the MobileBase robot (host-side)."""
    pass

@dataclass
class MobileBaseHostConfig:
    """Configuration for the server hosting the MobileBase robot."""
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 864000
    watchdog_timeout_ms: int = 2000
    max_loop_freq_hz: int = 50

@dataclass
class MobileBaseClientSettings:
    """Settings for the client to control the MobileBase robot remotely."""
    remote_ip: str = "0.0.0.0"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

@RobotConfig.register_subclass("mobile_base_client")
@dataclass
class MobileBaseClientConfig(RobotConfig, MobileBaseClientSettings):
    """Configuration for the client to control the MobileBase robot remotely."""
    pass