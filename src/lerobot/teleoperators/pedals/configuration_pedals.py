# Copyright 2025 Crab Robot Team. All rights reserved.

from dataclasses import dataclass


@dataclass
class PedalsTeleopConfig:
    """Configuration for the pedals teleoperator."""
    
    # Serial port for Arduino
    port: str = "/dev/arduino"
    
    # Baud rate for serial communication
    baud_rate: int = 115200
    
    # Connection timeout in seconds
    timeout: float = 1.0
