"""
Feetech STS3215 Smart Data Module

This module provides utilities for reading diagnostic/telemetry data from 
Feetech STS3215 servo motors, including:
- Current
- Temperature
- Voltage
- Load
- Velocity
- Status flags
- Moving state

These values are useful for monitoring motor health, detecting collisions,
implementing compliance control, and logging for analysis.
"""

from .smart_data_reader import (
    FeetechSmartDataReader,
    MotorSmartData,
    SmartDataBatch,
    SMART_DATA_REGISTERS,
    STATUS_FLAGS,
    read_motors_smart_data,
)

__all__ = [
    "FeetechSmartDataReader",
    "MotorSmartData",
    "SmartDataBatch",
    "SMART_DATA_REGISTERS",
    "STATUS_FLAGS",
    "read_motors_smart_data",
]
