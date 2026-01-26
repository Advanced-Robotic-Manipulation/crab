#!/bin/bash

# This script starts the MobileBase robot host on the onboard computer.
#
# IMPORTANT: Before running, ensure you have set the correct value for:
# - ROBOT_SERIAL_PORT
#
# You might need to give access to the USB serial port:
# sudo chmod 666 /dev/ttyCH340

# --- Configuration ---
# Serial port for the motor controller.
ROBOT_SERIAL_PORT="/dev/ttyCH340" # e.g., /dev/ttyCH340 or /dev/ttyUSB0

# --- Start the MobileBase Host ---
echo "Starting MobileBase robot host..."
python -m lerobot.robots.mobile_base.mobile_base_host \
    --robot.port="$ROBOT_SERIAL_PORT"
