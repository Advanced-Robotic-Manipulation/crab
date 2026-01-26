#!/bin/bash

# This script starts the teleoperation client for the MobileBase robot on the control computer.
#
# IMPORTANT: Before running, ensure you have set the correct value for:
# - ROBOT_REMOTE_IP

# --- Configuration ---
# IP address of the robot's onboard computer where mobile_base_host is running.
ROBOT_REMOTE_IP="192.168.1.102" # e.g., 192.168.1.102

# --- Start the Teleoperation Client ---
echo "Starting MobileBase teleoperation client (connecting to $ROBOT_REMOTE_IP)..."
python teleoperate.py \
    --robot.remote_ip="$ROBOT_REMOTE_IP"