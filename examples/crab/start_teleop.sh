#!/bin/bash

# --- Activate Conda Environment ---
source /home/orin/miniconda3/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then
    echo "Error: Failed to source conda.sh. Make sure conda is installed correctly."
    exit 1
fi

conda activate lerobot_vla
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'lerobot_vla'. Make sure it exists."
    exit 1
fi

# This script starts the teleoperation client for the Crab robot.
#
# Device symlinks are created by udev rules in /etc/udev/rules.d/99-robot-devices.rules
# Run 'check-robot-devices' to verify all devices are correctly mapped.

# --- Configuration ---
# IP address of the robot's onboard computer where crab_host is running.
ROBOT_REMOTE_IP="192.168.50.239"

# The ID of the Crab robot instance you want to connect to.
ROBOT_ID="my_awesome_crab_robot_20251217"

# Serial port for the left SO-101 leader arm (uses persistent symlink)
LEFT_LEADER_PORT="/dev/manipulator_left"
LEFT_LEADER_ID="left_leader_arm"

# Serial port for the right SO-101 leader arm (uses persistent symlink)
RIGHT_LEADER_PORT="/dev/manipulator_right"
RIGHT_LEADER_ID="right_leader_arm"

# Teleop settings
FPS=30
MAX_LINEAR_VELOCITY=0.8
MAX_ANGULAR_VELOCITY=1.0

# Connection settings
CONNECT_TIMEOUT_S=10
MAX_CONNECTION_RETRIES=3
RETRY_DELAY_S=2.0

# ZMQ ports (must match host)
ZMQ_CMD_PORT=5555
ZMQ_OBS_PORT=5556

# --- Helper function to check if host is reachable ---
check_host_reachable() {
    local ip=$1
    local port=$2
    local timeout=${3:-2}
    
    # Try to connect to the ZMQ observation port
    if command -v nc &> /dev/null; then
        nc -z -w $timeout $ip $port 2>/dev/null
        return $?
    elif command -v timeout &> /dev/null; then
        timeout $timeout bash -c "echo > /dev/tcp/$ip/$port" 2>/dev/null
        return $?
    else
        # Fallback: just ping the host
        ping -c 1 -W $timeout $ip &>/dev/null
        return $?
    fi
}

# --- Pre-flight checks ---
echo "=== Crab Teleop Client Pre-flight Checks ==="

# Check leader arm devices
echo "Checking leader arm devices..."
for device in "$LEFT_LEADER_PORT" "$RIGHT_LEADER_PORT"; do
    if [ ! -e "$device" ]; then
        echo "Error: Device $device not found!"
        echo "  Check USB connections and udev rules."
        exit 1
    else
        echo "  Found: $device"
    fi
done

# Check if host is reachable
echo "Checking if robot host is reachable at $ROBOT_REMOTE_IP..."
max_wait=30
wait_interval=2
waited=0

while [ $waited -lt $max_wait ]; do
    if check_host_reachable $ROBOT_REMOTE_IP $ZMQ_OBS_PORT 2; then
        echo "  Robot host is reachable!"
        break
    else
        echo "  Waiting for robot host... ($waited/$max_wait seconds)"
        sleep $wait_interval
        waited=$((waited + wait_interval))
    fi
done

if [ $waited -ge $max_wait ]; then
    echo ""
    echo "Warning: Could not confirm robot host is running."
    echo "  Make sure start_host.sh is running on the robot at $ROBOT_REMOTE_IP"
    echo "  Continuing anyway (teleop will retry connection)..."
    echo ""
fi

# --- Check for gamepad ---
echo "Checking for gamepad..."
if command -v python &> /dev/null; then
    gamepad_check=$(python -c "
import pygame
pygame.init()
pygame.joystick.init()
count = pygame.joystick.get_count()
if count > 0:
    js = pygame.joystick.Joystick(0)
    print(f'Found gamepad: {js.get_name()}')
else:
    print('No gamepad detected')
pygame.quit()
" 2>/dev/null)
    if [ -n "$gamepad_check" ]; then
        echo "  $gamepad_check"
    fi
fi

# --- Start the Teleoperation Client ---
echo ""
echo "=== Starting Crab Teleoperation Client ==="
echo "  Robot IP:     $ROBOT_REMOTE_IP"
echo "  Robot ID:     $ROBOT_ID"
echo "  Left leader:  $LEFT_LEADER_PORT (ID: $LEFT_LEADER_ID)"
echo "  Right leader: $RIGHT_LEADER_PORT (ID: $RIGHT_LEADER_ID)"
echo "  FPS:          $FPS"
echo ""

cd ~/anywhereVLA/workspaces/miilv_ws/lerobot

python examples/crab/teleoperate.py \
    --robot.remote_ip="$ROBOT_REMOTE_IP" \
    --robot.id="$ROBOT_ID" \
    --robot.port_zmq_cmd="$ZMQ_CMD_PORT" \
    --robot.port_zmq_observations="$ZMQ_OBS_PORT" \
    --robot.connect_timeout_s="$CONNECT_TIMEOUT_S" \
    --left_leader.port="$LEFT_LEADER_PORT" \
    --left_leader.id="$LEFT_LEADER_ID" \
    --right_leader.port="$RIGHT_LEADER_PORT" \
    --right_leader.id="$RIGHT_LEADER_ID" \
    --fps="$FPS" \
    --max_linear_velocity="$MAX_LINEAR_VELOCITY" \
    --max_angular_velocity="$MAX_ANGULAR_VELOCITY" \
    --connect_timeout_s="$CONNECT_TIMEOUT_S" \
    --max_connection_retries="$MAX_CONNECTION_RETRIES" \
    --retry_delay_s="$RETRY_DELAY_S"

exit_code=$?
echo ""
echo "Teleop client exited with code: $exit_code"

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Make sure start_host.sh is running on the robot"
    echo "  2. Check network connectivity: ping $ROBOT_REMOTE_IP"
    echo "  3. Check leader arm connections: ls -la $LEFT_LEADER_PORT $RIGHT_LEADER_PORT"
    echo "  4. Check ZMQ ports are not blocked by firewall"
fi

exit $exit_code
