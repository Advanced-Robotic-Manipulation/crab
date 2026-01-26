#!/bin/bash

# --- Activate Conda Environment ---
# This ensures the correct python interpreter and libraries are used.
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


# This script starts the Crab robot host on the onboard computer.
#
# Device symlinks are created by udev rules in /etc/udev/rules.d/99-robot-devices.rules
# Run 'check-robot-devices' to verify all devices are correctly mapped.

# --- Configuration ---
# A unique ID for your Crab robot.
ROBOT_ID="my_awesome_crab_robot_20251217"

# Serial port for the left SO-101 follower arm (uses persistent symlink)
LEFT_ARM_PORT="/dev/manipulator_left"

# Serial port for the right SO-101 follower arm (uses persistent symlink)
RIGHT_ARM_PORT="/dev/manipulator_right"

# Camera paths (use persistent symlinks)
MAIN_CAMERA="/dev/camera_realsense_rgb"
LEFT_ARM_CAMERA="/dev/camera_left"
RIGHT_ARM_CAMERA="/dev/camera_right"

# Camera settings
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# ZMQ ports
ZMQ_CMD_PORT=5555
ZMQ_OBS_PORT=5556

# --- Helper function to check if a port is in use ---
check_port() {
    local port=$1
    if command -v ss &> /dev/null; then
        ss -tuln 2>/dev/null | grep -q ":${port} "
    elif command -v netstat &> /dev/null; then
        netstat -tuln 2>/dev/null | grep -q ":${port} "
    else
        # If neither command is available, try lsof
        lsof -i :${port} &> /dev/null
    fi
    return $?
}

# --- Kill any existing process on ZMQ ports ---
cleanup_ports() {
    echo "Checking for processes using ZMQ ports..."
    
    for port in $ZMQ_CMD_PORT $ZMQ_OBS_PORT; do
        if check_port $port; then
            echo "  Port $port is in use. Attempting to free it..."
            # Find and kill processes using this port
            if command -v fuser &> /dev/null; then
                fuser -k ${port}/tcp 2>/dev/null
            else
                # Alternative using lsof
                local pids=$(lsof -t -i:${port} 2>/dev/null)
                if [ -n "$pids" ]; then
                    echo "  Killing PIDs: $pids"
                    kill -9 $pids 2>/dev/null
                fi
            fi
            sleep 1
            
            if check_port $port; then
                echo "Error: Could not free port $port. Please manually kill the process."
                exit 1
            fi
            echo "  Port $port is now free."
        fi
    done
    echo "All ZMQ ports are available."
}

# --- Cleanup function for graceful shutdown ---
cleanup() {
    echo ""
    echo "Shutting down Crab host..."
    # The Python script handles its own cleanup, but we ensure ZMQ ports are freed
    sleep 2
    echo "Cleanup complete."
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# --- Pre-flight checks ---
echo "=== Crab Robot Host Pre-flight Checks ==="

# Check device symlinks exist
echo "Checking device symlinks..."
for device in "$LEFT_ARM_PORT" "$RIGHT_ARM_PORT"; do
    if [ ! -e "$device" ]; then
        echo "Warning: Device $device not found!"
        echo "  Run 'check-robot-devices' to diagnose."
    else
        echo "  Found: $device"
    fi
done

# Check camera devices
echo "Checking camera devices..."
for device in "$MAIN_CAMERA" "$LEFT_ARM_CAMERA" "$RIGHT_ARM_CAMERA"; do
    if [ ! -e "$device" ]; then
        echo "Warning: Camera $device not found!"
    else
        echo "  Found: $device"
    fi
done

# Cleanup any stale port bindings
cleanup_ports

# --- Start the Crab Host ---
echo ""
echo "=== Starting Crab Robot Host ==="
echo "  Robot ID:  $ROBOT_ID"
echo "  Left arm:  $LEFT_ARM_PORT"
echo "  Right arm: $RIGHT_ARM_PORT"
echo "  Main cam:  $MAIN_CAMERA"
echo "  Left cam:  $LEFT_ARM_CAMERA"
echo "  Right cam: $RIGHT_ARM_CAMERA"
echo "  ZMQ Cmd Port:  $ZMQ_CMD_PORT"
echo "  ZMQ Obs Port:  $ZMQ_OBS_PORT"
echo ""

python -u -m lerobot.robots.crab.crab_host \
    --robot.id="$ROBOT_ID" \
    --robot.left_arm_port="$LEFT_ARM_PORT" \
    --robot.right_arm_port="$RIGHT_ARM_PORT" \
    --robot.cameras.main_camera.index_or_path="$MAIN_CAMERA" \
    --robot.cameras.main_camera.width="$CAMERA_WIDTH" \
    --robot.cameras.main_camera.height="$CAMERA_HEIGHT" \
    --robot.cameras.main_camera.fps="$CAMERA_FPS" \
    --robot.cameras.left_arm_camera.index_or_path="$LEFT_ARM_CAMERA" \
    --robot.cameras.left_arm_camera.width="$CAMERA_WIDTH" \
    --robot.cameras.left_arm_camera.height="$CAMERA_HEIGHT" \
    --robot.cameras.left_arm_camera.fps="$CAMERA_FPS" \
    --robot.cameras.right_arm_camera.index_or_path="$RIGHT_ARM_CAMERA" \
    --robot.cameras.right_arm_camera.width="$CAMERA_WIDTH" \
    --robot.cameras.right_arm_camera.height="$CAMERA_HEIGHT" \
    --robot.cameras.right_arm_camera.fps="$CAMERA_FPS" \
    --host.port_zmq_cmd="$ZMQ_CMD_PORT" \
    --host.port_zmq_observations="$ZMQ_OBS_PORT"

exit_code=$?
echo "Crab host exited with code: $exit_code"
exit $exit_code
