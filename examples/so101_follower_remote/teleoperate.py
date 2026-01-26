# !/usr/bin/env python

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

import time

from lerobot.robots.so101_follower import SO101FollowerClient, SO101FollowerClientConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    # Create the robot and teleoperator configurations
    robot_config = SO101FollowerClientConfig(remote_ip="192.168.50.239", id="left_follower")
    teleop_arm_config = SO101LeaderConfig(port="/dev/ttyUSB0", id="leader_left")

    # Initialize the robot and teleoperator
    robot = SO101FollowerClient(robot_config)
    leader_arm = SO101Leader(teleop_arm_config)

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on the robot: `python -m lerobot.robots.so101_follower.so101_follower_host --robot.id=my_so101_follower`
    robot.connect()
    leader_arm.connect()

    # Init rerun viewer
    init_rerun(session_name="so101_follower_teleop")

    if not robot.is_connected or not leader_arm.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        action = leader_arm.get_action()

        # Send action to robot
        _ = robot.send_action(action)

        # Visualize
        log_rerun_data(observation=observation, action=action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
