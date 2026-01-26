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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.crab import CrabClient, CrabClientConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# --- Replay Parameters ---
# TODO: Replace with your actual data and configuration
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"
EPISODE_INDEX = 0

# Robot connection settings
CRAB_REMOTE_IP = "192.168.50.239"


def main():
    # --- Initialize robot ---
    robot_config = CrabClientConfig(remote_ip=CRAB_REMOTE_IP)
    robot = CrabClient(robot_config)

    # --- Load dataset ---
    dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_INDEX])
    # Filter dataset to only include frames from the specified episode
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_INDEX)
    actions = episode_frames.select_columns(ACTION)

    # --- Connect to robot ---
    # To connect, the host script must be running on the robot:
    # ./start_host.sh (uses /dev/manipulator_left and /dev/manipulator_right)
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to the robot.")

    # --- Replay Loop ---
    log_say(f"Starting replay of episode {EPISODE_INDEX} from {HF_REPO_ID}...")
    for idx in range(len(episode_frames)):
        t0 = time.perf_counter()

        # Get recorded action from dataset
        action_array = actions[idx][ACTION]
        action_dict = {name: float(action_array[i]) for i, name in enumerate(dataset.features[ACTION]["names"])}

        # Send action to robot
        robot.send_action(action_dict)

        # Wait to maintain the original recording frequency
        precise_sleep(max(0.0, 1.0 / dataset.fps - (time.perf_counter() - t0)))

    log_say("Replay finished.")
    robot.disconnect()


if __name__ == "__main__":
    main()
