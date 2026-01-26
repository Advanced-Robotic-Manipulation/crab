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
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots.crab import CrabClient, CrabClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# --- Evaluation Parameters ---
# TODO: Replace with your actual model, dataset, and robot configurations
HF_POLICY_REPO_ID = "<hf_username>/<policy_repo_id>"
HF_DATASET_ID = "<hf_username>/<dataset_repo_id_for_stats>" # Used for pre/post-processing stats

# Robot connection settings
CRAB_REMOTE_IP = "192.168.50.239"

# Task settings
TASK_DESCRIPTION = "A description of the task to evaluate."
NUM_EPISODES = 5
EPISODE_TIME_SEC = 60
FPS = 30


def main():
    # --- Initialize robot and policy ---
    robot_config = CrabClientConfig(remote_ip=CRAB_REMOTE_IP)
    robot = CrabClient(robot_config)

    # Load policy from Hub
    policy = make_policy(HF_POLICY_REPO_ID)
    
    # Load pre/post-processors, using stats from the original training dataset
    preprocessor, postprocessor = make_pre_post_processors(HF_POLICY_REPO_ID, HF_DATASET_ID)


    # --- Create a dataset for logging evaluation episodes ---
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    eval_repo_id = f"{HF_POLICY_REPO_ID}-eval"
    dataset = LeRobotDataset.create(
        repo_id=eval_repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
    )

    # --- Connect to robot ---
    # To connect, the host script must be running on the robot:
    # ./start_host.sh (uses /dev/manipulator_left and /dev/manipulator_right)
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to the robot.")

    # --- Initialize Evaluation ---
    listener, events = init_keyboard_listener()
    init_rerun(session_name="crab_evaluate")

    # --- Evaluation Loop ---
    log_say(f"Starting evaluation of policy {HF_POLICY_REPO_ID}...")
    recorded_episodes = 0
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Running evaluation episode {recorded_episodes + 1}/{NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        if events["rerecord_episode"]:
            log_say("Re-recording episode.")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue
        
        if not events["stop_recording"]:
            dataset.save_episode()
            recorded_episodes += 1

    # --- Cleanup ---
    log_say("Evaluation finished.")
    robot.disconnect()
    listener.stop()
    
    # --- Finalize and Push to Hub ---
    # dataset.finalize()
    # dataset.push_to_hub()
    # print(f"Evaluation data saved to {eval_repo_id}")


if __name__ == "__main__":
    main()
