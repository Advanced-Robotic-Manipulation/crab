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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.crab import CrabClient, CrabClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# --- Recording Parameters ---
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 45
TASK_DESCRIPTION = "A description of the task being recorded."
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"  # TODO: Replace with your repo ID

# --- Device Configurations ---
# Robot connection settings
CRAB_REMOTE_IP = "192.168.50.239"

# Leader arm ports (using persistent symlinks)
LEFT_LEADER_PORT = "/dev/manipulator_left"
RIGHT_LEADER_PORT = "/dev/manipulator_right"


def main():
    # --- Initialize robot and teleop devices ---
    robot_config = CrabClientConfig(remote_ip=CRAB_REMOTE_IP)
    robot = CrabClient(robot_config)

    left_leader_config = SO101LeaderConfig(port=LEFT_LEADER_PORT, id="left_leader")
    right_leader_config = SO101LeaderConfig(port=RIGHT_LEADER_PORT, id="right_leader")
    left_leader = SO101Leader(left_leader_config)
    right_leader = SO101Leader(right_leader_config)

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_devices = [left_leader, right_leader, keyboard]

    # --- Create Dataset ---
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # --- Connect to devices ---
    robot.connect()
    for device in teleop_devices:
        device.connect()

    # --- Initialize Recording ---
    listener, events = init_keyboard_listener()
    init_rerun(session_name="crab_record")

    if not all(dev.is_connected for dev in [robot, *teleop_devices]):
        raise RuntimeError("One or more devices failed to connect.")

    # --- Custom Teleop Action Processor ---
    # This processor combines actions from all teleop devices
    def process_teleop_actions(actions: list[dict]):
        left_action, right_action, key_action = actions[0], actions[1], actions[2]
        # Add prefixes
        prefixed_left = {f"left_{k}": v for k, v in left_action.items()}
        prefixed_right = {f"right_{k}": v for k, v in right_action.items()}
        # Get base action from keyboard
        base_action = robot._from_keyboard_to_base_action(key_action)
        return {**prefixed_left, **prefixed_right, **base_action}

    _, robot_action_processor, robot_observation_processor = make_default_processors()


    # --- Recording Loop ---
    print("Starting record loop... Press 'r' to start/stop recording, 'q' to quit.")
    recorded_episodes = 0
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes + 1}/{NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=teleop_devices,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            # Pass a custom processor to combine actions from the 3 teleop devices
            teleop_action_processor=process_teleop_actions,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
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
    log_say("Stopping recording.")
    robot.disconnect()
    for device in teleop_devices:
        device.disconnect()
    listener.stop()

    # --- Finalize and Push to Hub ---
    # dataset.finalize()
    # dataset.push_to_hub()
    # print(f"Dataset pushed to hub at {HF_REPO_ID}")


if __name__ == "__main__":
    main()
