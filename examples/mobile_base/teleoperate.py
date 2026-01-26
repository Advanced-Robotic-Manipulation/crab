import time
import draccus
from dataclasses import dataclass

from lerobot.robots.mobile_base import MobileBaseClient, MobileBaseClientConfig
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

@dataclass
class TeleopConfig:
    robot: MobileBaseClientConfig
    gamepad: GamepadTeleopConfig = GamepadTeleopConfig(use_gripper=False)
    max_linear_velocity: float = 0.8
    max_angular_velocity: float = 1.0

@draccus.wrap()
def main(cfg: TeleopConfig):
    # --- Initialize robot and teleop device ---
    robot = MobileBaseClient(cfg.robot)
    gamepad = GamepadTeleop(cfg.gamepad)

    # --- Connect to devices ---
    robot.connect()
    gamepad.connect()

    if not all([robot.is_connected, gamepad.is_connected]):
        raise RuntimeError("One or more devices failed to connect.")

    print("Starting teleop loop...")
    print("Use left stick up/down for forward/backward, right stick left/right for turning.")
    
    while True:
        t0 = time.perf_counter()

        # Get actions from teleop device
        gamepad_action = gamepad.get_action()
        
        # Mapping from gamepad deltas to robot velocities.
        linear_velocity = -gamepad_action.get("delta_x", 0.0) * cfg.max_linear_velocity
        angular_velocity = -gamepad_action.get("delta_y", 0.0) * cfg.max_angular_velocity
        
        # --- DEBUGGING: Print raw gamepad input and calculated velocities ---
        print(f"Gamepad raw: dx={gamepad_action.get('delta_x', 0.0):.2f}, dy={gamepad_action.get('delta_y', 0.0):.2f} | "
              f"Vel: Lin={linear_velocity:.2f}, Ang={angular_velocity:.2f}", end='\r', flush=True)

        action = {
            "linear_velocity": float(linear_velocity),
            "angular_velocity": float(angular_velocity),
        }

        # --- Send action to robot ---
        robot.send_action(action)

        precise_sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))

if __name__ == "__main__":
    main()