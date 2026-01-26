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

import base64
import json
import logging
import signal
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from .config_crab import CrabConfig, CrabHostConfig
from .crab import Crab

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class CrabServerConfig:
    """Configuration for the Crab host script."""
    robot: CrabConfig
    host: CrabHostConfig = field(default_factory=CrabHostConfig)


class CrabHost:
    def __init__(self, config: CrabHostConfig):
        logger.info(f"Initializing ZMQ sockets...")
        logger.info(f"  Command port: {config.port_zmq_cmd}")
        logger.info(f"  Observation port: {config.port_zmq_observations}")
        
        self.zmq_context = zmq.Context()
        
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")
        logger.info(f"  Command socket bound to tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")
        logger.info(f"  Observation socket bound to tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz
        
        logger.info(f"ZMQ sockets initialized successfully")

    def disconnect(self):
        logger.info("Closing ZMQ sockets...")
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        logger.info("ZMQ sockets closed")


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    global _shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True


@draccus.wrap()
def main(cfg: CrabServerConfig):
    global _shutdown_requested
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    robot = None
    host = None
    
    try:
        logger.info("=" * 50)
        logger.info("CRAB HOST STARTING")
        logger.info("=" * 50)
        
        # Step 1: Configure robot
        logger.info("Step 1/3: Configuring Crab Robot...")
        logger.info(f"  Robot ID: {cfg.robot.id}")
        logger.info(f"  Left arm port: {cfg.robot.left_arm_port}")
        logger.info(f"  Right arm port: {cfg.robot.right_arm_port}")
        
        robot = Crab(cfg.robot)
        logger.info("  Robot object created")

        # Step 2: Connect to hardware (with graceful handling of optional components)
        logger.info("Step 2/3: Connecting to Crab hardware...")
        logger.info("  Arms are REQUIRED, other components are OPTIONAL")
        
        robot.connect()  # This now handles failures gracefully
        
        if not robot.is_connected:
            logger.error("Required components (arms) failed to connect. Cannot start host.")
            raise RuntimeError("Arms failed to connect")

        # Step 3: Start ZMQ server
        logger.info("Step 3/3: Starting Crab Host server...")
        host = CrabHost(cfg.host)

        logger.info("=" * 50)
        logger.info("CRAB HOST READY - Waiting for client connections")
        logger.info(f"  Connect client to this machine's IP on ports {cfg.host.port_zmq_cmd}/{cfg.host.port_zmq_observations}")
        if not robot.is_fully_connected:
            logger.warning("  NOTE: Some optional components are disabled (see above)")
        logger.info("=" * 50)

        last_cmd_time = time.time()
        watchdog_active = False
        loop_count = 0
        last_status_time = time.time()
        observations_sent = 0
        commands_received = 0
        
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < host.connection_time_s and not _shutdown_requested:
            loop_start_time = time.time()
            loop_count += 1
            
            # Process incoming commands
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
                commands_received += 1
            except zmq.Again:
                pass  # No new command
            except Exception as e:
                logger.error(f"Command processing failed: {e}")

            # Watchdog check
            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logger.warning(
                    f"Watchdog triggered: No command received for >{host.watchdog_timeout_ms}ms. Stopping robot."
                )
                robot.stop_base()
                try:
                    robot.left_arm.bus.disable_torque()
                    robot.right_arm.bus.disable_torque()
                except Exception as e:
                    logger.warning(f"Error disabling torque: {e}")
                watchdog_active = True

            # Get observation
            try:
                last_observation = robot.get_observation()
            except Exception as e:
                logger.error(f"Failed to get observation: {e}")
                continue

            # Encode images (only for connected cameras)
            for cam_key in robot.cameras:
                frame = last_observation.get(cam_key)
                if frame is not None:
                    try:
                        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if ret:
                            last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                        else:
                            last_observation[cam_key] = ""
                    except Exception as e:
                        logger.debug(f"Failed to encode {cam_key}: {e}")
                        last_observation[cam_key] = ""
                else:
                    last_observation[cam_key] = ""

            # Send observation
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
                observations_sent += 1
            except zmq.Again:
                pass  # No client connected, drop observation

            # Status update every 5 seconds
            if now - last_status_time >= 5.0:
                hz = loop_count / (now - start_time) if (now - start_time) > 0 else 0
                logger.info(f"Status: {loop_count} loops @ {hz:.1f}Hz | {observations_sent} obs sent | {commands_received} cmds recv")
                last_status_time = now

            # Rate limiting
            elapsed = time.time() - loop_start_time
            sleep_time = (1 / host.max_loop_freq_hz) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if _shutdown_requested:
            logger.info("Shutdown requested by signal")
        else:
            logger.info("Host cycle time reached")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down Crab Host...")
        if robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
                    logger.info("Robot disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")
        if host is not None:
            try:
                host.disconnect()
                logger.info("Host disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting host: {e}")
        logger.info("Crab Host shut down cleanly.")


if __name__ == "__main__":
    main()
