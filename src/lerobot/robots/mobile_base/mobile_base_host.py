import json
import logging
import time
from dataclasses import dataclass, field

import draccus
import zmq

from .config_mobile_base import MobileBaseConfig, MobileBaseHostConfig
from .mobile_base import MobileBase


@dataclass
class MobileBaseServerConfig:
    """Configuration for the MobileBase host script."""
    robot: MobileBaseConfig
    host: MobileBaseHostConfig = field(default_factory=MobileBaseHostConfig)


class MobileBaseHost:
    def __init__(self, config: MobileBaseHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: MobileBaseServerConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info("Configuring MobileBase Robot")
    robot = MobileBase(cfg.robot)

    logging.info("Connecting to MobileBase hardware")
    robot.connect()

    logging.info("Starting MobileBase Host server")
    host = MobileBaseHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for client commands...")
    try:
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass  # No new command
            except Exception as e:
                logging.error(f"Command processing failed: {e}")

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Watchdog triggered: No command received for >{host.watchdog_timeout_ms}ms. Stopping robot."
                )
                robot.stop()
                watchdog_active = True

            last_observation = robot.get_observation()
            
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping observation, no client connected.")

            elapsed = time.time() - loop_start_time
            sleep_time = (1 / host.max_loop_freq_hz) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Host cycle time reached.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting.")
    finally:
        print("Shutting down MobileBase Host.")
        robot.disconnect()
        host.disconnect()
        logging.info("MobileBase Host shut down cleanly.")


if __name__ == "__main__":
    main()
