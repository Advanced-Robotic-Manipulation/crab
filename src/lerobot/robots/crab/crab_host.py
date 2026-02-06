"""Crab Host - optimized for low latency."""
import base64
import json
import signal
import time
from dataclasses import dataclass, field

import cv2
import draccus
import numpy as np
import zmq

from .config_crab import CrabConfig, CrabHostConfig
from .crab import Crab

# JPEG quality for network transfer (lower = smaller/faster, 30-50 is fine for teleop)
JPEG_QUALITY = 35

@dataclass
class CrabServerConfig:
    robot: CrabConfig
    host: CrabHostConfig = field(default_factory=CrabHostConfig)

class CrabHost:
    def __init__(self, config: CrabHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.setsockopt(zmq.SNDHWM, 1)  # Only buffer 1 message
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")
        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()

_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True

@draccus.wrap()
def main(cfg: CrabServerConfig):
    global _shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    print(f"Connecting robot...", flush=True)
    robot = Crab(cfg.robot)
    robot.connect()
    host = CrabHost(cfg.host)
    print(f"Ready. cmd={cfg.host.port_zmq_cmd} obs={cfg.host.port_zmq_observations}", flush=True)
    
    last_cmd_time = time.time()
    watchdog_active = False
    loop_count = 0
    last_status = time.perf_counter()
    
    start = time.perf_counter()
    
    try:
        while (time.perf_counter() - start) < host.connection_time_s and not _shutdown:
            t0 = time.perf_counter()
            loop_count += 1
            
            # Commands
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                robot.send_action(json.loads(msg))
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass

            # Watchdog
            if (time.time() - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                robot.stop_base()
                try:
                    robot.left_arm.bus.disable_torque()
                    robot.right_arm.bus.disable_torque()
                except:
                    pass
                watchdog_active = True

            # Observation
            try:
                obs = robot.get_observation()
            except:
                continue

            # Encode images (low quality JPEG for fast transfer)
            for k in robot.cameras:
                f = obs.get(k)
                if f is not None:
                    ret, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    obs[k] = base64.b64encode(buf).decode("utf-8") if ret else ""
                else:
                    obs[k] = ""

            # Encode tactile matrices (base64 encoded raw bytes)
            for tactile_key in ["tactile_left", "tactile_right"]:
                matrix = obs.get(tactile_key)
                if matrix is not None and isinstance(matrix, np.ndarray):
                    # Convert to uint16 bytes and base64 encode
                    obs[tactile_key] = base64.b64encode(matrix.astype(np.uint16).tobytes()).decode("utf-8")
                else:
                    obs[tactile_key] = ""

            # Send
            try:
                host.zmq_observation_socket.send_string(json.dumps(obs), zmq.NOBLOCK)
            except zmq.Again:
                pass

            # Status
            if time.perf_counter() - last_status >= 5.0:
                hz = loop_count / (time.perf_counter() - start)
                tactile_status = "OK" if robot._tactile_connected else "OFF"
                print(f"{loop_count} loops @ {hz:.1f}Hz | tactile: {tactile_status}", flush=True)
                last_status = time.perf_counter()

            # Rate limit
            sleep = (1 / host.max_loop_freq_hz) - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    
    finally:
        print("Shutting down...", flush=True)
        if robot.is_connected:
            robot.disconnect()
        host.disconnect()

if __name__ == "__main__":
    main()
