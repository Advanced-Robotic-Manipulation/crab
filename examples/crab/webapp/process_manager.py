"""
Async process manager for Crab robot web control panel.

Manages host (Orin) and session (local teleop/record) lifecycles via
asyncio subprocesses with proper cleanup and log streaming.
"""

import asyncio
import enum
import json
import logging
import os
import signal
import socket
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIN_HOST = "orin"
ORIN_IP = "192.168.50.239"
ZMQ_OBS_PORT = 5556
LEROBOT_DIR = os.path.expanduser("~/AnywhereVLA/lerobot")
CRAB_DIR = f"{LEROBOT_DIR}/examples/crab"
CONDA_WRAP = 'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate mobile-robot && {cmd}"'
HOST_LOG_REMOTE = "/tmp/crab_host.log"
HOST_LOG_LOCAL = "/tmp/crab_host_remote.log"
RERUN_WEB_PORT = 9090
RERUN_GRPC_PORT = 9876
RECORD_STATE_FILE = "/tmp/crab_record_state.json"
DATASET_ROOT = "/mnt/drive/datasets"

TASKS = {
    0: "pick_place",
    1: "bimanual_handover",
    2: "stack",
    3: "pour",
    4: "insertion",
    5: "free_manipulation",
}

HARDNESS = {
    0: "soft",
    1: "medium",
    2: "hard",
}


# ---------------------------------------------------------------------------
# State enums
# ---------------------------------------------------------------------------
class HostState(str, enum.Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class SessionState(str, enum.Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"


class SessionMode(str, enum.Enum):
    TELEOPERATION = "Teleoperation"
    RECORDING = "Recording"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_local_ip() -> str:
    """Get the IP address of this machine as seen from the LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ORIN_IP, 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _conda_cmd(cmd: str) -> list[str]:
    """Wrap a command in conda activation for local execution."""
    return ["bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate mobile-robot && "
            f"export DISPLAY=${{DISPLAY:-:1}} && {cmd}"]


def _ssh_cmd(remote_cmd: str) -> list[str]:
    """Build SSH command to Orin."""
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5", ORIN_HOST, remote_cmd]


async def _read_stream(stream: asyncio.StreamReader, callback: Callable[[str], None]):
    """Read lines from an async stream and forward to callback."""
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                text = line.decode("utf-8", errors="replace").rstrip("\n\r")
                callback(text)
            except Exception:
                pass
    except (asyncio.CancelledError, ConnectionError):
        pass


async def _kill_process(proc: asyncio.subprocess.Process, timeout: float = 3.0):
    """Send SIGTERM, wait, then SIGKILL if needed."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    except ProcessLookupError:
        pass


# ---------------------------------------------------------------------------
# ProcessManager
# ---------------------------------------------------------------------------
class ProcessManager:
    """Singleton managing all Crab robot processes."""

    _instance: Optional["ProcessManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.host_state = HostState.STOPPED
        self.session_state = SessionState.IDLE
        self.session_mode: Optional[SessionMode] = None
        self.rebooting = False

        self._host_log_proc: Optional[asyncio.subprocess.Process] = None
        self._host_log_task: Optional[asyncio.Task] = None
        self._session_proc: Optional[asyncio.subprocess.Process] = None
        self._session_read_tasks: list[asyncio.Task] = []
        self._rerun_proc: Optional[asyncio.subprocess.Process] = None

        self.host_log_callback: Optional[Callable[[str], None]] = None
        self.session_log_callback: Optional[Callable[[str], None]] = None

        # Buffered log messages for UI polling (avoids cross-context issues)
        self._host_log_buffer: list[str] = []
        self._session_log_buffer: list[str] = []

        self.local_ip = get_local_ip()

    # ------------------------------------------------------------------
    # Stale process cleanup
    # ------------------------------------------------------------------
    async def kill_stale_processes(self):
        """Kill leftover processes from previous runs.
        Uses SIGTERM first for graceful shutdown (serial port cleanup),
        then SIGKILL as fallback.
        """
        local_patterns = [
            "examples/crab/teleoperate.py",
            "examples/crab/record.py",
        ]
        # SIGTERM local processes first
        for pat in local_patterns:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "pkill", "-f", pat,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            except Exception:
                pass

        # Kill stale host on Orin — SIGTERM first, wait, then SIGKILL
        try:
            await self._ssh_run(
                "pkill -f 'start_host[.]sh' 2>/dev/null; "
                "pkill -f '[c]rab_host' 2>/dev/null; true"
            )
            await asyncio.sleep(3)
            await self._ssh_run(
                "pkill -9 -f '[c]rab_host' 2>/dev/null; true"
            )
        except Exception:
            pass

        # Release serial ports on Orin
        try:
            await self._ssh_run("fuser -k /dev/ttyACM* /dev/ttyUSB* 2>/dev/null; true")
        except Exception:
            pass

        # SIGKILL local stragglers
        for pat in local_patterns:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "pkill", "-9", "-f", pat,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Host management
    # ------------------------------------------------------------------
    async def start_host(self):
        """Start crab_host on Orin via SSH.

        Key details:
        - Uses [c]rab_host bracket trick in pkill/pgrep to avoid self-matching
          (otherwise the SSH bash session's cmdline contains "crab_host" and
           pkill kills itself, pgrep counts itself).
        - Checks ZMQ first to detect an already-running host.
        - Kills start_host.sh FIRST (it respawns crab_host if killed alone).
        """
        if self.host_state not in (HostState.STOPPED, HostState.ERROR, HostState.STARTING):
            return
        self.host_state = HostState.STARTING
        self._log_host("Starting host on Orin...")

        try:
            # 1. Check if host is already running (ZMQ port reachable)
            already_up = await self._poll_zmq_ready(timeout=3)
            if already_up:
                self.host_state = HostState.RUNNING
                self._log_host("Host already RUNNING (ZMQ reachable)")
                await self._start_host_log_stream()
                return

            # 2. Graceful shutdown: SIGTERM first, then SIGKILL as fallback
            # SIGTERM lets crab_host close serial ports/cameras cleanly.
            # SIGKILL leaves them in dirty state → next start hangs.
            self._log_host("Sending SIGTERM to host processes...")
            await self._ssh_run(
                "pkill -f 'start_host[.]sh' 2>/dev/null; "
                "pkill -f '[c]rab_host' 2>/dev/null; "
                "true"
            )
            self._log_host("Waiting for graceful shutdown (5s)...")
            await asyncio.sleep(5)

            # 3. SIGKILL any survivors
            out = await self._ssh_run("pgrep -c -f '[c]rab_host' || true", capture=True)
            count = out.strip().split('\n')[-1]
            if count not in ("", "0"):
                self._log_host(f"{count} processes still alive, sending SIGKILL...")
                await self._ssh_run("pkill -9 -f '[c]rab_host' 2>/dev/null; true")
                await asyncio.sleep(2)
            else:
                self._log_host("All host processes stopped cleanly")

            # 4. Release any remaining serial port locks
            await self._ssh_run(
                "fuser -k /dev/ttyACM* /dev/ttyUSB* 2>/dev/null; true"
            )
            await asyncio.sleep(3)

            # 5. Launch new host — fire-and-forget SSH (don't wait for return)
            # SSH hangs even with nohup/disown/&, so we launch it and let it go.
            # The ZMQ poll below will detect when the host is actually ready.
            self._log_host("Launching start_host.sh on Orin...")
            start_cmd = (
                "cd ~/anywhereVLA/workspaces/miilv_ws/lerobot/examples/crab && "
                f"nohup bash start_host.sh > {HOST_LOG_REMOTE} 2>&1 </dev/null &"
            )
            self._start_ssh_proc = await asyncio.create_subprocess_exec(
                *_ssh_cmd(start_cmd),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
            # Don't wait — just give SSH a moment to connect and launch
            await asyncio.sleep(2)
            self._log_host("Host launch command sent, waiting for ZMQ readiness...")

            # 5. Start log streaming immediately so user sees progress
            await self._start_host_log_stream()

            # 6. Poll for ZMQ readiness (host takes ~20-30s for hardware init)
            ready = await self._poll_zmq_ready(timeout=90)
            if ready:
                self.host_state = HostState.RUNNING
                self._log_host("Host is RUNNING (ZMQ ready)")
            else:
                self.host_state = HostState.ERROR
                self._log_host("Host failed to become ready within timeout")

        except Exception as e:
            self.host_state = HostState.ERROR
            self._log_host(f"Error starting host: {e}")

    async def _ssh_run(self, remote_cmd: str, capture: bool = False, timeout: float = 15) -> str:
        """Run a command on Orin via SSH with timeout. Returns stdout if capture=True."""
        proc = await asyncio.create_subprocess_exec(
            *_ssh_cmd(remote_cmd),
            stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode().strip() if capture and stdout else ""
        except asyncio.TimeoutError:
            proc.kill()
            self._log_host(f"SSH command timed out ({timeout}s): {remote_cmd[:60]}...")
            return ""

    async def _poll_zmq_ready(self, timeout: float = 30) -> bool:
        """Poll ZMQ port on Orin to check if host is ready."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(ORIN_IP, ZMQ_OBS_PORT),
                    timeout=2,
                )
                writer.close()
                await writer.wait_closed()
                return True
            except Exception:
                await asyncio.sleep(1)
        return False

    async def _start_host_log_stream(self):
        """Start streaming host log from Orin."""
        await self._stop_host_log_stream()
        try:
            self._host_log_proc = await asyncio.create_subprocess_exec(
                *_ssh_cmd(f"tail -F -n 0 {HOST_LOG_REMOTE}"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            if self._host_log_proc.stdout:
                self._host_log_task = asyncio.create_task(
                    _read_stream(self._host_log_proc.stdout, self._log_host)
                )
        except Exception as e:
            self._log_host(f"Failed to start log stream: {e}")

    async def _stop_host_log_stream(self):
        """Stop host log streaming."""
        if self._host_log_task:
            self._host_log_task.cancel()
            try:
                await self._host_log_task
            except asyncio.CancelledError:
                pass
            self._host_log_task = None

        if self._host_log_proc:
            await _kill_process(self._host_log_proc)
            self._host_log_proc = None

    async def stop_host(self):
        """Stop crab_host on Orin."""
        if self.host_state not in (HostState.RUNNING, HostState.ERROR, HostState.STARTING):
            return
        self.host_state = HostState.STOPPING
        self._log_host("Stopping host on Orin...")

        try:
            await self._stop_host_log_stream()

            # Graceful SIGTERM first so host can close serial ports/cameras
            self._log_host("Sending SIGTERM...")
            await self._ssh_run(
                "pkill -f 'start_host[.]sh' 2>/dev/null; "
                "pkill -f '[c]rab_host' 2>/dev/null; true"
            )

            # Wait for graceful shutdown
            self._log_host("Waiting for graceful shutdown...")
            await asyncio.sleep(5)

            # SIGKILL any survivors
            await self._ssh_run(
                "pkill -9 -f '[c]rab_host' 2>/dev/null; true"
            )

            # Wait until ZMQ port is actually down
            self._log_host("Waiting for ZMQ port to close...")
            for _ in range(10):
                try:
                    _, writer = await asyncio.wait_for(
                        asyncio.open_connection(ORIN_IP, ZMQ_OBS_PORT), timeout=1)
                    writer.close()
                    await writer.wait_closed()
                    await asyncio.sleep(1)
                except Exception:
                    break  # Port closed — host is dead

            # Release serial ports
            await self._ssh_run("fuser -k /dev/ttyACM* /dev/ttyUSB* 2>/dev/null; true")
            await asyncio.sleep(2)

            self.host_state = HostState.STOPPED
            self._log_host("Host STOPPED")
        except Exception as e:
            self.host_state = HostState.ERROR
            self._log_host(f"Error stopping host: {e}")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    async def start_session(self, mode: SessionMode, task_id: int = 0,
                            hardness_id: int = 1, dataset_name: str = "",
                            num_episodes: int = 10):
        """Start a teleop or recording session."""
        if self.session_state != SessionState.IDLE:
            return
        self.session_state = SessionState.STARTING
        self.session_mode = mode
        self._log_session(f"Starting {mode.value} session...")

        try:
            # Kill stale Rerun from previous session (frees ports 9090/9876)
            await self._kill_stale_rerun()

            if mode == SessionMode.TELEOPERATION:
                # exec replaces bash with python so pkill signals don't
                # accidentally kill the bash wrapper (default SIGUSR1 = terminate)
                cmd = f"cd {LEROBOT_DIR} && exec python examples/crab/teleoperate.py"
            else:
                if not dataset_name:
                    task_name = TASKS.get(task_id, "unknown")
                    hardness_name = HARDNESS.get(hardness_id, "medium")
                    dataset_name = f"{task_name}_{hardness_name}"

                # exec replaces bash with sudo, so SIGUSR1/SIGUSR2 from
                # toggle_recording/save_and_quit go only to sudo→python,
                # not to a bash wrapper that would die and trigger _watch_session
                cmd = (
                    f"cd {LEROBOT_DIR} && "
                    f"exec sudo -E env \"PATH=$PATH\" "
                    f"python examples/crab/record.py "
                    f"--task-id {task_id} "
                    f"--hardness-id {hardness_id} "
                    f"--num-episodes {num_episodes} "
                    f"--repo-id {dataset_name} "
                    f"--root {DATASET_ROOT}"
                )

            # Teleop: set LEROBOT_RERUN_WEB_PORT for gRPC+web+native Rerun
            # Recording: DON'T set it — use rr.spawn() like start.sh
            # (gRPC approach breaks under sudo)
            if mode == SessionMode.TELEOPERATION:
                full_cmd = _conda_cmd(f"export LEROBOT_RERUN_WEB_PORT={RERUN_WEB_PORT} && {cmd}")
            else:
                full_cmd = _conda_cmd(cmd)
            self._session_proc = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=LEROBOT_DIR,
            )

            # Stream both stdout and stderr
            self._session_read_tasks = []
            if self._session_proc.stdout:
                self._session_read_tasks.append(
                    asyncio.create_task(
                        _read_stream(self._session_proc.stdout, self._log_session)
                    )
                )
            if self._session_proc.stderr:
                self._session_read_tasks.append(
                    asyncio.create_task(
                        _read_stream(self._session_proc.stderr, self._log_session)
                    )
                )

            self.session_state = SessionState.RUNNING
            self._log_session(f"{mode.value} session RUNNING (PID: {self._session_proc.pid})")

            # Monitor process exit in background
            asyncio.create_task(self._watch_session())

        except Exception as e:
            self.session_state = SessionState.IDLE
            self._log_session(f"Error starting session: {e}")

    async def _watch_session(self):
        """Wait for session process to exit and update state."""
        if not self._session_proc:
            return
        try:
            code = await self._session_proc.wait()
            if self.session_state == SessionState.RUNNING:
                self.session_state = SessionState.IDLE
                self._log_session(f"Session exited (code: {code})")
        except asyncio.CancelledError:
            pass

    async def stop_session(self):
        """Stop the current session."""
        if self.session_state not in (SessionState.RUNNING, SessionState.STARTING):
            return
        self.session_state = SessionState.STOPPING
        self._log_session("Stopping session...")

        try:
            # Cancel read tasks
            for t in self._session_read_tasks:
                t.cancel()
            self._session_read_tasks = []

            if self._session_proc:
                await _kill_process(self._session_proc, timeout=3.0)
                self._session_proc = None

            # Kill any stray processes
            for pat in ["examples/crab/teleoperate.py", "examples/crab/record.py"]:
                try:
                    p = await asyncio.create_subprocess_exec(
                        "pkill", "-9", "-f", pat,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await p.wait()
                except Exception:
                    pass

            # Kill stale Rerun processes (free ports 9090/9876 for next session)
            await self._kill_stale_rerun()

            self.session_state = SessionState.IDLE
            self._log_session("Session STOPPED")
        except Exception as e:
            self.session_state = SessionState.IDLE
            self._log_session(f"Error stopping session: {e}")

    # ------------------------------------------------------------------
    # Rerun cleanup
    # ------------------------------------------------------------------
    async def _kill_stale_rerun(self):
        """Kill stale Rerun viewer/server processes to free ports 9090/9876."""
        for pat in ["rerun", "lerobot_dataset_viz"]:
            try:
                p = await asyncio.create_subprocess_exec(
                    "pkill", "-f", pat,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await p.wait()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Recording episode control (signal-based)
    # ------------------------------------------------------------------
    def get_record_state(self) -> Optional[dict]:
        """Read recording state from the state file written by record.py."""
        try:
            with open(RECORD_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return None

    async def toggle_recording(self):
        """Send SIGUSR1 to record.py to toggle episode recording."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "pkill", "-USR1", "-f", "examples/crab/record.py",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._log_session("Sent toggle signal to record.py")
        except Exception as e:
            self._log_session(f"Failed to send toggle signal: {e}")

    async def save_and_quit_recording(self):
        """Send SIGUSR2 to record.py to save current episode and quit."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "pkill", "-USR2", "-f", "examples/crab/record.py",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._log_session("Sent save+quit signal to record.py")
        except Exception as e:
            self._log_session(f"Failed to send save+quit signal: {e}")

    async def discard_recording(self):
        """Send SIGRTMIN to record.py to discard current episode without saving.

        Uses PID from state file to target the python process directly,
        bypassing sudo (which does not forward real-time signals).
        """
        try:
            rstate = self.get_record_state()
            if not rstate or "pid" not in rstate:
                self._log_session("Cannot discard: no PID in record state")
                return
            pid = rstate["pid"]
            proc = await asyncio.create_subprocess_exec(
                "sudo", "kill", "-34", str(pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            self._log_session(f"Sent discard signal to record.py (pid {pid})")
        except Exception as e:
            self._log_session(f"Failed to send discard signal: {e}")

    # ------------------------------------------------------------------
    # Rerun viewer management
    # ------------------------------------------------------------------
    async def start_rerun_viewer(self, dataset_name: str, episode_index: int = 0):
        """Launch Rerun dataset visualizer — opens native viewer via rr.spawn()."""
        await self.stop_rerun_viewer()

        cmd = (
            f"cd {LEROBOT_DIR} && "
            f"lerobot-dataset-viz "
            f"--repo-id {dataset_name} "
            f"--episode-index {episode_index} "
            f"--root {DATASET_ROOT}/{dataset_name}"
        )

        try:
            self._rerun_proc = await asyncio.create_subprocess_exec(
                *_conda_cmd(cmd),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=LEROBOT_DIR,
            )
            # rr.spawn() opens native viewer; process exits after logging data
            await asyncio.sleep(2)
            if self._rerun_proc.returncode is not None and self._rerun_proc.returncode != 0:
                raise RuntimeError(
                    f"Rerun exited with code {self._rerun_proc.returncode}"
                )
            return True
        except Exception as e:
            logger.error(f"Failed to start Rerun viewer: {e}")
            self._rerun_proc = None
            raise

    async def stop_rerun_viewer(self):
        """Stop the Rerun viewer subprocess and any spawned native viewers."""
        if self._rerun_proc:
            await _kill_process(self._rerun_proc)
            self._rerun_proc = None

        # Kill stray viz processes AND native rerun viewers spawned by rr.spawn()
        for pat in ["lerobot_dataset_viz", "rerun"]:
            try:
                p = await asyncio.create_subprocess_exec(
                    "pkill", "-f", pat,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await p.wait()
            except Exception:
                pass

    @property
    def rerun_running(self) -> bool:
        return self._rerun_proc is not None and self._rerun_proc.returncode is None

    # ------------------------------------------------------------------
    # Reboot Orin
    # ------------------------------------------------------------------
    async def reboot_orin(self):
        """Reboot the Orin (full robot reboot). Nuclear option."""
        self.rebooting = True
        self._log_host("REBOOTING ORIN...")
        try:
            await self.stop_session()
        except Exception:
            pass
        try:
            await self._stop_host_log_stream()
        except Exception:
            pass
        self.host_state = HostState.STOPPED
        self.session_state = SessionState.IDLE
        await self._ssh_run("sudo reboot", timeout=5)
        self._log_host("Reboot command sent. Waiting for Orin to come back...")

        # Poll until Orin is reachable again (up to 120s)
        for i in range(24):
            await asyncio.sleep(5)
            self._log_host(f"Waiting... {(i+1)*5}s")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *_ssh_cmd("echo UP"),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    stdin=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                if stdout and b"UP" in stdout:
                    self._log_host(f"Orin is BACK after {(i+1)*5}s")
                    devs = await self._ssh_run(
                        "ls /dev/manipulator_* /dev/ttyACM* 2>/dev/null", capture=True
                    )
                    self._log_host(f"Devices: {devs}")
                    self.rebooting = False
                    return
            except Exception:
                continue
        self._log_host("Orin did not come back within 120s!")
        self.rebooting = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    async def cleanup_all(self):
        """Stop everything — called on app shutdown."""
        logger.info("Cleaning up all processes...")
        try:
            await self.stop_session()
        except Exception:
            pass
        try:
            await self.stop_rerun_viewer()
        except Exception:
            pass
        try:
            await self.stop_host()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_host(self, msg: str):
        logger.info(f"[HOST] {msg}")
        self._host_log_buffer.append(msg)

    def _log_session(self, msg: str):
        logger.info(f"[SESSION] {msg}")
        self._session_log_buffer.append(msg)

    def drain_host_log(self) -> list[str]:
        """Return and clear buffered host log messages."""
        msgs = self._host_log_buffer
        self._host_log_buffer = []
        return msgs

    def drain_session_log(self) -> list[str]:
        """Return and clear buffered session log messages."""
        msgs = self._session_log_buffer
        self._session_log_buffer = []
        return msgs
