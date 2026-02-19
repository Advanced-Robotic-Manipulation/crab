#!/usr/bin/env python3
"""
Crab Robot Unified Launcher
Starts both host (Orin) and teleop (Aibek) from a single command.
Supports both teleoperation and dataset recording modes.

Optimized for:
- Parallel device checks
- Robust SSH connection (host survives SSH drops)
- Non-blocking output streaming
- Fast startup

Usage:
    python crab_launcher.py              # Interactive mode selection
    python crab_launcher.py --check      # Only check devices
    python crab_launcher.py --host-only  # Only start host
    python crab_launcher.py --teleop-only # Only start teleop (assumes host running)
"""

import subprocess
import sys
import os
import signal
import time
import threading
import argparse
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

ORIN_IP = "192.168.50.239"
ORIN_USER = "orin"
ORIN_PASS = "1234"
ORIN_CRAB_DIR = "~/anywhereVLA/workspaces/miilv_ws/lerobot/examples/crab"

AIBEK_CRAB_DIR = Path(__file__).parent.resolve()
LOGS_DIR = AIBEK_CRAB_DIR / "logs"

ZMQ_OBS_PORT = 5556
HOST_READY_TIMEOUT = 30  # seconds to wait for host to start

# Dataset recording settings
HARDNESS_OPTIONS = {
    1: "soft",
    2: "medium",
    3: "hard",
}

TASK_OPTIONS = {
    1: "Task 1",
    2: "Task 2",
    3: "Task 3",
    4: "Task 4",
    5: "Task 5",
    6: "Task 6",
    7: "Task 7",
}

# =============================================================================
# HELPERS
# =============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'
    BOLD = '\033[1m'

def log(msg, color=Colors.NC, prefix="[LAUNCHER]"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}{prefix} [{timestamp}] {msg}{Colors.NC}", flush=True)

def log_host(msg):
    log(msg, Colors.CYAN, "[ORIN]")

def log_teleop(msg):
    log(msg, Colors.GREEN, "[AIBEK]")

def log_record(msg):
    log(msg, Colors.MAGENTA, "[RECORD]")

def log_error(msg):
    log(msg, Colors.RED, "[ERROR]")

def ssh_cmd(cmd, timeout=30):
    """Run command on Orin via SSH (synchronous, for simple commands)."""
    full_cmd = f"sshpass -p '{ORIN_PASS}' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {ORIN_USER}@{ORIN_IP} '{cmd}'"
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def check_orin_reachable():
    """Check if Orin is reachable."""
    log("Checking if Orin is reachable...")
    success, stdout, stderr = ssh_cmd("echo ok", timeout=10)
    if success and "ok" in stdout:
        log("Orin is reachable", Colors.GREEN)
        return True
    log_error(f"Cannot reach Orin: {stderr}")
    return False

def check_orin_devices():
    """Run check-robot-devices on Orin."""
    log_host("Checking robot devices...")
    success, stdout, stderr = ssh_cmd("check-robot-devices", timeout=15)
    
    for line in stdout.strip().split('\n'):
        if line.strip():
            log_host(line)
    
    return "All devices OK" in stdout

def check_aibek_devices():
    """Check devices on Aibek."""
    log_teleop("Checking local devices...")
    
    devices = {
        "/dev/leader_left": "Left Leader Arm",
        "/dev/leader_right": "Right Leader Arm", 
        "/dev/arduino": "Pedals (Arduino)",
    }
    
    all_ok = True
    for path, name in devices.items():
        if os.path.exists(path):
            target = os.path.realpath(path)
            log_teleop(f"  [OK] {name} -> {os.path.basename(target)}")
        else:
            log_error(f"  [FAIL] {name}: {path} not found")
            all_ok = False
    
    return all_ok

def check_devices_parallel():
    """Run Orin and Aibek device checks in parallel."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(check_orin_devices): "orin",
            executor.submit(check_aibek_devices): "aibek",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                log_error(f"Device check failed for {name}: {e}")
                results[name] = False
    
    return results.get("orin", False), results.get("aibek", False)

def wait_for_host_ready(timeout=HOST_READY_TIMEOUT):
    """Wait for host ZMQ port to be available."""
    log(f"Waiting for host to be ready (port {ZMQ_OBS_PORT})...")
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ORIN_IP, ZMQ_OBS_PORT))
            sock.close()
            if result == 0:
                log("Host is ready!", Colors.GREEN)
                return True
        except:
            pass
        time.sleep(0.3)  # Faster polling
        elapsed = int(time.time() - start)
        if elapsed > 0 and elapsed % 5 == 0:
            log(f"  Still waiting... ({elapsed}s)")
    
    log_error(f"Host did not become ready within {timeout}s")
    return False

# =============================================================================
# MODE SELECTION
# =============================================================================

def select_mode():
    """Ask user to select operation mode."""
    print()
    print(f"{Colors.BOLD}{'='*50}{Colors.NC}")
    print(f"{Colors.BOLD}  Select Operation Mode{Colors.NC}")
    print(f"{Colors.BOLD}{'='*50}{Colors.NC}")
    print()
    print(f"  {Colors.GREEN}1{Colors.NC} - Teleoperation only")
    print(f"  {Colors.MAGENTA}2{Colors.NC} - Dataset Recording")
    print()
    
    while True:
        try:
            choice = input(f"{Colors.BOLD}Enter choice (1 or 2): {Colors.NC}").strip()
            if choice in ['1', '2']:
                return int(choice)
            print(f"{Colors.RED}Invalid choice. Enter 1 or 2.{Colors.NC}")
        except KeyboardInterrupt:
            print()
            sys.exit(0)

def select_hardness():
    """Ask user to select object hardness."""
    print()
    print(f"{Colors.BOLD}  Select Object Hardness:{Colors.NC}")
    print()
    for num, name in HARDNESS_OPTIONS.items():
        print(f"  {Colors.YELLOW}{num}{Colors.NC} - {name}")
    print()
    
    while True:
        try:
            choice = input(f"{Colors.BOLD}Enter hardness (1-{len(HARDNESS_OPTIONS)}): {Colors.NC}").strip()
            if choice.isdigit() and int(choice) in HARDNESS_OPTIONS:
                return int(choice)
            print(f"{Colors.RED}Invalid choice. Enter 1-{len(HARDNESS_OPTIONS)}.{Colors.NC}")
        except KeyboardInterrupt:
            print()
            sys.exit(0)

def select_task():
    """Ask user to select task."""
    print()
    print(f"{Colors.BOLD}  Select Task:{Colors.NC}")
    print()
    for num, name in TASK_OPTIONS.items():
        print(f"  {Colors.CYAN}{num}{Colors.NC} - {name}")
    print()
    
    while True:
        try:
            choice = input(f"{Colors.BOLD}Enter task (1-{len(TASK_OPTIONS)}): {Colors.NC}").strip()
            if choice.isdigit() and int(choice) in TASK_OPTIONS:
                return int(choice)
            print(f"{Colors.RED}Invalid choice. Enter 1-{len(TASK_OPTIONS)}.{Colors.NC}")
        except KeyboardInterrupt:
            print()
            sys.exit(0)

def confirm_recording_settings(task_id, hardness_id):
    """Show selected settings and wait for confirmation."""
    task_name = TASK_OPTIONS[task_id]
    hardness_name = HARDNESS_OPTIONS[hardness_id]
    
    print()
    print(f"{Colors.BOLD}{'='*50}{Colors.NC}")
    print(f"{Colors.BOLD}  Recording Settings{Colors.NC}")
    print(f"{Colors.BOLD}{'='*50}{Colors.NC}")
    print()
    print(f"  Task:     {Colors.CYAN}{task_name}{Colors.NC} (ID: {task_id})")
    print(f"  Hardness: {Colors.YELLOW}{hardness_name}{Colors.NC} (ID: {hardness_id})")
    print()
    print(f"{Colors.BOLD}{'='*50}{Colors.NC}")
    print()
    print(f"Teleop will start first. When ready to record:")
    print(f"  - Press {Colors.GREEN}ENTER{Colors.NC} to start recording")
    print(f"  - Press {Colors.RED}Ctrl+C{Colors.NC} to cancel")
    print()

# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================

class ProcessManager:
    def __init__(self):
        self.host_proc = None
        self.teleop_proc = None
        self.record_proc = None
        self.shutdown_requested = False
        self._output_queues = {}
        self._reader_threads = []
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        if not self.shutdown_requested:
            self.shutdown_requested = True
            log("\nShutdown requested...", Colors.YELLOW)
            self.stop_all()
    
    def _enqueue_output(self, proc, q, prefix, color, log_file):
        """Non-blocking output reader thread."""
        try:
            with open(log_file, 'a') as f:
                for line in iter(proc.stdout.readline, ''):
                    if not line:
                        break
                    line = line.rstrip()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        # Print immediately (non-blocking)
                        print(f"{color}{prefix} [{timestamp}] {line}{Colors.NC}", flush=True)
                        f.write(f"[{timestamp}] {line}\n")
                        f.flush()
        except Exception as e:
            pass
    
    def start_host(self):
        """Start host on Orin via SSH with nohup (survives SSH disconnection)."""
        log_host("Starting Crab Host...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"host_{timestamp}.log"
        
        # Use nohup so host survives if SSH connection drops
        # Redirect output to a remote log file, then tail it locally
        remote_log = f"/tmp/crab_host_{timestamp}.log"
        
        # Start host with nohup in background on Orin
        start_cmd = f"cd {ORIN_CRAB_DIR} && nohup ./start_host.sh > {remote_log} 2>&1 & echo $!"
        success, stdout, stderr = ssh_cmd(start_cmd, timeout=10)
        
        if not success:
            log_error(f"Failed to start host: {stderr}")
            return False
        
        remote_pid = stdout.strip()
        log_host(f"Host started on Orin (PID: {remote_pid})")
        
        # Now tail the remote log for live output
        tail_cmd = f"sshpass -p '{ORIN_PASS}' ssh -o StrictHostKeyChecking=no {ORIN_USER}@{ORIN_IP} 'tail -f {remote_log}' 2>/dev/null"
        
        self.host_proc = subprocess.Popen(
            tail_cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Start non-blocking output reader
        t = threading.Thread(
            target=self._enqueue_output,
            args=(self.host_proc, None, "[ORIN]", Colors.CYAN, log_file),
            daemon=True
        )
        t.start()
        self._reader_threads.append(t)
        
        log_host(f"Log file: {log_file}")
        return True
    
    def start_teleop(self):
        """Start teleop locally."""
        log_teleop("Starting Teleop Client...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"teleop_{timestamp}.log"
        
        cmd = f"cd {AIBEK_CRAB_DIR} && ./start_teleop.sh 2>&1"
        
        self.teleop_proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Start non-blocking output reader
        t = threading.Thread(
            target=self._enqueue_output,
            args=(self.teleop_proc, None, "[AIBEK]", Colors.GREEN, log_file),
            daemon=True
        )
        t.start()
        self._reader_threads.append(t)
        
        log_teleop(f"Teleop process started (PID: {self.teleop_proc.pid})")
        log_teleop(f"Log file: {log_file}")
        return True
    
    def start_recording(self, task_id, hardness_id):
        """Start dataset recording."""
        log_record("Starting Dataset Recording...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"record_{timestamp}.log"
        
        cmd = f"cd {AIBEK_CRAB_DIR} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate mobile-robot && python record.py --task-id {task_id} --hardness-id {hardness_id} 2>&1"
        
        self.record_proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            executable='/bin/bash'
        )
        
        # Start non-blocking output reader
        t = threading.Thread(
            target=self._enqueue_output,
            args=(self.record_proc, None, "[RECORD]", Colors.MAGENTA, log_file),
            daemon=True
        )
        t.start()
        self._reader_threads.append(t)
        
        log_record(f"Recording process started (PID: {self.record_proc.pid})")
        log_record(f"Log file: {log_file}")
        return True
    
    def stop_teleop(self):
        """Stop teleop process."""
        if self.teleop_proc and self.teleop_proc.poll() is None:
            log_teleop("Stopping teleop...")
            self.teleop_proc.terminate()
            try:
                self.teleop_proc.wait(timeout=5)
            except:
                self.teleop_proc.kill()
    
    def stop_all(self):
        """Stop all processes."""
        log("Stopping all processes...", Colors.YELLOW)
        
        # Stop recording first
        if self.record_proc and self.record_proc.poll() is None:
            log_record("Stopping recording...")
            self.record_proc.terminate()
            try:
                self.record_proc.wait(timeout=5)
            except:
                self.record_proc.kill()
        
        # Stop teleop
        self.stop_teleop()
        
        # Stop host tail process
        if self.host_proc and self.host_proc.poll() is None:
            self.host_proc.terminate()
            try:
                self.host_proc.wait(timeout=2)
            except:
                self.host_proc.kill()
        
        # Kill host on Orin
        log_host("Stopping host on Orin...")
        ssh_cmd("pkill -f 'start_host.sh' 2>/dev/null; pkill -f 'crab_host' 2>/dev/null; pkill -f 'python.*crab' 2>/dev/null", timeout=5)
        
        log("All processes stopped", Colors.YELLOW)
    
    def wait(self):
        """Wait for processes to finish."""
        try:
            while not self.shutdown_requested:
                # Check if teleop died
                if self.teleop_proc and self.teleop_proc.poll() is not None:
                    log("Teleop exited, shutting down...", Colors.YELLOW)
                    break
                # Check if recording died  
                if self.record_proc and self.record_proc.poll() is not None:
                    log_record("Recording finished.")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()
    
    def wait_for_recording_start(self):
        """Wait for user to press Enter to start recording."""
        try:
            input(f"{Colors.BOLD}Press ENTER to start recording...{Colors.NC}")
            return True
        except KeyboardInterrupt:
            print()
            return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Crab Robot Unified Launcher")
    parser.add_argument("--check", action="store_true", help="Only check devices")
    parser.add_argument("--host-only", action="store_true", help="Only start host")
    parser.add_argument("--teleop-only", action="store_true", help="Only start teleop")
    args = parser.parse_args()
    
    LOGS_DIR.mkdir(exist_ok=True)
    
    log("=" * 50)
    log("Crab Robot Unified Launcher")
    log("=" * 50)
    
    # Check Orin reachable first (fast fail)
    if not check_orin_reachable():
        sys.exit(1)
    
    # Check devices in parallel for speed
    orin_ok, aibek_ok = check_devices_parallel()
    
    if args.check:
        if orin_ok and aibek_ok:
            log("All devices OK", Colors.GREEN)
        sys.exit(0 if (orin_ok and aibek_ok) else 1)
    
    if not orin_ok:
        log_error("Orin device check failed. Fix issues before continuing.")
        sys.exit(1)
    
    if not args.host_only and not aibek_ok:
        log_error("Aibek device check failed. Fix issues before continuing.")
        sys.exit(1)
    
    # Select mode (unless specific flags given)
    recording_mode = False
    task_id = None
    hardness_id = None
    
    if not args.host_only and not args.teleop_only:
        mode = select_mode()
        if mode == 2:
            recording_mode = True
            hardness_id = select_hardness()
            task_id = select_task()
            confirm_recording_settings(task_id, hardness_id)
    
    # Start processes
    pm = ProcessManager()
    
    if not args.teleop_only:
        if not pm.start_host():
            sys.exit(1)
        
        if not wait_for_host_ready():
            pm.stop_all()
            sys.exit(1)
    
    if not args.host_only:
        if not pm.start_teleop():
            pm.stop_all()
            sys.exit(1)
    
    log("=" * 50)
    if recording_mode:
        log("Teleop running. Prepare for recording.", Colors.GREEN)
    else:
        log("System running. Press Ctrl+C to stop.", Colors.GREEN)
    log("=" * 50)
    
    # If recording mode, wait for Enter then start recording
    if recording_mode:
        time.sleep(2)  # Give teleop time to initialize
        
        if pm.wait_for_recording_start():
            # Stop teleop before starting recording (record.py has its own teleop)
            pm.stop_teleop()
            time.sleep(1)
            
            if not pm.start_recording(task_id, hardness_id):
                pm.stop_all()
                sys.exit(1)
    
    pm.wait()

if __name__ == "__main__":
    main()
