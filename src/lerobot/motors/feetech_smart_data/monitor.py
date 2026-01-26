#!/usr/bin/env python3
"""
Feetech STS3215 Motor Smart Data Monitor

A comprehensive monitoring tool for viewing motor telemetry data in real-time.
Features:
- Pretty terminal UI with live updates
- Optional Rerun visualization with graphs
- JSON output mode for logging
- Alerts for over-temperature, over-current, errors

Usage:
    # Basic monitoring with TUI
    monitor-motor-smart-data

    # With rerun visualization
    monitor-motor-smart-data --rerun

    # Monitor specific arm only
    monitor-motor-smart-data --arm left

    # JSON output for logging
    monitor-motor-smart-data --json --interval 1000
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

from .smart_data_reader import FeetechSmartDataReader, SmartDataBatch

# Motor names for SO-101 arm
MOTOR_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}

# Alert thresholds
ALERT_TEMP_WARN = 45  # Celsius
ALERT_TEMP_CRIT = 55  # Celsius
ALERT_CURRENT_WARN = 500  # mA
ALERT_CURRENT_CRIT = 1000  # mA
ALERT_VOLTAGE_LOW = 6.0  # V
ALERT_VOLTAGE_HIGH = 8.5  # V


@dataclass
class MonitorConfig:
    """Configuration for the motor monitor."""
    left_port: str = "/dev/manipulator_left"
    right_port: str = "/dev/manipulator_right"
    baudrate: int = 1_000_000
    interval_ms: float = 100
    use_rerun: bool = False
    json_output: bool = False
    arm: str = "both"  # "left", "right", or "both"
    show_alerts: bool = True
    compact: bool = False


class TerminalUI:
    """Simple terminal UI for motor monitoring."""

    CLEAR = "\033[H\033[J"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

    @classmethod
    def clear(cls):
        print(cls.CLEAR, end="")

    @classmethod
    def color(cls, text: str, color: str) -> str:
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.RESET}"

    @classmethod
    def status_color(cls, value: float, warn: float, crit: float, invert: bool = False) -> str:
        """Return color code based on threshold."""
        if invert:
            if value <= crit:
                return cls.RED
            elif value <= warn:
                return cls.YELLOW
            return cls.GREEN
        else:
            if value >= crit:
                return cls.RED
            elif value >= warn:
                return cls.YELLOW
            return cls.GREEN


class RerunLogger:
    """Logger for sending motor data to Rerun visualization."""

    def __init__(self):
        self.initialized = False

    def init(self, session_name: str = "motor_smart_data"):
        """Initialize rerun connection."""
        try:
            import rerun as rr

            batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
            os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
            rr.init(session_name)
            memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
            rr.spawn(memory_limit=memory_limit)
            self.initialized = True
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize Rerun: {e}")
            return False

    def log_batch(self, arm_name: str, batch: SmartDataBatch):
        """Log a batch of motor data to rerun."""
        if not self.initialized:
            return

        try:
            import rerun as rr

            for motor_id, motor_data in batch.motors.items():
                motor_name = MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")
                prefix = f"motor/{arm_name}/{motor_name}"

                # Log each metric as a scalar (for graphs)
                if motor_data.temperature is not None:
                    rr.log(f"{prefix}/temperature_c", rr.Scalar(motor_data.temperature))

                if motor_data.current is not None:
                    rr.log(f"{prefix}/current_ma", rr.Scalar(motor_data.current))

                if motor_data.voltage is not None:
                    rr.log(f"{prefix}/voltage_v", rr.Scalar(motor_data.voltage))

                if motor_data.load is not None:
                    rr.log(f"{prefix}/load_percent", rr.Scalar(motor_data.load))

                if motor_data.velocity is not None:
                    rr.log(f"{prefix}/velocity", rr.Scalar(motor_data.velocity))

                if motor_data.position is not None:
                    rr.log(f"{prefix}/position", rr.Scalar(motor_data.position))

        except Exception as e:
            pass  # Don't crash on rerun errors


class MotorMonitor:
    """Main motor monitoring class."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.readers = {}
        self.rerun_logger = None
        self.alerts = []
        self.loop_count = 0
        self.start_time = time.time()

    def connect(self) -> bool:
        """Connect to motor buses."""
        success = True

        if self.config.arm in ("left", "both"):
            reader = FeetechSmartDataReader(
                self.config.left_port,
                self.config.baudrate
            )
            if reader.connect():
                ids = reader.scan()
                if ids:
                    self.readers["left"] = reader
                    print(f"Connected to LEFT arm: {len(ids)} motors")
                else:
                    print("Warning: No motors found on LEFT arm")
                    reader.disconnect()
                    success = False
            else:
                print(f"Error: Failed to connect to LEFT arm at {self.config.left_port}")
                success = False

        if self.config.arm in ("right", "both"):
            reader = FeetechSmartDataReader(
                self.config.right_port,
                self.config.baudrate
            )
            if reader.connect():
                ids = reader.scan()
                if ids:
                    self.readers["right"] = reader
                    print(f"Connected to RIGHT arm: {len(ids)} motors")
                else:
                    print("Warning: No motors found on RIGHT arm")
                    reader.disconnect()
                    success = False
            else:
                print(f"Error: Failed to connect to RIGHT arm at {self.config.right_port}")
                success = False

        if not self.readers:
            print("Error: No arms connected!")
            return False

        # Initialize rerun if requested
        if self.config.use_rerun:
            self.rerun_logger = RerunLogger()
            if not self.rerun_logger.init():
                print("Warning: Rerun disabled due to initialization failure")
                self.rerun_logger = None

        return success

    def disconnect(self):
        """Disconnect from all buses."""
        for name, reader in self.readers.items():
            reader.disconnect()
        self.readers.clear()

    def check_alerts(self, arm_name: str, batch: SmartDataBatch):
        """Check for alert conditions."""
        alerts = []

        for motor_id, m in batch.motors.items():
            motor_name = MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")
            prefix = f"{arm_name}/{motor_name}"

            # Temperature alerts
            if m.temperature is not None:
                if m.temperature >= ALERT_TEMP_CRIT:
                    alerts.append(f"CRITICAL: {prefix} temp={m.temperature}C")
                elif m.temperature >= ALERT_TEMP_WARN:
                    alerts.append(f"WARNING: {prefix} temp={m.temperature}C")

            # Current alerts
            if m.current is not None:
                if m.current >= ALERT_CURRENT_CRIT:
                    alerts.append(f"CRITICAL: {prefix} current={m.current:.0f}mA")
                elif m.current >= ALERT_CURRENT_WARN:
                    alerts.append(f"WARNING: {prefix} current={m.current:.0f}mA")

            # Voltage alerts
            if m.voltage is not None:
                if m.voltage <= ALERT_VOLTAGE_LOW:
                    alerts.append(f"WARNING: {prefix} voltage={m.voltage:.1f}V LOW")
                elif m.voltage >= ALERT_VOLTAGE_HIGH:
                    alerts.append(f"WARNING: {prefix} voltage={m.voltage:.1f}V HIGH")

            # Error flags
            if m.has_error:
                flags = [k for k, v in m.status_flags.items() if v]
                alerts.append(f"ERROR: {prefix} flags={flags}")

        return alerts

    def format_motor_row(self, arm: str, motor_id: int, m) -> str:
        """Format a single motor row for display."""
        ui = TerminalUI
        name = MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")

        # Color code values based on thresholds
        temp_color = ui.status_color(m.temperature or 0, ALERT_TEMP_WARN, ALERT_TEMP_CRIT)
        curr_color = ui.status_color(m.current or 0, ALERT_CURRENT_WARN, ALERT_CURRENT_CRIT)

        volt_ok = ALERT_VOLTAGE_LOW <= (m.voltage or 0) <= ALERT_VOLTAGE_HIGH
        volt_color = ui.GREEN if volt_ok else ui.YELLOW

        err_color = ui.RED if m.has_error else ui.GREEN

        temp_str = f"{m.temperature:3d}" if m.temperature is not None else "N/A"
        curr_str = f"{m.current:6.1f}" if m.current is not None else "  N/A "
        volt_str = f"{m.voltage:4.1f}" if m.voltage is not None else " N/A"
        load_str = f"{m.load:+5.1f}" if m.load is not None else " N/A "
        pos_str = f"{m.position:5d}" if m.position is not None else " N/A "
        vel_str = f"{m.velocity:5d}" if m.velocity is not None else " N/A "

        return (
            f" {arm:5s} | {motor_id} | {name:14s} | "
            f"{ui.color(temp_str, temp_color)} C | "
            f"{ui.color(curr_str, curr_color)} mA | "
            f"{ui.color(volt_str, volt_color)} V | "
            f"{load_str}% | "
            f"{pos_str} | {vel_str} | "
            f"{ui.color('ERR' if m.has_error else 'OK', err_color)}"
        )

    def render_tui(self, data):
        """Render the terminal UI."""
        ui = TerminalUI
        ui.clear()

        elapsed = time.time() - self.start_time
        hz = self.loop_count / elapsed if elapsed > 0 else 0

        # Header
        print(ui.bold("=" * 95))
        print(ui.bold("  FEETECH STS3215 MOTOR SMART DATA MONITOR"))
        print(ui.bold(f"  Loop: {self.loop_count} | Rate: {hz:.1f} Hz | Interval: {self.config.interval_ms}ms"))
        if self.rerun_logger:
            print(ui.color("  [RERUN ENABLED - View graphs in Rerun Viewer]", ui.CYAN))
        print(ui.bold("=" * 95))
        print()

        # Table header
        header = " Arm   | # | Motor          | Temp   | Current  | Volt  | Load   | Pos   | Vel   | Status"
        print(ui.bold(header))
        print("-" * 95)

        # Motor rows
        for arm_name, batch in data.items():
            for motor_id in sorted(batch.motors.keys()):
                m = batch.motors[motor_id]
                print(self.format_motor_row(arm_name, motor_id, m))
            print("-" * 95)

        # Alerts section
        if self.config.show_alerts and self.alerts:
            print()
            print(ui.bold(ui.color("ALERTS:", ui.YELLOW)))
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                if "CRITICAL" in alert:
                    print(f"  {ui.color(alert, ui.RED)}")
                else:
                    print(f"  {ui.color(alert, ui.YELLOW)}")

        print()
        print(ui.color("Press Ctrl+C to stop", ui.CYAN))

    def run(self):
        """Main monitoring loop."""
        try:
            while True:
                loop_start = time.time()
                self.loop_count += 1

                # Read data from all connected arms
                data = {}
                for arm_name, reader in self.readers.items():
                    try:
                        batch = reader.read_all()
                        data[arm_name] = batch

                        # Check for alerts
                        new_alerts = self.check_alerts(arm_name, batch)
                        self.alerts.extend(new_alerts)

                        # Log to rerun if enabled
                        if self.rerun_logger:
                            self.rerun_logger.log_batch(arm_name, batch)

                    except Exception as e:
                        print(f"Error reading {arm_name}: {e}")

                # Output based on mode
                if self.config.json_output:
                    output = {
                        "timestamp": time.time(),
                        "loop": self.loop_count,
                        "arms": {
                            name: batch.to_dict()
                            for name, batch in data.items()
                        }
                    }
                    print(json.dumps(output))
                else:
                    self.render_tui(data)

                # Maintain timing
                elapsed = time.time() - loop_start
                sleep_time = (self.config.interval_ms / 1000.0) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping monitor...")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Feetech STS3215 motor smart data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  monitor-motor-smart-data              # Monitor both arms with TUI
  monitor-motor-smart-data --rerun      # Enable Rerun visualization
  monitor-motor-smart-data --arm left   # Monitor only left arm
  monitor-motor-smart-data --json       # JSON output for logging
  monitor-motor-smart-data -i 50        # 50ms update interval
        """
    )

    parser.add_argument(
        "--left-port", "-l",
        default="/dev/manipulator_left",
        help="Left arm serial port (default: /dev/manipulator_left)"
    )
    parser.add_argument(
        "--right-port", "-r",
        default="/dev/manipulator_right",
        help="Right arm serial port (default: /dev/manipulator_right)"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=1_000_000,
        help="Serial baudrate (default: 1000000)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=100,
        help="Update interval in ms (default: 100)"
    )
    parser.add_argument(
        "--arm", "-a",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to monitor (default: both)"
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Enable Rerun visualization with graphs"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON (for logging/piping)"
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Disable alert checking and display"
    )

    args = parser.parse_args()

    config = MonitorConfig(
        left_port=args.left_port,
        right_port=args.right_port,
        baudrate=args.baudrate,
        interval_ms=args.interval,
        arm=args.arm,
        use_rerun=args.rerun,
        json_output=args.json,
        show_alerts=not args.no_alerts,
    )

    monitor = MotorMonitor(config)

    try:
        if not monitor.connect():
            sys.exit(1)
        monitor.run()
    finally:
        monitor.disconnect()


if __name__ == "__main__":
    main()
