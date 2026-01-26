#!/usr/bin/env python3
"""
CLI tool for reading Feetech STS3215 motor smart data.

Usage:
    read-motor-smart-data                           # Read both arms
    read-motor-smart-data --arm left                # Read left arm only
    read-motor-smart-data --json                    # JSON output
    python -m lerobot.motors.feetech_smart_data.cli --port /dev/manipulator_left
"""

import argparse
import json
import sys

from .smart_data_reader import FeetechSmartDataReader


def format_table(batch, show_raw=False):
    """Format batch data as a table."""
    lines = []
    header = "ID  | Pos    | Vel   | Load%  | Volt | Temp | mA    | Mov | Err"
    lines.append(header)
    lines.append("-" * len(header))

    for m in batch:
        load_str = f"{m.load:+6.1f}" if m.load is not None else "  N/A "
        curr_str = f"{m.current:5.1f}" if m.current is not None else " N/A "
        mov_str = "Y" if m.moving else "N"
        err_str = "Y" if m.has_error else "N"

        line = (
            f"{m.motor_id:3d} | {m.position:5d}  | {m.velocity:5d} | {load_str} | "
            f"{m.voltage:4.1f} | {m.temperature:4d} | {curr_str} | "
            f"{mov_str:3s} | {err_str}"
        )
        lines.append(line)

    return "\n".join(lines)


def read_arm(port, name, motor_ids=None, json_output=False):
    """Read and display data from one arm."""
    reader = FeetechSmartDataReader(port, 1_000_000)
    if not reader.connect():
        print(f"Error: Failed to connect to {name} at {port}", file=sys.stderr)
        return False

    try:
        if motor_ids is None:
            motor_ids = reader.scan()
            if not motor_ids:
                print(f"Error: No motors found on {name}", file=sys.stderr)
                return False

        batch = reader.read_all(motor_ids)

        if json_output:
            return batch.to_dict()
        else:
            print(f"\n{name} ({port})")
            print(f"Motors: {motor_ids}")
            print(f"Timestamp: {batch.timestamp:.3f}")
            print()
            print(format_table(batch))
            return True

    finally:
        reader.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Read smart data from Feetech STS3215 motors"
    )
    parser.add_argument(
        "--port", "-p",
        default=None,
        help="Serial port (e.g., /dev/manipulator_left). If not specified, reads based on --arm"
    )
    parser.add_argument(
        "--arm", "-a",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to read (default: both)"
    )
    parser.add_argument(
        "--baudrate", "-b",
        type=int,
        default=1_000_000,
        help="Baudrate (default: 1000000)"
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated motor IDs (default: scan all)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    # Parse motor IDs if specified
    motor_ids = None
    if args.ids:
        motor_ids = [int(x.strip()) for x in args.ids.split(",")]

    # If port is specified, use single-port mode
    if args.port:
        reader = FeetechSmartDataReader(args.port, args.baudrate)
        if not reader.connect():
            print(f"Error: Failed to connect to {args.port}", file=sys.stderr)
            sys.exit(1)

        try:
            if motor_ids is None:
                motor_ids = reader.scan()
                if not motor_ids:
                    print("Error: No motors found", file=sys.stderr)
                    sys.exit(1)

            batch = reader.read_all(motor_ids)

            if args.json:
                print(json.dumps(batch.to_dict(), indent=2))
            else:
                print(f"Port: {args.port}")
                print(f"Motors: {motor_ids}")
                print(f"Timestamp: {batch.timestamp:.3f}")
                print()
                print(format_table(batch))

        finally:
            reader.disconnect()
    else:
        # Multi-arm mode
        results = {}
        success = True

        if args.arm in ("left", "both"):
            result = read_arm(
                "/dev/manipulator_left",
                "LEFT ARM",
                motor_ids,
                args.json
            )
            if args.json and isinstance(result, dict):
                results["left"] = result
            elif not result:
                success = False

        if args.arm in ("right", "both"):
            result = read_arm(
                "/dev/manipulator_right",
                "RIGHT ARM",
                motor_ids,
                args.json
            )
            if args.json and isinstance(result, dict):
                results["right"] = result
            elif not result:
                success = False

        if args.json and results:
            print(json.dumps(results, indent=2))

        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
