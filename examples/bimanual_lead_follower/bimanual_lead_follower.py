#!/usr/bin/env python3

import math
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from types import FrameType
from typing import Literal

import tyro

current_file_path = os.path.dirname(os.path.abspath(__file__))


def check_can_interface(interface: str) -> bool:
    """Check if a CAN interface exists and is available"""
    try:
        # Check if interface exists in network interfaces
        result = subprocess.run(["ip", "link", "show", interface], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return False

        # Check if interface is UP
        if "state UP" in result.stdout or "state UNKNOWN" in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False

    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def check_all_can_interfaces() -> bool:
    """Check if all required CAN interfaces exist"""
    required_interfaces = ["can_follower_r", "can_leader_r", "can_follower_l", "can_leader_l"]

    missing_interfaces = []

    for interface in required_interfaces:
        if not check_can_interface(interface):
            missing_interfaces.append(interface)

    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")

    print("✓ All CAN interfaces are available")
    return True


def launch_gello_process(
    can_channel: str,
    gripper: str,
    mode: str | None = None,
    server_port: int | None = None,
    bilateral_kp: float = 0.0,
    bilateral_alpha: float = 0.0,
    feedback_mode: Literal["angle", "torque"] = "angle",
    bilateral_beta: float = 1.0,
    trace_latency: bool = False,
    record_j3: bool = False,
) -> "subprocess.Popen[bytes] | None":
    """Launch a single gello process with given parameters"""
    python_path = "python"
    script_path = os.path.join(current_file_path, "..", "minimum_gello", "minimum_gello.py")

    cmd = [python_path, os.path.expanduser(script_path), "--can_channel", can_channel, "--gripper", gripper]

    if mode:
        cmd.extend(["--mode", mode])

    if server_port:
        cmd.extend(["--server_port", str(server_port)])

    if mode == "leader":
        cmd.extend(
            [
                "--bilateral-kp",
                str(bilateral_kp),
                "--bilateral-alpha",
                str(bilateral_alpha),
                "--bilateral-beta",
                str(bilateral_beta),
                "--feedback-mode",
                feedback_mode,
            ]
        )
    if trace_latency:
        cmd.append("--trace-latency")
    if record_j3:
        cmd.append("--record-j3")

    print(f"Starting: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd, start_new_session=True)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


def _stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    """Stop every arm before waiting, then remove any surviving worker processes."""
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    previous_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        for process in processes:
            if process.poll() is None:
                print(f"Stopping process {process.pid}...")
                with suppress(ProcessLookupError):
                    process.send_signal(signal.SIGINT)

        deadline = time.monotonic() + 5.0
        for process in processes:
            if process.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))

        # Each minimum_gello parent owns a new session. Clean up portal workers even
        # if their parent exited before it could join them.
        for process in processes:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)

        deadline = time.monotonic() + 2.0
        for process in processes:
            if process.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))

        for process in processes:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            if process.poll() is None:
                process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def _request_shutdown(_signum: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main(
    bilateral_kp: float = 0.0,
    bilateral_alpha: float = 0.0,
    feedback_mode: Literal["angle", "torque"] = "angle",
    bilateral_beta: float = 1.0,
    trace_latency: bool = False,
    record_j3: bool = False,
) -> None:
    """Launch both arm pairs with angle or torque feedback."""
    if not math.isfinite(bilateral_kp) or bilateral_kp < 0.0:
        raise ValueError("--bilateral-kp must be a finite non-negative number")
    if not math.isfinite(bilateral_alpha) or bilateral_alpha < 0.0:
        raise ValueError("--bilateral-alpha must be a finite non-negative number")
    if not math.isfinite(bilateral_beta) or bilateral_beta < 0.0:
        raise ValueError("--bilateral-beta must be a finite non-negative number")
    if feedback_mode not in ("angle", "torque"):
        raise ValueError("--feedback-mode must be 'angle' or 'torque'")

    previous_sigterm = signal.signal(signal.SIGTERM, _request_shutdown)
    processes = []

    try:
        # First check if all CAN interfaces exist
        print("Checking CAN interfaces...")
        check_all_can_interfaces()

        # Define the processes to launch
        process_configs = [
            {
                "can_channel": "can_follower_r",
                "gripper": "linear_4310",
                "server_port": 1234,
                "trace_latency": trace_latency,
            },
            {
                "can_channel": "can_leader_r",
                "gripper": "yam_teaching_handle",
                "mode": "leader",
                "server_port": 1234,
                "bilateral_kp": bilateral_kp,
                "bilateral_alpha": bilateral_alpha,
                "bilateral_beta": bilateral_beta,
                "feedback_mode": feedback_mode,
                "trace_latency": trace_latency,
                "record_j3": record_j3,
            },
            {
                "can_channel": "can_follower_l",
                "gripper": "linear_4310",
                "server_port": 1235,
                "trace_latency": trace_latency,
            },
            {
                "can_channel": "can_leader_l",
                "gripper": "yam_teaching_handle",
                "mode": "leader",
                "server_port": 1235,
                "bilateral_kp": bilateral_kp,
                "bilateral_alpha": bilateral_alpha,
                "bilateral_beta": bilateral_beta,
                "feedback_mode": feedback_mode,
                "trace_latency": trace_latency,
            },
        ]

        # Launch all processes
        print("\nLaunching processes...")
        for config in process_configs:
            process = launch_gello_process(**config)
            if process:
                processes.append(process)
                print(f"✓ Started process {process.pid} for {config['can_channel']}")
            else:
                raise RuntimeError(f"Failed to start process for {config['can_channel']}")

        print(f"\n✓ Successfully launched {len(processes)} processes")
        print("Press Ctrl+C to stop all processes")

        # Wait for processes and handle termination
        try:
            while True:
                for process in processes:
                    exit_code = process.poll()
                    if exit_code is not None:
                        raise RuntimeError(f"Process {process.pid} exited with code {exit_code}; stopping all arms")

                time.sleep(1)

        except KeyboardInterrupt:
            print("\nReceived Ctrl+C, terminating all processes...")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        try:
            _stop_processes(processes)
            print("All processes terminated")
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    tyro.cli(main)
