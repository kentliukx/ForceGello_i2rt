import csv
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import portal
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.utils import RateRecorder, override_log_level

DEFAULT_ROBOT_PORT = 11333
_MAX_DOFS = 16
"""Upper bound on robot DOF. The shared pos array is sized to this; the active
slice is exposed through an int Value the worker fills after constructing the robot."""

_WORKER_LOOP_PERIOD_S = 0.002
"""Maximum follower command-queue wait and viewer-only polling interval.
The leader control worker waits for fresh CAN-backed joint state instead."""

_LEADER_IO_PERIOD_S = 0.002
"""Check the latest leader command and follower feedback about every 2 ms."""

_FEEDBACK_TIMEOUT_S = 0.2
_MAX_FEEDBACK_TORQUES_NM = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
_FOLLOWER_COMMAND_TIMEOUT_S = 0.5
_LATENCY_TRACE_EVERY = 10
_LATENCY_REPORT_PERIOD_S = 5.0
_LATENCY_FIELDS = (
    "trace_id",
    "joint",
    "leader_can_rx_ns",
    "leader_read_ns",
    "leader_queue_ns",
    "leader_io_ns",
    "follower_rpc_ns",
    "follower_worker_ns",
    "robot_update_ns",
    "driver_accept_ns",
    "follower_can_tx_ns",
)


def _feedback_source(
    mode: Literal["angle", "torque"],
    leader_pos: np.ndarray,
    follower_pos: np.ndarray,
    follower_eff: np.ndarray,
    leader_kp: np.ndarray,
) -> np.ndarray:
    """Return the torque-like signal to reflect from one of the two feedback modes."""
    if mode == "angle":
        return leader_kp * (leader_pos - follower_pos)
    if mode == "torque":
        return follower_eff
    raise ValueError(f"unsupported feedback mode: {mode}")


def _leader_feedback_torque(
    feedback_signal: np.ndarray,
    follower_vel: np.ndarray,
    angle_error: np.ndarray,
    bilateral_kp: float,
    bilateral_alpha: float,
    bilateral_beta: float,
) -> np.ndarray:
    """Apply speed attenuation, angle-error correction, and torque caps."""
    with np.errstate(over="ignore"):
        speed_attenuation = 1.0 + bilateral_alpha * np.abs(follower_vel)
        angle_correction = (2.0 / np.pi) * np.arctan(bilateral_beta * np.abs(angle_error))
        reflected_eff = -bilateral_kp * (feedback_signal * angle_correction / speed_attenuation)
    return np.clip(reflected_eff, -_MAX_FEEDBACK_TORQUES_NM, _MAX_FEEDBACK_TORQUES_NM)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class ClientRobot(Robot):
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, Any]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._client.get_observations().result()

    def close(self) -> None:
        """Tear down the underlying portal client (background loop thread + socket)."""
        self._client.close()

    def __enter__(self) -> "ClientRobot":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class YAMLeaderRobot:
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        qpos, can_rx_ns = self._robot.get_joint_pos_and_can_rx_ns()
        read_ns = time.monotonic_ns()
        return self._format_info(qpos, can_rx_ns, read_ns)

    def wait_for_info(
        self, last_generation: int, timeout_s: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
        """Wait for a new arm state before issuing another follower target."""
        snapshot = self._robot.wait_for_joint_state_and_can_rx_ns(last_generation, timeout_s)
        if snapshot is None:
            return None
        qpos, qvel, can_rx_ns, generation = snapshot
        qpos_with_gripper, button, can_rx_ns, read_ns = self._format_info(qpos, can_rx_ns, time.monotonic_ns())
        return qpos_with_gripper, np.concatenate([qvel, [0.0]]), button, can_rx_ns, read_ns, generation

    def _format_info(
        self, qpos: np.ndarray, can_rx_ns: np.ndarray, read_ns: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        if encoder_obs is None:
            # Populated only after the first CAN read in the background driver thread.
            raise RuntimeError("CAN device states not ready yet")
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs, can_rx_ns, read_ns

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def command_joint_torques(self, torques: np.ndarray) -> None:
        self._robot.command_joint_torques(torques)

    def enter_gravity_comp_idle(self) -> None:
        self._robot.enter_gravity_comp_idle()


# ---------------------------------------------------------------------------
# Args + Resources
# ---------------------------------------------------------------------------


@dataclass
class Args:
    arm: Literal["yam", "yam_pro", "yam_ultra", "yam_ultra_2", "big_yam", "no_arm"] = "yam"
    gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "flexible_4310", "yam_teaching_handle", "no_gripper"
    ] = "yam_teaching_handle"
    mode: Literal["follower", "leader", "visualizer_local", "visualizer_remote"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"
    bilateral_kp: float = 0.0
    bilateral_alpha: float = 1.0
    bilateral_beta: float = 1.0
    feedback_mode: Literal["angle", "torque"] = "angle"
    trace_latency: bool = False
    record_j3: bool = False
    sim: bool = False
    """Use SimRobot instead of real hardware."""
    ee_mass: Optional[float] = None
    """Override end-effector (link_6) mass in kg for gravity compensation. Defaults to the value in the XML."""


@dataclass
class Resources:
    """Resources accumulated during mode setup; ``cleanup`` drains them on exit.

    Mirrors the ``server_processes`` accumulator pattern from xdof/envs/launch.py:
    every spawn helper appends here so a single ``cleanup(resources)`` call tears
    down the mode regardless of how it exits."""

    processes: List[portal.Process] = field(default_factory=list)
    pos_shared: Optional[portal.SharedArray] = None
    vel_shared: Optional[portal.SharedArray] = None
    feedback_shared: Optional[portal.SharedArray] = None
    pid_shared: Optional[portal.SharedArray] = None
    feedback_time: Optional[Any] = None
    stop_event: Optional[Any] = None


def _spawn_into(processes: List[portal.Process], proc: portal.Process) -> portal.Process:
    """Start a portal.Process and append it to the cleanup pool."""
    proc.start()
    processes.append(proc)
    return proc


def _log_latency_summary(records: List[Dict[str, int]], channel: str) -> None:
    """Summarize traced samples; raw stage timestamps remain in the CSV."""
    for joint in sorted({record["joint"] for record in records}):
        totals_ms = np.array(
            [
                (record["follower_can_tx_ns"] - record["leader_can_rx_ns"]) / 1e6
                for record in records
                if record["joint"] == joint
            ]
        )
        p50, p95 = np.percentile(totals_ms, [50, 95])
        logging.info(
            "[CAN latency %s joint%d] n=%d CAN RX to TX p50=%.2f ms p95=%.2f ms max=%.2f ms",
            channel,
            joint,
            len(totals_ms),
            p50,
            p95,
            np.max(totals_ms),
        )


# ---------------------------------------------------------------------------
# Background workers. Each runs in its own portal.Process so the foreground
# loop (web IO server, web IO client, or MuJoCo viewer) never blocks on
# hardware or RPC.
# ---------------------------------------------------------------------------


def _yam_polling_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    vel_shared: Optional["portal.SharedArray"],
    feedback_shared: Optional["portal.SharedArray"],
    pid_shared: Optional["portal.SharedArray"],
    feedback_time: Any,
    n_dofs_value: Any,
    cmd_queue: Any,
    stop_event: Any,
    rate_name: str,
) -> None:
    """Owns the YAM hardware. Streams joint_pos into ``pos_shared`` at hardware
    rate; if ``cmd_queue`` is provided, drains pending commands onto the YAM.
    Publishes the actual DOF count via ``n_dofs_value`` once the YAM is up."""
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    yam = get_yam_robot(
        channel=args.can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        ee_mass=args.ee_mass,
        sim=args.sim,
    )
    if args.trace_latency and isinstance(yam, MotorChainRobot):
        yam.motor_chain.motor_interface.trace_timestamps_enabled = True
    n_dofs_value.value = yam.num_dofs()
    rate = RateRecorder(name=rate_name, report_interval=1.0)
    rate.start()
    last_command_time = 0.0
    following = False
    latency_file = None
    latency_writer = None
    if args.trace_latency and cmd_queue is not None and isinstance(yam, MotorChainRobot):
        latency_path = Path("/tmp") / f"i2rt_can_latency_{args.can_channel}_{os.getpid()}.csv"
        latency_file = latency_path.open("w", newline="")
        latency_writer = csv.DictWriter(latency_file, fieldnames=_LATENCY_FIELDS)
        latency_writer.writeheader()
        logging.info("CAN latency trace: %s", latency_path)
    next_latency_report = time.monotonic() + _LATENCY_REPORT_PERIOD_S
    while not stop_event.is_set():
        if cmd_queue is not None:
            # The queue wakes this worker as soon as a command arrives. A short timeout
            # still lets us publish feedback and check command/stop deadlines regularly.
            try:
                cmd = cmd_queue.get(timeout=_WORKER_LOOP_PERIOD_S)
            except queue.Empty:
                cmd = None
            # Commands are latest-wins: discard any backlog accumulated during the
            # previous hardware/feedback update.
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break
            if cmd is not None:
                if isinstance(cmd, dict):
                    state = {"pos": np.asarray(cmd["pos"]), "vel": np.asarray(cmd["vel"])}
                    if "trace" in cmd:
                        state["trace"] = dict(cmd["trace"], follower_worker_ns=time.monotonic_ns())
                    yam.command_joint_state(state)
                else:
                    yam.command_joint_pos(cmd)
                last_command_time = time.monotonic()
                following = True
            elif following and time.monotonic() - last_command_time > _FOLLOWER_COMMAND_TIMEOUT_S:
                if isinstance(yam, MotorChainRobot):
                    yam.enter_gravity_comp_idle()
                following = False
        try:
            if feedback_shared is not None:
                if isinstance(yam, MotorChainRobot):
                    feedback_obs = yam.get_force_feedback_observations()
                    pos = feedback_obs["joint_pos"]
                    vel = feedback_obs["joint_vel"]
                    external_eff = feedback_obs["joint_eff_external"]
                    pd_eff = feedback_obs["joint_eff_pd_est"]
                else:
                    obs = yam.get_observations()
                    pos = yam.get_joint_pos()
                    vel = obs["joint_vel"]
                    external_eff = obs["joint_eff"] - yam.get_motor_torques()
                    pd_eff = np.zeros_like(external_eff)
                feedback_shared.array[: len(external_eff)] = external_eff
                if pid_shared is not None:
                    pid_shared.array[: len(pd_eff)] = pd_eff
                vel_shared.array[: len(vel)] = vel
            else:
                pos = yam.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            if feedback_shared is not None:
                feedback_time.value = time.monotonic()
            rate.track()
        except Exception as e:
            logging.error(f"[{rate_name}] error: {e}")
            time.sleep(0.1)
        if latency_writer is not None and time.monotonic() >= next_latency_report:
            records = yam.motor_chain.drain_latency_records()
            if records:
                latency_writer.writerows(records)
                latency_file.flush()
                _log_latency_summary(records, args.can_channel)
            next_latency_report = time.monotonic() + _LATENCY_REPORT_PERIOD_S
        if cmd_queue is None:
            time.sleep(_WORKER_LOOP_PERIOD_S)  # viewer-only worker has no command queue to wait on

    if latency_writer is not None:
        records = yam.motor_chain.drain_latency_records()
        latency_writer.writerows(records)
        latency_file.close()
    if isinstance(yam, MotorChainRobot):
        yam.enter_gravity_comp_idle()
        time.sleep(0.05)
        yam.close()


def _rpc_polling_worker(
    server_port: int,
    server_host: str,
    pos_shared: "portal.SharedArray",
    stop_event: Any,
    rate_name: str,
) -> None:
    """Polls a remote follower over portal RPC and caches the latest joint_pos
    in ``pos_shared`` so the foreground viewer reads from memory."""
    override_log_level()
    client = ClientRobot(server_port, host=server_host)
    rate = RateRecorder(name=rate_name, report_interval=1.0)
    rate.start()
    while not stop_event.is_set():
        try:
            pos = client.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            rate.track()
        except Exception as e:
            logging.error(f"[{rate_name}] error: {e}")
            time.sleep(0.1)


def _leader_control_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    vel_shared: "portal.SharedArray",
    feedback_shared: "portal.SharedArray",
    pid_shared: "portal.SharedArray",
    feedback_time: Any,
    cmd_queue: Any,
    stop_event: Any,
    j3_csv_path: Optional[Path],
) -> None:
    """Owns the leader YAM hardware and runs the bilateral control
    loop. Reads the latest follower position and effort from shared memory
    (written by main's foreground web-IO loop) and pushes commands for the follower into
    ``cmd_queue`` (drained by main and forwarded over RPC). Never touches the
    network itself."""
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    yam = get_yam_robot(
        channel=args.can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        ee_mass=args.ee_mass,
        sim=args.sim,
    )
    if args.trace_latency:
        yam.motor_chain.motor_interface.trace_timestamps_enabled = True
    robot = YAMLeaderRobot(yam)
    leader_kp = np.asarray(yam.get_robot_info()["kp"][:6], dtype=float).copy()

    control_rate = RateRecorder(name="yam-leader control", report_interval=1.0)
    control_rate.start()
    j3_file = None
    j3_writer = None
    if j3_csv_path is not None:
        try:
            j3_file = j3_csv_path.open("w", newline="", buffering=1)
            j3_writer = csv.writer(j3_file)
            j3_writer.writerow(
                ("monotonic_ns", "leader_angle_rad", "follower_angle_rad", "follower_pd_est_nm", "leader_feedback_nm")
            )
            logging.info("J3 telemetry: %s", j3_csv_path)
        except OSError:
            logging.exception("Could not start J3 telemetry recording")
            if j3_file is not None:
                j3_file.close()
            j3_file = None
            j3_writer = None

    # The encoder device states are populated only after the first CAN read, so get_info() can
    # raise on startup. Retry until the first reading lands before entering the control loop.
    while not stop_event.is_set():
        try:
            initial_info = robot.wait_for_info(-1, 0.0)
            if initial_info is None:
                time.sleep(0.01)
                continue
            (
                current_joint_pos,
                current_joint_vel,
                current_button,
                current_rx_ns,
                leader_read_ns,
                last_generation,
            ) = initial_info
            break
        except RuntimeError:
            time.sleep(0.05)
    else:
        return  # stop requested before the encoder became ready
    current_follower_joint_pos = pos_shared.array.copy()
    logging.info(f"Current leader joint pos: {current_joint_pos}")
    logging.info(f"Current follower joint pos: {current_follower_joint_pos}")

    def slow_move(joint_pos: np.ndarray, duration: float = 3.0) -> None:
        start_pos = pos_shared.array.copy()
        steps = 100
        dt = duration / steps
        for i in range(steps):
            if stop_event.is_set():
                return
            cmd = joint_pos * i / steps + start_pos * (1 - i / steps)
            _put_latest_command(cmd_queue, cmd)
            time.sleep(dt)

    synchronized = False
    command_sequence = 0
    while not stop_event.is_set():
        info = robot.wait_for_info(last_generation, timeout_s=0.02)
        if info is None:
            if synchronized and time.monotonic() - feedback_time.value > _FEEDBACK_TIMEOUT_S:
                robot.enter_gravity_comp_idle()
            continue
        current_joint_pos, current_joint_vel, current_button, current_rx_ns, leader_read_ns, last_generation = info
        control_rate.track()
        if current_button[0] > 0.5:
            if not synchronized:
                slow_move(current_joint_pos)
                if stop_event.is_set():
                    break
            else:
                logging.info("clear bilateral feedback")
                robot.enter_gravity_comp_idle()
            synchronized = not synchronized
            while current_button[0] > 0.5 and not stop_event.is_set():
                info = robot.wait_for_info(last_generation, timeout_s=0.03)
                if info is not None:
                    (
                        current_joint_pos,
                        current_joint_vel,
                        current_button,
                        current_rx_ns,
                        leader_read_ns,
                        last_generation,
                    ) = info
            # Wait for a fresh arm sample after the button is released so the
            # first velocity target cannot come from before the alignment move.
            continue

        if synchronized:
            if not np.all(np.isfinite(current_joint_pos)) or not np.all(np.isfinite(current_joint_vel)):
                robot.enter_gravity_comp_idle()
                continue
            command_sequence += 1
            if args.trace_latency and command_sequence % _LATENCY_TRACE_EVERY == 0:
                trace = {
                    "trace_id": command_sequence,
                    "leader_can_rx_ns": current_rx_ns[:6].copy(),
                    "leader_read_ns": leader_read_ns,
                    "leader_queue_ns": time.monotonic_ns(),
                }
                _put_latest_command(
                    cmd_queue, {"pos": current_joint_pos.copy(), "vel": current_joint_vel.copy(), "trace": trace}
                )
            else:
                _put_latest_command(cmd_queue, {"pos": current_joint_pos.copy(), "vel": current_joint_vel.copy()})
            now = time.monotonic()
            feedback_fresh = now - feedback_time.value <= _FEEDBACK_TIMEOUT_S
            follower_pos = pos_shared.array[:6].copy()
            follower_pd = pid_shared.array[:6].copy() if j3_writer is not None else None
            record_ns = time.monotonic_ns() if j3_writer is not None else 0
            leader_eff = np.zeros(6)
            if feedback_fresh:
                external_eff = feedback_shared.array[:6].copy()
                follower_vel = vel_shared.array[:6].copy()
                if (
                    np.all(np.isfinite(current_joint_pos[:6]))
                    and np.all(np.isfinite(follower_pos))
                    and np.all(np.isfinite(external_eff))
                    and np.all(np.isfinite(follower_vel))
                ):
                    feedback_source = _feedback_source(
                        args.feedback_mode, current_joint_pos[:6], follower_pos, external_eff, leader_kp
                    )
                    # Positive source means the leader is ahead or the follower pushes forward.
                    leader_eff = _leader_feedback_torque(
                        feedback_source,
                        follower_vel,
                        current_joint_pos[:6] - follower_pos,
                        args.bilateral_kp,
                        args.bilateral_alpha,
                        args.bilateral_beta,
                    )
                    robot.command_joint_torques(leader_eff)
                else:
                    robot.enter_gravity_comp_idle()
            else:
                robot.enter_gravity_comp_idle()

            if j3_writer is not None:
                follower_angle = float(follower_pos[2]) if feedback_fresh else float("nan")
                follower_pd_nm = float(follower_pd[2]) if feedback_fresh else float("nan")
                try:
                    j3_writer.writerow(
                        (
                            record_ns,
                            float(current_joint_pos[2]),
                            follower_angle,
                            follower_pd_nm,
                            float(leader_eff[2]),
                        )
                    )
                except OSError:
                    logging.exception("J3 telemetry recording stopped")
                    j3_file.close()
                    j3_writer = None

    robot.enter_gravity_comp_idle()
    if j3_file is not None and not j3_file.closed:
        j3_file.close()
    time.sleep(0.05)
    yam.close()


# ---------------------------------------------------------------------------
# Foreground helpers and loops
# ---------------------------------------------------------------------------


def _wait_for_dof(n_dofs_value: Any, timeout_s: float = 30.0) -> int:
    """Block until the background worker publishes the active DOF count."""
    deadline = time.time() + timeout_s
    while n_dofs_value.value == 0:
        if time.time() > deadline:
            raise TimeoutError("worker did not publish DOF count within timeout")
        time.sleep(0.05)
    return int(n_dofs_value.value)


def _put_latest_command(cmd_queue: Any, item: Any) -> None:
    """Non-blocking latest-wins enqueue onto a (bounded) command queue.

    Never blocks the caller: if the queue is full (the hardware worker has not drained it
    yet), drop the stale item and replace it with the newest, so the follower always actuates
    the freshest target. A setpoint stream is latest-wins, not a work queue."""
    while True:
        try:
            cmd_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                cmd_queue.get_nowait()
            except queue.Empty:
                pass


def _build_follower_server(
    server_port: int,
    n_dofs_value: Any,
    pos_shared: "portal.SharedArray",
    vel_shared: "portal.SharedArray",
    feedback_shared: "portal.SharedArray",
    pid_shared: "portal.SharedArray",
    feedback_time: Any,
    cmd_queue: Any,
) -> portal.Server:
    """Build the portal.Server that exposes the follower YAM over RPC. Returned
    unstarted; caller does ``server.start()`` to block."""
    io_rate = RateRecorder(name=f"ServerRobot[{server_port}] io", report_interval=1.0)
    io_rate.start()
    io_lock = threading.Lock()

    def _tracked(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            with io_lock:
                io_rate.track()
            return fn(*a, **kw)

        return wrapper

    def _get_pos() -> np.ndarray:
        return pos_shared.array[: n_dofs_value.value].copy()

    def _cmd_pos(joint_pos: np.ndarray) -> None:
        _put_latest_command(cmd_queue, np.asarray(joint_pos))

    def _cmd_state(state: Dict[str, Any]) -> None:
        key = "position" if "position" in state else "pos"
        pos = np.asarray(state[key])
        vel = np.asarray(state.get("vel", np.zeros_like(pos)))
        if pos.shape != vel.shape or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            raise ValueError("joint position and velocity targets must have matching finite values")
        packet = {"pos": pos, "vel": vel}
        if "trace" in state:
            packet["trace"] = dict(state["trace"], follower_rpc_ns=time.monotonic_ns())
        _put_latest_command(cmd_queue, packet)

    def _get_obs() -> Dict[str, np.ndarray]:
        return {
            "joint_pos": _get_pos(),
            "joint_vel": vel_shared.array[: n_dofs_value.value].copy(),
            "joint_eff_external": feedback_shared.array[: n_dofs_value.value].copy(),
            "joint_eff_pd_est": pid_shared.array[: n_dofs_value.value].copy(),
            "sample_age_s": np.array(max(0.0, time.monotonic() - feedback_time.value)),
        }

    server = portal.Server(server_port)
    logging.info(f"Robot Server Binding to {server_port}")
    server.bind("num_dofs", _tracked(lambda: n_dofs_value.value))
    server.bind("get_joint_pos", _tracked(_get_pos))
    server.bind("command_joint_pos", _tracked(_cmd_pos))
    server.bind("command_joint_state", _tracked(_cmd_state))
    server.bind("get_observations", _tracked(_get_obs))
    return server


def _run_leader_io_loop(
    client_robot: ClientRobot,
    cmd_queue: Any,
    pos_shared: "portal.SharedArray",
    vel_shared: "portal.SharedArray",
    feedback_shared: "portal.SharedArray",
    pid_shared: "portal.SharedArray",
    feedback_time: Any,
    stop_event: Any,
    io_rate: RateRecorder,
) -> None:
    """Foreground loop for leader mode: drain commands from the bilateral
    control worker onto the follower over RPC, and pull the follower's latest
    joint state into shared memory so the worker can read it for bilateral feedback."""
    while not stop_event.is_set():
        loop_started = time.monotonic()
        # Drain to the newest command and drop any stale backlog: the control worker can enqueue
        # faster than each blocking RPC round-trip drains, so replaying every queued command would
        # make the follower lag further and further behind.
        cmd = None
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break
        try:
            if cmd is not None:
                if isinstance(cmd, dict):
                    state = {"pos": cmd["pos"], "vel": cmd["vel"]}
                    if "trace" in cmd:
                        state["trace"] = dict(cmd["trace"], leader_io_ns=time.monotonic_ns())
                    client_robot.command_joint_state(state)
                else:
                    client_robot.command_joint_pos(cmd)
            request_started = time.monotonic()
            obs = client_robot.get_observations()
            request_elapsed = time.monotonic() - request_started
            pos = np.asarray(obs["joint_pos"])
            vel = np.asarray(obs["joint_vel"])
            external_eff = np.asarray(obs["joint_eff_external"])
            pd_eff = np.asarray(obs.get("joint_eff_pd_est", np.full_like(external_eff, np.nan)))
            sample_age = float(obs["sample_age_s"])
            if not np.all(np.isfinite(pos)):
                raise ValueError("follower returned non-finite joint positions")
            pos_shared.array[: len(pos)] = pos
            if (
                sample_age + request_elapsed > _FEEDBACK_TIMEOUT_S
                or not np.all(np.isfinite(external_eff))
                or not np.all(np.isfinite(vel))
            ):
                feedback_time.value = 0.0
            else:
                feedback_shared.array[: len(external_eff)] = external_eff
                pid_shared.array[: len(pd_eff)] = pd_eff
                vel_shared.array[: len(vel)] = vel
                feedback_time.value = time.monotonic()
            io_rate.track()
        except Exception as e:
            # Ride out a transient follower/network blip. Without this the exception escapes to
            # run_leader, whose finally: cleanup() calls proc.kill() on the control worker while the
            # leader arm is energized — an abrupt loss of control of a powered arm.
            logging.error(f"[yam-leader web-port io] error: {e}")
            feedback_time.value = 0.0
            time.sleep(0.1)
        time.sleep(max(0.0, _LEADER_IO_PERIOD_S - (time.monotonic() - loop_started)))


def _run_viewer_loop(
    arm_type: ArmType,
    gripper_type: GripperType,
    get_pos_fn: Callable[[], np.ndarray],
) -> None:
    """Foreground MuJoCo viewer. Pulls joint_pos from ``get_pos_fn`` each frame
    — typically a snapshot of a SharedArray populated by a background worker."""
    xml_path = combine_arm_and_gripper_xml(arm_type, gripper_type)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt: float = 0.01
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            joint_pos = get_pos_fn()
            nq = model.nq
            n = min(len(joint_pos), nq)
            data.qpos[:n] = joint_pos[:n]

            for j in range(model.njnt):
                adr = model.jnt_qposadr[j]
                if adr >= n:
                    continue
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                    lo, hi = model.jnt_range[j]
                    data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)

            for i in range(model.neq):
                if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                adr1 = model.jnt_qposadr[model.eq_obj1id[i]]
                adr2 = model.jnt_qposadr[model.eq_obj2id[i]]
                coef = model.eq_data[i, :5]
                data.qpos[adr2] = np.polyval(coef[::-1], data.qpos[adr1])

            mujoco.mj_forward(model, data)
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(resources: Resources) -> None:
    """Stop workers, allow hardware commands to return to idle, then release shared memory."""
    logging.info("exiting")
    if resources.stop_event is not None:
        resources.stop_event.set()
    for proc in resources.processes:
        proc.join(timeout=2.0)
        if proc.running:
            proc.kill()
    if resources.pos_shared is not None:
        resources.pos_shared.close()
    if resources.vel_shared is not None:
        resources.vel_shared.close()
    if resources.feedback_shared is not None:
        resources.feedback_shared.close()
    if resources.pid_shared is not None:
        resources.pid_shared.close()
    logging.info("Processes terminated.")


# ---------------------------------------------------------------------------
# Per-mode runners. Each owns its own setup → foreground loop → cleanup, the
# same shape as ``main`` in xdof/envs/launch.py: build dependencies, call the
# control loop, wrap in try/finally with a single ``cleanup``.
# ---------------------------------------------------------------------------


def run_follower(args: Args) -> None:
    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        vel_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        feedback_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        pid_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        feedback_time=portal.mp.Value("d", 0.0),
        stop_event=portal.mp.Event(),
    )
    spawn = partial(_spawn_into, resources.processes)

    n_dofs_value = portal.mp.Value("i", 0)
    # maxsize=1 so the command queue holds only the latest setpoint; _cmd_pos/_cmd_state drop
    # stale commands on overflow (see _build_follower_server). A setpoint stream is latest-wins,
    # not a work queue — bounding it prevents any backlog from accumulating on this hop.
    cmd_queue = portal.mp.Queue(maxsize=1)

    spawn(
        portal.Process(
            _yam_polling_worker,
            args,
            resources.pos_shared,
            resources.vel_shared,
            resources.feedback_shared,
            resources.pid_shared,
            resources.feedback_time,
            n_dofs_value,
            cmd_queue,
            resources.stop_event,
            "follower yam-hardware",
            name="follower-yam-control",
        )
    )
    n_dofs = _wait_for_dof(n_dofs_value)
    logging.info(f"Follower YAM ready, num_dofs={n_dofs}")

    server = _build_follower_server(
        args.server_port,
        n_dofs_value,
        resources.pos_shared,
        resources.vel_shared,
        resources.feedback_shared,
        resources.pid_shared,
        resources.feedback_time,
        cmd_queue,
    )

    try:
        server.start()  # blocks
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("follower mode failed")
        raise
    finally:
        cleanup(resources)


def run_leader(args: Args) -> None:
    client_robot = ClientRobot(args.server_port, host=args.server_host)
    initial_follower_obs = client_robot.get_observations()
    initial_follower_pos = initial_follower_obs["joint_pos"]
    logging.info(f"Initial follower joint pos: {initial_follower_pos}")

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(len(initial_follower_pos),), dtype=np.float64),
        vel_shared=portal.SharedArray(shape=(len(initial_follower_pos),), dtype=np.float64),
        feedback_shared=portal.SharedArray(shape=(len(initial_follower_pos),), dtype=np.float64),
        pid_shared=portal.SharedArray(shape=(len(initial_follower_pos),), dtype=np.float64),
        feedback_time=portal.mp.Value("d", 0.0),
        stop_event=portal.mp.Event(),
    )
    resources.pos_shared.array[:] = initial_follower_pos
    cmd_queue = portal.mp.Queue(maxsize=1)
    j3_csv_path = Path("/tmp") / f"i2rt_j3_{args.can_channel}_{os.getpid()}.csv" if args.record_j3 else None

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _leader_control_worker,
            args,
            resources.pos_shared,
            resources.vel_shared,
            resources.feedback_shared,
            resources.pid_shared,
            resources.feedback_time,
            cmd_queue,
            resources.stop_event,
            j3_csv_path,
            name="yam-leader-control",
        )
    )

    io_rate = RateRecorder(name="yam-leader web-port io", report_interval=1.0)
    io_rate.start()
    try:
        _run_leader_io_loop(
            client_robot,
            cmd_queue,
            resources.pos_shared,
            resources.vel_shared,
            resources.feedback_shared,
            resources.pid_shared,
            resources.feedback_time,
            resources.stop_event,
            io_rate,
        )
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("leader mode failed")
        raise
    finally:
        cleanup(resources)
        client_robot.close()
        if j3_csv_path is not None and j3_csv_path.exists():
            plot_script = Path(__file__).resolve().parents[1] / "bimanual_lead_follower" / "plot_j3.py"
            try:
                subprocess.run(
                    [sys.executable, str(plot_script), "--csv-path", str(j3_csv_path)], check=True, timeout=15
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logging.exception("Could not plot J3 telemetry from %s", j3_csv_path)


def run_visualizer_local(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    n_dofs_value = portal.mp.Value("i", 0)

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _yam_polling_worker,
            args,
            resources.pos_shared,
            None,
            None,
            None,
            None,
            n_dofs_value,
            None,  # no cmd_queue — viewer doesn't send commands
            resources.stop_event,
            "viz-local yam-read",
            name="viz-local-yam",
        )
    )
    n_dofs = _wait_for_dof(n_dofs_value)
    logging.info(f"Viz YAM ready, num_dofs={n_dofs}")

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        _run_viewer_loop(arm_type, gripper_type, get_pos)
    except Exception:
        logging.exception("visualizer_local mode failed")
        raise
    finally:
        cleanup(resources)


def run_visualizer_remote(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    with ClientRobot(args.server_port, host=args.server_host) as bootstrap:
        n_dofs = bootstrap.num_dofs()
    logging.info(f"Remote follower num_dofs={n_dofs}")

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _rpc_polling_worker,
            args.server_port,
            args.server_host,
            resources.pos_shared,
            resources.stop_event,
            "viz-remote rpc-poll",
            name="viz-remote-rpc",
        )
    )

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        _run_viewer_loop(arm_type, gripper_type, get_pos)
    except Exception:
        logging.exception("visualizer_remote mode failed")
        raise
    finally:
        cleanup(resources)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    # Make RateRecorder's logging.info output (and our own logging calls) visible.
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)

    if arm_type == ArmType.NO_ARM:
        # Every mode here needs an arm model (real YAM for follower/leader, an XML for the
        # visualizers); NO_ARM has no XML path and otherwise crashes deep inside the utils.
        raise ValueError("--arm 'no_arm' is not supported by minimum_gello; all modes require an arm")

    if args.mode == "leader" and args.sim:
        raise ValueError("Leader mode requires real hardware (--sim is not supported)")

    if args.mode == "leader" and (not math.isfinite(args.bilateral_kp) or args.bilateral_kp < 0.0):
        raise ValueError("--bilateral-kp must be a finite non-negative number")

    if args.mode == "leader" and (not math.isfinite(args.bilateral_alpha) or args.bilateral_alpha < 0.0):
        raise ValueError("--bilateral-alpha must be a finite non-negative number")

    if args.mode == "leader" and (not math.isfinite(args.bilateral_beta) or args.bilateral_beta < 0.0):
        raise ValueError("--bilateral-beta must be a finite non-negative number")

    if args.mode == "leader" and args.feedback_mode not in ("angle", "torque"):
        raise ValueError("--feedback-mode must be 'angle' or 'torque'")

    runners: Dict[str, Callable[[Args], None]] = {
        "follower": run_follower,
        "leader": run_leader,
        "visualizer_local": run_visualizer_local,
        "visualizer_remote": run_visualizer_remote,
    }
    runner = runners.get(args.mode)
    if runner is None:
        raise ValueError(f"Invalid mode: {args.mode}")
    runner(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
