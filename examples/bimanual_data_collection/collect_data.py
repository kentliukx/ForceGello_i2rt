"""Dual YAM teleoperation + multi-rate recording; run from the i2rt repository root."""

import heapq
import json
import logging
import math
import multiprocessing as mp
import queue
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

# Allow `python examples/bimanual_data_collection/collect_data.py` as well as `python -m ...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bimanual_data_collection.cameras import camera_worker, list_cameras
from examples.bimanual_data_collection.export_implicitrdp import CartesianProjection
from examples.bimanual_data_collection.recording import (
    ButtonEdges,
    EpisodeWriter,
    PeriodicClock,
    request_shutdown,
    worker_signals,
)


# Fixed lab camera assignment; edit here if a camera is replaced or moved.
CAMERA_DEVICES = {
    "left_wrist": "realsense:135122079769",
    "right_wrist": "realsense:135122075425",
    "external": "realsense:036422060422",
}


@dataclass
class Args:
    output: Path = Path("data/demonstrations")
    list_cameras: bool = False
    control_hz: float = 60.0  # Followers and recording; leaders run at five times this rate.
    camera_hz: float = 20.0
    camera_capture_hz: int = 30
    camera_width: int = 640
    camera_height: int = 480
    jpeg_quality: int = 90
    arm: str = "yam"
    follower_gripper: str = "linear_4310"
    leader_gripper: str = "yam_teaching_handle"
    left_leader_can: str = "can_leader_l"
    right_leader_can: str = "can_leader_r"
    left_follower_can: str = "can_follower_l"
    right_follower_can: str = "can_follower_r"
    feedback_mode: Literal["angle", "torque"] = "angle"
    bilateral_kp: float = 1.0
    bilateral_alpha: float = 0.0
    bilateral_beta: float = 5.0
    yellow_button_index: int = 0
    white_button_index: int = 1
    print_buttons: bool = False
    alignment_s: float = 3.0
    stale_state_s: float = 0.2
    startup_timeout_s: float = 120.0
    drain_s: float = 1.0
    queue_size: int = 512
    wrench_damping: float = 0.01
    wrench_sign: float = 1.0


def _snapshot(robot: Any) -> dict[str, Any]:
    # Subtract the last submitted gravity term without recomputing it between control updates.
    return robot.get_force_feedback_observations(use_applied_gravity=True)


def _fresh(obs: dict[str, Any], now_ns: int, timeout_s: float) -> bool:
    timestamps = obs["can_rx_monotonic_ns"][:6]
    return bool(np.all(timestamps > 0) and 0 <= now_ns - int(np.min(timestamps)) <= timeout_s * 1e9)


class ArmPair:
    """One pair's state, owned entirely by the shared control thread."""

    def __init__(self, side: str, leader: Any, follower: Any, args: Args) -> None:
        from examples.minimum_gello.minimum_gello import YAMLeaderRobot

        self.side, self.leader, self.follower, self.args = side, leader, follower, args
        self.adapter = YAMLeaderRobot(leader)
        self.kp = np.asarray(leader.get_robot_info()["kp"][:6], dtype=float)
        self.edges = ButtonEdges()
        self.mode = "idle"
        self.last_buttons = None
        self.align_start_ns = 0
        self.align_start = self.align_target = None
        self.command = None
        self.command_ns = None
        self.follower_update_ns = 0
        self.feedback_torque = np.zeros(6)
        self.feedback_ns = 0

    def read(self, packets: Any) -> None:
        self.leader.refresh_state()
        self.follower.refresh_state()
        self.leader_obs = _snapshot(self.leader)
        self.follower_obs = _snapshot(self.follower)
        handle_pos, self.buttons, _, _ = self.adapter.get_info()
        self.pos = np.concatenate([self.leader_obs["joint_pos"][:6], handle_pos[-1:]])
        if not np.all(np.isfinite(self.pos)):
            raise ValueError(f"{self.side}: non-finite leader position")
        now_ns = time.monotonic_ns()
        if self.args.print_buttons and list(self.buttons) != self.last_buttons:
            print(f"{self.side} io_inputs={list(self.buttons)}", flush=True)
            self.last_buttons = list(self.buttons)
        if self.side == "right" and self.edges.rising(
            self.args.white_button_index, bool(self.buttons[self.args.white_button_index]), now_ns
        ):
            packets.put_nowait({"kind": "toggle", "monotonic_ns": now_ns})
        if self.edges.rising(self.args.yellow_button_index, bool(self.buttons[self.args.yellow_button_index]), now_ns):
            if self.mode == "idle":
                self.align_start = self.follower_obs["joint_pos"].copy()
                self.align_target = self.pos.copy()
                self.align_start_ns = now_ns
                self.mode = "aligning"
            else:
                self.mode = "idle"
            logging.info("%s yellow button: %s", self.side, self.mode)
        if self.mode != "idle" and not (
            _fresh(self.leader_obs, now_ns, self.args.stale_state_s)
            and _fresh(self.follower_obs, now_ns, self.args.stale_state_s)
        ):
            raise RuntimeError(f"{self.side}: CAN state is stale; stopping teleoperation")

    def update_follower(self) -> None:
        self.command = None
        self.command_ns = None
        if self.mode == "aligning":
            fraction = min(1.0, (time.monotonic_ns() - self.align_start_ns) / (self.args.alignment_s * 1e9))
            target = self.align_start + fraction * (self.align_target - self.align_start)
            self.command = {"pos": target, "vel": np.zeros_like(target)}
            if fraction >= 1.0:
                self.mode = "following"
        elif self.mode == "following":
            self.command = {
                "pos": self.pos.copy(),
                "vel": np.concatenate([self.leader_obs["joint_vel"][:6], [0.0]]),
            }
        if self.command is None:
            self.follower.enter_gravity_comp_idle()
        else:
            self.command["pos"] = self.follower._clip_robot_joint_pos_command(self.command["pos"].copy())
            self.command_ns = time.monotonic_ns()
            self.follower.command_joint_state(self.command)
        # Compute gravity and submit to CAN's command buffer, including when idle.
        self.follower.update()
        self.follower_update_ns = time.monotonic_ns()
        self.follower_obs = _snapshot(self.follower)

    def update_leader(self) -> None:
        from examples.minimum_gello.minimum_gello import (
            _feedback_source,
            _leader_angle_target,
            _leader_feedback_torque,
        )

        if self.mode == "following":
            target = self.follower_obs["joint_pos"][:6]
            if self.args.feedback_mode == "angle":
                target = _leader_angle_target(target, self.leader_obs["joint_vel"][:6])
            source = _feedback_source(
                self.args.feedback_mode,
                self.pos[:6],
                target,
                self.follower_obs["joint_eff_external"][:6],
                self.kp,
            )
            self.feedback_torque = _leader_feedback_torque(
                source,
                self.follower_obs["joint_vel"][:6],
                self.pos[:6] - target,
                self.args.bilateral_kp,
                self.args.bilateral_alpha,
                self.args.bilateral_beta,
            )
            if not np.all(np.isfinite(self.feedback_torque)):
                raise ValueError(f"{self.side}: non-finite feedback torque")
            self.leader.command_joint_torques(self.feedback_torque)
        else:
            self.leader.enter_gravity_comp_idle()
            self.feedback_torque = np.zeros(6)
        self.leader.update()
        self.feedback_ns = time.monotonic_ns()
        self.leader_obs = _snapshot(self.leader)

    def packet(self, index: int, scheduled_ns: int, missed: int) -> dict[str, Any]:
        return {
            "kind": "robot",
            "side": self.side,
            "monotonic_ns": time.monotonic_ns(),
            "timestamp_ns": time.time_ns(),
            "scheduled_monotonic_ns": scheduled_ns,
            "tick_index": index,
            "missed_periods": missed,
            "leader": self.leader_obs,
            "leader_joint_pos_with_gripper": np.concatenate([self.leader_obs["joint_pos"][:6], self.pos[-1:]]),
            "buttons": np.asarray(self.buttons).copy(),
            "follower": self.follower_obs,
            "command": self.command,
            "command_monotonic_ns": self.command_ns,
            "follower_update_monotonic_ns": self.follower_update_ns,
            "teleoperation_mode": self.mode,
            "leader_feedback_torque": self.feedback_torque.copy(),
            "leader_feedback_monotonic_ns": self.feedback_ns,
        }


def control_worker(args: Args, packets: Any, stop: Any, origin_ns: int) -> None:
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    worker_signals()
    robots, pairs = [], []
    try:
        for side in ("left", "right"):
            arms = {}
            for role in ("follower", "leader"):
                robot = get_yam_robot(
                    channel=getattr(args, f"{side}_{role}_can"),
                    arm_type=ArmType.from_string_name(args.arm),
                    gripper_type=GripperType.from_string_name(getattr(args, f"{role}_gripper")),
                    background_control=False,
                )
                robots.append(robot)
                arms[role] = robot
            if arms["leader"].num_dofs() != 6 or arms["follower"].num_dofs() != 7:
                raise ValueError("collection requires six-joint leaders and seven-joint followers including grippers")
            pairs.append(ArmPair(side, arms["leader"], arms["follower"], args))
        # Initialization may take seconds; do not count it as missed control ticks.
        clock = PeriodicClock(args.control_hz * 5, max(origin_ns, time.monotonic_ns()))
        iteration, missed_since_sample = 0, 0
        while not stop.is_set():
            if stop.wait(max(0.0, (clock.deadline_ns - time.monotonic_ns()) / 1e9)):
                break
            index, scheduled_ns, missed = clock.consume(time.monotonic_ns())
            missed_since_sample += missed
            for pair in pairs:
                pair.read(packets)
            follower_tick = iteration % 5 == 0
            if follower_tick:
                for pair in pairs:
                    pair.update_follower()
            # Both followers finish before either leader computes and submits its command.
            for pair in pairs:
                pair.update_leader()
            if follower_tick:
                if iteration == 0:
                    for pair in pairs:
                        packets.put_nowait({"kind": "ready", "name": pair.side})
                for pair in pairs:
                    packets.put_nowait(pair.packet(index, scheduled_ns, missed_since_sample))
                missed_since_sample = 0
            iteration += 1
    except BaseException as error:
        if not stop.is_set():
            try:
                packets.put_nowait({"kind": "error", "name": "control", "error": repr(error)})
            except queue.Full:
                pass
        raise
    finally:
        for robot in robots:
            try:
                robot.enter_gravity_comp_idle()
                robot.refresh_state()
                robot.update()
            except Exception:
                logging.exception("Could not idle arm during shutdown")
        for robot in reversed(robots):
            try:
                robot.close()
            except Exception:
                logging.exception("Could not close arm during shutdown")


class RecordingSession:
    """Recording state is independent of the two yellow-button teleoperation states."""

    def __init__(self, args: Args, metadata: dict[str, Any], projections: dict[str, Any] | None = None) -> None:
        self.args = args
        self.metadata = metadata
        self.active: EpisodeWriter | None = None
        self.closing: list[EpisodeWriter] = []
        self.projections = projections
        self.pending: list[tuple[int, int, dict[str, Any]]] = []
        self.sequence = 0
        self.flushed_until_ns = -1

    def toggle(self, timestamp_ns: int) -> None:
        if timestamp_ns <= self.flushed_until_ns:
            raise RuntimeError("button event exceeded the reorder window; increase --drain-s")
        if self.active is None:
            self.active = EpisodeWriter(self.args.output, timestamp_ns, self.metadata)
            print(f"RECORDING {self.active.path}", flush=True)
        else:
            self.active.end_ns = timestamp_ns
            self.closing.append(self.active)
            print(f"STOP RECORDING {self.active.path} (teleoperation unchanged)", flush=True)
            self.active = None

    def append(self, packet: dict[str, Any]) -> None:
        # A short reorder window lets white-button edges overtake already queued samples.
        # This prevents a sample after the stop edge being written before we learn about that edge.
        if packet["monotonic_ns"] <= self.flushed_until_ns:
            raise RuntimeError("sample exceeded the reorder window; increase --drain-s")
        self.sequence += 1
        heapq.heappush(self.pending, (packet["monotonic_ns"], self.sequence, packet))

    def _write_packet(self, packet: dict[str, Any]) -> None:
        episodes = self.closing + ([self.active] if self.active is not None else [])
        if not any(episode.accepts(packet["monotonic_ns"]) for episode in episodes):
            return
        if packet["kind"] == "robot" and self.projections is not None:
            pose, velocity, wrench = self.projections[packet["side"]].project(packet)
            packet["cartesian"] = {
                "tcp_pose": pose,
                "tcp_velocity": velocity,
                "tcp_wrench_estimate": wrench,
                "wrench_damping": self.args.wrench_damping,
                "wrench_sign": self.args.wrench_sign,
            }
        for episode in episodes:
            episode.append(packet)

    def flush(self, now_ns: int) -> None:
        cutoff = now_ns - round(self.args.drain_s * 1e9)
        while self.pending and self.pending[0][0] <= cutoff:
            _, _, packet = heapq.heappop(self.pending)
            self._write_packet(packet)
        self.flushed_until_ns = max(self.flushed_until_ns, cutoff)
        remaining = []
        for episode in self.closing:
            if cutoff >= episode.end_ns:
                print(f"SAVED {episode.finish(episode.end_ns)} {episode.counts}", flush=True)
            else:
                remaining.append(episode)
        self.closing = remaining
        if self.active is not None:
            self.active.flush()

    def close(self, reason: str) -> None:
        while self.pending:
            _, _, packet = heapq.heappop(self.pending)
            self._write_packet(packet)
        if self.active is not None:
            self.active.finish(self.active.end_ns or time.monotonic_ns(), status="interrupted", reason=reason)
            self.active = None
        for episode in self.closing:
            episode.finish(episode.end_ns, status="complete", reason=reason)
        self.closing.clear()


def validate(args: Args) -> dict[str, str]:
    cameras = dict(CAMERA_DEVICES)
    camera_ids = {
        str(Path(f"/dev/video{device}" if device.isdigit() else device).resolve())
        if not device.startswith("realsense:")
        else device
        for device in cameras.values()
    }
    if not all(cameras.values()) or len(camera_ids) != 3:
        raise ValueError("CAMERA_DEVICES must contain three distinct camera devices")
    for name in (
        "control_hz",
        "camera_hz",
        "camera_capture_hz",
        "alignment_s",
        "stale_state_s",
        "startup_timeout_s",
        "drain_s",
        "wrench_damping",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.camera_hz > args.camera_capture_hz:
        raise ValueError("camera_hz cannot exceed camera_capture_hz")
    if args.wrench_sign not in (-1.0, 1.0):
        raise ValueError("wrench_sign must be +1 or -1")
    for name in ("bilateral_kp", "bilateral_alpha", "bilateral_beta"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if {args.yellow_button_index, args.white_button_index} != {0, 1}:
        raise ValueError("yellow and white must use distinct handle input indices 0 and 1")
    channels = [getattr(args, f"{side}_{role}_can") for side in ("left", "right") for role in ("leader", "follower")]
    if len(set(channels)) != 4:
        raise ValueError("each of the four arms needs its own CAN channel")
    if not 1 <= args.jpeg_quality <= 100 or args.queue_size < 32 or min(args.camera_width, args.camera_height) < 1:
        raise ValueError("invalid JPEG quality, queue size or image dimensions")
    return cameras


def main(args: Args) -> None:
    if args.list_cameras:
        list_cameras()
        return
    cameras = validate(args)
    logging.basicConfig(level=logging.INFO)
    if args.bilateral_kp == 0:
        print("bilateral_kp=0: additional force feedback is disabled; supply your usual feedback gain.", flush=True)
    context = mp.get_context("spawn")
    packets = context.Queue(maxsize=args.queue_size)
    stop = context.Event()
    origin_ns = time.monotonic_ns()
    workers = [
        context.Process(target=camera_worker, args=(name, device, args, packets, stop, origin_ns), name=name)
        for name, device in cameras.items()
    ]
    workers.append(context.Process(target=control_worker, args=(args, packets, stop, origin_ns), name="control"))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    metadata = {
        "settings": json.loads(json.dumps(asdict(args), default=str)),
        "camera_devices": cameras,
        "git_revision": revision,
    }
    projections = {
        side: CartesianProjection(args.arm, args.follower_gripper, args.wrench_damping, args.wrench_sign)
        for side in ("left", "right")
    }
    metadata["control_timing"] = {
        "leader_hz": args.control_hz * 5,
        "follower_hz": args.control_hz,
        "follower_every_iterations": 5,
        "order": "read all CAN buffers; both followers when due; both leaders; robot samples when due",
        "tick_index": "leader scheduler slot; follower update every five executed iterations",
        "missed_periods": "skipped leader slots since the preceding robot sample",
        "gravity_residual": "measured effort minus last submitted gravity compensation term",
    }
    metadata["cartesian"] = {
        "site": "grasp_site",
        "pose": "[x,y,z,roll,pitch,yaw], m/rad, each arm base frame",
        "velocity": "[vx,vy,vz,wx,wy,wz], m/s and rad/s, each arm base frame",
        "wrench": "[Fx,Fy,Fz,Tx,Ty,Tz], N/Nm, local TCP, estimate from actuator effort residual",
        "damping": args.wrench_damping,
        "sign": args.wrench_sign,
    }
    session = RecordingSession(args, metadata, projections)
    ready: set[str] = set()
    last_seen: dict[str, int] = {}
    started_ns = time.monotonic_ns()
    next_flush = started_ns
    reason = "user stopped collector"
    previous_handler = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        for worker in workers:
            worker.start()
        while True:
            try:
                packet = packets.get(timeout=0.02)
            except queue.Empty:
                packet = None
            now_ns = time.monotonic_ns()
            if packet is not None:
                kind = packet["kind"]
                if kind == "error":
                    raise RuntimeError(f"{packet['name']}: {packet['error']}")
                if kind == "ready":
                    ready.add(packet["name"])
                    print(f"READY {packet['name']} ({len(ready)}/5)", flush=True)
                elif kind == "toggle":
                    if len(ready) == 5:
                        session.toggle(packet["monotonic_ns"])
                    else:
                        print("White button ignored: waiting for both arm pairs and all three cameras.", flush=True)
                else:
                    last_seen[packet.get("side", packet.get("camera"))] = packet["monotonic_ns"]
                    session.append(packet)
            for worker in workers:
                if worker.exitcode is not None:
                    raise RuntimeError(f"{worker.name} exited ({worker.exitcode})")
            if len(ready) < 5 and now_ns - started_ns > args.startup_timeout_s * 1e9:
                raise TimeoutError(f"startup timeout; ready devices: {sorted(ready)}")
            if len(ready) == 5:
                for name in ready:
                    if name in last_seen and now_ns - last_seen[name] > 3e9:
                        raise TimeoutError(f"no samples from {name} for 3 seconds")
            if now_ns >= next_flush:
                session.flush(now_ns)
                next_flush = now_ns + 250_000_000
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        reason = repr(error)
        raise
    finally:
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop.set()
        if session.active is not None:
            session.active.end_ns = time.monotonic_ns()
        try:
            # Drain producers during shutdown too; queued images may precede a recording boundary.
            deadline = time.monotonic() + args.drain_s
            while time.monotonic() < deadline:
                try:
                    packet = packets.get(timeout=0.05)
                    if packet["kind"] in ("robot", "camera"):
                        session.append(packet)
                except queue.Empty:
                    continue
            session.close(reason)
        finally:
            for worker in workers:
                if worker.pid is not None:
                    worker.join(timeout=3.0)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=2.0)
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=1.0)
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main(tyro.cli(Args))
