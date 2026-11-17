"""Run a joint-action RDP checkpoint directly on YAM, without ROS."""

import math
import multiprocessing as mp
import queue
import signal
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.bimanual_data_collection.camera_devices import CAMERA_DEVICES
from examples.bimanual_data_collection.cameras import camera_worker
from examples.bimanual_data_collection.recording import PeriodicClock, request_shutdown
from examples.rdp_policy.buttons import HandleButtons
from examples.rdp_policy.runtime import History, RDPModel, SlowPlanner

CAMERA_KEYS = {"external_img": "external", "left_wrist_img": "left_wrist", "right_wrist_img": "right_wrist"}
STATE_SHAPES = {
    "joint_pos": [7],
    "joint_vel": [7],
    "joint_effort": [6],
    "joint_effort_external": [6],
    "tcp_pose": [9],
    "tcp_vel": [6],
    "tcp_wrench": [6],
    "gripper_width": [1],
}


@dataclass
class Args:
    checkpoint: Path
    rdp_root: Path = REPO_ROOT.parent / "RDP"
    at_checkpoint: Path | None = None
    device: str = "cuda:0"
    side: Literal["left", "right"] = "right"
    can_channel: str | None = None  # Defaults to can_follower_r/l according to side.
    button_can_channel: str = "can_leader_r"
    yellow_button_index: int = 0
    button_timeout_s: float = 0.2
    arm: str = "yam"
    gripper: str = "linear_4310"
    external_camera: str = CAMERA_DEVICES["external"]
    right_wrist_camera: str = CAMERA_DEVICES["right_wrist"]
    left_wrist_camera: str = CAMERA_DEVICES["left_wrist"]
    control_hz: float = 20.0
    command_hz: float = 60.0
    inference_hz: float = 5.0
    inference_steps: int = 8
    camera_hz: float = 5.0
    camera_capture_hz: int = 30
    camera_width: int = 640
    camera_height: int = 480
    jpeg_quality: int = 90
    wrench_damping: float = 0.01
    wrench_sign: float = 1.0
    gripper_max_width_m: float = 0.095
    stale_state_s: float = 0.2
    stale_camera_s: float = 0.5
    startup_timeout_s: float = 30.0
    policy_timeout_s: float = 1.0
    max_steps: int | None = None
    validate_only: bool = False  # Load/validate checkpoint without opening CAN or cameras.


def validate(args: Args, model: Any) -> dict[str, Any]:
    for name in (
        "control_hz",
        "command_hz",
        "inference_hz",
        "camera_hz",
        "wrench_damping",
        "gripper_max_width_m",
        "stale_state_s",
        "stale_camera_s",
        "startup_timeout_s",
        "policy_timeout_s",
        "button_timeout_s",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    divisor = args.control_hz / args.inference_hz
    if divisor < 1 or not math.isclose(divisor, round(divisor)):
        raise ValueError("control_hz must be an integer multiple of inference_hz")
    command_divisor = args.command_hz / args.control_hz
    if command_divisor < 1 or not math.isclose(command_divisor, round(command_divisor)):
        raise ValueError("command_hz must be an integer multiple of control_hz")
    if not math.isclose(args.control_hz, model.control_hz):
        raise ValueError("control_hz must match the checkpoint training rate")
    visual_hz = getattr(model, "visual_hz", None)
    if visual_hz is not None and not math.isclose(args.camera_hz, visual_hz):
        raise ValueError("camera_hz must match the checkpoint visual sampling rate; retrain for 5 Hz")
    if args.inference_steps < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("inference_steps and max_steps must be positive")
    if args.yellow_button_index not in (0, 1):
        raise ValueError("yellow_button_index must be handle input 0 or 1")
    if args.camera_capture_hz < args.camera_hz or min(args.camera_width, args.camera_height) < 1:
        raise ValueError("invalid camera capture rate or dimensions")
    if not 1 <= args.jpeg_quality <= 100 or args.wrench_sign not in (-1.0, 1.0):
        raise ValueError("invalid JPEG quality or wrench sign")
    cameras = {}
    supported = {f"{args.side}_robot_{key}": shape for key, shape in STATE_SHAPES.items()}
    for key, meta in {**model.obs_meta, **model.extended_meta}.items():
        if meta.get("type", "low_dim") == "rgb":
            if (
                key not in CAMERA_KEYS
                or key in model.extended_meta
                or len(meta["shape"]) != 3
                or meta["shape"][0] != 3
            ):
                raise ValueError(f"unsupported RGB input: {key}")
            cameras[key] = meta
        elif key not in supported or list(meta["shape"]) != supported[key]:
            raise ValueError(f"unsupported state input or shape: {key}")
    if not args.validate_only:
        devices = [getattr(args, f"{CAMERA_KEYS[key]}_camera") for key in cameras]
        normalized = [
            device
            if device.startswith("realsense:")
            else str(Path(f"/dev/video{device}" if device.isdigit() else device).resolve())
            for device in devices
        ]
        if not all(devices) or len(set(normalized)) != len(devices):
            raise ValueError("specify a distinct device for every camera used by the checkpoint")
    return cameras


class LiveImages:
    """Reuse collection's camera processes; decode only new frames in the control process."""

    def __init__(self, args: Args, specs: dict[str, Any]) -> None:
        self.args, self.specs = args, specs
        context = mp.get_context("spawn")
        self.packets = context.Queue(maxsize=64)
        self.stop = context.Event()
        self.frames: dict[str, tuple[int, np.ndarray]] = {}
        self.future_packets: list[dict[str, Any]] = []
        origin_ns = time.monotonic_ns()
        self.workers = [
            context.Process(
                target=camera_worker,
                args=(
                    CAMERA_KEYS[key],
                    getattr(args, f"{CAMERA_KEYS[key]}_camera"),
                    args,
                    self.packets,
                    self.stop,
                    origin_ns,
                ),
                name=f"rdp-{CAMERA_KEYS[key]}",
            )
            for key in specs
        ]
        try:
            for worker in self.workers:
                worker.start()
        except BaseException:
            self.close()
            raise

    def read(self, now_ns: int, *, required: bool = True) -> dict[str, np.ndarray] | None:
        try:
            return self._read(now_ns)
        except (RuntimeError, ValueError):
            if required:
                raise
            return None

    def _read(self, now_ns: int) -> dict[str, np.ndarray] | None:
        import cv2

        packets, self.future_packets = self.future_packets, []
        while True:
            try:
                packets.append(self.packets.get_nowait())
            except queue.Empty:
                break
        for packet in packets:
            if packet["kind"] == "error":
                raise RuntimeError(f"{packet['name']}: {packet['error']}")
            if packet["kind"] != "camera":
                continue
            if packet["monotonic_ns"] > now_ns:
                self.future_packets.append(packet)
                continue
            key = f"{packet['camera']}_img"
            _, height, width = self.specs[key]["shape"]
            bgr = cv2.imdecode(np.frombuffer(packet["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot decode {key}")
            rgb = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
            self.frames[key] = (packet["monotonic_ns"], np.moveaxis(rgb, -1, 0).astype(np.float32) / 255)
        for worker in self.workers:
            if worker.exitcode is not None:
                raise RuntimeError(f"camera worker {worker.name} exited ({worker.exitcode})")
        if len(self.frames) != len(self.specs):
            return None
        # Frames may arrive while the queue is drained; validate against the current receive time.
        now_ns = max(now_ns, time.monotonic_ns())
        if any(now_ns - ns > self.args.stale_camera_s * 1e9 for ns, _ in self.frames.values()):
            raise RuntimeError("camera data is stale")
        return {key: value for key, (_, value) in self.frames.items()}

    def close(self) -> None:
        self.stop.set()
        for worker in self.workers:
            if worker.pid is not None:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=1.0)
        self.packets.close()


def joint_command(action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=float)
    if action.shape != (14,) or not np.all(np.isfinite(action)):
        raise ValueError("expected finite pos7/vel7 action")
    pos, vel = action[:7].copy(), action[7:].copy()
    pos[6] = np.clip(pos[6], 0.0, 1.0)
    vel[6] = 0.0
    return {"pos": pos, "vel": vel}


def interpolate_command(
    start: dict[str, np.ndarray], target: dict[str, np.ndarray], fraction: float
) -> dict[str, np.ndarray]:
    fraction = min(1.0, max(0.0, fraction))
    return {key: start[key] + fraction * (target[key] - start[key]) for key in ("pos", "vel")}


def control_loop(
    robot: Any,
    model: Any,
    planner: Any,
    projection: Any,
    images: Any,
    *,
    args: Args,
    stop: Any,
    buttons: Any,
) -> None:
    history = History(max(2 * model.horizon, model.past_steps))
    divisor = round(args.control_hz / args.inference_hz)
    command_divisor = round(args.command_hz / args.control_hz)
    clock = PeriodicClock(args.command_hz, time.monotonic_ns())
    started_ns = last_action_ns = time.monotonic_ns()
    enabled = active = False
    hold = None
    segment_start = segment_target = None
    segment_index = None
    last_policy_index = -1
    count = 0
    next_report_ns = started_ns
    while not stop.is_set() and (args.max_steps is None or count < args.max_steps):
        if stop.wait(max(0.0, (clock.deadline_ns - time.monotonic_ns()) / 1e9)):
            break
        index, _, missed = clock.consume(time.monotonic_ns())
        policy_index = index // command_divisor
        policy_tick = policy_index != last_policy_index
        last_policy_index = policy_index
        robot.refresh_state()
        state = robot.get_force_feedback_observations(use_applied_gravity=True)
        now_ns = time.monotonic_ns()
        can_ns = state["can_rx_monotonic_ns"][:6]
        if np.any(can_ns <= 0) or np.any(can_ns > now_ns) or now_ns - int(np.min(can_ns)) > args.stale_state_s * 1e9:
            raise RuntimeError("robot CAN feedback is stale")
        for _ in range(buttons.consume_toggles(started_ns)):
            enabled = not enabled
            history.rows.clear()
            planner.clear()
            active = False
            segment_start = segment_target = segment_index = None
            last_action_ns = now_ns
            hold = state["joint_pos"].copy() if enabled else None
            print("Yellow: policy enabled" if enabled else "Yellow: gravity-compensation idle", flush=True)
        if not enabled:
            robot.enter_gravity_comp_idle()
            robot.update()  # Clear PD targets before decoding/draining any queued camera frames.
            images.read(now_ns, required=False)
        else:
            if segment_target is not None:
                command = interpolate_command(
                    segment_start, segment_target, (index - segment_index) / command_divisor
                )
            else:
                command = {"pos": hold.copy(), "vel": np.zeros(7)}
            hold = command["pos"].copy()
            robot.command_joint_state(command)
            robot.update()
            if policy_tick:
                frame_values = images.read(now_ns, required=True)
                action = None
                if frame_values is not None:
                    observations = projection.project(state)
                    observations.update(frame_values)
                    history.append(policy_index, observations)
                    if len(history.rows) >= model.past_steps and policy_index % divisor == 0:
                        planner.submit(
                            policy_index, history.observations(model.obs_meta, model.n_obs_steps, model.ratio)
                        )
                    plan = planner.latest()
                    # A visual latent from observation t becomes available to
                    # the fast proprio decoder no earlier than tick t + 1.
                    if plan is not None and plan.anchor_step >= policy_index:
                        plan = None
                    if plan is not None:
                        temporal = history.temporal(plan, model.horizon, model.extended_meta)
                        if temporal is not None:
                            action = model.decode(plan, temporal)
                if action is None:
                    timeout = args.policy_timeout_s if active else args.startup_timeout_s
                    if now_ns - last_action_ns > timeout * 1e9:
                        raise TimeoutError(
                            "no usable RDP plan; check inference speed, history continuity and model worker"
                        )
                    target = {"pos": hold.copy(), "vel": np.zeros(7)}
                else:
                    target = joint_command(action)
                    active = True
                    last_action_ns = now_ns
                segment_start, segment_target, segment_index = command, target, index
        count += 1
        if now_ns >= next_report_ns:
            mode = "running" if enabled and active else "warming up" if enabled else "gravity idle; waiting for yellow"
            print(f"RDP {mode}: tick={index}, missed={missed}", flush=True)
            next_report_ns = now_ns + 5_000_000_000


def close_robot(robot: Any) -> None:
    try:
        robot.enter_gravity_comp_idle()
        robot.refresh_state()
        robot.update()
    finally:
        robot.close()


def main(args: Args) -> None:
    model = RDPModel(
        args.checkpoint,
        args.rdp_root,
        args.device,
        args.side,
        at_checkpoint=args.at_checkpoint,
        inference_steps=args.inference_steps,
    )
    specs = validate(args, model)
    print(
        f"Loaded {args.side} joint RDP: observation={args.control_hz} Hz, "
        f"command={args.command_hz} Hz, diffusion={args.inference_hz} Hz, action=14D"
    )
    if args.validate_only:
        return
    import torch
    from post_process_forcegello import ArmKinematics

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    torch.set_num_threads(1)
    arm, gripper = ArmType.from_string_name(args.arm), GripperType.from_string_name(args.gripper)
    projection = ArmKinematics(
        Path(combine_arm_and_gripper_xml(arm, gripper)),
        args.wrench_damping,
        args.wrench_sign,
        args.gripper_max_width_m,
        args.side,
    )
    images = planner = robot = buttons = None
    previous_handler = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        buttons = HandleButtons(args.button_can_channel, args.yellow_button_index, args.button_timeout_s)
        # Gripper calibration can take seconds. Start cameras afterwards so their queues cannot fill during it.
        channel = args.can_channel or f"can_follower_{args.side[0]}"
        robot = get_yam_robot(channel=channel, arm_type=arm, gripper_type=gripper, background_control=False)
        images = LiveImages(args, specs)
        planner = SlowPlanner(model)
        control_loop(
            robot,
            model,
            planner,
            projection,
            images,
            args=args,
            stop=threading.Event(),
            buttons=buttons,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # ExitStack runs every cleanup even if one fails; clear robot commands before waiting for workers.
        with ExitStack() as cleanup:
            cleanup.callback(signal.signal, signal.SIGTERM, previous_handler)
            if buttons is not None:
                cleanup.callback(buttons.close)
            if images is not None:
                cleanup.callback(images.close)
            if planner is not None:
                cleanup.callback(planner.close)
            if robot is not None:
                cleanup.callback(close_robot, robot)


if __name__ == "__main__":
    main(tyro.cli(Args))
