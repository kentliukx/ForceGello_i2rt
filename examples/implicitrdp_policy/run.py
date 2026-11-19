"""Run a ForceGello joint-command ImplicitRDP checkpoint without ROS."""

import math
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

I2RT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(I2RT_ROOT))

from examples.bimanual_data_collection.camera_devices import CAMERA_DEVICES
from examples.bimanual_data_collection.recording import PeriodicClock, request_shutdown
from examples.rdp_policy.buttons import HandleButtons
from examples.rdp_policy.run import LiveImages, close_robot


@dataclass
class Args:
    checkpoint: Path
    implicitrdp_root: Path = I2RT_ROOT.parent / "ImplicitRDP"
    device: str = "cuda:0"
    side: Literal["left", "right"] = "right"
    can_channel: str | None = None
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
    validate_only: bool = False
    debug_actions: bool = False


class Model:
    def __init__(self, checkpoint: Path, device: str, side: str) -> None:
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        cfg = payload["cfg"]
        meta = cfg.task.shape_meta
        expected_state = {f"{side}_robot_joint_pos", f"{side}_robot_joint_vel", f"{side}_robot_tcp_wrench"}
        expected_obs = expected_state | {"external_img", f"{side}_wrist_img"}
        if (
            meta.action.get("space") != "joint"
            or meta.action.get("key") != f"{side}_robot_command"
            or list(meta.action.shape) != [7]
        ):
            raise ValueError("checkpoint must be trained for the selected arm's absolute 7D joint command")
        if set(meta.obs.keys()) != expected_obs or set(meta.extended_obs.keys()) != expected_state:
            raise ValueError(
                "checkpoint observations must be two cameras plus joint position, velocity and TCP wrench"
            )
        if (
            float(cfg.task.control_hz) != 20
            or float(cfg.task.visual_hz) != 5
            or int(cfg.task.dataset.sample_phase) != 0
            or int(cfg.task.dataset.visual_latency_steps) != 1
            or int(cfg.n_obs_steps) != 5
            or int(cfg.horizon) != 20
            or int(cfg.n_latency_steps) != 0
        ):
            raise ValueError("checkpoint timing must be 20 Hz joints, 5 Hz delayed images, phase 0")
        if any("tcp_pose" in key for key in meta.obs.keys()) or any(
            "tcp_pose" in key for key in meta.extended_obs.keys()
        ):
            raise ValueError("checkpoint still contains TCP pose observations; retrain with the joint task")
        if cfg.task.dataset.delta_action or cfg.task.dataset.relative_action:
            raise ValueError("checkpoint actions must be absolute joint commands")
        self.obs_meta = OmegaConf.to_container(meta.obs, resolve=True)
        self.extended_meta = OmegaConf.to_container(meta.extended_obs, resolve=True)
        self.obs_steps = int(cfg.n_obs_steps)
        self.policy = hydra.utils.instantiate(cfg.policy)
        state_key = "ema_model" if cfg.training.use_ema and "ema_model" in payload["state_dicts"] else "model"
        self.policy.load_state_dict(payload["state_dicts"][state_key])
        self.policy.eval().to(device)
        self.device = device

    def predict(self, obs: dict[str, np.ndarray], extended: dict[str, np.ndarray]) -> np.ndarray:
        obs_tensors = {
            key: self.torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(self.device)
            for key, value in obs.items()
        }
        extended_tensors = {
            key: self.torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(self.device)
            for key, value in extended.items()
        }
        with self.torch.inference_mode():
            action = self.policy.predict_action(obs_tensors, extended_obs_dict=extended_tensors)["action"][0, 0]
        result = action.detach().cpu().numpy().copy()
        if result.shape != (7,) or not np.all(np.isfinite(result)):
            raise ValueError("policy returned a non-finite or non-7D joint command")
        return result


class History:
    def __init__(self, obs_steps: int, obs_meta: dict, extended_meta: dict) -> None:
        self.rows = deque(maxlen=obs_steps + 1)
        self.obs_steps = obs_steps
        self.obs_meta = obs_meta
        self.extended_meta = extended_meta

    def clear(self) -> None:
        self.rows.clear()

    def append(self, index: int, row: dict[str, np.ndarray]) -> None:
        if self.rows and index != self.rows[-1][0] + 1:
            self.clear()
        self.rows.append((index, {key: value.copy() for key, value in row.items()}))

    def input(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
        if len(self.rows) < self.obs_steps + 1:
            return None
        rows = dict(self.rows)
        anchor = self.rows[-1][0]
        obs = {}
        for key, meta in self.obs_meta.items():
            # A 5 Hz frame is held for four 20 Hz ticks. The visual observation
            # is one tick older than the proprioception observation.
            if meta.get("type") == "rgb":
                ids = [anchor - self.obs_steps + step for step in range(self.obs_steps)]
            else:
                ids = [anchor - self.obs_steps + 1 + step for step in range(self.obs_steps)]
            obs[key] = np.stack([rows[i][key] for i in ids])
        extended = {
            key: np.stack([rows[i][key] for i in range(anchor - self.obs_steps + 1, anchor + 1)])
            for key in self.extended_meta
        }
        return obs, extended


def interpolate_joint_command(
    start: np.ndarray, target: np.ndarray, tick: int, anchor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reach the next 20 Hz target over three 60 Hz commands."""
    fraction = min(1.0, max(0.0, (tick - anchor * 3) / 3))
    delta = target - start
    velocity = delta / 0.05
    velocity[6] = 0.0
    return start + fraction * delta, velocity


def validate(args: Args, model: Model) -> dict[str, dict]:
    if (
        not math.isclose(args.control_hz, 20)
        or not math.isclose(args.command_hz, 60)
        or not math.isclose(args.camera_hz, 5)
    ):
        raise ValueError("this checkpoint requires 20 Hz inference, 60 Hz commands, and 5 Hz images")
    for name in (
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
    if args.wrench_sign not in (-1, 1) or args.yellow_button_index not in (0, 1):
        raise ValueError("invalid wrench sign or button index")
    if args.camera_capture_hz < args.camera_hz or min(args.camera_width, args.camera_height) < 1:
        raise ValueError("invalid camera capture settings")
    if not 1 <= args.jpeg_quality <= 100 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("invalid JPEG quality or max steps")
    expected_lowdim = {
        f"{args.side}_robot_joint_pos": [7],
        f"{args.side}_robot_joint_vel": [7],
        f"{args.side}_robot_tcp_wrench": [6],
    }
    cameras = {}
    for key, meta in {**model.obs_meta, **model.extended_meta}.items():
        if meta.get("type") == "rgb":
            if key not in ("external_img", "right_wrist_img", "left_wrist_img") or key in model.extended_meta:
                raise ValueError(f"unsupported camera observation: {key}")
            if len(meta["shape"]) != 3 or meta["shape"][0] != 3 or int(meta.get("hold_steps", 1)) != 4:
                raise ValueError(f"camera shape/hold rate differs from training: {key}")
            cameras[key] = meta
        elif key not in expected_lowdim or list(meta["shape"]) != expected_lowdim[key]:
            raise ValueError(f"unsupported proprioception observation: {key}")
    if not args.validate_only:
        devices = [getattr(args, f"{key.removesuffix('_img')}_camera") for key in cameras]
        if not all(devices) or len(set(devices)) != len(devices):
            raise ValueError("camera devices must be distinct")
    return cameras


def run(args: Args, model: Model, cameras: dict[str, dict]) -> None:
    import torch

    from examples.bimanual_data_collection.post_process_forcegello import ArmKinematics
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    arm, gripper = ArmType.from_string_name(args.arm), GripperType.from_string_name(args.gripper)
    projection = ArmKinematics(
        Path(combine_arm_and_gripper_xml(arm, gripper)),
        args.wrench_damping,
        args.wrench_sign,
        args.gripper_max_width_m,
        args.side,
    )
    torch.set_num_threads(1)
    robot = images = buttons = None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="implicitrdp-infer")
    previous_handler = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        buttons = HandleButtons(args.button_can_channel, args.yellow_button_index, args.button_timeout_s)
        robot = get_yam_robot(
            channel=args.can_channel or f"can_follower_{args.side[0]}",
            arm_type=arm,
            gripper_type=gripper,
            background_control=False,
        )
        images = LiveImages(args, cameras)
        history = History(model.obs_steps, model.obs_meta, model.extended_meta)
        clock = PeriodicClock(args.command_hz, time.monotonic_ns())
        started_ns = last_action_ns = time.monotonic_ns()
        enabled = active = False
        epoch = 0
        future = None
        future_anchor = future_epoch = -1
        previous_target = None
        segment_start = segment_target = None
        segment_anchor = -1
        hold = None
        last_policy_index = -1
        last_camera = None
        camera_tick = 0
        next_report_ns = started_ns
        count = 0
        while args.max_steps is None or count < args.max_steps:
            time.sleep(max(0, (clock.deadline_ns - time.monotonic_ns()) / 1e9))
            index, _, missed = clock.consume(time.monotonic_ns())
            policy_index = index // 3
            policy_tick = policy_index != last_policy_index
            last_policy_index = policy_index
            robot.refresh_state()
            state = robot.get_force_feedback_observations(use_applied_gravity=True)
            now_ns = time.monotonic_ns()
            can_ns = state["can_rx_monotonic_ns"][:6]
            if (
                np.any(can_ns <= 0)
                or np.any(can_ns > now_ns)
                or now_ns - int(np.min(can_ns)) > args.stale_state_s * 1e9
            ):
                raise RuntimeError("follower CAN feedback is stale")
            for _ in range(buttons.consume_toggles(started_ns)):
                enabled = not enabled
                epoch += 1
                history.clear()
                active = False
                previous_target = segment_start = segment_target = None
                last_camera = None
                camera_tick = 0
                last_action_ns = now_ns
                hold = state["joint_pos"].copy() if enabled else None
                print("Yellow: policy enabled" if enabled else "Yellow: gravity-compensation idle", flush=True)
            if not enabled:
                robot.enter_gravity_comp_idle()
                robot.update()
                images.read(now_ns, required=False)
                count += 1
                continue

            # Take a finished prediction only while its t+50 ms target is still
            # ahead. The worker never blocks the 60 Hz motor command loop.
            if future is not None and future.done():
                if future_epoch == epoch:
                    proposed = future.result()
                future = None
                if future_epoch == epoch and index < (future_anchor + 1) * 3:
                    previous_target = hold.copy() if previous_target is None else previous_target
                    segment_start, segment_target = previous_target.copy(), proposed
                    segment_anchor = future_anchor
                    previous_target = proposed.copy()
                    last_action_ns = now_ns
                    active = True
                    if args.debug_actions:
                        print(
                            f"ImplicitRDP obs={future_anchor} target={future_anchor + 1} "
                            f"delta={np.array2string(proposed[:6] - segment_start[:6], precision=4)}",
                            flush=True,
                        )
                elif future_epoch == epoch and args.debug_actions:
                    print(f"ImplicitRDP discarded late prediction for obs {future_anchor}", flush=True)

            if segment_target is not None and index <= (segment_anchor + 1) * 3:
                command_pos, command_vel = interpolate_joint_command(
                    segment_start, segment_target, index, segment_anchor
                )
            else:
                command_pos, command_vel = hold.copy(), np.zeros(7)
            hold = command_pos.copy()
            robot.command_joint_state({"pos": command_pos, "vel": command_vel})
            robot.update()

            if policy_tick:
                frames = images.read(now_ns, required=True)
                if frames is not None:
                    if last_camera is None or camera_tick % 4 == 0:
                        last_camera = frames
                    camera_tick += 1
                    row = projection.project(state)
                    row.update(last_camera)
                    history.append(policy_index, row)
                    inputs = history.input()
                    if inputs is not None and future is None:
                        future_anchor, future_epoch = policy_index, epoch
                        future = pool.submit(model.predict, *inputs)
                timeout = args.policy_timeout_s if active else args.startup_timeout_s
                if now_ns - last_action_ns > timeout * 1e9:
                    raise TimeoutError("no timely ImplicitRDP action; check inference latency and camera/CAN streams")
            if now_ns >= next_report_ns:
                print(
                    f"ImplicitRDP {'running' if active else 'warming up'}: tick={index}, missed={missed}", flush=True
                )
                next_report_ns = now_ns + 5_000_000_000
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        with ExitStack() as cleanup:
            cleanup.callback(signal.signal, signal.SIGTERM, previous_handler)
            if buttons is not None:
                cleanup.callback(buttons.close)
            if images is not None:
                cleanup.callback(images.close)
            if robot is not None:
                cleanup.callback(close_robot, robot)
        pool.shutdown(wait=False, cancel_futures=True)


def main(args: Args) -> None:
    if not (args.implicitrdp_root / "ImplicitRDP" / "policy").is_dir():
        raise FileNotFoundError(f"ImplicitRDP repository not found: {args.implicitrdp_root}")
    sys.path.insert(0, str(args.implicitrdp_root.resolve()))
    model = Model(args.checkpoint, args.device, args.side)
    cameras = validate(args, model)
    print("Loaded 7D joint-command ImplicitRDP: 20 Hz inference, 5 Hz images, 60 Hz commands", flush=True)
    if not args.validate_only:
        run(args, model, cameras)


if __name__ == "__main__":
    main(tyro.cli(Args))
