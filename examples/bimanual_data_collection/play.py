"""Replay one recorded bimanual episode in MuJoCo and three camera windows.

Run from the repository root with ``python examples/bimanual_data_collection/play.py``.
Only saved files are read; this module never opens a CAN interface.
"""

import json
import math
import multiprocessing as mp
import queue
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.bimanual_data_collection.export_implicitrdp import CartesianProjection
from examples.bimanual_data_collection.recording import read_robot_stream
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

CAMERAS = ("left_wrist", "right_wrist", "external")
MAX_FORCE_DOT_OFFSET_M = 0.20


@dataclass
class Args:
    episode: Path = REPO_ROOT / "data/demonstrations/1"
    speed: float = 1.0
    loop: bool = True
    cartesian_kp: float = 200.0  # Virtual Cartesian stiffness in N/m (translation only).
    validate_only: bool = False  # Load data and build the model without opening any windows.


def _index_at(times: np.ndarray, timestamp_ns: int) -> int:
    """Show the latest recorded sample; hold the first one before its timestamp."""
    return max(0, int(np.searchsorted(times, timestamp_ns, side="right")) - 1)


def load_episode(episode: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = json.loads((episode / "metadata.json").read_text())
    if metadata.get("status") == "recording":
        raise ValueError("finish recording this episode before playing it")
    start_ns = int(metadata["start_monotonic_ns"])
    end_ns = int(metadata["end_monotonic_ns"])
    if end_ns <= start_ns:
        raise ValueError("episode has no positive duration")

    robots = {}
    for side in ("left", "right"):
        rows = list(read_robot_stream(episode / f"robot_{side}.pkl"))
        times = np.array([row["monotonic_ns"] for row in rows], dtype=np.int64)
        if not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError(f"{side} robot stream is empty or not time ordered")
        for row in rows:
            q = np.asarray(row["follower"]["joint_pos"], dtype=float)
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                raise ValueError(f"{side} robot stream contains an invalid follower pose")
        robots[side] = (times, rows)

    cameras: dict[str, list[dict[str, Any]]] = {name: [] for name in CAMERAS}
    for line in (episode / "cameras.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["camera"] not in cameras:
            raise ValueError(f"unknown camera {row['camera']!r}")
        cameras[row["camera"]].append(row)
    tracks = {}
    for name, rows in cameras.items():
        times = np.array([row["monotonic_ns"] for row in rows], dtype=np.int64)
        if not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError(f"{name} camera stream is empty or not time ordered")
        for row in rows:
            frame = (episode / row["path"]).resolve()
            if not frame.is_relative_to(episode.resolve()) or not frame.is_file():
                raise FileNotFoundError(f"missing or invalid {name} frame: {row['path']}")
        tracks[name] = (times, rows)
    return metadata, robots, tracks


def build_scene(metadata: dict[str, Any]) -> tuple[mujoco.MjModel, mujoco.MjData]:
    settings = metadata["settings"]
    arm = ArmType.from_string_name(settings["arm"])
    gripper = GripperType.from_string_name(settings["follower_gripper"])
    single_arm_xml = combine_arm_and_gripper_xml(arm, gripper)
    scene = mujoco.MjSpec()
    scene.modelname = "Bimanual episode replay"
    scene.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, -0.04],
        size=[2, 2, 0.1],
        rgba=[0.8, 0.8, 0.8, 1],
        contype=0,
        conaffinity=0,
    )
    # +Y points left: the right arm base is 0.60 m to the right of the left base.
    # Their orientations are assumed parallel; other extrinsics are not calibrated.
    for side, y in (("left", 0.3), ("right", -0.3)):
        mount = scene.worldbody.add_frame(name=f"{side}_mount", pos=[0, y, 0])
        scene.attach(mujoco.MjSpec.from_file(single_arm_xml), prefix=f"{side}_", frame=mount)
        marker = scene.worldbody.add_body(name=f"{side}_force_dot", mocap=True)
        marker.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],
            rgba=[0.0, 0.85, 1.0, 1.0] if side == "left" else [1.0, 0.25, 0.1, 1.0],
            contype=0,
            conaffinity=0,
        )
    model = scene.compile()
    for side in ("left", "right"):
        for number in range(1, 8):
            model.joint(f"{side}_joint{number}")
    return model, mujoco.MjData(model)


def set_robot_pose(model: mujoco.MjModel, data: mujoco.MjData, side: str, sample: dict[str, Any]) -> None:
    q = np.asarray(sample["follower"]["joint_pos"], dtype=float)
    for number in range(1, 7):
        joint = model.joint(f"{side}_joint{number}")
        data.qpos[joint.qposadr[0]] = q[number - 1]
    # The recorded gripper is normalized [0, 1]; the MJCF finger sliders use metres.
    opening = float(np.clip(q[6], 0.0, 1.0))
    for number in (7, 8):
        joint = model.joint(f"{side}_joint{number}")
        low, high = joint.range
        data.qpos[joint.qposadr[0]] = low + opening * (high - low)


def set_force_dot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    sample: dict[str, Any],
    projection: CartesianProjection,
    cartesian_kp: float,
) -> None:
    """Show the virtual target whose Cartesian Kp force opposes the estimated external force."""
    _, _, wrench_tcp = projection.project(sample)
    site = data.site(f"{side}_grasp_site")
    # At equilibrium, tau_measured - tau_gravity = -J.T @ force_on_robot.
    # F_cmd = Kp * (target - TCP) = -force_on_robot; undo the recording's wrench_sign.
    force_on_robot = -(site.xmat.reshape(3, 3) @ np.asarray(wrench_tcp[:3], dtype=float)) / projection.wrench_sign
    offset = -force_on_robot / cartesian_kp
    length = float(np.linalg.norm(offset))
    if not np.isfinite(length):
        raise ValueError(f"{side} estimated TCP force is not finite")
    if length > MAX_FORCE_DOT_OFFSET_M:
        offset *= MAX_FORCE_DOT_OFFSET_M / length
    marker_id = model.body(f"{side}_force_dot").mocapid[0]
    data.mocap_pos[marker_id] = site.xpos + offset


def show_camera(episode: Path, name: str, row: dict[str, Any], elapsed_s: float, paused: bool) -> None:
    import cv2

    image = cv2.imread(str(episode / row["path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode {row['path']}")
    label = f"{name}  {elapsed_s:5.2f}s" + ("  PAUSED" if paused else "")
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 390), 36), (0, 0, 0), -1)
    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imshow(name, image)


def camera_worker(
    episode: Path,
    cameras: dict[str, Any],
    states: Any,
    commands: Any,
    stop: Any,
) -> None:
    import cv2

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    last_camera = dict.fromkeys(cameras, -1)
    last_paused = None
    try:
        for window_index, name in enumerate(CAMERAS):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 480, 360)
            cv2.moveWindow(name, 20 + window_index * 500, 40)
        while not stop.is_set():
            try:
                state = states.get(timeout=0.01)
                while True:
                    try:
                        state = states.get_nowait()
                    except queue.Empty:
                        break
            except queue.Empty:
                state = None
            if state is not None:
                timestamp_ns, elapsed_s, paused = state
                for name, (times, rows) in cameras.items():
                    index = _index_at(times, timestamp_ns)
                    if index != last_camera[name] or paused != last_paused:
                        show_camera(episode, name, rows[index], elapsed_s, paused)
                        last_camera[name] = index
                last_paused = paused
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                commands.put("quit")
                break
            if key == ord("r"):
                commands.put("restart")
            elif key == ord(" "):
                commands.put("pause")
    except BaseException as error:
        if not stop.is_set():
            commands.put(("error", repr(error)))
        raise
    finally:
        cv2.destroyAllWindows()


def _send_latest(states: Any, state: tuple[int, float, bool]) -> None:
    try:
        states.put_nowait(state)
    except queue.Full:
        try:
            states.get_nowait()
        except queue.Empty:
            return
        try:
            states.put_nowait(state)
        except queue.Full:
            pass


def main(args: Args) -> None:
    if not math.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("--speed must be finite and positive")
    if not math.isfinite(args.cartesian_kp) or args.cartesian_kp <= 0:
        raise ValueError("--cartesian-kp must be finite and positive (N/m)")
    episode = args.episode.expanduser().resolve()
    metadata, robots, cameras = load_episode(episode)
    model, data = build_scene(metadata)
    cartesian = metadata.get("cartesian", {})
    projections = {
        side: CartesianProjection(
            metadata["settings"]["arm"],
            metadata["settings"]["follower_gripper"],
            float(cartesian.get("damping", 0.01)),
            float(cartesian.get("sign", 1.0)),
        )
        for side in robots
    }
    duration_s = (int(metadata["end_monotonic_ns"]) - int(metadata["start_monotonic_ns"])) / 1e9
    print(f"Episode: {episode} ({duration_s:.2f}s)", flush=True)
    for side, (times, _) in robots.items():
        print(f"  {side}: {len(times)} robot samples", flush=True)
    for name, (times, _) in cameras.items():
        print(f"  {name}: {len(times)} camera frames", flush=True)
    if args.validate_only:
        for side, (_, rows) in robots.items():
            set_robot_pose(model, data, side, rows[0])
        mujoco.mj_forward(model, data)
        for side, (_, rows) in robots.items():
            set_force_dot(model, data, side, rows[0], projections[side], args.cartesian_kp)
        mujoco.mj_forward(model, data)
        print("Episode and bimanual MuJoCo model OK", flush=True)
        return

    context = mp.get_context("spawn")
    states = context.Queue(maxsize=2)
    commands = context.Queue()
    stop = context.Event()
    camera_process = context.Process(
        target=camera_worker, args=(episode, cameras, states, commands, stop), name="replay-cameras"
    )
    camera_process.start()
    previous_sigterm = signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    try:
        print("Camera window keys: Space pause/resume, R restart, Q or Esc quit. Close MuJoCo to quit.", flush=True)
        start_wall = time.monotonic()
        paused_s = 0.0
        paused = False
        last_robot = dict.fromkeys(robots, -1)
        last_cycle = -1
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.35]
                viewer.cam.distance = 1.7
                viewer.cam.azimuth = 130
                viewer.cam.elevation = -25
            while viewer.is_running() and not stop.is_set():
                now = time.monotonic()
                while True:
                    try:
                        command = commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == "quit":
                        return
                    if isinstance(command, tuple) and command[0] == "error":
                        raise RuntimeError(f"camera windows failed: {command[1]}")
                    if command == "restart":
                        start_wall = now
                        paused_s = 0.0
                        paused = False
                        last_cycle = -1
                    elif command == "pause":
                        if paused:
                            start_wall = now - paused_s / args.speed
                            paused = False
                        else:
                            paused_s = (now - start_wall) * args.speed
                            paused = True
                if camera_process.exitcode is not None:
                    if camera_process.exitcode == 0:
                        return
                    raise RuntimeError(f"camera windows exited unexpectedly ({camera_process.exitcode})")
                total_s = paused_s if paused else (now - start_wall) * args.speed
                cycle = int(total_s / duration_s)
                if not args.loop and cycle:
                    break
                elapsed_s = total_s % duration_s
                if cycle != last_cycle:
                    last_robot = dict.fromkeys(robots, -1)
                    last_cycle = cycle
                    print(f"Replay loop {cycle + 1}", flush=True)
                timestamp_ns = int(metadata["start_monotonic_ns"]) + round(elapsed_s * 1e9)

                changed = False
                current_rows = {}
                for side, (times, rows) in robots.items():
                    index = _index_at(times, timestamp_ns)
                    if index != last_robot[side]:
                        with viewer.lock():
                            set_robot_pose(model, data, side, rows[index])
                        last_robot[side] = index
                        changed = True
                    current_rows[side] = rows[index]
                if changed:
                    with viewer.lock():
                        mujoco.mj_forward(model, data)
                        for side, sample in current_rows.items():
                            set_force_dot(model, data, side, sample, projections[side], args.cartesian_kp)
                        mujoco.mj_forward(model, data)
                    viewer.sync()
                _send_latest(states, (timestamp_ns, elapsed_s, paused))
                time.sleep(0.004)
    except KeyboardInterrupt:
        print("Playback stopped", flush=True)
    finally:
        stop.set()
        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()
            camera_process.join(timeout=2)
        states.close()
        commands.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main(tyro.cli(Args))
