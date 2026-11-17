"""Convert ForceGello episodes to a reusable symmetric dual-arm Zarr dataset."""

import bisect
import json
import math
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import tyro
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")
CAMERAS = ("external", "left_wrist", "right_wrist")
IMAGE_KEYS = tuple(f"{name}_img" for name in CAMERAS)
I2RT_ROOT = Path(__file__).resolve().parents[2]


def read_robot_stream(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("rb") as stream:
        while True:
            try:
                rows.append(pickle.load(stream))
            except EOFError:
                break
    return sorted(rows, key=lambda row: row["monotonic_ns"])


class ArmKinematics:
    """Compute FK, base-frame Jacobian and local TCP wrench at every aligned sample."""

    def __init__(
        self,
        xml_path: Path,
        damping: float,
        wrench_sign: float,
        gripper_max_width_m: float,
        side: str,
    ) -> None:
        self.side = side
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("grasp_site").id
        self.qpos_indices = [
            self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)
        ]
        self.dof_indices = [
            self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 7)
        ]
        self.damping, self.wrench_sign = damping, wrench_sign
        self.gripper_max_width_m = gripper_max_width_m

    def pose(self, joint_pos: np.ndarray) -> np.ndarray:
        """Return grasp_site position and first two rotation columns in the arm base."""
        q = np.asarray(joint_pos, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("TCP forward kinematics requires six finite joint angles")
        self.data.qpos[self.qpos_indices] = q
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.site].reshape(3, 3)
        return np.concatenate(
            [self.data.site_xpos[self.site], rotation[:, :2].T.flatten()]
        ).astype(np.float32)

    def project(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        q = np.asarray(obs["joint_pos"], dtype=np.float64)
        dq = np.asarray(obs["joint_vel"], dtype=np.float64)
        measured = np.asarray(obs["joint_eff_measured"], dtype=np.float64)[:6]
        residual = np.asarray(obs["joint_eff_external"], dtype=np.float64)[:6]
        if (
            q.shape != (7,)
            or dq.shape != (7,)
            or measured.shape != (6,)
            or residual.shape != (6,)
        ):
            raise ValueError(
                "expected follower q/dq with 7 entries and arm effort with 6 entries"
            )
        if not all(np.all(np.isfinite(value)) for value in (q, dq, measured, residual)):
            raise ValueError("non-finite follower state or effort")
        pose = self.pose(q[:6])
        rotation = self.data.site_xmat[self.site].reshape(3, 3).copy()
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros_like(linear)
        mujoco.mj_jacSite(self.model, self.data, linear, angular, self.site)
        jacobian = np.vstack([linear, angular])[:, self.dof_indices]
        # tau_residual ~= J.T @ wrench_base; damping bounds amplification near singularities.
        wrench_base = self.wrench_sign * np.linalg.solve(
            jacobian @ jacobian.T + self.damping**2 * np.eye(6), jacobian @ residual
        )
        wrench_tcp = np.concatenate(
            [rotation.T @ wrench_base[:3], rotation.T @ wrench_base[3:]]
        )
        width = np.array([q[6] * self.gripper_max_width_m])
        values = {
            "tcp_pose": pose,
            "tcp_vel": (jacobian @ dq[:6]).astype(np.float32),
            "tcp_wrench": wrench_tcp.astype(np.float32),
            "gripper_width": width.astype(np.float32),
            "joint_pos": q.astype(np.float32),
            "joint_vel": dq.astype(np.float32),
            "joint_effort": measured.astype(np.float32),
            "joint_effort_external": residual.astype(np.float32),
            "jacobian": jacobian.astype(np.float32),
        }
        return {f"{self.side}_robot_{key}": value for key, value in values.items()}


def leader_target(packet: dict[str, Any]) -> np.ndarray:
    """Keep the recorded leader state as an observation, separate from follower commands."""
    q = np.asarray(packet["leader_joint_pos_with_gripper"], dtype=np.float32)
    dq = np.asarray(packet["leader"]["joint_vel"], dtype=np.float32)
    if (
        q.shape != (7,)
        or dq.shape != (6,)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(dq))
    ):
        raise ValueError(
            "leader command requires finite q7 (arm + handle gripper) and six measured joint velocities"
        )
    return np.concatenate([q, dq, np.zeros(1, dtype=np.float32)])


def follower_command(packet: dict[str, Any]) -> tuple[np.ndarray, np.uint8]:
    """Read the submitted position target, or mark an idle frame as invalid."""
    submitted = packet.get("command")
    if submitted is None:
        pos = packet["follower"]["joint_pos"]
        valid = 0
    else:
        pos = submitted["pos"]
        valid = 1
    pos = np.asarray(pos, dtype=np.float32)
    if pos.shape != (7,) or not np.all(np.isfinite(pos)):
        raise ValueError("follower command requires seven finite joint positions")
    return pos.copy(), np.uint8(valid)


def tcp_command_delta(
    observed_pose: np.ndarray, target_pose: np.ndarray, gripper: float, valid: bool
) -> np.ndarray:
    """Same-frame base-frame TCP command offset; inactive arms have zero motion."""
    if not valid:
        return np.array([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)
    obs_x, obs_y = observed_pose[3:6], observed_pose[6:9]
    tgt_x, tgt_y = target_pose[3:6], target_pose[6:9]
    obs_rotation = np.column_stack((obs_x, obs_y, np.cross(obs_x, obs_y)))
    tgt_rotation = np.column_stack((tgt_x, tgt_y, np.cross(tgt_x, tgt_y)))
    relative = tgt_rotation @ obs_rotation.T
    return np.r_[
        target_pose[:3] - observed_pose[:3],
        Rotation.from_matrix(relative).as_euler("xyz"),
        gripper,
    ].astype(np.float32)


def preceding_index(
    timestamps: list[int], timestamp_ns: int, max_age_ns: int
) -> int | None:
    index = bisect.bisect_right(timestamps, timestamp_ns) - 1
    if index < 0 or timestamp_ns - timestamps[index] > max_age_ns:
        return None
    return index


def aligned_segments(
    robots: dict[str, list[dict[str, Any]]],
    cameras: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    hz: float,
    robot_age_ns: int,
    camera_age_ns: int,
) -> Iterator[list[tuple[int, tuple[int, ...], tuple[int, ...]]]]:
    """Align both measured streams causally; keep missing-state gaps out of training chunks."""
    robot_times = {
        side: [int(row["monotonic_ns"]) for row in robots[side]] for side in SIDES
    }
    camera_times = {
        name: [int(row["monotonic_ns"]) for row in cameras[name]] for name in CAMERAS
    }
    all_times = [*robot_times.values(), *camera_times.values()]
    if any(not values for values in all_times):
        raise ValueError(
            "each episode needs both robot streams and all three RGB streams"
        )
    origin = int(metadata["start_monotonic_ns"])
    start = max(*(values[0] for values in all_times), origin)
    end = min(
        *(values[-1] + 1 for values in robot_times.values()),
        int(metadata["end_monotonic_ns"]),
    )
    tick = max(0, math.ceil((start - origin) * hz / 1e9))
    segment = []
    while True:
        timestamp_ns = origin + round(tick * 1e9 / hz)
        if timestamp_ns >= end:
            break
        indices = tuple(
            preceding_index(robot_times[side], timestamp_ns, robot_age_ns)
            for side in SIDES
        )
        frames = tuple(
            preceding_index(camera_times[name], timestamp_ns, camera_age_ns)
            for name in CAMERAS
        )
        if any(index is None for index in (*indices, *frames)):
            if len(segment) >= 2:
                yield segment
            segment = []
        else:
            if segment and any(
                index != previous + 1
                for index, previous in zip(indices, segment[-1][1])
            ):
                # Keep each segment on a continuous robot sample sequence.
                if len(segment) >= 2:
                    yield segment
                segment = []
            segment.append((timestamp_ns, indices, frames))
        tick += 1
    if len(segment) >= 2:
        yield segment


class ReplayWriter:
    def __init__(self, path: Path, attrs: dict[str, Any]) -> None:
        self.root = zarr.open_group(str(path), mode="w-")
        self.root.attrs.update(attrs)
        self.data = self.root.create_group("data")
        self.ends = self.root.create_group("meta").create_dataset(
            "episode_ends", shape=(0,), chunks=(1024,), dtype="int64", compressor=None
        )
        self.count = 0

    def append(self, rows: list[dict[str, np.ndarray]]) -> None:
        for key in rows[0]:
            values = np.stack([row[key] for row in rows])
            if key not in self.data:
                chunk_length = 1 if key in IMAGE_KEYS else 256
                self.data.create_dataset(
                    key,
                    shape=(0,) + values.shape[1:],
                    chunks=(chunk_length,) + values.shape[1:],
                    dtype=values.dtype,
                    compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
                )
            self.data[key].append(values)
        self.count += len(rows)

    def end_episode(self) -> None:
        self.ends.append(np.array([self.count], dtype=np.int64))


def load_model(
    metadata: dict[str, Any], i2rt_root: Path, robot_xml: Path | None
) -> Path:
    if robot_xml is not None:
        return robot_xml
    # Only compose the model. No robot is constructed and no CAN interface is opened.
    sys.path.insert(0, str(i2rt_root.resolve()))
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    settings = metadata["settings"]
    return Path(
        combine_arm_and_gripper_xml(
            ArmType.from_string_name(settings["arm"]),
            GripperType.from_string_name(settings["follower_gripper"]),
        )
    )


def main(
    input_dir: Path,
    output_dir: Path = I2RT_ROOT / "data/forcegello_zarr",
    hz: float = 60.0,
    image_width: int = 320,
    image_height: int = 240,
    damping: float = 0.01,
    wrench_sign: float = 1.0,
    gripper_max_width_m: float = 0.095,
    max_robot_age_s: float = 0.05,
    max_camera_age_s: float = 0.10,
    i2rt_root: Path = I2RT_ROOT,
    robot_xml: Path | None = None,
    batch_size: int = 32,
) -> None:
    """Export same-timestamp dual-arm observations and submitted follower targets."""
    if not all(
        math.isfinite(x) and x > 0
        for x in (hz, damping, gripper_max_width_m, max_robot_age_s, max_camera_age_s)
    ):
        raise ValueError(
            "rate, damping, gripper stroke and maximum sample ages must be finite and positive"
        )
    if wrench_sign not in (-1.0, 1.0) or min(image_width, image_height, batch_size) < 1:
        raise ValueError("invalid wrench sign, image dimensions or batch size")
    if (output_dir / "replay_buffer.zarr").exists():
        raise FileExistsError(output_dir / "replay_buffer.zarr")
    candidates = (
        [input_dir]
        if (input_dir / "metadata.json").is_file()
        else sorted(
            (p for p in input_dir.iterdir() if p.is_dir()),
            key=lambda p: (
                not p.name.isdigit(),
                int(p.name) if p.name.isdigit() else p.name,
            ),
        )
    )
    episodes = []
    for path in candidates:
        if path.name.endswith(".partial") or not (path / "metadata.json").is_file():
            continue
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["status"] == "complete":
            episodes.append((path, metadata))
    if not episodes:
        raise ValueError("no complete ForceGello episodes found")
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "replay_buffer.zarr.partial"
    writer = ReplayWriter(
        partial,
        {
            "schema": "forcegello.bimanual_command.v1",
            "sample_hz": hz,
            "image_color": "RGB",
            "camera_order": list(CAMERAS),
            "image_shape": [image_height, image_width, 3],
            "command": "submitted follower position target at the same aligned timestamp; per-arm pos7, global left7+right7; idle entries copy measured follower position and have command_valid=0",
            "command_layout": [
                "q1..q6",
                "normalized gripper target",
            ],
            "tcp_command": "FK of submitted follower q1..q6 in arm base; [x,y,z,R[:,0],R[:,1]], m and rotation columns",
            "tcp_command_delta": "same-time target minus follower TCP observation in arm base; [dx,dy,dz,roll,pitch,yaw,absolute gripper]; R_delta=Rz(yaw) Ry(pitch) Rx(roll); idle motion is exactly zero",
            "training_action": "future training action not stored; downstream loaders choose their prediction horizon and rebase target commands on the selected observation",
            "robot_order": list(SIDES),
            "pose": "each arm base frame; [x,y,z,R[:,0],R[:,1]], m and rotation columns",
            "velocity": "each arm base frame; [vx,vy,vz,wx,wy,wz], m/s and rad/s",
            "wrench": "local grasp_site frame; [Fx,Fy,Fz,Tx,Ty,Tz], N and Nm; actuator-residual estimate",
            "wrench_equation": "F_base = sign * solve(J J^T + damping^2 I, J tau_residual); rotate force and moment into TCP",
            "effort_residual": "recorded measured arm effort minus last submitted gravity term; includes friction/dynamics/tracking",
            "jacobian": "base-frame grasp_site geometric Jacobian; rows [linear,angular], columns joint1..joint6",
            "gripper": "normalized measured opening * gripper_max_width_m; nominal conversion",
            "gripper_max_width_m": gripper_max_width_m,
            "wrench_damping": damping,
            "wrench_sign": wrench_sign,
            "alignment": "60 Hz by default; latest preceding robot/image only; split gaps; no teleoperation-state filter",
            "max_robot_age_s": max_robot_age_s,
            "max_camera_age_s": max_camera_age_s,
        },
    )
    sources = []
    for episode, metadata in episodes:
        robots = {
            side: read_robot_stream(episode / f"robot_{side}.pkl") for side in SIDES
        }
        cameras: dict[str, list[dict[str, Any]]] = {name: [] for name in CAMERAS}
        for line in (episode / "cameras.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["camera"] in cameras:
                cameras[row["camera"]].append(row)
        for rows in cameras.values():
            rows.sort(key=lambda row: row["monotonic_ns"])
        xml = load_model(metadata, i2rt_root, robot_xml)
        kinematics = {
            side: ArmKinematics(xml, damping, wrench_sign, gripper_max_width_m, side)
            for side in SIDES
        }
        image_cache = {}
        for segment in aligned_segments(
            robots,
            cameras,
            metadata,
            hz,
            round(max_robot_age_s * 1e9),
            round(max_camera_age_s * 1e9),
        ):
            pending = []
            for timestamp_ns, indices, frame_indices in segment:
                row = {}
                for side, index in zip(SIDES, indices):
                    packet = robots[side][index]
                    row.update(kinematics[side].project(packet["follower"]))
                    leader = leader_target(packet)
                    row[f"{side}_leader_joint_pos"] = leader[:7].copy()
                    row[f"{side}_leader_joint_vel"] = leader[7:].copy()
                    command, valid = follower_command(packet)
                    row[f"{side}_robot_command"] = command
                    row[f"{side}_robot_command_valid"] = np.asarray(valid)
                    target_pose = kinematics[side].pose(command[:6])
                    row[f"{side}_robot_tcp_command"] = target_pose
                    row[f"{side}_robot_tcp_command_delta"] = tcp_command_delta(
                        row[f"{side}_robot_tcp_pose"], target_pose, command[6], bool(valid)
                    )
                row["command"] = np.concatenate(
                    [row[f"{side}_robot_command"] for side in SIDES]
                )
                robot = robots["right"][indices[1]]
                row["monotonic_ns"] = np.asarray(timestamp_ns, dtype=np.int64)
                row["timestamp"] = np.asarray(
                    (robot["timestamp_ns"] + timestamp_ns - robot["monotonic_ns"])
                    / 1e9,
                    dtype=np.float64,
                )
                row["robot_source_monotonic_ns"] = np.asarray(
                    [
                        robots[side][index]["monotonic_ns"]
                        for side, index in zip(SIDES, indices)
                    ],
                    dtype=np.int64,
                )
                source_times = []
                for name, key, frame_index in zip(CAMERAS, IMAGE_KEYS, frame_indices):
                    frame = cameras[name][frame_index]
                    source_times.append(frame["monotonic_ns"])
                    path = (episode / frame["path"]).resolve()
                    if not path.is_relative_to(episode.resolve()):
                        raise ValueError(f"image path escapes episode: {path}")
                    if name not in image_cache or image_cache[name][0] != path:
                        image = cv2.imread(str(path))
                        if image is None:
                            raise ValueError(f"cannot decode {path}")
                        image = cv2.cvtColor(
                            cv2.resize(image, (image_width, image_height)),
                            cv2.COLOR_BGR2RGB,
                        )
                        image_cache[name] = (path, image)
                    row[key] = image_cache[name][1]
                row["camera_source_monotonic_ns"] = np.asarray(
                    source_times, dtype=np.int64
                )
                pending.append(row)
                if len(pending) == batch_size:
                    writer.append(pending)
                    pending.clear()
            if pending:
                writer.append(pending)
            writer.end_episode()
            sources.append(
                {
                    "episode": str(episode.resolve()),
                    "start_monotonic_ns": segment[0][0],
                    "end_monotonic_ns": segment[-1][0],
                    "settings": metadata["settings"],
                }
            )
        print(f"Processed {episode}: {writer.count} total samples", flush=True)
    if writer.count == 0:
        raise ValueError(
            "no segments contain at least two aligned samples; check clocks and sample ages"
        )
    writer.root.attrs["sources"] = sources
    writer.root.attrs["episode_count"] = len(sources)
    partial.rename(output_dir / "replay_buffer.zarr")
    print(
        f"Saved {writer.count} samples / {len(sources)} segments to {output_dir / 'replay_buffer.zarr'}"
    )


if __name__ == "__main__":
    tyro.cli(main)
