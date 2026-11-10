"""Export a multi-rate episode to ImplicitRDP's sensor_message_list pickle layout."""

import bisect
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bimanual_data_collection.recording import read_robot_stream


def preceding(
    samples: list[dict[str, Any]], timestamps: list[int], timestamp_ns: int, max_age_ns: int
) -> dict[str, Any] | None:
    index = bisect.bisect_right(timestamps, timestamp_ns) - 1
    if index < 0 or timestamp_ns - timestamps[index] > max_age_ns:
        return None
    return samples[index]


class CartesianProjection:
    """FK and a damped wrench estimate, expressed at grasp_site in the local TCP frame."""

    def __init__(self, arm: str, gripper: str, damping: float, wrench_sign: float) -> None:
        import mujoco

        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        self.mujoco = mujoco
        xml = combine_arm_and_gripper_xml(ArmType.from_string_name(arm), GripperType.from_string_name(gripper))
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("grasp_site").id
        self.damping = damping
        self.wrench_sign = wrench_sign
        self.arm_qpos = [self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
        self.arm_dofs = [self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 7)]

    def project(self, sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.spatial.transform import Rotation

        cached = sample.get("cartesian")
        if cached is not None and cached["wrench_damping"] == self.damping:
            return (
                cached["tcp_pose"],
                cached["tcp_velocity"],
                cached["tcp_wrench_estimate"] * (self.wrench_sign / cached["wrench_sign"]),
            )
        obs = sample["follower"]
        self.data.qpos[self.arm_qpos] = obs["joint_pos"][:6]
        self.mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.site].reshape(3, 3).copy()
        pose = np.concatenate([self.data.site_xpos[self.site], Rotation.from_matrix(rotation).as_euler("xyz")])
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros_like(linear)
        self.mujoco.mj_jacSite(self.model, self.data, linear, angular, self.site)
        jacobian = np.vstack([linear, angular])[:, self.arm_dofs]
        velocity = jacobian @ obs["joint_vel"][:6]
        residual = obs["joint_eff_external"][:6]
        wrench_base = self.wrench_sign * np.linalg.solve(
            jacobian @ jacobian.T + self.damping**2 * np.eye(6), jacobian @ residual
        )
        wrench_tcp = np.concatenate([rotation.T @ wrench_base[:3], rotation.T @ wrench_base[3:]])
        return pose.astype(np.float32), velocity.astype(np.float32), wrench_tcp.astype(np.float32)


def make_sensor_message(
    timestamp_ns: int,
    robots: dict[str, dict[str, Any]],
    images: dict[str, np.ndarray],
    projections: dict[str, Any],
    gripper_max_width_m: float,
) -> SimpleNamespace:
    fields: dict[str, Any] = {"timestamp": timestamp_ns / 1e9}
    for side in ("left", "right"):
        pose, velocity, wrench = projections[side].project(robots[side])
        fields[f"{side}RobotTCP"] = pose
        fields[f"{side}RobotTCPVel"] = velocity
        fields[f"{side}RobotTCPWrench"] = wrench
        # Width is a documented nominal stroke conversion. No calibrated gripper force is available.
        fields[f"{side}RobotGripperState"] = np.array(
            [robots[side]["follower"]["joint_pos"][6] * gripper_max_width_m, np.nan], dtype=np.float32
        )
    for name, prefix in (
        ("external", "externalCamera"),
        ("left_wrist", "leftWristCamera"),
        ("right_wrist", "rightWristCamera"),
    ):
        fields[f"{prefix}RGB"] = images[name]
        fields[f"{prefix}PointCloud"] = np.zeros((0, 6), dtype=np.float32)
    # The upstream postprocessor reads tactile fields even with only RGB cameras.
    # They are explicitly marked unavailable in export_metadata, not real measurements.
    for side in ("left", "right"):
        for finger in (1, 2):
            fields[f"{side}GripperCameraRGB{finger}"] = np.zeros((24, 32, 3), dtype=np.uint8)
            fields[f"{side}GripperCameraMarker{finger}"] = np.zeros((63, 3), dtype=np.float32)
            fields[f"{side}GripperCameraMarkerOffset{finger}"] = np.zeros((63, 3), dtype=np.float32)
    return SimpleNamespace(**fields)


def main(
    episode: Path,
    output: Path,
    max_robot_age_s: float = 0.05,
    max_camera_age_s: float = 0.10,
    damping: float = 0.01,
    wrench_sign: float = 1.0,
    gripper_max_width_m: float = 0.095,
) -> None:
    """Export one trajectory (20 Hz anchored to left-wrist frames); keep the native 60 Hz source."""
    import cv2

    if output.exists():
        raise FileExistsError(output)
    if min(max_robot_age_s, max_camera_age_s, damping, gripper_max_width_m) <= 0 or wrench_sign not in (-1.0, 1.0):
        raise ValueError("ages/damping/width must be positive; wrench_sign must be +1 or -1")
    metadata = json.loads((episode / "metadata.json").read_text())
    if metadata["status"] == "recording":
        raise ValueError("finish the trajectory before exporting it")
    settings = metadata["settings"]
    robots = {
        side: sorted(read_robot_stream(episode / f"robot_{side}.pkl"), key=lambda row: row["monotonic_ns"])
        for side in ("left", "right")
    }
    robot_times = {side: [row["monotonic_ns"] for row in rows] for side, rows in robots.items()}
    cameras: dict[str, list[dict[str, Any]]] = {name: [] for name in ("left_wrist", "right_wrist", "external")}
    for line in (episode / "cameras.jsonl").read_text().splitlines():
        row = json.loads(line)
        cameras[row["camera"]].append(row)
    for rows in cameras.values():
        rows.sort(key=lambda row: row["monotonic_ns"])
    camera_times = {name: [row["monotonic_ns"] for row in rows] for name, rows in cameras.items()}
    projections = {
        side: CartesianProjection(settings["arm"], settings["follower_gripper"], damping, wrench_sign)
        for side in robots
    }
    messages = []
    alignment = []
    skipped = 0
    for anchor in cameras["left_wrist"]:
        timestamp_ns = anchor["monotonic_ns"]
        states = {
            side: preceding(rows, robot_times[side], timestamp_ns, round(max_robot_age_s * 1e9))
            for side, rows in robots.items()
        }
        frames = {
            name: preceding(rows, camera_times[name], timestamp_ns, round(max_camera_age_s * 1e9))
            for name, rows in cameras.items()
        }
        if any(row is None for row in (*states.values(), *frames.values())):
            skipped += 1
            continue
        images = {}
        for name, row in frames.items():
            image = cv2.imread(str(episode / row["path"]))
            if image is None:
                raise RuntimeError(f"cannot decode {row['path']}")
            images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        messages.append(make_sensor_message(anchor["timestamp_ns"], states, images, projections, gripper_max_width_m))
        alignment.append(
            {
                "monotonic_ns": timestamp_ns,
                "robot_monotonic_ns": {side: row["monotonic_ns"] for side, row in states.items()},
                "camera_monotonic_ns": {name: row["monotonic_ns"] for name, row in frames.items()},
                "command": {side: row["command"] for side, row in states.items()},
            }
        )
    if len(messages) < 2:
        raise ValueError("fewer than two aligned frames; check camera and robot sample timestamps")
    payload = {
        "sensor_message_list": SimpleNamespace(sensorMessages=messages),
        "alignment": alignment,
        "export_metadata": {
            "source": str(episode.resolve()),
            "source_schema": metadata["schema"],
            "skipped_frames": skipped,
            "nominal_hz": settings["camera_hz"],
            "color_order": "RGB",
            "tcp_site": "grasp_site",
            "pose_frame": "each arm base (not a shared calibrated world frame)",
            "velocity_frame": "each arm base; [vx,vy,vz,wx,wy,wz]",
            "wrench_frame": "local TCP",
            "wrench_source": "damped Jacobian projection of effort residual",
            "wrench_sign": wrench_sign,
            "wrench_damping": damping,
            "gripper_width": f"nominal normalized opening * {gripper_max_width_m} m; verify actual stroke",
            "gripper_force": "unavailable (NaN)",
            "tactile": "unavailable (zero placeholders)",
            "point_cloud": "unavailable (empty arrays)",
            "robot_source_hz": settings["control_hz"],
            "alignment": "latest preceding sample only; no future observations",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(output)
    print(f"Saved {len(messages)} frames to {output}; skipped {skipped}; original 60 Hz streams retained.")


if __name__ == "__main__":
    tyro.cli(main)
