"""Turn a base-frame TCP displacement into a follower joint target."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose9_matrix(pose: np.ndarray) -> np.ndarray:
    """Decode [xyz, first rotation column, second rotation column]."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (9,) or not np.all(np.isfinite(pose)):
        raise ValueError("expected a finite 9D TCP pose")
    x, y = pose[3:6], pose[6:9]
    rotation = np.column_stack((x, y, np.cross(x, y)))
    if np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-3:
        raise ValueError("invalid TCP pose rotation")
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pose[:3]
    return matrix


class TCPDeltaIK:
    def __init__(self, model_path: str) -> None:
        import mink
        from i2rt.robots.kinematics import Kinematics

        self.kinematics = Kinematics(model_path, "grasp_site")
        model = self.kinematics._configuration.model
        self.arm_qpos_indices = np.array(
            [model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)], dtype=int
        )
        self.nq = model.nq
        self.limits = [mink.ConfigurationLimit(model)]

    def solve(
        self, action: np.ndarray, measured_joint_pos: np.ndarray, measured_pose: np.ndarray
    ) -> np.ndarray:
        """Return pos7. Rebase each predicted displacement on the current observation."""
        action = np.asarray(action, dtype=np.float64)
        current = np.asarray(measured_joint_pos, dtype=np.float64)
        if (
            action.shape != (7,) or current.shape != (7,)
            or not np.all(np.isfinite(action)) or not np.all(np.isfinite(current))
        ):
            raise ValueError("expected finite TCP delta7 and measured joint pos7")
        target = pose9_matrix(measured_pose)
        target[:3, 3] += action[:3]
        target[:3, :3] = Rotation.from_euler('xyz', action[3:6]).as_matrix() @ target[:3, :3]
        seed = np.zeros(self.nq)
        seed[self.arm_qpos_indices] = current[:6]
        success, solution = self.kinematics.ik(
            target, "grasp_site", init_q=seed, limits=self.limits,
            max_iters=60, pos_threshold=1e-3, ori_threshold=1e-2,
        )
        arm_target = np.asarray(solution, dtype=np.float64)[self.arm_qpos_indices]
        if not success or not np.all(np.isfinite(arm_target)):
            raise ValueError("TCP inverse kinematics did not converge")
        return np.r_[arm_target, np.clip(action[6], 0.0, 1.0)]
