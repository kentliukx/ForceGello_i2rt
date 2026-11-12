import mujoco
import numpy as np

from examples.bimanual_data_collection.export_implicitrdp import CartesianProjection
from examples.bimanual_data_collection.play import build_scene, set_force_dot, set_robot_pose


def test_replay_force_dot_uses_world_direction_and_caps_display_offset() -> None:
    metadata = {"settings": {"arm": "yam", "follower_gripper": "linear_4310"}}
    model, data = build_scene(metadata)
    projection = CartesianProjection("yam", "linear_4310", damping=0.01, wrench_sign=1.0)
    sample = {
        "follower": {"joint_pos": np.array([0.2, 1.0, 1.2, -0.3, 0.2, 0.4, 0.5])},
        "cartesian": {
            "tcp_pose": np.zeros(6),
            "tcp_velocity": np.zeros(6),
            "tcp_wrench_estimate": np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]),
            "wrench_damping": 0.01,
            "wrench_sign": 1.0,
        },
    }
    set_robot_pose(model, data, "right", sample)
    mujoco.mj_forward(model, data)
    site = data.site("right_grasp_site")
    expected_force = site.xmat.reshape(3, 3) @ np.array([3.0, 4.0, 0.0])
    expected = site.xpos + expected_force / 100.0
    set_force_dot(model, data, "right", sample, projection, cartesian_kp=100.0)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("right_force_dot").xpos, expected, atol=1e-9)
    np.testing.assert_allclose(100.0 * (data.body("right_force_dot").xpos - site.xpos), expected_force, atol=1e-9)

    # Recording wrench_sign changes stored values, but not the virtual controller target.
    opposite_sign = CartesianProjection("yam", "linear_4310", damping=0.01, wrench_sign=-1.0)
    set_force_dot(model, data, "right", sample, opposite_sign, cartesian_kp=100.0)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("right_force_dot").xpos, expected, atol=1e-9)

    sample["cartesian"]["tcp_wrench_estimate"] *= 100
    set_force_dot(model, data, "right", sample, projection, cartesian_kp=100.0)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(np.linalg.norm(data.body("right_force_dot").xpos - site.xpos), 0.20, atol=1e-9)
