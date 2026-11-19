"""Check the observation and motor-command clocks without hardware."""

import numpy as np

from examples.implicitrdp_policy.run import History, interpolate_joint_command


def test_visual_delay_and_joint_command_interpolation() -> None:
    history = History(
        5,
        {"external_img": {"type": "rgb"}, "right_robot_joint_pos": {"type": "low_dim"}},
        {"right_robot_joint_pos": {"type": "low_dim"}},
    )
    for tick in range(6):
        history.append(
            tick,
            {
                "external_img": np.full((3, 4, 4), (tick // 4) * 4, dtype=np.float32),
                "right_robot_joint_pos": np.full(7, tick, dtype=np.float32),
            },
        )
    obs, extended = history.input()
    np.testing.assert_array_equal(obs["right_robot_joint_pos"][:, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(obs["external_img"][:, 0, 0, 0], [0, 0, 0, 0, 4])
    np.testing.assert_array_equal(extended["right_robot_joint_pos"][:, 0], [1, 2, 3, 4, 5])
    start = np.zeros(7)
    target = np.array([0.06, 0, 0, 0, 0, 0, 0.5])
    pos, vel = interpolate_joint_command(start, target, 2, 0)
    np.testing.assert_allclose(pos, target * 2 / 3)
    np.testing.assert_allclose(vel, [1.2, 0, 0, 0, 0, 0, 0])
    pos, _ = interpolate_joint_command(start, target, 3, 0)
    np.testing.assert_allclose(pos, target)
