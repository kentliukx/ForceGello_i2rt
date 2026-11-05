import numpy as np

from examples.minimum_gello.minimum_gello import _feedback_source, _leader_feedback_torque


def test_angle_mode_tracks_follower_with_leader_base_kp() -> None:
    source = _feedback_source(
        "angle",
        leader_pos=np.array([0.3, -0.2]),
        follower_pos=np.array([0.2, -0.1]),
        follower_eff=np.array([99.0, 99.0]),
        leader_kp=np.array([80.0, 10.0]),
    )

    np.testing.assert_allclose(source, [8.0, -1.0])
    np.testing.assert_allclose(
        _leader_feedback_torque(
            np.array([source[0], source[1], 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, -0.1, 0.0, 0.0, 0.0, 0.0]),
            bilateral_kp=0.1,
            bilateral_alpha=1.0,
            bilateral_beta=1.0,
        )[:2],
        np.array([-0.4, 0.05]) * (2.0 / np.pi) * np.arctan(0.1),
    )


def test_torque_mode_keeps_follower_torque_signal() -> None:
    source = _feedback_source(
        "torque",
        leader_pos=np.array([0.3]),
        follower_pos=np.array([0.2]),
        follower_eff=np.array([1.5]),
        leader_kp=np.array([80.0]),
    )

    np.testing.assert_allclose(source, [1.5])


def test_speed_attenuation_is_per_joint_and_direction_independent() -> None:
    effort = np.array([2.0, 2.0, 2.0, -2.0, 0.0, 0.0])
    velocity = np.array([0.0, 1.0, -1.0, -3.0, 0.0, 0.0])

    torque = _leader_feedback_torque(
        effort,
        velocity,
        np.ones(6),
        bilateral_kp=0.5,
        bilateral_alpha=1.0,
        bilateral_beta=1.0,
    )

    np.testing.assert_allclose(torque, [-0.5, -0.25, -0.25, 0.125, 0.0, 0.0])


def test_zero_alpha_disables_attenuation_and_preserves_torque_caps() -> None:
    torque = _leader_feedback_torque(
        np.array([100.0, -100.0, 0.0, 100.0, -100.0, 0.0]),
        np.array([5.0, -5.0, 0.0, 5.0, -5.0, 0.0]),
        np.ones(6),
        bilateral_kp=1.0,
        bilateral_alpha=0.0,
        bilateral_beta=1.0,
    )

    np.testing.assert_allclose(torque, [-10.0, 10.0, 0.0, -5.0, 5.0, 0.0])


def test_gain_above_one_still_respects_torque_caps() -> None:
    torque = _leader_feedback_torque(
        np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0]),
        np.zeros(6),
        np.ones(6),
        bilateral_kp=10.0,
        bilateral_alpha=1.0,
        bilateral_beta=1.0,
    )

    np.testing.assert_allclose(torque, [-10.0, 10.0, 0.0, -5.0, 5.0, 0.0])


def test_angle_correction_is_zero_at_alignment_and_increases_with_error() -> None:
    effort = np.ones(6)
    velocity = np.zeros(6)
    angle_error = np.array([0.0, 0.1, -1.0, 0.1, -1.0, 10.0])

    torque = _leader_feedback_torque(effort, velocity, angle_error, 1.0, 0.0, 1.0)
    expected = -(2.0 / np.pi) * np.arctan(np.abs(angle_error))

    np.testing.assert_allclose(torque, expected)
    assert torque[0] == 0.0


def test_beta_zero_disables_feedback_in_both_modes() -> None:
    torque = _leader_feedback_torque(np.ones(6), np.zeros(6), np.ones(6), 100.0, 0.0, 0.0)

    np.testing.assert_allclose(torque, np.zeros(6))
