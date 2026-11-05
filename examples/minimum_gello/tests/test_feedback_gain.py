from unittest.mock import patch

import numpy as np

from examples.minimum_gello.minimum_gello import _feedback_source, _leader_angle_target, _leader_feedback_torque


def test_angle_and_torque_modes_pass_entry_point_validation() -> None:
    from examples.minimum_gello import minimum_gello

    for mode in ("angle", "torque"):
        args = minimum_gello.Args(mode="leader", feedback_mode=mode)
        with patch.object(minimum_gello, "run_leader") as run_leader:
            minimum_gello.main(args)
        run_leader.assert_called_once_with(args)


def test_angle_mode_tracks_follower_with_leader_base_kp() -> None:
    source = _feedback_source(
        "angle",
        leader_pos=np.array([0.3, -0.2]),
        target_pos=np.array([0.2, -0.1]),
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
        np.array([-0.4, 0.05]) * (2.0 / np.pi) * np.arctan(10.0),
    )


def test_torque_mode_keeps_follower_torque_signal() -> None:
    source = _feedback_source(
        "torque",
        leader_pos=np.array([0.3]),
        target_pos=np.array([0.2]),
        follower_eff=np.array([1.5]),
        leader_kp=np.array([80.0]),
    )

    np.testing.assert_allclose(source, [1.5])


def test_angle_target_leads_follower_by_20_ms_of_current_leader_velocity() -> None:
    follower_pos = np.array([0.2, -0.1, 0.0, 0.0, 0.0, 0.0])
    leader_vel = np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    target = _leader_angle_target(follower_pos, leader_vel)
    np.testing.assert_allclose(target, [0.22, -0.14, 0.0, 0.0, 0.0, 0.0])

    # With equal measured angles, a moving leader feels the predicted target.
    error = follower_pos - target
    source = _feedback_source("angle", follower_pos, target, np.zeros(6), np.array([80.0, 10.0, 0, 0, 0, 0]))
    np.testing.assert_allclose(source[:2], [-1.6, 0.4])
    feedback = _leader_feedback_torque(
        source,
        np.zeros(6),
        error,
        bilateral_kp=0.1,
        bilateral_alpha=0.0,
        bilateral_beta=1.0,
    )
    assert feedback[0] > 0.0
    assert feedback[1] < 0.0
    np.testing.assert_allclose(_feedback_source("angle", target, target, np.zeros(6), np.ones(6)), np.zeros(6))


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

    correction = (2.0 / np.pi) * np.arctan(100.0)
    np.testing.assert_allclose(torque, np.array([-1.0, -0.5, -0.5, 0.25, 0.0, 0.0]) * correction)


def test_zero_alpha_disables_attenuation_without_feedback_torque_caps() -> None:
    torque = _leader_feedback_torque(
        np.array([100.0, -100.0, 0.0, 100.0, -100.0, 0.0]),
        np.array([5.0, -5.0, 0.0, 5.0, -5.0, 0.0]),
        np.ones(6),
        bilateral_kp=1.0,
        bilateral_alpha=0.0,
        bilateral_beta=1.0,
    )

    correction = (2.0 / np.pi) * np.arctan(100.0)
    np.testing.assert_allclose(torque, np.array([-100.0, 100.0, 0.0, -100.0, 100.0, 0.0]) * correction)


def test_gain_above_one_scales_feedback_without_software_caps() -> None:
    torque = _leader_feedback_torque(
        np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0]),
        np.zeros(6),
        np.ones(6),
        bilateral_kp=10.0,
        bilateral_alpha=1.0,
        bilateral_beta=1.0,
    )

    correction = (2.0 / np.pi) * np.arctan(100.0)
    np.testing.assert_allclose(torque, np.array([-20.0, 20.0, 0.0, -20.0, 20.0, 0.0]) * correction)


def test_angle_correction_is_zero_at_alignment_and_increases_with_error() -> None:
    effort = np.ones(6)
    velocity = np.zeros(6)
    angle_error = np.array([0.0, 0.1, -1.0, 0.1, -1.0, 10.0])

    torque = _leader_feedback_torque(effort, velocity, angle_error, 1.0, 0.0, 1.0)
    expected = -(2.0 / np.pi) * np.arctan(100.0 * np.abs(angle_error))

    np.testing.assert_allclose(torque, expected)
    assert torque[0] == 0.0


def test_beta_zero_disables_angle_attenuation() -> None:
    torque = _leader_feedback_torque(np.ones(6), np.zeros(6), np.ones(6), 100.0, 0.0, 0.0)

    np.testing.assert_allclose(torque, np.full(6, -100.0))


def test_larger_beta_reduces_feedback() -> None:
    torques = [
        _leader_feedback_torque(np.ones(6), np.zeros(6), np.full(6, 0.05), 1.0, 0.0, beta)[0]
        for beta in (1.0, 10.0, 100.0)
    ]

    assert abs(torques[0]) > abs(torques[1]) > abs(torques[2])
