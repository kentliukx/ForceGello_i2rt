import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from i2rt.robots.motor_chain_robot import MotorChainRobot


def test_measured_effort_excludes_commanded_gravity_compensation() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._state_lock = threading.Lock()
    robot._joint_state = SimpleNamespace(
        pos=np.array([0.1, 0.2]),
        vel=np.array([0.4, -0.5]),
        eff=np.array([3.0, -1.0]),
        pd_torque_est=np.array([0.5, -0.6]),
        can_rx_monotonic_ns=np.array([123, 456]),
    )
    robot.gravity_comp_factor = np.array([1.0, 1.1])
    robot._compute_gravity_compensation = lambda state: np.array([1.0, 2.0])

    obs = robot.get_force_feedback_observations()

    np.testing.assert_allclose(obs["joint_eff_external"], [2.0, -3.2])
    np.testing.assert_allclose(obs["joint_pos"], [0.1, 0.2])
    np.testing.assert_allclose(obs["joint_vel"], [0.4, -0.5])
    np.testing.assert_allclose(obs["joint_eff_measured"], [3.0, -1.0])
    np.testing.assert_allclose(obs["joint_eff_pd_est"], [0.5, -0.6])
    np.testing.assert_array_equal(obs["can_rx_monotonic_ns"], [123, 456])


def test_torque_feedback_command_uses_no_position_stiffness() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot.motor_chain = [None, None]
    robot._command_lock = threading.Lock()
    robot._grav_comp_kd = np.array([0.1, 0.2])
    robot._update_wakeup = threading.Event()
    trace = {"trace_id": 10, "follower_can_rx_ns": np.array([1, 2])}

    robot.command_joint_torques(np.array([-0.4, 0.6]), trace=trace)

    np.testing.assert_allclose(robot._commands.torques, [-0.4, 0.6])
    np.testing.assert_allclose(robot._commands.kp, [0.0, 0.0])
    np.testing.assert_allclose(robot._commands.kd, [0.1, 0.2])
    assert robot._commands.trace is trace
    assert robot._update_wakeup.is_set()

    robot._torque_command_deadline = time.monotonic() - 1.0
    expired = robot._snapshot_joint_commands()

    np.testing.assert_allclose(expired.torques, [0.0, 0.0])
    np.testing.assert_allclose(expired.kp, [0.0, 0.0])
    np.testing.assert_allclose(expired.kd, [0.1, 0.2])


def test_joint_state_command_sends_velocity_target_to_motor() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot.motor_chain = [None, None]
    robot._command_lock = threading.Lock()
    robot._update_wakeup = threading.Event()
    robot._joint_limits = None
    robot.remapper = SimpleNamespace(
        to_robot_joint_pos_space=lambda values: values,
        to_robot_joint_vel_space=lambda values: values,
    )
    robot._kp = np.array([80.0, 80.0])
    robot._kd = np.array([2.0, 2.0])
    trace = {"trace_id": 1}

    robot.command_joint_state({"pos": np.array([0.1, -0.2]), "vel": np.array([0.3, -0.4]), "trace": trace})

    np.testing.assert_array_equal(robot._commands.pos, [0.1, -0.2])
    np.testing.assert_array_equal(robot._commands.vel, [0.3, -0.4])
    np.testing.assert_array_equal(robot._commands.kd, [2.0, 2.0])
    assert robot._commands.trace is trace
    assert robot._torque_command_deadline is None
    assert robot._update_wakeup.is_set()
