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
    )
    robot.gravity_comp_factor = np.array([1.0, 1.1])
    robot._compute_gravity_compensation = lambda state: np.array([1.0, 2.0])

    obs = robot.get_force_feedback_observations()

    np.testing.assert_allclose(obs["joint_eff_external"], [2.0, -3.2])
    np.testing.assert_allclose(obs["joint_pos"], [0.1, 0.2])
    np.testing.assert_allclose(obs["joint_vel"], [0.4, -0.5])
    np.testing.assert_allclose(obs["joint_eff_pd_est"], [0.5, -0.6])


def test_torque_feedback_command_uses_no_position_stiffness() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot.motor_chain = [None, None]
    robot._command_lock = threading.Lock()
    robot._grav_comp_kd = np.array([0.1, 0.2])

    robot.command_joint_torques(np.array([-0.4, 0.6]))

    np.testing.assert_allclose(robot._commands.torques, [-0.4, 0.6])
    np.testing.assert_allclose(robot._commands.kp, [0.0, 0.0])
    np.testing.assert_allclose(robot._commands.kd, [0.1, 0.2])

    robot._torque_command_deadline = time.monotonic() - 1.0
    expired = robot._snapshot_joint_commands()

    np.testing.assert_allclose(expired.torques, [0.0, 0.0])
    np.testing.assert_allclose(expired.kp, [0.0, 0.0])
    np.testing.assert_allclose(expired.kd, [0.1, 0.2])
