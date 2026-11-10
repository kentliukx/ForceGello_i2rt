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


def test_manual_control_refreshes_can_without_submitting_commands_or_recomputing_gravity() -> None:
    class Chain:
        running = True
        submissions = 0
        closed = False

        def __init__(self) -> None:
            self.states = [
                SimpleNamespace(
                    pos=0.0,
                    vel=0.0,
                    eff=4.0,
                    temp_mos=20.0,
                    temp_rotor=20.0,
                    timestamp=0.0,
                    rx_monotonic_ns=100,
                    pd_torque_est=0.0,
                    state_update_seq=1,
                )
                for _ in range(6)
            ]

        def __len__(self) -> int:
            return 6

        def read_states(self) -> list:
            return self.states

        def set_commands(self, torques: np.ndarray, **kwargs: Any) -> list:
            self.submissions += 1
            return self.states

        def close(self) -> None:
            self.closed = True

    chain = Chain()
    robot = MotorChainRobot(
        motor_chain=cast(Any, chain),
        use_gravity_comp=False,
        joint_limits=np.tile([-2.0, 2.0], (6, 1)),
        background_control=False,
    )
    gravity_states = []

    def gravity(state: Any) -> np.ndarray:
        gravity_states.append(state.pos.copy())
        return state.pos.copy()

    robot._compute_gravity_compensation = gravity
    try:
        assert robot._server_thread is None
        assert chain.submissions == 1  # startup primes the CAN buffer
        for motor in chain.states:
            motor.pos, motor.rx_monotonic_ns, motor.state_update_seq = 0.5, 200, 2
        robot.refresh_state()
        obs = robot.get_force_feedback_observations(use_applied_gravity=True)
        np.testing.assert_allclose(obs["joint_pos"], 0.5)
        np.testing.assert_array_equal(obs["can_rx_monotonic_ns"], np.full(6, 200))
        assert not gravity_states and chain.submissions == 1
        robot.command_joint_torques(np.ones(6))
        robot.update()
        assert len(gravity_states) == 1 and chain.submissions == 2
        np.testing.assert_allclose(robot.get_motor_torques(), 1.5)
        # The next CAN sample changes, but the submitted gravity term remains 0.5.
        for motor in chain.states:
            motor.pos, motor.state_update_seq = 0.8, 3
        robot.refresh_state()
        obs = robot.get_force_feedback_observations(use_applied_gravity=True)
        np.testing.assert_allclose(obs["joint_pos"], 0.8)
        np.testing.assert_allclose(obs["joint_eff_gravity"], 0.5)
        np.testing.assert_allclose(obs["joint_eff_external"], 3.5)
        assert len(gravity_states) == 1 and chain.submissions == 2
    finally:
        robot.close()
    assert chain.closed
