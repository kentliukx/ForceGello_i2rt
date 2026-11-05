"""State-update notification used by the bimanual leader control loop."""

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot


def test_wait_for_joint_state_wakes_on_new_generation() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._state_lock = threading.Lock()
    robot._state_condition = threading.Condition(robot._state_lock)
    robot._joint_state_generation = 3
    robot._joint_state = SimpleNamespace(pos=np.array([0.1]), can_rx_monotonic_ns=np.array([123]))

    assert robot.wait_for_joint_pos_and_can_rx_ns(3, timeout_s=0.001) is None

    def publish() -> None:
        time.sleep(0.01)
        with robot._state_condition:
            robot._joint_state = SimpleNamespace(pos=np.array([0.2]), can_rx_monotonic_ns=np.array([456]))
            robot._joint_state_generation = 4
            robot._state_condition.notify_all()

    publisher = threading.Thread(target=publish)
    publisher.start()
    try:
        snapshot = robot.wait_for_joint_pos_and_can_rx_ns(3, timeout_s=0.2)
    finally:
        publisher.join(timeout=1.0)

    assert snapshot is not None
    pos, can_rx_ns, generation = snapshot
    assert generation == 4
    np.testing.assert_array_equal(pos, [0.2])
    np.testing.assert_array_equal(can_rx_ns, [456])
    pos[0] = 9.0
    assert robot._joint_state.pos[0] == 0.2


def test_position_and_velocity_snapshot_are_from_same_state() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._state_lock = threading.Lock()
    robot._state_condition = threading.Condition(robot._state_lock)
    robot._joint_state_generation = 8
    robot._joint_state = SimpleNamespace(
        pos=np.array([0.2, -0.3]), vel=np.array([0.4, -0.5]), can_rx_monotonic_ns=np.array([123, 456])
    )

    snapshot = robot.wait_for_joint_state_and_can_rx_ns(7, timeout_s=0.0)

    assert snapshot is not None
    pos, vel, can_rx_ns, generation = snapshot
    assert generation == 8
    np.testing.assert_array_equal(pos, [0.2, -0.3])
    np.testing.assert_array_equal(vel, [0.4, -0.5])
    np.testing.assert_array_equal(can_rx_ns, [123, 456])
    vel[0] = 10.0
    assert robot._joint_state.vel[0] == 0.4


def test_robot_notifies_only_after_fresh_driver_feedback() -> None:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._state_lock = threading.Lock()
    robot._state_condition = threading.Condition(robot._state_lock)
    robot._joint_state_generation = 0
    robot._joint_state = SimpleNamespace(state_update_seq=1)
    robot._mcap_lock = threading.Lock()
    robot._mcap_recorder = None
    robot._joint_state_saver = None

    def check_limits() -> None:
        pass

    def make_state(seq: int) -> SimpleNamespace:
        return SimpleNamespace(state_update_seq=seq)

    state_update_seq = 1

    def set_commands(*args: Any, **kwargs: Any) -> int:
        return state_update_seq

    robot._check_current_qpos_in_joint_limits = check_limits
    robot._motor_state_to_joint_state = make_state
    robot.motor_chain = SimpleNamespace(set_commands=set_commands)
    commands = JointCommands.init_all_zero(1)

    with robot._state_lock:
        robot._update_joint_state(np.zeros(1), commands)
    assert robot._joint_state_generation == 0

    state_update_seq = 2
    with robot._state_lock:
        robot._update_joint_state(np.zeros(1), commands)
    assert robot._joint_state_generation == 1
