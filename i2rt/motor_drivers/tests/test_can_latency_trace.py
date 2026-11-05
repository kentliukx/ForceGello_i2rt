import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Any

import can
import numpy as np

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import ControlMode, DMChainCanInterface, MotorCmd
from i2rt.motor_drivers.utils import FeedbackFrameInfo, ReceiveMode


def test_can_timestamps_surround_send_and_receive() -> None:
    interface = CanInterface.__new__(CanInterface)
    interface.bus = SimpleNamespace(
        send=lambda message: None,
        recv=lambda timeout: can.Message(arbitration_id=0x11, data=bytes(8), is_extended_id=False),
    )
    interface.receive_mode = ReceiveMode.p16
    interface.use_buffered_reader = False
    interface.trace_timestamps_enabled = True
    before_ns = time.monotonic_ns()

    interface._send_message_get_response(0x01, 1, [0] * 8)

    after_ns = time.monotonic_ns()
    assert before_ns <= interface._last_tx_monotonic_ns <= interface._last_rx_monotonic_ns <= after_ns


def test_can_receive_timestamp_is_recorded_without_latency_trace() -> None:
    interface = CanInterface.__new__(CanInterface)
    interface.bus = SimpleNamespace(
        recv=lambda timeout: can.Message(arbitration_id=0x11, data=bytes(8), is_extended_id=False)
    )
    interface.use_buffered_reader = False
    interface.trace_timestamps_enabled = False
    interface._last_rx_monotonic_ns = 0
    before_ns = time.monotonic_ns()

    interface._receive_message()

    assert before_ns <= interface._last_rx_monotonic_ns <= time.monotonic_ns()


def test_driver_records_first_send_for_each_trace_and_joint() -> None:
    driver = DMChainCanInterface.__new__(DMChainCanInterface)
    driver.motor_list = [(1, "DM4340")]
    driver.motor_direction = np.ones(1)
    driver.motor_offset = np.zeros(1)
    driver._latency_records = deque(maxlen=100)
    driver._latency_lock = threading.Lock()
    driver._last_logged_trace_id = [-1]
    tx_ns = time.monotonic_ns()
    driver.motor_interface = SimpleNamespace(
        control_mode=ControlMode.MIT,
        set_control=lambda **kwargs: FeedbackFrameInfo(
            id=1,
            error_code="0x1",
            error_message="normal",
            position=0.0,
            velocity=0.0,
            torque=0.0,
            temperature_mos=0.0,
            temperature_rotor=0.0,
            tx_monotonic_ns=tx_ns,
        )
    )
    trace: dict[str, Any] = {
        "trace_id": 7,
        "leader_can_rx_ns": np.array([tx_ns - 3_000_000]),
        "leader_read_ns": tx_ns - 2_000_000,
        "leader_queue_ns": tx_ns - 1_900_000,
        "leader_io_ns": tx_ns - 1_800_000,
        "follower_rpc_ns": tx_ns - 1_000_000,
        "follower_worker_ns": tx_ns - 900_000,
        "robot_update_ns": tx_ns - 800_000,
        "driver_accept_ns": tx_ns - 700_000,
    }
    command = MotorCmd(pos=0.1, vel=0.2, kp=2.0, kd=0.5, trace=trace)

    feedback = driver._set_commands([command])
    driver._set_commands([command])
    records = driver.drain_latency_records()

    assert np.isclose(feedback[0].pd_torque_est, 0.3)
    assert len(records) == 1
    assert records[0]["trace_id"] == 7
    assert records[0]["joint"] == 1
    assert records[0]["leader_can_rx_ns"] == tx_ns - 3_000_000
    assert records[0]["follower_can_tx_ns"] == tx_ns
    assert driver.drain_latency_records() == []
