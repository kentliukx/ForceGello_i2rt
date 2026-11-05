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


def test_can_request_skips_unrelated_frame_without_resending() -> None:
    interface = CanInterface.__new__(CanInterface)
    received = iter(
        [
            can.Message(arbitration_id=0x12, data=bytes(8), is_extended_id=False),
            can.Message(arbitration_id=0x11, data=bytes(8), is_extended_id=False),
        ]
    )
    sent: list[can.Message] = []
    interface.bus = SimpleNamespace(send=sent.append, recv=lambda timeout: next(received, None))
    interface.receive_mode = ReceiveMode.p16
    interface.use_buffered_reader = False
    interface.trace_timestamps_enabled = False

    reply = interface._send_message_get_response(0x01, 1, [0] * 8)

    assert reply.arbitration_id == 0x11
    assert len(sent) == 1


def test_new_command_is_accepted_during_can_exchange() -> None:
    driver = DMChainCanInterface.__new__(DMChainCanInterface)
    driver.motor_list = [(1, "DM4340")]
    driver.command_lock = threading.RLock()
    driver.state_lock = threading.Lock()
    driver.state_updated_event = threading.Event()
    driver.commands = [MotorCmd(torque=0.0)]
    driver.running = True
    driver.enable_auto_recovery = False
    driver.same_bus_device_driver = None
    driver._state_update_seq = 0
    driver._report_interval = 30.0
    driver._update_absolute_positions = lambda _feedback: None

    class Recorder:
        def __enter__(self) -> "Recorder":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def track(self) -> None:
            pass

    driver._rate_recorder = Recorder()
    started = threading.Event()
    release = threading.Event()
    accepted = threading.Event()

    def exchange(_commands: list[MotorCmd]) -> list[SimpleNamespace]:
        started.set()
        if not release.wait(2.0):
            raise AssertionError("test CAN exchange was not released")
        driver.running = False
        return [SimpleNamespace(error_code="0x1")]

    driver._set_commands = exchange
    control_thread = threading.Thread(target=driver._set_torques_and_update_state, daemon=True)
    control_thread.start()
    assert started.wait(1.0)

    def update_command() -> None:
        driver.set_commands(np.array([1.0]), get_state=False)
        accepted.set()

    update_thread = threading.Thread(target=update_command, daemon=True)
    update_thread.start()
    try:
        assert accepted.wait(1.0), "command update blocked on the CAN round trip"
    finally:
        release.set()
        update_thread.join(timeout=1.0)
        control_thread.join(timeout=1.0)

    assert not control_thread.is_alive()
    assert driver.state_updated_event.is_set()
    assert driver.commands[0].torque == 1.0


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


def test_driver_records_follower_feedback_to_leader_can_send() -> None:
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
        set_control=lambda **_kwargs: FeedbackFrameInfo(
            id=1,
            error_code="0x1",
            error_message="normal",
            position=0.0,
            velocity=0.0,
            torque=0.0,
            temperature_mos=0.0,
            temperature_rotor=0.0,
            tx_monotonic_ns=tx_ns,
        ),
    )
    trace = {
        "trace_id": 10,
        "follower_can_rx_ns": np.array([tx_ns - 4_000_000]),
        "follower_publish_ns": tx_ns - 3_000_000,
        "follower_rpc_get_ns": tx_ns - 2_000_000,
        "leader_rpc_rx_ns": tx_ns - 1_500_000,
        "leader_feedback_command_ns": tx_ns - 1_000_000,
        "robot_update_ns": tx_ns - 500_000,
        "driver_accept_ns": tx_ns - 400_000,
    }

    driver._set_commands([MotorCmd(torque=0.2, trace=trace)])

    records = driver.drain_latency_records()
    assert len(records) == 1
    assert records[0]["joint"] == 1
    assert records[0]["follower_can_rx_ns"] == tx_ns - 4_000_000
    assert records[0]["leader_can_tx_ns"] == tx_ns
    assert "follower_can_tx_ns" not in records[0]
