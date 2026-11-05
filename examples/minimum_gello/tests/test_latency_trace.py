import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from portal import packlib

from examples.minimum_gello import minimum_gello
from i2rt.robots.motor_chain_robot import MotorChainRobot


def test_trace_survives_rpc_serialization_and_follower_queue() -> None:
    trace = {
        "trace_id": 4,
        "leader_can_rx_ns": np.arange(6, dtype=np.int64) + 100,
        "leader_read_ns": 200,
        "leader_queue_ns": 210,
        "leader_io_ns": 220,
    }
    packet = {"pos": np.arange(7, dtype=float), "vel": np.arange(7, dtype=float) * 0.1, "trace": trace}
    decoded = packlib.unpack(b"".join(packlib.pack((packet,))))[0]
    command_queue = queue.Queue(maxsize=1)

    class FakeServer:
        def __init__(self, port: int) -> None:
            self.methods = {}

        def bind(self, name: str, method: object) -> None:
            self.methods[name] = method

    with patch.object(minimum_gello.portal, "Server", FakeServer):
        sample_trace = np.array([100, 101, 102, 103, 104, 105, 200], dtype=np.int64)
        server = minimum_gello._build_follower_server(
            1234,
            SimpleNamespace(value=7),
            SimpleNamespace(array=np.zeros(7)),
            SimpleNamespace(array=np.zeros(7)),
            SimpleNamespace(array=np.zeros(7)),
            SimpleNamespace(array=np.full(7, 2.5)),
            SimpleNamespace(value=time.monotonic()),
            command_queue,
            SimpleNamespace(array=sample_trace),
        )
    server.methods["command_joint_state"](decoded)
    received = command_queue.get_nowait()

    np.testing.assert_array_equal(received["pos"], packet["pos"])
    np.testing.assert_array_equal(received["vel"], packet["vel"])
    np.testing.assert_array_equal(server.methods["get_observations"]()["joint_eff_measured"], np.full(7, 2.5))
    np.testing.assert_array_equal(server.methods["get_observations"]()["feedback_trace_ns"], sample_trace)
    np.testing.assert_array_equal(received["trace"]["leader_can_rx_ns"], trace["leader_can_rx_ns"])
    assert int(received["trace"]["follower_rpc_ns"]) > 0


def test_leader_io_attaches_send_stage_to_traced_command() -> None:
    command_queue = queue.Queue(maxsize=1)
    trace = {"trace_id": 1, "leader_can_rx_ns": np.ones(6, dtype=np.int64)}
    command_queue.put({"pos": np.zeros(7), "vel": np.ones(7) * 0.2, "trace": trace})
    stop_event = SimpleNamespace(stopped=False)
    stop_event.is_set = lambda: stop_event.stopped

    class FakeClient:
        sent = None

        def command_joint_state(self, packet: dict) -> None:
            self.sent = packet

        def get_observations(self) -> dict:
            stop_event.stopped = True
            sample_ns = time.monotonic_ns() - 1_000_000
            return {
                "joint_pos": np.zeros(7),
                "joint_vel": np.zeros(7),
                "joint_eff_external": np.zeros(7),
                "joint_eff_measured": np.full(7, 2.5),
                "sample_age_s": np.array(0.0),
                "feedback_trace_ns": np.array([sample_ns] * 6 + [sample_ns + 200_000], dtype=np.int64),
                "follower_rpc_get_ns": np.array(time.monotonic_ns()),
            }

    client = FakeClient()
    def shared() -> SimpleNamespace:
        return SimpleNamespace(array=np.zeros(7))
    measured_shared = shared()
    feedback_trace_shared = SimpleNamespace(array=np.zeros(9, dtype=np.int64))
    minimum_gello._run_leader_io_loop(
        client,
        command_queue,
        shared(),
        shared(),
        shared(),
        measured_shared,
        SimpleNamespace(value=0.0),
        stop_event,
        SimpleNamespace(track=lambda: None),
        feedback_trace_shared,
    )

    assert client.sent is not None
    np.testing.assert_array_equal(client.sent["vel"], np.ones(7) * 0.2)
    np.testing.assert_array_equal(measured_shared.array, np.full(7, 2.5))
    assert int(client.sent["trace"]["leader_io_ns"]) > 0
    assert feedback_trace_shared.array[0] > 0
    assert feedback_trace_shared.array[6] <= feedback_trace_shared.array[7] <= feedback_trace_shared.array[8]


def test_feedback_latency_trace_requires_ordered_can_sample() -> None:
    sample = np.array([100, 101, 102, 103, 104, 105, 200, 300, 400], dtype=np.int64)

    trace = minimum_gello._make_feedback_latency_trace(sample, trace_id=10, command_ns=500)

    assert trace is not None
    np.testing.assert_array_equal(trace["follower_can_rx_ns"], sample[:6])
    assert trace["leader_feedback_command_ns"] == 500
    invalid = sample.copy()
    invalid[7] = 99
    assert minimum_gello._make_feedback_latency_trace(invalid, trace_id=10, command_ns=500) is None


def test_follower_publishes_can_times_with_feedback_sample() -> None:
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot.motor_chain = SimpleNamespace(motor_interface=SimpleNamespace(trace_timestamps_enabled=False))
    robot.num_dofs = lambda: 7
    stop_event = threading.Event()
    can_rx_ns = np.arange(6, dtype=np.int64) + 100

    def observe() -> dict:
        stop_event.set()
        return {
            "joint_pos": np.zeros(7),
            "joint_vel": np.zeros(7),
            "joint_eff_measured": np.ones(7),
            "joint_eff_external": np.ones(7) * 0.5,
            "can_rx_monotonic_ns": can_rx_ns,
        }

    robot.get_force_feedback_observations = observe
    robot.enter_gravity_comp_idle = lambda: None
    robot.close = lambda: None
    shared = lambda size, dtype: SimpleNamespace(array=np.zeros(size, dtype=dtype))
    sample_trace = shared(7, np.int64)
    n_dofs = SimpleNamespace(value=0)
    with (
        patch.object(minimum_gello, "get_yam_robot", return_value=robot),
        patch.object(minimum_gello, "RateRecorder", return_value=Mock()),
        patch.object(minimum_gello, "override_log_level"),
    ):
        minimum_gello._yam_polling_worker(
            minimum_gello.Args(trace_latency=True),
            shared(16, float),
            shared(16, float),
            shared(16, float),
            shared(16, float),
            SimpleNamespace(value=0.0),
            n_dofs,
            None,
            stop_event,
            "test-feedback",
            sample_trace,
        )

    assert n_dofs.value == 7
    np.testing.assert_array_equal(sample_trace.array[:6], can_rx_ns)
    assert sample_trace.array[6] > can_rx_ns[-1]


def test_leader_io_wakes_for_command_before_next_feedback_poll() -> None:
    packet = {"pos": np.zeros(7), "vel": np.zeros(7)}

    class WakeQueue:
        def __init__(self) -> None:
            self.waited = False

        def get(self, timeout: float) -> dict:
            if timeout > 0 and not self.waited:
                self.waited = True
                return packet
            raise queue.Empty

        def get_nowait(self) -> dict:
            raise queue.Empty

    stop_event = SimpleNamespace(stopped=False)
    stop_event.is_set = lambda: stop_event.stopped
    wake_queue = WakeQueue()

    class Client:
        polls = 0
        sent = None

        def command_joint_state(self, command: dict) -> None:
            self.sent = command

        def get_observations(self) -> dict:
            self.polls += 1
            if self.polls == 2:
                stop_event.stopped = True
            return {
                "joint_pos": np.zeros(7),
                "joint_vel": np.zeros(7),
                "joint_eff_external": np.zeros(7),
                "joint_eff_measured": np.zeros(7),
                "sample_age_s": np.array(0.0),
            }

    client = Client()
    shared = lambda: SimpleNamespace(array=np.zeros(7))
    minimum_gello._run_leader_io_loop(
        client,
        wake_queue,
        shared(),
        shared(),
        shared(),
        shared(),
        SimpleNamespace(value=0.0),
        stop_event,
        SimpleNamespace(track=lambda: None),
    )

    assert wake_queue.waited
    assert client.polls == 2
    assert client.sent is not None
