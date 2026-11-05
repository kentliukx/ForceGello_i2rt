import queue
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from portal import packlib

from examples.minimum_gello import minimum_gello


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
        server = minimum_gello._build_follower_server(
            1234,
            SimpleNamespace(value=7),
            SimpleNamespace(array=np.zeros(7)),
            SimpleNamespace(array=np.zeros(7)),
            SimpleNamespace(array=np.zeros(7)),
            SimpleNamespace(array=np.full(7, 2.5)),
            SimpleNamespace(value=time.monotonic()),
            command_queue,
        )
    server.methods["command_joint_state"](decoded)
    received = command_queue.get_nowait()

    np.testing.assert_array_equal(received["pos"], packet["pos"])
    np.testing.assert_array_equal(received["vel"], packet["vel"])
    np.testing.assert_array_equal(server.methods["get_observations"]()["joint_eff_measured"], np.full(7, 2.5))
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
            return {
                "joint_pos": np.zeros(7),
                "joint_vel": np.zeros(7),
                "joint_eff_external": np.zeros(7),
                "joint_eff_measured": np.full(7, 2.5),
                "sample_age_s": np.array(0.0),
            }

    client = FakeClient()
    def shared() -> SimpleNamespace:
        return SimpleNamespace(array=np.zeros(7))
    measured_shared = shared()
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
    )

    assert client.sent is not None
    np.testing.assert_array_equal(client.sent["vel"], np.ones(7) * 0.2)
    np.testing.assert_array_equal(measured_shared.array, np.full(7, 2.5))
    assert int(client.sent["trace"]["leader_io_ns"]) > 0
