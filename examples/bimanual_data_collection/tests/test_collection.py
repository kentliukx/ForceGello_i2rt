import json
import pickle
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from examples.bimanual_data_collection.collect_data import (
    Args,
    ArmPair,
    RecordingSession,
    validate,
)
from examples.bimanual_data_collection.export_implicitrdp import CartesianProjection, preceding
from examples.bimanual_data_collection.export_implicitrdp import main as export_episode
from examples.bimanual_data_collection.recording import ButtonEdges, EpisodeWriter, PeriodicClock, read_robot_stream


def test_clock_has_no_drift_and_never_catches_up_with_command_bursts() -> None:
    clock = PeriodicClock(60, start_ns=1_000_000_000)
    for tick in range(600):
        expected = 1_000_000_000 + round(tick * 1e9 / 60)
        assert clock.deadline_ns == expected
        index, scheduled, skipped = clock.consume(expected + 50_000)
        assert (index, scheduled, skipped) == (tick, expected, 0)
    now = clock.deadline_ns + 100_000_000
    index, _, skipped = clock.consume(now)
    assert skipped == 6
    assert index == 606
    assert clock.deadline_ns > now


def test_white_and_yellow_edges_are_independent_and_debounced() -> None:
    edges = ButtonEdges()
    assert edges.rising(1, True, 1_000_000_000)
    assert not edges.rising(1, True, 1_020_000_000)
    assert edges.rising(0, True, 1_020_000_000)
    assert not edges.rising(1, False, 1_040_000_000)
    assert not edges.rising(1, True, 1_060_000_000)
    edges.rising(1, False, 1_200_000_000)
    assert edges.rising(1, True, 1_300_000_000)


def robot_packet(side: str, ns: int) -> dict[str, Any]:
    return {
        "kind": "robot",
        "side": side,
        "monotonic_ns": ns,
        "timestamp_ns": 1_700_000_000_000_000_000 + ns,
        "teleoperation_mode": "following",
        "command": {"pos": np.ones(7), "vel": np.zeros(7)},
        "follower": {
            "joint_pos": np.array([0.2, 1.0, 1.2, -0.3, 0.2, 0.4, 0.5]),
            "joint_vel": np.zeros(7),
            "joint_eff_external": np.zeros(7),
        },
    }


def test_episode_boundaries_keep_late_packets_and_do_not_duplicate_between_trajectories(tmp_path: Path) -> None:
    args = Args(output=tmp_path, drain_s=0.001)
    session = RecordingSession(args, {"settings": {}})
    session.toggle(100)
    session.append(robot_packet("left", 99))
    session.append(robot_packet("left", 101))
    session.toggle(200)
    session.toggle(250)
    # An IPC packet from before the stop can arrive after the next episode starts.
    session.append(robot_packet("right", 199))
    session.append(robot_packet("right", 200))
    session.append(robot_packet("right", 251))
    session.toggle(300)
    session.flush(2_000_000)
    assert [row["monotonic_ns"] for row in read_robot_stream(tmp_path / "1/robot_left.pkl")] == [101]
    assert [row["monotonic_ns"] for row in read_robot_stream(tmp_path / "1/robot_right.pkl")] == [199]
    assert [row["monotonic_ns"] for row in read_robot_stream(tmp_path / "2/robot_right.pkl")] == [251]
    assert json.loads((tmp_path / "2/metadata.json").read_text())["status"] == "complete"


def test_stop_event_can_arrive_after_later_samples_without_leaking_them_into_episode(tmp_path: Path) -> None:
    session = RecordingSession(Args(output=tmp_path, drain_s=0.001), {})
    session.append(robot_packet("left", 110))  # data can also arrive before the START event
    session.toggle(100)
    session.append(robot_packet("left", 210))
    session.toggle(200)  # STOP arrives after a sample belonging outside this episode
    session.flush(2_000_000)
    assert [row["monotonic_ns"] for row in read_robot_stream(tmp_path / "1/robot_left.pkl")] == [110]
    with pytest.raises(RuntimeError, match="reorder window"):
        session.append(robot_packet("left", 150))


def test_interrupted_trajectory_is_saved_and_marked(tmp_path: Path) -> None:
    session = RecordingSession(Args(output=tmp_path), {})
    session.toggle(0)
    session.append(robot_packet("left", 1))
    session.close("camera disconnected")
    metadata = json.loads((tmp_path / "1/metadata.json").read_text())
    assert metadata["status"] == "interrupted"
    assert metadata["reason"] == "camera disconnected"
    assert len(list(read_robot_stream(tmp_path / "1/robot_left.pkl"))) == 1


def test_asof_alignment_never_reads_future_and_rejects_stale_samples() -> None:
    samples = [{"value": 1}, {"value": 2}]
    assert preceding(samples, [100, 200], 150, 60) == samples[0]
    assert preceding(samples, [100, 200], 99, 60) is None
    assert preceding(samples, [100, 200], 190, 60) is None


class FakeRobot:
    def __init__(self, leader: bool) -> None:
        self.leader = leader
        self.buttons = [False, False]
        self.torques = []
        self.closed = False

    def num_dofs(self) -> int:
        return 6 if self.leader else 7

    def get_robot_info(self) -> dict:
        return {"kp": np.ones(6) * 80}

    def refresh_state(self) -> None:
        pass

    def get_force_feedback_observations(self, use_applied_gravity: bool = False) -> dict:
        assert use_applied_gravity
        n = self.num_dofs()
        return {
            "joint_pos": np.ones(n) * (0.1 if self.leader else 0.0),
            "joint_vel": np.zeros(n),
            "joint_eff_measured": np.ones(n) * 2,
            "joint_eff_external": np.ones(n) * 2,
            "can_rx_monotonic_ns": np.full(n, time.monotonic_ns() - 1_000_000),
        }

    def command_joint_state(self, command: dict) -> None:
        pass

    def command_joint_torques(self, torque: np.ndarray) -> None:
        self.torques.append(torque.copy())

    def enter_gravity_comp_idle(self) -> None:
        pass

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        return pos

    def update(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeAdapter:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot

    def get_info(self) -> tuple:
        return np.ones(7) * 0.1, self.robot.buttons, np.full(6, time.monotonic_ns()), time.monotonic_ns()


@pytest.mark.parametrize("mode", ["angle", "torque"])
def test_force_feedback_and_white_record_button_do_not_toggle_yellow_sync(monkeypatch: Any, mode: str) -> None:
    from examples.minimum_gello import minimum_gello

    monkeypatch.setattr(minimum_gello, "YAMLeaderRobot", FakeAdapter)
    leader, follower = FakeRobot(True), FakeRobot(False)
    pair = ArmPair("right", leader, follower, Args(feedback_mode=mode, bilateral_kp=0.05))
    pair.mode = "following"
    leader.buttons = [False, True]
    packets: queue.Queue = queue.Queue()
    pair.read(packets)
    pair.update_follower()
    pair.update_leader()
    assert pair.mode == "following"
    assert packets.get_nowait()["kind"] == "toggle"
    source = np.ones(6) * (2 if mode == "torque" else 80 * 0.1)
    expected = minimum_gello._leader_feedback_torque(source, np.zeros(6), np.ones(6) * 0.1, 0.05, 0.0, 1.0)
    np.testing.assert_allclose(leader.torques[-1], expected)
    leader.buttons = [True, True]
    pair.read(packets)
    assert pair.mode == "idle"  # yellow still controls teleoperation independently
    assert packets.empty()  # holding white does not start a second episode


@pytest.mark.parametrize(
    ("left_mode", "right_mode"),
    [
        ("following", "following"),
        ("idle", "following"),
        ("following", "idle"),
        ("idle", "idle"),
        ("aligning", "following"),
        ("following", "aligning"),
    ],
)
def test_multirate_recording_exports_implicitrdp_without_losing_60hz_source(
    tmp_path: Path, left_mode: str, right_mode: str
) -> None:
    import cv2

    settings = {"arm": "yam", "follower_gripper": "linear_4310", "control_hz": 60, "camera_hz": 20}
    writer = EpisodeWriter(tmp_path / "raw", 0, {"settings": settings})
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # BGR red; exported SensorMessage images must be RGB red.
    success, jpeg = cv2.imencode(".jpg", image)
    assert success
    for tick in range(60):
        ns = round(tick * 1e9 / 60)
        for side in ("left", "right"):
            packet = robot_packet(side, ns)
            packet["teleoperation_mode"] = left_mode if side == "left" else right_mode
            if packet["teleoperation_mode"] == "idle":
                packet["command"] = None
            writer.append(packet)
        if tick % 3 == 0:
            for name in ("left_wrist", "right_wrist", "external"):
                writer.append(
                    {
                        "kind": "camera",
                        "camera": name,
                        "monotonic_ns": ns,
                        "timestamp_ns": 1_700_000_000_000_000_000 + ns,
                        "jpeg": jpeg.tobytes(),
                    }
                )
    episode = writer.finish(1_000_000_000)
    output = tmp_path / "implicit/1.pkl"
    export_episode(episode, output)
    with output.open("rb") as stream:
        data = pickle.load(stream)
    messages = data["sensor_message_list"].sensorMessages
    assert len(messages) == 20
    assert len(list(read_robot_stream(episode / "robot_left.pkl"))) == 60
    assert len(list(read_robot_stream(episode / "robot_right.pkl"))) == 60
    assert messages[0].leftRobotTCP.shape == (6,)
    assert messages[0].rightRobotTCP.shape == (6,)
    assert messages[0].leftRobotTCPWrench.shape == (6,)
    assert messages[0].rightRobotTCPWrench.shape == (6,)
    assert messages[0].leftWristCameraRGB.shape == (24, 32, 3)
    assert messages[0].rightWristCameraRGB.shape == (24, 32, 3)
    assert messages[0].externalCameraRGB[0, 0, 0] > 240
    assert messages[0].externalCameraRGB[0, 0, 2] < 10
    assert messages[0].leftRobotGripperState[0] == pytest.approx(0.0475)
    assert np.isnan(messages[0].leftRobotGripperState[1])
    assert messages[1].timestamp > messages[0].timestamp
    for alignment in data["alignment"]:
        assert max(alignment["robot_monotonic_ns"].values()) <= alignment["monotonic_ns"]
        for side, mode in (("left", left_mode), ("right", right_mode)):
            assert (alignment["command"][side] is None) == (mode == "idle")
    with pytest.raises(FileExistsError):
        export_episode(episode, output)


def test_cartesian_projection_reconstructs_jacobian_effort_and_local_frame() -> None:
    projection = CartesianProjection("yam", "linear_4310", damping=1e-8, wrench_sign=1.0)
    sample = robot_packet("left", 0)
    projection.project(sample)
    linear = np.zeros((3, projection.model.nv))
    angular = np.zeros_like(linear)
    projection.mujoco.mj_jacSite(projection.model, projection.data, linear, angular, projection.site)
    jacobian = np.vstack([linear, angular])[:, projection.arm_dofs]
    wrench = np.array([1.0, -2.0, 3.0, 0.1, -0.2, 0.3])
    sample["follower"]["joint_eff_external"][:6] = jacobian.T @ wrench
    _, _, local = projection.project(sample)
    rotation = projection.data.site_xmat[projection.site].reshape(3, 3)
    np.testing.assert_allclose(local, np.concatenate([rotation.T @ wrench[:3], rotation.T @ wrench[3:]]), atol=1e-5)


def test_fixed_camera_assignment_and_invalid_rates_are_rejected() -> None:
    assert validate(Args()) == {
        "left_wrist": "realsense:135122075425",
        "right_wrist": "realsense:135122079769",
        "external": "realsense:036422060422",
    }
    with pytest.raises(ValueError):
        validate(Args(control_hz=float("nan")))


@pytest.mark.parametrize("overrun", [False, True])
def test_unified_loop_orders_both_followers_before_leaders_every_five_iterations(
    monkeypatch: Any, overrun: bool
) -> None:
    from examples.bimanual_data_collection import collect_data
    from examples.minimum_gello import minimum_gello
    from i2rt.robots import get_robot

    now = [1_000_000_000]
    updates = []
    robots = {}
    packets: queue.Queue = queue.Queue()

    class Stop:
        iterations = 0
        done = False

        def is_set(self) -> bool:
            return self.done

        def wait(self, seconds: float) -> bool:
            now[0] += round(seconds * 1e9)
            if overrun and self.iterations == 2:
                now[0] += 20_000_000
            self.iterations += 1
            return self.done

    stop = Stop()

    class Robot(FakeRobot):
        def __init__(self, channel: str) -> None:
            super().__init__("leader" in channel)
            self.channel = channel
            self.buttons = [True, False]  # begin alignment, without a blocking sleep

        def get_force_feedback_observations(self, use_applied_gravity: bool = False) -> dict:
            obs = super().get_force_feedback_observations(use_applied_gravity)
            obs["can_rx_monotonic_ns"][:] = now[0] - 1_000
            return obs

        def update(self) -> None:
            if not stop.done:
                updates.append((stop.iterations - 1, self.channel))
            if self.channel == "can_leader_r" and stop.iterations == 300:
                stop.done = True

    def make_robot(channel: str, **kwargs: Any) -> Robot:
        assert kwargs["background_control"] is False
        robots[channel] = Robot(channel)
        return robots[channel]

    monkeypatch.setattr(collect_data, "worker_signals", lambda: None)
    monkeypatch.setattr(collect_data.time, "monotonic_ns", lambda: now[0])
    monkeypatch.setattr(minimum_gello, "YAMLeaderRobot", FakeAdapter)
    monkeypatch.setattr(get_robot, "get_yam_robot", make_robot)
    collect_data.control_worker(Args(alignment_s=0.001), packets, stop, now[0])
    for iteration in range(300):
        expected = ["can_follower_l", "can_follower_r"] if iteration % 5 == 0 else []
        expected += ["can_leader_l", "can_leader_r"]
        assert [name for tick, name in updates if tick == iteration] == expected
    rows = [row for row in packets.queue if row["kind"] == "robot"]
    assert len(rows) == 120  # 60 samples per pair for 300 executed iterations
    assert all(robot.closed for robot in robots.values())
    for row in rows:
        assert row["follower_update_monotonic_ns"] <= row["leader_feedback_monotonic_ns"] <= row["monotonic_ns"]
        assert np.max(row["leader"]["can_rx_monotonic_ns"]) <= row["monotonic_ns"]
        assert np.max(row["follower"]["can_rx_monotonic_ns"]) <= row["monotonic_ns"]
    left = [row for row in rows if row["side"] == "left"]
    if overrun:
        assert sum(row["missed_periods"] for row in left) > 0
    else:
        assert [row["tick_index"] for row in left] == list(range(0, 300, 5))
        assert left[-1]["monotonic_ns"] - left[0]["monotonic_ns"] == round(295e9 / 300)


def test_camera_drains_acquisition_stream_and_keeps_one_frame_per_20hz_slot(monkeypatch: Any) -> None:
    import cv2

    from examples.bimanual_data_collection import cameras

    stop = threading.Event()
    packets: queue.Queue = queue.Queue()

    class Capture:
        frames = 0
        closed = False

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            return True

        def read(self) -> tuple:
            self.frames += 1
            if self.frames == 30:
                stop.set()
            return True, np.full((24, 32, 3), self.frames, dtype=np.uint8)

        def release(self) -> None:
            self.closed = True

    capture = Capture()
    monkeypatch.setattr(cameras, "worker_signals", lambda: None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: capture)
    monkeypatch.setattr(
        cameras,
        "time",
        SimpleNamespace(
            monotonic_ns=lambda: round(capture.frames * 1e9 / 30),
            time_ns=lambda: 1_700_000_000_000_000_000 + round(capture.frames * 1e9 / 30),
        ),
    )
    cameras.camera_worker("external", "0", Args(camera_hz=20), packets, stop, 0)
    rows = [row for row in list(packets.queue) if row["kind"] == "camera"]
    assert capture.frames == 30 and capture.closed
    assert [row["tick_index"] for row in rows] == list(range(21))
    assert len(rows) < capture.frames
    assert all(row["missed_periods"] == 0 for row in rows)
    assert all(row["monotonic_ns"] >= row["scheduled_monotonic_ns"] for row in rows)


def test_collector_persists_cartesian_data_at_robot_rate(tmp_path: Path) -> None:
    projection = CartesianProjection("yam", "linear_4310", damping=0.01, wrench_sign=1.0)
    session = RecordingSession(Args(output=tmp_path), {}, {"left": projection})
    session.toggle(0)
    session.append(robot_packet("left", 1))
    session.toggle(2)
    session.close("test")
    row = next(read_robot_stream(tmp_path / "1/robot_left.pkl"))
    assert row["cartesian"]["tcp_pose"].shape == (6,)
    assert row["cartesian"]["tcp_wrench_estimate"].shape == (6,)
    row["follower"]["joint_pos"][:6] = 0  # cached collection-time FK must be reused during export
    cached_pose, _, _ = projection.project(row)
    np.testing.assert_array_equal(cached_pose, row["cartesian"]["tcp_pose"])
