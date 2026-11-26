import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from examples.rdp_policy import run
from examples.rdp_policy.runtime import History, Plan, RDPModel, SlowPlanner


def test_history_matches_training_visual_stride_and_never_pads_future_force() -> None:
    history = History(64)
    for step in range(5, 14):
        history.append(step, {"force": np.full(6, step)})
    observations = history.observations(["force"], 2, 3)
    np.testing.assert_array_equal(observations["force"][:, 0], [10, 13])
    plan = Plan(anchor_step=10, start_step=5, latent=np.zeros(2))
    temporal = history.temporal(plan, 32, ["force"])
    np.testing.assert_array_equal(temporal["force"][:, 0], np.arange(5, 14))
    assert history.temporal(plan, 8, ["force"]) is None  # expired chunk
    history.append(16, {"force": np.full(6, 16)})
    assert len(history.rows) == 1
    assert history.temporal(plan, 32, ["force"]) is None  # missing force frames


def test_history_visual_5hz_and_proprio_20hz_use_different_steps() -> None:
    history = History(128)
    for step in range(40):
        history.append(step, {"image": np.full(1, step // 4 * 4), "joint": np.full(7, step)})
    observations = history.observations(
        {
            "image": {"n_obs_steps": 2, "temporal_downsample_ratio": 4},
            "joint": {"n_obs_steps": 2, "temporal_downsample_ratio": 1},
        },
        2,
        4,
    )
    np.testing.assert_array_equal(observations["image"][:, 0], [32, 36])
    np.testing.assert_array_equal(observations["joint"][:, 0], [38, 39])
    temporal = history.temporal(Plan(35, 12, np.zeros(1)), 64, ["joint"])
    np.testing.assert_array_equal(temporal["joint"][:, 0], np.arange(12, 40))


def test_visual_latent_from_t_first_targets_the_command_at_t_plus_two() -> None:
    import torch

    model = RDPModel.__new__(RDPModel)
    model.torch = torch
    model.device = "cpu"
    model.past_steps = 8
    model.slow_obs_latency_steps = 1
    model.policy = SimpleNamespace(
        predict_action=lambda *args, **kwargs: {"action": torch.zeros(1, 1, 4)}
    )
    model.ratio = 4
    plan = model.infer(20, {"joint": np.zeros((2, 7), dtype=np.float32)})
    assert plan.anchor_step == 20
    assert plan.start_step == 14
    history = History(32)
    for step in range(14, 22):
        history.append(step, {"joint": np.full(7, step)})
    temporal = history.temporal(plan, 32, ["joint"])
    np.testing.assert_array_equal(temporal["joint"][:, 0], np.arange(14, 22))
    # Decoder index 7 sees fast state 21 and predicts command time 22.
    assert temporal["joint"][-1, 0] == 21


def test_joint_commands_keep_position_and_velocity_units_and_gripper_convention() -> None:
    action = np.arange(7, dtype=float) / 10
    previous = action - 0.01
    command = run.joint_command(action, previous, 20)
    np.testing.assert_allclose(command["pos"], action)
    np.testing.assert_allclose(command["vel"][:6], np.full(6, 0.2))
    assert command["vel"][6] == 0
    action[6] = 2
    assert run.joint_command(action, previous, 20)["pos"][6] == 1
    action[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run.joint_command(action, previous, 20)


def test_joint_command_clips_differenced_arm_velocity_per_joint() -> None:
    action = np.array([0.2, -0.2, 0.1, -0.1, 0.05, -0.05, 0.7])
    command = run.joint_command(action, np.zeros(7), 20)
    np.testing.assert_allclose(command["pos"], action)
    np.testing.assert_allclose(command["vel"], [2, -2, 2, -2, 1, -1, 0])


def test_command_interpolation_reaches_each_target_on_the_next_20hz_observation(monkeypatch: Any) -> None:
    now = [1_000_000_000]
    commands, submitted_ns = [], []

    class Stop:
        step = -1

        def is_set(self) -> bool:
            return False

        def wait(self, duration: float) -> bool:
            now[0] += round(duration * 1e9)
            self.step += 1
            return False

    stop = Stop()

    class Robot:
        def refresh_state(self) -> None:
            pass

        def get_force_feedback_observations(self, use_applied_gravity: bool) -> dict:
            assert use_applied_gravity
            return {
                "joint_pos": np.full(7, stop.step // 3),
                "can_rx_monotonic_ns": np.full(7, now[0] - 100),
            }

        def command_joint_state(self, command: dict) -> None:
            commands.append((stop.step, command))

        def update(self) -> None:
            submitted_ns.append(now[0])

    class Model:
        horizon, past_steps, n_obs_steps, ratio = 32, 1, 1, 1
        obs_meta = {"right_robot_joint_pos": {"shape": [7]}}
        extended_meta = {"right_robot_joint_pos": {"shape": [7]}}

        def decode(self, plan: Plan, temporal: dict) -> np.ndarray:
            target = 10 * (temporal["right_robot_joint_pos"][-1, 0] + 1)
            return np.full(7, target)

    planner = SimpleNamespace(submit=lambda *args: None, latest=lambda: Plan(0, 0, np.zeros(1)), clear=lambda: None)
    projection = SimpleNamespace(project=lambda state: {"right_robot_joint_pos": state["joint_pos"]})
    monkeypatch.setattr(run.time, "monotonic_ns", lambda: now[0])
    run.control_loop(
        Robot(),
        Model(),
        planner,
        projection,
        SimpleNamespace(read=lambda *args, **kwargs: {}),
        args=run.Args(checkpoint=Path("unused"), max_steps=7, inference_hz=20),
        stop=stop,
        buttons=SimpleNamespace(consume_toggles=lambda since_ns: int(stop.step == 0)),
    )
    assert [step for step, _ in commands] == list(range(7))
    np.testing.assert_allclose(
        [command["pos"][0] for _, command in commands], [0, 0, 0, 0, 20 / 3, 40 / 3, 20]
    )
    np.testing.assert_allclose(
        [command["vel"][0] for _, command in commands], [0, 0, 0, 0, 2, 2, 2]
    )
    np.testing.assert_allclose(np.diff(submitted_ns) / 1e9, 1 / 60, atol=1e-8)


def test_slow_inference_is_asynchronous_and_replaces_pending_observations() -> None:
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class Model:
        def infer(self, step: int, obs: dict) -> Plan:
            calls.append(step)
            if step == 0:
                entered.set()
                assert release.wait(1)
            else:
                completed.set()
            return Plan(step, step - 5, np.zeros(1))

    planner = SlowPlanner(Model())
    try:
        planner.submit(0, {})
        assert entered.wait(1)
        assert planner.latest() is None
        planner.submit(1, {})
        planner.submit(2, {})
        release.set()
        assert completed.wait(1)
    finally:
        release.set()
        planner.close()
    assert calls == [0, 2]
    assert planner.latest().anchor_step == 2


@pytest.mark.parametrize("presses,steps", [([0], 25), ([0, 15, 20], 35), ([], 25)])
def test_control_loop_keeps_20hz_and_compensates_delayed_latent_with_current_force(
    monkeypatch: Any, presses: list[int], steps: int
) -> None:
    now = [1_000_000_000]
    events, commands, submitted_ns = [], [], []

    class Stop:
        step = -1

        def is_set(self) -> bool:
            return False

        def wait(self, duration: float) -> bool:
            now[0] += round(duration * 1e9)
            self.step += 1
            return False

    stop = Stop()

    class Robot:
        def get_joint_pos(self) -> np.ndarray:
            return np.zeros(7)

        def refresh_state(self) -> None:
            events.append((stop.step, "read"))

        def get_force_feedback_observations(self, use_applied_gravity: bool) -> dict:
            assert use_applied_gravity
            return {"joint_pos": np.full(7, stop.step), "can_rx_monotonic_ns": np.full(7, now[0] - 100)}

        def command_joint_state(self, command: dict) -> None:
            events.append((stop.step, "command"))
            commands.append((stop.step, now[0], command))

        def enter_gravity_comp_idle(self) -> None:
            events.append((stop.step, "idle"))

        def update(self) -> None:
            events.append((stop.step, "gravity-and-submit"))
            submitted_ns.append(now[0])

    class Model:
        horizon, past_steps, n_obs_steps, ratio = 32, 6, 2, 3
        obs_meta = {"right_robot_joint_pos": {"shape": [7]}}
        extended_meta = {"right_robot_tcp_wrench": {"shape": [6]}}

        def decode(self, plan: Plan, temporal: dict) -> np.ndarray:
            force = temporal["right_robot_tcp_wrench"][:, 0]
            # The decoder observes only start..current and its last command targets current+1.
            np.testing.assert_array_equal(force, np.arange(plan.start_step, stop.step + 1))
            return np.full(7, force[-1] + 1)

    class Planner:
        pending = active = None

        old = None

        def clear(self) -> None:
            self.old = self.active  # simulate a late result returning from an old in-flight inference
            self.active = self.pending = None

        def submit(self, step: int, obs: dict) -> None:
            np.testing.assert_array_equal(obs["right_robot_joint_pos"][:, 0], [step - 3, step])
            self.pending = Plan(step, step - 5, np.zeros(1))

        def latest(self) -> Plan | None:
            if self.pending is not None and stop.step >= self.pending.anchor_step + 3:
                self.active = self.pending
                self.pending = None
            return self.active if self.active is not None else self.old

    def read_images(ns: int, *, required: bool) -> dict:
        if not required:
            assert events[-1] == (stop.step, "gravity-and-submit")
        return {}

    projection = SimpleNamespace(
        project=lambda state: {
            "right_robot_joint_pos": state["joint_pos"],
            "right_robot_tcp_wrench": np.full(6, stop.step),
        }
    )
    monkeypatch.setattr(run.time, "monotonic_ns", lambda: now[0])
    run.control_loop(
        Robot(),
        Model(),
        Planner(),
        projection,
        SimpleNamespace(read=read_images),
        args=run.Args(checkpoint=Path("unused"), max_steps=steps, inference_hz=2, command_hz=20),
        stop=stop,
        buttons=SimpleNamespace(consume_toggles=lambda since_ns: int(stop.step in presses)),
    )
    enabled = False
    command_steps = []
    for step in range(steps):
        if step in presses:
            enabled = not enabled
        command_steps.append(enabled)
        expected = "command" if enabled else "idle"
        assert [name for index, name in events if index == step] == ["read", expected, "gravity-and-submit"]
    assert len(commands) == sum(command_steps)
    assert len(submitted_ns) == steps
    np.testing.assert_allclose(np.diff(submitted_ns) / 1e9, 1 / 20, atol=1e-8)
    by_step = {step: command for step, _, command in commands}
    if presses:
        assert by_step[12]["pos"][0] == 0
        assert by_step[13]["pos"][0] == 0  # observation 13 is decoded, but not sent yet
        assert by_step[14]["pos"][0] == 14  # observation 13 targets control slot 14
        assert by_step[14]["vel"][0] == 2
        assert by_step[14]["vel"][6] == 0
    if 15 in presses:
        assert not any(step in by_step for step in range(15, 20))
        assert by_step[20]["pos"][0] == 20  # restart holds the new pose, not an old prediction
        assert by_step[21]["vel"][0] == 0
        assert by_step[33]["pos"][0] == 20
        assert by_step[34]["pos"][0] == 34


def test_stale_robot_state_never_sends_policy_command() -> None:
    submitted = []
    robot = SimpleNamespace(
        get_joint_pos=lambda: np.zeros(7),
        refresh_state=lambda: None,
        get_force_feedback_observations=lambda **kw: {"can_rx_monotonic_ns": np.zeros(7)},
        command_joint_state=submitted.append,
    )
    model = SimpleNamespace(horizon=32, past_steps=6)
    with pytest.raises(RuntimeError, match="CAN feedback"):
        run.control_loop(
            robot,
            model,
            None,
            None,
            None,
            args=run.Args(checkpoint=Path("unused"), max_steps=1),
            stop=threading.Event(),
            buttons=None,
        )
    assert not submitted


def test_validate_opens_only_model_selected_camera_devices() -> None:
    model = SimpleNamespace(
        control_hz=20,
        obs_meta={
            "external_img": {"type": "rgb", "shape": [3, 240, 320]},
            "right_wrist_img": {"type": "rgb", "shape": [3, 240, 320]},
            "right_robot_joint_pos": {"shape": [7]},
        },
        extended_meta={"right_robot_tcp_wrench": {"shape": [6]}},
    )
    args = run.Args(checkpoint=Path("unused"), external_camera="0", right_wrist_camera="2")
    assert set(run.validate(args, model)) == {"external_img", "right_wrist_img"}
    args.right_wrist_camera = "/dev/video0"
    with pytest.raises(ValueError, match="distinct"):
        run.validate(args, model)
    args.right_wrist_camera = "2"
    args.command_hz = 55
    with pytest.raises(ValueError, match="command_hz"):
        run.validate(args, model)
    args.command_hz = 60
    args.control_hz = 50
    with pytest.raises(ValueError):
        run.validate(args, model)


def test_live_camera_rgb_preprocessing_and_stale_detection(monkeypatch: Any) -> None:
    cv2 = pytest.importorskip("cv2")

    now = 1_000_000_000
    monkeypatch.setattr(run.time, "monotonic_ns", lambda: now)
    images = run.LiveImages.__new__(run.LiveImages)
    images.args = run.Args(checkpoint=Path("unused"))
    images.specs = {key: {"shape": [3, 24, 32]} for key in ("external_img", "right_wrist_img")}
    images.packets = queue.Queue()
    images.frames = {}
    images.future_packets = []
    images.workers = []
    bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    success, jpeg = cv2.imencode(".jpg", bgr)
    assert success
    images.packets.put({"kind": "camera", "camera": "external", "jpeg": jpeg.tobytes(), "monotonic_ns": now})
    assert images.read(now) is None
    images.packets.put({"kind": "camera", "camera": "right_wrist", "jpeg": jpeg.tobytes(), "monotonic_ns": now})
    obs = images.read(now)
    for rgb in obs.values():
        assert rgb.shape == (3, 24, 32) and rgb.dtype == np.float32
        assert rgb[0, 0, 0] > 0.98 and rgb[2, 0, 0] < 0.02
    images.packets.put({"kind": "camera", "camera": "right_wrist", "jpeg": jpeg.tobytes(), "monotonic_ns": now + 50_000_000})
    assert images.read(now) is not None
    assert images.frames["right_wrist_img"][0] == now
    assert images.read(now + 50_000_000) is not None
    assert images.frames["right_wrist_img"][0] == now + 50_000_000
    with pytest.raises(RuntimeError, match="camera data is stale"):
        images.read(now + 600_000_000)
    assert images.read(now + 600_000_000, required=False) is None
