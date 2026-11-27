import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from examples.next.train import Windows, fit, split_runs
from i2rt.utils.next_torque import (
    FEATURES,
    FORMAT,
    TORQUE_TARGET,
    NextEstimator,
    build_model,
    next_features,
)


def checkpoint(path: Path, history: int = 3, free: float = 1.0) -> None:
    cfg = {"hidden_size": 4, "num_layers": 1, "head_hidden": 8, "dropout": 0.0}
    model = build_model(cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head[-1].bias.fill_(free)
    torch.save(
        {
            "format": FORMAT,
            "features": FEATURES,
            "side": "right",
            "hz": 60.0,
            "history": history,
            "velocity_target": "zero",
            "model": cfg,
            "torque_target": TORQUE_TARGET,
            "model_state_dict": model.state_dict(),
            "x_mean": torch.zeros(18),
            "x_std": torch.ones(18),
            "y_mean": torch.zeros(6),
            "y_std": torch.ones(6),
        },
        path,
    )


def test_next_raw_nm_sign_features_and_gap_reset(tmp_path: Path) -> None:
    path = tmp_path / "right.pt"
    checkpoint(path)
    estimator = NextEstimator(path, 60, "right")
    q, v, cmd, measured = np.ones(7), np.ones(7) * 2, np.ones(7) * 4, np.ones(7) * 10
    gravity = np.full(7, 2.0)
    np.testing.assert_array_equal(next_features(q, v, cmd), [1] * 6 + [2] * 6 + [3] * 6)
    for i in range(2):
        assert estimator.update(q, v, cmd, measured, 1_000_000_000 + round(i * 1e9 / 60), gravity) is None
    free, external = estimator.update(q, v, cmd, measured, 1_000_000_000 + round(2e9 / 60), gravity)
    np.testing.assert_array_equal(free, np.ones(6))
    np.testing.assert_array_equal(external, np.full(6, 7))  # 10 - 2 - 1: gravity subtracted once.
    assert estimator.update(q, v, cmd, measured, 2_000_000_000, gravity) is None
    with pytest.raises(ValueError, match="side"):
        NextEstimator(path, 60, "left")
    with pytest.raises(ValueError, match="frequency"):
        NextEstimator(path, 20, "right")


def test_training_windows_are_causal_and_splits_have_no_shared_frames() -> None:
    x = np.repeat(np.arange(100)[:, None], 18, axis=1).astype(np.float32)
    y = np.repeat(np.arange(100)[:, None], 6, axis=1).astype(np.float32)
    train, val = split_runs([(x, y)], 10)
    assert train[0][0][-1, 0] < val[0][0][0, 0]
    windows = Windows([(x, y)], 10)
    features, measured = windows[4]
    assert features[-1, 0] == measured[0] == 13


def test_calibration_training_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    episode = tmp_path / "raw" / "1"
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "settings": {"next_calibration": True, "control_hz": 60.0},
            }
        )
    )
    with (episode / "robot_right.pkl").open("wb") as stream:
        for i in range(120):
            q = np.ones(6, dtype=np.float32) * np.sin(i / 30)
            pickle.dump(
                {
                    "teleoperation_mode": "following",
                    "leader_feedback_torque": np.zeros(6),
                    "command": {"vel": np.zeros(7)},
                    "next_sample": {
                        "joint_pos": q,
                        "joint_vel": np.ones(6),
                        "joint_cmd": q + 0.01,
                        "measured_joint_torque": q * 0.2 + 1,
                        "applied_gravity_torque": np.ones(6),
                        "monotonic_ns": 1_000_000_000 + round(i * 1e9 / 60),
                    },
                },
                stream,
            )
    output = tmp_path / "right.pt"
    result = fit(tmp_path / "raw", output, "right", 60, 5, 2, 32, "cpu", 0)
    assert np.isfinite(result["val_rmse_nm"]).all()
    estimator = NextEstimator(output, 60, "right")
    assert estimator.history == 5
    assert output.with_suffix(".metrics.json").is_file()
    saved = torch.load(output, map_location="cpu", weights_only=True)
    assert saved["epoch"] == result["epoch"] == 2
    assert saved["val_loss"] == result["val_mse_normalized"]


@pytest.mark.parametrize("calibration", [False, True])
def test_pair_uses_preceding_command_zero_velocity_and_next_feedback(
    tmp_path: Path, monkeypatch: Any, calibration: bool
) -> None:
    from examples.bimanual_data_collection.collect_data import Args, ArmPair
    from examples.bimanual_data_collection.tests.test_collection import FakeAdapter, FakeRobot
    from examples.minimum_gello import minimum_gello

    path = tmp_path / "right.pt"
    checkpoint(path, history=1, free=0.5)
    monkeypatch.setattr(minimum_gello, "YAMLeaderRobot", FakeAdapter)
    leader, follower = FakeRobot(True), FakeRobot(False)
    sent = []
    follower.command_joint_state = sent.append
    pair = ArmPair(
        "right",
        leader,
        follower,
        Args(
            feedback_mode="next",
            next_calibration=calibration,
            next_right_checkpoint=path,
            bilateral_kp=2,
        ),
    )
    pair.mode = "following"
    pair.pos = np.full(7, 0.2)
    pair.follower_obs = follower.get_force_feedback_observations(True)
    pair.leader_obs = leader.get_force_feedback_observations(True)
    pair.command = {"pos": np.full(7, 0.1), "vel": np.zeros(7)}
    pair.prepare_next()
    pair.update_follower()
    pair.update_leader()
    np.testing.assert_array_equal(pair.next_sample["joint_cmd"], np.full(6, 0.1))
    np.testing.assert_array_equal(sent[-1]["pos"], np.full(7, 0.2))
    np.testing.assert_array_equal(sent[-1]["vel"], np.zeros(7))
    if calibration:
        assert pair.next_estimator is None and not leader.torques
        np.testing.assert_array_equal(pair.feedback_torque, np.zeros(6))
    else:
        np.testing.assert_allclose(pair.follower_obs["joint_eff_external"][:6], np.full(6, 0.5))
        np.testing.assert_allclose(leader.torques[-1], np.full(6, -1.0))


def test_rdp_velocity_targets_zero_with_large_position_changes() -> None:
    from examples.rdp_policy.run import joint_command

    cmd = joint_command(np.full(7, 0.2), np.ones(7), np.zeros(7), 20)
    np.testing.assert_allclose(cmd["pos"][:6], 1.2)
    np.testing.assert_array_equal(cmd["vel"], np.zeros(7))


def test_rdp_updates_next_at_60hz_before_new_command_and_samples_its_output(monkeypatch: Any) -> None:
    from types import SimpleNamespace

    from examples.rdp_policy import run
    from examples.rdp_policy.runtime import Plan

    now, commands, seen = [1_000_000_000], [], []

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

        def get_force_feedback_observations(self, **kwargs: Any) -> dict:
            return {
                "joint_pos": np.zeros(7),
                "joint_vel": np.zeros(7),
                "joint_eff_measured": np.full(7, 10.0),
                "joint_eff_gravity": np.full(7, 2.0),
                "joint_eff_external": np.full(7, 8.0),
                "can_rx_monotonic_ns": np.full(7, now[0] - 100),
            }

        def command_joint_state(self, command: dict) -> None:
            commands.append(command)

        def update(self) -> None:
            pass

    class Estimator:
        def reset(self) -> None:
            pass

        def update(self, q: Any, qdot: Any, command: Any, measured: Any, ns: int, gravity: Any) -> tuple:
            expected = commands[-1]["pos"] if commands else np.zeros(7)
            np.testing.assert_array_equal(command, expected)
            np.testing.assert_array_equal(measured, np.full(7, 10.0))
            np.testing.assert_array_equal(gravity, np.full(7, 2.0))
            seen.append(ns)
            return np.ones(6), np.full(6, 7.0)

    class Model:
        horizon, past_steps, n_obs_steps, ratio = 32, 1, 1, 1
        obs_meta = {"right_robot_joint_pos": {"shape": [7]}}
        extended_meta = {"right_robot_joint_effort_external": {"shape": [6]}}

        def decode(self, plan: Any, temporal: dict) -> np.ndarray:
            np.testing.assert_array_equal(temporal["right_robot_joint_effort_external"][-1], np.full(6, 7.0))
            return np.full(7, 0.1)

    monkeypatch.setattr(run.time, "monotonic_ns", lambda: now[0])
    run.control_loop(
        Robot(),
        Model(),
        SimpleNamespace(
            submit=lambda *args: None, latest=lambda: Plan(0, 0, np.zeros(1), np.zeros(7)), clear=lambda: None
        ),
        run.JointStateProjection("right"),
        SimpleNamespace(read=lambda *args, **kwargs: {}),
        args=run.Args(checkpoint=Path("unused"), max_steps=10, inference_hz=20),
        stop=stop,
        buttons=SimpleNamespace(consume_toggles=lambda ns: int(stop.step == 0)),
        next_estimator=Estimator(),
    )
    assert len(seen) == 10 and len(commands) == 10
    np.testing.assert_allclose(np.diff(seen), 1e9 / 60, atol=1)
    assert any(np.any(command["pos"][:6] != 0) for command in commands)
    for command in commands:
        np.testing.assert_array_equal(command["vel"], np.zeros(7))


def test_zarr_command_zero_target_velocity_preserves_measured_leader_velocity() -> None:
    from examples.bimanual_data_collection.post_process_forcegello import leader_target

    packet = {
        "leader_joint_pos_with_gripper": np.ones(7),
        "leader": {"joint_vel": np.ones(6)},
        "follower_velocity_target": "zero",
    }
    np.testing.assert_array_equal(leader_target(packet), np.concatenate([np.ones(7), np.zeros(7)]))
    np.testing.assert_array_equal(packet["leader"]["joint_vel"], np.ones(6))
