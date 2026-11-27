"""ROS-free NEXT on gravity-compensated motor torque.

Method: https://github.com/philiphan0109/factr2_next (FACTR 2).
All arm positions/velocities/torques here are in rad, rad/s and Nm; no gripper.
"""

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = "q,qdot,q_command_minus_q"
FORMAT = "i2rt_next_v1"
TORQUE_TARGET = "measured_minus_applied_gravity"


def next_features(q: np.ndarray, qdot: np.ndarray, command: np.ndarray) -> np.ndarray:
    values = [np.asarray(v, dtype=np.float32)[:6] for v in (q, qdot, command)]
    if any(v.shape != (6,) or not np.all(np.isfinite(v)) for v in values):
        raise ValueError("NEXT requires six finite arm joints in rad and rad/s")
    return np.concatenate([values[0], values[1], values[2] - values[0]])


def build_model(config: dict[str, Any]) -> Any:
    import torch

    class TorqueModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(config.get("hidden_size", 128))
            layers = int(config.get("num_layers", 2))
            self.lstm = torch.nn.LSTM(
                18,
                hidden,
                layers,
                batch_first=True,
                dropout=float(config.get("dropout", 0.1)) if layers > 1 else 0.0,
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden, int(config.get("head_hidden", 256))),
                torch.nn.ReLU(),
                torch.nn.Linear(int(config.get("head_hidden", 256)), 6),
            )

        def forward(self, x: Any) -> Any:
            y, _ = self.lstm(x)  # Fresh hidden state for every sliding window, as in NEXT.
            return self.head(y[:, -1])

    return TorqueModel()


class NextEstimator:
    """Fixed-rate, causal sliding windows; never silently fall back to gravity residuals."""

    def __init__(self, checkpoint: Path, hz: float, side: str) -> None:
        import torch

        torch.set_num_threads(1)
        self.checkpoint = Path(checkpoint)
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if ckpt.get("format") != FORMAT or ckpt.get("features") != FEATURES:
            raise ValueError("unsupported NEXT checkpoint or input convention")
        if ckpt.get("side") != side or not math.isclose(ckpt["hz"], hz):
            raise ValueError("NEXT checkpoint must match the arm side and sampling frequency")
        if ckpt.get("velocity_target") != "zero":
            raise ValueError("NEXT checkpoint was not trained with zero velocity targets")
        if ckpt.get("torque_target") != TORQUE_TARGET:
            raise ValueError(
                "NEXT checkpoint must learn measured torque minus applied gravity (subtract gravity once)"
            )
        self.hz = hz
        self.history = int(ckpt["history"])
        if self.history < 1:
            raise ValueError("NEXT history must be positive")
        self.model = build_model(ckpt["model"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.norm = {}
        for key, size in (("x_mean", 18), ("x_std", 18), ("y_mean", 6), ("y_std", 6)):
            value = np.asarray(ckpt[key], dtype=np.float32)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid NEXT normalization: {key}")
            if key.endswith("std") and np.any(value <= 0):
                raise ValueError(f"invalid NEXT normalization scale: {key}")
            self.norm[key] = value
        self.torch = torch
        self.rows: deque[np.ndarray] = deque(maxlen=self.history)
        self.last_ns = 0
        # Warm PyTorch kernels before opening CAN or engaging feedback.
        with torch.inference_mode():
            self.model(torch.zeros(1, self.history, 18))

    def reset(self) -> None:
        self.rows.clear()
        self.last_ns = 0

    def update(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        command: np.ndarray,
        measured: np.ndarray,
        timestamp_ns: int,
        gravity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        measured = np.asarray(measured, dtype=float)[:6]
        if measured.shape != (6,) or not np.all(np.isfinite(measured)):
            raise ValueError("NEXT requires raw measured motor torque in Nm")
        gravity = np.asarray(gravity, dtype=float)[:6]
        if gravity.shape != (6,) or not np.all(np.isfinite(gravity)):
            raise ValueError("NEXT requires the applied gravity compensation torque in Nm")
        if self.last_ns and not 0 < timestamp_ns - self.last_ns <= 1.75e9 / self.hz:
            self.reset()
        self.last_ns = timestamp_ns
        self.rows.append(next_features(q, qdot, command))
        if len(self.rows) < self.history:
            return None
        x = (np.stack(self.rows) - self.norm["x_mean"]) / self.norm["x_std"]
        with self.torch.inference_mode():
            normalized = self.model(self.torch.from_numpy(x).unsqueeze(0))[0].numpy()
        free = normalized * self.norm["y_std"] + self.norm["y_mean"]
        external = (measured - gravity) - free
        if not np.all(np.isfinite(external)):
            raise ValueError("non-finite NEXT output")
        return free.copy(), external.copy()
