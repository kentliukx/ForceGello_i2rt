"""Train NEXT on contact-free i2rt recordings. Does not open robot hardware."""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bimanual_data_collection.recording import read_robot_stream
from i2rt.utils.next_torque import FEATURES, FORMAT, TORQUE_TARGET, build_model, next_features


def read_runs(input_dir: Path, side: str, hz: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split at episode boundaries, disengagements, missing samples and scheduler gaps."""
    episodes = [input_dir] if (input_dir / "metadata.json").is_file() else sorted(input_dir.iterdir())
    runs = []

    def finish(xs: list[np.ndarray], ys: list[np.ndarray]) -> None:
        if xs:
            runs.append((np.stack(xs), np.stack(ys)))
            xs.clear()
            ys.clear()

    for path in episodes:
        if not (path / "metadata.json").is_file() or path.name.endswith(".partial"):
            continue
        meta = json.loads((path / "metadata.json").read_text())
        settings = meta["settings"]
        if meta["status"] != "complete":
            continue
        if not settings.get("next_calibration", False):
            raise ValueError(f"{path}: not a NEXT free-motion recording; contact data must not supervise free torque")
        if not math.isclose(settings["control_hz"], hz):
            raise ValueError(f"{path}: sample rate differs from --hz")
        xs, ys, last_ns = [], [], 0

        for packet in read_robot_stream(path / f"robot_{side}.pkl"):
            sample = packet.get("next_sample")
            if sample is None or packet["teleoperation_mode"] == "idle":
                finish(xs, ys)
                last_ns = 0
                continue
            if np.any(np.asarray(packet["leader_feedback_torque"]) != 0):
                raise ValueError(f"{path}: haptic feedback was active in NEXT calibration data")
            command = packet["command"]
            if command is not None and np.any(np.asarray(command["vel"]) != 0):
                raise ValueError(f"{path}: nonzero velocity targets in calibration data")
            ns = int(sample["monotonic_ns"])
            if last_ns and not 0 < ns - last_ns <= 1.75e9 / hz:
                finish(xs, ys)
            last_ns = ns
            x = next_features(sample["joint_pos"], sample["joint_vel"], sample["joint_cmd"])
            y = np.asarray(sample["measured_joint_torque"], dtype=np.float32) - np.asarray(
                sample["applied_gravity_torque"], dtype=np.float32
            )
            if y.shape != (6,) or not np.all(np.isfinite(y)):
                raise ValueError(f"{path}: invalid measured Nm")
            xs.append(x)
            ys.append(y)
        finish(xs, ys)
    return runs


class Windows(torch.utils.data.Dataset):
    def __init__(self, runs: list[tuple[np.ndarray, np.ndarray]], history: int) -> None:
        self.runs, self.history = runs, history
        counts = [max(0, len(x) - history + 1) for x, _ in runs]
        self.ends = np.cumsum(counts)
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError("not enough continuous samples for NEXT history")

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        run = int(np.searchsorted(self.ends, index, side="right"))
        start = index - (int(self.ends[run - 1]) if run else 0)
        x, y = self.runs[run]
        # y is measured torque at the last INPUT timestep, never a future command.
        return torch.from_numpy(x[start : start + self.history]), torch.from_numpy(y[start + self.history - 1])


def split_runs(
    runs: list[tuple[np.ndarray, np.ndarray]], history: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    # Hold out a contiguous tail in each run. No history window straddles the split.
    train, val = [], []
    for x, y in runs:
        cut = int(len(x) * 0.8)
        if cut < history or len(x) - cut < history:
            continue
        train.append((x[:cut], y[:cut]))
        val.append((x[cut:], y[cut:]))
    if not train:
        raise ValueError("record longer continuous free-space runs (at least 5 * history samples)")
    return train, val


def fit(
    input_dir: Path,
    output: Path,
    side: str,
    hz: float,
    history: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_runs, val_runs = split_runs(read_runs(input_dir, side, hz), history)
    raw_x = np.concatenate([x for x, _ in train_runs])
    raw_y = np.concatenate([y for _, y in train_runs])
    norm = {
        "x_mean": raw_x.mean(0),
        "x_std": np.maximum(raw_x.std(0), 1e-5),
        "y_mean": raw_y.mean(0),
        "y_std": np.maximum(raw_y.std(0), 1e-5),
    }

    def normalize(runs: list[tuple[np.ndarray, np.ndarray]]) -> Windows:
        return Windows(
            [
                (
                    ((x - norm["x_mean"]) / norm["x_std"]).astype(np.float32),
                    ((y - norm["y_mean"]) / norm["y_std"]).astype(np.float32),
                )
                for x, y in runs
            ],
            history,
        )

    train_loader = torch.utils.data.DataLoader(normalize(train_runs), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(normalize(val_runs), batch_size=batch_size)
    config = {"hidden_size": 128, "num_layers": 2, "head_hidden": 256, "dropout": 0.1}
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output.parent.mkdir(parents=True, exist_ok=True)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * len(x)
        model.eval()
        val_sum, squared_nm = 0.0, np.zeros(6)
        with torch.inference_mode():
            for x, y in val_loader:
                error = model(x.to(device)) - y.to(device)
                val_sum += error.square().mean().item() * len(x)
                squared_nm += np.sum((error.cpu().numpy() * norm["y_std"]) ** 2, axis=0)
        val_loss = val_sum / len(val_loader.dataset)
        rmse = np.sqrt(squared_nm / len(val_loader.dataset))
        row = {
            "epoch": epoch,
            "train_mse_normalized": train_sum / len(train_loader.dataset),
            "val_mse_normalized": val_loss,
            "val_rmse_nm": rmse.tolist(),
        }
        losses.append(row)
        print(
            f"NEXT {side} {epoch}/{epochs}: train={row['train_mse_normalized']:.6f} "
            f"val={val_loss:.6f} RMSE Nm={np.round(rmse, 4)}",
            flush=True,
        )
        if not math.isfinite(val_loss):
            raise ValueError("non-finite NEXT training loss")
        # The default runtime path always contains the latest completed epoch.
        ckpt = {
            "format": FORMAT,
            "features": FEATURES,
            "side": side,
            "hz": hz,
            "history": history,
            "velocity_target": "zero",
            "model": config,
            "torque_target": TORQUE_TARGET,
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            **{k: torch.from_numpy(v.copy()) for k, v in norm.items()},
            "epoch": epoch,
            "val_loss": val_loss,
            "val_rmse_nm": rmse.tolist(),
        }
        temporary = output.with_suffix(".tmp")
        torch.save(ckpt, temporary)
        temporary.replace(output)
        output.with_suffix(".metrics.json").write_text(json.dumps(losses, indent=2))
    return losses[-1]


def main(
    input_dir: Path = Path("data/next/free_motion"),
    output_dir: Path = Path("data/next"),
    side: str = "both",
    hz: float = 60.0,
    history: int = 50,
    epochs: int = 20,
    batch_size: int = 256,
    device: str = "cuda:0",
    seed: int = 0,
) -> None:
    if (
        side not in ("left", "right", "both")
        or min(history, epochs, batch_size) < 1
        or not math.isfinite(hz)
        or hz <= 0
    ):
        raise ValueError("invalid NEXT side/rate/history/training parameters")
    torch.set_num_threads(1)
    for arm in ("left", "right") if side == "both" else (side,):
        fit(input_dir, output_dir / f"{arm}.pt", arm, hz, history, epochs, batch_size, device, seed)


if __name__ == "__main__":
    tyro.cli(main)
