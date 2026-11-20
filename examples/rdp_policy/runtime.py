"""ROS-free checkpoint loading, observation history and slow/fast RDP inference."""

import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Plan:
    anchor_step: int
    start_step: int
    latent: np.ndarray


class History:
    def __init__(self, capacity: int) -> None:
        self.rows: deque[tuple[int, dict[str, np.ndarray]]] = deque(maxlen=capacity)

    def append(self, step: int, obs: dict[str, np.ndarray]) -> None:
        if self.rows and step != self.rows[-1][0] + 1:
            self.rows.clear()
        self.rows.append((step, obs))

    def observations(self, keys: Any, steps: int, ratio: int) -> dict[str, np.ndarray]:
        rows = dict(self.rows)
        anchor = self.rows[-1][0]
        result = {}
        for key in keys:
            meta = keys[key] if isinstance(keys, dict) else {}
            key_ratio = int(meta.get("temporal_downsample_ratio", ratio))
            key_steps = int(meta.get("n_obs_steps", steps))
            indices = [anchor - i * key_ratio for i in reversed(range(key_steps))]
            result[key] = np.stack([rows[index][key] for index in indices])
        return result

    def temporal(self, plan: Plan, horizon: int, keys: Any) -> dict[str, np.ndarray] | None:
        if not self.rows:
            return None
        end = self.rows[-1][0]
        rows = dict(self.rows)
        if plan.anchor_step > end or plan.start_step < self.rows[0][0] or end - plan.start_step >= horizon:
            return None
        indices = range(plan.start_step, end + 1)
        return {key: np.stack([rows[index][key] for index in indices]) for key in keys}


def resolve_at_checkpoint(checkpoint: Path, configured: str, rdp_root: Path, override: Path | None) -> Path:
    if override is not None:
        candidates = [override]
    else:
        candidates = [
            Path(configured),
            rdp_root / configured,
            checkpoint.parent.parent.parent / "at/checkpoints/latest.ckpt",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("AT checkpoint not found; specify --at-checkpoint with the AT used in training")


class RDPModel:
    """Slow diffusion on the requested device; independent proprioceptive AT on CPU."""

    def __init__(
        self,
        checkpoint: Path,
        rdp_root: Path,
        device: str,
        side: str,
        *,
        at_checkpoint: Path | None = None,
        inference_steps: int = 8,
    ) -> None:
        import sys

        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        torch.set_num_threads(1)
        sys.path.insert(0, str(rdp_root.resolve()))
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        cfg = payload["cfg"]
        meta = cfg.task.shape_meta
        if meta.action.get("space") != "joint" or list(meta.action.shape) != [7]:
            raise ValueError("this runner requires a single-arm 7D joint-position RDP checkpoint")
        if meta.action.get("key") != f"{side}_robot_command" or cfg.task.dataset.relative_action:
            raise ValueError("checkpoint arm/command key must match --side and use absolute joint targets")
        if "latent_diffusion" not in cfg.policy._target_ or not cfg.policy.at.use_rnn_decoder:
            raise ValueError("expected Latent Diffusion Policy with a proprioceptive GRU AT decoder")
        expected_feedback = {f"{side}_robot_joint_pos", f"{side}_robot_joint_vel"}
        if set(meta.extended_obs) != expected_feedback or any(
            list(meta.extended_obs[key].shape) != [7] for key in expected_feedback
        ):
            raise ValueError("AT decoder must use only 7D joint position and velocity feedback")
        path = resolve_at_checkpoint(checkpoint, str(cfg.at_load_dir), rdp_root, at_checkpoint)
        cfg.policy.at.load_dir = str(path)
        cfg.policy.at.device = device
        self.policy = hydra.utils.instantiate(cfg.policy)
        state_key = "ema_model" if cfg.training.use_ema and "ema_model" in payload["state_dicts"] else "model"
        self.policy.load_state_dict(payload["state_dicts"][state_key])
        self.policy.eval().to(device)
        self.policy.at.set_normalizer(self.policy.normalizer)
        self.policy.at.normalizer.to(device)
        self.policy.num_inference_steps = inference_steps

        self.fast_at = hydra.utils.instantiate(cfg.policy.at, device="cpu")
        self.fast_at.set_normalizer(self.policy.normalizer)
        self.fast_at.normalizer.to("cpu")
        self.fast_at.eval()
        self.obs_meta = OmegaConf.to_container(meta.obs, resolve=True)
        self.extended_meta = OmegaConf.to_container(meta.extended_obs, resolve=True)
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.ratio = int(cfg.dataset_obs_temporal_downsample_ratio)
        self.past_steps = self.n_obs_steps * self.ratio
        self.slow_obs_latency_steps = int(cfg.task.dataset.get("slow_obs_latency_steps", 0))
        if not 0 <= self.slow_obs_latency_steps < self.past_steps:
            raise ValueError("checkpoint has invalid slow_obs_latency_steps")
        self.horizon = int(cfg.horizon)
        self.control_hz = float(cfg.task.get("control_hz", 60.0))
        visual_rates = {
            self.control_hz / int(value.get("temporal_downsample_ratio", self.ratio))
            for value in self.obs_meta.values()
            if value.get("type") == "rgb"
        }
        if len(visual_rates) > 1:
            raise ValueError("all visual inputs must use the same frame rate")
        self.visual_hz = next(iter(visual_rates), None)
        self.torch = torch
        self.device = device

    def infer(self, anchor_step: int, observations: dict[str, np.ndarray]) -> Plan:
        tensors = {
            key: self.torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(self.device)
            for key, value in observations.items()
        }
        with self.torch.inference_mode():
            result = self.policy.predict_action(
                tensors, dataset_obs_temporal_downsample_ratio=self.ratio, return_latent_action=True
            )
            latent = result["action"][0, 0].detach().cpu().numpy().copy()
        if not np.all(np.isfinite(latent)):
            raise ValueError("non-finite diffusion latent")
        return Plan(
            anchor_step,
            anchor_step - self.past_steps + 1 + self.slow_obs_latency_steps,
            latent,
        )

    def decode(self, plan: Plan, temporal: dict[str, np.ndarray]) -> np.ndarray:
        tensors = {
            key: self.torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0) for key, value in temporal.items()
        }
        with self.torch.inference_mode():
            condition = self.fast_at.get_temporal_cond(tensors)
            latent = self.torch.from_numpy(plan.latent).unsqueeze(0)
            decoded = self.fast_at.get_action_from_latent_with_temporal_cond(latent, condition)
            # The last observed joint state predicts the next 20 Hz command.
            action = self.fast_at.normalizer["action"].unnormalize(decoded)[0, -1].numpy().copy()
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("decoder must return a finite 7D joint command")
        return action


class SlowPlanner:
    """Latest observation wins; diffusion never runs inside the robot control loop."""

    def __init__(self, model: RDPModel) -> None:
        self.model = model
        self.jobs: queue.Queue[tuple[int, dict[str, np.ndarray]] | None] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.plan: Plan | None = None
        self.error: BaseException | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="rdp-slow-diffusion", daemon=True)
        self.thread.start()

    def submit(self, step: int, observations: dict[str, np.ndarray]) -> None:
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put_nowait((step, observations))

    def latest(self) -> Plan | None:
        with self.lock:
            if self.error is not None:
                raise RuntimeError("RDP diffusion worker failed") from self.error
            return self.plan

    def clear(self) -> None:
        """Discard pending work and published plans; in-flight results retain their old step numbers."""
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        with self.lock:
            self.plan = None

    def _run(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    job = self.jobs.get(timeout=0.1)
                except queue.Empty:
                    continue
                if job is not None:
                    plan = self.model.infer(*job)
                    with self.lock:
                        self.plan = plan
        except BaseException as error:
            with self.lock:
                self.error = error

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)
