import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
hydra = pytest.importorskip("hydra")
pytest.importorskip("diffusers")
pytest.importorskip("peft")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from examples.rdp_policy import run
from examples.rdp_policy.runtime import RDPModel, resolve_at_checkpoint

RDP_ROOT = Path(__file__).resolve().parents[4] / "RDP"
if not (RDP_ROOT / "reactive_diffusion_policy").is_dir():
    pytest.skip("requires adjacent ForceGello RDP checkout", allow_module_level=True)
sys.path.insert(0, str(RDP_ROOT))


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer
    from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

    torch.set_num_threads(1)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    directory = tmp_path / "run"
    at_path = directory / "at/checkpoints/latest.ckpt"
    at_path.parent.mkdir(parents=True)
    with initialize_config_dir(config_dir=str(RDP_ROOT / "reactive_diffusion_policy/config"), version_base=None):
        cfg = compose(
            config_name="train_forcegello_right_rdp",
            overrides=[
                "training.device=cpu",
                "task.image_shape=[3,32,32]",
                "policy.obs_encoder.resize_shape=[32,32]",
                "policy.down_dims=[16,32,64]",
                f"at_load_dir={at_path}",
            ],
        )
    OmegaConf.resolve(cfg)
    tokenizer = hydra.utils.instantiate(cfg.policy.at, load_dir=None)
    torch.save({"state_dicts": {"model": tokenizer.state_dict()}}, at_path)
    model = hydra.utils.instantiate(cfg.policy)
    rng = np.random.default_rng(42)
    normalizer = LinearNormalizer()
    for key, meta in cfg.task.shape_meta.obs.items():
        if meta.type == "rgb":
            normalizer[key] = get_image_range_normalizer()
        else:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                rng.normal(size=(64, meta.shape[0])).astype(np.float32)
            )
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(rng.normal(size=(64, 14)).astype(np.float32))
    normalizer["latent_action"] = SingleFieldLinearNormalizer.create_fit(rng.normal(size=(64, 16)).astype(np.float32))
    model.set_normalizer(normalizer)
    path = directory / "rdp/checkpoints/latest.ckpt"
    path.parent.mkdir(parents=True)
    # Exercise relocation fallback instead of relying on the absolute training path.
    cfg.at_load_dir = "/old/training/location/latest.ckpt"
    cfg.policy.at.load_dir = cfg.at_load_dir
    torch.save({"cfg": cfg, "state_dicts": {"model": model.state_dict(), "ema_model": model.state_dict()}}, path)
    return path


def test_actual_checkpoint_latent_and_force_reactive_cpu_decoder(checkpoint: Path) -> None:
    model = RDPModel(checkpoint, RDP_ROOT, "cpu", "right", inference_steps=2)
    observations = {key: np.zeros((2, *meta["shape"]), dtype=np.float32) for key, meta in model.obs_meta.items()}
    plan = model.infer(30, observations)
    assert plan.anchor_step == 30 and plan.start_step == 23
    assert model.policy.num_inference_steps == 2
    assert model.visual_hz == 5 and model.control_hz == 20
    with pytest.raises(ValueError, match="visual sampling rate"):
        run.validate(run.Args(checkpoint=checkpoint, device="cpu", camera_hz=20, validate_only=True), model)
    temporal = {key: np.zeros((11, *meta["shape"]), dtype=np.float32) for key, meta in model.extended_meta.items()}
    action1 = model.decode(plan, temporal)
    temporal["right_robot_tcp_wrench"][-1] = 3
    action2 = model.decode(plan, temporal)
    assert action1.shape == action2.shape == (14,)
    assert np.all(np.isfinite(action2)) and not np.allclose(action1, action2)
    temporal["right_robot_joint_pos"][-1] = 1
    temporal["right_robot_joint_vel"][-1] = -1
    action3 = model.decode(plan, temporal)
    assert not np.allclose(action2, action3)  # proprioception reaches the reactive decoder too
    assert all(parameter.device.type == "cpu" for parameter in model.fast_at.decoder.parameters())


def test_validate_only_never_opens_hardware(checkpoint: Path, monkeypatch: Any) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("validation must not open a camera")

    monkeypatch.setattr(run, "LiveImages", forbidden)
    run.main(run.Args(checkpoint=checkpoint, rdp_root=RDP_ROOT, device="cpu", validate_only=True))


def test_wrong_arm_checkpoint_rejected_before_hardware(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="arm/action key"):
        RDPModel(checkpoint, RDP_ROOT, "cpu", "left")
    with pytest.raises(FileNotFoundError):
        resolve_at_checkpoint(checkpoint, "/missing/file", RDP_ROOT, Path("/missing/override"))
