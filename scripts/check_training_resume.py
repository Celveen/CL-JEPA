"""CPU regression check: a resumed optimizer step matches uninterrupted training."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import Trainer


def one_step(trainer, obs, act):
    trainer.model.train()
    optimizers = [trainer.encoder_optimizer, trainer.predictor_optimizer,
                  trainer.action_encoder_optimizer]
    for optimizer in optimizers:
        optimizer.zero_grad()
    loss = trainer.model(obs, act)[3]
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    return loss.detach()


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="pushobj-resume-") as tmp:
        with hydra.initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.1"):
            full = hydra.compose(config_name="train", return_hydra_config=True, overrides=[
                "env=pusht", "encoder=scratch_resnet_gem", "env.num_workers=0",
                f"env.dataset.data_path={ROOT / 'data/pushobj_bias_16k'}", "env.dataset.n_rollout=1",
                "training.batch_size=2", "training.mixed_precision=no", "training.epochs=3",
                "has_decoder=false", "model.train_decoder=false", "predictor.depth=1",
                "predictor.heads=2", "predictor.mlp_dim=128",
            ])
        full.hydra.mode = RunMode.RUN
        HydraConfig.instance().set_config(full)
        cfg = OmegaConf.create({k: v for k, v in full.items() if k != "hydra"})
        os.chdir(tmp)
        trainer = Trainer(cfg)
        obs, act, _ = next(iter(trainer.dataloaders["train"]))
        one_step(trainer, obs, act)
        trainer.epoch = 1
        trainer.save_ckpt()
        saved = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        expected_loss = one_step(trainer, obs, act)
        expected = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.wandb_run.finish()

        restored = Trainer(cfg)
        assert restored.epoch == 1
        for k, v in restored.model.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)
        assert restored.action_encoder_optimizer.state_dict()["state"]
        actual_loss = one_step(restored, obs, act)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for k, v in restored.model.state_dict().items():
            torch.testing.assert_close(v, expected[k], rtol=0, atol=0)

        observed_epochs = []
        with patch.object(restored, "monitor_jobs"), \
             patch.object(restored, "train", side_effect=lambda: observed_epochs.append(restored.epoch)), \
             patch.object(restored, "val"), patch.object(restored, "logs_flash"), \
             patch.object(restored, "save_ckpt", return_value=(None, "test", 0)):
            restored.run()
        assert observed_epochs == [2, 3], observed_epochs

        z = {"visual": torch.ones(2, 3, 1, 384), "proprio": torch.ones(2, 3, 10)}
        assert all(torch.isfinite(v) for v in restored.err_eval(z, z).values())
        stationary = torch.zeros(2, 4, 1, 384, requires_grad=True)
        stationary_loss = restored.model.total_curvature(stationary)
        assert stationary_loss.item() == 0
        stationary_loss.backward()
        assert torch.isfinite(stationary.grad).all()
        restored.wandb_run.finish()
        print("PASS: all model weights restored; next loss/weights exactly match uninterrupted training")
        print("PASS: optimizer moments restored; total epoch limit respected; next-step metrics finite")


if __name__ == "__main__":
    main()
