"""Train SeeAnythingFar.

    python train.py                                  # nuScenes, Sonata + DINOv3 (see config/config.yaml)
    python train.py --config-name smoke              # synthetic end-to-end check, CPU
    python train.py trainer.devices=4 data.batch_size=4 pc_range='[-102.4,-102.4,-5,102.4,102.4,3]'
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

try:
    import pytorch_lightning as pl  # noqa: E402
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
except ImportError:
    import lightning.pytorch as pl  # noqa: E402
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402

from seeanythingfar.utils.build import make_loader, make_system  # noqa: E402


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(cfg.seed, workers=True)
    out_dir = HydraConfig.get().runtime.output_dir

    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=False)
    system = make_system(cfg)

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(out_dir, "checkpoints"), monitor=cfg.checkpoint.monitor,
                        mode=cfg.checkpoint.mode, save_last=True, save_top_k=3),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(default_root_dir=out_dir, callbacks=callbacks, **cfg.trainer)
    trainer.fit(system, train_loader, val_loader, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
