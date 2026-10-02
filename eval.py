"""Evaluate a checkpoint on the validation split.

    python eval.py +ckpt=outputs/.../checkpoints/last.ckpt
    python eval.py +ckpt=... system.nuscenes_export=results/nusc.json +official=true +nusc_version=v1.0-trainval

Always reports the built-in range-binned mAP (and mIoU). With ``+official=true`` the
predictions are also exported and scored by the nuScenes devkit (mAP, NDS).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

try:
    import pytorch_lightning as pl  # noqa: E402
except ImportError:
    import lightning.pytorch as pl  # noqa: E402

from seeanythingfar.utils.build import make_loader, make_system  # noqa: E402


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    official = cfg.get("official", False)
    if official and not cfg.system.nuscenes_export:
        raise ValueError("+official=true needs system.nuscenes_export=<path.json>")

    system = make_system(cfg)
    if cfg.get("ckpt"):
        state = torch.load(cfg.ckpt, map_location="cpu")
        system.load_state_dict(state.get("state_dict", state))
    trainer = pl.Trainer(**{k: v for k, v in cfg.trainer.items() if k not in ("max_epochs",)}, logger=False)
    metrics = trainer.validate(system, make_loader(cfg, "val", shuffle=False))[0]
    for k in sorted(metrics):
        print(f"{k:40s} {metrics[k]:.4f}")

    if official:
        from seeanythingfar.evaluation.nuscenes_eval import run_official_eval
        out_dir = os.path.join(os.path.dirname(os.path.abspath(cfg.system.nuscenes_export)), "nuscenes_eval")
        run_official_eval(cfg.system.nuscenes_export, cfg.dataset.data_root,
                          cfg.get("nusc_version", "v1.0-trainval"), "val", out_dir)


if __name__ == "__main__":
    main()
