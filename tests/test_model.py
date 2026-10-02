"""Full network + config composition. Needs einops (DINO_FPN) and hydra."""
import os

import pytest
import torch

pytest.importorskip("einops")
pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from seeanythingfar.datasets.collate import collate_fn  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def _cfg(name, overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def test_main_config_composes():
    cfg = _cfg("config")
    assert cfg.model.backbone_3d.backend == "sonata"
    assert cfg.model.head.pc_range == cfg.model.backbone_3d.pc_range
    assert cfg.system.class_names[0] == "car"


def test_smoke_model_end_to_end():
    cfg = _cfg("smoke")
    model = instantiate(cfg.model)
    ds = instantiate(cfg.dataset.train)
    batch = collate_fn([ds[0], ds[1]])
    preds = model(batch)
    loss, logs = model.loss(batch, preds)
    assert torch.isfinite(loss) and "loss_seg" in logs
    loss.backward()
    assert all(p.grad is None for p in model.backbone_2d.dino.parameters())   # DINO frozen
    assert model.backbone_2d.fpn_up[0].weight.grad is not None
    model.eval()
    res = model.predict(batch)
    assert res[0]["seg_pred"].shape[0] == batch["points"][0].shape[0]


def test_mismatched_bev_grid_is_rejected():
    cfg = _cfg("smoke", ["model.backbone_3d.bev_stride=4"])
    with pytest.raises(Exception, match="cell"):
        instantiate(cfg.model)
