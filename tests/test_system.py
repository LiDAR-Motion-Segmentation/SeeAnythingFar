"""Lightning fit on synthetic data; mirrors `python train.py --config-name smoke`."""
import pytest

pytest.importorskip("einops")
pl = pytest.importorskip("pytorch_lightning")

from test_model import _cfg  # noqa: E402

from seeanythingfar.utils.build import make_loader, make_system  # noqa: E402


def test_fit_and_validate(tmp_path):
    cfg = _cfg("smoke")
    system = make_system(cfg)
    trainer = pl.Trainer(default_root_dir=str(tmp_path), logger=False, enable_checkpointing=False,
                         **{k: v for k, v in cfg.trainer.items()})
    trainer.fit(system, make_loader(cfg, "train", True), make_loader(cfg, "val", False))
    metrics = trainer.callback_metrics
    assert "val/mAP/all" in metrics and "val/mIoU" in metrics
