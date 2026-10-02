from typing import Dict, List, Optional, Sequence

import torch

try:
    import pytorch_lightning as pl
except ImportError:  # the unified `lightning` package
    import lightning.pytorch as pl

from seeanythingfar.evaluation.metrics import DetectionEvaluator, SegmentationEvaluator
from seeanythingfar.models.detector import SeeAnythingFar


class MultiModalDetectorSystem(pl.LightningModule):
    """Training / validation loop around :class:`SeeAnythingFar`.

    Validation logs the training losses, range-binned centre-distance mAP and (if a
    segmentation head exists) mIoU. With ``nuscenes_export`` set, predictions are also
    written as a nuScenes submission json for ``evaluation/nuscenes_eval.py``.
    """

    def __init__(
        self,
        model: SeeAnythingFar,
        class_names: Sequence[str],
        seg_class_names: Optional[Sequence[str]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        scheduler: str = "onecycle",           # 'onecycle' | 'cosine' | 'none'
        warmup_pct: float = 0.4,
        range_bins: Sequence[float] = (0.0, 30.0, 50.0, float("inf")),
        nuscenes_export: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.class_names = list(class_names)
        self.det_eval = DetectionEvaluator(self.class_names, range_bins=range_bins)
        self.seg_eval = SegmentationEvaluator(list(seg_class_names)) if seg_class_names and model.seg_head is not None else None
        self.nuscenes_export = nuscenes_export
        self._export: Dict[str, List[Dict]] = {}

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        preds = self.model(batch)
        loss, logs = self.model.loss(batch, preds)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, batch_size=len(batch["points"]),
                      prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss", loss.detach(), prog_bar=True, batch_size=len(batch["points"]))
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch)
        bs = len(batch["points"])
        if any(len(b) for b in batch["gt_boxes"]):
            _, logs = self.model.loss(batch, preds)
            self.log_dict({f"val/{k}": v for k, v in logs.items()}, batch_size=bs, sync_dist=True)

        results = self.model.predict(batch, preds)
        for i, r in enumerate(results):
            self.det_eval.update(r["boxes"], r["scores"], r["labels"], batch["gt_boxes"][i], batch["gt_labels"][i])
            seg = batch.get("seg_labels")
            if self.seg_eval is not None and seg is not None and seg[i] is not None:
                self.seg_eval.update(r["seg_pred"], seg[i])
            if self.nuscenes_export:
                from seeanythingfar.evaluation.nuscenes_eval import to_nuscenes_boxes
                meta = batch["meta"][i]
                self._export[meta["token"]] = to_nuscenes_boxes(r, meta, self.class_names)

    def on_validation_epoch_end(self):
        # metrics are accumulated per process; with multiple GPUs these are rank-local
        metrics = self.det_eval.compute()
        if self.seg_eval is not None:
            metrics.update(self.seg_eval.compute())
        self.log_dict({f"val/{k}": torch.tensor(v, dtype=torch.float32) for k, v in metrics.items()})
        self.det_eval.reset()
        if self.seg_eval is not None:
            self.seg_eval.reset()
        if self.nuscenes_export and self._export:
            from seeanythingfar.evaluation.nuscenes_eval import write_submission
            write_submission(self._export, self.nuscenes_export, use_camera=self.model.backbone_2d is not None)
            self._export = {}

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        sched = self.hparams.scheduler
        if sched == "none":
            return opt
        total = self.trainer.estimated_stepping_batches
        if sched == "onecycle":
            s = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.learning_rate, total_steps=total,
                                                    pct_start=self.hparams.warmup_pct, div_factor=10.0)
        elif sched == "cosine":
            s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        else:
            raise ValueError(sched)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": s, "interval": "step"}}
