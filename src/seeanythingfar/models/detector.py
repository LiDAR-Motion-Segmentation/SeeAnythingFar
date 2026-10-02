"""Full SeeAnythingFar network: backbones -> TransFusion detection + linear segmentation."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from seeanythingfar.models.base import BaseBackbone2D, BaseBackbone3D, BaseFusionHead, BaseSegHead


class SeeAnythingFar(nn.Module):
    """
    Batch format (see ``datasets/collate.py``):
        points      list of B tensors (N_i, C)            LiDAR frame, [x, y, z, intensity, ...]
        images      (B, N_cam, 3, H, W) in [0, 1] or None  camera-less runs skip image fusion
        lidar2img   (B, N_cam, 4, 4)                       for the network-input image size
        image_hw    (H, W)
        gt_boxes    list of B tensors (M_i, 9)             training / eval only
        gt_labels   list of B tensors (M_i,)
        seg_labels  list of B tensors (N_i,) or None       255 = ignore
    """

    def __init__(
        self,
        backbone_3d: BaseBackbone3D,
        head: BaseFusionHead,
        backbone_2d: Optional[BaseBackbone2D] = None,
        seg_head: Optional[BaseSegHead] = None,
        seg_loss_weight: float = 1.0,
        seg_ignore_index: int = 255,
    ):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.backbone_2d = backbone_2d
        self.head = head
        self.seg_head = seg_head
        self.seg_loss_weight = seg_loss_weight
        self.seg_ignore_index = seg_ignore_index

        # the BEV map the 3D backbone produces must be the grid the head was built for
        b_cell, h_cell = getattr(backbone_3d, "bev_cell_size", None), getattr(head, "cell_size", None)
        if b_cell is not None and h_cell is not None and [float(x) for x in b_cell] != [float(x) for x in h_cell]:
            raise ValueError(f"backbone_3d BEV cell {b_cell} (bev_voxel_size x bev_stride) != head bev_cell_size {h_cell}")
        b_rng, h_rng = getattr(backbone_3d, "pc_range", None), getattr(head, "pc_range", None)
        if b_rng is not None and h_rng is not None and [float(x) for x in b_rng] != [float(x) for x in h_rng]:
            raise ValueError(f"backbone_3d pc_range {b_rng} != head pc_range {h_rng}")

    def forward(self, batch: Dict) -> Dict:
        point_tokens, bev = self.backbone_3d(batch["points"])

        image_features, calib = None, None
        images = batch.get("images")
        if self.backbone_2d is not None and images is not None:
            B, N = images.shape[:2]
            feats = self.backbone_2d(images.flatten(0, 1))
            image_features = {k: v.reshape(B, N, *v.shape[1:]) for k, v in feats.items()}
            calib = {"lidar2img": batch["lidar2img"], "image_hw": batch["image_hw"]}

        out = {"det": self.head(bev, image_features, calib)}
        out["seg_logits"] = self.seg_head(point_tokens) if self.seg_head is not None else None
        return out

    def loss(self, batch: Dict, preds: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total, logs = self.head.loss(preds["det"], batch["gt_boxes"], batch["gt_labels"])
        seg_labels = batch.get("seg_labels")
        if preds["seg_logits"] is not None and seg_labels is not None:
            labels = _concat_seg_labels(seg_labels, batch["points"], self.seg_ignore_index)
            if (labels != self.seg_ignore_index).any():
                loss_seg = F.cross_entropy(preds["seg_logits"], labels, ignore_index=self.seg_ignore_index)
            else:
                loss_seg = preds["seg_logits"].sum() * 0
            total = total + self.seg_loss_weight * loss_seg
            logs["loss_seg"] = loss_seg.detach()
        logs["loss"] = total.detach()
        return total, logs

    @torch.no_grad()
    def predict(self, batch: Dict, preds: Optional[Dict] = None) -> List[Dict[str, torch.Tensor]]:
        """Per-sample dicts with 'boxes', 'scores', 'labels' and, if segmenting, 'seg_pred' (N_i,)."""
        if preds is None:
            preds = self(batch)
        results = self.head.decode(preds["det"])
        if preds["seg_logits"] is not None:
            sizes = [p.shape[0] for p in batch["points"]]
            for r, logits in zip(results, preds["seg_logits"].split(sizes)):
                r["seg_pred"] = logits.argmax(-1)
        return results


def _concat_seg_labels(seg_labels, points, ignore_index) -> torch.Tensor:
    """Concatenate per-sample labels, filling samples without labels with ``ignore_index``."""
    out = []
    for lab, pts in zip(seg_labels, points):
        if lab is None:
            lab = torch.full((pts.shape[0],), ignore_index, dtype=torch.long, device=pts.device)
        out.append(lab.long())
    return torch.cat(out)
