from typing import Sequence, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from seeanythingfar.utils.box_ops import aligned_iou_3d


class HungarianAssigner3D:
    """One-to-one query <-> ground-truth matching (TransFusion / DETR style).

    Cost = focal classification cost + normalised BEV centre L1 + (negative) IoU.
    IoU uses axis-aligned envelopes (:func:`aligned_iou_3d`) so no compiled ops are needed;
    the reference TransFusion uses rotated IoU here.
    """

    def __init__(self, cls_weight: float = 0.15, reg_weight: float = 0.25, iou_weight: float = 0.25,
                 alpha: float = 0.25, gamma: float = 2.0):
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.iou_weight = iou_weight
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def assign(
        self,
        pred_boxes: torch.Tensor,     # (Q, 9) metric
        cls_logits: torch.Tensor,     # (Q, num_classes)
        gt_boxes: torch.Tensor,       # (M, 9)
        gt_labels: torch.Tensor,      # (M,)
        pc_range: Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``assigned`` (Q,) with the matched GT index or -1, and the matched IoUs (Q,)."""
        num_q = pred_boxes.shape[0]
        assigned = torch.full((num_q,), -1, dtype=torch.long, device=pred_boxes.device)
        ious_out = pred_boxes.new_zeros(num_q)
        if gt_boxes.shape[0] == 0 or num_q == 0:
            return assigned, ious_out

        p = cls_logits.sigmoid()
        eps = 1e-12
        neg = -(1 - p + eps).log() * (1 - self.alpha) * p.pow(self.gamma)
        pos = -(p + eps).log() * self.alpha * (1 - p).pow(self.gamma)
        cls_cost = (pos[:, gt_labels] - neg[:, gt_labels]) * self.cls_weight

        lo = pred_boxes.new_tensor(pc_range[0:2])
        span = pred_boxes.new_tensor(pc_range[3:5]) - lo
        reg_cost = torch.cdist((pred_boxes[:, :2] - lo) / span, (gt_boxes[:, :2] - lo) / span, p=1) * self.reg_weight

        iou = aligned_iou_3d(pred_boxes, gt_boxes)
        cost = cls_cost + reg_cost - iou * self.iou_weight

        rows, cols = linear_sum_assignment(cost.float().cpu().numpy())
        rows = torch.as_tensor(rows, device=pred_boxes.device, dtype=torch.long)
        cols = torch.as_tensor(cols, device=pred_boxes.device, dtype=torch.long)
        assigned[rows] = cols
        ious_out[rows] = iou[rows, cols]
        return assigned, ious_out
