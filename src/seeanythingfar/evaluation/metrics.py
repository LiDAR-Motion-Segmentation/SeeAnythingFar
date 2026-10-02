"""Dependency-free metrics, usable during validation without the nuScenes devkit.

* :class:`DetectionEvaluator` - nuScenes-style AP (BEV centre-distance matching, thresholds
  0.5/1/2/4 m, AP from recall >= 0.1 with precision floor 0.1), reported overall and per
  **range bin**, since how detection decays with distance is the point of this project.
  It does not compute the TP error terms (ATE/ASE/AOE/AVE/AAE) or NDS; use
  ``evaluation/nuscenes_eval.py`` for the official numbers.
* :class:`SegmentationEvaluator` - confusion-matrix mIoU.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np


def _ap_from_matches(tp: np.ndarray, num_gt: int, min_recall: float = 0.1, min_precision: float = 0.1) -> float:
    """nuScenes ``calc_ap`` on score-sorted true-positive flags."""
    if num_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(1 - tp)
    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    rec = tp_c / num_gt
    rec_interp = np.linspace(0, 1, 101)
    prec_interp = np.interp(rec_interp, rec, prec, right=0)  # no precision envelope, as in nuScenes
    prec_interp = prec_interp[round(100 * min_recall) + 1:]
    prec_interp = np.clip(prec_interp - min_precision, 0, None)
    return float(prec_interp.mean() / (1 - min_precision))


class DetectionEvaluator:
    def __init__(self, class_names: Sequence[str], dist_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
                 range_bins: Sequence[float] = (0.0, 30.0, 50.0, float("inf"))):
        self.class_names = list(class_names)
        self.dist_thresholds = list(dist_thresholds)
        self.range_bins = list(range_bins)
        self.reset()

    def reset(self):
        self.preds: List[Dict[str, np.ndarray]] = []
        self.gts: List[Dict[str, np.ndarray]] = []

    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """All arrays for one sample; boxes (M, >=2) in the LiDAR frame."""
        a = lambda x: np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)  # noqa: E731
        self.preds.append({"xy": a(pred_boxes)[:, :2].reshape(-1, 2), "score": a(pred_scores).reshape(-1),
                           "label": a(pred_labels).reshape(-1)})
        self.gts.append({"xy": a(gt_boxes)[:, :2].reshape(-1, 2), "label": a(gt_labels).reshape(-1)})

    def _ap(self, cls: int, thr: float, lo: float, hi: float) -> float:
        scores, flags, num_gt = [], [], 0
        for p, g in zip(self.preds, self.gts):
            gm = (g["label"] == cls)
            gxy = g["xy"][gm]
            gxy = gxy[_in_range(gxy, lo, hi)]
            pm = (p["label"] == cls)
            pxy, ps = p["xy"][pm], p["score"][pm]
            keep = _in_range(pxy, lo, hi)
            pxy, ps = pxy[keep], ps[keep]
            num_gt += len(gxy)
            order = np.argsort(-ps)
            taken = np.zeros(len(gxy), dtype=bool)
            for j in order:
                tp = 0
                if len(gxy):
                    d = np.linalg.norm(gxy - pxy[j], axis=1)
                    d[taken] = np.inf
                    k = int(np.argmin(d))
                    if d[k] < thr:
                        taken[k] = True
                        tp = 1
                scores.append(ps[j])
                flags.append(tp)
        order = np.argsort(-np.asarray(scores)) if scores else np.zeros(0, dtype=int)
        return _ap_from_matches(np.asarray(flags, dtype=np.float64)[order], num_gt)

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        bins = [(0.0, float("inf"), "all")] + [
            (lo, hi, f"{int(lo)}m+" if hi == float("inf") else f"{int(lo)}-{int(hi)}m")
            for lo, hi in zip(self.range_bins[:-1], self.range_bins[1:])
        ]
        for lo, hi, tag in bins:
            per_class = []
            for c, name in enumerate(self.class_names):
                aps = [self._ap(c, t, lo, hi) for t in self.dist_thresholds]
                ap = float(np.nanmean(aps)) if not all(np.isnan(aps)) else float("nan")
                if tag == "all":
                    out[f"AP/{name}"] = ap
                per_class.append(ap)
            valid = [x for x in per_class if not np.isnan(x)]
            out[f"mAP/{tag}"] = float(np.mean(valid)) if valid else float("nan")
        return out


def _in_range(xy: np.ndarray, lo: float, hi: float) -> np.ndarray:
    r = np.linalg.norm(xy, axis=1)
    return (r >= lo) & (r < hi)


class SegmentationEvaluator:
    def __init__(self, class_names: Sequence[str], ignore_index: int = 255):
        self.class_names = list(class_names)
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        n = len(self.class_names)
        self.confusion = np.zeros((n, n), dtype=np.int64)

    def update(self, pred, target):
        a = lambda x: np.asarray(x.detach().cpu() if hasattr(x, "detach") else x).reshape(-1)  # noqa: E731
        pred, target = a(pred), a(target)
        m = target != self.ignore_index
        n = len(self.class_names)
        self.confusion += np.bincount(target[m] * n + pred[m], minlength=n * n).reshape(n, n)

    def compute(self) -> Dict[str, float]:
        tp = np.diag(self.confusion).astype(np.float64)
        denom = self.confusion.sum(0) + self.confusion.sum(1) - tp
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        out = {f"IoU/{n}": float(v) for n, v in zip(self.class_names, iou)}
        out["mIoU"] = float(np.nanmean(iou)) if not np.isnan(iou).all() else float("nan")
        return out
