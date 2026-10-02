import numpy as np

from seeanythingfar.evaluation.metrics import DetectionEvaluator, SegmentationEvaluator


def test_perfect_detections_score_one_and_bins():
    ev = DetectionEvaluator(["a", "b"])
    gt = np.array([[10.0, 0, 0], [40.0, 0, 0], [0, 60.0, 0]])
    ev.update(gt, np.array([0.9, 0.8, 0.7]), np.array([0, 1, 0]), gt, np.array([0, 1, 0]))
    m = ev.compute()
    for k in ("mAP/all", "mAP/0-30m", "mAP/30-50m", "mAP/50m+", "AP/a", "AP/b"):
        assert abs(m[k] - 1.0) < 1e-9, k


def test_misses_and_false_positives():
    ev = DetectionEvaluator(["a"])
    gt = np.array([[10.0, 0, 0], [20.0, 0, 0]])
    pred = np.array([[10.2, 0, 0], [50.0, 50.0, 0]])                  # one TP, one far FP
    ev.update(pred, np.array([0.9, 0.95]), np.array([0, 0]), gt, np.array([0, 0]))
    m = ev.compute()
    assert 0.0 < m["mAP/all"] < 1.0


def test_segmentation_miou():
    ev = SegmentationEvaluator(["x", "y", "z"])
    target = np.array([0, 0, 1, 1, 255])
    ev.update(np.array([0, 0, 1, 0, 2]), target)
    m = ev.compute()
    assert m["IoU/x"] == 2 / 3 and m["IoU/y"] == 0.5 and np.isnan(m["IoU/z"])
