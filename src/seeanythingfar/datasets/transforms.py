"""Global LiDAR augmentations that keep boxes and camera calibration consistent.

Every augmentation is a rigid(+scale) map ``p_new = T @ p_old`` of the LiDAR frame, so the
cameras still see the same pixels if ``lidar2img`` becomes ``lidar2img @ inv(T)``.
Images themselves are never warped.
"""
from typing import Dict, Optional, Sequence

import numpy as np


def apply_global_transform(sample: Dict, T: np.ndarray, yaw_fn, vel_fn, size_scale: float = 1.0) -> Dict:
    """Apply 4x4 ``T`` to points / boxes / calibration of a sample (in place)."""
    pts = sample["points"]
    pts[:, :3] = pts[:, :3] @ T[:3, :3].T + T[:3, 3]

    boxes = sample.get("gt_boxes")
    if boxes is not None and len(boxes):
        boxes[:, :3] = boxes[:, :3] @ T[:3, :3].T + T[:3, 3]
        boxes[:, 3:6] *= size_scale
        boxes[:, 6] = yaw_fn(boxes[:, 6])
        boxes[:, 7:9] = vel_fn(boxes[:, 7:9])

    if sample.get("lidar2img") is not None:
        sample["lidar2img"] = sample["lidar2img"] @ np.linalg.inv(T).astype(sample["lidar2img"].dtype)
    sample["aug_matrix"] = T @ sample.get("aug_matrix", np.eye(4))
    return sample


class GlobalAugment:
    def __init__(self, rot_range: Sequence[float] = (-0.785, 0.785), scale_range: Sequence[float] = (0.95, 1.05),
                 flip_x_prob: float = 0.5, flip_y_prob: float = 0.5, seed: Optional[int] = None):
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.flip_x_prob = flip_x_prob
        self.flip_y_prob = flip_y_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: Dict) -> Dict:
        if self.rng.random() < self.flip_y_prob:  # mirror across the x axis: y -> -y
            T = np.diag([1.0, -1.0, 1.0, 1.0])
            apply_global_transform(sample, T, lambda y: -y, lambda v: v * np.array([1.0, -1.0], dtype=v.dtype))
        if self.rng.random() < self.flip_x_prob:  # mirror across the y axis: x -> -x
            T = np.diag([-1.0, 1.0, 1.0, 1.0])
            apply_global_transform(sample, T, lambda y: np.pi - y, lambda v: v * np.array([-1.0, 1.0], dtype=v.dtype))

        theta = self.rng.uniform(*self.rot_range)
        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        R2 = T[:2, :2].astype(np.float32)
        apply_global_transform(sample, T, lambda y: y + theta, lambda v: v @ R2.T)

        scale = self.rng.uniform(*self.scale_range)
        T = np.diag([scale, scale, scale, 1.0])
        apply_global_transform(sample, T, lambda y: y, lambda v: v * scale, size_scale=scale)

        if sample.get("gt_boxes") is not None and len(sample["gt_boxes"]):
            sample["gt_boxes"][:, 6] = limit_period(sample["gt_boxes"][:, 6])
        return sample


def limit_period(yaw: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def filter_boxes_by_range(boxes: np.ndarray, labels: np.ndarray, pc_range: Sequence[float]):
    keep = (
        (boxes[:, 0] >= pc_range[0]) & (boxes[:, 0] < pc_range[3])
        & (boxes[:, 1] >= pc_range[1]) & (boxes[:, 1] < pc_range[4])
    )
    return boxes[keep], labels[keep]
