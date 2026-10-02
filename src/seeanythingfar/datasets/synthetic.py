"""Procedural scenes for smoke-testing the full pipeline without nuScenes on disk.

A ground plane plus random boxes whose surfaces are sampled as LiDAR points, six cameras
in a ring at the LiDAR origin, and images in which every box's projection is painted in a
class-specific colour, so image features do carry signal. Not a benchmark.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from seeanythingfar.datasets.transforms import GlobalAugment, filter_boxes_by_range
from seeanythingfar.utils.box_ops import box_corners


def ring_cameras(num_cams: int, image_hw, fov_deg: float = 70.0, height: float = 1.5) -> np.ndarray:
    """(N, 4, 4) lidar2img for cameras evenly spaced in yaw, looking horizontally outwards."""
    H, W = image_hw
    f = (W / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    K = np.array([[f, 0, W / 2, 0], [0, f, H / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mats = []
    for i in range(num_cams):
        yaw = 2 * np.pi * i / num_cams
        z = np.array([np.cos(yaw), np.sin(yaw), 0.0])   # optical axis
        x = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # image right
        y = np.array([0.0, 0.0, -1.0])                  # image down
        R = np.stack([x, y, z], axis=1)                 # camera -> lidar rotation
        t = np.array([0.0, 0.0, height])
        lidar2cam = np.eye(4)
        lidar2cam[:3, :3] = R.T
        lidar2cam[:3, 3] = -R.T @ t
        mats.append(K @ lidar2cam)
    return np.stack(mats).astype(np.float32)


class SyntheticDataset(Dataset):
    def __init__(
        self,
        length: int = 16,
        pc_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_classes: int = 10,
        num_seg_classes: int = 16,
        max_objects: int = 12,
        points_per_object: int = 400,
        ground_points: int = 4000,
        image_size: Optional[Sequence[int]] = (224, 384),
        num_cams: int = 6,
        augment: Optional[GlobalAugment] = None,
        seed: int = 0,
    ):
        self.length = length
        self.pc_range = list(pc_range)
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.max_objects = max_objects
        self.points_per_object = points_per_object
        self.ground_points = ground_points
        self.image_size = tuple(image_size) if image_size is not None else None
        self.num_cams = num_cams
        self.augment = augment
        self.seed = seed
        self.ground_seg_class = num_seg_classes - 1
        self.palette = np.random.default_rng(1234).uniform(0.2, 1.0, (num_classes, 3)).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Dict:
        rng = np.random.default_rng(self.seed * 100003 + idx)
        r = 0.8 * min(self.pc_range[3], self.pc_range[4])
        n = int(rng.integers(1, self.max_objects + 1))
        labels = rng.integers(0, self.num_classes, n)
        dims = np.stack([rng.uniform(1.0, 5.0, n), rng.uniform(0.8, 2.5, n), rng.uniform(1.0, 2.5, n)], 1)
        xy = rng.uniform(-r, r, (n, 2))
        z = dims[:, 2] / 2 - 1.8                        # resting on a ground plane at z = -1.8
        yaw = rng.uniform(-np.pi, np.pi, n)
        vel = rng.normal(0, 2, (n, 2))
        boxes = np.concatenate([xy, z[:, None], dims, yaw[:, None], vel], 1).astype(np.float32)

        pts, seg = [], []
        for i in range(n):  # points on the box surfaces
            local = rng.uniform(-0.5, 0.5, (self.points_per_object, 3))
            face = rng.integers(0, 3, self.points_per_object)
            local[np.arange(self.points_per_object), face] = np.sign(local[np.arange(self.points_per_object), face]) * 0.5
            local *= dims[i]
            c, s = np.cos(yaw[i]), np.sin(yaw[i])
            world = local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + boxes[i, :3]
            pts.append(world)
            seg.append(np.full(len(world), labels[i] % (self.num_seg_classes - 1)))
        ground = np.concatenate([rng.uniform(-r, r, (self.ground_points, 2)),
                                 np.full((self.ground_points, 1), -1.8)], 1)
        pts.append(ground)
        seg.append(np.full(len(ground), self.ground_seg_class))
        xyz = np.concatenate(pts).astype(np.float32)
        intensity = rng.uniform(0, 255, (len(xyz), 1)).astype(np.float32)
        points = np.concatenate([xyz, intensity, np.zeros_like(intensity)], 1)

        sample = {"points": points, "gt_boxes": boxes, "gt_labels": labels.astype(np.int64),
                  "seg_labels": np.concatenate(seg).astype(np.int64), "meta": {"token": f"synthetic_{idx}"}}
        if self.image_size is not None:
            sample["lidar2img"] = ring_cameras(self.num_cams, self.image_size)
            sample["images"] = self._render(boxes, labels, sample["lidar2img"], rng)
            sample["image_hw"] = self.image_size

        if self.augment is not None:
            sample = self.augment(sample)
        sample.pop("aug_matrix", None)
        sample["gt_boxes"], sample["gt_labels"] = filter_boxes_by_range(sample["gt_boxes"], sample["gt_labels"], self.pc_range)
        return sample

    def _render(self, boxes, labels, lidar2img, rng):
        """Paint each box's projected 2D envelope into every camera that sees it."""
        H, W = self.image_size
        imgs = rng.uniform(0, 0.15, (self.num_cams, 3, H, W)).astype(np.float32)
        corners = box_corners(torch.from_numpy(boxes)).numpy()  # (n, 8, 3)
        homo = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], -1)
        for c in range(self.num_cams):
            proj = homo @ lidar2img[c].T
            for i in range(len(boxes)):
                d = proj[i, :, 2]
                if (d <= 0.5).any():
                    continue
                u, v = proj[i, :, 0] / d, proj[i, :, 1] / d
                u0, u1 = int(np.clip(u.min(), 0, W)), int(np.clip(u.max(), 0, W))
                v0, v1 = int(np.clip(v.min(), 0, H)), int(np.clip(v.max(), 0, H))
                if u1 > u0 and v1 > v0:
                    imgs[c, :, v0:v1, u0:u1] = self.palette[labels[i]][:, None, None]
        return imgs
