"""nuScenes LiDAR + 6-camera dataset with detection boxes and (optional) lidarseg labels.

Reads the info pickles written by ``tools/create_nuscenes_infos.py``; the nuScenes devkit
is only needed to create those infos and for the official evaluation, not for training.
"""
import os
import pickle
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from seeanythingfar.datasets.constants import CAMERA_NAMES, NUSCENES_DET_CLASSES, SEG_IGNORE_INDEX
from seeanythingfar.datasets.transforms import GlobalAugment, filter_boxes_by_range


class NuScenesDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        info_path: str,
        pc_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        class_names: Sequence[str] = NUSCENES_DET_CLASSES,
        image_size: Optional[Sequence[int]] = (448, 800),   # (H, W) network input; None = LiDAR only
        cameras: Sequence[str] = CAMERA_NAMES,
        max_sweeps: int = 1,                                # 1 = keyframe only
        load_seg: bool = True,
        augment: Optional[GlobalAugment] = None,
        min_lidar_pts: int = 1,
        load_interval: int = 1,                             # subsample infos, e.g. 4 for quick runs
    ):
        self.data_root = data_root
        with open(info_path, "rb") as f:
            data = pickle.load(f)
        self.infos = data["infos"][::load_interval]
        self.seg_map = np.asarray(data["lidarseg_map"], dtype=np.uint8) if data.get("lidarseg_map") is not None else None
        self.pc_range = list(pc_range)
        self.class_names = list(class_names)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.cameras = list(cameras)
        self.max_sweeps = max_sweeps
        self.load_seg = load_seg and self.seg_map is not None
        self.augment = augment
        self.min_lidar_pts = min_lidar_pts

    def __len__(self):
        return len(self.infos)

    def _path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.data_root, rel)

    def _load_points(self, info) -> np.ndarray:
        """(N, 5): x, y, z, intensity, time_lag. Keyframe points come first, then sweeps."""
        pts = np.fromfile(self._path(info["lidar_path"]), dtype=np.float32).reshape(-1, 5)[:, :4]
        out = [np.concatenate([pts, np.zeros((len(pts), 1), np.float32)], 1)]
        for sweep in info.get("sweeps", [])[: self.max_sweeps - 1]:
            sp = np.fromfile(self._path(sweep["lidar_path"]), dtype=np.float32).reshape(-1, 5)[:, :4]
            sp = sp[np.linalg.norm(sp[:, :2], axis=1) > 1.0]  # drop returns from the ego vehicle
            T = sweep["sweep2lidar"]
            sp[:, :3] = sp[:, :3] @ T[:3, :3].T + T[:3, 3]
            out.append(np.concatenate([sp, np.full((len(sp), 1), sweep["time_lag"], np.float32)], 1))
        return np.concatenate(out, 0)

    def _load_images(self, info):
        H, W = self.image_size
        images, lidar2img = [], []
        for name in self.cameras:
            cam = info["cams"][name]
            img = Image.open(self._path(cam["img_path"])).convert("RGB")
            sx, sy = W / img.width, H / img.height
            img = np.asarray(img.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
            images.append(img.transpose(2, 0, 1))
            K = np.eye(4)
            K[:3, :3] = cam["intrinsic"]
            K[0] *= sx
            K[1] *= sy
            lidar2img.append(K @ cam["lidar2cam"])
        return np.stack(images).astype(np.float32), np.stack(lidar2img).astype(np.float32)

    def __getitem__(self, idx) -> Dict:
        info = self.infos[idx]
        points = self._load_points(info)
        n_key = int((points[:, 4] == 0).sum()) if self.max_sweeps > 1 else len(points)

        names = np.asarray(info.get("gt_names", []))
        boxes = np.asarray(info.get("gt_boxes", np.zeros((0, 9))), dtype=np.float32).reshape(-1, 9)
        keep = np.isin(names, self.class_names)
        if "num_lidar_pts" in info:
            keep &= np.asarray(info["num_lidar_pts"]) >= self.min_lidar_pts
        boxes = boxes[keep]
        boxes[:, 7:9] = np.nan_to_num(boxes[:, 7:9])  # some annotations lack velocity
        labels = np.array([self.class_names.index(n) for n in names[keep]], dtype=np.int64)

        sample = {"points": points, "gt_boxes": boxes, "gt_labels": labels,
                  "meta": {"token": info["token"], "lidar2ego": info["lidar2ego"], "ego2global": info["ego2global"]}}

        if self.load_seg:
            seg = np.full(len(points), SEG_IGNORE_INDEX, dtype=np.int64)
            if info.get("lidarseg_path"):
                raw = np.fromfile(self._path(info["lidarseg_path"]), dtype=np.uint8)
                seg[:n_key] = self.seg_map[raw]
            sample["seg_labels"] = seg

        if self.image_size is not None:
            sample["images"], sample["lidar2img"] = self._load_images(info)
            sample["image_hw"] = self.image_size

        if self.augment is not None:
            sample = self.augment(sample)
        sample["meta"]["aug_matrix"] = sample.pop("aug_matrix", np.eye(4))
        sample["gt_boxes"], sample["gt_labels"] = filter_boxes_by_range(sample["gt_boxes"], sample["gt_labels"], self.pc_range)
        return sample
