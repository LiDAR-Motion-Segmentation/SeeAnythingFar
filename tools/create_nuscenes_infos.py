"""Build the info pickles read by ``seeanythingfar.datasets.nuscenes.NuScenesDataset``.

    python tools/create_nuscenes_infos.py --data-root /path/to/nuscenes --version v1.0-trainval \
        --out-dir /path/to/nuscenes --max-sweeps 10

Writes ``nuscenes_infos_{train,val}.pkl`` (or ``_test`` / ``mini_*`` variants). Paths inside
are relative to ``--data-root``. Needs ``nuscenes-devkit`` (pip install nuscenes-devkit).

Per sample: LiDAR path + sweeps (with sweep->keyframe transforms), all 6 cameras (path,
intrinsics, lidar->camera extrinsics using each camera's own ego pose), GT boxes in the
LiDAR frame as (x, y, z, l, w, h, yaw, vx, vy) and the lidarseg label path if available.
"""
import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from seeanythingfar.datasets.constants import (CAMERA_NAMES, GENERAL_TO_DET, GENERAL_TO_SEG,  # noqa: E402
                                               NUSCENES_SEG_CLASSES, SEG_IGNORE_INDEX)


def transform(translation, rotation, inverse=False):
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    return transform_matrix(translation, Quaternion(rotation), inverse=inverse)


def sensor_to_global(nusc, sd_token):
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    s2e = transform(cs["translation"], cs["rotation"])
    e2g = transform(pose["translation"], pose["rotation"])
    return sd, cs, s2e, e2g


def split_scenes(nusc, version):
    from nuscenes.utils import splits
    if version == "v1.0-trainval":
        names = {"train": splits.train, "val": splits.val}
    elif version == "v1.0-mini":
        names = {"mini_train": splits.mini_train, "mini_val": splits.mini_val}
    elif version == "v1.0-test":
        names = {"test": splits.test}
    else:
        raise ValueError(version)
    by_name = {s["name"]: s["token"] for s in nusc.scene}
    return {k: {by_name[n] for n in v if n in by_name} for k, v in names.items()}


def lidarseg_map(nusc):
    """general category index (as stored in lidarseg .bin files) -> challenge class or 255."""
    table = np.full(256, SEG_IGNORE_INDEX, dtype=np.uint8)
    for name, idx in nusc.lidarseg_name2idx_mapping.items():
        if name in GENERAL_TO_SEG:
            table[idx] = NUSCENES_SEG_CLASSES.index(GENERAL_TO_SEG[name])
    return table


def make_info(nusc, sample, data_root, max_sweeps, with_ann, has_lidarseg):
    from pyquaternion import Quaternion

    lidar_token = sample["data"]["LIDAR_TOP"]
    sd, cs, lidar2ego, ego2global = sensor_to_global(nusc, lidar_token)
    lidar2global = ego2global @ lidar2ego
    global2lidar = np.linalg.inv(lidar2global)
    rel = lambda p: os.path.relpath(p, data_root)  # noqa: E731

    info = {
        "token": sample["token"],
        "scene_token": sample["scene_token"],
        "timestamp": sample["timestamp"],
        "lidar_path": rel(nusc.get_sample_data_path(lidar_token)),
        "lidar2ego": lidar2ego.astype(np.float32),
        "ego2global": ego2global.astype(np.float32),
        "sweeps": [],
        "cams": {},
    }

    # previous non-keyframe sweeps, expressed in the keyframe LiDAR frame
    cur = sd
    while len(info["sweeps"]) < max_sweeps - 1 and cur["prev"]:
        cur = nusc.get("sample_data", cur["prev"])
        _, _, s2e, e2g = sensor_to_global(nusc, cur["token"])
        info["sweeps"].append({
            "lidar_path": rel(nusc.get_sample_data_path(cur["token"])),
            "sweep2lidar": (global2lidar @ e2g @ s2e).astype(np.float32),
            "time_lag": float(sd["timestamp"] - cur["timestamp"]) * 1e-6,
        })

    for cam in CAMERA_NAMES:
        cam_token = sample["data"][cam]
        csd, ccs, cam2ego, cam_e2g = sensor_to_global(nusc, cam_token)
        lidar2cam = np.linalg.inv(cam_e2g @ cam2ego) @ lidar2global
        info["cams"][cam] = {
            "img_path": rel(nusc.get_sample_data_path(cam_token)),
            "intrinsic": np.asarray(ccs["camera_intrinsic"], dtype=np.float32),
            "lidar2cam": lidar2cam.astype(np.float32),
            "width": csd["width"],
            "height": csd["height"],
        }

    if has_lidarseg:
        try:
            info["lidarseg_path"] = nusc.get("lidarseg", lidar_token)["filename"]
        except KeyError:
            info["lidarseg_path"] = None

    if with_ann:
        boxes, names, num_pts = [], [], []
        r_lidar = np.linalg.inv(lidar2global[:3, :3])
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:
                continue
            box = nusc.get_box(ann_token)
            center = global2lidar[:3, :3] @ box.center + global2lidar[:3, 3]
            rot = Quaternion(matrix=r_lidar) * box.orientation
            heading = rot.rotation_matrix @ np.array([1.0, 0.0, 0.0])
            yaw = np.arctan2(heading[1], heading[0])
            vel = nusc.box_velocity(ann_token)  # global frame, NaN if undefined
            vel = r_lidar @ np.nan_to_num(vel)
            w, l, h = box.wlh
            boxes.append([*center, l, w, h, yaw, vel[0], vel[1]])
            names.append(GENERAL_TO_DET.get(ann["category_name"], "ignore"))
            num_pts.append(ann["num_lidar_pts"])
        info["gt_boxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 9)
        info["gt_names"] = np.asarray(names)
        info["num_lidar_pts"] = np.asarray(num_pts, dtype=np.int64)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-sweeps", type=int, default=10)
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    has_lidarseg = hasattr(nusc, "lidarseg") and len(getattr(nusc, "lidarseg", [])) > 0
    seg_table = lidarseg_map(nusc) if has_lidarseg else None
    with_ann = args.version != "v1.0-test"

    os.makedirs(args.out_dir, exist_ok=True)
    for split, scene_tokens in split_scenes(nusc, args.version).items():
        infos = [make_info(nusc, s, args.data_root, args.max_sweeps, with_ann, has_lidarseg)
                 for s in tqdm(nusc.sample, desc=split) if s["scene_token"] in scene_tokens]
        out = os.path.join(args.out_dir, f"nuscenes_infos_{split}.pkl")
        with open(out, "wb") as f:
            pickle.dump({"version": args.version, "lidarseg_map": seg_table, "infos": infos}, f)
        print(f"wrote {len(infos)} infos -> {out}")


if __name__ == "__main__":
    main()
