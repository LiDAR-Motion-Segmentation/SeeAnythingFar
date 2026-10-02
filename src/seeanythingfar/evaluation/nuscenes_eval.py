"""Official nuScenes detection evaluation (mAP, NDS). Needs ``nuscenes-devkit``."""
import json
import os
from typing import Dict, List, Sequence

import numpy as np

# the attribute each class gets when the model does not predict one (as in CenterPoint/TransFusion)
DEFAULT_ATTRIBUTE = {
    "car": "vehicle.parked", "truck": "vehicle.parked", "construction_vehicle": "vehicle.parked",
    "bus": "vehicle.moving", "trailer": "vehicle.parked", "motorcycle": "cycle.without_rider",
    "bicycle": "cycle.without_rider", "pedestrian": "pedestrian.moving", "barrier": "", "traffic_cone": "",
}


def to_nuscenes_boxes(result: Dict, meta: Dict, class_names: Sequence[str]) -> List[Dict]:
    """One sample's predictions (LiDAR frame) -> nuScenes submission entries (global frame)."""
    from pyquaternion import Quaternion

    lidar2global = np.asarray(meta["ego2global"]) @ np.asarray(meta["lidar2ego"])
    inv_aug = np.linalg.inv(np.asarray(meta.get("aug_matrix", np.eye(4))))
    T = lidar2global @ inv_aug
    R = T[:3, :3]
    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    labels = result["labels"].detach().cpu().numpy()
    out = []
    for b, s, l in zip(boxes, scores, labels):
        name = class_names[int(l)]
        center = R @ b[:3] + T[:3, 3]
        yaw_q = Quaternion(axis=[0, 0, 1], radians=float(b[6]))
        rot = Quaternion(matrix=R / np.cbrt(np.linalg.det(R))) * yaw_q
        vel = R @ np.array([b[7], b[8], 0.0])
        attr = DEFAULT_ATTRIBUTE.get(name, "")
        if name in ("car", "truck", "construction_vehicle", "trailer") and np.hypot(vel[0], vel[1]) > 0.2:
            attr = "vehicle.moving"
        out.append({
            "sample_token": meta["token"],
            "translation": center.tolist(),
            "size": [float(b[4]), float(b[3]), float(b[5])],  # nuScenes wants (w, l, h)
            "rotation": rot.elements.tolist(),
            "velocity": vel[:2].tolist(),
            "detection_name": name,
            "detection_score": float(s),
            "attribute_name": attr,
        })
    return out


def write_submission(entries: Dict[str, List[Dict]], path: str, use_camera: bool = True) -> str:
    sub = {"meta": {"use_camera": use_camera, "use_lidar": True, "use_radar": False,
                    "use_map": False, "use_external": False},
           "results": entries}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sub, f)
    return path


def run_official_eval(result_path: str, data_root: str, version: str, eval_set: str, output_dir: str) -> Dict:
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    evaluator = NuScenesEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=result_path,
                             eval_set=eval_set, output_dir=output_dir, verbose=True)
    return evaluator.main(render_curves=False)
