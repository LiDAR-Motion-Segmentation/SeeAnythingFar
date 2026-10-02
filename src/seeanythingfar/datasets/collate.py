from typing import Dict, List

import numpy as np
import torch


def collate_fn(samples: List[Dict]) -> Dict:
    """Variable-size fields (points, boxes, seg labels) stay lists; cameras are stacked."""
    def t(x, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)

    batch = {
        "points": [t(s["points"]) for s in samples],
        "gt_boxes": [t(s["gt_boxes"]).reshape(-1, 9) for s in samples],
        "gt_labels": [t(s["gt_labels"], torch.long) for s in samples],
        "meta": [s.get("meta", {}) for s in samples],
    }
    if any(s.get("seg_labels") is not None for s in samples):
        batch["seg_labels"] = [None if s.get("seg_labels") is None else t(s["seg_labels"], torch.long)
                               for s in samples]
    if samples[0].get("images") is not None:
        batch["images"] = torch.stack([t(s["images"]) for s in samples])
        batch["lidar2img"] = torch.stack([t(s["lidar2img"]) for s in samples])
        batch["image_hw"] = tuple(samples[0]["image_hw"])
    return batch


def move_to_device(batch: Dict, device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and (torch.is_tensor(v[0]) or v[0] is None):
            out[k] = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in v]
        else:
            out[k] = v
    return out
