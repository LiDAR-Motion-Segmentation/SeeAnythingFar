"""Render one validation sample: BEV (points + GT/pred boxes) and every camera with projected boxes.

    python visualize.py +index=0                         # ground truth only
    python visualize.py +ckpt=.../last.ckpt +index=0 +score_threshold=0.3 +out=viz
    python visualize.py --config-name smoke +index=0     # synthetic data

Writes ``<out>/<token>_bev.png`` and ``<out>/<token>_cams.png``. Points are coloured by
predicted segmentation class when a checkpoint with a segmentation head is given, else by height.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import hydra  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from seeanythingfar.datasets.collate import collate_fn, move_to_device  # noqa: E402
from seeanythingfar.utils.box_ops import box_corners  # noqa: E402
from seeanythingfar.utils.build import make_system  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_bev(ax, points, colors, gt, pred, pc_range):
    ax.scatter(points[:, 0], points[:, 1], s=0.2, c=colors, cmap="tab20" if colors.dtype.kind in "iu" else "viridis")
    for boxes, color in ((gt, "lime"), (pred, "red")):
        if len(boxes) == 0:
            continue
        corners = box_corners(torch.as_tensor(boxes, dtype=torch.float32)).numpy()[:, :4, :2]
        for c, b in zip(corners, boxes):
            ax.plot(*np.vstack([c, c[:1]]).T, color=color, lw=0.8)
            ax.plot([b[0], (c[0, 0] + c[3, 0]) / 2], [b[1], (c[0, 1] + c[3, 1]) / 2], color=color, lw=0.8)  # heading
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect("equal")
    ax.set_title("BEV  (green = GT, red = prediction)")


def draw_cam(ax, image, lidar2img, gt, pred):
    H, W = image.shape[1:]
    ax.imshow(image.transpose(1, 2, 0).clip(0, 1))
    for boxes, color in ((gt, "lime"), (pred, "red")):
        if len(boxes) == 0:
            continue
        corners = box_corners(torch.as_tensor(boxes, dtype=torch.float32)).numpy()
        homo = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], -1) @ lidar2img.T
        for c in homo:
            if (c[:, 2] < 0.5).any():
                continue
            uv = c[:, :2] / c[:, 2:3]
            if ((uv[:, 0] < 0) | (uv[:, 0] >= W) | (uv[:, 1] < 0) | (uv[:, 1] >= H)).all():
                continue
            for i, j in EDGES:
                ax.plot(uv[[i, j], 0], uv[[i, j], 1], color=color, lw=0.6)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    index = cfg.get("index", 0)
    out = cfg.get("out", "viz")
    thr = cfg.get("score_threshold", 0.3)
    os.makedirs(out, exist_ok=True)

    dataset = instantiate(cfg.dataset.val)
    batch = collate_fn([dataset[index]])
    points = batch["points"][0].numpy()
    gt = batch["gt_boxes"][0].numpy()
    pred = np.zeros((0, 9), np.float32)
    colors = points[:, 2]

    if cfg.get("ckpt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        system = make_system(cfg)
        state = torch.load(cfg.ckpt, map_location="cpu")
        system.load_state_dict(state.get("state_dict", state))
        model = system.model.to(device).eval()
        result = model.predict(move_to_device(batch, device))[0]
        keep = result["scores"] > thr
        pred = result["boxes"][keep].cpu().numpy()
        if "seg_pred" in result:
            colors = result["seg_pred"].cpu().numpy()

    token = batch["meta"][0].get("token", str(index))
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_bev(ax, points, colors, gt, pred, cfg.pc_range)
    fig.savefig(os.path.join(out, f"{token}_bev.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if "images" in batch:
        images, l2i = batch["images"][0].numpy(), batch["lidar2img"][0].numpy()
        n = images.shape[0]
        cols = 3
        fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(6 * cols, 3.5 * ((n + cols - 1) // cols)))
        for i, ax in enumerate(np.atleast_1d(axes).ravel()):
            if i < n:
                draw_cam(ax, images[i], l2i[i], gt, pred)
            else:
                ax.axis("off")
        fig.savefig(os.path.join(out, f"{token}_cams.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote visualisations for {token} to {out}/")


if __name__ == "__main__":
    main()
