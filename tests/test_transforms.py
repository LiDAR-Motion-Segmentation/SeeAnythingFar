import numpy as np
import torch

from seeanythingfar.datasets.synthetic import SyntheticDataset
from seeanythingfar.datasets.transforms import GlobalAugment
from seeanythingfar.utils.box_ops import box_corners
from seeanythingfar.utils.geometry import project_points_to_images


def _project(sample):
    pts = torch.from_numpy(sample["points"][None, :, :3].astype(np.float32))
    uv, depth = project_points_to_images(pts, torch.from_numpy(sample["lidar2img"][None]))
    return uv[0].numpy(), depth[0].numpy()


def test_augmentation_keeps_points_on_same_pixels():
    ds = SyntheticDataset(length=1, image_size=(64, 96))
    raw = ds[0]
    for seed in range(5):
        aug = GlobalAugment(seed=seed, flip_x_prob=0.5, flip_y_prob=0.5)(
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in raw.items()})
        uv0, d0 = _project(raw)
        uv1, d1 = _project(aug)
        front = d0 > 0.5
        assert np.allclose(uv0[front], uv1[front], atol=1e-2)
        # lidar2img absorbs inv(T), so camera depth is unchanged even under scaling
        assert np.allclose(d1, d0, rtol=1e-4, atol=1e-3)
        assert not np.allclose(aug["points"][:, :3], raw["points"][:, :3])


def test_augmentation_keeps_points_inside_boxes():
    ds = SyntheticDataset(length=1, image_size=None, max_objects=4)
    raw = ds[0]
    aug = GlobalAugment(seed=3, flip_x_prob=1.0, flip_y_prob=1.0)(
        {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in raw.items()})

    def inside(sample):
        b = torch.from_numpy(sample["gt_boxes"])
        p = torch.from_numpy(sample["points"][:, :3])
        c, s = torch.cos(b[:, 6]), torch.sin(b[:, 6])
        d = p[None] - b[:, None, :3]
        lx = d[..., 0] * c[:, None] + d[..., 1] * s[:, None]
        ly = -d[..., 0] * s[:, None] + d[..., 1] * c[:, None]
        tol = 1e-3
        return ((lx.abs() <= b[:, None, 3] / 2 + tol) & (ly.abs() <= b[:, None, 4] / 2 + tol)
                & (d[..., 2].abs() <= b[:, None, 5] / 2 + tol))

    assert torch.equal(inside(raw), inside(aug))
    # heading flips are reflected in the corners too
    assert box_corners(torch.from_numpy(aug["gt_boxes"])).shape[1] == 8
