import math

import numpy as np
import torch

from seeanythingfar.datasets.synthetic import ring_cameras
from seeanythingfar.utils.box_ops import (aligned_iou_3d, box_corners, decode_boxes, draw_gaussian,
                                          encode_boxes)
from seeanythingfar.utils.geometry import (bev_to_metric, metric_to_bev, project_points_to_images,
                                           scatter_mean_to_bev, visible_in_image)

PC = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def test_projection_hits_principal_point():
    H, W = 128, 224
    l2i = torch.from_numpy(ring_cameras(6, (H, W)))[None]          # cam 0 looks along +x
    pts = torch.tensor([[[10.0, 0.0, 1.5], [-10.0, 0.0, 1.5]]])     # in front of / behind cam 0
    uv, depth = project_points_to_images(pts, l2i)
    assert torch.allclose(uv[0, 0, 0], torch.tensor([W / 2, H / 2]), atol=1e-3)
    assert depth[0, 0, 0] > 0 and depth[0, 0, 1] < 0
    vis = visible_in_image(uv, depth, (H, W))
    assert vis[0, 0, 0] and not vis[0, 0, 1]
    assert vis[0, 3, 1]                                            # cam 3 looks along -x


def test_point_above_projects_up():
    H, W = 128, 224
    l2i = torch.from_numpy(ring_cameras(6, (H, W)))[None]
    uv, _ = project_points_to_images(torch.tensor([[[10.0, 0.0, 3.0]]]), l2i)
    assert uv[0, 0, 0, 1] < H / 2       # higher in the world = smaller row
    uv, _ = project_points_to_images(torch.tensor([[[10.0, 1.0, 1.5]]]), l2i)
    assert uv[0, 0, 0, 0] < W / 2       # +y is to the left for a camera looking along +x


def test_bev_roundtrip_and_scatter_mean():
    xy = torch.tensor([[0.05, 0.05], [0.15, 0.1], [10.0, -3.0], [100.0, 0.0]])
    cells = metric_to_bev(xy, PC, [0.2, 0.2])
    assert torch.allclose(bev_to_metric(cells, PC, [0.2, 0.2]), xy, atol=1e-5)
    feats = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    bev = scatter_mean_to_bev(feats, xy, torch.zeros(4, dtype=torch.long), 1, PC, [0.2, 0.2], (512, 512))
    assert bev[0, 0, 256, 256] == 2.0                    # mean of the two points in that cell
    assert bev.sum() == 2.0 + 5.0                        # the out-of-range point is dropped


def test_box_code_roundtrip():
    boxes = torch.tensor([[3.0, -4.0, 0.5, 4.2, 1.9, 1.6, 0.7, 1.0, -2.0],
                          [-20.0, 10.0, -1.0, 0.6, 0.6, 1.8, -2.9, 0.0, 0.0]])
    out = decode_boxes(encode_boxes(boxes, PC, [0.4, 0.4]), PC, [0.4, 0.4])
    assert torch.allclose(out, boxes, atol=1e-5)


def test_aligned_iou_and_corners():
    b = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(aligned_iou_3d(b, b), torch.ones(1, 1))
    shifted = b.clone()
    shifted[0, 0] = 2.0
    assert torch.allclose(aligned_iou_3d(b, shifted), torch.tensor([[1 / 3]]), atol=1e-6)
    rot = b.clone()
    rot[0, 6] = math.pi / 2
    c = box_corners(rot)[0]
    assert torch.allclose(c[:, 0].abs().max(), torch.tensor(1.0)) and torch.allclose(c[:, 1].abs().max(), torch.tensor(2.0))


def test_draw_gaussian_peak_and_border():
    hm = torch.zeros(10, 10)
    draw_gaussian(hm, (0, 9), 3)
    assert hm[9, 0] == 1.0 and hm.max() == 1.0 and (hm > 0).sum() < 49
