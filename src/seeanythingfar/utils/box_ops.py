"""3D box helpers.

Box layout used everywhere: ``(x, y, z, l, w, h, yaw, vx, vy)`` in the LiDAR frame,
with ``(x, y, z)`` the geometric centre, ``l`` along the heading, ``yaw`` measured
from +x towards +y, and ``(vx, vy)`` in m/s.

The regression code predicted by the head (``CODE_SIZE = 10``):
``(cx_cell, cy_cell, z, log l, log w, log h, sin yaw, cos yaw, vx, vy)``
where ``cx_cell, cy_cell`` are continuous BEV cell coordinates of the head's grid.
"""
import math
from typing import Sequence

import torch

CODE_SIZE = 10


def encode_boxes(boxes: torch.Tensor, pc_range: Sequence[float], cell_size: Sequence[float]) -> torch.Tensor:
    """(M, 9) boxes -> (M, 10) regression targets."""
    code = boxes.new_zeros(boxes.shape[0], CODE_SIZE)
    code[:, 0] = (boxes[:, 0] - pc_range[0]) / cell_size[0]
    code[:, 1] = (boxes[:, 1] - pc_range[1]) / cell_size[1]
    code[:, 2] = boxes[:, 2]
    code[:, 3:6] = boxes[:, 3:6].clamp(min=1e-3).log()
    code[:, 6] = torch.sin(boxes[:, 6])
    code[:, 7] = torch.cos(boxes[:, 6])
    code[:, 8:10] = boxes[:, 7:9]
    return code


def decode_boxes(code: torch.Tensor, pc_range: Sequence[float], cell_size: Sequence[float]) -> torch.Tensor:
    """(..., 10) regression code -> (..., 9) metric boxes."""
    x = code[..., 0] * cell_size[0] + pc_range[0]
    y = code[..., 1] * cell_size[1] + pc_range[1]
    dims = code[..., 3:6].exp()
    yaw = torch.atan2(code[..., 6], code[..., 7])
    return torch.cat([x[..., None], y[..., None], code[..., 2:3], dims, yaw[..., None], code[..., 8:10]], dim=-1)


def box_corners(boxes: torch.Tensor) -> torch.Tensor:
    """(M, >=7) boxes -> (M, 8, 3) corners. Bottom face first, counter-clockwise."""
    signs = boxes.new_tensor([
        [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
        [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
    ]) * 0.5
    local = signs[None] * boxes[:, None, 3:6]
    c, s = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
    rot = torch.stack([
        torch.stack([c, -s, torch.zeros_like(c)], -1),
        torch.stack([s, c, torch.zeros_like(c)], -1),
        torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], -1),
    ], dim=1)  # (M, 3, 3)
    return torch.einsum("mij,mkj->mki", rot, local) + boxes[:, None, :3]


def aligned_iou_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise 3D IoU of the *axis-aligned envelopes* of rotated boxes, (Ma, Mb).

    An approximation of rotated IoU that needs no compiled ops; used only as a matching
    cost and a training diagnostic, never as an evaluation metric.
    """
    def envelope(boxes):
        c, s = torch.cos(boxes[:, 6]).abs(), torch.sin(boxes[:, 6]).abs()
        ex = boxes[:, 3] * c + boxes[:, 4] * s
        ey = boxes[:, 3] * s + boxes[:, 4] * c
        half = torch.stack([ex, ey, boxes[:, 5]], dim=1) * 0.5
        return boxes[:, :3] - half, boxes[:, :3] + half

    if a.shape[0] == 0 or b.shape[0] == 0:
        return a.new_zeros(a.shape[0], b.shape[0])
    amin, amax = envelope(a)
    bmin, bmax = envelope(b)
    inter = (torch.min(amax[:, None], bmax[None]) - torch.max(amin[:, None], bmin[None])).clamp(min=0).prod(-1)
    vol_a = (amax - amin).prod(-1)
    vol_b = (bmax - bmin).prod(-1)
    return inter / (vol_a[:, None] + vol_b[None] - inter).clamp(min=1e-6)


def gaussian_radius(height: float, width: float, min_overlap: float = 0.1) -> float:
    """CenterNet Gaussian radius (in cells) for a box of the given size (in cells)."""
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: torch.Tensor, center_xy: Sequence[int], radius: int) -> None:
    """In-place max-blend of a 2D Gaussian into ``heatmap`` (H, W) at integer ``(x, y)``."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    r = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=heatmap.dtype)
    gauss = torch.exp(-(r[None, :] ** 2 + r[:, None] ** 2) / (2 * sigma ** 2))

    x, y = int(center_xy[0]), int(center_xy[1])
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    region = heatmap[y - top:y + bottom, x - left:x + right]
    patch = gauss[radius - top:radius + bottom, radius - left:radius + right]
    torch.max(region, patch, out=region)
