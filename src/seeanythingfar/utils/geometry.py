"""Geometric bridge: LiDAR-frame <-> image-plane and LiDAR-frame <-> BEV-grid conversions.

Conventions used across the code base:
    * Points live in the LiDAR frame: x forward, y left, z up (nuScenes LIDAR_TOP).
    * ``lidar2img`` is a (4, 4) matrix ``K_4x4 @ lidar2cam`` that maps homogeneous
      LiDAR points to ``(u*d, v*d, d, 1)``, where ``d`` is depth along the camera axis.
    * The BEV grid is indexed ``[y_idx, x_idx]`` (rows follow y, columns follow x), and
      BEV positions are expressed in *cell units* (continuous ``x_idx``, ``y_idx``).
"""
from typing import Sequence, Tuple

import torch


def project_points_to_images(
    points: torch.Tensor, lidar2img: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project LiDAR-frame points into every camera.

    Args:
        points: (B, Q, 3) points in the LiDAR frame.
        lidar2img: (B, N_cam, 4, 4) projection matrices.
    Returns:
        uv: (B, N_cam, Q, 2) pixel coordinates (u = column, v = row).
        depth: (B, N_cam, Q) depth along the optical axis (<= 0 means behind the camera).
    """
    homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (B, Q, 4)
    cam = torch.einsum("bnij,bqj->bnqi", lidar2img, homo)                 # (B, N, Q, 4)
    depth = cam[..., 2]
    uv = cam[..., :2] / depth.clamp(min=eps).unsqueeze(-1)
    return uv, depth


def visible_in_image(
    uv: torch.Tensor, depth: torch.Tensor, image_hw: Sequence[int], min_depth: float = 0.1
) -> torch.Tensor:
    """Boolean mask (same leading shape as ``depth``) of projections that land inside the image."""
    h, w = image_hw
    return (
        (depth > min_depth)
        & (uv[..., 0] >= 0) & (uv[..., 0] < w)
        & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    )


def pixel_to_grid_sample(uv: torch.Tensor, image_hw: Sequence[int]) -> torch.Tensor:
    """Pixel coordinates -> the [-1, 1] coordinates ``F.grid_sample`` expects (align_corners=False)."""
    h, w = image_hw
    x = uv[..., 0] / w * 2.0 - 1.0
    y = uv[..., 1] / h * 2.0 - 1.0
    return torch.stack([x, y], dim=-1)


def metric_to_bev(xy: torch.Tensor, pc_range: Sequence[float], cell_size: Sequence[float]) -> torch.Tensor:
    """Metric (x, y) -> continuous BEV cell coordinates (x_idx, y_idx).

    ``pc_range`` is ``[x_min, y_min, z_min, x_max, y_max, z_max]``; ``cell_size`` is the
    metric size of one cell of the grid in question (voxel size x feature-map stride).
    """
    out = torch.empty_like(xy)
    out[..., 0] = (xy[..., 0] - pc_range[0]) / cell_size[0]
    out[..., 1] = (xy[..., 1] - pc_range[1]) / cell_size[1]
    return out


def bev_to_metric(cells: torch.Tensor, pc_range: Sequence[float], cell_size: Sequence[float]) -> torch.Tensor:
    """Inverse of :func:`metric_to_bev`."""
    out = torch.empty_like(cells)
    out[..., 0] = cells[..., 0] * cell_size[0] + pc_range[0]
    out[..., 1] = cells[..., 1] * cell_size[1] + pc_range[1]
    return out


def bev_grid_centers(h: int, w: int, device=None) -> torch.Tensor:
    """(H*W, 2) cell-centre coordinates (x_idx + 0.5, y_idx + 0.5), row-major over [y, x]."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2) + 0.5


def scatter_mean_to_bev(
    feats: torch.Tensor,
    xy: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    pc_range: Sequence[float],
    cell_size: Sequence[float],
    grid_hw: Tuple[int, int],
) -> torch.Tensor:
    """Average point features into a dense BEV grid.

    Several points usually fall into one cell, so a plain indexed assignment would keep an
    arbitrary one; this averages them instead.

    Args:
        feats: (N, C) point features.
        xy: (N, 2) metric x, y of each point.
        batch_idx: (N,) long tensor with the sample index of each point.
    Returns:
        (B, C, H, W) BEV feature map.
    """
    h, w = grid_hw
    c = feats.shape[1]
    x_idx = ((xy[:, 0] - pc_range[0]) / cell_size[0]).floor().long()
    y_idx = ((xy[:, 1] - pc_range[1]) / cell_size[1]).floor().long()
    keep = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)

    flat = batch_idx[keep] * (h * w) + y_idx[keep] * w + x_idx[keep]
    summed = feats.new_zeros(batch_size * h * w, c).index_add_(0, flat, feats[keep])
    count = feats.new_zeros(batch_size * h * w).index_add_(0, flat, feats.new_ones(flat.shape[0]))
    bev = summed / count.clamp(min=1).unsqueeze(1)
    return bev.view(batch_size, h, w, c).permute(0, 3, 1, 2).contiguous()


def rotation_z(angle: float) -> torch.Tensor:
    """3x3 rotation about +z (counter-clockwise, radians)."""
    a = torch.as_tensor(angle, dtype=torch.float64)
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
