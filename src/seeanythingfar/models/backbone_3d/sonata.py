import torch
import torch.nn as nn
from typing import List, Optional, Sequence, Tuple

from seeanythingfar.models.base import BaseBackbone3D
from seeanythingfar.utils.geometry import scatter_mean_to_bev


class SonataEncoder(BaseBackbone3D):
    """
    Wrapper for the Sonata (self-supervised Point Transformer V3) encoder.

    Pipeline: raw points -> per-sample grid sampling -> Sonata -> unpool back to every
    input point (point tokens, used for segmentation) -> scatter-mean into a BEV grid ->
    small conv neck (BEV feature map, used for detection queries).

    ``backend='mlp'`` replaces Sonata with a per-point MLP. It exists only so the rest of
    the stack can be developed and tested without the ``sonata`` package (spconv,
    torch_scatter, optionally flash-attn); do not train real models with it.
    """
    def __init__(
        self,
        in_channels: int = 4,                       # X, Y, Z, intensity
        embed_dim: int = 256,                       # dimension of point tokens and BEV channels
        pc_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_voxel_size: Sequence[float] = (0.2, 0.2),  # metric size of a scatter cell
        bev_stride: int = 2,                        # neck downsampling; head cell = voxel * stride
        num_neck_blocks: int = 2,
        backend: str = 'sonata',                    # 'sonata' | 'mlp'
        pretrained: bool = True,
        freeze: bool = True,
        grid_size: float = 0.05,                    # Sonata input voxelisation (metres)
        intensity_scale: float = 255.0,             # raw intensity range, mapped to [-1, 1]
        num_concat_levels: int = 2,                 # unpooling levels whose features are concatenated
        sonata_feat_dim: int = 1088,                # width after unpooling; 512+384+192 for the
                                                    # released checkpoint with 2 concat levels
        sonata_name: str = 'sonata',
        sonata_repo_id: str = 'facebook/sonata',
        download_root: Optional[str] = None,        # keep checkpoints off a full home filesystem
        enable_flash: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.pc_range = list(pc_range)
        self.bev_voxel_size = list(bev_voxel_size)
        self.bev_stride = bev_stride
        self.backend = backend
        self.freeze = freeze and backend == 'sonata'
        self.grid_size = grid_size
        self.intensity_scale = intensity_scale
        self.num_concat_levels = num_concat_levels

        # Spatial dimensions of the scatter grid and of the output BEV map
        self.scatter_w = round((self.pc_range[3] - self.pc_range[0]) / self.bev_voxel_size[0])
        self.scatter_h = round((self.pc_range[4] - self.pc_range[1]) / self.bev_voxel_size[1])
        assert self.scatter_w % bev_stride == 0 and self.scatter_h % bev_stride == 0
        self.bev_hw = (self.scatter_h // bev_stride, self.scatter_w // bev_stride)
        # metric size of one cell of the returned BEV map; the head needs this
        self.bev_cell_size = [v * bev_stride for v in self.bev_voxel_size]

        if backend == 'sonata':
            self.ptv3_core = self._build_sonata(sonata_name, sonata_repo_id, pretrained,
                                                download_root, enable_flash)
            if self.freeze:
                for p in self.ptv3_core.parameters():
                    p.requires_grad = False
                self.ptv3_core.eval()
            # (a LazyLinear would break optimizer construction, which happens before any forward)
            self.sonata_feat_dim = sonata_feat_dim
            self.proj = nn.Sequential(nn.Linear(sonata_feat_dim, embed_dim), nn.LayerNorm(embed_dim))
        elif backend == 'mlp':
            self.ptv3_core = None
            self.proj = nn.Sequential(
                nn.Linear(in_channels, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
                nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
            )
        else:
            raise ValueError(f"unknown backend {backend!r}")

        # Conv neck to smooth the sparse scattered BEV and bring it to the head stride
        layers = []
        for i in range(num_neck_blocks):
            stride = bev_stride if i == 0 else 1
            layers += [
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            ]
        if not layers and bev_stride > 1:
            layers.append(nn.AvgPool2d(bev_stride))
        self.bev_neck = nn.Sequential(*layers)

    @staticmethod
    def _build_sonata(name, repo_id, pretrained, download_root, enable_flash) -> nn.Module:
        """Instantiates the official Sonata/PTv3 encoder (https://github.com/facebookresearch/sonata)."""
        try:
            import sonata
        except ImportError as e:
            raise ImportError(
                "backend='sonata' needs the `sonata` package "
                "(pip install git+https://github.com/facebookresearch/sonata, plus spconv and "
                "torch_scatter). Use backend='mlp' for dependency-free smoke tests."
            ) from e
        if not pretrained:
            raise NotImplementedError("Sonata is only wired up as a pretrained encoder")
        custom_config = dict(enable_flash=enable_flash)
        if not enable_flash:
            # without flash attention the released checkpoint needs smaller patches to fit memory
            custom_config["enc_patch_size"] = [1024] * 5
        kwargs = dict(repo_id=repo_id, custom_config=custom_config)
        if download_root is not None:
            kwargs["download_root"] = download_root
        return sonata.load(name, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.ptv3_core.eval()
        return self

    def forward(self, points_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points_list: List of length B. Each tensor is shape (N_i, C) with
                         C = [X, Y, Z, intensity, ...]; extra channels are ignored.
        Returns:
            point_tokens: (Total_N, embed_dim), in the concatenated order of points_list
            bev_features: (B, embed_dim, H_bev, W_bev)
        """
        batch_size = len(points_list)
        device = points_list[0].device

        points = torch.cat([p[:, :self.in_channels] for p in points_list], dim=0)
        batch_idx = torch.cat([
            torch.full((p.shape[0],), b, dtype=torch.long, device=device)
            for b, p in enumerate(points_list)
        ])

        if self.backend == 'sonata':
            point_tokens = self.proj(self._sonata_point_features(points, batch_idx, batch_size))
        else:
            point_tokens = self.proj(points)

        # BEV projection: only points inside the vertical range contribute
        z = points[:, 2]
        in_z = (z >= self.pc_range[2]) & (z < self.pc_range[5])
        bev = scatter_mean_to_bev(
            point_tokens[in_z], points[in_z, :2], batch_idx[in_z], batch_size,
            self.pc_range, self.bev_voxel_size, (self.scatter_h, self.scatter_w),
        )
        bev_features = self.bev_neck(bev)

        # Return both the point-level tokens (for Segmentation)
        # and the dense grid (for Detection/TransFusion)
        return point_tokens, bev_features

    def _sonata_point_features(self, points: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Grid-sample, run Sonata, and return raw (unprojected) features for every input point."""
        coord = points[:, :3]

        # per-sample CenterShift(apply_z=True), as in Sonata's default transform
        shifted = torch.empty_like(coord)
        grid_min = torch.empty_like(coord)
        for b in range(batch_size):
            m = batch_idx == b
            c = coord[m]
            lo, hi = c.min(0).values, c.max(0).values
            shift = torch.stack([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            shifted[m] = c - shift
            grid_min[m] = shifted[m].min(0).values

        grid = torch.floor((shifted - grid_min) / self.grid_size).long()  # (N, 3), >= 0
        g = 1 << 16
        key = ((batch_idx * g + grid[:, 0]) * g + grid[:, 1]) * g + grid[:, 2]
        _, inverse = torch.unique(key, return_inverse=True)  # sorted, so voxels are grouped by batch
        num_vox = int(inverse.max()) + 1

        def voxel_mean(x):
            s = x.new_zeros(num_vox, x.shape[1]).index_add_(0, inverse, x)
            n = x.new_zeros(num_vox).index_add_(0, inverse, x.new_ones(x.shape[0]))
            return s / n.unsqueeze(1)

        # Sonata feature layout: coord(3) + color(3) + normal(3). LiDAR has neither colour nor
        # normals, so intensity stands in for grey colour and normals are zero.
        intensity = points[:, 3:4] / self.intensity_scale * 2.0 - 1.0
        feat = torch.cat([shifted, intensity.expand(-1, 3), torch.zeros_like(shifted)], dim=1)

        vox_coord = voxel_mean(shifted)
        vox_feat = voxel_mean(feat)
        vox_grid = torch.zeros(num_vox, 3, dtype=torch.long, device=points.device)
        vox_grid[inverse] = grid
        vox_batch = torch.zeros(num_vox, dtype=torch.long, device=points.device)
        vox_batch[inverse] = batch_idx
        offset = torch.bincount(vox_batch, minlength=batch_size).cumsum(0)

        data = dict(coord=vox_coord, grid_coord=vox_grid, feat=vox_feat, offset=offset)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            point = self.ptv3_core(data)
            # unpool back to the input resolution, concatenating multi-scale features
            for _ in range(self.num_concat_levels):
                if "pooling_parent" not in point.keys():
                    break
                parent = point.pop("pooling_parent")
                inv = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
                point = parent
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inv = point.pop("pooling_inverse")
                parent.feat = point.feat[inv]
                point = parent
            vox_out = point.feat

        if vox_out.shape[1] != self.sonata_feat_dim:
            raise ValueError(
                f"Sonata produced {vox_out.shape[1]}-d features but sonata_feat_dim="
                f"{self.sonata_feat_dim}; set model.backbone_3d.sonata_feat_dim={vox_out.shape[1]}")
        return vox_out[inverse]
