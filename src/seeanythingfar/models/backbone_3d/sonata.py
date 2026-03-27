import torch
import torch.nn as nn
from typing import List, Tuple

from src.seeanythingfar.models.base import BaseBackbone3D

class SonataEncoder(BaseBackbone3D):
    """
    Wrapper for the Sonata (Pre-trained PTv3) model
    Handles sparse batching and BEV projection
    """
    def __init__(
      self,
      in_channels: int = 4,                                     # X, Y, Z, Reflectance
      embed_dim: int = 256,                                     # output dimension of PTv3 tokens
      grid_extents: List[float] = [-51.2, -51.2, 51.2, 51.2],   # [min_x, min_y, max_x, max_y]
      voxel_size: List[float] = [0.1, 0.1],                     # [size_x, size_y] in meters
      pretrained: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.grid_extents = grid_extents
        self.voxel_size = voxel_size
        
        # Calculate the spatial dimensions of the BEV grid
        self.bev_width = int((grid_extents[2] - grid_extents[0]) / voxel_size[0])
        self.bev_height = int((grid_extents[3] - grid_extents[1]) / voxel_size[1])
        
        # return high resolution point features from PTv3
        self.ptv3_core = self._build_ptv3_core(in_channels, embed_dim, pretrained)
        
        # a small 2D convolution to smooth the BEV features after scattering
        self.bev_neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )