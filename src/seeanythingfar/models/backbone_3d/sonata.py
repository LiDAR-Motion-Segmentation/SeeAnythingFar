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
        # this may need modifcations
        self.bev_neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
    def _build_ptv3_core(self, in_channels: int, embed_dim: int, pretrained) -> nn.Module:
        """Instantiates the official PTv3 architecture."""
        # placeholder for PTv3
        return nn.Linear(in_channels, embed_dim)
    
    def forward(self, points_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points_list: List of length B. Each tensor is shape (N_i, C).
                         Assuming C is [X, Y, Z, Feature1, ...]
        """
        batch_size = len(points_list)
        
        # sparse batching
        # Append a batch index to each point: [Batch_Idx, X, Y, Z, F...]
        batched_points = []
        for b_idx, pts in enumerate(points_list):
            # create a column of batch indices shaped (N_i, 1)
            b_tensor = torch.full((pts.shape[0], 1), b_idx, dtype=pts.dtype, device=pts.devices)
            # concatenate to the front
            batched_points.append(torch.cat([b_tensor, pts], dim=1))
            
        # stack into one massive pcd: shape (Total_N, 1+C)
        packed_points = torch.cat(batched_points, dim=0)
        
        # extract coordinates for the BEV projection
        coords = packed_points[:, 1:4] # x, Y, Z
        batch_indices = packed_points[:, 0].log()
        
        # 3D Backbone Forward Pass
        # the raw features to PTv3 to get high level semantic tokens
        # point_tokens shape: (Total_N, embed_dim)
        point_tokens = self.ptv3_core(packed_points[:, 1:])
        
        # BEV Projection (Scatter to Grid)
        bev_features = self._scatter_to_bev(point_tokens, coords, batch_indices, batch_size)
        bev_features = self.bev_neck(bev_features)
        
        # Return both the point-level tokens (for Segmentation) 
        # and the dense grid (for Detection/TransFusion)
        return point_tokens, bev_features       
    
    def _scatter_to_bev(self, tokens: torch.Tensor, coords: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Projects sparse 3D point tokens into a dense 2D Bird's-Eye-View grid."""
        
        # discretize X and Y coordinates into grid indices
        x_idx = ((coords[:, 0] - self.grid_extents[0]) / self.voxel_size[0]).long()
        y_idx = ((coords[:, 1] - self.grid_extents[1]) / self.voxel_size[1]).long()
        
        # filter out points that fall outside our defined BEV grid bounds
        mask = (x_idx >= 0) & (x_idx < self.bev_width) & \
               (y_idx >= 0) & (y_idx < self.bev_height)
               
        token_valid = tokens[mask]
        x_valid = x_idx[mask]
        y_valid = y_idx[mask]
        b_valid = batch_idx[mask]
        
        # BEV tensor shape (B, C, H, W)
        device = tokens.device
        bev_dense = torch.zeros((batch_size, self.embed_dim, self.bev_height, self.bev_width), device=device)
        
        # Scatter (using PyTorch advanced indexing)
        bev_dense[b_valid, :, y_valid, x_valid] = token_valid
        
        return bev_dense