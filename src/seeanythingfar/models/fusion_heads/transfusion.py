import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from src.seeanythingfar.models.base import BaseFusionHead

class TransFusionHead(BaseFusionHead):
    """
    Two stage multi model Fusion Head
    Stage 1: CenterPoint BEV Heatmaps -> 3D Queries
    Stage 2: Cross Attention with DINOv3 Image Features
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        num_queries: int = 200,  # Top-K proposals to keep
        image_feature_scale: str = 'stride_8'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.target_scale = image_feature_scale
        
        # stage 1: CenterPoint Proposal Heads (BEV Space)
        # Predicts object probability (X, Y)
        self.heatmap_head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # predicts absolute z height
        self.z_regression = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # predicts box dimensions (L, W, H)
        self.dim_regression = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1) 
        
        # stage 2: Transfusion CrossAttention
        # Standard Multi-Head Attention: 
        # Query = 3D Proposals, Key/Value = 2D Image Features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=8,
            batch_first=True
        )
        
        # MLP to refine the boudning boxes after fusion
        self.final_box_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 7) # dx, dy, dz, L, W, H, yaw
        )
        
    def forward(self,
                bev_features: torch.Tensor,
                image_features: Dict[str, torch.Tensor],
                calibration: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        B = bev_features.size(0)
        
        # BEV Query generation (centerpoint)
        heatmaps = torch.sigmoid(self.heatmap_head(bev_features))  # [B, Classes, H, W]
        z_heights = self.z_regression(bev_features)                # [B, 1, H, W]
        
        # Extract the Top-K hottest pixels to act as our 3D Queries
        # queries shape: [B, num_queries, embed_dim]
        # query_centers shape: [B, num_queries, 3] -> (X, Y, Z)
        queries, query_centers = self._get_topk_proposals(
            bev_features, heatmaps, z_heights, self.num_queries
        )
        
        # target the high-res Dinov3 feature
        target_img_feat = image_features[self.target_scale]
        
        
        
    
    def _get_topk_proposals():
        pass
    
    def _project_and_sample():
        pass
    