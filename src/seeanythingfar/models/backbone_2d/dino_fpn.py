import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange
import timm

from src.seeanythingfar.models.base import BaseBackbone2D

class DINO_FPN(BaseBackbone2D):
    """
    DINOv3 Backbone with a Simple Feature Pyramid Network.
    Converts 1D patch tokens into 2D multi-scale spatial grids.
    """
    def __init__(
        self,
        model_size: str = 'vits14',
        out_channel: int = 256,
        freeze_dino: bool = True
    ):
        super().__init__()
        
        self.dino = timm.create_model('vit_small_patch14_dinov3.lvd1689m', pretrained=True)
        self.patch_size = 14
        
        # Dimensions mapping based on model size
        embed_dims = {'vits14': 384, 'vitb14': 768, 'vitl14': 1024}
        self.embed_dim = embed_dims[model_size]
        
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval() # Ensure dropout/batchnorm are locked
            
        # Simple Feature Pyramid Network (ViTDet style)
        # TransFusion usually expects features at different resolutions.
        # DINO gives us a 1/14 resolution grid. We will project it to 256 channels,
        # and create multi-scale maps (e.g., standardizing to something like 1/8 and 1/16)
        
        # scale 1: upsample from 1/14 to ~1/8 using Transposed Conv
        self.fpn_up = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, out_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        
        # scale 2: Keep roughly the same resolution (1/14 projected to 256 dims)
        self.fpn_mid = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        
        # scale 3: Downsample from 1/14 to ~1/32 using strided Conv
        self.fpn_down = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, 3, H, W). 
                    Note: H and W should ideally be divisible by 14.
        Returns:
            Dict of multi-scale feature maps.
        """
        
        B, C, H, W = images.shape
        
        # calculate spatial dimensions of the grid
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        
        # Extract features from DINOv2
        # We use a context manager to enforce no-gradients if frozen
        with torch.set_grad_enabled(next(self.parameters()).required_grad):
            features = self.dino.forward_features(images)
            
            patch_tokens = features['x_norm_patchtokens'] # Shape: (B, N, D)
            
        # spatial reshape (Token to Grid)
        spatial_grid = rearrange(
            patch_tokens,
            'b (h w) d -> b d h w',
            h=h_grid,
            w=w_grid 
        )
        
        # building the feature pyramid
        multi_scale_features = {
            'stride_8': self.fpn_up(spatial_grid),    # High-res for small distant objects
            'stride_16': self.fpn_mid(spatial_grid),  # Base DINO resolution
            'stride_32': self.fpn_down(spatial_grid)  # Low-res for global context
        }
        
        return multi_scale_features