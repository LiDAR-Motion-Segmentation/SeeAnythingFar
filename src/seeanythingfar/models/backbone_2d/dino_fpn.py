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