import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange
import timm

from seeanythingfar.models.base import BaseBackbone2D

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINO_FPN(BaseBackbone2D):
    """
    DINOv3 (or DINOv2) ViT backbone with a Simple Feature Pyramid Network (ViTDet style).
    Converts 1D patch tokens into 2D multi-scale spatial grids.

    Images are expected in [0, 1]; ImageNet normalisation happens here so the dataset does
    not need to know which backbone is used.
    """
    def __init__(
        self,
        model_name: str = 'vit_small_patch16_dinov3.lvd1689m',
        out_channels: int = 256,
        freeze_dino: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        try:
            # DINOv2 in timm needs dynamic_img_size for non-square inputs; DINOv3 (RoPE) is
            # resolution-agnostic already and may not accept the kwarg.
            self.dino = timm.create_model(model_name, pretrained=pretrained, dynamic_img_size=True)
        except TypeError:
            self.dino = timm.create_model(model_name, pretrained=pretrained)

        patch = self.dino.patch_embed.patch_size
        self.patch_size = patch[0] if isinstance(patch, (tuple, list)) else patch
        self.embed_dim = self.dino.embed_dim
        # cls token + register tokens precede the patch tokens
        self.num_prefix_tokens = getattr(self.dino, 'num_prefix_tokens', 1)
        self.freeze_dino = freeze_dino

        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()  # Ensure dropout is locked

        self.register_buffer('pixel_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        # SimpleFPN over the single 1/P token grid (P = patch size).
        # scale 1: upsample 1/P -> 1/(P/2), high-res for small distant objects
        self.fpn_up = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # scale 2: keep the base 1/P resolution, project to out_channels
        self.fpn_mid = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # scale 3: downsample 1/P -> 1/(2P), low-res global context
        self.fpn_down = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        p = self.patch_size
        self.strides = {f'stride_{p // 2}': p // 2, f'stride_{p}': p, f'stride_{2 * p}': 2 * p}

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_dino:
            self.dino.eval()  # a frozen backbone stays in eval mode
        return self

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, 3, H, W) in [0, 1].
                    H and W must be divisible by the patch size.
        Returns:
            Dict of multi-scale feature maps keyed 'stride_{s}'.
        """
        B, C, H, W = images.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"image size {(H, W)} must be divisible by patch size {self.patch_size}"

        h_grid = H // self.patch_size
        w_grid = W // self.patch_size

        images = (images - self.pixel_mean) / self.pixel_std
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_dino):
            tokens = self.dino.forward_features(images)  # (B, prefix + h*w, D)
        patch_tokens = tokens[:, self.num_prefix_tokens:]

        # spatial reshape (Token to Grid)
        spatial_grid = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=h_grid, w=w_grid)

        names = list(self.strides)
        return {
            names[0]: self.fpn_up(spatial_grid),
            names[1]: self.fpn_mid(spatial_grid),
            names[2]: self.fpn_down(spatial_grid),
        }
