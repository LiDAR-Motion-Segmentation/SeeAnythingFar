from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class BaseBackbone2D(nn.Module, ABC):
    """
    Contract for any 2D visual feature extractor.
    Must accept batched images and return multi-scale dense feature maps.
    """
    @abstractmethod
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, 3, H, W)
        Returns:
            Dict mapping scale names (e.g., 'p2', 'p3') to feature tensors.
        """
        pass

class BaseBackbone3D(nn.Module, ABC):
    """
    Contract for any 3D point cloud feature extractor.
    Must accept raw points and return both point-level tokens (for segmentation)
    and a dense BEV map (for detection queries).
    """
    @abstractmethod
    def forward(self, points: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: List of length B, where each element is a point cloud (N, C)
        Returns:
            Tuple containing:
                - point_tokens: Tensor of shape (Total_N, D) for segmentation
                - bev_features: Tensor of shape (B, C_bev, H_bev, W_bev)
        """
        pass

class BaseFusionHead(nn.Module, ABC):
    """
    Contract for the multi-modal 3D detection head (e.g., TransFusion).
    """
    @abstractmethod
    def forward(
        self, 
        bev_features: torch.Tensor, 
        image_features: Dict[str, torch.Tensor],
        calibrations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_features: 3D queries/features (B, C_bev, H_bev, W_bev)
            image_features: Output from BaseBackbone2D
            calibrations: Intrinsics and extrinsics for 3D-to-2D projection
        Returns:
            Dict containing predicted bounding boxes, classes, and scores.
        """
        pass