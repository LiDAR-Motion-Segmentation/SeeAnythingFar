from typing import Optional

import torch
import torch.nn as nn

from seeanythingfar.models.base import BaseSegHead


class LinearDecoder(BaseSegHead):
    """Linear point-wise segmentation decoder over the 3D backbone's point tokens.

    The architecture diagram marks this decoder frozen, i.e. a pretrained linear probe.
    Freezing only makes sense with ``ckpt_path`` pointing at such weights, so ``freeze``
    without a checkpoint is rejected rather than silently training nothing.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 16, freeze: bool = False,
                 ckpt_path: Optional[str] = None):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            self.linear.load_state_dict(state.get("state_dict", state))
        if freeze:
            if ckpt_path is None:
                raise ValueError("freeze=True needs ckpt_path with pretrained linear-probe weights")
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, point_tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(point_tokens)
