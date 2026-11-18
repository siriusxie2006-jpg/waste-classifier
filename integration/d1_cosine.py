"""D1 cosine-softmax head for the E0 256-D post-dropout embedding (fc[2] output)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CosineSoftmax(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 6,
                 scale: float = 31.0, angular_margin: float = 0.0):
        super().__init__()
        if feature_dim != 256 or num_classes != 6:
            raise ValueError("D1 is fixed to 256 features and six classes")
        if scale <= 0 or angular_margin != 0.0:
            raise ValueError("This D1 integration requires scale>0 and margin=0")
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.scale = float(scale)
        self.angular_margin = float(angular_margin)

    @torch.no_grad()
    def initialize_from_e0(self, linear: nn.Linear) -> None:
        if linear.weight.shape != self.weight.shape:
            raise ValueError("E0 final Linear weight must have shape [6,256]")
        self.weight.copy_(linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 256:
            raise ValueError("Expected [N,256] feature tensor")
        x = F.normalize(features, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x, w, bias=None)
