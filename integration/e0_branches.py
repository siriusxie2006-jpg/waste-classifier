"""Read-only E0 adapter exposing the 256-D feature for D1/CDS smoke tests."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class E0Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True),
                                      nn.Dropout(0.4), nn.Linear(256, 6))

    def load_e0_state(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state, strict=True)
        for name, p in self.model.named_parameters():
            p.requires_grad_(name.startswith("fc."))

    def set_mode(self, training: bool) -> None:
        self.model.train(training)
        for name, child in self.model.named_children():
            child.train(training if name == "fc" else False)

    def forward_with_embedding(self, images: torch.Tensor):
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(images))))
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        pooled = torch.flatten(m.avgpool(x), 1)
        linear = m.fc[0](pooled)
        embedding = F.relu(linear, inplace=False)
        dropped = m.fc[2](embedding)
        logits = m.fc[3](dropped)
        return embedding, dropped, logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_with_embedding(images)[2]
