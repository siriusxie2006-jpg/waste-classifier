"""Small cross-stage fusion variant; inspired by existing multi-scale/gating work."""
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleHead(nn.Module):
    def __init__(self, classifier, adaptive):
        super().__init__()
        self.classifier = classifier
        self.project_detail = nn.Linear(256, 512)
        self.gate = nn.Sequential(nn.Linear(768, 32), nn.ReLU(), nn.Linear(32, 1)) if adaptive else None
        if self.gate is not None:
            # Start at the same 50/50 mixture as the fixed-fusion ablation.
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward_with_gate(self, detail, semantic):
        aligned = F.relu(self.project_detail(detail))
        if self.gate is None:
            weight = semantic.new_full((semantic.shape[0], 1), .5)
        else:
            weight = self.gate(torch.cat([detail, semantic], dim=1)).sigmoid()
        fused = weight*aligned + (1-weight)*semantic
        return self.classifier(fused), weight

    def forward(self, detail, semantic):
        return self.forward_with_gate(detail, semantic)[0]


class MultiScaleResNet18(nn.Module):
    """Keep the backbone names so existing freezing and strict loading still apply."""
    gradcam_target = 'layer3'  # Both direct-detail and semantic paths depend on layer3.

    def __init__(self, backbone, adaptive=True):
        super().__init__()
        for name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            setattr(self, name, getattr(backbone, name))
        self.fc = MultiScaleHead(backbone.fc, adaptive)

    def forward_features(self, image):
        value = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        value = self.layer2(self.layer1(value))
        detail = self.layer3(value)
        semantic = self.layer4(detail)
        return self.avgpool(detail).flatten(1), self.avgpool(semantic).flatten(1)

    def forward_with_gate(self, image):
        detail, semantic = self.forward_features(image)
        return self.fc.forward_with_gate(detail, semantic)

    def forward(self, image):
        detail, semantic = self.forward_features(image)
        return self.fc(detail, semantic)
