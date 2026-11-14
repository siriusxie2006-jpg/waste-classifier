"""A same-stage residual head with almost exactly the fixed-fusion parameter count."""
from torch import nn


class ResidualCapacityHead(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        # 1025*h + 512 parameters; h=128 is nearest to the fusion's 131584 extras.
        self.residual = nn.Sequential(nn.Linear(512, 128), nn.ReLU(),
                                      nn.Linear(128, 512), nn.ReLU())

    def forward(self, semantic):
        return self.classifier(.5*self.residual(semantic)+.5*semantic)
