"""Loss assembly adapters shared by independent CDS and combined E3 training."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .cds_loss import ConfusionDirectedSqueezeLoss
    from .d1_cosine import CosineSoftmax
else:
    from cds_loss import ConfusionDirectedSqueezeLoss
    from d1_cosine import CosineSoftmax


class D1Branch(nn.Module):
    """Replace only E0's final affine classifier with cosine-softmax logits."""
    def __init__(self, e0_final_linear: nn.Linear):
        super().__init__()
        self.classifier = CosineSoftmax()
        self.classifier.initialize_from_e0(e0_final_linear)

    def forward(self, dropped_embedding: torch.Tensor, labels: torch.Tensor):
        logits = self.classifier(dropped_embedding)
        return logits, F.cross_entropy(logits, labels)

    def optimizer_parameters(self, e0_model: nn.Module):
        """Freeze the unused E0 affine output layer and return only active D1 params."""
        for parameter in e0_model.fc[3].parameters():
            parameter.requires_grad_(False)
        active = list(e0_model.fc[0].parameters()) + list(self.classifier.parameters())
        if any(not parameter.requires_grad for parameter in active):
            raise RuntimeError("D1 optimizer parameter list contains a frozen parameter")
        return active


class CDSBranch(nn.Module):
    """Classification loss plus CDS; defaults to the original CE objective."""
    def __init__(self, theta: float = 0.30, rho0: float = 0.10,
                 classification_loss: nn.Module | None = None):
        super().__init__()
        self.loss = ConfusionDirectedSqueezeLoss(theta=theta, rho0=rho0)
        self.classification_loss = (classification_loss if classification_loss is not None
                                    else nn.CrossEntropyLoss())

    def forward(self, embedding: torch.Tensor, logits: torch.Tensor,
                labels: torch.Tensor, lambda_cds: float | None):
        if lambda_cds is None or not torch.isfinite(torch.tensor(float(lambda_cds))) or lambda_cds < 0:
            raise ValueError("A reviewed finite lambda_cds is required before combining losses")
        ce = F.cross_entropy(logits, labels)
        classification = self.classification_loss(logits, labels)
        cds = self.loss(embedding, labels, logits)
        return {'logits': logits, 'ce': ce, 'classification': classification, 'cds': cds,
                'total': classification + float(lambda_cds) * cds}

    @torch.no_grad()
    def update_centers_after_batch(self, embedding: torch.Tensor,
                                   labels: torch.Tensor) -> torch.Tensor:
        """Call after the current batch loss/optimizer update, not before it."""
        return self.loss.update_centers(embedding.detach(), labels)
