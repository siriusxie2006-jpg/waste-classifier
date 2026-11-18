"""CDS integration using the corrected stop-gradient ledger.

Confuser b: nearest competing normalized EMA prototype (the corrected
definition's explicit rule). Weight w: detached full-class softmax probability
of b from the ordinary classifier logits. Only the normalized embedding z in
tau is differentiable. EMA centers are raw class means; prototype normalization
is applied only while forming geometry.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConfusionDirectedSqueezeLoss(nn.Module):
    def __init__(self, num_classes: int = 6, feature_dim: int = 256,
                 theta: float = 0.30, rho0: float = 0.10, eps: float = 1e-6):
        super().__init__()
        if num_classes != 6 or feature_dim != 256:
            raise ValueError("CDS integration is fixed to six classes and 256-D")
        if not 0.0 < theta < 0.5 or not 0.0 < rho0 <= 1.0:
            raise ValueError("Invalid theta or EMA update rate")
        self.num_classes, self.feature_dim = num_classes, feature_dim
        self.theta, self.rho0, self.eps = float(theta), float(rho0), float(eps)
        self.register_buffer("centers", torch.zeros(num_classes, feature_dim))
        self.register_buffer("seen", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("class_update_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("total_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_update_norm", torch.zeros(()))

    @torch.no_grad()
    def initialize_from_train_features(self, features: torch.Tensor,
                                       labels: torch.Tensor) -> None:
        self._validate(features, labels)
        z = F.normalize(features.detach(), dim=1, eps=self.eps)
        for k in range(self.num_classes):
            rows = z[labels == k]
            if rows.shape[0] == 0:
                raise ValueError(f"Missing Train class {k} during center initialization")
            self.centers[k].copy_(rows.mean(0))
            self.seen[k] = True
        self.class_update_counts.zero_()
        self.total_updates.zero_()
        self.last_update_norm.zero_()

    @torch.no_grad()
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Count-adjusted EMA; no optimizer participation; absent classes skipped."""
        self._validate(features, labels)
        z = F.normalize(features.detach(), dim=1, eps=self.eps)
        before = self.centers.clone()
        for k in labels.unique().tolist():
            batch = z[labels == k]
            mean = batch.mean(0)
            n_k = int(batch.shape[0])
            rho_k = 1.0 - (1.0 - self.rho0) ** n_k
            if not bool(self.seen[k]):
                self.centers[k].copy_(mean)
                self.seen[k] = True
            else:
                self.centers[k].mul_(1.0 - rho_k).add_(mean, alpha=rho_k)
            self.class_update_counts[k] += 1
        self.total_updates += 1
        delta = torch.linalg.vector_norm(self.centers - before)
        self.last_update_norm.copy_(delta)
        if not torch.isfinite(self.centers).all() or not torch.isfinite(delta):
            raise FloatingPointError("CDS EMA center update became non-finite")
        return delta

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                logits: torch.Tensor) -> torch.Tensor:
        self._validate(features, labels)
        if logits.shape != (features.shape[0], self.num_classes):
            raise ValueError("logits must have shape [N,6]")
        if not bool(self.seen.all()):
            raise RuntimeError("Initialize CDS centers from Train features first")
        z = F.normalize(features, dim=1, eps=self.eps)
        # All center-derived quantities and discrete selections are targets.
        mu = self.centers.detach().to(device=features.device, dtype=features.dtype)
        mu_hat = F.normalize(mu, dim=1, eps=self.eps).detach()
        scores = z.detach() @ mu_hat.T
        scores = scores.masked_fill(F.one_hot(labels.long(), self.num_classes).bool(), -torch.inf)
        b = scores.argmax(dim=1).detach()
        a = labels.long().detach()
        mu_a, mu_b = mu_hat[a].detach(), mu_hat[b].detach()
        vector = (mu_a - mu_b).detach()
        raw_d = torch.linalg.vector_norm(vector, dim=1)
        if not torch.isfinite(raw_d).all():
            raise FloatingPointError("CDS prototype distance is non-finite")
        degenerate = raw_d < 0.05
        d = raw_d.clamp_min(self.eps).detach()
        u = (vector / d[:, None]).detach()
        # Confidence weights are explicitly stopped, including normalization Z.
        probs = logits.detach().softmax(dim=1)
        w = probs.gather(1, b[:, None]).squeeze(1).detach()
        # A nearly collapsed prototype pair makes tau's 1/d^2 scale unsafe.
        # Exclude only affected anchors; all other anchors retain their gradients.
        w = w.masked_fill(degenerate, 0.0).detach()
        zsum = w.sum().clamp_min(self.eps).detach()
        # Only z remains live in the geometric score; centers/u/d/b/w are constants.
        tau = ((mu_a - z) * u).sum(dim=1) / d
        tau = tau.masked_fill(degenerate, 0.0)
        loss = (w * F.relu(tau - self.theta).square()).sum() / zsum
        if (not torch.isfinite(tau).all() or not torch.isfinite(w).all()
                or not torch.isfinite(loss)):
            raise FloatingPointError("CDS loss intermediates or result are non-finite")
        return loss

    def _validate(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [N,256]")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0] or features.shape[0] == 0:
            raise ValueError("labels must match non-empty feature batch")
        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN/Inf")
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("labels outside [0,6)")
