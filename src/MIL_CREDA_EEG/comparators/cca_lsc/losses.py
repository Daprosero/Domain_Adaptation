"""Paper-based CCA-LSC bag objectives; see REPLICATION.md for explicit choices."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["7.1", "7.2", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cca_calibration_uses_raw_confidence", "cca_prior_uses_confident_bags_only",
                   "cca_weighted_centroids_and_memory", "cca_ratios_have_closed_form",
                   "cca_adversary_reverses_only_feature_gradient", "cca_degenerate_terms_are_finite"],
}

EPS = 1e-8


def label_shift(source_prior: torch.Tensor, target_prior: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 5: (C,), (C,) -> target/source; all source classes required."""
    if source_prior.shape != target_prior.shape or (source_prior <= 0).any():
        raise ValueError("matching priors and positive source mass required")
    return target_prior / source_prior


def calibration_weights(shift: torch.Tensor, hm: float = 1.5) -> torch.Tensor:
    """Paper Eq. 6: (C,) -> (C,); bounded, not normalized."""
    if hm <= 0 or not torch.isfinite(torch.tensor(hm)) or (shift < 0).any():
        raise ValueError("hm must be finite positive and shift nonnegative")
    return 1 / (hm + torch.exp(-torch.sqrt(shift)))


@torch.no_grad()
def calibrated_pseudo(probability: torch.Tensor, weights: torch.Tensor) -> tuple:
    """Paper Eqs. 7,8: (B,C),(C,) -> labels (B,), RAW confidence (B,)."""
    labels = (probability * weights).argmax(1)
    return labels, probability.gather(1, labels[:, None]).squeeze(1)


@torch.no_grad()
def estimate_prior(probability: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Sec. 3.4/4.6: counts of argmax labels with raw max probability > threshold.

    No confident bag: uniform fallback, an unspecified numerical policy.
    """
    if probability.ndim != 2 or not len(probability) or not 0 <= threshold <= 1:
        raise ValueError("nonempty (bags,classes) probabilities and threshold in [0,1] required")
    confidence, labels = probability.max(1)
    counts = torch.bincount(labels[confidence > threshold], minlength=probability.shape[1])
    counts = counts.to(probability)
    return counts / counts.sum() if counts.sum() > 0 else torch.full_like(counts, 1 / len(counts))


def weighted_centroids(features: torch.Tensor, labels: torch.Tensor,
                       confidence: torch.Tensor, classes: int) -> tuple:
    """Sec. 3.2: (B,d),(B,),(B,) -> (C,d), presence (C,).

    Confidence-normalized mean; absent classes are zero and masked out.
    """
    membership = F.one_hot(labels, classes).to(features) * confidence.detach()[:, None]
    mass = membership.sum(0)
    return membership.T @ features / mass.clamp_min(EPS)[:, None], mass > 0


class CentroidMemory(nn.Module):
    """Sec. 3.2 moving centroids: present classes EMA, missing classes retained.

    First observation initializes the class. Current centroids retain gradients;
    persistent history is detached after each step (no cross-step graph).
    """
    def __init__(self, classes: int, width: int, decay: float = 0.9):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError("centroid decay must be in [0,1)")
        self.decay = decay
        self.register_buffer("centroids", torch.zeros(classes, width, dtype=torch.float64))
        self.register_buffer("seen", torch.zeros(classes, dtype=torch.bool))

    def forward(self, features, labels, confidence):
        current, present = weighted_centroids(features, labels, confidence, len(self.seen))
        mixed = torch.where(self.seen.clone()[:, None],
                            self.decay * self.centroids.clone() + (1 - self.decay) * current, current)
        value = torch.where(present[:, None], mixed, self.centroids.clone())
        active = self.seen | present
        with torch.no_grad():
            self.centroids.copy_(value.detach())
            self.seen.copy_(active)
        return value, active.clone()


def centroid_alignment(source, target, active_source, active_target):
    """Paper Eq. 3 (Eq. 2's attraction numerator): ratio of Euclidean sums."""
    distance = torch.cdist(source, target)
    valid = active_source[:, None] & active_target[None, :]
    same = torch.eye(len(source), dtype=torch.bool, device=source.device)
    return distance[valid & same].sum() / distance[valid & ~same].sum().clamp_min(EPS)


def discriminative_alignment(source, target, labels_source, labels_target,
                              confidence_source, confidence_target):
    """Paper Eq. 4: confidence-weighted Euclidean pair means, same/different."""
    distance = torch.cdist(source, target)
    weight = (confidence_source.detach()[:, None] * confidence_target.detach()[None, :]).sqrt()
    same = labels_source[:, None] == labels_target[None, :]
    weighted = distance * weight
    attraction = weighted[same].sum() / same.sum().clamp_min(1)
    repulsion = weighted[~same].sum() / (~same).sum().clamp_min(1)
    # No cross-class pairs: undefined paper ratio, choose a graph-connected zero.
    return torch.where(same.any() & (~same).any(), attraction / repulsion.clamp_min(EPS),
                       distance.sum() * 0)


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


def domain_adversarial(discriminator, source, target):
    """Paper Eq. 1 saddle objective: BCE for D, negative BCE gradient for target G.

    Source G is frozen by plan 7.3; D still trains on both domains. Equal
    domain means implement the two expectations (not a pooled-size mean).
    """
    s = discriminator(source.detach()).reshape(-1)
    t = discriminator(_Reverse.apply(target)).reshape(-1)
    return (F.binary_cross_entropy_with_logits(s, torch.zeros_like(s))
            + F.binary_cross_entropy_with_logits(t, torch.ones_like(t)))
