"""CLUST source-free auxiliary objectives; equations refer to Nature HTML.

Ambiguous source-free combined prototypes and numeric guards are documented in
REPLICATION.md. This is the plan §7.4 controlled variant, not published CLUST.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["7.1", "7.2", "7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["clust_probabilities_favor_nearest_prototype",
                   "clust_centroids_use_confident_bags_and_preserve_missing_classes",
                   "clust_transport_has_closed_form_and_finite_gradients",
                   "clust_consistency_is_symmetric_and_zero_at_equality",
                   "clust_information_maximization_has_known_limits",
                   "clust_concentration_and_alpha_are_finite_at_degeneracy",
                   "clust_ema_preserves_missing_classes"],
}
EPS = 1e-6


def cosine_distance(objects, prototypes):
    """Eq.2/6 distance interpretation: [B,d],[K,d] -> [B,K]."""
    return 1 - F.normalize(objects, dim=-1) @ F.normalize(prototypes, dim=-1).T


def prototype_probabilities(objects, prototypes):
    """Eq.2: nearest-prototype probabilities, with cosine distance."""
    return (-cosine_distance(objects, prototypes)).softmax(-1)


def confidence_assignments(probabilities, threshold):
    """Eq.3/4: per-class confidence mask, never target ground truth."""
    if not math.isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise ValueError("confidence must be finite in [0.5,1]")
    return probabilities.detach() >= threshold


def target_centroids(features, membership, fallback):
    """Eq.4 target-only mean. Empty classes retain their previous EMA row."""
    mask = membership.to(features.dtype)
    counts = mask.sum(0)
    means = mask.T @ features / counts.clamp_min(1).unsqueeze(1)
    return torch.where(counts[:, None] > 0, means, fallback.detach()), counts


def ema_update(old, current, active, momentum):
    """Eq.5, with no update for unsupported classes; returns detached state."""
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("EMA momentum must be finite in [0,1)")
    return torch.where(active[:, None], momentum * old + (1 - momentum) * current,
                       old).detach()


def concentration(features, prototypes, membership):
    """Eq.24: [B,d],[K,d],[B,K] -> tau[K], detached concentration."""
    mask = membership.to(features.dtype)
    n = mask.sum(0)
    delta = torch.exp(-n) / math.sqrt(2 * math.pi)
    squared = (features[:, None, :] - prototypes[None, :, :]).square().sum(-1)
    tau = (mask * squared).sum(0) / (n * (n + delta).log()).clamp_min(EPS)
    return torch.where(n > 0, tau.clamp_min(EPS), torch.ones_like(n)).detach()


def conditional_probability(objects, prototypes, tau):
    """Eq.6: uniform class prior cancels; softmax across K, not B."""
    return (-cosine_distance(objects, prototypes) / tau.clamp_min(EPS)).softmax(-1)


def transport_cost(features, prototypes, membership, tau):
    """Eq.7/8/11: class-set squared cost, expectation over bags."""
    squared = (features[:, None, :] - prototypes[None, :, :]).square().sum(-1)
    return (conditional_probability(features, prototypes, tau) * squared * membership).sum() / features.shape[0]


def clustering_terms(features, source, target, membership):
    """Eq.11: two feature transports and two same-class prototype transports.

    Eq.4's combined source-free centroid is unspecified: use equal prototype
    weights, not source observations or stored source counts.
    """
    tau_s = concentration(features, source, membership)
    tau_t = concentration(features, target, membership)
    combined = (source + target) / 2
    tau = concentration(features, combined, membership)
    feature = transport_cost(features, source, membership, tau_s)
    feature = feature + transport_cost(features, target, membership, tau_t)
    active = membership.any(0)
    per_class = features.new_zeros(source.shape[0])
    for objects in (source, target):
        probability = conditional_probability(objects, combined, tau).diagonal()
        per_class = per_class + probability * (objects - combined).square().sum(-1)
    per_class = per_class * active
    return feature, per_class.mean(), per_class


def clustering_loss(feature, semantic, alpha):
    """Eq.12."""
    return (1 - alpha) * feature + alpha * semantic


def symmetric_kl(p, q):
    """Eq.13 recovered from HTML: half the two KL directions, mean over bags."""
    log_ratio = p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log()
    return (0.5 * ((p - q) * log_ratio).sum(-1)).mean()


def information_maximization(p):
    """Eq.14 with gamma=1: H(per bag) - H(mean bag prediction)."""
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
    mean = p.mean(0)
    return entropy + (mean * mean.clamp_min(1e-12).log()).sum()


def objective(feature, semantic, consistency, information, alpha, weights):
    """Eq.15 source-free: no supervised source summand; plan §7.7 weights."""
    return (weights[0] * clustering_loss(feature, semantic, alpha)
            + weights[1] * consistency + weights[2] * information)


def a_distance(loss):
    """Eq.21/22 plus explicit nonnegative guard (not stated by paper)."""
    return (2 * (1 - 2 * loss.detach())).clamp_min(0)


def dynamic_alpha(feature, per_class_semantic):
    """Eq.23; zero denominator has no estimate and falls back to initial .5."""
    a, b = a_distance(feature), a_distance(per_class_semantic).mean()
    total = a + b
    return float(torch.where(total > EPS, b / total.clamp_min(EPS), total.new_tensor(.5)))
