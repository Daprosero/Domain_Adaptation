"""Xu et al. §3.2.4 Eq. (3), source-anchored cross-domain bag contrast."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cluda_equation_three_factor_domain_mean_and_direction",
                   "cluda_absent_positive_and_single_class_are_finite_zero_gradient"],
}

POSITIVE_EPS = 1e-6


def cross_domain_contrastive(source, target, labels, pseudo, temperature):
    """Eq. (3): (S,D),(T,D),(S,),(T,) -> (scalar, absent-positive count).

    Each source row is one singleton subject-domain; mean over domains, keeping
    the paper's 1/2 factor. Cosine similarity follows official models.py:280.
    Parent-approved finite extension: floor relative positive probability at
    1e-6; absent-class anchors contribute a constant and zero gradient.
    Stable logsumexp avoids exp overflow at the protocol's temperature .01.
    """
    if (source.ndim != 2 or target.ndim != 2 or not len(source) or not len(target)
            or source.shape[1] != target.shape[1] or not source.shape[1]):
        raise ValueError("contrast requires nonempty matrices with the same feature width")
    if labels.shape != (len(source),) or pseudo.shape != (len(target),):
        raise ValueError("one label per source bag and pseudo-label per target bag required")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("non-finite contrastive embeddings")
    logits = F.normalize(source, dim=1) @ F.normalize(target, dim=1).T / temperature
    mask = labels[:, None] == pseudo[None, :]
    present = mask.any(dim=1)
    # Avoid logsumexp(all -inf), whose backward is NaN even behind torch.where.
    safe_mask = mask | ~present[:, None]
    log_mass = (logits.masked_fill(~safe_mask, -torch.inf).logsumexp(dim=1)
                - logits.logsumexp(dim=1))
    log_mass = torch.where(present, log_mass, torch.full_like(log_mass, math.log(POSITIVE_EPS)))
    log_mass = log_mass.clamp(min=math.log(POSITIVE_EPS), max=0.)
    return -.5*log_mass.mean(), int((~present).sum().item())
