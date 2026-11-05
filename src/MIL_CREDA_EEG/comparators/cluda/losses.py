"""Xu et al. official models.py:225–282 contrastive losses, adapted to bags."""
from __future__ import annotations

import math
from itertools import combinations

import torch

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cluda_official_symmetric_loss_includes_target_target_pairs",
                   "cluda_excludes_no_positive_anchors_and_keeps_official_positive_clamp",
                   "cluda_singleton_source_pairs_are_inert_without_floor_saturation",
                   "cluda_domain_averages_match_official_loops"],
}

POSITIVE_EPS = 1e-6
SOURCE_TEMPERATURE = .03  # Official models.py:208, main.py:193; not a search axis.
SOURCE_SOURCE_RATIO = .5  # Official main.py:88–91, beta2 = beta1 / 2.


def _normalized(features):
    if (features.ndim != 2 or len(features) < 2 or not features.shape[1]
            or not torch.isfinite(features).all()):
        raise ValueError("contrast requires a finite matrix with at least two bags")
    norms = features.norm(dim=1, keepdim=True)
    if (norms == 0).any():
        raise ValueError("official cosine contrast is undefined for zero-norm embeddings")
    return features / norms


def supervised_contrastive(features, labels, temperature):
    """Official models.py:257–282: (N,D),(N,) -> scalar, no-positive anchor count.

    Diagonal excluded, symmetric anchors, same-class positive mass SUM (not mean).
    Floor the UNNORMALIZED numerator at 1e-6, not a relative probability floor.
    Logsumexp is algebraically identical, including possible negative losses when
    the clamped numerator exceeds the denominator. L81 follow-up: exclude
    no-positive anchors from the mean. They have no anchor loss but may still
    be candidates for valid anchors. Fail explicitly on zero-norm embeddings.
    """
    normalized = _normalized(features)
    if labels.shape != (len(features),):
        raise ValueError("one label per contrastive bag required")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = normalized @ normalized.T / temperature
    off_diagonal = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = (labels[:, None] == labels[None, :]) & off_diagonal
    present = positive.any(dim=0)
    # Safe fallback avoids all-minus-infinity logsumexp backward for absent classes.
    safe_positive = positive | (~present[None, :] & off_diagonal)
    log_positive = logits.masked_fill(~safe_positive, -torch.inf).logsumexp(dim=0)
    log_positive = torch.where(present, log_positive,
                               torch.full_like(log_positive, math.log(POSITIVE_EPS)))
    log_positive = log_positive.clamp_min(math.log(POSITIVE_EPS))
    log_denominator = logits.masked_fill(~off_diagonal, -torch.inf).logsumexp(dim=0)
    excluded = int((~present).sum().item())
    if not present.any():
        return logits.sum()*0., excluded
    return (log_denominator[present]-log_positive[present]).mean(), excluded


def _domains(source, labels, domains):
    if source.ndim != 2 or len(source) < 1 or labels.shape != (len(source),):
        raise ValueError("nonempty source matrix and one label per source bag required")
    if domains is None:
        return [source[i:i+1] for i in range(len(source))], [labels[i:i+1] for i in range(len(source))]
    if len(domains) != len(source):
        raise ValueError("one subject-domain id per source bag required")
    indices = {}
    for i, domain in enumerate(domains):
        indices.setdefault(domain, []).append(i)
    return [source[idx] for idx in indices.values()], [labels[idx] for idx in indices.values()]


def cross_domain_contrastive(source, target, labels, pseudo, temperature, domains=None):
    """Official models.py:231–241: mean over source domains of SupCon([source_i,target]).

    A source subject is one domain, possibly repeated by the shared source queue.
    Target-target pairs are included separately in every domain's symmetric loss.
    """
    if (target.ndim != 2 or len(target) < 1 or source.ndim != 2
            or source.shape[1] != target.shape[1] or pseudo.shape != (len(target),)):
        raise ValueError("source/target feature widths and per-bag labels must agree")
    groups, classes = _domains(source, labels, domains)
    values, missing = [], 0
    for group, group_labels in zip(groups, classes):
        value, count = supervised_contrastive(torch.cat((group, target)),
                                              torch.cat((group_labels, pseudo)), temperature)
        values.append(value)
        missing += count
    return torch.stack(values).mean(), missing


def source_source_contrastive(source, labels, domains=None, *, diagnostics=None):
    """Official models.py:243–249: mean SupCon over every unordered source-domain pair.

    The frozen source representations must remain connected to the PROJECTOR.
    Fixed T2=.03 is retained independently of the searched ST temperature.
    """
    groups, classes = _domains(source, labels, domains)
    if len(groups) < 2:
        raise ValueError("source-source contrast requires at least two source domains")
    values, missing = [], 0
    zero_loss_pairs = 0
    for i, j in combinations(range(len(groups)), 2):
        value, count = supervised_contrastive(torch.cat((groups[i], groups[j])),
                                              torch.cat((classes[i], classes[j])), SOURCE_TEMPERATURE)
        values.append(value)
        missing += count
        zero_loss_pairs += int(abs(float(value.detach())) < 1e-12)
    if diagnostics is not None:
        diagnostics.update(pairs=len(values), zero_loss_pairs=zero_loss_pairs)
    return torch.stack(values).mean(), missing


def contrastive_objective(source_ce, source_target, source_source, weight):
    """Official main.py:88–91 ratio, L79 constant scale instead of gamma ramp."""
    return source_ce + weight*(source_target + SOURCE_SOURCE_RATIO*source_source)
