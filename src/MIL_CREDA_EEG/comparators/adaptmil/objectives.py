"""AdaptMIL Eqs.1–4 and mean-pool MK-MMD, with explicit estimator choices."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["adaptmil_mmd_matches_closed_form_and_symmetries",
                   "adaptmil_equations_one_to_four_match_their_terms",
                   "adaptmil_alignment_never_uses_attention_or_target_labels",
                   "adaptmil_bandwidth_is_training_only_and_degenerate_safe",
                   "adaptmil_mmd_refuses_invalid_inputs"],
}

# Unspecified by the paper: a mean mixture of three Gaussian bandwidths.
KERNEL_SCALES = (0.5, 1.0, 2.0)


def _positive(value, name):
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def multi_kernel_mmd(source: torch.Tensor, target: torch.Tensor, sigma: float):
    """Appendix A MK-MMD; chosen biased V-statistic, unequal batches allowed.

    Inputs (Ns,d), (Nt,d); returns scalar. No normalization of embeddings.
    The paper does not specify the estimator or kernel mixture; not claimed
    numerically equivalent to TLlib's estimator.
    """
    _positive(sigma, "sigma")
    if (source.ndim != 2 or target.ndim != 2 or source.shape[0] == 0
            or target.shape[0] == 0 or source.shape[1] != target.shape[1]
            or source.shape[1] == 0):
        raise ValueError("MMD requires nonempty (bags, shared features) matrices")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("MMD inputs must be finite")
    # Direct differences avoid cancellation of large, nearly equal norms.
    def kernel(a, b):
        distance = (a[:, None] - b[None, :]).square().sum(dim=-1)
        return torch.stack([torch.exp(-distance / (2 * (sigma * s) ** 2))
                            for s in KERNEL_SCALES]).mean(dim=0)
    return kernel(source, source).mean() + kernel(target, target).mean() - 2 * kernel(
        source, target).mean()


@torch.no_grad()
def median_bandwidth(source: torch.Tensor, target: torch.Tensor, multiplier: float):
    """Plan §7.7: multiplier times joint TRAINING bag median, fixed once.

    Scope/cadence and collapse fallback 1 are unspecified implementation choices.
    """
    _positive(multiplier, "kernelMultiplier")
    vectors = torch.cat((source, target))
    if not torch.isfinite(vectors).all():
        raise ValueError("bandwidth inputs must be finite")
    distances = torch.pdist(vectors)
    positive = distances[distances > 0]
    median = float(positive.median()) if positive.numel() else 1.0
    return multiplier * median


def uda_objective(logits, labels, source, target, weight: float, sigma: float):
    """Paper Eq.1: mean task CE plus weighted domain alignment."""
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("MMD weight must be finite and nonnegative")
    return F.cross_entropy(logits, labels) + weight * multi_kernel_mmd(source, target, sigma)


def single_stage_objective(model, source_bags, labels, target_bags, weight, sigma):
    """Paper Eq.2 audit/reference path: DA after learned MIL aggregation."""
    source = model.representations(source_bags, "source")
    target = model.representations(target_bags, "target")
    return uda_objective(model.head.linear(source), labels, source, target, weight, sigma)


def stage_one_objective(model, source_bags, labels, target_bags, weight, sigma):
    """Paper Eq.3, HTML L_stage1: mean(P(z)) for CE and alignment."""
    source = model.representations(source_bags, "source", mean=True)
    target = model.representations(target_bags, "target", mean=True)
    return uda_objective(model.head.linear(source), labels, source, target, weight, sigma)


def stage_two_objective(model, source_bags, labels):
    """Paper Eq.4: mean source-bag CE after attention(P(z))."""
    return F.cross_entropy(model.logits(source_bags, "source"), labels)


def alignment_objective(model, source_bags, target_bags, weight, sigma):
    """Plan §7.3 controlled adaptation: Eq.3's DA summand only.

    No labels accepted; attention and head are never called. Source features
    detached because the standardized source side is frozen.
    """
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("MMD weight must be finite and nonnegative")
    with torch.no_grad():
        source = model.representations(source_bags, "source", mean=True)
    target = model.representations(target_bags, "target", mean=True)
    mmd = multi_kernel_mmd(source, target, sigma)
    return {"objective": weight * mmd, "mmd": mmd}
