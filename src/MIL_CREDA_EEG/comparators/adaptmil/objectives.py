"""AdaptMIL Eqs.1–4, mean-pool MK-MMD and the phase-2 LMMD, with explicit estimator choices."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["7.1", "7.3", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["adaptmil_mmd_matches_closed_form_and_symmetries",
                   "adaptmil_equations_one_to_four_match_their_terms",
                   "adaptmil_lmmd_is_zero_when_class_conditionals_coincide",
                   "adaptmil_lmmd_class_weights_are_normalized_per_class",
                   "adaptmil_lmmd_weights_come_from_target_soft_predictions",
                   "adaptmil_lmmd_gradient_flows_only_into_the_target_encoder",
                   "adaptmil_lmmd_refuses_invalid_inputs",
                   "adaptmil_bandwidth_is_training_only_and_degenerate_safe",
                   "adaptmil_mmd_refuses_invalid_inputs"],
}

# Unspecified by the paper: a mean mixture of three Gaussian bandwidths.
KERNEL_SCALES = (0.5, 1.0, 2.0)


def _positive(value, name):
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _matrices(source, target):
    if (source.ndim != 2 or target.ndim != 2 or source.shape[0] == 0
            or target.shape[0] == 0 or source.shape[1] != target.shape[1]
            or source.shape[1] == 0):
        raise ValueError("MMD requires nonempty (bags, shared features) matrices")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("MMD inputs must be finite")


def mixture_kernel(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    """Mean of Gaussians exp(-||u-v||^2 / (2 (sigma s)^2)), s in KERNEL_SCALES; (n,m)."""
    _positive(sigma, "sigma")
    # Direct differences avoid cancellation of large, nearly equal norms.
    distance = (a[:, None] - b[None, :]).square().sum(dim=-1)
    return torch.stack([torch.exp(-distance / (2 * (sigma * s) ** 2))
                        for s in KERNEL_SCALES]).mean(dim=0)


def multi_kernel_mmd(source: torch.Tensor, target: torch.Tensor, sigma: float):
    """Appendix A MK-MMD; chosen biased V-statistic, unequal batches allowed.

    Inputs (Ns,d), (Nt,d); returns scalar. No normalization of embeddings.
    The paper does not specify the estimator or kernel mixture; not claimed
    numerically equivalent to TLlib's estimator.
    """
    _positive(sigma, "sigma")
    _matrices(source, target)
    return (mixture_kernel(source, source, sigma).mean()
            + mixture_kernel(target, target, sigma).mean()
            - 2 * mixture_kernel(source, target, sigma).mean())


@torch.no_grad()
def lmmd_class_weights(source_labels: torch.Tensor, target_probability: torch.Tensor,
                       classes: int):
    """LMMD class weights (Zhu et al. 2021 Eq. 4, Transfer Learning Repo `cal_weight`).

    Source bag i weighs one-hot(y_i)_c / sum_i one-hot(y_i)_c; target bag j weighs
    p_jc / sum_j p_jc, p the soft prediction, detached. Only classes present among
    the source labels AND the target argmax are kept, as the repository does; returns
    (Ns,K), (Nt,K) for the K kept classes, each column summing to one.
    """
    if (source_labels.ndim != 1 or target_probability.ndim != 2
            or target_probability.shape[1] != classes
            or ((source_labels < 0) | (source_labels >= classes)).any()):
        raise ValueError("LMMD needs (Ns,) labels in [0, classes) and (Nt, classes) predictions")
    probability = target_probability.detach()
    if (not torch.isfinite(probability).all() or (probability < 0).any()
            or not torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]),
                                  atol=1e-6)):
        raise ValueError("target predictions must be finite rows of the simplex")
    source = F.one_hot(source_labels, classes).to(probability.dtype)
    kept = (source.sum(0) > 0) & (torch.bincount(probability.argmax(1),
                                                 minlength=classes) > 0)
    source, probability = source[:, kept], probability[:, kept]
    return source / source.sum(0), probability / probability.sum(0).clamp_min(1e-12)


def local_mmd(source: torch.Tensor, target: torch.Tensor, source_labels: torch.Tensor,
              target_probability: torch.Tensor, sigma: float, classes: int = 2):
    """Zhu et al. (TNNLS 2021) LMMD, Eq. 5 over bags: mean over kept classes c of
    ||sum_i ws_ic phi(s_i) - sum_j wt_jc phi(t_j)||^2 in the module's mixture kernel.

    Biased V-statistic like `multi_kernel_mmd`; no kept class gives a zero still
    connected to the target graph. The class weights carry no gradient.
    """
    _positive(sigma, "sigma")
    _matrices(source, target)
    if source_labels.shape != (len(source),) or target_probability.shape[0] != len(target):
        raise ValueError("LMMD needs one label per source row and one prediction per target row")
    ws, wt = lmmd_class_weights(source_labels, target_probability, classes)
    if not ws.shape[1]:
        return target.sum() * 0
    ws, wt = ws.to(source.dtype), wt.to(target.dtype)
    terms = ((ws * (mixture_kernel(source, source, sigma) @ ws)).sum()
             + (wt * (mixture_kernel(target, target, sigma) @ wt)).sum()
             - 2 * (ws * (mixture_kernel(source, target, sigma) @ wt)).sum())
    return terms / ws.shape[1]


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


def alignment_objective(model, source_bags, source_labels, target_bags, weight, sigma):
    """Plan §7.3 controlled adaptation: AdaptMIL-LMMD on mean-pooled P(z) bags.

    Source bags weigh by their labels; target bags by the soft prediction of the
    training target encoder through the frozen native attention and head, no
    threshold, no marginal term. Source features and the class weights are
    detached: only the target encoder receives gradient. No target labels exist here.
    """
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("LMMD weight must be finite and nonnegative")
    with torch.no_grad():
        source = model.representations(source_bags, "source", mean=True)
    # One target pass feeds both the alignment (mean) and the weights (attention).
    projected = [model.aggregator.project(h) for h in model.embed(target_bags, "target")]
    target = torch.stack([h.mean(dim=0) for h in projected])
    with torch.no_grad():
        probability = model.head(torch.stack([model.aggregator(h)[0] for h in projected]))
    lmmd = local_mmd(source, target, source_labels, probability, sigma, model.classes)
    return {"objective": weight * lmmd, "lmmd": lmmd}
