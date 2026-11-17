"""Paper-based UDA-DDA objectives over bag representations; see REPLICATION.md.

Tang et al., Neurocomputing 658 (2025) 131715: marginal MMD (Eqs. 2-5),
conditional CMMD (Eq. 6), the dynamic confidence threshold (Eq. 7), the
source cross-entropy (Eq. 8) and the MMD weight schedule from tau_h to tau_l
(Sec. 3.4). The CMMD weight beta of Eq. 9, a gate on the source loss, is not
implemented: plan v60 Sec. 7.7 removes it. Every epoch boundary of the paper is
read on its own 100-epoch scale and rescaled to the phase-2 budget (plan 7.4).
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md",
    "sections": ["7.1", "7.2", "7.4", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["uda_mmd_is_eq5_and_the_eq4_trace_form", "uda_cmmd_is_eq6_per_class",
                   "uda_kernel_is_gaussian_with_median_bandwidth",
                   "uda_confidence_schedule_is_eq7_rescaled", "uda_alpha_decays_from_tau_h_to_tau_l",
                   "uda_degenerate_terms_are_finite"],
}

#: Sec. 3.4 tau_h > tau_l; Sec. 4.2 values {1, 0.01}, ordering resolved in REPLICATION.md.
TAU_HIGH, TAU_LOW = 1.0, 0.01
#: Sec. 3.3 (Eq. 7 prose) stage boundaries on the paper's 100-epoch run: [0,10), [10,40), [40,85], >85.
PAPER_EPOCHS = 100
STAGE_BOUNDS = (10, 40, 85)
EPS = 1e-12


def _matrix(x: torch.Tensor, name: str) -> None:
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be a finite (bags, features) matrix")


def gaussian_kernel(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    """Sec. 3.2: k(u, v) = exp(-||u - v||^2 / sigma); (n,d),(m,d) -> (n,m)."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("kernel sigma must be finite and positive")
    # Direct differences avoid the cancellation of the expanded-norm form.
    return torch.exp(-(a[:, None] - b[None, :]).square().sum(-1) / sigma)


def mmd2(source: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    """Eq. 5 (equivalently tr(K M) with Eq. 4's M): biased V-statistic, scalar."""
    _matrix(source, "source")
    _matrix(target, "target")
    if not len(source) or not len(target) or source.shape[1] != target.shape[1]:
        raise ValueError("MMD needs nonempty source and target with equal width")
    return (gaussian_kernel(source, source, sigma).mean()
            + gaussian_kernel(target, target, sigma).mean()
            - 2 * gaussian_kernel(source, target, sigma).mean())


def mmd_matrix(n_source: int, n_target: int, dtype=torch.float64) -> torch.Tensor:
    """Eq. 4 as read in the PDF text layer: 1/n_s^2, 1/n_t^2, -1/(n_s n_t)."""
    n = n_source + n_target
    m = torch.full((n, n), -1.0 / (n_source * n_target), dtype=dtype)
    m[:n_source, :n_source] = 1.0 / n_source ** 2
    m[n_source:, n_source:] = 1.0 / n_target ** 2
    return m


def cmmd2(source: torch.Tensor, target: torch.Tensor, source_labels: torch.Tensor,
          target_labels: torch.Tensor, classes: int, sigma: float) -> torch.Tensor:
    """Eq. 6: (1/C) sum_c MMD^2 between class-c source and class-c pseudo-labelled target.

    A class with no source or no selected target bag contributes zero (Eq. 6 is
    undefined there; unspecified choice). An empty selection returns a zero that
    stays connected to the target graph, so backward still runs.
    """
    _matrix(source, "source")
    if target.ndim != 2 or source_labels.shape != (len(source),) or target_labels.shape != (len(target),):
        raise ValueError("CMMD needs (n,d) features with one label per row")
    total = target.sum() * 0 + source.sum() * 0
    for c in range(classes):
        s, t = source[source_labels == c], target[target_labels == c]
        if len(s) and len(t):
            total = total + mmd2(s, t, sigma)
    return total / classes


@torch.no_grad()
def median_bandwidth(source: torch.Tensor, target: torch.Tensor, multiplier: float) -> float:
    """Plan 7.7 generic rule: multiplier x median pairwise SQUARED distance (pooled).

    Squared because the paper's kernel divides ||u-v||^2 by sigma. No positive
    distance (collapsed representations): fallback 1.0, an unspecified choice.
    """
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("kernelMultiplier must be finite and positive")
    pooled = torch.cat((source, target))
    _matrix(pooled, "bandwidth input")
    distances = torch.pdist(pooled).square()
    positive = distances[distances > 0]
    return multiplier * (float(positive.median()) if positive.numel() else 1.0)


def paper_epoch(epoch: int, epochs: int) -> float:
    """Plan 7.4 rescaling: phase-2 epoch e of E sits at paper epoch 100 e / E."""
    if epochs < 1 or not 0 <= epoch < epochs:
        raise ValueError("epoch must lie in [0, epochs)")
    return PAPER_EPOCHS * epoch / epochs


def stage(epoch: int, epochs: int) -> int:
    """Eq. 7 prose: 0 for [0,10), 1 for [10,40), 2 for [40,85], 3 beyond (paper epochs)."""
    p = paper_epoch(epoch, epochs)
    low, mid, high = STAGE_BOUNDS
    return 0 if p < low else 1 if p < mid else 2 if p <= high else 3


def confidence_threshold(epoch: int, epochs: int, middle: float, late: float) -> float:
    """Eq. 7 schedule tau(e): 0, middle (paper 0.5), late (paper 0.75), 1."""
    return (0.0, float(middle), float(late), 1.0)[stage(epoch, epochs)]


@torch.no_grad()
def confident_pseudo_labels(probability: torch.Tensor, threshold: float) -> tuple:
    """Eq. 7: argmax pseudo-labels and the mask max-probability > tau (strict, as the code)."""
    if probability.ndim != 2 or not 0 <= threshold <= 1:
        raise ValueError("(bags, classes) probabilities and a threshold in [0,1] required")
    confidence, labels = probability.max(1)
    return labels, confidence > threshold


def mmd_weight_schedule(epoch: int, epochs: int, high: float = TAU_HIGH,
                        low: float = TAU_LOW) -> float:
    """Sec. 3.4 alpha schedule: tau_h in stage 0, geometric decay to tau_l across
    stage 1 (paper epochs 10-40), tau_l from paper epoch 40 on. Shape unspecified."""
    if not 0 < low <= high:
        raise ValueError("0 < tau_l <= tau_h required")
    p, (start, end, _) = paper_epoch(epoch, epochs), STAGE_BOUNDS
    if p < start:
        return high
    if p >= end:
        return low
    return high * (low / high) ** ((p - start) / (end - start))


def source_cross_entropy(probability: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. 8 over source bags: mean -log p(y); the head returns probabilities."""
    return F.nll_loss(probability.clamp_min(EPS).log(), labels)
