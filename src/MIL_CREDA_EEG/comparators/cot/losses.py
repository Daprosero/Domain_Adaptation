"""COT CVPR 2023 §4.1 Eqs. 7, 8, 11, 12; finite unsmoothed dual."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["7.1", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "cot_equations_have_closed_form_values_and_gradients",
        "cot_dual_matches_uniform_transport_and_is_gauge_invariant",
        "cot_dual_ascent_detaches_cost_and_descent_detaches_potentials",
        "cot_dual_window_is_first_ten_percent_with_ceiling",
    ],
}


def squared_cost(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """§4.1: (N,d),(K,d)->(N,K), squared L2, no normalization."""
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError("cost needs two matrices with the same feature dimension")
    if not x.shape[0] or not centers.shape[0]:
        raise ValueError("cost needs nonempty point sets")
    return (x[:, None, :] - centers[None, :, :]).square().sum(-1)


def source_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. 7: mean cross-entropy of source BAG logits and source BAG labels."""
    return F.cross_entropy(logits, labels.long())


def cluster_loss(target: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Eq. 8: sum of squared distances to nearest learnable target center."""
    return squared_cost(target, centers).min(dim=1).values.sum()


def ot_dual(cost: torch.Tensor, potentials: torch.Tensor) -> torch.Tensor:
    """Eq. 11: (classes,K),(K,)->scalar; uniform target/source cluster masses.

    Callers detach cost for dual ascent, potentials for representation descent.
    This is not Sinkhorn, and does not implement Eq. 9's inconsistent marginals.
    """
    if cost.ndim != 2 or potentials.shape != (cost.shape[1],) or not cost.numel():
        raise ValueError("dual needs nonempty (classes,K) cost and (K,) potentials")
    return potentials.mean() + (cost-potentials[None, :]).min(dim=1).values.mean()


def cot_objective(ce: torch.Tensor, cluster: torch.Tensor, ot: torch.Tensor,
                  alpha_cluster: float, alpha_ot: float) -> torch.Tensor:
    """Eq. 12; in the controlled variant CE is read-only, not target supervision."""
    return ce + alpha_cluster*cluster + alpha_ot*ot


def maximize_dual(cost: torch.Tensor, potentials: torch.nn.Parameter, *,
                  lr: float, steps: int) -> None:
    """Algorithm 1 line 7: only lambda changes; finite-budget SGD subgradient ascent.

    Mean-centering chooses a gauge without changing Eq. 11. No convergence claim.
    """
    if not math.isfinite(lr) or lr <= 0 or steps < 1 or int(steps) != steps:
        raise ValueError("dual ascent needs positive finite lr and integer steps")
    optimizer = torch.optim.SGD([potentials], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        (-ot_dual(cost.detach(), potentials)).backward()
        optimizer.step()
        with torch.no_grad():
            potentials.sub_(potentials.mean())
    optimizer.zero_grad(set_to_none=True)


def dual_active(batch_index: int, batches: int) -> bool:
    """Algorithm 1 following prose: first 10% iterations per epoch, rounded up."""
    if batches < 1 or not 0 <= batch_index < batches:
        raise ValueError("batch index must lie in a nonempty epoch")
    return batch_index < math.ceil(.1*batches)
