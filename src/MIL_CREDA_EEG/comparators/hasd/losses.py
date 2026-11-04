"""HASD §2.2's four loss displays; see REPLICATION.md for undefined conventions."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["7.1", "7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["hasd_gram_sgir_is_unsquared_sum_and_rotation_invariant",
                   "hasd_pacr_is_sum_of_squared_attention_differences",
                   "hasd_sinkhorn_balanced_marginals_and_uniform_known_case",
                   "hasd_partial_transport_mass_and_closed_form",
                   "hasd_das_includes_entropy_and_both_marginal_kl_terms",
                   "hasd_total_is_exact_weighted_sum"],
}

EPSILON = .001  # Ambiguity: paper prose calls this gamma; equation says epsilon.
TAU = .1  # Unspecified; controlled plan requires partial OT, not label-based gating.
MAX_ITERATIONS = 10000
TOLERANCE = 1e-9


@torch.no_grad()
def sinkhorn_plan(cost: torch.Tensor, epsilon: float = EPSILON, tau: float = TAU,
                  *, max_iterations: int = MAX_ITERATIONS,
                  tolerance: float = TOLERANCE) -> torch.Tensor:
    """DAS: log-domain SK with uniform bag marginals and optional KL relaxation.

    tau>0 minimizes <P,C> + eps sum P(log P-1) + tau*(KL(P1,a)+KL(P^T1,b)).
    tau=0 means the paper's *balanced* DAS branch: exact marginal constraints.
    Solve detached; the envelope derivative of the converged objective is P.
    """
    if (not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(tau) or tau < 0
            or max_iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("positive finite epsilon/tolerance/iterations and nonnegative tau required")
    if cost.ndim != 2 or min(cost.shape) < 1 or not torch.isfinite(cost).all():
        raise ValueError("cost must be finite and nonempty")
    cost = cost.detach().to(torch.float64)
    log_a = cost.new_full((cost.shape[0],), -math.log(cost.shape[0]))
    log_b = cost.new_full((cost.shape[1],), -math.log(cost.shape[1]))
    kernel = -cost / epsilon
    log_u, log_v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    rho = tau / (tau + epsilon) if tau else 1.
    for _ in range(max_iterations):
        old_u, old_v = log_u, log_v
        log_u = rho * (log_a - torch.logsumexp(kernel + log_v[None, :], dim=1))
        log_v = rho * (log_b - torch.logsumexp(kernel + log_u[:, None], dim=0))
        if tau:
            # Optimize the dual translation gauge exactly. This leaves P unchanged
            # but removes the nearly-unit contraction when tau >> epsilon.
            shift = .5 * tau / epsilon * (
                torch.logsumexp(log_a - epsilon / tau * log_u, dim=0)
                - torch.logsumexp(log_b - epsilon / tau * log_v, dim=0))
            log_u = log_u + shift
            log_v = log_v - shift
        if max(float((log_u - old_u).abs().max()),
               float((log_v - old_v).abs().max())) < tolerance:
            plan = (kernel + log_u[:, None] + log_v[None, :]).exp()
            if not torch.isfinite(plan).all():
                raise RuntimeError("nonfinite HASD transport plan")
            return plan
    raise RuntimeError(f"HASD Sinkhorn did not converge in {max_iterations} iterations")


def _kl(mass: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # Generalized KL, defined also at mass=0; positive reference by construction.
    return (torch.xlogy(mass, mass) - mass * reference.log() - mass + reference).sum()


def das_loss(current: torch.Tensor, source: torch.Tensor, *, epsilon: float = EPSILON,
             tau: float = TAU) -> dict[str, torch.Tensor]:
    """HASD DAS display, reversed direction / bag-level cost by plan §§7.3–7.4.

    (target bags,d), (source bags,d) -> scalar plus detached plan diagnostics.
    No target labels, hard pseudolabels, classifier updates or hidden CE term.
    """
    if current.ndim != 2 or source.ndim != 2 or current.shape[1] != source.shape[1]:
        raise ValueError("DAS needs two matrices with the same feature width")
    current, source = current.to(torch.float64), source.to(torch.float64)
    cost = 1 - F.normalize(current, dim=1, eps=1e-12) @ F.normalize(source, dim=1, eps=1e-12).T
    plan = sinkhorn_plan(cost, epsilon, tau)
    transport = (plan * cost).sum()
    entropy = (torch.xlogy(plan, plan) - plan).sum()
    partial = _kl(plan.sum(1), plan.new_full((plan.shape[0],), 1 / plan.shape[0]))
    partial = partial + _kl(plan.sum(0), plan.new_full((plan.shape[1],), 1 / plan.shape[1]))
    return {"loss": transport + epsilon * entropy + tau * partial,
            "transport": transport, "entropy": entropy, "partial": partial, "plan": plan}


def gram(x: torch.Tensor) -> torch.Tensor:
    """HASD SGIR display: G(X)=XX^T (prototype axis, not feature axis)."""
    return x @ x.T


def _paired(before: Sequence[torch.Tensor], after: Sequence[torch.Tensor]):
    if not before or len(before) != len(after):
        raise ValueError("matching nonempty lists of bags required")
    if any(a.shape != b.shape for a, b in zip(before, after)):
        raise ValueError("corresponding bags must have matching prototype/attention shapes")
    return zip(before, after)


def sgir_loss(before: Sequence[torch.Tensor], after: Sequence[torch.Tensor]) -> torch.Tensor:
    """HASD SGIR display: sum_n ||G(before_n)-G(after_n)||_F, NOT squared."""
    return torch.stack([torch.linalg.matrix_norm(gram(a) - gram(b), ord="fro")
                        for a, b in _paired(before, after)]).sum()


def pacr_loss(before: Sequence[torch.Tensor], after: Sequence[torch.Tensor]) -> torch.Tensor:
    """HASD PACR display: sum over bags and corresponding prototype attentions."""
    return torch.stack([(a - b).square().sum() for a, b in _paired(before, after)]).sum()


def total_loss(das: torch.Tensor, sgir: torch.Tensor, pacr: torch.Tensor,
               lambda_sgir: float = 1., lambda_pacr: float = 1.) -> torch.Tensor:
    """HASD total display: DAS + lambda1 SGIR + lambda2 PACR; no other terms."""
    if any(not math.isfinite(v) or v < 0 for v in (lambda_sgir, lambda_pacr)):
        raise ValueError("regularizer weights must be finite and nonnegative")
    return das + lambda_sgir * sgir + lambda_pacr * pacr
