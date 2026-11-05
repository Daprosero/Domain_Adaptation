"""CASUAL Eqs. 11–16, evaluated on bags (plan v54 §§7.2–7.4)."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["7.1", "7.2", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["casual_losses_have_closed_form_values",
                   "casual_joint_features_preserve_conditioning_and_gradients",
                   "casual_support_matches_sets_not_masses",
                   "casual_vat_is_input_bounded_and_mode_isolated",
                   "casual_objective_is_the_weighted_sum"],
}


def _log(p):
    return p.clamp_min(torch.finfo(p.dtype).tiny).log()


def classification_loss(probabilities, labels):
    """Paper Eq. 11: ordinary mean bag CE (not MIL-CREDA's normalized CE)."""
    return F.nll_loss(_log(probabilities), labels.long())


def conditional_entropy(probabilities):
    """Paper Eq. 12: entropy averaged over bags, natural logarithm."""
    return -(probabilities * _log(probabilities)).sum(-1).mean()


def joint_features(features, probabilities):
    """G after Eq. 15: (B,D) outer (B,K) -> (B,D*K), feature-major."""
    if features.ndim != 2 or probabilities.ndim != 2 or features.shape[0] != probabilities.shape[0]:
        raise ValueError("joint features need matching (bags, features/classes) matrices")
    return (features.unsqueeze(2) * probabilities.unsqueeze(1)).flatten(1)


def entropy_weights(probabilities):
    """Entropy conditioning cited in CASUAL: explicit detached CDAN-style assumption."""
    h = -(probabilities * _log(probabilities)).sum(-1)
    weights = 1 + (-h).exp()
    return (weights / weights.sum()).detach()


def discriminator_loss(source_logits, target_logits, source_probs=None, target_probs=None):
    """Paper Eq. 14: source=1, target=0; independent domain means (or weights)."""
    s = source_logits.reshape(-1)
    t = target_logits.reshape(-1)
    if not s.numel() or not t.numel():
        raise ValueError("both domain batches must be nonempty")
    ls, lt = F.softplus(-s), F.softplus(t)
    return ((ls.mean() if source_probs is None else (ls * entropy_weights(source_probs)).sum())
            + (lt.mean() if target_probs is None else (lt * entropy_weights(target_probs)).sum()))


def support_alignment(source_scores, target_scores, metric="squared_l2"):
    """Paper Eq. 15 / empirical Eq. 3: nearest opposite support in 1D."""
    s, t = source_scores.reshape(-1), target_scores.reshape(-1)
    if not s.numel() or not t.numel():
        raise ValueError("both supports must be nonempty")
    distances = (s[:, None] - t[None, :]).abs()
    if metric == "squared_l2":
        distances = distances.square()
    elif metric != "l1":
        raise ValueError("metric must be squared_l2 or l1")
    return distances.min(1).values.mean() + distances.min(0).values.mean()


@contextmanager
def deterministic_predictions(model):
    """VAT implementation choice: no dropout mismatch or BN-buffer updates."""
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for m, mode in modes.items():
            m.training = mode


def _unit(x):
    return x / torch.linalg.vector_norm(x).clamp_min(torch.finfo(x.dtype).tiny)


def _kl(reference, prediction):
    return (reference * (_log(reference) - _log(prediction))).sum(-1).mean()


def virtual_adversarial_loss(model, windows, radius, *, xi=0.01, iterations=1,
                             return_perturbations=False):
    """Paper Eq. 13, target part: perturb windows, compare bag predictions.

    Each bag is one sample: its full (m,C,T) perturbation has L2 norm radius.
    Power iteration computes input gradients only; it never fills parameter.grad.
    The final perturbed prediction retains its target-trunk parameter graph.
    """
    if not windows or not torch.isfinite(torch.tensor(float(radius))) or radius < 0:
        raise ValueError("VAT needs nonempty bags and a nonnegative finite radius")
    if xi <= 0 or iterations < 1:
        raise ValueError("VAT xi and iteration count must be positive")
    with deterministic_predictions(model):
        with torch.no_grad():
            reference = model.predict(windows, "target").detach()
        directions = [_unit(torch.randn_like(x)) for x in windows]
        for _ in range(iterations):
            probes = [(xi * d).detach().requires_grad_() for d in directions]
            divergence = _kl(reference, model.predict([x + d for x, d in zip(windows, probes)], "target"))
            gradients = torch.autograd.grad(divergence, probes)
            directions = [_unit(g.detach()) for g in gradients]
        perturbations = [(float(radius) * d).detach() for d in directions]
        loss = _kl(reference, model.predict([x + r for x, r in zip(windows, perturbations)], "target"))
    return (loss, perturbations) if return_perturbations else loss


def adaptation_objective(source_ce, alignment, entropy, vat, *, lambda_align,
                         lambda_entropy, lambda_vat):
    """Paper Eq. 16; source CE is a constant in plan v54's frozen-source variant."""
    return source_ce + lambda_align * alignment + lambda_entropy * entropy + lambda_vat * vat
