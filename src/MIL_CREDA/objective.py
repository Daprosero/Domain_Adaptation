"""The supervised term and the complete MIL-CREDA objective.

Supervision exists only at the source bag level (Eq. 18). Eq. (39) combines it
with the two bounded adaptation terms: the supervised loss is fixed as the
reference with coefficient one, because the global scale of the objective does
not change its minimizer, and only the two adaptation terms carry free
non-negative coefficients. Both coefficients at zero reduce the formulation to
source-only supervised training.
"""

from __future__ import annotations

import torch

from MIL_CREDA import as_matrix, as_tensor

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["3", "5"],
    "equations": ["18", "39"],
    "invariants": [
        "source_loss_matches_negative_log_likelihood_of_the_observed_class",
        "source_loss_non_negative",
        "objective_reduces_to_source_only_when_coefficients_vanish",
        "objective_monotone_in_each_adaptation_term",
    ],
}


def source_loss(
    G_source: torch.Tensor, Y_source: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Implement Eq. (18): the per-bag averaged negative log-likelihood.

    `Y_source` holds one-hot bag labels, so the inner sum selects the observed
    class. The stabilizer is normalized by its own maximum, (g + eps)/(1 + eps),
    so the argument never exceeds one: the logarithm stays non-positive, the
    term stays non-negative, and a bag predicted with certainty costs exactly
    zero. Adding eps inside the logarithm alone would make a certain, correct
    prediction score -ln(1 + eps) < 0.
    """
    if epsilon <= 0:
        raise ValueError("the source stabilizer must be strictly positive")
    G = as_matrix(G_source)
    Y = as_matrix(Y_source)
    if G.shape != Y.shape:
        raise ValueError("one label vector per bag, over the same classes, is required")
    if G.shape[0] == 0:
        raise ValueError("the supervised term needs at least one source bag")
    return -(Y * torch.log((G + epsilon) / (1.0 + epsilon))).sum() / G.shape[0]


def total_objective(
    supervised: float | torch.Tensor,
    global_term: float | torch.Tensor,
    local_term: float | torch.Tensor,
    lambda_global: float,
    lambda_local: float,
) -> torch.Tensor:
    """Implement Eq. (39): L_src + lambda_glob L_glob + lambda_loc L_loc.

    Both coefficients must be non-negative: a negative one would reward
    misalignment.

    This is the scalar an optimizer is handed. It is a tensor rather than a
    number for exactly that reason: casting it to a float here would sever the
    three terms from the encoder, the relevance selector and the classifier that
    produced them, and leave nothing to differentiate.
    """
    if lambda_global < 0 or lambda_local < 0:
        raise ValueError("both balance coefficients must be non-negative")
    return (
        as_tensor(supervised)
        + lambda_global * as_tensor(global_term)
        + lambda_local * as_tensor(local_term)
    )
