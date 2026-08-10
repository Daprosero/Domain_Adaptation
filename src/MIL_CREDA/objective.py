"""The supervised term and the complete MIL-CREDA objective.

Supervision exists only at the source bag level (Eq. 18). Eq. (39) combines it
with the two bounded adaptation terms: the supervised loss is fixed as the
reference with coefficient one, because the global scale of the objective does
not change its minimizer, and only the two adaptation terms carry free
non-negative coefficients. Both coefficients at zero reduce the formulation to
source-only supervised training.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "research-concept-r14.md",
    "sections": ["3", "5"],
    "equations": ["18", "39"],
    "invariants": [
        "source_loss_matches_negative_log_likelihood_of_the_observed_class",
        "objective_reduces_to_source_only_when_coefficients_vanish",
        "objective_monotone_in_each_adaptation_term",
    ],
}


def source_loss(G_source: np.ndarray, Y_source: np.ndarray, epsilon: float = 1e-8) -> float:
    """Implement Eq. (18): the per-bag averaged negative log-likelihood.

    `Y_source` holds one-hot bag labels, so the inner sum selects the observed
    class. `epsilon` is the numerical stabilizer of the logarithm.
    """
    if epsilon <= 0:
        raise ValueError("the source stabilizer must be strictly positive")
    G = np.atleast_2d(np.asarray(G_source, dtype=float))
    Y = np.atleast_2d(np.asarray(Y_source, dtype=float))
    if G.shape != Y.shape:
        raise ValueError("one label vector per bag, over the same classes, is required")
    if G.shape[0] == 0:
        raise ValueError("the supervised term needs at least one source bag")
    return float(-np.sum(Y * np.log(G + epsilon)) / G.shape[0])


def total_objective(
    supervised: float,
    global_term: float,
    local_term: float,
    lambda_global: float,
    lambda_local: float,
) -> float:
    """Implement Eq. (39): L_src + lambda_glob L_glob + lambda_loc L_loc.

    Both coefficients must be non-negative: a negative one would reward
    misalignment.
    """
    if lambda_global < 0 or lambda_local < 0:
        raise ValueError("both balance coefficients must be non-negative")
    return float(supervised + lambda_global * global_term + lambda_local * local_term)
