"""The global dependency score and the bounded loss it becomes.

Section 5 contrasts, class by class, the conditional geometry of both domains.
Eq. (32) trace-normalizes the three structures, Eq. (33) averages the two
marginal entropies against the entropy of the mixed structure, Eqs. (34)-(35)
give conservative extremes for that score, and Eq. (36) maps it affinely onto
[0, 1] with negative slope, so minimizing the loss maximizes the score.
Eq. (37) aggregates the active classes convexly, and yields zero when none is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from MIL_CREDA.renyi import quadratic_entropy, trace_normalize

__provenance__ = {
    "revision": "research-concept-r14.md",
    "sections": ["5"],
    "equations": ["32", "33", "34", "35", "36", "37"],
    "invariants": [
        "score_within_conservative_bounds",
        "class_global_loss_in_unit_interval",
        "global_loss_is_convex_combination",
        "global_loss_zero_when_no_active_class",
    ],
}


def is_active(K_s_c: np.ndarray, K_t_weighted: np.ndarray, K_mix_weighted: np.ndarray) -> bool:
    """The membership test for C_act of Eq. (37).

    A class participates only when bags exist in both domains and all three
    matrices that must be normalized have positive trace; otherwise the score
    is undefined for it and Eq. (37) excludes it rather than defaulting it.
    """
    matrices = (K_s_c, K_t_weighted, K_mix_weighted)
    if any(np.asarray(M).size == 0 for M in matrices):
        return False
    return all(float(np.trace(np.asarray(M, dtype=float))) > 0.0 for M in matrices)


def dependency_score(
    K_s_c: np.ndarray, K_t_weighted: np.ndarray, K_mix_weighted: np.ndarray
) -> float:
    """Implement Eqs. (32)-(33): J_c = (H_2(s) + H_2(t)) / 2 - H_2(mix).

    A surrogate score for structural dependency, not a conventional mutual
    information: the joint structure is the conditional mixed matrix, not a
    Hadamard product of paired marginals. Both marginals carry the same sign
    and the same coefficient, so neither is privileged.
    """
    H_source = quadratic_entropy(trace_normalize(K_s_c))
    H_target = quadratic_entropy(trace_normalize(K_t_weighted))
    H_mixed = quadratic_entropy(trace_normalize(K_mix_weighted))
    return float(0.5 * (H_source + H_target) - H_mixed)


def conservative_bounds(n_source: int, n_target: int) -> tuple[float, float]:
    """Implement Eq. (35): L_c = -ln(n_s + n_t) and U_c = (ln n_s + ln n_t) / 2.

    Obtained by combining the permitted extremes of each entropy separately, so
    they are conservative and possibly not tight: the three matrices are not
    independent and their extremes need not be reachable at once.
    """
    if n_source < 1 or n_target < 1:
        raise ValueError("both domains must contribute at least one bag to the class")
    lower = float(-np.log(n_source + n_target))
    upper = float(0.5 * (np.log(n_source) + np.log(n_target)))
    return lower, upper


def class_global_loss(score: float, lower: float, upper: float) -> float:
    """Implement Eq. (36): l_glob,c = (U_c - J_c) / (U_c - L_c), in [0, 1].

    The decreasing affine map sends U_c to zero and L_c to one. Because each
    class is normalized by its own bounds, classes of different cardinality end
    up on a common numerical scale — though equal normalized values do not mean
    the same absolute degree of alignment across classes.
    """
    span = upper - lower
    if span <= 0.0:
        raise ValueError("the conservative bounds must satisfy U_c > L_c")
    return float((upper - score) / span)


def global_loss(
    class_losses: Sequence[float], weights: Sequence[float] | None = None
) -> float:
    """Implement Eq. (37): the convex aggregation over the active classes.

    With no active class the term imposes no penalty and the unit-sum condition
    does not apply. Uniform weights treat every active class alike.
    """
    losses = np.asarray(list(class_losses), dtype=float)
    if losses.size == 0:
        return 0.0
    if weights is None:
        omega = np.full(losses.size, 1.0 / losses.size)
    else:
        omega = np.asarray(list(weights), dtype=float)
        if omega.size != losses.size:
            raise ValueError("one weight per active class is required")
        if np.any(omega < 0.0):
            raise ValueError("the aggregation weights must be non-negative")
        if not np.isclose(omega.sum(), 1.0):
            raise ValueError("the aggregation weights must sum to one")
    return float(omega @ losses)
