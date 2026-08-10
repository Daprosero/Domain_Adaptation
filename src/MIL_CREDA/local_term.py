"""Local alignment: each target subject against a source reference built for it.

The global term aligns aggregated conditional distributions but leaves every
source bag of a class interchangeable. Section 4 resolves the subject-to-subject
correspondence with the SAME bag kernel: Eq. (28) spreads relevance among the
source bags of one class, Eq. (29) weights those distributions by the target's
own class scores so the total mass is one, Eq. (30) builds a personalized
reference in H_I, and Eq. (31) measures the squared RKHS distance to it using
kernel evaluations alone. Eq. (38) normalizes and aggregates it by confidence.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["4", "5"],
    "equations": ["28", "29", "30", "31", "38"],
    "invariants": [
        "correspondence_simplex_per_class",
        "correspondence_order_preserving",
        "total_correspondence_sums_to_one",
        "local_distance_bounded_by_two",
        "local_distance_zero_iff_reference_matches",
        "local_loss_in_unit_interval",
    ],
}


def class_correspondence(
    similarities: np.ndarray, source_idx: np.ndarray, tau: float
) -> np.ndarray:
    """Implement Eq. (28): pi_{ji|c}, softmax over the source bags of one class.

    `similarities` holds kappa^B(B_i^s, B_j^t) for every source bag i and this
    one target bag j. Normalization happens strictly inside S_c. The class
    coverage assumption S_c != empty is strict: without it the denominator does
    not exist and there is no default value that replaces it.
    """
    if tau <= 0:
        raise ValueError("the local temperature tau_loc must be strictly positive")
    source_idx = np.asarray(source_idx, dtype=int)
    if source_idx.size == 0:
        raise ValueError("a class with no source bag leaves pi_{ji|c} undefined")
    scores = np.asarray(similarities, dtype=float).ravel()[source_idx] / tau
    shifted = np.exp(scores - scores.max())
    return shifted / shifted.sum()


def total_correspondence(
    similarities: np.ndarray,
    source_labels: np.ndarray,
    target_scores: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Implement Eq. (29): pi_{ji} = g_{j,c_i} * pi_{ji|c_i}, over all source bags.

    Hierarchical by construction: the classifier's affinity fixes each class's
    total mass and the within-class correspondence distributes it among that
    class's subjects. Keeping the full distribution g_j^t, rather than the hard
    pseudo-label alone, is what preserves the classifier's uncertainty here.
    No second global normalization is needed.
    """
    source_labels = np.asarray(source_labels).ravel()
    g = np.asarray(target_scores, dtype=float).ravel()
    pi = np.zeros(source_labels.size, dtype=float)
    for class_id, mass in enumerate(g):
        members = np.flatnonzero(source_labels == class_id)
        if members.size == 0:
            if mass > 0.0:
                raise ValueError(
                    f"class {class_id} carries mass but has no source bag; "
                    "the local correspondence is undefined at this step"
                )
            continue
        pi[members] = mass * class_correspondence(similarities, members, tau)
    return pi


def local_distance(
    self_similarity: float,
    cross_similarities: np.ndarray,
    source_gram: np.ndarray,
    pi: np.ndarray,
) -> float:
    """Implement Eq. (31): d_j^2 = ||Psi(B_j^t) - mu_j^s||^2, expanded in the kernel.

    The three terms are the target's self-similarity, its weighted approach to
    the compatible source subjects, and the internal geometry of the source
    prototype. Only kernel evaluations and the weights pi_{ji} appear; the
    explicit coordinates of Phi_sigma are never required.
    """
    pi = np.asarray(pi, dtype=float).ravel()
    cross = np.asarray(cross_similarities, dtype=float).ravel()
    K_ss = np.asarray(source_gram, dtype=float)
    return float(self_similarity - 2.0 * (pi @ cross) + pi @ K_ss @ pi)


def normalized_local_distance(squared_distance: float) -> float:
    """Implement the first half of Eq. (38): l_loc,j = d_j^2 / 2, in [0, 1].

    The constant is the tight one. The triangle inequality alone would give 4,
    but the Gaussian instance kernel is strictly positive, so every inner
    product in H_I is, and both Psi(B_j^t) and mu_j^s are convex combinations of
    those images with non-negative weights. Hence <Psi, mu> > 0 and
    d_j^2 = ||Psi||^2 - 2<Psi, mu> + ||mu||^2 < 2, which the sweep approaches.
    """
    return float(squared_distance / 2.0)


def local_loss(
    squared_distances: np.ndarray, target_weights: np.ndarray, epsilon: float = 1e-8
) -> float:
    """Implement Eq. (38): the confidence-weighted aggregation of l_loc,j.

    The stabilizer keeps the quotient defined when the total confidence is
    tiny, and drives the term to zero in that case: if no pseudo-label is
    reliable, local alignment imposes no penalty. Aggregation acts only at this
    external level, leaving beta and pi untouched inside d_j^2.
    """
    if epsilon <= 0:
        raise ValueError("the local stabilizer must be strictly positive")
    losses = np.array(
        [normalized_local_distance(d2) for d2 in np.asarray(squared_distances, dtype=float)],
        dtype=float,
    )
    w = np.asarray(target_weights, dtype=float).ravel()
    if losses.size != w.size:
        raise ValueError("one confidence per target bag is required")
    if losses.size == 0:
        return 0.0
    return float((w @ losses) / (w.sum() + epsilon))
