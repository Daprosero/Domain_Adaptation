"""Instance relevance inside a bag, and the bag representation it induces.

A subject is a bag whose instances carry no label of their own and do not
contribute equally. Eq. (14) scores each embedding with a shared relevance
selector, Eq. (15) turns those logits into weights normalized strictly WITHIN
the bag, and Eq. (16) collapses the bag into a convex combination of its
instance embeddings.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "research-concept-r14.md",
    "sections": ["3"],
    "equations": ["14", "15", "16"],
    "invariants": ["bag_weights_on_simplex", "bag_embedding_permutation_invariant"],
}


def relevance_logits(
    H: np.ndarray, V_R: np.ndarray, b_R: np.ndarray, v_R: np.ndarray
) -> np.ndarray:
    """Implement Eq. (14): nu = v_R^T tanh(V_R h + b_R), applied per instance.

    `H` is (m, d), one embedding per row. The same parameters serve source and
    target, so an embedding produces the same logit regardless of its domain.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    hidden = np.tanh(H @ np.asarray(V_R, dtype=float).T + np.asarray(b_R, dtype=float))
    return hidden @ np.asarray(v_R, dtype=float)


def bag_weights(logits: np.ndarray) -> np.ndarray:
    """Implement Eq. (15): exponential normalization over the instances of ONE bag.

    Never across bags and never mixing domains: the weights express relative
    relevance among the instances of a single subject.
    """
    logits = np.asarray(logits, dtype=float).ravel()
    # Shifting by the maximum is the standard stabilization; it leaves the
    # normalized weights unchanged because the shift cancels in the ratio.
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


def bag_embedding(H: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Implement Eq. (16): z = sum_a beta_a h_a, a convex combination in R^d.

    Invariant under permutations of the bag's instances, since the sum pairs
    each weight with its own embedding.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    weights = np.asarray(weights, dtype=float).ravel()
    if H.shape[0] != weights.shape[0]:
        raise ValueError("one weight per instance is required")
    return weights @ H
