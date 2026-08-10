"""Target pseudo-labels and the confidence that down-weights the unreliable ones.

Source labels are observed; target pseudo-labels are estimated and can be
wrong. Eq. (22) assigns a class by a deterministic maximum rule, and Eq. (24)
measures the uncertainty of the bag's class distribution with its quadratic
Renyi entropy, turning it into a confidence in [0, 1] through the ln C
normalizer. That normalizer is why the formulation requires C >= 2.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "research-concept-r14.md",
    "sections": ["3"],
    "equations": ["22", "24"],
    "invariants": [
        "pseudolabel_deterministic_tie_break",
        "confidence_in_unit_interval",
        "confidence_one_at_onehot_zero_at_uniform",
    ],
}


def pseudolabel(g: np.ndarray) -> int:
    """Implement Eq. (22): the deterministic decision rule on the bag's class scores.

    A class of maximal probability, with a fixed tie-break so the rule is a
    function: the lowest index among the maximizers. Returns a 0-based index
    into the C classes.
    """
    g = np.asarray(g, dtype=float).ravel()
    return int(np.argmax(g))


def probability_quadratic_entropy(g: np.ndarray) -> float:
    """The estimator of Eq. (24): H_2(g) = -ln(sum_c g_c^2).

    Bounded by 0 <= H_2 <= ln C over the simplex: the lower end at a one-hot
    distribution, the upper end at the uniform one.
    """
    g = np.asarray(g, dtype=float).ravel()
    mass = float(np.sum(g**2))
    if mass <= 0.0:
        raise ValueError("a class distribution cannot be identically zero")
    return float(-np.log(mass))


def bag_confidence(g: np.ndarray, n_classes: int | None = None) -> float:
    """Implement Eq. (24): w = 1 - H_2(g) / ln C, the normalized complement.

    A prediction concentrated on one class gets confidence near one; a uniform
    prediction gets zero. C >= 2 is required because ln C is the normalizer.
    """
    g = np.asarray(g, dtype=float).ravel()
    C = g.size if n_classes is None else int(n_classes)
    if C < 2:
        raise ValueError("the confidence normalizer ln C requires at least two classes")
    return float(1.0 - probability_quadratic_entropy(g) / np.log(C))


def confidences(G: np.ndarray) -> np.ndarray:
    """The confidence of every target bag, one row of `G` per bag."""
    G = np.atleast_2d(np.asarray(G, dtype=float))
    return np.array([bag_confidence(row, G.shape[1]) for row in G], dtype=float)
