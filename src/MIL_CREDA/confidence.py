"""Target pseudo-labels and the confidence that down-weights the unreliable ones.

Source labels are observed; target pseudo-labels are estimated and can be
wrong. Eq. (22) assigns a class by a deterministic maximum rule, and Eq. (24)
measures the uncertainty of the bag's class distribution with its quadratic
Renyi entropy, turning it into a confidence in [0, 1] through the ln C
normalizer. That normalizer is why the formulation requires C >= 2.
"""

from __future__ import annotations

import math

import torch

from MIL_CREDA import as_matrix, as_tensor

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["3"],
    "equations": ["22", "24"],
    "invariants": [
        "pseudolabel_deterministic_tie_break",
        "confidence_in_unit_interval",
        "confidence_one_at_onehot_zero_at_uniform",
    ],
}


def pseudolabel(g: torch.Tensor) -> int:
    """Implement Eq. (22): the deterministic decision rule on the bag's class scores.

    A class of maximal probability, with a fixed tie-break so the rule is a
    function: the lowest index among the maximizers. Returns a 0-based index
    into the C classes.

    The tie-break is written out rather than delegated to `argmax`, whose choice
    among equal maxima is a backend detail. Eq. (22) fixes it, so the code does.
    """
    g = as_tensor(g).reshape(-1)
    maximizers = torch.nonzero(g == g.max(), as_tuple=False).reshape(-1)
    return int(maximizers.min())


def probability_quadratic_entropy(g: torch.Tensor) -> torch.Tensor:
    """The estimator of Eq. (24): H_2(g) = -ln(sum_c g_c^2).

    Bounded by 0 <= H_2 <= ln C over the simplex: the lower end at a one-hot
    distribution, the upper end at the uniform one.
    """
    g = as_tensor(g).reshape(-1)
    mass = (g * g).sum()
    if mass <= 0.0:
        raise ValueError("a class distribution cannot be identically zero")
    return -torch.log(mass)


def bag_confidence(g: torch.Tensor, n_classes: int | None = None) -> torch.Tensor:
    """Implement Eq. (24): w = 1 - H_2(g) / ln C, the normalized complement.

    A prediction concentrated on one class gets confidence near one; a uniform
    prediction gets zero. C >= 2 is required because ln C is the normalizer.
    """
    g = as_tensor(g).reshape(-1)
    C = g.numel() if n_classes is None else int(n_classes)
    if C < 2:
        raise ValueError("the confidence normalizer ln C requires at least two classes")
    return 1.0 - probability_quadratic_entropy(g) / math.log(C)


def confidences(G: torch.Tensor) -> torch.Tensor:
    """The confidence of every target bag, one row of `G` per bag.

    Evaluated over the whole batch at once rather than bag by bag: Eq. (24) is
    the same expression for each row, and every one of these weights multiplies
    a block of Eq. (26), so they are all on the gradient's path.
    """
    G = as_matrix(G)
    C = G.shape[1]
    if C < 2:
        raise ValueError("the confidence normalizer ln C requires at least two classes")
    mass = (G * G).sum(dim=1)
    if bool((mass <= 0.0).any()):
        raise ValueError("a class distribution cannot be identically zero")
    entropies = -torch.log(mass)
    return 1.0 - entropies / math.log(C)
