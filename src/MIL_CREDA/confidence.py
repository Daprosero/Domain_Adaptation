"""Target pseudo-labels and the confidence that down-weights the AMBIGUOUS ones.

Source labels are observed; target pseudo-labels are estimated and can be
wrong. Eq. (22) assigns a class by a deterministic maximum rule, and Eq. (24)
measures the uncertainty of the bag's class distribution with its quadratic
Renyi entropy, turning it into a confidence in [0, 1] through the ln C
normalizer. That normalizer is why the formulation requires C >= 2.

**It weighs AMBIGUITY, not ERROR, and that distinction is the whole property.**
`w` is a function of `g_j^t` and of nothing else, so it cannot see the true
label -- which, this being unsupervised domain adaptation, does not exist to be
seen. The consequence is the one a reader gets backwards: a bag whose posterior
concentrates on the WRONG class is confidently wrong and enters `D_c^t`
(Eq. 25) at nearly full weight. Only bags sitting BETWEEN classes are weighed
down.

This is not a defect. A target label is unobservable, so no term here can
penalize having gone to the wrong class; the same blindness is why the local
term of Section 4 lowers its loss for any coherent source neighbourhood rather
than for the right one. It is written down because without it "confidence"
reads as "correctness", and pinned by `confidence_is_blind_to_the_true_label`
because a property nothing asserts is a sentence, not a guarantee.

**Measured on the toyset** (`Results/Toyset/full`, arm `W-con`, seed 0, 81
target bags): in `toroide` 3 of the 6 mislabelled bags carry `w > 0.9`, one of
them 0.9988, while in `lunas` and `rollo` none exceeds 0.9. So confidence
correlates with correctness in two families and not in the third: an accident
of the data, never a guarantee of the equation.
"""

from __future__ import annotations

import math

import torch

from MIL_CREDA import as_matrix, as_tensor

__provenance__ = {
    "revision": "research-concept-r34.md",
    "sections": ["3"],
    "equations": ["22", "24"],
    "invariants": [
        "pseudolabel_deterministic_tie_break",
        "confidence_in_unit_interval",
        "confidence_one_at_onehot_zero_at_uniform",
        "confidence_is_blind_to_the_true_label",
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

    **Concentrated on WHICH class is not part of it.** `g` is the only argument
    and the expression is symmetric under permuting its entries, so two bags
    whose posteriors are permutations of one another carry the same `w` -- one
    of them may be right and the other confidently wrong. See the module
    docstring for why that is a property of unsupervised adaptation rather than
    a gap in it.
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
