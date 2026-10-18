"""Local alignment: each target subject against a source reference built for it.

The global term aligns aggregated conditional distributions but leaves every
source bag of a class interchangeable. Section 4 resolves the subject-to-subject
correspondence with the SAME bag kernel: Eq. (28) spreads relevance among the
source bags of one class, Eq. (29) weights those distributions by the target's
own class scores so the total mass is one, Eq. (30) builds a personalized
reference in H_I, and Eq. (31) measures the squared RKHS distance to it using
kernel evaluations alone. Eq. (38) normalizes it and takes the uniform mean.
"""

from __future__ import annotations

import torch

from MIL_CREDA import as_index, as_tensor

__provenance__ = {
    "revision": "research-concept-r34.md",
    "sections": ["4", "5"],
    "equations": ["28", "29", "30", "31", "38"],
    "invariants": [
        "correspondence_simplex_per_class",
        "correspondence_order_preserving",
        "total_correspondence_sums_to_one",
        "local_distance_bounded_by_two",
        "local_distance_zero_iff_reference_matches",
        "local_loss_in_unit_interval",
        "local_loss_is_the_uniform_mean_and_ignores_confidence",
    ],
}


def class_correspondence(
    similarities: torch.Tensor, source_idx: torch.Tensor, tau: float | torch.Tensor
) -> torch.Tensor:
    """Implement Eq. (28): pi_{ji|c}, softmax over the source bags of one class.

    `similarities` holds kappa^B(B_i^s, B_j^t) for every source bag i and this
    one target bag j. Normalization happens strictly inside S_c. The class
    coverage assumption S_c != empty is strict: without it the denominator does
    not exist and there is no default value that replaces it.
    """
    if tau <= 0:
        raise ValueError("the local temperature tau_loc must be strictly positive")
    scores = as_tensor(similarities).reshape(-1)
    members = as_index(source_idx, device=scores.device)
    if members.numel() == 0:
        raise ValueError("a class with no source bag leaves pi_{ji|c} undefined")
    return torch.softmax(scores[members] / tau, dim=0)


def total_correspondence(
    similarities: torch.Tensor,
    source_labels: torch.Tensor,
    target_scores: torch.Tensor,
    tau: float | torch.Tensor,
) -> torch.Tensor:
    """Implement Eq. (29): pi_{ji} = g_{j,c_i} * pi_{ji|c_i}, over all source bags.

    Hierarchical by construction: the classifier's affinity fixes each class's
    total mass and the within-class correspondence distributes it among that
    class's subjects. Keeping the full distribution g_j^t, rather than the hard
    pseudo-label alone, is what preserves the classifier's uncertainty here.
    No second global normalization is needed.
    """
    scores = as_tensor(similarities).reshape(-1)
    labels = as_index(source_labels, device=scores.device)
    g = as_tensor(target_scores).reshape(-1)
    pi = torch.zeros(labels.numel(), dtype=scores.dtype, device=scores.device)
    for class_id, mass in enumerate(g):
        members = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
        if members.numel() == 0:
            if mass > 0.0:
                raise ValueError(
                    f"class {class_id} carries mass but has no source bag; "
                    "the local correspondence is undefined at this step"
                )
            continue
        # Out-of-place, so the mass g_{j,c} and the within-class distribution
        # both stay on the path back to the classifier and the bag kernel.
        pi = pi.index_put((members,), mass * class_correspondence(scores, members, tau))
    return pi


def local_distance(
    self_similarity: float | torch.Tensor,
    cross_similarities: torch.Tensor,
    source_gram: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """Implement Eq. (31): d_j^2 = ||Psi(B_j^t) - mu_j^s||^2, expanded in the kernel.

    The three terms are the target's self-similarity, its weighted approach to
    the compatible source subjects, and the internal geometry of the source
    prototype. Only kernel evaluations and the weights pi_{ji} appear; the
    explicit coordinates of Phi_sigma are never required.
    """
    pi = as_tensor(pi).reshape(-1)
    cross = as_tensor(cross_similarities).reshape(-1)
    K_ss = as_tensor(source_gram)
    return as_tensor(self_similarity) - 2.0 * (pi @ cross) + pi @ K_ss @ pi


def normalized_local_distance(
    squared_distance: float | torch.Tensor,
) -> torch.Tensor:
    """Implement the first half of Eq. (38): l_loc,j = d_j^2 / 2, in [0, 1].

    The constant is the tight one. The triangle inequality alone would give 4,
    but the Gaussian instance kernel is strictly positive, so every inner
    product in H_I is, and both Psi(B_j^t) and mu_j^s are convex combinations of
    those images with non-negative weights. Hence <Psi, mu> > 0 and
    d_j^2 = ||Psi||^2 - 2<Psi, mu> + ||mu||^2 < 2, which the sweep approaches.

    Elementwise, so one subject and a whole batch of them read the same.
    """
    return as_tensor(squared_distance) / 2.0


def local_loss(squared_distances: torch.Tensor) -> torch.Tensor:
    """Implement Eq. (38): the UNIFORM mean of l_loc,j over the target bags.

    Every target subject contributes the same weight, so `L_loc` estimates the
    mean local discrepancy over the target distribution.

    **It was confidence-weighted until r31 and r32 removed the weights, which
    is why this signature lost two arguments.** The old form was
    `sum_j w_j^t l_loc,j / (sum_j w_j^t + eps_loc)`, and the revision's own
    reason for dropping it is that `w_j^t` is a DIFFERENTIABLE function of the
    classifier's outputs (Eq. 24): a weight the model computes can be lowered
    on the worst-aligned subjects instead of aligning them, which lowers
    `L_loc` without moving the geometry. A uniform mean has no such channel.
    `eps_loc` went with the denominator it stabilized; nothing reads it now.

    The pseudo-label's uncertainty did not leave the term, it moved inside it:
    `total_correspondence` (Eq. 29) keeps the whole `g_j^t`, so a subject with
    an uncertain prediction is compared against a reference that mixes the
    classes in proportion to that uncertainty. Aggregation still acts only at
    this external level, leaving beta and pi untouched inside `d_j^2`.
    """
    losses = normalized_local_distance(as_tensor(squared_distances).reshape(-1))
    if losses.numel() == 0:
        return torch.zeros((), dtype=losses.dtype, device=losses.device)
    return losses.mean()
