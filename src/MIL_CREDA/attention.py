"""Instance relevance inside a bag, and the bag representation it induces.

A subject is a bag whose instances carry no label of their own and do not
contribute equally. Eq. (14) scores each embedding with a shared relevance
selector, Eq. (15) turns those logits into weights normalized strictly WITHIN
the bag, and Eq. (16) collapses the bag into a convex combination of its
instance embeddings.
"""

from __future__ import annotations

import torch

from MIL_CREDA import as_matrix, as_tensor

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["3"],
    "equations": ["14", "15", "16"],
    "invariants": ["bag_weights_on_simplex", "bag_embedding_permutation_invariant"],
}


def relevance_logits(
    H: torch.Tensor, V_R: torch.Tensor, b_R: torch.Tensor, v_R: torch.Tensor
) -> torch.Tensor:
    """Implement Eq. (14): nu = v_R^T tanh(V_R h + b_R), applied per instance.

    `H` is (m, d), one embedding per row. The same parameters serve source and
    target, so an embedding produces the same logit regardless of its domain —
    and, now that they are tensors, one set of learnable parameters does.
    """
    H = as_matrix(H)
    hidden = torch.tanh(H @ as_tensor(V_R).transpose(0, 1) + as_tensor(b_R))
    return hidden @ as_tensor(v_R)


def bag_weights(logits: torch.Tensor) -> torch.Tensor:
    """Implement Eq. (15): exponential normalization over the instances of ONE bag.

    Never across bags and never mixing domains: the weights express relative
    relevance among the instances of a single subject. `softmax` over the single
    instance axis IS that equation, and it applies the same shift by the maximum
    the proposal's stabilized form does — the shift cancels in the ratio.
    """
    return torch.softmax(as_tensor(logits).reshape(-1), dim=0)


def bag_embedding(H: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Implement Eq. (16): z = sum_a beta_a h_a, a convex combination in R^d.

    Invariant under permutations of the bag's instances, since the sum pairs
    each weight with its own embedding.
    """
    H = as_matrix(H)
    weights = as_tensor(weights).reshape(-1)
    if H.shape[0] != weights.shape[0]:
        raise ValueError("one weight per instance is required")
    return weights @ H
