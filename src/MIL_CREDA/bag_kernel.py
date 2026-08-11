"""The relevance-weighted bag kernel: the single geometry of the whole method.

Alignment is not measured on the Euclidean representation z of Eq. (16) but in
the RKHS H_I induced by the instance kernel. Eq. (20) represents a bag as the
relevance-weighted mean of its instances' images, and Eq. (21) evaluates the
inner product of two such representations as a double sum over instance pairs.

The same kernel feeds the global term (Eq. 23) and the local correspondence
(Eq. 28), so both operate on one geometry rather than two.
"""

from __future__ import annotations

from typing import Sequence

import torch

from MIL_CREDA import DTYPE, as_tensor
from MIL_CREDA.kernels import gaussian_kernel

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["3"],
    "equations": ["20", "21"],
    "invariants": [
        "bag_kernel_symmetric",
        "bag_kernel_psd",
        "bag_self_similarity_at_most_one",
        "bag_kernel_reduces_to_instance_kernel_on_singletons",
    ],
}

Bag = tuple  # (H, weights): embeddings (m, d) and their in-bag weights (m,)


def bag_kernel(
    H_u: torch.Tensor,
    weights_u: torch.Tensor,
    H_v: torch.Tensor,
    weights_v: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Implement Eq. (21): kappa^B(B_u, B_v) = sum_{a,b} beta_a beta_b kappa^I(h_a, h_b).

    Equivalently the inner product in H_I of the two representations Psi of
    Eq. (20), which is what makes it positive semidefinite. No explicit
    coordinates of Phi_sigma are ever needed.
    """
    K_instances = gaussian_kernel(H_u, H_v, sigma)
    beta_u = as_tensor(weights_u).reshape(-1)
    beta_v = as_tensor(weights_v).reshape(-1)
    if tuple(K_instances.shape) != (beta_u.numel(), beta_v.numel()):
        raise ValueError("each bag needs exactly one weight per instance")
    return beta_u @ K_instances @ beta_v


def bag_kernel_matrix(
    bags_rows: Sequence[Bag], bags_cols: Sequence[Bag], sigma: float | torch.Tensor
) -> torch.Tensor:
    """Every pairwise bag kernel value between two collections of bags.

    Assembled with `stack` rather than by writing into a preallocated buffer, so
    each entry keeps the path back to the embeddings and the bandwidth that
    produced it: the whole matrix is one node of the graph, not a constant.
    """
    bags_rows, bags_cols = list(bags_rows), list(bags_cols)
    if not bags_rows or not bags_cols:
        return _empty(bags_rows, bags_cols)
    return torch.stack([
        torch.stack([bag_kernel(H_u, w_u, H_v, w_v, sigma) for H_v, w_v in bags_cols])
        for H_u, w_u in bags_rows
    ])


def bag_gram(bags: Sequence[Bag], sigma: float | torch.Tensor) -> torch.Tensor:
    """The Gram matrix of one collection of bags under kappa^B."""
    return bag_kernel_matrix(bags, bags, sigma)


def _empty(bags_rows: list[Bag], bags_cols: list[Bag]) -> torch.Tensor:
    """The correctly shaped matrix when one side contributes no bag at all.

    A class present in one domain and absent from the other reaches Eq. (23)
    this way; Eq. (37) then excludes it. The dtype and device follow whichever
    side does have a bag, so an empty block still composes with the rest.
    """
    reference = next((as_tensor(H) for H, _ in bags_rows + bags_cols), None)
    shape = (len(bags_rows), len(bags_cols))
    if reference is None:
        return torch.empty(shape, dtype=DTYPE)
    return torch.empty(shape, dtype=reference.dtype, device=reference.device)
