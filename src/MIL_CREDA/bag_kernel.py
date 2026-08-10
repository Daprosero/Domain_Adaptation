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

import numpy as np

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
    H_u: np.ndarray,
    weights_u: np.ndarray,
    H_v: np.ndarray,
    weights_v: np.ndarray,
    sigma: float,
) -> float:
    """Implement Eq. (21): kappa^B(B_u, B_v) = sum_{a,b} beta_a beta_b kappa^I(h_a, h_b).

    Equivalently the inner product in H_I of the two representations Psi of
    Eq. (20), which is what makes it positive semidefinite. No explicit
    coordinates of Phi_sigma are ever needed.
    """
    K_instances = gaussian_kernel(H_u, H_v, sigma)
    beta_u = np.asarray(weights_u, dtype=float).ravel()
    beta_v = np.asarray(weights_v, dtype=float).ravel()
    if K_instances.shape != (beta_u.size, beta_v.size):
        raise ValueError("each bag needs exactly one weight per instance")
    return float(beta_u @ K_instances @ beta_v)


def bag_kernel_matrix(
    bags_rows: Sequence[Bag], bags_cols: Sequence[Bag], sigma: float
) -> np.ndarray:
    """Every pairwise bag kernel value between two collections of bags."""
    return np.array(
        [
            [bag_kernel(H_u, w_u, H_v, w_v, sigma) for H_v, w_v in bags_cols]
            for H_u, w_u in bags_rows
        ],
        dtype=float,
    ).reshape(len(bags_rows), len(bags_cols))


def bag_gram(bags: Sequence[Bag], sigma: float) -> np.ndarray:
    """The Gram matrix of one collection of bags under kappa^B."""
    return bag_kernel_matrix(bags, bags, sigma)
