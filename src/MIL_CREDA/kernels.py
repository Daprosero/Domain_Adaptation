"""The Gaussian kernel, used both on raw inputs and on instance embeddings.

Section 1 introduces it as the kernel of the RKHS the whole construction lives
in; Eq. (19) reuses the same form on the embeddings produced by the encoder.
Nothing else in MIL-CREDA introduces a second kernel or a second bandwidth.
"""

from __future__ import annotations

import torch

from MIL_CREDA import as_matrix

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["1", "3"],
    "equations": ["2", "19"],
    "invariants": ["kernel_unit_diagonal_and_bounded", "kernel_psd"],
}


def gaussian_kernel(
    X: torch.Tensor, Y: torch.Tensor, sigma: float | torch.Tensor
) -> torch.Tensor:
    """Implement Eq. (2)/(19): kappa_sigma(x, x') = exp(-||x - x'||^2 / (2 sigma^2)).

    `X` is (n, d), `Y` is (m, d); the result is the (n, m) matrix of pairwise
    kernel values. The proposal requires sigma > 0. The bandwidth may itself be a
    tensor: nothing here assumes it is a constant, so a caller is free to learn
    it alongside the encoder.
    """
    if sigma <= 0:
        raise ValueError("the bandwidth sigma must be strictly positive")
    X = as_matrix(X)
    Y = as_matrix(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("both sets must live in the same ambient dimension")
    squared = (
        (X * X).sum(dim=1).unsqueeze(1)
        - 2.0 * (X @ Y.transpose(0, 1))
        + (Y * Y).sum(dim=1).unsqueeze(0)
    )
    # The identity above is exact in real arithmetic but can drift a hair below
    # zero in floating point for coincident points; the distance never is. The
    # clamp is flat there, which is also the correct derivative: a coincident
    # pair is at the minimum of the squared distance.
    return torch.exp(-squared.clamp_min(0.0) / (2.0 * sigma**2))


def gram(X: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """The Gram matrix of one sample set: kappa_sigma(x_i, x_j) for all i, j."""
    return gaussian_kernel(X, X, sigma)
