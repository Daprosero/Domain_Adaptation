"""The Gaussian kernel, used both on raw inputs and on instance embeddings.

Section 1 introduces it as the kernel of the RKHS the whole construction lives
in; Eq. (19) reuses the same form on the embeddings produced by the encoder.
Nothing else in MIL-CREDA introduces a second kernel or a second bandwidth.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["1", "3"],
    "equations": ["2", "19"],
    "invariants": ["kernel_unit_diagonal_and_bounded", "kernel_psd"],
}


def gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Implement Eq. (2)/(19): kappa_sigma(x, x') = exp(-||x - x'||^2 / (2 sigma^2)).

    `X` is (n, d), `Y` is (m, d); the result is the (n, m) matrix of pairwise
    kernel values. The proposal requires sigma > 0.
    """
    if sigma <= 0:
        raise ValueError("the bandwidth sigma must be strictly positive")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if X.shape[1] != Y.shape[1]:
        raise ValueError("both sets must live in the same ambient dimension")
    squared = (
        np.sum(X**2, axis=1)[:, None]
        - 2.0 * (X @ Y.T)
        + np.sum(Y**2, axis=1)[None, :]
    )
    # The identity above is exact in real arithmetic but can drift a hair below
    # zero in floating point for coincident points; the distance never is.
    return np.exp(-np.maximum(squared, 0.0) / (2.0 * sigma**2))


def gram(X: np.ndarray, sigma: float) -> np.ndarray:
    """The Gram matrix of one sample set: kappa_sigma(x_i, x_j) for all i, j."""
    return gaussian_kernel(X, X, sigma)
