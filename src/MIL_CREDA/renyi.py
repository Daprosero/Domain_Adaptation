"""Matrix-based Renyi entropy over trace-normalized Gram matrices.

Section 2 builds the estimator: a Parzen window (Eq. 4) turns the unknown
density into pairwise kernel evaluations, the information potential collapses
to a Gram sum (Eqs. 7-8), and normalizing by the trace produces a unit-trace
positive semidefinite matrix on whose spectrum the entropy is defined (Eq. 9).
MIL-CREDA operates at alpha = 2, where Eq. (10) gives a closed Frobenius form
that avoids the spectral decomposition entirely.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "research-concept-r14.md",
    "sections": ["2"],
    "equations": ["7", "8", "9", "10", "11", "12"],
    "invariants": [
        "trace_one_after_normalization",
        "h2_frobenius_matches_spectral",
        "h2_bounded_by_log_n",
    ],
}


def information_potential(K: np.ndarray, dimension: int, sigma: float) -> float:
    """Implement Eqs. (7)-(8): V_2 = Z_{d,sigma} / N^2 * 1^T K 1.

    `K` carries the entries kappa_{sqrt(2) sigma}(x_i, x_j), as Eq. (7) requires:
    the convolution of two Gaussians of width sigma widens the bandwidth by
    sqrt(2). The constant Z_{d,sigma} = (pi sigma^2)^{d/2} cancels under the
    trace normalization of Eq. (9), which is why nothing downstream carries it.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    Z = (np.pi * sigma**2) ** (dimension / 2.0)
    return float(Z * K.sum() / (n**2))


def trace_normalize(K: np.ndarray) -> np.ndarray:
    """Turn a Gram matrix into the unit-trace matrix A = K / tr(K) of Eq. (9).

    The proposal only defines the entropy for a matrix with positive trace; a
    class whose trace vanishes is excluded from the active set instead (Eq. 37).
    """
    K = np.asarray(K, dtype=float)
    trace = np.trace(K)
    if trace <= 0.0:
        raise ValueError("a Gram matrix with non-positive trace has no normalization")
    return K / trace


def renyi_entropy(A: np.ndarray, alpha: float) -> float:
    """Implement Eq. (9): H_alpha(A) = ln(sum_i lambda_i^alpha) / (1 - alpha).

    Defined for alpha > 0, alpha != 1. This is the general form; the operative
    formulation fixes alpha = 2 and uses `quadratic_entropy` instead.
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("Renyi's order must satisfy alpha > 0 and alpha != 1")
    eigenvalues = np.linalg.eigvalsh(np.asarray(A, dtype=float))
    # A is positive semidefinite by construction; negative eigenvalues here are
    # rounding, and raising them to a fractional power would produce NaN.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return float(np.log(np.sum(eigenvalues**alpha)) / (1.0 - alpha))


def quadratic_entropy(A: np.ndarray) -> float:
    """Implement Eq. (10): H_2(A) = -ln(tr(A^T A)) = -ln ||A||_F^2.

    Quadratic cost in N against the cubic cost of a spectral decomposition, and
    a direct expression for backpropagation.
    """
    A = np.asarray(A, dtype=float)
    frobenius_squared = float(np.sum(A * A))
    if frobenius_squared <= 0.0:
        raise ValueError("the quadratic entropy needs a non-zero matrix")
    return float(-np.log(frobenius_squared))


def joint_normalized(K_x: np.ndarray, K_y: np.ndarray) -> np.ndarray:
    """Implement the joint matrix of Eq. (11): (K_x . K_y) / tr(K_x . K_y).

    The Hadamard product requires both marginals to have the same size and to
    be paired sample by sample. Section 5 states explicitly that the global
    term does NOT use this construction: across domains no such pairing exists,
    so the mixed matrix of Eq. (27) takes its place.
    """
    return trace_normalize(np.asarray(K_x, dtype=float) * np.asarray(K_y, dtype=float))


def matrix_mutual_information(K_x: np.ndarray, K_y: np.ndarray, alpha: float) -> float:
    """Implement Eq. (12): I = H(K_x) + H(K_y) - H(K_x, K_y).

    Kept as the informational antecedent Section 5 cites for the shape of the
    global score. It is not evaluated as a loss anywhere in the formulation.
    """
    return (
        renyi_entropy(trace_normalize(K_x), alpha)
        + renyi_entropy(trace_normalize(K_y), alpha)
        - renyi_entropy(joint_normalized(K_x, K_y), alpha)
    )
