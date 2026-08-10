"""Class-conditional Gram blocks, confidence weighting, and the mixed matrix.

The conditional character of MIL-CREDA does not come from an explicit
conditional entropy: it comes from building Gram matrices class by class
(Eq. 23). Confidence weighting is applied to every block that involves target
bags and to none that does not (Eqs. 25-26), because source labels are observed
rather than estimated. Eq. (27) assembles the three blocks into one matrix
whose size tolerates n_c^s != n_c^t.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "research-concept-r16.md",
    "sections": ["3"],
    "equations": ["23", "25", "26", "27"],
    "invariants": [
        "weighted_target_equals_diagonal_congruence",
        "source_block_never_weighted",
        "mixed_matrix_symmetric_psd",
    ],
}


def class_indices(labels: np.ndarray, class_id: int) -> np.ndarray:
    """The index set S_c or T_c: the bags assigned to one class."""
    return np.flatnonzero(np.asarray(labels).ravel() == class_id)


def conditional_blocks(
    K_ss: np.ndarray,
    K_st: np.ndarray,
    K_tt: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implement Eq. (23): the three per-class blocks of the bag kernel.

    Rows of K_c^{st} are source bags and columns are target bags, as the
    proposal states. A single bandwidth governs all three, since all three come
    from the same instance kernel.
    """
    source_idx = np.asarray(source_idx, dtype=int)
    target_idx = np.asarray(target_idx, dtype=int)
    K_s = np.asarray(K_ss, dtype=float)[np.ix_(source_idx, source_idx)]
    K_st_c = np.asarray(K_st, dtype=float)[np.ix_(source_idx, target_idx)]
    K_t = np.asarray(K_tt, dtype=float)[np.ix_(target_idx, target_idx)]
    return K_s, K_st_c, K_t


def weighted_blocks(
    K_st_c: np.ndarray, K_t_c: np.ndarray, target_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Implement Eq. (26): K~_c^t = D K_c^t D and K~_c^{st} = K_c^{st} D.

    `D` is the diagonal matrix of Eq. (25) built from the confidences of the
    target bags of this class. The source block is deliberately absent here.
    """
    D = np.diag(np.asarray(target_weights, dtype=float).ravel())
    return np.asarray(K_st_c, dtype=float) @ D, D @ np.asarray(K_t_c, dtype=float) @ D


def mixed_matrix(
    K_s_c: np.ndarray, K_st_weighted: np.ndarray, K_t_weighted: np.ndarray
) -> np.ndarray:
    """Implement Eq. (27): the (n_s + n_t) square block matrix of one class.

    Its size is what keeps the global score valid when the two domains
    contribute different numbers of bags: unlike the Hadamard product of
    Eq. (11), it needs no sample-to-sample pairing.
    """
    K_s_c = np.asarray(K_s_c, dtype=float)
    K_st_weighted = np.asarray(K_st_weighted, dtype=float)
    K_t_weighted = np.asarray(K_t_weighted, dtype=float)
    return np.block([[K_s_c, K_st_weighted], [K_st_weighted.T, K_t_weighted]])
