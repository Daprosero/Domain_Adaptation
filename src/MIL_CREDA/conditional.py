"""Class-conditional Gram blocks, confidence weighting, and the mixed matrix.

The conditional character of MIL-CREDA does not come from an explicit
conditional entropy: it comes from building Gram matrices class by class
(Eq. 23). Confidence weighting is applied to every block that involves target
bags and to none that does not (Eqs. 25-26), because source labels are observed
rather than estimated. Eq. (27) assembles the three blocks into one matrix
whose size tolerates n_c^s != n_c^t.
"""

from __future__ import annotations

import torch

from MIL_CREDA import as_index, as_tensor

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


def class_indices(labels: torch.Tensor, class_id: int) -> torch.Tensor:
    """The index set S_c or T_c: the bags assigned to one class."""
    labels = as_index(labels)
    return torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)


def conditional_blocks(
    K_ss: torch.Tensor,
    K_st: torch.Tensor,
    K_tt: torch.Tensor,
    source_idx: torch.Tensor,
    target_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implement Eq. (23): the three per-class blocks of the bag kernel.

    Rows of K_c^{st} are source bags and columns are target bags, as the
    proposal states. A single bandwidth governs all three, since all three come
    from the same instance kernel.
    """
    return (
        _submatrix(K_ss, source_idx, source_idx),
        _submatrix(K_st, source_idx, target_idx),
        _submatrix(K_tt, target_idx, target_idx),
    )


def weighted_blocks(
    K_st_c: torch.Tensor, K_t_c: torch.Tensor, target_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implement Eq. (26): K~_c^t = D K_c^t D and K~_c^{st} = K_c^{st} D.

    `D` is the diagonal matrix of Eq. (25) built from the confidences of the
    target bags of this class. The source block is deliberately absent here.
    """
    D = torch.diag(as_tensor(target_weights).reshape(-1))
    return as_tensor(K_st_c) @ D, D @ as_tensor(K_t_c) @ D


def mixed_matrix(
    K_s_c: torch.Tensor, K_st_weighted: torch.Tensor, K_t_weighted: torch.Tensor
) -> torch.Tensor:
    """Implement Eq. (27): the (n_s + n_t) square block matrix of one class.

    Its size is what keeps the global score valid when the two domains
    contribute different numbers of bags: unlike the Hadamard product of
    Eq. (11), it needs no sample-to-sample pairing.
    """
    K_s_c = as_tensor(K_s_c)
    K_st_w = as_tensor(K_st_weighted)
    K_t_w = as_tensor(K_t_weighted)
    top = torch.cat((K_s_c, K_st_w), dim=1)
    bottom = torch.cat((K_st_w.transpose(0, 1), K_t_w), dim=1)
    return torch.cat((top, bottom), dim=0)


def _submatrix(
    M: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """The block M[rows, cols], with the index sets moved beside the matrix."""
    M = as_tensor(M)
    return M[as_index(rows, device=M.device)][:, as_index(cols, device=M.device)]
