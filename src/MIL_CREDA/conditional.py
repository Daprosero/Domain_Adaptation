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
    "revision": "research-concept-r34.md",
    "sections": ["3"],
    "equations": ["23", "25", "26", "27"],
    "invariants": [
        "weighted_target_equals_diagonal_congruence",
        "source_block_never_weighted",
        "mixed_matrix_symmetric_psd",
        # The axis `weighted_blocks` used to pin with no name. See its
        # docstring for what each pole decides.
        "weight_cross_block_off_leaves_the_cross_block_unweighted",
        "weight_cross_block_freezes_a_zero_confidence_bag",
        "weight_cross_block_defaults_to_the_axis_value_creda_declares",
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
    K_st_c: torch.Tensor,
    K_t_c: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    weight_cross_block: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implement Eq. (26): K~_c^t = D K_c^t D and K~_c^{st} = K_c^{st} D.

    `D` is the diagonal matrix of Eq. (25) built from the confidences of the
    target bags of this class. The source block is deliberately absent here.

    **`weight_cross_block` is an axis, and it used to be pinned here with no
    name.** `True` -- the default -- is what r31 describes and what is
    published: the weight enters the CROSS block too, from the right and only
    on the target side, since rows of `K_c^{st}` are source bags and columns
    are target bags. `False` is CREDA's original (`CREDALoss`'s own default),
    which weights the target block and feeds the cross block in raw.

    **The difference is not cosmetic, and it is not a tightening either.** With
    the cross block weighted, a bag of confidence zero has its row and column
    emptied out of BOTH blocks that reach it, so the global term's gradient
    with respect to that bag is exactly zero and the bag stays FROZEN where it
    is. Without, the cross block still reaches it and the term can still repair
    it. CREDA names this the axis separating *who defines the geometry* from
    *whom the term repairs*, and observes that one number decides both when the
    opposite would suit each. Nothing here chooses between them; what this
    parameter buys is that a run can say which one it used.

    Under `False` the cross block is returned as it arrived -- not a clone, the
    same tensor `as_tensor` hands back -- so the setting costs nothing when it
    is off. The target block is weighted either way: Eq. (26)'s congruence on
    `K_c^t` is not what this axis is about.
    """
    D = torch.diag(as_tensor(target_weights).reshape(-1))
    K_st_c = as_tensor(K_st_c)
    cross = K_st_c @ D if weight_cross_block else K_st_c
    return cross, D @ as_tensor(K_t_c) @ D


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
