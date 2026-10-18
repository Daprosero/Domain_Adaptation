"""DSAN / LMMD adapter (Zhu et al., IEEE TNNLS) over the vendored easezyc code.

Reproduces UDA/pytorch1.0/DSAN/DSAN.py L29: `lmmd_loss.get_loss(source, target,
s_label, softmax(t_label))`. Returns the LMMD term only; upstream main.py L42-43
multiplies it by `args.weight * lambd` with lambd = 2/(1+exp(-10*epoch/nepoch))-1.

Upstream defaults (lmmd.py L6): kernel_type='rbf', kernel_mul=2.0, kernel_num=5,
fix_sigma=None (bandwidth = mean pairwise squared distance, detached).

Requires n_s == n_t: `cal_weight` multiplies the (n_t, C) target weights by a
(n_s, C) mask (lmmd.py L71-73) and `get_loss` slices the kernel at the source
size. The upstream result is a float32 tensor of shape (1,); it is returned as a
0-d tensor.
"""
from __future__ import annotations

import torch

from .upstream_lmmd import LMMD_loss


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: DSAN/LMMD (Zhu et al., TNNLS 2021), Eq. 7 and '
        '8 through the vendored easezyc/deep-transfer-learning@2969ed6 code; '
        'it implements no equation of the CREDA research concept (see '
        'CITATIONS.md)'
    ),
    "invariants": [
        "lmmd_is_zero_when_target_matches_source_and_predicts_its_labels",
        "lmmd_refuses_unequal_batches",
    ],
}


class LMMD:
    def __init__(self, feature_dim: int, num_classes: int, kernel_type: str = "rbf", kernel_mul: float = 2.0,
                 kernel_num: int = 5, fix_sigma=None) -> None:
        self.loss = LMMD_loss(class_num=num_classes, kernel_type=kernel_type, kernel_mul=kernel_mul,
                              kernel_num=kernel_num, fix_sigma=fix_sigma)

    def __call__(self, f_s, f_t, g_s, g_t, y_s):
        if f_s.size(0) != f_t.size(0):
            raise ValueError("LMMD requires n_s == n_t (upstream masks target weights with a source-sized mask)")
        out = self.loss.get_loss(f_s, f_t, y_s, torch.nn.functional.softmax(g_t, dim=1))
        return out.reshape(())
