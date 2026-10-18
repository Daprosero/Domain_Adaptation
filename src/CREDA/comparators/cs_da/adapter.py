"""CS-DA adapter (Yin et al., UAI 2024) over the vendored ywzcode/CS-adv code.

Reproduces train_image_officehome_cs.py L291-302 (Step A) for ONE classifier:
L2-normalise features and logits, then
    CS(Z_s, Z_t, 1, median_sigma=0)                                  (marginal, L296)
  + CondCSD(Z_s, Z_t, out_s, out_t, sigma=1, median_sigma=0)        (conditional, L300-302)
Upstream runs two classifiers (MCD-style) and sums one CondCSD per classifier
(L300-306); the common contract carries one prediction, so one term is computed.

`include_marginal` defaults to True: the paper's Step-1 objective (Eq. 15) is
L_cls + lambda * D_CS(p_s(z), p_t(z)) + beta * sum_n D_CCS(...), with lambda = beta = 1
(Sec. 4.2), which is what the script adds unweighted. The conditional term
conditions on the L2-normalised LOGITS, not on one-hot labels (L297-298).
Both terms are 0 for identical source and target batches by construction.
"""
from __future__ import annotations

import torch.nn.functional as F

from .upstream_csutils import CS, CondCSD


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: CS+CCS (Yin et al., UAI 2024), Eq. 11 and 13 '
        'through the vendored ywzcode/CS-adv@79d010b code; it implements no '
        'equation of the CREDA research concept (see CITATIONS.md)'
    ),
    "invariants": [
        "cs_da_is_zero_on_identical_batches",
        "cs_da_marginal_flag_matches_paper_default",
    ],
}


class CSDA:
    def __init__(self, feature_dim: int, num_classes: int, sigma: float = 1, include_marginal: bool = True) -> None:
        self.sigma = sigma
        self.include_marginal = include_marginal

    def __call__(self, f_s, f_t, g_s, g_t, y_s):
        Z_s, Z_t = F.normalize(f_s, p=2, dim=1), F.normalize(f_t, p=2, dim=1)
        output_s1, output_t1 = F.normalize(g_s, p=2, dim=1), F.normalize(g_t, p=2, dim=1)
        loss = CondCSD(Z_s, Z_t, output_s1, output_t1, sigma=self.sigma, median_sigma=0)
        if self.include_marginal:
            loss = CS(Z_s, Z_t, self.sigma, median_sigma=0) + loss
        return loss
