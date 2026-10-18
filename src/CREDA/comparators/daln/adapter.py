"""DALN adapter (Chen et al., CVPR 2022) over the vendored xiaoachen98/DALN code.

Reproduces `NuclearWassersteinDiscrepancy.forward` (upstream nwd.py L13-19) and
the README "Core code" usage: features of the concatenated source+target batch go
through the warm-start GRL, then through the task classifier's head, the output
is split in two halves, and the returned value is

    -n_discrepancy(y_s, y_t) = -(||softmax(y_s)||_* - ||softmax(y_t)||_*) / n_t

i.e. the README's `discrepancy_loss = -discrepancy(f)`: minus the empirical NWD
of paper Eq. (7)-(8) (the code takes the nuclear norm of the whole batch
prediction matrix, not per sample). The paper's min_G max_C game, Eq. (10), comes
from the GRL: the head receives d(-NWD) and so maximises NWD, the features
receive the reversed gradient and so minimise it. The trade-off lambda (paper
Eq. (12), lambda = 1) is NOT baked in: the README multiplies it outside the loss.

Caller contract (differs from the other comparators): the critic IS the shared
classifier head h, re-run on GRL(f). The precomputed logits g_s, g_t = h(f) carry
no GRL, so they cannot be used and are ignored; pass the head module instead:

    method(f_s, f_t, g_s, g_t, y_s, head=h)

`head` is required (TypeError otherwise). It is not stored: the adapter owns no
trainable parameter, and h's parameters already belong to the main optimiser.
h runs on torch.cat((f_s, f_t)), as upstream runs `classifier.head` on the joint
batch (any BatchNorm in h therefore sees joint statistics, as upstream).

State: the upstream `WarmStartGradientReverseLayer(alpha=1, lo=0, hi=1,
max_iters=1000, auto_step=True)` (nwd.py L10). Each call advances its iteration
counter, so call once per training step. The GRL coefficient is 0 on the first
call (features get no alignment gradient, the head still does) and rises towards
hi = 1 as 2/(1+exp(-i/1000)) - 1.

Requires n_s == n_t: upstream splits with `y.chunk(2, dim=0)` (nwd.py L16) and
divides both nuclear norms by the target size (L21).
"""
from __future__ import annotations

import torch
from torch import nn

from .upstream_nwd import NuclearWassersteinDiscrepancy


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: DALN (Chen et al., CVPR 2022), Eq. 7, 8 and 10 '
        'through the vendored xiaoachen98/DALN@e40ca55 code; it implements no '
        'equation of the CREDA research concept (see CITATIONS.md)'
    ),
    "invariants": [
        "value_is_negative_hand_written_nwd",
        "owns_no_trainable_parameters_and_does_not_capture_the_head",
        "head_gets_plain_gradient_and_features_get_reversed_scaled_gradient",
    ],
}


class DALN(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        # The upstream module is built with a parameter-free placeholder classifier:
        # it contributes its GRL (with upstream's own arguments) and its
        # `n_discrepancy`; the real classifier is the caller's head, per call.
        self.nwd = NuclearWassersteinDiscrepancy(nn.Identity())
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, f_s, f_t, g_s=None, g_t=None, y_s=None, head: nn.Module | None = None):
        if head is None:
            raise TypeError("DALN needs head=<the shared classifier head>: the critic is h(GRL(f)), "
                            "which the precomputed logits g_s, g_t cannot provide")
        if f_s.size(0) != f_t.size(0):
            raise ValueError("DALN requires n_s == n_t (upstream splits the batch with chunk(2))")
        f = torch.cat((f_s, f_t), dim=0)
        f_grl = self.nwd.grl(f)
        y = head(f_grl)
        y_s_logits, y_t_logits = y.chunk(2, dim=0)
        return -self.nwd.n_discrepancy(y_s_logits, y_t_logits)
