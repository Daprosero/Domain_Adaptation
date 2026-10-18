"""CDAN+E adapter (Long et al., NeurIPS 2018) over the vendored thuml/CDAN code.

Reproduces pytorch/train_image.py L177-182: concatenate source and target
features and logits, softmax, `Entropy`, then `CDAN(..., entropy, calc_coeff(i),
random_layer)`. Returns the transfer loss only (upstream multiplies it by
`trade_off`, default 1.0, L190).

State: owns the `AdversarialNetwork` discriminator (hidden 1024, train_image.py
L115-118) and an optional `RandomLayer` (random_dim 1024, L242).

Multilinear map (paper Eq. 8): `use_random=None` (default) picks the randomized
map T_(.) (Eq. 6, `RandomLayer`, d = random_dim) when feature_dim * num_classes
> 4096, else the full map T_(x) (outer product). At (256, 31), 7936 > 4096, so
the default is randomized: 2,100,225 own trainable parameters versus 9,178,113
with the full map. `use_random=True/False` overrides the rule; this deviates
from the upstream CLI default `--random False` (L208), which follows Eq. 8 only
when the user sets the flag. The random matrices are fixed buffers (not
parameters): they move with `.to()` and are saved in `state_dict`. Its parameters must be added to the optimiser (the
GRL hook inside `AdversarialNetwork.forward` makes one backward train both
sides). `iter_num` counts calls and feeds `calc_coeff` like the loop index `i`.

Requires n_s == n_t: upstream builds the domain labels as `size // 2` ones then
zeros (loss.py L30-31) and splits the entropy masks at `size // 2` (L35-40).
"""
from __future__ import annotations

import torch
from torch import nn

from .upstream_loss import CDAN, Entropy
from .upstream_network import AdversarialNetwork, RandomLayer, calc_coeff


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        'third-party baseline: CDAN+E (Long et al., NeurIPS 2018), Eq. 4, 6, '
        '8 and 10 through the vendored thuml/CDAN@c904cd3 code; it implements '
        'no equation of the CREDA research concept (see CITATIONS.md)'
    ),
    "invariants": [
        "default_at_256_31_uses_random_map",
        "threshold_boundary",
        "matrices_are_buffers_in_state_dict_not_parameters",
        "math_matches_eq6",
    ],
}


class CDANE(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_size: int = 1024, use_random: bool | None = None,
                 random_dim: int = 1024, use_entropy: bool = True) -> None:
        super().__init__()
        if use_random is None:
            use_random = feature_dim * num_classes > 4096  # paper Eq. 8
        if use_random:
            self.random_layer = RandomLayer([feature_dim, num_classes], random_dim)
            self.ad_net = AdversarialNetwork(random_dim, hidden_size)
        else:
            self.random_layer = None
            self.ad_net = AdversarialNetwork(feature_dim * num_classes, hidden_size)
        self.use_entropy = use_entropy
        self.iter_num = 0

    def forward(self, f_s, f_t, g_s, g_t, y_s, coeff=None):
        if f_s.size(0) != f_t.size(0):
            raise ValueError("CDAN requires n_s == n_t (upstream splits the batch at size // 2)")
        features = torch.cat((f_s, f_t), dim=0)
        outputs = torch.cat((g_s, g_t), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        if coeff is None:
            coeff = calc_coeff(self.iter_num)
        self.iter_num += 1
        if self.use_entropy:
            entropy = Entropy(softmax_out)
            return CDAN([features, softmax_out], self.ad_net, entropy, coeff, self.random_layer)
        return CDAN([features, softmax_out], self.ad_net, None, None, self.random_layer)
