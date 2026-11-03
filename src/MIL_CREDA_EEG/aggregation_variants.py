"""K3's aggregation variants: MIL-CREDA with another bag aggregation and nothing else.

Experiments plan v40, Sec. 6.3, over proposal r34 Eqs. 13-38. The aggregation
table trains MIL-CREDA with five mechanisms: its own relevance-plus-consensus
attention (r34 Eqs. 14-16, `mil_creda_model.MILCREDAModel`, unchanged) and the
four variants here, each the published pooling of Ilse, Tomczak and Welling,
"Attention-based Deep Multiple Instance Learning", ICML 2018,
https://proceedings.mlr.press/v80/ilse18a.html, read in the ingested copy
`guidance/introduction/ilse2018/ilse2018.md` (the equation numbers below are that
copy's):

- "published attention": Eq. 8, a_k = softmax_k(w^T tanh(V h_k)), and Eq. 7,
  z = sum_k a_k h_k. The original, non-gated form: no bias, no constraint on w,
  no temperature, no consensus.
- "gated attention": Eq. 9, a_k = softmax_k(w^T (tanh(V h_k) * sigm(U h_k))), and
  Eq. 7. V and U have the same size, as the paper states.
- "max": Eq. 5, z_m = max_k h_km, feature by feature.
- "mean": Eq. 6, z = (1/K) sum_k h_k.

V, U are L x M and w is 1 x L with L = `ATTENTION_WIDTH` (128, the width of the
own attention; the paper tested 64, 128 and 256), Glorot-initialized, as the
paper initializes every layer.

**Everything else is MIL-CREDA's.** The subclass keeps the two EEGNet-8,2
trunks, the common head, sigma (one fixed number, `fix_sigma`), and every
function of the objective: `adaptation_terms`, the bag kernel of Eqs. 17-18 and
the global and local terms read the pairs `(H, weights)` and the representation
this class returns, exactly as they read the own attention's. Only `bag_pairs`
(the in-bag weights), `represent` (the bag representation), the attention
parameters and their constraint change. `gamma` and `tau_att` do not exist in a
variant: they are recorded as not applicable (`NOT_APPLICABLE`), never reused.

**Implementation choice the plan does not fix: the weights of max pooling.**
The bag kernel (Eq. 18) weighs the instance kernel by in-bag weights, and max
pooling has none. A variant's weights are those its representation reads: for
max, the share of the M features whose maximum each instance attains (a convex
combination, constant almost everywhere, so no gradient flows through it; the
gradient flows through the embeddings in the kernel and through the max). Owed
to /experimental-deliberation for confirmation, with uniform weights as the
alternative.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from MIL_CREDA.attention import bag_embedding
from MIL_CREDA_EEG import eegnet
from MIL_CREDA_EEG.mil_creda_model import ATTENTION_WIDTH, CLASSES, MILCREDAModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["6.3", "7.3"],
    "experiments": ["K3-ablacion"],
    "equations": ["13", "17", "18", "19", "20"],
    "sources": {"Ilse, Tomczak and Welling 2018 (ICML), ingested copy "
                "guidance/introduction/ilse2018/ilse2018.md":
                ["Eq. 5 (max)", "Eq. 6 (mean)", "Eq. 7 (weighted sum)",
                 "Eq. 8 (attention)", "Eq. 9 (gated attention)"]},
    "invariants": [
        "the_default_model_is_unchanged",
        "each_aggregation_is_its_published_form",
        "the_variants_differ_only_in_the_aggregation",
        "the_parameters_a_variant_lacks_are_not_applicable",
        "an_aggregation_variant_trains_both_phases",
        "the_variant_trainers_refuse_an_unknown_variant",
    ],
}

#: Sec. 6.3's four variants (`mechanisms.AGGREGATIONS` without "own"), their published form.
FORMS = {
    "published attention": "Ilse et al. 2018 Eqs. 7-8: a_k = softmax_k(w^T tanh(V h_k)), "
                           "z = sum_k a_k h_k",
    "gated attention": "Ilse et al. 2018 Eqs. 7 and 9: a_k = softmax_k(w^T (tanh(V h_k) "
                       "* sigm(U h_k))), z = sum_k a_k h_k",
    "max": "Ilse et al. 2018 Eq. 5: z_m = max_k h_km",
    "mean": "Ilse et al. 2018 Eq. 6: z = (1/K) sum_k h_k",
}
VARIANTS = tuple(FORMS)
#: MIL-CREDA's phase-1 keys that a variant does not have (Eqs. 15-16).
NOT_APPLICABLE = ("attentionGamma", "attentionTemperature")
MAX_WEIGHTS_RULE = ("share of the features whose maximum each instance attains "
                    "(implementation choice, owed to /experimental-deliberation)")


class UnknownAggregation(ValueError):
    """An aggregation Sec. 6.3 does not declare as a variant."""


def check_aggregation(aggregation: str) -> str:
    if aggregation not in FORMS:
        raise UnknownAggregation(f"the aggregation variants are {list(VARIANTS)} (the own "
                                 f"attention is MILCREDAModel); got {aggregation!r}")
    return aggregation


class AggregationVariantModel(MILCREDAModel):
    """MIL-CREDA with one of Sec. 6.3's published aggregations in place of Eqs. 14-16."""

    def __init__(self, source_channels: int, target_channels: int, *, aggregation: str,
                 classes: int = len(CLASSES), attention_width: int = ATTENTION_WIDTH):
        nn.Module.__init__(self)       # not MILCREDAModel's: no V_R, b_R, v_R, gamma, tau_att
        self.aggregation = check_aggregation(aggregation)
        self.classes = int(classes)
        self.attention_gamma = None
        self.attention_temperature = None
        self.source_trunk = eegnet.EEGNetTrunk(source_channels)
        self.target_trunk = eegnet.EEGNetTrunk(target_channels)
        d = self.source_trunk.output_dim
        self._attention: list[str] = []
        if aggregation in ("published attention", "gated attention"):
            self.V = self._glorot(attention_width, d)
            self._attention.append("V")
            if aggregation == "gated attention":
                self.U = self._glorot(attention_width, d)
                self._attention.append("U")
            self.w = self._glorot(1, attention_width)
            self._attention.append("w")
        self.head = eegnet.EEGNetHead(d, self.classes)
        self._sigma: float | None = None

    @staticmethod
    def _glorot(rows: int, cols: int) -> nn.Parameter:
        p = nn.Parameter(torch.empty(rows, cols, dtype=torch.float64))
        nn.init.xavier_uniform_(p)
        return p

    def attention_parameters(self) -> list[nn.Parameter]:
        return [getattr(self, name) for name in self._attention]

    def constrain(self, *modules: str) -> None:
        """Max norm on what moved; the published aggregations carry no constraint."""
        for name in modules:
            if name != "attention":
                getattr(self, name).constrain()

    def weights(self, H: torch.Tensor) -> torch.Tensor:
        """The in-bag weights the aggregation reads, on the simplex."""
        m = H.shape[0]
        if m == 0:
            raise ValueError("a bag needs at least one instance")
        if self.aggregation == "mean":
            return torch.full((m,), 1.0 / m, dtype=H.dtype, device=H.device)
        if self.aggregation == "max":
            winners = H.detach().argmax(dim=0)
            return torch.bincount(winners, minlength=m).to(H.dtype) / H.shape[1]
        hidden = torch.tanh(H @ self.V.T)
        if self.aggregation == "gated attention":
            hidden = hidden * torch.sigmoid(H @ self.U.T)
        return torch.softmax((hidden @ self.w.T).reshape(-1), dim=0)

    def bag_pairs(self, H: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(H_k, self.weights(H_k)) for H_k in H]

    def represent(self, pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if self.aggregation == "max":
            return torch.stack([H.max(dim=0).values for H, _ in pairs])
        return torch.stack([bag_embedding(H, w) for H, w in pairs])
