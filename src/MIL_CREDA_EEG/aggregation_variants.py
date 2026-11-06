"""K3's aggregation variants: MIL-CREDA with another bag aggregation and nothing else.

Experiments plan v59, Sec. 6.3, over proposal r44 Eqs. 13-38. The aggregation
table trains MIL-CREDA with five mechanisms: its own relevance-plus-consensus
attention (r44 Eqs. 14-16, `mil_creda_model.MILCREDAModel`, unchanged) and the
four variants here (`VARIANTS`, in the table's order):

- "own without consensus" (`WITHOUT_CONSENSUS`): MIL-CREDA with gamma = 0 and
  everything else identical: the relevance selector with its bias b_R and its
  l1 constraint, and the attention temperature selected for full MIL-CREDA. It
  is `MILCREDAModel` itself built at `attention_gamma=0` (`build`), not a new
  aggregation; it carries the criterion of Secs. 6.1 and 6.3, because the
  consensus is its only difference from the own attention.

The other three are the published poolings of Ilse, Tomczak and Welling,
"Attention-based Deep Multiple Instance Learning", ICML 2018,
https://proceedings.mlr.press/v80/ilse18a.html, read in the ingested copy
`guidance/introduction/ilse2018/ilse2018.md` (the equation numbers below are that
copy's), reported without criterion (`FORMS`, `AggregationVariantModel`):

- "published attention": Eq. 8, a_k = softmax_k(w^T tanh(V h_k)), and Eq. 7,
  z = sum_k a_k h_k. The original, non-gated form: no bias, no constraint on w,
  no temperature, no consensus.
- "gated attention": Eq. 9, a_k = softmax_k(w^T (tanh(V h_k) * sigm(U h_k))), and
  Eq. 7. V and U have the same size, as the paper states.
- "mean": Eq. 6, z = (1/K) sum_k h_k.

Max pooling is not a variant (Sec. 6.3, Sec. 9): it has no per-instance weights
for the bag kernel.

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
published variant: they are recorded as not applicable (`NOT_APPLICABLE`),
never reused. The own attention without consensus keeps both, gamma set to 0
(`variant_hyper`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from MIL_CREDA.attention import bag_embedding
from MIL_CREDA_EEG import eegnet
from MIL_CREDA_EEG.mil_creda_model import ATTENTION_WIDTH, CLASSES, MILCREDAModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["6.3", "7.3"],
    "experiments": ["K3-ablacion"],
    "equations": ["13", "17", "18", "19", "20"],
    "sources": {"Ilse, Tomczak and Welling 2018 (ICML), ingested copy "
                "guidance/introduction/ilse2018/ilse2018.md":
                ["Eq. 6 (mean)", "Eq. 7 (weighted sum)",
                 "Eq. 8 (attention)", "Eq. 9 (gated attention)"]},
    "invariants": [
        "the_default_model_is_unchanged",
        "each_aggregation_is_its_published_form",
        "the_own_without_consensus_is_mil_creda_at_gamma_zero",
        "the_variants_differ_only_in_the_aggregation",
        "the_parameters_a_variant_lacks_are_not_applicable",
        "an_aggregation_variant_trains_both_phases",
        "the_variant_trainers_refuse_an_unknown_variant",
    ],
}

#: Sec. 6.3: MIL-CREDA with gamma = 0 and everything else identical.
WITHOUT_CONSENSUS = "own without consensus"
#: Sec. 6.3's published poolings, reported without criterion: their published form.
FORMS = {
    "published attention": "Ilse et al. 2018 Eqs. 7-8: a_k = softmax_k(w^T tanh(V h_k)), "
                           "z = sum_k a_k h_k",
    "gated attention": "Ilse et al. 2018 Eqs. 7 and 9: a_k = softmax_k(w^T (tanh(V h_k) "
                       "* sigm(U h_k))), z = sum_k a_k h_k",
    "mean": "Ilse et al. 2018 Eq. 6: z = (1/K) sum_k h_k",
}
#: Sec. 6.3's four variants (`mechanisms.AGGREGATIONS` without "own"), in the table's order.
VARIANTS = (WITHOUT_CONSENSUS, *FORMS)
WITHOUT_CONSENSUS_FORM = ("r44 Eqs. 15-16 at gamma = 0: relevance with bias b_R, l1 "
                          "constraint and full MIL-CREDA's selected temperature tau_att")
#: MIL-CREDA's phase-1 keys that a published variant does not have (Eqs. 15-16).
NOT_APPLICABLE = ("attentionGamma", "attentionTemperature")


class UnknownAggregation(ValueError):
    """An aggregation Sec. 6.3 does not declare as a variant."""


def check_aggregation(aggregation: str) -> str:
    if aggregation not in VARIANTS:
        raise UnknownAggregation(f"the aggregation variants are {list(VARIANTS)} (the own "
                                 f"attention is MILCREDAModel); got {aggregation!r}")
    return aggregation


def not_applicable(aggregation: str) -> tuple[str, ...]:
    """The phase-1 keys a variant lacks: none for the own attention without consensus."""
    return () if check_aggregation(aggregation) == WITHOUT_CONSENSUS else NOT_APPLICABLE


def variant_hyper(aggregation: str, hyper: Mapping) -> dict:
    """Full MIL-CREDA's selection as the variant reads it: gamma 0, or NaN where absent."""
    if check_aggregation(aggregation) == WITHOUT_CONSENSUS:
        return {**hyper, "attentionGamma": 0.0}
    return {**hyper, **{k: float("nan") for k in NOT_APPLICABLE}}


def build(aggregation: str, source_channels: int, target_channels: int, hyper: Mapping):
    """The variant's model; `hyper` is `variant_hyper`'s."""
    if check_aggregation(aggregation) == WITHOUT_CONSENSUS:
        if hyper["attentionGamma"] != 0:
            raise ValueError("the own attention without consensus has gamma = 0")
        return MILCREDAModel(source_channels, target_channels, attention_gamma=0.0,
                             attention_temperature=hyper["attentionTemperature"])
    return AggregationVariantModel(source_channels, target_channels, aggregation=aggregation)


class AggregationVariantModel(MILCREDAModel):
    """MIL-CREDA with one of Sec. 6.3's published aggregations in place of Eqs. 14-16."""

    def __init__(self, source_channels: int, target_channels: int, *, aggregation: str,
                 classes: int = len(CLASSES), attention_width: int = ATTENTION_WIDTH):
        nn.Module.__init__(self)       # not MILCREDAModel's: no V_R, b_R, v_R, gamma, tau_att
        if aggregation not in FORMS:
            raise UnknownAggregation(f"the published poolings are {list(FORMS)}; the own "
                                     f"attention without consensus is MILCREDAModel at gamma "
                                     f"0 (`build`); got {aggregation!r}")
        self.aggregation = aggregation
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
        hidden = torch.tanh(H @ self.V.T)
        if self.aggregation == "gated attention":
            hidden = hidden * torch.sigmoid(H @ self.U.T)
        return torch.softmax((hidden @ self.w.T).reshape(-1), dim=0)

    def bag_pairs(self, H: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(H_k, self.weights(H_k)) for H_k in H]

    def represent(self, pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return torch.stack([bag_embedding(H, w) for H, w in pairs])
