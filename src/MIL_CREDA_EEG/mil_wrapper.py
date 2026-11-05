"""The common MIL wrapper of the non-MIL comparators, on EEG.

Experiments plan v50, Secs. 7.2 and 7.3, over proposal r38, Eqs. 15, 16, 19
and 20. The six comparators that are not MIL methods (UDA-DDA, CCA-LSC,
CASUAL, COT, CS+CCS and CLUST in its source-free variant) all receive the same
attention wrapper, and this is it: each window is encoded by the EEGNet-8,2
trunk, a learned tanh relevance score passes through a softmax over the
windows of the bag, the bag is the weighted sum of the window vectors, and the
common head classifies the bag once. That is Eq. 15 with gamma = 0: no
in-bag kernel consensus, no local subject matching, none of MIL-CREDA's
normalized terms (Sec. 7.2).

**One phase-1 group, shared.** Sec. 7.3 declares four phase-1 groups; this
module is the architecture of the common one (`GROUP`), whose single selected
checkpoint per transfer and rotation is shared by the six comparators.
AdaptMIL and HASD keep their native aggregators (their own groups), and
MIL-CREDA its own attention (`mil_creda_model.MILCREDAModel`).

**Reuse, not a second model.** `CommonMILModel` is `MILCREDAModel` with the
consensus weight fixed at 0 and the attention temperature fixed at 1, so its
two trunks (Eq. 13, one per domain with its own channel count), its single
attention shared by both domains and its common head are exactly MIL-CREDA's,
and `two_phase.train_source_phase`, `bridge` and `evaluate` run on it
unchanged. The relevance logits are `MIL_CREDA.attention.relevance_logits`
at gamma = 0, the pinned identity "relevance_logit_reduces_to_relevance_at_
gamma_zero" of that module; at gamma = 0 the consensus term is multiplied by
zero, so the kernel width handed to it (`NEUTRAL_SIGMA`) changes no weight,
and this model never reads a fitted sigma.

**The consensus is not computed at gamma = 0** (`zero_consensus_logits`, code
review should-fix). Eq. 15 multiplies the in-bag kernel consensus by gamma,
so at gamma exactly 0 its kernel is skipped and the logit is the relevance
term alone; the outputs and gradients are the same bytes as Eq. 15 evaluated
in full, because adding `0 * consensus` to a finite relevance adds nothing.

**Implementation choice the plan does not fix.** Sec. 7.2 writes "a learned
tanh relevance score through a softmax", with no temperature; the softmax is
taken at temperature 1 (`COMMON_TEMPERATURE`), and nothing is searched for the
common group in phase 1 but `lr1` and `wd1` (Sec. 7.7).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from MIL_CREDA.attention import bag_weights, relevance_component, relevance_logits
from MIL_CREDA_EEG.mil_creda_model import ATTENTION_WIDTH, CLASSES, MILCREDAModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["7.2", "7.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "equations": ["13", "15", "16", "19", "20"],
    "invariants": [
        "the_common_attention_is_eq_15_with_gamma_zero_and_no_consensus",
        "the_bag_is_the_attention_weighted_sum_and_the_head_applies_once_per_bag",
        "the_common_wrapper_fixes_gamma_and_temperature",
        "the_common_wrapper_is_the_phase_one_architecture_of_the_six_non_mil_comparators",
        "gamma_zero_skips_the_consensus_kernel",
        "the_gamma_zero_shortcut_is_byte_identical_in_outputs_and_gradients",
    ],
}

#: Sec. 7.3: the phase-1 group this architecture is.
GROUP = "common"
#: Sec. 7.1: the six non-MIL comparators that share it (Sec. 7.2).
COMPARATORS = ("UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS",
               "CLUST — variante source-free")
#: Eq. 15 with gamma = 0 (Sec. 7.2) and a softmax at temperature 1.
COMMON_GAMMA = 0.0
COMMON_TEMPERATURE = 1.0
#: The width handed to Eq. 15's consensus, which gamma = 0 multiplies by zero.
NEUTRAL_SIGMA = 1.0
#: The phase-1 keys of `two_phase.check_hyper` that the common group fixes rather
#: than searches. `sigmaMultiplier` only scales the width phase 1 measures and
#: stores; this model never reads it.
COMMON_PHASE_ONE_FIXED = {"attentionGamma": COMMON_GAMMA,
                          "attentionTemperature": COMMON_TEMPERATURE,
                          "sigmaMultiplier": 1.0}


def zero_consensus_logits(H: torch.Tensor, V_R: torch.Tensor, b_R: torch.Tensor,
                          v_R: torch.Tensor, gamma: float, sigma: float) -> torch.Tensor:
    """Eq. 15, without computing the consensus kernel when gamma is exactly 0."""
    if gamma == 0:
        return relevance_component(H, V_R, b_R, v_R)
    return relevance_logits(H, V_R, b_R, v_R, gamma, sigma)


class CommonMILModel(MILCREDAModel):
    """Two EEGNet trunks, one tanh-relevance attention (gamma = 0), one common head."""

    def __init__(self, source_channels: int, target_channels: int, *,
                 classes: int = len(CLASSES), attention_width: int = ATTENTION_WIDTH):
        super().__init__(source_channels, target_channels, attention_gamma=COMMON_GAMMA,
                         attention_temperature=COMMON_TEMPERATURE, classes=classes,
                         attention_width=attention_width)

    def bag_pairs(self, H: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Eqs. 15 (gamma = 0) and 16 per bag: (embeddings, in-bag weights)."""
        out = []
        for H_k in H:
            logits = zero_consensus_logits(H_k, self.V_R, self.b_R, self.v_R, COMMON_GAMMA,
                                           NEUTRAL_SIGMA)
            out.append((H_k, bag_weights(logits, COMMON_TEMPERATURE)))
        return out
