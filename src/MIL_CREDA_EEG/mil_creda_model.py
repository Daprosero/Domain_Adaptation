"""MIL-CREDA on EEG: two EEGNet trunks, one shared attention, one common head.

Experiments plan v54, Secs. 3, 7.2 and 7.3, over proposal r38, Eqs. 13-39.
This module wires the proposal's objects to the EEG instance encoder; it does
not re-implement any of them. Every equation is called from `MIL_CREDA`:

- Eq. 13: two independent EEGNet-8,2 trunks (`eegnet.EEGNetTrunk`), one per
  domain, each with its domain's channel count (29 for CGX, 14 for EPOC X);
  no weight is shared between them.
- Eqs. 14-16: ONE relevance selector `R_phi` (`V_R`, `b_R`, `v_R`), the same
  for source and target instances, with consensus weight `gamma` and attention
  temperature `tau_att` (`attention.relevance_logits`, `attention.bag_weights`).
  `||v_R||_1 <= 1` is imposed at construction and after every step
  (`attention.project_onto_l1_ball`).
- Eqs. 17-19: the bag kernel (`bag_kernel.bag_kernel_matrix`) and the bag
  representation (`attention.bag_embedding`).
- Eq. 20: the common head (`eegnet.EEGNetHead`), applied once per bag.
- Eqs. 21-38 in `adaptation_terms` and `source_term`: the supervised term
  (`objective.source_loss`), pseudo-labels and confidences
  (`confidence.pseudolabel`, `confidence.confidences`), the conditional blocks
  with the confidence weighting of Eqs. 25-27 (`conditional.*`), the balanced
  global term of Eqs. 28-33 (`global_term.*`, `renyi.quadratic_entropy`) and the
  local term of Eqs. 34-38 (`local_term.*`).

**One bandwidth, a fixed number.** The instance kernel of Eq. 14 that the
attention's consensus and the bag kernel both read uses one `sigma`, a Python
float fixed once by `fix_sigma` and never a tensor of the graph. A second call
with another value is refused (`SigmaAlreadyFixed`), so phase 2 cannot move the
width phase 1 used (plan Sec. 7.7: `sigma2 = sigma1`).

**Numerics.** The trunks run in float32; their embeddings are cast to float64
before any kernel, as the `MIL_CREDA` package's invariants are stated at
double-precision tolerances. The attention and the head live in float64.

**Implementation choices the plan does not fix.** The selector's intermediate
width `d_R` is 128, the toy-set precedent (`MIL_CREDA_Benchmark.config.
ATTENTION_WIDTH`); the instance embeddings enter the kernel as they leave the
trunk, with no extra normalization (r38 states none); the confidence weight
enters the cross block too (`conditional.weighted_blocks` default, r38 Eq. 26).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from MIL_CREDA.attention import (bag_embedding, bag_weights, project_onto_l1_ball,
                                 relevance_logits)
from MIL_CREDA.bag_kernel import bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import confidences, pseudolabel
from MIL_CREDA.global_term import (balanced_dependency_score, balanced_mixture,
                                   class_global_loss_cosh, global_loss, is_active)
from MIL_CREDA.local_term import local_distance, local_loss, total_correspondence
from MIL_CREDA.objective import source_loss
from MIL_CREDA.renyi import quadratic_entropy
from MIL_CREDA_EEG import eegnet

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["3.2", "3.3", "7.2", "7.3", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K3-ablacion"],
    "equations": ["13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
                  "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
                  "35", "36", "37", "38"],
    "invariants": [
        "source_and_target_trunks_are_independent_eegnets_with_their_own_channels",
        "one_shared_attention_serves_both_domains",
        "the_bag_representation_is_eq_19_over_eq_15_16_weights_at_the_fixed_sigma",
        "the_head_is_applied_once_per_bag",
        "sigma_is_a_fixed_number_set_once",
        "the_l1_constraint_on_the_attention_holds_from_construction",
        "the_adaptation_terms_lie_in_their_declared_bounds",
        "a_class_no_target_bag_routes_to_is_charged_the_maximum",
        "the_local_correspondence_needs_every_class_in_the_source_batch",
    ],
}

#: Bag labels in class-index order (the manifest's `label`).
CLASSES = ("no", "yes")
ATTENTION_WIDTH = 128
#: Eq. 21's stabilizer, as in the toy set (`MIL_CREDA_Benchmark.config.EPSILON`).
EPSILON_SOURCE = 1e-8
#: Declared ranges of the normalized terms (Eqs. 21, 33, 38).
TERM_BOUNDS = {"source": (0.0, 1.0), "global": (0.0, 1.0), "local": (0.0, 1.0)}
DOMAINS = ("source", "target")


class SigmaNotFixed(RuntimeError):
    """The kernel was read before its bandwidth was fixed."""


class SigmaAlreadyFixed(RuntimeError):
    """A second, different bandwidth was offered after the first was fixed."""


class SourceBatchMissingClass(ValueError):
    """r38 Sec. 5: the local correspondence needs S_c non-empty for every class."""


class MILCREDAModel(nn.Module):
    """theta_s, theta_t, phi (one attention) and psi (one head)."""

    def __init__(self, source_channels: int, target_channels: int, *,
                 attention_gamma: float, attention_temperature: float,
                 classes: int = len(CLASSES), attention_width: int = ATTENTION_WIDTH):
        super().__init__()
        if attention_gamma < 0:
            raise ValueError("attention_gamma must be non-negative (Eq. 15)")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive (Eq. 16)")
        self.classes = int(classes)
        self.attention_gamma = float(attention_gamma)
        self.attention_temperature = float(attention_temperature)
        self.source_trunk = eegnet.EEGNetTrunk(source_channels)
        self.target_trunk = eegnet.EEGNetTrunk(target_channels)
        d = self.source_trunk.output_dim
        self.V_R = nn.Parameter(torch.empty(attention_width, d, dtype=torch.float64))
        self.b_R = nn.Parameter(torch.zeros(attention_width, dtype=torch.float64))
        self.v_R = nn.Parameter(torch.empty(attention_width, dtype=torch.float64))
        nn.init.xavier_uniform_(self.V_R)
        nn.init.normal_(self.v_R, std=0.1)
        project_onto_l1_ball(self.v_R)
        self.head = eegnet.EEGNetHead(d, self.classes)
        self._sigma: float | None = None

    # ------------------------------------------------------------ bandwidth

    @property
    def sigma(self) -> float:
        if self._sigma is None:
            raise SigmaNotFixed("the instance-kernel bandwidth has not been fixed")
        return self._sigma

    @property
    def sigma_fixed(self) -> bool:
        return self._sigma is not None

    def fix_sigma(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"sigma must be a positive finite number, got {value!r}")
        if self._sigma is not None and self._sigma != value:
            raise SigmaAlreadyFixed(f"sigma is fixed at {self._sigma!r}; {value!r} refused")
        self._sigma = value
        return value

    # ------------------------------------------------------------ parameters

    def attention_parameters(self) -> list[nn.Parameter]:
        return [self.V_R, self.b_R, self.v_R]

    def trunk(self, domain: str) -> eegnet.EEGNetTrunk:
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}")
        return self.source_trunk if domain == "source" else self.target_trunk

    def constrain(self, *modules: str) -> None:
        """Max-norm and l1 constraints after an optimizer step, on what moved."""
        for name in modules:
            if name == "attention":
                project_onto_l1_ball(self.v_R)
            else:
                getattr(self, name).constrain()

    # ------------------------------------------------------------ forward pieces

    def embed(self, bags: Sequence[torch.Tensor], domain: str,
              chunk: int | None = None) -> list[torch.Tensor]:
        """Eq. 13 per instance, every bag of `bags` through one trunk pass.

        `chunk` bounds how many windows go through the trunk at once (for
        evaluation without gradient); BatchNorm in train mode then normalizes
        over each chunk instead of the whole batch.
        """
        trunk = self.trunk(domain)
        counts = [int(b.shape[0]) for b in bags]
        flat = torch.cat(list(bags), dim=0)
        if chunk is None or flat.shape[0] <= chunk:
            out = trunk(flat)
        else:
            out = torch.cat([trunk(flat[i:i + chunk]) for i in range(0, flat.shape[0], chunk)])
        return list(torch.split(out.to(torch.float64), counts))

    def bag_pairs(self, H: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Eqs. 15-16 per bag: (embeddings, in-bag weights) at the fixed sigma."""
        sigma = self.sigma
        out = []
        for H_k in H:
            logits = relevance_logits(H_k, self.V_R, self.b_R, self.v_R,
                                      self.attention_gamma, sigma)
            out.append((H_k, bag_weights(logits, self.attention_temperature)))
        return out

    @staticmethod
    def represent(pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Eq. 19 for every bag, stacked: (bags, d)."""
        return torch.stack([bag_embedding(H, w) for H, w in pairs])

    def classify(self, Z: torch.Tensor) -> torch.Tensor:
        """Eq. 20: the head, once, over the stacked bag representations."""
        return self.head(Z)

    def predict(self, bags: Sequence[torch.Tensor], domain: str,
                chunk: int | None = None) -> torch.Tensor:
        return self.classify(self.represent(self.bag_pairs(self.embed(bags, domain, chunk))))


def one_hot(labels: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), classes).to(torch.float64)


def source_term(model: MILCREDAModel, G_source: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
    """Eq. 21, normalized by B_src, over the source bags of a batch."""
    return source_loss(G_source, one_hot(labels, model.classes), EPSILON_SOURCE)


def adaptation_terms(model: MILCREDAModel,
                     pairs_s: Sequence[tuple[torch.Tensor, torch.Tensor]],
                     labels_s: torch.Tensor,
                     pairs_t: Sequence[tuple[torch.Tensor, torch.Tensor]],
                     tau_local: float) -> dict:
    """Eqs. 22-38 over one source batch and one target batch.

    Returns the global term (Eq. 33), the local term (Eq. 38), the target
    predictions they read, and the per-class routing. Both terms keep their
    graph back to the target trunk; nothing here sums them, since Eq. 39
    carries a coefficient for each.
    """
    labels_s = labels_s.reshape(-1).long()
    source_counts = [int((labels_s == c).sum()) for c in range(model.classes)]
    missing = [CLASSES[c] if c < len(CLASSES) else c
               for c, n in enumerate(source_counts) if n == 0]
    if missing:
        raise SourceBatchMissingClass(
            f"the source batch holds no bag of class(es) {missing}; Eq. 34 is undefined")
    sigma = model.sigma
    K_ss = bag_kernel_matrix(pairs_s, pairs_s, sigma)
    K_st = bag_kernel_matrix(pairs_s, pairs_t, sigma)
    K_tt = bag_kernel_matrix(pairs_t, pairs_t, sigma)

    G_t = model.classify(model.represent(pairs_t))                 # Eq. 20
    w_t = confidences(G_t)                                         # Eq. 24
    pseudo = torch.tensor([pseudolabel(g) for g in G_t], device=K_ss.device)   # Eq. 22
    labels_dev = labels_s.to(K_ss.device)

    class_losses = []
    for c in range(model.classes):
        s_idx = torch.nonzero(labels_dev == c, as_tuple=False).reshape(-1)
        t_idx = torch.nonzero(pseudo == c, as_tuple=False).reshape(-1)
        if s_idx.numel() == 0 or t_idx.numel() == 0:
            class_losses.append(None)                              # not in C_act
            continue
        K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, s_idx, t_idx)  # Eq. 23
        K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w_t[t_idx])                # Eq. 26
        if not is_active(K_s_c, K_t_w, mixed_matrix(K_s_c, K_st_w, K_t_w)):       # Eq. 27
            class_losses.append(None)
            continue
        breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_w, K_t_w)              # Eq. 28
        score = balanced_dependency_score(K_s_c, K_st_w, K_t_w)                   # Eq. 29
        class_losses.append(class_global_loss_cosh(                                # Eq. 32
            score, quadratic_entropy(breve_s), quadratic_entropy(breve_t)))
    global_value = global_loss(class_losses, source_counts)                       # Eq. 33

    distances = []
    for j in range(len(pairs_t)):
        cross = K_st[:, j]
        pi = total_correspondence(cross, labels_dev, G_t[j], tau_local)           # Eq. 35
        distances.append(local_distance(K_tt[j, j], cross, K_ss, pi))             # Eq. 37
    local_value = local_loss(torch.stack(distances))                              # Eq. 38

    return {"global": global_value.to(G_t.device), "local": local_value, "G_target": G_t,
            "pseudo": pseudo, "confidence_mean": float(w_t.detach().mean()),
            "active_classes": sum(loss is not None for loss in class_losses),
            "source_counts": source_counts}
