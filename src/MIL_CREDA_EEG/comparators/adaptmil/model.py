"""Native projected ABMIL on the shared EEGNet (AdaptMIL §3.3; plan §7.3).

This is the BALLADEER controlled variant, not AdaptMIL's original scheduler.
"""
from __future__ import annotations

import torch
from torch import nn

from MIL_CREDA_EEG import eegnet
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["7.2", "7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion"],
    "invariants": ["adaptmil_native_pool_is_permutation_invariant",
                   "adaptmil_uses_shared_eegnet_and_one_bag_head",
                   "adaptmil_projection_and_attention_have_finite_gradients",
                   "adaptmil_refuses_empty_bags_and_unknown_domains",
                   "adaptmil_mean_pooling_preserves_bag_boundaries"],
}


class NativeABMIL(nn.Module):
    """P followed by Ilse (2018) Eq.8 ungated attention (AdaptMIL §3.3).

    Appendix A fixes both dimensions to the instance embedding dimension.
    Linear+ReLU projection is unspecified by AdaptMIL; see REPLICATION.md.
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(dimension, dimension, dtype=torch.float64),
                                        nn.ReLU())
        self.V = nn.Linear(dimension, dimension, bias=False, dtype=torch.float64)
        self.w = nn.Linear(dimension, 1, bias=False, dtype=torch.float64)
        for layer in (self.projection[0], self.V, self.w):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.projection[0].bias)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)

    def weights(self, projected: torch.Tensor) -> torch.Tensor:
        # Ilse Eq.8: softmax over instances, not features or bags.
        if projected.ndim != 2 or projected.shape[0] == 0:
            raise ValueError("ABMIL requires a nonempty (instances, features) bag")
        return self.w(torch.tanh(self.V(projected))).squeeze(-1).softmax(dim=0)

    def forward(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(projected)
        return (weights[:, None] * projected).sum(dim=0), weights


class AdaptMILModel(nn.Module):
    """Two independent EEGNets, one shared native ABMIL and one EEGNetHead."""

    def __init__(self, source_channels: int, target_channels: int):
        super().__init__()
        self.classes = 2
        self.source_trunk = eegnet.EEGNetTrunk(source_channels)
        self.target_trunk = eegnet.EEGNetTrunk(target_channels)
        self.aggregator = NativeABMIL(self.source_trunk.output_dim)
        self.head = eegnet.EEGNetHead(self.source_trunk.output_dim, self.classes)
        self._adaptation = False

    def trunk(self, domain: str):
        if domain not in ("source", "target"):
            raise ValueError(f"unknown domain {domain!r}")
        return self.source_trunk if domain == "source" else self.target_trunk

    def embed(self, bags, domain: str, chunk: int | None = None):
        if not bags or any(b.ndim != 3 or b.shape[0] == 0 for b in bags):
            raise ValueError("expected nonempty bags of (instances, channels, samples)")
        # Reuse the existing flatten/encode/split implementation, including float64 cast.
        return MILCREDAModel.embed(self, bags, domain, chunk)

    def attention_parameters(self):
        # The shared bridge freezes every native aggregation parameter, P included.
        return list(self.aggregator.parameters())

    def constrain(self, *modules: str):
        for name in modules:
            if name != "attention":  # native ABMIL has no MIL-CREDA L1 constraint
                eegnet.apply_max_norm(getattr(self, name))

    def representations(self, bags, domain: str, *, mean: bool = False,
                        chunk: int | None = None):
        projected = [self.aggregator.project(h) for h in self.embed(bags, domain, chunk)]
        if mean:
            # AdaptMIL §3.3 / Eq.3: no attention module on the alignment path.
            return torch.stack([h.mean(dim=0) for h in projected])
        return torch.stack([self.aggregator(h)[0] for h in projected])

    def logits(self, bags, domain: str, *, mean: bool = False,
               chunk: int | None = None):
        return self.head.linear(self.representations(bags, domain, mean=mean, chunk=chunk))

    def predict(self, bags, domain: str, chunk: int | None = None):
        return self.head(self.representations(bags, domain, chunk=chunk))

    def train(self, mode: bool = True):
        super().train(mode)
        if self._adaptation:
            # Frozen parameters alone do not freeze BatchNorm buffers/dropout.
            self.source_trunk.eval()
            self.aggregator.eval()
            self.head.eval()
        return self
