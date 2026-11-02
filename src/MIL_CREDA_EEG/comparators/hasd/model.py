"""HASD native ABMIL on shared EEGNet; controlled variant, plan §§7.3–7.4."""
from __future__ import annotations

import torch
from torch import nn

from MIL_CREDA_EEG import eegnet
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["7.1", "7.2", "7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion"],
    "invariants": ["hasd_native_attention_shapes_and_permutation",
                   "hasd_phase_one_has_finite_gradients_for_every_trainable_parameter"],
}

GROUP = "HASD"
ATTENTION_WIDTH = 128  # HASD unspecified; Ilse §4 includes this candidate.


class NativeAttention(nn.Module):
    """HASD §2.1, citing Ilse §2.4 Eqs.7–8, ungated, without bias."""

    def __init__(self, dimension: int):
        super().__init__()
        self.V = nn.Parameter(torch.empty(ATTENTION_WIDTH, dimension, dtype=torch.float64))
        self.w = nn.Parameter(torch.empty(ATTENTION_WIDTH, dtype=torch.float64))
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.w.unsqueeze(0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 2 or not len(h):
            raise ValueError("native attention needs a nonempty (instances, features) matrix")
        return torch.softmax(torch.tanh(h @ self.V.T) @ self.w, dim=0)


class HASDModel(nn.Module):
    """Two independent EEGNet trunks, a single native aggregator and the common head.

    The shape plumbing and sigma metadata API are reused from MILCREDAModel,
    not its consensus attention or objective. Sigma is unused compatibility
    metadata for the existing source-phase trainer (see REPLICATION.md).
    """

    trunk = MILCREDAModel.trunk
    represent = staticmethod(MILCREDAModel.represent)
    classify = MILCREDAModel.classify
    predict = MILCREDAModel.predict
    fix_sigma = MILCREDAModel.fix_sigma
    sigma = MILCREDAModel.sigma
    sigma_fixed = MILCREDAModel.sigma_fixed

    def __init__(self, source_channels: int, target_channels: int):
        super().__init__()
        self.classes = 2
        self.source_trunk = eegnet.EEGNetTrunk(source_channels)
        self.target_trunk = eegnet.EEGNetTrunk(target_channels)
        self.attention = NativeAttention(self.source_trunk.output_dim)
        self.head = eegnet.EEGNetHead(self.source_trunk.output_dim, self.classes)
        self._sigma = None

    def attention_parameters(self):
        return list(self.attention.parameters())

    def embed(self, bags, domain, chunk=None):
        # HASD §2.1 representation display: ragged bags, no labels on windows.
        if not bags or any(b.ndim != 3 or b.shape[0] == 0 for b in bags):
            raise ValueError("nonempty bags of windows required")
        if chunk is not None and chunk < 1:
            raise ValueError("chunk must be positive")
        return MILCREDAModel.embed(self, bags, domain, chunk)

    def bag_pairs(self, embeddings):
        return [(h, self.attention(h)) for h in embeddings]

    def constrain(self, *modules):
        for name in modules:
            if name != "attention":  # native ABMIL has no MIL-CREDA l1 projection
                eegnet.apply_max_norm(getattr(self, name))
