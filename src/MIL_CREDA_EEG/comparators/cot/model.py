"""COT §4.1 training auxiliaries on the existing common MIL inference path."""
from __future__ import annotations

import torch
from torch import nn

from MIL_CREDA_EEG.mil_wrapper import CommonMILModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v58.md",
    "sections": ["7.2", "7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cot_uses_common_inference_and_four_auxiliary_centers"],
}

Q = 2
CENTER_NOISE_STD = .01


class COTModel(CommonMILModel):
    """Plan §7.4: frozen common-head rows, four auxiliary target clusters.

    No inference override: inherited predict uses ONLY EEGNet, common attention,
    and common head. Centers and potentials are training state, never diagnosis.
    """
    def __init__(self, source_channels: int, target_channels: int):
        super().__init__(source_channels, target_channels)
        self.target_centers = nn.Parameter(torch.zeros(Q*self.classes,
                                                       self.source_trunk.output_dim,
                                                       dtype=torch.float64))
        self.potentials = nn.Parameter(torch.zeros(Q*self.classes, dtype=torch.float64))

    @property
    def source_prototypes(self) -> torch.Tensor:
        """Paper §4.1 W rows, not a copied tensor that can silently drift."""
        return self.head.linear.weight

    @torch.no_grad()
    def initialize_centers(self, generator: torch.Generator) -> None:
        """Unspecified in paper: source-row repeats with seeded tie-breaking noise."""
        base = self.source_prototypes.repeat_interleave(Q, dim=0)
        noise = torch.randn(base.shape, generator=generator, dtype=base.dtype)
        self.target_centers.copy_(base + CENTER_NOISE_STD*noise.to(base.device))
        self.potentials.zero_()
