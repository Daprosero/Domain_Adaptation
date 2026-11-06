"""CLUDA §3.2.4 training projector, on the shared MIL inference architecture."""
from torch import nn

from MIL_CREDA_EEG.mil_wrapper import CommonMILModel

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["7.2", "7.3", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["cluda_projector_is_training_only_and_differentiable"],
}


class BagProjector(nn.Sequential):
    """Each source-only or target-only batch needs two bags in training mode."""
    def forward(self, x):
        if self.training and len(x) < 2:
            raise ValueError("CLUDA projector training requires at least two bags")
        return super().forward(x)


class CLUDAModel(CommonMILModel):
    """Common predictor plus a freshly initialized, inference-unused projector.

    Paper §3.2.4 / Table 1: two linear maps and nonlinearity; architecture
    details audited from official models.py:167-181, not vendored. The projector
    remains trainable in phase 2 through both official contrastive losses (L81).
    The source-source term trains this projector, never the frozen source trunk.
    """
    def __init__(self, source_channels: int, target_channels: int):
        super().__init__(source_channels, target_channels)
        width = self.source_trunk.output_dim
        self.projector = BagProjector(
            nn.Linear(width, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, affine=False),
            nn.Linear(1024, 256, bias=False),
            nn.BatchNorm1d(256, affine=False),
        ).double()
