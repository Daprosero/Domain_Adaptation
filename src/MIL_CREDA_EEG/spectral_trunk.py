"""A spectral instance encoder for the adaptation test, outside plan v60.

Each window becomes its log absolute and log relative band power per channel (1-4, 4-8,
8-13, 13-30, 30-45 Hz at 128 Hz), the features that generalized across subjects in the
band-power check, then one learnable linear map. Layer normalization keeps no running
statistics, so no BatchNorm issue reaches it. It keeps the `EEGNetTrunk` interface.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from MIL_CREDA_EEG import eegnet

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
RATE, WIDTH = 128, 64
LAYERS = ("project",)
CHANNEL_DEPENDENT = ("project",)


class SpectralTrunk(nn.Module):
    """(n, channels, 512) -> (n, WIDTH)."""

    def __init__(self, channels: int, samples: int = eegnet.SAMPLES):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        freqs = torch.fft.rfftfreq(self.samples, 1 / RATE)
        masks = torch.stack([(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]).float()
        self.register_buffer("masks", masks / masks.sum(1, keepdim=True), persistent=False)
        self.norm = nn.LayerNorm(2 * len(BANDS) * self.channels)
        self.project = nn.Linear(2 * len(BANDS) * self.channels, WIDTH)
        self.output_dim = WIDTH

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.project.reset_parameters()

    def constrain(self) -> None:
        """No constraint."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(-1, keepdim=True)
        power = torch.fft.rfft(x, dim=-1).abs() ** 2 @ self.masks.T          # (n, C, bands)
        relative = power / power.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.cat([torch.log(power.clamp_min(1e-12)), torch.log(relative.clamp_min(1e-12))],
                         -1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.norm(self.features(x)))


@contextlib.contextmanager
def installed():
    """`eegnet` builds this trunk while the block runs, and the plan's one afterwards."""
    names = {"EEGNetTrunk": SpectralTrunk, "LAYERS": LAYERS, "CHANNEL_DEPENDENT": CHANNEL_DEPENDENT,
             "HEAD_MAX_NORM": 1.0}
    saved = {k: getattr(eegnet, k) for k in names}
    for key, value in names.items():
        setattr(eegnet, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(eegnet, key, value)
