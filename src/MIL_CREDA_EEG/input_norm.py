"""A trainable normalization at the EEGNet input, for the phase-1 check, outside plan v59.

Amplitude and offset differ across tasks and devices. MIL-CREDA already keeps one trunk per
domain, so a normalization layer at each trunk's input is domain-specific by construction.
`batch` normalizes each channel with batch statistics and a learned scale and shift; `layer`
normalizes each window over channels and time, then applies a learned per-channel scale and
shift, and keeps no running statistics.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from MIL_CREDA_EEG import eegnet


class WindowNorm(nn.Module):
    """Each window to zero mean and unit variance over channels and time, then per-channel affine."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std * self.weight[:, None] + self.bias[:, None]


def trunk_class(kind: str, base: type) -> type:
    class NormTrunk(base):
        def __init__(self, channels: int, samples: int = eegnet.SAMPLES):
            super().__init__(channels, samples)
            self.input_norm = (nn.BatchNorm1d(self.channels) if kind == "batch"
                               else WindowNorm(self.channels))

        def reset_parameters(self) -> None:
            super().reset_parameters()
            if hasattr(self, "input_norm"):
                self.input_norm.reset_parameters()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(self.input_norm(x))

    NormTrunk.__name__ = f"{kind.capitalize()}NormTrunk"
    return NormTrunk


@contextlib.contextmanager
def installed(kind: str):
    """EEGNet with an input normalization while the block runs, the plan's one afterwards."""
    saved = {k: getattr(eegnet, k) for k in ("EEGNetTrunk", "LAYERS", "CHANNEL_DEPENDENT")}
    eegnet.EEGNetTrunk = trunk_class(kind, saved["EEGNetTrunk"])
    eegnet.LAYERS = ("input_norm", *saved["LAYERS"])
    eegnet.CHANNEL_DEPENDENT = (*saved["CHANNEL_DEPENDENT"], "input_norm")
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(eegnet, key, value)
