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


class LogPowerTrunk(eegnet.EEGNetTrunk):
    """EEGNet with block 1's ELU and average pooling replaced by log of mean square.

    The spatially filtered signal is squared, averaged over the block-1 pooling window and
    logged: a log band power per filter, the operation of FBCNet and ShallowConvNet. Located
    reason: learning from windows without log compression fails across subjects here.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.channels or x.shape[2] != self.samples:
            raise ValueError(f"expected (n, {self.channels}, {self.samples}) windows")
        h = self.bn1(self.conv_temporal(eegnet._same(x.unsqueeze(1), eegnet.KERNEL_LENGTH)))
        h = self.bn2(self.depthwise(h))
        h = self.drop1(torch.log(self.pool1(h * h).clamp_min(1e-6)))
        h = self.bn3(self.separable_point(self.separable_depth(eegnet._same(h, eegnet.SEPARABLE_LENGTH))))
        h = self.drop2(self.pool2(self.elu2(h)))
        return h.flatten(1)


@contextlib.contextmanager
def log_power():
    """EEGNet with the log-power block 1 while the block runs, the plan's one afterwards."""
    plain = eegnet.EEGNetTrunk
    eegnet.EEGNetTrunk = LogPowerTrunk
    try:
        yield
    finally:
        eegnet.EEGNetTrunk = plain


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
