"""ShallowConvNet as an instance encoder for the backbone test, outside plan v59.

Schirrmeister et al. (2017, https://doi.org/10.1002/hbm.23730): temporal convolution, spatial
convolution, square, mean pooling, log. The block computes a learned log band power, so it
carries the spectral prior that transferred in the band-power check. The paper's 250 Hz
sizes are halved for 128 Hz: kernel 13, pool 38. The stride of 16 gives 29 steps per window.
It keeps the `EEGNetTrunk` interface, so attention, kernels and alignment stay untouched.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from MIL_CREDA_EEG import eegnet

FILTERS, KERNEL, POOL, STRIDE = 40, 13, 38, 16
DROPOUT = 0.5
LAYERS = ("conv_temporal", "conv_spatial", "bn")
CHANNEL_DEPENDENT = ("conv_spatial",)


class ShallowTrunk(nn.Module):
    """(n, channels, 512) -> (n, FILTERS * 29)."""

    def __init__(self, channels: int, samples: int = eegnet.SAMPLES):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        self.conv_temporal = nn.Conv2d(1, FILTERS, (1, KERNEL))
        self.conv_spatial = nn.Conv2d(FILTERS, FILTERS, (self.channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(FILTERS)
        self.drop = nn.Dropout(DROPOUT)
        steps = (self.samples - KERNEL + 1 - POOL) // STRIDE + 1
        self.output_dim = FILTERS * steps

    def constrain(self) -> None:
        """No max-norm constraint in the original network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Both convolutions are linear, so one fused kernel gives the same output and never
        # holds the (n, FILTERS, channels, time) activation.
        spatial = self.conv_spatial.weight[..., 0]
        temporal = self.conv_temporal.weight[:, 0, 0]
        kernel = torch.einsum("oic,ik->ock", spatial, temporal).unsqueeze(1)
        bias = torch.einsum("oic,i->o", spatial, self.conv_temporal.bias)
        h = self.bn(F.conv2d(x.unsqueeze(1), kernel, bias))
        h = F.avg_pool2d(h * h, (1, POOL), stride=(1, STRIDE))
        return self.drop(torch.log(h.clamp_min(1e-6))).flatten(1)


@contextlib.contextmanager
def installed():
    """`eegnet` builds this trunk while the block runs, and the plan's one afterwards."""
    names = {"EEGNetTrunk": ShallowTrunk, "LAYERS": LAYERS, "CHANNEL_DEPENDENT": CHANNEL_DEPENDENT,
             "HEAD_MAX_NORM": 1.0}
    saved = {k: getattr(eegnet, k) for k in names}
    for key, value in names.items():
        setattr(eegnet, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(eegnet, key, value)
