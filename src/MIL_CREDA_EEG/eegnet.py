"""EEGNet-8,2 split in two: the instance encoder and the common predictive head.

Experiments plan v60, Sec. 7.3. The configuration is EEGNet-8,2 of Lawhern et
al. (2018, https://arxiv.org/abs/1611.08024) as its official implementation
fixes it (https://raw.githubusercontent.com/vlawhern/arl-eegmodels/master/EEGModels.py):
F1 = 8 temporal filters, D = 2 spatial filters per temporal filter, F2 = 16
separable filters; temporal kernel of 64 samples (half the 128 Hz rate);
depthwise spatial convolution of size (channels, 1) with max norm 1; average
pooling of 4 and of 8 samples and a separable kernel of 16, all fixed by the
paper for 128 Hz and not rescaled; BatchNorm, ELU and no bias. Dropout is 0.25,
the paper's cross-subject value (the code ships 0.5, its within-subject value).
The configuration is not searched.

**Split, as the plan states.** The convolutional trunk is the instance encoder:
one 512-sample window becomes one vector of F2 * 512 / 32 = 256 coordinates.
The final linear layer with softmax is the common head, with the classifier's
max norm 0.25, and it is applied once per bag on the bag representation (Eq. 19
and Eq. 20 of the proposal), never per window. `EEGNetHead` is that layer and
nothing else; who calls it, and on what, is `mil_creda_model`'s business.

**Standard initialization.** Every layer starts as a fresh network of the
official code starts: Keras' default Glorot-uniform kernels, computed with
Keras' own fan convention (for a depthwise kernel `(C, 1, F1, D)`: fan_in =
F1 * C, fan_out = D * C, which differs from PyTorch's `xavier_uniform_` on the
same weight), zero biases, BatchNorm at weight 1, bias 0, running mean 0 and
running variance 1. BatchNorm takes Keras' defaults too: momentum 0.99 (0.01 in
PyTorch's convention) and epsilon 1e-3. The max-norm constraints are applied
once at construction, so a fresh network already satisfies them; Keras applies
them only after updates, and a fresh Dense layer of 256 inputs draws row norms
near 1.4, above the head's 0.25.

**The layer copy of phase 2 (Sec. 7.3).** `copy_compatible_layers` copies every
layer with its parameters and BatchNorm buffers when the two trunks have the
same channel count (level and task arms). When they differ (device arm), the
only layer whose size depends on the channel count is the depthwise spatial
convolution, one weight per electrode: every other layer is copied and that one
keeps the standard initialization of a new network, in both directions. No
electrode mapping, and never zeros (with every weight at zero the filters of
that layer would start identical and never differentiate). It returns which
layers were copied.

**Official-code commit.** Sec. 7.3 asks the implementation to record the
commit of the official code it follows: vlawhern/arl-eegmodels at 4a512e5
(2022-05-02), the last commit touching `EEGModels.py` and the HEAD of `master`
when this unit read it (resolved 2026-11-02).
"""

from __future__ import annotations

import contextlib

import math

import torch
from torch import nn
from torch.nn import functional as F

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md",
    "sections": ["7.3", "3.1", "5.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K3-ablacion"],
    "equations": ["13", "20"],
    "invariants": [
        "the_trunk_maps_a_window_to_one_embedding_for_both_montages",
        "the_configuration_is_eegnet_8_2",
        "the_max_norm_constraints_hold_after_a_step",
        "a_fresh_network_satisfies_its_max_norm_constraints",
        "the_layer_copy_copies_every_layer_and_buffer_when_channels_match",
        "the_layer_copy_keeps_the_depthwise_standard_init_when_channels_differ",
        "the_standard_init_is_glorot_uniform_with_keras_fans",
    ],
}

#: Sec. 3.1: four seconds at 128 Hz.
SAMPLES = 512
F1, D, F2 = 8, 2, 16
KERNEL_LENGTH = 64
SEPARABLE_LENGTH = 16
POOL1, POOL2 = 4, 8
DROPOUT = 0.25
DEPTHWISE_MAX_NORM = 1.0
HEAD_MAX_NORM = 0.25
#: Keras BatchNormalization defaults (momentum 0.99 is 0.01 in PyTorch's sense).
BN_MOMENTUM, BN_EPS = 0.01, 1e-3
EMBEDDING_DIM = F2 * (SAMPLES // (POOL1 * POOL2))
#: The trunk's layers that carry parameters or buffers, in forward order.
LAYERS = ("conv_temporal", "bn1", "depthwise", "bn2", "separable_depth",
          "separable_point", "bn3")
#: The one layer whose shape depends on the channel count.
CHANNEL_DEPENDENT = ("depthwise",)
OFFICIAL_CODE = {
    "url": "https://raw.githubusercontent.com/vlawhern/arl-eegmodels/"
           "4a512e503198db2010848813ead9afbf8cd54c97/EEGModels.py",
    "paper": "https://arxiv.org/abs/1611.08024",
    "commit": "4a512e503198db2010848813ead9afbf8cd54c97",
    "commitNote": "last commit touching EEGModels.py (2022-05-02) and HEAD of master on "
                  "2026-11-02",
}


def _glorot_uniform_(weight: torch.Tensor, fan_in: int, fan_out: int) -> None:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.uniform_(-limit, limit)


def _max_norm_(weight: torch.Tensor, max_norm: float) -> None:
    """Keras `max_norm` on one weight: every output unit's norm at most `max_norm`.

    The output unit is dim 0 in PyTorch for both the depthwise kernel
    `(F1*D, 1, C, 1)` (norm over the electrodes, Keras axis 0 of `(C, 1, F1, D)`)
    and the linear weight `(classes, d)` (norm over inputs, Keras axis 0 of
    `(d, classes)`).
    """
    with torch.no_grad():
        weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=max_norm))


def _same(h: torch.Tensor, kernel: int) -> torch.Tensor:
    """Keras `padding="same"` along time: (k - 1) zeros, the extra one on the right."""
    total = kernel - 1
    return F.pad(h, (total // 2, total - total // 2))


@contextlib.contextmanager
def configured(f1: int = F1, d: int = D, kernel_length: int = KERNEL_LENGTH,
               dropout: float = DROPOUT):
    """Build trunks with another EEGNet setting (F2 = F1 x D, the paper's rule): every
    `EEGNetTrunk` constructed inside keeps it; trunks built outside keep EEGNet-8,2."""
    global F1, D, F2, KERNEL_LENGTH, DROPOUT
    saved = F1, D, F2, KERNEL_LENGTH, DROPOUT
    F1, D, F2, KERNEL_LENGTH, DROPOUT = int(f1), int(d), int(f1) * int(d), int(kernel_length), \
        float(dropout)
    try:
        yield
    finally:
        F1, D, F2, KERNEL_LENGTH, DROPOUT = saved


class EEGNetTrunk(nn.Module):
    """EEGNet-8,2 up to its flatten: (n, channels, 512) -> (n, 256)."""

    def __init__(self, channels: int, samples: int = SAMPLES):
        super().__init__()
        if samples % (POOL1 * POOL2):
            raise ValueError(f"samples must be a multiple of {POOL1 * POOL2}")
        self.channels = int(channels)
        self.samples = int(samples)
        # The configuration is read once, here, from the module constants: a caller that
        # needs another EEGNet setting rebinds them around construction (`configured`).
        self.f1, self.d, self.f2 = int(F1), int(D), int(F2)
        self.kernel_length, self.dropout = int(KERNEL_LENGTH), float(DROPOUT)
        self.conv_temporal = nn.Conv2d(1, self.f1, (1, self.kernel_length), bias=False)
        self.bn1 = nn.BatchNorm2d(self.f1, momentum=BN_MOMENTUM, eps=BN_EPS)
        self.depthwise = nn.Conv2d(self.f1, self.f1 * self.d, (self.channels, 1), groups=self.f1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.f1 * self.d, momentum=BN_MOMENTUM, eps=BN_EPS)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, POOL1))
        self.drop1 = nn.Dropout(self.dropout)
        self.separable_depth = nn.Conv2d(self.f1 * self.d, self.f1 * self.d, (1, SEPARABLE_LENGTH),
                                         groups=self.f1 * self.d, bias=False)
        self.separable_point = nn.Conv2d(self.f1 * self.d, self.f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.f2, momentum=BN_MOMENTUM, eps=BN_EPS)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, POOL2))
        self.drop2 = nn.Dropout(self.dropout)
        self.output_dim = self.f2 * (self.samples // (POOL1 * POOL2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """The standard initialization of a fresh network (module docstring)."""
        C = self.channels
        _glorot_uniform_(self.conv_temporal.weight, self.kernel_length, self.kernel_length * self.f1)
        _glorot_uniform_(self.depthwise.weight, self.f1 * C, self.d * C)
        _glorot_uniform_(self.separable_depth.weight, SEPARABLE_LENGTH * self.f1 * self.d,
                         SEPARABLE_LENGTH)
        _glorot_uniform_(self.separable_point.weight, self.f1 * self.d, self.f2)
        for bn in (self.bn1, self.bn2, self.bn3):
            bn.reset_parameters()
        self.constrain()

    def constrain(self) -> None:
        _max_norm_(self.depthwise.weight, DEPTHWISE_MAX_NORM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.channels or x.shape[2] != self.samples:
            raise ValueError(f"expected (n, {self.channels}, {self.samples}) windows, "
                             f"got {tuple(x.shape)}")
        h = self.bn1(self.conv_temporal(_same(x.unsqueeze(1), self.kernel_length)))
        h = self.drop1(self.pool1(self.elu1(self.bn2(self.depthwise(h)))))
        h = self.bn3(self.separable_point(self.separable_depth(_same(h, SEPARABLE_LENGTH))))
        h = self.drop2(self.pool2(self.elu2(h)))
        return h.flatten(1)


class EEGNetHead(nn.Module):
    """EEGNet's final dense layer with softmax: the common head, z -> g in the simplex."""

    def __init__(self, in_features: int = EMBEDDING_DIM, classes: int = 2,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self.linear = nn.Linear(in_features, classes, dtype=dtype)
        _glorot_uniform_(self.linear.weight, in_features, classes)
        with torch.no_grad():
            self.linear.bias.zero_()
        self.constrain()

    def constrain(self) -> None:
        _max_norm_(self.linear.weight, HEAD_MAX_NORM)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.linear(z), dim=-1)


def apply_max_norm(*modules: nn.Module) -> None:
    """Re-impose every EEGNet max-norm constraint after an optimizer step."""
    for module in modules:
        module.constrain()


def copy_compatible_layers(source: EEGNetTrunk, target: EEGNetTrunk) -> dict:
    """Sec. 7.3: initialize `target` from `source` where shapes are compatible.

    Copies parameters and buffers (BatchNorm statistics included) layer by
    layer. With equal channel counts every layer is copied; otherwise the
    depthwise spatial convolution keeps `target`'s own standard initialization.
    Any other shape mismatch is refused: it would mean the two trunks are not
    two EEGNet-8,2 of the plan.
    """
    if source.samples != target.samples:
        raise ValueError("source and target trunks must read windows of the same length")
    same = source.channels == target.channels
    copied, kept = [], []
    with torch.no_grad():
        for name in LAYERS:
            if not same and name in CHANNEL_DEPENDENT:
                kept.append(name)
                continue
            s_layer, t_layer = getattr(source, name), getattr(target, name)
            s_state = s_layer.state_dict()
            for key, value in t_layer.state_dict().items():
                if value.shape != s_state[key].shape:
                    raise ValueError(f"{name}.{key}: {tuple(s_state[key].shape)} vs "
                                     f"{tuple(value.shape)} is not a compatible layer")
                value.copy_(s_state[key])
            copied.append(name)
    return {"copied": copied, "kept_initialization": kept,
            "channels": {"source": source.channels, "target": target.channels},
            "rule": "all layers" if same else "all layers but the depthwise spatial "
                    "convolution, which keeps the standard initialization"}
