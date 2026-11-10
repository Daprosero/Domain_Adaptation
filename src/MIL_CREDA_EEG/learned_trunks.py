"""Two learned instance encoders for the phase-1 check, outside the plan's protocol.

TSMTrunk follows TSMNet (Kobler et al., NeurIPS 2022, arXiv 2206.01323): a temporal filter bank and
a spatial projection, the window's covariance, a Stiefel bilinear map (BiMap), eigenvalue
rectification (ReEig) and the matrix logarithm (LogEig), vectorized. The domain-specific SPD batch
norm of TSMNet is left out on purpose: here every adaptation step belongs to MIL-CREDA.

TGARTrunk ports T-GARNet (Salazar, Alvarez-Meza et al.; code github.com/dannasalazar11/Msc_thesis,
TGARNet/models/TGARNet.py, BSD-2-Clause) from Keras: layer norm over channels, a transformer
encoder over time with the channels as features, three Gaussian channel-connectivity kernels
combined convexly, three 3x3 convolutions and a dense projection. Its sigmas are scaled to the
median channel distance of the first batch (the original searched them with Optuna); its
alpha-Renyi mutual information regularizer is not included in this first check.

Both read windows of (channels, 512) at 128 Hz and return WIDTH values per window. Inputs are
standardized per channel by `input_standard_of` (the source training windows).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrizations

from MIL_CREDA_EEG import eegnet

WIDTH = 64


class ChannelStandard(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (torch.nan_to_num(x) - self.mean[:, None]) / self.std[:, None]


def input_standard_of(trunk, bags) -> ChannelStandard:
    with torch.no_grad():
        x = torch.nan_to_num(torch.cat([b.windows for b in bags]).double())
        m, s = x.mean(dim=(0, 2)), x.std(dim=(0, 2)).clamp_min(1e-6)
    return ChannelStandard(m.float(), s.float()).to(next(trunk.parameters()).device)


class _SymLog(torch.autograd.Function):
    """ReEig + LogEig with the Loewner-matrix backward of TSMNet: a safe divided difference
    replaces 1/(l_i - l_j) when two eigenvalues coincide, so the gradient stays finite."""

    @staticmethod
    def forward(ctx, S, eps):
        w, V = torch.linalg.eigh(S)
        wc = w.clamp_min(eps)
        ctx.save_for_backward(w, wc, V)
        ctx.eps = eps
        return V @ torch.diag_embed(torch.log(wc)) @ V.transpose(-1, -2)

    @staticmethod
    def backward(ctx, G):
        w, wc, V = ctx.saved_tensors
        f, fp = torch.log(wc), torch.where(w > ctx.eps, 1.0 / wc, torch.zeros_like(wc))
        dl = wc[..., :, None] - wc[..., None, :]
        df = f[..., :, None] - f[..., None, :]
        same = dl.abs() < 1e-8 * wc.abs().amax(-1, keepdim=True)[..., None].clamp_min(1e-12)
        L = torch.where(same, fp[..., :, None].expand_as(dl), df / torch.where(same, torch.ones_like(dl), dl))
        Gs = 0.5 * (G + G.transpose(-1, -2))
        return V @ (L * (V.transpose(-1, -2) @ Gs @ V)) @ V.transpose(-1, -2), None


def _sym_fn(S, fn, eps=1e-4):
    return _SymLog.apply(S, eps)


class TSMTrunk(nn.Module):
    def __init__(self, channels: int, samples: int = eegnet.SAMPLES, filters: int = 4,
                 spatial: int = 20, subspace: int = 10):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        self.norm = nn.Identity()
        self.temporal = nn.Conv2d(1, filters, (1, 25), padding=(0, 12), bias=False)
        self.spatial = nn.Conv2d(filters, spatial, (self.channels, 1), bias=False)
        self.bimap = parametrizations.orthogonal(nn.Linear(spatial, subspace, bias=False))
        self.iu = torch.triu_indices(subspace, subspace)
        self.out_norm = nn.LayerNorm(self.iu.shape[1])
        self.project = nn.Linear(self.iu.shape[1], WIDTH)
        self.output_dim = WIDTH

    def constrain(self) -> None:
        return None

    def forward(self, x):
        x = torch.nan_to_num(self.norm(x)).unsqueeze(1)                # (n, 1, C, T)
        z = self.spatial(self.temporal(x)).squeeze(2).double()       # (n, spatial, T)
        z = z - z.mean(-1, keepdim=True)
        S = z @ z.transpose(1, 2) / z.shape[-1]
        S = 0.5 * (S + S.transpose(1, 2))
        S = S + 1e-3 * torch.eye(S.shape[-1], dtype=S.dtype, device=S.device) * S.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
        W = self.bimap.weight.double()                               # (subspace, spatial), orthonormal rows
        S = W @ S @ W.T
        S = 0.5 * (S + S.transpose(1, 2))
        L = _sym_fn(S, torch.log)                                    # ReEig + LogEig
        v = L[:, self.iu[0], self.iu[1]].float()
        return self.project(self.out_norm(v))


class _Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.zeros(()))


class TGARTrunk(nn.Module):
    def __init__(self, channels: int, samples: int = eegnet.SAMPLES, scales=(0.5, 1.0, 2.0)):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        heads = next(h for h in (3, 2, 1) if self.channels % h == 0)
        self.norm = nn.Identity()
        self.in_norm = nn.LayerNorm(self.channels)
        self.encoder = nn.TransformerEncoderLayer(self.channels, heads, dim_feedforward=128,
                                                  dropout=0.1, batch_first=True)
        self.out_norm = nn.LayerNorm(self.channels)
        self.scale = _Scale()
        self.register_buffer("scales", torch.tensor(scales))
        self.mix = nn.Parameter(torch.zeros(len(scales)))
        self.conv1 = nn.Conv2d(1, 5, 3, padding=1)
        self.conv2 = nn.Conv2d(5, 3, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3, 3, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(3)
        self.drop = nn.Dropout(0.3)
        self.project = nn.Linear(3 * self.channels * self.channels, WIDTH)
        self.output_dim = WIDTH

    def constrain(self) -> None:
        return None

    def kernels(self, x):
        h = self.out_norm(self.encoder(self.in_norm(torch.nan_to_num(self.norm(x)).transpose(1, 2))))   # (n, T, C)
        h = h.transpose(1, 2)                                                          # (n, C, T)
        d2 = torch.cdist(h, h) ** 2
        if float(self.scale.value) == 0.0:
            with torch.no_grad():
                off = d2[:, ~torch.eye(self.channels, dtype=torch.bool, device=d2.device)]
                self.scale.value.fill_(float(off.median().clamp_min(1e-6)))
        sig2 = (self.scales ** 2 * self.scale.value / 2.0).to(d2.dtype)
        return torch.exp(-d2[:, None] / (2.0 * sig2[None, :, None, None]))            # (n, G, C, C)

    def forward(self, x):
        K = self.kernels(x)
        k = (torch.softmax(self.mix, 0)[None, :, None, None] * K).sum(1, keepdim=True)
        z = torch.relu(self.conv1(k))
        z = self.bn2(torch.relu(self.conv2(z)))
        z = self.bn3(torch.relu(self.conv3(z)))
        return self.project(self.drop(z.flatten(1)))


class _KerasEncoder(nn.Module):
    """keras_nlp TransformerEncoder as STF-KernelSHAP's InspectableTransformerEncoder builds it:
    multi-head self-attention with key_dim = value_dim = hidden // heads, no dropout, post-norm
    (epsilon 1e-6), then a ReLU feed-forward of `intermediate` units, post-norm."""

    def __init__(self, hidden: int, heads: int, intermediate: int):
        super().__init__()
        self.heads, self.key_dim = int(heads), max(1, int(hidden) // int(heads))
        width = self.heads * self.key_dim
        self.query, self.key, self.value = (nn.Linear(hidden, width) for _ in range(3))
        self.output = nn.Linear(width, hidden)
        self.attention_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.ff_in, self.ff_out = nn.Linear(hidden, intermediate), nn.Linear(intermediate, hidden)
        self.ff_norm = nn.LayerNorm(hidden, eps=1e-6)
        for layer in (self.query, self.key, self.value, self.output, self.ff_in, self.ff_out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):                                       # (n, T, hidden)
        n, t, _ = x.shape
        split = lambda z: z.view(n, t, self.heads, self.key_dim).transpose(1, 2)
        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        weights = torch.softmax(q @ k.transpose(-1, -2) / self.key_dim ** 0.5, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(n, t, -1)
        h = self.attention_norm(x + self.output(attended))
        return self.ff_norm(h + self.ff_out(torch.relu(self.ff_in(h))))


class TGARSTFTrunk(nn.Module):
    """T-GARNet as STF-KernelSHAP defines it (github.com/Daprosero/STF-KernelSHAP,
    src/stf_kernelshap/modeling/models.py TGARNet, layers.py): layer norm over channels
    (epsilon 1e-3), the transformer encoder over time, layer norm, ONE Gaussian channel kernel
    (C x C), its normalized order-2 Renyi entropy, two 3x3 SELU convolutions each followed by
    batch norm, flatten and dropout; the dense max-norm output is the common head
    (HEAD_MAX_NORM = norm_rate). STF's defaults: filters 3, heads 3, intermediate 128, dropout
    .3, norm_rate .25. STF searched sigma in [1, 20]; here sigma^2 = median off-diagonal squared
    channel distance / 2 of the first batch (the default sigma 1 leaves a diagonal kernel on 512
    normalized samples). Each forward appends its mean entropy to `entropies`, which the
    Renyi term of the loss consumes (`renyi_loss`)."""

    def __init__(self, channels: int, samples: int = eegnet.SAMPLES, filters: int = 3,
                 heads: int = 3, intermediate: int = 128, dropout: float = 0.3):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        self.norm = nn.Identity()
        self.in_norm = nn.LayerNorm(self.channels, eps=1e-3)
        self.encoder = _KerasEncoder(self.channels, heads, intermediate)
        self.out_norm = nn.LayerNorm(self.channels, eps=1e-3)
        self.scale = _Scale()
        self.conv1 = nn.Conv2d(1, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters, eps=1e-3, momentum=0.01)
        for conv in (self.conv1, self.conv2):
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        self.drop = nn.Dropout(dropout)
        self.output_dim = filters * self.channels * self.channels
        self.entropies: list[torch.Tensor] = []

    def constrain(self) -> None:
        return None

    def forward(self, x):
        h = self.out_norm(self.encoder(self.in_norm(torch.nan_to_num(self.norm(x)).transpose(1, 2))))
        h = h.transpose(1, 2)                                                          # (n, C, T)
        d2 = torch.cdist(h, h) ** 2
        if float(self.scale.value) == 0.0:
            with torch.no_grad():
                off = d2[:, ~torch.eye(self.channels, dtype=torch.bool, device=d2.device)]
                self.scale.value.fill_(float(off.median().clamp_min(1e-6)) / 2.0)
        K = torch.exp(-d2 / (2.0 * self.scale.value.to(d2.dtype)))                   # (n, C, C)
        A = K / torch.diagonal(K, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-8)[:, None, None]
        H = -torch.log((A * A).sum((-1, -2)).clamp_min(1e-8)) / np.log(max(self.channels, 2))
        self.entropies.append(H.mean())
        z = self.bn1(torch.nn.functional.selu(self.conv1(K[:, None])))
        z = self.bn2(torch.nn.functional.selu(self.conv2(z)))
        return self.drop(z.flatten(1))


@contextlib.contextmanager
def renyi_loss(classification_weight: float = 0.5):
    """STF's TGARNet objective: w x the classification loss + (1 - w) x the batch's mean Renyi
    entropy (minimized), on the source trunk's forward of that batch."""
    from MIL_CREDA_EEG import two_phase
    plain = two_phase.source_term

    def term(model, G, labels):
        base = plain(model, G, labels)
        ent = getattr(model.source_trunk, "entropies", None)
        if not ent:
            return base
        H = torch.stack(ent).mean().to(base.dtype)
        ent.clear()
        return classification_weight * base + (1.0 - classification_weight) * H

    two_phase.source_term = term
    try:
        yield
    finally:
        two_phase.source_term = plain


TRUNKS = {
    "tsm": (TSMTrunk, ("norm", "temporal", "spatial", "bimap", "out_norm", "project"), ("norm", "spatial")),
    "tgar": (TGARTrunk, ("norm", "in_norm", "encoder", "out_norm", "scale", "conv1", "conv2", "bn2", "conv3",
                         "bn3", "project"), ("norm", "in_norm", "encoder", "out_norm", "project")),
    "tgarstf": (TGARSTFTrunk, ("norm", "in_norm", "encoder", "out_norm", "scale", "conv1", "bn1",
                               "conv2", "bn2"), ("norm", "in_norm", "encoder", "out_norm")),
}
#: The common head's max norm per trunk (STF's TGARNet: norm_rate .25).
HEAD_MAX_NORM = {"tgarstf": 0.25}


@contextlib.contextmanager
def installed(kind: str):
    """`eegnet` builds this trunk while the block runs, and the plan's one afterwards."""
    cls, layers, dependent = TRUNKS[kind]
    names = {"EEGNetTrunk": cls, "LAYERS": layers, "CHANNEL_DEPENDENT": dependent,
             "HEAD_MAX_NORM": HEAD_MAX_NORM.get(kind, 1.0)}
    saved = {k: getattr(eegnet, k) for k in names}
    for key, value in names.items():
        setattr(eegnet, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(eegnet, key, value)
