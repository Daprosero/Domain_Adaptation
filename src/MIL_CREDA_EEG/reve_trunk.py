"""A frozen REVE instance encoder for the phase-1 check, outside the plan's protocol.

REVE (El Ouahidi et al., NeurIPS 2025, arXiv 2510.21585) is a pretrained EEG encoder that takes
any electrode layout through 3D positions looked up by electrode name. Here it stays frozen: each
128 Hz window is resampled to the 200 Hz REVE was pretrained at, z-scored per channel and clipped
at 15 standard deviations (the model card's preprocessing), encoded, and mean-pooled over channels
and patches into one 512-value vector. Only the standardization and the linear projection on top
are trained, exactly as for the spectral encoder, so the two encoders are compared under the same
MIL-CREDA phase 1. Weights: Hugging Face `brain-bzh/reve-base` (REVE Responsible Use License).
"""
from __future__ import annotations

import contextlib
import hashlib

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn

from MIL_CREDA_EEG import eegnet
from MIL_CREDA_EEG.modality import CGX_CHANNELS, EMOTIV_CHANNELS

REVISION = "33841986f6cc1438d9dcdbca5a2137d60f6007d9"
EMBED, WIDTH, CHUNK = 512, 64, 64
LAYERS = ("project",)
CHANNEL_DEPENDENT = ()
_MODEL: dict = {}
_CACHE: dict = {}


def _encoder(device: torch.device):
    """The frozen REVE and its position bank, loaded once per device."""
    if device not in _MODEL:
        from transformers import AutoModel
        bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
        model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True,
                                          revision=REVISION).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL[device] = (bank, model)
    return _MODEL[device]


def channel_names(channels: int) -> tuple:
    names = {len(EMOTIV_CHANNELS): EMOTIV_CHANNELS, len(CGX_CHANNELS): CGX_CHANNELS}
    if channels not in names:
        raise ValueError(f"no montage with {channels} channels")
    return names[channels]


class ReveTrunk(nn.Module):
    """(n, channels, 512 at 128 Hz) -> (n, WIDTH); REVE frozen, its embeddings cached."""

    def __init__(self, channels: int, samples: int = eegnet.SAMPLES):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        self.names = list(channel_names(self.channels))
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)   # carries the device
        self.norm = nn.LayerNorm(EMBED)
        self.project = nn.Linear(EMBED, WIDTH)
        self.output_dim = WIDTH

    @property
    def masks(self) -> torch.Tensor:
        """The device carrier `spectral_check.standard_of` reads, as on the spectral trunk."""
        return self.anchor

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.project.reset_parameters()

    def constrain(self) -> None:
        return None

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        # One cache entry per window: the model concatenates the windows of several bags
        # in a different combination at every step, so a whole-batch key would never repeat.
        rows = x.detach().cpu().numpy()
        keys = [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in rows]
        missing = [i for i, k in enumerate(keys) if k not in _CACHE]
        if missing:
            device = self.anchor.device
            bank, model = _encoder(device)
            w = resample_poly(rows[missing].astype(np.float64), 25, 16, axis=-1)   # 128 -> 200 Hz
            w = (w - w.mean(-1, keepdims=True)) / w.std(-1, keepdims=True).clip(1e-8)
            w = torch.from_numpy(np.clip(w, -15, 15)).float().to(device)
            pos = bank(self.names).to(device)
            for i in range(0, len(w), CHUNK):
                part = w[i:i + CHUNK]
                emb = model(part, pos.expand(len(part), -1, -1)).mean(dim=(1, 2)).cpu()
                for j, e in zip(missing[i:i + CHUNK], emb):
                    _CACHE[keys[j]] = e
        return torch.stack([_CACHE[k] for k in keys]).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.norm(self.features(x)))


@contextlib.contextmanager
def installed():
    """`eegnet` builds this trunk while the block runs, and the plan's one afterwards."""
    names = {"EEGNetTrunk": ReveTrunk, "LAYERS": LAYERS, "CHANNEL_DEPENDENT": CHANNEL_DEPENDENT,
             "HEAD_MAX_NORM": 1.0}
    saved = {k: getattr(eegnet, k) for k in names}
    for key, value in names.items():
        setattr(eegnet, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(eegnet, key, value)
