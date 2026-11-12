"""IEEE cross-subject pilot without the common average reference.

Diagnostic outside the plan: the same `xpilot.smoke(experiment="ieee", ...)` run, with the
v63 instance rule's average-reference step removed (1-45 Hz band-pass and the <= 60 window
rule kept). It decides whether the average reference stays in the pipeline: if phase 2 still
beats predict on IEEE without it, the step is dropped.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
from scipy.signal import sosfiltfilt

from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import xds_pilot as X
from MIL_CREDA_EEG import xpilot as XP


def instances_no_car(w: np.ndarray) -> torch.Tensor:
    """`xds_pilot.instances` without the average reference."""
    w = w[::max(1, math.ceil(len(w) / V.WINDOWS))][:V.WINDOWS]
    w = sosfiltfilt(V.SOS, w - w.mean(-1, keepdims=True), axis=-1)
    return torch.from_numpy(np.ascontiguousarray(w)).float()


@contextlib.contextmanager
def no_average_reference():
    saved = X.instances
    X.instances = instances_no_car
    try:
        yield
    finally:
        X.instances = saved


def smoke(config, product=None, **kwargs) -> dict:
    with no_average_reference():
        return XP.smoke(config, product, **kwargs)
