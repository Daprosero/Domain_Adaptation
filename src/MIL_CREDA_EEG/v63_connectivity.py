"""Plan v63, Sec. 6.4: phase-locking graphs and their spectral descriptors.

Each window (channels x 512 samples at 128 Hz) of the analysis copy is re-referenced to its
common average, band-passed in theta (4-8 Hz), alpha (8-13 Hz) and beta (13-30 Hz)
(4th-order Butterworth, zero phase), its phase taken with the Hilbert transform, and the
PLV computed between every pair of channels, the diagonal set to zero and no threshold.
A subject's graph, per band, is the mean of its windows' PLV weighted by a set of window
weights (attention, uniform or permuted); its descriptors per band are each electrode's
degree and eigenvector centrality and the spectral radius, concatenated over the bands
into one vector per subject. The model never receives any of this.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v63.md",
    "sections": ["6.4"],
    "experiments": ["K4-conectividad", "C-dispositivo"],
    "invariants": [
        "the_plv_matrix_is_symmetric_in_zero_one_with_a_zero_diagonal",
        "the_graph_is_the_weighted_mean_of_the_window_plv",
        "the_descriptors_are_degree_centrality_and_spectral_radius_per_band",
    ],
}

RATE = 128
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
_SOS = {k: butter(4, list(v), "bandpass", fs=RATE, output="sos") for k, v in BANDS.items()}


def window_plv(windows: np.ndarray) -> dict[str, np.ndarray]:
    """(windows, channels, samples) -> band -> (windows, channels, channels) PLV."""
    x = np.asarray(windows, float)
    x = x - x.mean(1, keepdims=True)                     # common average, analysis copy
    out = {}
    for band, sos in _SOS.items():
        z = np.exp(1j * np.angle(hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)))
        plv = np.abs(np.einsum("wct,wdt->wcd", z, z.conj())) / x.shape[-1]
        idx = np.arange(x.shape[1])
        plv[:, idx, idx] = 0.0
        out[band] = plv
    return out


def weighted_graph(plv: dict[str, np.ndarray], weights: Sequence[float]) -> dict[str, np.ndarray]:
    w = np.asarray(weights, float)
    if w.ndim != 1 or len(w) != next(iter(plv.values())).shape[0] or w.sum() <= 0:
        raise ValueError("one nonnegative weight per window, not all zero")
    w = w / w.sum()
    return {b: np.einsum("w,wcd->cd", w, m) for b, m in plv.items()}


def descriptors(graph: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for band in BANDS:
        G = graph[band]
        values, vectors = np.linalg.eigh((G + G.T) / 2)
        top = vectors[:, -1]
        top = top * np.sign(top.sum() or 1.0)
        top = top / (np.linalg.norm(top) or 1.0)
        parts += [G.sum(1), top, [values[-1]]]
    return np.concatenate([np.asarray(p, float) for p in parts])


def subject_descriptors(windows: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    return descriptors(weighted_graph(window_plv(windows), weights))
