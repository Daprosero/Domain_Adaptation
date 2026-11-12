"""Realistic simulated EEG for testing when MIL-CREDA's phase 2 can help.

SEREEGA's principle (Krol et al., J Neurosci Methods 2018) in Python: dipolar cortical
sources in a head model, each with its own activity, projected to the electrodes by a
lead field. The head model is MNE-Python's spherical one (volume conduction, no MRI
download); each source signal is NeuroDSP's aperiodic 1/f background plus bursty theta,
alpha and beta oscillations (Cole et al., JOSS 2019).

What can be controlled, and why:
* class effect (ADHD-like): frontal theta up, frontal beta down, by `effect`;
* subject variability: individual alpha frequency, aperiodic exponent, source gains;
* device (domain): montage (19-channel 10-20 or a 14-channel EMOTIV-like set), per-channel
  gain, sensor noise and overall amplitude scale, grown by `shift`;
* concept shift: `invert` flips the sign of the class effect in a domain.
A subject is one bag: 4 s windows at 50% overlap at 128 Hz (60 windows), the instances.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

from MIL_CREDA_EEG import two_phase

RATE, WINDOW, WINDOWS = 128, 512, 60
SECONDS = (WINDOWS + 1) * WINDOW // 2 // RATE + 1          # 60 windows at a 2 s step
TEN_TWENTY = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
              "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz")
EMOTIV = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
MONTAGES = {"1020": TEN_TWENTY, "emotiv": EMOTIV}
#: source name -> (azimuth deg, elevation deg) on a 6 cm shell; frontal ones carry the class
SOURCES = {"frontal-L": (120, 35), "frontal-R": (60, 35), "central-L": (165, 55),
           "central-R": (15, 55), "parietal": (270, 50), "occipital": (270, 15)}
FRONTAL = ("frontal-L", "frontal-R")


@functools.lru_cache(maxsize=4)
def lead_field(montage: str) -> np.ndarray:
    """(channels, sources) fixed-orientation (radial) lead field of a spherical head."""
    import mne
    names = list(MONTAGES[montage])
    info = mne.create_info(names, RATE, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    sphere = mne.make_sphere_model("auto", "auto", info, verbose=False)
    c = np.asarray(sphere["r0"])
    rr, nn = [], []
    for az, el in SOURCES.values():
        a, e = np.deg2rad(az), np.deg2rad(el)
        u = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        rr.append(c + 0.06 * u)
        nn.append(u)
    src = mne.setup_volume_source_space(pos={"rr": np.array(rr), "nn": np.array(nn)},
                                        sphere=sphere, verbose=False)
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, meg=False, eeg=True,
                                    verbose=False)
    G = fwd["sol"]["data"]                                   # (channels, 3 * sources)
    return np.stack([G[:, 3 * i:3 * i + 3] @ n for i, n in enumerate(nn)], axis=1)


def source_signals(rng, label: int, effect: float, invert: bool) -> np.ndarray:
    from neurodsp.sim import sim_combined
    n = SECONDS * RATE
    iaf = rng.normal(10.0, 1.0)
    exponent = rng.normal(-1.8, 0.2)
    sign = -1.0 if invert else 1.0
    out = []
    for name in SOURCES:
        # resting-EEG-like proportions: 1/f dominates, alpha strongest over the back
        theta, beta = 0.4, 0.25
        alpha = 0.3 if name in FRONTAL else 3.0 if name in ("parietal", "occipital") else 0.8
        if name in FRONTAL and label == 1:
            theta *= 1.0 + sign * effect
            beta *= max(0.05, 1.0 - sign * effect / 2)
        comps = {"sim_powerlaw": {"exponent": exponent, "f_range": (1, None)},
                 "sim_bursty_oscillation": [{"freq": rng.normal(6.0, 0.5)},
                                            {"freq": iaf}, {"freq": rng.normal(20.0, 2.0)}]}
        x = sim_combined(SECONDS, RATE, comps, component_variances=[1.0, theta, alpha, beta])
        out.append(x[:n] * rng.lognormal(0.0, 0.2))           # per-subject source gain
    return np.stack(out)                                     # (sources, samples)


def device(montage: str, shift: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    c = len(MONTAGES[montage])
    return {"montage": montage, "gain": rng.lognormal(0.0, 0.3 * shift, c),
            "noise": 0.05 + 0.3 * shift, "scale": float(np.exp(shift * rng.normal(0, 0.7)))}


def subject_bag(name, label, dev, effect, invert, seed) -> two_phase.BagTensor:
    rng = np.random.default_rng(seed)
    x = lead_field(dev["montage"]) @ source_signals(rng, label, effect, invert)
    x = x / (np.std(x) + 1e-12)
    x = dev["scale"] * (dev["gain"][:, None] * x + dev["noise"] * rng.standard_normal(x.shape))
    x = x - x.mean(0, keepdims=True)                         # common average reference
    starts = range(0, x.shape[1] - WINDOW + 1, WINDOW // 2)
    w = np.stack([x[:, s:s + WINDOW] for s in starts])[:WINDOWS]
    return two_phase.BagTensor(name, dev["montage"], int(label),
                               torch.from_numpy(np.ascontiguousarray(w)).float())


def domain(tag, n_subjects, dev, effect, invert=False, seed=0) -> list[two_phase.BagTensor]:
    """`n_subjects` subjects, balanced classes, of one simulated domain."""
    return [subject_bag(f"{tag}-S{i:03d}", i % 2, dev, effect, invert, seed * 100003 + i)
            for i in range(n_subjects)]
