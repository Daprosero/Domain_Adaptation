"""Realistic simulated EEG for testing when MIL-CREDA's phase 2 can help.

SEREEGA's principle (Krol et al., J Neurosci Methods 2018) in Python: dipolar cortical
sources in a head model, each with its own activity, projected to the electrodes by a
lead field. The head model is MNE-Python's spherical one (volume conduction, no MRI
download); each source signal is NeuroDSP's aperiodic 1/f background plus bursty theta,
alpha and beta oscillations (Cole et al., JOSS 2019).

What can be controlled, and why:
* class effect (ADHD-like): frontal theta up, frontal beta down, by `effect` (the internal
  knob). Experiments should state the effect as a target Cohen's d of the subject-level
  theta/beta ratio (TBR) and translate it with `effect_for_d`: the literature reports
  d = 0.62 (ages 6-18) and 0.75 (6-13) (Arns, Conners & Kraemer 2013, J Atten Disord),
  itself considered an overestimate, while `effect=1.0` gives d of about 3.
* subject variability: individual alpha frequency, aperiodic exponent, source gains;
  `identity_seed` lets the same subject (same identity and class) appear in two domains;
* device (domain): montage (19-channel 10-20 or a 14-channel EMOTIV-like set), per-channel
  gain, sensor noise and overall amplitude scale, grown by `shift`;
* concept shift: `invert` flips the sign of the class effect in a domain;
* artifacts: eye blinks (frontal) and muscle bursts (temporal) at `artifact_rate` per
  minute, independent of class and domain unless `artifact_class_corr` is set;
* event windows: `event_fraction` concentrates the class effect in a known subset of
  windows.
Per-window masks (artifact, artifact type, event) and the subject identity are kept in
the side registry `SIM_MASKS`, keyed by bag subject name (`masks_of`).
A subject is one bag: 4 s windows at 50% overlap at 128 Hz (60 windows), the instances.

Measured effect -> d mapping (`effect_for_d` defaults: paired calibration of 1000 subjects,
seed 0, clean 10-20 device at shift 0, Fz TBR, no artifacts, no events):
    effect 0.05 -> d 0.16, 0.10 -> d 0.33, 0.15 -> d 0.49, 0.20 -> d 0.64,
    0.30 -> d 0.95, 0.50 -> d 1.51, 1.00 -> d 2.62;
    d 0.30 -> effect 0.0918, d 0.60 -> 0.1864, d 0.62 -> 0.1929, d 0.75 -> 0.2351;
    with event_fraction 0.2: d 0.30 -> 0.0895, d 0.60 -> 0.1782.
An independent 1000-subject sample labelled as `domain` labels it measured d 0.32 and 0.62
at the effects for 0.3 and 0.6. Sensor noise lowers the d a device records: the EMOTIV-like
target at shift 1 records less than the clean source at the same effect.

Events hold fixed the time-averaged frontal theta and beta power multipliers of class 1
(`event_multipliers`: the multiplier outside event time keeps EVENT_OUTSIDE of the change,
inside it carries the rest), so the recording's mean band powers match the uniform effect;
the subject-level d still moves slightly (a mean of window ratios is not a ratio of mean
powers), which `effect_for_d(..., event_fraction=f)` recalibrates.
"""
from __future__ import annotations

import contextlib
import functools

import numpy as np
import torch
from scipy.signal import butter, get_window, sosfiltfilt, welch

from MIL_CREDA_EEG import two_phase

RATE, WINDOW, WINDOWS = 128, 512, 60
SECONDS = (WINDOWS + 1) * WINDOW // 2 // RATE + 1          # 60 windows at a 2 s step
STEP = WINDOW // 2
SPAN = (WINDOWS - 1) * STEP + WINDOW                       # samples the windows cover (122 s)
TEN_TWENTY = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
              "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz")
EMOTIV = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
MONTAGES = {"1020": TEN_TWENTY, "emotiv": EMOTIV}
#: source name -> (azimuth deg, elevation deg) on a 6 cm shell; frontal ones carry the class
SOURCES = {"frontal-L": (120, 35), "frontal-R": (60, 35), "central-L": (165, 55),
           "central-R": (15, 55), "parietal": (270, 50), "occipital": (270, 15)}
FRONTAL = ("frontal-L", "frontal-R")
#: TBR bands (Hz, half-open) and the fallback frontal channels when the asked one is absent
THETA, BETA = (4.0, 8.0), (13.0, 30.0)
FRONTAL_FALLBACK = ("F3", "F4")
#: artifacts: blink share, blink duration (s) and peak (x EEG std), muscle duration and std
BLINK_SHARE = 0.5
BLINK_SECONDS, BLINK_PEAK = (0.2, 0.5), (6.0, 10.0)
BLINK_FALLOFF = 0.045                                      # m, exp falloff from Fpz
MUSCLE_SECONDS, MUSCLE_STD, MUSCLE_BAND = (0.5, 2.0), (2.0, 4.0), (20.0, 45.0)
MUSCLE_CHANNELS = {"L": {"T7": 1.0, "FC5": 0.8, "F7": 0.7},
                   "R": {"T8": 1.0, "FC6": 0.8, "F8": 0.7}}
MUSCLE_BOTH_SIDES = 0.2
RATE_SHAPE = 4.0                     # per-subject artifact rate ~ Gamma(4, rate / 4): mean = rate
#: events: share of the class effect kept outside event time, and the edge ramp (s)
EVENT_OUTSIDE = 0.25
EVENT_RAMP = 0.25
#: side registry, bag subject name -> masks and identity (latest generation wins)
SIM_MASKS: dict[str, dict] = {}


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


@functools.lru_cache(maxsize=4)
def blink_weights(montage: str) -> np.ndarray:
    """Per-channel blink weight: exponential falloff with distance from Fpz, 1 at Fp1/Fp2."""
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    dist = lambda ch: float(np.linalg.norm(pos[ch] - pos["Fpz"]))
    ref = dist("Fp1")
    return np.array([min(1.0, np.exp(-(dist(c) - ref) / BLINK_FALLOFF)) for c in MONTAGES[montage]])


@contextlib.contextmanager
def _global_seed(seed: int):
    """NeuroDSP draws from NumPy's global generator: seed it per subject, then restore it."""
    state = np.random.get_state()
    np.random.seed(int(seed) % 2 ** 32)
    try:
        yield
    finally:
        np.random.set_state(state)


# ------------------------------------------------------------------ subjects


def subject_identity(rng) -> dict:
    """What makes a subject: IAF, aperiodic exponent, per-source theta/beta peak and gain.
    Drawn in the order the original generator drew them."""
    iaf, exponent = rng.normal(10.0, 1.0), rng.normal(-1.8, 0.2)
    theta, beta, gain = [], [], []
    for _ in SOURCES:
        theta.append(rng.normal(6.0, 0.5))
        beta.append(rng.normal(20.0, 2.0))
        gain.append(rng.lognormal(0.0, 0.2))
    return {"iaf": float(iaf), "exponent": float(exponent), "thetaFreq": [float(v) for v in theta],
            "betaFreq": [float(v) for v in beta], "gain": [float(v) for v in gain]}


def source_components(ident: dict) -> np.ndarray:
    """(sources, 4, samples) unit-variance components (1/f, theta, alpha, beta), drawn from
    the global generator exactly as `sim_combined` draws them."""
    from neurodsp.sim import sim_bursty_oscillation, sim_powerlaw
    n = SECONDS * RATE
    out = []
    for j, _ in enumerate(SOURCES):
        out.append(np.stack([
            sim_powerlaw(SECONDS, RATE, exponent=ident["exponent"], f_range=(1, None), variance=1.0)[:n],
            sim_bursty_oscillation(SECONDS, RATE, freq=ident["thetaFreq"][j], variance=1.0)[:n],
            sim_bursty_oscillation(SECONDS, RATE, freq=ident["iaf"], variance=1.0)[:n],
            sim_bursty_oscillation(SECONDS, RATE, freq=ident["betaFreq"][j], variance=1.0)[:n]]))
    return np.stack(out)


def _base_variances(name: str) -> np.ndarray:
    # resting-EEG-like proportions: 1/f dominates, alpha strongest over the back
    alpha = 0.3 if name in FRONTAL else 3.0 if name in ("parietal", "occipital") else 0.8
    return np.array([1.0, 0.4, alpha, 0.25])


def class_multipliers(effect: float, invert: bool) -> tuple[float, float]:
    """Frontal (theta, beta) variance multipliers of class 1."""
    sign = -1.0 if invert else 1.0
    return max(0.05, 1.0 + sign * effect), max(0.05, 1.0 - sign * effect / 2)


def event_multipliers(m: float, p: float, outside: float = EVENT_OUTSIDE) -> tuple[float, float]:
    """(inside, outside) multipliers that keep the time-averaged multiplier at `m` when a
    share `p` of the time is event time and `outside` of the change is kept outside."""
    out = 1.0 + outside * (m - 1.0)
    inside = (m - (1.0 - p) * out) / p if p > 0 else m
    return max(0.05, inside), out


def event_cover(rng, fraction: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Pick round(fraction * WINDOWS) windows; event time is the union of their spans.
    Returns the smoothed time mask r(t) in [0, 1], each window's covered share, and the
    picked windows."""
    n = SECONDS * RATE
    k = int(round(fraction * WINDOWS))
    picked = sorted(int(i) for i in rng.choice(WINDOWS, size=k, replace=False)) if k else []
    hard = np.zeros(n)
    for i in picked:
        hard[i * STEP:i * STEP + WINDOW] = 1.0
    cover = np.array([hard[i * STEP:i * STEP + WINDOW].mean() for i in range(WINDOWS)])
    ramp = max(1, int(EVENT_RAMP * RATE))
    kern = np.hanning(2 * ramp + 1)
    soft = np.clip(np.convolve(hard, kern / kern.sum(), mode="same"), 0.0, 1.0) if k else hard
    return soft, cover, picked


def subject_parts(label, dev, seed, *, identity_seed=None, index=0, event_fraction=0.0) -> dict:
    """Everything one subject's recording is made of, before the class effect is applied.
    `seed` drives the recording (oscillation bursts, sensor noise, events); the identity
    comes from `identity_seed` when given, else from the recording's own generator."""
    rng = np.random.default_rng(seed)
    ident = (subject_identity(np.random.default_rng([int(identity_seed), int(index), 0x1D]))
             if identity_seed is not None else subject_identity(rng))
    with _global_seed(seed):
        comps = source_components(ident)
    noise = rng.standard_normal((len(MONTAGES[dev["montage"]]), comps.shape[-1]))
    erng = np.random.default_rng([int(seed), 0xE7])
    soft, cover, picked = event_cover(erng, event_fraction)
    return {"label": int(label), "identity": ident, "components": comps, "noise": noise,
            "eventTime": soft, "eventCover": cover, "eventWindows": picked}


def amplitudes(label, effect, invert, event_time, event_fraction,
               outside=EVENT_OUTSIDE) -> np.ndarray:
    """(sources, 4, samples or 1) amplitude of each unit-variance component."""
    a = np.stack([np.sqrt(_base_variances(s)) for s in SOURCES])[:, :, None]
    if label != 1:
        return a
    mt, mb = class_multipliers(effect, invert)
    a = np.broadcast_to(a, (len(SOURCES), 4, event_time.shape[0] if event_fraction > 0 else 1)).copy()
    p = float(event_time[:SPAN].mean()) if event_fraction > 0 else 0.0
    for j, s in enumerate(SOURCES):
        if s not in FRONTAL:
            continue
        for k, m in ((1, mt), (3, mb)):
            base = np.sqrt(_base_variances(s)[k])
            if p > 0:
                inside, out = event_multipliers(m, p, outside)
                a[j, k] = base * (np.sqrt(out) + (np.sqrt(inside) - np.sqrt(out)) * event_time)
            else:
                a[j, k] = base * np.sqrt(m)
    return a


def source_signals(parts, effect: float, invert: bool, event_fraction=0.0,
                   outside=EVENT_OUTSIDE) -> np.ndarray:
    a = amplitudes(parts["label"], effect, invert, parts["eventTime"], event_fraction, outside)
    gain = np.asarray(parts["identity"]["gain"])[:, None]
    return (a * parts["components"]).sum(1) * gain           # (sources, samples)


def device(montage: str, shift: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    c = len(MONTAGES[montage])
    return {"montage": montage, "gain": rng.lognormal(0.0, 0.3 * shift, c),
            "noise": 0.05 + 0.3 * shift, "scale": float(np.exp(shift * rng.normal(0, 0.7)))}


# ------------------------------------------------------------------ artifacts


def artifact_rate_of(rng, label, rate, class_corr) -> float:
    """Per-subject artifacts per minute: Gamma with mean `rate`, the same for both classes
    unless `class_corr` (in [0, 1]) raises class 1 to rate (1 + c) and lowers class 0."""
    mean = rate * (1.0 + class_corr if label == 1 else 1.0 - class_corr)
    return float(rng.gamma(RATE_SHAPE, mean / RATE_SHAPE)) if mean > 0 else 0.0


def artifacts(rng, montage, rate_per_minute) -> tuple[np.ndarray, list[dict]]:
    """(channels, samples) artifact signal in EEG-std units and the inserted events."""
    names = list(MONTAGES[montage])
    n = SECONDS * RATE
    x = np.zeros((len(names), n))
    count = int(rng.poisson(rate_per_minute * SPAN / RATE / 60.0))
    sos = butter(4, MUSCLE_BAND, btype="bandpass", fs=RATE, output="sos")
    events = []
    for _ in range(count):
        if rng.random() < BLINK_SHARE:
            d = int(rng.uniform(*BLINK_SECONDS) * RATE)
            start = int(rng.integers(0, SPAN - d))
            shape = np.sin(np.pi * np.arange(d) / max(d - 1, 1))       # half-sine
            x[:, start:start + d] += rng.uniform(*BLINK_PEAK) * blink_weights(montage)[:, None] * shape
            events.append({"type": "blink", "start": start, "stop": start + d})
        else:
            d = int(rng.uniform(*MUSCLE_SECONDS) * RATE)
            start = int(rng.integers(0, SPAN - d))
            sides = ("L", "R") if rng.random() < MUSCLE_BOTH_SIDES else (("L", "R")[int(rng.integers(2))],)
            env = np.hanning(d)
            amp = rng.uniform(*MUSCLE_STD)
            for side in sides:
                for ch, w in MUSCLE_CHANNELS[side].items():
                    if ch in names:
                        burst = sosfiltfilt(sos, rng.standard_normal(d + 2 * RATE))[RATE:RATE + d]
                        burst = burst / (burst.std() + 1e-12)
                        x[names.index(ch), start:start + d] += amp * w * env * burst
            events.append({"type": "muscle", "start": start, "stop": start + d})
    return x, events


def window_masks(events) -> tuple[list[bool], list[str]]:
    """Per window: does it overlap any artifact, and of which type(s)."""
    flag, kind = [], []
    for i in range(WINDOWS):
        a, b = i * STEP, i * STEP + WINDOW
        hit = sorted({e["type"] for e in events if e["start"] < b and e["stop"] > a})
        flag.append(bool(hit))
        kind.append("+".join(hit) if hit else "none")
    return flag, kind


# ------------------------------------------------------------------ bags


def _windows(x) -> np.ndarray:
    starts = range(0, x.shape[1] - WINDOW + 1, STEP)
    return np.stack([x[:, s:s + WINDOW] for s in starts])[:WINDOWS]


def subject_bag(name, label, dev, effect, invert, seed, *, identity_seed=None, index=0,
                artifact_rate=0.0, artifact_class_corr=0.0, event_fraction=0.0,
                event_outside=EVENT_OUTSIDE) -> two_phase.BagTensor:
    parts = subject_parts(label, dev, seed, identity_seed=identity_seed, index=index,
                          event_fraction=event_fraction)
    x = lead_field(dev["montage"]) @ source_signals(parts, effect, invert, event_fraction, event_outside)
    x = x / (np.std(x) + 1e-12)
    arng = np.random.default_rng([int(seed), 0xA7])          # artifacts: never the label's draws
    rate = artifact_rate_of(arng, label, artifact_rate, artifact_class_corr)
    art, events = artifacts(arng, dev["montage"], rate) if rate > 0 else (0.0, [])
    x = x + art
    x = dev["scale"] * (dev["gain"][:, None] * x + dev["noise"] * parts["noise"])
    x = x - x.mean(0, keepdims=True)                         # common average reference
    w = _windows(x)
    flag, kind = window_masks(events)
    cover = parts["eventCover"]
    SIM_MASKS[name] = {"label": int(label), "montage": dev["montage"], "identity": parts["identity"],
                       "artifact": flag, "artifactType": kind, "artifactEvents": events,
                       "artifactRate": rate,
                       "event": [bool(c >= 1.0) for c in cover] if event_fraction > 0 else [False] * WINDOWS,
                       "eventCover": [round(float(c), 3) for c in cover],
                       "eventWindows": parts["eventWindows"]}
    return two_phase.BagTensor(name, dev["montage"], int(label),
                               torch.from_numpy(np.ascontiguousarray(w)).float())


def domain(tag, n_subjects, dev, effect, invert=False, seed=0, *, identity_seed=None,
           artifact_rate=0.0, artifact_class_corr=0.0, event_fraction=0.0,
           event_outside=EVENT_OUTSIDE) -> list[two_phase.BagTensor]:
    """`n_subjects` subjects, balanced classes, of one simulated domain. With the same
    `identity_seed`, two domains hold the same subjects (identity and class) under their own
    devices and recordings; subject i is `<tag>-S<i>` in both, so the suffix pairs them."""
    return [subject_bag(f"{tag}-S{i:03d}", i % 2, dev, effect, invert, seed * 100003 + i,
                        identity_seed=identity_seed, index=i, artifact_rate=artifact_rate,
                        artifact_class_corr=artifact_class_corr, event_fraction=event_fraction,
                        event_outside=event_outside)
            for i in range(n_subjects)]


def masks_of(bags) -> dict:
    """Registry entries of `bags` (subject name -> per-window masks), JSON-ready."""
    keys = ("artifact", "artifactType", "event", "eventCover", "artifactRate")
    return {b.subject: {k: SIM_MASKS[b.subject][k] for k in keys} for b in bags if b.subject in SIM_MASKS}


# ------------------------------------------------------------------ TBR effect size


def tbr_channels(montage: str, channel: str = "Fz") -> list[int]:
    names = list(MONTAGES[montage])
    if channel in names:
        return [names.index(channel)]
    return [names.index(c) for c in FRONTAL_FALLBACK if c in names]


def _band_masks():
    f = np.fft.rfftfreq(WINDOW // 2, 1.0 / RATE)
    return (f >= THETA[0]) & (f < THETA[1]), (f >= BETA[0]) & (f < BETA[1])


def subject_tbr(windows, idx) -> float:
    """Mean over windows of theta power / beta power (Welch, 2 s Hann segments), the powers
    averaged over the channels `idx`."""
    th, be = _band_masks()
    _, p = welch(np.asarray(windows, dtype=np.float64)[:, idx], fs=RATE, nperseg=WINDOW // 2, axis=-1)
    p = p.mean(1)                                            # (windows, freqs)
    return float(np.mean(p[:, th].sum(-1) / p[:, be].sum(-1)))


def cohens_d(values, labels) -> float:
    v, y = np.asarray(values, float), np.asarray(labels)
    a, b = v[y == 1], v[y == 0]
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


def tbr_effect_size(bags, channel: str = "Fz") -> float:
    """Cohen's d (class 1 minus class 0, pooled sd) of the per-subject TBR at `channel`, or
    the F3/F4 average when the montage has no such channel (EMOTIV)."""
    tbr = [subject_tbr(b.windows.numpy(), tbr_channels(b.condition, channel)) for b in bags]
    return cohens_d(tbr, [int(b.label) for b in bags])


class _Calibration:
    """The TBR of a fixed calibration sample as an exact function of `effect`: the recording
    is linear in the component amplitudes, so each window's band power is a quadratic form
    c(effect)^T M c(effect) over a few fixed traces (common random numbers across effects).

    `d(effect, paired=False)` is the d of the sample as `domain` labels it (subject i has
    class i % 2), equal to `tbr_effect_size(domain(...))` on the same seed. `paired=True`
    scores every subject both as class 0 and as class 1 (same identity, same recording): the
    same population d, with far less sampling noise, and exactly 0 at effect 0."""

    def __init__(self, dev, n, seed, channel, event_fraction, outside):
        self.dev, self.fraction, self.outside = dev, event_fraction, outside
        L = lead_field(dev["montage"])
        C = L.shape[0]
        idx = tbr_channels(dev["montage"], channel)
        th, be = _band_masks()
        taper = get_window("hann", WINDOW // 2)
        fj = [list(SOURCES).index(s) for s in FRONTAL]
        self.subjects = []
        for i in range(n):
            label = i % 2
            parts = subject_parts(label, dev, seed * 100003 + i, index=i, event_fraction=event_fraction)
            gain = np.asarray(parts["identity"]["gain"])
            a0 = amplitudes(0, 0.0, False, parts["eventTime"], 0.0)[:, :, 0]
            comps = parts["components"]
            # fixed part: every component at its base amplitude except the frontal theta/beta
            fixed_src = (a0[:, :, None] * comps).sum(1)
            var = []                                         # (source j, band k, event part?)
            for j in fj:
                for k in (1, 3):
                    fixed_src[j] -= a0[j, k] * comps[j, k]
                    var.append((j, k, comps[j, k] * gain[j]))
                    if event_fraction > 0:
                        var.append((j, k, comps[j, k] * gain[j] * parts["eventTime"]))
            X0 = L @ (fixed_src * gain[:, None])
            U = np.stack([L[:, j] for j, _, _ in var])       # (traces, channels)
            T = np.stack([t for _, _, t in var])             # (traces, samples)
            n_s = X0.size
            stats = {"x0x0": float((X0 ** 2).sum()), "x0t": np.einsum("kc,cs,ks->k", U, X0, T),
                     "uu": U @ U.T, "tt": T @ T.T, "sx0": float(X0.sum()),
                     "su": U.sum(1), "st": T.sum(1), "n": n_s}
            g = self.dev["gain"]
            M = {}
            for ch in idx:
                car = np.full(C, -1.0 / C)                   # common average of channel ch
                car[ch] += 1.0
                h = g * car                                  # device gain acts before the noise
                r0 = h @ X0
                m = car @ parts["noise"]
                tr = (U @ h)[:, None] * T                     # (traces, samples) projected
                B = np.vstack([r0[None], tr, m[None]])[:, :SPAN]
                segs = []
                for w in range(WINDOWS):
                    seg = B[:, w * STEP:w * STEP + WINDOW]
                    for s0 in range(0, WINDOW - WINDOW // 2 + 1, WINDOW // 4):
                        z = seg[:, s0:s0 + WINDOW // 2]
                        segs.append(np.fft.rfft((z - z.mean(1, keepdims=True)) * taper, axis=-1))
                F = np.stack(segs).reshape(WINDOWS, -1, B.shape[0], th.size)  # (w, seg, trace, f)
                cross = lambda band: np.einsum("wsif,wsjf->wij", F[..., band], F[..., band].conj()).real
                M[ch] = (cross(th), cross(be))
            self.subjects.append({"label": label, "var": [(j, k) for j, k, _ in var], "stats": stats,
                                  "M": M, "parts_time": parts["eventTime"]})

    def coefficients(self, s, effect, label):
        """Amplitude of every varying trace for this subject at `effect` as class `label`."""
        out = []
        for j, k in s["var"][::2 if self.fraction > 0 else 1]:
            base = np.sqrt(_base_variances(list(SOURCES)[j])[k])
            m = class_multipliers(effect, False)[0 if k == 1 else 1] if label == 1 else 1.0
            if self.fraction > 0:
                p = float(s["parts_time"][:SPAN].mean())
                inside, out_m = event_multipliers(m, p, self.outside) if label == 1 else (1.0, 1.0)
                out += [base * np.sqrt(out_m), base * (np.sqrt(inside) - np.sqrt(out_m))]
            else:
                out.append(base * np.sqrt(m))
        return np.asarray(out)

    def tbr(self, s, effect, label) -> float:
        a, st = self.coefficients(s, effect, label), s["stats"]
        total = st["x0x0"] + 2 * a @ st["x0t"] + a @ (st["uu"] * st["tt"]) @ a
        mean = (st["sx0"] + a @ (st["su"] * st["st"])) / st["n"]
        sigma = np.sqrt(total / st["n"] - mean ** 2) + 1e-12
        c = np.concatenate([[1.0 / sigma], a / sigma, [self.dev["noise"]]])
        pt = sum(np.einsum("i,wij,j->w", c, Mt, c) for Mt, _ in s["M"].values())
        pb = sum(np.einsum("i,wij,j->w", c, Mb, c) for _, Mb in s["M"].values())
        return float(np.mean(pt / pb))

    def d(self, effect, paired=True) -> float:
        if not paired:
            return cohens_d([self.tbr(s, effect, s["label"]) for s in self.subjects],
                            [s["label"] for s in self.subjects])
        ones = [self.tbr(s, effect, 1) for s in self.subjects]
        zeros = [self.tbr(s, 0.0, 0) for s in self.subjects]
        return cohens_d(ones + zeros, [1] * len(ones) + [0] * len(zeros))


@functools.lru_cache(maxsize=16)
def _calibration(montage, shift, dev_seed, n, seed, channel, event_fraction, outside):
    return _Calibration(device(montage, shift, dev_seed), n, seed, channel, event_fraction, outside)


def effect_for_d(d_target: float, device_spec="1020", n: int = 1000, seed: int = 0, *,
                 shift: float = 0.0, device_seed: int = 0, channel: str = "Fz",
                 event_fraction: float = 0.0, event_outside: float = EVENT_OUTSIDE,
                 hi: float = 4.0, tol: float = 1e-4) -> float:
    """The `effect` whose subject-level TBR Cohen's d equals `d_target` on a calibration
    sample of `n` subjects (balanced, seed `seed`) recorded with `device(device_spec, shift,
    device_seed)`, no artifacts; bisection on the paired, common-random-number d(effect)
    (see `_Calibration`), which increases with effect and is 0 at effect 0."""
    cal = _calibration(device_spec, float(shift), int(device_seed), int(n), int(seed), channel,
                       float(event_fraction), float(event_outside))
    if d_target <= cal.d(0.0):
        return 0.0
    lo, top = 0.0, float(hi)
    if cal.d(top) < d_target:
        raise ValueError(f"d {d_target} not reached at effect {top} (d {cal.d(top):.3f})")
    while top - lo > tol:
        mid = (lo + top) / 2
        lo, top = (mid, top) if cal.d(mid) < d_target else (lo, mid)
    return round((lo + top) / 2, 5)


def calibration_d(effect: float, device_spec="1020", n: int = 1000, seed: int = 0, *,
                  paired: bool = True, **kw) -> float:
    """d(effect) on the calibration sample `effect_for_d` bisects (`paired=True`), or on the
    same sample labelled as `domain` labels it (`paired=False`)."""
    cal = _calibration(device_spec, float(kw.get("shift", 0.0)), int(kw.get("device_seed", 0)),
                       int(n), int(seed), kw.get("channel", "Fz"), float(kw.get("event_fraction", 0.0)),
                       float(kw.get("event_outside", EVENT_OUTSIDE)))
    return cal.d(float(effect), paired)
