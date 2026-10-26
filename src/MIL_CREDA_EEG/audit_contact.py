"""Contact-quality audit, asymmetric across devices by necessity.

Experiments plan v02, Sec. 2.5. Emotiv writes quality indicators per channel
and per session; CGX writes none, so its criterion has to come from the signal.

**Emotiv.** The defect is falling below the acquisition threshold the data
paper itself declares: "an acceptable threshold being no lower than 90%" of
signal quality. `EQ.OVERALL` is that signal-quality percentage; `CQ.Overall`,
the contact percentage the setup turned green before recording, is measured
against the same number so both readings are visible. The threshold is the
paper's; how a record-level verdict aggregates windows is NOT declared, so this
module reports the median and the fraction of four-second windows whose median
falls below, and turns neither into an exclusion.

**CGX.** No column and no declared threshold exist. The signal-derived
measurement is mains pick-up: dry electrodes with poor contact have high
impedance and pick up more 50 Hz (the acquisition country's mains, Sec. 2.7).
The ratio of 49-51 Hz power to in-band power is reported per channel, with no
cut.

The audit of this audit -- whether its marks correlate with the label, because
a child with the disorder plausibly moves and touches the sensors more -- is
run over these distributions in `defect_log`, measured and declared, not fixed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v02.md",
    "sections": ["2.5"],
    "experiments": ["2.5"],
    "invariants": [
        "emotiv_quality_below_the_declared_ninety_is_counted_per_window",
        "cgx_contact_has_no_quality_column_so_it_is_read_from_the_signal",
    ],
}

#: Data paper, Technical Validation: signal quality "no lower than 90%".
EMOTIV_QUALITY_THRESHOLD = 90.0
WINDOW_S = 4.0
LINE_HZ = 50.0
BAND = (0.2, 45.0)


def emotiv_quality(timestamps: np.ndarray, quality: dict[str, np.ndarray],
                   threshold: float = EMOTIV_QUALITY_THRESHOLD,
                   window_s: float = WINDOW_S,
                   channels: tuple[str, ...] = ()) -> dict:
    """Median and windowed fraction below the threshold, overall and per channel."""
    timestamps = np.asarray(timestamps, dtype=float)
    windows = np.floor((timestamps - timestamps[0]) / window_s).astype(np.int64)
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
    n_windows = int(np.floor((timestamps[-1] - timestamps[0] + step) / window_s + 1e-9))
    report: dict = {"threshold": float(threshold), "window_s": float(window_s)}
    for key in ("EQ.OVERALL", "CQ.Overall"):
        values = quality.get(key)
        if values is None or not np.isfinite(values).any():
            report[key] = None
            continue
        values = np.asarray(values, dtype=float)
        ok = np.isfinite(values) & (windows < max(n_windows, 1))
        per_window = {}
        for w, v in zip(windows[ok], values[ok]):
            per_window.setdefault(int(w), []).append(v)
        medians = np.array([np.median(v) for v in per_window.values()])
        finite = values[np.isfinite(values)]
        report[key] = {
            "median": float(np.median(finite)),
            "sample_fraction_below": float(np.mean(finite < threshold)),
            "windows_with_reading": int(len(medians)),
            "window_fraction_below": (float(np.mean(medians < threshold))
                                      if len(medians) else None),
        }
    report["per_channel_contact_median"] = {
        ch: float(np.nanmedian(quality[f"CQ.{ch}"])) for ch in channels
        if f"CQ.{ch}" in quality and np.isfinite(quality[f"CQ.{ch}"]).any()}
    report["per_channel_signal_quality_median"] = {
        ch: float(np.nanmedian(quality[f"EQ.{ch}"])) for ch in channels
        if f"EQ.{ch}" in quality and np.isfinite(quality[f"EQ.{ch}"]).any()}
    return report


def line_noise_ratio(signal: np.ndarray, rate: float, line: float = LINE_HZ,
                     window_s: float = WINDOW_S) -> np.ndarray:
    """Per channel: power within 1 Hz of the mains line over in-band power."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal, axis=0)
    signal = np.nan_to_num(signal)
    freqs, power = welch(signal, fs=rate, nperseg=min(len(signal), int(rate * window_s)),
                         axis=0)
    line_power = power[(freqs >= line - 1) & (freqs <= line + 1)].sum(axis=0)
    in_band = power[(freqs >= BAND[0]) & (freqs <= BAND[1])].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(in_band > 0, line_power / in_band, np.nan)
