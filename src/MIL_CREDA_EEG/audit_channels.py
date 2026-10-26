"""Channel audit: MARK, never act.

Experiments plan v02, Sec. 2.4. Per record, a channel is defective when it is

1. absent although the rest of its cohort carries it;
2. flat;
3. saturated;
4. of a variance out of range against its peers;
5. without spectral content in the declared band.

**This module marks and reports. It does not interpolate, drop a channel or
exclude a record**: what to do with a defective channel is decided later, on
three statistics only the files answer (Sec. 10.1), before any model is
trained and never after seeing performance.

**Which criteria are marked and which are only measured.** The first three
have a definition that needs no number: a channel either is in the header or
is not, has zero variance or does not, touches the device's rail or does not.
Criteria 4 and 5 need a cut the plan has not instantiated. For them the module
reports a robust z-score against the record's own channels -- log standard
deviation for (4), the fraction of in-band power above 13 Hz for (5) -- and
marks nothing. The distribution is the deliverable; the cut is the operator's.

Criterion 5 is measured as the high-band fraction because of the reason the
plan gives for it: a badly seated electrode can carry plenty of low-frequency
drift, so it is neither flat nor saturated and its total variance looks normal,
while above the low part of the band it is noise floor. The band is the one
Sec. 2.7 declares (0.2-45 Hz); 13 Hz is where the beta half of the clinical
theta/beta marker starts.

**Whole record or segments.** For flatness and saturation the fraction of
four-second windows affected is reported per channel, because attention weighs
windows and not channels: only a failure by stretches leaves clean windows to
prefer (Sec. 10.1).

Two more measurements belong here because Sec. 2.1 sent the columns to audit
instead of discard: whether the mastoid reference reads near zero against the
scalp (then the channels are already referenced to it), and whether the
bipolar externals carry ocular activity (then they become the best tool of
Sec. 2.8).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v02.md",
    "sections": ["2.4"],
    "experiments": ["2.4"],
    "invariants": [
        "a_flat_channel_is_marked_flat_and_nothing_is_removed",
        "a_channel_at_the_device_rail_is_marked_saturated_with_its_windows",
        "variance_and_spectral_outliers_are_measured_not_marked",
        "a_channel_missing_from_its_cohort_is_reported_by_name",
        "a_reference_already_subtracted_reads_near_zero_against_the_scalp",
        "an_ocular_external_correlates_with_frontopolar_more_than_occipital",
    ],
}

#: Band declared by Sec. 2.7, and the start of its high (beta) part.
BAND = (0.2, 45.0)
HIGH_BAND = (13.0, 45.0)
#: Ocular activity lives at the bottom of the band (blinks, saccades).
OCULAR_BAND = (0.2, 4.0)
WINDOW_S = 4.0

#: Device rails OBSERVED in the release: the values a clipped sample takes.
#: Emotiv's 14-bit range sits at [0, 8401.922852] uV; CGX writes a railed or
#: disconnected input as -833333.3125 uV (seen on every ExG column).
RAILS = {"EMOTIV": (0.0, 8401.922852), "CGX": (-833333.0, 833333.0)}

#: CGX channels used as frontopolar and occipital references for the
#: ocular measurement of the bipolar externals.
FRONTOPOLAR = ("Fp1", "Fp2", "Fpz", "AF7", "AF8")
OCCIPITAL = ("O1", "O2", "Oz", "PO7", "PO8")


def _robust_z(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if finite.sum() < 3:
        return out
    median = np.median(values[finite])
    mad = 1.4826 * np.median(np.abs(values[finite] - median))
    if mad > 0:
        out[finite] = (values[finite] - median) / mad
    return out


def _band_power(x: np.ndarray, rate: float, nperseg: int,
                bands: tuple[tuple[float, float], ...]) -> list[float]:
    freqs, power = welch(x - np.mean(x), fs=rate, nperseg=min(len(x), nperseg))
    return [float(power[(freqs >= lo) & (freqs <= hi)].sum()) for lo, hi in bands]


def bandpass(x: np.ndarray, rate: float, band: tuple[float, float] = BAND) -> np.ndarray:
    """Zero-phase band-pass, for MEASUREMENT only; the record is not modified."""
    high = min(band[1], 0.45 * rate)
    sos = butter(4, [band[0], high], btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def channel_metrics(signal: np.ndarray, channels: list[str], rate: float,
                    rails: tuple[float, float], window_s: float = WINDOW_S) -> dict:
    """Per channel: flatness, saturation, variance and spectral position vs peers."""
    signal = np.asarray(signal, dtype=float)
    size = int(round(rate * window_s))
    n_windows = len(signal) // size
    out: dict = {}
    log_std = np.full(len(channels), np.nan)
    high = np.full(len(channels), np.nan)
    for j, name in enumerate(channels):
        x = signal[:, j]
        finite = np.isfinite(x)
        if not finite.any():
            out[name] = {"all_missing": True, "std": None, "flat": False,
                         "saturated_fraction": None, "flat_window_fraction": None,
                         "saturated_window_fraction": None, "high_band_fraction": None}
            continue
        x = np.where(finite, x, np.nanmedian(x))
        std = float(np.std(x))
        saturated = (x <= rails[0]) | (x >= rails[1])
        if n_windows:
            windows = x[: n_windows * size].reshape(n_windows, size)
            flat_w = float(np.mean(windows.std(axis=1) == 0))
            sat_w = float(np.mean(saturated[: n_windows * size]
                                  .reshape(n_windows, size).any(axis=1)))
        else:
            flat_w = sat_w = None
        fraction = None
        if std > 0:
            in_band, upper = _band_power(x, rate, size, (BAND, HIGH_BAND))
            fraction = upper / in_band if in_band > 0 else None
            log_std[j] = np.log10(std)
            high[j] = fraction if fraction is not None else np.nan
        out[name] = {"all_missing": False, "std": std, "flat": std == 0,
                     "saturated_fraction": float(saturated.mean()),
                     "flat_window_fraction": flat_w,
                     "saturated_window_fraction": sat_w,
                     "high_band_fraction": fraction,
                     "missing_fraction": float(1 - finite.mean())}
    for j, z in enumerate(_robust_z(log_std)):
        out[channels[j]]["log_std_robust_z"] = float(z) if np.isfinite(z) else None
    for j, z in enumerate(_robust_z(high)):
        out[channels[j]]["high_band_robust_z"] = float(z) if np.isfinite(z) else None
    return out


def mark_channels(metrics: dict) -> dict[str, list[str]]:
    """The criteria with a definition and no cut: missing data, flat, saturated."""
    marks = {}
    for name, m in metrics.items():
        found = []
        if m.get("all_missing"):
            found.append("all_missing")
        if m.get("flat"):
            found.append("flat")
        if m.get("saturated_fraction"):
            found.append("saturated")
        marks[name] = found
    return marks


def missing_vs_cohort(channel_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    """Channels the rest of the cohort carries and this record does not."""
    cohort = set().union(*map(set, channel_sets.values())) if channel_sets else set()
    return {rid: sorted(cohort - set(chs)) for rid, chs in channel_sets.items()}


def same_montage(channels_by_model: dict[str, list[str]]) -> bool:
    """Every model lists the same channels in the same order."""
    lists = [tuple(chs) for chs in channels_by_model.values()]
    return len(set(lists)) <= 1


def reference_metrics(reference: np.ndarray, scalp: np.ndarray, rate: float) -> dict:
    """Does the mastoid column read near zero against the scalp channels?"""
    reference = np.asarray(reference, dtype=float)
    scalp = np.asarray(scalp, dtype=float)
    ref_band = bandpass(reference, rate)
    scalp_band = bandpass(scalp, rate)
    scalp_std = float(np.median(scalp_band.std(axis=0)))
    ref_std = float(ref_band.std())
    mean_scalp = scalp_band.mean(axis=1)
    corr = (float(np.corrcoef(ref_band, mean_scalp)[0, 1])
            if ref_std > 0 and mean_scalp.std() > 0 else None)
    return {"raw_std": float(reference.std()),
            "raw_median_abs": float(np.median(np.abs(reference))),
            "band_std": ref_std, "scalp_median_band_std": scalp_std,
            "band_std_ratio_to_scalp": ref_std / scalp_std if scalp_std > 0 else None,
            "band_corr_with_scalp_mean": corr}


def bipolar_metrics(external: np.ndarray, frontal: np.ndarray, occipital: np.ndarray,
                    rate: float, rail: float = 833333.0) -> dict:
    """Ocular signature of a bipolar external: low-band correlation, front vs back."""
    external = np.asarray(external, dtype=float)
    constant = bool(np.nanstd(external) == 0)
    report = {"constant": constant,
              "at_rail_fraction": float(np.mean(np.abs(external) >= rail)),
              "raw_std": float(np.nanstd(external)),
              "frontal_abs_corr": None, "occipital_abs_corr": None,
              "frontal_minus_occipital_corr": None}
    if constant:
        return report
    ext = bandpass(external, rate, OCULAR_BAND)
    front = bandpass(np.asarray(frontal, dtype=float), rate, OCULAR_BAND)
    back = bandpass(np.asarray(occipital, dtype=float), rate, OCULAR_BAND)

    def mean_abs_corr(block: np.ndarray) -> float:
        return float(np.mean([abs(np.corrcoef(ext, block[:, k])[0, 1])
                              for k in range(block.shape[1])]))

    report["frontal_abs_corr"] = mean_abs_corr(front)
    report["occipital_abs_corr"] = mean_abs_corr(back)
    report["frontal_minus_occipital_corr"] = (report["frontal_abs_corr"]
                                              - report["occipital_abs_corr"])
    return report
