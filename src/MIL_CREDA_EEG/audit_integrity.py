"""Sample integrity: holes in the stream and samples the device made up.

Experiments plan v56, Sec. 2.3. Both device families carry a sample counter,
so a lost sample is visible as a jump in it; Emotiv also flags, per sample,
whether it was interpolated. The plan declares three defects:

* a lost fraction above the declared threshold;
* an interpolated fraction above the declared threshold;
* a contiguous hole longer than one window.

**Only the third has a number in the plan.** The window is four seconds
(Sec. 3.1), so a hole longer than that is marked here. The two fraction
thresholds are not instantiated anywhere in v02, so this module reports the
fractions and marks nothing on them (Sec. 10.1): inventing a cut after seeing
the distribution is exactly what the plan forbids.

The defect is of a SEGMENT, not of a record. What a later step needs is which
windows a hole touches, so that is reported too, never a verdict on the record.

**The counter wraps.** Both devices count modulo a power of two (CGX and Emotiv
at 128 Hz: 0..127). A jump is read modulo that, so `127 -> 3` is three lost
samples, not a negative step. A hole of a whole counter period or more is
invisible to the counter, which is why the timestamps are read as well and the
longer of the two readings is the hole.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v56.md",
    "sections": ["2.3"],
    "experiments": ["2.3"],
    "invariants": [
        "lost_samples_are_counted_across_a_counter_wrap",
        "a_gap_longer_than_one_window_is_marked",
        "a_short_gap_is_measured_and_not_marked",
        "interpolated_fraction_and_the_windows_it_touches",
    ],
}

#: Instance window of the plan (Sec. 3.1), in seconds.
WINDOW_S = 4.0


def counter_modulus(counter: np.ndarray) -> int:
    """The counter's period: the smallest power of two above its largest value."""
    top = int(np.nanmax(counter))
    return 1 << max(top, 1).bit_length()


def lost_before_each_sample(counter: np.ndarray, modulus: int) -> np.ndarray:
    """Samples lost immediately before each sample (0 for the first)."""
    counter = np.asarray(counter, dtype=float)
    step = np.mod(np.diff(counter), modulus)
    lost = np.where(step > 0, step - 1, 0)
    return np.r_[0, lost].astype(np.int64)


def window_index(timestamps: np.ndarray, window_s: float = WINDOW_S) -> np.ndarray:
    return np.floor((timestamps - timestamps[0]) / window_s).astype(np.int64)


def complete_windows(timestamps: np.ndarray, rate: float,
                     window_s: float = WINDOW_S) -> int:
    span = float(timestamps[-1] - timestamps[0]) + 1.0 / rate
    return int(np.floor(span / window_s + 1e-9))


def audit_integrity(timestamps: np.ndarray, counter: np.ndarray | None,
                    interpolated: np.ndarray | None, rate: float,
                    window_s: float = WINDOW_S) -> dict:
    """Lost and interpolated samples, the longest hole, and the windows touched."""
    timestamps = np.asarray(timestamps, dtype=float)
    n = len(timestamps)
    windows = window_index(timestamps, window_s)
    report: dict = {"n_samples": int(n), "rate_used": float(rate),
                    "window_s": float(window_s),
                    "n_windows": complete_windows(timestamps, rate, window_s)}
    if counter is not None and np.isfinite(counter).any():
        modulus = counter_modulus(counter)
        lost = lost_before_each_sample(counter, modulus)
        duplicated = int(np.sum(np.mod(np.diff(counter), modulus) == 0))
        report.update({
            "counter_modulus": modulus, "n_lost": int(lost.sum()),
            "lost_fraction": float(lost.sum() / (n + lost.sum())),
            "n_lost_events": int(np.count_nonzero(lost)),
            "n_repeated_counter": duplicated,
            "longest_gap_samples": int(lost.max()) if n else 0,
            "windows_touched_by_loss": int(len(np.unique(windows[lost > 0]))),
        })
        counter_gap_s = report["longest_gap_samples"] / rate
    else:
        report.update({"counter_modulus": None, "n_lost": None, "lost_fraction": None,
                       "n_lost_events": None, "n_repeated_counter": None,
                       "longest_gap_samples": None, "windows_touched_by_loss": None})
        counter_gap_s = 0.0
    dt = np.diff(timestamps)
    timestamp_gap_s = float(max(dt.max() - 1.0 / rate, 0.0)) if n > 1 else 0.0
    report["longest_timestamp_step_s"] = float(dt.max()) if n > 1 else 0.0
    report["n_backward_timestamps"] = int(np.sum(dt < 0)) if n > 1 else 0
    report["longest_gap_s"] = float(max(counter_gap_s, timestamp_gap_s))
    report["gap_exceeds_window"] = bool(report["longest_gap_s"] > window_s)
    if interpolated is not None:
        flag = np.nan_to_num(np.asarray(interpolated, dtype=float)) > 0
        report.update({
            "n_interpolated": int(flag.sum()),
            "interpolated_fraction": float(flag.mean()) if n else 0.0,
            "windows_touched_by_interpolation": int(len(np.unique(windows[flag]))),
        })
    else:
        report.update({"n_interpolated": None, "interpolated_fraction": None,
                       "windows_touched_by_interpolation": None})
    return report
