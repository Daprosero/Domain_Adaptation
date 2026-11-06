"""Four-second instances, and the side data each one carries.

Experiments plan v55, Secs. 3.1, 2.3, 2.4, 2.5, 2.8 and 2.1. An instance is a
four-second window without overlap, cut from the signal already filtered and
resampled (Sec. 2.7): 512 samples at 128 Hz. The partial last window is
discarded, without a threshold (Sec. 3.1), and so is the first window of every
record, the filter transient (Sec. 2.7: five time constants of a 0.2 Hz
high-pass are about four seconds).

**Windows are blocks of samples, the audit's own convention.** Raw window `k`
holds raw samples `[k * R, (k + 1) * R)` with `R = 4 * rate` (2000 for CGX,
1024 for EPOC+ at 256 Hz, 512 at 128 Hz), and the processed window `k` holds
exactly the 512 resampled samples of that span, because every device's rate
divides into 128 Hz by a rational factor that maps `R` onto 512. The number of
complete windows is the audit's `len // R`. This is what lets every mark below
call the audit's own function and come out identical to it.

Per window, side data only -- the signal is never touched by a mark:

* **integrity** (Sec. 2.3): marked when the window holds at least one lost or
  interpolated sample, with the lost and interpolated fractions beside it. A
  contiguous hole longer than a window marks every window it touches. In
  sample space a hole is a splice between two samples, so the windows it
  touches are the one holding the sample before it and the one holding the
  sample after it; the time it spanned holds no samples and makes no window.
  The hole is read from the counter and from the timestamps, the longer of the
  two, exactly as `audit_integrity` reads it.
* **flat / saturated** per channel (Sec. 2.4): zero variance within the
  window, or at least one sample at the device rail -- `audit_channels`' own
  definitions and rails.
* **contact** per channel, Emotiv only (Sec. 2.5): the window's median
  `CQ.<channel>` below the top level 4 (`audit_contact.window_levels`), and
  the record-level majority rule from `audit_contact.contact_rule`, over all
  of the record's windows as the audit counted it.
* **ocular** (Sec. 2.8): frontal peak-to-peak and the 0.5-4 Hz
  frontal/posterior ratio, from `audit_ocular.window_measures` on the raw
  record at its native rate, as the audit measured them. Values, never a cut.
* **focus** (Sec. 2.1), Emotiv only: the window median of `PM.Focus.Scaled`,
  audit metadata for the optional contrast of Sec. 6.1, never in the signal.
* **event_window** (plan v42, Sec. 6.1), Slackline levels only: the window
  contains the instant of a K1 behavioural event (`k1_events`), no neighbour.
  The EEG of a level starts at the level start, so `window_start_s` -- raw
  window `k` starts at `4k` seconds, nothing is trimmed before windowing -- is
  also seconds from the level start: an event at `generalTime` `t` falls in raw
  window `floor(t / 4)`, kept window `floor(t / 4) - 1`. An event in the
  discarded first window or past the last complete one marks nothing.
"""

from __future__ import annotations

import warnings

import numpy as np

from MIL_CREDA_EEG import audit_contact, audit_ocular, k1_events
from MIL_CREDA_EEG.audit_channels import RAILS
from MIL_CREDA_EEG.audit_integrity import counter_modulus, lost_before_each_sample
from MIL_CREDA_EEG.modality import EEGRecord
from MIL_CREDA_EEG.preprocess_filter import TARGET_RATE, preprocess, scalp_signal

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["3.1", "2.3", "2.4", "2.5", "2.8", "2.1"],
    "experiments": ["3.1", "2.3", "2.4", "2.5", "2.8", "2.1"],
    "invariants": [
        "four_second_windows_of_512_samples_drop_the_first_and_the_partial_last",
        "the_window_holds_the_processed_samples_of_its_span",
        "a_cgx_record_maps_2000_raw_samples_onto_each_512_sample_window",
        "integrity_marks_equal_the_audits_definition",
        "a_gap_longer_than_a_window_marks_every_window_it_touches",
        "channel_marks_equal_the_audits_flat_and_saturated_definitions",
        "contact_marks_equal_the_audits_rule_per_window_and_per_record",
        "ocular_measures_equal_the_audits_and_cut_nothing",
        "kept_marks_are_the_audit_marks_without_the_first_window",
        "focus_is_kept_as_a_per_window_median_and_never_in_the_signal",
        "marking_never_modifies_the_signal",
        "an_event_marks_the_kept_window_holding_its_general_time",
        "the_event_mark_never_touches_another_array",
    ],
}

WINDOW_S = 4.0
WINDOW_SAMPLES = int(TARGET_RATE * WINDOW_S)
#: Leading windows discarded as the filter transient (Sec. 2.7).
DISCARDED_LEADING = 1
#: Plan v18, Sec. 3.2: a bag needs at least this many useful windows, already
#: without the first and the partial last. One floor for every condition; the
#: exclusion itself is `exclusions.bag_floor_reasons`.
BAG_FLOOR = 68

#: Plan v42, Sec. 6.1: K1's behavioural event mark (`mechanisms.EVENT_MARK`).
EVENT_MARK = "event_window"

#: Per-window side arrays, aligned with the kept windows.
WINDOW_ARRAYS = ("integrity_mark", "lost_fraction", "interpolated_fraction", "long_gap",
                 "flat", "saturated", "contact_bad", "frontal_ptp", "low_ratio",
                 "rail_window", "focus_median", EVENT_MARK)


def raw_window_size(rate: float, window_s: float = WINDOW_S) -> int:
    return int(round(rate * window_s))


def complete_windows(n_samples: int, rate: float) -> int:
    """The audit's count: complete blocks of raw samples."""
    return int(n_samples) // raw_window_size(rate)


def _per_window_sum(values: np.ndarray, size: int, n_windows: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)[: n_windows * size]
    return values.reshape(n_windows, size).sum(axis=1)


def integrity_marks(timestamps: np.ndarray, counter: np.ndarray | None,
                    interpolated: np.ndarray | None, rate: float, n_windows: int) -> dict:
    """Sec. 2.3 per window: lost and interpolated fractions, long holes, the mark."""
    timestamps = np.asarray(timestamps, dtype=float)
    n = len(timestamps)
    size = raw_window_size(rate)
    if counter is not None and np.isfinite(counter).any():
        lost = lost_before_each_sample(counter, counter_modulus(counter)).astype(float)
    else:
        lost = np.zeros(n)
    flag = (np.nan_to_num(np.asarray(interpolated, dtype=float)) > 0
            if interpolated is not None else np.zeros(n, dtype=bool))
    lost_in = _per_window_sum(lost, size, n_windows)
    interp_in = _per_window_sum(flag, size, n_windows)
    # The hole before sample i, in seconds: the longer of the counter and timestamp readings.
    step = np.r_[1.0 / rate, np.diff(timestamps)] if n else np.array([])
    gap_s = np.maximum(lost / rate, step - 1.0 / rate)
    long_gap = np.zeros(n_windows, dtype=bool)
    for i in np.flatnonzero(gap_s > WINDOW_S):
        for w in {(i - 1) // size, i // size}:
            if 0 <= w < n_windows:
                long_gap[w] = True
    lost_fraction = lost_in / (size + lost_in)
    interpolated_fraction = interp_in / size
    return {"integrity_mark": (lost_in > 0) | (interp_in > 0) | long_gap,
            "lost_fraction": lost_fraction, "interpolated_fraction": interpolated_fraction,
            "long_gap": long_gap, "gap_exceeds_window": bool((gap_s > WINDOW_S).any())}


def channel_marks(signal: np.ndarray, rate: float, rails: tuple[float, float],
                  n_windows: int) -> dict:
    """Sec. 2.4 per window and channel: flat (zero variance) and saturated (at the rail)."""
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, np.nanmedian(x, axis=0))
    size = raw_window_size(rate)
    blocks = x[: n_windows * size].reshape(n_windows, size, x.shape[1])
    return {"flat": blocks.std(axis=1) == 0,
            "saturated": ((blocks <= rails[0]) | (blocks >= rails[1])).any(axis=1)}


def contact_marks(cq: np.ndarray, rate: float, n_windows: int) -> dict:
    """Sec. 2.5 per window and channel, and the record-level majority rule."""
    levels, _ = audit_contact.window_levels(cq, rate, WINDOW_S)
    with np.errstate(invalid="ignore"):
        bad = np.isfinite(levels) & (levels < audit_contact.CQ_TOP)
    rule = audit_contact.contact_rule(cq, rate, WINDOW_S)
    return {"contact_bad": bad[:n_windows], "contact_meets_rule": rule["meets_rule"],
            "contact_n_bad_channels": rule["n_bad_channels"]}


def ocular_measures(signal: np.ndarray, channels: list[str], family: str, rate: float,
                    n_windows: int) -> dict:
    """Sec. 2.8 per window, from the audit's own measurement on the raw record."""
    used = audit_ocular.channels_used(channels, family)
    measures = audit_ocular.window_measures(signal, channels, rate, used["frontal"],
                                            used["posterior"], RAILS[family], WINDOW_S)
    if len(measures["frontal_ptp"]) != n_windows:
        nan = np.full(n_windows, np.nan)
        return {"frontal_ptp": nan, "low_ratio": nan.copy(),
                "rail_window": np.zeros(n_windows, dtype=bool)}
    return {"frontal_ptp": measures["frontal_ptp"], "low_ratio": measures["low_ratio"],
            "rail_window": measures["rail_windows"]}


def focus_medians(focus: np.ndarray, rate: float, n_windows: int) -> np.ndarray:
    """Window median of the sparse `PM.Focus.Scaled` stream; NaN where it never updated."""
    size = raw_window_size(rate)
    blocks = np.asarray(focus, dtype=float)[: n_windows * size].reshape(n_windows, size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(blocks, axis=1)


def emotiv_cq(record: EEGRecord, channels: list[str]) -> np.ndarray | None:
    """Per-sample `CQ.<channel>` columns in the given channel order, or None."""
    if not all(f"CQ.{c}" in record.quality for c in channels):
        return None
    return np.column_stack([record.quality[f"CQ.{c}"] for c in channels])


def record_windows(record: EEGRecord, cq: np.ndarray | None = None,
                   focus: np.ndarray | None = None, keep_first: bool = False,
                   events: list[float] | None = None) -> dict:
    """The processed windows of one record and every per-window side array.

    `keep_first=True` keeps the transient window, so marks can be laid beside
    the audit's own per-window readings; the processed output never keeps it.
    `events` are K1 event instants in seconds from the level start; when given,
    `EVENT_MARK` marks each window that contains one. Without it no key is added.
    """
    rate = float(record.nominal_rate)
    scalp, channels = scalp_signal(record)
    n_windows = complete_windows(len(scalp), rate)
    processed = preprocess(scalp, rate)
    signal = (processed[: n_windows * WINDOW_SAMPLES]
              .reshape(n_windows, WINDOW_SAMPLES, len(channels))
              .transpose(0, 2, 1).astype(np.float32))
    side: dict = {}
    side.update(integrity_marks(record.timestamps, record.counter, record.interpolated,
                                rate, n_windows))
    side.update(channel_marks(scalp, rate, RAILS[record.family], n_windows))
    side.update(ocular_measures(scalp, channels, record.family, rate, n_windows))
    if record.family == "EMOTIV":
        if cq is None:
            cq = emotiv_cq(record, channels)
        if cq is not None:
            side.update(contact_marks(cq, rate, n_windows))
        if focus is not None:
            side["focus_median"] = focus_medians(focus, rate, n_windows)
    first = 0 if keep_first else DISCARDED_LEADING
    index = np.arange(first, n_windows)
    out = {"channels": channels, "rate": rate, "target_rate": TARGET_RATE,
           "n_windows_total": int(n_windows), "n_windows": int(len(index)),
           "window_index": index, "window_start_s": index * WINDOW_S,
           "signal": signal[first:]}
    for key, value in side.items():
        out[key] = value[first:] if key in WINDOW_ARRAYS else value
    if events is not None:
        out[EVENT_MARK] = k1_events.event_windows(events, out["window_start_s"], WINDOW_S)
    return out
