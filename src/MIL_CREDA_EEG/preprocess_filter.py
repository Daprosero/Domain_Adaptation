"""Band-pass, then resample: the first step that transforms the signal.

Experiments plan v59, Secs. 2.1 and 2.7. Per record, only the scalp channels
enter -- 29 for CGX and 14 for Emotiv, picked by name in the canonical order
`modality` declares -- and A2, the bipolar externals and every other column
never do. No device is re-referenced, nothing is decomposed into components,
nothing is interpolated.

**The order is a constraint, not a preference.** The continuous recording is
band-passed to 0.2-45 Hz first and only then resampled to 128 Hz: resampling
first would fold whatever sits above the new Nyquist back into the good band,
irreversibly. The filter runs on the continuous recording, never per window,
because the 0.2 Hz edge has a time constant of about 0.8 s; its transient only
affects the start of the record, which `windows` discards.

The band-pass is the audit's own (`audit_channels.bandpass`: fourth-order
Butterworth, zero-phase), so the band the audit measured is the band the
encoder gets. Resampling is polyphase with the exact rational factor of each
device: CGX 500 -> 128 (32/125), EPOC+ at 256 -> 128 (1/2), and 128 Hz records
pass through unchanged.

A non-finite sample refuses instead of being filled: filling is interpolation,
which Sec. 2.7 does not allow, and the audit measured none in the release.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from MIL_CREDA_EEG.audit_channels import BAND, bandpass
from MIL_CREDA_EEG.modality import CGX_CHANNELS, EMOTIV_CHANNELS, EEGRecord

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["2.1", "2.7"],
    "experiments": ["2.1", "2.7"],
    "invariants": [
        "a_slow_drift_below_the_band_is_removed_before_resampling",
        "content_above_45_hz_is_cut_even_when_the_rate_allows_it",
        "output_is_at_128_hz_with_the_raw_duration",
        "the_filter_is_the_audits_zero_phase_band_pass",
        "the_input_array_is_never_modified",
        "preprocess_is_filter_then_resample_and_the_order_is_observable",
        "non_finite_samples_refuse_instead_of_being_filled",
        "scalp_channels_only_in_canonical_order_and_never_a2_or_exg",
        "a_non_scalp_column_in_the_signal_refuses",
        "a_missing_scalp_channel_refuses",
    ],
}

#: The common rate of Sec. 2.7.
TARGET_RATE = 128.0

#: Scalp channels per device family, in the order the encoder receives them.
SCALP = {"CGX": CGX_CHANNELS, "EMOTIV": EMOTIV_CHANNELS}


def scalp_signal(record: EEGRecord) -> tuple[np.ndarray, list[str]]:
    """The record's scalp channels only, in canonical order; refuses anything else."""
    expected = SCALP[record.family]
    foreign = [c for c in record.channels if c not in expected]
    if foreign:
        raise ValueError(f"{record.path}: {foreign} is not a scalp channel of {record.family}")
    missing = [c for c in expected if c not in record.channels]
    if missing:
        raise ValueError(f"{record.path}: missing scalp channels {missing}")
    index = {c: k for k, c in enumerate(record.channels)}
    return np.asarray(record.signal)[:, [index[c] for c in expected]], list(expected)


def filter_continuous(signal: np.ndarray, rate: float) -> np.ndarray:
    """0.2-45 Hz zero-phase band-pass over the whole recording (a new array)."""
    signal = np.asarray(signal, dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError("non-finite samples in the signal; Sec. 2.7 does not fill them")
    return bandpass(signal, rate, BAND)


def resample(signal: np.ndarray, rate: float, target: float = TARGET_RATE) -> np.ndarray:
    """Polyphase resampling by the exact rational factor `target / rate`."""
    factor = Fraction(target / rate).limit_denominator(1000)
    if factor == 1:
        return np.array(signal, dtype=float, copy=True)
    return resample_poly(signal, factor.numerator, factor.denominator, axis=0)


def preprocess(signal: np.ndarray, rate: float) -> np.ndarray:
    """Sec. 2.7 in its declared order: filter the continuous signal, then resample."""
    return resample(filter_continuous(signal, rate), rate)
