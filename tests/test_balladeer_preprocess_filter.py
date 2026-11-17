"""Experiments plan v05, Sec. 2.7 (band-pass, then resample to 128 Hz), on synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest

from MIL_CREDA_EEG import modality, preprocess_filter


def _power_at(x: np.ndarray, rate: float, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    return float(spectrum[np.argmin(np.abs(freqs - freq))])


def test_a_slow_drift_below_the_band_is_removed_before_resampling():
    # resample_poly alone keeps everything below the new Nyquist, so a 0.05 Hz
    # drift survives it; only the 0.2 Hz high-pass edge can remove it.
    rate = 500.0
    t = np.arange(int(120 * rate)) / rate
    drift = 200.0 * np.sin(2 * np.pi * 0.05 * t)
    alpha = 10.0 * np.sin(2 * np.pi * 10.0 * t)
    x = np.column_stack([drift + alpha, drift + alpha])
    out = preprocess_filter.preprocess(x, rate)
    target = preprocess_filter.TARGET_RATE
    middle = out[len(out) // 4: 3 * len(out) // 4, 0]
    raw_middle = (drift + alpha)[len(t) // 4: 3 * len(t) // 4]
    assert _power_at(middle, target, 0.05) < 1e-3 * _power_at(raw_middle, rate, 0.05)
    assert _power_at(middle, target, 10.0) > 0.5 * _power_at(raw_middle, rate, 10.0) * (
        target / rate) ** 2


def test_content_above_45_hz_is_cut_even_when_the_rate_allows_it():
    # The audit's filter is a fourth-order Butterworth run forwards and back:
    # 45 Hz is its -6 dB point, not a brick wall. 55 Hz keeps about 2 % of its
    # power, 80 Hz well under 0.1 %; the 10 Hz passband is untouched.
    rate = 500.0
    t = np.arange(int(60 * rate)) / rate
    x = (10 * np.sin(2 * np.pi * 10 * t) + 10 * np.sin(2 * np.pi * 55 * t)
         + 10 * np.sin(2 * np.pi * 80 * t))[:, None]
    filtered = preprocess_filter.filter_continuous(x, rate)[:, 0]
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    ratio = {f: _power_at(filtered[middle], rate, f) / _power_at(x[middle, 0], rate, f)
             for f in (10.0, 55.0, 80.0)}
    assert ratio[10.0] > 0.99
    assert ratio[55.0] < 0.05
    assert ratio[80.0] < 1e-3


@pytest.mark.parametrize("rate,seconds", [(500.0, 30.0), (256.0, 30.0), (128.0, 30.0),
                                          (500.0, 30.001)])
def test_output_is_at_128_hz_with_the_raw_duration(rate, seconds):
    n = int(round(seconds * rate))
    x = np.random.default_rng(0).normal(size=(n, 3))
    out = preprocess_filter.preprocess(x, rate)
    assert out.shape == (int(np.ceil(n * 128.0 / rate)), 3)


def test_the_filter_is_the_audits_zero_phase_band_pass():
    from MIL_CREDA_EEG.audit_channels import bandpass
    x = np.random.default_rng(1).normal(size=(5000, 2))
    np.testing.assert_array_equal(preprocess_filter.filter_continuous(x, 500.0),
                                  bandpass(x, 500.0))


def test_the_input_array_is_never_modified():
    x = np.random.default_rng(2).normal(size=(5000, 2))
    before = x.copy()
    preprocess_filter.preprocess(x, 500.0)
    np.testing.assert_array_equal(x, before)


def test_non_finite_samples_refuse_instead_of_being_filled():
    x = np.random.default_rng(3).normal(size=(5000, 2))
    x[10, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        preprocess_filter.preprocess(x, 500.0)


def _record(family: str, channels: list[str], extras: dict | None = None):
    n = 1000
    return modality.EEGRecord(
        path="x", family=family, device="CGX" if family == "CGX" else "EPOCX",
        columns=[], channels=channels,
        signal=np.arange(n * len(channels), dtype=float).reshape(n, len(channels)),
        timestamps=np.arange(n) / 500.0, counter=None, interpolated=None,
        nominal_rate=500.0, extras=extras or {})


def test_scalp_channels_only_in_canonical_order_and_never_a2_or_exg():
    shuffled = list(reversed(modality.CGX_CHANNELS))
    record = _record("CGX", shuffled, {"A2": np.zeros(1000), "ExG 1": np.zeros(1000)})
    signal, channels = preprocess_filter.scalp_signal(record)
    assert channels == list(modality.CGX_CHANNELS)
    assert not {"A2", "ExG 1", "ExG 2"} & set(channels)
    assert signal.shape == (1000, 29)
    np.testing.assert_array_equal(signal[:, 0], record.signal[:, shuffled.index("AF7")])


def test_emotiv_keeps_its_fourteen_scalp_channels():
    record = _record("EMOTIV", list(modality.EMOTIV_CHANNELS))
    _, channels = preprocess_filter.scalp_signal(record)
    assert channels == list(modality.EMOTIV_CHANNELS)


def test_a_non_scalp_column_in_the_signal_refuses():
    record = _record("CGX", list(modality.CGX_CHANNELS) + ["A2"])
    with pytest.raises(ValueError, match="not a scalp channel"):
        preprocess_filter.scalp_signal(record)


def test_a_missing_scalp_channel_refuses():
    record = _record("EMOTIV", list(modality.EMOTIV_CHANNELS)[:-1])
    with pytest.raises(ValueError, match="missing scalp"):
        preprocess_filter.scalp_signal(record)


def test_preprocess_is_filter_then_resample_and_the_order_is_observable():
    # resample_poly anti-aliases on its own, so the order changes no band
    # wholesale; it does change the samples. Pin the declared order exactly,
    # and prove the reversed order is distinguishable on this input.
    rate = 500.0
    t = np.arange(int(20 * rate)) / rate
    rng = np.random.default_rng(4)
    x = np.column_stack([rng.normal(size=len(t)) + 50 * np.sin(2 * np.pi * 0.1 * t)
                         for _ in range(2)])
    declared = preprocess_filter.resample(preprocess_filter.filter_continuous(x, rate), rate)
    reversed_order = preprocess_filter.filter_continuous(preprocess_filter.resample(x, rate),
                                                         preprocess_filter.TARGET_RATE)
    out = preprocess_filter.preprocess(x, rate)
    np.testing.assert_array_equal(out, declared)
    assert np.max(np.abs(out - reversed_order)) > 1e-3
