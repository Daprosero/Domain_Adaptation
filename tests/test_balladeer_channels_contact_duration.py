"""Experiments plan v02, Secs. 2.4-2.6, on synthetic signals."""

from __future__ import annotations

import numpy as np

from MIL_CREDA_EEG import audit_channels, audit_contact, audit_duration

RATE = 128.0


def _eeg(n_channels: int = 6, seconds: float = 40.0, seed: int = 0) -> np.ndarray:
    """Pink-ish noise plus a 10 Hz and a 20 Hz component on every channel."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    base = (np.cumsum(rng.normal(size=(len(t), n_channels)), axis=0) * 0.05
            + rng.normal(size=(len(t), n_channels)))
    return 4000 + base + 3 * np.sin(2 * np.pi * 10 * t)[:, None] \
        + 2 * np.sin(2 * np.pi * 20 * t)[:, None]


RAILS = (0.0, 8401.922852)


# ---------------------------------------------------------------- 2.4 channels


def test_a_flat_channel_is_marked_flat_and_nothing_is_removed():
    x = _eeg()
    x[:, 2] = 4100.0
    report = audit_channels.channel_metrics(x, ["a", "b", "c", "d", "e", "f"],
                                            RATE, RAILS)
    marks = audit_channels.mark_channels(report)
    assert marks["c"] == ["flat"]
    assert all(marks[ch] == [] for ch in "abdef")
    assert report["c"]["flat_window_fraction"] == 1.0
    assert x.shape == (len(x), 6)


def test_a_channel_at_the_device_rail_is_marked_saturated_with_its_windows():
    x = _eeg()
    x[: int(4 * RATE), 1] = RAILS[1]
    report = audit_channels.channel_metrics(x, list("abcdef"), RATE, RAILS)
    assert audit_channels.mark_channels(report)["b"] == ["saturated"]
    assert np.isclose(report["b"]["saturated_window_fraction"], 1 / 10)


def test_variance_and_spectral_outliers_are_measured_not_marked():
    """Plan Sec. 2.4 declares the criteria and no cut: report the z, mark nothing."""
    x = _eeg()
    x[:, 4] = 4000 + (x[:, 4] - 4000) * 50
    t = np.arange(len(x)) / RATE
    drift = np.cumsum(np.random.default_rng(3).normal(size=len(x))) * 0.5
    x[:, 5] = 4000 + drift + 0.0 * t
    report = audit_channels.channel_metrics(x, list("abcdef"), RATE, RAILS)
    assert report["e"]["log_std_robust_z"] > 3
    assert report["f"]["high_band_fraction"] < report["a"]["high_band_fraction"]
    assert report["f"]["high_band_robust_z"] < 0
    marks = audit_channels.mark_channels(report)
    assert marks["e"] == [] and marks["f"] == []


def test_a_channel_missing_from_its_cohort_is_reported_by_name():
    missing = audit_channels.missing_vs_cohort(
        {"r1": ["a", "b", "c"], "r2": ["a", "c"], "r3": ["a", "b", "c"]})
    assert missing == {"r1": [], "r2": ["b"], "r3": []}


def test_two_models_share_a_montage_only_with_the_same_channels():
    assert audit_channels.same_montage({"EPOCX": ["a", "b"], "EPOCPLUS": ["a", "b"]})
    assert not audit_channels.same_montage({"EPOCX": ["a", "b"], "EPOCPLUS": ["a"]})


def test_a_reference_already_subtracted_reads_near_zero_against_the_scalp():
    x = _eeg(seed=1)
    referenced = audit_channels.reference_metrics(np.zeros(len(x)) + 1e-9, x, RATE)
    live = audit_channels.reference_metrics(x[:, 0] * 1.0, x[:, 1:], RATE)
    assert referenced["band_std_ratio_to_scalp"] < 1e-6
    assert live["band_std_ratio_to_scalp"] > 0.3


def test_a_bipolar_external_at_a_constant_rail_carries_nothing():
    x = _eeg(seed=2)
    report = audit_channels.bipolar_metrics(np.full(len(x), -833333.3125),
                                            x[:, :2], x[:, 2:4], RATE)
    assert report["constant"] is True
    assert report["frontal_minus_occipital_corr"] is None


def test_an_ocular_external_correlates_with_frontopolar_more_than_occipital():
    rng = np.random.default_rng(5)
    n = int(60 * RATE)
    blink = np.zeros(n)
    blink[rng.choice(n - 64, 30, replace=False)] = 1
    blink = np.convolve(blink, np.hanning(48) * 200, mode="same")
    front = np.c_[blink + rng.normal(size=n), blink * 0.8 + rng.normal(size=n)]
    back = rng.normal(size=(n, 2))
    exg = blink * 1.2 + rng.normal(size=n)
    report = audit_channels.bipolar_metrics(exg, front, back, RATE)
    assert report["constant"] is False
    assert report["frontal_minus_occipital_corr"] > 0.5


# ---------------------------------------------------------------- 2.5 contact


def test_emotiv_quality_below_the_declared_ninety_is_counted_per_window():
    n = int(40 * RATE)
    t = np.arange(n) / RATE
    eq = np.full(n, np.nan)
    eq[::64] = 100.0
    eq[(t >= 8) & (t < 16) & ~np.isnan(eq)] = 50.0
    report = audit_contact.emotiv_quality(t, {"EQ.OVERALL": eq, "CQ.Overall": eq})
    assert report["threshold"] == 90.0
    assert np.isclose(report["EQ.OVERALL"]["window_fraction_below"], 2 / 10)
    assert report["EQ.OVERALL"]["median"] == 100.0


def test_cgx_contact_has_no_quality_column_so_it_is_read_from_the_signal():
    rate = 500.0
    t = np.arange(int(20 * rate)) / rate
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(len(t), 3))
    noisy = clean.copy()
    noisy[:, 0] += 30 * np.sin(2 * np.pi * 50 * t)
    low = audit_contact.line_noise_ratio(clean, rate)
    high = audit_contact.line_noise_ratio(noisy, rate)
    assert high[0] > 10 * low[0]
    assert np.allclose(high[1:], low[1:])


# ---------------------------------------------------------------- 2.6 duration


def test_nominal_duration_of_each_task_comes_from_its_own_source():
    flags = {"slackline_levels_flags_info": [
        {"level": "Level1", "flags": [{"flag_spawn_time": 2}, {"flag_spawn_time": 300}]}]}
    assert audit_duration.nominal_duration("Slackline", 1, flags) == 302.0
    assert audit_duration.nominal_duration("Robots", None, flags) == 280.0
    assert audit_duration.nominal_duration("Cognifit", None, flags) is None


def test_duration_reports_ratio_and_window_count_without_a_cut():
    t = np.arange(int(150 * RATE)) / RATE
    report = audit_duration.audit_duration(t, RATE, nominal=300.0)
    assert np.isclose(report["duration_s"], 150.0)
    assert np.isclose(report["ratio_to_nominal"], 0.5)
    assert report["n_windows"] == 37
    assert report["marks"] == []
