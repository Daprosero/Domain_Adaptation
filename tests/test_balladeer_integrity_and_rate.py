"""Experiments plan v02, Secs. 2.2 and 2.3, on synthetic counters and clocks."""

from __future__ import annotations

import numpy as np

from MIL_CREDA_EEG import audit_integrity, audit_rate


def _clock(n: int, rate: float, start: float = 0.0) -> np.ndarray:
    return start + np.arange(n) / rate


# ---------------------------------------------------------------- 2.3 integrity


def test_counter_modulus_is_the_power_of_two_above_the_largest_value():
    assert audit_integrity.counter_modulus(np.arange(300) % 128) == 128
    assert audit_integrity.counter_modulus(np.arange(600) % 256) == 256


def test_lost_samples_are_counted_across_a_counter_wrap():
    counter = np.r_[np.arange(120, 128), np.arange(3, 10)]  # 127 -> 3 loses 0,1,2
    lost = audit_integrity.lost_before_each_sample(counter, 128)
    assert lost.sum() == 3 and lost[8] == 3


def test_a_contiguous_counter_reports_no_loss():
    report = audit_integrity.audit_integrity(
        _clock(2000, 128.0), np.arange(2000) % 128, np.zeros(2000), 128.0)
    assert report["n_lost"] == 0 and report["lost_fraction"] == 0.0
    assert report["windows_touched_by_loss"] == 0
    assert report["gap_exceeds_window"] is False


def test_interpolated_fraction_and_the_windows_it_touches():
    n = 128 * 20
    interpolated = np.zeros(n)
    interpolated[[10, 11, 128 * 9]] = 1
    report = audit_integrity.audit_integrity(
        _clock(n, 128.0), np.arange(n) % 128, interpolated, 128.0)
    assert report["n_interpolated"] == 3
    assert np.isclose(report["interpolated_fraction"], 3 / n)
    assert report["windows_touched_by_interpolation"] == 2
    assert report["n_windows"] == 5


def test_a_gap_longer_than_one_window_is_marked():
    rate = 128.0
    first = _clock(1000, rate)
    second = _clock(1000, rate, start=first[-1] + 5.0)
    timestamps = np.r_[first, second]
    counter = np.r_[np.arange(1000), np.arange(1000) + 1000 + 639] % 128
    report = audit_integrity.audit_integrity(timestamps, counter, None, rate)
    assert report["gap_exceeds_window"] is True
    assert report["longest_gap_s"] > 4.0


def test_a_short_gap_is_measured_and_not_marked():
    counter = np.r_[np.arange(500), np.arange(510, 1500)] % 128
    timestamps = np.r_[_clock(500, 128.0), _clock(990, 128.0, start=510 / 128.0)]
    report = audit_integrity.audit_integrity(timestamps, counter, None, 128.0)
    assert report["n_lost"] == 10
    assert report["longest_gap_samples"] == 10
    assert report["gap_exceeds_window"] is False
    assert report["windows_touched_by_loss"] == 1


# ---------------------------------------------------------------- 2.2 rate


def test_nearest_menu_rate_is_taken_in_log_space():
    assert audit_rate.nearest_in_menu(128.07, (128.0, 256.0)) == 128.0
    assert audit_rate.nearest_in_menu(64.0, (128.0, 256.0)) == 128.0
    assert audit_rate.nearest_in_menu(250.0, (128.0, 256.0)) == 256.0


def test_a_record_at_its_nominal_menu_rate_is_not_a_defect():
    report = audit_rate.audit_rate(_clock(40000, 128.0), np.arange(40000) % 128,
                                   128.0, audit_rate.EMOTIV_MENU)
    assert report["nearest_menu_rate"] == 128.0
    assert report["nearest_equals_nominal"] is True
    assert report["counter_contiguous"] is True
    assert report["hard_defect"] is False


def test_a_rate_half_the_nominal_is_flagged_as_packet_loss():
    n = 20000
    timestamps = _clock(n, 64.0)
    counter = (np.arange(n) * 2) % 128
    report = audit_rate.audit_rate(timestamps, counter, 128.0, audit_rate.EMOTIV_MENU)
    assert report["half_of_nominal"] is True
    assert report["counter_contiguous"] is False
    assert np.isclose(report["counter_corrected_rate"], 128.0, rtol=1e-3)


def test_a_nominal_rate_outside_the_menu_is_a_hard_defect():
    report = audit_rate.audit_rate(_clock(5000, 100.0), np.arange(5000) % 128,
                                   100.0, audit_rate.EMOTIV_MENU)
    assert report["nominal_in_menu"] is False
    assert report["hard_defect"] is True


def test_a_legitimate_high_menu_rate_is_configuration_not_defect():
    report = audit_rate.audit_rate(_clock(40000, 256.0), np.arange(40000) % 256,
                                   256.0, audit_rate.EMOTIV_MENU)
    assert report["hard_defect"] is False
    assert report["nearest_menu_rate"] == 256.0


def test_a_long_hole_lowers_the_span_rate_but_not_the_step_rate():
    """A 30 s hole drags the span-derived rate down; the median step keeps the
    configured rate, which tells a hole apart from a slower configuration.

    The categorical rate rule does not fire on a 24 % deficit (its nearest
    ratio class is still 1); the hole itself is 2.3's mark, not 2.2's."""
    first = _clock(6000, 128.0)
    second = _clock(6000, 128.0, start=first[-1] + 30.0)
    report = audit_rate.audit_rate(np.r_[first, second], None, 128.0,
                                   audit_rate.EMOTIV_MENU)
    assert report["effective_rate"] < 100
    assert np.isclose(report["median_step_rate"], 128.0)
    assert report["ratio_class"] == 1.0 and report["hard_defect"] is False
