"""Experiments plan v04, Sec. 2.5: the operator-approved record rule, counted only."""

from __future__ import annotations

import numpy as np

from MIL_CREDA_EEG import audit_contact

RATE = 128.0
N_WINDOWS = 10
N_CHANNELS = 14


def _cq(bad_windows_per_channel: list[int]) -> np.ndarray:
    """Per-sample CQ at green (4), with the first `k` windows of each channel at level 2."""
    n = int(RATE * audit_contact.WINDOW_S)
    cq = np.full((N_WINDOWS * n, len(bad_windows_per_channel)), 4.0)
    for ch, k in enumerate(bad_windows_per_channel):
        cq[:k * n, ch] = 2.0
    return cq


def test_eight_of_fourteen_channels_bad_in_most_windows_meets_the_rule():
    cq = _cq([6] * 8 + [0] * 6)
    before = cq.copy()
    result = audit_contact.contact_rule(cq, RATE)
    assert result["n_windows"] == N_WINDOWS
    assert result["n_bad_channels"] == 8
    assert result["meets_rule"] is True
    np.testing.assert_array_equal(cq, before)


def test_seven_of_fourteen_channels_bad_does_not_meet_the_rule():
    cq = _cq([10] * 7 + [0] * 7)
    before = cq.copy()
    result = audit_contact.contact_rule(cq, RATE)
    assert result["n_bad_channels"] == 7
    assert result["meets_rule"] is False
    np.testing.assert_array_equal(cq, before)


def test_a_channel_bad_in_exactly_half_the_windows_is_not_bad():
    cq = _cq([5] * N_CHANNELS)
    before = cq.copy()
    result = audit_contact.contact_rule(cq, RATE)
    assert result["bad_window_fraction"] == [0.5] * N_CHANNELS
    assert result["n_bad_channels"] == 0
    assert result["meets_rule"] is False
    np.testing.assert_array_equal(cq, before)


def test_a_window_whose_median_is_green_is_not_bad_even_with_a_dip():
    # One window: 3 of 4 samples-quarters at 4, one quarter at 0 -> median 4.
    n = int(RATE * audit_contact.WINDOW_S)
    cq = np.full((N_WINDOWS * n, N_CHANNELS), 4.0)
    for w in range(N_WINDOWS):
        cq[w * n:w * n + n // 4, :] = 0.0
    result = audit_contact.contact_rule(cq, RATE)
    assert result["bad_window_fraction"] == [0.0] * N_CHANNELS
    assert result["meets_rule"] is False
