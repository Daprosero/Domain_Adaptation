"""Experiments plan v50, Sec. 2.6: the restarted-or-concatenated mark."""

from __future__ import annotations

import numpy as np

from MIL_CREDA_EEG import audit_duration

RATE = 128.0


def _clean_timestamps(seconds: float = 120.0) -> np.ndarray:
    return 1_700_000_000.0 + np.arange(int(seconds * RATE)) / RATE


def test_a_backward_timestamp_marks_a_restarted_or_concatenated_session():
    first = _clean_timestamps(60.0)
    second = first[0] + 10.0 + np.arange(int(60 * RATE)) / RATE  # clock goes back
    report = audit_duration.audit_duration(np.r_[first, second], RATE, nominal=280.0)
    assert report["marks"] == ["restarted_or_concatenated"]
    assert report["n_backward_timestamp_steps"] == 1


def test_a_clean_session_is_not_marked():
    report = audit_duration.audit_duration(_clean_timestamps(), RATE, nominal=280.0)
    assert report["marks"] == []
    assert report["n_backward_timestamp_steps"] == 0


def test_a_counter_wrap_or_a_long_session_alone_is_not_a_restart():
    # A session three times its nominal, timestamps monotone: measured, not marked.
    report = audit_duration.audit_duration(_clean_timestamps(840.0), RATE, nominal=280.0)
    assert np.isclose(report["ratio_to_nominal"], 3.0)
    assert report["marks"] == []


def test_the_mark_excludes_nothing_and_the_floor_is_not_pending_here():
    first = _clean_timestamps(60.0)
    second = first[0] + np.arange(int(60 * RATE)) / RATE
    marked = audit_duration.audit_duration(np.r_[first, second], RATE, nominal=280.0)
    clean = audit_duration.audit_duration(first, RATE, nominal=280.0)
    assert marked["marks"] == ["restarted_or_concatenated"]
    assert marked["n_windows"] == clean["n_windows"] == 15
    assert not any("exclu" in key for key in marked)
    assert "bag_cardinality_floor" not in marked["pending"]
