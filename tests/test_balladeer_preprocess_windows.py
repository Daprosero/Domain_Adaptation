"""Experiments plan v05, Secs. 3.1, 2.3, 2.4, 2.5 and 2.8: windows and their side marks.

The marks are pinned against the audit's own functions on the same synthetic
record, so a record the audit marked is marked identically here.
"""

from __future__ import annotations

import numpy as np
import pytest

from MIL_CREDA_EEG import audit_contact, audit_ocular, modality, windows
from MIL_CREDA_EEG.audit_channels import RAILS, channel_metrics
from MIL_CREDA_EEG.audit_integrity import audit_integrity

EMOTIV = list(modality.EMOTIV_CHANNELS)


def _emotiv_record(seconds: float = 41.0, rate: float = 128.0, seed: int = 0):
    """A synthetic Emotiv record with a hole, interpolated samples, a flat and a railed channel."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * rate))
    signal = 4200 + rng.normal(scale=5.0, size=(n, len(EMOTIV)))
    signal[:, EMOTIV.index("T7")] = 4100.0                     # flat
    signal[int(9.5 * rate), EMOTIV.index("O1")] = RAILS["EMOTIV"][1]   # railed once
    counter = np.arange(n) % 128
    counter = counter.astype(float)
    # Three samples lost before sample at 13.2 s: the counter jumps by 4.
    cut = int(13.2 * rate)
    counter[cut:] = (counter[cut:] + 3) % 128
    interpolated = np.zeros(n)
    interpolated[int(21.1 * rate): int(21.1 * rate) + 5] = 1
    timestamps = np.arange(n) / rate
    cq = np.full((n, len(EMOTIV)), 4.0)
    cq[: int(28 * rate), EMOTIV.index("F3")] = 2.0              # bad contact, first 28 s
    record = modality.EEGRecord(
        path="synthetic", family="EMOTIV", device="EPOCX", columns=[],
        channels=EMOTIV, signal=signal, timestamps=timestamps, counter=counter,
        interpolated=interpolated, nominal_rate=rate,
        quality={f"CQ.{c}": cq[:, k] for k, c in enumerate(EMOTIV)})
    return record, cq


def test_four_second_windows_of_512_samples_drop_the_first_and_the_partial_last():
    record, cq = _emotiv_record(seconds=41.0)
    out = windows.record_windows(record, cq=cq)
    # 41 s -> 10 complete raw windows; the first is the filter transient.
    assert out["n_windows_total"] == 10
    assert out["signal"].shape == (9, 14, 512)
    assert out["signal"].dtype == np.float32
    np.testing.assert_array_equal(out["window_index"], np.arange(1, 10))
    np.testing.assert_allclose(out["window_start_s"], np.arange(1, 10) * 4.0)


def test_the_window_holds_the_processed_samples_of_its_span():
    record, cq = _emotiv_record(seconds=41.0)
    out = windows.record_windows(record, cq=cq)
    from MIL_CREDA_EEG import preprocess_filter
    processed = preprocess_filter.preprocess(record.signal, 128.0)
    np.testing.assert_array_equal(out["signal"][0], processed[512:1024].T.astype(np.float32))


def test_a_cgx_record_maps_2000_raw_samples_onto_each_512_sample_window():
    rate = 500.0
    n = int(29.999 * rate)
    signal = np.random.default_rng(0).normal(size=(n, 29))
    record = modality.EEGRecord(
        path="s", family="CGX", device="CGX", columns=[], channels=list(modality.CGX_CHANNELS),
        signal=signal, timestamps=np.arange(n) / rate, counter=np.arange(n) % 128.0,
        interpolated=None, nominal_rate=rate)
    out = windows.record_windows(record)
    assert out["n_windows_total"] == 7
    assert out["signal"].shape == (6, 29, 512)
    assert "contact_bad" not in out and "focus_median" not in out


def test_integrity_marks_equal_the_audits_definition():
    record, cq = _emotiv_record()
    out = windows.record_windows(record, cq=cq, keep_first=True)
    audit = audit_integrity(record.timestamps, record.counter, record.interpolated, 128.0)
    loss = out["lost_fraction"] > 0
    interp = out["interpolated_fraction"] > 0
    assert int(loss.sum()) == audit["windows_touched_by_loss"] == 1
    assert int(interp.sum()) == audit["windows_touched_by_interpolation"] == 1
    assert np.flatnonzero(loss).tolist() == [3]          # 13.2 s -> window 3
    assert np.flatnonzero(interp).tolist() == [5]        # 21.1 s -> window 5
    np.testing.assert_array_equal(out["integrity_mark"], loss | interp)
    assert out["lost_fraction"][3] == pytest.approx(3 / (512 + 3))
    assert out["interpolated_fraction"][5] == pytest.approx(5 / 512)
    assert out["gap_exceeds_window"] == audit["gap_exceeds_window"] is False


def test_a_gap_longer_than_a_window_marks_every_window_it_touches():
    record, cq = _emotiv_record()
    cut = int(17.0 * 128)
    record.counter[cut:] = (record.counter[cut:] + 600) % 128   # 600 lost: 4.7 s
    record.timestamps[cut:] += 600 / 128.0
    out = windows.record_windows(record, cq=cq, keep_first=True)
    audit = audit_integrity(record.timestamps, record.counter, record.interpolated, 128.0)
    assert out["gap_exceeds_window"] is True and audit["gap_exceeds_window"] is True
    # Sample `cut` opens window 4 (17 s / 4 s); its predecessor closes window 3.
    assert out["long_gap"].tolist().count(True) >= 1
    assert out["long_gap"][4] and out["integrity_mark"][4]
    cut2 = int(16.0 * 128)       # a gap exactly at a window border touches both sides
    record2, cq2 = _emotiv_record()
    record2.counter[cut2:] = (record2.counter[cut2:] + 600) % 128
    record2.timestamps[cut2:] += 600 / 128.0
    out2 = windows.record_windows(record2, cq=cq2, keep_first=True)
    assert out2["long_gap"][3] and out2["long_gap"][4]


def test_channel_marks_equal_the_audits_flat_and_saturated_definitions():
    record, cq = _emotiv_record()
    out = windows.record_windows(record, cq=cq, keep_first=True)
    metrics = channel_metrics(record.signal, EMOTIV, 128.0, RAILS["EMOTIV"])
    for k, ch in enumerate(EMOTIV):
        assert out["flat"][:, k].mean() == metrics[ch]["flat_window_fraction"]
        assert out["saturated"][:, k].mean() == metrics[ch]["saturated_window_fraction"]
    assert out["flat"][:, EMOTIV.index("T7")].all()
    assert np.flatnonzero(out["saturated"][:, EMOTIV.index("O1")]).tolist() == [2]


def test_contact_marks_equal_the_audits_rule_per_window_and_per_record():
    record, cq = _emotiv_record()
    out = windows.record_windows(record, cq=cq, keep_first=True)
    rule = audit_contact.contact_rule(cq, 128.0)
    np.testing.assert_allclose(out["contact_bad"].mean(axis=0), rule["bad_window_fraction"])
    assert out["contact_bad"][:, EMOTIV.index("F3")].tolist() == [True] * 7 + [False] * 3
    assert out["contact_meets_rule"] == rule["meets_rule"] is False


def test_ocular_measures_equal_the_audits_and_cut_nothing():
    record, cq = _emotiv_record()
    out = windows.record_windows(record, cq=cq, keep_first=True)
    used = audit_ocular.channels_used(EMOTIV, "EMOTIV")
    audit = audit_ocular.window_measures(record.signal, EMOTIV, 128.0, used["frontal"],
                                         used["posterior"], RAILS["EMOTIV"])
    np.testing.assert_array_equal(out["frontal_ptp"], audit["frontal_ptp"])
    np.testing.assert_array_equal(out["low_ratio"], audit["low_ratio"])
    assert len(out["frontal_ptp"]) == out["n_windows_total"]


def test_kept_marks_are_the_audit_marks_without_the_first_window():
    record, cq = _emotiv_record()
    full = windows.record_windows(record, cq=cq, keep_first=True)
    kept = windows.record_windows(record, cq=cq)
    for key in windows.WINDOW_ARRAYS:
        if key in full:
            np.testing.assert_array_equal(kept[key], full[key][1:])


def test_focus_is_kept_as_a_per_window_median_and_never_in_the_signal():
    record, cq = _emotiv_record()
    focus = np.full(len(record.timestamps), np.nan)
    focus[::64] = np.arange(len(focus[::64]), dtype=float)
    out = windows.record_windows(record, cq=cq, focus=focus, keep_first=True)
    assert out["focus_median"][0] == pytest.approx(np.nanmedian(focus[:512]))
    assert out["signal"].shape[1] == 14


def test_marking_never_modifies_the_signal():
    record, cq = _emotiv_record()
    before = record.signal.copy()
    marked = windows.record_windows(record, cq=cq)
    record_clean, cq_clean = _emotiv_record()
    record_clean.counter = np.arange(len(record_clean.counter)) % 128.0
    record_clean.interpolated = np.zeros(len(record_clean.counter))
    clean = windows.record_windows(record_clean, cq=np.full_like(cq_clean, 4.0))
    np.testing.assert_array_equal(record.signal, before)
    # Same samples, different marks: the windows are bit-identical.
    np.testing.assert_array_equal(marked["signal"], clean["signal"])
    assert marked["integrity_mark"].any() and not clean["integrity_mark"].any()
