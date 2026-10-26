"""Experiments plan v02, Sec. 2.1: three destinations for every EEG column.

Synthetic headers only; nothing here reads the BALLADEER files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import modality
from MIL_CREDA_EEG.modality import (
    AUDIT, DISCARD, SIGNAL, UNCLASSIFIED, classify_column, classify_file,
    read_record,
)

EMOTIV_COLUMNS = (
    ["Timestamp", "OriginalTimestamp", "EEG.Counter", "EEG.Interpolated"]
    + [f"EEG.{c}" for c in modality.EMOTIV_CHANNELS]
    + ["EEG.RawCq", "EEG.Battery", "EEG.BatteryPercent", "MarkerIndex",
       "MarkerType", "MarkerValueInt", "EEG.MarkerHardware"]
    + [f"CQ.{c}" for c in modality.EMOTIV_CHANNELS] + ["CQ.Overall"]
    + ["EQ.SampleRateQuality", "EQ.OVERALL"]
    + [f"EQ.{c}" for c in modality.EMOTIV_CHANNELS]
    + ["MOT.CounterMems", "MOT.AccX", "PM.Focus.Scaled", "PM.Attention.Raw",
       "POW.AF3.Theta"]
)

CGX_COLUMNS = (
    ["timestamps"] + [f"{c}(uV)" for c in modality.CGX_CHANNELS]
    + ["A2(uV)", "ExG 1(uV)", "ExG 2(uV)", "ACC32(mg)", "ACC33(mg)",
       "ACC34(mg)", "Packet Counter", "TRIGGER"]
)


def write_emotiv(path: Path, n: int = 64, rate: int = 128,
                 drop: tuple[str, ...] = ()) -> Path:
    """A minimal EmotivPRO export: one metadata line, then the column header."""
    columns = [c for c in EMOTIV_COLUMNS if c not in drop]
    rng = np.random.default_rng(0)
    header = (f"title:UB9999, start timestamp:1000.0, stop timestamp:"
              f"{1000.0 + n / rate}, headset type:EPOCX, headset serial:X, "
              f"headset firmware:1, channels:{len(columns)}, sampling rate:"
              f"eeg_{rate};mot_32;pm_0.1;pow_8, samples:{n}     , version:2.2, ")
    lines = [header, ",".join(columns)]
    for i in range(n):
        row = []
        for c in columns:
            if c in ("Timestamp", "OriginalTimestamp"):
                row.append(f"{1000.0 + i / rate:.6f}")
            elif c == "EEG.Counter":
                row.append(str(i % 128))
            elif c == "EEG.Interpolated":
                row.append("0")
            elif c.startswith("EEG.") and c[4:] in modality.EMOTIV_CHANNELS:
                row.append(f"{4200 + rng.normal():.4f}")
            elif c in ("EQ.OVERALL", "CQ.Overall"):
                row.append("100" if i % 8 == 0 else "")
            else:
                row.append("")
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_cgx(path: Path, n: int = 200, rate: float = 500.0) -> Path:
    rng = np.random.default_rng(1)
    lines = [",".join(CGX_COLUMNS)]
    for i in range(n):
        values = [f"{100.0 + i / rate:.7f}"]
        values += [f"{rng.normal() * 10:.4f}" for _ in modality.CGX_CHANNELS]
        values += [f"{rng.normal():.4f}", "-833333.3125", "-833333.3125",
                   "1", "2", "3", str(i % 128), "0"]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_scalp_channels_go_to_signal_and_only_they_do():
    for c in modality.EMOTIV_CHANNELS:
        assert classify_column("EMOTIV", f"EEG.{c}")[0] == SIGNAL
    for c in modality.CGX_CHANNELS:
        assert classify_column("CGX", f"{c}(uV)")[0] == SIGNAL
    signal = [c for c in EMOTIV_COLUMNS if classify_column("EMOTIV", c)[0] == SIGNAL]
    assert len(signal) == 14
    signal = [c for c in CGX_COLUMNS if classify_column("CGX", c)[0] == SIGNAL]
    assert len(signal) == 29


def test_counters_timestamps_quality_and_interpolation_go_to_audit():
    for c in ("Timestamp", "OriginalTimestamp", "EEG.Counter", "EEG.Interpolated",
              "EEG.RawCq", "CQ.AF3", "CQ.Overall", "EQ.SampleRateQuality",
              "EQ.OVERALL", "EQ.F7"):
        assert classify_column("EMOTIV", c)[0] == AUDIT, c
    for c in ("timestamps", "Packet Counter"):
        assert classify_column("CGX", c)[0] == AUDIT, c


def test_reference_and_bipolar_externals_go_to_audit_not_discard():
    assert classify_column("CGX", "A2(uV)") == (AUDIT, "reference")
    assert classify_column("CGX", "ExG 1(uV)") == (AUDIT, "bipolar_external")
    assert classify_column("CGX", "ExG 2(uV)") == (AUDIT, "bipolar_external")


def test_motion_triggers_battery_and_proprietary_derivations_are_discarded():
    for c in ("EEG.Battery", "EEG.BatteryPercent", "MarkerIndex", "MarkerType",
              "MarkerValueInt", "EEG.MarkerHardware", "MOT.AccX",
              "MOT.CounterMems", "PM.Focus.Scaled", "POW.AF3.Theta"):
        assert classify_column("EMOTIV", c)[0] == DISCARD, c
    for c in ("ACC32(mg)", "TRIGGER"):
        assert classify_column("CGX", c)[0] == DISCARD, c


def test_the_proprietary_attention_score_is_named_as_such():
    assert classify_column("EMOTIV", "PM.Attention.Raw") == (
        DISCARD, "proprietary_attention_score")
    assert classify_column("EMOTIV", "PM.Focus.Scaled") == (
        DISCARD, "proprietary_attention_score")
    assert classify_column("EMOTIV", "PM.Stress.Raw") == (
        DISCARD, "proprietary_derived")


def test_an_unknown_column_is_unclassified_never_silently_kept():
    assert classify_column("EMOTIV", "EEG.Foo")[0] == UNCLASSIFIED
    assert classify_column("CGX", "Mystery")[0] == UNCLASSIFIED
    with pytest.raises(ValueError):
        classify_column("NOPE", "x")


def test_files_eye_tracking_and_wristband_are_discarded_game_logs_kept():
    assert classify_file("EYE_TRACKING")[0] == DISCARD
    assert classify_file("EMBRACEPLUS")[0] == DISCARD
    assert classify_file("GAME_DATA")[0] == AUDIT
    assert classify_file("TAGS")[0] == AUDIT
    assert classify_file("DEMOGRAPHICS")[0] == AUDIT
    assert classify_file("EEG")[0] == SIGNAL


def test_emotiv_suffix_names_the_exported_streams():
    assert modality.emotiv_export_streams("X_EPOCX_1_2023.md.pm.bp.csv") == [
        "motion_data", "performance_metrics", "band_power"]
    assert modality.emotiv_export_streams("X_EPOCX_1_2023.md.bp.csv") == [
        "motion_data", "band_power"]


def test_read_record_splits_an_emotiv_file_by_destination(tmp_path):
    path = write_emotiv(tmp_path / "UB9999_EPOCX_1_2023.01.01T00.00.00+01.00.md.pm.bp.csv")
    record = read_record(path)
    assert record.family == "EMOTIV" and record.device == "EPOCX"
    assert record.channels == list(modality.EMOTIV_CHANNELS)
    assert record.signal.shape == (64, 14)
    assert record.nominal_rate == 128.0
    assert record.counter is not None and record.interpolated is not None
    assert "EQ.OVERALL" in record.quality
    assert record.header["samples"] == "64"


def test_read_record_splits_a_cgx_file_by_destination(tmp_path):
    record = read_record(write_cgx(tmp_path / "UB9999_EEG_CGX_2023_01_01T00.00.00.csv"))
    assert record.family == "CGX" and record.device == "CGX"
    assert record.signal.shape == (200, 29)
    assert set(record.extras) == {"A2", "ExG 1", "ExG 2"}
    assert record.nominal_rate == 500.0
    assert record.interpolated is None


def test_column_table_lists_every_header_column_once(tmp_path):
    path = write_emotiv(tmp_path / "UB9999_EPOCX_1_2023.01.01T00.00.00+01.00.md.pm.bp.csv")
    table = modality.column_table(path)
    assert [row["column"] for row in table] == EMOTIV_COLUMNS
    assert all(row["destination"] != UNCLASSIFIED for row in table)
