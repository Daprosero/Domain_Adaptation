"""Experiments plan v02, Sec. 2.10: the defect log, on synthetic records."""

from __future__ import annotations

import json

import pandas as pd

from MIL_CREDA_EEG import defect_log
from test_balladeer_modality import write_cgx, write_emotiv

DEMO = pd.DataFrame([
    {"subject": "S1", "label": "yes", "sex": "M", "age": 9, "group": 1},
    {"subject": "S2", "label": "yes", "sex": "M", "age": 10, "group": 1},
    {"subject": "S3", "label": "no", "sex": "F", "age": 11, "group": 2},
    {"subject": "S4", "label": "no", "sex": "F", "age": 12, "group": 2},
    {"subject": "S5", "label": "undetermined", "sex": "M", "age": 8, "group": 2},
])


def _records() -> pd.DataFrame:
    rows = [
        # subject, record, readable, rate hard, gap, channel marks, missing, below
        ("S1", "r1", True, False, False, 0, 0, True),
        ("S1", "r2", True, True, False, 1, 0, False),
        ("S2", "r3", False, None, None, None, None, None),
        ("S3", "r4", True, False, True, 0, 1, False),
        ("S4", "r5", True, False, False, 0, 0, False),
        ("S5", "r6", True, False, False, 2, 0, True),
    ]
    frame = pd.DataFrame(rows, columns=[
        "subject", "record_id", "readable", "rate_hard_defect", "gap_exceeds_window",
        "n_marked_channels", "n_missing_vs_cohort", "quality_any_window_below"])
    frame["family"] = "EMOTIV"
    return frame


def test_hard_defects_and_incompatibilities_are_kept_apart():
    frame = defect_log.classify_records(_records())
    by_id = frame.set_index("record_id")
    assert by_id.loc["r3", "hard_defects"] == ["unreadable"]
    assert by_id.loc["r2", "hard_defects"] == ["rate_hard_defect"]
    assert by_id.loc["r4", "hard_defects"] == []
    assert by_id.loc["r4", "incompatibilities"] == ["channel_missing_vs_cohort"]
    assert by_id.loc["r2", "incompatibilities"] == []


def test_flow_table_counts_subjects_left_after_declared_hard_defects():
    flow = defect_log.flow_table(defect_log.classify_records(_records()), "EMOTIV")
    flow = flow.set_index("step")
    assert flow.loc["inicial", "subjects_left"] == 5
    assert flow.loc["lectura", "records_hard_defect"] == 1
    assert flow.loc["lectura", "subjects_left"] == 4      # S2 had one record only
    assert flow.loc["2.2 tasa", "records_hard_defect"] == 1
    assert flow.loc["2.2 tasa", "subjects_left"] == 4     # S1 keeps r1
    assert flow.loc["2.3 integridad", "records_marked"] == 1
    assert flow.loc["2.3 integridad", "subjects_left"] == 4  # a mark, not an exclusion
    assert flow.loc["2.4 canales", "records_marked"] == 3
    assert flow.loc["2.5 contacto", "records_marked"] == 2


def test_cross_with_label_counts_marked_subjects_per_group():
    marked = pd.Series({"S1": True, "S2": True, "S3": False, "S4": False, "S5": True})
    cross = defect_log.cross_with_demographics(marked, DEMO)
    label = cross["label"].set_index("label")
    assert label.loc["yes", "marked"] == 2 and label.loc["yes", "unmarked"] == 0
    assert label.loc["no", "marked"] == 0 and label.loc["no", "unmarked"] == 2
    assert label.loc["undetermined", "marked"] == 1
    assert 0 <= cross["tests"]["label_yes_vs_no_fisher_p"] <= 1
    assert "sex_fisher_p" in cross["tests"] and "age_mannwhitney_p" in cross["tests"]


def test_continuous_cross_compares_label_sex_and_age():
    values = pd.Series({"S1": 3.0, "S2": 4.0, "S3": 0.0, "S4": 1.0, "S5": 2.0})
    result = defect_log.continuous_cross(values, DEMO)
    assert result["median_yes"] == 3.5 and result["median_no"] == 0.5
    assert result["age_spearman_rho"] < 0


def test_audit_record_of_an_unreadable_file_is_a_hard_defect_not_a_crash(tmp_path):
    path = tmp_path / "UB9_EPOCX_1_2023.01.01T00.00.00+01.00.md.pm.bp.csv"
    path.write_text("garbage\n", encoding="utf-8")
    record = defect_log.audit_record(str(path), {"record_id": "x", "task": "Robots",
                                                 "level": None}, {})
    assert record["readable"] is False and record["error"]


def test_audit_record_end_to_end_on_synthetic_files(tmp_path):
    emotiv = write_emotiv(tmp_path / "UB9_EPOCX_1_2023.01.01T00.00.00+01.00.md.pm.bp.csv",
                          n=128 * 12)
    record = defect_log.audit_record(str(emotiv), {"record_id": "e", "task": "Robots",
                                                   "level": None}, {})
    assert record["readable"] is True
    assert record["rate"]["nearest_menu_rate"] == 128.0
    assert record["integrity"]["n_lost"] == 0
    assert record["duration"]["nominal_s"] == 280.0
    assert set(record["channels_audit"]) == set(record["channels"])
    json.dumps(record)
    cgx = write_cgx(tmp_path / "UB9_EEG_CGX_2023_01_01T00.00.00.csv", n=500 * 9)
    record = defect_log.audit_record(str(cgx), {"record_id": "c", "task": "Slackline",
                                                "level": 1}, {})
    assert record["readable"] is True
    assert record["bipolar"]["ExG 1"]["constant"] is True
    assert record["reference"]["band_std_ratio_to_scalp"] > 0
    assert len(record["line_noise"]) == 29
    json.dumps(record)


def test_run_audit_reuses_its_cache(tmp_path):
    raw = tmp_path / "raw"
    session = raw / "UB0001" / "AttentionRobotsDesktop" / "1"
    session.mkdir(parents=True)
    write_emotiv(session / "UB0001_EPOCX_1_2023.01.01T00.00.00+01.00.md.pm.bp.csv",
                 n=128 * 10)
    out = tmp_path / "out"
    first = defect_log.run_audit(raw, out, workers=1)
    assert len(first) == 1
    cached = next((out / "records").glob("*.json"))
    cached.write_text(json.dumps({**json.loads(cached.read_text()), "marker": 1}))
    second = defect_log.run_audit(raw, out, workers=1)
    assert second[0]["marker"] == 1
