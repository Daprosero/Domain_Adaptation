"""Experiments plan v05, Secs. 2.2, 2.4, 2.5 and 2.10: what preprocessing excludes, and its flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from MIL_CREDA_EEG import exclusions


def test_a_record_with_most_channels_dead_in_every_window_has_no_eeg():
    flat = np.zeros((10, 29), dtype=bool)
    saturated = np.zeros((10, 29), dtype=bool)
    flat[:, :20] = True
    saturated[:, 20:25] = True            # 25 of 29 dead, as UB0058 Slackline-L6
    assert exclusions.dead_channels(flat, saturated) == 25
    assert exclusions.record_without_eeg(25, 29) is True


def test_a_channel_dead_in_all_but_one_window_is_not_dead():
    flat = np.ones((10, 29), dtype=bool)
    flat[3, :] = False
    assert exclusions.dead_channels(flat, np.zeros_like(flat)) == 0


def test_a_minority_of_dead_channels_is_a_channel_defect_not_a_record_without_eeg():
    assert exclusions.record_without_eeg(14, 29) is False   # 14 is not more than half
    assert exclusions.record_without_eeg(7, 14) is False
    assert exclusions.record_without_eeg(8, 14) is True


def test_reasons_follow_plan_order_and_only_the_three_declared_rules_exclude():
    assert exclusions.exclusion_reasons(False, 0, 29, None) == []
    assert exclusions.exclusion_reasons(True, 0, 14, True) == [
        "2.2 rate_hard_defect", "2.5 majority_bad_contact"]
    assert exclusions.exclusion_reasons(False, 25, 29, None) == ["2.4 no_eeg"]
    assert exclusions.exclusion_reasons(False, 3, 14, False) == []


def _manifest() -> pd.DataFrame:
    rows = [
        # record_id, subject, family, reasons, label
        ("r1", "S1", "CGX", [], "yes"),
        ("r2", "S1", "CGX", ["2.4 no_eeg"], "yes"),
        ("r3", "S2", "CGX", ["2.4 no_eeg"], "undetermined"),
        ("r4", "S3", "CGX", [], "undetermined"),
        ("e1", "S4", "EMOTIV", ["2.2 rate_hard_defect", "2.5 majority_bad_contact"], "no"),
        ("e2", "S5", "EMOTIV", ["2.5 majority_bad_contact"], "yes"),
        ("e3", "S5", "EMOTIV", [], "yes"),
        ("e4", "S6", "EMOTIV", [], "no"),
    ]
    frame = pd.DataFrame(rows, columns=["record_id", "subject", "family", "reasons", "label"])
    return exclusions.finalize_manifest(frame)


def test_the_excluded_set_is_exactly_the_records_with_a_reason():
    manifest = _manifest()
    assert manifest.loc[manifest["excluded"], "record_id"].tolist() == ["r2", "r3", "e1", "e2"]
    assert manifest.set_index("record_id").loc["e1", "exclusion_step"] == "2.2 tasa"


def test_undetermined_subjects_are_flagged_and_never_dropped():
    manifest = _manifest().set_index("record_id")
    assert bool(manifest.loc["r4", "undetermined"]) is True
    assert bool(manifest.loc["r4", "excluded"]) is False
    assert bool(manifest.loc["r3", "undetermined"]) is True
    assert manifest.loc["r3", "exclusion_step"] == "2.4 registro sin EEG"


def test_flow_table_counts_records_and_subjects_left_after_each_step():
    flow = exclusions.flow_table(_manifest(), "EMOTIV").set_index("step")
    assert flow.loc["inicial", "records_left"] == 4
    assert flow.loc["inicial", "subjects_left"] == 3
    assert flow.loc["2.2 tasa", "records_excluded"] == 1
    assert flow.loc["2.2 tasa", "subjects_left"] == 2
    # e1 already left at 2.2; 2.5 only counts what it newly removes.
    assert flow.loc["2.5 contacto", "records_excluded"] == 1
    assert flow.loc["2.5 contacto", "records_meeting_rule"] == 2
    assert flow.loc["final", "records_left"] == 2
    assert flow.loc["final", "subjects_left"] == 2
    assert flow.loc["final", "undetermined_subjects_flagged"] == 0
    cgx = exclusions.flow_table(_manifest(), "CGX").set_index("step")
    assert cgx.loc["final", "subjects_left"] == 2
    assert cgx.loc["final", "undetermined_subjects_flagged"] == 1


def test_hard_defects_and_incompatibilities_are_crossed_apart():
    demo = pd.DataFrame({"subject": ["S1", "S2", "S3", "S4", "S5", "S6"],
                         "label": ["yes", "undetermined", "undetermined", "no", "yes", "no"],
                         "sex": ["M", "F", "M", "F", "M", "F"],
                         "age": [8, 9, 10, 11, 12, 13]})
    crosses = exclusions.exclusion_crosses(_manifest(), demo)
    kinds = set(crosses["kind"])
    assert kinds == {"hard_defect", "incompatibility"}
    hard = crosses[(crosses["kind"] == "hard_defect") & (crosses["family"] == "EMOTIV")
                   & (crosses["by"] == "label")].set_index("value")
    assert hard.loc["yes", "excluded"] == 1 and hard.loc["no", "excluded"] == 1
    incompat = crosses[crosses["kind"] == "incompatibility"]
    assert (incompat["excluded"].fillna(0) == 0).all()


def test_the_focus_score_is_read_under_either_emotiv_name(tmp_path):
    for name in ("PM.Focus.Scaled", "PM.Attention.Scaled"):
        path = tmp_path / f"UB0001_EPOCX_1_{name}.csv"
        path.write_text("title:x, sampling rate:eeg_128\n"
                        f"Timestamp,EEG.AF3,{name}\n0,1,\n1,2,0.5\n", encoding="utf-8")
        values, column = exclusions.read_focus(str(path))
        assert column == name
        assert np.isnan(values[0]) and values[1] == 0.5
    bare = tmp_path / "UB0002_EPOCX_1_bare.csv"
    bare.write_text("title:x\nTimestamp,EEG.AF3\n0,1\n", encoding="utf-8")
    assert exclusions.read_focus(str(bare)) == (None, None)
