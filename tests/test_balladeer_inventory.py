"""Inventory of the BALLADEER tree, on a synthetic miniature of its layout."""

from __future__ import annotations

import json
from pathlib import Path

from MIL_CREDA_EEG import inventory


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_tree(root: Path) -> Path:
    demo = [
        {"user": "UB0001", "age": 9, "gender": 1, "group": 1, "diagnosed": "yes"},
        {"user": "UB0002", "age": 12, "gender": 2, "group": 2, "diagnosed": "no"},
        {"user": "UB0003", "age": 10, "gender": 1, "group": 2, "diagnosed": "undetermined"},
    ]
    _touch(root / "users_demographics.json", json.dumps(demo))
    _touch(root / "README.md")
    _touch(root / "UB0001" / "SlacklineLvl1" / "100" / "UB0001_EEG_CGX_2023_01_01T10.00.00.csv")
    _touch(root / "UB0001" / "SlacklineLvl1" / "100" / "UB0001_TAGS_2023_01_01T10.00.00.csv")
    _touch(root / "UB0001" / "SlacklineLvl6" / "101" / "UB0001_EEG_CGX_2023_01_01T10.10.00.csv")
    _touch(root / "UB0001" / "AttentionRobotsDesktop" / "200"
           / "UB0001_EPOCX_123_2023.01.02T10.00.00+01.00.md.pm.bp.csv")
    _touch(root / "UB0001" / "AttentionRobotsDesktop" / "200"
           / "UB0001_EYE_TRACKING_DATA_2023_01_02T10.00.00+01.00.csv")
    _touch(root / "UB0001" / "AttentionRobotsDesktop" / "200" / ".DS_Store")
    _touch(root / "UB0002" / "Cognifit" / "300"
           / "UB0002_EPOCPLUS_123_2023.01.03T10.00.00+01.00.md.pm.bp.csv")
    _touch(root / "UB0002" / "SlacklineLvl1" / "301" / "UB0002_EEG_CGX_2023_01_03T11.00.00.csv")
    _touch(root / "UB0003" / "Cognifit" / "400"
           / "UB0003_Cognifit_EPOCX_9_2023.01.04T10.00.00+01.00.md.pm.bp.csv")
    return root


def test_file_kinds_are_read_off_the_name():
    assert inventory.file_kind("UB1_EEG_CGX_2023.csv") == "EEG"
    assert inventory.file_kind("UB1_EPOCPLUS_9_2023.md.pm.bp.csv") == "EEG"
    assert inventory.file_kind("UB1_TAGS_CGX_2023.csv") == "TAGS"
    assert inventory.file_kind("UB1_GAME_DATA_2023.csv") == "GAME_DATA"
    assert inventory.file_kind("UB1_EYE_TRACKING_DATA_2023.csv") == "EYE_TRACKING"
    assert inventory.file_kind("strange.bin") == "OTHER"


def test_index_has_one_row_per_session_file_and_skips_finder_debris(tmp_path):
    index = inventory.build_index(make_tree(tmp_path))
    assert len(index) == 8
    assert ".DS_Store" not in set(index["file"])
    eeg = inventory.eeg_records(index)
    assert len(eeg) == 6
    assert set(eeg["condition"]) == {"Slackline-L1/CGX", "Slackline-L6/CGX",
                                     "Robots/EMOTIV", "Cognifit/EMOTIV"}
    assert eeg["record_id"].is_unique


def test_demographics_map_label_sex_and_group(tmp_path):
    demo = inventory.load_demographics(make_tree(tmp_path))
    row = demo.set_index("subject").loc["UB0002"]
    assert (row["label"], row["sex"], row["group"], row["age"]) == ("no", "F", 2, 12)


def test_indeterminate_count_and_the_group_it_landed_in(tmp_path):
    demo = inventory.load_demographics(make_tree(tmp_path))
    summary = inventory.indeterminate_summary(demo)
    assert summary["count"] == 1
    assert summary["by_group"] == {"2": 1}


def test_condition_overlap_counts_subjects_shared_by_two_conditions(tmp_path):
    eeg = inventory.eeg_records(inventory.build_index(make_tree(tmp_path)))
    overlap = inventory.condition_overlap(eeg)
    assert overlap.loc["Slackline-L1/CGX", "Slackline-L1/CGX"] == 2
    assert overlap.loc["Slackline-L1/CGX", "Slackline-L6/CGX"] == 1
    assert overlap.loc["Slackline-L1/CGX", "Cognifit/EMOTIV"] == 1
    assert overlap.loc["Robots/EMOTIV", "Cognifit/EMOTIV"] == 0
    assert (overlap.values == overlap.values.T).all()


def test_subjects_per_condition_and_device_match_the_tree(tmp_path):
    eeg = inventory.eeg_records(inventory.build_index(make_tree(tmp_path)))
    table = inventory.subjects_per_condition(eeg)
    assert table.loc["Cognifit", "EPOCX"] == 1
    assert table.loc["Cognifit", "EPOCPLUS"] == 1
    assert table.loc["Slackline-L1", "CGX"] == 2
