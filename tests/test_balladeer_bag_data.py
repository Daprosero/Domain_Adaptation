"""Experiments plan v40, Secs. 3.2 and 4.2: a bag is one subject in one condition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MIL_CREDA_EEG import bag_data, partition

CGX_CH = ["Fp1", "Fp2", "O1"]
EMO_CH = ["AF3", "AF4", "O1"]


def _write_npz(root: Path, record_id: str, channels: list[str], n_windows: int,
               seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    relative = f"signals/{record_id}.npz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ch = len(channels)
    np.savez(path,
             signal=rng.normal(size=(n_windows, n_ch, 512)).astype(np.float32),
             window_index=np.arange(1, n_windows + 1),
             integrity_mark=np.zeros(n_windows, bool),
             flat=np.zeros((n_windows, n_ch), bool),
             saturated=np.zeros((n_windows, n_ch), bool),
             contact_bad=np.zeros((n_windows, n_ch), bool),
             frontal_ptp=np.ones(n_windows),
             channels=np.array(channels), record_id=np.array(record_id))
    return relative


def _fixture(root: Path, extra_rows: list[tuple] = (),
             n_folds: int = partition.N_FOLDS) -> tuple[pd.DataFrame, Path]:
    """Six subjects; UB0003's Robots record is excluded; one EPOC+ record."""
    rows = [
        # subject, condition, device, label, sex, excluded
        ("UB0001", "Slackline-L6/CGX", "CGX", "yes", "M", False),
        ("UB0001", "Robots/EMOTIV", "EPOCX", "yes", "M", False),
        ("UB0002", "Slackline-L1/CGX", "CGX", "no", "F", False),
        ("UB0002", "Slackline-L11/CGX", "CGX", "no", "F", False),
        ("UB0003", "Slackline-L6/CGX", "CGX", "yes", "F", False),
        ("UB0003", "Robots/EMOTIV", "EPOCX", "yes", "F", True),
        ("UB0004", "Cognifit/EMOTIV", "EPOCX", "no", "M", False),
        ("UB0005", "Robots/EMOTIV", "EPOCPLUS", "yes", "M", True),
        ("UB0006", "Robots/EMOTIV", "EPOCX", "no", "M", False),
        *extra_rows,
    ]
    records = []
    for k, (subject, condition, device, label, sex, excluded) in enumerate(rows):
        rid = f"{subject}__{condition.split('/')[0]}__{device}"
        family = "CGX" if device == "CGX" else "EMOTIV"
        npz = None
        if not excluded:
            npz = _write_npz(root, rid, CGX_CH if family == "CGX" else EMO_CH, 70 + k, k)
        records.append({"record_id": rid, "subject": subject, "condition": condition,
                        "family": family, "device": device, "label": label, "sex": sex,
                        "excluded": excluded, "n_windows": 70 + k, "npz": npz})
    manifest = pd.DataFrame(records)
    manifest.to_csv(root / "manifest.csv", index=False)
    part = root / "folds.json"
    doc = partition.build_partition(partition.survivors(manifest), n_folds=n_folds)
    part.write_bytes(partition.serialize(doc))
    return manifest, part


def test_bags_are_built_only_from_non_excluded_records(tmp_path):
    manifest, part = _fixture(tmp_path)
    bags = bag_data.load_bags(tmp_path, part)
    kept = set(manifest.loc[~manifest.excluded, "record_id"])
    assert {b.record_id for b in bags.bags} == kept
    assert ("UB0003", "Robots") not in {(b.subject, b.condition) for b in bags.bags}
    assert all(b.device != "EPOCPLUS" for b in bags.bags)


def test_a_bag_is_one_subject_in_one_condition(tmp_path):
    _, part = _fixture(tmp_path)
    bags = bag_data.load_bags(tmp_path, part)
    keys = [(b.subject, b.condition) for b in bags.bags]
    assert len(keys) == len(set(keys))
    assert bags.get("UB0001", "L6").condition == "L6"
    assert bags.get("UB0001", "Robots").device == "EPOCX"


def test_two_records_for_one_bag_are_refused(tmp_path):
    _, part = _fixture(tmp_path, [("UB0001", "Slackline-L6/CGX", "CGX", "yes", "M", False)])
    with pytest.raises(bag_data.BagDataError, match="UB0001"):
        bag_data.load_bags(tmp_path, part)


def test_a_bag_carries_its_windows_marks_label_sex_and_fold(tmp_path):
    manifest, part = _fixture(tmp_path)
    doc = partition.load_partition(part)
    bags = bag_data.load_bags(tmp_path, part)
    bag = bags.get("UB0002", "L1")
    assert (bag.label, bag.sex, bag.fold) == ("no", "F", doc["folds"]["UB0002"])
    windows = bag.windows()
    assert windows.shape == (bag.n_windows, len(CGX_CH), 512)
    assert bag.channels == tuple(CGX_CH)
    marks = bag.marks()
    assert "signal" not in marks and "channels" not in marks
    assert marks["flat"].shape == (bag.n_windows, len(CGX_CH))
    assert marks["integrity_mark"].shape == (bag.n_windows,)


def test_a_bag_carries_the_fold_of_the_frozen_partition(tmp_path):
    _, part = _fixture(tmp_path)
    doc = partition.load_partition(part)
    bags = bag_data.load_bags(tmp_path, part)
    assert bags.partition_sha256 == doc["partition_sha256"]
    for bag in bags.bags:
        assert bag.fold == doc["folds"][bag.subject]
    assert bags.rotations == doc["rotations"]


def test_a_partition_hash_mismatch_is_refused(tmp_path):
    _, part = _fixture(tmp_path)
    with pytest.raises(partition.PartitionMismatch):
        bag_data.load_bags(tmp_path, part, expected_partition_sha256="0" * 64)
    good = partition.current_partition_hash(part)
    assert bag_data.load_bags(tmp_path, part, expected_partition_sha256=good).bags


def test_a_subject_outside_the_partition_is_refused(tmp_path):
    manifest, part = _fixture(tmp_path)
    extra = manifest.iloc[[0]].assign(subject="UB0099", record_id="UB0099__x__CGX",
                                      npz=_write_npz(tmp_path, "UB0099__x__CGX", CGX_CH, 70))
    pd.concat([manifest, extra]).to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(bag_data.BagDataError, match="UB0099"):
        bag_data.load_bags(tmp_path, part)


def test_arm_conditions_are_the_declared_ones():
    assert bag_data.ARM_CONDITIONS == {"level": ("L1", "L6", "L11"),
                                       "task": ("Robots", "CogniFit"),
                                       "device": ("L6", "Robots")}
    assert bag_data.CONDITIONS["L6"] == ("Slackline-L6/CGX", "CGX")
    assert bag_data.CONDITIONS["Robots"] == ("Robots/EMOTIV", "EPOCX")
    assert bag_data.CONDITIONS["CogniFit"] == ("Cognifit/EMOTIV", "EPOCX")
    assert all(device != "EPOCPLUS" for _, device in bag_data.CONDITIONS.values())


def test_the_holders_of_a_condition_are_its_bag_subjects(tmp_path):
    _, part = _fixture(tmp_path)
    bags = bag_data.load_bags(tmp_path, part)
    assert bags.holders("Robots") == {"UB0001", "UB0006"}
    assert bags.holders("L6") == {"UB0001", "UB0003"}


def test_the_bags_read_the_current_four_fold_partition_by_default():
    assert bag_data.default_partition_path() == (bag_data.REPOSITORY / "MIL-CREDA" / "Partition"
                                                 / "folds_4_v50.json")
    assert bag_data.default_partition_path() == bag_data.REPOSITORY / partition.PARTITION_RELATIVE


def test_a_v40_format_partition_is_refused_as_not_current(tmp_path):
    _, part = _fixture(tmp_path)
    doc = json.loads(part.read_text(encoding="utf-8"))
    doc["format"] = "balladeer-folds/v1"
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    part.write_bytes(partition.serialize(doc))
    with pytest.raises(partition.PartitionError, match="balladeer-folds/v1"):
        bag_data.load_bags(tmp_path, part)


def test_a_five_fold_partition_is_refused_as_not_current(tmp_path):
    _, part = _fixture(tmp_path, n_folds=5)
    with pytest.raises(partition.PartitionError, match="5 folds"):
        bag_data.load_bags(tmp_path, part)
