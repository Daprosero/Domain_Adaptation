"""Experiments plan v18, Sec. 4.2: the five-fold partition, generated once and frozen."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pytest

from MIL_CREDA_EEG import partition


def _survivors(n: int = 60) -> pd.DataFrame:
    labels = ["yes", "no"]
    sexes = ["M", "M", "F"]
    rows = [{"subject": f"UB{i:04d}", "label": labels[i % 2], "sex": sexes[i % 3]}
            for i in range(1, n + 1)]
    return pd.DataFrame(rows)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame([
        ("a1", "UB0001", "yes", "M", False),
        ("a2", "UB0001", "yes", "M", True),
        ("b1", "UB0002", "no", "F", True),        # every record excluded: not a survivor
        ("c1", "UB0003", "no", "M", False),
    ], columns=["record_id", "subject", "label", "sex", "excluded"])


def test_survivors_are_the_subjects_with_a_kept_record():
    survivors = partition.survivors(_manifest())
    assert survivors.to_dict("records") == [
        {"subject": "UB0001", "label": "yes", "sex": "M"},
        {"subject": "UB0003", "label": "no", "sex": "M"}]


def test_folds_are_stratified_by_label_and_sex_and_rotate():
    doc = partition.build_partition(_survivors())
    folds = doc["folds"]
    assert sorted(folds) == sorted(_survivors()["subject"])
    assert set(folds.values()) == set(range(partition.N_FOLDS))
    sizes = Counter(folds.values())
    assert max(sizes.values()) - min(sizes.values()) <= 1
    table = _survivors().assign(fold=lambda t: t["subject"].map(folds))
    for _, stratum in table.groupby(["label", "sex"]):
        per_fold = Counter(stratum["fold"])
        counts = [per_fold.get(k, 0) for k in range(partition.N_FOLDS)]
        assert max(counts) - min(counts) <= 1
    rotations = doc["rotations"]
    assert sorted(rotations) == [str(r) for r in range(partition.N_FOLDS)]
    tests = [rotations[str(r)]["test_fold"] for r in range(partition.N_FOLDS)]
    validations = [rotations[str(r)]["validation_fold"] for r in range(partition.N_FOLDS)]
    assert sorted(tests) == sorted(validations) == list(range(partition.N_FOLDS))
    for r in range(partition.N_FOLDS):
        rotation = rotations[str(r)]
        assert rotation["test_fold"] == r
        assert rotation["validation_fold"] == (r + 1) % partition.N_FOLDS
        assert sorted(rotation["train_folds"]) == sorted(
            set(range(partition.N_FOLDS)) - {r, (r + 1) % partition.N_FOLDS})
        roles = rotation["test"] + rotation["validation"] + rotation["train"]
        assert sorted(roles) == sorted(folds)               # disjoint and complete
        assert all(folds[s] == r for s in rotation["test"])
    assert doc["fingerprint"]["seed"] == partition.PARTITION_SEED
    assert doc["fingerprint"]["code_version"] == partition.code_version()
    assert doc["partition_sha256"] == partition.partition_hash(doc)


def test_two_runs_write_the_same_file_byte_for_byte(tmp_path):
    first, second = tmp_path / "a" / "folds.json", tmp_path / "b" / "folds.json"
    assert partition.freeze_partition(_survivors(), first)["status"] == "created"
    assert partition.freeze_partition(_survivors(), second)["status"] == "created"
    assert first.read_bytes() == second.read_bytes()
    before = first.stat().st_mtime_ns
    again = partition.freeze_partition(_survivors().sample(frac=1, random_state=3), first)
    assert again["status"] == "unchanged"
    assert first.stat().st_mtime_ns == before
    stored = json.loads(first.read_text(encoding="utf-8"))
    assert stored["partition_sha256"] == partition.current_partition_hash(first)


def test_a_changed_survivor_set_refuses_naming_it(tmp_path):
    path = tmp_path / "folds.json"
    partition.freeze_partition(_survivors(), path)
    original = path.read_bytes()
    fewer = _survivors().iloc[1:]
    with pytest.raises(partition.PartitionChanged, match="sobrevivientes") as error:
        partition.freeze_partition(fewer, path)
    assert "UB0001" in str(error.value)
    relabelled = _survivors()
    relabelled.loc[relabelled["subject"] == "UB0001", "sex"] = "F"
    with pytest.raises(partition.PartitionChanged, match="etiqueta o sexo"):
        partition.freeze_partition(relabelled, path)
    assert path.read_bytes() == original


def test_a_changed_seed_refuses_naming_it(tmp_path):
    path = tmp_path / "folds.json"
    partition.freeze_partition(_survivors(), path)
    with pytest.raises(partition.PartitionChanged, match="semilla") as error:
        partition.freeze_partition(_survivors(), path, seed=partition.PARTITION_SEED + 1)
    assert "sobrevivientes" not in str(error.value)


def test_regeneration_needs_the_flag_and_a_new_name_and_never_overwrites(tmp_path):
    path = tmp_path / "folds.json"
    partition.freeze_partition(_survivors(), path)
    original = path.read_bytes()
    fewer = _survivors().iloc[1:]
    with pytest.raises(partition.PartitionError, match="nombre nuevo"):
        partition.freeze_partition(fewer, path, regenerate=True)
    with pytest.raises(partition.PartitionError, match="bandera"):
        partition.freeze_partition(fewer, path, new_name="folds-v2.json")
    with pytest.raises(partition.PartitionError, match="sobrescrib"):
        partition.freeze_partition(fewer, path, regenerate=True, new_name="folds.json")
    result = partition.freeze_partition(fewer, path, regenerate=True, new_name="folds-v2.json")
    assert result["status"] == "regenerated"
    assert result["path"] == tmp_path / "folds-v2.json"
    assert path.read_bytes() == original
    with pytest.raises(partition.PartitionError, match="sobrescrib"):
        partition.freeze_partition(fewer, path, regenerate=True, new_name="folds-v2.json")


def test_a_result_with_another_partition_hash_is_rejected_on_load(tmp_path):
    path = tmp_path / "folds.json"
    partition.freeze_partition(_survivors(), path)
    current = partition.current_partition_hash(path)
    assert partition.check_result({"partition_sha256": current, "auc": 0.7}, path) == current
    with pytest.raises(partition.PartitionMismatch, match="0" * 64):
        partition.check_result({"partition_sha256": "0" * 64}, path)
    with pytest.raises(partition.PartitionMismatch, match="no registra"):
        partition.check_result({"auc": 0.7}, path)


def test_a_tampered_file_is_refused(tmp_path):
    path = tmp_path / "folds.json"
    partition.freeze_partition(_survivors(), path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["folds"]["UB0001"] = (doc["folds"]["UB0001"] + 1) % partition.N_FOLDS
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(partition.PartitionError, match="hash"):
        partition.current_partition_hash(path)
