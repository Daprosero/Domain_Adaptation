"""Experiments plan v50, Sec. 4.2: the four-fold partition, generated once and frozen."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from MIL_CREDA_EEG import partition, preprocess

REPOSITORY = Path(__file__).resolve().parents[1]
#: sha256 of the five-fold file frozen at 7a75dd8; it is kept, never rewritten.
FIVE_FOLD_FILE_SHA256 = "8a723bc4e04c4e5c7753c9c4fbff1a15dd026fd6d34984848da79bafe19d32cb"
#: sha256 of the v40 four-fold file (`folds_4.json`) and its own partition hash; it is kept,
#: never rewritten, and replaced by the v50 regeneration under a new name.
V40_FILE_SHA256 = "ee46d4d43eaa54fac14b9fa48b35df90c32590cd8b7c1a0239bb882a2af2fcdc"
V40_PARTITION_SHA256 = "abc9012bad78eaaa542c08a1e38e2f35d110e00f33739d76853624297d8735d1"


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
    assert sorted(tests) == list(range(partition.N_FOLDS))
    for r in range(partition.N_FOLDS):
        rotation = rotations[str(r)]
        assert rotation["test_fold"] == r
        assert rotation["non_test_folds"] == [k for k in range(partition.N_FOLDS) if k != r]
        roles = rotation["test"] + rotation["non_test"]
        assert sorted(roles) == sorted(folds)               # disjoint and complete
        assert all(folds[s] == r for s in rotation["test"])
        assert all(folds[s] in rotation["non_test_folds"] for s in rotation["non_test"])
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


# ------------------------------------------------------------- plan v50: four folds


def _rotations_store_one_test_fold_and_three_non_test_folds_each_validated_once(doc):
    rotations = doc["rotations"]
    assert sorted(rotations) == ["0", "1", "2", "3"]
    assert Counter(v["test_fold"] for v in rotations.values()) == Counter(range(4))
    for v in rotations.values():
        test, non_test = v["test_fold"], v["non_test_folds"]
        assert non_test == [k for k in range(4) if k != test]
        assert "validation_fold" not in v and "train_folds" not in v
        assert "validation" not in v and "train" not in v
        splits = v["inner_splits"]
        assert [s["inner"] for s in splits] == [0, 1, 2]
        assert [s["validation_fold"] for s in splits] == non_test     # each validates once
        for s in splits:
            assert s["train_folds"] == [k for k in non_test if k != s["validation_fold"]]
            assert test not in (s["validation_fold"], *s["train_folds"])
        assert v["non_test"] == sorted(x for x, k in doc["folds"].items() if k in non_test)


def test_four_folds_rotate_with_one_test_fold_and_three_non_test_folds_each_validated_once():
    assert partition.N_FOLDS == 4
    doc = partition.build_partition(_survivors())
    assert doc["n_folds"] == 4 and set(doc["folds"].values()) == {0, 1, 2, 3}
    assert "v50" in doc["plan"] and doc["format"] == "balladeer-folds/v2"
    _rotations_store_one_test_fold_and_three_non_test_folds_each_validated_once(doc)


def test_the_fold_count_is_a_parameter_recorded_in_the_document():
    five = partition.build_partition(_survivors(), n_folds=5)
    assert five["n_folds"] == 5 and len(five["rotations"]) == 5
    assert set(five["folds"].values()) == set(range(5))
    assert five["partition_sha256"] != partition.build_partition(_survivors())["partition_sha256"]


def test_the_current_partition_is_one_declared_four_fold_file():
    assert partition.PARTITION_RELATIVE == Path("MIL-CREDA") / "Partition" / "folds_4_v50.json"
    assert partition.FIVE_FOLD_RELATIVE == Path("MIL-CREDA") / "Partition" / "folds.json"
    assert partition.V40_RELATIVE == Path("MIL-CREDA") / "Partition" / "folds_4.json"
    assert preprocess.PARTITION_RELATIVE == partition.PARTITION_RELATIVE
    doc = partition.load_partition(REPOSITORY / partition.PARTITION_RELATIVE)
    assert partition.check_current(doc) is doc


def test_the_committed_partition_stores_the_v50_rotations():
    doc = partition.load_partition(REPOSITORY / partition.PARTITION_RELATIVE)
    assert "v50" in doc["plan"]
    _rotations_store_one_test_fold_and_three_non_test_folds_each_validated_once(doc)


def test_the_v50_partition_keeps_the_fold_of_every_subject_of_the_v40_one():
    v40 = partition.load_partition(REPOSITORY / partition.V40_RELATIVE)
    v50 = partition.load_partition(REPOSITORY / partition.PARTITION_RELATIVE)
    assert v50["folds"] == v40["folds"]
    assert v50["fingerprint"]["survivors"] == v40["fingerprint"]["survivors"]
    assert v50["fingerprint"]["seed"] == v40["fingerprint"]["seed"] == partition.PARTITION_SEED
    assert v50["partition_sha256"] != V40_PARTITION_SHA256
    for r, rotation in v40["rotations"].items():
        assert v50["rotations"][r]["test"] == rotation["test"]
        assert v50["rotations"][r]["non_test"] == sorted(rotation["validation"]
                                                         + rotation["train"])


def test_the_v40_four_fold_file_is_kept_and_refused_as_current(tmp_path):
    path = REPOSITORY / partition.V40_RELATIVE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V40_FILE_SHA256
    doc = partition.load_partition(path)                 # its own hash still holds
    assert doc["partition_sha256"] == V40_PARTITION_SHA256
    with pytest.raises(partition.PartitionError, match="balladeer-folds/v1"):
        partition.check_current(doc)
    copy = tmp_path / path.name
    shutil.copyfile(path, copy)
    survivors = pd.DataFrame(doc["fingerprint"]["survivors"])
    with pytest.raises(partition.PartitionChanged, match="versión del código"):
        partition.freeze_partition(survivors, copy, seed=doc["fingerprint"]["seed"])
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == V40_FILE_SHA256


def test_a_result_bound_to_the_v40_partition_is_rejected_against_the_current_one():
    current = REPOSITORY / partition.PARTITION_RELATIVE
    with pytest.raises(partition.PartitionMismatch, match=V40_PARTITION_SHA256):
        partition.check_result({"partition_sha256": V40_PARTITION_SHA256}, current)


def test_the_five_fold_file_is_kept_and_refused_as_current(tmp_path):
    five = REPOSITORY / partition.FIVE_FOLD_RELATIVE
    assert hashlib.sha256(five.read_bytes()).hexdigest() == FIVE_FOLD_FILE_SHA256
    doc = partition.load_partition(five)                 # its own hash still holds
    with pytest.raises(partition.PartitionError, match="5 folds"):
        partition.check_current(doc)
    copy = tmp_path / "folds.json"
    shutil.copyfile(five, copy)
    survivors = pd.DataFrame(doc["fingerprint"]["survivors"])
    with pytest.raises(partition.PartitionChanged, match="versión del código"):
        partition.freeze_partition(survivors, copy, seed=doc["fingerprint"]["seed"])
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == FIVE_FOLD_FILE_SHA256


def test_regenerating_under_a_new_name_never_touches_the_five_fold_file(tmp_path):
    five = REPOSITORY / partition.FIVE_FOLD_RELATIVE
    copy = tmp_path / "folds.json"
    shutil.copyfile(five, copy)
    survivors = pd.DataFrame(partition.load_partition(five)["fingerprint"]["survivors"])
    result = partition.freeze_partition(survivors, copy, regenerate=True,
                                        new_name="folds_4.json")
    assert result["status"] == "regenerated" and result["path"] == tmp_path / "folds_4.json"
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == FIVE_FOLD_FILE_SHA256
    assert partition.load_partition(result["path"])["n_folds"] == 4


def test_the_committed_four_fold_partition_reruns_byte_for_byte(tmp_path):
    path = REPOSITORY / partition.PARTITION_RELATIVE
    stored = partition.load_partition(path)
    survivors = pd.DataFrame(stored["fingerprint"]["survivors"])
    seed = stored["fingerprint"]["seed"]
    assert seed == partition.PARTITION_SEED
    assert partition.serialize(partition.build_partition(survivors, seed)) == path.read_bytes()
    copy = tmp_path / path.name
    shutil.copyfile(path, copy)
    assert partition.freeze_partition(survivors, copy, seed=seed)["status"] == "unchanged"
    # The five-fold file's survivors are the current file's: only the rule changed.
    five = partition.load_partition(REPOSITORY / partition.FIVE_FOLD_RELATIVE)
    assert five["fingerprint"]["survivors"] == stored["fingerprint"]["survivors"]
