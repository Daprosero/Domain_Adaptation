"""Experiments plan v50, Secs. 4.2, 4.5, 7.3 and 7.7: selection by nested cross-validation.

One Optuna study per strategy, transfer and outer fold; each trial is scored on the three
inner splits of the outer fold's non-test folds, per epoch, by the balanced accuracy of the
subject-level counts pooled over the splits; the label-free rule pools the target
predictions the same way. Every number here is scripted or synthetic, never a measured
result.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import assignment, bag_data, partition, search
from MIL_CREDA_EEG import two_phase as tp

LEVEL = assignment.Transfer("level", "L1", "L11")
SCHEDULE = {"epochs1": 1, "epochs2": 3, "bagsPerStep": 2}


def _bagset(root, rotations=None) -> bag_data.BagSet:
    """24 subjects over 4 folds, L1 and L11 holders (as the search tests build it)."""
    rng = np.random.default_rng(0)
    bags = []
    for i in range(24):
        subject, fold, label = f"UB{i:04d}", i % 4, ("yes", "no")[(i // 4) % 2]
        for condition in {0: ("L1",), 1: ("L11",), 2: ("L1", "L11")}[i % 3]:
            path = root / f"{subject}_{condition}.npz"
            np.savez(path, signal=rng.normal(size=(3, 3, 512)).astype(np.float32))
            bags.append(bag_data.Bag(subject=subject, condition=condition,
                                     record_id=f"{subject}-{condition}", label=label, sex="M",
                                     fold=fold, device="CGX", channels=("a", "b", "c"),
                                     n_windows=3, path=path))
    rotations = rotations or {str(r): partition.rotation_roles(r) for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256="x" * 64, rotations=rotations)


def _probability(subject: str, params, epoch: int) -> float:
    digest = hashlib.sha256(f"{subject}:{params['lr2']:.12g}:{epoch}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _scripted_trainer(seen: list):
    """A phase-2 trainer whose predictions depend on the subject, the trial and the epoch,
    never on a label; it records which subjects each call validated on."""

    def fake(strategy, data, rotation, seed, record, params, schedule, device="cpu", *,
             label_free=False):
        d = data.rotations[rotation]
        seen.append({"rotation": rotation, "valid": [b.subject for b in d["target_valid"]],
                     "train": [b.subject for b in d["target_train"]]})
        validation, history = [], []
        for e in range(schedule["epochs2"]):
            p = [_probability(b.subject, params, e) for b in d["target_valid"]]
            G = torch.tensor([[1 - x, x] for x in p], dtype=torch.float64)
            validation.append(tp.validation_record(e, d["target_valid"], G))
            history.append({"epoch": e, "val_balanced_accuracy": float("nan"),
                            "label_free_score": float("nan"), "loss": 0.0})
        return {"result": {"checkpoint": {"sigma": 1.0}, "history": history,
                           "validation": validation, "best_epoch": len(history) - 1,
                           "epochs": len(history)}}

    return fake


def _study(tmp_path, data, rotation, seen, monkeypatch, n_trials=4, name="s"):
    monkeypatch.setattr(search, "train_phase_two", _scripted_trainer(seen))
    monkeypatch.setattr(search, "_check_phase_one_record", lambda *a, **k: None)
    return search.run_phase_two_study("MIL-CREDA", data, rotation, ({}, {"sigma": 1.0}),
                                      SCHEDULE, root=tmp_path / name, n_trials=n_trials)


def _flip(data, subjects):
    """The same search data with the labels of `subjects` flipped everywhere."""
    def bags(rows):
        return {k: [tp.BagTensor(b.subject, b.condition, 1 - b.label, b.windows)
                    if b.subject in subjects else b for b in v] for k, v in rows.items()}

    return search.SearchData(transfer=data.transfer, partition_sha256=data.partition_sha256,
                             folds=data.folds,
                             rotations={q: bags(v) for q, v in data.rotations.items()},
                             inner={k: bags(v) for k, v in data.inner.items()})


def test_the_labels_of_r_test_subjects_cannot_change_r_selection(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    data = search.search_data(LEVEL, bags)
    folds = bags.folds()
    for r in range(4):
        seen = []
        first = _study(tmp_path, data, r, seen, monkeypatch, name=f"a{r}")
        test = {s for s, f in folds.items() if f == r}
        # the study of outer fold r never trains or validates on its test subjects ...
        assert seen and all(not (set(c["valid"]) | set(c["train"])) & test for c in seen)
        assert {c["rotation"] for c in seen} == {r}
        flipped_data = _flip(data, test)
        flipped = _study(tmp_path, flipped_data, r, [], monkeypatch, name=f"f{r}")
        for rule in search.RULES:
            before, after = search.select(first, rule), search.select(flipped, rule)
            assert (after["trial"], after["epoch"], after["score"]) == \
                (before["trial"], before["epoch"], before["score"])
        # ... while the same flip reaches the study of another outer fold, whose F holds them
        other = (r + 1) % 4
        base = _study(tmp_path, data, other, [], monkeypatch, name=f"o{r}")
        moved = _study(tmp_path, flipped_data, other, [], monkeypatch, name=f"m{r}")
        assert [t.user_attrs["pooledBalancedAccuracy"] for t in base.trials] != \
            [t.user_attrs["pooledBalancedAccuracy"] for t in moved.trials]


def test_the_objective_is_the_best_epochs_pooled_balanced_accuracy_over_the_inner_splits(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    seen = []
    study = _study(tmp_path, data, 2, seen, monkeypatch, n_trials=3)
    assert len(seen) == 3 * 3                              # three inner splits per trial
    for t in study.trials:
        series = t.user_attrs["pooledBalancedAccuracy"]
        assert len(series) == SCHEDULE["epochs2"]
        assert t.value == max(series) and t.user_attrs["bestEpoch"] == int(np.argmax(series))
        assert t.user_attrs["epochs"] == t.user_attrs["bestEpoch"] + 1
    # recomputed by hand: pooled counts over the three validating folds, epoch by epoch
    t = study.trials[0]
    labels, by_epoch = {}, {e: {} for e in range(SCHEDULE["epochs2"])}
    for i in range(3):
        for b in data.inner[(2, i)]["target_valid"]:
            labels[b.subject] = b.label
            for e in by_epoch:
                by_epoch[e][b.subject] = int(_probability(b.subject, t.params, e) > 0.5)
    validated = sorted(labels)
    assert validated == sorted(b.subject for b in data.rotations[2]["target_train"])
    expected = [tp.balanced_accuracy([labels[s] for s in validated],
                                     [by_epoch[e][s] for s in validated]) for e in by_epoch]
    assert t.user_attrs["pooledBalancedAccuracy"] == pytest.approx(expected)


def test_the_label_free_score_is_recorded_per_trial_and_epoch(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    study = _study(tmp_path, data, 1, [], monkeypatch, n_trials=3)
    every = {b.subject for i in range(3) for b in data.inner[(1, i)]["target_valid"]}
    flipped = _study(tmp_path, _flip(data, every), 1, [], monkeypatch, n_trials=3, name="f")
    for a, b in zip(study.trials, flipped.trials):
        assert len(a.user_attrs["pooledLabelFree"]) == SCHEDULE["epochs2"]
        assert a.user_attrs["pooledLabelFree"] == b.user_attrs["pooledLabelFree"]
        assert a.user_attrs["label_free_score"] == max(a.user_attrs["pooledLabelFree"])
        assert a.user_attrs["labelFreeBestEpoch"] == int(np.argmax(a.user_attrs["pooledLabelFree"]))
    assert [t.user_attrs["pooledBalancedAccuracy"] for t in study.trials] != \
        [t.user_attrs["pooledBalancedAccuracy"] for t in flipped.trials]
    free, again = search.select(study, "label-free"), search.select(flipped, "label-free")
    assert (free["trial"], free["epoch"]) == (again["trial"], again["epoch"])
    assert math.isfinite(free["score"])


def test_the_study_records_its_outer_fold_and_inner_splits(tmp_path, monkeypatch):
    # The inner splits are the ones the partition document stores, read, never rebuilt.
    v40 = {str(r): {"test_fold": r, "validation_fold": (r + 1) % 4,
                    "train_folds": [(r + 2) % 4, (r + 3) % 4]} for r in range(4)}
    (tmp_path / "v40").mkdir()
    with pytest.raises(assignment.AssignmentError, match="inner splits"):
        search.search_data(LEVEL, _bagset(tmp_path / "v40", v40))
    rotations = {str(r): partition.rotation_roles(r) for r in range(4)}
    data = search.search_data(LEVEL, _bagset(tmp_path, rotations))
    assert data.folds == {r: {"test_fold": r, "inner_splits": rotations[str(r)]["inner_splits"],
                              "inner_validation_folds": [f for f in range(4) if f != r]}
                          for r in range(4)}
    study = _study(tmp_path, data, 3, [], monkeypatch, n_trials=1)
    attrs = json.loads(json.dumps(study.user_attrs))
    assert attrs["outerFold"] == 3 and attrs["testFold"] == 3
    assert attrs["innerSplits"] == rotations["3"]["inner_splits"]
    chosen = search.select(study)
    assert chosen["testFold"] == chosen["excludedRotation"] == 3
    assert chosen["selectionRotations"] == [0, 1, 2]
