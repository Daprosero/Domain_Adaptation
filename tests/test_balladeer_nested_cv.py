"""Experiments plan v50, Secs. 4.2, 4.5, 7.3 and 7.7: the nested cross-validation.

Per outer test fold k, F is the other three frozen folds; the three inner splits each
validate one fold of F and train on the other two. Every training runs a fixed number of
epochs (no early stopping); the search records validation per epoch and the refit runs
exactly E epochs with no validation set. The selection score is the balanced accuracy of
the subject-level confusion counts pooled over the three inner splits, per epoch; the
label-free rule pools the target predictions the same way and never reads a label.
"""

from __future__ import annotations

import math

import numpy as np
import optuna
import pytest
import torch

from MIL_CREDA_EEG import assignment, bag_data, partition, search
from MIL_CREDA_EEG import two_phase as tp
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel

CPU = torch.device("cpu")
LEVEL = assignment.Transfer("level", "L1", "L11")
SCHEDULE = {"epochs1": 3, "epochs2": 3, "bagsPerStep": 2}
PHASE_ONE = {"lr1": 1e-2, "wd1": 1e-4, "attentionGamma": 0.5, "attentionTemperature": 1.0,
             "sigmaMultiplier": 1.0}
HYPER = {**PHASE_ONE, "lr2": 1e-2, "wd2": 1e-4, "tauLocal": 1.0, "epochs1": 3, "epochs2": 3,
         "bagsPerStep": 4}


def _bagset(root, sha: str = "x" * 64) -> bag_data.BagSet:
    """24 subjects over 4 folds; L1 and L11 holders, some holding both (3 channels)."""
    rng = np.random.default_rng(0)
    bags = []
    for i in range(24):
        subject, fold, label = f"UB{i:04d}", i % 4, ("yes", "no")[(i // 4) % 2]
        held = {0: ("L1",), 1: ("L11",), 2: ("L1", "L11")}[i % 3]
        for condition in held:
            path = root / f"{subject}_{condition}.npz"
            np.savez(path, signal=rng.normal(size=(3, 3, 512)).astype(np.float32))
            bags.append(bag_data.Bag(subject=subject, condition=condition,
                                     record_id=f"{subject}-{condition}", label=label, sex="M",
                                     fold=fold, device="CGX", channels=("a", "b", "c"),
                                     n_windows=3, path=path))
    rotations = {str(r): partition.rotation_roles(r) for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256=sha, rotations=rotations)


def _bags(n: int, channels: int, seed: int, prefix: str) -> list[tp.BagTensor]:
    g = torch.Generator().manual_seed(seed)
    return [tp.BagTensor(f"{prefix}{i}", "c", i % 2,
                         torch.randn(3, channels, 512, generator=g)) for i in range(n)]


# ------------------------------------------------------------------ inner splits


def test_the_inner_splits_never_hold_the_test_fold_and_validate_each_non_test_fold_once():
    for k in range(partition.N_FOLDS):
        splits = assignment.inner_splits(partition.rotation_roles(k))
        assert splits == partition.rotation_roles(k)["inner_splits"]
        assert len(splits) == partition.N_FOLDS - 1 == 3
        non_test = sorted(f for f in range(partition.N_FOLDS) if f != k)
        assert sorted(s["validation_fold"] for s in splits) == non_test
        for s in splits:
            assert k not in s["train_folds"] and s["validation_fold"] != k
            assert sorted([*s["train_folds"], s["validation_fold"]]) == non_test
            assert len(s["train_folds"]) == 2


def test_inner_splits_that_do_not_validate_each_non_test_fold_once_are_refused():
    rotation = partition.rotation_roles(0)
    twice = {**rotation, "inner_splits": [{**s, "validation_fold": 1}
                                          for s in rotation["inner_splits"]]}
    with pytest.raises(assignment.AssignmentError, match="once"):
        assignment.inner_splits(twice)
    with pytest.raises(assignment.AssignmentError, match="inner splits"):
        assignment.inner_splits({"test_fold": 0, "validation_fold": 1, "train_folds": [2, 3]})


def test_the_inner_roles_validate_every_non_test_subject_exactly_once(tmp_path):
    bags = _bagset(tmp_path)
    a = assignment.assign_from_bags(LEVEL, bags)
    for r in range(partition.N_FOLDS):
        view = a.rotation(r)
        for side in ("source", "target"):
            test = {b.subject for b in assignment.side_bags(a, bags, side, r, "test")}
            non_test = {b.subject for b in assignment.side_bags(a, bags, side, r, "nontest")}
            assert non_test == view[f"{side}_nontest"]
            assert not test & non_test
            validated = []
            for i in range(3):
                train = {b.subject for b in assignment.side_bags(
                    a, bags, side, r, "inner_train", split=i)}
                valid = [b.subject for b in assignment.side_bags(
                    a, bags, side, r, "inner_validation", split=i)]
                assert not train & set(valid) and not (train | set(valid)) & test
                assert train | set(valid) == non_test
                validated += valid
            assert sorted(validated) == sorted(non_test)          # each exactly once
    with pytest.raises(assignment.AssignmentError, match="split"):
        assignment.side_bags(a, bags, "source", 0, "inner_train")


def test_the_frozen_rotation_roles_still_read_as_before(tmp_path):
    """The rotation's own roles (nontest, test) are unchanged by the inner ones."""
    bags = _bagset(tmp_path)
    a = assignment.assign_from_bags(LEVEL, bags)
    for r in range(4):
        view = a.rotation(r)
        for side in ("source", "target"):
            for role in assignment.ROLES:
                got = assignment.side_bags(a, bags, side, r, role)
                assert [b.subject for b in got] == sorted(view[f"{side}_{role}"])
                assert all(b.condition == a.conditions[b.subject] for b in got)
        with pytest.raises(assignment.AssignmentError, match="no inner split"):
            assignment.side_bags(a, bags, "source", r, "nontest", split=0)


def test_the_search_data_holds_the_inner_splits_and_a_refit_view_without_validation(
        tmp_path):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    for r in range(4):
        refit = data.rotations[r]
        assert refit["source_valid"] == [] and refit["target_valid"] == []
        test_fold = data.folds[r]["test_fold"]
        for i in range(3):
            view = data.inner_view(r, i)
            assert list(view.rotations) == [r]
            d = view.rotations[r]
            assert d["source_valid"] and d["target_valid"]
            subjects = {b.subject for key in d for b in d[key]}
            assert subjects <= {b.subject for key in refit for b in refit[key]}
            assert test_fold not in {int(s[2:]) % 4 for s in subjects}


# ------------------------------------------------------------------ fixed epochs


def test_the_refit_runs_exactly_e_epochs_with_no_validation():
    torch.manual_seed(0)
    model = MILCREDAModel(3, 2, attention_gamma=0.5, attention_temperature=1.0)
    out = tp.train_source_phase(model, _bags(6, 3, 1, "s"), [], {**HYPER, "epochs1": 3},
                                torch.Generator().manual_seed(0), CPU)
    assert out["epochs"] == 3 and len(out["history"]) == 3
    assert out["validation"] == []
    assert out["checkpoint"]["epoch"] == 2
    assert all(math.isnan(h["val_balanced_accuracy"]) for h in out["history"])
    joint = tp.bridge(model)
    two = tp.adapt_target_phase(model, out, joint["frozen_state"], _bags(6, 3, 1, "s"),
                                _bags(6, 2, 3, "t"), [], {**HYPER, "epochs2": 2},
                                torch.Generator().manual_seed(0), CPU)
    assert two["epochs"] == 2 and two["validation"] == [] and two["checkpoint"]["epoch"] == 1


def test_the_search_records_validation_every_epoch_and_never_stops_early(monkeypatch):
    torch.manual_seed(0)
    model = MILCREDAModel(3, 2, attention_gamma=0.5, attention_temperature=1.0)
    valid = _bags(4, 3, 2, "v")
    out = tp.train_source_phase(model, _bags(6, 3, 1, "s"), valid, {**HYPER, "epochs1": 4},
                                torch.Generator().manual_seed(0), CPU)
    assert out["epochs"] == 4 and [v["epoch"] for v in out["validation"]] == [0, 1, 2, 3]
    for v in out["validation"]:
        assert v["subjects"] == [b.subject for b in valid]
        assert v["labels"] == [b.label for b in valid]
        assert len(v["probabilities"]) == len(valid)
    assert "patience1" not in tp.HYPER_KEYS and "patience1" not in tp.SCHEDULE_KEYS


# ------------------------------------------------------------------ pooled selection


def _split(subjects, labels, predicted, epoch=0):
    return {"epoch": epoch, "subjects": list(subjects), "labels": list(labels),
            "probabilities": [[1.0 - p, float(p)] for p in predicted]}


def test_the_pooled_count_balanced_accuracy_is_not_the_mean_of_the_split_ones():
    # Imbalanced: split 0 has one positive, split 1 has five, split 2 has two.
    splits = [_split(["a", "b", "c", "d"], [1, 0, 0, 0], [0, 0, 0, 0]),
              _split(["e", "f", "g", "h", "i", "j"], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]),
              _split(["k", "l", "m"], [1, 1, 0], [1, 0, 0])]
    per_split = [tp.balanced_accuracy(s["labels"], np.argmax(s["probabilities"], 1))
                 for s in splits]
    labels = [y for s in splits for y in s["labels"]]
    predicted = [int(np.argmax(p)) for s in splits for p in s["probabilities"]]
    pooled = tp.pooled_balanced_accuracy(splits)
    assert pooled == tp.balanced_accuracy(labels, predicted)
    assert not math.isclose(pooled, float(np.mean(per_split)))
    with pytest.raises(ValueError, match="once"):
        tp.pooled_balanced_accuracy([splits[0], splits[0]])


def _trial(value, attrs):
    return optuna.trial.create_trial(
        params={"lr2": 1e-3}, distributions={"lr2": search.LR2}, value=value,
        user_attrs=attrs)


def test_the_selection_is_the_argmax_of_the_pooled_score_over_trial_and_epoch():
    study = optuna.create_study(direction="maximize")
    study.add_trial(_trial(0.7, {"pooledBalancedAccuracy": [0.5, 0.7, 0.6],
                                 "pooledLabelFree": [0.1, 0.2, 0.3]}))
    study.add_trial(_trial(0.8, {"pooledBalancedAccuracy": [0.6, 0.6, 0.8],
                                 "pooledLabelFree": [0.4, 0.0, 0.0]}))
    primary = search.select(study, "primary")
    assert (primary["trial"], primary["epoch"], primary["epochs"]) == (1, 2, 3)
    assert primary["score"] == 0.8
    free = search.select(study, "label-free")
    assert (free["trial"], free["epoch"], free["epochs"]) == (1, 0, 1)
    # ties: the lowest trial, then the earliest epoch
    tie = optuna.create_study(direction="maximize")
    tie.add_trial(_trial(0.5, {"pooledBalancedAccuracy": [0.4, 0.5, 0.5],
                               "pooledLabelFree": [0.0, 0.0, 0.0]}))
    tie.add_trial(_trial(0.5, {"pooledBalancedAccuracy": [0.5, 0.5, 0.5],
                               "pooledLabelFree": [0.0, 0.0, 0.0]}))
    chosen = search.select(tie, "primary")
    assert (chosen["trial"], chosen["epoch"]) == (0, 1)


def test_the_label_free_selection_reads_no_target_label():
    splits = [_split(["a", "b"], [1, 0], [0.9, 0.2]), _split(["c", "d"], [0, 1], [0.6, 0.3]),
              _split(["e"], [1], [0.8])]
    flipped = [{**s, "labels": [1 - y for y in s["labels"]]} for s in splits]
    unlabelled = [{k: v for k, v in s.items() if k != "labels"} for s in splits]
    score = tp.pooled_label_free_score(splits)
    assert score == tp.pooled_label_free_score(flipped) == tp.pooled_label_free_score(
        unlabelled)
    G = torch.tensor([p for s in splits for p in s["probabilities"]], dtype=torch.float64)
    assert score == pytest.approx(tp.label_free_score(G))


def test_the_pooled_series_check_each_validated_subject_once_per_epoch():
    a = [_split(["a"], [1], [0.9], 0), _split(["a"], [1], [0.1], 1)]
    b = [_split(["b"], [0], [0.1], 0), _split(["b"], [0], [0.9], 1)]
    c = [_split(["c"], [1], [0.8], 0), _split(["c"], [1], [0.8], 1)]
    series = search.pooled_series([a, b, c])
    assert series["pooledBalancedAccuracy"] == [1.0, 0.25]
    assert len(series["pooledLabelFree"]) == 2
    with pytest.raises(search.SearchError, match="epochs"):
        search.pooled_series([a, b, c[:1]])


# ------------------------------------------------------------------ studies and units


def test_one_study_per_strategy_transfer_and_outer_fold(tmp_path):
    paths = {search.study_path(tmp_path, LEVEL, 2, "MIL-CREDA", r) for r in range(4)}
    assert len(paths) == 4
    assert all(f"__outer{r}" in str(search.study_path(tmp_path, LEVEL, 1, "common", r))
               for r in range(4))
    with pytest.raises(search.SearchError, match="outer"):
        search.study_path(tmp_path, LEVEL, 1, "common", 4)
