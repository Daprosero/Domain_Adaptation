"""Experiments plan v58, Secs. 4.2, 4.5, 7.3 and 7.7: selection by nested cross-validation.

One Optuna study per strategy, transfer and outer fold; each phase-2 trial is scored on the
three inner splits of the outer fold's non-test folds, per epoch, by the centroid rule: the
source validation bags assigned to the nearest centroid of the target validation bags'
pseudo-classes, the counts pooled over the splits and the balanced accuracy computed once;
EnsV reads the pooled target predictions of every trial and epoch. Neither reads a target
label. Every number here is scripted or synthetic, never a measured result.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import optuna
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


def _representation(subject: str, params, epoch: int) -> list[float]:
    """A target bag's scripted Z: it moves with the trial and the epoch."""
    p = _probability(subject, params, epoch)
    return [4.0 * p - 2.0, _probability(subject + "z", params, epoch)]


def _scripted_trainer(seen: list):
    """A phase-2 trainer whose predictions and representations depend on the subject, the
    trial and the epoch, never on a label; it records which subjects each call validated
    on."""

    def fake(strategy, data, rotation, seed, record, params, schedule, device="cpu"):
        d = data.rotations[rotation]
        seen.append({"rotation": rotation, "valid": [b.subject for b in d["target_valid"]],
                     "train": [b.subject for b in d["target_train"]]})
        validation, history = [], []
        for e in range(schedule["epochs2"]):
            p = [_probability(b.subject, params, e) for b in d["target_valid"]]
            G = torch.tensor([[1 - x, x] for x in p], dtype=torch.float64)
            Z = torch.tensor([_representation(b.subject, params, e) for b in d["target_valid"]],
                             dtype=torch.float64)
            validation.append(tp.validation_record(e, d["target_valid"], G, Z))
            history.append({"epoch": e, "val_balanced_accuracy": float("nan"), "loss": 0.0})
        return {"result": {"checkpoint": {"sigma": 1.0}, "history": history,
                           "validation": validation, "best_epoch": len(history) - 1,
                           "epochs": len(history)}}

    return fake


def _source_z(subject: str) -> list[float]:
    p = int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return [4.0 * p - 2.0, 0.5]


def _scripted_sources(calls: list):
    """The source pass: per inner split the source validation bags with their SOURCE
    labels and a fixed representation per subject (the frozen source side)."""

    def fake(strategy, data, rotation, record, device="cpu"):
        calls.append(rotation)
        return [{"subjects": [b.subject for b in data.inner[(rotation, i)]["source_valid"]],
                 "labels": [b.label for b in data.inner[(rotation, i)]["source_valid"]],
                 "representations": [_source_z(b.subject)
                                     for b in data.inner[(rotation, i)]["source_valid"]]}
                for i in range(search.N_INNER)]

    return fake


def _study(tmp_path, data, rotation, seen, monkeypatch, n_trials=4, name="s", calls=None):
    monkeypatch.setattr(search, "train_phase_two", _scripted_trainer(seen))
    monkeypatch.setattr(search, "source_representations",
                        _scripted_sources(calls if calls is not None else []))
    monkeypatch.setattr(search, "_check_phase_one_record", lambda *a, **k: None)
    return search.run_phase_two_study("MIL-CREDA", data, rotation, ({}, {"sigma": 1.0}),
                                      SCHEDULE, root=tmp_path / name, n_trials=n_trials)


def _flip(data, subjects, roles=None):
    """The same search data with the labels of `subjects` flipped (in `roles`, everywhere
    by default)."""
    def bags(rows):
        return {k: [tp.BagTensor(b.subject, b.condition, 1 - b.label, b.windows)
                    if b.subject in subjects and (roles is None or k in roles) else b
                    for b in v] for k, v in rows.items()}

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
        # ... while the same flip reaches the study of another outer fold, whose F holds
        # them: as source subjects, their labels are what the centroid rule scores against
        other = (r + 1) % 4
        base = _study(tmp_path, data, other, [], monkeypatch, name=f"o{r}")
        moved = _study(tmp_path, flipped_data, other, [], monkeypatch, name=f"m{r}")
        assert [t.user_attrs["pooledCentroidBalancedAccuracy"] for t in base.trials] != \
            [t.user_attrs["pooledCentroidBalancedAccuracy"] for t in moved.trials]


def test_the_objective_is_the_best_epochs_pooled_centroid_score_over_the_inner_splits(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    seen, calls = [], []
    study = _study(tmp_path, data, 2, seen, monkeypatch, n_trials=3, calls=calls)
    assert len(seen) == 3 * 3                              # three inner splits per trial
    assert calls == [2]                     # the source pass: once per outer fold's study
    for t in study.trials:
        series = t.user_attrs["pooledCentroidBalancedAccuracy"]
        assert len(series) == SCHEDULE["epochs2"]
        assert t.value == max(series) and t.user_attrs["bestEpoch"] == int(np.argmax(series))
        assert t.user_attrs["epochs"] == t.user_attrs["bestEpoch"] + 1
        assert "pooledBalancedAccuracy" not in t.user_attrs
        assert "splitBalancedAccuracy" not in t.user_attrs
    # recomputed by hand: per split, the target pseudo-class centroids, each source bag to
    # the nearest one; counts pooled over the three validating folds, epoch by epoch
    t = study.trials[0]
    validated = sorted(b.subject for i in range(3) for b in data.inner[(2, i)]["target_valid"])
    assert validated == sorted(b.subject for b in data.rotations[2]["target_train"])
    expected = []
    for e in range(SCHEDULE["epochs2"]):
        real, assigned = [], []
        for i in range(3):
            targets = data.inner[(2, i)]["target_valid"]
            pseudo = np.array([int(_probability(b.subject, t.params, e) > 0.5)
                               for b in targets])
            Z = np.array([_representation(b.subject, t.params, e) for b in targets])
            centroids = {c: Z[pseudo == c].mean(0) for c in (0, 1) if np.any(pseudo == c)}
            for b in data.inner[(2, i)]["source_valid"]:
                z = np.array(_source_z(b.subject))
                assigned.append(min(centroids, key=lambda c: np.sum((z - centroids[c]) ** 2)))
                real.append(b.label)
        expected.append(tp.balanced_accuracy(real, assigned))
    assert t.user_attrs["pooledCentroidBalancedAccuracy"] == pytest.approx(expected)


def test_the_ensv_probabilities_are_recorded_per_trial_and_epoch(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    study = _study(tmp_path, data, 1, [], monkeypatch, n_trials=3)
    order = [b.subject for i in range(3) for b in data.inner[(1, i)]["target_valid"]]
    for t in study.trials:
        assert t.user_attrs["ensvSubjects"] == order
        np.testing.assert_allclose(
            t.user_attrs["ensvProbabilities"],
            [[[1 - _probability(s, t.params, e), _probability(s, t.params, e)] for s in order]
             for e in range(SCHEDULE["epochs2"])])
        assert "representations" not in json.dumps(t.user_attrs)
    chosen = search.select(study, "ensv")
    assert chosen["rule"] == "ensv" and math.isfinite(chosen["score"])


def test_no_target_label_of_f_changes_either_phase_two_selection(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    every = {b.subject for b in data.rotations[1]["target_train"]}
    flipped = _flip(data, every, roles=("target_train", "target_valid"))
    a = _study(tmp_path, data, 1, [], monkeypatch, n_trials=3)
    b = _study(tmp_path, flipped, 1, [], monkeypatch, n_trials=3, name="f")
    for x, y in zip(a.trials, b.trials):
        for key in ("pooledCentroidBalancedAccuracy", "ensvSubjects", "ensvProbabilities"):
            assert x.user_attrs[key] == y.user_attrs[key], key
    for rule in search.RULES:
        before, after = search.select(a, rule), search.select(b, rule)
        assert (after["trial"], after["epoch"], after["score"]) == \
            (before["trial"], before["epoch"], before["score"]), rule
    # a source label of F does move the primary rule (the flip test is not vacuous)
    sources = {b.subject for i in range(3) for b in data.inner[(1, i)]["source_valid"]}
    c = _study(tmp_path, _flip(data, sources, roles=("source_valid",)), 1, [], monkeypatch,
               n_trials=3, name="c")
    assert [t.user_attrs["pooledCentroidBalancedAccuracy"] for t in a.trials] != \
        [t.user_attrs["pooledCentroidBalancedAccuracy"] for t in c.trials]


def test_a_study_scored_by_an_earlier_rule_is_refused(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    study = _study(tmp_path, data, 0, [], monkeypatch, n_trials=1)
    study.set_user_attr("objective", "the best epoch's target-validation balanced accuracy")
    with pytest.raises(search.SearchError, match="selection rule changed in plan v58"):
        _study(tmp_path, data, 0, [], monkeypatch, n_trials=2)


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


def _spied_study(tmp_path, data, rotation, monkeypatch, log, captured=None, n_trials=3):
    """`_study` with the scripted trainer and source pass logging, in call order, each
    ("train" or "source", outer fold) and, in `captured`, what the source pass returned."""
    trainer, sources = _scripted_trainer([]), _scripted_sources([])

    def train(strategy, data_, rotation_, *args, **kwargs):
        log.append(("train", rotation_))
        return trainer(strategy, data_, rotation_, *args, **kwargs)

    def source(strategy, data_, rotation_, *args, **kwargs):
        log.append(("source", rotation_))
        out = sources(strategy, data_, rotation_, *args, **kwargs)
        if captured is not None:
            captured.append(out)
        return out

    monkeypatch.setattr(search, "train_phase_two", train)
    monkeypatch.setattr(search, "source_representations", source)
    monkeypatch.setattr(search, "_check_phase_one_record", lambda *a, **k: None)
    return search.run_phase_two_study("MIL-CREDA", data, rotation, ({}, {"sigma": 1.0}),
                                      SCHEDULE, root=tmp_path / f"spied{rotation}",
                                      n_trials=n_trials)


def test_no_target_label_changes_either_phase_two_selection(tmp_path, monkeypatch):
    """Every target label of the transfer flipped, in every outer fold, inner split and
    target role: both rules choose the same (trial, epoch) with the same score."""
    data = search.search_data(LEVEL, _bagset(tmp_path))
    target = {b.subject for view in [*data.rotations.values(), *data.inner.values()]
              for role in ("target_train", "target_valid") for b in view[role]}
    flipped = _flip(data, target, roles=("target_train", "target_valid"))
    assert [b.label for b in flipped.rotations[0]["target_train"]] == \
        [1 - b.label for b in data.rotations[0]["target_train"]]          # the flip is real
    for rotation in (0, 3):
        a = _study(tmp_path, data, rotation, [], monkeypatch, n_trials=3, name=f"a{rotation}")
        b = _study(tmp_path, flipped, rotation, [], monkeypatch, n_trials=3,
                   name=f"b{rotation}")
        assert [t.value for t in a.trials] == [t.value for t in b.trials]
        for rule in search.RULES:
            x, y = search.select(a, rule), search.select(b, rule)
            assert (x["trial"], x["epoch"], x["score"]) == (y["trial"], y["epoch"], y["score"])


def test_the_ensv_selection_reads_every_completed_trial_of_the_study(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    study = _study(tmp_path, data, 1, [], monkeypatch, n_trials=4)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == 4
    # by hand: every (trial, epoch) of every completed trial is a candidate; the mean of
    # their probabilities gives the pseudo-labels; the best agreement wins, the lowest trial
    # and earliest epoch on ties
    candidates = [(t.number, e, np.asarray(P, dtype=np.float64)) for t in complete
                  for e, P in enumerate(t.user_attrs["ensvProbabilities"])]
    assert len(candidates) == 4 * SCHEDULE["epochs2"]
    pseudo = np.mean([P for _, _, P in candidates], axis=0).argmax(axis=1)
    score, trial, epoch = max((float(np.mean(P.argmax(axis=1) == pseudo)), -n, -e)
                              for n, e, P in candidates)
    chosen = search.select(study, "ensv")
    assert (chosen["trial"], chosen["epoch"]) == (-trial, -epoch)
    assert chosen["score"] == pytest.approx(score)
    # a trial that did not complete is no candidate, whatever its probabilities say
    opposite = [[1.0, 0.0] if c == 1 else [0.0, 1.0] for c in pseudo]
    inverted = [opposite] * 50
    with_failed = np.mean([P for _, _, P in candidates] + [np.asarray(opposite)] * 50,
                          axis=0).argmax(axis=1)
    assert not np.array_equal(with_failed, pseudo)          # counted, it would move the vote
    study.add_trial(optuna.trial.create_trial(
        state=optuna.trial.TrialState.FAIL,
        user_attrs={"ensvSubjects": complete[0].user_attrs["ensvSubjects"],
                    "ensvProbabilities": inverted}))
    again = search.select(study, "ensv")
    assert (again["trial"], again["epoch"], again["score"]) == \
        (chosen["trial"], chosen["epoch"], chosen["score"])


def test_the_phase_two_objective_is_the_centroid_rule_and_reads_no_target_label(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    study = _study(tmp_path, data, 0, [], monkeypatch, n_trials=2)
    sources = _scripted_sources([])("MIL-CREDA", data, 0, None)
    trainer = _scripted_trainer([])
    for t in study.trials:
        per_split = [trainer("MIL-CREDA", data.inner_view(0, i), 0, search.SEARCH_SEED, None,
                             t.params, SCHEDULE)["result"]["validation"]
                     for i in range(search.N_INNER)]
        expected = search.pooled_centroid_series(per_split, sources)
        assert t.user_attrs["pooledCentroidBalancedAccuracy"] == pytest.approx(expected)
        assert t.value == pytest.approx(max(expected))
        # the same records with their labels removed, or flipped, score the same
        unlabelled = [[{k: v for k, v in r.items() if k != "labels"} for r in split]
                      for split in per_split]
        flipped = [[{**r, "labels": [1 - y for y in r["labels"]]} for r in split]
                   for split in per_split]
        assert search.pooled_centroid_series(unlabelled, sources) == expected
        assert search.pooled_centroid_series(flipped, sources) == expected
    assert any("centroid" in str(v) for v in study.user_attrs.values())


def test_the_source_representations_are_computed_once_per_outer_fold_and_inner_split(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    log, captured = [], []
    for rotation in (0, 2):
        _spied_study(tmp_path, data, rotation, monkeypatch, log, captured, n_trials=3)
    # one source pass per outer fold's study, before its first training, whatever the
    # number of trials; three trainings (one per inner split) per trial
    assert [x for x in log if x[0] == "source"] == [("source", 0), ("source", 2)]
    for rotation in (0, 2):
        mine = [x for x in log if x[1] == rotation]
        assert mine[0] == ("source", rotation)
        assert mine.count(("train", rotation)) == 3 * search.N_INNER
    # each pass holds one entry per inner split: that split's source validation bags
    for rotation, out in zip((0, 2), captured):
        assert len(out) == search.N_INNER
        for i, split in enumerate(out):
            bags = data.inner[(rotation, i)]["source_valid"]
            assert split["subjects"] == [b.subject for b in bags]
            assert split["labels"] == [b.label for b in bags]
