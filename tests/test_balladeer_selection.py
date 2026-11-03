"""Experiments plan v43, Secs. 4.5, 7.3 and 7.7: hyperparameter selection per test rotation.

One Optuna study per strategy and transfer, the same trials scored in all four rotations;
for the test fold of rotation r the configuration is chosen by mean minus sd over the three
rotations whose validation fold is not r's test fold. Every number here is scripted or
synthetic, never a measured result.
"""

from __future__ import annotations

import json

import numpy as np
import optuna
import pytest

from MIL_CREDA_EEG import assignment, bag_data, partition, search
from MIL_CREDA_EEG import two_phase as tp

LEVEL = assignment.Transfer("level", "L1", "L11")
SCHEDULE = {"epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 2}


def _robust3(values):
    return float(np.mean(values) - np.std(values, ddof=1))


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
    rotations = rotations or {str(r): {"test_fold": r, "validation_fold": (r + 1) % 4,
                                       "train_folds": [(r + 2) % 4, (r + 3) % 4]}
                              for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256="x" * 64, rotations=rotations)


def _scripted_study(values, free=None):
    """A study whose trials carry scripted per-rotation primary and label-free values."""
    study = optuna.create_study(direction="maximize")
    for i, rotation_values in enumerate(values):
        trial = study.ask()
        trial.suggest_float("lr2", 1e-5, 1e-2, log=True)
        trial.set_user_attr("rotationValues", list(rotation_values))
        trial.set_user_attr("labelFreeRotationValues",
                            list(free[i] if free else rotation_values))
        study.tell(trial, float(np.mean(rotation_values) - np.std(rotation_values, ddof=1)))
    return study


FOLDS = {r: {"test_fold": r, "validation_fold": (r + 1) % 4} for r in range(4)}


def test_the_excluded_rotation_is_derived_from_the_frozen_partition():
    folds = {f"S{i:02d}": i % 4 for i in range(16)}
    doc = partition.rotations(folds)
    for r, rot in doc.items():
        kept = search.selection_rotations(doc, int(r))
        assert len(kept) == partition.N_FOLDS - 1 and int(r) in kept
        excluded = [q for q in range(4) if q not in kept]
        # the one rotation left out is exactly the one validating on r's test subjects
        assert len(excluded) == 1
        assert set(doc[str(excluded[0])]["validation"]) == set(rot["test"])
        for q in kept:
            assert not set(doc[str(q)]["validation"]) & set(rot["test"])
    # derived, never an offset: another layout moves the excluded rotation with it
    other = {r: {"test_fold": r, "validation_fold": (r + 3) % 4} for r in range(4)}
    assert search.selection_rotations(other, 0) == [0, 2, 3]
    assert search.selection_rotations(FOLDS, 0) == [0, 1, 2]
    with pytest.raises(search.SearchError, match="three"):
        search.selection_rotations({r: {"test_fold": r, "validation_fold": 1}
                                    for r in range(4)}, 0)


def test_the_selection_is_per_test_rotation_by_mean_minus_sd_over_three_rotations():
    values = [(0.9, 0.9, 0.9, 0.1), (0.5, 0.5, 0.5, 0.95), (0.7, 0.7, 0.7, 0.7)]
    study = _scripted_study(values)
    chosen = search.select_per_rotation(study, FOLDS, "primary")
    for r in range(4):
        kept = search.selection_rotations(FOLDS, r)
        scores = [_robust3([v[q] for q in kept]) for v in values]
        best = int(np.argmax(scores))
        assert chosen[r]["trial"] == best
        assert chosen[r]["selectionRotations"] == kept
        assert chosen[r]["excludedRotation"] == [q for q in range(4) if q not in kept][0]
        assert chosen[r]["score"] == pytest.approx(scores[best], abs=1e-15)
        assert chosen[r]["params"] == study.trials[best].params
    # rotation 0 excludes rotation 3 and picks the flat-high trial; rotation 1 excludes
    # rotation 0 and may not: the rotations really receive different configurations
    assert chosen[0]["trial"] == 0
    assert len({c["trial"] for c in chosen.values()}) > 1


def test_the_label_free_rule_applies_the_same_exclusion():
    primary = [(0.6, 0.6, 0.6, 0.6), (0.6, 0.6, 0.6, 0.6)]
    free = [(0.2, 0.2, 0.2, 0.9), (0.3, 0.3, 0.3, 0.0)]
    study = _scripted_study(primary, free)
    chosen = search.select_per_rotation(study, FOLDS, "label-free")
    assert chosen[0]["trial"] == 1                  # rotation 3 is excluded for test fold 0
    assert chosen[0]["rule"] == "label-free"
    assert chosen[0]["selectionRotations"] == [0, 1, 2]
    with pytest.raises(search.SearchError, match="rule"):
        search.select_per_rotation(study, FOLDS, "test")


def _labelled_trainer(calls):
    """A phase-2 trainer whose validation BA depends on the params AND on the labels of the
    target-validation bags it is given, as a real one does."""

    def fake(strategy, data, rotation, seed, record, params, schedule, device):
        bags = data.rotations[rotation]["target_valid"]
        labels = [b.label for b in bags]
        predicted = [(int(b.subject[2:]) + int(params["lr2"] > 1e-3)) % 2 for b in bags]
        ba = tp.balanced_accuracy(labels, predicted) if len(set(labels)) > 1 else \
            float(np.mean(np.asarray(labels) == np.asarray(predicted)))
        value = 0.5 * ba + 0.5 * float(np.log10(params["wd2"]) / -7.0) + 0.01 * rotation
        calls.append(rotation)
        return {"result": {"checkpoint": {"valid_balanced_accuracy": value,
                                          "label_free_score": 0.0, "sigma": 1.0},
                           "history": [{"label_free_score": 0.0}], "best_epoch": 0,
                           "epochs": 1}}

    return fake


def test_the_labels_of_r_test_subjects_cannot_change_r_selection(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    data = search.search_data(LEVEL, bags)
    monkeypatch.setattr(search, "train_phase_two", _labelled_trainer([]))
    monkeypatch.setattr(search, "_check_phase_one_records", lambda *a, **k: None)
    records = {r: ({}, {"sigma": 1.0}) for r in range(4)}
    # fewer trials than GPSampler's ten random start-up trials: the same configurations are
    # tried whatever the scores, so only the selection can differ
    first = search.run_phase_two_study("MIL-CREDA", data, records, SCHEDULE,
                                       root=tmp_path / "a", n_trials=6)
    for r in range(4):
        # half of r's test subjects: flipping every label of a fold could leave a balanced
        # accuracy of 0.5 where it was, and prove nothing
        test_subjects = set(sorted(s for s, f in bags.folds().items() if f == r)[::2])
        flipped_data = search.SearchData(
            transfer=data.transfer, partition_sha256=data.partition_sha256,
            folds=data.folds, rotations={q: {**rot, "target_valid": [
                tp.BagTensor(b.subject, b.condition, 1 - b.label, b.windows)
                if b.subject in test_subjects else b for b in rot["target_valid"]]}
                for q, rot in data.rotations.items()})
        flipped = search.run_phase_two_study("MIL-CREDA", flipped_data, records, SCHEDULE,
                                             root=tmp_path / f"flip{r}", n_trials=6)
        assert [t.params for t in flipped.trials] == [t.params for t in first.trials]
        excluded = search.select_per_rotation(first, data.folds)[r]["excludedRotation"]
        # the flip did reach the rotation validating on r's test subjects ...
        assert [t.user_attrs["rotationValues"][excluded] for t in flipped.trials] != \
            [t.user_attrs["rotationValues"][excluded] for t in first.trials]
        # ... and rotation r's selection is unchanged
        before = search.select_per_rotation(first, data.folds)[r]
        after = search.select_per_rotation(flipped, flipped_data.folds)[r]
        assert (after["trial"], after["params"], after["score"]) == \
            (before["trial"], before["params"], before["score"])


def test_the_study_records_each_rotations_folds_from_the_partition(tmp_path, monkeypatch):
    rotations = {str(r): {"test_fold": r, "validation_fold": (r + 3) % 4,
                          "train_folds": [(r + 1) % 4, (r + 2) % 4]} for r in range(4)}
    data = search.search_data(LEVEL, _bagset(tmp_path, rotations))
    assert data.folds == {r: {"test_fold": r, "validation_fold": (r + 3) % 4}
                          for r in range(4)}
    monkeypatch.setattr(search, "train_phase_two", _labelled_trainer([]))
    monkeypatch.setattr(search, "_check_phase_one_records", lambda *a, **k: None)
    study = search.run_phase_two_study("MIL-CREDA", data, {r: ({}, {}) for r in range(4)},
                                       SCHEDULE, root=tmp_path, n_trials=1)
    assert json.loads(json.dumps(study.user_attrs["rotationFolds"])) == {
        str(r): {"test_fold": r, "validation_fold": (r + 3) % 4} for r in range(4)}
