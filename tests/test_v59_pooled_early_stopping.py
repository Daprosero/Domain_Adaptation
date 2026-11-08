"""Experiments plan v60, Secs. 4.5 and 7.6: pooled early stopping of every search trial.

Each search trial trains up to `search.EPOCH_CAP` epochs and stops when the POOLED
per-epoch selection score of its phase has not improved for `search.PATIENCE[phase]`
consecutive epochs (phase 1: the pooled source-validation balanced accuracy over the
three inner splits; phase 2: the pooled centroid score over the three inner splits). The
trial's selected epoch is the best epoch of its pooled series; the refit trains exactly
that epoch count, with no early stopping. Every number here is scripted or synthetic.
"""

from __future__ import annotations

import hashlib
import math
import threading

import numpy as np
import optuna
import pytest
import torch

from MIL_CREDA_EEG import assignment, search
from MIL_CREDA_EEG import two_phase as tp
from test_balladeer_search import _bagset, _sources

CPU = torch.device("cpu")
LEVEL = assignment.Transfer("level", "L1", "L11")


def _split_of(data, rotation, bags, side):
    names = [b.subject for b in bags]
    for i in range(search.N_INNER):
        if [b.subject for b in data.inner[(rotation, i)][f"{side}_valid"]] == names:
            return i
    raise AssertionError("not an inner split's validation")


def _phase_one_fake(data, probability, trained):
    """A phase-1 trainer shaped like the real ones: it loops over `tp.epoch_range`, calls
    `tp.validation_record` once per epoch and appends its history after it.
    `probability(split, epoch, bag)` is the positive-class probability; `trained[split]`
    counts the epochs it actually trained."""

    def fake(group, data_, rotation, seed, hyper, device="cpu"):
        bags = data_.rotations[rotation]["source_valid"]
        split = _split_of(data, rotation, bags, "source")
        validation, history = [], []
        for e in tp.epoch_range(hyper["epochs1"]):
            trained[split] = trained.get(split, 0) + 1
            p = [probability(split, e, b) for b in bags]
            G = torch.tensor([[1 - x, x] for x in p], dtype=torch.float64)
            validation.append(tp.validation_record(e, bags, G))
            history.append({"epoch": e, "loss": 1.0 / (e + 1),
                            "val_balanced_accuracy": tp.balanced_accuracy(
                                [b.label for b in bags], G.argmax(1).tolist())})
        return {"result": {"validation": validation, "history": history,
                           "checkpoint": {"epoch": len(history) - 1}}}

    return fake


def _phase_two_fake(trained, probability=None):
    """A phase-2 trainer shaped like the real ones (`tp.epoch_range`, one
    `tp.validation_record` with Z per epoch, history after it)."""
    probability = probability or (lambda s, params, e: 0.5)

    def fake(strategy, data_, rotation, seed, record, params, schedule, device="cpu"):
        bags = data_.rotations[rotation]["target_valid"]
        key = tuple(b.subject for b in bags)
        validation, history = [], []
        for e in tp.epoch_range(schedule["epochs2"]):
            trained.setdefault(key, []).append(e)
            p = [probability(b.subject, params, e) for b in bags]
            G = torch.tensor([[1 - x, x] for x in p], dtype=torch.float64)
            Z = torch.tensor([[x] for x in p], dtype=torch.float64)
            validation.append(tp.validation_record(e, bags, G, Z))
            history.append({"epoch": e, "loss": 0.0, "val_balanced_accuracy": math.nan})
        return {"result": {"checkpoint": {"sigma": 1.0}, "history": history,
                           "validation": validation}}

    return fake


def _scripted_centroid(monkeypatch, series):
    """The pooled centroid score of epoch e is `series[e]` (the records' epoch)."""
    def score(records, sources):
        epochs = {int(r["epoch"]) for r in records}
        assert len(records) == search.N_INNER and len(epochs) == 1
        return float(series[epochs.pop()])

    monkeypatch.setattr(tp, "pooled_centroid_balanced_accuracy", score)


def _phase_two(monkeypatch, tmp_path, data, trained, *, strategy="MIL-CREDA", epochs=20,
               n_trials=1, probability=None, name="s"):
    monkeypatch.setattr(search, "train_phase_two", _phase_two_fake(trained, probability))
    monkeypatch.setattr(search, "source_representations", _sources)
    monkeypatch.setattr(search, "_check_phase_one_record", lambda *a, **k: None)
    if strategy not in search.PHASE_TWO_TRAINERS:
        monkeypatch.setitem(search.PHASE_TWO_TRAINERS, strategy, print)
    return search.run_phase_two_study(strategy, data, 0, ({}, {"sigma": 1.0}),
                                      {"epochs1": 1, "epochs2": epochs, "bagsPerStep": 2},
                                      root=tmp_path / name, n_trials=n_trials)


def test_the_caps_and_patience_values_are_those_of_v59():
    assert search.EPOCH_CAP == 100
    assert search.PATIENCE == {1: 30, 2: 10}
    from MIL_CREDA_EEG.comparators.cca_lsc import training as cca
    assert search.MIN_TRAINED_EPOCHS == {"CCA-LSC": cca.WARMUP_EPOCHS + 1} == {"CCA-LSC": 4}


def test_a_trial_stops_after_exactly_patience_epochs_without_pooled_improvement(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    series = [0.5, 0.6, 0.8, 0.7, 0.8, 0.75] + [0.79] * 30
    _scripted_centroid(monkeypatch, series)
    trained = {}
    study = _phase_two(monkeypatch, tmp_path, data, trained, epochs=30)
    trial = study.trials[0]
    # best at index 2; epochs 3..12 (ten) do not improve strictly; the trial stops there
    patience = search.PATIENCE[2]
    assert [len(v) for v in trained.values()] == [3 + patience] * search.N_INNER
    assert trial.user_attrs["pooledCentroidBalancedAccuracy"] == pytest.approx(
        series[:3 + patience])
    assert trial.user_attrs["trainedEpochs"] == 3 + patience
    assert trial.user_attrs["reachedCap"] is False
    assert trial.user_attrs["bestEpoch"] == 2 and trial.user_attrs["epochs"] == 3
    assert trial.value == pytest.approx(0.8)
    assert search.select(study)["epochs"] == 3
    # the EnsV candidates are the (trial, epoch) pairs actually trained
    assert len(trial.user_attrs["ensvProbabilities"]) == 3 + patience


def test_the_phase_one_stop_uses_the_pooled_score_not_one_split(tmp_path, monkeypatch):
    """Split 0 gets one more 'yes' subject right every epoch while split 1 loses one: the
    split-0 score improves, the pooled counts do not move, so the trial stops `patience`
    epochs after epoch 0 (a per-split rule would have kept split 0 training)."""
    monkeypatch.setitem(search.PATIENCE, 1, 1)
    data = search.search_data(LEVEL, _bagset(tmp_path))
    yes = {i: [b.subject for b in data.inner[(0, i)]["source_valid"] if b.label == 1]
           for i in range(search.N_INNER)}
    assert min(len(yes[0]), len(yes[1])) >= 1

    def probability(split, epoch, bag):
        if bag.label == 0:
            return 0.0
        k = yes[split].index(bag.subject)
        if split == 0:
            return 1.0 if k < epoch else 0.0                  # improves with the epoch
        if split == 1:
            return 1.0 if k < len(yes[1]) - epoch else 0.0    # worsens by as much
        return 1.0

    trained = {}
    monkeypatch.setattr(search, "train_phase_one", _phase_one_fake(data, probability, trained))
    study = search.run_phase_one_study("common", data, 0,
                                       {"epochs1": 20, "epochs2": 1, "bagsPerStep": 2},
                                       root=tmp_path, n_trials=1)
    trial = study.trials[0]
    split0 = trial.user_attrs["splitBalancedAccuracy"][0]
    assert split0[1] > split0[0]                              # one split did improve
    assert len(set(trial.user_attrs["pooledBalancedAccuracy"])) == 1
    assert trained == {0: 2, 1: 2, 2: 2}
    assert trial.user_attrs["trainedEpochs"] == 2 and trial.user_attrs["bestEpoch"] == 0
    assert trial.user_attrs["trainLoss"] == pytest.approx([1.0, 0.5])


def test_the_cap_of_100_is_respected(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    _scripted_centroid(monkeypatch, [e / 1000 for e in range(200)])   # always improving
    trained = {}
    study = _phase_two(monkeypatch, tmp_path, data, trained, epochs=search.EPOCH_CAP)
    trial = study.trials[0]
    assert [len(v) for v in trained.values()] == [search.EPOCH_CAP] * search.N_INNER
    assert trial.user_attrs["trainedEpochs"] == search.EPOCH_CAP
    assert trial.user_attrs["reachedCap"] is True
    assert trial.user_attrs["bestEpoch"] == search.EPOCH_CAP - 1
    with pytest.raises(search.SearchError, match="cap"):
        _phase_two(monkeypatch, tmp_path, data, {}, epochs=search.EPOCH_CAP + 1, name="over")
    with pytest.raises(search.SearchError, match="cap"):
        search.run_phase_one_study("common", data, 0, {"epochs1": search.EPOCH_CAP + 1,
                                                       "epochs2": 1, "bagsPerStep": 2},
                                   root=tmp_path / "over1", n_trials=1)
    assert not (tmp_path / "over" / "Results").exists()


def test_cca_lsc_trains_its_minimum_epochs_before_it_can_stop(tmp_path, monkeypatch):
    monkeypatch.setitem(search.PATIENCE, 2, 1)
    data = search.search_data(LEVEL, _bagset(tmp_path))
    _scripted_centroid(monkeypatch, [0.5] * 20)                       # never improves
    trained = {}
    study = _phase_two(monkeypatch, tmp_path, data, trained, strategy="CCA-LSC", name="cca")
    assert study.trials[0].user_attrs["trainedEpochs"] == 4
    assert [len(v) for v in trained.values()] == [4] * search.N_INNER
    trained = {}
    study = _phase_two(monkeypatch, tmp_path, data, trained, strategy="MIL-CREDA", name="mc")
    assert study.trials[0].user_attrs["trainedEpochs"] == 2


def _probability(subject, params, epoch):
    digest = hashlib.sha256(f"{subject}:{params['lr2']:.12g}:{epoch}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def test_target_labels_never_change_a_selection_under_early_stopping(tmp_path, monkeypatch):
    from test_balladeer_selection import _flip
    monkeypatch.setitem(search.PATIENCE, 2, 2)
    data = search.search_data(LEVEL, _bagset(tmp_path))
    targets = {b.subject for v in data.inner.values() for b in v["target_valid"]}
    runs = []
    for name, d in (("a", data), ("b", _flip(data, targets))):
        trained = {}
        study = _phase_two(monkeypatch, tmp_path, d, trained, epochs=12, n_trials=3,
                           probability=_probability, name=name)
        runs.append((study, [t.user_attrs["trainedEpochs"] for t in study.trials]))
    (a, ea), (b, eb) = runs
    assert ea == eb and any(e < 12 for e in ea)
    for rule in search.RULES:
        x, y = search.select(a, rule), search.select(b, rule)
        assert (x["trial"], x["epoch"], x["score"]) == (y["trial"], y["epoch"], y["score"])


def test_the_coordinated_splits_keep_the_trainers_numerics(tmp_path):
    """Real MIL-CREDA phase-1 trainer: the three splits trained in lockstep give exactly
    the validation records and losses the serial runs give."""
    data = search.search_data(LEVEL, _bagset(tmp_path))
    schedule = {"epochs1": 3, "epochs2": 1, "bagsPerStep": 2}
    study = search.run_phase_one_study("MIL-CREDA", data, 0, schedule, root=tmp_path,
                                       n_trials=1)
    trial = study.trials[0]
    hyper = search.phase_one_hyper("MIL-CREDA", trial.params, schedule)
    serial = [search.train_phase_one("MIL-CREDA", data.inner_view(0, i), 0,
                                     search.SEARCH_SEED, hyper, CPU)["result"]
              for i in range(search.N_INNER)]
    assert trial.user_attrs["pooledBalancedAccuracy"] == \
        search.pooled_balanced_accuracy_series([r["validation"] for r in serial])
    assert trial.user_attrs["trainLoss"] == search._mean_loss([r["history"] for r in serial])
    assert trial.user_attrs["trainedEpochs"] == 3


def test_a_real_trainer_stops_with_the_pooled_rule(tmp_path, monkeypatch):
    monkeypatch.setitem(search.PATIENCE, 1, 1)
    data = search.search_data(LEVEL, _bagset(tmp_path))
    schedule = {"epochs1": 8, "epochs2": 1, "bagsPerStep": 2}
    study = search.run_phase_one_study("MIL-CREDA", data, 0, schedule, root=tmp_path,
                                       n_trials=2)
    for trial in study.trials:
        series = trial.user_attrs["pooledBalancedAccuracy"]
        n = trial.user_attrs["trainedEpochs"]
        assert len(series) == n
        best = int(np.nanargmax(series))
        assert n == 8 or n == best + 2          # patience 1 after the last improvement
        assert trial.user_attrs["bestEpoch"] == best


def test_the_refit_trains_exactly_the_selected_epochs_with_no_early_stopping(tmp_path,
                                                                            monkeypatch):
    monkeypatch.setitem(search.PATIENCE, 1, 1)
    data = search.search_data(LEVEL, _bagset(tmp_path))
    schedule = search.refit_schedule({"epochs1": 9, "epochs2": 1, "bagsPerStep": 2}, 1, 6)
    hyper = search.phase_one_hyper("MIL-CREDA", {"lr1": 1e-3, "wd1": 1e-5,
                                                 "attentionGamma": 0.5,
                                                 "attentionTemperature": 1.0,
                                                 "sigmaMultiplier": 1.0}, schedule)
    out = search.train_phase_one("MIL-CREDA", data, 0, search.PHASE_ONE_SEED, hyper, CPU)
    assert out["result"]["epochs"] == 6 and out["result"]["validation"] == []
    # epoch_range outside a coordinated search is range itself
    assert list(tp.epoch_range(5)) == [0, 1, 2, 3, 4]


def test_a_failing_split_fails_the_trial_and_releases_the_others(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    _scripted_centroid(monkeypatch, [e / 100 for e in range(50)])
    base = _phase_two_fake({})

    def flaky(strategy, data_, rotation, seed, record, params, schedule, device="cpu"):
        if _split_of(data, rotation, data_.rotations[rotation]["target_valid"],
                     "target") == 1:
            for e in tp.epoch_range(schedule["epochs2"]):
                if e == 2:
                    raise RuntimeError("split 1 broke")
                bags = data_.rotations[rotation]["target_valid"]
                G = torch.full((len(bags), 2), 0.5, dtype=torch.float64)
                tp.validation_record(e, bags, G, G)
        return base(strategy, data_, rotation, seed, record, params, schedule, device)

    monkeypatch.setattr(search, "train_phase_two", flaky)
    monkeypatch.setattr(search, "source_representations", _sources)
    monkeypatch.setattr(search, "_check_phase_one_record", lambda *a, **k: None)
    before = threading.active_count()
    with pytest.raises(RuntimeError, match="split 1 broke"):
        search.run_phase_two_study("MIL-CREDA", data, 0, ({}, {"sigma": 1.0}),
                                   {"epochs1": 1, "epochs2": 10, "bagsPerStep": 2},
                                   root=tmp_path, n_trials=1)
    study = optuna.load_study(study_name=search.study_path(tmp_path, LEVEL, 2, "MIL-CREDA",
                                                           0).stem,
                              storage=f"sqlite:///{search.study_path(tmp_path, LEVEL, 2, 'MIL-CREDA', 0)}")
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert threading.active_count() == before


def test_the_stopping_diagnostic_reports_cap_and_last_epoch_fractions(tmp_path, monkeypatch):
    monkeypatch.setitem(search.PATIENCE, 2, 2)
    data = search.search_data(LEVEL, _bagset(tmp_path))
    calls = {"n": 0}

    def score(records, sources):
        e = int(records[0]["epoch"])
        trial = calls["trial"]
        return e / 10 if trial == 0 else 0.5                   # trial 0 improves to the cap

    monkeypatch.setattr(tp, "pooled_centroid_balanced_accuracy", score)
    original = search.run_phase_two_study
    studies = []
    for k in range(2):
        calls["trial"] = k
        studies.append(_phase_two(monkeypatch, tmp_path, data, {}, epochs=6, name=f"d{k}"))
    row = search.stopping_diagnostic("MIL-CREDA", studies)
    assert row["studies"] == 2 and row["trials"] == 2
    assert row["trialsReachedCap"] == 1 and row["fractionTrialsReachedCap"] == 0.5
    # study 0 selects its trial's last trained epoch (the cap), study 1 epoch 1 of 3
    assert row["selectedAtLastTrainedEpoch"]["primary"] == {"selections": 2, "atLast": 1,
                                                            "fraction": 0.5}
    assert set(row["selectedAtLastTrainedEpoch"]) == {"primary", "ensv"}
    assert original is search.run_phase_two_study


def test_a_v58_study_is_refused_naming_v59(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    _scripted_centroid(monkeypatch, [0.5] * 20)
    study = _phase_two(monkeypatch, tmp_path, data, {}, epochs=3, name="old")
    study.set_user_attr("objective", "plan v60, Sec. 4.5, no target label: ...")
    with pytest.raises(search.SearchError, match="v59"):
        _phase_two(monkeypatch, tmp_path, data, {}, epochs=3, n_trials=2, name="old")
