"""Experiments plan v50, Secs. 4.2, 4.5, 7.3 and 7.7: the Optuna searches of both phases,
one study per group or strategy, transfer and outer fold, on the three inner splits."""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import optuna
import pytest
import torch
from optuna.distributions import CategoricalDistribution, FloatDistribution

from MIL_CREDA_EEG import assignment, bag_data, model_store, partition, search
from MIL_CREDA_EEG import two_phase as tp

CPU = torch.device("cpu")
LEVEL = assignment.Transfer("level", "L1", "L11")
SCHEDULE = {"epochs1": 1, "epochs2": 1, "bagsPerStep": 2}
PHASE_ONE = {"lr1": 1e-2, "wd1": 1e-4, "attentionGamma": 0.5, "attentionTemperature": 1.0,
             "sigmaMultiplier": 1.0}


def _partition_file(root):
    doc = {"format": "tiny"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    path = root / "partition.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, doc[partition.HASH_FIELD]


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


def _records(bags, epochs, score):
    """Scripted validation records: `score(subject, epoch)` is the positive probability."""
    return [{"epoch": e, "subjects": [b.subject for b in bags],
             "labels": [b.label for b in bags],
             "probabilities": [[1 - score(b.subject, e), score(b.subject, e)] for b in bags]}
            for e in range(epochs)]


def test_the_sampler_is_gpsampler_with_the_fixed_seed_and_no_pruner(tmp_path, monkeypatch):
    seen = []
    real = optuna.samplers.GPSampler

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(search.optuna.samplers, "GPSampler", spy)
    study = search.make_study(tmp_path / "s.db", "s")
    assert seen == [{"seed": search.SAMPLER_SEED}]
    assert isinstance(study.sampler, real)
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert search.N_TRIALS == 20
    for fn in (search.run_phase_one_study, search.run_phase_two_study):
        assert inspect.signature(fn).parameters["n_trials"].default == 20
    with pytest.raises(search.SearchError, match="exists"):
        search.make_study(tmp_path / "s.db", "s")


def test_the_search_seed_is_not_a_report_seed():
    # Plan v42 Sec. 4.6: a single report seed, seed 1, distinct from the search seed 0.
    assert search.REPORT_SEEDS == (1,)
    assert search.SEARCH_SEED not in search.REPORT_SEEDS


def test_the_spaces_are_exactly_those_of_sec_7_7():
    lr1 = FloatDistribution(1e-4, 1e-1, log=True)
    wd = FloatDistribution(1e-7, 1e-2, log=True)
    shared1 = {"lr1": lr1, "wd1": wd}
    assert search.PHASE_ONE_SPACES == {
        "common": shared1, "AdaptMIL": shared1,
        "MIL-CREDA": {**shared1, "attentionGamma": FloatDistribution(0.0, 1.0),
                      "attentionTemperature": FloatDistribution(0.1, 10.0, log=True),
                      "sigmaMultiplier": CategoricalDistribution((0.5, 0.75, 1.0, 1.5))}}
    assert search.SHARED_PHASE_TWO == {"lr2": FloatDistribution(1e-5, 1e-2, log=True),
                                       "wd2": wd}
    assert search.phase_two_space("MIL-CREDA") == {
        **search.SHARED_PHASE_TWO, "tauLocal": FloatDistribution(0.1, 10.0, log=True)}
    assert search.LOSS_WEIGHT == FloatDistribution(1e-3, 10.0, log=True)
    assert search.KERNEL_MULTIPLIER == CategoricalDistribution((0.5, 0.75, 1.0, 1.5))
    assert search.CONFIDENCE_THRESHOLD == FloatDistribution(0.5, 0.95)
    # The two comparator-specific axes of Sec. 7.7's closing paragraph.
    assert search.EMA_MOMENTUM_COMPLEMENT == FloatDistribution(1e-3, 1e-1, log=True)
    rho = sorted(1 - b for b in (search.EMA_MOMENTUM_COMPLEMENT.low,
                                 search.EMA_MOMENTUM_COMPLEMENT.high))
    assert rho == pytest.approx([0.9, 0.999])
    assert search.VAT_RADIUS_FRACTION == FloatDistribution(0.01, 1.0, log=True)
    assert search.CONTRASTIVE_TEMPERATURE == FloatDistribution(0.01, 1.0, log=True)


def test_the_comparator_specific_axes_are_admitted_only_for_their_own_strategy(monkeypatch):
    monkeypatch.setattr(search, "PHASE_TWO_TRAINERS", dict(search.PHASE_TWO_TRAINERS))
    monkeypatch.setattr(search, "PHASE_TWO_EXTRA", dict(search.PHASE_TWO_EXTRA))
    clust, casual, cluda = "CLUST — variante source-free", "CASUAL", "CLUDA"
    assert set(search.OWN_AXES) == {clust, casual, cluda} <= set(model_store.STRATEGIES)
    assert search.OWN_AXES[clust] == (search.EMA_MOMENTUM_COMPLEMENT,)
    assert search.OWN_AXES[casual] == (search.VAT_RADIUS_FRACTION,)
    assert search.OWN_AXES[cluda] == (search.CONTRASTIVE_TEMPERATURE,)
    for strategy, axis in ((casual, search.EMA_MOMENTUM_COMPLEMENT),
                           (clust, search.VAT_RADIUS_FRACTION),
                           ("COT", search.EMA_MOMENTUM_COMPLEMENT),
                           ("UDA-DDA", search.VAT_RADIUS_FRACTION)):
        with pytest.raises(search.SearchError, match="generic"):
            search.register_phase_two(strategy, {"axis": axis}, print)
        assert strategy not in search.PHASE_TWO_EXTRA
    search.register_phase_two(clust, {"w": search.LOSS_WEIGHT,
                                      "ema": search.EMA_MOMENTUM_COMPLEMENT}, print)
    search.register_phase_two(casual, {"r": search.VAT_RADIUS_FRACTION}, print)
    assert search.phase_two_space(clust)["ema"] == search.EMA_MOMENTUM_COMPLEMENT
    assert search.phase_two_space(casual)["r"] == search.VAT_RADIUS_FRACTION


def test_no_comparator_phase_two_is_wired_and_the_hook_takes_only_the_generic_rule(
        monkeypatch):
    assert set(search.PHASE_TWO_TRAINERS) == {"MIL-CREDA"}
    for strategy in model_store.STRATEGIES:
        if strategy != "MIL-CREDA":
            with pytest.raises(search.NotWired):
                search.phase_two_space(strategy)
    monkeypatch.setattr(search, "PHASE_TWO_TRAINERS", dict(search.PHASE_TWO_TRAINERS))
    monkeypatch.setattr(search, "PHASE_TWO_EXTRA", dict(search.PHASE_TWO_EXTRA))
    with pytest.raises(search.SearchError, match="generic"):
        search.register_phase_two("COT", {"wCluster": FloatDistribution(1e-4, 1.0)}, print)
    with pytest.raises(search.SearchError):
        search.register_phase_two("COT", {"lr2": search.LOSS_WEIGHT}, print)
    with pytest.raises(search.SearchError):
        search.register_phase_two("MIL-CREDA", {"w": search.LOSS_WEIGHT}, print)
    search.register_phase_two("COT", {"wCluster": search.LOSS_WEIGHT,
                                      "kernel": search.KERNEL_MULTIPLIER,
                                      "threshold": search.CONFIDENCE_THRESHOLD}, print)
    assert set(search.phase_two_space("COT")) == {"lr2", "wd2", "wCluster", "kernel",
                                                  "threshold"}


def test_epoch_caps_have_no_default(tmp_path):
    for fn in (search.run_phase_one_study, search.run_phase_two_study):
        assert inspect.signature(fn).parameters["schedule"].default is inspect.Parameter.empty
    data = search.search_data(LEVEL, _bagset(tmp_path))
    with pytest.raises(ValueError, match="schedule"):
        search.run_phase_one_study("MIL-CREDA", data, 0, {"epochs1": 1}, root=tmp_path)
    # a schedule that still carries the retired patience is refused, never ignored
    with pytest.raises(ValueError, match="schedule"):
        search.run_phase_one_study("MIL-CREDA", data, 0, {**SCHEDULE, "patience1": 1},
                                   root=tmp_path)
    assert not (tmp_path / "Results").exists()


def test_the_phase_one_objective_pools_the_source_validation_of_the_three_inner_splits(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    calls = []

    def fake(group, data_, rotation, seed, hyper, device):
        d = data_.rotations[rotation]
        calls.append((rotation, seed, tuple(b.subject for b in d["source_valid"])))
        # epoch 1 predicts every validated subject right, epoch 0 says "no" to all
        records = _records(d["source_valid"], hyper["epochs1"],
                           lambda s, e: float(e == 1) * {0: 0.0, 1: 1.0}[
                               next(b.label for b in d["source_valid"] if b.subject == s)])
        return {"result": {"validation": records,
                           "history": [{"val_balanced_accuracy": 0.0, "loss": 1.0}
                                       for _ in records]}}

    monkeypatch.setattr(search, "train_phase_one", fake)
    study = search.run_phase_one_study("common", data, 1, {**SCHEDULE, "epochs1": 3},
                                       root=tmp_path, n_trials=1)
    trial = study.trials[0]
    assert [(r, seed) for r, seed, _ in calls] == [(1, search.SEARCH_SEED)] * 3
    validated = [s for _, _, subjects in calls for s in subjects]
    assert sorted(validated) == sorted(b.subject for b in data.rotations[1]["source_train"])
    assert trial.user_attrs["pooledBalancedAccuracy"] == [0.5, 1.0, 0.5]
    assert trial.value == 1.0 and trial.user_attrs["bestEpoch"] == 1
    assert trial.user_attrs["epochs"] == 2 and trial.user_attrs["trainLoss"] == [1.0] * 3
    assert search.select(study)["epochs"] == 2


def test_the_search_never_loads_a_test_fold_bag(tmp_path):
    bags = _bagset(tmp_path)
    a = assignment.assign_from_bags(LEVEL, bags)
    # Every subject is a test subject in some rotation, so no bag can be removed; a bag
    # read through any test role instead is recorded by a spy on the accessor.
    roles = []
    real = assignment.side_bags

    def spy(a_, bags_, side, r, role, **kwargs):
        roles.append(role)
        return real(a_, bags_, side, r, role, **kwargs)

    search.assignment_mod.side_bags, saved = spy, search.assignment_mod.side_bags
    try:
        data = search.search_data(LEVEL, bags)
    finally:
        search.assignment_mod.side_bags = saved
    assert set(roles) == {"nontest", "inner_train", "inner_validation"}
    assert sorted(data.rotations) == [0, 1, 2, 3]
    assert sorted(data.inner) == [(r, i) for r in range(4) for i in range(3)]
    for (r, _), sides in [*((k, v) for k, v in data.inner.items()),
                          *(((r, None), v) for r, v in data.rotations.items())]:
        assert set(sides) == {"source_train", "source_valid", "target_train", "target_valid"}
        view = a.rotation(r)
        held = {b.subject for side in sides.values() for b in side}
        assert not held & (view["source_test"] | view["target_test"])


def test_the_phase_two_mil_creda_sigma_comes_from_the_phase_one_checkpoint(tmp_path,
                                                                          monkeypatch):
    assert not {k for k in search.phase_two_space("MIL-CREDA")} & {
        "sigma", "sigma2", "sigmaMultiplier", "kernelSigma"}
    path_p, sha = _partition_file(tmp_path)
    data = search.search_data(LEVEL, _bagset(tmp_path, sha))
    hyper = search.phase_one_hyper("MIL-CREDA", PHASE_ONE, SCHEDULE)
    trained = search.train_phase_one("MIL-CREDA", data, 0, search.SEARCH_SEED, hyper, CPU)
    saved = search.save_phase_one(tmp_path / "Models", "MIL-CREDA", data, 0,
                                  search.SEARCH_SEED, trained)
    record = model_store.load(saved, partition_path=path_p)
    seen = []
    real = tp.adapt_target_phase

    def spy(model, phase_one, *args, **kwargs):
        seen.append((model.sigma, phase_one["checkpoint"]["sigma"]))
        return real(model, phase_one, *args, **kwargs)

    monkeypatch.setattr(search.two_phase, "adapt_target_phase", spy)
    out = search.train_phase_two("MIL-CREDA", data, 0, search.SEARCH_SEED, record,
                                 {"lr2": 1e-3, "wd2": 1e-5, "tauLocal": 1.0}, SCHEDULE, CPU)
    sigma1 = record[1]["sigma"]
    assert sigma1 == trained["result"]["checkpoint"]["sigma"]
    assert seen == [(sigma1, sigma1)] and out["result"]["checkpoint"]["sigma"] == sigma1
    saved2 = search.save_phase_two(tmp_path / "Models", "MIL-CREDA", data, 0,
                                   search.SEARCH_SEED, out)
    state2, meta2 = model_store.load(saved2, partition_path=path_p)
    assert meta2["sigma"] == sigma1 and meta2["hyper"]["tauLocal"] == 1.0
    assert all(torch.equal(state2[k], v) for k, v in out["model"].state_dict().items())
    no_sigma = (record[0], {**record[1], "sigma": None})
    with pytest.raises(search.SearchError, match="sigma"):
        search.train_phase_two("MIL-CREDA", data, 0, search.SEARCH_SEED, no_sigma,
                               {"lr2": 1e-3, "wd2": 1e-5, "tauLocal": 1.0}, SCHEDULE, CPU)
    with pytest.raises(search.SearchError):
        search.train_phase_two("MIL-CREDA", data, 0, search.SEARCH_SEED, record,
                               {"lr2": 1e-3, "wd2": 1e-5, "tauLocal": 1.0, "sigma": 2.0},
                               SCHEDULE, CPU)


def test_phase_two_starts_only_from_the_outer_folds_one_phase_one_checkpoint(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    good = ({}, {"phase": 1, "name": "MIL-CREDA", "transfer": LEVEL.name, "rotation": 2,
                 "seed": search.PHASE_ONE_SEED, partition.HASH_FIELD: data.partition_sha256,
                 "sigma": 1.0, "state_sha256": "s" * 64})
    search._check_phase_one_record("MIL-CREDA", data, 2, good)
    for change in ({"seed": search.SEARCH_SEED}, {"name": "common"},
                   {partition.HASH_FIELD: "y" * 64}, {"transfer": "L11->L1"},
                   {"rotation": 1}):
        bad = ({}, {**good[1], **change})
        with pytest.raises(search.SearchError):
            search.run_phase_two_study("MIL-CREDA", data, 2, bad, SCHEDULE, root=tmp_path,
                                       n_trials=1)
    assert not (tmp_path / "Results").exists()
    # every trial, in each of the three inner splits, starts from that one record
    used = []

    def fake(strategy, data_, rotation, seed, record, params, schedule, device):
        used.append((rotation, seed, record is good))
        d = data_.rotations[rotation]
        return {"result": {"checkpoint": {"sigma": 1.0},
                           "validation": _records(d["target_valid"], 1, lambda s, e: 0.7),
                           "history": [{"val_balanced_accuracy": 0.5}]}}

    monkeypatch.setattr(search, "train_phase_two", fake)
    study = search.run_phase_two_study("MIL-CREDA", data, 2, good, SCHEDULE, root=tmp_path,
                                       n_trials=2)
    assert used == [(2, search.SEARCH_SEED, True)] * 6
    assert study.user_attrs["phaseOneCheckpoint"] == search.phase_one_checkpoint_identity(good[1])
    # a stored study started from another checkpoint is not continued
    other = ({}, {**good[1], "state_sha256": "t" * 64})
    with pytest.raises(search.SearchError, match="phaseOneCheckpoint"):
        search.run_phase_two_study("MIL-CREDA", data, 2, other, SCHEDULE, root=tmp_path,
                                   n_trials=3)


def test_unwired_groups_and_strategies_refuse_before_creating_storage(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    monkeypatch.delitem(model_store.BUILDERS, "AdaptMIL", raising=False)
    with pytest.raises(model_store.GroupNotRegistered):
        search.run_phase_one_study("AdaptMIL", data, 0, SCHEDULE, root=tmp_path, n_trials=1)
    with pytest.raises(search.NotWired):
        search.run_phase_two_study("COT", data, 0, ({}, {}), SCHEDULE, root=tmp_path,
                                   n_trials=1)
    assert not (tmp_path / "Results").exists()


def test_a_two_trial_smoke_study_completes_and_writes_its_sqlite(tmp_path):
    path_p, sha = _partition_file(tmp_path)
    data = search.search_data(LEVEL, _bagset(tmp_path, sha))
    root = tmp_path / "MIL-CREDA"
    schedule = {**SCHEDULE, "epochs1": 2, "epochs2": 2}
    studies = {group: search.run_phase_one_study(group, data, 0, schedule, root=root,
                                                 n_trials=2)
               for group in ("common", "MIL-CREDA")}
    for group, study in studies.items():
        path = search.study_path(root, LEVEL, 1, group, 0)
        assert path.is_file() and path.parent == root / "Results" / "hpo"
        loaded = optuna.load_study(study_name=path.stem, storage=f"sqlite:///{path}")
        assert [t.state for t in loaded.trials] == [optuna.trial.TrialState.COMPLETE] * 2
        assert loaded.user_attrs["sampler"] == "GPSampler"
        assert loaded.user_attrs["samplerSeed"] == search.SAMPLER_SEED
        assert loaded.user_attrs["searchSeed"] == search.SEARCH_SEED
        assert loaded.user_attrs["outerFold"] == 0
    chosen = search.select(studies["MIL-CREDA"])
    best = search.phase_one_hyper("MIL-CREDA", chosen["params"],
                                  search.refit_schedule(schedule, 1, chosen["epochs"]))
    trained = search.train_phase_one("MIL-CREDA", data, 0, search.PHASE_ONE_SEED, best, CPU)
    # the refit: every non-test fold, exactly E1 epochs, no validation
    assert trained["result"]["epochs"] == chosen["epochs"]
    assert trained["result"]["validation"] == []
    saved = search.save_phase_one(root / "Models", "MIL-CREDA", data, 0, search.PHASE_ONE_SEED,
                                  trained)
    record = model_store.load(saved, partition_path=path_p)
    study = search.run_phase_two_study("MIL-CREDA", data, 0, record, schedule, root=root,
                                       n_trials=2)
    path = search.study_path(root, LEVEL, 2, "MIL-CREDA", 0)
    assert path.is_file() and path.name == "L1-to-L11__phase2__MIL-CREDA__outer0.db"
    assert len(study.trials) == 2
    for trial in study.trials:
        assert set(trial.params) == {"lr2", "wd2", "tauLocal"}
        assert math.isfinite(trial.user_attrs["label_free_score"])
        assert len(trial.user_attrs["pooledLabelFree"]) == schedule["epochs2"]
        assert trial.user_attrs["sigmaBySplit"] == [record[1]["sigma"]] * 3


def _lowest(space):
    return {k: (d.low if isinstance(d, FloatDistribution) else d.choices[0])
            for k, d in space.items()}


def test_a_registered_phase_one_trainer_replaces_the_shared_one_for_its_group_only(
        tmp_path, monkeypatch):
    monkeypatch.setattr(search, "PHASE_ONE_TRAINERS", dict(search.PHASE_ONE_TRAINERS))
    monkeypatch.setattr(model_store, "BUILDERS", dict(model_store.BUILDERS))
    model_store.BUILDERS["AdaptMIL"] = model_store.BUILDERS["common"]
    calls = []

    def trainer(name):
        def run(model, source_train, source_valid, hyper, generator, device, *, seed, fold,
                transfer):
            calls.append((name, fold, transfer, len(source_valid)))
            return {"history": [{"epoch": 1}], "checkpoint": {}}
        return run

    monkeypatch.setattr(tp, "train_source_phase", trainer("shared"))
    search.register_phase_one("AdaptMIL", trainer("native"))
    data = search.search_data(LEVEL, _bagset(tmp_path))
    for group in ("AdaptMIL", "common"):
        hyper = search.phase_one_hyper(group, _lowest(search.PHASE_ONE_SPACES[group]),
                                       SCHEDULE)
        out = search.train_phase_one(group, data, 0, search.SEARCH_SEED, hyper, CPU)
        assert out["result"]["history"] == [{"epoch": 1, "model": f"{group}:phase1"}]
        search.train_phase_one(group, data.inner_view(0, 1), 0, search.SEARCH_SEED, hyper,
                               CPU)
    valid = len(data.inner[(0, 1)]["source_valid"])
    assert valid > 0
    assert calls == [("native", 0, LEVEL.name, 0), ("native", 0, LEVEL.name, valid),
                     ("shared", 0, LEVEL.name, 0), ("shared", 0, LEVEL.name, valid)]


def test_only_a_group_with_its_own_unit_registers_a_phase_one_trainer(monkeypatch):
    monkeypatch.setattr(search, "PHASE_ONE_TRAINERS", dict(search.PHASE_ONE_TRAINERS))
    assert search.PHASE_ONE_TRAINERS == {}
    for group in ("common", "MIL-CREDA", "UDA-DDA"):
        with pytest.raises(search.SearchError):
            search.register_phase_one(group, print)
    search.register_phase_one("AdaptMIL", print)
    assert search.PHASE_ONE_TRAINERS == {"AdaptMIL": print}


def _scripted_phase_one(fail_at=None):
    calls = []

    def fake(group, data_, rotation, seed, hyper, device):
        calls.append(rotation)
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError("the session was cut")
        bags = data_.rotations[rotation]["source_valid"]
        records = _records(bags, hyper["epochs1"], lambda s, e: 0.5 + 0.4 * hyper["lr1"])
        return {"result": {"validation": records,
                           "history": [{"val_balanced_accuracy": 0.5} for _ in records]}}

    return fake


def test_an_interrupted_study_of_the_same_identity_continues_to_its_declared_count(
        tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    # the second trial dies in its third inner split: one complete trial and one failed
    monkeypatch.setattr(search, "train_phase_one", _scripted_phase_one(fail_at=6))
    with pytest.raises(RuntimeError, match="cut"):
        search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=3)
    path = search.study_path(tmp_path, LEVEL, 1, "common", 0)
    cut = optuna.load_study(study_name=path.stem, storage=f"sqlite:///{path}")
    states = [t.state for t in cut.trials]
    assert states == [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL]
    monkeypatch.setattr(search, "train_phase_one", _scripted_phase_one())
    study = search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=3)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == 3
    # the continuation does not replay the sampler's first draws
    params = [json.dumps(t.params, sort_keys=True) for t in study.trials]
    assert len(set(params)) == len(params)
    assert study.user_attrs["resumedAfter"] == [2]
    # a complete study is returned as it is, never extended
    again = search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=3)
    assert len(again.trials) == len(study.trials)


def test_a_study_of_another_identity_is_refused_naming_what_differs(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    monkeypatch.setattr(search, "train_phase_one", _scripted_phase_one())
    search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=1)
    with monkeypatch.context() as m:
        m.setattr(model_store, "code_version", lambda: "0" * 64)
        with pytest.raises(search.SearchError, match="code_version"):
            search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=1)
    with monkeypatch.context() as m:
        m.setitem(search.__provenance__, "revision", "old-v01.md")
        with pytest.raises(search.SearchError, match="revision"):
            search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=1)
    other = search.SearchData(transfer=data.transfer, partition_sha256="y" * 64,
                              rotations=data.rotations, folds=data.folds, inner=data.inner)
    with pytest.raises(search.SearchError, match="partition_sha256"):
        search.run_phase_one_study("common", other, 0, SCHEDULE, root=tmp_path, n_trials=1)
    with pytest.raises(search.SearchError, match="schedule"):
        search.run_phase_one_study("common", data, 0, {**SCHEDULE, "epochs1": 2},
                                   root=tmp_path, n_trials=1)
    with pytest.raises(search.SearchError, match="nTrials"):
        search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=2)
    # another outer fold is another study, in its own file
    search.run_phase_one_study("common", data, 1, SCHEDULE, root=tmp_path, n_trials=2)


def test_a_study_records_the_identity_it_is_checked_against(tmp_path, monkeypatch):
    data = search.search_data(LEVEL, _bagset(tmp_path))
    monkeypatch.setattr(search, "train_phase_one", _scripted_phase_one())
    study = search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=1)
    assert study.user_attrs["code_version"] == model_store.code_version()
    assert study.user_attrs["partition_sha256"] == data.partition_sha256
    assert study.user_attrs["revision"] == search.__provenance__["revision"]
    assert study.user_attrs["revision"].endswith("v56.md")
    assert study.user_attrs["outerFold"] == 0 and study.user_attrs["testFold"] == 0
    assert study.user_attrs["innerSplits"] == partition.rotation_roles(0)["inner_splits"]


def test_a_study_stops_between_trials_when_asked_and_the_rerun_completes_it(
        tmp_path, monkeypatch):
    """A deadline hook stops a study between trials; the next call continues it."""
    data = search.search_data(LEVEL, _bagset(tmp_path))
    monkeypatch.setattr(search, "train_phase_one", _scripted_phase_one())
    asked = []

    def stop_after_one(study):
        asked.append(len(study.trials))
        return len(study.trials) >= 1

    study = search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=3,
                                       should_stop=stop_after_one)
    states = [t.state for t in study.trials]
    assert states == [optuna.trial.TrialState.COMPLETE]
    assert asked == [0, 1], "asked before the first trial and after each one"
    # a hook that stops at once runs no trial at all
    held = search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=3,
                                      should_stop=lambda s: True)
    assert len(held.trials) == 1
    done = search.run_phase_one_study("common", data, 0, SCHEDULE, root=tmp_path, n_trials=3)
    assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in done.trials) == 3
    # no hook: the study runs its declared count in one call, as before
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    whole = search.run_phase_one_study("common", data, 0, SCHEDULE, root=fresh, n_trials=2)
    assert "resumedAfter" not in whole.user_attrs and len(whole.trials) == 2
