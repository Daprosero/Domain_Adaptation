"""Experiments plan v58, Secs. 4.5, 7.3 and 7.7: the phase-2 selection rules.

The primary rule is the centroid rule: per inner split and epoch, the adapted model's
target validation predictions are pseudo-labels, their representations Z (the vector
entering the predictive head) give one centroid per pseudo-class, and the source
validation bags, through the frozen source side, are assigned to the nearest centroid and
scored against their real SOURCE labels; the counts are pooled over the three inner splits
and the balanced accuracy is computed once. The secondary rule is EnsV, on the same
predictions. Neither reads a target label, and neither trains anything (Sec. 7.7).

On an inner split every phase-2 trainer (MIL-CREDA and each comparator) runs its whole epoch
cap, keeps the last epoch and records every epoch's target-validation predictions and
representations; on a refit it runs exactly the epochs it is given. Rehearsed on a SYNTHETIC
set on CPU: seeded noise with a weak class signal; no number it yields is a measured result.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import replace

import numpy as np
import optuna
import pytest
import torch

from MIL_CREDA_EEG import campaign, eegnet, model_store, search, two_phase
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel

CPU = torch.device("cpu")
SEED = 1
ROTATION = 1
SPLIT = 0
#: A rehearsal schedule, not the pilot's: enough epochs, and learning rates at the top of
#: the Sec. 7.7 ranges, for the predictions to move between phase-2 epochs.
#: CCA-LSC needs more phase-2 epochs than its three warm-up epochs.
SCHEDULE = {"epochs1": 6, "epochs2": 5, "bagsPerStep": 4}
LR1, LR2 = 1e-2, 1e-2


# ------------------------------------------------------------------ the centroid rule


def test_the_centroid_rule_assigns_each_source_bag_to_the_nearest_pseudo_class_centroid():
    # target pseudo-labels: argmax -> [0, 0, 1]; centroids (0, 0) and (10, 0)
    target_Z = [[-1.0, 0.0], [1.0, 0.0], [10.0, 0.0]]
    P = [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]
    source_Z = [[0.5, 0.0], [9.0, 1.0], [6.0, 0.0], [4.0, 0.0]]
    source_labels = [0, 1, 0, 1]
    counts = two_phase.centroid_counts(target_Z, P, source_Z, source_labels)
    # assigned: 0, 1, 1 (6 is nearer 10 than 0), 0 (4 is nearer 0)
    assert counts.tolist() == [[1, 1], [1, 1]]
    assert two_phase.balanced_accuracy_of_counts(counts) == 0.5
    perfect = two_phase.centroid_counts(target_Z, P, source_Z, [0, 1, 1, 0])
    assert two_phase.balanced_accuracy_of_counts(perfect) == 1.0


def test_a_collapse_to_one_pseudo_class_scores_one_half_with_two_classes():
    P = [[0.9, 0.1], [0.7, 0.3], [0.6, 0.4]]          # every target bag predicts class 0
    counts = two_phase.centroid_counts(np.random.default_rng(0).normal(size=(3, 4)), P,
                                       np.random.default_rng(1).normal(size=(6, 4)),
                                       [0, 1, 0, 1, 1, 0])
    assert counts[:, 1].sum() == 0                     # class 1 has no centroid
    assert two_phase.balanced_accuracy_of_counts(counts) == 0.5


def test_a_class_absent_from_the_source_labels_does_not_crash():
    counts = two_phase.centroid_counts([[0.0], [5.0]], [[0.8, 0.2], [0.1, 0.9]],
                                       [[0.1], [4.0]], [1, 1])
    assert counts.tolist() == [[0, 0], [1, 1]]
    assert two_phase.balanced_accuracy_of_counts(counts) == 0.5


def _target(subjects, P, Z, labels=None, epoch=0):
    return {"epoch": epoch, "subjects": list(subjects), "probabilities": P,
            "representations": Z, "labels": labels or [0] * len(subjects)}


def _source(subjects, Z, labels):
    return {"subjects": list(subjects), "representations": Z, "labels": labels}


def test_the_pooled_centroid_score_sums_the_counts_and_reads_no_target_label():
    targets = [_target("ab", [[0.9, 0.1], [0.1, 0.9]], [[0.0], [1.0]], [0, 1]),
               _target("cd", [[0.9, 0.1], [0.2, 0.8]], [[0.0], [1.0]], [1, 1]),
               _target("e", [[0.3, 0.7]], [[2.0]], [0])]
    sources = [_source(["s1", "s2", "s3"], [[0.1], [0.2], [0.9]], [0, 0, 1]),
               _source(["s4"], [[0.9]], [0]),
               _source(["s5", "s6"], [[0.0], [5.0]], [1, 1])]
    pooled = two_phase.pooled_centroid_balanced_accuracy(targets, sources)
    total = sum(two_phase.centroid_counts(t["representations"], t["probabilities"],
                                          s["representations"], s["labels"], 2)
                for t, s in zip(targets, sources))
    assert pooled == two_phase.balanced_accuracy_of_counts(total)
    per_split = [two_phase.balanced_accuracy_of_counts(two_phase.centroid_counts(
        t["representations"], t["probabilities"], s["representations"], s["labels"], 2))
        for t, s in zip(targets, sources)]
    assert not math.isclose(pooled, float(np.nanmean(per_split)))
    flipped = [{**t, "labels": [1 - y for y in t["labels"]]} for t in targets]
    unlabelled = [{k: v for k, v in t.items() if k != "labels"} for t in targets]
    assert two_phase.pooled_centroid_balanced_accuracy(flipped, sources) == pooled
    assert two_phase.pooled_centroid_balanced_accuracy(unlabelled, sources) == pooled
    with pytest.raises(ValueError, match="once"):
        two_phase.pooled_centroid_balanced_accuracy([targets[0], targets[0]], sources[:2])


def test_the_device_arm_writes_both_domains_in_one_representation_space():
    torch.manual_seed(0)
    model = MILCREDAModel(29, 14, attention_gamma=0.5, attention_temperature=1.0)
    model.fix_sigma(1.0)
    g = torch.Generator().manual_seed(0)
    source = [two_phase.BagTensor(f"s{i}", "c", i % 2, torch.randn(3, 29, eegnet.SAMPLES,
                                                                     generator=g))
              for i in range(4)]
    target = [two_phase.BagTensor(f"t{i}", "c", i % 2, torch.randn(3, 14, eegnet.SAMPLES,
                                                                     generator=g))
              for i in range(3)]
    Zs = two_phase.represent(model, source, "source", CPU)
    Zt = two_phase.represent(model, target, "target", CPU)
    assert Zs.shape == (4, model.source_trunk.output_dim) == (4, 256)
    assert Zt.shape == (3, 256)
    # Z is the vector the head reads: the head of Z is the prediction
    torch.testing.assert_close(model.classify(Zt), two_phase.evaluate(model, target, "target",
                                                                      CPU).to(torch.float64))
    P = two_phase.evaluate(model, target, "target", CPU)
    counts = two_phase.centroid_counts(Zt, P, Zs, [b.label for b in source])
    assert counts.sum() == 4


# ------------------------------------------------------------------ every trainer


@pytest.fixture(scope="module")
def registries():
    """Comparator registration mutates global search state; restore it afterwards."""
    saved = {(m, name): dict(getattr(m, name)) for m, name in (
        (search, "PHASE_TWO_TRAINERS"), (search, "PHASE_TWO_EXTRA"),
        (search, "PHASE_ONE_TRAINERS"), (model_store, "BUILDERS"))}
    campaign.register(model_store.STRATEGIES)
    yield
    for (m, name), value in saved.items():
        getattr(m, name).clear()
        getattr(m, name).update(value)


def _sample(space) -> dict:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    return search.suggest(study.ask(), space)


@pytest.fixture(scope="module")
def setting(tmp_path_factory, registries):
    tmp = tmp_path_factory.mktemp("phase-two-rules")
    cfg = campaign.write_synthetic_set(tmp / "synthetic", seed=0, subjects=24)
    bags = campaign.load_bags(cfg)
    transfer = campaign.transfers_of({"transfers": ["L1->L6"]})[0]
    data = search.search_data(transfer, bags)
    records = {}
    for group in model_store.GROUPS:
        params = {**_sample(search.PHASE_ONE_SPACES[group]), "lr1": LR1}
        hyper = search.phase_one_hyper(group, params, SCHEDULE)
        trained = search.train_phase_one(group, data, ROTATION, SEED, hyper)
        path = search.save_phase_one(tmp / "Models", group, data, ROTATION, SEED, trained)
        records[group] = model_store.load(path, partition_path=campaign._partition_path(cfg))
    return data, records


_TRAINED: dict = {}


def _train(setting, strategy, flipped=False, refit=False):
    """One seeded phase-2 training on the inner split (or on the refit view, two epochs),
    cached per (strategy, flipped target labels, refit)."""
    key = (strategy, flipped, refit)
    if key not in _TRAINED:
        base, records = setting
        record = records[search.PHASE_ONE_GROUP_OF[strategy]]
        params = {**_sample(search.phase_two_space(strategy)), "lr2": LR2}
        data = base if refit else (_flipped(base) if flipped else base).inner_view(
            ROTATION, SPLIT)
        schedule = search.refit_schedule(SCHEDULE, 2, 2) if refit else SCHEDULE
        _TRAINED[key] = search.train_phase_two(strategy, data, ROTATION, SEED, record, params,
                                               schedule)
    return _TRAINED[key]


def _flipped(data: search.SearchData) -> search.SearchData:
    """The same data with every target label of the inner split flipped."""
    view = dict(data.inner[(ROTATION, SPLIT)])
    for role in ("target_train", "target_valid"):
        view[role] = [replace(b, label=1 - b.label) for b in view[role]]
    return replace(data, inner={**data.inner, (ROTATION, SPLIT): view})


STRATEGIES = list(model_store.STRATEGIES)


def test_no_trainer_offers_a_label_free_option(registries):
    for strategy, trainer in search.PHASE_TWO_TRAINERS.items():
        assert "label_free" not in inspect.signature(trainer).parameters, strategy
    assert not hasattr(two_phase, "LabelFreeBest")
    assert not hasattr(two_phase, "label_free_score")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_trainer_records_predictions_and_representations_per_epoch(setting, strategy):
    trained = _train(setting, strategy)
    out = trained["result"]
    valid = setting[0].inner[(ROTATION, SPLIT)]["target_valid"]
    assert out["epochs"] == SCHEDULE["epochs2"] == len(out["history"])
    assert out["best_epoch"] == SCHEDULE["epochs2"] - 1 == out["checkpoint"]["epoch"]
    assert [v["epoch"] for v in out["validation"]] == list(range(SCHEDULE["epochs2"]))
    for v in out["validation"]:
        assert v["subjects"] == [b.subject for b in valid]
        assert len(v["probabilities"]) == len(valid)
        assert np.asarray(v["representations"]).shape == (len(valid), 256)
    # the last record is the kept model's: its predictions and its representations
    last = out["validation"][-1]
    np.testing.assert_allclose(
        last["representations"], two_phase.represent(trained["model"], valid, "target", CPU))
    np.testing.assert_allclose(
        last["probabilities"],
        two_phase.evaluate(trained["model"], valid, "target", CPU).to(torch.float64))
    for h in out["history"]:
        assert not any("label_free" in k for k in h)
    assert out["selection"] == two_phase.SELECTION


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_trainer_refits_exactly_e_epochs_without_validation(setting, strategy):
    out = _train(setting, strategy, refit=True)["result"]
    assert out["epochs"] == 2 == len(out["history"]) and out["validation"] == []
    assert out["checkpoint"]["epoch"] == 1
    assert all(h["val_balanced_accuracy"] != h["val_balanced_accuracy"]
               for h in out["history"])                                  # NaN: no validation


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_target_label_changes_what_either_rule_reads(setting, strategy):
    base = _train(setting, strategy)["result"]["validation"]
    flipped = _train(setting, strategy, True)["result"]["validation"]
    for key in ("subjects", "probabilities", "representations"):
        assert [v[key] for v in flipped] == [v[key] for v in base], key


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_source_pass_is_the_frozen_source_side_of_every_trial(setting, strategy):
    data, records = setting
    sources = search.source_representations(
        strategy, data, ROTATION, records[search.PHASE_ONE_GROUP_OF[strategy]])
    assert len(sources) == search.N_INNER
    bags = data.inner[(ROTATION, SPLIT)]["source_valid"]
    assert sources[SPLIT]["subjects"] == [b.subject for b in bags]
    assert sources[SPLIT]["labels"] == [b.label for b in bags]
    # the trained phase-2 model's source side gives the same representations
    np.testing.assert_allclose(
        sources[SPLIT]["representations"],
        two_phase.represent(_train(setting, strategy)["model"], bags, "source", CPU),
        rtol=1e-6, atol=1e-9)
    score = two_phase.pooled_centroid_balanced_accuracy(
        [_train(setting, strategy)["result"]["validation"][-1]], [sources[SPLIT]])
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_phase_two_selection_reads_no_target_label(setting, strategy):
    """With every target label of the inner split flipped, the recorded labels change and
    the centroid score of every epoch does not."""
    data, records = setting
    sources = search.source_representations(
        strategy, data, ROTATION, records[search.PHASE_ONE_GROUP_OF[strategy]])
    base = _train(setting, strategy)["result"]["validation"]
    flipped = _train(setting, strategy, True)["result"]["validation"]
    assert [r["labels"] for r in flipped] == [[1 - y for y in r["labels"]] for r in base]
    for b, f in zip(base, flipped, strict=True):
        assert two_phase.pooled_centroid_balanced_accuracy([f], [sources[SPLIT]]) == \
            two_phase.pooled_centroid_balanced_accuracy([b], [sources[SPLIT]])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_representation_is_the_vector_entering_the_predictive_head(setting, strategy):
    """`represent` gives Z, and the model's own head applied to Z gives exactly the
    probabilities `evaluate` predicts."""
    model = _train(setting, strategy)["model"]
    valid = setting[0].inner[(ROTATION, SPLIT)]["target_valid"]
    Z = two_phase.represent(model, valid, "target", CPU)
    G = two_phase.evaluate(model, valid, "target", CPU)
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            dtype = next(model.head.parameters()).dtype
            through_head = model.head(Z.to(dtype))
    finally:
        for m, mode in modes.items():
            m.train(mode)
    np.testing.assert_allclose(through_head.to(torch.float64).numpy(),
                               G.to(torch.float64).numpy(), rtol=1e-5, atol=1e-7)


def test_the_source_pass_runs_in_the_search_never_in_a_trainer(setting, monkeypatch):
    """No registered phase-2 trainer calls the source pass: with it replaced by one that
    refuses, every trainer still trains; and none of their sources names it."""
    for strategy, trainer in search.PHASE_TWO_TRAINERS.items():
        assert "source_representations" not in inspect.getsource(trainer), strategy
        module = inspect.getmodule(trainer)
        if module is not search:
            assert "source_representations" not in inspect.getsource(module), strategy

    def refuse(*args, **kwargs):
        raise AssertionError("a phase-2 trainer ran the source pass")

    monkeypatch.setattr(search, "source_representations", refuse)
    base, records = setting
    for strategy in STRATEGIES:
        record = records[search.PHASE_ONE_GROUP_OF[strategy]]
        params = {**_sample(search.phase_two_space(strategy)), "lr2": LR2}
        out = search.train_phase_two(strategy, base, ROTATION, SEED, record, params,
                                     search.refit_schedule(SCHEDULE, 2, 2))   # a refit
        assert out["result"]["epochs"] == 2, strategy
