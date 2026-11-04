"""Experiments plan v50 Secs. 4.5 and 7.7: every phase-2 trainer on an inner split and on a
refit.

On an inner split each trainer (MIL-CREDA and each comparator) runs its whole epoch cap,
keeps the last epoch and records every epoch's target-validation predictions, which the
search pools over the three inner splits; the secondary rule "no usa etiquetas del
objetivo", so those predictions must not depend on a target-validation label. On a refit
(every non-test fold, no validation bags) it runs exactly the epochs it is given. The
per-run label-free state (`two_phase.LabelFreeBest`) is still kept when asked, and only
then.

Rehearsed on a SYNTHETIC set on CPU: seeded noise with a weak class signal; no number it
yields is a measured result.
"""

from __future__ import annotations

from dataclasses import replace

import optuna
import pytest
import torch

from MIL_CREDA_EEG import campaign, model_store, search, two_phase

SEED = 1
ROTATION = 1
SPLIT = 0
#: A rehearsal schedule, not the pilot's: enough epochs, and learning rates at the top of
#: the Sec. 7.7 ranges, for the predictions to move between phase-2 epochs (with one
#: epoch each, every model here predicted one class at every epoch and all epochs tied).
#: CCA-LSC needs more phase-2 epochs than its three warm-up epochs.
SCHEDULE = {"epochs1": 6, "epochs2": 5, "bagsPerStep": 4}
LR1, LR2 = 1e-2, 1e-2


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
    tmp = tmp_path_factory.mktemp("label-free-epoch")
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


def _train(setting, strategy, flipped=False, refit=False, **kwargs):
    """One seeded phase-2 training on the inner split (or on the refit view, two epochs),
    cached per (strategy, flipped labels, refit, kwargs)."""
    key = (strategy, flipped, refit, tuple(sorted(kwargs.items())))
    if key not in _TRAINED:
        base, records = setting
        record = records[search.PHASE_ONE_GROUP_OF[strategy]]
        params = {**_sample(search.phase_two_space(strategy)), "lr2": LR2}
        data = base if refit else (_flipped(base) if flipped else base).inner_view(
            ROTATION, SPLIT)
        schedule = search.refit_schedule(SCHEDULE, 2, 2) if refit else SCHEDULE
        _TRAINED[key] = search.train_phase_two(strategy, data, ROTATION, SEED, record, params,
                                               schedule, **kwargs)
    return _TRAINED[key]


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            _equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_equal, a, b))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True                                  # NaN placeholders match themselves
    return a == b


def _flipped(data: search.SearchData) -> search.SearchData:
    """The same data with every target-validation label of the inner split flipped."""
    view = dict(data.inner[(ROTATION, SPLIT)])
    view["target_valid"] = [replace(b, label=1 - b.label) for b in view["target_valid"]]
    return replace(data, inner={**data.inner, (ROTATION, SPLIT): view})


STRATEGIES = list(model_store.STRATEGIES)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_asking_for_the_label_free_state_changes_nothing_of_the_default(setting, strategy):
    default = _train(setting, strategy)
    asked = _train(setting, strategy, label_free=True)
    assert "label_free_best" not in default["result"]
    assert "label_free_best" in asked["result"]
    rest = {k: v for k, v in asked["result"].items() if k != "label_free_best"}
    for key in ("checkpoint", "history", "best_epoch", "epochs", "selection"):
        assert _equal(default["result"].get(key), rest.get(key)), key
    assert _equal(default["model"].state_dict(), asked["model"].state_dict())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_trainer_runs_its_cap_keeps_the_last_epoch_and_records_validation(setting,
                                                                                strategy):
    out = _train(setting, strategy)["result"]
    valid = setting[0].inner[(ROTATION, SPLIT)]["target_valid"]
    assert out["epochs"] == SCHEDULE["epochs2"] == len(out["history"])
    assert out["best_epoch"] == SCHEDULE["epochs2"] - 1 == out["checkpoint"]["epoch"]
    assert [v["epoch"] for v in out["validation"]] == list(range(SCHEDULE["epochs2"]))
    for v in out["validation"]:
        assert v["subjects"] == [b.subject for b in valid]
        assert len(v["probabilities"]) == len(valid)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_trainer_refits_exactly_e_epochs_without_validation(setting, strategy):
    out = _train(setting, strategy, refit=True)["result"]
    assert out["epochs"] == 2 == len(out["history"]) and out["validation"] == []
    assert out["checkpoint"]["epoch"] == 1
    assert all(h["val_balanced_accuracy"] != h["val_balanced_accuracy"]
               for h in out["history"])                                  # NaN: no validation


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_label_free_state_is_the_earliest_best_label_free_epoch(setting, strategy):
    out = _train(setting, strategy, label_free=True)
    free = out["result"]["label_free_best"]
    scores = [h["label_free_score"] for h in out["result"]["history"]]
    assert len(scores) == SCHEDULE["epochs2"]
    best = max(scores)
    assert free["epoch"] == scores.index(best)        # ties: the earliest
    assert free["label_free_score"] == best
    assert free["rule"] == two_phase.LABEL_FREE_EPOCH_RULE
    import copy
    model = copy.deepcopy(out["model"])
    model.load_state_dict(free["state"])
    target_valid = setting[0].inner[(ROTATION, SPLIT)]["target_valid"]
    G = two_phase.evaluate(model, target_valid, "target", torch.device("cpu"))
    assert two_phase.label_free_score(G) == pytest.approx(best, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_target_validation_label_changes_the_label_free_epoch(setting, strategy):
    base = _train(setting, strategy, label_free=True)["result"]
    flipped = _train(setting, strategy, True, label_free=True)["result"]
    assert flipped["label_free_best"]["epoch"] == base["label_free_best"]["epoch"]
    assert _equal(flipped["label_free_best"]["state"], base["label_free_best"]["state"])
    # the predictions the search pools are those of the same models: the pooled label-free
    # score (which reads no label) is the same, epoch by epoch
    assert [v["probabilities"] for v in flipped["validation"]] == \
        [v["probabilities"] for v in base["validation"]]
    assert [two_phase.pooled_label_free_score([v]) for v in flipped["validation"]] == \
        [two_phase.pooled_label_free_score([v]) for v in base["validation"]]


def test_the_flip_is_seen_by_the_primary_rule(setting):
    """The flip test is not vacuous: on this fixture, flipping the target-validation labels
    moves the pooled balanced accuracy of some strategy, whose label-free series stays."""
    def series(result):
        return [two_phase.pooled_balanced_accuracy([v]) for v in result["validation"]]

    moved = [s for s in STRATEGIES
             if series(_train(setting, s, label_free=True)["result"])
             != series(_train(setting, s, True, label_free=True)["result"])]
    assert moved


def test_the_keeper_breaks_ties_by_the_earliest_epoch_as_the_primary_rule_does():
    """Mirrors the primary rule's `if accuracy > best`: equal scores keep the first epoch;
    asked for nothing, it keeps nothing."""
    module = torch.nn.Linear(1, 1, bias=False)
    keeper = two_phase.LabelFreeBest(True)
    for epoch, score in enumerate([0.2, 0.5, 0.5, 0.1]):
        with torch.no_grad():
            module.weight.fill_(float(epoch))
        keeper.offer(epoch, score, module)
    kept = keeper.checkpoint()
    assert kept["epoch"] == 1 and kept["label_free_score"] == 0.5
    assert float(kept["state"]["weight"]) == 1.0
    assert kept["rule"] == two_phase.LABEL_FREE_EPOCH_RULE
    idle = two_phase.LabelFreeBest(False)
    idle.offer(0, 0.9, module)
    assert idle.checkpoint() is None
