"""Experiments plan v44 Sec. 4.5 and 7.7: the label-free rule picks its phase-2 epoch too.

Phase 2 runs its whole epoch budget and keeps the best epoch "con la regla de la
sección 4.5" (Sec. 7.7). The secondary rule "no usa etiquetas del objetivo", so the
model it selects must keep the epoch with the best label-free score, never the epoch
the target-validation balanced accuracy chose. Every phase-2 trainer (MIL-CREDA and
each comparator) retains that state when asked, and only then.

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
#: A rehearsal schedule, not the pilot's: enough epochs, and learning rates at the top of
#: the Sec. 7.7 ranges, for the predictions to move between phase-2 epochs (with one
#: epoch each, every model here predicted one class at every epoch and all epochs tied).
#: CCA-LSC needs more phase-2 epochs than its three warm-up epochs.
SCHEDULE = {"epochs1": 6, "patience1": 6, "epochs2": 5, "bagsPerStep": 4}
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


def _train(setting, strategy, flipped=False, **kwargs):
    """One seeded phase-2 training, cached per (strategy, flipped labels, kwargs)."""
    key = (strategy, flipped, tuple(sorted(kwargs.items())))
    if key not in _TRAINED:
        base, records = setting
        record = records[search.PHASE_ONE_GROUP_OF[strategy]]
        params = {**_sample(search.phase_two_space(strategy)), "lr2": LR2}
        _TRAINED[key] = search.train_phase_two(
            strategy, _flipped(base) if flipped else base, ROTATION, SEED, record, params,
            SCHEDULE, **kwargs)
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
    """The same data with every target-validation label of the rotation flipped."""
    view = dict(data.rotations[ROTATION])
    view["target_valid"] = [replace(b, label=1 - b.label) for b in view["target_valid"]]
    return replace(data, rotations={**data.rotations, ROTATION: view})


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
def test_the_label_free_state_is_the_earliest_best_label_free_epoch(setting, strategy):
    out = _train(setting, strategy, label_free=True)
    free = out["result"]["label_free_best"]
    scores = [h["label_free_score"] for h in out["result"]["history"]]
    assert len(scores) == SCHEDULE["epochs2"]
    best = max(scores)
    assert free["epoch"] == scores.index(best)        # ties: the earliest, as the primary
    assert free["label_free_score"] == best
    assert free["rule"] == two_phase.LABEL_FREE_EPOCH_RULE
    import copy
    model = copy.deepcopy(out["model"])
    model.load_state_dict(free["state"])
    target_valid = setting[0].rotations[ROTATION]["target_valid"]
    G = two_phase.evaluate(model, target_valid, "target", torch.device("cpu"))
    assert two_phase.label_free_score(G) == pytest.approx(best, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_target_validation_label_changes_the_label_free_epoch(setting, strategy):
    out = _train(setting, strategy, label_free=True)["result"]["label_free_best"]
    flipped = _train(setting, strategy, True, label_free=True)["result"]["label_free_best"]
    assert flipped["epoch"] == out["epoch"]
    assert flipped["label_free_score"] == out["label_free_score"]
    assert _equal(flipped["state"], out["state"])


def test_the_flip_is_seen_by_the_primary_rule(setting):
    """The flip test is not vacuous: on this fixture, flipping the target-validation labels
    moves the primary epoch of some strategy, whose label-free epoch stays."""
    moved = [s for s in STRATEGIES
             if _train(setting, s, label_free=True)["result"]["best_epoch"]
             != _train(setting, s, True, label_free=True)["result"]["best_epoch"]]
    assert moved
    differ = [s for s in STRATEGIES
              if _train(setting, s, label_free=True)["result"]["label_free_best"]["epoch"]
              != _train(setting, s, label_free=True)["result"]["best_epoch"]]
    assert differ


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
