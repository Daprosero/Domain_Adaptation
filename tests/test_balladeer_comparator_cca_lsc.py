"""CCA-LSC paper equations and controlled BALLADEER adaptation invariants."""
import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from MIL_CREDA_EEG import eegnet, mil_wrapper, model_store, search, two_phase
from MIL_CREDA_EEG.comparators import cca_lsc
from MIL_CREDA_EEG.comparators.cca_lsc import losses, training
from MIL_CREDA_EEG.comparators.cca_lsc.smoke import fixture, run


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def params():
    return {**training.DEFAULTS, "lr2": 0.001, "wd2": 0.}


def schedule():
    return {"epochs1": 1, "epochs2": 4, "bagsPerStep": 4}


def test_cca_calibration_uses_raw_confidence():
    p = torch.tensor([[0.51, 0.49], [0.99, 0.01]], dtype=torch.float64, requires_grad=True)
    shift = losses.label_shift(torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9]))
    weights = losses.calibration_weights(shift)
    labels, confidence = losses.calibrated_pseudo(p, weights)
    assert labels.tolist() == [1, 0]
    assert confidence.tolist() == [0.49, 0.99]
    assert not confidence.requires_grad
    identity = losses.calibration_weights(torch.ones(2))
    assert losses.calibrated_pseudo(p, identity)[0].tolist() == [0, 0]
    torch.testing.assert_close(weights, 1 / (1.5 + torch.exp(-shift.sqrt())))


def test_cca_prior_uses_confident_bags_only():
    p = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.5, 0.5]])
    torch.testing.assert_close(losses.estimate_prior(p), torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(losses.estimate_prior(p, 0.8), torch.tensor([1., 0.]))
    torch.testing.assert_close(losses.estimate_prior(p, 0.99), torch.tensor([0.5, 0.5]))


def test_cca_weighted_centroids_and_memory():
    f = torch.tensor([[0., 0.], [4., 0.], [0., 8.]], dtype=torch.float64, requires_grad=True)
    y, w = torch.tensor([0, 0, 1]), torch.tensor([0.25, 0.75, 1.])
    memory = losses.CentroidMemory(2, 2, 0.5)
    c, active = memory(f, y, w)
    torch.testing.assert_close(c, torch.tensor([[3., 0.], [0., 8.]], dtype=torch.float64))
    assert active.tolist() == [True, True]
    c.sum().backward()
    assert torch.isfinite(f.grad).all() and f.grad.abs().sum() > 0
    previous = memory.centroids.clone()
    next_features = torch.tensor([[5., 2.]], dtype=torch.float64, requires_grad=True)
    new, _ = memory(next_features, torch.tensor([0]), torch.ones(1))
    torch.testing.assert_close(new[0], torch.tensor([4., 1.], dtype=torch.float64))
    torch.testing.assert_close(new[1], previous[1])
    new.sum().backward()
    assert next_features.grad.abs().sum() > 0
    assert not memory.centroids.requires_grad


def test_cca_ratios_have_closed_form():
    source = torch.tensor([[0.], [4.]], dtype=torch.float64)
    target = torch.tensor([[1.], [5.]], dtype=torch.float64, requires_grad=True)
    active = torch.ones(2, dtype=torch.bool)
    # same distances 1+1, different distances 5+3.
    torch.testing.assert_close(losses.centroid_alignment(source, target, active, active),
                               torch.tensor(0.25, dtype=torch.float64))
    labels, w = torch.tensor([0, 1]), torch.ones(2, dtype=torch.float64)
    dfa = losses.discriminative_alignment(source, target, labels, labels, w, w)
    torch.testing.assert_close(dfa, torch.tensor(0.25, dtype=torch.float64))
    dfa.backward()
    assert torch.isfinite(target.grad).all() and target.grad.abs().sum() > 0
    assert losses.centroid_alignment(source, source, active, active) == 0


def test_cca_adversary_reverses_only_feature_gradient():
    d = nn.Linear(2, 1, dtype=torch.float64)
    s = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[3., 4.]], dtype=torch.float64, requires_grad=True)
    loss = losses.domain_adversarial(d, s, t)
    loss.backward()
    actual_t, actual_d = t.grad.clone(), d.weight.grad.clone()
    assert s.grad is None
    d.zero_grad()
    t.grad = None
    raw = (nn.functional.binary_cross_entropy_with_logits(d(s.detach()), torch.zeros(1, 1, dtype=s.dtype))
           + nn.functional.binary_cross_entropy_with_logits(d(t), torch.ones(1, 1, dtype=t.dtype)))
    raw.backward()
    torch.testing.assert_close(actual_t, -t.grad)
    torch.testing.assert_close(actual_d, d.weight.grad)
    assert actual_t.abs().sum() > 0


def test_cca_degenerate_terms_are_finite():
    source = torch.zeros(2, 3, dtype=torch.float64)
    target = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    active = torch.ones(2, dtype=torch.bool)
    a = losses.centroid_alignment(source, target, active, active)
    b = losses.discriminative_alignment(source, target, torch.zeros(2, dtype=torch.long),
                                        torch.zeros(2, dtype=torch.long), torch.ones(2), torch.ones(2))
    (a + b).backward()
    assert torch.isfinite(a + b) and torch.isfinite(target.grad).all()
    absent = torch.zeros(2, dtype=torch.bool)
    assert losses.centroid_alignment(source, target, active, absent) == 0


@pytest.mark.parametrize("source_channels,target_channels", [(29, 14), (14, 29), (29, 29)])
def test_cca_phase_two_freezes_common_predictor(source_channels, target_channels):
    model, source, _, target, valid = fixture(source_channels, target_channels)
    assert isinstance(model.source_trunk, eegnet.EEGNetTrunk)
    assert isinstance(model.head, eegnet.EEGNetHead)
    frozen = two_phase.phase_one_state(model)
    result = training.adapt(model, source, target, valid, params(), schedule(), seed=11)
    assert all(torch.equal(v, two_phase.phase_one_state(model)[k]) for k, v in frozen.items())
    assert result["layer_copy"]["kept_initialization"] == (
        ["depthwise"] if source_channels != target_channels else [])
    assert model.source_trunk.training is False
    assert all(not p.requires_grad for p in model.source_trunk.parameters())
    assert all(not p.requires_grad for p in model.attention_parameters())
    assert all(not p.requires_grad for p in model.head.parameters())
    trainable = list(model.target_trunk.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert any(p.grad.abs().sum() > 0 for p in trainable)
    assert any(not torch.equal(v, model.target_trunk.state_dict()[k])
               for k, v in model.source_trunk.state_dict().items()
               if k != "depthwise.weight")
    assert result["epochs"] == 4


class UnreadableLabel:
    def __int__(self):
        raise AssertionError("target training label read")
    def __eq__(self, other):
        raise AssertionError("target training label compared")


def test_cca_target_training_labels_are_unread():
    a, source, _, target, valid = fixture()
    b = copy.deepcopy(a)
    poisoned = [replace(bag, label=UnreadableLabel()) for bag in target]
    first = training.adapt(a, source, target, valid, params(), schedule(), seed=11)
    second = training.adapt(b, source, poisoned, valid, params(), schedule(), seed=11)
    assert isinstance(poisoned[0].label, UnreadableLabel)
    assert first["history"] == second["history"]
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_cca_warmup_precedes_fixed_prior(monkeypatch):
    model, source, _, target, valid = fixture()
    calls = []
    original = training.estimate_prior
    def measured(p, threshold):
        calls.append((p.clone(), threshold))
        return original(p, threshold)
    monkeypatch.setattr(training, "estimate_prior", measured)
    out = training.adapt(model, source, target, valid, params(), schedule(), seed=11)
    assert [h["calibrated"] for h in out["history"]] == [False, False, False, True]
    assert len(calls) == 1 and calls[0][0].shape == (4, 2)
    torch.testing.assert_close(out["target_prior"], original(*calls[0]))
    generator = torch.Generator().manual_seed(8)
    imbalanced = source + [source[0]] * 8
    queues = two_phase.ClassQueues(imbalanced, 2, generator)
    drawn = queues.draw(10)
    assert [imbalanced[i].label for i in drawn].count(0) == 10
    assert [imbalanced[i].label for i in drawn].count(1) == 10


def test_cca_search_hook_uses_common_checkpoint():
    old_extra, old_trainers = dict(search.PHASE_TWO_EXTRA), dict(search.PHASE_TWO_TRAINERS)
    try:
        cca_lsc.register()
        space = search.phase_two_space("CCA-LSC")
        assert set(space) == set(params())
        assert all(space[k] == search.LOSS_WEIGHT for k in training.DEFAULTS if k != "priorThreshold")
        assert search.PHASE_ONE_GROUP_OF["CCA-LSC"] == "common"
        assert search.PHASE_TWO_TRAINERS["CCA-LSC"] is training.train_phase_two
        from MIL_CREDA_EEG.assignment import Transfer
        model, s, sv, t, tv = fixture()
        data = search.SearchData(Transfer("device", "L6", "Robots"), "hash", {
            0: {"source_train": s, "source_valid": sv, "target_train": t, "target_valid": tv}})
        record = (two_phase.phase_one_state(model), {
            "phase": 1, "name": "common", "transfer": data.transfer.name, "rotation": 0,
            "seed": search.PHASE_ONE_SEED, "partition_sha256": "hash", "hyper": {}})
        trained = search.train_phase_two("CCA-LSC", data, 0, 11, record, params(), schedule())
        assert isinstance(trained["model"], mil_wrapper.CommonMILModel)
        assert trained["result"]["checkpoint"]["hyper"]["calibrationHm"] == 1.5
        wrong = (record[0], {**record[1], "rotation": 1})
        with pytest.raises(search.SearchError, match="checkpoint"):
            training.train_phase_two(data, 0, 11, wrong, params(), schedule())
    finally:
        search.PHASE_TWO_EXTRA.clear(); search.PHASE_TWO_EXTRA.update(old_extra)
        search.PHASE_TWO_TRAINERS.clear(); search.PHASE_TWO_TRAINERS.update(old_trainers)


def test_cca_keeps_the_last_epoch_and_records_validation_per_epoch(monkeypatch):
    model, source, _, target, valid = fixture()
    # Actual epoch weights snapshotted at the real validation seam; epoch 1 scores best,
    # and v50 still keeps the last epoch (no early stopping, no restore).
    snapshots = []
    real_eval = two_phase.evaluate
    def evaluate(m, bags, domain, device):
        out = real_eval(m, bags, domain, device)
        if bags is valid:
            snapshots.append(copy.deepcopy(m.target_trunk.state_dict()))
        return out
    scores = iter([0.1, 0.9, 0.3, 0.2, 0.9])
    monkeypatch.setattr(two_phase, "evaluate", evaluate)
    monkeypatch.setattr(two_phase, "balanced_accuracy", lambda *_: next(scores))
    out = training.adapt(model, source, target, valid, params(), schedule(), seed=11)
    last = schedule()["epochs2"] - 1
    assert out["best_epoch"] == last == out["checkpoint"]["epoch"]
    assert all(torch.equal(v, model.target_trunk.state_dict()[k])
               for k, v in snapshots[last].items())
    assert any(not torch.equal(v, snapshots[last][k]) for k, v in snapshots[1].items())
    assert out["checkpoint"]["label_free_score"] == out["history"][last]["label_free_score"]
    assert [v["epoch"] for v in out["validation"]] == list(range(schedule()["epochs2"]))


@pytest.mark.parametrize("change", [{"lr2": -1}, {"wd2": -1}, {"centroidWeight": -1},
                                    {"calibrationHm": 0}, {"priorThreshold": 2},
                                    {"adversarialWeight": float("nan")}])
def test_cca_invalid_inputs_fail_closed(change):
    with pytest.raises(ValueError):
        training.check_params({**params(), **change})


def test_cca_short_schedule_and_leakage_refuse():
    model, s, _, t, v = fixture()
    with pytest.raises(ValueError, match="calibrated epoch"):
        training.adapt(model, s, t, v, params(), {**schedule(), "epochs2": 3}, seed=11)
    # a refit runs exactly the selected E2, which may end inside the warm-up
    refit = training.adapt(fixture()[0], s, t, [], params(), {**schedule(), "epochs2": 2},
                           seed=11)
    assert refit["epochs"] == 2 and refit["validation"] == [] and refit["target_prior"] is None
    with pytest.raises(ValueError, match="disjoint"):
        training.adapt(model, s, t, [replace(v[0], subject=t[0].subject)], params(), schedule(), seed=11)


def test_cca_objective_weights_are_effective():
    model, source, _, target, _ = fixture()
    two_phase.bridge(model)
    state = training.AlignmentState(256, 2)
    weights = torch.ones(2, dtype=torch.float64)
    torch.manual_seed(7)
    first = training.adaptation_step(model, state, source, target, weights, params(), torch.device("cpu"))
    # All three components contribute via the declared coefficients.
    expected = (first["source_ce"] + 3 * first["centroid"]
                + 0.6 * first["discriminative"] + first["adversarial"])
    torch.testing.assert_close(first["loss"], expected)
    first["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in state.discriminator.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.target_trunk.parameters())


def test_cca_package_is_a_controlled_variant():
    assert cca_lsc.adapt is training.adapt
    assert cca_lsc.__provenance__["sections"] == ["7.1", "7.4"]
    assert training.WARMUP_EPOCHS == 3


def test_cca_synthetic_smoke_executes(capsys):
    result = run()
    assert result["epochs"] == 5
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in result["history"])
    assert len({h["loss"] for h in result["history"]}) > 1
    assert "synthetic CPU only" in capsys.readouterr().out
