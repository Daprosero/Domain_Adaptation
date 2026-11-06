"""UDA-DDA paper equations and controlled BALLADEER adaptation invariants (CPU)."""
import copy
import importlib
import math
from dataclasses import replace

import pytest
import torch

from MIL_CREDA_EEG import assignment, eegnet, mil_wrapper, model_store, search, two_phase
from MIL_CREDA_EEG.comparators import uda_dda
from MIL_CREDA_EEG.comparators.uda_dda import losses, training
from MIL_CREDA_EEG.comparators.uda_dda.smoke import fixture, run


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def registries():
    """Save and restore every global registry a comparator may touch."""
    saved = (dict(model_store.BUILDERS), dict(search.PHASE_ONE_TRAINERS),
             dict(search.PHASE_TWO_TRAINERS), dict(search.PHASE_TWO_EXTRA))
    yield
    for target, old in zip((model_store.BUILDERS, search.PHASE_ONE_TRAINERS,
                            search.PHASE_TWO_TRAINERS, search.PHASE_TWO_EXTRA), saved):
        target.clear()
        target.update(old)


def params(**change):
    return {**training.DEFAULTS, "lr2": 0.001, "wd2": 0., **change}


def schedule(epochs=4):
    return {"epochs1": 1, "epochs2": epochs, "bagsPerStep": 4}


def f64(values):
    return torch.tensor(values, dtype=torch.float64)


# ------------------------------------------------------------------ equations


def test_uda_mmd_is_eq5_and_the_eq4_trace_form():
    source, target = f64([[0.], [1.]]), f64([[3.], [5.], [6.]])
    sigma = 2.0
    k = lambda a, b: math.exp(-(a - b) ** 2 / sigma)
    s, t = [0., 1.], [3., 5., 6.]
    expected = (sum(k(a, b) for a in s for b in s) / 4 + sum(k(a, b) for a in t for b in t) / 9
                - 2 * sum(k(a, b) for a in s for b in t) / 6)
    value = losses.mmd2(source, target, sigma)
    assert value.item() == pytest.approx(expected, rel=1e-12)
    # Eq. 3/4 trace form with the corrected M: diag blocks 1/n_s^2, 1/n_t^2, off -1/(n_s n_t).
    pooled = torch.cat((source, target))
    trace = torch.trace(losses.gaussian_kernel(pooled, pooled, sigma) @ losses.mmd_matrix(2, 3))
    torch.testing.assert_close(value, trace)
    m = losses.mmd_matrix(2, 3)
    assert m[0, 0] == 1 / 4 and m[3, 3] == 1 / 9 and m[0, 4] == -1 / 6
    assert m.sum().abs() < 1e-12      # mean embeddings difference: M 1 = 0 in total mass
    assert losses.mmd2(source, source.clone(), sigma).abs() < 1e-12


def test_uda_cmmd_is_eq6_per_class():
    source = f64([[0.], [1.], [4.]])
    target = f64([[0.5], [3.], [9.]]).requires_grad_()
    ys, yt = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])
    sigma = 3.0
    value = losses.cmmd2(source, target, ys, yt, 2, sigma)
    expected = (losses.mmd2(source[:2], target[:1], sigma)
                + losses.mmd2(source[2:], target[1:], sigma)) / 2
    torch.testing.assert_close(value, expected)
    value.backward()
    assert torch.isfinite(target.grad).all() and target.grad.abs().sum() > 0
    # A class absent from the selected target contributes zero, still divided by C.
    only0 = losses.cmmd2(source, target[:1].detach(), ys, yt[:1], 2, sigma)
    torch.testing.assert_close(only0, losses.mmd2(source[:2], target[:1].detach(), sigma) / 2)


def test_uda_kernel_is_gaussian_with_median_bandwidth():
    a, b = f64([[0., 0.]]), f64([[3., 4.]])
    assert losses.gaussian_kernel(a, b, 5.).item() == pytest.approx(math.exp(-25 / 5))
    source, target = f64([[0.], [1.]]), f64([[3.]])
    # squared distances 1, 9, 4 -> median 4.
    assert losses.median_bandwidth(source, target, 0.5) == 2.0
    assert losses.median_bandwidth(f64([[1.]]), f64([[1.]]), 0.75) == 0.75
    with pytest.raises(ValueError):
        losses.gaussian_kernel(a, b, 0.)


def test_uda_confidence_schedule_is_eq7_rescaled():
    at = lambda e, n=100: losses.confidence_threshold(e, n, 0.5, 0.75)
    # The paper's own 100-epoch run reproduces its stage boundaries exactly.
    assert [at(e) for e in (0, 9, 10, 39, 40, 85, 86, 99)] == [0, 0, .5, .5, .75, .75, 1, 1]
    # Ten epochs: paper epochs 0,10,...,90.
    assert [losses.stage(e, 10) for e in range(10)] == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    p = f64([[0.9, 0.1], [0.6, 0.4], [0.75, 0.25]])
    labels, keep = losses.confident_pseudo_labels(p, 0.75)
    assert labels.tolist() == [0, 0, 0] and keep.tolist() == [True, False, False]  # strict >
    assert losses.confident_pseudo_labels(p, 1.0)[1].sum() == 0
    assert losses.confident_pseudo_labels(p, 0.0)[1].all()


def test_uda_alpha_decays_from_tau_h_to_tau_l():
    values = [losses.mmd_weight_schedule(e, 100) for e in range(100)]
    assert values[:10] == [1.0] * 10 and values[40:] == [0.01] * 60
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[25] == pytest.approx(0.1)  # geometric midpoint of 1 -> 0.01
    assert losses.mmd_weight_schedule(0, 1) == 1.0


def test_uda_confidence_thresholds_are_sorted_so_the_schedule_never_inverts():
    assert training.ordered_thresholds(params(confidenceA=0.9, confidenceB=0.6)) == (0.6, 0.9)
    assert training.ordered_thresholds(params(confidenceA=0.6, confidenceB=0.9)) == (0.6, 0.9)
    runs = []
    for a, b in ((0.9, 0.6), (0.6, 0.9)):
        model, source, _, target, valid = fixture()
        out = training.adapt(model, source, target, valid,
                             params(confidenceA=a, confidenceB=b), schedule(10), seed=13)
        by_stage = {h["stage"]: h["threshold"] for h in out["history"]}
        assert by_stage == {0: 0.0, 1: 0.6, 2: 0.9, 3: 1.0}
        assert out["checkpoint"]["thresholds"] == {"middle": 0.6, "late": 0.9}
        runs.append([h["threshold"] for h in out["history"]])
    assert runs[0] == runs[1]
    assert all(x <= y for x, y in zip(runs[0], runs[0][1:]))


def test_uda_degenerate_terms_are_finite():
    source = torch.zeros(2, 3, dtype=torch.float64)
    target = torch.zeros(0, 3, dtype=torch.float64, requires_grad=True)
    empty = losses.cmmd2(source, target, torch.tensor([0, 1]), torch.zeros(0, dtype=torch.long), 2, 1.)
    assert empty.item() == 0 and empty.requires_grad
    empty.backward()
    same = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    value = losses.mmd2(source, same, losses.median_bandwidth(source, same.detach(), 1.))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(same.grad).all()


# ------------------------------------------------------------------ training


@pytest.mark.parametrize("source_channels,target_channels", [(29, 14), (14, 29), (29, 29)])
def test_uda_phase_two_freezes_common_predictor(source_channels, target_channels):
    model, source, _, target, valid = fixture(source_channels, target_channels)
    assert isinstance(model.source_trunk, eegnet.EEGNetTrunk)
    frozen = two_phase.phase_one_state(model)
    result = training.adapt(model, source, target, valid, params(), schedule(), seed=13)
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
    assert result["epochs"] == 4


class UnreadableLabel:
    def __int__(self):
        raise AssertionError("target training label read")
    def __index__(self):
        raise AssertionError("target training label read")
    def __eq__(self, other):
        raise AssertionError("target training label compared")
    __hash__ = object.__hash__


def test_uda_target_training_labels_are_unread():
    a, source, _, target, valid = fixture()
    b = copy.deepcopy(a)
    poisoned = [replace(bag, label=UnreadableLabel()) for bag in target]
    first = training.adapt(a, source, target, valid, params(), schedule(), seed=13)
    second = training.adapt(b, source, poisoned, valid, params(), schedule(), seed=13)
    assert isinstance(poisoned[0].label, UnreadableLabel)
    assert first["history"] == second["history"]
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_uda_bandwidth_is_the_detached_step_median(monkeypatch):
    model, source, _, target, valid = fixture()
    calls = []
    original = losses.median_bandwidth
    def measured(s, t, multiplier):
        calls.append((s.requires_grad, t.requires_grad, multiplier))
        return original(s, t, multiplier)
    monkeypatch.setattr(losses, "median_bandwidth", measured)
    out = training.adapt(model, source, target, valid, params(kernelMultiplier=0.75),
                         schedule(), seed=13)
    # one call per optimisation step (4 target bags, 4 per step -> 1 step/epoch)
    assert len(calls) == out["epochs"] == 4
    assert all(c == (False, False, 0.75) for c in calls)
    assert all(h["sigma"] > 0 for h in out["history"])


def test_uda_search_hook_uses_common_checkpoint(registries):
    assert "UDA-DDA" not in search.PHASE_TWO_TRAINERS
    uda_dda.register()
    space = search.phase_two_space("UDA-DDA")
    assert set(space) == set(params())
    assert space["mmdWeight"] == space["cmmdWeight"] == search.LOSS_WEIGHT
    assert space["kernelMultiplier"] == search.KERNEL_MULTIPLIER
    assert space["confidenceA"] == space["confidenceB"] == search.CONFIDENCE_THRESHOLD
    assert search.PHASE_ONE_GROUP_OF["UDA-DDA"] == "common"
    assert search.PHASE_TWO_TRAINERS["UDA-DDA"] is training.train_phase_two
    model, s, sv, t, tv = fixture()
    data = search.SearchData(assignment.Transfer("device", "L6", "Robots"), "hash", {
        0: {"source_train": s, "source_valid": sv, "target_train": t, "target_valid": tv}})
    record = (two_phase.phase_one_state(model), {
        "phase": 1, "name": "common", "transfer": data.transfer.name, "rotation": 0,
        "seed": search.PHASE_ONE_SEED, "partition_sha256": "hash", "hyper": {}})
    trained = search.train_phase_two("UDA-DDA", data, 0, 13, record, params(), schedule())
    assert isinstance(trained["model"], mil_wrapper.CommonMILModel)
    assert trained["result"]["checkpoint"]["hyper"]["confidenceB"] == 0.75
    assert trained["result"]["checkpoint"]["thresholds"] == {"middle": 0.5, "late": 0.75}
    for wrong in ({"rotation": 1}, {"name": "AdaptMIL"}, {"partition_sha256": "other"}):
        with pytest.raises(search.SearchError, match="checkpoint"):
            training.train_phase_two(data, 0, 13, (record[0], {**record[1], **wrong}),
                                     params(), schedule())


def test_uda_keeps_the_last_epoch_and_records_validation_per_epoch(monkeypatch):
    model, source, _, target, valid = fixture()
    snapshots = []
    real_eval = two_phase.evaluate
    def evaluate(m, bags, domain, device):
        out = real_eval(m, bags, domain, device)
        if bags is valid:
            snapshots.append(copy.deepcopy(m.target_trunk.state_dict()))
        return out
    scores = iter([0.1, 0.9, 0.9, 0.2])
    monkeypatch.setattr(two_phase, "evaluate", evaluate)
    monkeypatch.setattr(two_phase, "balanced_accuracy", lambda *_: next(scores))
    out = training.adapt(model, source, target, valid, params(), schedule(), seed=13)
    last = schedule()["epochs2"] - 1
    # v50: no early stopping, no restore: the last epoch is kept whatever scored best
    assert out["best_epoch"] == last == out["checkpoint"]["epoch"]
    assert all(torch.equal(v, model.target_trunk.state_dict()[k])
               for k, v in snapshots[last].items())
    assert any(not torch.equal(v, snapshots[last][k]) for k, v in snapshots[1].items())
    assert out["checkpoint"]["label_free_score"] == out["history"][last]["label_free_score"]
    assert [v["epoch"] for v in out["validation"]] == list(range(last + 1))


@pytest.mark.parametrize("change", [{"lr2": -1}, {"wd2": -1}, {"mmdWeight": -1},
                                    {"cmmdWeight": float("nan")}, {"kernelMultiplier": 0},
                                    {"confidenceA": 1.5}, {"confidenceB": -0.1}])
def test_uda_invalid_inputs_fail_closed(change):
    with pytest.raises(ValueError):
        training.check_params(params(**change))


def test_uda_missing_key_and_leakage_refuse():
    with pytest.raises(ValueError, match="exactly"):
        training.check_params({k: v for k, v in params().items() if k != "cmmdWeight"})
    model, s, _, t, v = fixture()
    with pytest.raises(ValueError, match="disjoint"):
        training.adapt(model, s, t, [replace(v[0], subject=t[0].subject)], params(), schedule(), seed=13)
    with pytest.raises(ValueError, match="common MIL"):
        training.adapt(object(), s, t, v, params(), schedule(), seed=13)


def test_uda_objective_weights_are_effective():
    model, source, _, target, _ = fixture()
    two_phase.bridge(model)
    torch.manual_seed(7)
    p = params(mmdWeight=2.0, cmmdWeight=3.0)
    terms = training.adaptation_step(model, source, target, p, 0.5, 0.0, torch.device("cpu"))
    expected = terms["source_ce"] + 2.0 * 0.5 * terms["mmd"] + 3.0 * terms["cmmd"]
    torch.testing.assert_close(terms["loss"], expected)
    assert terms["selected"] == 1.0 and terms["cmmd"] > 0
    assert not terms["source_ce"].requires_grad
    terms["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.target_trunk.parameters())


def test_uda_cmmd_has_no_source_loss_gate(monkeypatch):
    # Plan v42 Sec. 7.7: the Eq. 9 gate is gone, CMMD weighs cmmdWeight from phase-2 start.
    assert not hasattr(losses, "cmmd_weight") and not hasattr(losses, "RHO")
    assert "beta" not in training.TERMS
    model, source, _, target, valid = fixture()
    two_phase.bridge(model)
    # A source loss far above the paper's rho_1 = 0.15 would have closed the gate.
    monkeypatch.setattr(losses, "source_cross_entropy",
                        lambda p, y: torch.tensor(5.0, dtype=torch.float64))
    on = training.adaptation_step(model, source, target, params(cmmdWeight=3.0), 0.5, 0.0,
                                  torch.device("cpu"))
    assert "beta" not in on and on["cmmd"] > 0
    torch.testing.assert_close(on["loss"], on["source_ce"] + 0.5 * on["mmd"] + 3.0 * on["cmmd"])
    monkeypatch.undo()
    model, source, _, target, valid = fixture()
    out = training.adapt(model, source, target, valid, params(), schedule(), seed=13)
    first = out["history"][0]
    assert first["stage"] == 0 and first["cmmd"] > 0 and "beta" not in first


def test_uda_package_is_a_controlled_variant_and_import_registers_nothing(registries):
    before = (dict(search.PHASE_TWO_TRAINERS), dict(search.PHASE_TWO_EXTRA),
              dict(search.PHASE_ONE_TRAINERS), dict(model_store.BUILDERS))
    importlib.reload(uda_dda)
    importlib.reload(training)
    after = (dict(search.PHASE_TWO_TRAINERS), dict(search.PHASE_TWO_EXTRA),
             dict(search.PHASE_ONE_TRAINERS), dict(model_store.BUILDERS))
    assert before == after and "UDA-DDA" not in search.PHASE_TWO_TRAINERS
    assert uda_dda.__provenance__["sections"] == ["7.1", "7.4"]
    # The package, its trainer, its smoke and its losses answer to v55 (Secs. 4.5, 7.7:
    # fixed epochs, no patience).
    for module, revision in ((uda_dda, "v55"), (training, "v55"), (losses, "v55"),
                             (importlib.import_module(
                                 "MIL_CREDA_EEG.comparators.uda_dda.smoke"), "v55")):
        assert module.__provenance__["revision"] == (
            f"experiments-plan-experimental-mil-creda-sobre-balladeer-{revision}.md")


def test_uda_synthetic_smoke_executes(capsys):
    result = run()
    assert result["epochs"] == 10
    assert [h["stage"] for h in result["history"]] == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    assert all(math.isfinite(h["loss"]) for h in result["history"])
    assert len({round(h["mmd"], 9) for h in result["history"]}) > 1
    assert "synthetic CPU only" in capsys.readouterr().out
