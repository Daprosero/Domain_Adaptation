"""Experiments plan v40, Secs. 4.5, 7.3, 7.7: MIL-CREDA's two phases on EEG bags."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import assignment, bag_data, eegnet, two_phase as tp
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel, SigmaAlreadyFixed

CPU = torch.device("cpu")
HYPER = {"lr1": 1e-2, "wd1": 1e-4, "attentionGamma": 0.5, "attentionTemperature": 1.0,
         "sigmaMultiplier": 1.0, "lr2": 1e-2, "wd2": 1e-4, "tauLocal": 1.0,
         "epochs1": 2, "patience1": 2, "epochs2": 2, "bagsPerStep": 4}


def _bags(n: int, channels: int, seed: int, prefix: str) -> list[tp.BagTensor]:
    g = torch.Generator().manual_seed(seed)
    return [tp.BagTensor(f"{prefix}{i}", "c", i % 2,
                         torch.randn(3 + i % 2, channels, eegnet.SAMPLES, generator=g))
            for i in range(n)]


def _setup(source_channels: int = 3, target_channels: int = 2):
    torch.manual_seed(0)
    model = MILCREDAModel(source_channels, target_channels, attention_gamma=0.5,
                          attention_temperature=1.0)
    data = {"source_train": _bags(6, source_channels, 1, "s"),
            "source_valid": _bags(4, source_channels, 2, "v"),
            "target_train": _bags(6, target_channels, 3, "t"),
            "target_valid": _bags(4, target_channels, 4, "w")}
    return model, data, torch.Generator().manual_seed(0)


def _phase_one(model, data, generator, hyper=HYPER):
    return tp.train_source_phase(model, data["source_train"], data["source_valid"], hyper,
                                 generator, CPU)


def _through_bridge(hyper=HYPER, source_channels=3, target_channels=2):
    model, data, generator = _setup(source_channels, target_channels)
    phase1 = _phase_one(model, data, generator, hyper)
    joint = tp.bridge(model)
    return model, data, generator, phase1, joint


def _phase_two(model, data, generator, phase1, joint, hyper=HYPER):
    return tp.adapt_target_phase(model, phase1, joint["frozen_state"], data["source_train"],
                                 data["target_train"], data["target_valid"], hyper,
                                 generator, CPU)


def test_the_hyperparameters_are_exactly_the_declared_ones():
    assert tp.check_hyper(HYPER) == HYPER
    for extra in ("lambdaGlobal", "lambdaLocal", "ramp", "sigma2", "kernelSigma"):
        with pytest.raises(ValueError, match="not declared"):
            tp.check_hyper({**HYPER, extra: 1.0})
    with pytest.raises(ValueError, match="missing"):
        tp.check_hyper({k: v for k, v in HYPER.items() if k != "tauLocal"})
    with pytest.raises(ValueError):
        tp.check_hyper({**HYPER, "bagsPerStep": 1})


def _scripted(monkeypatch, name, values, model=None, states=None):
    calls = iter(values)

    def fake(*args, **kwargs):
        if states is not None:
            states.append(copy.deepcopy(model.state_dict()))
        value = next(calls)
        return value(model) if callable(value) else value

    monkeypatch.setattr(tp, name, fake)


def test_phase_one_restores_the_best_epochs_weights_and_proves_it(monkeypatch):
    model, data, generator = _setup()
    states = []
    restored_ok = (lambda m: 0.9 if all(torch.equal(m.state_dict()[k], states[1][k])
                                        for k in tp.phase_one_state(m)) else 0.0)
    _scripted(monkeypatch, "balanced_accuracy", [0.5, 0.9, 0.6, 0.6, restored_ok],
              model, states)
    out = _phase_one(model, data, generator, {**HYPER, "epochs1": 6, "patience1": 2})
    assert out["best_epoch"] == 1 and out["epochs"] == 4
    assert out["restored_valid_balanced_accuracy"] == out["best_valid_balanced_accuracy"] == 0.9
    now = model.state_dict()
    assert all(torch.equal(now[k], states[1][k]) for k in tp.phase_one_state(model))
    assert not all(torch.equal(now[k], states[3][k]) for k in tp.phase_one_state(model))
    assert tp._states_equal(out["checkpoint"]["state"], tp.phase_one_state(model))


def test_a_restore_that_does_not_reproduce_the_best_accuracy_refuses(monkeypatch):
    model, data, generator = _setup()
    _scripted(monkeypatch, "balanced_accuracy", [0.5, 0.9, 0.8])
    with pytest.raises(tp.RestoreFailed):
        _phase_one(model, data, generator, {**HYPER, "epochs1": 2})


def test_sigma_is_fixed_during_phase_one_and_stored_with_the_checkpoint():
    model, data, generator = _setup()
    out = _phase_one(model, data, generator, {**HYPER, "epochs1": 3, "patience1": 3})
    sigma = out["checkpoint"]["sigma"]
    assert sigma == model.sigma == out["sigma"]["sigma"] > 0
    assert [r["sigma"] for r in out["history"]] == [sigma] * out["epochs"]
    with pytest.raises(SigmaAlreadyFixed):
        model.fix_sigma(sigma * 1.01)


def test_the_phase_one_sigma_is_the_multiplier_times_the_within_bag_median():
    model, data, _ = _setup()
    before = {k: v.clone() for k, v in model.source_trunk.state_dict().items()}
    reference = copy.deepcopy(model.source_trunk)
    reference.train()
    for m in reference.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    with torch.no_grad():
        pooled = []
        for b in data["source_train"]:
            d = torch.cdist(*(2 * [reference(b.windows).double()]))
            pooled.append(d[~torch.eye(d.shape[0], dtype=torch.bool)])
    median = float(torch.cat(pooled).median())
    out = tp.phase_one_sigma(model, data["source_train"], 1.5, CPU)
    assert out["withinBagMedian"] == pytest.approx(median, rel=1e-12)
    assert model.sigma == out["sigma"] == pytest.approx(1.5 * median, rel=1e-12)
    after = model.source_trunk.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)      # no trace left


@pytest.mark.parametrize("target_channels", [3, 2])
def test_the_bridge_copies_compatible_layers_and_freezes_the_source_side(target_channels):
    model, _, _, _, joint = _through_bridge(target_channels=target_channels)
    kept = joint["layer_copy"]["kept_initialization"]
    assert kept == ([] if target_channels == 3 else ["depthwise"])
    frozen = [*model.source_trunk.parameters(), *model.attention_parameters(),
              *model.head.parameters()]
    assert all(not p.requires_grad for p in frozen)
    assert all(p.requires_grad for p in model.target_trunk.parameters())
    assert not any(m.training for m in model.source_trunk.modules())
    assert model.target_trunk.training
    src, tgt = model.source_trunk.state_dict(), model.target_trunk.state_dict()
    for name in tgt:
        if target_channels == 3 or not name.startswith("depthwise."):
            assert torch.equal(src[name], tgt[name]), name


def test_phase_two_updates_only_the_target_trunk():
    model, data, generator, phase1, joint = _through_bridge()
    start = {k: v.clone() for k, v in model.target_trunk.state_dict().items()}
    _phase_two(model, data, generator, phase1, joint)
    assert tp._states_equal(tp.phase_one_state(model), joint["frozen_state"])
    moved = model.target_trunk.state_dict()
    assert any(not torch.equal(start[k], moved[k]) for k in start)
    tampered = {k: v.clone() for k, v in joint["frozen_state"].items()}
    tampered["head.linear.bias"] += 1.0
    model2, data2, generator2, phase1b, _ = _through_bridge()
    with pytest.raises(tp.FrozenPartsMoved):
        _phase_two(model2, data2, generator2, phase1b, {"frozen_state": tampered})


def test_sigma_in_phase_two_is_exactly_the_phase_one_value():
    model, data, generator, phase1, joint = _through_bridge()
    out = _phase_two(model, data, generator, phase1, joint)
    sigma1 = phase1["checkpoint"]["sigma"]
    assert model.sigma == sigma1 and out["checkpoint"]["sigma"] == sigma1
    assert [r["sigma"] for r in out["history"]] == [sigma1] * out["epochs"]
    other = {**phase1, "checkpoint": {**phase1["checkpoint"], "sigma": sigma1 * 2}}
    with pytest.raises(SigmaAlreadyFixed):
        _phase_two(model, data, generator, other, joint)


def test_the_adaptation_coefficients_are_one_with_no_ramp():
    model, data, generator, phase1, joint = _through_bridge()
    out = _phase_two(model, data, generator, phase1, joint, {**HYPER, "epochs2": 3})
    assert tp.LAMBDA_GLOBAL == tp.LAMBDA_LOCAL == 1.0
    assert [(r["lambda_global"], r["lambda_local"]) for r in out["history"]] == [(1.0, 1.0)] * 3
    step = tp.phase_two_step(model, data["source_train"][:4], data["target_train"][:4],
                             1.0, CPU)
    assert torch.equal(step["objective"], step["global"] + step["local"])


def test_the_source_term_does_not_change_the_phase_two_gradient():
    model, data, _, _, _ = _through_bridge()
    step = tp.phase_two_step(model, data["source_train"][:4], data["target_train"][:4],
                             1.0, CPU)
    params = list(model.target_trunk.parameters())
    without = torch.autograd.grad(step["objective"], params, retain_graph=True)
    with_ce = torch.autograd.grad(step["objective"] + step["source_ce"], params)
    assert float(step["source_ce"]) > 0.0
    assert all(torch.equal(a, b) for a, b in zip(without, with_ce))


def test_the_best_phase_two_epoch_is_chosen_by_target_validation_balanced_accuracy(
        monkeypatch):
    model, data, generator, phase1, joint = _through_bridge()
    states = []
    original = tp.balanced_accuracy

    def scripted(labels, predicted, _it=iter([0.4, 0.8, 0.6])):
        states.append({k: v.clone() for k, v in model.target_trunk.state_dict().items()})
        return next(_it)

    monkeypatch.setattr(tp, "balanced_accuracy", scripted)
    monkeypatch.setattr(tp, "label_free_score", lambda G, _it=iter([0.5, 0.1, 0.9]): next(_it))
    out = _phase_two(model, data, generator, phase1, joint, {**HYPER, "epochs2": 3})
    monkeypatch.setattr(tp, "balanced_accuracy", original)
    assert out["epochs"] == 3                         # the cap runs whole
    assert out["best_epoch"] == 1 and out["checkpoint"]["valid_balanced_accuracy"] == 0.8
    assert out["selection"]["best_label_free_epoch"] == 2
    assert out["checkpoint"]["label_free_score"] == 0.1
    assert tp._states_equal(model.target_trunk.state_dict(), states[1])
    assert not tp._states_equal(model.target_trunk.state_dict(), states[2])


def test_the_label_free_score_is_entropy_of_the_mean_minus_mean_entropy():
    confident = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    assert tp.label_free_score(confident) == pytest.approx(math.log(2))
    assert tp.label_free_score(torch.full((4, 2), 0.5)) == pytest.approx(0.0, abs=1e-12)
    one_class = torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float64)
    assert tp.label_free_score(one_class) == pytest.approx(0.0, abs=1e-12)
    G = torch.tensor([[0.9, 0.1], [0.3, 0.7]], dtype=torch.float64)
    h = lambda p: -(p * p.log()).sum(-1)
    assert tp.label_free_score(G) == pytest.approx(float(h(G.mean(0)) - h(G).mean()))


def test_history_records_carry_the_dashboard_keys(tmp_path):
    model, data, generator, phase1, joint = _through_bridge()
    phase2 = _phase_two(model, data, generator, phase1, joint)
    records = phase1["history"] + phase2["history"]
    assert {r["model"] for r in records} == {"MIL-CREDA:phase1", "MIL-CREDA:phase2"}
    for r in records:
        assert {"epoch", "seed", "fold", "model"} <= set(r)
        numeric = {k for k, v in r.items() if isinstance(v, float)}
        assert {"loss", "val_balanced_accuracy"} <= numeric
    for r in phase2["history"]:
        assert {"global", "local", "source_ce", "label_free_score", "global_bound_low",
                "global_bound_high", "local_bound_low", "local_bound_high"} <= set(r)
        assert r["terms_within_bounds"] == 1.0
    path = tp.write_history(records, tmp_path / "Results" / "history_milcreda.json")
    assert json.loads(path.read_text()) == records
    with pytest.raises(ValueError):
        tp.write_history(records, tmp_path / "Results" / "curves.json")


def _bagset(root) -> bag_data.BagSet:
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
    rotations = {str(r): {"test_fold": r, "validation_fold": (r + 1) % 4,
                          "train_folds": [(r + 2) % 4, (r + 3) % 4]} for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256="x" * 64, rotations=rotations)


def test_test_predictions_are_only_for_target_test_fold_subjects(tmp_path):
    bags = _bagset(tmp_path)
    transfer = assignment.Transfer("level", "L1", "L11")
    hyper = {**HYPER, "epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 2}
    out = tp.run_transfer_rotation(transfer, 1, 3, hyper, "cpu", bags=bags)
    view = assignment.assign_from_bags(transfer, bags).rotation(1)
    subjects = [p["subject"] for p in out["predictions"]]
    assert subjects and set(subjects) == view["target_test"]
    assert len(subjects) == len(set(subjects))
    assert not set(subjects) & (view["source_train"] | view["source_validation"]
                                | view["source_test"])
    labels = bags.labels()
    for p in out["predictions"]:
        assert p["direction"] == "L1->L11" and 0.0 <= p["probability"] <= 1.0
        assert p["label"] == ("no", "yes").index(labels[p["subject"]])
    assert out["sigma"] == out["checkpoints"]["phase1"]["sigma"]
    assert out["checkpoints"]["phase2"]["sigma"] == out["sigma"]
    assert {r["fold"] for r in out["history"]} == {1} and {r["seed"] for r in out["history"]} == {3}
