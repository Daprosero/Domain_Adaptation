"""CPU invariants for Xu et al. ESWA CLUDA, controlled bag-level variant."""
import copy
import importlib
import math

import pytest
import torch
from optuna.distributions import FloatDistribution

from MIL_CREDA_EEG import assignment, model_store, search, two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
from MIL_CREDA_EEG.comparators.cluda.losses import cross_domain_contrastive
from MIL_CREDA_EEG.comparators.cluda.model import CLUDAModel
from MIL_CREDA_EEG.comparators.cluda.training import (
    DEFAULT_PARAMS, train_phase_two, register, load_phase_two, source_digest,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture_data(source_channels=29, target_channels=14, seed=19):
    torch.manual_seed(seed)
    def bags(prefix, channels):
        return [two_phase.BagTensor(f"{prefix}{i}", "synthetic", i % 2,
                                    torch.randn(2+i % 2, channels, 512)) for i in range(4)]
    d = {"source_train": bags("s", source_channels),
         "source_valid": bags("sv", source_channels),
         "target_train": bags("t", target_channels),
         "target_valid": bags("v", target_channels)}
    data = search.SearchData(assignment.Transfer("device", "L6", "Robots"),
                             "synthetic-partition", {0: d})
    model = CommonMILModel(source_channels, target_channels)
    record = (two_phase.phase_one_state(model), {
        "phase": 1, "name": "common", "transfer": data.transfer.name,
        "rotation": 0, "seed": search.PHASE_ONE_SEED,
        "partition_sha256": data.partition_sha256, "channels": data.channels(),
        "hyper": {"lr1": .001, "wd1": 0.},
    })
    return data, record


def schedule(epochs=2):
    return dict(epochs1=1, epochs2=epochs, bagsPerStep=4)


def official_reference(feats, labels, temperature):
    # Independent literal scalar sums: official models.py:264-290, diagonal excluded.
    normalized = feats / feats.norm(dim=1, keepdim=True)
    rows = []
    for anchor in range(len(feats)):
        denominator = sum(torch.exp((normalized[anchor]*normalized[j]).sum()/temperature)
                          for j in range(len(feats)) if j != anchor)
        positives = [torch.exp((normalized[anchor]*normalized[j]).sum()/temperature)
                     for j in range(len(feats)) if j != anchor and labels[j] == labels[anchor]]
        numerator = sum(positives, feats.new_zeros(())).clamp_min(1e-6)
        if positives:
            rows.append(-torch.log(numerator/denominator))
    return torch.stack(rows).mean() if rows else feats.sum()*0


def test_cluda_official_symmetric_loss_includes_target_target_pairs():
    source = torch.tensor([[1., 0.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0., 1.], [1., 1.], [-1., 1.]], dtype=torch.float64, requires_grad=True)
    labels, pseudo = torch.tensor([1]), torch.tensor([0, 0, 1])
    value, missing = cross_domain_contrastive(source, target, labels, pseudo, .7)
    expected = official_reference(torch.cat((source, target)), torch.cat((labels, pseudo)), .7)
    torch.testing.assert_close(value, expected)
    assert missing == 0  # the two target class-0 bags are positives for one another
    value.backward()
    for tensor in (source, target):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    scaled, _ = cross_domain_contrastive(source*7, target*3, labels, pseudo, .7)
    torch.testing.assert_close(scaled, value)


def test_cluda_excludes_no_positive_anchors_and_keeps_official_positive_clamp():
    from MIL_CREDA_EEG.comparators.cluda.losses import supervised_contrastive
    z = torch.tensor([[1., 0.], [.5, 1.]], dtype=torch.float64, requires_grad=True)
    value, missing = supervised_contrastive(z, torch.tensor([0, 1]), .07)
    assert missing == 2 and value.item() == 0
    value.backward()
    assert torch.count_nonzero(z.grad) == 0
    # Every anchor has positives; the numerator floor is inactive in this batch.
    positive_batch = torch.tensor([[1., 0.], [1., .1], [0., 1.], [.1, 1.]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1])
    ordinary, missing = supervised_contrastive(positive_batch, labels, .07)
    assert missing == 0 and ordinary.item() >= 0
    torch.testing.assert_close(ordinary, official_reference(positive_batch, labels, .07))
    # Retain the official active floor even for positives: no GLOBAL >=0 claim.
    opposite = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    saturated, _ = supervised_contrastive(opposite, torch.tensor([0, 0]), .03)
    assert saturated.item() == pytest.approx(-1/.03-math.log(1e-6))
    for invalid in (torch.zeros(2, 2), torch.ones(1, 2)):
        with pytest.raises(ValueError):
            supervised_contrastive(invalid, torch.zeros(len(invalid), dtype=torch.long), .07)


def test_cluda_source_source_trains_only_projector_and_coefficient_ratio():
    from MIL_CREDA_EEG.comparators.cluda import training
    data, record = fixture_data(14, 14)
    model = CLUDAModel(14, 14)
    model.load_state_dict(record[0], strict=False)
    two_phase.bridge(model)
    model.projector.train()
    source = data.rotations[0]["source_train"]
    # Queue repeats provide a multi-bag subject-domain; unlike singleton pairs it can train.
    terms = training.adaptation_step(model, [source[0], source[0], source[1], source[2]],
                                    data.rotations[0]["target_train"], DEFAULT_PARAMS, "cpu")
    assert terms["source_source"].requires_grad
    terms["source_source"].backward()
    assert all(p.grad is None for p in model.source_trunk.parameters())
    assert all(p.grad is None for p in model.target_trunk.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.projector.parameters())
    torch.testing.assert_close(terms["loss"], terms["source_ce"] + DEFAULT_PARAMS["contrastiveWeight"] *
                               (terms["contrastive"] + .5*terms["source_source"]))
    from MIL_CREDA_EEG.comparators.cluda.losses import contrastive_objective
    ce, st, ss = (torch.tensor(v, dtype=torch.float64) for v in (2., 3., 7.))
    assert contrastive_objective(ce, st, ss, .4).item() == pytest.approx(4.6)


def test_cluda_singleton_source_pairs_are_inert_without_floor_saturation():
    from MIL_CREDA_EEG.comparators.cluda.losses import source_source_contrastive
    source = torch.tensor([[1., 0.], [1., .1], [0., 1.], [.1, 1.]],
                          dtype=torch.float64, requires_grad=True)
    diagnostics = {}
    ss, excluded = source_source_contrastive(source, torch.tensor([0, 0, 1, 1]),
                                             diagnostics=diagnostics)
    assert ss.item() == pytest.approx(0., abs=1e-14)
    ss.backward()
    assert source.grad.abs().sum() < 1e-12
    assert excluded == 8  # four different-class pairs, two excluded anchors each
    assert diagnostics["zero_loss_pairs"] == diagnostics["pairs"] == 6


def test_cluda_domain_averages_match_official_loops():
    from MIL_CREDA_EEG.comparators.cluda.losses import source_source_contrastive
    torch.manual_seed(42)
    source = torch.randn(4, 5, dtype=torch.float64)
    target = torch.randn(3, 5, dtype=torch.float64)
    labels, pseudo = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0])
    st, _ = cross_domain_contrastive(source, target, labels, pseudo, .2)
    expected_st = torch.stack([official_reference(torch.cat((source[i:i+1], target)),
                              torch.cat((labels[i:i+1], pseudo)), .2) for i in range(4)]).mean()
    torch.testing.assert_close(st, expected_st)
    ss, _ = source_source_contrastive(source, labels)
    expected_ss = torch.stack([official_reference(source[[i, j]], labels[[i, j]], .03)
                               for i in range(4) for j in range(i+1, 4)]).mean()
    torch.testing.assert_close(ss, expected_ss)
    # Repeated draws of the SAME subject stay in one domain, not independent domains.
    domains = ["a", "a", "b", "c"]
    grouped, _ = cross_domain_contrastive(source, target, labels, pseudo, .2, domains)
    expected_grouped = torch.stack([official_reference(torch.cat((source[idx], target)),
                                  torch.cat((labels[idx], pseudo)), .2)
                                  for idx in ([0, 1], [2], [3])]).mean()
    torch.testing.assert_close(grouped, expected_grouped)


def test_cluda_projector_is_training_only_and_differentiable(monkeypatch):
    torch.manual_seed(8)
    common = CommonMILModel(14, 14)
    model = CLUDAModel(14, 14)
    model.load_state_dict(common.state_dict(), strict=False)
    common.eval(); model.eval()
    x = [torch.randn(2, 14, 512), torch.randn(3, 14, 512)]
    torch.testing.assert_close(model.predict(x, "target"), common.predict(x, "target"))
    before = model.predict(x, "target")
    with torch.no_grad():
        for p in model.projector.parameters():
            p.add_(10)
    torch.testing.assert_close(model.predict(x, "target"), before)
    monkeypatch.setattr(model.projector, "forward", lambda *args: (_ for _ in ()).throw(
        AssertionError("projector called at inference")))
    torch.testing.assert_close(model.predict(x, "target"), before)
    fresh = CLUDAModel(14, 14)
    with pytest.raises(ValueError, match="at least two bags"):
        fresh.projector(torch.zeros(1, 256, dtype=torch.float64))
    # All projector parameters and target inputs receive finite nonzero gradients.
    model = CLUDAModel(14, 14)
    z = torch.randn(6, 256, dtype=torch.float64, requires_grad=True)
    e = model.projector(z)
    assert e.shape == (6, 256)
    loss, _ = cross_domain_contrastive(e[:2], e[2:], torch.tensor([0, 1]),
                                        torch.tensor([0, 1, 0, 1]), .07)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    for p in model.projector.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_cluda_phase_two_freezes_common_state_and_trains_target_and_projector(monkeypatch):
    from MIL_CREDA_EEG.comparators.cluda import training
    data, record = fixture_data()
    # Fix both pseudo-classes to exercise the nondegenerate gradient path, not confidence selection.
    monkeypatch.setattr(training, "pseudo_labels", lambda logits: torch.arange(len(logits),
                        device=logits.device) % 2)
    out = train_phase_two(data, 0, 19, record, DEFAULT_PARAMS, schedule())
    model = out["model"]
    for k, v in record[0].items():
        assert torch.equal(model.state_dict()[k], v), k
    assert out["layer_copy"]["kept_initialization"] == ["depthwise"]
    for component in (model.target_trunk, model.projector):
        assert any(p.grad.abs().sum() > 0 for p in component.parameters())
        for name, p in component.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(not p.requires_grad and p.grad is None for p in model.source_trunk.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.head.parameters())
    assert all(p.grad is None for p in model.attention_parameters())
    assert not model.source_trunk.training
    assert not torch.equal(record[0]["source_trunk.conv_temporal.weight"],
                           model.target_trunk.conv_temporal.weight)
    assert all("source_zero_loss_pair_fraction" in row for row in out["result"]["history"])


def test_cluda_argmax_has_no_threshold_and_target_training_labels_are_unread():
    from MIL_CREDA_EEG.comparators.cluda.training import pseudo_labels
    logits = torch.tensor([[.001, 0.], [0., .001]], requires_grad=True)
    assert pseudo_labels(logits).tolist() == [0, 1]
    assert not pseudo_labels(logits).requires_grad
    data, record = fixture_data(14, 14)
    first = train_phase_two(data, 0, 0, record, DEFAULT_PARAMS, schedule(1))
    d = dict(data.rotations[0])
    d["target_train"] = [two_phase.BagTensor(b.subject, b.condition, object(), b.windows)
                           for b in d["target_train"]]
    assert all(not isinstance(b.label, int) for b in d["target_train"])
    changed = search.SearchData(data.transfer, data.partition_sha256, {0: d})
    second = train_phase_two(changed, 0, 0, record, DEFAULT_PARAMS, schedule(1))
    for k, v in first["model"].state_dict().items():
        assert torch.equal(v, second["model"].state_dict()[k]), k
    assert first["result"]["history"] == second["result"]["history"]


def test_cluda_last_epoch_validation_and_refit(monkeypatch):
    data, record = fixture_data(14, 14)
    answers = [torch.tensor([[.6, .4], [.4, .6]]*2, dtype=torch.float64),
               torch.tensor([[.01, .99], [.99, .01]]*2, dtype=torch.float64)]
    outputs = iter(answers)
    monkeypatch.setattr(two_phase, "evaluate", lambda *args: next(outputs))
    result = train_phase_two(data, 0, 0, record, DEFAULT_PARAMS, schedule(2))["result"]
    assert result["best_epoch"] == 1 == result["checkpoint"]["epoch"]
    assert [v["probabilities"] for v in result["validation"]] == [a.tolist() for a in answers]
    for row in result["history"]:
        assert row["loss"] == pytest.approx(row["source_ce"] +
                                             DEFAULT_PARAMS["contrastiveWeight"]*(row["contrastive"] + .5*row["source_source"]))
        assert row["anchors_without_positive"] >= 0
        assert len(row["step_anchors_without_positive"]) == 1
    d = {**data.rotations[0], "source_valid": [], "target_valid": []}
    refit_data = search.SearchData(data.transfer, data.partition_sha256, {0: d})
    monkeypatch.setattr(two_phase, "evaluate", lambda *args: (_ for _ in ()).throw(AssertionError("evaluated")))
    refit = train_phase_two(refit_data, 0, 1, record, DEFAULT_PARAMS, schedule(2))["result"]
    assert refit["epochs"] == 2 and refit["validation"] == []


def test_cluda_rejects_foreign_records_invalid_parameters_and_leaking_roles():
    data, record = fixture_data()
    for key, value in [("name", "MIL-CREDA"), ("phase", 2), ("rotation", 1),
                       ("seed", 0), ("partition_sha256", "foreign"),
                       ("transfer", "wrong"), ("channels", {"source": 14, "target": 29})]:
        assert record[1][key] != value
        with pytest.raises(ValueError):
            train_phase_two(data, 0, 0, (record[0], {**record[1], key: value}),
                            DEFAULT_PARAMS, schedule(1))
    for key, value in [("contrastiveWeight", -1), ("lr2", 0), ("wd2", -1),
                       ("temperature", 0), ("temperature", float("nan")), ("extra", 1)]:
        with pytest.raises(ValueError):
            train_phase_two(data, 0, 0, record, {**DEFAULT_PARAMS, key: value}, schedule(1))
    d = copy.copy(data.rotations[0]); d["target_valid"] = d["target_train"]
    assert {b.subject for b in d["target_valid"]} & {b.subject for b in d["target_train"]}
    with pytest.raises(ValueError, match="training/validation"):
        train_phase_two(search.SearchData(data.transfer, data.partition_sha256, {0: d}),
                        0, 0, record, DEFAULT_PARAMS, schedule(1))
    for s, t in [(torch.zeros(0, 2), torch.zeros(2, 2)),
                 (torch.zeros(2, 3), torch.zeros(2, 2))]:
        with pytest.raises(ValueError):
            cross_domain_contrastive(s, t, torch.tensor([0, 1]), torch.tensor([0, 1]), .07)


def test_cluda_separate_projector_batches_skip_singleton_target_and_require_source_classes(monkeypatch):
    from MIL_CREDA_EEG.comparators.cluda import training
    data, record = fixture_data(14, 14)
    d = dict(data.rotations[0])
    base = d["target_train"][0]
    d["target_train"] = [*d["target_train"], two_phase.BagTensor("t-extra", base.condition,
                                                               object(), base.windows)]
    changed = search.SearchData(data.transfer, data.partition_sha256, {0: d})
    calls, visited = [], []
    original_step = training.adaptation_step
    def step(model, source, target, params, device):
        visited.extend(b.subject for b in target)
        hook = model.projector.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
        try:
            return original_step(model, source, target, params, device)
        finally:
            hook.remove()
    monkeypatch.setattr(training, "adaptation_step", step)
    result = train_phase_two(changed, 0, 0, record, DEFAULT_PARAMS, schedule(1))["result"]
    row = result["history"][0]
    assert row["skipped_singleton_target_steps"] == 1
    assert row["target_bags_contributing"] == 4
    assert len(visited) == len(set(visited)) == 4
    assert set(visited) < {b.subject for b in d["target_train"]}
    assert calls == [4, 4]  # separate target and source forwards, not a joint eight-bag BN batch
    monkeypatch.setattr(training, "CLUDAModel", lambda *args: (_ for _ in ()).throw(
        AssertionError("model built before role validation")))
    for role, bags, message in [("target_train", d["target_train"][:1], "at least two bags"),
                                 ("source_train", [b for b in d["source_train"] if b.label == 0], "no bag of class")]:
        invalid = {**d, role: bags}
        assert len(bags) < 2 if role == "target_train" else all(b.label == 0 for b in bags)
        with pytest.raises(ValueError, match=message):
            train_phase_two(search.SearchData(data.transfer, data.partition_sha256, {0: invalid}),
                            0, 0, record, DEFAULT_PARAMS, schedule(1))


def test_cluda_search_registration_is_explicit(monkeypatch):
    # Integration e5de342 owns these shared declarations; test the real join.
    assert "CLUDA" in model_store.STRATEGIES
    assert search.PHASE_ONE_GROUP_OF["CLUDA"] == "common"
    assert search.OWN_AXES["CLUDA"] == (search.CONTRASTIVE_TEMPERATURE,)
    monkeypatch.setattr(search, "PHASE_TWO_EXTRA", dict(search.PHASE_TWO_EXTRA))
    monkeypatch.setattr(search, "PHASE_TWO_TRAINERS", dict(search.PHASE_TWO_TRAINERS))
    before = dict(search.PHASE_TWO_TRAINERS)
    importlib.reload(importlib.import_module("MIL_CREDA_EEG.comparators.cluda"))
    assert search.PHASE_TWO_TRAINERS == before
    register()
    assert search.PHASE_TWO_TRAINERS["CLUDA"] is train_phase_two
    space = search.phase_two_space("CLUDA")
    assert set(space) == set(DEFAULT_PARAMS)
    assert space["contrastiveWeight"] == search.LOSS_WEIGHT
    assert space["temperature"] == FloatDistribution(.01, 1., log=True)


def test_cluda_saved_state_round_trips_and_checks_source_digest(tmp_path, monkeypatch):
    data, record = fixture_data(14, 14)
    out = train_phase_two(data, 0, 0, record, DEFAULT_PARAMS, schedule(1))
    monkeypatch.setattr(model_store, "STRATEGIES", (*model_store.STRATEGIES, "CLUDA"))
    path = search.save_phase_two(tmp_path, "CLUDA", data, 0, 0, out)
    monkeypatch.setattr(model_store.partition, "check_result", lambda *args: None)
    restored, meta = load_phase_two(path)
    assert meta["hyper"]["cluda_source_sha256"] == source_digest()
    for k, v in out["model"].state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), k
    import json
    meta["hyper"]["cluda_source_sha256"] = "foreign"
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="CLUDA source"):
        load_phase_two(path)


def test_cluda_synthetic_two_phase_smoke():
    from MIL_CREDA_EEG.comparators.cluda.smoke import run
    result = run()
    values = [r["loss"] for r in result["history"]]
    assert all(math.isfinite(v) for v in values)
    assert max(values)-min(values) > 1e-5


def test_cluda_every_provenance_invariant_has_a_test():
    for name in ("losses", "model", "training", "smoke", "__init__"):
        module = importlib.import_module(f"MIL_CREDA_EEG.comparators.cluda.{name}")
        assert module.__provenance__["revision"] == "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md"
        for invariant in module.__provenance__["invariants"]:
            assert callable(globals().get(f"test_{invariant}")), invariant
