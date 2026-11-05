"""COT mathematical and BALLADEER controlled-variant invariants (CPU)."""
from pathlib import Path
import importlib

import pytest
import torch

from MIL_CREDA_EEG import assignment, model_store, search, two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
from MIL_CREDA_EEG.comparators.cot.losses import (
    squared_cost, source_cross_entropy, cluster_loss, ot_dual, cot_objective,
    maximize_dual, dual_active,
)
from MIL_CREDA_EEG.comparators.cot.model import COTModel
from MIL_CREDA_EEG.comparators.cot.training import (
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
    def bags(prefix, channels, labels):
        return [two_phase.BagTensor(f"{prefix}{i}", "synthetic", y,
                                   torch.randn(2+i % 2, channels, 512))
                for i, y in enumerate(labels)]
    d = {"source_train": bags("s", source_channels, [0, 1]),
         "source_valid": bags("sv", source_channels, [0, 1]),
         "target_train": bags("t", target_channels, [0, 1]),
         "target_valid": bags("v", target_channels, [0, 1])}
    data = search.SearchData(assignment.Transfer("device", "L6", "Robots"),
                             "synthetic-partition", {0: d})
    model = CommonMILModel(source_channels, target_channels)
    record = (two_phase.phase_one_state(model), {
        "phase": 1, "name": "common", "transfer": data.transfer.name,
        "rotation": 0, "seed": search.PHASE_ONE_SEED, "partition_sha256": data.partition_sha256,
        "channels": data.channels(), "hyper": {"lr1": .001, "wd1": 0.},
    })
    return data, record


def schedule(epochs=2):
    return dict(epochs1=1, epochs2=epochs, bagsPerStep=2)


def test_cot_equations_have_closed_form_values_and_gradients():
    x = torch.tensor([[0., 0.], [2., 0.]], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[0., 0.], [3., 0.]], dtype=torch.float64, requires_grad=True)
    cost = squared_cost(x, centers)
    torch.testing.assert_close(cost, torch.tensor([[0., 9.], [4., 1.]], dtype=torch.float64))
    loss = cluster_loss(x, centers)
    assert float(loss.detach()) == 1.
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([[0., 0.], [-2., 0.]], dtype=x.dtype))
    torch.testing.assert_close(centers.grad, torch.tensor([[0., 0.], [2., 0.]], dtype=x.dtype))
    ce = source_cross_entropy(torch.zeros(2, 2, dtype=x.dtype), torch.tensor([0, 1]))
    torch.testing.assert_close(ce, torch.log(torch.tensor(2., dtype=x.dtype)))
    torch.testing.assert_close(cot_objective(ce, loss.detach(), ce*0, .2, .2), ce+.2)
    # Sum over bags (not mean), and no feature normalization.
    torch.testing.assert_close(cluster_loss(x.repeat(2, 1), centers), 2*loss)
    torch.testing.assert_close(cluster_loss(2*x, 2*centers), 4*loss)


def test_cot_dual_matches_uniform_transport_and_is_gauge_invariant():
    cost = torch.tensor([[0., 0., 4., 4.], [4., 4., 0., 0.]], dtype=torch.float64)
    lam = torch.zeros(4, dtype=cost.dtype, requires_grad=True)
    assert float(ot_dual(cost, lam).detach()) == 0.
    torch.testing.assert_close(ot_dual(cost, lam+123.), ot_dual(cost, lam))
    # Same cost to every target: known uniform source expectation.
    cost2 = torch.tensor([[1., 1., 1., 1.], [3., 3., 3., 3.]], dtype=cost.dtype)
    assert float(ot_dual(cost2, lam).detach()) == 2.
    # Independently solve primal with consistent marginals to cross-check Eq. 11.
    from scipy.optimize import linprog
    c = torch.tensor([[0., 1., 7., 9.], [5., 2., 0., 1.]], dtype=cost.dtype)
    A = torch.zeros(6, 8, dtype=cost.dtype)
    for i in range(2):
        A[i, i*4:(i+1)*4] = 1
    for j in range(4):
        A[2+j, j::4] = 1
    primal = linprog(c.flatten().numpy(), A_eq=A.numpy(),
                     b_eq=[.5, .5, .25, .25, .25, .25], bounds=(0, None), method="highs")
    assert primal.success
    p = torch.nn.Parameter(torch.zeros(4, dtype=cost.dtype))
    maximize_dual(c, p, lr=.1, steps=2000)
    assert abs(float(ot_dual(c, p).detach())-primal.fun) < .03


def test_cot_dual_ascent_detaches_cost_and_descent_detaches_potentials():
    cost = torch.tensor([[0., 1., 2., 3.], [2., 0., 1., 3.]],
                        dtype=torch.float64, requires_grad=True)
    potentials = torch.nn.Parameter(torch.zeros(4, dtype=cost.dtype))
    before = ot_dual(cost, potentials).detach()
    maximize_dual(cost, potentials, lr=.1, steps=4)
    assert cost.grad is None
    assert not torch.equal(potentials.detach(), torch.zeros_like(potentials))
    assert ot_dual(cost, potentials).detach() > before
    assert abs(float(potentials.detach().mean())) < 1e-14
    potentials.grad = None
    ot_dual(cost, potentials.detach()).backward()
    assert potentials.grad is None
    assert cost.grad is not None and torch.isfinite(cost.grad).all()
    assert cost.grad.abs().sum() > 0


def test_cot_dual_window_is_first_ten_percent_with_ceiling():
    assert [dual_active(i, 20) for i in range(20)] == [True]*2+[False]*18
    assert [dual_active(i, 3) for i in range(3)] == [True, False, False]
    with pytest.raises(ValueError):
        dual_active(0, 0)


def test_cot_uses_common_inference_and_four_auxiliary_centers():
    torch.manual_seed(8)
    common = CommonMILModel(14, 14)
    cot = COTModel(14, 14)
    cot.load_state_dict(common.state_dict(), strict=False)
    common.eval(); cot.eval()
    x = [torch.randn(2, 14, 512), torch.randn(3, 14, 512)]
    torch.testing.assert_close(cot.predict(x, "source"), common.predict(x, "source"))
    assert cot.target_centers.shape == (4, 256)
    assert cot.potentials.shape == (4,)
    assert cot.source_prototypes.data_ptr() == cot.head.linear.weight.data_ptr()
    before = cot.predict(x, "target")
    with torch.no_grad():
        cot.target_centers.add_(100)
        cot.potentials.add_(100)
    torch.testing.assert_close(cot.predict(x, "target"), before)


def test_cot_phase_two_freezes_shared_parts_and_has_finite_target_gradients():
    data, record = fixture_data()
    out = train_phase_two(data, 0, 19, record, DEFAULT_PARAMS, schedule())
    model = out["model"]
    for k, v in record[0].items():
        assert torch.equal(model.state_dict()[k], v), k
    assert out["layer_copy"]["kept_initialization"] == ["depthwise"]
    for n, p in model.target_trunk.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert any(p.grad.abs().sum() > 0 for p in model.target_trunk.parameters())
    assert model.target_centers.grad is not None
    assert torch.isfinite(model.target_centers.grad).all()
    assert model.potentials.grad is None
    assert all(not p.requires_grad for p in model.source_trunk.parameters())
    assert all(not p.requires_grad for p in model.head.parameters())
    assert not model.source_trunk.training
    assert all(p.grad is None for p in model.attention_parameters())
    copied = record[0]["source_trunk.conv_temporal.weight"]
    assert not torch.equal(copied, model.target_trunk.conv_temporal.weight)


def test_cot_training_is_label_blind_and_reproducible():
    data, record = fixture_data(14, 14)
    out1 = train_phase_two(data, 0, 19, record, DEFAULT_PARAMS, schedule(1))
    d = dict(data.rotations[0])
    d["target_train"] = [two_phase.BagTensor(b.subject, b.condition, object(), b.windows)
                          for b in d["target_train"]]
    changed = search.SearchData(data.transfer, data.partition_sha256, {0: d})
    out2 = train_phase_two(changed, 0, 19, record, DEFAULT_PARAMS, schedule(1))
    for k, v in out1["model"].state_dict().items():
        assert torch.equal(v, out2["model"].state_dict()[k]), k
    assert out1["result"]["history"] == out2["result"]["history"]


def test_cot_selection_and_history_use_shared_metrics():
    data, record = fixture_data(14, 14)
    out = train_phase_two(data, 0, 19, record, DEFAULT_PARAMS, schedule(3))
    result = out["result"]
    h = result["history"]
    assert len(h) == 3
    assert result["best_epoch"] == 2 == result["checkpoint"]["epoch"]   # the last, kept
    G = two_phase.evaluate(out["model"], data.rotations[0]["target_valid"],
                           "target", torch.device("cpu"))
    assert result["checkpoint"]["valid_balanced_accuracy"] == two_phase.balanced_accuracy(
        [0, 1], G.argmax(1).tolist())
    assert result["checkpoint"]["label_free_score"] == two_phase.label_free_score(G)
    for row in h:
        assert {"epoch", "seed", "fold", "model", "loss", "cluster", "ot", "source_ce",
                "cluster_share", "ot_share"} <= row.keys()
        assert row["model"] == "COT:phase2"
        assert row["loss"] == pytest.approx(row["source_ce"]+.2*row["cluster"]+.2*row["ot"])


def test_cot_rejects_foreign_phase_one_records_and_bad_parameters():
    data, record = fixture_data()
    for key, value in [("name", "MIL-CREDA"), ("phase", 2), ("rotation", 1),
                       ("seed", 8), ("partition_sha256", "foreign"), ("transfer", "wrong")]:
        with pytest.raises(ValueError):
            train_phase_two(data, 0, 19, (record[0], {**record[1], key: value}),
                            DEFAULT_PARAMS, schedule(1))
    for key, value in [("alphaCluster", -1), ("lr2", 0), ("dualLR", float("nan")),
                       ("dualMultiplier", .1), ("extra", 1)]:
        with pytest.raises(ValueError):
            train_phase_two(data, 0, 19, record, {**DEFAULT_PARAMS, key: value}, schedule(1))


def test_cot_search_registration_is_explicit_and_restorable():
    old_extra = dict(search.PHASE_TWO_EXTRA)
    old_trainers = dict(search.PHASE_TWO_TRAINERS)
    try:
        register()
        assert search.PHASE_TWO_TRAINERS["COT"] is train_phase_two
        space = search.phase_two_space("COT")
        assert set(space) == set(DEFAULT_PARAMS)
        assert space["alphaCluster"] == search.LOSS_WEIGHT
        assert space["dualMultiplier"] == search.KERNEL_MULTIPLIER
        assert search.PHASE_ONE_GROUP_OF["COT"] == "common"
    finally:
        search.PHASE_TWO_EXTRA.clear(); search.PHASE_TWO_EXTRA.update(old_extra)
        search.PHASE_TWO_TRAINERS.clear(); search.PHASE_TWO_TRAINERS.update(old_trainers)


def test_cot_saved_state_round_trips_with_auxiliaries_and_digest(tmp_path, monkeypatch):
    data, record = fixture_data(14, 14)
    out = train_phase_two(data, 0, 19, record, DEFAULT_PARAMS, schedule(1))
    path = search.save_phase_two(tmp_path, "COT", data, 0, 19, out)
    # Exercise shared store validation without inventing a real partition file.
    monkeypatch.setattr(model_store.partition, "check_result",
                        lambda meta, p: None if meta["partition_sha256"] ==
                        data.partition_sha256 else (_ for _ in ()).throw(ValueError()))
    restored, meta = load_phase_two(path)
    for k, v in out["model"].state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), k
    assert meta["hyper"]["cot_source_sha256"] == source_digest()
    import json
    sidecar = path.with_suffix(".json")
    meta["hyper"]["cot_source_sha256"] = "foreign"
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="COT source"):
        load_phase_two(path)


def test_cot_synthetic_two_phase_smoke():
    import hashlib
    seed = int(hashlib.sha256(b"COT:synthetic:rotation0").hexdigest()[:8], 16) % 2**31
    data, record = fixture_data(29, 14, seed=seed)
    model = CommonMILModel(29, 14)
    hyper = search.phase_one_hyper("common", {"lr1": .001, "wd1": 0.}, schedule(3))
    phase1 = two_phase.train_source_phase(
        model, data.rotations[0]["source_train"], data.rotations[0]["source_valid"],
        hyper, torch.Generator().manual_seed(seed), torch.device("cpu"))
    record = (phase1["checkpoint"]["state"], record[1])
    out = train_phase_two(data, 0, seed, record, DEFAULT_PARAMS, schedule(3))
    losses = [row["loss"] for row in out["result"]["history"]]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert max(losses)-min(losses) > 1e-5
    assert out["model"].predict([torch.randn(2, 14, 512)], "target").shape == (1, 2)
    print({"data": "seeded synthetic BALLADEER shapes, not real data", "device": "cpu",
           "seed": seed, "phase1_loss": phase1["history"][0]["loss"], "phase2_losses": losses})


def test_cot_keeps_the_last_epoch_and_records_validation_per_epoch(monkeypatch):
    data, record = fixture_data(14, 14)
    answers = [torch.tensor([[.6, .4], [.4, .6]], dtype=torch.float64),
               torch.tensor([[.01, .99], [.99, .01]], dtype=torch.float64)]
    outputs = iter(answers)
    monkeypatch.setattr(two_phase, "evaluate", lambda *args: next(outputs))
    out = train_phase_two(data, 0, 19, record, DEFAULT_PARAMS, schedule(2))["result"]
    assert out["best_epoch"] == 1 == out["checkpoint"]["epoch"]
    assert "secondary_checkpoint" not in out
    assert [v["probabilities"] for v in out["validation"]] == [a.tolist() for a in answers]
    # the refit: no validation bags, nothing evaluated, exactly the epochs given
    d = {**data.rotations[0], "source_valid": [], "target_valid": []}
    refit_data = search.SearchData(data.transfer, data.partition_sha256, {0: d})
    monkeypatch.setattr(two_phase, "evaluate", lambda *args: (_ for _ in ()).throw(
        AssertionError("evaluated")))
    refit = train_phase_two(refit_data, 0, 19, record, DEFAULT_PARAMS, schedule(2))["result"]
    assert refit["epochs"] == 2 and refit["validation"] == []


def test_cot_rejects_leaking_roles_and_empty_batches():
    data, record = fixture_data()
    d = dict(data.rotations[0])
    b = d["target_train"][0]
    d["target_valid"] = [two_phase.BagTensor(b.subject, b.condition, b.label, b.windows)]
    assert {x.subject for x in d["target_train"]} & {x.subject for x in d["target_valid"]}
    changed = search.SearchData(data.transfer, data.partition_sha256, {0: d})
    with pytest.raises(ValueError, match="training/validation"):
        train_phase_two(changed, 0, 19, record, DEFAULT_PARAMS, schedule(1))
    with pytest.raises(ValueError, match="nonempty"):
        squared_cost(torch.zeros(0, 2), torch.zeros(4, 2))
    with pytest.raises(ValueError, match="same feature"):
        squared_cost(torch.zeros(2, 3), torch.zeros(4, 2))
    with pytest.raises(ValueError, match="dual needs"):
        ot_dual(torch.zeros(2, 4), torch.zeros(2))


def test_cot_every_provenance_invariant_has_a_test():
    for name in ("losses", "model", "training", "__init__"): 
        module = importlib.import_module(f"MIL_CREDA_EEG.comparators.cot.{name}")
        assert module.__provenance__["experiments"] == ["A-nivel", "B-tarea", "C-dispositivo"]
        for invariant in module.__provenance__["invariants"]:
            assert callable(globals().get(f"test_{invariant}")), invariant
