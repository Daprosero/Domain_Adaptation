"""HASD controlled-variant invariants; synthetic EEG only, never partition writes."""
import copy
import math

import pytest
import torch

from MIL_CREDA_EEG import eegnet, model_store, search, two_phase
from MIL_CREDA_EEG.comparators.hasd import HASDModel, register
from MIL_CREDA_EEG.comparators.hasd.losses import (
    das_loss, gram, pacr_loss, sgir_loss, sinkhorn_plan, total_loss,
)
from MIL_CREDA_EEG.comparators.hasd.prototypes import cluster_assignments, centroids
from MIL_CREDA_EEG.comparators.hasd.training import (
    adaptation_step, begin_adaptation, train_phase_two,
)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(19)
    torch.set_num_threads(1)


def bags(channels=14, poison=False):
    return [two_phase.BagTensor(str(i), "synthetic", -999 if poison else i % 2,
                               torch.randn(3 + i, channels, 512)) for i in range(2)]


def params():
    return {"lr2": 1e-4, "wd2": 1e-6, "lambda_sgir": .001,
            "lambda_pacr": .1, "epsilon": .01, "tau": .1}


def test_hasd_native_attention_shapes_and_permutation():
    model = HASDModel(29, 14).eval()
    assert isinstance(model.source_trunk, eegnet.EEGNetTrunk)
    assert isinstance(model.head, eegnet.EEGNetHead)
    x = [b.windows for b in bags(29)]
    h = model.embed(x, "source")
    assert [tuple(v.shape) for v in h] == [(3, 256), (4, 256)]
    pairs = model.bag_pairs(h)
    expected = torch.softmax(torch.tanh(h[0] @ model.attention.V.T) @ model.attention.w, 0)
    torch.testing.assert_close(pairs[0][1], expected)
    torch.testing.assert_close(pairs[0][1].sum(), torch.tensor(1., dtype=torch.float64))
    z = model.represent(pairs)
    assert z.shape == (2, 256)
    torch.testing.assert_close(z[0], (h[0] * expected[:, None]).sum(0))
    p = model.predict(x, "source")
    assert p.shape == (2, 2)
    torch.testing.assert_close(p, model.predict([v.flip(0) for v in x], "source"))
    torch.testing.assert_close(p, model.head(z))
    with pytest.raises(ValueError):
        model.embed([torch.empty(0, 29, 512)], "source")
    with pytest.raises(ValueError):
        model.trunk("wrong")


def test_hasd_phase_one_has_finite_gradients_for_every_trainable_parameter():
    model = HASDModel(14, 14)
    p = model.predict([b.windows for b in bags()], "source")
    loss = -p[torch.arange(2), torch.tensor([0, 1])].log().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("target_trunk"):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert sum(float(p.grad.abs().sum()) for p in model.attention.parameters()) > 0


def test_hasd_prototypes_are_label_free_centroids_with_fixed_membership():
    h = torch.tensor([[0., 0.], [0., 2.], [10., 0.], [10., 2.]], dtype=torch.float64)
    ids = cluster_assignments(h, k=2)
    c = centroids(h.requires_grad_(), ids)
    torch.testing.assert_close(c[c[:, 0].argsort()], torch.tensor([[0., 1.], [10., 1.]], dtype=h.dtype))
    c.sum().backward()
    torch.testing.assert_close(h.grad, torch.full_like(h, .5))
    ids2 = cluster_assignments(h.detach(), k=2)
    assert torch.equal(ids, ids2)
    assert cluster_assignments(h.detach(), k=10).unique().numel() == 4
    assert cluster_assignments(torch.ones(5, 2), k=3).unique().numel() >= 1
    with pytest.raises(ValueError):
        cluster_assignments(h, k=0)


def test_hasd_gram_sgir_is_unsquared_sum_and_rotation_invariant():
    h = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    moved = h * 2
    torch.testing.assert_close(gram(h), torch.tensor([[5., 11.], [11., 25.]], dtype=h.dtype))
    expected = torch.linalg.matrix_norm(3 * gram(h), ord="fro")
    torch.testing.assert_close(sgir_loss([h, h], [moved, moved]), 2 * expected)
    q = torch.tensor([[0., -1.], [1., 0.]], dtype=h.dtype)
    torch.testing.assert_close(sgir_loss([h], [h @ q]), torch.tensor(0., dtype=h.dtype))
    assert sgir_loss([h], [moved]) > 0
    with pytest.raises(ValueError):
        sgir_loss([h], [moved, moved])


def test_hasd_pacr_is_sum_of_squared_attention_differences():
    a = torch.tensor([.25, .75], dtype=torch.float64)
    b = torch.tensor([.5, .5], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(pacr_loss([a], [b]), torch.tensor(.125, dtype=a.dtype))
    pacr_loss([a], [b]).backward()
    torch.testing.assert_close(b.grad, torch.tensor([.5, -.5], dtype=a.dtype))
    assert pacr_loss([a], [a]) == 0


def test_hasd_sinkhorn_balanced_marginals_and_uniform_known_case():
    cost = torch.zeros(2, 3, dtype=torch.float64)
    p = sinkhorn_plan(cost, epsilon=.1, tau=0)
    torch.testing.assert_close(p, torch.full_like(cost, 1 / 6))
    cost = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.float64)
    p = sinkhorn_plan(cost, epsilon=.1, tau=0)
    torch.testing.assert_close(p.sum(0), torch.full((2,), .5, dtype=cost.dtype))
    torch.testing.assert_close(p.sum(1), torch.full((2,), .5, dtype=cost.dtype))
    assert p[0, 0] > p[0, 1]


def test_hasd_partial_transport_mass_and_closed_form():
    cost = torch.tensor([[2.]], dtype=torch.float64)
    eps, tau = .1, .5
    p = sinkhorn_plan(cost, epsilon=eps, tau=tau)
    torch.testing.assert_close(p, torch.tensor([[math.exp(-2 / (eps + 2 * tau))]], dtype=cost.dtype), atol=1e-8, rtol=1e-7)
    assert p.sum() < 1
    with pytest.raises(ValueError):
        sinkhorn_plan(cost, epsilon=0, tau=tau)
    with pytest.raises(ValueError):
        sinkhorn_plan(cost, epsilon=eps, tau=-1)


def test_hasd_das_includes_entropy_and_both_marginal_kl_terms():
    x = torch.tensor([[1., 0.]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0., 1.]], dtype=torch.float64)
    result = das_loss(x, y, epsilon=.1, tau=.5)
    mass = math.exp(-1 / 1.1)
    expected = mass + .1 * mass * (math.log(mass) - 1) + 1. * (mass * math.log(mass) - mass + 1)
    torch.testing.assert_close(result["loss"], torch.tensor(expected, dtype=x.dtype), atol=1e-8, rtol=1e-7)
    result["loss"].backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    # Envelope gradient at converged optimum equals a central finite difference.
    def f(t):
        return das_loss(torch.tensor([[1., t]], dtype=x.dtype), y, epsilon=.1, tau=.5)["loss"]
    numerical = (f(1e-5) - f(-1e-5)) / 2e-5
    torch.testing.assert_close(x.grad[0, 1], numerical, atol=1e-7, rtol=1e-6)


def test_hasd_total_is_exact_weighted_sum():
    d, s, p = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (2., 3., 5.))
    result = total_loss(d, s, p, .2, .4)
    torch.testing.assert_close(result, torch.tensor(4.6, dtype=torch.float64))
    result.backward()
    assert (d.grad.item(), s.grad.item(), p.grad.item()) == (1., .2, .4)


@pytest.mark.parametrize("source_channels,target_channels", [(14, 14), (29, 14), (14, 29)])
def test_hasd_bridge_freezes_all_but_target_and_keeps_reference(source_channels, target_channels):
    model = HASDModel(source_channels, target_channels)
    initial_spatial = model.target_trunk.depthwise.weight.detach().clone()
    context = begin_adaptation(model)
    if source_channels != target_channels:
        torch.testing.assert_close(model.target_trunk.depthwise.weight, initial_spatial)
    else:
        torch.testing.assert_close(model.target_trunk.depthwise.weight, model.source_trunk.depthwise.weight)
    frozen = copy.deepcopy(context.frozen_state)
    reference = copy.deepcopy(context.reference.state_dict())
    target = bags(target_channels, poison=True)
    source = bags(source_channels, poison=True)
    model.eval()  # identity at the beginning, no dropout/BN mismatch
    step = adaptation_step(model, context, source, target, params(), torch.device("cpu"))
    assert step["sgir"] == 0 and step["pacr"] == 0
    optimizer = torch.optim.Adam(model.target_trunk.parameters(), lr=1e-4)
    optimizer.zero_grad()
    step["objective"].backward()
    for name, p in model.named_parameters():
        if name.startswith("target_trunk"):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None and not p.requires_grad, name
    optimizer.step()
    model.constrain("target_trunk")
    assert any(not torch.equal(v, reference[k]) for k, v in model.target_trunk.state_dict().items())
    for k, v in context.reference.state_dict().items():
        assert torch.equal(v, reference[k])
    for k, v in two_phase.phase_one_state(model).items():
        assert torch.equal(v, frozen[k])
    step2 = adaptation_step(model, context, source, target, params(), torch.device("cpu"))
    assert step2["sgir"] > 0 and step2["pacr"] > 0
    for term in ("sgir", "pacr"):
        gradients = torch.autograd.grad(step2[term], list(model.target_trunk.parameters()),
                                        retain_graph=True)
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_hasd_target_training_labels_are_never_read():
    model = HASDModel(14, 14).eval()
    context = begin_adaptation(model)
    model.eval()
    source, target = bags(), bags()
    first = adaptation_step(model, context, source, target, params(), torch.device("cpu"))
    changed = [two_phase.BagTensor(b.subject, b.condition, 888, b.windows) for b in target]
    second = adaptation_step(model, context, source, changed, params(), torch.device("cpu"))
    torch.testing.assert_close(first["objective"], second["objective"])
    assert [b.label for b in target] != [b.label for b in changed]


def test_hasd_phase_two_keeps_the_last_epoch_and_records_validation_per_epoch():
    model = HASDModel(14, 14)
    schedule = {"epochs1": 2, "epochs2": 2, "bagsPerStep": 2}
    hyper = search.phase_one_hyper("HASD", {"lr1": .001, "wd1": 1e-6}, schedule)
    train, valid = bags(), bags()
    phase1 = two_phase.train_source_phase(model, train, valid, hyper,
                    torch.Generator().manual_seed(0), torch.device("cpu"))
    assert phase1["epochs"] == 2 and phase1["checkpoint"]["epoch"] == 1
    context = begin_adaptation(model)
    result = train_phase_two(model, context, train, bags(poison=True), valid,
                            params(), schedule, seed=0, fold=0, transfer="synthetic")
    assert len(result["history"]) == 2 == len(result["validation"])
    # v50: the last epoch is kept (no early stopping, no restore)
    assert result["checkpoint"]["epoch"] == 1 == result["best_epoch"]
    assert result["checkpoint"]["valid_balanced_accuracy"] == \
        result["history"][-1]["val_balanced_accuracy"]
    for row in result["history"]:
        assert {"epoch", "seed", "fold", "model", "label_free_score", "loss"} <= row.keys()
        assert row["model"] == "HASD:phase2"
        assert math.isfinite(row["loss"])
    prediction = two_phase.evaluate(model, valid, "target", torch.device("cpu"))
    assert result["checkpoint"]["label_free_score"] == two_phase.label_free_score(prediction)
    assert result["checkpoint"]["valid_balanced_accuracy"] == two_phase.balanced_accuracy(
        [b.label for b in valid], prediction.argmax(1).tolist())


def test_hasd_registration_uses_existing_store_and_search_hooks():
    # Registration is explicit: package import must not mutate sibling registries.
    old_builder = model_store.BUILDERS.get("HASD")
    old_trainer = search.PHASE_TWO_TRAINERS.get("HASD")
    old_extra = search.PHASE_TWO_EXTRA.get("HASD")
    try:
        register()
        model = model_store.build_model("HASD", 29, 14, {})
        assert type(model) is HASDModel
        assert set(search.phase_two_space("HASD")) == set(params())
    finally:
        for mapping, old in [(model_store.BUILDERS, old_builder),
                             (search.PHASE_TWO_TRAINERS, old_trainer),
                             (search.PHASE_TWO_EXTRA, old_extra)]:
            if old is None:
                mapping.pop("HASD", None)
            else:
                mapping["HASD"] = old


def test_hasd_sinkhorn_high_tau_low_epsilon_satisfies_stationarity():
    cost = torch.rand(3, 4, dtype=torch.float64)
    p = sinkhorn_plan(cost, epsilon=.001, tau=10.)
    residual = (cost + .001 * p.clamp_min(1e-300).log()
                + 10 * (p.sum(1) * 3).log()[:, None]
                + 10 * (p.sum(0) * 4).log()[None, :])
    assert residual[p > 1e-200].abs().max() < 1e-7
    with pytest.raises(RuntimeError, match="did not converge"):
        sinkhorn_plan(cost, epsilon=.001, tau=10., max_iterations=1)


def test_hasd_search_adapter_and_checkpoint_roundtrip(tmp_path):
    import json
    from MIL_CREDA_EEG import assignment, partition
    from MIL_CREDA_EEG.comparators.hasd.training import train_search_phase_two

    doc = {"format": "synthetic"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    partition_path = tmp_path / "synthetic-partition.json"
    partition_path.write_text(json.dumps(doc))
    transfer = assignment.Transfer("level", "L1", "L6")
    channels = {"source": 14, "target": 14}
    model = HASDModel(14, 14).eval()
    schedule = {"epochs1": 1, "epochs2": 1, "bagsPerStep": 2}
    hyper = search.phase_one_hyper("HASD", {"lr1": .001, "wd1": 1e-6}, schedule)
    state = two_phase.phase_one_state(model)
    path = model_store.save(tmp_path / "Models", transfer=transfer, rotation=0, seed=0,
             phase=1, name="HASD", state=state, hyper=hyper,
             partition_sha256=doc[partition.HASH_FIELD], channels=channels)
    record = model_store.load(path, partition_path=partition_path)
    d = {"source_train": bags(), "source_valid": bags(),
         "target_train": bags(poison=True), "target_valid": bags()}
    data = search.SearchData(transfer, doc[partition.HASH_FIELD], {0: d})
    result = train_search_phase_two(data, 0, 0, record, params(), schedule)
    assert result["strategy"] == "HASD"
    assert result["result"]["checkpoint"]["epoch"] == 0
    for k, v in state.items():
        assert torch.equal(v, result["model"].state_dict()[k])
    for key, value in [("rotation", 1), ("seed", 9), ("name", "common"),
                       ("partition_sha256", "wrong"), ("transfer", "L6->L1")]:
        changed = {**record[1], key: value}
        assert changed[key] != record[1][key]
        with pytest.raises(search.SearchError):
            train_search_phase_two(data, 0, 0, (record[0], changed), params(), schedule)
    bad_state = {**record[0], "not_a_weight": torch.ones(1)}
    with pytest.raises(search.SearchError):
        train_search_phase_two(data, 0, 0, (bad_state, record[1]), params(), schedule)
    # Builder registration is local and restored, so shared tests keep their baseline state.
    old = model_store.BUILDERS.get("HASD")
    try:
        model_store.BUILDERS["HASD"] = lambda s, t, h: HASDModel(s, t)
        loaded, meta = model_store.load_model(path, partition_path=partition_path)
        expected = two_phase.evaluate(model, d["source_valid"], "source", torch.device("cpu"))
        actual = two_phase.evaluate(loaded, d["source_valid"], "source", torch.device("cpu"))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        path2 = search.save_phase_two(tmp_path / "Models", "HASD", data, 0, 0, result)
        saved, meta2 = model_store.load(path2, partition_path=partition_path)
        rebuilt = HASDModel(14, 14)
        rebuilt.load_state_dict(saved)
        predicted = two_phase.evaluate(rebuilt, d["target_valid"], "target", torch.device("cpu"))
        torch.testing.assert_close(predicted, two_phase.evaluate(result["model"], d["target_valid"], "target", torch.device("cpu")), atol=0, rtol=0)
        assert meta2["phase"] == 2
    finally:
        if old is None:
            model_store.BUILDERS.pop("HASD", None)
        else:
            model_store.BUILDERS["HASD"] = old


def test_hasd_synthetic_smoke_executes_both_phases():
    from MIL_CREDA_EEG.comparators.hasd.smoke import run
    result = run()
    assert result["device"] == "cpu" and result["prediction_finite"]
    assert result["prediction_shape"] == [4, 2]
    assert result["kept_initialization"] == ["depthwise"]
    assert len(result["phase1_loss"]) == 2 and len(result["phase2_loss"]) == 3
    assert result["phase1_loss"][0] != result["phase1_loss"][-1]
    assert result["phase2_loss"][0] != result["phase2_loss"][-1]


def test_hasd_runner_uses_shared_assignment_and_heldout_test(tmp_path):
    import numpy as np
    from MIL_CREDA_EEG import assignment, bag_data
    from MIL_CREDA_EEG.comparators.hasd.training import run_transfer_rotation
    items = []
    for fold in range(4):
        for condition in ("L1", "L6"):
            for label in ("no", "yes"):
                subject = f"{condition}-{fold}-{label}"
                path = tmp_path / f"{subject}.npz"
                np.savez(path, signal=torch.randn(3, 14, 512).numpy())
                items.append(bag_data.Bag(subject, condition, subject, label, "F", fold,
                             "synthetic", tuple(str(c) for c in range(14)), 3, path))
    rotations = {str(r): {"test_fold": r, "validation_fold": (r + 1) % 4,
                         "train_folds": [(r + 2) % 4, (r + 3) % 4]} for r in range(4)}
    cohort = bag_data.BagSet(tuple(items), "synthetic-hash-not-a-real-partition", rotations)
    schedule = {"epochs1": 1, "epochs2": 1, "bagsPerStep": 2}
    hyper = search.phase_one_hyper("HASD", {"lr1": .001, "wd1": 1e-6}, schedule)
    transfer = assignment.Transfer("level", "L1", "L6")
    out = run_transfer_rotation(transfer, 0, 7, hyper, params(), schedule, bags=cohort)
    assert out["partition_sha256"] == cohort.partition_sha256
    assert {p["subject"] for p in out["predictions"]} == {"L6-0-no", "L6-0-yes"}
    assert all(p["direction"] == "L1->L6" for p in out["predictions"])
    assert out["phase1"]["history"][0]["model"] == "HASD:phase1"
    assert out["phase2"]["history"][0]["model"] == "HASD:phase2"
