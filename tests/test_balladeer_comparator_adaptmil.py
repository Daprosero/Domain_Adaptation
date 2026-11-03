"""AdaptMIL paper mechanisms and plan-v40 controlled-variant invariants, CPU only."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from MIL_CREDA_EEG import assignment, eegnet, model_store, partition, search, two_phase
import MIL_CREDA_EEG.comparators.adaptmil as adaptmil
from MIL_CREDA_EEG.comparators.adaptmil import model as model_module
from MIL_CREDA_EEG.comparators.adaptmil import objectives as obj
from MIL_CREDA_EEG.comparators.adaptmil import training as tr

SCHEDULE = {"epochs1": 2, "patience1": 1, "epochs2": 2, "bagsPerStep": 2}
PARAMS1 = {"lr1": 1e-3, "wd1": 1e-5}
PARAMS2 = {"lr2": 1e-3, "wd2": 1e-5, "mmdWeight": 2.5, "kernelMultiplier": 1.0}


@pytest.fixture(autouse=True)
def isolated_runtime():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(34)
    builders = dict(model_store.BUILDERS)
    trainers = dict(search.PHASE_TWO_TRAINERS)
    phase_one = dict(search.PHASE_ONE_TRAINERS)
    spaces = copy.deepcopy(search.PHASE_TWO_EXTRA)
    yield
    search.PHASE_ONE_TRAINERS.clear()
    search.PHASE_ONE_TRAINERS.update(phase_one)
    model_store.BUILDERS.clear()
    model_store.BUILDERS.update(builders)
    search.PHASE_TWO_TRAINERS.clear()
    search.PHASE_TWO_TRAINERS.update(trainers)
    search.PHASE_TWO_EXTRA.clear()
    search.PHASE_TWO_EXTRA.update(spaces)
    torch.set_num_threads(threads)


def bags(prefix, channels=14, windows=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [two_phase.BagTensor(f"{prefix}{i}", prefix, i % 2,
                               torch.randn(windows + i, channels, 512, generator=generator))
            for i in range(2)]


def data_fixture():
    d = {"source_train": bags("s"), "source_valid": bags("v", seed=1),
         "target_train": bags("t", seed=2), "target_valid": bags("u", seed=3)}
    return search.SearchData(assignment.Transfer("task", "Robots", "CogniFit"), "test-sha",
                             {r: d for r in range(4)})


def state(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def trained_source(data=None):
    data = data or data_fixture()
    trained = tr.train_phase_one(data, 0, 0, PARAMS1, SCHEDULE)
    return data, trained


def record_for(data, trained):
    return trained["result"]["checkpoint"]["state"], {
        "phase": 1, "name": "AdaptMIL", "transfer": data.transfer.name,
        "rotation": 0, "seed": 0, "partition_sha256": data.partition_sha256,
        "adaptmil_source_digest": tr.source_digest()}


def test_adaptmil_native_pool_is_permutation_invariant():
    pool = adaptmil.NativeABMIL(5)
    h = torch.randn(7, 5, dtype=torch.float64)
    p = pool.project(h)
    z, w = pool(p)
    order = torch.tensor([3, 1, 6, 0, 4, 2, 5])
    zp, wp = pool(p[order])
    torch.testing.assert_close(zp, z)
    torch.testing.assert_close(wp, w[order])
    torch.testing.assert_close(w.sum(), torch.ones((), dtype=w.dtype))
    assert bool((w > 0).all())
    # Closed-form uniform attention must give the mean, not a sum.
    with torch.no_grad():
        pool.w.weight.zero_()
    torch.testing.assert_close(pool(p)[0], p.mean(0))
    torch.testing.assert_close(pool(p)[1], torch.full((7,), 1/7, dtype=p.dtype))


@pytest.mark.parametrize("channels", [(14, 14), (29, 14), (14, 29)])
def test_adaptmil_uses_shared_eegnet_and_one_bag_head(channels):
    model = adaptmil.AdaptMILModel(*channels).eval()
    assert type(model.source_trunk) is eegnet.EEGNetTrunk
    assert type(model.head) is eegnet.EEGNetHead
    calls = []
    handle = model.head.register_forward_hook(lambda m, args, out: calls.append(args[0].shape))
    for domain, c in zip(("source", "target"), channels):
        batch = [b.windows for b in bags(domain, c)]
        p = model.predict(batch, domain)
        assert p.shape == (2, 2)
        torch.testing.assert_close(p.sum(1), torch.ones(2, dtype=p.dtype))
        assert calls[-1] == (2, 256)
        torch.testing.assert_close(model.predict(batch, domain, chunk=2), p)
    handle.remove()
    assert len(calls) == 4
    assert not set(map(id, model.source_trunk.parameters())) & set(map(id, model.target_trunk.parameters()))
    assert model.aggregator.V.weight.shape == (256, 256)
    assert model.aggregator.projection[0].weight.shape == (256, 256)


def test_adaptmil_projection_and_attention_have_finite_gradients():
    model = adaptmil.AdaptMILModel(14, 14)
    source = bags("s")
    loss = obj.stage_two_objective(model, [b.windows for b in source], torch.tensor([0, 1]))
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("target_trunk."):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert sum(float(p.grad.abs().sum()) for p in model.aggregator.parameters()) > 0


def test_adaptmil_mean_pooling_preserves_bag_boundaries(monkeypatch):
    model = adaptmil.AdaptMILModel(14, 14).eval()
    first = torch.arange(3 * 256, dtype=torch.float64).reshape(3, 256) / 256
    second = torch.ones(7, 256, dtype=torch.float64) * 4
    monkeypatch.setattr(model, "embed", lambda *args: [first, second])
    actual = model.representations([torch.zeros(1)], "source", mean=True)
    expected = torch.stack([model.aggregator.project(first).sum(0) / 3,
                            model.aggregator.project(second).sum(0) / 7])
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[0], actual[1])


def test_adaptmil_refuses_empty_bags_and_unknown_domains():
    model = adaptmil.AdaptMILModel(14, 14)
    for batch in ([], [torch.empty(0, 14, 512)]):
        with pytest.raises(ValueError, match="nonempty"):
            model.predict(batch, "source")
    with pytest.raises(ValueError, match="domain"):
        model.trunk("other")
    with pytest.raises(ValueError, match="nonempty"):
        model.aggregator(torch.empty(0, 256, dtype=torch.float64))


def test_adaptmil_mmd_matches_closed_form_and_symmetries():
    s = torch.tensor([[0.]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[2.]], dtype=torch.float64, requires_grad=True)
    expected = sum(2 - 2 * math.exp(-4 / (2*k*k)) for k in obj.KERNEL_SCALES) / 3
    torch.testing.assert_close(obj.multi_kernel_mmd(s, t, 1), torch.tensor(expected, dtype=s.dtype))
    torch.testing.assert_close(obj.multi_kernel_mmd(s, s, 1), torch.zeros((), dtype=s.dtype))
    x, y = torch.randn(3, 5, dtype=s.dtype), torch.randn(4, 5, dtype=s.dtype)
    a = obj.multi_kernel_mmd(x, y, .7)
    torch.testing.assert_close(a, obj.multi_kernel_mmd(y.flip(0), x.flip(0), .7))
    torch.testing.assert_close(a, obj.multi_kernel_mmd(x + 7, y + 7, .7))
    assert a >= -1e-14
    assert torch.autograd.gradcheck(lambda u, v: obj.multi_kernel_mmd(u, v, .8), (s, t))


def test_adaptmil_equations_one_to_four_match_their_terms():
    model = adaptmil.AdaptMILModel(14, 14).eval()
    s = [b.windows for b in bags("s")]
    t = [b.windows for b in bags("t", seed=1)]
    labels = torch.tensor([0, 1])
    for mean, function in ((False, obj.single_stage_objective), (True, obj.stage_one_objective)):
        zs = model.representations(s, "source", mean=mean)
        zt = model.representations(t, "target", mean=mean)
        expected = F.cross_entropy(model.head.linear(zs), labels) + 2.5 * obj.multi_kernel_mmd(zs, zt, .5)
        torch.testing.assert_close(function(model, s, labels, t, 2.5, .5), expected)
        torch.testing.assert_close(obj.uda_objective(model.head.linear(zs), labels, zs, zt, 2.5, .5), expected)
    torch.testing.assert_close(obj.stage_two_objective(model, s, labels),
                               F.cross_entropy(model.logits(s, "source"), labels))
    torch.testing.assert_close(obj.stage_one_objective(model, s, labels, t, 0, .5),
                               F.cross_entropy(model.logits(s, "source", mean=True), labels))


def test_adaptmil_alignment_never_uses_attention_or_target_labels(monkeypatch):
    model = adaptmil.AdaptMILModel(14, 14).eval()
    tr.bridge(model)
    model.eval()
    s, t = [b.windows for b in bags("s")], [b.windows for b in bags("t", seed=1)]
    a = obj.alignment_objective(model, s, t, 2.5, 1)["objective"]
    monkeypatch.setattr(model.aggregator, "forward", lambda *a: pytest.fail("attention called"))
    monkeypatch.setattr(model.head, "forward", lambda *a: pytest.fail("head called"))
    with torch.no_grad():
        model.aggregator.w.weight.add_(100)
    b = obj.alignment_objective(model, s, t, 2.5, 1)["objective"]
    torch.testing.assert_close(a, b)
    b.backward()
    assert model.aggregator.w.weight.grad is None
    for name, p in model.named_parameters():
        if name.startswith("target_trunk."):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
    model.zero_grad(set_to_none=True)
    disabled = obj.alignment_objective(model, s, t, 0, 1)["objective"]
    disabled.backward()
    assert float(disabled.detach()) == 0
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.target_trunk.parameters())


def test_adaptmil_bandwidth_is_training_only_and_degenerate_safe():
    s, t = torch.tensor([[0.], [2.]]), torch.tensor([[4.]])
    assert obj.median_bandwidth(s, t, .5) == 1
    assert obj.median_bandwidth(torch.zeros(2, 3), torch.zeros(1, 3), .75) == .75
    model = adaptmil.AdaptMILModel(14, 14)
    tr.bridge(model)
    original = state(model)
    sigma = tr._bandwidth(model, bags("s"), bags("t", seed=1), 1, torch.device("cpu"))
    assert sigma > 0 and math.isfinite(sigma)
    assert equal(original, state(model))  # no BatchNorm update during measurement


def test_adaptmil_bandwidth_is_measured_in_the_mode_adaptation_trains_in():
    """The width comes from the representations phase 2 actually aligns: BatchNorm on
    batch statistics, dropout off -- MIL-CREDA's sigma rule (two_phase
    .within_bag_median_distance). An eval-mode width saturates the train-mode kernel."""
    model = adaptmil.AdaptMILModel(14, 14)
    tr.bridge(model)
    source, target = bags("s"), bags("t", seed=1)
    sigma = tr._bandwidth(model, source, target, 1, torch.device("cpu"))
    modes = {m: m.training for m in model.modules()}
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    with torch.no_grad():
        s = model.representations([b.windows for b in source], "source", mean=True,
                                  chunk=two_phase.EVAL_CHUNK)
        t = model.representations([b.windows for b in target], "target", mean=True,
                                  chunk=two_phase.EVAL_CHUNK)
    for m, mode in modes.items():
        m.train(mode)
    assert sigma == pytest.approx(obj.median_bandwidth(s, t, 1), rel=1e-6)


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf")])
def test_adaptmil_mmd_refuses_invalid_inputs(sigma):
    with pytest.raises(ValueError):
        obj.multi_kernel_mmd(torch.zeros(2, 3), torch.zeros(2, 3), sigma)
    with pytest.raises(ValueError):
        obj.multi_kernel_mmd(torch.empty(0, 3), torch.zeros(2, 3), 1)
    with pytest.raises(ValueError):
        obj.multi_kernel_mmd(torch.zeros(2, 3), torch.zeros(2, 4), 1)
    with pytest.raises(ValueError):
        obj.multi_kernel_mmd(torch.full((2, 3), float("nan")), torch.zeros(2, 3), 1)


def test_adaptmil_source_pretraining_restores_best_native_checkpoint(monkeypatch):
    model = adaptmil.AdaptMILModel(14, 14)
    # Make a later epoch worse; prove restoration does not retain the final epoch.
    snapshots = []
    answers = [torch.tensor([[.9, .1], [.1, .9]]),
               torch.tensor([[.1, .9], [.9, .1]]),
               torch.tensor([[.9, .1], [.1, .9]])]
    def evaluate(*args):
        snapshots.append(two_phase.phase_one_state(model))
        return answers[len(snapshots)-1]
    monkeypatch.setattr(two_phase, "evaluate", evaluate)
    target = state(model.target_trunk)
    out = tr.train_source_phase(model, bags("s"), bags("v", seed=1),
                                 search.phase_one_hyper("AdaptMIL", PARAMS1, SCHEDULE),
                                 torch.Generator().manual_seed(0), "cpu")
    assert out["best_epoch"] == 0 and out["epochs"] == 2
    assert not equal(snapshots[0], snapshots[1])
    assert equal(two_phase.phase_one_state(model), snapshots[0])
    assert equal(target, state(model.target_trunk))
    assert out["restored_valid_balanced_accuracy"] == 1
    assert set(out["history"][0]) >= {"epoch", "fold", "seed", "model", "loss"}


@pytest.mark.parametrize("channels", [(14, 14), (29, 14), (14, 29)])
def test_adaptmil_bridge_freezes_all_but_target_and_preserves_buffers(channels):
    model = adaptmil.AdaptMILModel(*channels)
    depth = model.target_trunk.depthwise.weight.detach().clone()
    out = tr.bridge(model)
    for name, p in model.named_parameters():
        assert p.requires_grad == name.startswith("target_trunk.")
    if channels[0] == channels[1]:
        torch.testing.assert_close(model.target_trunk.depthwise.weight, model.source_trunk.depthwise.weight)
    else:
        assert "depthwise" in out["layer_copy"]["kept_initialization"]
        torch.testing.assert_close(model.target_trunk.depthwise.weight, depth)
        assert torch.count_nonzero(depth) > 0
    model.train()
    assert not model.source_trunk.training and not model.aggregator.training
    assert not model.head.training and model.target_trunk.training
    before = two_phase.phase_one_state(model)
    model.predict([b.windows for b in bags("s", channels[0])], "source")
    assert equal(before, two_phase.phase_one_state(model))


def test_adaptmil_adaptation_updates_only_target_with_label_blind_loss():
    _, trained = trained_source()
    model = trained["model"]
    tr.bridge(model)
    original = copy.deepcopy(model)
    target_before = state(model.target_trunk)
    frozen_before = two_phase.phase_one_state(model)
    s, t, valid = bags("s"), bags("t", seed=2), bags("u", seed=3)
    a = tr.adapt_target_phase(model, s, t, valid, PARAMS2, SCHEDULE,
                              torch.Generator().manual_seed(9), "cpu")
    # Poison training target labels. Every selected state and loss must remain identical.
    poisoned = [replace(b, label=123456) for b in t]
    torch.manual_seed(999)
    # Reset dropout RNG identically for both repeat runs below.
    left, right = copy.deepcopy(original), copy.deepcopy(original)
    torch.manual_seed(55)
    b = tr.adapt_target_phase(left, s, t, valid, PARAMS2, SCHEDULE,
                              torch.Generator().manual_seed(9), "cpu")
    torch.manual_seed(55)
    c = tr.adapt_target_phase(right, s, poisoned, valid, PARAMS2, SCHEDULE,
                              torch.Generator().manual_seed(9), "cpu")
    assert [b.label for b in poisoned] != [b.label for b in t]
    assert equal(state(left), state(right)) and b["history"] == c["history"]
    assert not equal(target_before, state(model.target_trunk))
    assert equal(frozen_before, two_phase.phase_one_state(model))
    assert a["epochs"] == SCHEDULE["epochs2"]
    assert len({row["sigma"] for row in a["history"]}) == 1
    assert all(math.isfinite(row["loss"]) for row in a["history"])


def test_adaptmil_validation_selects_both_rules_without_training_on_labels(monkeypatch):
    model = adaptmil.AdaptMILModel(14, 14)
    tr.bridge(model)
    primary_snapshots = []
    answers = [torch.tensor([[.9, .1], [.1, .9]], dtype=torch.float64),
               torch.tensor([[.01, .99], [.99, .01]], dtype=torch.float64)]
    def evaluate(*args):
        primary_snapshots.append(state(model.target_trunk))
        return answers[len(primary_snapshots)-1]
    monkeypatch.setattr(two_phase, "evaluate", evaluate)
    out = tr.adapt_target_phase(model, bags("s"), bags("t", seed=2), bags("u", seed=3),
                                PARAMS2, SCHEDULE, torch.Generator().manual_seed(0), "cpu")
    assert out["checkpoint"]["epoch"] == 0
    assert out["secondary_checkpoint"]["epoch"] == 1
    assert not equal(primary_snapshots[0], primary_snapshots[1])
    assert equal(state(model.target_trunk), primary_snapshots[0])
    assert equal(out["secondary_checkpoint"]["target_state"], primary_snapshots[1])


def test_adaptmil_search_hook_restarts_from_matching_phase_one_state():
    adaptmil.register()
    data, trained = trained_source()
    record = record_for(data, trained)
    before = {k: v.clone() for k, v in record[0].items()}
    a = search.train_phase_two("AdaptMIL", data, 0, 0, record, PARAMS2, SCHEDULE)
    b = search.train_phase_two("AdaptMIL", data, 0, 0, record, PARAMS2, SCHEDULE)
    assert equal(before, record[0])
    assert equal(state(a["model"]), state(b["model"]))
    assert a["result"]["checkpoint"]["valid_balanced_accuracy"] >= 0
    for key, bad in (("rotation", 1), ("seed", 7), ("partition_sha256", "foreign"),
                     ("transfer", "wrong"), ("name", "common")):
        assert record[1][key] != bad
        with pytest.raises(ValueError, match="record"):
            tr.train_phase_two(data, 0, 0, (record[0], {**record[1], key: bad}), PARAMS2, SCHEDULE)


def test_adaptmil_store_checks_comparator_code_identity(tmp_path, monkeypatch):
    data, trained = trained_source()
    doc = {"format": "test"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    pp = tmp_path / "partition.json"
    pp.write_text(json.dumps(doc))
    data = replace(data, partition_sha256=doc[partition.HASH_FIELD])
    path = tr.save_model(tmp_path, data, 0, 0, 1, trained)
    loaded, meta = tr.load_model(path, partition_path=pp)
    assert equal(two_phase.phase_one_state(loaded), trained["result"]["checkpoint"]["state"])
    assert meta["adaptmil_source_digest"] == tr.source_digest()
    assert set(meta["hyper"]) == {"lr1", "wd1", *two_phase.SCHEDULE_KEYS}
    assert all(math.isfinite(value) for value in meta["hyper"].values())
    # Native architecture rebuild through the unmodified generic model store.
    adaptmil.register()
    generic, _ = model_store.load_model(path, partition_path=pp)
    assert isinstance(generic, adaptmil.AdaptMILModel)
    with pytest.raises(model_store.ModelExists):
        tr.save_model(tmp_path, data, 0, 0, 1, trained)
    monkeypatch.setattr(tr, "source_digest", lambda: "changed")
    with pytest.raises(model_store.CodeVersionMismatch):
        tr.load_model(path, partition_path=pp)
    # The phase-2 trainer takes the record `model_store.load` returned; that load is
    # where foreign code is refused, through the store's code_version.
    monkeypatch.setattr(model_store, "code_version", lambda: "0" * 64)
    with pytest.raises(model_store.CodeVersionMismatch):
        model_store.load(path, partition_path=pp)


def test_adaptmil_registration_is_explicit_and_uses_shared_search_spaces():
    # Importing the package never registers placeholder groups behind the parent's back.
    before = dict(search.PHASE_TWO_TRAINERS)
    assert before.get("AdaptMIL") is None
    adaptmil.register()
    assert search.PHASE_ONE_TRAINERS["AdaptMIL"] is adaptmil.train_source_phase
    model = model_store.build_model("AdaptMIL", 14, 29, {})
    assert type(model) is adaptmil.AdaptMILModel
    space = search.phase_two_space("AdaptMIL")
    assert set(space) == tr.PHASE_TWO_KEYS
    assert space["mmdWeight"] == search.LOSS_WEIGHT
    assert space["kernelMultiplier"] == search.KERNEL_MULTIPLIER
    assert space["lr2"] == search.LR2


def test_adaptmil_training_guards_hyperparameters_and_roles():
    model = adaptmil.AdaptMILModel(14, 14)
    s, t, v = bags("s"), bags("t"), bags("u")
    with pytest.raises(ValueError, match="bridge"):
        tr.adapt_target_phase(model, s, t, v, PARAMS2, SCHEDULE,
                              torch.Generator(), "cpu")
    tr.bridge(model)
    for params, schedule in (({**PARAMS2, "extra": 1}, SCHEDULE),
                              ({**PARAMS2, "kernelMultiplier": 0}, SCHEDULE),
                              ({**PARAMS2, "mmdWeight": -1}, SCHEDULE),
                              ({**PARAMS2, "lr2": float("nan")}, SCHEDULE),
                              (PARAMS2, {**SCHEDULE, "epochs2": 0}),
                              (PARAMS2, {**SCHEDULE, "bagsPerStep": 1})):
        with pytest.raises(ValueError):
            tr.adapt_target_phase(model, s, t, v, params, schedule, torch.Generator(), "cpu")
    with pytest.raises(ValueError, match="overlap"):
        tr.adapt_target_phase(model, s, t, t, PARAMS2, SCHEDULE, torch.Generator(), "cpu")
    with pytest.raises(ValueError, match="nonempty"):
        tr.adapt_target_phase(model, s, [], v, PARAMS2, SCHEDULE, torch.Generator(), "cpu")
    model.aggregator.w.weight.requires_grad_(True)
    assert model.aggregator.w.weight.requires_grad
    with pytest.raises(ValueError, match="only target"):
        tr.adapt_target_phase(model, s, t, v, PARAMS2, SCHEDULE, torch.Generator(), "cpu")


def test_adaptmil_every_declared_invariant_has_a_test():
    from MIL_CREDA_EEG.comparators.adaptmil import __provenance__ as package_provenance
    for provenance in (package_provenance, model_module.__provenance__,
                       obj.__provenance__, tr.__provenance__):
        assert "experiments" in provenance
        for invariant in provenance["invariants"]:
            assert callable(globals().get("test_" + invariant)), invariant


def test_adaptmil_phase_two_reads_the_checkpoint_the_campaign_stores(tmp_path):
    """The campaign stores phase 1 through `search.save_phase_one` and reads it back
    with `model_store.load`; AdaptMIL's phase 2 must accept exactly that record."""
    adaptmil.register()
    data, trained = trained_source()
    doc = {"format": "test"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    pp = tmp_path / "partition.json"
    pp.write_text(json.dumps(doc))
    data = replace(data, partition_sha256=doc[partition.HASH_FIELD])
    path = search.save_phase_one(tmp_path, "AdaptMIL", data, 0, 0, trained)
    record = model_store.load(path, partition_path=pp)
    out = search.train_phase_two("AdaptMIL", data, 0, 0, record, PARAMS2, SCHEDULE)
    assert out["result"]["checkpoint"]["valid_balanced_accuracy"] >= 0
