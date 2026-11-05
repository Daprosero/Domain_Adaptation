"""CS+CCS estimators (Yin et al., UAI 2024) and controlled BALLADEER phase-2 invariants (CPU)."""
import copy
import importlib
import json
import math
import sys
from dataclasses import replace

import pytest
import torch

from MIL_CREDA_EEG import assignment, eegnet, mil_wrapper, model_store, search, two_phase
from MIL_CREDA_EEG.comparators.cs_ccs import divergences, smoke, training
from MIL_CREDA_EEG.comparators.cs_ccs.divergences import (ccs_divergence, cs_divergence,
                                                          median_bandwidth, normalise)
from MIL_CREDA_EEG.comparators.cs_ccs.smoke import fixture

D = torch.float64


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def registries():
    """Save and restore every global registry a comparator can touch."""
    saved = (dict(model_store.BUILDERS), dict(search.PHASE_ONE_TRAINERS),
             dict(search.PHASE_TWO_TRAINERS), {k: dict(v) for k, v in search.PHASE_TWO_EXTRA.items()})
    yield
    for target, old in zip((model_store.BUILDERS, search.PHASE_ONE_TRAINERS,
                            search.PHASE_TWO_TRAINERS, search.PHASE_TWO_EXTRA), saved):
        target.clear()
        target.update(old)


def params(**over):
    return {**training.DEFAULTS, "lr2": 0.001, "wd2": 0., **over}


def schedule(epochs=3):
    return {"epochs1": 1, "epochs2": epochs, "bagsPerStep": 4}


def search_data(source_channels=29, target_channels=14, seed=11):
    model, s, sv, t, tv = fixture(source_channels, target_channels, seed)
    data = search.SearchData(assignment.Transfer("device", "L6", "Robots"), "synthetic-hash", {
        0: {"source_train": s, "source_valid": sv, "target_train": t, "target_valid": tv}})
    record = (two_phase.phase_one_state(model), {
        "phase": 1, "name": "common", "transfer": data.transfer.name, "rotation": 0,
        "seed": search.PHASE_ONE_SEED, "partition_sha256": "synthetic-hash",
        "channels": data.channels(),
        "hyper": {"lr1": .001, "wd1": 0.}})
    return data, record


# ------------------------------------------------------------------ Eqs. 11 and 13


def direct_cs(zs, zt, sigma):
    """Eq. 11 transcribed literally, non-log kernels."""
    k = lambda a, b: torch.exp(-torch.cdist(a, b) ** 2 / (2 * sigma ** 2))
    M, N = len(zs), len(zt)
    return (torch.log(k(zs, zs).sum() / M ** 2) + torch.log(k(zt, zt).sum() / N ** 2)
            - 2 * torch.log(k(zs, zt).sum() / (M * N)))


def direct_ccs_terms(zs, zt, ys, yt, sz, sy):
    """The four log terms of Eq. 13 transcribed literally (rows j, columns i)."""
    k = lambda a, b, s: torch.exp(-torch.cdist(a, b) ** 2 / (2 * s ** 2))
    Ks, Kt, Kst, Kts = k(zs, zs, sz), k(zt, zt, sz), k(zs, zt, sz), k(zt, zs, sz)
    Ls, Lt, Lst, Lts = k(ys, ys, sy), k(yt, yt, sy), k(ys, yt, sy), k(yt, ys, sy)
    self_s = torch.log(((Ks * Ls).sum(1) / Ks.sum(1) ** 2).sum())
    self_t = torch.log(((Kt * Lt).sum(1) / Kt.sum(1) ** 2).sum())
    cross_st = torch.log(((Kst * Lst).sum(1) / (Ks.sum(1) * Kst.sum(1))).sum())
    cross_ts = torch.log(((Kts * Lts).sum(1) / (Kts.sum(1) * Kt.sum(1))).sum())
    return self_s, self_t, cross_st, cross_ts


def test_cs_divergence_has_closed_form_on_single_points():
    zs = torch.tensor([[0., 0.]], dtype=D)
    zt = torch.tensor([[3., 4.]], dtype=D, requires_grad=True)
    value = cs_divergence(zs, zt, 2.0)
    # M = N = 1: -2 log exp(-25 / 8) = 25 / 4.
    torch.testing.assert_close(value, torch.tensor(25 / 4, dtype=D))
    value.backward()
    torch.testing.assert_close(zt.grad, 2 * zt.detach() / 4)


def test_ccs_divergence_has_closed_form_on_single_points():
    zs, zt = torch.tensor([[0., 0.]], dtype=D), torch.tensor([[5., 1.]], dtype=D)
    ys = torch.tensor([[1., 0.]], dtype=D)
    yt = torch.tensor([[0., 1.]], dtype=D, requires_grad=True)
    value = ccs_divergence(zs, zt, ys, yt, 3.0, 0.5)
    # M = N = 1: the z kernels cancel and D = -2 log L^st = ||ys - yt||^2 / sigma_y^2.
    torch.testing.assert_close(value, torch.tensor(2 / 0.25, dtype=D))
    value.backward()
    assert torch.isfinite(yt.grad).all() and yt.grad.abs().sum() > 0


def test_divergences_vanish_on_identical_samples():
    torch.manual_seed(3)
    z, y = torch.randn(5, 4, dtype=D), torch.softmax(torch.randn(5, 2, dtype=D), 1)
    torch.testing.assert_close(cs_divergence(z, z.clone(), 1.3), torch.zeros((), dtype=D))
    torch.testing.assert_close(ccs_divergence(z, z.clone(), y, y.clone(), 1.3, 0.7),
                               torch.zeros((), dtype=D))
    # Same marginal, different conditional: CS stays zero, CCS does not.
    flipped = y.flip(1)
    torch.testing.assert_close(cs_divergence(z, z.clone(), 1.3), torch.zeros((), dtype=D))
    assert ccs_divergence(z, z.clone(), y, flipped, 1.3, 0.7) > 1e-3


def test_cs_divergence_is_nonnegative():
    gen = torch.Generator().manual_seed(5)
    for _ in range(50):
        m, n = torch.randint(1, 7, (2,), generator=gen).tolist()
        zs = torch.randn(m, 3, generator=gen, dtype=D)
        zt = torch.randn(n, 3, generator=gen, dtype=D) + torch.rand(1, generator=gen, dtype=D)
        assert cs_divergence(zs, zt, float(torch.rand(1, generator=gen)) + 0.1) >= -1e-12


def test_ccs_estimator_matches_eq_13_transcription_and_is_symmetric():
    torch.manual_seed(7)
    zs, zt = torch.randn(6, 3, dtype=D), torch.randn(4, 3, dtype=D) + 0.5
    ys, yt = torch.softmax(torch.randn(6, 2, dtype=D), 1), torch.softmax(torch.randn(4, 2, dtype=D), 1)
    self_s, self_t, cross_st, cross_ts = direct_ccs_terms(zs, zt, ys, yt, 1.1, 0.4)
    value = ccs_divergence(zs, zt, ys, yt, 1.1, 0.4)
    torch.testing.assert_close(value, self_s + self_t - cross_st - cross_ts)
    # Eq. 86: the mean of the two one-sided estimators Eq. 84 (-2 cross_st) and Eq. 85 (-2 cross_ts).
    eq84, eq85 = self_s + self_t - 2 * cross_st, self_s + self_t - 2 * cross_ts
    torch.testing.assert_close(value, (eq84 + eq85) / 2)
    torch.testing.assert_close(value, ccs_divergence(zt, zs, yt, ys, 1.1, 0.4))
    torch.testing.assert_close(cs_divergence(zs, zt, 1.1), direct_cs(zs, zt, 1.1))
    torch.testing.assert_close(cs_divergence(zs, zt, 1.1), cs_divergence(zt, zs, 1.1))


def test_divergences_stay_finite_where_direct_sums_underflow():
    zs = torch.zeros(3, 2, dtype=D)
    zt = torch.full((3, 2), 60., dtype=D, requires_grad=True)
    ys = torch.tensor([[1., 0.]] * 3, dtype=D)
    yt = torch.tensor([[0., 1.]] * 3, dtype=D)
    assert not torch.isfinite(direct_cs(zs, zt.detach(), 1.0))
    value = cs_divergence(zs, zt, 1.0)
    torch.testing.assert_close(value, torch.tensor(7200., dtype=D))  # ||60,60||^2 / sigma^2
    ccs = ccs_divergence(zs, zt, ys, yt, 1.0, 0.1)
    (value + ccs).backward()
    assert torch.isfinite(ccs) and torch.isfinite(zt.grad).all()


def test_kernel_inputs_are_l2_normalised():
    torch.manual_seed(2)
    x = torch.randn(5, 7, dtype=D)
    torch.testing.assert_close(normalise(x).norm(dim=1), torch.ones(5, dtype=D))
    torch.testing.assert_close(normalise(3.5 * x), normalise(x))
    model, s, _, t, _ = fixture()
    zs, ys, raw = training.kernel_inputs(model, s, "source", "cpu")
    torch.testing.assert_close(zs.norm(dim=1), torch.ones(len(s), dtype=D))
    torch.testing.assert_close(ys, normalise(model.classify(raw)))
    torch.testing.assert_close(ys.norm(dim=1), torch.ones(len(s), dtype=D))
    with pytest.raises(ValueError):
        normalise(torch.ones(3))


def test_median_bandwidth_is_multiplier_times_positive_median():
    points = torch.tensor([[0.], [0.], [1.], [3.]], dtype=D)
    # positive distances 1, 1, 3, 3, 2 -> median 2 (the zero pair is excluded).
    assert median_bandwidth(points, 0.75) == 1.5
    assert median_bandwidth(torch.zeros(4, 2, dtype=D), 1.5) == 1.5 * divergences.PAPER_SIGMA
    assert median_bandwidth(torch.zeros(1, 2, dtype=D), 1.0) == divergences.PAPER_SIGMA
    for bad in (0., -1., math.inf):
        with pytest.raises(ValueError):
            median_bandwidth(points, bad)
    with pytest.raises(ValueError):
        cs_divergence(points, points, 0.)


# ------------------------------------------------------------------ controlled phase 2


@pytest.mark.parametrize("source_channels,target_channels", [(29, 14), (14, 29), (29, 29)])
def test_cs_ccs_phase_two_freezes_common_predictor(source_channels, target_channels):
    model, source, _, target, valid = fixture(source_channels, target_channels)
    frozen = two_phase.phase_one_state(model)
    result = training.adapt(model, source, target, valid, params(), schedule(), seed=11)
    assert all(torch.equal(v, two_phase.phase_one_state(model)[k]) for k, v in frozen.items())
    assert result["layer_copy"]["kept_initialization"] == (
        ["depthwise"] if source_channels != target_channels else [])
    assert isinstance(model.head, eegnet.EEGNetHead) and model.source_trunk.training is False
    assert all(not p.requires_grad for p in [*model.source_trunk.parameters(),
                                             *model.attention_parameters(),
                                             *model.head.parameters()])
    trainable = list(model.target_trunk.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert any(p.grad.abs().sum() > 0 for p in trainable)
    assert result["epochs"] == 3


class UnreadableLabel:
    def __int__(self):
        raise AssertionError("target training label read")

    def __index__(self):
        raise AssertionError("target training label read")

    def __eq__(self, other):
        raise AssertionError("target training label compared")

    __hash__ = None


def test_cs_ccs_target_training_labels_are_unread():
    a, source, _, target, valid = fixture()
    b = copy.deepcopy(a)
    poisoned = [replace(bag, label=UnreadableLabel()) for bag in target]
    first = training.adapt(a, source, target, valid, params(), schedule(), seed=11)
    second = training.adapt(b, source, poisoned, valid, params(), schedule(), seed=11)
    assert first["history"] == second["history"]
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    # Flipping every target training label changes nothing either.
    model, *_ = fixture()
    flipped = [replace(bag, label=1 - bag.label) for bag in target]
    third = training.adapt(model, source, flipped, valid, params(), schedule(), seed=11)
    assert third["history"] == first["history"]


def test_cs_ccs_bandwidths_are_fixed_from_training_bags_at_phase_two_start(monkeypatch):
    model, source, _, target, valid = fixture()
    reference = copy.deepcopy(model)
    two_phase.bridge(reference)
    expected = training.phase_two_bandwidths(reference, source, target, params(), "cpu")
    # Independent reading: every source and target TRAINING bag, alone, in eval mode.
    reference.eval()
    with torch.no_grad():
        rows = [training.kernel_inputs(reference, [b], d, "cpu")
                for d, bags in (("source", source), ("target", target)) for b in bags]
    two_phase.set_phase_two_modes(reference)
    assert expected == {"zBandwidth": median_bandwidth(torch.cat([r[0] for r in rows]), 1.),
                        "yBandwidth": 1.0}
    # The bandwidth pass leaves BatchNorm statistics and modes untouched.
    stats = {k: v.clone() for k, v in reference.state_dict().items()}
    two_phase.set_phase_two_modes(reference)
    training.phase_two_bandwidths(reference, source, target, params(), "cpu")
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in stats.items())
    assert reference.target_trunk.training and not reference.source_trunk.training
    calls = []
    original = training.phase_two_bandwidths
    monkeypatch.setattr(training, "phase_two_bandwidths",
                        lambda *a, **k: calls.append(a) or original(*a, **k))
    result = training.adapt(model, source, target, valid, params(), schedule(), seed=11)
    assert len(calls) == 1 and calls[0][1] is source and calls[0][2] is target
    assert result["bandwidths"] == expected
    assert all(h["zBandwidth"] == expected["zBandwidth"] and
               h["yBandwidth"] == expected["yBandwidth"] for h in result["history"])
    assert result["checkpoint"]["hyper"]["zBandwidth"] == expected["zBandwidth"]
    model2, *_ = fixture()
    doubled = training.adapt(model2, source, target, valid, params(zKernelMultiplier=1.5),
                             schedule(1), seed=11)
    torch.testing.assert_close(doubled["bandwidths"]["zBandwidth"], 1.5 * expected["zBandwidth"])
    assert doubled["bandwidths"]["yBandwidth"] == 1.0


def test_cs_ccs_prediction_kernel_is_fixed_at_sigma_one_on_normalised_predictions(monkeypatch):
    # Plan v42 Sec. 7.7: sigma = 1 on the L2-normalised softmax predictions, never searched.
    assert training.PREDICTION_SIGMA == 1.0
    assert "yKernelMultiplier" not in training.DEFAULTS
    model, source, _, target, valid = fixture()
    seen = []
    original = training.ccs_divergence
    def spy(zs, zt, ys, yt, sigma_z, sigma_y):
        seen.append((sigma_y, ys.detach(), yt.detach()))
        return original(zs, zt, ys, yt, sigma_z, sigma_y)
    monkeypatch.setattr(training, "ccs_divergence", spy)
    for multiplier in (0.5, 1.5):
        training.adapt(copy.deepcopy(model), source, target, valid,
                       params(zKernelMultiplier=multiplier), schedule(1), seed=11)
    assert seen and all(sigma == 1.0 for sigma, *_ in seen)
    for _, ys, yt in seen:
        for y in (ys, yt):
            torch.testing.assert_close(y.norm(dim=1), torch.ones(len(y), dtype=y.dtype))


def test_cs_ccs_objective_is_eq_90_with_frozen_source_ce():
    model, source, _, target, _ = fixture()
    two_phase.bridge(model)
    bands = training.phase_two_bandwidths(model, source, target, params(), "cpu")
    terms = training.adaptation_step(model, source, target, params(csWeight=0.3, ccsWeight=2.),
                                     bands, "cpu")
    torch.testing.assert_close(terms["loss"],
                               terms["source_ce"] + 0.3 * terms["cs"] + 2. * terms["ccs"])
    assert not terms["source_ce"].requires_grad
    zs, ys, raw = training.kernel_inputs(model, source, "source", "cpu")
    labels = torch.tensor([b.label for b in source])
    torch.testing.assert_close(terms["source_ce"],
                               torch.nn.functional.cross_entropy(model.head.linear(raw), labels))
    # lambda = beta = 0: Eq. 90 reduces to the frozen CE and moves nothing.
    zero = training.adaptation_step(model, source, target, params(csWeight=0., ccsWeight=0.),
                                    bands, "cpu")
    assert not zero["loss"].requires_grad or all(
        g is None or g.abs().sum() == 0 for g in torch.autograd.grad(
            zero["loss"], list(model.target_trunk.parameters()), allow_unused=True))
    # Each weight is effective on its own.
    for over in ({"csWeight": 1., "ccsWeight": 0.}, {"csWeight": 0., "ccsWeight": 1.}):
        step = training.adaptation_step(model, source, target, params(**over), bands, "cpu")
        grads = torch.autograd.grad(step["loss"], list(model.target_trunk.parameters()),
                                    allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_cs_ccs_keeps_the_last_epoch_and_records_validation_per_epoch(monkeypatch):
    model, source, _, target, valid = fixture()
    scores = iter([0.25, 0.75, 0.5])
    frees = iter([0.6, 0.1, 0.2])
    monkeypatch.setattr(training.two_phase, "balanced_accuracy",
                        lambda *a: next(scores))
    monkeypatch.setattr(training.two_phase, "label_free_score", lambda *a: next(frees))
    result = training.adapt(model, source, target, valid, params(), schedule(3), seed=11)
    # v50: no early stopping, no restore: the last epoch is kept whatever scored best
    assert result["best_epoch"] == 2 == result["checkpoint"]["epoch"]
    assert "secondary_checkpoint" not in result
    final = model.target_trunk.state_dict()
    assert all(torch.equal(v, final[k]) for k, v in result["checkpoint"]["target_state"].items())
    assert all(torch.equal(v, final[k]) for k, v in result["final_target_state"].items())
    assert result["checkpoint"]["valid_balanced_accuracy"] == 0.5
    assert [v["epoch"] for v in result["validation"]] == [0, 1, 2]
    # the refit: no validation bags, nothing scored, exactly the epochs given
    model2, *_ = fixture()
    monkeypatch.setattr(training.two_phase, "balanced_accuracy",
                        lambda *a: (_ for _ in ()).throw(AssertionError("scored")))
    refit = training.adapt(model2, source, target, [], params(), schedule(2), seed=11)
    assert refit["epochs"] == 2 and refit["validation"] == []


def test_cs_ccs_invalid_inputs_fail_closed():
    model, source, _, target, valid = fixture()
    bad = [params(extra=1.), {k: v for k, v in params().items() if k != "csWeight"},
           params(lr2=0.), params(wd2=-1.), params(csWeight=-1.), params(ccsWeight=math.nan),
           params(zKernelMultiplier=2.), params(yKernelMultiplier=1.)]
    for p in bad:
        with pytest.raises(ValueError):
            training.adapt(copy.deepcopy(model), source, target, valid, p, schedule(), seed=11)
    with pytest.raises(ValueError):
        training.adapt(copy.deepcopy(model), source, target, valid, params(),
                       {**schedule(), "bagsPerStep": 1}, seed=11)
    overlap = [replace(target[0], subject=source[0].subject), *target[1:]]
    leak = [replace(valid[0], subject=target[0].subject), *valid[1:]]
    # an empty validation role is a refit (v50 Sec. 4.5), not an error
    for args in ((source, overlap, valid), (source, target, leak), ([], target, valid),
                 (source, [], valid)):
        with pytest.raises(ValueError):
            training.adapt(copy.deepcopy(model), *args, params(), schedule(), seed=11)
    native = model_store.build_model("MIL-CREDA", 29, 14, {
        **mil_wrapper.COMMON_PHASE_ONE_FIXED, "attentionGamma": .5})
    with pytest.raises(ValueError, match="common MIL"):
        training.adapt(native, source, target, valid, params(), schedule(), seed=11)
    with pytest.raises(ValueError):
        ccs_divergence(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, 2), torch.ones(2, 2), 1, 1)
    with pytest.raises(ValueError):
        cs_divergence(torch.ones(0, 3), torch.ones(2, 3), 1.)


def test_cs_ccs_package_is_a_controlled_variant(registries):
    for name in [m for m in sys.modules if m.startswith("MIL_CREDA_EEG.comparators.cs_ccs")]:
        del sys.modules[name]
    before = (dict(search.PHASE_TWO_TRAINERS), dict(search.PHASE_TWO_EXTRA),
              dict(model_store.BUILDERS))
    package = importlib.import_module("MIL_CREDA_EEG.comparators.cs_ccs")
    assert (dict(search.PHASE_TWO_TRAINERS), dict(search.PHASE_TWO_EXTRA),
            dict(model_store.BUILDERS)) == before
    assert "controlled" in package.__doc__ and "not a reproduction" in package.__doc__
    # The trainer, its smoke and its divergences answer to v50 (Secs. 4.5, 7.7: fixed epochs,
    # no patience); the package module itself is still bound to v42.
    for module, revision in (("", "v42"), (".divergences", "v50"), (".training", "v50"),
                             (".smoke", "v50")):
        prov = importlib.import_module("MIL_CREDA_EEG.comparators.cs_ccs" + module).__provenance__
        assert prov["revision"] == (
            f"experiments-plan-experimental-mil-creda-sobre-balladeer-{revision}.md")
    assert "CS+CCS" in mil_wrapper.COMPARATORS and "CS+CCS" in model_store.STRATEGIES


def test_cs_ccs_search_registration_is_explicit_and_restorable(registries):
    assert "CS+CCS" not in search.PHASE_TWO_TRAINERS
    data, record = search_data()
    with pytest.raises(search.SearchError):
        search.train_phase_two("CS+CCS", data, 0, 11, record, params(), schedule())
    training.register()
    space = search.phase_two_space("CS+CCS")
    assert set(space) == set(params())
    assert space["csWeight"] == space["ccsWeight"] == search.LOSS_WEIGHT
    assert space["zKernelMultiplier"] == search.KERNEL_MULTIPLIER
    assert [k for k, d in space.items() if d == search.KERNEL_MULTIPLIER] == ["zKernelMultiplier"]
    assert search.PHASE_ONE_GROUP_OF["CS+CCS"] == mil_wrapper.GROUP
    trained = search.train_phase_two("CS+CCS", data, 0, 11, record, params(), schedule(2))
    assert isinstance(trained["model"], mil_wrapper.CommonMILModel)
    assert trained["result"]["epochs"] == 2 and trained["strategy"] == "CS+CCS"


def test_cs_ccs_search_hook_rejects_foreign_phase_one_records():
    data, record = search_data()
    for key, value in (("rotation", 1), ("seed", 12), ("name", "MIL-CREDA"), ("phase", 2),
                       ("partition_sha256", "other"), ("transfer", "other"),
                       ("channels", {"source": 29, "target": 29})):
        with pytest.raises(search.SearchError, match="common phase-1"):
            training.train_phase_two(data, 0, 11, (record[0], {**record[1], key: value}),
                                     params(), schedule())
    foreign = dict(record[0])
    foreign["extra.weight"] = torch.zeros(1)
    with pytest.raises(search.SearchError, match="incompatible"):
        training.train_phase_two(data, 0, 11, (foreign, record[1]), params(), schedule())


def test_cs_ccs_saved_state_round_trips(tmp_path, monkeypatch):
    data, record = search_data(14, 14)
    out = training.train_phase_two(data, 0, 11, record, params(), schedule(1))
    path = search.save_phase_two(tmp_path, "CS+CCS", data, 0, 11, out)
    monkeypatch.setattr(model_store.partition, "check_result",
                        lambda meta, p: None if meta["partition_sha256"] ==
                        data.partition_sha256 else (_ for _ in ()).throw(ValueError()))
    restored, meta = training.load_phase_two(path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in out["model"].state_dict().items())
    assert meta["hyper"]["zBandwidth"] == out["result"]["bandwidths"]["zBandwidth"]
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    sidecar = path.with_suffix(".json")
    stored = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**stored, "name": "COT"}))
    with pytest.raises(ValueError, match="CS\\+CCS"):
        training.load_phase_two(path)


def test_cs_ccs_training_is_reproducible():
    a, source, _, target, valid = fixture()
    b = copy.deepcopy(a)
    first = training.adapt(a, source, target, valid, params(), schedule(), seed=11)
    second = training.adapt(b, source, target, valid, params(), schedule(), seed=11)
    assert first["history"] == second["history"]
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_cs_ccs_synthetic_smoke_executes(capsys):
    phase1, result = smoke.run(epochs1=2, epochs2=2)
    out = capsys.readouterr().out
    assert "synthetic CPU only" in out and "phase2_ccs" in out
    losses = [h["loss"] for h in result["history"]]
    assert all(math.isfinite(v) for v in losses) and len(losses) == 2
    assert len(phase1["history"]) == 2
