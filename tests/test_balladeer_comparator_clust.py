"""Invariants for the source-free controlled CLUST comparator only."""
import json
from types import SimpleNamespace
import math

import pytest
import torch
from torch.nn import functional as F

from MIL_CREDA_EEG import assignment, model_store, search, two_phase
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
from MIL_CREDA_EEG.comparators import clust
from MIL_CREDA_EEG.comparators.clust import losses, training


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(17)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def params():
    return dict(lr2=.001, wd2=.0001, clusteringWeight=1., consistencyWeight=1.,
                informationWeight=1., confidence=.5, emaMomentum=.9)


def schedule():
    return dict(epochs1=1, patience1=1, epochs2=2, bagsPerStep=4)


def bags(prefix="t", channels=14):
    return [two_phase.BagTensor(prefix + str(i), prefix, i % 2,
                                torch.randn(2 + i % 2, channels, 512)) for i in range(4)]


def model():
    m = CommonMILModel(29, 14)
    two_phase.bridge(m)
    return m


def fitted():
    m = model()
    result = training.adapt_target_phase(m, bags(), bags("v"), params(), schedule(),
                                         torch.Generator().manual_seed(17))
    return m, result


def test_clust_probabilities_favor_nearest_prototype():
    p = losses.prototype_probabilities(torch.eye(2), torch.eye(2))
    assert p.argmax(-1).tolist() == [0, 1]
    assert torch.allclose(p.sum(-1), torch.ones(2))
    assert p[0, 0] == pytest.approx(math.e / (math.e + 1))
    q = losses.conditional_probability(torch.eye(2), torch.eye(2), torch.ones(2))
    assert torch.allclose(p, q)


def test_clust_centroids_use_confident_bags_and_preserve_missing_classes():
    z = torch.tensor([[1., 2.], [3., 4.], [90., 90.]], requires_grad=True)
    mask = losses.confidence_assignments(torch.tensor([[.9, .1], [.8, .2], [.5, .5]]), .6)
    fallback = torch.tensor([[-5., -5.], [-6., -6.]])
    centroid, counts = losses.target_centroids(z, mask, fallback)
    assert counts.tolist() == [2, 0]
    assert torch.equal(centroid, torch.tensor([[2., 3.], [-6., -6.]]))
    centroid.sum().backward()
    assert torch.equal(z.grad, torch.tensor([[.5, .5], [.5, .5], [0., 0.]]))


def test_clust_transport_has_closed_form_and_finite_gradients():
    z = torch.tensor([[2., 0.], [0., 2.]], dtype=torch.float64, requires_grad=True)
    prototypes = torch.eye(2, dtype=torch.float64)
    membership = torch.eye(2, dtype=torch.bool)
    cost = losses.transport_cost(z, prototypes, membership, torch.ones(2))
    assert cost.item() == pytest.approx(math.e / (math.e + 1))
    target, _ = losses.target_centroids(z, membership, prototypes)
    feature, sem, per = losses.clustering_terms(z, prototypes, target, membership)
    total = losses.objective(feature, sem, cost * 0, cost * 0, .25, [1, 1, 1])
    assert torch.allclose(total, .75 * feature + .25 * sem)
    assert per.shape == (2,)
    total.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    zero = losses.transport_cost(z, target, membership, torch.ones(2))
    assert zero.item() == 0


def test_clust_consistency_is_symmetric_and_zero_at_equality():
    p = torch.tensor([[.2, .8], [.5, .5]], dtype=torch.float64)
    q = torch.tensor([[.7, .3], [.1, .9]], dtype=torch.float64, requires_grad=True)
    actual = losses.symmetric_kl(p, q)
    expected = ((p * (p / q).log()).sum(-1) + (q * (q / p).log()).sum(-1)).mean() / 2
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual, losses.symmetric_kl(q, p))
    assert actual > 0 and losses.symmetric_kl(p, p).item() == 0
    actual.backward()
    assert torch.isfinite(q.grad).all()


def test_clust_information_maximization_has_known_limits():
    assert losses.information_maximization(torch.ones(4, 2) / 2).item() == pytest.approx(0)
    assert losses.information_maximization(torch.tensor([[1., 0.]]).repeat(4, 1)).item() == 0
    assert losses.information_maximization(torch.eye(2)).item() == pytest.approx(-math.log(2))


def test_clust_concentration_and_alpha_are_finite_at_degeneracy():
    z = torch.tensor([[1., 0.], [3., 0.]])
    prototypes = torch.tensor([[2., 0.], [0., 5.]])
    mask = torch.tensor([[True, False], [True, False]])
    tau = losses.concentration(z, prototypes, mask)
    expected = 1 / math.log(2 + math.exp(-2) / math.sqrt(2 * math.pi))
    assert tau[0].item() == pytest.approx(expected)
    assert tau[1].item() == 1
    assert losses.dynamic_alpha(torch.tensor(.1), torch.tensor([.2, .2])) == pytest.approx(1.2 / 2.8)
    assert losses.dynamic_alpha(torch.tensor(4.), torch.tensor([5., 5.])) == .5
    assert torch.equal(losses.a_distance(torch.tensor([0., .5, 1.])), torch.tensor([2., 0., 0.]))
    all_empty = torch.zeros(2, 2, dtype=torch.bool)
    f, s, _ = losses.clustering_terms(z.requires_grad_(), prototypes, prototypes, all_empty)
    assert f.item() == 0 and s.item() == 0
    (f + s).backward()
    assert torch.isfinite(z.grad).all()


def test_clust_ema_preserves_missing_classes():
    old, current = torch.ones(2, 3), torch.full((2, 3), 3.)
    actual = losses.ema_update(old, current, torch.tensor([True, False]), .75)
    assert torch.equal(actual, torch.tensor([[1.5] * 3, [1.] * 3]))
    assert not actual.requires_grad


class Unlabeled:
    def __init__(self, windows):
        self.windows = windows
    @property
    def label(self):
        raise AssertionError("training target label read")


def test_clust_adaptation_never_reads_training_labels_or_source_bags():
    m = model()
    def forbidden(*args, **kwargs):
        raise AssertionError("source trunk called during adaptation")
    m.source_trunk.forward = forbidden
    unlabeled = [Unlabeled(b.windows) for b in bags()]
    result = training.adapt_target_phase(m, unlabeled, bags("v"), params(), schedule(),
                                         torch.Generator().manual_seed(17))
    assert result["epochs"] == 2


def test_clust_only_target_encoder_changes_and_all_gradients_are_finite():
    m = model()
    before = training.clone_state(m)
    m.clust_auxiliary = training.Auxiliary(m)
    two_phase.set_phase_two_modes(m)
    step = training.adaptation_step(m, bags(), params(), torch.device("cpu"))
    step["objective"].backward()
    for p in m.target_trunk.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert sum(p.grad.abs().sum() for p in m.target_trunk.parameters()) > 0
    for p in [*m.source_trunk.parameters(), *m.head.parameters(), *m.attention_parameters(),
              *m.clust_auxiliary.teacher.parameters()]:
        assert p.grad is None
    del m.clust_auxiliary
    training.adapt_target_phase(m, bags(), bags("v"), params(), schedule(),
                                torch.Generator().manual_seed(17))
    after = m.state_dict()
    assert any(not torch.equal(v, after[k]) for k, v in before.items() if k.startswith("target_trunk."))
    assert all(torch.equal(v, after[k]) for k, v in before.items() if not k.startswith("target_trunk."))
    assert m.target_trunk.depthwise.weight.flatten(1).norm(dim=1).max() <= 1.000001


def test_clust_teacher_uses_exact_ema_and_never_predicts_diagnosis():
    m = model()
    aux = training.Auxiliary(m)
    old = next(aux.teacher.parameters()).clone()
    with torch.no_grad():
        next(m.target_trunk.parameters()).add_(2)
    aux.update(m, aux.target + 2, torch.tensor([True, False]), .75)
    assert torch.allclose(next(aux.teacher.parameters()), old + .5)
    m.clust_auxiliary = aux
    m.eval()
    inputs = [bags()[0].windows]
    expected = m.head(training.representations(m, inputs))
    with torch.no_grad():
        aux.target.fill_(1000)
        next(aux.teacher.parameters()).fill_(1000)
    assert torch.equal(m.predict(inputs, "target"), expected)


def test_clust_epoch_selection_restores_both_rules_and_dashboard_history(monkeypatch):
    m = model()
    accuracies, free = iter([.9, .2]), iter([.1, .8])
    monkeypatch.setattr(two_phase, "balanced_accuracy", lambda *a: next(accuracies))
    monkeypatch.setattr(two_phase, "label_free_score", lambda *a: next(free))
    result = training.adapt_target_phase(m, bags(), bags("v"), params(), schedule(),
                                         torch.Generator().manual_seed(17), seed=7, fold=2,
                                         transfer="L6->Robots")
    assert result["best_epoch"] == 0 and result["secondary_checkpoint"]["epoch"] == 1
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in result["checkpoint"]["state"].items())
    assert any(not torch.equal(v, result["secondary_checkpoint"]["state"][k])
               for k, v in m.state_dict().items() if k.startswith("target_trunk."))
    row = result["history"][0]
    assert (row["seed"], row["fold"], row["transfer"]) == (7, 2, "L6->Robots")
    assert row["model"].endswith(":phase2") and math.isfinite(row["loss"])


@pytest.mark.parametrize("key,value", [("lr2", 0), ("wd2", -1), ("confidence", .1),
                                       ("emaMomentum", 1), ("informationWeight", float("nan"))])
def test_clust_rejects_invalid_parameters_and_empty_inputs(key, value):
    p = params(); p[key] = value
    with pytest.raises(ValueError):
        training.check_parameters(p)
    with pytest.raises(ValueError, match="nonempty"):
        training.adapt_target_phase(model(), [], bags(), params(), schedule(), torch.Generator())


def test_clust_auxiliary_state_round_trips():
    m, result = fitted()
    restored = training.restore_model(result["checkpoint"]["state"], 29, 14)
    inputs = [bags()[0].windows]
    assert torch.equal(m.predict(inputs, "target"), restored.predict(inputs, "target"))
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in m.state_dict().items())


def test_clust_registration_is_explicit_and_uses_generic_spaces():
    old_trainers, old_extra = dict(search.PHASE_TWO_TRAINERS), dict(search.PHASE_TWO_EXTRA)
    try:
        clust.register()
        space = search.phase_two_space(training.NAME)
        assert set(space) == set(training.PARAMETERS)
        assert space["emaMomentum"] == search.CONFIDENCE_THRESHOLD
        assert search.PHASE_ONE_GROUP_OF[training.NAME] == "common"
    finally:
        search.PHASE_TWO_TRAINERS.clear(); search.PHASE_TWO_TRAINERS.update(old_trainers)
        search.PHASE_TWO_EXTRA.clear(); search.PHASE_TWO_EXTRA.update(old_extra)


def test_clust_search_reuses_common_checkpoint_and_never_reads_source_data():
    class TargetOnly(dict):
        def __getitem__(self, key):
            assert key in ("target_train", "target_valid"), "source data accessed"
            return super().__getitem__(key)
    source = CommonMILModel(29, 14)
    transfer = assignment.Transfer("device", "L6", "Robots")
    data = SimpleNamespace(transfer=transfer, partition_sha256="hash", rotations={0: TargetOnly(
        target_train=[Unlabeled(b.windows) for b in bags()], target_valid=bags("v"))})
    meta = dict(phase=1, name="common", rotation=0, seed=17, transfer=transfer.name,
                partition_sha256="hash", channels={"source": 29, "target": 14})
    record = two_phase.phase_one_state(source), meta
    out = clust.train_phase_two(data, 0, 17, record, params(), schedule())
    assert out["layer_copy"]["kept_initialization"] == ["depthwise"]
    assert torch.equal(out["model"].head.linear.weight, source.head.linear.weight)
    bad_meta = {**meta, "rotation": 1}
    assert bad_meta != meta
    with pytest.raises(search.SearchError):
        clust.train_phase_two(data, 0, 17, (record[0], bad_meta), params(), schedule())


def test_clust_store_checks_comparator_code_version(monkeypatch, tmp_path):
    m, result = fitted()
    partition = model_store.partition
    doc = {"format": "tiny", "tag": "CLUST store invariant"}
    sha = partition.partition_hash(doc)
    doc[partition.HASH_FIELD] = sha
    partition_path = tmp_path / "partition.json"
    partition_path.write_text(json.dumps(doc))
    trained = {"model": m, "result": result, "channels": {"source": 29, "target": 14}}
    kwargs = dict(trained=trained, transfer=assignment.Transfer("device", "L6", "Robots"),
                  rotation=0, seed=17, partition_sha256=sha)
    path = clust.save(tmp_path / "primary", **kwargs)
    rebuilt, meta = clust.load(path, partition_path=partition_path)
    assert meta["selection"] == "primary"
    assert all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in m.state_dict().items())
    secondary_path = clust.save(tmp_path / "secondary", selection="secondary", **kwargs)
    secondary, meta = clust.load(secondary_path, partition_path=partition_path)
    assert meta["selection"] == "secondary"
    assert all(torch.equal(v, secondary.state_dict()[k])
               for k, v in result["secondary_checkpoint"]["state"].items())
    with pytest.raises(model_store.ModelExists):
        clust.save(tmp_path / "primary", **kwargs)
    monkeypatch.setattr(clust, "code_version", lambda: "changed")
    with pytest.raises(model_store.CodeVersionMismatch):
        clust.load(path, partition_path=partition_path)


def test_clust_smoke_uses_seeded_balladeer_shapes_on_cpu(capsys):
    from MIL_CREDA_EEG.comparators.clust.smoke import run
    result = run()
    assert result["device"] == "cpu" and result["seed"] == 17
    assert len(result["phase2_loss"]) == 3
    assert all(math.isfinite(v) for v in result["phase2_loss"])
    assert len(set(result["phase2_loss"])) > 1
    assert "29/14,512" in result["shape"]
    assert "synthetic" in capsys.readouterr().out
