"""CPU invariants for CASUAL's controlled bag-level BALLADEER variant."""
import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from MIL_CREDA_EEG import assignment, model_store, search, two_phase
from MIL_CREDA_EEG.eegnet import EEGNetTrunk, EEGNetHead
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel
from MIL_CREDA_EEG.comparators.casual import losses as L, training as T


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    yield
    torch.set_num_threads(threads)


def params():
    return dict(lr2=0.001, wd2=1e-5, lambda_align=1., lambda_entropy=0.1,
                lambda_vat=0.1, vat_radius_fraction=0.5)


def schedule(epochs=2):
    return dict(epochs1=1, patience1=1, epochs2=epochs, bagsPerStep=2)


def bags(channels, prefix="s", n=4):
    return [two_phase.BagTensor(f"{prefix}{i}", "synthetic", i % 2,
                               torch.randn(2 + i % 2, channels, 512)) for i in range(n)]


def model(s=29, t=14):
    m = CommonMILModel(s, t)
    two_phase.bridge(m)
    return m


def test_casual_losses_have_closed_form_values():
    p = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    assert L.classification_loss(p, torch.tensor([0, 1])).item() == pytest.approx(0.69314718056)
    assert L.conditional_entropy(p).item() == pytest.approx(0.69314718056)
    assert L.conditional_entropy(torch.eye(2, dtype=torch.float64)).item() == 0
    assert L.discriminator_loss(torch.zeros(2), torch.zeros(3)).item() == pytest.approx(2 * 0.69314718056)
    logits = torch.tensor([1., -2.], dtype=torch.float64)
    probs = torch.tensor([[0.9, 0.1], [0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    w = L.entropy_weights(probs)
    assert w[0] > w[1]
    assert w.sum().item() == pytest.approx(1.)
    assert not w.requires_grad
    expected = (torch.nn.functional.softplus(-logits) * w).sum() + torch.tensor(0.5).exp().log1p()
    assert L.discriminator_loss(logits, torch.tensor([0.5]), probs, probs[:1]).item() == pytest.approx(expected.item())


def test_casual_joint_features_preserve_conditioning_and_gradients():
    z = torch.tensor([[1., 2.]], requires_grad=True)
    p = torch.tensor([[0.25, 0.75]], requires_grad=True)
    j = L.joint_features(z, p)
    torch.testing.assert_close(j, torch.tensor([[0.25, 0.75, 0.5, 1.5]]))
    j.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))
    torch.testing.assert_close(p.grad, torch.tensor([[3., 3.]]))
    with pytest.raises(ValueError):
        L.joint_features(z, p.repeat(2, 1))
    # Swapping class probabilities changes the joint even if features do not move.
    assert not torch.equal(j, L.joint_features(z, p.flip(1)))


def test_casual_support_matches_sets_not_masses():
    s = torch.tensor([0., 1., 1.], dtype=torch.float64)
    t = torch.tensor([0., 0., 1., 1., 1.], dtype=torch.float64)
    assert L.support_alignment(s, t).item() == 0
    a = torch.tensor([0., 0.2], requires_grad=True)
    b = torch.tensor([0.8], requires_grad=True)
    v = L.support_alignment(a, b)
    assert v.item() == pytest.approx((0.64 + 0.36) / 2 + 0.36)
    torch.testing.assert_close(v, L.support_alignment(b, a))
    assert L.support_alignment(a, b, "l1").item() == pytest.approx(1.3)
    v.backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0
    for x, y in [(s[:0], t), (s, t[:0])]:
        with pytest.raises(ValueError):
            L.support_alignment(x, y)
    with pytest.raises(ValueError):
        L.support_alignment(s, t, "wasserstein")


def test_casual_vat_is_input_bounded_and_mode_isolated():
    m = model()
    two_phase.set_phase_two_modes(m)
    modes = {n: x.training for n, x in m.named_modules()}
    buffers = {n: b.clone() for n, b in m.named_buffers()}
    windows = [b.windows for b in bags(14, "t", 2)]
    zero = L.virtual_adversarial_loss(m, windows, 0.)
    assert abs(zero.item()) < 1e-12
    value, perturbations = L.virtual_adversarial_loss(m, windows, 1., return_perturbations=True)
    assert value.item() > 0
    for x, r in zip(windows, perturbations):
        assert r.shape == x.shape
        assert torch.linalg.vector_norm(r).item() == pytest.approx(1., abs=1e-5)
        assert not r.requires_grad
    assert modes == {n: x.training for n, x in m.named_modules()}
    for n, b in m.named_buffers():
        torch.testing.assert_close(b, buffers[n], rtol=0, atol=0)
    assert all(p.grad is None for p in m.parameters())
    value.backward()
    for p in m.target_trunk.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert sum(p.grad.abs().sum() for p in m.target_trunk.parameters()) > 0
    with pytest.raises(ValueError):
        L.virtual_adversarial_loss(m, windows, -1)
    with torch.no_grad():
        m.head.linear.weight.zero_()
        m.head.linear.bias.zero_()
    assert L.virtual_adversarial_loss(m, windows, 1.).item() == pytest.approx(0., abs=1e-14)


def test_casual_objective_is_the_weighted_sum():
    a, b, c, d = [torch.tensor(float(i), requires_grad=True) for i in (1, 2, 3, 4)]
    v = L.adaptation_objective(a, b, c, d, lambda_align=2., lambda_entropy=3., lambda_vat=4.)
    assert v.item() == 30.
    v.backward()
    assert [x.grad.item() for x in (a, b, c, d)] == [1., 2., 3., 4.]


@pytest.mark.parametrize("s,t", [(29, 14), (14, 29), (14, 14)])
def test_casual_alternation_updates_only_allowed_parameters(s, t):
    m = model(s, t)
    frozen = model_store.state_digest(two_phase.phase_one_state(m))
    target = model_store.state_digest(m.target_trunk.state_dict())
    dis = T.DomainDiscriminator()
    old_dis = model_store.state_digest(dis.state_dict())
    op = torch.optim.Adam(m.target_trunk.parameters(), lr=0.001)
    dop = torch.optim.Adam(dis.parameters(), lr=0.001)
    original_step = dop.step
    # Running BN buffers legitimately move on the target forward, before either optimizer.
    # Compare target parameters, not BN buffers, at the auxiliary update boundary.
    initial_parameters = {n: p.clone() for n, p in m.target_trunk.named_parameters()}
    def checked_discriminator_step(*args, **kwargs):
        assert all(p.grad is None for p in m.target_trunk.parameters())
        for n, p in m.target_trunk.named_parameters():
            torch.testing.assert_close(p, initial_parameters[n], rtol=0, atol=0)
        return original_step(*args, **kwargs)
    with patch.object(dop, "step", side_effect=checked_discriminator_step):
        out = T.alternating_step(m, dis, bags(s, n=3), bags(t, "t", 2), params(), op, dop,
                                 vat_radius=0.5)
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())
    assert out["vat"] > 0 and out["entropy"] > 0
    assert frozen == model_store.state_digest(two_phase.phase_one_state(m))
    assert target != model_store.state_digest(m.target_trunk.state_dict())
    assert old_dis != model_store.state_digest(dis.state_dict())
    for p in m.target_trunk.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.source_trunk.parameters())
    assert all(p.grad is None for p in dis.parameters())
    # Auxiliary Eq.17 reaches every discriminator parameter independently.
    ds = dis(torch.randn(3, 512, dtype=torch.float64))
    dt = dis(torch.randn(2, 512, dtype=torch.float64))
    L.discriminator_loss(ds, dt).backward()
    for p in dis.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    G = two_phase.evaluate(m, bags(t, "v", 2), "target", torch.device("cpu"))
    assert G.shape == (2, 2)
    torch.testing.assert_close(G.sum(1), torch.ones(2, dtype=G.dtype))
    assert isinstance(m.source_trunk, EEGNetTrunk) and isinstance(m.head, EEGNetHead)
    assert torch.linalg.vector_norm(m.target_trunk.depthwise.weight.flatten(1), dim=1).max() <= 1.000001


def data_record():
    data = search.SearchData(assignment.Transfer("device", "L6", "Robots"), "synthetic-hash", {
        0: dict(source_train=bags(29), source_valid=bags(29, "sv"),
                target_train=bags(14, "t"), target_valid=bags(14, "v"))})
    m = CommonMILModel(29, 14)
    meta = dict(phase=1, name="common", rotation=0, seed=19, transfer="L6->Robots",
                partition_sha256="synthetic-hash", channels=dict(source=29, target=14))
    return data, (two_phase.phase_one_state(m), meta)


def test_casual_checkpoint_identity_is_checked():
    data, (state, meta) = data_record()
    for key, value in dict(name="MIL-CREDA", phase=2, rotation=1, seed=0,
                           transfer="Robots->L6", partition_sha256="other",
                           channels=dict(source=14, target=29)).items():
        with pytest.raises(ValueError):
            T.train_phase_two(data, 0, 19, (state, {**meta, key: value}), params(), schedule())
    with pytest.raises(ValueError):
        T.train_phase_two(data, 0, 19, ({}, meta), params(), schedule())


def test_casual_selection_runs_full_budget_and_keeps_both_states():
    data, record = data_record()
    scores = [torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64),
              torch.tensor([[0.4, 0.6], [0.6, 0.4]], dtype=torch.float64),
              torch.tensor([[0.99, 0.01], [0.01, 0.99]], dtype=torch.float64)]
    # Validation labels distinguish the primary epoch; free score distinguishes another.
    data.rotations[0]["target_valid"] = bags(14, "v", 2)
    with patch.object(two_phase, "evaluate", side_effect=scores):
        trained = T.train_phase_two(data, 0, 19, record, params(), schedule(3))
    result = trained["result"]
    assert result["epochs"] == 3
    assert result["best_epoch"] == 0
    assert result["selection"]["best_label_free_epoch"] == 2
    assert result["history"][0]["val_balanced_accuracy"] == 1.
    assert result["history"][1]["val_balanced_accuracy"] == 0.
    for r in result["history"]:
        assert {"model", "seed", "fold", "epoch", "transfer", "loss", "label_free_score"} <= r.keys()
    for k, v in trained["model"].target_trunk.state_dict().items():
        torch.testing.assert_close(v, result["checkpoint"]["target_state"][k], rtol=0, atol=0)
    assert model_store.state_digest(result["checkpoint"]["target_state"]) != model_store.state_digest(result["label_free_checkpoint"]["target_state"])


def test_casual_training_does_not_read_target_labels():
    class Unlabelled:
        def __init__(self, b):
            self.windows = b.windows
        @property
        def label(self):
            raise AssertionError("target training labels accessed")
    data, record = data_record()
    data.rotations[0]["target_train"] = [Unlabelled(b) for b in data.rotations[0]["target_train"]]
    trained = T.train_phase_two(data, 0, 19, record, params(), schedule(1))
    assert trained["result"]["epochs"] == 1


def test_casual_search_hook_uses_common_group_and_the_radius_fraction_axis():
    before = dict(search.PHASE_TWO_TRAINERS), dict(search.PHASE_TWO_EXTRA)
    try:
        with pytest.raises(TypeError):
            T.register(search.LOSS_WEIGHT)  # no caller-supplied radius any more
        T.register()
        assert search.PHASE_ONE_GROUP_OF["CASUAL"] == "common"
        space = search.phase_two_space("CASUAL")
        assert set(space) == T.PARAM_KEYS
        assert space["vat_radius_fraction"] == search.VAT_RADIUS_FRACTION
        for k in ("lambda_align", "lambda_entropy", "lambda_vat"):
            assert space[k] == search.LOSS_WEIGHT
        assert search.PHASE_TWO_TRAINERS["CASUAL"] is T.train_phase_two
    finally:
        search.PHASE_TWO_TRAINERS.clear()
        search.PHASE_TWO_TRAINERS.update(before[0])
        search.PHASE_TWO_EXTRA.clear()
        search.PHASE_TWO_EXTRA.update(before[1])


def test_casual_vat_radius_is_the_fraction_of_the_median_target_training_window_norm(
        monkeypatch):
    import statistics
    data, record = data_record()
    target = data.rotations[0]["target_train"]
    expected = statistics.median(float(torch.linalg.vector_norm(w.double()))
                                 for b in target for w in b.windows)
    assert T.typical_window_norm(target) == pytest.approx(expected, rel=1e-12)
    # The median of an even count is the mean of the two middle norms, not the lower one.
    two = [two_phase.BagTensor("x", "s", 0, torch.stack([torch.full((1, 4), 1.),
                                                         torch.full((1, 4), 2.)]))]
    assert T.typical_window_norm(two) == pytest.approx(3.)
    measured, radii = [], []
    real_norm, real_vat = T.typical_window_norm, T.virtual_adversarial_loss
    monkeypatch.setattr(T, "typical_window_norm",
                        lambda b: measured.append(b) or real_norm(b))
    monkeypatch.setattr(T, "virtual_adversarial_loss",
                        lambda m, w, r: radii.append(r) or real_vat(m, w, r))
    trained = T.train_phase_two(data, 0, 19, record, {**params(), "vat_radius_fraction": 0.1},
                                schedule(2))
    assert len(measured) == 1 and measured[0] is target  # once, target training bags only
    assert radii and all(r == pytest.approx(0.1 * expected, rel=1e-12) for r in radii)
    for row in trained["result"]["history"]:
        assert row["typical_window_norm"] == pytest.approx(expected, rel=1e-12)
        assert row["vat_radius"] == pytest.approx(0.1 * expected, rel=1e-12)
    with pytest.raises(ValueError):
        T.typical_window_norm([])


def test_casual_import_does_not_register_or_fork_common_model():
    old = dict(search.PHASE_TWO_TRAINERS)
    importlib.reload(importlib.import_module("MIL_CREDA_EEG.comparators.casual"))
    assert old == search.PHASE_TWO_TRAINERS
    assert T.CommonMILModel is CommonMILModel
    assert T.two_phase.bridge is two_phase.bridge


def test_casual_models_round_trip_with_comparator_digest(tmp_path):
    data, record = data_record()
    trained = T.train_phase_two(data, 0, 19, record, params(), schedule(1))
    path = T.save_final_model(tmp_path, data, 0, 19, trained)
    # Shared partition validation tested through the real load join; synthetic fixture has no partition file.
    with patch("MIL_CREDA_EEG.partition.check_result") as check:
        loaded, meta = T.load_final_model(path, partition_path=tmp_path / "folds.json")
        check.assert_called_once()
    assert meta["partition_sha256"] == data.partition_sha256
    assert model_store.state_digest(loaded.state_dict()) == model_store.state_digest(trained["model"].state_dict())
    test_bags = bags(14, "test", 2)
    torch.testing.assert_close(two_phase.evaluate(loaded, test_bags, "target", torch.device("cpu")),
                               two_phase.evaluate(trained["model"], test_bags, "target", torch.device("cpu")), rtol=0, atol=0)
    sidecar = path.with_suffix(".json")
    contents = json.loads(sidecar.read_text())
    contents["casual_code_digest"] = "changed"
    sidecar.write_text(json.dumps(contents))
    with patch("MIL_CREDA_EEG.partition.check_result"), pytest.raises(model_store.CodeVersionMismatch):
        T.load_final_model(path)


@pytest.mark.parametrize("key,value", [("lr2", 0), ("vat_radius_fraction", float("nan")),
                                       ("vat_radius_fraction", 0),
                                       ("lambda_align", -1), ("wd2", -1)])
def test_casual_invalid_parameters_refuse(key, value):
    with pytest.raises(ValueError):
        T.check_params({**params(), key: value})
    with pytest.raises(ValueError):
        T.check_params({**params(), "unplanned_axis": 1})


def test_casual_smoke_runs_common_phase_one_and_cpu_adaptation():
    from MIL_CREDA_EEG.comparators.casual.smoke import run
    result = run()
    assert result["device"] == "cpu"
    assert result["phase1_epochs"] == 1 and result["phase2_epochs"] == 3
    assert len(set(result["phase2_loss"])) > 1
    assert all(x > 0 for x in result["phase2_vat"])


def test_casual_provenance_invariants_have_tests():
    from pathlib import Path
    package = Path(T.__file__).parent
    text = Path(__file__).read_text()
    for path in package.glob("*.py"):
        module = importlib.import_module("MIL_CREDA_EEG.comparators.casual." + path.stem) if path.stem != "__init__" else importlib.import_module("MIL_CREDA_EEG.comparators.casual")
        assert module.__provenance__["experiments"] == ["A-nivel", "B-tarea", "C-dispositivo"]
        for invariant in module.__provenance__["invariants"]:
            assert f"def test_{invariant}(" in text
