"""Experiments plan v40, Secs. 6.1-6.3: the mechanism analyses over stored models.

Synthetic bags and untrained models only: no number here is a measured result.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from MIL_CREDA.bag_kernel import bag_kernel_matrix
from MIL_CREDA_EEG import assignment, bag_data, mechanisms as mech, model_store, partition
from MIL_CREDA_EEG import statistics as st
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel
from MIL_CREDA_EEG.mil_wrapper import CommonMILModel

CHANNELS = 3


def _bag(root, subject, condition, *, windows=6, label="yes", fold=0, marks=None, seed=0):
    rng = np.random.default_rng(seed)
    path = root / f"{subject}_{condition}.npz"
    marks = marks if marks is not None else {
        "integrity_mark": np.array([True] + [False] * (windows - 1)),
        "contact_bad": np.zeros((windows, CHANNELS), dtype=bool)}
    np.savez(path, signal=rng.normal(size=(windows, CHANNELS, 512)).astype(np.float32),
             channels=np.array(["a", "b", "c"]), **marks)
    return bag_data.Bag(subject=subject, condition=condition, record_id=f"{subject}-{condition}",
                        label=label, sex="M", fold=fold, device="CGX",
                        channels=("a", "b", "c"), n_windows=windows, path=path)


def _model(sigma=2.0):
    torch.manual_seed(0)
    m = MILCREDAModel(CHANNELS, CHANNELS, attention_gamma=0.5, attention_temperature=1.0)
    m.fix_sigma(sigma)
    return m.eval()


def test_k1_reads_only_conditions_with_an_event_stream(tmp_path):
    assert mech.EVENT_CONDITIONS == ("L1", "L6", "L11", "Robots")
    bag = _bag(tmp_path, "S1", "CogniFit")
    with pytest.raises(mech.MechanismError, match="event stream"):
        mech.event_windows(bag)
    with pytest.raises(mech.MechanismError, match="event stream"):
        mech.k1_subject_rows(_model(), [bag], model_name="M",
                             transfer=assignment.Transfer("task", "Robots", "CogniFit"),
                             rotation=0, seed=1, bands=((4, 8), (13, 30)))


def test_a_marked_window_is_an_integrity_or_contact_mark(tmp_path):
    contact = np.zeros((5, CHANNELS), dtype=bool)
    contact[3, 2] = True
    bag = _bag(tmp_path, "S1", "L1", windows=5,
               marks={"integrity_mark": np.array([False, True, False, False, False]),
                      "contact_bad": contact})
    assert mech.marked_windows(bag).tolist() == [False, True, False, True, False]
    no_contact = _bag(tmp_path, "S2", "L1", windows=3,
                      marks={"integrity_mark": np.array([True, False, False])})
    assert mech.marked_windows(no_contact).tolist() == [True, False, False]


def test_an_unrecorded_event_mark_is_refused_not_guessed(tmp_path):
    bag = _bag(tmp_path, "S1", "L1")
    with pytest.raises(mech.NotRecorded, match="event_window"):
        mech.event_windows(bag)
    rows = mech.k1_subject_rows(_model(), [bag], model_name="M",
                                transfer=assignment.Transfer("level", "L11", "L1"), rotation=0,
                                seed=1, bands=((4, 8), (13, 30)))
    assert rows[0]["event_minus_rest"] is None and "event_window" in rows[0]["event_not_recorded"]
    events = _bag(tmp_path, "S2", "L1", windows=4,
                  marks={"integrity_mark": np.zeros(4, bool),
                         "event_window": np.array([1, 0, 0, 1], bool)})
    assert mech.event_windows(events).tolist() == [True, False, False, True]


def test_the_attention_weights_are_the_models_own_and_sum_to_one(tmp_path):
    from MIL_CREDA_EEG.comparators.adaptmil.model import AdaptMILModel
    x = torch.randn(7, CHANNELS, 512)
    model = _model()
    w = mech.attention_weights(model, x, "target")
    with torch.no_grad():
        ref = model.bag_pairs(model.embed([x], "target"))[0][1].numpy()
    assert np.allclose(w, ref) and w.sum() == pytest.approx(1.0)
    common = CommonMILModel(CHANNELS, CHANNELS).eval()
    assert mech.attention_weights(common, x).sum() == pytest.approx(1.0)
    adapt = AdaptMILModel(CHANNELS, CHANNELS).eval()
    wa = mech.attention_weights(adapt, x)
    with torch.no_grad():
        H = adapt.embed([x], "target")[0]
        assert np.allclose(wa, adapt.aggregator.weights(adapt.aggregator.project(H)).numpy())
    assert not model.training


def test_the_weight_contrast_is_marked_minus_clean_and_skips_subjects_without_marks():
    w = np.array([0.1, 0.2, 0.3, 0.4])
    assert mech.weight_contrast(w, [True, False, False, True]) == pytest.approx(0.25 - 0.25)
    assert mech.weight_contrast(w, [True, True, False, False]) == pytest.approx(0.15 - 0.35)
    assert mech.weight_contrast(w, [False] * 4) is None
    assert mech.weight_contrast(w, [True] * 4) is None


def test_the_band_ratio_reads_the_declared_bands_and_has_no_default():
    assert inspect.signature(mech.band_ratio).parameters["bands"].default is \
        inspect.Parameter.empty
    t = np.arange(512) / 128.0
    theta = np.sin(2 * np.pi * 6 * t)
    beta = 0.5 * np.sin(2 * np.pi * 20 * t)
    windows = np.stack([np.stack([theta + beta] * 2), np.stack([beta] * 2)])
    ratio = mech.band_ratio(windows, ((4, 8), (13, 30)))
    assert ratio[0] == pytest.approx(4.0, rel=1e-6)          # power ratio (1 / 0.5)^2
    assert ratio[1] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(mech.MechanismError):
        mech.band_ratio(windows, ((8, 4), (13, 30)))


def test_k1_averages_over_seeds_before_testing_and_holms_the_three_criterion_tests():
    rows = []
    for seed, shift in ((1, 0.0), (2, 0.2)):
        for i in range(8):
            rows.append({"model": "M", "transfer": "L11->L1", "pair": "L1|L11", "seed": seed,
                         "subject": f"S{i}", "marked_minus_clean": -0.1 * (i + 1) + shift,
                         "event_minus_rest": 0.05 * (i + 1), "weight_ratio_spearman": 0.1})
    own = mech.average_over_seeds(rows, mech.K1_VALUES)
    assert own[0]["marked_minus_clean"] == pytest.approx(np.mean([-0.1, 0.1]))
    assert own[0]["seeds"] == [1, 2]
    common = [{**r, "model": "C", "marked_minus_clean": 0.0} for r in own]
    with pytest.raises(mech.MechanismError, match="scope"):
        mech.k1_tests(own, common, own="M", common="C", scope=None)
    table = mech.k1_tests(own, common, own="M", common="C", scope="pooled")
    assert len(table) == 4
    criterion = table[:3]
    assert st.holm([r["p_value"] for r in criterion]) == [r["p_adjusted"] for r in criterion]
    assert table[3]["p_adjusted"] is None and table[3]["criterion"].startswith("none")
    assert all(r["experiment"] == "K1-atencion" for r in table)
    per_pair = mech.k1_tests(own, common, own="M", common="C", scope="per-pair")
    assert {r["comparison"].split(":")[1] for r in per_pair} == {"L1|L11"}


def _assignment(tmp_path):
    bags, holders, folds, labels = [], {"L1": set(), "L6": set()}, {}, {}
    for i in range(16):
        s, fold, label = f"P{i:02d}", i % 4, ("yes", "no")[(i // 4) % 2]
        folds[s], labels[s] = fold, label
        for c in (("L1", "L6") if i % 2 == 0 else (("L1",) if i % 4 == 1 else ("L6",))):
            holders[c].add(s)
            bags.append(_bag(tmp_path, s, c, label=label, fold=fold, seed=i))
    rotations = {str(r): {"test_fold": r, "validation_fold": (r + 1) % 4,
                          "train_folds": [(r + 2) % 4, (r + 3) % 4]} for r in range(4)}
    bs = bag_data.BagSet(bags=tuple(bags), partition_sha256="x" * 64, rotations=rotations)
    a = assignment.assign(assignment.Transfer("level", "L1", "L6"), holders, folds, labels,
                          rotations, 7, "x" * 64)
    return a, bs


def test_k2_reads_the_other_condition_only_through_the_probe(tmp_path, monkeypatch):
    a, bs = _assignment(tmp_path)
    calls = []
    real = assignment.probe_bag
    monkeypatch.setattr(mech.assignment_mod, "probe_bag",
                        lambda *args: calls.append(args[2]) or real(*args))
    pairs = mech.probe_pairs(a, bs, 0)
    assert [s for s, _, _ in pairs] == sorted(a.probe_subjects(0))
    assert sorted(calls) == sorted(a.probe_subjects(0))
    for s, target, source in pairs:
        assert (target.condition, source.condition) == ("L6", "L1")
        assert a.conditions[s] in (target.condition, source.condition)
    with pytest.raises(assignment.ProbeRefused):
        real(a, bs, "P01", 0)


def test_the_kernel_distance_is_the_rkhs_distance_at_the_frozen_sigma():
    model = _model(sigma=3.0)
    t = [torch.randn(4, CHANNELS, 512) for _ in range(3)]
    s = [torch.randn(5, CHANNELS, 512) for _ in range(3)]
    D = mech.kernel_distances(model, t, s)
    with torch.no_grad():
        pt = model.bag_pairs(model.embed(t, "target"))
        ps = model.bag_pairs(model.embed(s, "source"))
        k = bag_kernel_matrix(pt[1:2], pt[1:2], 3.0)[0, 0] + bag_kernel_matrix(
            ps[2:3], ps[2:3], 3.0)[0, 0] - 2 * bag_kernel_matrix(pt[1:2], ps[2:3], 3.0)[0, 0]
    assert D.shape == (3, 3) and D[1, 2] == pytest.approx(float(k.clamp_min(0).sqrt()))
    same = mech.kernel_distances(model, t, [x for x in t])
    assert np.all(same >= 0)
    E = mech.euclidean_distances(model, t, s)
    assert E.shape == (3, 3) and np.all(E >= 0)


def test_the_separation_is_others_minus_own():
    D = np.array([[1.0, 3.0, 5.0], [2.0, 0.5, 2.0], [1.0, 1.0, 4.0]])
    rows = mech.separation(D, ["a", "b", "c"])
    assert [r["separation"] for r in rows] == pytest.approx([4 - 1, 2 - 0.5, 1 - 4])
    with pytest.raises(mech.MechanismError):
        mech.separation(np.zeros((1, 1)), ["a"])


def test_k2_tests_the_two_declared_transfers_with_holm_over_four():
    assert {k: v.name for k, v in mech.K2_DECLARED.items()} == {
        "smaller shift": "L1->L6", "larger shift": "L6->Robots"}
    rows = [{"transfer": t, "subject": f"S{i}", "separation": 0.1 * (i + 1)}
            for t in ("L1->L6", "L6->Robots") for i in range(8)]
    out = mech.k2_tests(rows, None)
    assert len(out) == 4
    assert out[1]["p_value"] is None and "not trained" in out[1]["not_computed"]
    assert out[0]["p_adjusted"] == pytest.approx(min(1.0, 2 * out[0]["p_value"]))
    weaker = [{**r, "separation": r["separation"] / 2} for r in rows]
    full = mech.k2_tests(rows, weaker)
    assert all(r["p_value"] is not None for r in full)
    assert "m=4" in full[0]["correction"]


def test_the_bound_control_reads_every_recorded_step():
    history = [{"model": "MIL-CREDA:phase2", "global_min": g0, "global_max": g1,
                "local_min": l0, "local_max": l1, "global_bound_low": 0.0,
                "global_bound_high": 1.0, "local_bound_low": 0.0, "local_bound_high": 1.0}
               for g0, g1, l0, l1 in ((0.2, 0.4, 0.1, 0.3), (0.05, 0.6, 0.2, 1.2))]
    history.append({"model": "MIL-CREDA:phase1", "global_min": -5})
    rows = mech.bound_control(history)
    assert rows[0] == {"term": "global", "bound_low": 0.0, "bound_high": 1.0,
                       "observed_min": 0.05, "observed_max": 0.6, "within": True,
                       "epochs_read": 2}
    assert rows[1]["observed_max"] == 1.2 and rows[1]["within"] is False
    with pytest.raises(mech.NotRecorded):
        mech.bound_control([])


def test_a_stored_phase_two_model_is_rebuilt_with_every_parameter(tmp_path):
    doc = {"format": "tiny"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    ppath = tmp_path / "p.json"
    ppath.write_text(json.dumps(doc))
    model = _model(sigma=1.5)
    hyper = {"attentionGamma": 0.5, "attentionTemperature": 1.0}
    transfer = assignment.Transfer("level", "L1", "L11")
    state = {**model.state_dict(), "auxiliary.weight": torch.zeros(2)}
    path = model_store.save(tmp_path / "M", transfer=transfer, rotation=0, seed=1, phase=2,
                            name="MIL-CREDA", state=state, hyper=hyper,
                            partition_sha256=doc[partition.HASH_FIELD],
                            channels={"source": CHANNELS, "target": CHANNELS}, sigma=1.5)
    rebuilt, meta, extra = mech.load_phase_two(path, partition_path=ppath)
    assert extra == ["auxiliary.weight"] and rebuilt.sigma == 1.5
    for k, v in model.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[k], v)
    partial = {k: v for k, v in model.state_dict().items() if not k.startswith("head")}
    path2 = model_store.save(tmp_path / "M", transfer=transfer, rotation=1, seed=1, phase=2,
                             name="MIL-CREDA", state=partial, hyper=hyper,
                             partition_sha256=doc[partition.HASH_FIELD],
                             channels={"source": CHANNELS, "target": CHANNELS}, sigma=1.5)
    with pytest.raises(mech.MechanismError, match="lacks"):
        mech.load_phase_two(path2, partition_path=ppath)


def test_the_variant_trainers_refuse_an_unknown_variant():
    from MIL_CREDA_EEG import aggregation_variants, two_phase
    assert mech.ABLATIONS == ("global term only", "local term only", "both terms")
    assert mech.AGGREGATIONS == ("own", "published attention", "gated attention", "max",
                                 "mean")
    with pytest.raises(two_phase.UnknownTermSelection):
        mech.train_ablation_variant(None, 0, 1, None, {}, {}, "no local term")
    with pytest.raises(aggregation_variants.UnknownAggregation):
        mech.train_aggregation_variant(None, 0, 1, "own", {}, {}, {})
