"""Experiments plan v42, Secs. 6.1-6.3: the mechanism analyses over stored models.

Synthetic bags and untrained models only: no number here is a measured result.
"""

from __future__ import annotations

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


def test_the_event_contrast_reads_only_the_slackline_levels_of_the_level_arm(tmp_path):
    assert mech.EVENT_CONDITIONS == ("L1", "L6", "L11")
    assert mech.K1_CONTRASTS == {"level": ("contact", "event"), "task": ("contact",),
                                 "device": ("contact",)}
    for condition in ("CogniFit", "Robots"):
        with pytest.raises(mech.MechanismError, match="event stream"):
            mech.event_windows(_bag(tmp_path, "S1", condition))
    events = {"integrity_mark": np.array([True, False, False, False]),
              "event_window": np.array([0, 1, 0, 0], bool)}
    # the task and device arms run only the contact contrast, even with a Slackline target
    for transfer in (assignment.Transfer("task", "Robots", "CogniFit"),
                     assignment.Transfer("device", "Robots", "L6")):
        bag = _bag(tmp_path, "S2", transfer.target, windows=4, marks=events)
        rows = mech.k1_subject_rows(_model(), [bag], model_name="M", transfer=transfer,
                                    rotation=0, seed=1)
        assert rows[0]["arm"] == transfer.arm and rows[0]["marked_minus_clean"] is not None
        assert rows[0]["event_minus_rest"] is None and rows[0]["event_not_recorded"] is None
    level = mech.k1_subject_rows(_model(), [_bag(tmp_path, "S3", "L6", windows=4,
                                                 marks=events)], model_name="M",
                                 transfer=assignment.Transfer("level", "L1", "L6"), rotation=0,
                                 seed=1)
    assert level[0]["event_minus_rest"] is not None
    assert "weight_ratio_spearman" not in level[0]
    assert not hasattr(mech, "band_ratio") and "bands" not in str(
        __import__("inspect").signature(mech.k1_subject_rows))


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
                                seed=1)
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


def _k1_rows(model, arm, transfer, *, shift=0.0, n=8):
    return [{"model": model, "arm": arm, "transfer": transfer, "pair": "x", "seed": 1,
             "rotation": i % 4, "subject": f"{transfer}:S{i}",
             "marked_minus_clean": -0.1 * (i + 1) + shift,
             "event_minus_rest": 0.05 * (i + 1) if arm == "level" else None,
             "event_not_recorded": None} for i in range(n)]


def test_k1_reads_one_report_seed_and_holms_the_three_level_criterion_tests():
    without = "MIL-CREDA [own without consensus]"
    assert mech.K1_WITHOUT_CONSENSUS == without
    assert set(mech.K1_EXCLUDED) == {"mean"} and "uniform" in mech.K1_EXCLUDED["mean"]
    by_model = {
        "MIL-CREDA": _k1_rows("MIL-CREDA", "level", "L1->L11")
        + _k1_rows("MIL-CREDA", "task", "Robots->CogniFit"),
        without: _k1_rows(without, "level", "L1->L11", shift=0.05)
        + _k1_rows(without, "task", "Robots->CogniFit", shift=0.05),
        "MIL-CREDA [published attention]": _k1_rows("MIL-CREDA [published attention]",
                                                    "level", "L1->L11", shift=0.1)}
    table = mech.k1_tests(by_model)
    level = [r for r in table if r["comparison"].startswith("K1-atencion:level:")]
    criterion = [r for r in level if r["criterion"] == mech.K1_CRITERION]
    assert [r["comparison"] for r in criterion] == [
        "K1-atencion:level:marked vs clean, MIL-CREDA",
        f"K1-atencion:level:marked vs clean, MIL-CREDA vs {without}",
        "K1-atencion:level:event vs rest, MIL-CREDA"]
    assert st.holm([r["p_value"] for r in criterion]) == [r["p_adjusted"] for r in criterion]
    assert all("m=3" in r["correction"] for r in criterion)
    task = [r for r in table if r["comparison"].startswith("K1-atencion:task:")]
    assert [r["comparison"] for r in task] == [
        "K1-atencion:task:marked vs clean, MIL-CREDA",
        f"K1-atencion:task:marked vs clean, MIL-CREDA vs {without}"]
    assert all(r["p_adjusted"] is None and r["criterion"].startswith("none") for r in task)
    references = [r for r in table if "published attention" in r["comparison"]]
    assert [r["comparison"] for r in references] == [
        "K1-atencion:level:marked vs clean, MIL-CREDA [published attention]",
        "K1-atencion:level:event vs rest, MIL-CREDA [published attention]"]
    assert all(r["p_adjusted"] is None and r["criterion"].startswith("none")
               for r in references)
    assert all(r["experiment"] == "K1-atencion" for r in table)
    # the paired contrast is MIL-CREDA minus the variant without consensus, per subject
    paired = criterion[1]
    assert paired["n"] == 8 and paired["median"] == pytest.approx(-0.05)
    # one report seed: a subject twice for one model is refused, never averaged
    twice = {**by_model, "MIL-CREDA": by_model["MIL-CREDA"] + by_model["MIL-CREDA"][:1]}
    with pytest.raises(mech.MechanismError, match="once"):
        mech.k1_tests(twice)
    assert not hasattr(mech, "average_over_seeds") and not hasattr(mech, "K1_SCOPES")
    # without the variant's model the paired contrast is not computed, and says why
    alone = mech.k1_tests({"MIL-CREDA": by_model["MIL-CREDA"]})
    row = next(r for r in alone if r["comparison"].endswith(f"vs {without}"))
    assert row["p_value"] is None and "not stored" in row["not_computed"]


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
    assert mech.K2_NO_LOCAL == "MIL-CREDA [global term only]"
    rows = [{"model": "MIL-CREDA", "transfer": t, "subject": f"S{i}",
             "separation": 0.1 * (i + 1)} for t in ("L1->L6", "L6->Robots") for i in range(8)]
    out = mech.k2_tests(rows, None)
    assert len(out) == 4
    assert out[1]["p_value"] is None and "not trained" in out[1]["not_computed"]
    assert out[0]["p_adjusted"] == pytest.approx(min(1.0, 2 * out[0]["p_value"]))
    weaker = [{**r, "model": mech.K2_NO_LOCAL, "separation": r["separation"] / 2}
              for r in rows]
    full = mech.k2_tests(rows, weaker)
    assert all(r["p_value"] is not None for r in full)
    assert "m=4" in full[0]["correction"]
    assert full[1]["comparison"] == ("K2-emparejamiento:L1->L6 (smaller shift):MIL-CREDA vs "
                                     "MIL-CREDA [global term only]")
    assert full[1]["median"] == pytest.approx(np.median([0.05 * (i + 1) for i in range(8)]))
    # a declared transfer without the variant is not computed, and says so
    half = mech.k2_tests(rows, [r for r in weaker if r["transfer"] == "L1->L6"])
    assert half[1]["p_value"] is not None
    assert half[3]["p_value"] is None and "not trained" in half[3]["not_computed"]
    with pytest.raises(mech.MechanismError, match="once"):
        mech.k2_tests(rows + rows[:1], weaker)


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


def test_a_phase_two_model_of_another_plan_revision_is_refused_on_load(tmp_path):
    doc = {"format": "tiny"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    ppath = tmp_path / "p.json"
    ppath.write_text(json.dumps(doc))
    model = _model(sigma=1.5)
    path = model_store.save(tmp_path / "M", transfer=assignment.Transfer("level", "L1", "L11"),
                            rotation=0, seed=1, phase=2, name="MIL-CREDA",
                            state=model.state_dict(),
                            hyper={"attentionGamma": 0.5, "attentionTemperature": 1.0},
                            partition_sha256=doc[partition.HASH_FIELD],
                            channels={"source": CHANNELS, "target": CHANNELS}, sigma=1.5)
    assert mech.load_phase_two(path, partition_path=ppath)[1]["revision"] == \
        model_store.__provenance__["revision"]
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["revision"] = "experiments-plan-experimental-mil-creda-sobre-balladeer-v43.md"
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(model_store.RevisionMismatch, match="v43"):
        mech.load_phase_two(path, partition_path=ppath)
    with pytest.raises(model_store.RevisionMismatch):
        mech.load_phase_two(path, partition_path=ppath, allow_code_mismatch=True)


def test_the_variant_trainers_refuse_an_unknown_variant():
    from MIL_CREDA_EEG import aggregation_variants, two_phase
    assert mech.ABLATIONS == ("global term only", "local term only", "both terms")
    assert mech.AGGREGATIONS == ("own", "own without consensus", "published attention",
                                 "gated attention", "mean")
    with pytest.raises(two_phase.UnknownTermSelection):
        mech.train_ablation_variant(None, 0, 1, None, {}, {}, "no local term")
    with pytest.raises(aggregation_variants.UnknownAggregation):
        mech.train_aggregation_variant(None, 0, 1, "own", {}, {}, {})
    with pytest.raises(aggregation_variants.UnknownAggregation):
        mech.train_aggregation_variant(None, 0, 1, "max", {}, {}, {})


def test_a_stored_k3_variant_model_is_rebuilt_for_its_attention(tmp_path):
    from MIL_CREDA_EEG import aggregation_variants as av
    doc = {"format": "tiny"}
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    ppath = tmp_path / "p.json"
    ppath.write_text(json.dumps(doc))
    transfer = assignment.Transfer("level", "L1", "L11")
    hyper = av.variant_hyper("own without consensus",
                             {"attentionGamma": 0.5, "attentionTemperature": 0.8})
    torch.manual_seed(0)
    model = av.build("own without consensus", CHANNELS, CHANNELS, hyper)
    model.fix_sigma(1.5)
    path = model_store.save(tmp_path / "M", transfer=transfer, rotation=0, seed=1, phase=2,
                            name=mech.K1_WITHOUT_CONSENSUS, state=model.state_dict(),
                            hyper=hyper, partition_sha256=doc[partition.HASH_FIELD],
                            channels={"source": CHANNELS, "target": CHANNELS}, sigma=1.5)
    rebuilt, meta, extra = mech.load_phase_two(path, partition_path=ppath)
    assert type(rebuilt) is MILCREDAModel and rebuilt.attention_gamma == 0.0
    assert rebuilt.attention_temperature == 0.8 and rebuilt.sigma == 1.5 and extra == []
    x = torch.randn(5, CHANNELS, 512)
    assert np.allclose(mech.attention_weights(rebuilt, x), mech.attention_weights(model, x))


# ------------------------------------------------------------------ plan v44


def test_the_provenance_binds_plan_v50():
    assert mech.__provenance__["revision"] == (
        "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md")


def test_every_k1_and_k2_wilcoxon_row_carries_hodges_lehmann_and_its_interval():
    table = mech.k1_tests({"MIL-CREDA": _k1_rows("MIL-CREDA", "level", "L1->L11")})
    rows = [{"model": "MIL-CREDA", "transfer": t, "subject": f"S{i}",
             "separation": 0.1 * (i + 1)} for t in ("L1->L6", "L6->Robots") for i in range(8)]
    weaker = [{**r, "model": mech.K2_NO_LOCAL, "separation": r["separation"] / 2}
              for r in rows]
    k2 = mech.k2_tests(rows, weaker)
    tested = [r for r in table + k2 if r["n"]]
    assert len(tested) >= 2 + 4 and all(r["test"] == st.TEST_WILCOXON for r in tested)
    for r in tested:
        assert r["ci_low"] <= r["hodges_lehmann"] <= r["ci_high"]
        assert r["ci_name"] == "Hodges-Lehmann, exact Wilcoxon signed-rank 95% interval"
    # Sec. 6.2's paired rows: MIL-CREDA minus its global-term-only variant, half the separation
    paired = st.hodges_lehmann([0.05 * (i + 1) for i in range(8)])
    assert k2[1]["hodges_lehmann"] == pytest.approx(paired["estimate"])
    assert (k2[1]["ci_low"], k2[1]["ci_high"]) == (pytest.approx(paired["ci_low"]),
                                                   pytest.approx(paired["ci_high"]))
    row = table[0]
    values = [-0.1 * (i + 1) for i in range(8)]
    hl = st.hodges_lehmann(values)
    assert row["hodges_lehmann"] == pytest.approx(hl["estimate"])
    assert (row["ci_low"], row["ci_high"]) == (pytest.approx(hl["ci_low"]),
                                               pytest.approx(hl["ci_high"]))
    assert row["effect_size"] == pytest.approx(-1.0)          # the rank-biserial stays
    out = mech.k1_test_table(table)
    assert list(out[0]) == list(mech.K1_TABLE_COLUMNS) == [
        "Contraste", "Estadístico", "Valor de p",
        "Tamaño del efecto, correlación biserial de rangos", "Estimador de Hodges-Lehmann",
        "Intervalo de confianza al 95 % de Wilcoxon"]
    assert out[0]["Estimador de Hodges-Lehmann"] == pytest.approx(hl["estimate"])


def test_the_k1_and_k2_test_tables_carry_the_plans_headers():
    k1 = mech.k1_test_table(mech.k1_tests(
        {"MIL-CREDA": _k1_rows("MIL-CREDA", "level", "L1->L11")}))
    assert list(mech.K1_TABLE_COLUMNS) == [
        "Contraste", "Estadístico", "Valor de p",
        "Tamaño del efecto, correlación biserial de rangos", "Estimador de Hodges-Lehmann",
        "Intervalo de confianza al 95 % de Wilcoxon"]
    assert k1 and all(list(r) == list(mech.K1_TABLE_COLUMNS) for r in k1)
    rows = [{"model": "MIL-CREDA", "transfer": t, "subject": f"S{i}",
             "separation": 0.1 * (i + 1)} for t in ("L1->L6", "L6->Robots") for i in range(8)]
    weaker = [{**r, "model": mech.K2_NO_LOCAL, "separation": r["separation"] / 2}
              for r in rows]
    tests = mech.k2_tests(rows, weaker)
    hl = st.hodges_lehmann([0.1 * (i + 1) for i in range(8)])
    assert tests[0]["hodges_lehmann"] == pytest.approx(hl["estimate"])
    assert (tests[0]["ci_low"], tests[0]["ci_high"]) == (pytest.approx(hl["ci_low"]),
                                                         pytest.approx(hl["ci_high"]))
    table = mech.k2_test_table(tests)
    assert list(mech.K2_TABLE_COLUMNS) == [
        "Prueba de Wilcoxon", "Estadístico", "Valor de p crudo", "Valor de p ajustado por Holm",
        "Tamaño del efecto, correlación biserial de rangos", "Estimador de Hodges-Lehmann",
        "Intervalo de confianza al 95 % de Wilcoxon", "Sujetos de la sonda"]
    assert [r["Prueba de Wilcoxon"] for r in table] == [
        "Corrimiento menor: separación de MIL-CREDA mayor que cero",
        "Corrimiento mayor: separación de MIL-CREDA mayor que cero",
        "Corrimiento menor: MIL-CREDA contra sin término local",
        "Corrimiento mayor: MIL-CREDA contra sin término local"]     # Sec. 6.2's row order
    assert table[2]["Valor de p crudo"] == tests[1]["p_value"]
    assert all(list(r) == list(mech.K2_TABLE_COLUMNS) for r in table)
    assert table[0]["Valor de p ajustado por Holm"] == tests[0]["p_adjusted"]
    assert table[0]["Sujetos de la sonda"] == 8
    assert table[0]["Intervalo de confianza al 95 % de Wilcoxon"] == (
        f"[{hl['ci_low']:.4f}, {hl['ci_high']:.4f}]")
