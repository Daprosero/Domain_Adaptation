"""Plan v63 stage C/D/E analyses: pure rules, then an end-to-end read of a synthetic campaign."""

import json

import numpy as np
import pytest

from MIL_CREDA_EEG import bag_data, campaign, partition
from MIL_CREDA_EEG import v63_analysis as An
from MIL_CREDA_EEG import v63_analysis_attention as K1
from MIL_CREDA_EEG import v63_analysis_device as Dv
from MIL_CREDA_EEG import v63_analysis_matching as K2
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_connectivity as G
from MIL_CREDA_EEG import v63_saved as S
from MIL_CREDA_EEG import v63_setup as V


# ------------------------------------------------------------------ connectivity


def test_the_plv_matrix_is_symmetric_in_zero_one_with_a_zero_diagonal():
    x = np.random.default_rng(0).normal(size=(3, 4, 512))
    plv = G.window_plv(x)
    for m in plv.values():
        assert m.shape == (3, 4, 4) and np.allclose(m, m.transpose(0, 2, 1))
        assert (m >= 0).all() and (m <= 1 + 1e-9).all() and np.allclose(m[:, range(4), range(4)], 0)


def test_the_graph_is_the_weighted_mean_of_the_window_plv():
    plv = {"theta": np.stack([np.zeros((2, 2)), np.ones((2, 2))])}
    assert np.allclose(G.weighted_graph(plv, [1, 3])["theta"], 0.75)
    with pytest.raises(ValueError):
        G.weighted_graph(plv, [0, 0])


def test_the_descriptors_are_degree_centrality_and_spectral_radius_per_band():
    x = np.random.default_rng(1).normal(size=(5, 4, 512))
    d = G.subject_descriptors(x, np.ones(5))
    assert d.shape == (3 * (2 * 4 + 1),) and np.isfinite(d).all()


# ------------------------------------------------------------------ rules


def test_the_class_separation_leaves_the_subject_out_of_the_centres():
    desc = {"a": np.array([0.0]), "b": np.array([1.0]), "c": np.array([10.0]), "d": np.array([11.0])}
    sep = Dv.class_separation(desc, {"a": 0, "b": 0, "c": 1, "d": 1})
    assert sep["a"] == pytest.approx(10.5 - 1.0)
    assert np.isnan(Dv.class_separation({"a": desc["a"], "c": desc["c"]}, {"a": 0, "c": 1})["a"])


def test_the_predicted_class_reads_no_target_label():
    e = {"seeds": {1: ({"probabilities": [0.7], "labels": [0]}, 0),
                   2: ({"probabilities": [0.5], "labels": [0]}, 0)}}
    assert S.predicted_class(e) == 1


def test_a_marked_contrast_needs_marked_and_clean_windows():
    assert np.isnan(K1.contrast([0.5, 0.5], np.array([True, True])))
    assert np.isnan(K1.contrast([0.5, 0.5], None))
    assert K1.contrast([0.1, 0.9], np.array([True, False])) == pytest.approx(-0.8)
    w = {"integrity": [False, True], "contactBad": [True, False], "event": None}
    assert K1.window_flags(w, "marked").tolist() == [True, True]
    assert K1.window_flags(w, "event") is None


def test_the_deletion_never_removes_every_window():
    assert K1.removal_count(2, 0.25) == 1 and K1.removal_count(40, 0.10) == 4
    assert K1.removal_count(3, 0.9) == 2


def test_the_figure_subject_is_the_median_contrast():
    assert S.median_rule({"a": 3.0, "b": 1.0, "c": 2.0}) == "c"
    assert S.median_rule({"a": 1.0, "b": 2.0, "c": float("nan"), "d": 4.0, "e": 3.0}) == "b"
    assert S.median_rule({}) is None


def test_the_separation_is_mean_other_minus_own_distance():
    D = np.array([[1.0, 3.0, 5.0], [2.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
    assert K2.separation(D).tolist() == pytest.approx([3.0, 2.0, 0.0])


def test_the_reading_condition_is_entropy_below_0_9_or_margin_at_least_1():
    m = lambda e, g: {"entropyRatio": e, "marginRatio": g, "ownNearest": 1.0}
    assert K2.readable({"a": m(0.95, 0.5), "b": m(0.95, 0.4)})["readable"] is False
    assert K2.readable({"a": m(0.5, 0.5)})["readable"] is True
    assert K2.readable({"a": m(0.95, 1.0)})["readable"] is True


def test_the_permuted_correspondence_stays_within_the_class():
    pi = np.array([0.5, 0.3, 0.2])
    p = K2.permuted_within_class(pi, np.random.default_rng(0))
    assert sorted(p) == sorted(pi)


def test_the_best_rival_is_chosen_by_mean_pooled_balanced_accuracy(tmp_path, monkeypatch):
    good = {(0, 1): {"labels": [0, 1], "probabilities": [0.2, 0.8]}}
    bad = {(0, 1): {"labels": [0, 1], "probabilities": [0.8, 0.2]}}
    monkeypatch.setattr(S, "records", lambda p, t, m: good if m == "COT" else bad)
    assert Dv.best_rival(tmp_path, V.transfer("L6->Robots")) == "COT"


def test_holm_groups_join_only_strategies_holm_does_not_separate():
    ranking = {"average_rank": {"a": 1.0, "b": 2.0, "c": 3.0},
               "pairs": [{"a": "a", "b": "b", "reject": False}, {"a": "a", "b": "c", "reject": True},
                         {"a": "b", "b": "c", "reject": False}]}
    assert An.holm_groups(ranking) == [["a", "b"], ["b", "c"]]


# ------------------------------------------------------------------ end to end


def _bagset(root):
    rng = np.random.default_rng(0)
    bags = []
    for i in range(40):
        s, fold, label = f"UB{i:04d}", i % 4, ("yes", "no")[(i // 4) % 2]
        for c in {0: ("L6",), 1: ("Robots",), 2: ("L6", "Robots")}[i % 3]:
            n = 6
            marks = {"window_index": np.arange(n), "window_start_s": np.arange(n) * 4.0,
                     "integrity_mark": np.zeros(n, bool), "rail_window": np.zeros(n, bool)}
            if c == "Robots":
                cb = np.zeros((n, 12), bool); cb[0, 0] = True
                marks["contact_bad"] = cb
            else:
                marks["event_window"] = np.array([True, False, False, True, False, False])
            path = root / f"{s}_{c}.npz"
            sig = rng.normal(size=(n, 12, 512)) + (0.5 if c == "Robots" else 0.0)
            np.savez(path, signal=sig.astype(np.float32), **marks)
            bags.append(bag_data.Bag(subject=s, condition=c, record_id=f"{s}-{c}", label=label,
                                     sex="M", fold=fold, device="CGX", channels=V.SHARED,
                                     n_windows=n, path=path))
    rot = {str(r): partition.rotation_roles(r) for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256="x" * 64, rotations=rot)


class _Cov:
    def of(self, bags):
        return {"ages": [10.0] * len(bags), "frontalAmplitude": [1.0] * len(bags)}


@pytest.fixture(scope="module")
def campaign_product(tmp_path_factory):
    root = tmp_path_factory.mktemp("v63mech")
    bags = _bagset(root)
    cfg = C.check_config({"schedule": {"epochs1": 2, "epochs2": 2, "bagsPerStep": 4}, "nTrials": 1,
                          "seeds": [1, 2], "strategies": ["MIL-CREDA", "AdaptMIL", "UDA-DDA"],
                          "permutations": 10, "threads": 1, "marginSeconds": 0, "ensvRefits": False})
    campaign.register(cfg["strategies"])
    product = root / "P"
    with V.installed():
        for r in (0, 1):
            C.run_performance(V.transfer("L6->Robots"), r, cfg, product, bags, V.Prepared(bags),
                              _Cov(), None)
    return product, bags


def test_the_gsa_comparisons_are_paired_wilcoxon_with_holm(campaign_product):
    product, bags = campaign_product
    with V.installed():
        out = Dv.analyze(product, bags, figures=True)
    rows = json.loads((product / "Results" / "significance_v63_gsa.json").read_text())
    assert out["rows"] == len(rows) == 2
    assert all(r["test"].startswith("Wilcoxon") and r["correction"].startswith("holm") for r in rows)
    assert {r["versus"] for r in rows} == {"predict", rows[0]["bestRival"]}
    assert any(p.endswith("umap_L6-to-Robots.pdf") for p in out["figures"])


def test_t1_t2_t3_share_one_holm_family(campaign_product):
    product, bags = campaign_product
    with V.installed():
        out = K1.analyze(product, bags, figures=False)
    rows = json.loads((product / "Results" / "significance_v63_K1.json").read_text())
    crit = [r for r in rows if r["family"] == "K1 criterion"]
    assert [r["test_id"] for r in crit] == ["T1", "T2", "T3"]
    assert all("m=" in r["correction"] for r in crit)
    dev = [r for r in rows if r["family"] == "K1 references" and r.get("arm") == "device"]
    assert dev and all(r["eligibleSubjects"] > 0 for r in dev)
    dele = [r for r in rows if r["family"] == "K1 deletion"]
    assert {(r["q"], r["versus"]) for r in dele} == {(q, v) for q in K1.FRACTIONS
                                                     for v in ("random", "adaptmil")}


def test_the_contrast_is_averaged_over_the_seeds_per_subject(campaign_product):
    product, _ = campaign_product
    c = K1.subject_contrasts(product, S.MIL_CREDA, ("L6->Robots",), "marked")
    recs = S.by_subject(S.records(product, V.transfer("L6->Robots"), S.MIL_CREDA))
    key = next(iter(c))
    e = recs[key[1]]
    vals = [K1.contrast(r["attention"][k], K1.window_flags(r["windows"][k], "marked"))
            for r, k in e["seeds"].values()]
    assert c[key] == pytest.approx(float(np.nanmean(vals))) and len(e["seeds"]) == 2


def test_the_four_k2_tests_share_one_holm_family(campaign_product):
    product, bags = campaign_product
    with V.installed():
        out = K2.analyze(product, bags, figures=True, comparators=True)
    rows = json.loads((product / "Results" / "significance_v63_K2_K4.json").read_text())
    crit = [r for r in rows if r["family"] == "K2 criterion"]
    assert len(crit) == 2 and all(r["transfer"] == "L6->Robots" for r in crit)
    assert all("m=2" in r["correction"] for r in crit)      # Robots->L6 holds no records here
    diag = out["diagnostics"][0]
    assert {"medianEntropyRatio", "medianMarginRatio", "readable"} <= set(diag)
    assert any(r["family"] == "K4 part 3, support" for r in rows)
    assert any(r["family"] == "K2 comparators, no criterion" for r in rows)
    assert out["figure"] and out["figure"].endswith(".pdf")


def test_the_displacement_figure_subject_is_the_median_push(campaign_product):
    product, bags = campaign_product
    store = S.Store(product, bags)
    with V.installed():
        m = K2.probe_measures(store, V.transfer("L6->Robots"))
    pick = S.median_rule({("L6->Robots", s): v["pushMIL-CREDA"] for s, v in m.items()})
    assert pick is not None and pick[1] in m


def test_a_record_set_holds_each_target_subject_once_per_seed(campaign_product):
    product, _ = campaign_product
    subjects = S.by_subject(S.records(product, V.transfer("L6->Robots"), S.MIL_CREDA))
    assert subjects and all(sorted(e["seeds"]) == [1, 2] for e in subjects.values())
