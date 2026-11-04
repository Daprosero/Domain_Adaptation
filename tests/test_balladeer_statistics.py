"""Experiments plan v50, Secs. 4.4 and 4.6: metrics, summaries and contrasts.

Every number here is synthetic: seeded random predictions, never a measured result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from MIL_CREDA_EEG import statistics as st


def _records(n=24, seeds=(1,), rotations=4, direction="A->B", strategy="M", rng=None,
             skill=0.3):
    rng = rng or np.random.default_rng(0)
    labels = np.array([i % 2 for i in range(n)])
    out = []
    for seed in seeds:
        noise = rng.normal(scale=0.3, size=n)
        p = np.clip(0.5 + skill * (labels - 0.5) * 2 * 0.5 + noise * 0.5, 0.01, 0.99)
        for i in range(n):
            # both classes in every rotation, as the class-stratified partition gives
            out.append({"strategy": strategy, "seed": seed, "rotation": (i // 2) % rotations,
                        "subject": f"S{i:03d}", "direction": direction,
                        "probability": float(p[i]), "label": int(labels[i])})
    return out


def test_metrics_are_subject_level_and_balanced_accuracy_is_the_mean_recall():
    labels = [1, 1, 1, 0]
    probs = [0.9, 0.8, 0.2, 0.7]
    m = st.subject_metrics(labels, probs)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == 0.0
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 0) / 2)
    assert m["auc"] == pytest.approx(roc_auc_score(labels, probs))
    assert m["n"] == 4 and m["n_positive"] == 3
    # a tie at 0.5 goes to class 0, as the argmax of the two-class prediction
    assert st.predicted_labels([0.5, 0.5000001]).tolist() == [0, 1]
    # kappa: always-majority scores zero
    assert st.subject_metrics([1, 1, 1, 0], [0.9] * 4)["kappa"] == pytest.approx(0.0)
    rng = np.random.default_rng(9)
    y, p = rng.integers(0, 2, 30), rng.uniform(size=30)
    assert st.subject_metrics(y, p)["kappa"] == pytest.approx(
        cohen_kappa_score(y, st.predicted_labels(p)))


def test_a_subject_predicted_twice_in_one_seed_is_refused():
    r = _records(n=4, seeds=(1,))
    with pytest.raises(st.StatisticsError, match="more than once"):
        st.check_once(r + r[:1])
    st.check_once(r + [{**r[0], "seed": 2}])


def test_the_pooled_summary_is_one_value_from_the_one_report_seed():
    r = _records()
    m = st.pooled(r)
    assert m == st.subject_metrics([x["label"] for x in r], [x["probability"] for x in r],
                                   [st.cell_of(x) for x in r])
    # Sec. 4.6: one report seed, no reduction between seeds; a second seed is refused
    with pytest.raises(st.StatisticsError, match="one report seed"):
        st.pooled(_records(seeds=(1, 2)))
    assert not hasattr(st, "pooled_by_seed") and not hasattr(st, "summarize_seeds")


def test_the_per_fold_summary_is_mean_and_sd_over_the_rotations():
    r = _records()
    f = st.per_fold(r)
    assert sorted(f["rotations"]) == [0, 1, 2, 3]
    expected = st.subject_metrics([x["label"] for x in r if x["rotation"] == 2],
                                  [x["probability"] for x in r if x["rotation"] == 2])
    assert f["rotations"][2]["balanced_accuracy"] == pytest.approx(
        expected["balanced_accuracy"])
    values = [f["rotations"][k]["balanced_accuracy"] for k in range(4)]
    assert f["balanced_accuracy"]["mean"] == pytest.approx(np.mean(values))
    assert f["balanced_accuracy"]["sd"] == pytest.approx(np.std(values, ddof=1))
    with pytest.raises(st.StatisticsError, match="one report seed"):
        st.per_fold(_records(seeds=(1, 2)))


def test_a_pair_pools_each_subject_from_the_direction_where_it_is_target():
    a = _records(n=4, direction="A->B")[:4]
    b = [{**x, "subject": f"T{i}", "direction": "B->A"} for i, x in enumerate(a)]
    pooled = {**{x["subject"]: "A->B" for x in a}, **{x["subject"]: "B->A" for x in b}}
    assert len(st.pool_pair(a + b, pooled)) == 8
    wrong = [{**b[0], "direction": "A->B"}] + b[1:]
    with pytest.raises(st.StatisticsError, match="direction"):
        st.pool_pair(a + wrong, pooled)
    with pytest.raises(st.StatisticsError, match="without a prediction"):
        st.pool_pair(a + b[1:], pooled)


def test_the_contrasts_read_one_report_seed_and_no_reduction():
    import inspect
    assert "reduction" not in inspect.signature(st.pair_contrast).parameters
    assert not hasattr(st, "SEED_REDUCTIONS") and not hasattr(st, "reduce_seeds")
    a = _records(n=8, strategy="A")
    b = _records(n=8, strategy="B", rng=np.random.default_rng(1))
    with pytest.raises(st.StatisticsError, match="one report seed"):
        st.pair_contrast(_records(n=8, seeds=(1, 2)), b, name_a="A", name_b="B", scope="s",
                         test_train_ratio=0.3, hypothesis="h", experiment="e", resamples=20)
    with pytest.raises(st.StatisticsError, match="same report seed"):
        st.pair_contrast(a, _records(n=8, seeds=(2,)), name_a="A", name_b="B", scope="s",
                         test_train_ratio=0.3, hypothesis="h", experiment="e", resamples=20)
    rows = st.pair_contrast(a, b, name_a="A", name_b="B", scope="s", test_train_ratio=0.3,
                            hypothesis="h", experiment="e", resamples=20)
    assert rows[0]["report_seed"] == 1 and "seed_reduction" not in rows[0]
    assert "[seed" not in rows[0]["comparison"]


def test_the_bootstrap_is_paired_and_keeps_every_stratum_size():
    rng = np.random.default_rng(1)
    y = np.array([0] * 7 + [1] * 13)
    strata = [(int(v), "A->B" if i % 3 else "B->A") for i, v in enumerate(y)]
    pa, pb = rng.uniform(size=20), rng.uniform(size=20)
    out = st.paired_bootstrap(y, pa, pb, strata, resamples=500, seed=3)
    groups = st._strata_index(strata)
    assert out["strata_sizes"] == [len(g) for g in groups]
    # the replicate index is drawn inside each stratum, so class counts never change:
    rng2 = np.random.default_rng(3)
    index = np.concatenate([g[rng2.integers(0, len(g), size=(500, len(g)))] for g in groups],
                           axis=1)
    assert np.all((y[index] == 1).sum(axis=1) == 13)
    ba_a, _ = st._ba_kappa(y[index], st.predicted_labels(pa)[index])
    ba_b, _ = st._ba_kappa(y[index], st.predicted_labels(pb)[index])
    assert np.allclose(out["replicates"], ba_a - ba_b)        # same indices for both models


def test_the_bootstrap_reports_difference_interval_kappa_and_discordant_subjects():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    pa = [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6]           # perfect
    pb = [0.6, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.4]           # two errors
    out = st.paired_bootstrap(y, pa, pb, [v for v in y], resamples=2000, seed=0)
    assert out["difference"] == pytest.approx(1.0 - 0.75)
    assert out["discordant"] == 2
    assert out["kappa_difference"] == pytest.approx(1.0 - 0.5)
    assert out["ci_low"] <= out["difference"] <= out["ci_high"]
    assert 0 < out["p_value"] <= 1


def test_identical_models_give_a_zero_difference_and_p_one():
    rng = np.random.default_rng(2)
    y = [0, 1] * 10
    p = rng.uniform(size=20)
    out = st.paired_bootstrap(y, p, p, y, resamples=300, seed=0)
    assert out["difference"] == 0 and out["discordant"] == 0 and out["p_value"] == 1.0
    assert out["auc_difference"] == 0 and out["auc_p_value"] == 1.0


def test_nadeau_bengio_reduces_to_the_one_sample_t_at_zero_ratio():
    d = [0.05, 0.02, 0.08, -0.01]
    plain = st.nadeau_bengio(d, 0.0)
    ref = sps.ttest_1samp(d, 0.0)
    assert plain["t"] == pytest.approx(ref.statistic) and plain["p_value"] == pytest.approx(
        ref.pvalue)
    corrected = st.nadeau_bengio(d, 0.25)
    assert corrected["se"] == pytest.approx(math.sqrt((1 / 4 + 0.25) * np.var(d, ddof=1)))
    assert corrected["p_value"] > plain["p_value"] and corrected["df"] == 3


def test_holm_is_the_step_down_adjustment():
    assert st.holm([0.01, 0.04, 0.03, None]) == pytest.approx([0.03, 0.06, 0.06, None])
    assert st.holm([0.5, 0.6]) == [1.0, 1.0]
    rows = [{"p_value": 0.01}, {"p_value": 0.02}]
    st.apply_holm(rows, "A-nivel")
    assert rows[0]["p_adjusted"] == pytest.approx(0.02) and "holm" in rows[0]["correction"]


def test_friedman_ranks_with_one_as_best_and_nemenyi_gives_the_critical_difference():
    scores = {"M": [0.9, 0.8], "X": [0.7, 0.75], "Y": [0.5, 0.6], "Z": [0.6, 0.6]}
    out = st.friedman_nemenyi(scores)
    assert out["average_rank"]["M"] == 1.0
    assert out["average_rank"]["Y"] == pytest.approx((4 + 3.5) / 2)   # ties share ranks
    assert out["position"]["M"] == 1
    assert out["critical_difference"] == pytest.approx(2.569 * math.sqrt(4 * 5 / 12))
    ref = sps.friedmanchisquare(*scores.values())
    assert out["statistic"] == pytest.approx(ref.statistic)


def test_wilcoxon_reports_the_rank_biserial_effect():
    d = [0.3, 0.2, 0.1, -0.05, 0.4, 0.0]
    out = st.wilcoxon_signed_rank(d)
    ref = sps.wilcoxon(d, zero_method="wilcox")
    assert out["p_value"] == pytest.approx(ref.pvalue)
    # ranks of |0.3,0.2,0.1,0.05,0.4| = 4,3,2,1,5 -> (14 - 1) / 15
    assert out["rank_biserial"] == pytest.approx(13 / 15)
    assert out["n"] == 6 and out["n_nonzero"] == 5
    assert st.wilcoxon_signed_rank([0, 0])["p_value"] == 1.0


def test_significance_rows_carry_exactly_the_dashboard_keys():
    a = _records(strategy="MIL-CREDA", rng=np.random.default_rng(5))
    b = _records(strategy="X", rng=np.random.default_rng(6), skill=0.1)
    rows = st.pair_contrast(a, b, name_a="MIL-CREDA", name_b="X", scope="L1|L11",
                            test_train_ratio=0.3,
                            hypothesis="A-nivel", experiment="A-nivel", resamples=200)
    assert [r["test"] for r in rows] == [st.TEST_BOOTSTRAP, st.TEST_KAPPA, st.TEST_AUC_AVG,
                                         st.TEST_NADEAU_BENGIO]
    for r in rows:
        assert set(st.SIGNIFICANCE_KEYS) | set(st.OPTIONAL_KEYS) <= set(r)
        assert list(r)[:len(st.SIGNIFICANCE_KEYS)] == list(st.SIGNIFICANCE_KEYS)
        assert r["experiment"] == "A-nivel"
        assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in r.values())
    assert "discordant" in rows[0] and "kappa_difference" in rows[0]


# ------------------------------------------------------------------ plan v50


def test_the_provenance_binds_plan_v50():
    assert st.__provenance__["revision"] == (
        "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md")


def test_hodges_lehmann_and_its_exact_wilcoxon_interval_on_one_to_eight():
    # Walsh averages of 1..8: (i + j) / 2 for i <= j, 36 of them, symmetric about 4.5.
    out = st.hodges_lehmann([1, 2, 3, 4, 5, 6, 7, 8])
    assert out["estimate"] == pytest.approx(4.5)
    # n = 8: P(T+ <= 3) = 5/256 <= 0.025 < P(T+ <= 4) = 7/256, so k = 4 and the
    # interval is the 4th and the 33rd Walsh averages, (1+3)/2 and (8+6)/2.
    assert out["k"] == 4 and out["walsh"] == 36
    assert (out["ci_low"], out["ci_high"]) == (pytest.approx(2.0), pytest.approx(7.0))
    assert out["coverage"] == pytest.approx(1 - 2 * 5 / 256)


def test_hodges_lehmann_is_the_walsh_median_and_too_few_subjects_give_no_interval():
    out = st.hodges_lehmann([1, 2, 10])      # Walsh: 1, 1.5, 2, 5.5, 6, 10
    assert out["estimate"] == pytest.approx((2 + 5.5) / 2)
    assert out["ci_low"] is None and out["ci_high"] is None and "n = 3" in out["note"]
    rng = np.random.default_rng(4)
    d = rng.normal(0.2, 1, size=11)
    walsh = sorted((d[i] + d[j]) / 2 for i in range(11) for j in range(i, 11))
    got = st.hodges_lehmann(d)
    assert got["estimate"] == pytest.approx(np.median(walsh))
    assert got["ci_low"] == pytest.approx(walsh[got["k"] - 1])
    assert got["ci_high"] == pytest.approx(walsh[len(walsh) - got["k"]])
    # n = 11 at 95%: the signed-rank lower critical value is 10, so k = 11.
    assert got["k"] == 11


def test_wilcoxon_reports_hodges_lehmann_beside_the_rank_biserial():
    d = [0.3, 0.2, 0.1, -0.05, 0.4, 0.0, 0.25, 0.15]
    out = st.wilcoxon_signed_rank(d)
    hl = st.hodges_lehmann(d)
    assert out["hodges_lehmann"] == pytest.approx(hl["estimate"])
    assert (out["ci_low"], out["ci_high"]) == (hl["ci_low"], hl["ci_high"])
    assert out["rank_biserial"] == st.wilcoxon_signed_rank(d)["rank_biserial"]


def test_the_kappa_p_value_equals_a_direct_recomputation():
    rng = np.random.default_rng(7)
    y = np.array([i % 2 for i in range(30)])
    pa = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.25, 30), 0, 1)
    pb = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.25, 30), 0, 1)
    strata = [(int(v), "d1" if i < 15 else "d2") for i, v in enumerate(y)]
    out = st.paired_bootstrap(y, pa, pb, strata, resamples=3000, seed=11)
    groups = st._strata_index(strata)
    rng2 = np.random.default_rng(11)
    index = np.concatenate([g[rng2.integers(0, len(g), size=(3000, len(g)))]
                            for g in groups], axis=1)
    ka = np.array([cohen_kappa_score(y[i], st.predicted_labels(pa)[i]) for i in index])
    kb = np.array([cohen_kappa_score(y[i], st.predicted_labels(pb)[i]) for i in index])
    kdiff = ka - kb
    lower = (1 + np.sum(kdiff <= 0)) / 3001
    upper = (1 + np.sum(kdiff >= 0)) / 3001
    assert out["kappa_p_value"] == pytest.approx(min(1.0, 2 * min(lower, upper)))
    # the balanced-accuracy p value is unchanged by the kappa one
    diff = out["replicates"]
    lo, hi = (1 + np.sum(diff <= 0)) / 3001, (1 + np.sum(diff >= 0)) / 3001
    assert out["p_value"] == pytest.approx(min(1.0, 2 * min(lo, hi)))


def test_pair_contrast_emits_the_primary_and_the_three_secondary_rows():
    a = _records(strategy="MIL-CREDA", rng=np.random.default_rng(5))
    b = _records(strategy="X", rng=np.random.default_rng(6), skill=0.1)
    rows = st.pair_contrast(a, b, name_a="MIL-CREDA", name_b="X", scope="L1|L11",
                            test_train_ratio=0.3, hypothesis="A-nivel", experiment="A-nivel",
                            resamples=200)
    assert [r["test"] for r in rows] == [st.TEST_BOOTSTRAP, st.TEST_KAPPA, st.TEST_AUC_AVG,
                                         st.TEST_NADEAU_BENGIO]
    assert [r["family"] for r in rows] == [st.FAMILY_PRIMARY] + [st.FAMILY_SECONDARY] * 3
    boot = st.paired_bootstrap(
        [r["label"] for r in sorted(a, key=lambda r: r["subject"])],
        [r["probability"] for r in sorted(a, key=lambda r: r["subject"])],
        [r["probability"] for r in sorted(b, key=lambda r: r["subject"])],
        [(r["label"], r["direction"])
         for r in sorted(a, key=lambda r: r["subject"])],
        cells=[st.cell_of(r) for r in sorted(a, key=lambda r: r["subject"])],
        resamples=200)
    kappa = rows[1]
    assert kappa["p_value"] == pytest.approx(boot["kappa_p_value"])
    assert kappa["effect_size"] == pytest.approx(boot["kappa_difference"])
    assert (kappa["ci_low"], kappa["ci_high"]) == (pytest.approx(boot["kappa_ci_low"]),
                                                   pytest.approx(boot["kappa_ci_high"]))
    assert len({r["contrast"] for r in rows}) == 1
    # the balanced-accuracy row reads the same replicates, unchanged by the AUC_avg one
    assert rows[0]["p_value"] == pytest.approx(boot["p_value"])
    assert rows[0]["effect_size"] == pytest.approx(boot["difference"])
    assert rows[0]["strata"] == "class x direction"
    assert rows[2]["strata"] == "class x rotation-and-direction cell"


def test_holm_runs_within_each_metric_and_no_row_stays_uncorrected():
    rows = []
    for k, rival in enumerate(("X", "Y", "Z")):
        rows += st.pair_contrast(
            _records(strategy="MIL-CREDA", rng=np.random.default_rng(5)),
            _records(strategy=rival, rng=np.random.default_rng(10 + k), skill=0.05 * k),
            name_a="MIL-CREDA", name_b=rival, scope="L1|L11", test_train_ratio=0.3,
            hypothesis="A-nivel", experiment="A-nivel", resamples=200)
    primary_only = [dict(r) for r in rows if r["test"] == st.TEST_BOOTSTRAP]
    st.apply_holm(primary_only, "A-nivel MIL-CREDA vs rivals, bootstrap")
    st.apply_contrast_families(rows, "A-nivel MIL-CREDA vs rivals")
    assert all(r["correction"] != "none" for r in rows)
    for test in (st.TEST_BOOTSTRAP, st.TEST_KAPPA, st.TEST_AUC_AVG, st.TEST_NADEAU_BENGIO):
        family = [r for r in rows if r["test"] == test]
        assert len(family) == 3
        assert [r["p_adjusted"] for r in family] == pytest.approx(
            st.holm([r["p_value"] for r in family]))
        assert all("m=3" in r["correction"] for r in family)
    # the primary family is the one the success criteria read, unchanged
    assert [r["p_adjusted"] for r in rows if r["test"] == st.TEST_BOOTSTRAP] == [
        r["p_adjusted"] for r in primary_only]


def test_nemenyi_critical_difference_for_nine_strategies_and_ten_blocks():
    # Demsar 2006, Table 5(a): q_0.05 = 3.102 for k = 9; CD = q sqrt(k(k+1) / 6N).
    rng = np.random.default_rng(1)
    scores = {f"S{i}": list(rng.uniform(0.5, 0.9, 10)) for i in range(9)}
    out = st.friedman_nemenyi(scores)
    assert out["q_alpha"] == 3.102
    assert out["critical_difference"] == pytest.approx(3.102 * math.sqrt(1.5))
    assert out["critical_difference"] == pytest.approx(3.799, abs=5e-4)


def test_the_arm_ranking_is_descriptive_with_no_test():
    scores = {"MIL-CREDA": [0.9, 0.8], "X": [0.7, 0.75], "Y": [0.5, 0.6], "Z": [0.6, 0.6]}
    out = st.descriptive_ranking(scores)
    assert set(out) == {"average_rank", "position", "top_three", "blocks", "k", "test"}
    assert out["test"] is None
    assert out["average_rank"] == st.friedman_nemenyi(scores)["average_rank"]
    assert out["position"]["MIL-CREDA"] == 1 and out["top_three"]["MIL-CREDA"] is True
    assert out["position"]["Z"] == 3 and out["top_three"]["Z"] is True      # the boundary
    assert out["top_three"]["Y"] is False
    # two strategies are enough: nothing is tested
    assert st.descriptive_ranking({"A": [0.5], "B": [0.6]})["position"] == {"A": 2, "B": 1}


def test_the_global_ranking_joins_strategies_whose_rank_gap_does_not_exceed_the_cd():
    ranks = {"A": 1.0, "B": 2.0, "C": 4.5, "D": 5.0}
    assert st.nemenyi_groups(ranks, 3.5) == [["A", "B", "C"], ["B", "C", "D"]]
    assert st.nemenyi_groups(ranks, 0.5) == [["C", "D"]]
    assert st.nemenyi_groups(ranks, 0.1) == []


def test_the_balanced_accuracy_and_kappa_contrast_without_cells_is_unchanged():
    # touched behavior: paired_bootstrap called as before (no cells) gives the same BA
    # and kappa replicates as a direct recomputation on the same index
    rng = np.random.default_rng(12)
    y = np.array([i % 2 for i in range(20)])
    pa, pb = rng.uniform(size=20), rng.uniform(size=20)
    strata = [(int(v), "d") for v in y]
    out = st.paired_bootstrap(y, pa, pb, strata, resamples=300, seed=5)
    groups = st._strata_index(strata)
    rng2 = np.random.default_rng(5)
    index = np.concatenate([g[rng2.integers(0, len(g), size=(300, len(g)))] for g in groups],
                           axis=1)
    ba_a, k_a = st._ba_kappa(y[index], st.predicted_labels(pa)[index])
    ba_b, k_b = st._ba_kappa(y[index], st.predicted_labels(pb)[index])
    assert np.allclose(out["replicates"], ba_a - ba_b)
    assert out["kappa_p_value"] == pytest.approx(st._bootstrap_p((k_a - k_b)[
        np.isfinite(k_a - k_b)]))
    assert out["auc_a"] == pytest.approx(roc_auc_score(y, pa)) and out["cells"] == 1
