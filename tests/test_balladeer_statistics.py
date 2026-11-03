"""Experiments plan v42, Secs. 4.4 and 4.6: metrics, summaries and contrasts.

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
            out.append({"strategy": strategy, "seed": seed, "rotation": i % rotations,
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
    assert m == st.subject_metrics([x["label"] for x in r], [x["probability"] for x in r])
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
    assert st.delong(y, p, p)["p_value"] == 1.0


def test_delong_matches_its_structural_components_definition():
    rng = np.random.default_rng(4)
    y = np.array([0] * 9 + [1] * 11)
    a = rng.normal(size=20) + y
    b = rng.normal(size=20) + 0.5 * y
    a[3] = a[12]                                              # a tie across classes
    out = st.delong(y, a, b)

    def psi(x, z):
        return 1.0 if x > z else 0.5 if x == z else 0.0

    def components(s):
        pos, neg = s[y == 1], s[y == 0]
        v10 = np.array([np.mean([psi(x, z) for z in neg]) for x in pos])
        v01 = np.array([np.mean([psi(x, z) for x in pos]) for z in neg])
        return v10, v01

    (a10, a01), (b10, b01) = components(a), components(b)
    m, n = 11, 9
    var = (np.var(a10, ddof=1) + np.var(b10, ddof=1) - 2 * np.cov(a10, b10)[0, 1]) / m \
        + (np.var(a01, ddof=1) + np.var(b01, ddof=1) - 2 * np.cov(a01, b01)[0, 1]) / n
    assert out["auc_a"] == pytest.approx(roc_auc_score(y, a))
    assert out["auc_b"] == pytest.approx(roc_auc_score(y, b))
    assert out["variance"] == pytest.approx(var)
    z = (out["auc_a"] - out["auc_b"]) / math.sqrt(var)
    assert out["p_value"] == pytest.approx(2 * sps.norm.sf(abs(z)))


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
    assert [r["test"] for r in rows] == [st.TEST_BOOTSTRAP, st.TEST_DELONG,
                                         st.TEST_NADEAU_BENGIO]
    for r in rows:
        assert set(st.SIGNIFICANCE_KEYS) | set(st.OPTIONAL_KEYS) <= set(r)
        assert list(r)[:len(st.SIGNIFICANCE_KEYS)] == list(st.SIGNIFICANCE_KEYS)
        assert r["experiment"] == "A-nivel"
        assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in r.values())
    assert "discordant" in rows[0] and "kappa_difference" in rows[0]
