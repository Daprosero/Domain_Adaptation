"""Experiments plan v50, Sec. 4.6: AUC_avg per rotation-and-direction cell and its contrast.

Every number here is synthetic: seeded random predictions, never a measured result.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from MIL_CREDA_EEG import results
from MIL_CREDA_EEG import statistics as st


def _miscalibrated(strategy="M", n_per_cell=12, rng_seed=0, separation=1.0):
    """Two directions x four rotations; each rotation's model is shifted by its own
    offset, so every cell ranks well but the pooled probabilities mix calibrations."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for direction in ("A->B", "B->A"):
        for rotation in range(4):
            offset = 0.15 * rotation - 0.2
            for i in range(n_per_cell):
                label = i % 2
                score = separation * (label - 0.5) + rng.normal(0, 0.4)
                p = float(1 / (1 + np.exp(-(score * 2)))) * 0.6 + 0.2 + offset
                out.append({"strategy": strategy, "seed": 1, "rotation": rotation,
                            "direction": direction, "label": label,
                            "subject": f"{direction}:{rotation}:{i}",
                            "probability": float(np.clip(p, 0.001, 0.999))})
    return out


def _cell_aucs(records):
    cells = {}
    for r in records:
        cells.setdefault((r["rotation"], r["direction"]), []).append(r)
    return {c: roc_auc_score([r["label"] for r in rs], [r["probability"] for r in rs])
            for c, rs in cells.items()}


def test_auc_avg_is_the_mean_of_the_cell_aucs_and_differs_from_the_pooled_auc():
    records = _miscalibrated()
    expected = np.mean(list(_cell_aucs(records).values()))
    pooled_auc = roc_auc_score([r["label"] for r in records],
                               [r["probability"] for r in records])
    summary = st.pooled(records)
    assert summary["auc"] == pytest.approx(expected)
    assert abs(summary["auc"] - pooled_auc) > 0.02        # miscalibration penalizes pooling
    assert summary["auc_cells"] == {f"rotation {c[0]}, {c[1]}": pytest.approx(v)
                                    for c, v in _cell_aucs(records).items()}
    # without cells, one cell: the plain AUC
    labels = [r["label"] for r in records]
    probs = [r["probability"] for r in records]
    assert st.subject_metrics(labels, probs)["auc"] == pytest.approx(pooled_auc)
    assert st.auc_avg(labels, probs, [(r["rotation"], r["direction"]) for r in records]
                      )["auc"] == pytest.approx(expected)


def test_balanced_accuracy_and_kappa_stay_on_pooled_hard_predictions():
    records = _miscalibrated()
    labels = [r["label"] for r in records]
    probs = [r["probability"] for r in records]
    plain = st.subject_metrics(labels, probs)
    summary = st.pooled(records)
    for metric in ("balanced_accuracy", "kappa", "sensitivity", "specificity"):
        assert summary[metric] == pytest.approx(plain[metric])


def test_the_auc_contrast_is_the_cell_bootstrap_of_auc_avg_on_the_ba_replicates():
    a = _miscalibrated("MIL-CREDA", rng_seed=1, separation=1.4)
    b = _miscalibrated("X", rng_seed=2, separation=0.6)
    rows = st.pair_contrast(a, b, name_a="MIL-CREDA", name_b="X", scope="A|B",
                            test_train_ratio=0.3, hypothesis="A-nivel", experiment="A-nivel",
                            resamples=400)
    assert [r["test"] for r in rows] == [st.TEST_BOOTSTRAP, st.TEST_KAPPA, st.TEST_AUC_AVG,
                                         st.TEST_NADEAU_BENGIO]
    auc_row = rows[2]
    ca, cb = _cell_aucs(a), _cell_aucs(b)
    assert auc_row["auc_a"] == pytest.approx(np.mean(list(ca.values())))
    assert auc_row["effect_size"] == pytest.approx(np.mean(list(ca.values()))
                                                   - np.mean(list(cb.values())))
    assert auc_row["family"] == st.FAMILY_SECONDARY
    assert auc_row["cells"] == 8 and auc_row["resamples"] == 400

    # recompute: the same strata (class x rotation x direction) and the same index as BA
    ra = {r["subject"]: r for r in a}
    rb = {r["subject"]: r for r in b}
    subjects = sorted(ra)
    y = np.array([ra[s]["label"] for s in subjects])
    pa = np.array([ra[s]["probability"] for s in subjects])
    pb = np.array([rb[s]["probability"] for s in subjects])
    cells = [(ra[s]["rotation"], ra[s]["direction"]) for s in subjects]
    strata = [(ra[s]["label"], ra[s]["direction"], ra[s]["rotation"]) for s in subjects]
    boot = st.paired_bootstrap(y, pa, pb, strata, cells=cells, resamples=400)
    assert auc_row["p_value"] == pytest.approx(boot["auc_p_value"])
    groups = st._strata_index(strata)
    rng = np.random.default_rng(st.BOOTSTRAP_SEED)
    index = np.concatenate([g[rng.integers(0, len(g), size=(400, len(g)))] for g in groups],
                           axis=1)
    ba_a, _ = st._ba_kappa(y[index], st.predicted_labels(pa)[index])
    ba_b, _ = st._ba_kappa(y[index], st.predicted_labels(pb)[index])
    assert np.allclose(boot["replicates"], ba_a - ba_b)
    cell_of = np.array([hash(c) for c in cells])
    for k in (0, 17, 399):
        row = index[k]
        diffs = []
        for c in set(cells):
            cols = row[cell_of[row] == hash(c)]
            diffs.append(roc_auc_score(y[cols], pa[cols]) - roc_auc_score(y[cols], pb[cols]))
        assert boot["auc_replicates"][k] == pytest.approx(np.mean(diffs))
    lo, hi = np.percentile(boot["auc_replicates"], [2.5, 97.5])
    assert (auc_row["ci_low"], auc_row["ci_high"]) == (pytest.approx(lo), pytest.approx(hi))


def test_a_stratum_spanning_two_cells_is_refused():
    y = [0, 1, 0, 1]
    with pytest.raises(st.StatisticsError, match="cell"):
        st.paired_bootstrap(y, [0.2, 0.8, 0.3, 0.7], [0.4, 0.6, 0.5, 0.5], y,
                            cells=["c1", "c1", "c2", "c2"], resamples=10)


def test_the_cell_auc_handles_ties_as_midranks():
    y = np.array([[0, 1, 0, 1, 1]])
    s = np.array([[0.3, 0.3, 0.1, 0.9, 0.5]])
    assert st._auc_rows(y, s)[0] == pytest.approx(roc_auc_score(y[0], s[0]))
    assert sps.rankdata(s[0])[0] == 2.5


def test_no_delong_test_remains():
    assert not hasattr(st, "delong") and not hasattr(st, "TEST_DELONG")
    assert all("DeLong" not in t for t in st.CONTRAST_TESTS)
    assert all("DeLong" not in c for c in results.SIGNIFICANCE_TABLE_COLUMNS)
    assert "DeLong" not in (st.__doc__ or "") and "DeLong" not in (results.__doc__ or "")
