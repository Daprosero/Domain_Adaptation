"""Plan v63: metrics and declared tests over the per-(method, seed) records."""

import json

import numpy as np
import pytest

from MIL_CREDA_EEG import statistics
from MIL_CREDA_EEG import v63_analysis as An


def _data(n=40, seeds=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    cells = (np.arange(n) // (n // 4)) % 4
    good = np.stack([np.clip(y * 0.6 + 0.2 + rng.normal(0, .05, n), 0, 1) for _ in range(seeds)])
    rand = np.stack([rng.uniform(size=n) for _ in range(seeds)])
    return y, cells, good, rand


def test_the_bootstrap_pairs_both_methods_on_the_same_resampled_subjects_in_every_seed():
    y, cells, good, rand = _data()
    same = An.multi_seed_bootstrap(y, cells, good, good, resamples=200)
    assert same["ba_difference"] == 0 and same["ba_ci_low"] == same["ba_ci_high"] == 0
    better = An.multi_seed_bootstrap(y, cells, good, rand, resamples=200)
    assert better["ba_difference"] > 0 and better["ba_ci_low"] > 0
    assert better["seeds"] == 3 and better["n"] == 40


def test_the_bootstrap_strata_are_the_classes():
    y = np.array([0] * 5 + [1] * 30)
    groups = An._groups(y.tolist())
    idx = An._index(groups, np.random.default_rng(0), 50)
    assert all((y[row] == 0).sum() == 5 for row in idx)


def test_non_inferiority_is_one_sided_against_minus_the_margin():
    y, cells, good, _ = _data()
    r = An.multi_seed_bootstrap(y, cells, good, good, resamples=300)
    assert r["ni_p_value"] < 0.05 and r["margin"] == 0.03
    worse = good.copy()
    worse[:, y == 1] -= 0.5
    r2 = An.multi_seed_bootstrap(y, cells, np.clip(worse, 0, 1), good, resamples=300)
    assert r2["ni_p_value"] > 0.05


def test_holm_runs_over_the_transfers_of_one_family():
    rows = [{"p_value": p} for p in (0.01, 0.04, 0.03)]
    statistics.apply_holm(rows, "f")
    assert [r["p_adjusted"] for r in rows] == pytest.approx([0.03, 0.06, 0.06])


def test_the_friedman_post_hoc_is_holm_over_the_pairs():
    scores = {"a": [0.9, 0.8, 0.85, 0.9], "b": [0.6, 0.5, 0.55, 0.6], "c": [0.7, 0.7, 0.7, 0.71]}
    out = An.friedman_holm(scores)
    assert len(out["pairs"]) == 3 and out["average_rank"]["a"] == 1.0
    adj = statistics.holm([p["p_value"] for p in out["pairs"]])
    assert [p["p_adjusted"] for p in out["pairs"]] == adj


def test_the_age_adjusted_auc_is_the_auc_of_the_residuals():
    ages = np.linspace(6, 14, 30)
    res = An.age_residual(0.1 * ages + 0.3, ages)
    assert np.allclose(res, 0, atol=1e-12)


def _record(path, transfer, method, seed, rotation, y, p):
    n = len(y)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "transfer": transfer, "method": method, "seed": seed, "rotation": rotation,
        "subjects": [f"{transfer}-{rotation}-{i}" for i in range(n)], "labels": y,
        "probabilities": p, "ages": list(np.linspace(6, 14, n)),
        "frontalAmplitude": list(np.linspace(10, 30, n)),
        "targetRepresentations": [[float(v), 1.0] for v in p],
        "sourceRepresentations": [[float(v) + 2.0, 0.0] for v in p]}))


def test_a_number_in_a_table_is_read_from_a_record(tmp_path):
    product = tmp_path / "P"
    root = product / "Results" / "v63"
    (root / "S0-codificador").mkdir(parents=True)
    for d in ("L6", "Robots"):
        (root / "S0-codificador" / f"target_filter_{d}.json").write_text(json.dumps(
            {"auc": 0.8, "p_value": 0.01, "subjects": 40, "passes": True}))
    (root / "F-corrimiento").mkdir()
    (root / "F-corrimiento" / "shift_L6-to-Robots.json").write_text(json.dumps(
        {"transfer": "L6->Robots", "auc": 0.9, "p_value": 0.001, "classification": "shift",
         "gateByRotation": [0.8] * 4}))
    rng = np.random.default_rng(0)
    for method, good in (("MIL-CREDA", True), ("predict", False), ("COT", False)):
        for seed in (1, 2):
            for r in range(4):
                y = [0, 1] * 5
                p = [0.2 + 0.6 * v for v in y] if good else list(rng.uniform(size=10))
                _record(root / "C-dispositivo" / "runs" / "L6-to-Robots" / f"outer{r}"
                        / f"{method}__seed{seed}.json", "L6->Robots", method, seed, r, y, p)
    summary = An.analyze(product, resamples=100)
    assert summary["transfersWithResults"] == ["L6->Robots"]
    sig = json.loads((product / "Results" / "significance_v63.json").read_text())
    keys = {"comparison", "test", "statistic", "p_value", "p_adjusted", "correction",
            "effect_size", "effect_size_name", "ci_low", "ci_high", "n", "alpha"}
    assert sig and all(keys <= set(r) for r in sig)
    sup = [r for r in sig if r.get("family", "").startswith("superiority")]
    assert len(sup) == 1 and sup[0]["statistic"] > 0
    table = (root / "C-dispositivo" / "tables" / "performance.csv").read_text().splitlines()
    assert len(table) == 1 + 3 and "balanced_accuracy" in table[0]
