"""Plan v63: the metrics and the declared tests of the performance experiments.

Secs. 4.4, 4.6, 4.7, 5.4 and 0 (part 1). Reads only the per-(method, seed) records the
units leave under `Results/v63/` (`v63_campaign`) and writes, in the product folder:

- `Results/v63/S0-codificador/tables/target_filter.csv` and
  `Results/v63/F-corrimiento/tables/shift_filter.csv`: one row per domain / transfer;
- `Results/v63/<experiment>/tables/performance.csv`: per transfer and method, the mean and
  standard deviation over the report seeds of the balanced accuracy on the pooled hard
  predictions (primary), the AUC averaged over the rotation cells, the AUC of the residuals
  of the score after a linear regression on age (averaged over the cells the same way), the
  Spearman rho of the score with age and with frontal amplitude, and the AUC of a domain
  classifier on the bag representations of the source and target test subjects;
- `Results/significance_v63.json`: the confirmatory family (MIL-CREDA minus predict:
  superiority in balanced accuracy on the "shift" transfers, Holm over them; non-inferiority
  in AUC with margin 0.03 on the "no-shift" transfers, one-sided, Holm over them), the
  supporting family (MIL-CREDA minus each rival, Holm over the transfers within each
  rival), and the Friedman test over the nine adapted strategies with Holm post hoc;
- `Results/v63/summary.json`: the headline per experiment, every number traced to a table.

The bootstrap is subject-level, paired and stratified by class (Sec. 4.6): each replicate
resamples subjects and recomputes, for both methods and on the same subjects, the metric in
every seed, then averages over the seeds. The AUC replicate resamples within each rotation
cell and class. Nothing here decides a verdict: it reports the tests the protocol declares.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from MIL_CREDA_EEG import model_store, statistics
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["0", "4.4", "4.6", "4.7", "5.4"],
    "experiments": ["S0-codificador", "F-corrimiento", "A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "the_bootstrap_pairs_both_methods_on_the_same_resampled_subjects_in_every_seed",
        "the_bootstrap_strata_are_the_classes",
        "non_inferiority_is_one_sided_against_minus_the_margin",
        "holm_runs_over_the_transfers_of_one_family",
        "the_friedman_post_hoc_is_holm_over_the_pairs",
        "the_age_adjusted_auc_is_the_auc_of_the_residuals",
        "a_number_in_a_table_is_read_from_a_record",
    ],
}

MARGIN = 0.03
ALPHA = 0.05
RESAMPLES = statistics.BOOTSTRAP_RESAMPLES
BOOT_SEED = statistics.BOOTSTRAP_SEED
PREDICT = "predict"
REFERENCE = "MIL-CREDA"
DOMAIN_C = 0.1


# ------------------------------------------------------------------ loading


#: The refits whose training diverged, as their records name them (`v63_campaign`).
FAILED: list[dict] = []


def load_records(product: Path) -> dict:
    """(transfer, method) -> seed -> pooled record over the four rotations."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    FAILED.clear()
    for exp in V.EXPERIMENT_OF_ARM.values():
        for path in sorted((product / "Results" / "v63" / exp / "runs").glob("*/outer*/*.json")):
            r = json.loads(path.read_text())
            if r.get("failed"):
                FAILED.append({k: r.get(k) for k in ("transfer", "method", "seed", "rotation",
                                                     "failed")})
                continue
            cell = out[(r["transfer"], r["method"])][int(r["seed"])]
            n = len(r["subjects"])
            cell["subjects"] += r["subjects"]
            cell["labels"] += r["labels"]
            cell["probabilities"] += r["probabilities"]
            cell["cells"] += [int(r["rotation"])] * n
            cell["ages"] += r["ages"]
            cell["frontal"] += r["frontalAmplitude"]
            cell["target"] += r["targetRepresentations"]
            cell["source"] += r["sourceRepresentations"]
            cell["rotations"].append(int(r["rotation"]))
    return out


def aligned(seeds: Mapping) -> dict:
    """Seed records sorted by subject, checked to hold the same subjects in every seed."""
    out = {}
    order = None
    for seed in sorted(seeds):
        c = seeds[seed]
        idx = np.argsort(c["subjects"], kind="stable")
        subjects = [c["subjects"][i] for i in idx]
        if len(set(subjects)) != len(subjects):
            raise V.V63Error("a target subject is predicted twice in one seed")
        if order is None:
            order = subjects
        elif subjects != order:
            raise V.V63Error("the seeds of one method do not hold the same subjects")
        out[seed] = {k: (np.asarray(c[k])[idx] if k not in ("source", "rotations") else c[k])
                     for k in c}
    return out


# ------------------------------------------------------------------ metrics


def ba(y, p) -> float:
    return float(statistics._ba_kappa(np.asarray(y)[None], statistics.predicted_labels(p)[None])[0][0])


def age_residual(scores, ages) -> np.ndarray:
    s, a = np.asarray(scores, float), np.asarray(ages, float)
    ok = np.isfinite(a)
    X = np.c_[np.ones(ok.sum()), a[ok]]
    beta = np.linalg.lstsq(X, s[ok], rcond=None)[0]
    out = np.full(len(s), np.nan)
    out[ok] = s[ok] - X @ beta
    return out


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or len(set(x[ok])) < 2 or len(set(y[ok])) < 2:
        return float("nan")
    return float(sps.spearmanr(x[ok], y[ok]).statistic)


def domain_auc(source: Sequence, target: Sequence, seed: int = 0) -> float:
    Zs, Zt = np.asarray(source, float), np.asarray(target, float)
    X = np.r_[Zs, Zt]
    d = np.r_[np.zeros(len(Zs)), np.ones(len(Zt))]
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=DOMAIN_C, max_iter=2000)),
                          X, d, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(statistics._auc_rows(d[None].astype(int), p[None])[0])


def seed_metrics(c: Mapping) -> dict:
    y, p, cells = c["labels"], c["probabilities"], c["cells"]
    res = age_residual(p, c["ages"])
    ok = np.isfinite(res)
    return {"balanced_accuracy": ba(y, p),
            "auc": statistics.auc_avg(y, p, list(cells))["auc"],
            "auc_age_removed": statistics.auc_avg(np.asarray(y)[ok], res[ok],
                                                  list(np.asarray(cells)[ok]))["auc"],
            "rho_age": spearman(p, c["ages"]), "rho_frontal": spearman(p, c["frontal"]),
            "domain_auc": domain_auc(c["source"], c["target"])}


def summarize(per_seed: Mapping[int, Mapping]) -> dict:
    keys = next(iter(per_seed.values())).keys()
    out = {}
    for k in keys:
        v = np.asarray([per_seed[s][k] for s in per_seed], float)
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if len(v) else None
        out[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
    return out


# ------------------------------------------------------------------ tests


def _index(groups: Sequence[np.ndarray], rng, R: int) -> np.ndarray:
    return np.concatenate([g[rng.integers(0, len(g), size=(R, len(g)))] for g in groups], axis=1)


def _groups(keys: Sequence) -> list[np.ndarray]:
    by: dict = defaultdict(list)
    for i, k in enumerate(keys):
        by[k].append(i)
    return [np.asarray(v) for _, v in sorted(by.items(), key=lambda kv: str(kv[0]))]


def multi_seed_bootstrap(y, cells, A: np.ndarray, B: np.ndarray, *, resamples: int = RESAMPLES,
                         seed: int = BOOT_SEED, margin: float = MARGIN) -> dict:
    """A minus B; `A`, `B` (seeds, subjects) probabilities on the same subjects."""
    y, cells = np.asarray(y, int), np.asarray(cells)
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.shape != B.shape or A.shape[1] != len(y):
        raise V.V63Error("both methods hold the same seeds and subjects")
    idx = _index(_groups(y.tolist()), np.random.default_rng(seed), resamples)
    hard = lambda M: statistics.predicted_labels(M.ravel()).reshape(M.shape)
    HA, HB = hard(A), hard(B)
    ba_of = lambda H, ix: np.mean([statistics._ba_kappa(y[ix], H[k][ix])[0] for k in range(len(H))], 0)
    diff = ba_of(HA, idx) - ba_of(HB, idx)
    observed = float(ba_of(HA, np.arange(len(y))[None])[0] - ba_of(HB, np.arange(len(y))[None])[0])
    aidx = _index(_groups(list(zip(y.tolist(), cells.tolist()))), np.random.default_rng(seed),
                  resamples)
    col_cell = cells[aidx[0]]

    def auc_of(M, ix):
        per_seed = []
        for k in range(len(M)):
            per_cell = [statistics._auc_rows(y[ix[:, col_cell == c]], M[k][ix[:, col_cell == c]])
                        for c in np.unique(cells)]
            per_seed.append(np.nanmean(np.stack(per_cell), 0))
        return np.mean(per_seed, 0)

    adiff = auc_of(A, aidx) - auc_of(B, aidx)
    obs_auc = lambda M: float(np.mean([statistics.auc_avg(y, M[k], list(cells))["auc"]
                                       for k in range(len(M))]))
    aobs = obs_auc(A) - obs_auc(B)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    alo, ahi = np.nanpercentile(adiff, [2.5, 97.5])
    fin = adiff[np.isfinite(adiff)]
    return {"ba_difference": observed, "ba_ci_low": float(lo), "ba_ci_high": float(hi),
            "ba_p_value": statistics._bootstrap_p(diff),
            "auc_difference": aobs, "auc_ci_low": float(alo), "auc_ci_high": float(ahi),
            "auc_p_value": statistics._bootstrap_p(fin),
            "ni_p_value": float((1 + np.sum(fin <= -margin)) / (1 + len(fin))),
            "ni_lower_95": float(np.percentile(fin, 5)), "ni_replicates": fin,
            "discordant": int(np.mean([(HA[k] != HB[k]).sum() for k in range(len(HA))])),
            "n": int(len(y)), "seeds": int(len(A)), "resamples": int(resamples),
            "margin": margin}


def friedman_holm(scores: Mapping[str, Sequence[float]], alpha: float = ALPHA) -> dict:
    """Friedman over blocks with Holm on every pair of average ranks (Demsar 2006)."""
    names = list(scores)
    T = np.asarray([scores[k] for k in names], float)
    k, N = T.shape
    ranks = np.vstack([sps.rankdata(-T[:, j]) for j in range(N)]).T
    R = ranks.mean(1)
    stat, p = sps.friedmanchisquare(*T) if k >= 3 and N >= 2 else (float("nan"), float("nan"))
    se = math.sqrt(k * (k + 1) / (6.0 * N))
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    z = [abs(R[names.index(a)] - R[names.index(b)]) / se for a, b in pairs]
    pv = [float(2 * sps.norm.sf(v)) for v in z]
    adj = statistics.holm(pv)
    return {"statistic": float(stat), "p_value": float(p), "k": k, "blocks": N,
            "average_rank": dict(zip(names, map(float, R))),
            "pairs": [{"a": a, "b": b, "z": float(zz), "p_value": pp, "p_adjusted": pa,
                       "reject": bool(pa is not None and pa < alpha)}
                      for (a, b), zz, pp, pa in zip(pairs, z, pv, adj)]}


# ------------------------------------------------------------------ the report


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def _read(paths) -> dict:
    return {p.stem.split("_", 2)[-1]: json.loads(p.read_text()) for p in sorted(paths)}


def analyze(product: str | Path, *, resamples: int = RESAMPLES) -> dict:
    product = Path(product)
    root = product / "Results" / "v63"
    targets = {k: v for k, v in _read((root / "S0-codificador").glob("target_filter_*.json")).items()}
    shifts = {json.loads(p.read_text())["transfer"]: json.loads(p.read_text())
              for p in sorted((root / "F-corrimiento").glob("shift_*.json"))}
    _write_csv(root / "S0-codificador" / "tables" / "target_filter.csv",
               [{"domain": d, "auc": v["auc"], "p_value": v["p_value"], "subjects": v["subjects"],
                 "passes": v["passes"]} for d, v in targets.items()])
    _write_csv(root / "F-corrimiento" / "tables" / "shift_filter.csv",
               [{"transfer": t, "auc": v["auc"], "p_value": v["p_value"],
                 "classification": v["classification"],
                 "gate_by_rotation": json.dumps(v["gateByRotation"])} for t, v in shifts.items()])
    records = load_records(product)
    perf_rows, metrics = [], {}
    for (t, method), seeds in sorted(records.items()):
        if any(len(set(c["rotations"])) != 4 for c in seeds.values()):
            continue                                   # a transfer whose four folds are not in
        al = aligned(seeds)
        per = {s: seed_metrics(c) for s, c in al.items()}
        metrics[(t, method)] = (al, summarize(per))
        perf_rows.append({"experiment": V.experiment_of(V.transfer(t)), "transfer": t,
                          "method": method, "seeds": len(per), "n": len(al[min(al)]["labels"]),
                          **metrics[(t, method)][1]})
    for exp in V.EXPERIMENT_OF_ARM.values():
        rows = [r for r in perf_rows if r["experiment"] == exp]
        if rows:
            _write_csv(root / exp / "tables" / "performance.csv", rows)
    sig = []
    passing = [t for t in V.TRANSFERS if targets.get(t.target, {}).get("passes")]
    fam = {"shift": [], "no-shift": []}
    for t in passing:
        if (t.name, REFERENCE) not in metrics or (t.name, PREDICT) not in metrics:
            continue
        cls = shifts.get(t.name, {}).get("classification")
        a, b = metrics[(t.name, REFERENCE)][0], metrics[(t.name, PREDICT)][0]
        s0 = min(a)
        res = multi_seed_bootstrap(a[s0]["labels"], a[s0]["cells"],
                                   np.stack([a[s]["probabilities"] for s in sorted(a)]),
                                   np.stack([b[s]["probabilities"] for s in sorted(b)]),
                                   resamples=resamples)
        exp = V.experiment_of(t)
        if cls in fam:
            fam[cls].append((t, exp, res))
    for t, exp, res in fam["shift"]:
        sig.append(statistics.significance_row(
            comparison=f"MIL-CREDA vs predict, BA, {t.name}", test="paired bootstrap by subject, "
            "stratified by class, mean over seeds", statistic=res["ba_difference"],
            p_value=res["ba_p_value"], effect_size=res["ba_difference"],
            effect_size_name="balanced accuracy difference", ci_low=res["ba_ci_low"],
            ci_high=res["ba_ci_high"], n=res["n"], hypothesis="F-corrimiento", experiment=exp,
            family="superiority (shift transfers)", transfer=t.name, discordant=res["discordant"]))
    statistics.apply_holm([r for r in sig if r.get("family", "").startswith("superiority")],
                          "superiority over the shift transfers")
    ni = []
    for t, exp, res in fam["no-shift"]:
        ni.append(statistics.significance_row(
            comparison=f"MIL-CREDA vs predict, AUC non-inferiority, {t.name}",
            test="one-sided paired bootstrap of the AUC difference against -0.03",
            statistic=res["auc_difference"], p_value=res["ni_p_value"],
            effect_size=res["auc_difference"], effect_size_name="AUC difference",
            ci_low=res["ni_lower_95"], ci_high=None, n=res["n"], hypothesis="F-corrimiento",
            experiment=exp, family="non-inferiority (no-shift transfers)", transfer=t.name,
            margin=MARGIN))
    statistics.apply_holm(ni, "non-inferiority over the no-shift transfers")
    sig += ni
    rivals = sorted({m for (t, m) in metrics if m not in (REFERENCE, PREDICT)
                     and not m.startswith("phase1") and "[EnsV]" not in m})
    for rival in rivals:
        rows = []
        for t in passing:
            if (t.name, rival) not in metrics or (t.name, REFERENCE) not in metrics:
                continue
            a, b = metrics[(t.name, REFERENCE)][0], metrics[(t.name, rival)][0]
            s0 = min(a)
            res = multi_seed_bootstrap(a[s0]["labels"], a[s0]["cells"],
                                       np.stack([a[s]["probabilities"] for s in sorted(a)]),
                                       np.stack([b[s]["probabilities"] for s in sorted(b)]),
                                       resamples=resamples)
            rows.append(statistics.significance_row(
                comparison=f"MIL-CREDA vs {rival}, BA, {t.name}",
                test="paired bootstrap by subject, stratified by class, mean over seeds",
                statistic=res["ba_difference"], p_value=res["ba_p_value"],
                effect_size=res["ba_difference"], effect_size_name="balanced accuracy difference",
                ci_low=res["ba_ci_low"], ci_high=res["ba_ci_high"], n=res["n"],
                hypothesis=V.experiment_of(t), experiment=V.experiment_of(t),
                family=f"support: MIL-CREDA vs {rival}", transfer=t.name))
        sig += statistics.apply_holm(rows, f"MIL-CREDA vs {rival} over the transfers")
    adapted = [REFERENCE, *rivals]
    blocks = [t.name for t in passing if all((t.name, m) in metrics for m in adapted)]
    ranking = None
    if len(adapted) >= 3 and len(blocks) >= 2:
        ranking = friedman_holm({m: [metrics[(b, m)][1]["balanced_accuracy"] for b in blocks]
                                 for m in adapted})
        sig.append(statistics.significance_row(
            comparison="ranking of the nine adapted strategies, BA", test="Friedman",
            statistic=ranking["statistic"], p_value=ranking["p_value"], effect_size=None,
            effect_size_name="average rank", ci_low=None, ci_high=None, n=len(blocks),
            hypothesis="A-nivel", experiment="A-nivel", family="ranking",
            average_rank=ranking["average_rank"]))
        for pr in ranking["pairs"]:
            sig.append(statistics.significance_row(
                comparison=f"rank {pr['a']} vs {pr['b']}", test="Friedman post hoc z, Holm",
                statistic=pr["z"], p_value=pr["p_value"], p_adjusted=pr["p_adjusted"],
                correction=f"holm (pairs, m={len(ranking['pairs'])})", effect_size=None,
                effect_size_name="average rank difference", ci_low=None, ci_high=None,
                n=len(blocks), hypothesis="A-nivel", experiment="A-nivel", family="ranking"))
    (product / "Results").mkdir(parents=True, exist_ok=True)
    (product / "Results" / "significance_v63.json").write_text(
        json.dumps(sig, indent=1, default=float), encoding="utf-8")
    summary = {"revision": V.REVISION,
               "targetFilter": {d: {"auc": v["auc"], "p_value": v["p_value"], "passes": v["passes"]}
                                for d, v in targets.items()},
               "excludedTransfers": [t.name for t in V.TRANSFERS
                                     if t.target in targets and not targets[t.target]["passes"]],
               "shiftFilter": {t: v["classification"] for t, v in shifts.items()},
               "transfersWithResults": sorted({t for t, _ in metrics}),
               "failedRefits": list(FAILED),
               "tables": ["S0-codificador/tables/target_filter.csv",
                          "F-corrimiento/tables/shift_filter.csv",
                          *(f"{e}/tables/performance.csv" for e in V.EXPERIMENT_OF_ARM.values())],
               "significance": "Results/significance_v63.json",
               "ranking": ranking and {"blocks": blocks, "average_rank": ranking["average_rank"],
                                       "p_value": ranking["p_value"]}}
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=float),
                                       encoding="utf-8")
    return summary
