"""Stage 06, ADHD graph analysis (paper Figures `fig:ieee-gsa` and `fig:ieee-pair-descriptors`).

Protocol v84 sec 8.2/8.4. For every ADHD unit (`ieee-r<rotation>h<half>`), the broadband phase-locking value
(PLV) of every window of the target-test and source subjects, with the per-window attention of MIL-CREDA's
cef pick as the v82 ADHD pool (stage 04) saved it: no retraining. Then

* GSA-XGB: subject graph = mean PLV of a window set (all windows; top / bottom 25 % attention; random 25 %,
  ten draws); 39 descriptors (degree and eigenvector centrality per electrode, spectral radius); XGBoost with
  the IWINAC 2026 configuration and the six descriptors of highest gain per training fold, stratified 5-fold x
  10 repeats; paired bootstrap of BA(set) - BA(all) over subjects, Holm over three contrasts.
* Pair descriptors: attention-weighted graph of the ADHD pair of `fig:ieee-alignment` (unit ieee-r3h1, target
  IEEE-v236, source IEEE-v265) and the correlation of its 39 descriptors among all pairs of the unit.

The code ports the campaign scripts verbatim (MIL-CREDA-EEG/Results/v84-gsa/gsa84.py and xgb84.py, untracked)
and the figure scripts paper/.paper-writing/figures/ieee-gsa/fig_gsa84.py and
ieee-pair-descriptors/fig_pair84.py. The campaign ran it on a local workstation, not on Kaggle (about 12 s per
unit for the PLV, gsa84-report.json).

Stored summary (MIL-CREDA-EEG/Results/paper/06_adhd_graph_analysis/summary/):
  pick_attention.npz  the only stage-04 fields this stage reads, per unit: att / attOffsets of the
                      target_test and source sides and Ptest of MIL-CREDA-<trial>.npz (the cef pick).
  graphs.npz          per target-test subject (121) the mean PLV graph of every window set.
  pair_graphs.npz     attention-weighted graphs of every target-test and source subject of ieee-r3h1.
  xgb84.json          the GSA-XGB result the paper reports (the campaign's file).
  gsa84-report.json   per unit: the pick, its BA/AUC recomputed from the saved Ptest, run time.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from MIL_CREDA_EEG import repro

STAGE = "06_adhd_graph_analysis"
#: the ten ADHD units, as `ieee_pool.UNITS` (5 rotations x 2 halves)
UNITS = tuple(f"ieee-r{r}h{h}" for r in range(5) for h in range(2))
#: the 121 target-test subjects across the ten units (xgb84.py asserts it)
N_SUBJECTS = 121
CH = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz")
NAMES = [f"deg {c}" for c in CH] + [f"ec {c}" for c in CH] + ["SR"]
#: xgb84.py: quarter kept, random draws, CV repeats, bootstrap resamples, descriptors kept per fold
FRAC, DRAWS, REPEATS, BOOT, TOP = 0.25, 10, 10, 10000, 6
#: xgb84.py: the IWINAC 2026 XGBoost configuration
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", n_estimators=200, max_depth=3,
                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=4)
#: fig_pair84.py: the ADHD pair of fig:ieee-alignment
PAIR_UNIT, PAIR_TARGET, PAIR_SOURCE = "ieee-r3h1", "IEEE-v236", "IEEE-v265"
#: fig_gsa84.py: pairs drawn per head; one column of the IJNS page (pt)
NE, TW = 20, 226.7
POS = {"Fp1": (-0.31, 0.95), "Fp2": (0.31, 0.95), "F7": (-0.81, 0.59), "F3": (-0.39, 0.5), "Fz": (0, 0.4),
       "F4": (0.39, 0.5), "F8": (0.81, 0.59), "T7": (-1, 0), "C3": (-0.5, 0), "Cz": (0, 0), "C4": (0.5, 0),
       "T8": (1, 0), "P7": (-0.81, -0.59), "P3": (-0.39, -0.5), "Pz": (0, -0.4), "P4": (0.39, -0.5),
       "P8": (0.81, -0.59), "O1": (-0.31, -0.95), "O2": (0.31, -0.95)}
SIDES = ("target_test", "source")
CONDITIONS = ("all", "top", "bottom") + tuple(f"random {d}" for d in range(DRAWS))
#: the cef pick of MIL-CREDA per unit (trial, BA, AUC on the target-test half), tracked
PICKS_FILE = repro.eeg_dir() / "Manifests" / "attention_influence" / "ieee_cef_picks.json"
#: the stage-04 fields of MIL-CREDA-<trial>.npz this stage reads
POOL_FIELDS = tuple(f"{k}[{s}]" for s in SIDES for k in ("att", "attOffsets")) + ("Ptest",)


def summary_dir() -> Path:
    return repro.results_dir(STAGE) / "summary"


def picks() -> dict:
    return json.loads(PICKS_FILE.read_text())["units"]


# ---------------------------------------------------------------- PLV per window (gsa84.py, verbatim)

def plv_broadband(windows):
    """Common average, broadband Hilbert phase, |mean phase difference| per channel pair, zero diagonal."""
    from scipy.signal import hilbert
    x = np.asarray(windows, float)
    x = x - x.mean(1, keepdims=True)
    z = np.exp(1j * np.angle(hilbert(x, axis=-1)))
    p = np.abs(np.einsum("wct,wdt->wcd", z, z.conj())) / x.shape[-1]
    i = np.arange(x.shape[1]); p[:, i, i] = 0.0
    return p.astype(np.float32)


def _ba_cut(y, p1, cut):
    yh = (p1 >= cut).astype(int)
    return float(np.mean([(yh[y == k] == k).mean() for k in (0, 1)]))


def _key(unit, field):
    return f"{unit}__{field.replace('[', '_').replace(']', '')}"


def condense_pool(pool_dir, out=None) -> Path:
    """Copy from stage 04's raw outputs (the ix-r*h*-s* job folders) the fields this stage reads into
    `pick_attention.npz` (a few hundred kB). Run once after the ADHD pool, before `run`."""
    out = Path(out or summary_dir()); out.mkdir(parents=True, exist_ok=True)
    store = {}
    for unit, pk in picks().items():
        a = np.load(_pool_npz(pool_dir, unit, pk["trial"]))
        for f in POOL_FIELDS:
            store[_key(unit, f)] = a[f]
        store[f"{unit}__trial"] = np.array(pk["trial"])
        store[f"{unit}__cut"] = np.array(_pool_cut(pool_dir, unit))
    path = out / "pick_attention.npz"
    np.savez_compressed(path, **store)
    return path


def _pool_npz(pool_dir, unit, trial) -> Path:
    r, h = unit[6], unit[8]
    hits = list(Path(pool_dir).glob(f"ix-r{r}h{h}-s*/ieeepool-{unit}-sh*of8/MIL-CREDA-{trial}.npz"))
    assert len(hits) == 1, (unit, hits)
    return hits[0]


def _pool_cut(pool_dir, unit) -> float:
    """The phase-1 cut the pool ran with, as each shard's ieeepool-<unit>-sh*of8.json records it. It is not the
    cut of the tracked MIL-CREDA-EEG/Manifests/ieee_pool_bases (commit ce382b3): the v82 pool ran on the bases
    of commit d7a4ef3e (branch ieee-pool-slim), whose cuts differ; the windows and configs are the same."""
    cuts = {json.loads(f.read_text())["cut"] for f in Path(pool_dir).glob(f"ix-*/ieeepool-{unit}-sh*of8.json")}
    assert len(cuts) == 1, (unit, cuts)
    return float(cuts.pop())


def _attention(unit, trial, pool_dir=None, summary=None) -> dict:
    """The pick's attention, Ptest and the pool's cut: from stage 04's job folders or pick_attention.npz."""
    if pool_dir is not None:
        a = np.load(_pool_npz(pool_dir, unit, trial))
        return {**{f: a[f] for f in POOL_FIELDS}, "cut": _pool_cut(pool_dir, unit)}
    a = np.load(Path(summary or summary_dir()) / "pick_attention.npz")
    if int(a[f"{unit}__trial"]) != int(trial):
        raise ValueError(f"pick_attention.npz holds trial {int(a[f'{unit}__trial'])} for {unit}, not {trial}")
    return {**{f: a[_key(unit, f)] for f in POOL_FIELDS}, "cut": float(a[f"{unit}__cut"])}


def unit_job(unit, out, pool_dir=None, ieee_root=None) -> dict:
    """gsa84.py's unit_job: PLV and attention per window of both sides; checks the saved Ptest."""
    from sklearn.metrics import roc_auc_score
    from MIL_CREDA_EEG import ieee_pool as I
    t0 = time.time()
    pk = picks()[unit]
    a = _attention(unit, pk["trial"], pool_dir)
    s = I._setup(unit, ["MIL-CREDA"], None, 1, ieee_root)   # the tracked seed-1 base: the unit's config and sides
    try:
        res, report = {}, {"unit": unit, "trial": pk["trial"]}
        for side in SIDES:
            bags = s["sides"][side]
            att, off = a[f"att[{side}]"], a[f"attOffsets[{side}]"]
            assert len(off) == len(bags) + 1, (unit, side, len(off), len(bags))
            plv, w, n = [], [], []
            for k, b in enumerate(bags):
                x = b.windows.numpy(); wk = att[off[k]:off[k + 1]]
                assert len(wk) == x.shape[0], (unit, side, b.subject, len(wk), x.shape)
                plv.append(plv_broadband(x)); w.append(wk); n.append(x.shape[0])
            res[f"{side}_plv"] = np.concatenate(plv); res[f"{side}_att"] = np.concatenate(w).astype(np.float64)
            res[f"{side}_n"] = np.array(n); res[f"{side}_labels"] = np.array([int(b.label) for b in bags])
            res[f"{side}_subjects"] = np.array([b.subject for b in bags])
        y = res["target_test_labels"]; p = np.asarray(a["Ptest"], float)
        p1 = p[:, 1] if p.ndim == 2 else p
        report["BA"] = _ba_cut(y, p1, a["cut"]); report["AUC"] = float(roc_auc_score(y, p1))
        report["diff"] = max(abs(report["BA"] - pk["BA"]), abs(report["AUC"] - pk["AUC"]))
        report["reproduces"] = report["diff"] <= 1e-4
        Path(out).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(out) / f"gsa84-{unit}.npz", **res)
    finally:
        s["stack"].close()
    report["seconds"] = round(time.time() - t0, 1)
    (Path(out) / f"gsa84-report-{unit}.json").write_text(json.dumps(report, indent=1))
    return report


def run(split=None, smoke=False, out=None, pool_dir=None, ieee_root=None) -> list[dict]:
    """PLV per window for `repro.portion(UNITS, split)`; `smoke` runs only the first unit of the portion.
    Attention comes from `pool_dir` (stage 04's job folders) when given, else from the stored
    pick_attention.npz. Writes runs/gsa84-<unit>.npz (about 3.5 MB each) and one report per unit."""
    root = ieee_root or repro.ieee_root()
    out = Path(out or repro.results_dir(STAGE) / "runs")
    units = repro.portion(UNITS, split)
    if smoke:
        units = units[:1]
    reports = []
    for u in units:
        r = unit_job(u, out, pool_dir, root)
        print(u, "trial", r["trial"], "BA", round(r["BA"], 5), "AUC", round(r["AUC"], 5),
              "reproduces", r["reproduces"], f"{r['seconds']} s", flush=True)
        reports.append(r)
    return reports


# ---------------------------------------------------------------- graphs and GSA-XGB (xgb84.py, verbatim)

def descriptors(G):
    vals, vecs = np.linalg.eigh((G + G.T) / 2)
    top = vecs[:, -1]; top = top * np.sign(top.sum() or 1.0); top = top / (np.linalg.norm(top) or 1.0)
    return np.concatenate([G.sum(1), top, [vals[-1]]])


def graphs_from_runs(runs) -> tuple[dict, dict]:
    """Per target-test subject the mean PLV of every window set, and the attention-weighted graphs of the
    pair unit, from the gsa84-<unit>.npz files in `runs`."""
    subs, y, units, G = [], [], [], {k: [] for k in CONDITIONS}
    pair = None
    for f in sorted(Path(runs).glob("gsa84-ieee-*.npz")):
        a = np.load(f); n = a["target_test_n"]; off = np.concatenate([[0], np.cumsum(n)])
        subs += a["target_test_subjects"].tolist(); y += a["target_test_labels"].tolist(); units += [f.stem[6:]] * len(n)
        for k in range(len(n)):
            P = a["target_test_plv"][off[k]:off[k + 1]].astype(float); w = a["target_test_att"][off[k]:off[k + 1]]
            q = int(round(FRAC * n[k])); o = np.argsort(w, kind="stable")
            G["all"].append(P.mean(0)); G["top"].append(P[o[-q:]].mean(0)); G["bottom"].append(P[o[:q]].mean(0))
            for dr in range(DRAWS):
                G[f"random {dr}"].append(P[np.random.default_rng(1000 + dr).permutation(n[k])[:q]].mean(0))
        if f.stem[6:] == PAIR_UNIT:   # fig_pair84.py: each subject's graph weighted by its attention
            pair = {}
            for side in SIDES:
                n2 = a[f"{side}_n"]; o2 = np.concatenate([[0], np.cumsum(n2)])
                pair[f"{side}_subjects"] = a[f"{side}_subjects"]
                pair[f"{side}_graphs"] = np.array([
                    np.einsum("w,wcd->cd", (w := a[f"{side}_att"][o2[k]:o2[k + 1]]) / w.sum(),
                              a[f"{side}_plv"][o2[k]:o2[k + 1]].astype(float)) for k in range(len(n2))])
    graphs = {"subjects": np.array(subs), "labels": np.array(y), "units": np.array(units)}
    graphs.update({k: np.array(v) for k, v in G.items()})
    return graphs, pair


def gsa_xgb(graphs, repeats=REPEATS, draws=DRAWS, boot=BOOT, full=True) -> dict:
    """xgb84.py on stored graphs; `full=False` (smoke) drops the 121-subject check."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    y = np.asarray(graphs["labels"]); subs = [str(s) for s in graphs["subjects"]]
    if full:
        assert len(y) == N_SUBJECTS and len(set(subs)) == N_SUBJECTS, (len(y), len(set(subs)))
    X = {k: np.array([descriptors(G) for G in graphs[k]]) for k in ("all", "top", "bottom")}
    X.update({f"random {d}": np.array([descriptors(G) for G in graphs[f"random {d}"]]) for d in range(draws)})
    xgb = lambda rep: XGBClassifier(**XGB_PARAMS, random_state=rep)

    def true_prob(F, chosen):
        p = np.zeros(len(y))
        for rep in range(repeats):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(F, y):
                rank = xgb(rep).fit(F[tr], y[tr]).get_booster().get_score(importance_type="gain")
                gain = np.array([rank.get(f"f{i}", 0.0) for i in range(F.shape[1])])
                sel = np.argsort(-gain, kind="stable")[:TOP]; chosen.update(NAMES[i] for i in sel)
                q = xgb(rep).fit(F[tr][:, sel], y[tr]).predict_proba(F[te][:, sel])[:, 1]
                p[te] += np.where(y[te] == 1, q, 1 - q) / repeats
        return p

    def ba(c, i=None):
        c, t = (c, y) if i is None else (c[i], y[i])
        return float(np.mean([c[t == k].mean() for k in (0, 1)]))

    def holm(p):
        adj, run_ = {}, 0.0
        for i, k in enumerate(sorted(p, key=p.get)):
            run_ = max(run_, min(1.0, p[k] * (len(p) - i))); adj[k] = run_
        return adj

    sel = {k: Counter() for k in ("all", "top", "bottom", "random")}
    P = {k: true_prob(X[k], sel[k]) for k in ("all", "top", "bottom")}
    P["random"] = np.mean([true_prob(X[f"random {d}"], sel["random"]) for d in range(draws)], axis=0)
    c = {k: (v > 0.5).astype(float) for k, v in P.items()}
    s1 = {k: np.where(y == 1, v, 1 - v) for k, v in P.items()}
    idx = [np.flatnonzero(y == k) for k in (0, 1)]
    rng = np.random.default_rng(1); diffs = {k: [] for k in ("top", "bottom", "random")}
    for _ in range(boot):
        i = np.concatenate([rng.choice(g, len(g)) for g in idx])
        for k in diffs:
            diffs[k].append(ba(c[k], i) - ba(c["all"], i))
    pb = {k: float(min(1.0, 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0)))) for k, v in diffs.items()}
    ph = holm(pb)
    folds = {"all": 5 * repeats, "top": 5 * repeats, "bottom": 5 * repeats, "random": 5 * repeats * draws}
    return {"subjects": subs, "labels": y.tolist(), "units": [str(u) for u in graphs["units"]], "descriptorNames": NAMES,
            "BA": {k: ba(v) for k, v in c.items()}, "AUC": {k: float(roc_auc_score(y, v)) for k, v in s1.items()},
            "minusAll": {k: {"BA": ba(c[k]) - ba(c["all"]),
                             "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                             "pBoot": pb[k], "pHolm": ph[k]} for k, v in diffs.items()},
            "selection": {k: {n: v / folds[k] for n, v in sel[k].most_common()} for k in sel},
            "trueProb": {k: [round(float(x), 5) for x in v] for k, v in P.items()}}


def merge(out=None, runs=None, smoke=False) -> Path:
    """Join the per-unit PLV files (copied from every account into one `runs` folder) into the stage
    summary: graphs.npz, pair_graphs.npz, gsa84-report.json, and xgb84.json from the GSA-XGB analysis
    (about 3 min on 4 CPU cores). `smoke` accepts a partial set of units and runs XGBoost with 1 repeat,
    2 random draws and 200 bootstrap resamples, only to prove the wiring."""
    runs = Path(runs or repro.results_dir(STAGE) / "runs")
    out = Path(out or summary_dir()); out.mkdir(parents=True, exist_ok=True)
    present = sorted(f.stem[6:] for f in runs.glob("gsa84-ieee-*.npz"))
    if not smoke and present != sorted(UNITS):
        raise FileNotFoundError(f"{runs} holds {len(present)} of the {len(UNITS)} units; missing "
                                f"{sorted(set(UNITS) - set(present))}. Copy every account's runs/ here first.")
    graphs, pair = graphs_from_runs(runs)
    np.savez_compressed(out / "graphs.npz", **graphs)
    if pair is not None:
        np.savez_compressed(out / "pair_graphs.npz", **pair)
    reports = [json.loads((runs / f"gsa84-report-{u}.json").read_text()) for u in present
               if (runs / f"gsa84-report-{u}.json").exists()]
    (out / "gsa84-report.json").write_text(json.dumps(reports, indent=1))
    res = gsa_xgb_isolated(graphs, 1, 2, 200, full=False) if smoke else gsa_xgb_isolated(graphs)
    (out / "xgb84.json").write_text(json.dumps(res, indent=1))
    return out


def gsa_xgb_isolated(graphs, *args, **kwargs) -> dict:
    """`gsa_xgb` in a fresh spawned process. On macOS, XGBoost crashes (segmentation fault, two OpenMP
    runtimes) when torch was imported first in the same process, as it is after `run`; the child imports
    XGBoost without torch. The result is the same as calling `gsa_xgb` directly."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(gsa_xgb, {k: np.asarray(v) for k, v in dict(graphs).items()}, *args, **kwargs).result()


def store_campaign_summary(campaign_dir, pool_dir, out=None) -> Path:
    """Build the tracked summary from the paper campaign's own files (Results/v84-gsa and the stage-04
    ieee-out folder): the graphs from its gsa84-*.npz, its xgb84.json and report as they are."""
    out = Path(out or summary_dir()); out.mkdir(parents=True, exist_ok=True)
    condense_pool(pool_dir, out)
    graphs, pair = graphs_from_runs(campaign_dir)
    np.savez_compressed(out / "graphs.npz", **graphs)
    np.savez_compressed(out / "pair_graphs.npz", **pair)
    for f in ("xgb84.json", "gsa84-report.json"):
        shutil.copyfile(Path(campaign_dir) / f, out / f)
    return out


# ---------------------------------------------------------------- figures and tables (no training)

def _load(summary=None):
    d = Path(summary or summary_dir())
    g = np.load(d / "graphs.npz"); pg = np.load(d / "pair_graphs.npz")
    return d, g, pg, json.loads((d / "xgb84.json").read_text())


def rank_biserial(graphs) -> dict:
    """fig_gsa84.py: per electrode pair, rank-biserial correlation of Mann-Whitney, ADHD vs control."""
    from scipy.stats import mannwhitneyu
    y = np.asarray(graphs["labels"]); iu = np.triu_indices(len(CH), 1)
    rb = {}
    for k, key in (("all", "all"), ("top", "top"), ("random", "random 0")):
        E = np.asarray(graphs[key])[:, iu[0], iu[1]]
        U = np.array([mannwhitneyu(E[y == 1, e], E[y == 0, e]).statistic for e in range(E.shape[1])])
        rb[k] = 2 * U / ((y == 1).sum() * (y == 0).sum()) - 1
    return rb


def gsa_summary(graphs, xg) -> dict:
    """fig_gsa84_summary.json: the 20 strongest pairs per window set and the ringed electrodes."""
    rb = rank_biserial(graphs); iu = np.triu_indices(len(CH), 1)
    out = {}
    for k, v in rb.items():
        strong = np.argsort(-np.abs(v))[:NE]
        marked = {n.split(" ")[1] for n in list(xg["selection"][k].keys())[:6] if " " in n}
        out[k] = {"edges": [[CH[iu[0][e]], CH[iu[1][e]], round(float(v[e]), 3)] for e in strong],
                  "meanAbsTop": float(np.abs(v[strong]).mean()), "nHigherADHD": int((v[strong] > 0).sum()),
                  "marked": sorted(marked)}
    out["vmax"] = float(max(np.sort(np.abs(v))[-NE:].max() for v in rb.values()))
    return out


def pair_summary(pg) -> dict:
    """pair84.json (fig_pair84.py): descriptor and centrality correlation of the pair, percentile among pairs."""
    n = len(CH)
    D = {(side, str(s)): descriptors(G) for side in SIDES
         for s, G in zip(pg[f"{side}_subjects"], pg[f"{side}_graphs"])}
    t, s = D[("target_test", PAIR_TARGET)], D[("source", PAIR_SOURCE)]
    cor = lambda u, v: float(np.corrcoef(u, v)[0, 1])
    tg, sg = [str(i) for i in pg["target_test_subjects"]], [str(j) for j in pg["source_subjects"]]
    allc = np.array([cor(D[("target_test", i)], D[("source", j)]) for i in tg for j in sg])
    allcc = np.array([cor(D[("target_test", i)][n:2 * n], D[("source", j)][n:2 * n]) for i in tg for j in sg])
    return {"unit": PAIR_UNIT, "target": PAIR_TARGET, "source": PAIR_SOURCE, "descriptorCorrelation": cor(t, s),
            "centralityCorrelation": cor(t[n:2 * n], s[n:2 * n]),
            "percentileAmongAllPairs": float((allc < cor(t, s)).mean()), "meanCorrelationAllPairs": float(allc.mean()),
            "nPairs": int(len(allc)), "centralityPercentile": float((allcc < cor(t[n:2 * n], s[n:2 * n])).mean()),
            "target_centrality": t[n:2 * n].tolist(), "source_centrality": s[n:2 * n].tolist()}


def _style(aspect):
    import matplotlib.pyplot as plt
    from MIL_CREDA_EEG import ps_figures as F
    F.apply_style(10, 1.0, text_width_pt=TW, aspect=aspect); plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({k: 10.0 for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                                             "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
                                             "figure.titlesize")})  # journal floor: text >= 10 pt


def _fit_width_and_save(fig, path):
    from MIL_CREDA_EEG import ps_figures as F
    for _ in range(8):                          # widen or narrow the canvas until the tight crop is TW wide
        fig.draw_without_rendering(); w = fig.get_tightbbox().width + 2 * F.PAD_INCHES
        if abs(w * 72 - TW) < 0.3: break
        fig.set_size_inches(fig.get_size_inches()[0] + TW / 72 - w, fig.get_size_inches()[1])
    F.save_figure(fig, path, png_preview=True)


def _fig_gsa(graphs, xg, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    rb = rank_biserial(graphs); iu = np.triu_indices(len(CH), 1)
    vmax = max(np.sort(np.abs(v))[-NE:].max() for v in rb.values())
    _style(0.66)
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    for ax, (k, lab) in zip(axs, (("all", "raw EEG"), ("top", "highest-attention 25 %"))):   # random: text only
        ax.add_patch(Circle((0, 0), 1.18, fill=False, lw=0.8, color="0.3"))
        ax.plot([-0.12, 0, 0.12], [1.17, 1.3, 1.17], color="0.3", lw=0.8)
        for e in np.argsort(-np.abs(rb[k]))[:NE][::-1]:
            a, b = CH[iu[0][e]], CH[iu[1][e]]; r = rb[k][e]
            ax.plot(*zip(POS[a], POS[b]), color="tab:red" if r > 0 else "tab:blue",
                    lw=0.3 + 2.2 * abs(r) / vmax, alpha=0.85, zorder=2)
        marked = {n.split(" ")[1] for n in list(xg["selection"][k].keys())[:6] if " " in n}
        for c in CH:
            ring = c in marked
            ax.scatter(*POS[c], s=95, color="white", edgecolors="k", lw=1.2 if ring else 0.4, zorder=3)
            ax.text(*POS[c], c, ha="center", va="center", fontsize=5.5, zorder=4,   # map labels, not running text
                    fontweight="bold" if ring else "normal")
        ax.set_xlim(-1.19, 1.19); ax.set_ylim(-1.25, 1.33); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlabel(f"{lab}\nBA {xg['BA'][k]:.3f}", labelpad=1)
    _fit_width_and_save(fig, path)
    plt.close(fig)


def _fig_pair(ps, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(CH); xx = np.arange(n)
    _style(0.44)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(xx - 0.2, ps["target_centrality"], 0.4, color="tab:red"); ax.bar(xx + 0.2, ps["source_centrality"], 0.4, color="0.6")
    ax.set_xticks(xx, CH, rotation=90); ax.set_ylabel("eigenvector\ncentrality"); ax.set_xlim(-0.6, n - 0.4)
    _fit_width_and_save(fig, path)
    plt.close(fig)


def render(summary_dir=None, out_dir=None) -> dict:
    """ieee-gsa and ieee-pair-descriptors (PDF + PNG) and the GSA-XGB / pair tables (.tex + .csv), from the
    stored summary only."""
    d, g, pg, xg = _load(summary_dir)
    out = Path(out_dir or repro.results_dir(STAGE))
    figs, tabs = out / "figures", out / "tables"; figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    gs, ps = gsa_summary(g, xg), pair_summary(pg)
    _fig_gsa(g, xg, figs / "ieee-gsa.pdf")
    _fig_pair(ps, figs / "ieee-pair-descriptors.pdf")
    (tabs / "fig_gsa84_summary.json").write_text(json.dumps(gs, indent=1))
    (tabs / "pair84.json").write_text(json.dumps({k: v for k, v in ps.items() if not k.endswith("centrality")}, indent=1))
    labels = {"all": "All windows (raw EEG)", "top": "Highest-attention 25 %", "bottom": "Lowest-attention 25 %",
              "random": "Random 25 % (10 draws)"}
    rows = []
    for k in ("all", "top", "bottom", "random"):
        m = xg["minusAll"].get(k)
        rows.append([labels[k], xg["BA"][k], xg["AUC"][k], m and m["BA"], m and m["ci95"][0], m and m["ci95"][1],
                     m and m["pHolm"], "; ".join(f"{n} {f:.2f}" for n, f in list(xg["selection"][k].items())[:6])])
    head = ["window set", "BA", "AUC", "BA - BA(all)", "CI95 low", "CI95 high", "p Holm", "top-6 descriptors (fraction of folds)"]
    import csv
    with open(tabs / "gsa_xgb.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([head] + rows)
    f3 = lambda v: "--" if v is None else f"{v:.3f}"
    tex = ["\\begin{tabular}{lrrrrr}", "\\hline",
           "Window set & BA & AUC & $\\Delta$BA vs all & 95\\,\\% CI & $p_{\\mathrm{Holm}}$ \\\\", "\\hline"]
    for r in rows:
        ci = "--" if r[4] is None else f"[{r[4]:.3f}, {r[5]:.3f}]"
        tex.append(f"{r[0].replace('%', chr(92) + '%')} & {f3(r[1])} & {f3(r[2])} & {f3(r[3])} & {ci} & {f3(r[6])} \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (tabs / "gsa_xgb.tex").write_text("\n".join(tex) + "\n")
    pr = [["descriptor correlation of the pair", ps["descriptorCorrelation"]],
          ["percentile among the unit's pairs", ps["percentileAmongAllPairs"]],
          ["mean descriptor correlation of all pairs", ps["meanCorrelationAllPairs"]],
          ["centrality correlation of the pair", ps["centralityCorrelation"]],
          ["centrality percentile among the unit's pairs", ps["centralityPercentile"]],
          ["pairs in the unit", ps["nPairs"]]]
    with open(tabs / "pair_descriptors.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["quantity", "value"]] + pr)
    (tabs / "pair_descriptors.tex").write_text("\n".join(
        ["\\begin{tabular}{lr}", "\\hline"] + [f"{q} & {v:.3f} \\\\" if isinstance(v, float) else f"{q} & {v} \\\\"
                                               for q, v in pr] + ["\\hline", "\\end{tabular}"]) + "\n")
    return {"figures": sorted(str(p) for p in figs.iterdir()), "tables": sorted(str(p) for p in tabs.iterdir())}


def paper_checks(summary_dir=None) -> list[tuple[str, float, float, float]]:
    """The stage's numbers as main.tex states them (rounded as printed; tolerance 0 on the rounded value)."""
    d, g, pg, xg = _load(summary_dir)
    gs, ps = gsa_summary(g, xg), pair_summary(pg)
    m = xg["minusAll"]
    f4cz = lambda k: next(e[2] for e in gs[k]["edges"] if {e[0], e[1]} == {"F4", "Cz"})
    r3, r2 = (lambda v: round(v, 3)), (lambda v: round(v, 2))
    return [
        # main.tex l.477: "F4--Cz, led with all windows and with the highest-attention quarter, as it did with a
        # random quarter (rank-biserial correlation $-0.50$, $-0.48$ and $-0.43$)"
        ("F4-Cz rank-biserial, all windows (l.477)", r2(f4cz("all")), -0.50, 0.0),
        ("F4-Cz rank-biserial, top 25 % (l.477)", r2(f4cz("top")), -0.48, 0.0),
        ("F4-Cz rank-biserial, random 25 % (l.477)", r2(f4cz("random")), -0.43, 0.0),
        ("F4-Cz leads all / top / random (l.477)",
         float(all(set(gs[k]["edges"][0][:2]) == {"F4", "Cz"} for k in ("all", "top", "random"))), 1.0, 0.0),
        # l.477: "0.727 and 0.780 against 0.645 and 0.672 with all windows, a difference of 0.083 with a 95 %
        # bootstrap interval of [0.008, 0.158]"; l.58 and l.547 repeat 0.645 -> 0.727
        ("XGBoost BA, top 25 % (l.477)", r3(xg["BA"]["top"]), 0.727, 0.0),
        ("XGBoost AUC, top 25 % (l.477)", r3(xg["AUC"]["top"]), 0.780, 0.0),
        ("XGBoost BA, all windows (l.477)", r3(xg["BA"]["all"]), 0.645, 0.0),
        ("XGBoost AUC, all windows (l.477)", r3(xg["AUC"]["all"]), 0.672, 0.0),
        ("BA top - all (l.477)", r3(m["top"]["BA"]), 0.083, 0.0),
        ("BA top - all, CI95 low (l.477)", r3(m["top"]["ci95"][0]), 0.008, 0.0),
        ("BA top - all, CI95 high (l.477)", r3(m["top"]["ci95"][1]), 0.158, 0.0),
        # l.477: "A random quarter reached 0.686 and the discarded quarter 0.595, neither different from all
        # windows ($p_{\mathrm{Holm}}=0.442$)" ... "that gain remains a trend ($p_{\mathrm{Holm}}=0.107$)"
        ("XGBoost BA, random 25 % (l.477)", r3(xg["BA"]["random"]), 0.686, 0.0),
        ("XGBoost BA, bottom 25 % (l.477)", r3(xg["BA"]["bottom"]), 0.595, 0.0),
        ("p Holm, random - all (l.477)", r3(m["random"]["pHolm"]), 0.442, 0.0),
        ("p Holm, bottom - all (l.477)", r3(m["bottom"]["pHolm"]), 0.442, 0.0),
        ("p Holm, top - all (l.477)", r3(m["top"]["pHolm"]), 0.107, 0.0),
        # l.507: "The 39 graph descriptors of the ADHD pair ... correlated at 0.988, above 90 % of the pairs of the
        # unit, although the degrees make every pair correlate highly (mean 0.973), and their centrality maps at
        # 0.68, above 82 %."
        ("pair descriptor correlation (l.507)", r3(ps["descriptorCorrelation"]), 0.988, 0.0),
        ("pair above 90 % of the unit's pairs (l.507)", float(ps["percentileAmongAllPairs"] > 0.90), 1.0, 0.0),
        ("mean descriptor correlation, all pairs (l.507)", r3(ps["meanCorrelationAllPairs"]), 0.973, 0.0),
        ("pair centrality correlation (l.507)", r2(ps["centralityCorrelation"]), 0.68, 0.0),
        ("pair centrality above 82 % of pairs (l.507)", float(ps["centralityPercentile"] > 0.82), 1.0, 0.0),
    ]
