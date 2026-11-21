"""Stage 02, synthetic criteria pool (paper Table `tab:selection-dev` and Figure `fig:synthetic-map`).

Protocol v82, sections 5 (A-mapa, H1) and 7 (S-seleccion, H3). Phase 1 is frozen: every method of every
synthetic unit starts from the unit's single v82 final-rule base (`<unit>-s1.pt`). For each of the eight
methods, `criteria_pool.run` trains the same 100 seeded random phase-2 configurations for 100 epochs and
records, per trial, the balanced accuracy at the base's cut on 120 fresh target subjects (the oracle, true
labels only the simulator has) and the target-train and source-test outputs from which every label-free
criterion is computed offline. The campaign ran it on Kaggle as family `sx`: 24 units x 8 trial shards =
192 CPU notebooks (run-config.json of `sx-<unit>-<shard>`: seed 1, shards 8, procs 4, threads 1, repository
commit 820ce461 of ieee-pool-slim, whose criteria_pool/oracle_screen/xcompare/xpilot are byte-identical to
this repository's).

Analysis, ported from the campaign scripts (MIL-CREDA-EEG/Results/v82-pool24, untracked):
* `trial_rows` = crit3.py + export_trials.py: one row per (unit, method, finite trial) with the eight
  criterion scores, `pool` (BA on the 120-subject evaluation pool) and `test` (BA on the 12 target-test
  subjects). The v82 file was built by the Results/v77-pool24/crit3.py copy, not the v74 one: only the v77
  copy writes the `trial` field (present in every v82 row) and sorts each pool by trial (ties: first
  maximum in trial order); the v74 copy has neither.
* `selection_summary` = v77-pool24/h3_pool.py (the criteria rows of the table; its v82 output is
  v82-pool24/h3_pool_summary.json).
* `methods_summary` = v82-pool24/h1_pool24.py (the method rank row and the Phase 1 column).
* `map_cells` = the cef pick minus the frozen phase-1 base per (class effect, shift) cell, mean of the two
  rotations (v82-pool24/pool24_map_cells.json); drawn by paper/.paper-writing/figures/
  synthetic-map-methods/make_map.py, ported in `render`.
* `picks` = v82-pool24/mkpicks82.py: MIL-CREDA's cef pick per unit and the comparators' picks on unit
  d1p5-s1p5-r0, with their settings. Stages 03 (synthetic mechanisms) and 05 (attention influence) read
  them from this stage's summary.
* `phase1_bases` = v82-pool24/eval_bases.py: phase-1 BA of each frozen base on the same pool.

Stored summary (MIL-CREDA-EEG/Results/paper/02_synthetic_criteria_pool/summary/), copied from the v82
campaign (the raw sx outputs are not kept):
  criteria_pool_trials.json.gz  the campaign's criteria_pool_trials.json, byte for byte, gzip-compressed
  phase1_bases.json             eval_bases.py output (phase-1 BA/AUC of each base on the pool)
  milcreda_picks.json, comparator_picks.json  mkpicks82.py output (for stages 03 and 05)
  selection.json, methods.json, map_cells.json  derived here from the two files above
The v82 bases themselves are in inputs/bases/ (from tag archive/ieee-pool-slim, commit 820ce461).
"""
from __future__ import annotations

import gzip
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from MIL_CREDA_EEG import repro

STAGE = "02_synthetic_criteria_pool"
#: the 24 synthetic units: class effect d, shift s, rotation r (eval_bases.py's order)
UNIT_KEYS = tuple(f"d{d}-s{s}-r{r}" for d in ("0p6", "1", "1p5", "2p5") for s in ("0p5", "1p5", "3") for r in (0, 1))
#: run-config.json of every sx-<unit>-<shard> job (scratchpad kaggle-jobs/ieee-slim/tools/kaggle/sx-*)
SEED, SHARDS, PROCS, THREADS = 1, 8, 4, 1
#: one work item = one Kaggle notebook of the campaign: (unit, trial shard)
UNITS = tuple((u, sh) for u in UNIT_KEYS for sh in range(SHARDS))
METHODS = ("MIL-CREDA", "UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS", "CLUST", "CLUDA")
COMPARATORS = METHODS[1:]
#: crit3.py's criteria, in its order; NAMES as h3_pool.py prints them
CRITERIA = ("cef", "BNM", "IM", "ClassAMI", "entropy", "SND", "source BA", "centroid BA")
NAMES = {"cef": "cef", "IM": "IM", "BNM": "BNM", "ClassAMI": "ClassAMI", "source BA": "Source accuracy",
         "centroid BA": "Centroid rule", "SND": "SND", "entropy": "Entropy"}
#: the paper's row labels for tab:selection-dev
PAPER_LABEL = {"cef": "cef", "centroid BA": "Centroid score", "source BA": "Source accuracy", "BNM": "BNM",
               "IM": "IM", "ClassAMI": "ClassAMI", "entropy": "Entropy", "SND": "SND"}
#: mkpicks82.py: the unit whose comparator picks the mechanism figures use
FIGURE_UNIT = "d1p5-s1p5-r0"
D_LEVELS, SHIFTS, ADHD_BAND = (0.6, 1.0, 1.5, 2.5), (0.5, 1.5, 3.0), 0.6
#: the smoke item: one unit, MIL-CREDA, the first trial, 2 epochs, 12 pool subjects
SMOKE = dict(unit_key=FIGURE_UNIT, trials=1, shards=1, shard=0, methods=("MIL-CREDA",), epochs=2,
             eval_subjects=12, procs=1, threads=4)
PICK_SOURCE = ("criteria pool v82 (protocol v82 final-rule bases, ieee-pool-slim 820ce461 lineage), seed 1; "
               "cef = exp entropy of argmax histogram of Pt; first max over finite trials in trial order")


def stage_dir() -> Path:
    return repro.results_dir(STAGE)


def summary_dir() -> Path:
    return stage_dir() / "summary"


def bases_dir() -> Path:
    """The 24 v82 final-rule phase-1 bases (tag archive/ieee-pool-slim, commit 820ce461)."""
    return stage_dir() / "inputs" / "bases"


def _key(unit: str) -> tuple:
    d, s, _ = unit.split("-")
    return float(d[1:].replace("p", ".")), float(s[1:].replace("p", "."))


# ------------------------------------------------------------------------------------------- run

def check_bases(bases, units=UNIT_KEYS) -> None:
    """Refuse bases whose cut differs from the v82 base the paper used (phase1_bases.json)."""
    import torch
    from MIL_CREDA_EEG import spectral_trunk
    want = json.loads((summary_dir() / "phase1_bases.json").read_text())["units"]
    for u in units:
        with spectral_trunk.installed():
            b = torch.load(Path(bases) / f"{u}-s1.pt", weights_only=False)
        if abs(float(b["cut"]) - want[u]["cut"]) > 1e-12:
            raise ValueError(f"{bases}/{u}-s1.pt is not the v82 base (cut {b['cut']} != {want[u]['cut']}); "
                             f"use {bases_dir()} (MIL-CREDA-EEG/Manifests/oracle_bases holds older bases)")


def run(split=None, smoke: bool = False, out=None, bases=None) -> list[str]:
    """Train the phase-2 pool of every (unit, shard) in `repro.portion(UNITS, split)` with the campaign's
    parameters (criteria_pool defaults: 100 trials x 100 epochs, 120 pool subjects; seed 1, 8 shards).
    Each item writes `<out>/sx-<unit>-<shard>/criteria-<unit>-k1-sh<shard>of8.json` and its npz arrays.
    `smoke=True` runs one MIL-CREDA trial of 2 epochs on 12 pool subjects into `<out>/smoke` (`out` defaults
    to a new temporary folder for the smoke run). Raw outputs are large (one npz per trial): keep `raw/` out
    of git."""
    from MIL_CREDA_EEG import criteria_pool as CP
    bases = Path(bases) if bases else bases_dir()
    if smoke:                       # a throw-away folder unless one is given: smoke rows never join the summary
        out = Path(out) if out else Path(tempfile.mkdtemp(prefix="stage02-smoke-"))
        s = dict(SMOKE)
        check_bases(bases, [s["unit_key"]])
        r = CP.run(s.pop("unit_key"), SEED, s.pop("shard"), workdir=out / "smoke", bases=bases, **s)
        return [str(out / "smoke" / r["file"])]
    out = Path(out) if out else stage_dir() / "raw"
    done = []
    items = repro.portion(UNITS, split)
    check_bases(bases, sorted({u for u, _ in items}))
    for u, sh in items:
        r = CP.run(u, SEED, sh, shards=SHARDS, workdir=out / f"sx-{u}-{sh}", bases=bases, procs=PROCS,
                   threads=THREADS)
        done.append(str(out / f"sx-{u}-{sh}" / r["file"]))
    return done


# ------------------------------------------------------------------------------- crit3.py criteria

def _ent(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(-1)


def cef(pt) -> float:
    h = np.bincount(pt.argmax(1), minlength=2) / len(pt)
    h = h[h > 0]
    return float(np.exp(-(h * np.log(h)).sum()))


def _centroid(zt, pt, zs, ys):
    from sklearn.metrics import balanced_accuracy_score
    q = pt.argmax(1)
    if any(not np.any(q == c) for c in (0, 1)):
        return 0.0
    c = np.stack([zt[q == k].mean(0) for k in (0, 1)])
    d = ((zs[:, None] - c[None]) ** 2).sum(-1)
    return float(balanced_accuracy_score(ys, d.argmin(1)))


def _snd(zt, tau=0.05):
    z = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-12)
    s = z @ z.T / tau
    np.fill_diagonal(s, -np.inf)
    s = np.exp(s - s.max(1, keepdims=True))
    s /= s.sum(1, keepdims=True)
    return float(_ent(s).mean())


def _class_ami(zt, pt):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score
    q = pt.argmax(1)
    if len(np.unique(q)) < 2:
        return 0.0
    k = KMeans(2, n_init=10, random_state=0).fit_predict(zt)
    return float(adjusted_mutual_info_score(q, k))


def criteria(a, ys) -> dict:
    """crit3.py's CRITERIA, verbatim: the eight label-free scores of one trial's saved outputs."""
    from sklearn.metrics import balanced_accuracy_score
    return {"cef": cef(a["Pt"]),
            "BNM": float(np.linalg.svd(a["Pt"], compute_uv=False).sum() / len(a["Pt"])),
            "IM": float(_ent(a["Pt"].mean(0)) - _ent(a["Pt"]).mean()),
            "ClassAMI": _class_ami(a["Zt"], a["Pt"]),
            "entropy": float(-_ent(a["Pt"]).mean()),
            "SND": _snd(a["Zt"]),
            "source BA": float(balanced_accuracy_score(ys, a["Ps"].argmax(1))),
            "centroid BA": _centroid(a["Zt"], a["Pt"], a["Zs"], ys)}


def _raw_trials(raw):
    """(unit, method, trial, row, npz path, source labels) of every finite trial with saved arrays."""
    for f in sorted(Path(raw).rglob("criteria-*.json")):
        d = json.loads(f.read_text())
        adir = f.with_suffix("")
        for m, v in d["methods"].items():
            for i, r in v["trials"].items():
                p = adir / f"{m.split()[0]}-{i}.npz"
                if r.get("nonFinite") or not p.exists():
                    continue
                yield d["unitKey"], m.split()[0], int(i), r, p, d["sourceTestLabels"]


def trial_rows(raw) -> dict:
    """crit3.py + export_trials.py: the criteria_pool_trials.json content from raw run outputs."""
    pools = defaultdict(list)
    for u, m, i, r, p, ys in _raw_trials(raw):
        with np.load(p) as a:
            row = {"trial": i, "pool": r["BApool"], "test": r["BAtest12"], **criteria(a, np.asarray(ys))}
        pools[(u, m)].append(row)
    for k in pools:                     # ties: the first maximum in trial order, never file-listing order
        pools[k].sort(key=lambda r: r["trial"])
    rows = [{"unit": u, "method": m, **{k: float(v) for k, v in r.items()}}
            for (u, m), rs in sorted(pools.items()) for r in rs]
    return {"source": "crit3.py criteria (verbatim), criteria-pool commit af71c57, seed 1",
            "criteria": list(CRITERIA), "rows": rows}


def picks(raw) -> tuple[dict, dict]:
    """mkpicks82.py: MIL-CREDA's cef pick per unit and the comparators' picks on FIGURE_UNIT."""
    rows = defaultdict(list)
    for u, m, i, r, p, _ in _raw_trials(raw):
        with np.load(p) as a:
            rows[(u, m)].append((i, cef(a["Pt"]), r))

    def pick(key):
        rs = sorted(rows[key], key=lambda x: x[0])
        c = np.array([x[1] for x in rs])
        i, cv, r = rs[int(np.flatnonzero(c == c.max())[0])]
        return {"trial": i, "cef": cv, "BApool": r["BApool"], "AUCpool": r["AUCpool"], "BAtest12": r["BAtest12"],
                "params": r["params"]}
    units = sorted({u for u, _ in rows})
    mil = {"source": PICK_SOURCE, "units": {u: pick((u, "MIL-CREDA")) for u in units if (u, "MIL-CREDA") in rows}}
    comp = {"source": PICK_SOURCE, "units": {FIGURE_UNIT: {m: pick((FIGURE_UNIT, m)) for m in COMPARATORS
                                                           if (FIGURE_UNIT, m) in rows}}}
    return mil, comp


def phase1_bases(bases=None, units=UNIT_KEYS) -> dict:
    """eval_bases.py: phase-1 BA/AUC of each frozen base on the unit's 120-subject pool and 12 target-test
    subjects, built as criteria_pool._setup builds them (no training; about one minute per unit)."""
    import torch
    from MIL_CREDA_EEG import criteria_pool as CP, xcompare as XC
    out = {}
    for u in units:
        s = CP._setup(u, CP.METHODS, CP.O.EVAL_SUBJECTS, str(bases or bases_dir()), 6)
        with torch.no_grad():
            pool = XC.snapshot(s["b"]["prepared"], s["pool"], s["cut"], s["dev"])
            test = XC.snapshot(s["b"]["prepared"], s["sides"]["target_test"], s["cut"], s["dev"])
        s["stack"].close()
        out[u] = {"BApool": pool["BA"], "AUCpool": pool["AUC"], "BAtest12": test["BA"], "AUCtest12": test["AUC"],
                  "cut": s["cut"]}
    return {"source": "eval_bases.py, protocol v82 final-rule bases; model = the prepared copy, frozen phase 1, "
                      "seed 1", "units": out}


# ------------------------------------------------------------------------------- derived summaries

def _pools(trials: dict) -> dict:
    pools = defaultdict(list)
    for r in trials["rows"]:
        pools[(r["unit"], r["method"])].append(r)
    for rows in pools.values():
        rows.sort(key=lambda r: r["trial"])       # ties: first maximum in trial order
    return pools


def _first_max(rows, c):
    s = np.array([r[c] for r in rows])
    return rows[int(np.flatnonzero(s == s.max())[0])]


def _holm(raw: dict) -> dict:
    holm, run_ = {}, 0.0
    for i, c in enumerate(sorted(raw, key=raw.get)):
        run_ = max(run_, min(1.0, raw[c] * (len(raw) - i)))
        holm[c] = run_
    return holm


def _cliff(a, b) -> float:
    return float(np.mean(np.sign(a[:, None] - b[None, :])))


def selection_summary(trials: dict) -> dict:
    """h3_pool.py (v77 copy, run on the v82 trials): each criterion's pick per (unit, method), mean BA and
    mean rank (over the criteria plus the oracle), unit-level Friedman, Wilcoxon cef vs each rival with
    monotone Holm over the 7 rivals, unpaired Cliff's delta."""
    from scipy import stats
    pools = _pools(trials)
    crit = list(NAMES)
    pick = {c: {} for c in crit + ["oracle"]}
    ranks = defaultdict(list)
    for key, rows in pools.items():
        ba = np.array([r["pool"] for r in rows])
        got = {c: _first_max(rows, c)["pool"] for c in crit}
        got["oracle"] = ba.max()
        for c, x in got.items():
            pick[c][key] = x
        for c, x in zip(got, stats.rankdata([-got[c] for c in got])):
            ranks[c].append(x)
    units = sorted({u for u, _ in pools})
    # rounded so float summation noise never decides whether two unit differences tie
    U = {c: np.round(np.array([np.mean([pick[c][(u, m)] for m in METHODS]) for u in units]), 12) for c in crit}
    fried = stats.friedmanchisquare(*[U[c] for c in crit]).pvalue
    rivals = [c for c in crit if c != "cef"]
    raw = {c: stats.wilcoxon(U["cef"], U[c]).pvalue for c in rivals}
    holm = _holm(raw)
    table = []
    for c in crit + ["oracle"]:
        row = {"criterion": NAMES.get(c, "Oracle (upper bound)"), "key": c,
               "methods": {m: round(float(np.mean([pick[c][(u, m)] for u in units])), 3) for m in METHODS},
               "meanBA": round(float(np.mean(list(pick[c].values()))), 3), "meanRank": round(float(np.mean(ranks[c])), 2)}
        if c in rivals:
            row.update(pRaw=float(raw[c]), pHolm=float(holm[c]), cliff=round(_cliff(U["cef"], U[c]), 2),
                       unitWins=int((U["cef"] > U[c]).sum()), unitLosses=int((U["cef"] < U[c]).sum()),
                       meanDiff=round(float((U["cef"] - U[c]).mean()), 3))
        table.append(row)
    table.sort(key=lambda r: r["meanRank"])
    # main.tex line 340: wins at d_C >= 1.5 and the mean gap at d_C <= 1 (unit-level BA)
    hi = [i for i, u in enumerate(units) if _key(u)[0] >= 1.5]
    lo = [i for i, u in enumerate(units) if _key(u)[0] <= 1.0]
    split = {NAMES[c]: {"winsHighD": int((U["cef"][hi] > U[c][hi]).sum()), "nHighD": len(hi),
                        "meanDiffLowD": float((U["cef"][lo] - U[c][lo]).mean())} for c in rivals}
    return {"protocol": "experiments-plan-experimental-mil-creda-sobre-balladeer-v82.md", "experiment": "S-seleccion",
            "hypothesis": "H3", "units": units, "nUnits": len(units), "pairs": len(pools), "alpha": 0.05,
            "friedmanP": float(fried), "table": table, "byClassEffect": split}


def methods_summary(trials: dict, phase1: dict) -> dict:
    """h1_pool24.py: cef picks each method's configuration per unit; ranks over the nine (eight methods +
    phase 1); Friedman, Nemenyi CD (q = 3.102), Wilcoxon MIL-CREDA vs each with monotone Holm over 8, Cliff's
    delta; MIL-CREDA vs phase 1 in units d >= 1. Also the oracle-pick BA per method."""
    from scipy import stats
    pools = _pools(trials)
    P1 = phase1["units"]
    units = sorted({u for u, _ in pools})
    names = list(METHODS) + ["Phase 1"]
    BA = np.round(np.array([[_first_max(pools[(u, m)], "cef")["pool"] for m in METHODS] + [P1[u]["BApool"]]
                            for u in units]), 12)
    R = np.array([stats.rankdata(-r) for r in BA])
    fr = stats.friedmanchisquare(*BA.T).pvalue
    cd = 3.102 * np.sqrt(9 * 10 / (6 * len(units)))
    raw = {j: stats.wilcoxon(BA[:, 0], BA[:, j]).pvalue for j in range(1, 9)}
    holm = _holm(raw)
    cl = {j: round(_cliff(BA[:, 0], BA[:, j]), 2) for j in range(1, 9)}
    sel = [i for i, u in enumerate(units) if not u.startswith("d0p6")]
    dd = BA[sel, 0] - BA[sel, 8]
    oracle = {m: round(float(np.mean([max(r["pool"] for r in pools[(u, m)]) for u in units])), 3) for m in METHODS}
    return {"protocol": "experiments-plan-experimental-mil-creda-sobre-balladeer-v82.md", "experiment": "A-mapa",
            "hypothesis": "H1", "units": units, "nUnits": len(units), "pick": "cef", "friedmanP": float(fr),
            "nemenyiCD": round(float(cd), 2),
            "table": [{"method": names[j], "meanRank": round(float(R[:, j].mean()), 2),
                       "meanBA": round(float(BA[:, j].mean()), 3)} for j in range(9)],
            "milcredaVs": [{"method": names[j], "p": float(raw[j]), "pHolm": float(holm[j]), "cliffDelta": cl[j],
                            "wins": int((BA[:, 0] > BA[:, j]).sum()), "ties": int((BA[:, 0] == BA[:, j]).sum()),
                            "losses": int((BA[:, 0] < BA[:, j]).sum())} for j in range(1, 9)],
            "phase1Cells": {"cells": "d >= 1, shift >= 0.5", "n": len(sel), "positive": int((dd > 0).sum()),
                            "zero": int((dd == 0).sum()), "negative": int((dd < 0).sum()),
                            "meanDiff": round(float(dd.mean()), 3), "p": float(stats.wilcoxon(dd).pvalue)},
            "oracleMeanBA": oracle}


def map_cells(trials: dict, phase1: dict) -> dict:
    """pool24_map_cells.json: cef pick BApool minus the frozen phase-1 base, mean of rotations 0 and 1, per
    (class effect, shift) cell, in make_map.py's layout (dBA[method][d index][shift index])."""
    pools = _pools(trials)
    P1 = phase1["units"]
    dBA, nrot = {}, {}
    for m in sorted(METHODS):
        g = defaultdict(list)
        for (u, mm), rows in sorted(pools.items()):       # rotation 0 before rotation 1
            if mm == m:
                g[_key(u)].append(_first_max(rows, "cef")["pool"] - P1[u]["BApool"])
        # np.round (scale, round half to even), the campaign's rounding: Python's round() differs from it on
        # five cells whose mean lies at a half of the third decimal
        dBA[m] = [[float(np.round(np.mean(g[(d, s)]), 3)) for s in SHIFTS] for d in D_LEVELS]
        nrot[m] = [[len(g[(d, s)]) for s in SHIFTS] for d in D_LEVELS]
    return {"source": "protocol v82, criteria pool, 24 units, seed 1, 100 random trials x 100 epochs, cef pick; BA at "
                      "source cut on 120-subject eval pool; phase 1 = frozen v82 base on same pool (eval_bases.py); "
                      "cef ties: first maximum in trial order",
            "value": "method (cef pick) minus phase 1, mean of rotations 0 and 1", "d": list(D_LEVELS),
            "shift": list(SHIFTS), "adhdBandRow": ADHD_BAND, "dBA": dBA, "nRotations": nrot}


def load_trials(summary=None) -> dict:
    with gzip.open(Path(summary or summary_dir()) / "criteria_pool_trials.json.gz", "rt") as f:
        return json.load(f)


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1))


def derive(summary=None) -> None:
    """Write selection.json, methods.json and map_cells.json from the trials and phase-1 files."""
    s = Path(summary or summary_dir())
    trials, phase1 = load_trials(s), json.loads((s / "phase1_bases.json").read_text())
    _dump(selection_summary(trials), s / "selection.json")
    _dump(methods_summary(trials, phase1), s / "methods.json")
    _dump(map_cells(trials, phase1), s / "map_cells.json")


def merge(raw=None, out=None, rebuild_phase1: bool = False, bases=None) -> Path:
    """Join the portions under `raw` (every `criteria-*.json` below it, from any number of accounts) into
    the stage summary. Without raw outputs it keeps the stored summary. The derived tables need all 24
    units x 8 methods; a partial set writes the trials and picks only and says what is missing."""
    raw = Path(raw) if raw else stage_dir() / "raw"
    out = Path(out) if out else summary_dir()
    out.mkdir(parents=True, exist_ok=True)
    if not raw.exists() or not any(raw.rglob("criteria-*.json")):
        print(f"no raw outputs under {raw}: keeping the stored summary in {out}")
        return out
    trials = trial_rows(raw)
    with gzip.open(out / "criteria_pool_trials.json.gz", "wt") as f:
        json.dump(trials, f)
    mil, comp = picks(raw)
    _dump(mil, out / "milcreda_picks.json")
    _dump(comp, out / "comparator_picks.json")
    if rebuild_phase1:
        _dump(phase1_bases(bases), out / "phase1_bases.json")
    elif not (out / "phase1_bases.json").exists():
        shutil.copy(summary_dir() / "phase1_bases.json", out / "phase1_bases.json")
    have = {(r["unit"], r["method"]) for r in trials["rows"]}
    missing = [(u, m) for u in UNIT_KEYS for m in METHODS if (u, m) not in have]
    if missing:
        print(f"{len(trials['rows'])} trials; {len(missing)} of 192 unit x method pools missing "
              f"(e.g. {missing[0]}): derived tables not written")
        return out
    derive(out)
    print(f"{len(trials['rows'])} trials over 192 pools -> {out}")
    return out


# ---------------------------------------------------------------------------------------- render

def _f3(v) -> str:
    return f"{v:.3f}"


def table_rows(summary=None) -> tuple[list, list]:
    """tab:selection-dev as rows of cells (strings, LaTeX bold marks applied) and as plain CSV rows."""
    s = Path(summary or summary_dir())
    sel = json.loads((s / "selection.json").read_text())
    met = json.loads((s / "methods.json").read_text())
    crit = [r for r in sel["table"] if r["key"] != "oracle"]
    oracle = next(r for r in sel["table"] if r["key"] == "oracle")
    phase1 = next(r for r in met["table"] if r["method"] == "Phase 1")
    best = {m: max(r["methods"][m] for r in crit) for m in METHODS}
    best_ba, best_rank = max(r["meanBA"] for r in crit), min(r["meanRank"] for r in crit)
    bold = lambda txt, on: f"\\textbf{{{txt}}}" if on else txt
    tex, csv = [], []
    for r in crit:
        ph = _f3(phase1["meanBA"]) if r["key"] == "cef" else ""
        pd = "--" if r["key"] == "cef" else f"{r['pHolm']:.3f} / {r['cliff']:+.2f}"
        tex.append([PAPER_LABEL[r["key"]]] + [bold(_f3(r["methods"][m]), r["methods"][m] == best[m]) for m in METHODS]
                   + [ph, bold(_f3(r["meanBA"]), r["meanBA"] == best_ba), pd,
                      bold(f"{r['meanRank']:.2f}", r["meanRank"] == best_rank)])
        csv.append([PAPER_LABEL[r["key"]]] + [_f3(r["methods"][m]) for m in METHODS] + [ph, _f3(r["meanBA"]),
                   "" if r["key"] == "cef" else f"{r['pHolm']:.3f}", "" if r["key"] == "cef" else f"{r['cliff']:+.2f}",
                   f"{r['meanRank']:.2f}"])
    tex.append(None)
    tex.append(["Oracle"] + [_f3(oracle["methods"][m]) for m in METHODS] + ["", _f3(oracle["meanBA"]), "--",
                                                                           f"{oracle['meanRank']:.2f}"])
    csv.append(["Oracle"] + [_f3(oracle["methods"][m]) for m in METHODS] + ["", _f3(oracle["meanBA"]), "", "",
                                                                           f"{oracle['meanRank']:.2f}"])
    rk = {r["method"]: r["meanRank"] for r in met["table"]}
    vs = {r["method"]: r for r in met["milcredaVs"]}
    best_m = min(rk.values())
    tex.append(None)
    tex.append(["Method rank"] + [bold(f"{rk[m]:.2f}", rk[m] == best_m) for m in list(METHODS) + ["Phase 1"]]
               + ["", "", ""])
    tex.append(["\\shortstack[l]{$p_{\\mathrm{Holm}}$ / $\\delta$\\\\(method)}", "--"]
               + [f"\\shortstack{{{vs[m]['pHolm']:.3f}\\\\{vs[m]['cliffDelta']:+.2f}}}" for m in list(COMPARATORS) + ["Phase 1"]]
               + ["", "", ""])
    csv.append(["Method rank"] + [f"{rk[m]:.2f}" for m in list(METHODS) + ["Phase 1"]] + ["", "", "", ""])
    csv.append(["pHolm (method)", ""] + [f"{vs[m]['pHolm']:.3f}" for m in list(COMPARATORS) + ["Phase 1"]] + ["", "", "", ""])
    csv.append(["delta (method)", ""] + [f"{vs[m]['cliffDelta']:+.2f}" for m in list(COMPARATORS) + ["Phase 1"]]
               + ["", "", "", ""])
    return tex, csv


def _table_tex(rows) -> str:
    head = ("Criterion & \\shortstack{MIL-\\\\CREDA} & \\shortstack{UDA-\\\\DDA} & \\shortstack{CCA-\\\\LSC} & CASUAL & COT & "
            "\\shortstack{CS+\\\\CCS} & CLUST & CLUDA & \\shortstack{Phase\\\\1} & \\shortstack{Mean\\\\BA} & "
            "\\shortstack{$p_{\\mathrm{Holm}}$ / $\\delta$\\\\(criterion)} & \\shortstack{Mean\\\\rank} \\\\")
    lines = ["\\begin{table*}[!t]", "\\centering", "\\footnotesize",
             "\\caption{Selection criteria (rows) and methods (columns) on the synthetic pool: BA of the picked "
             "configuration, mean over 24 units. $p_{\\mathrm{Holm}}$ / $\\delta$: against cef (criteria) and against "
             "MIL-CREDA (methods). Bold: best per column.}", "\\label{tab:selection-dev}",
             "\\setlength{\\tabcolsep}{1.5pt}",
             "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccccccccccc}", "\\toprule", head, "\\midrule"]
    lines += ["\\midrule" if r is None else " & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def draw_map(cells: dict, out_pdf: Path) -> dict:
    """paper/.paper-writing/figures/synthetic-map-methods/make_map.py, verbatim except the paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    d = cells
    assert set(list(COMPARATORS) + ["MIL-CREDA"]) == set(d["dBA"]), sorted(d["dBA"])
    panels = [("MIL-CREDA", np.array(d["dBA"]["MIL-CREDA"])),
              ("Comparators (mean)", np.mean([np.array(d["dBA"][m]) for m in COMPARATORS], axis=0))]
    FS = 10                                    # journal floor: every printed text >= 10 pt
    W_PT, PAD = 226.7, 0.02                    # one column: (486 - columnsep 1.15 cm) / 2, included unscaled
    with plt.rc_context({"font.size": FS, "pdf.fonttype": 42, "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.9), sharey=True, gridspec_kw={"wspace": 0.08})
        norm = TwoSlopeNorm(vmin=-0.12, vcenter=0.0, vmax=0.5)
        cmap = plt.get_cmap("RdBu")
        fmt = lambda v: ("−" if v < -0.005 else "") + f"{abs(v):.2f}".lstrip("0")
        for ax, (m, z) in zip(axes, panels):
            z = z[::-1]                              # top row = largest d
            ax.imshow(z, cmap=cmap, norm=norm, aspect="auto")
            for r in range(z.shape[0]):
                for c in range(z.shape[1]):
                    ax.text(c, r, fmt(z[r, c]), ha="center", va="center", fontsize=FS,
                            color="white" if abs(z[r, c]) > 0.3 else "black")
            ax.set_title(m, fontsize=FS, pad=3)
            band = len(d["d"]) - 1 - d["d"].index(d["adhdBandRow"])
            ax.add_patch(Rectangle((-0.5, band - 0.5), len(d["shift"]), 1, fill=False, ls="--", lw=1.0, ec="black"))
            ax.set_xticks(range(len(d["shift"])), [f"{s:g}" for s in d["shift"]])
            ax.set_yticks(range(len(d["d"])), [f"{v:g}" for v in d["d"][::-1]])
            ax.tick_params(length=2, width=0.6, pad=1.5, labelsize=FS)
            ax.set_xlabel("Shift", fontsize=FS, labelpad=1)
        axes[0].set_ylabel("Class effect", fontsize=FS, labelpad=1)
        for _ in range(8):                         # widen or narrow the canvas until the tight crop is W_PT wide
            w = fig.get_tightbbox().width + 2 * PAD
            if abs(w * 72 - W_PT) < 0.3:
                break
            fig.set_size_inches(fig.get_size_inches()[0] + W_PT / 72 - w, fig.get_size_inches()[1])
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02, transparent=True)
        fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=150)
        plt.close(fig)
    return {m: np.round(z, 3).tolist() for m, z in panels}


def render(summary_dir=None, out_dir=None) -> dict:
    """tables/selection-dev.{tex,csv} and figures/synthetic-map-methods.{pdf,png} from the summary only."""
    import csv as _csv
    s = Path(summary_dir) if summary_dir else globals()["summary_dir"]()
    o = Path(out_dir) if out_dir else stage_dir()
    (o / "tables").mkdir(parents=True, exist_ok=True)
    (o / "figures").mkdir(parents=True, exist_ok=True)
    tex, csv_rows = table_rows(s)
    (o / "tables" / "selection-dev.tex").write_text(_table_tex(tex))
    with open(o / "tables" / "selection-dev.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["Criterion", *METHODS, "Phase 1", "Mean BA", "pHolm (criterion)", "delta (criterion)", "Mean rank"])
        w.writerows(csv_rows)
    panels = draw_map(json.loads((s / "map_cells.json").read_text()), o / "figures" / "synthetic-map-methods.pdf")
    return {"table": str(o / "tables" / "selection-dev.tex"), "figure": str(o / "figures" / "synthetic-map-methods.pdf"),
            "panels": panels}


# ----------------------------------------------------------------------------------- paper checks

def paper_checks(summary_dir=None) -> list[tuple[str, float, float, float]]:
    """(name, obtained, paper, tolerance). Paper values quoted from the paper main.tex (MIL-CREDA/paper/main.tex) (line given);
    obtained values are rounded to the paper's printed precision, so the tolerance is 0."""
    s = Path(summary_dir) if summary_dir else globals()["summary_dir"]()
    sel = json.loads((s / "selection.json").read_text())
    met = json.loads((s / "methods.json").read_text())
    cells = json.loads((s / "map_cells.json").read_text())
    crit = {r["key"]: r for r in sel["table"]}
    mt = {r["method"]: r for r in met["table"]}
    vs = {r["method"]: r for r in met["milcredaVs"]}
    r3 = lambda v: round(float(v), 3)
    r2 = lambda v: round(float(v), 2)
    rows = [
        # main.tex line 340 (text of tab:selection-dev)
        ("SND mean BA (l.340)", r3(crit["SND"]["meanBA"]), 0.504, 0),
        ("cef mean BA (l.340)", r3(crit["cef"]["meanBA"]), 0.689, 0),
        ("oracle mean BA (l.340)", r3(crit["oracle"]["meanBA"]), 0.748, 0),
        ("cef mean rank (l.340)", r2(crit["cef"]["meanRank"]), 4.71, 0),
        ("criteria Friedman p (l.340)", r3(sel["friedmanP"]), 0.006, 0),
        ("cef vs ClassAMI pHolm (l.357)", r3(crit["ClassAMI"]["pHolm"]), 0.006, 0),
        ("cef vs BNM pHolm (l.355)", r3(crit["BNM"]["pHolm"]), 0.014, 0),
        ("cef vs IM pHolm (l.356)", r3(crit["IM"]["pHolm"]), 0.014, 0),
        ("cef vs Centroid pHolm (l.353)", r3(crit["centroid BA"]["pHolm"]), 0.079, 0),
        ("cef vs Source acc. pHolm (l.354)", r3(crit["source BA"]["pHolm"]), 0.079, 0),
        ("cef vs Entropy pHolm (l.358)", r3(crit["entropy"]["pHolm"]), 0.079, 0),
        ("cef vs SND pHolm (l.359)", r3(crit["SND"]["pHolm"]), 0.065, 0),
        ("min wins of cef at dC>=1.5 (l.340)", min(v["winsHighD"] for v in sel["byClassEffect"].values()), 10, 0),
        ("max wins of cef at dC>=1.5 (l.340)", max(v["winsHighD"] for v in sel["byClassEffect"].values()), 12, 0),
        ("max |gap to cef| at dC<=1 (l.340)", r3(max(abs(v["meanDiffLowD"]) for v in sel["byClassEffect"].values())),
         0.014, 0),
        # main.tex line 370 (ranking of the methods) and the table's method rows (l.363-364)
        ("MIL-CREDA mean rank (l.370)", r2(mt["MIL-CREDA"]["meanRank"]), 3.33, 0),
        ("MIL-CREDA mean BA (l.370)", r3(mt["MIL-CREDA"]["meanBA"]), 0.721, 0),
        ("UDA-DDA mean rank (l.370)", r2(mt["UDA-DDA"]["meanRank"]), 3.79, 0),
        ("UDA-DDA mean BA (l.370)", r3(mt["UDA-DDA"]["meanBA"]), 0.707, 0),
        ("Phase 1 mean BA (l.370)", r3(mt["Phase 1"]["meanBA"]), 0.576, 0),
        ("Phase 1 mean rank (l.363)", r2(mt["Phase 1"]["meanRank"]), 5.79, 0),
        ("methods Friedman p (l.370)", r3(met["friedmanP"]), 0.005, 0),
        ("Nemenyi CD (l.370)", r2(met["nemenyiCD"]), 2.45, 0),
        ("MIL-CREDA vs CCA-LSC pHolm (l.370)", r3(vs["CCA-LSC"]["pHolm"]), 0.013, 0),
        ("MIL-CREDA vs CLUST pHolm (l.370)", r3(vs["CLUST"]["pHolm"]), 0.007, 0),
        ("MIL-CREDA vs Phase 1 pHolm (l.370)", r3(vs["Phase 1"]["pHolm"]), 0.077, 0),
        ("MIL-CREDA oracle BA (l.370)", r3(crit["oracle"]["methods"]["MIL-CREDA"]), 0.781, 0),
    ]
    # main.tex line 380 (fig:synthetic-map): the four cells d_C >= 1.5 x shift in {1.5, 3}, and the rest
    mil = np.array(cells["dBA"]["MIL-CREDA"])
    comp = np.mean([np.array(cells["dBA"][m]) for m in COMPARATORS], axis=0)
    hi = np.zeros(mil.shape, bool)
    for i, d in enumerate(cells["d"]):
        for j, sh in enumerate(cells["shift"]):
            hi[i, j] = d >= 1.5 and sh >= 1.5
    rows += [("map: MIL-CREDA gain, dC>=1.5 & shift>=1.5 (l.380)", r2(mil[hi].mean()), 0.41, 0),
             ("map: comparators gain, dC>=1.5 & shift>=1.5 (l.380)", r2(comp[hi].mean()), 0.33, 0),
             ("map: MIL-CREDA above comparators in those 4 cells (l.380)", int((mil[hi] > comp[hi]).sum()), 4, 0),
             ("map: MIL-CREDA gain, other cells (l.380)", r3(mil[~hi].mean()), 0.015, 0),
             ("map: max |gain| in the ADHD band row (l.380)",
              float(np.max(np.abs(np.r_[mil[cells['d'].index(ADHD_BAND)], comp[cells['d'].index(ADHD_BAND)]])) <= 0.06),
              1.0, 0)]
    return rows
