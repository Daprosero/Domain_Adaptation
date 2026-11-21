"""Stage 04: the shared-pool protocol on EEG Data for ADHD/Control Children (paper Table
`tab:ieee-ranking` and the ADHD ranking and bootstrap numbers of Section `sec:res-adhd`).

Pipeline, as the paper ran it (protocol v82, Kaggle job family `ix`, commit d7a4ef3e of the
`ieee-pool-slim` branch, tag `archive/ieee-pool-slim`):

* 10 cross-subject units `ieee-r{0-4}h{0,1}` (rotation r of five subject folds, target-test
  half h) x 8 shards of the 100 trials = 80 CPU jobs `ix-r{r}h{h}-s{s}`, each running
  `python -m MIL_CREDA_EEG.ieee_pool run <unit> <shard>` (`ieee_pool.run`). This module calls that
  same entrypoint; it never re-implements training.
* Every method starts phase 2 from the unit's frozen seed-1 phase-1 base. The paper's bases are the
  ones of commit d7a4ef3e (`MIL-CREDA-EEG/Manifests/ieee_pool_bases` at that commit), copied into
  `Results/paper/04_adhd_pool/bases/`; the ones tracked today under `Manifests/ieee_pool_bases`
  are an older set (commit ce382b3) and do not reproduce the paper's phase-1 row.
* The data is the public Kaggle dataset `daprosero/mi-tdah-dataset` (`repro.ieee_root()`), and the
  preprocessing runs on the fly (`xpilot.ieee_bags`: 50 Hz notch, 4 s windows with 50 % overlap,
  at most 60 windows, 1-45 Hz band-pass, average reference; splits by
  `xpilot.cross_subject_sides`).

`merge` condenses the raw outputs (1.9 GB for the full campaign) to the per-trial fields the
analysis needs and recomputes the analysis of `Results/v82-ieee-pool/ieee_pool_tables.py` and
`ieee_bootstrap.py`; `render` writes the table from the stored summary only.
"""
from __future__ import annotations

import csv
import glob
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from MIL_CREDA_EEG import repro

STAGE = "04_adhd_pool"

#: `ieee_pool.UNITS`, `ieee_pool.SHARDS`, `criteria_pool.METHODS`, repeated here so that importing
#: this module (merge, render, tests) needs neither torch nor the training code.
IEEE_UNITS = tuple(f"ieee-r{r}h{h}" for r in range(5) for h in range(2))
SHARDS = 8
METHODS = ("MIL-CREDA", "UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS", "CLUST — variante source-free",
           "CLUDA")
#: the 80 (unit, shard) work items, in the order of the Kaggle jobs ix-r{r}h{h}-s{s}
UNITS = tuple((u, s) for u in IEEE_UNITS for s in range(SHARDS))

#: Parameters of every paper job. Source: `run.kwargs` of
#: scratchpad kaggle-jobs/ieee-slim/tools/kaggle/ix-r*h*-s*/run-config.json
#: (procs 4, shards 8, threads 1, unit_key, shard) and the defaults of `ieee_pool.run` at commit
#: d7a4ef3e that those kwargs leave in place (trials = criteria_pool.TRIALS = 100,
#: epochs = criteria_pool.EPOCHS2 = 100, every method of criteria_pool.METHODS, mid=True,
#: phase-2 seed ieee_pool.PHASE2_SEED = 1, base seed 1). Recorded again in every output JSON
#: (`trials`, `epochs2`, `phase2Seed`, `baseSeed`, `procs`, `threads`).
PARAMS = {"shards": SHARDS, "trials": 100, "epochs": 100, "procs": 4, "threads": 1, "mid": True,
          "methods": list(METHODS), "phase2Seed": 1, "baseSeed": 1, "accelerator": "cpu"}
#: the smoke run: one unit, one shard, trial 0 of each method, 2 epochs, one process
SMOKE = {"unit": "ieee-r0h0", "shard": 0, "trials": SHARDS, "epochs": 2, "procs": 1}

#: Table names: the pool's method keys -> the paper's row labels
LABELS = {"MIL-CREDA": "MIL-CREDA", "UDA-DDA": "UDA-DDA", "CCA-LSC": "CCA-LSC", "CASUAL": "CASUAL",
          "COT": "COT", "CS+CCS": "CS+CCS", "CLUST — variante source-free": "CLUST (source-free)",
          "CLUDA": "CLUDA", "phase 1": "Phase 1 (source only)"}
COLUMNS = list(METHODS) + ["phase 1"]
#: The unit order of the bootstrap's draws. `ieee_bootstrap.py` met the units in the unsorted
#: `glob` order of its analysis machine; the generator is shared across units, so this order fixes
#: the resamples. With it the bootstrap reproduces `Results/v82-ieee-pool/ieee_bootstrap.json`
#: exactly (sorted order gives the same observed value and interval to 4 decimals, and a
#: P(B <= 0) of 0.3978 instead of 0.4008; the paper prints only the value and the interval).
BOOTSTRAP_ORDER = ("ieee-r2h1", "ieee-r1h0", "ieee-r0h1", "ieee-r1h1", "ieee-r0h0", "ieee-r3h0", "ieee-r2h0",
                   "ieee-r3h1", "ieee-r4h1", "ieee-r4h0")
ARRAY_SIDES = ("target", "target_test", "source_test", "source")


def stage_dir() -> Path:
    return repro.results_dir(STAGE)


def summary_path(summary_dir=None) -> Path:
    return Path(summary_dir) if summary_dir else stage_dir() / "summary"


def bases_dir() -> Path:
    """The paper's ten frozen phase-1 bases (commit d7a4ef3e)."""
    return stage_dir() / "bases"


def smoke_dir() -> Path:
    """Where a smoke run writes by default: the system temp folder, never the repository."""
    import tempfile

    return Path(tempfile.gettempdir()) / "mil-creda-repro" / STAGE / "raw-smoke"


def job_name(unit: str, shard: int) -> str:
    return f"ix-{unit.removeprefix('ieee-')}-s{int(shard)}"


# ------------------------------------------------------------------ run

def run(split=None, smoke: bool = False, out=None, ieee_root=None, procs: int | None = None) -> list[dict]:
    """Run `repro.portion(UNITS, split)` with `ieee_pool.run`, one folder per (unit, shard) under
    `out` (default `Results/paper/04_adhd_pool/raw/`; `smoke_dir()` for a smoke run), laid out like the paper's `ieee-out/`.
    A finished trial is never rerun: `ieee_pool.run` resumes from its own JSON."""
    from MIL_CREDA_EEG import ieee_pool as I     # torch and the training code: imported on demand

    root = ieee_root or repro.ieee_root()
    out = Path(out) if out else (smoke_dir() if smoke else stage_dir() / "raw")
    done = []
    if smoke:
        work = out / job_name(SMOKE["unit"], SMOKE["shard"])
        done.append(I.run(SMOKE["unit"], SMOKE["shard"], SHARDS, SMOKE["trials"], METHODS, work, bases_dir(),
                          SMOKE["epochs"], SMOKE["procs"], 1, True, root))
        return done
    for unit, shard in repro.portion(UNITS, split):
        work = out / job_name(unit, shard)
        done.append(I.run(unit, shard, PARAMS["shards"], PARAMS["trials"], METHODS, work, bases_dir(),
                          PARAMS["epochs"], procs or PARAMS["procs"], PARAMS["threads"], PARAMS["mid"], root))
    return done


# ------------------------------------------------------------------ merge

def cef_counts(pt: np.ndarray) -> tuple[int, int]:
    """Predicted-class counts of the target adaptation bags (argmax of Pt), the only input of cef."""
    c = np.bincount(np.asarray(pt).argmax(1), minlength=2)
    return int(c[0]), int(c[1])


def cef(n0: int, n1: int) -> float:
    """Effective number of predicted classes, exactly as `ieee_pool_tables.cef` computes it."""
    h = np.array([n0, n1]) / (n0 + n1)
    h = h[h > 0]
    return float(np.exp(-(h * np.log(h)).sum()))


def phase1_subjects(bases=None) -> dict:
    """Per unit, the base's phase-1 prediction on the target-test half (subjects, labels,
    probabilities, cut), as `Results/v82-ieee-pool/phase1_subjects.json` holds it."""
    import torch

    from MIL_CREDA_EEG import spectral_trunk

    out = {}
    with spectral_trunk.installed():
        for u in IEEE_UNITS:
            b = torch.load(Path(bases or bases_dir()) / f"{u}-s1.pt", weights_only=False)
            p = b["predict"]
            out[u] = {"subjects": p["subjects"], "labels": p["labels"], "P": p["probabilities"],
                      "cut": float(b["cut"]), "tauAtt1": float(b["phase1"]["tauAtt1"]), "predictBA": p["BA"]}
    return out


def condense(raw, summary_dir=None, partial: bool = False) -> Path:
    """Join the portions under `raw` (any folder holding `*/ieeepool-*.json` and their array folders)
    into the condensed summary: trials.csv (one row per unit, method and trial), units.json (each
    unit's metadata and phase-1 prediction), ptest.npz (every trial's target-test probabilities)
    and pick_arrays.npz (all saved arrays of MIL-CREDA's cef and oracle picks, read by stage 06)."""
    files = sorted(glob.glob(str(Path(raw) / "*" / "ieeepool-*.json")))
    seen = {(json.loads(Path(f).read_text())["unitKey"], int(Path(f).stem.split("-sh")[1].split("of")[0])): f
            for f in files}
    missing = [job_name(u, s) for u, s in UNITS if (u, s) not in seen]
    if missing and not partial:
        raise FileNotFoundError(f"{len(missing)} of {len(UNITS)} portions missing under {raw}, "
                                f"e.g. {missing[:3]}; copy every account's raw folder there first")
    sd = summary_path(summary_dir)
    sd.mkdir(parents=True, exist_ok=True)
    rows, units, ptest = [], {}, defaultdict(dict)
    for (u, _s), f in sorted(seen.items()):
        d = json.loads(Path(f).read_text())
        meta = {k: v for k, v in d.items() if k not in ("methods", "shard", "procs", "threads")}
        meta.pop("baseCommit", None)
        units.setdefault(u, meta)
        arr = Path(f).with_suffix("")
        for m, v in d["methods"].items():
            for i, r in v["trials"].items():
                row = {"unit": u, "method": m, "trial": int(i), "status": "ok", "n0": "", "n1": "",
                       "BA": "", "AUC": "", "kappa": ""}
                if r.get("nonFinite") or "BAtest" not in r:
                    row["status"] = "nonFinite"
                else:
                    a = np.load(arr / f"{m.split(' ')[0]}-{i}.npz")
                    if not np.isfinite(a["Pt"]).all():
                        row["status"] = "PtNonFinite"
                    else:
                        row["n0"], row["n1"] = cef_counts(a["Pt"])
                    row.update(BA=r["BAtest"], AUC=r["AUCtest"], kappa=r["kappaTest"])
                    ptest[(u, m)][int(i)] = np.asarray(a["Ptest"], np.float32)
                rows.append(row)
    rows.sort(key=lambda r: (r["unit"], METHODS.index(r["method"]), r["trial"]))
    with open(sd / "trials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["unit"])
        w.writeheader()
        w.writerows(rows)
    packed = {}
    for (u, m), by_trial in ptest.items():
        n = len(units[u]["targetTestLabels"])
        x = np.full((PARAMS["trials"], n), np.nan, np.float32)
        for i, p in by_trial.items():
            x[i] = p
        packed[f"{u}|{m}"] = x
    np.savez_compressed(sd / "ptest.npz", **packed)
    (sd / "units.json").write_text(json.dumps(units, indent=1))
    if not missing:
        analyse(sd)
        _pick_arrays(seen, sd)
    return sd


def _pick_arrays(seen: dict, sd: Path) -> None:
    """Every array the pool saved for MIL-CREDA's cef and oracle picks (attention of the four sides,
    representations, probabilities), the input of the graph analysis (stage 06)."""
    S = json.loads((sd / "ieee_pool_summary.json").read_text())
    out = {}
    for tag in ("cef", "oracle"):
        for u in IEEE_UNITS:
            t = S[f"{tag}Picks"][f"{u}|MIL-CREDA"]["trial"]
            hits = [Path(f).with_suffix("") / f"MIL-CREDA-{t}.npz" for (uu, _), f in seen.items() if uu == u]
            hit = [h for h in hits if h.exists()]
            assert len(hit) == 1, (u, t, hit)
            a = np.load(hit[0])
            for k in a.files:
                out[f"{tag}|{u}|{k}"] = a[k]
    np.savez_compressed(sd / "pick_arrays.npz", **out)


def read_trials(summary_dir=None) -> dict:
    """{(unit, method): [row, ...]} of the finite trials, as `ieee_pool_tables.py` keeps them."""
    pools = defaultdict(list)
    with open(summary_path(summary_dir) / "trials.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            pools[(r["unit"], r["method"])].append(
                dict(trial=int(r["trial"]), cef=cef(int(r["n0"]), int(r["n1"])), BA=float(r["BA"]),
                     AUC=float(r["AUC"]), kappa=float(r["kappa"])))
    return pools


def _holm(raw: dict) -> dict:
    out, run_ = {}, 0.0
    for i, k in enumerate(sorted(raw, key=raw.get)):
        run_ = max(run_, min(1.0, raw[k] * (len(raw) - i)))
        out[k] = run_
    return out


def _table(src: dict, p1: dict, units: list) -> dict:
    """`ieee_pool_tables.table`: Friedman over the 9 columns with units as blocks, Wilcoxon of
    MIL-CREDA against each column, Holm over the 8 contrasts, Cliff's delta, W/T/L."""
    from scipy import stats

    get = lambda u, m, x: p1[u][x] if m == "phase 1" else src[(u, m)][x]
    BA = np.round(np.array([[get(u, m, "BA") for m in COLUMNS] for u in units]), 12)
    rk = np.mean([stats.rankdata(-r) for r in BA], axis=0)
    fr = stats.friedmanchisquare(*BA.T).pvalue
    raw = {m: stats.wilcoxon(BA[:, 0], BA[:, j]).pvalue for j, m in enumerate(COLUMNS) if j}
    ho = _holm(raw)
    out = []
    for j, m in enumerate(COLUMNS):
        auc = np.mean([get(u, m, "AUC") for u in units])
        ka = np.mean([get(u, m, "kappa") for u in units])
        cl = float(np.mean(np.sign(BA[:, 0][:, None] - BA[:, j][None, :])))
        w, l = int((BA[:, 0] > BA[:, j]).sum()), int((BA[:, 0] < BA[:, j]).sum())
        row = dict(method=LABELS[m], key=m, BA=round(float(BA[:, j].mean()), 3), AUC=round(float(auc), 3),
                   kappa=round(float(ka), 3), rank=round(float(rk[j]), 2), BAexact=float(BA[:, j].mean()))
        if j:
            row.update(pRaw=float(raw[m]), pHolm=float(ho[m]), cliff=round(cl, 2), WTL=f"{w}/{len(units) - w - l}/{l}")
        out.append(row)
    return dict(friedmanP=float(fr), rows=out)


def balanced_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean recall of the two classes (`sklearn.metrics.balanced_accuracy_score` for two classes
    that are both present, which the within-class bootstrap guarantees)."""
    return 0.5 * (float((yhat[y == 0] == 0).mean()) + float((yhat[y == 1] == 1).mean()))


def bootstrap(y_m_p: dict, resamples: int = 10000, seed: int = 0) -> dict:
    """`ieee_bootstrap.py`: paired bootstrap by subject of BA(MIL-CREDA cef pick) - BA(phase 1),
    subjects resampled with replacement within unit and class, mean over units, percentile 95 %
    interval. Same generator calls in the same order as the paper's script."""
    data = list(y_m_p.values())
    rng = np.random.default_rng(seed)
    obs = np.mean([balanced_accuracy(y, m) - balanced_accuracy(y, p) for y, m, p in data])
    B = np.empty(resamples)
    for b in range(resamples):
        diffs = []
        for y, m, p in data:
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (0, 1)])
            diffs.append(balanced_accuracy(y[idx], m[idx]) - balanced_accuracy(y[idx], p[idx]))
        B[b] = np.mean(diffs)
    lo, hi = np.percentile(B, [2.5, 97.5])
    return {"observed": round(float(obs), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "resamples": int(resamples), "pBootstrapLE0": float(np.mean(B <= 0))}


def analyse(summary_dir=None, resamples: int = 10000) -> dict:
    """Recompute `ieee_pool_summary.json` and `ieee_bootstrap.json` from the condensed trials."""
    sd = summary_path(summary_dir)
    units_meta = json.loads((sd / "units.json").read_text())
    P1 = json.loads((sd / "phase1_subjects.json").read_text())
    pools = read_trials(sd)
    units = sorted(units_meta)
    assert len(units) == 10 and len(pools) == 80, (len(units), len(pools))
    p1 = {u: units_meta[u]["phase1Predict"] for u in units}
    pick, orc = {}, {}
    for k, rows in pools.items():
        rows.sort(key=lambda r: r["trial"])
        s = np.array([r["cef"] for r in rows])
        pick[k] = rows[int(np.flatnonzero(s == s.max())[0])]     # first maximum of cef in trial order
        orc[k] = max(rows, key=lambda r: r["BA"])                 # best test BA (upper bound only)
    res = {"protocol": "experiments-plan-experimental-mil-creda-sobre-balladeer-v82.md", "experiment": "A-ieee",
           "units": units, "cef": _table(pick, p1, units), "oracle": _table(orc, p1, units),
           "cefPicks": {f"{u}|{m}": pick[(u, m)] for u, m in sorted(pick)},
           "oraclePicks": {f"{u}|{m}": orc[(u, m)] for u, m in sorted(orc)},
           "nFinite": {f"{u}|{m}": len(pools[(u, m)]) for u, m in sorted(pools)},
           "collapsedCefPicks": sum(1 for r in pick.values() if r["cef"] == 1.0)}
    (sd / "ieee_pool_summary.json").write_text(json.dumps(res, indent=1))
    pt = np.load(sd / "ptest.npz")
    data = {}
    for u in BOOTSTRAP_ORDER:
        meta, t = units_meta[u], pick[(u, "MIL-CREDA")]["trial"]
        assert meta["targetTestSubjects"] == P1[u]["subjects"], u
        y, cut = np.array(meta["targetTestLabels"]), float(meta["cut"])
        m = (pt[f"{u}|MIL-CREDA"][t] > cut).astype(int)
        p = (np.array(P1[u]["P"]) > cut).astype(int)
        assert abs(balanced_accuracy(y, p) - P1[u]["predictBA"]) < 1e-4, u
        assert abs(balanced_accuracy(y, m) - pick[(u, "MIL-CREDA")]["BA"]) < 1e-4, u
        data[u] = (y, m, p)
    boot = bootstrap(data, resamples)
    (sd / "ieee_bootstrap.json").write_text(json.dumps(boot, indent=1))
    return {"summary": res, "bootstrap": boot}


def merge(out=None, raw=None, partial: bool = False) -> Path:
    """Join the raw portions (default `Results/paper/04_adhd_pool/raw/`, where every account's
    `ix-*` folders are copied) into `out` (default `Results/paper/04_adhd_pool/summary/`), with the
    phase-1 subject predictions read from the paper's bases."""
    raw = Path(raw) if raw else stage_dir() / "raw"
    sd = summary_path(out)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "phase1_subjects.json").write_text(json.dumps(phase1_subjects(), indent=None))
    return condense(raw, sd, partial=partial)


# ------------------------------------------------------------------ render

def _num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def _delta(x: float) -> str:
    return f"$-${abs(x):.2f}" if x < 0 else f"+{x:.2f}"


def table_rows(S: dict) -> list[dict]:
    """The rows of `tab:ieee-ranking`, ordered by cef mean rank (stable on the pool's column order)."""
    cef_rows = {r["key"]: r for r in S["cef"]["rows"]}
    orc_rows = {r["key"]: r for r in S["oracle"]["rows"]}
    order = sorted(COLUMNS, key=lambda m: cef_rows[m]["rank"])
    return [{"key": m, "cef": cef_rows[m], "oracle": orc_rows[m]} for m in order]


def latex_table(S: dict) -> str:
    rows = table_rows(S)
    best = {"cr": min(r["cef"]["rank"] for r in rows), "cb": max(r["cef"]["BA"] for r in rows),
            "ca": max(r["cef"]["AUC"] for r in rows), "ck": max(r["cef"]["kappa"] for r in rows),
            "or": min(r["oracle"]["rank"] for r in rows), "ob": max(r["oracle"]["BA"] for r in rows)}
    bold = lambda s, on: f"\\textbf{{{s}}}" if on else s
    fr_o = S["oracle"]["friedmanP"]
    fr_o = "$<$0.001" if fr_o < 0.001 else _num(fr_o)
    lines = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize",
             r"\caption{Methods on ADHD/Control Children with the configuration picked by cef (left) and with the "
             rf"best test BA (right). Friedman $p$: {_num(S['cef']['friedmanP'])} with cef and {fr_o} with the best "
             r"test BA. Bold: best per metric.}",
             r"\label{tab:ieee-ranking}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccccc}", r"\toprule",
             r" & \multicolumn{5}{c}{cef (ten units)} & \multicolumn{3}{c}{Best test BA (ten units)} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
             r"Method & Rank & BA & AUC & $\kappa$ & $p_{\mathrm{Holm}}$ / $\delta$ & Rank & BA & "
             r"$p_{\mathrm{Holm}}$ / $\delta$ \\", r"\midrule"]
    for r in rows:
        c, o = r["cef"], r["oracle"]
        ctest = "--" if r["key"] == "MIL-CREDA" else f"{_num(c['pHolm'])} / {_delta(c['cliff'])}"
        otest = "--" if r["key"] == "MIL-CREDA" else f"{_num(o['pHolm'])} / {_delta(o['cliff'])}"
        cells = [LABELS[r["key"]], bold(_num(c["rank"], 2), c["rank"] == best["cr"]),
                 bold(_num(c["BA"]), c["BA"] == best["cb"]), bold(_num(c["AUC"]), c["AUC"] == best["ca"]),
                 bold(_num(c["kappa"]), c["kappa"] == best["ck"]), ctest,
                 bold(_num(o["rank"], 2), o["rank"] == best["or"]), bold(_num(o["BA"]), o["BA"] == best["ob"]), otest]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def render(summary_dir=None, out_dir=None) -> dict:
    """Write `tables/ieee-ranking.{tex,csv}` and `tables/ieee-bootstrap.csv` from the stored summary.
    This stage's paper item is a table only (it has no figure)."""
    sd = summary_path(summary_dir)
    S = json.loads((sd / "ieee_pool_summary.json").read_text())
    boot = json.loads((sd / "ieee_bootstrap.json").read_text())
    td = Path(out_dir) / "tables" if out_dir else stage_dir() / "tables"
    td.mkdir(parents=True, exist_ok=True)
    tex = latex_table(S)
    (td / "ieee-ranking.tex").write_text(tex)
    with open(td / "ieee-ranking.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "cef_rank", "cef_BA", "cef_AUC", "cef_kappa", "cef_pHolm", "cef_cliff", "cef_WTL",
                    "oracle_rank", "oracle_BA", "oracle_AUC", "oracle_kappa", "oracle_pHolm", "oracle_cliff",
                    "oracle_WTL"])
        for r in table_rows(S):
            c, o = r["cef"], r["oracle"]
            w.writerow([LABELS[r["key"]], c["rank"], c["BA"], c["AUC"], c["kappa"], c.get("pHolm", ""),
                        c.get("cliff", ""), c.get("WTL", ""), o["rank"], o["BA"], o["AUC"], o["kappa"],
                        o.get("pHolm", ""), o.get("cliff", ""), o.get("WTL", "")])
    with open(td / "ieee-bootstrap.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "observed", "ci95_low", "ci95_high", "resamples", "p_bootstrap_le0"])
        w.writerow(["BA(MIL-CREDA cef) - BA(phase 1)", boot["observed"], *boot["ci95"], boot["resamples"],
                    boot["pBootstrapLE0"]])
    return {"tex": td / "ieee-ranking.tex", "csv": td / "ieee-ranking.csv", "bootstrap": td / "ieee-bootstrap.csv",
            "table": tex}


# ------------------------------------------------------------------ paper checks

#: Values printed in paper/main.tex (line numbers of the version this module was written against):
#: Table tab:ieee-ranking, lines 457-465 (rank, BA, AUC, kappa, pHolm / delta with cef; rank, BA,
#: pHolm / delta with the best test BA); caption, line 449 (Friedman p); Section sec:res-adhd,
#: line 445 (bootstrap +0.008 [-0.058, 0.075]; gains 0.11 to 0.17; gaps 0.12 to 0.21).
PAPER_TABLE = {
    # method: (cef rank, BA, AUC, kappa, pHolm, delta, oracle rank, BA, pHolm, delta)
    "MIL-CREDA": (4.10, 0.663, 0.720, 0.325, None, None, 3.75, 0.820, None, None),
    "CCA-LSC": (4.60, 0.654, 0.648, 0.307, 1.000, 0.05, 5.10, 0.794, 1.000, 0.25),
    "phase 1": (4.60, 0.655, 0.717, 0.309, 1.000, 0.02, 8.10, 0.655, 0.062, 0.82),
    "CS+CCS": (4.65, 0.646, 0.716, 0.292, 1.000, 0.06, 5.70, 0.763, 0.438, 0.49),
    "CASUAL": (4.85, 0.638, 0.640, 0.276, 1.000, 0.13, 5.05, 0.796, 1.000, 0.16),
    "CLUDA": (5.40, 0.630, 0.706, 0.259, 1.000, 0.17, 3.45, 0.829, 1.000, -0.04),
    "UDA-DDA": (5.50, 0.623, 0.717, 0.245, 1.000, 0.25, 5.65, 0.764, 0.562, 0.40),
    "COT": (5.55, 0.605, 0.705, 0.209, 1.000, 0.30, 3.90, 0.819, 1.000, 0.05),
    "CLUST — variante source-free": (5.75, 0.604, 0.628, 0.206, 0.500, 0.28, 4.30, 0.808, 1.000, 0.17),
}
PAPER_TEXT = {"cef Friedman p": 0.849, "bootstrap observed": 0.008, "bootstrap CI low": -0.058,
              "bootstrap CI high": 0.075, "oracle gain over phase 1, min": 0.11,
              "oracle gain over phase 1, max": 0.17, "oracle - cef gap, min": 0.12, "oracle - cef gap, max": 0.21}


def paper_checks(summary_dir=None) -> list[tuple[str, float, float, float]]:
    """(name, obtained, paper, tolerance) rows. Obtained values are rounded to the precision the
    paper prints, so a match within tolerance 0 means the paper's number is reproduced."""
    sd = summary_path(summary_dir)
    S = json.loads((sd / "ieee_pool_summary.json").read_text())
    boot = json.loads((sd / "ieee_bootstrap.json").read_text())
    c = {r["key"]: r for r in S["cef"]["rows"]}
    o = {r["key"]: r for r in S["oracle"]["rows"]}
    rows = []
    for m, (cr, cb, ca, ck, cp, cd, orank, ob, op, od) in PAPER_TABLE.items():
        n = LABELS[m]
        rows += [(f"{n} cef rank", round(c[m]["rank"], 2), cr, 0.0), (f"{n} cef BA", round(c[m]["BA"], 3), cb, 0.0),
                 (f"{n} cef AUC", round(c[m]["AUC"], 3), ca, 0.0), (f"{n} cef kappa", round(c[m]["kappa"], 3), ck, 0.0),
                 (f"{n} best-test rank", round(o[m]["rank"], 2), orank, 0.0),
                 (f"{n} best-test BA", round(o[m]["BA"], 3), ob, 0.0)]
        if cp is not None:
            rows += [(f"{n} cef pHolm", round(c[m]["pHolm"], 3), cp, 0.0),
                     (f"{n} cef delta", round(c[m]["cliff"], 2), cd, 0.0),
                     (f"{n} best-test pHolm", round(o[m]["pHolm"], 3), op, 0.0),
                     (f"{n} best-test delta", round(o[m]["cliff"], 2), od, 0.0)]
    rows.append(("cef Friedman p", round(S["cef"]["friedmanP"], 3), PAPER_TEXT["cef Friedman p"], 0.0))
    # the caption prints "<0.001": checked as obtained in [0, 0.001]
    rows.append(("best-test Friedman p (paper: <0.001)", S["oracle"]["friedmanP"], 0.0, 0.001))
    rows += [("bootstrap observed", round(boot["observed"], 3), PAPER_TEXT["bootstrap observed"], 0.0),
             ("bootstrap CI low", round(boot["ci95"][0], 3), PAPER_TEXT["bootstrap CI low"], 0.0),
             ("bootstrap CI high", round(boot["ci95"][1], 3), PAPER_TEXT["bootstrap CI high"], 0.0)]
    gain = [o[m]["BA"] - o["phase 1"]["BA"] for m in METHODS]
    gap = [o[m]["BA"] - c[m]["BA"] for m in METHODS]
    rows += [("oracle gain over phase 1, min", round(min(gain), 2), PAPER_TEXT["oracle gain over phase 1, min"], 0.0),
             ("oracle gain over phase 1, max", round(max(gain), 2), PAPER_TEXT["oracle gain over phase 1, max"], 0.0),
             ("oracle - cef gap, min", round(min(gap), 2), PAPER_TEXT["oracle - cef gap, min"], 0.0),
             ("oracle - cef gap, max", round(max(gap), 2), PAPER_TEXT["oracle - cef gap, max"], 0.0)]
    return rows


def install_bases(source) -> Path:
    """Copy ten seed-1 bases (`ieee-r*h*-s1.pt`) into `bases_dir()`."""
    d = bases_dir()
    d.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(source).glob("ieee-r*h*-s1.pt")):
        shutil.copy2(f, d / f.name)
    return d
