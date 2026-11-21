"""Stage 03, synthetic mechanisms: the aggregation and adaptation-term ablations of MIL-CREDA on the
24 shared-pool synthetic units, the attention read-out and the own-pair alignment (paper:
Tables tab:synthetic-attention-variants and tab:synthetic-term-ablation, Figures
fig:synthetic-attention-signal and fig:synthetic-alignment, and the numbers of their paragraphs).

Pipeline. Every unit is one call to `MIL_CREDA_EEG.pool_aggregation.run`, the entrypoint of the paper's
Kaggle jobs `am-<unit>` (tools/kaggle/am-*/run-config.json: module MIL_CREDA_EEG.pool_aggregation,
function run, kwargs procs=4, threads=1, unit_key=<unit>; commit d75c6c2, CPU, branch ieee-pool-slim).
Everything else is the entrypoint's own default: all eight variants (`pool_aggregation.ALL`), 100
phase-2 epochs (`criteria_pool.EPOCHS2`), 120 evaluation subjects (`oracle_screen.EVAL_SUBJECTS`) and,
on the figure unit d1p5-s1p5-r0 only, the UDA-DDA run at its own cef pick.

Inputs (tracked here under `inputs/`, because this branch's `Manifests/` holds older ones):
* `inputs/milcreda_picks.json` and `inputs/comparator_picks.json`: the v82 cef picks the campaign
  trained (`Manifests/pool_aggregation/*` at commit d75c6c2, identical to tag archive/ieee-pool-slim,
  built by Results/v82-pool24/mkpicks82.py);
* `inputs/bases/<unit>-s1.pt`: the 24 frozen v82 final-rule phase-1 bases those jobs loaded
  (`Manifests/oracle_bases/*` at d75c6c2; their `commit` field equals each run's `baseCommit`).

Outputs: `raw/am-<unit>/` per unit (as the Kaggle jobs wrote them), then `merge` writes `summary/`
(the 24 aggregation records gzipped, the figure unit's attention and representation sidecars, the
signal excerpt and UMAP embedding of the two figures, and `statistics.json`, the port of the
campaign's Results/v82-mechanisms/mech_pool24.py), and `render` draws `figures/` and `tables/` from
`summary/` alone.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from MIL_CREDA_EEG import repro

STAGE = "03_synthetic_mechanisms"
FIGURE_UNIT = "d1p5-s1p5-r0"          # pool_aggregation.FIGURE_UNIT
FIGURE_METHOD = "UDA-DDA"             # pool_aggregation.FIGURE_METHOD
#: the paper jobs' kwargs (tools/kaggle/am-<unit>/run-config.json, run.kwargs)
RUN_KWARGS = {"procs": 4, "threads": 1}
#: smoke: one non-figure unit, two variants, one epoch, six evaluation subjects, one process
SMOKE_KWARGS = {"variants": ("own", "mean"), "epochs": 1, "eval_subjects": 6, "procs": 1, "threads": 1,
                "figure": False}

AGG = ["own", "own without consensus", "published attention", "gated attention", "mean", "sum"]
TER = ["full", "global only", "local only", "no adaptation"]
ATTENTION = AGG[:4]
LABELS = {"own": "Rel.\\ + consensus", "own without consensus": "Relevance only",
          "published attention": "Attention (Ilse)", "gated attention": "Gated (Ilse)", "mean": "Mean", "sum": "Sum",
          "no adaptation": "None", "global only": "Global only", "local only": "Local only", "full": "Global + local"}
STATES = ("start", "fullMid", "fullEnd", "globalOnlyEnd")
PDF_META = {"CreationDate": None}      # byte-stable PDFs: a re-render leaves the tracked files unchanged


def stage_dir() -> Path:
    return repro.results_dir(STAGE)


def inputs_dir() -> Path:
    return stage_dir() / "inputs"


def _picks() -> dict:
    return json.loads((inputs_dir() / "milcreda_picks.json").read_text())


#: the 24 units, in a fixed order (sorted unit keys of the v82 picks)
UNITS: list[str] = sorted(json.loads((repro.eeg_dir() / "Results" / "paper" / STAGE / "inputs"
                                      / "milcreda_picks.json").read_text())["units"])


def d_value(unit: str) -> float:
    return float(unit.split("-")[0][1:].replace("p", "."))


# ----------------------------------------------------------------------------------------- run
def run(split=None, smoke: bool = False, out=None) -> list[dict]:
    """Train this portion's units with `pool_aggregation.run` at the paper jobs' settings; each unit
    writes `<out>/am-<unit>/` (default `raw/`; for `smoke`, a fresh temporary folder outside the
    repository, so a wiring check never mixes with real results). A rerun resumes: the entrypoint
    skips variants whose rows are already on disk."""
    import tempfile

    from MIL_CREDA_EEG import pool_aggregation as PA

    picks, comparators = inputs_dir() / "milcreda_picks.json", inputs_dir() / "comparator_picks.json"
    bases = inputs_dir() / "bases"
    units = [UNITS[0]] if smoke else repro.portion(UNITS, split)
    out = Path(out) if out else (Path(tempfile.mkdtemp(prefix="stage03-smoke-")) if smoke else stage_dir() / "raw")
    kwargs = SMOKE_KWARGS if smoke else RUN_KWARGS
    done = []
    for u in units:
        r = PA.run(u, workdir=out / f"am-{u}", bases=str(bases), picks=str(picks),
                   comparator_picks=str(comparators), **kwargs)
        done.append({"unit": u, "dir": str(out / f"am-{u}"), **r})
    return done


# --------------------------------------------------------------------------------------- merge
def merge(out=None, raw=None) -> Path:
    """Join every portion's `am-<unit>/` folders (copied into one `raw/` folder, or the campaign's
    agg-out/) into `summary/`; refuses unless all 24 units are present."""
    raw = Path(raw) if raw else stage_dir() / "raw"
    out = Path(out) if out else stage_dir() / "summary"
    files = {u: raw / f"am-{u}" / f"aggregation-{u}.json" for u in UNITS}
    missing = [u for u, f in files.items() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(UNITS)} units have no aggregation record under {raw}, "
                                f"e.g. {missing[:3]}")
    (out / "aggregation").mkdir(parents=True, exist_ok=True)
    for u, f in files.items():
        with gzip.open(out / "aggregation" / f"aggregation-{u}.json.gz", "wt") as g:
            g.write(f.read_text())
    fu = out / "figure-unit"
    fu.mkdir(exist_ok=True)
    for name in ("attention", "umap"):
        shutil.copyfile(raw / f"am-{FIGURE_UNIT}" / f"{name}-{FIGURE_UNIT}.npz", fu / f"{name}-{FIGURE_UNIT}.npz")
    signal_excerpt(fu / f"attention-{FIGURE_UNIT}.npz", fu / "signal-excerpt.npz")
    umap_embedding(fu / f"umap-{FIGURE_UNIT}.npz", fu / "umap-embedding.npz")
    stats = statistics(out)
    (out / "statistics.json").write_text(json.dumps(stats, indent=1))
    return out


def load_records(summary_dir=None) -> dict:
    d = Path(summary_dir) if summary_dir else stage_dir() / "summary"
    recs = {}
    for u in UNITS:
        with gzip.open(d / "aggregation" / f"aggregation-{u}.json.gz", "rt") as g:
            recs[u] = json.load(g)
    return recs


def _pick_subject(z) -> int:
    """attfig_ideal.py's rule, fixed before reading any weight: the first ADHD evaluation subject, in
    subject order, with class-event windows and artifacts of at least two types."""
    lab, off, typ, ev = z["labels"], z["offsets"], z["artifactType"], z["event"]
    return next(i for i in range(len(z["subjects"])) if lab[i] == 1 and ev[off[i]:off[i + 1]].any()
                and len(set(typ[off[i]:off[i + 1]]) - {"none"}) >= 2)


EXCERPT_WINDOWS = 15                   # attfig_ideal.py: L = 15 consecutive 2-s hops


def signal_excerpt(attention_npz, out_npz) -> None:
    """The figure subject's AF3 and T7 traces over the excerpt of attfig_ideal.py (the first 30-s span
    holding a blink, a muscle artifact and a class event), rebuilt from the simulator with the unit's
    frozen base; refuses when the rebuilt masks differ from the Kaggle run's sidecar."""
    from MIL_CREDA_EEG import criteria_pool as C, eeg_sim as E, oracle_screen as O

    z = np.load(attention_npz)
    i = _pick_subject(z)
    s = slice(z["offsets"][i], z["offsets"][i + 1])
    subj, types, evs, art = str(z["subjects"][i]), list(z["artifactType"][s]), list(z["event"][s]), z["artifact"][s]
    st = C._setup(FIGURE_UNIT, ["MIL-CREDA"], O.EVAL_SUBJECTS, str(inputs_dir() / "bases"), 1)
    try:
        bag = next(b for b in st["pool"] if b.subject == subj)
        m = E.SIM_MASKS[subj]
        if (list(np.asarray(m["event"], bool)) != [bool(x) for x in evs]
                or list(np.asarray(m["artifact"], bool)) != [bool(x) for x in art]):
            raise ValueError("the simulated masks differ from the Kaggle run's")
        W = bag.windows.numpy()
    finally:
        st["stack"].close()
    K, hop, L = W.shape[0], W.shape[2] // 2, EXCERPT_WINDOWS
    start = next(k for k in range(K - L + 1) if any("blink" in t for t in types[k:k + L])
                 and any("muscle" in t for t in types[k:k + L]) and any(evs[k:k + L]))
    names = list(E.EMOTIV)
    sig = np.concatenate([W[k, :, :hop] for k in range(start, start + L)], axis=1)
    np.savez_compressed(out_npz, subject=subj, start=start, hop=hop, rate=E.RATE, windows=K,
                        AF3=sig[names.index("AF3")].astype(np.float32), T7=sig[names.index("T7")].astype(np.float32))


def umap_embedding(umap_npz, out_npz) -> None:
    """alignfig82.py / alignfig_col.py's embedding: one UMAP (n_neighbors=15, min_dist=0.3,
    random_state=0) fitted on the four states stacked (target test bags, then source bags)."""
    import umap

    z = np.load(umap_npz)
    tt = z["split"] == "target_test"
    Z = np.concatenate([np.concatenate([z[f"{k}:target"][tt], z[f"{k}:source"]])
                        for k in ("start", "fullMid", "fullEnd", "globalOnlyEnd")])
    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(Z)
    np.savez_compressed(out_npz, embedding=emb, umap_version=umap.__version__)


# ---------------------------------------------------------------------------------- statistics
def _holm(raw: dict) -> dict:
    out, run_ = {}, 0.0
    for i, k in enumerate(sorted(raw, key=raw.get)):
        run_ = max(run_, min(1.0, raw[k] * (len(raw) - i)))
        out[k] = run_
    return out


def _cliff(a, b) -> float:
    return float(np.mean(np.sign(a[:, None] - b[None, :])))


def _family(recs, names, get, ref) -> dict:
    from scipy import stats

    units = UNITS
    big = np.array([d_value(u) >= 1.5 for u in units])
    V = {k: np.round(np.array([get(recs[u], k) for u in units]), 12) for k in names}
    M = np.column_stack([V[k] for k in names])
    rk = np.mean([stats.rankdata(-r) for r in M], axis=0)
    raw = {k: float(stats.wilcoxon(V[ref], V[k]).pvalue) for k in names if k != ref}
    ho = _holm(raw)
    rows = {}
    for i, k in enumerate(names):
        w, l = int((V[ref] > V[k]).sum()), int((V[ref] < V[k]).sum())
        rows[k] = {"BA": float(V[k].mean()), "BAdHigh": float(V[k][big].mean()), "BAdLow": float(V[k][~big].mean()),
                   "rank": float(rk[i]), "perUnit": [float(x) for x in V[k]]}
        if k != ref:
            rows[k].update(pRaw=raw[k], pHolm=ho[k], cliff=_cliff(V[ref], V[k]), wins=w, ties=len(units) - w - l,
                           losses=l)
    return {"friedmanP": float(stats.friedmanchisquare(*M.T).pvalue), "reference": ref, "rows": rows}


def _term(d, k):
    return d["families"]["aggregation"]["own"]["BApool"] if k == "full" else d["families"]["terms"][k]["BApool"]


def _attention(recs) -> dict:
    from scipy import stats

    big = np.array([d_value(u) >= 1.5 for u in UNITS])
    out = {}
    for k in ATTENTION:
        res = {}
        for m in ("artifactAUC", "eventAUC"):
            x = np.array([np.nanmean([s[m] for s in recs[u]["families"]["aggregation"][k]["attention"]["subjects"]
                                      if s[m] is not None]) for u in UNITS])
            res[m] = (x, float(stats.wilcoxon(x - 0.5, alternative="greater").pvalue))
        ho = _holm({m: res[m][1] for m in res})
        out[k] = {m: {"mean": float(x.mean()), "dHigh": float(x[big].mean()), "dLow": float(x[~big].mean()),
                      "p": p, "pHolm": ho[m], "unitsAbove": int((x > 0.5).sum())} for m, (x, p) in res.items()}
    return out


def _own_pair(recs) -> dict:
    from scipy import stats

    big = np.array([d_value(u) >= 1.5 for u in UNITS])
    S = {k: np.array([np.mean(recs[u]["families"]["ownPair"]["states"][k]["distances"]) for u in UNITS])
         for k in ("start", "fullEnd", "globalOnlyEnd")}
    closer = S["fullEnd"] < S["globalOnlyEnd"]
    return {**{k: {"mean": float(v.mean()), "median": float(np.median(v)), "dHigh": float(v[big].mean()),
                   "dLow": float(v[~big].mean())} for k, v in S.items()},
            "closerUnits": int(closer.sum()), "closerDHigh": int(closer[big].sum()), "closerDLow": int(closer[~big].sum()),
            "pLess": float(stats.wilcoxon(S["fullEnd"], S["globalOnlyEnd"], alternative="less").pvalue),
            "ratioMedian": float(np.median(S["fullEnd"] / S["globalOnlyEnd"]))}


def _figure_unit(d: Path) -> dict:
    """Weights of MIL-CREDA (own) and UDA-DDA on the figure unit: on the excerpt subject and over the
    evaluation subjects (paired Wilcoxon over subjects)."""
    from scipy import stats

    z = np.load(d / "figure-unit" / f"attention-{FIGURE_UNIT}.npz")
    off, lab, ev, art = z["offsets"], z["labels"], z["event"], z["artifact"]
    names = {"MIL-CREDA": "own", "UDA-DDA": FIGURE_METHOD}
    per = {}
    for nm, key in names.items():
        w = z[f"weights[{key}]"]
        r = {k: [] for k in ("event", "eventClean", "eventArtifact", "artifact", "uniformEvent", "uniformArtifact")}
        for i in range(len(lab)):
            s = slice(off[i], off[i + 1])
            wi, e, a = w[s], ev[s], art[s]
            if lab[i] == 1:
                r["event"].append(wi[e].sum()); r["eventClean"].append(wi[e & ~a].sum())
                r["eventArtifact"].append(wi[e & a].sum()); r["uniformEvent"].append(e.mean())
            if a.any():
                r["artifact"].append(wi[a].sum()); r["uniformArtifact"].append(a.mean())
        per[nm] = {k: np.array(v, float) for k, v in r.items()}
    over = {"nEventSubjects": len(per["MIL-CREDA"]["event"]), "nArtifactSubjects": len(per["MIL-CREDA"]["artifact"]),
            "uniformEvent": float(per["MIL-CREDA"]["uniformEvent"].mean()),
            "uniformArtifact": float(per["MIL-CREDA"]["uniformArtifact"].mean())}
    for k in ("event", "eventClean", "eventArtifact", "artifact"):
        over[k] = {nm: float(per[nm][k].mean()) for nm in names}
        over[k]["pWilcoxon"] = float(stats.wilcoxon(per["MIL-CREDA"][k], per["UDA-DDA"][k]).pvalue)
    x = np.load(d / "figure-unit" / "signal-excerpt.npz")
    i = _pick_subject(z)
    s = slice(off[i], off[i + 1])
    e = ev[s]
    start, hop, rate = int(x["start"]), int(x["hop"]), float(x["rate"])
    first = start + next(k for k in range(EXCERPT_WINDOWS) if e[start + k])
    while e[first + 1]:                # the window right after the excerpt's first run of event windows
        first += 1
    one = {"subject": str(z["subjects"][i]), "excerptStartSeconds": start * hop / rate, "windows": int(len(e)),
           "uniformOnEvents": float(e.mean()), "afterFirstEventsSeconds": (first + 1) * hop / rate}
    for nm, key in names.items():
        w = z[f"weights[{key}]"][s]
        one[nm] = {"weightOnEvents": float(w[e].sum()), "weightAfterFirstEvents": float(w[first + 1])}
    return {"overSubjects": over, "excerptSubject": one}


def _alignment(d: Path) -> dict:
    """alignfig82.py's highlighted pairs: the first target test subject of each class (by name)."""
    z = np.load(d / "figure-unit" / f"umap-{FIGURE_UNIT}.npz")
    tt = z["split"] == "target_test"
    tnames, yt = list(z["subjects"][tt]), z["labels"][tt]
    pick = [sorted([s for s, y in zip(tnames, yt) if y == c])[0] for c in (0, 1)]
    allt = list(z["subjects"])
    dist = lambda k, s: float(np.linalg.norm(z[f"{k}:target"][allt.index(s)] - z[f"{k}:pair"][allt.index(s)]))
    return {"highlighted": pick, "distances": {s: {k: dist(k, s) for k in STATES} for s in pick}}


def statistics(summary_dir=None) -> dict:
    """Every number of the stage, from `summary/` alone (the port of mech_pool24.py)."""
    d = Path(summary_dir) if summary_dir else stage_dir() / "summary"
    recs = load_records(d)
    return {
        "units": UNITS,
        "aggregation": _family(recs, AGG, lambda r, k: r["families"]["aggregation"][k]["BApool"], "own"),
        "terms": _family(recs, TER, _term, "full"),
        "pickBApool": float(np.mean([recs[u]["pickBApool"] for u in UNITS])),
        "notReproducing": [u for u in UNITS if not recs[u]["reproduces"]],
        "attention": _attention(recs),
        "ownPair": _own_pair(recs),
        "figureUnit": _figure_unit(d),
        "alignment": _alignment(d),
        "epochs2": int(recs[FIGURE_UNIT]["epochs2"]),
        "trainingSeconds": float(sum(v.get("seconds", 0.0) for u in UNITS
                                     for fam in ("aggregation", "terms", "figure")
                                     for v in (recs[u]["families"].get(fam) or recs[u].get(fam) or {}).values()
                                     if isinstance(v, dict))),
    }


def load_statistics(summary_dir=None) -> dict:
    d = Path(summary_dir) if summary_dir else stage_dir() / "summary"
    return json.loads((d / "statistics.json").read_text())


# -------------------------------------------------------------------------------------- render
def _b(x: str, bold: bool) -> str:
    return f"\\textbf{{{x}}}" if bold else x


def tables(st: dict, out_dir: Path) -> dict:
    """The two tables as LaTeX (the paper's layout) and CSV (full precision)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ag, at = st["aggregation"]["rows"], st["attention"]
    best_ba = max(ag[k]["BA"] for k in AGG); best_rank = min(ag[k]["rank"] for k in AGG)
    best_art = max(at[k]["artifactAUC"]["mean"] for k in ATTENTION)
    best_ev = max(at[k]["eventAUC"]["mean"] for k in ATTENTION)
    lines, csv = [], ["aggregation,BA,pHolm,cliff,rank,artifactAUC,artifactAUC_pHolm,eventAUC,eventAUC_pHolm"]
    for k in AGG:
        r = ag[k]
        ph = "ref." if k == "own" else f"{r['pHolm']:.2f}/{r['cliff']:+.2f}"
        cells = [_b(f"{r['BA']:.2f}", r["BA"] == best_ba), ph, _b(f"{r['rank']:.2f}", r["rank"] == best_rank)]
        if k in at:
            for m, best in (("artifactAUC", best_art), ("eventAUC", best_ev)):
                a = at[k][m]
                cells.append(_b(f"{a['mean']:.2f}", a["mean"] == best) + ("$^{*}$" if a["pHolm"] < 0.05 else ""))
            csv.append(",".join(str(x) for x in (k, r["BA"], r.get("pHolm", ""), r.get("cliff", ""), r["rank"],
                                                 at[k]["artifactAUC"]["mean"], at[k]["artifactAUC"]["pHolm"],
                                                 at[k]["eventAUC"]["mean"], at[k]["eventAUC"]["pHolm"])))
        else:
            cells += ["--", "--"]
            csv.append(",".join(str(x) for x in (k, r["BA"], r.get("pHolm", ""), r.get("cliff", ""), r["rank"], "", "", "", "")))
        lines.append(f"{LABELS[k]} & " + " & ".join(cells) + " \\\\")
    att = ("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lccccc}\n\\toprule\n"
           "Aggregation & BA & $p_{\\mathrm{Holm}}$/$\\delta$ & Rank & \\shortstack{Artif.\\\\AUC} & "
           "\\shortstack{Event\\\\AUC} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular*}\n")
    (out_dir / "synthetic-attention-variants.tex").write_text(att)
    (out_dir / "synthetic-attention-variants.csv").write_text("\n".join(csv) + "\n")
    te = st["terms"]["rows"]; best = max(te[k]["BA"] for k in TER)
    order = ["no adaptation", "global only", "local only", "full"]
    lines = [f"{LABELS[k]} & {_b(f'{te[k]["BA"]:.3f}', te[k]['BA'] == best)} & "
             f"{'ref.' if k == 'full' else format(te[k]['pHolm'], '.3f')} \\\\" for k in order]
    term = ("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcc}\n\\toprule\n"
            "MIL-CREDA adaptation terms & BA & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular*}\n")
    (out_dir / "synthetic-term-ablation.tex").write_text(term)
    csv = ["terms,BA,BA_dHigh,BA_dLow,rank,pHolm"] + [
        ",".join(str(x) for x in (k, te[k]["BA"], te[k]["BAdHigh"], te[k]["BAdLow"], te[k]["rank"], te[k].get("pHolm", "")))
        for k in order]
    (out_dir / "synthetic-term-ablation.csv").write_text("\n".join(csv) + "\n")
    return {"attention": att, "terms": term}


def _fit_width(fig, w_pt: float, pad: float, rect) -> None:
    """The figure scripts' loop: widen or narrow the canvas until the tight crop is w_pt wide."""
    for _ in range(8):
        fig.tight_layout(rect=rect(fig))
        w = fig.get_tightbbox().width + 2 * pad
        if abs(w * 72 - w_pt) < 0.3:
            break
        fig.set_size_inches(fig.get_size_inches()[0] + w_pt / 72 - w, fig.get_size_inches()[1])


def attention_figure(summary_dir: Path, out_pdf: Path) -> None:
    """Port of paper/.paper-writing/figures/synthetic-attention-signal/attfig_col.py (the one-column
    version printed in the paper), drawn from the stored sidecar and signal excerpt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    z = np.load(summary_dir / "figure-unit" / f"attention-{FIGURE_UNIT}.npz")
    x = np.load(summary_dir / "figure-unit" / "signal-excerpt.npz")
    i = _pick_subject(z)
    s = slice(z["offsets"][i], z["offsets"][i + 1])
    wm, wu, types, evs = z["weights[own]"][s], z[f"weights[{FIGURE_METHOD}]"][s], list(z["artifactType"][s]), list(z["event"][s])
    K = len(wm); wi = np.asarray(evs, float); wi = wi / wi.sum()
    start, hop, rate, L = int(x["start"]), int(x["hop"]), float(x["rate"]), EXCERPT_WINDOWS
    idx = list(range(start, start + L))
    sig = {"AF3": x["AF3"], "T7": x["T7"]}
    t = start * hop / rate + np.arange(len(sig["AF3"])) / rate
    W_PT, PAD = 226.7, 0.02
    with plt.rc_context({"font.family": "serif", "font.size": 10}):
        fig, axes = plt.subplots(2, 1, figsize=(3.15, 3.3), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})

        def shade(ax):
            for k in idx:
                x0 = k * hop / rate
                if types[k] != "none":
                    ax.axvspan(x0, x0 + hop / rate, color="tab:red", alpha=0.13, lw=0)
                if evs[k]:
                    ax.axvspan(x0, x0 + hop / rate, color="tab:green", alpha=0.13, lw=0)
        ax = axes[0]; shade(ax)
        off = 0.6 * (np.ptp(sig["AF3"]) + np.ptp(sig["T7"])) / 2
        ax.plot(t, sig["AF3"] + off, color="0.15", lw=0.5); ax.plot(t, sig["T7"] - off, color="tab:blue", lw=0.5)
        ax.set_ylabel("EEG (a.u.)"); ax.set_yticks([off, -off]); ax.set_yticklabels(["AF3", "T7"])
        ax.get_yticklabels()[1].set_color("tab:blue")
        ax = axes[1]; shade(ax)
        for nm, w, col, ls in (("Ideal", wi, "0.25", "--"), ("MIL-CREDA", wm, "tab:orange", "-"),
                               ("UDA-DDA", wu, "tab:purple", "-")):
            ax.step([k * hop / rate for k in idx] + [(start + L) * hop / rate], list(w[idx]) + [w[start + L - 1]],
                    where="post", color=col, lw=1.4, ls=ls, label=nm)
        ax.axhline(1 / K, color="0.3", ls=":", lw=0.8)
        ax.set_ylabel("Attention"); axes[-1].set_xlabel("Time (s)")
        fig.legend(handles=[Patch(color="tab:red", alpha=0.3, label="Artifact window"),
                            Patch(color="tab:green", alpha=0.3, label="Class-event window"),
                            Line2D([], [], color="0.3", ls=":", label="Uniform weight"),
                            Line2D([], [], color="0.25", ls="--", lw=1.4, label="Ideal"),
                            Line2D([], [], color="tab:orange", lw=1.4, label="MIL-CREDA"),
                            Line2D([], [], color="tab:purple", lw=1.4, label="UDA-DDA")],
                   loc="upper center", ncol=2, frameon=False, columnspacing=0.8, handlelength=1.5)
        LEG = 0.70
        _fit_width(fig, W_PT, PAD, lambda f: (0, 0, 1, 1 - LEG / f.get_size_inches()[1]))
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02, transparent=True, metadata=PDF_META)
        fig.savefig(out_pdf.with_suffix(".png"), dpi=170)
        plt.close(fig)


def alignment_figure(summary_dir: Path, out_pdf: Path, epochs2: int) -> None:
    """Port of paper/.paper-writing/figures/synthetic-alignment/alignfig_col.py (the one-column version
    printed in the paper), drawn from the stored representations and their stored UMAP embedding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z = np.load(summary_dir / "figure-unit" / f"umap-{FIGURE_UNIT}.npz")
    E = np.load(summary_dir / "figure-unit" / "umap-embedding.npz")["embedding"]
    ep = int(epochs2); mid = ep // 2
    tt = z["split"] == "target_test"
    subj, pair, yt_all = z["subjects"], z["pairs"], z["labels"]
    src_names = list(z["source:subjects"])
    nT, nS = int(tt.sum()), len(src_names); per = nT + nS
    tnames, yt, tpairs = list(subj[tt]), yt_all[tt], list(pair[tt])
    pick = [sorted([s for s, y in zip(tnames, yt) if y == c])[0] for c in (0, 1)]
    W_PT, PAD, FS = 226.7, 0.02, 10
    with plt.rc_context({"font.family": "serif", "font.size": FS}):
        fig, axes = plt.subplots(1, 2, figsize=(3.15, 2.05), gridspec_kw={"width_ratios": [1.25, 1]})
        ax = axes[0]
        S0, S1, S2 = (E[k * per:(k + 1) * per] for k in range(3))
        for e, al in ((S0, 0.18), (S2, 0.4)):
            et, es = e[:nT], e[nT:]
            ax.scatter(*es.T, s=7, marker="o", facecolors="none", edgecolors="0.45", lw=0.5, alpha=al)
            ax.scatter(*et.T, s=9, marker="^", color="0.45", alpha=al, lw=0)
        for s in pick:
            i = tnames.index(s); j = src_names.index(tpairs[i]); col = "tab:red" if yt[i] else "tab:blue"
            path = np.array([S0[i], S1[i], S2[i]])
            ax.plot(*path.T, color=col, lw=0.9, ls=":", zorder=4)
            ax.plot([S2[i, 0], S2[nT + j, 0]], [S2[i, 1], S2[nT + j, 1]], color="0.1", lw=1.0, zorder=4)
            ax.scatter(*S0[i], s=34, marker="^", facecolors="white", edgecolors=col, lw=1.0, zorder=5)
            ax.scatter(*S1[i], s=34, marker="^", facecolors=matplotlib.colors.to_rgba(col, 0.35), edgecolors=col,
                       lw=0.8, zorder=5)
            ax.scatter(*S2[i], s=40, marker="^", color=col, edgecolors="k", lw=0.8, zorder=5)
            ax.scatter(*S2[nT + j], s=36, marker="o", color=col, edgecolors="k", lw=0.8, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        lo, hi = E[:3 * per].min(0), E[:3 * per].max(0); pad = 0.05 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax = axes[1]
        allt = list(subj)
        dist = lambda k, s: float(np.linalg.norm(z[f"{k}:target"][allt.index(s)] - z[f"{k}:pair"][allt.index(s)]))
        for s in pick:
            col = "tab:red" if yt[tnames.index(s)] else "tab:blue"
            ax.plot([0, mid, ep], [dist(k, s) for k in ("start", "fullMid", "fullEnd")], "-o", color=col, ms=4)
            ax.plot([0, ep], [dist("start", s), dist("globalOnlyEnd", s)], "--s", color=col, ms=4, alpha=0.6)
        ax.set_yscale("log"); ax.set_xticks([0, mid, ep]); ax.set_xlabel("phase-2 epoch")
        ax.set_ylabel("own-pair distance")
        ax.set_yticks([0.5, 1, 2, 5]); ax.set_yticklabels(["0.5", "1", "2", "5"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.legend(handles=[Line2D([], [], color="0.3", marker="o", ms=4, label="with local term"),
                            Line2D([], [], color="0.3", marker="s", ms=4, ls="--", alpha=0.6, label="without")],
                   loc="lower center", ncol=2, fontsize=FS, frameon=False, borderpad=0.0, columnspacing=1.0,
                   handletextpad=0.4, borderaxespad=0.15)
        LEG = 0.22
        _fit_width(fig, W_PT, PAD, lambda f: (0, LEG / f.get_size_inches()[1], 1, 1))
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=PAD, transparent=True, metadata=PDF_META)
        fig.savefig(out_pdf.with_suffix(".png"), dpi=170)
        plt.close(fig)


def render(summary_dir=None, out_dir=None) -> dict:
    """Figures (PDF+PNG) and tables (LaTeX+CSV) from `summary/` alone; no training."""
    d = Path(summary_dir) if summary_dir else stage_dir() / "summary"
    o = Path(out_dir) if out_dir else stage_dir()
    st = load_statistics(d)
    (o / "figures").mkdir(parents=True, exist_ok=True)
    attention_figure(d, o / "figures" / "synthetic-attention-signal.pdf")
    alignment_figure(d, o / "figures" / "synthetic-alignment.pdf", st["epochs2"])
    tables(st, o / "tables")
    return {"figures": sorted(str(p) for p in (o / "figures").glob("*")),
            "tables": sorted(str(p) for p in (o / "tables").glob("*"))}


# -------------------------------------------------------------------------------- paper checks
def paper_checks(summary_dir=None) -> list[tuple[str, float, float, float]]:
    """(name, obtained, paper, tolerance) for the numbers paper/main.tex states for this stage; each
    obtained value is rounded to the paper's printed precision, so the tolerance is 0. Line numbers
    are those of main.tex at commit 1898be5 of the paper workspace."""
    st = load_statistics(summary_dir)
    ag, te, at = st["aggregation"]["rows"], st["terms"]["rows"], st["attention"]
    op, fu, al = st["ownPair"], st["figureUnit"], st["alignment"]
    ov, one = fu["overSubjects"], fu["excerptSubject"]
    h0, h1 = (al["distances"][s] for s in al["highlighted"])
    rows = [
        # Table tab:synthetic-attention-variants (main.tex lines 394-399)
        *[(f"table attention: BA {k}", ag[k]["BA"], v, 2) for k, v in zip(AGG, (0.72, 0.71, 0.70, 0.69, 0.52, 0.52))],
        *[(f"table attention: rank {k}", ag[k]["rank"], v, 2) for k, v in zip(AGG, (2.71, 3.17, 3.21, 3.15, 4.60, 4.17))],
        *[(f"table attention: pHolm {k}", ag[k]["pHolm"], v, 2) for k, v in zip(AGG[1:], (0.31, 0.61, 0.61, 0.01, 0.02))],
        *[(f"table attention: delta {k}", ag[k]["cliff"], v, 2) for k, v in zip(AGG[1:], (0.04, 0.01, 0.03, 0.38, 0.33))],
        *[(f"table attention: artifact AUC {k}", at[k]["artifactAUC"]["mean"], v, 2)
          for k, v in zip(ATTENTION, (0.57, 0.53, 0.52, 0.54))],
        *[(f"table attention: event AUC {k}", at[k]["eventAUC"]["mean"], v, 2)
          for k, v in zip(ATTENTION, (0.63, 0.65, 0.62, 0.62))],
        *[(f"table attention: artifact AUC significant {k}", float(at[k]["artifactAUC"]["pHolm"] < 0.05), v, 0)
          for k, v in zip(ATTENTION, (1, 0, 0, 0))],
        *[(f"table attention: event AUC significant {k}", float(at[k]["eventAUC"]["pHolm"] < 0.05), 1, 0) for k in ATTENTION],
        # paragraph after the table (line 402)
        ("BA own (rel. + consensus)", ag["own"]["BA"], 0.721, 3),
        ("BA mean", ag["mean"]["BA"], 0.517, 3), ("BA sum", ag["sum"]["BA"], 0.523, 3),
        # the paper prints 0.008 for a Holm p of 0.00749 (mech82.log shows 0.0075, rounded up once more):
        # a last-digit rounding of the paper, not of this run, hence the one non-zero tolerance
        ("pHolm mean", ag["mean"]["pHolm"], 0.008, 3, 0.001), ("pHolm sum", ag["sum"]["pHolm"], 0.016, 3),
        ("BA gap own - relevance only", ag["own"]["BA"] - ag["own without consensus"]["BA"], 0.012, 3),
        ("BA gap own - gated (Ilse)", ag["own"]["BA"] - ag["gated attention"]["BA"], 0.033, 3),
        ("artifact AUC own", at["own"]["artifactAUC"]["mean"], 0.566, 3),
        ("artifact AUC own pHolm", at["own"]["artifactAUC"]["pHolm"], 0.011, 3),
        ("event AUC own", at["own"]["eventAUC"]["mean"], 0.628, 3),
        ("event AUC own pHolm", at["own"]["eventAUC"]["pHolm"], 0.012, 3),
        ("artifact AUC relevance only", at["own without consensus"]["artifactAUC"]["mean"], 0.534, 3),
        ("event AUC own at dC<=1", at["own"]["eventAUC"]["dLow"], 0.461, 3),
        ("artifact AUC own at dC<=1", at["own"]["artifactAUC"]["dLow"], 0.557, 3),
        # attention-signal paragraph (line 411)
        ("excerpt subject: weight after the first events, MIL-CREDA", one["MIL-CREDA"]["weightAfterFirstEvents"], 0.12, 2),
        ("excerpt subject: weight after the first events, UDA-DDA", one["UDA-DDA"]["weightAfterFirstEvents"], 0.04, 2),
        ("excerpt subject: window after the first events, start (s)", one["afterFirstEventsSeconds"], 10, 0),
        ("excerpt subject: % weight on events, MIL-CREDA", 100 * one["MIL-CREDA"]["weightOnEvents"], 67, 0),
        ("excerpt subject: % weight on events, UDA-DDA", 100 * one["UDA-DDA"]["weightOnEvents"], 87, 0),
        ("excerpt subject: % uniform weight on events", 100 * one["uniformOnEvents"], 20, 0),
        ("subjects: weight on events, MIL-CREDA", ov["event"]["MIL-CREDA"], 0.64, 2),
        ("subjects: weight on events, UDA-DDA", ov["event"]["UDA-DDA"], 0.68, 2),
        ("subjects: uniform weight on events", ov["uniformEvent"], 0.23, 2),
        ("subjects with ADHD", ov["nEventSubjects"], 60, 0),
        ("subjects: artifact-free events, MIL-CREDA", ov["eventClean"]["MIL-CREDA"], 0.54, 2),
        ("subjects: artifact-free events, UDA-DDA", ov["eventClean"]["UDA-DDA"], 0.55, 2),
        ("subjects: artifact-free events p", ov["eventClean"]["pWilcoxon"], 0.16, 2),
        ("subjects holding artifacts", ov["nArtifactSubjects"], 117, 0),
        ("subjects: artifact weight, UDA-DDA", ov["artifact"]["UDA-DDA"], 0.211, 3),
        ("subjects: uniform artifact weight", ov["uniformArtifact"], 0.215, 3),
        ("subjects: artifact weight, MIL-CREDA", ov["artifact"]["MIL-CREDA"], 0.183, 3),
        ("subjects: artifact weight p", ov["artifact"]["pWilcoxon"], 0.002, 3),
        ("subjects: event weight p", ov["event"]["pWilcoxon"], 0.006, 3),
        ("subjects: artifact-covered events, MIL-CREDA", ov["eventArtifact"]["MIL-CREDA"], 0.10, 2),
        ("subjects: artifact-covered events, UDA-DDA", ov["eventArtifact"]["UDA-DDA"], 0.13, 2),
        ("subjects: artifact-covered events not significant", float(ov["eventArtifact"]["pWilcoxon"] >= 0.05), 1, 0),
        # Table tab:synthetic-term-ablation (lines 424-427) and its paragraph (line 431)
        *[(f"table terms: BA {k}", te[k]["BA"], v, 3) for k, v in zip(TER, (0.721, 0.717, 0.669, 0.576))],
        *[(f"table terms: pHolm {k}", te[k]["pHolm"], v, 3) for k, v in zip(TER[1:], (0.453, 0.155, 0.038))],
        ("terms Friedman p", st["terms"]["friedmanP"], 0.099, 3),
        *[(f"terms BA at dC>=1.5 {k}", te[k]["BAdHigh"], v, 3) for k, v in zip(TER, (0.943, 0.940, 0.841, 0.649))],
        # alignment paragraph (line 441)
        ("class-0 pair distance start", h0["start"], 2.86, 2), ("class-1 pair distance start", h1["start"], 6.24, 2),
        ("class-0 pair distance end (local term)", h0["fullEnd"], 0.41, 2),
        ("class-1 pair distance end (local term)", h1["fullEnd"], 1.16, 2),
        ("class-0 pair distance end (global only)", h0["globalOnlyEnd"], 0.58, 2),
        ("class-1 pair distance end (global only)", h1["globalOnlyEnd"], 0.96, 2),
        ("own-pair distance start, mean over units", op["start"]["mean"], 6.80, 2),
        ("own-pair distance end with local term", op["fullEnd"]["mean"], 2.48, 2),
        ("own-pair distance end without local term", op["globalOnlyEnd"]["mean"], 2.64, 2),
        ("units closer with the local term", op["closerUnits"], 21, 0),
        ("one-sided Wilcoxon p < 0.001", float(op["pLess"] < 0.001), 1, 0),
        ("local term only BA at dC>=1.5", te["local only"]["BAdHigh"], 0.841, 3),
    ]
    return [(r[0], round(float(r[1]), r[3]), float(r[2]), float(r[4]) if len(r) > 4 else 0.0) for r in rows]
