"""Stage 05, attention influence and local alignment: Table `tab:ieee-best-subject` and Figure
`fig:ieee-alignment` of the paper, with the influence numbers of its text.

What the paper ran (Kaggle job folders `ai-*` of the ieee-pool-slim branch, CPU, one thread each):

* `ai-<unit>` (24 jobs): `attention_influence.run(unit_key, threads=1)` on every synthetic pool unit,
  MIL-CREDA's cef pick retrained from its frozen seed-1 base (`Manifests/oracle_bases`, picks in
  `Manifests/pool_aggregation/milcreda_picks.json`); influence of removing the top, bottom, random,
  class-event and artifact windows.
* `ai-ieee-cef`: `attention_influence.run_ieee_all(tag="cef", threads=1)`, the ten ADHD/Control Children
  units, label-free pick (`Manifests/attention_influence/ieee_cef_picks.json`).
* `ai-ieee-oracle2`: `run_ieee_all(tag="oracle", threads=1)`, the best-test (oracle) pick
  (`ieee_oracle_picks.json`); `ai-ieee-oracle` was its first launch, superseded by commit 0ed4e0d.
* `ai-ieee-align`: `attention_influence.run_alignment_all(threads=1)`, the cef pick trained with and
  without the local term, bag representations at the start, middle and end of phase 2.

Every run refuses to measure unless the retrained pick reproduces the reported balanced accuracy (and
AUC on ADHD). That holds on Kaggle's CPU, where the picks were trained; on another platform the
100-epoch phase 2 can drift (measured on an Apple-silicon Mac: no ADHD unit reproduces), so
`run(strict=False)` measures anyway and records the gap in `reproduction`.

`merge` condenses the raw outputs into `Results/paper/05_attention_influence/summary/`; `render`
draws the figure and writes the table from that summary only; `paper_checks` compares the numbers with
main.tex. The stored summary was merged from the paper's own outputs (`Results/v83-influence`).
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from MIL_CREDA_EEG import repro

STAGE = "05_attention_influence"

#: the 24 synthetic pool units, in the order of the MIL-CREDA picks manifest and the `ai-*` job folders
SYNTHETIC = tuple(f"d{d}-s{s}-r{r}" for d in ("0p6", "1", "1p5", "2p5") for s in ("0p5", "1p5", "3")
                  for r in (0, 1))
#: the ten ADHD/Control Children units (`ieee_pool.UNITS`)
IEEE = tuple(f"ieee-r{r}h{h}" for r in range(5) for h in range(2))
#: the work items, in a fixed order: (family, unit)
UNITS = (tuple(("synthetic", u) for u in SYNTHETIC) + tuple(("cef", u) for u in IEEE)
         + tuple(("oracle", u) for u in IEEE) + tuple(("align", u) for u in IEEE))
#: raw output folder of each family, the layout of the paper's own outputs (Results/v83-influence)
FOLDER = {"synthetic": "ai-{unit}", "cef": "ai-ieee-cef", "oracle": "ai-ieee-oracle2", "align": "ai-ieee-align"}
#: the file each run writes in its folder
FILE = {"synthetic": "influence-{unit}.json", "cef": "influence-{unit}-cef.json",
        "oracle": "influence-{unit}-oracle.json", "align": "alignment-{unit}.json"}
#: the item the smoke run trains (about 15 s on a laptop CPU): the ADHD cef pick of the figure's unit
SMOKE_ITEM = ("cef", "ieee-r3h1")
STAGES, EPOCHS = ("start", "mid", "end"), (0, 50, 100)
#: printed width of the figure: one column of the IJNS layout, (486 pt - columnsep 1.15 cm) / 2
COLUMN_PT = 226.7
#: Kaggle-measured seconds per item (the `seconds` field of the paper's outputs, one CPU thread)
SECONDS = {"synthetic": 410.0, "cef": 44.0, "oracle": 42.0, "align": 118.0}


def raw_path(family: str, unit: str, root) -> Path:
    return Path(root) / FOLDER[family].format(unit=unit) / FILE[family].format(unit=unit)


@contextlib.contextmanager
def _lenient(AI):
    """Let the runs measure even when the retrained pick does not reproduce the reported numbers."""
    saved = AI.TOLERANCE, AI.IEEE_TOLERANCE
    AI.TOLERANCE = AI.IEEE_TOLERANCE = float("inf")
    try:
        yield saved
    finally:
        AI.TOLERANCE, AI.IEEE_TOLERANCE = saved


def run(split=None, smoke: bool = False, out=None, strict: bool = True, threads: int = 1,
        manifests=None) -> Path:
    """Run this account's portion of UNITS with the paper's entrypoints and parameters (threads=1 as in
    every `ai-*` run-config.json); raw outputs go to `out` (default `<stage>/raw`) in the paper's layout.
    `strict=False` measures even when the retrained pick does not reproduce its reported numbers (the
    gap stays in `reproduction`, `reproduces` keeps the paper's tolerance). `smoke=True` runs only
    SMOKE_ITEM, non-strict, into a temporary folder unless `out` is given. `manifests` points to another
    `MIL-CREDA-EEG/Manifests` folder (frozen bases and picks); None uses the repository's own."""
    from MIL_CREDA_EEG import attention_influence as AI

    M = Path(manifests) if manifests else None
    syn = dict(bases=M / "oracle_bases", picks=M / "pool_aggregation" / "milcreda_picks.json") if M else {}
    ieee = (lambda tag: dict(bases=M / "ieee_pool_bases",
                             picks=M / "attention_influence" / f"ieee_{tag}_picks.json")) if M else (lambda tag: {})
    align = dict(bases=M / "ieee_pool_bases") if M else {}

    repro.ieee_root()
    if smoke:
        items, strict = [SMOKE_ITEM], False
        out = Path(out) if out else Path(tempfile.mkdtemp(prefix="stage05-smoke-"))
    else:
        items = repro.portion(UNITS, split)
        out = Path(out) if out else repro.results_dir(STAGE) / "raw"
    calls = {"synthetic": lambda u, w: AI.run(u, threads, w, **syn),
             "cef": lambda u, w: AI.run_ieee(u, "cef", threads, w, **ieee("cef")),
             "oracle": lambda u, w: AI.run_ieee(u, "oracle", threads, w, **ieee("oracle")),
             "align": lambda u, w: AI.run_alignment(u, threads, w, **align)}
    for family, unit in items:
        work = raw_path(family, unit, out).parent
        print(f"{family} {unit} -> {work}", flush=True)
        with (contextlib.nullcontext((AI.TOLERANCE, AI.IEEE_TOLERANCE)) if strict else _lenient(AI)) as tol:
            try:
                calls[family](unit, work)
            except RuntimeError as e:  # strict refusal: the file is written without measurements
                print(f"  refused: {e}")
        f = raw_path(family, unit, out)
        if not strict and f.exists():
            d = json.loads(f.read_text())
            d["strict"] = False
            d["reproduces"] = bool(d["reproduction"] <= (tol[0] if family == "synthetic" else tol[1]))
            f.write_text(json.dumps(d))
        d = json.loads(f.read_text())
        print(f"  reproduction gap {d['reproduction']:.3g}, reproduces {d['reproduces']}, {d['seconds']} s")
    return out


# ---------------------------------------------------------------------------------------------------
# merge


def _read(family, unit, raw):
    f = raw_path(family, unit, raw)
    return json.loads(f.read_text()) if f.exists() else None


def merge(out=None, raw=None) -> Path:
    """Join the raw outputs of every portion found under `raw` (default `<stage>/raw`; copy every
    account's folders there first) into the summary folder `out` (default `<stage>/summary`)."""
    raw = Path(raw) if raw else repro.results_dir(STAGE) / "raw"
    out = Path(out) if out else repro.results_dir(STAGE) / "summary"
    out.mkdir(parents=True, exist_ok=True)
    influence = {"synthetic": {}, "cef": {}, "oracle": {}}
    meta, reps, status = {}, {}, {"missing": [], "notMeasured": [], "notReproduced": []}
    for family, unit in UNITS:
        d = _read(family, unit, raw)
        if d is None:
            status["missing"].append(f"{family}:{unit}")
            continue
        if not d.get("reproduces"):
            status["notReproduced"].append(f"{family}:{unit} ({d['reproduction']:.3g})")
        if family != "align":
            if "rows" not in d:
                status["notMeasured"].append(f"{family}:{unit}")
                continue
            influence[family][unit] = {k: d[k] for k in d if k not in ("stepTotals", "threads")}
            continue
        if "sourceLabels" not in d:
            status["notMeasured"].append(f"{family}:{unit}")
            continue
        src = list(d["sourceLabels"])
        meta[unit] = {k: d[k] for k in ("unitKey", "trial", "params", "trainSeed", "pickBA", "pickAUC", "BAtest",
                                        "AUCtest", "reproduction", "reproduces", "subjects", "labels", "seconds")}
        meta[unit].update(sourceSubjects=src, sourceLabels=[int(d["sourceLabels"][s]) for s in src])
        for tag in ("local", "noLocal"):
            for st in STAGES:
                for side in ("t", "s"):
                    reps[f"{unit}|{tag}|{st}|{side}"] = np.asarray(d[tag][st][side], np.float64)
    (out / "influence.json").write_text(json.dumps(influence))
    (out / "alignment.json").write_text(json.dumps(meta, indent=1))
    np.savez_compressed(out / "alignment_reps.npz", **reps)
    status["raw"] = str(raw.resolve().relative_to(repro.repo_root())) if raw.resolve().is_relative_to(
        repro.repo_root()) else "outside the repository"
    (out / "status.json").write_text(json.dumps(status, indent=1))
    summary = statistics(out)
    (out / "statistics.json").write_text(json.dumps(summary, indent=1))
    if meta:
        (out / "alignment_pair.json").write_text(json.dumps(alignment_pair(out), indent=1))
    print(f"merged into {out}: {sum(map(len, influence.values()))} influence units, {len(meta)} alignment "
          f"units; missing {len(status['missing'])}, not measured {len(status['notMeasured'])}, "
          f"not reproduced {len(status['notReproduced'])}")
    return out


# ---------------------------------------------------------------------------------------------------
# statistics (the paper's analysis, Results/v83-influence/analyze_ai.py)


def _summary_dir(summary_dir) -> Path:
    return Path(summary_dir) if summary_dir else repro.results_dir(STAGE) / "summary"


def _test(pairs) -> dict:
    """removed vs random: means, ratio, share above, one-sided paired Wilcoxon (analyze_ai.test)."""
    from scipy.stats import wilcoxon
    p = np.array([q for q in pairs if q is not None], float)
    d = p[:, 0] - p[:, 1]
    pv = float(wilcoxon(p[:, 0], p[:, 1], alternative="greater").pvalue) if np.any(d) else 1.0
    return {"n": int(len(p)), "removed": float(p[:, 0].mean()), "random": float(p[:, 1].mean()),
            "ratio": float(p[:, 0].mean() / p[:, 1].mean()), "above": float(np.mean(d > 0)), "p": pv}


def _toward(rows, key) -> float:
    """Share of subjects whose removal moved the logit toward the true class (analyze_ai.toward)."""
    t = [np.sign(r[key]) * (2 * r["label"] - 1) for r in rows if key in r and abs(r[key]) > 1e-9]
    return float(np.mean(np.array(t) > 0))


def _unit_mean(rows, key, rand):
    x = [(abs(r[key]), float(np.mean(np.abs(r[rand])))) for r in rows if key in r]
    return tuple(np.mean(np.array(x), 0)) if x else None


def _d(unit: str) -> float:
    return float(unit.split("-")[0][1:].replace("p", "."))


def statistics(summary_dir=None) -> dict:
    """Every number of the table and the text, from the stored influence rows."""
    inf = json.loads((_summary_dir(summary_dir) / "influence.json").read_text())
    out = {}
    syn = inf["synthetic"]
    for name, sel in (("all", lambda u: True), ("d>=1.5", lambda u: _d(u) >= 1.5), ("d<=1", lambda u: _d(u) <= 1)):
        U = [u for u in SYNTHETIC if u in syn and sel(u)]
        if not U:
            continue
        rows = [r for u in U for r in syn[u]["rows"]]
        out[f"synthetic {name}"] = {
            "units": len(U),
            "event": _test([_unit_mean(syn[u]["rows"], "event", "eventRandom") for u in U]),
            "artifact": _test([_unit_mean(syn[u]["rows"], "artifact", "artifactRandom") for u in U]),
            "top": _test([_unit_mean(syn[u]["rows"], "top", "random") for u in U]),
            "bottom": _test([_unit_mean(syn[u]["rows"], "bottom", "random") for u in U]),
            "towardTop": _toward(rows, "top"), "towardEvent": _toward(rows, "event")}
    for tag in ("cef", "oracle"):
        U = [u for u in IEEE if u in inf[tag]]
        if not U:
            continue
        rows = [r for u in U for r in inf[tag][u]["rows"]]
        out[f"ADHD {tag}"] = {
            "units": len(U), "subjects": len(rows),
            "top": _test([(abs(r["top"]), np.mean(np.abs(r["random"]))) for r in rows]),
            "bottom": _test([(abs(r["bottom"]), np.mean(np.abs(r["random"]))) for r in rows]),
            "topUnits": _test([_unit_mean(inf[tag][u]["rows"], "top", "random") for u in U]),
            "towardTop": _toward(rows, "top"), "towardBottom": _toward(rows, "bottom")}
    return out


# ---------------------------------------------------------------------------------------------------
# alignment pair (rule of the paper's figure script, fig_align_col.py)


def _load_alignment(summary_dir=None):
    s = _summary_dir(summary_dir)
    meta = json.loads((s / "alignment.json").read_text())
    with np.load(s / "alignment_reps.npz") as z:
        reps = {k: z[k] for k in z.files}
    R = {u: {tag: {st: (reps[f"{u}|{tag}|{st}|t"], reps[f"{u}|{tag}|{st}|s"]) for st in STAGES}
             for tag in ("local", "noLocal")} for u in meta}
    return meta, R


def _dist(zt, zs):
    return np.linalg.norm(zt[:, None] - zs[None], axis=2)


def shortening(R_unit) -> float:
    """Mean over the target test subjects of how much the local term shortened, at the end of phase 2,
    the distance to the nearest source bag (nearest with the term): mean of D_without - D_with."""
    D, Dn = _dist(*R_unit["local"]["end"]), _dist(*R_unit["noLocal"]["end"])
    nn = D.argmin(1)
    i = np.arange(len(nn))
    return float(np.mean(Dn[i, nn] - D[i, nn]))


def alignment_pair(summary_dir=None) -> dict:
    """The unit where the local term most shortened the distance to the nearest source bag (main.tex,
    Methods) and, per class, the same-class pair whose final distance shrank most relative to training
    without that term (fig_align_col.py): its subjects and its distances at the start, middle and end."""
    meta, R = _load_alignment(summary_dir)
    by_unit = {u: shortening(R[u]) for u in meta}
    unit = max(by_unit, key=by_unit.get)
    m, Ru = meta[unit], R[unit]
    ys, yt = np.array(m["sourceLabels"]), np.array(m["labels"])
    D, Dn = _dist(*Ru["local"]["end"]), _dist(*Ru["noLocal"]["end"])
    nn = D.argmin(1)
    score = np.array([D[i, nn[i]] / Dn[i, nn[i]] for i in range(len(yt))])
    same = ys[nn] == yt
    pairs = {}
    for c in (0, 1):
        cand = np.flatnonzero((yt == c) & same)
        i = int(cand[np.argmin(score[cand])])
        j = int(nn[i])
        pairs[str(c)] = {"targetIndex": i, "sourceIndex": j, "target": m["subjects"][i],
                         "source": m["sourceSubjects"][j], "sourceLabel": int(ys[j]),
                         "distance": {tag: [float(_dist(*Ru[tag][st])[i, j]) for st in STAGES] for tag in Ru}}
    return {"unit": unit, "shorteningByUnit": by_unit, "pairs": pairs,
            "nearestSourceLabelAgreement": float(np.mean(same)), "epochs": list(EPOCHS)}


# ---------------------------------------------------------------------------------------------------
# render


TABLE_ROWS = (("Random 25\\,\\%", "Random 25%"), ("Highest 25\\,\\%", "Highest 25%"),
              ("Lowest 25\\,\\%", "Lowest 25%"), ("Class-event", "Class-event"))


def table_values(st: dict) -> list[list]:
    """The table of tab:ieee-best-subject: rows random, highest, lowest, class-event; columns ADHD cef,
    ADHD best test (oracle), synthetic 24 units, synthetic d_C >= 1.5 (None: not applicable)."""
    a, o, s, h = st["ADHD cef"], st["ADHD oracle"], st["synthetic all"], st["synthetic d>=1.5"]
    return [[a["top"]["random"], o["top"]["random"], s["top"]["random"], h["top"]["random"]],
            [a["top"]["removed"], o["top"]["removed"], s["top"]["removed"], h["top"]["removed"]],
            [a["bottom"]["removed"], o["bottom"]["removed"], s["bottom"]["removed"], h["bottom"]["removed"]],
            [None, None, s["event"]["removed"], h["event"]["removed"]]]


def _write_table(st, tables: Path):
    vals = table_values(st)
    col_max = [max(v[c] for v in vals if v[c] is not None) for c in range(4)]
    fmt = lambda v, c: "--" if v is None else (f"\\textbf{{{v:.3f}}}" if v == col_max[c] else f"{v:.3f}")
    body = "\n".join(f"{tex} & " + " & ".join(fmt(v, c) for c, v in enumerate(row)) + " \\\\"
                     for (tex, _), row in zip(TABLE_ROWS, vals))
    tex = ("\\setlength{\\tabcolsep}{2pt}\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcccc}\n"
           "\\toprule\n & \\multicolumn{2}{c}{ADHD} & \\multicolumn{2}{c}{Synthetic} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
           "Windows removed & cef & Best test & 24 units & $d_{\\mathrm{C}}\\geq1.5$ \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular*}}\n")
    (tables / "tab_ieee_best_subject.tex").write_text(tex)
    lines = ["windows_removed,adhd_cef,adhd_best_test,synthetic_24_units,synthetic_dC_ge_1p5"]
    lines += [csv + "," + ",".join("" if v is None else repr(v) for v in row) for (_, csv), row in zip(TABLE_ROWS, vals)]
    (tables / "tab_ieee_best_subject.csv").write_text("\n".join(lines) + "\n")


def _draw_alignment(summary_dir, figures: Path) -> list[Path]:
    """Port of paper/.paper-writing/figures/ieee-alignment/fig_align_col.py (the script behind the
    paper's current ieee-alignment.pdf), reading the stored summary instead of the old scratch copy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import umap
    from MIL_CREDA_EEG import ps_figures as F

    meta, R = _load_alignment(summary_dir)
    sel = alignment_pair(summary_dir)
    unit, Ru = sel["unit"], R[sel["unit"]]
    ys, yt = np.array(meta[unit]["sourceLabels"]), np.array(meta[unit]["labels"])
    pairs = {int(c): (p["targetIndex"], p["sourceIndex"]) for c, p in sel["pairs"].items()}
    curve = {int(c): p["distance"] for c, p in sel["pairs"].items()}
    col = {0: "tab:blue", 1: "tab:red"}
    F.apply_style(10, 1.0, text_width_pt=COLUMN_PT, aspect=0.62)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({k: 10.0 for k in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                                           "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
                                           "figure.titlesize")})  # journal floor: text >= 10 pt
    fig, axes = plt.subplots(1, 2, constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    nT = len(yt)
    per = nT + len(ys)  # one UMAP over the three stages; start faded, end solid, in one panel
    EJ = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(
        np.concatenate([np.concatenate(Ru["local"][st]) for st in STAGES]))
    S0, S1, S2 = (EJ[k * per:(k + 1) * per] for k in range(3))
    ax = axes[0]
    for e, al in ((S0, 0.18), (S2, 0.4)):  # other bags in grey, start fainter than end
        et, es = e[:nT], e[nT:]
        ax.scatter(*es.T, s=7, marker="o", facecolors="none", edgecolors="0.45", lw=0.5, alpha=al)
        ax.scatter(*et.T, s=9, marker="^", color="0.45", alpha=al, lw=0)
    for c, (i, j) in pairs.items():
        ax.plot(*np.array([S0[i], S1[i], S2[i]]).T, color=col[c], lw=0.9, ls=":", zorder=4)
        ax.plot([S2[i, 0], S2[nT + j, 0]], [S2[i, 1], S2[nT + j, 1]], color="0.1", lw=0.9, zorder=4)
        ax.scatter(*S0[i], s=34, marker="^", facecolors="white", edgecolors=col[c], lw=1.0, zorder=5)
        ax.scatter(*S1[i], s=34, marker="^", facecolors=matplotlib.colors.to_rgba(col[c], 0.35),
                   edgecolors=col[c], lw=0.8, zorder=5)
        ax.scatter(*S2[i], s=40, marker="^", color=col[c], edgecolors="k", lw=0.7, zorder=5)
        ax.scatter(*S2[nT + j], s=34, marker="o", color=col[ys[j]], edgecolors="k", lw=0.7, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    pad = 0.05 * np.ptp(EJ, 0)
    ax.set_xlim(EJ[:, 0].min() - pad[0], EJ[:, 0].max() + pad[0])
    ax.set_ylim(EJ[:, 1].min() - pad[1], EJ[:, 1].max() + pad[1])
    ax = axes[1]
    for c in (0, 1):
        ax.plot(EPOCHS, curve[c]["local"], "-o", color=col[c], ms=4, lw=1.2)
        ax.plot(EPOCHS, curve[c]["noLocal"], "--s", color=col[c], ms=4, lw=1.0, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xticks(EPOCHS)
    ax.set_xlabel("phase-2 epoch")
    ax.set_ylabel("pair distance")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.legend(handles=[Line2D([], [], color="0.4", lw=1.2, marker="o", ms=4, label="with local term"),
                        Line2D([], [], color="0.4", lw=1.0, ls="--", marker="s", ms=4, alpha=0.6, label="without")],
               loc="outside lower center", ncol=2, frameon=False, borderpad=0.0, columnspacing=1.0,
               handlelength=1.6, borderaxespad=0.15)
    for _ in range(8):  # widen or narrow the canvas until the tight crop is the column wide
        fig.draw_without_rendering()
        w = fig.get_tightbbox().width + 2 * F.PAD_INCHES
        if abs(w * 72 - COLUMN_PT) < 0.3:
            break
        fig.set_size_inches(fig.get_size_inches()[0] + COLUMN_PT / 72 - w, fig.get_size_inches()[1])
    pdf = figures / "ieee-alignment.pdf"
    saved = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"  # a fixed PDF creation date: re-rendering leaves the file unchanged
    try:
        F.save_figure(fig, pdf, png_preview=False)
    finally:
        if saved is None:
            os.environ.pop("SOURCE_DATE_EPOCH")
        else:
            os.environ["SOURCE_DATE_EPOCH"] = saved
    png = figures / "ieee-alignment.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=F.PAD_INCHES)
    plt.close(fig)
    return [pdf, png]


def render(summary_dir=None, out_dir=None) -> dict:
    """Draw fig:ieee-alignment and write tab:ieee-best-subject (LaTeX and CSV) from the summary only."""
    s = _summary_dir(summary_dir)
    out_dir = Path(out_dir) if out_dir else repro.results_dir(STAGE)
    figures, tables = out_dir / "figures", out_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    st = statistics(s)
    _write_table(st, tables)
    pair = alignment_pair(s)
    (tables / "alignment_pair.json").write_text(json.dumps(pair, indent=1))
    drawn = _draw_alignment(s, figures)
    return {"figures": [str(p) for p in drawn],
            "tables": [str(tables / n) for n in ("tab_ieee_best_subject.tex", "tab_ieee_best_subject.csv",
                                                 "alignment_pair.json")]}


# ---------------------------------------------------------------------------------------------------
# paper checks


def paper_checks(summary_dir=None) -> list[tuple[str, float, float, float]]:
    """(name, obtained, paper, tolerance). Values are rounded to the digits main.tex prints, so the
    tolerance is 0 for everything rendered from the stored summary. Paper lines are main.tex's."""
    st = statistics(summary_dir)
    pair = alignment_pair(summary_dir)
    r3 = lambda v: round(float(v), 3)
    r2 = lambda v: round(float(v), 2)
    pct = lambda v: round(100 * float(v))
    vals = table_values(st)
    names = ("random", "highest", "lowest", "class-event")
    paper = ((0.356, 0.289, 0.262, 0.307), (1.374, 1.105, 1.344, 2.066), (0.186, 0.211, 0.246, 0.363),
             (None, None, 0.672, 1.114))  # main.tex lines 492-495 (Table tab:ieee-best-subject)
    cols = ("ADHD cef", "ADHD best test", "synthetic 24 units", "synthetic dC>=1.5")
    rows = [(f"table {n}: {c}", r3(vals[k][j]), p, 0.0)
            for k, n in enumerate(names) for j, (c, p) in enumerate(zip(cols, paper[k])) if p is not None]
    a, o = st["ADHD cef"], st["ADHD oracle"]
    s, h = st["synthetic all"], st["synthetic d>=1.5"]
    rows += [  # main.tex line 480
        ("text: class-event random baseline, 24 units", r3(s["event"]["random"]), 0.246, 0.0),
        ("text: class-event random baseline, dC>=1.5", r3(h["event"]["random"]), 0.284, 0.0),
        ("text: class-event p < 0.001, 24 units (1 = yes)", float(s["event"]["p"] < 0.001), 1.0, 0.0),
        ("text: class-event p < 0.001, dC>=1.5 (1 = yes)", float(h["event"]["p"] < 0.001), 1.0, 0.0),
        ("text: ADHD highest above random, units, cef", a["topUnits"]["above"] * a["units"], 10.0, 0.0),
        ("text: ADHD highest above random, units, best", o["topUnits"]["above"] * o["units"], 10.0, 0.0),
        ("text: ADHD highest above random, % subjects, cef", pct(a["top"]["above"]), 81.0, 0.0),
        ("text: ADHD highest above random, % subjects, best", pct(o["top"]["above"]), 83.0, 0.0),
        ("text: ADHD highest p < 0.001, cef (1 = yes)", float(a["top"]["p"] < 0.001), 1.0, 0.0),
        ("text: ADHD highest p < 0.001, best (1 = yes)", float(o["top"]["p"] < 0.001), 1.0, 0.0),
        ("text: toward true class, % subjects, best (40)", pct(o["towardTop"]), 40.0, 0.0),
        ("text: toward true class, % subjects, cef (50)", pct(a["towardTop"]), 50.0, 0.0),
    ]
    c0, c1 = pair["pairs"]["0"]["distance"], pair["pairs"]["1"]["distance"]
    rows += [  # main.tex line 507
        ("figure: control pair end, with local term", r2(c0["local"][-1]), 0.42, 0.0),
        ("figure: ADHD pair end, with local term", r2(c1["local"][-1]), 0.66, 0.0),
        ("figure: control pair start, without", r2(c0["noLocal"][0]), 1.22, 0.0),
        ("figure: control pair end, without", r2(c0["noLocal"][-1]), 2.07, 0.0),
        ("figure: ADHD pair start, without", r2(c1["noLocal"][0]), 1.06, 0.0),
        ("figure: ADHD pair end, without", r2(c1["noLocal"][-1]), 1.76, 0.0),
    ]
    return rows


def full_run_cost() -> dict:
    """Serial CPU time of the full stage from the Kaggle-measured seconds per item (one thread)."""
    n = {f: sum(1 for g, _ in UNITS if g == f) for f in SECONDS}
    return {"items": n, "seconds": {f: n[f] * SECONDS[f] for f in n},
            "totalHours": sum(n[f] * SECONDS[f] for f in n) / 3600}
