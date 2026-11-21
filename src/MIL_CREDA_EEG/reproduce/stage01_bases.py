"""Stage 01: the frozen phase-1 base models every later stage starts phase 2 from.

Phase 1 is fitted once per unit, with seed 1, and frozen (its cut, its source model and the
bridged, target-initialised copy the methods adapt). The paper used 34 bases:

* 24 synthetic units `d{d}-s{shift}-r{r}` (class effect d in {0.6, 1, 1.5, 2.5}, shift in
  {0.5, 1.5, 3}, rotations 0 and 1). One work item per (d, rotation): `MIL_CREDA_EEG.p1_syn.run`
  searches phase 1 with TPE (20 trials, sampler seed `xpilot.SAMPLER_SEED + rotation`, scored on the
  shift-1.5 unit, since the source does not depend on the shift) and fits the seed-1 base of the
  three shifts. Kaggle family `sp1-d{d}-r{r}` (8 jobs, run.kwargs {"d", "rotation"}, defaults
  trials=20, threads=4; `sp1b-d1p5-r0` re-ran d=1.5 rotation 0).
* 10 ADHD units `ieee-r{r}h{h}` (rotation r of the five subject folds of the IEEE ADHD/Control
  cohort, target-test half h). One work item per rotation: `MIL_CREDA_EEG.p1_variants.run(r, "A")`,
  the full TPE search over `spectral_search.SPACE` (the final rule of the paper's Section on
  training: lr 1e-4..1e-2, l2 1e-4..1e-1, width {32, 64, 128}, tauAtt1 0.01..1, epochs
  {50, 100, 200}; 20 trials) followed by `ieee_pool.build_base` for both halves. Kaggle family
  `p1A-r{r}` (run.kwargs {"rotation", "variant": "A"}, defaults trials=20, threads=4).

For each synthetic base the stage also measures the frozen phase-1 prediction on the unit's
120-subject evaluation pool, exactly as `Results/v82-pool24/eval_bases.py` did
(`criteria_pool._setup` then `xcompare.snapshot` of the prepared copy): the first-phase BA the
paper quotes for the synthetic study is the mean of that pool BA over the 24 units.

The stored summary `Results/paper/01_phase1_bases/summary/` was built from the bases the paper's
phase-2 campaigns actually read (tag `archive/ieee-pool-slim`: synthetic bases of commits
8f71c6f/00a9189, ADHD bases of commit d7a4ef3) and the evaluation pool of
`Results/v82-pool24/phase1_bases.json`. Regenerated bases go to
`Results/paper/01_phase1_bases/regenerated/bases/{oracle_bases,ieee_pool_bases}/`, the same layout
as `MIL-CREDA-EEG/Manifests/`, so a downstream stage can be pointed at them with `bases_dir(True)`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

from MIL_CREDA_EEG import repro

STAGE = "01_phase1_bases"
DS = ("0p6", "1", "1p5", "2p5")
SHIFTS = ("0p5", "1p5", "3")
ROTATIONS_SYN = (0, 1)
ROTATIONS_IEEE = tuple(range(5))
#: one work item = one phase-1 search plus the bases it freezes; synthetic first, as in the paper
UNITS = tuple([("synthetic", d, r) for d in DS for r in ROTATIONS_SYN]
              + [("adhd", None, r) for r in ROTATIONS_IEEE])
#: the paper runs' parameters: p1_syn.run / p1_variants.run defaults used by the sp1 and p1A jobs
TRIALS = 20
THREADS = 4
VARIANT = "A"
EVAL_THREADS = 6          # eval_bases.py used CP._setup(..., 6); threads do not change the pool
#: the smoke items: one ADHD rotation and one synthetic (d, rotation), each with a single TPE trial,
#: and the evaluation pool of one synthetic base (about 5 minutes on a laptop CPU)
SMOKE_UNITS = (("adhd", None, 0), ("synthetic", "1p5", 0))
SMOKE_TRIALS = 1
#: tolerance of the paper checks on a regenerated summary (CPU platform and torch build differ)
REGEN_TOLERANCE = 0.05
FIELDS = ("unitKey", "family", "d", "shift", "rotation", "half", "lr1", "l2", "width", "tauAtt1", "epochs",
          "searchScore", "cut", "arm", "sourceTestBA", "sourceTestAUC", "targetTestBA", "targetTestAUC",
          "targetTestKappa", "evalPoolBA", "evalPoolAUC", "commit", "sha256")


def unit_name(unit) -> str:
    family, d, r = unit
    return f"sp1-d{d}-r{r}" if family == "synthetic" else f"p1{VARIANT}-r{r}"


def base_keys(unit) -> list[str]:
    family, d, r = unit
    if family == "synthetic":
        return [f"d{d}-s{s}-r{r}" for s in SHIFTS]
    return [f"ieee-r{r}h{h}" for h in (0, 1)]


ALL_KEYS = tuple(k for u in UNITS for k in base_keys(u))


def stage_dir() -> Path:
    return repro.results_dir(STAGE)


def summary_dir() -> Path:
    return stage_dir() / "summary"


def regenerated_dir() -> Path:
    return stage_dir() / "regenerated"


def bases_dir(regenerated: bool = False, family: str = "synthetic") -> Path:
    """Where a downstream stage reads its phase-1 bases: the tracked `Manifests/oracle_bases`
    (synthetic) or `Manifests/ieee_pool_bases` (adhd), or the same folders regenerated here."""
    name = {"synthetic": "oracle_bases", "adhd": "ieee_pool_bases"}[family]
    root = regenerated_dir() / "bases" if regenerated else repro.eeg_dir() / "Manifests"
    return root / name


# ----------------------------------------------------------------------------------------- run

def _eval_pool(key: str, bases: Path) -> dict:
    """Results/v82-pool24/eval_bases.py for one synthetic base: BA and AUC of the frozen prepared
    copy at the base's cut on the unit's 120-subject evaluation pool (no training)."""
    import torch
    from MIL_CREDA_EEG import criteria_pool as CP, xcompare as XC

    s = CP._setup(key, CP.METHODS, CP.O.EVAL_SUBJECTS, str(bases), EVAL_THREADS)
    try:
        with torch.no_grad():
            pool = XC.snapshot(s["b"]["prepared"], s["pool"], s["cut"], s["dev"])
    finally:
        s["stack"].close()
    return {"BApool": pool["BA"], "AUCpool": pool["AUC"], "cut": s["cut"]}


def _run_unit(unit, bases: Path, trials: int, evaluate: int | None) -> dict:
    """`evaluate`: how many of a synthetic unit's bases get their evaluation pool (None = all)."""
    family, d, r = unit
    t0 = time.time()
    if family == "synthetic":
        from MIL_CREDA_EEG import p1_syn

        out = bases / "oracle_bases"
        rec = p1_syn.run(d, r, workdir=out, trials=trials, threads=THREADS)
        keys = base_keys(unit)[:evaluate] if evaluate is not None else base_keys(unit)
        if keys:
            ev = {k: _eval_pool(k, out) for k in keys}
            (out / f"eval-d{d}-r{r}.json").write_text(json.dumps(ev, indent=1))
    else:
        from MIL_CREDA_EEG import p1_variants

        repro.ieee_root()
        out = bases / "ieee_pool_bases"
        rec = p1_variants.run(r, VARIANT, workdir=out, trials=trials, threads=THREADS)
        for k in base_keys(unit):              # p1_variants writes the bases under <workdir>/A/
            shutil.move(str(out / VARIANT / f"{k}-s1.pt"), str(out / f"{k}-s1.pt"))
        shutil.rmtree(out / VARIANT, ignore_errors=True)
    rec = {"unit": unit_name(unit), "bases": base_keys(unit), "trials": int(trials),
           "seconds": round(time.time() - t0, 1), **{k: v for k, v in rec.items() if k in ("params", "score")}}
    print(json.dumps(rec), flush=True)
    return rec


def run(split=None, smoke: bool = False, out=None) -> list[dict]:
    """Regenerate the bases of `repro.portion(UNITS, split)` into `<out>/bases/` (default
    `Results/paper/01_phase1_bases/regenerated`, or `.../smoke` with `smoke=True`, which runs
    SMOKE_UNITS with a single TPE trial only to prove the wiring)."""
    if smoke:
        out = Path(out) if out else stage_dir() / "smoke"
        items, trials = list(SMOKE_UNITS), SMOKE_TRIALS
    else:
        out = Path(out) if out else regenerated_dir()
        items, trials = repro.portion(UNITS, split), TRIALS
    bases = out / "bases"
    for name in ("oracle_bases", "ieee_pool_bases"):
        (bases / name).mkdir(parents=True, exist_ok=True)
    return [_run_unit(u, bases, trials, evaluate=1 if smoke else None) for u in items]


# --------------------------------------------------------------------------------------- merge

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(x, n=5):
    return None if x is None else round(float(x), n)


def summarize(oracle_dir, ieee_dir, eval_pool: dict | None = None, searches: dict | None = None) -> dict:
    """The summary of every base found in the two folders (missing ones are skipped, so a partial
    portion merges too). `eval_pool` maps a synthetic key to {"BApool", "AUCpool"}; `searches` maps
    a search name (phase1-d{d}-r{r} or phase1-A-r{r}) to its record."""
    import torch
    from MIL_CREDA_EEG import spectral_trunk

    eval_pool, searches = eval_pool or {}, searches or {}
    rows, subjects = [], {}
    for key in ALL_KEYS:
        syn = not key.startswith("ieee")
        path = Path(oracle_dir if syn else ieee_dir) / f"{key}-s1.pt"
        if not path.exists():
            continue
        with spectral_trunk.installed():
            b = torch.load(path, weights_only=False)
        p, pr, st = b["phase1"], b["predict"], b["sourceTest"]
        if syn:
            d, s, r = key.split("-")
            d, shift, rot, half = d[1:].replace("p", "."), s[1:].replace("p", "."), int(r[1:]), None
            search = searches.get(f"phase1-d{d.replace('.', 'p')}-r{rot}", {})
        else:
            d, shift, rot, half = None, None, int(b["rotation"]), int(b["unitKwargs"]["half"])
            search = searches.get(f"phase1-{VARIANT}-r{rot}", {})
            subjects[key] = {"subjects": pr.get("subjects"), "labels": pr.get("labels"),
                             "probabilities": [_round(v, 6) for v in pr.get("probabilities", [])]}
        ev = eval_pool.get(key, {})
        rows.append({"unitKey": key, "family": "synthetic" if syn else "adhd", "d": d, "shift": shift,
                     "rotation": rot, "half": half, "lr1": _round(p["lr1"], 8), "l2": _round(p["l2"], 8),
                     "width": int(p["width"]), "tauAtt1": _round(p["tauAtt1"], 8), "epochs": int(p["epochs"]),
                     "searchScore": search.get("score"), "cut": _round(b["cut"], 6), "arm": b["arm"],
                     "sourceTestBA": st.get("BA"), "sourceTestAUC": st.get("AUC"), "targetTestBA": pr["BA"],
                     "targetTestAUC": pr["AUC"], "targetTestKappa": pr["kappa"],
                     "evalPoolBA": ev.get("BApool"), "evalPoolAUC": ev.get("AUCpool"),
                     "commit": (b.get("commit") or "")[:8], "sha256": _sha256(path)})
    return {"bases": rows, "adhdSubjects": subjects, "searches": searches}


def _searches_in(*dirs) -> dict:
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("phase1-*.json")):
            out[f.stem] = json.loads(f.read_text())
    return out


def _eval_in(d) -> dict:
    out = {}
    for f in sorted(Path(d).glob("eval-*.json")):
        out.update(json.loads(f.read_text()))
    return out


def write_summary(summary: dict, out_dir, source: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase1_bases.json").write_text(json.dumps({"source": source, **summary}, indent=1))
    with open(out_dir / "phase1_bases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(summary["bases"])
    return out_dir


def merge(out=None) -> Path:
    """Summarize the regenerated bases under `<out>/bases/` (copy the `bases/` folders of every
    account's portion into the same place first) into `<out>/summary/`. The tracked summary of the
    paper's bases (`Results/paper/01_phase1_bases/summary/`) is never overwritten."""
    out = Path(out) if out else regenerated_dir()
    o, i = out / "bases" / "oracle_bases", out / "bases" / "ieee_pool_bases"
    s = summarize(o, i, _eval_in(o), _searches_in(o, i))
    missing = [k for k in ALL_KEYS if k not in {r["unitKey"] for r in s["bases"]}]
    if missing:
        print(f"{len(missing)} of {len(ALL_KEYS)} bases not regenerated yet, e.g. {missing[0]}")
    return write_summary(s, out / "summary", f"stage01_bases.merge of {out / 'bases'}")


def load_summary(summary_dir_=None) -> dict:
    return json.loads((Path(summary_dir_ or summary_dir()) / "phase1_bases.json").read_text())


# -------------------------------------------------------------------------------------- render

def _fmt(x, n=3):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def render(summary_dir_=None, out_dir=None) -> dict:
    """Table of the 34 phase-1 bases (settings, cut, phase-1 metrics) as CSV and LaTeX, from the
    summary only. Returns the paths written."""
    s = load_summary(summary_dir_)
    out = Path(out_dir) if out_dir else stage_dir()
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cols = ("unitKey", "lr1", "l2", "width", "tauAtt1", "epochs", "cut", "sourceTestBA", "targetTestBA",
            "targetTestAUC", "targetTestKappa", "evalPoolBA", "evalPoolAUC")
    with open(tables / "phase1_bases.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in s["bases"]:
            w.writerow([r[c] for c in cols])
    head = ("Unit & lr & $\\ell_2$ & Width & $\\tau$ & Epochs & Cut & BA$_{\\mathrm{src}}$ & BA & AUC & "
            "$\\kappa$ & BA$_{\\mathrm{pool}}$ & AUC$_{\\mathrm{pool}}$ \\\\")
    lines = ["% generated by MIL_CREDA_EEG.reproduce.stage01_bases.render from the stored summary",
             "\\begin{tabular}{lcccccccccccc}", "\\toprule", head, "\\midrule"]
    for fam, label in (("synthetic", "Synthetic (24 units)"), ("adhd", "ADHD/Control Children (10 units)")):
        lines.append(f"\\multicolumn{{13}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for r in (r for r in s["bases"] if r["family"] == fam):
            lines.append(" & ".join([r["unitKey"], f"{r['lr1']:.2e}", f"{r['l2']:.2e}", str(r["width"]),
                                     f"{r['tauAtt1']:.3f}", str(r["epochs"]), _fmt(r["cut"]),
                                     _fmt(r["sourceTestBA"]), _fmt(r["targetTestBA"]), _fmt(r["targetTestAUC"]),
                                     _fmt(r["targetTestKappa"]), _fmt(r["evalPoolBA"]),
                                     _fmt(r["evalPoolAUC"])]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (tables / "phase1_bases.tex").write_text("\n".join(lines))
    with open(tables / "phase1_paper_numbers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("quantity", "value", "paper", "main.tex line"))
        for (name, got, paper, _), line in zip(paper_checks(summary_dir_), PAPER_LINES):
            w.writerow((name, got, paper, line))
    return {"csv": tables / "phase1_bases.csv", "tex": tables / "phase1_bases.tex",
            "numbers": tables / "phase1_paper_numbers.csv"}


# -------------------------------------------------------------------------------------- checks

#: paper/main.tex, quoted: L370 "Every method exceeded the first phase (0.576) on average";
#: L459 "Phase 1 (source only) & 4.60 & 0.655 & 0.717 & 0.309 & ..." (Table tab:ieee-ranking, cef block)
PAPER = (("synthetic first phase, mean BA over the 24 units (evaluation pool)", "synthetic", "evalPoolBA", 0.576),
         ("ADHD Phase 1 (source only), mean BA over the 10 units", "adhd", "targetTestBA", 0.655),
         ("ADHD Phase 1 (source only), mean AUC over the 10 units", "adhd", "targetTestAUC", 0.717),
         ("ADHD Phase 1 (source only), mean kappa over the 10 units", "adhd", "targetTestKappa", 0.309))
PAPER_LINES = ("main.tex:370", "main.tex:459", "main.tex:459", "main.tex:459")


def _mean(rows, family, field):
    vals = [r[field] for r in rows if r["family"] == family and r[field] is not None]
    return round(sum(vals) / len(vals), 3) if vals else float("nan")


def paper_checks(summary_dir_=None) -> list[tuple[str, float, float, float]]:
    """(name, obtained, paper, tolerance): tolerance 0 on the stored summary, REGEN_TOLERANCE on a
    regenerated one (pass its `summary/` folder)."""
    rows = load_summary(summary_dir_)["bases"]
    tol = 0.0 if summary_dir_ is None or Path(summary_dir_).resolve() == summary_dir().resolve() else REGEN_TOLERANCE
    return [(name, _mean(rows, fam, field), paper, tol) for name, fam, field, paper in PAPER]


def compare_regenerated(regen_summary_dir=None, tolerance: float = REGEN_TOLERANCE) -> list[tuple]:
    """Means of the regenerated bases next to the paper's bases over the same units (so a partial
    portion compares too): cut, source-test BA, target-test BA and AUC, and evaluation-pool BA."""
    regen = load_summary(regen_summary_dir or regenerated_dir() / "summary")["bases"]
    paper = {r["unitKey"]: r for r in load_summary()["bases"]}
    out = []
    for fam in ("synthetic", "adhd"):
        got = [r for r in regen if r["family"] == fam]
        ref = [paper[r["unitKey"]] for r in got]
        if not got:
            continue
        for field in ("cut", "sourceTestBA", "targetTestBA", "targetTestAUC", "evalPoolBA"):
            if fam == "adhd" and field == "evalPoolBA":
                continue
            a, b = _mean(got, fam, field), _mean(ref, fam, field)
            if a == a and b == b:                # skip NaN (e.g. evaluation pool not computed)
                out.append((f"{fam} mean {field} ({len(got)} bases)", a, b, tolerance))
    return out


def manifests_match() -> dict:
    """Which tracked `Manifests/*_bases/*.pt` files are byte-identical to the bases the paper used
    (the sha256 recorded in the stored summary)."""
    paper = {r["unitKey"]: r["sha256"] for r in load_summary()["bases"]}
    same, differ = [], []
    for key, sha in paper.items():
        path = bases_dir(False, "adhd" if key.startswith("ieee") else "synthetic") / f"{key}-s1.pt"
        (same if path.exists() and _sha256(path) == sha else differ).append(key)
    return {"identical": same, "different": differ}
