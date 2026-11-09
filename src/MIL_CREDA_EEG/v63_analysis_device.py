"""Plan v63, Sec. 5.3 qualitative reading [exp:C-dispositivo]: UMAP and graph descriptors.

**UMAP.** For each direction of the core device pair, four panels of the bag
representations of the source and target TEST subjects (first report seed): predict and
MIL-CREDA, before and after phase 2. Before is the phase-1 model (the common group for
predict, MIL-CREDA's own group); after is the phase-2 model (for predict, which has no
phase 2, the same representations, labelled so). Colour is the domain, marker the class.
It illustrates the domain-classifier AUC of Sec. 4.4 and replaces nothing.

**Graph descriptors.** On every device transfer with records, for predict, MIL-CREDA and
the best rival by mean pooled balanced accuracy over the seeds: each target test subject's
PLV graph descriptors (`v63_connectivity`), its windows weighted by that method's attention
averaged over the seeds. A subject's class separation is its distance to the centre of the
other predicted class minus its distance to the centre of its own predicted class (the
centres over the transfer's other target test subjects, the subject left out; the predicted
class is the seed-mean probability above 0.5, so no target label enters). MIL-CREDA is
compared with each of the other two by a paired Wilcoxon over the subjects, Holm over the
comparisons, rank-biserial correlation and the Hodges-Lehmann estimate with its 95%
interval. A supporting reading: it decides no criterion.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from MIL_CREDA_EEG import model_store
from MIL_CREDA_EEG import v63_analysis as An
from MIL_CREDA_EEG import v63_connectivity as G
from MIL_CREDA_EEG import v63_saved as S
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["5.3"],
    "experiments": ["C-dispositivo"],
    "invariants": [
        "the_class_separation_leaves_the_subject_out_of_the_centres",
        "the_predicted_class_reads_no_target_label",
        "the_best_rival_is_chosen_by_mean_pooled_balanced_accuracy",
        "the_gsa_comparisons_are_paired_wilcoxon_with_holm",
    ],
}

EXPERIMENT = "C-dispositivo"
RIVALS = tuple(s for s in model_store.STRATEGIES if s != S.MIL_CREDA)


def mean_ba(recs: Mapping) -> float:
    seeds = sorted({s for _, s in recs})
    vals = []
    for seed in seeds:
        y = [v for (r, s), rec in recs.items() if s == seed for v in rec["labels"]]
        p = [v for (r, s), rec in recs.items() if s == seed for v in rec["probabilities"]]
        vals.append(An.ba(y, p))
    return float(np.mean(vals)) if vals else float("nan")


def best_rival(product: Path, t) -> str | None:
    scores = {m: mean_ba(S.records(product, t, m)) for m in RIVALS}
    scores = {m: v for m, v in scores.items() if np.isfinite(v)}
    return max(scores, key=lambda m: (scores[m], -RIVALS.index(m))) if scores else None


def class_separation(desc: Mapping[str, np.ndarray], predicted: Mapping[str, int]) -> dict:
    out = {}
    for s, x in desc.items():
        own = [desc[o] for o in desc if o != s and predicted[o] == predicted[s]]
        other = [desc[o] for o in desc if o != s and predicted[o] != predicted[s]]
        if not own or not other:
            out[s] = float("nan")
            continue
        out[s] = float(np.linalg.norm(x - np.mean(other, 0)) - np.linalg.norm(x - np.mean(own, 0)))
    return out


def method_descriptors(store: S.Store, t, recs: Mapping) -> tuple[dict, dict]:
    subjects = S.by_subject(recs)
    desc, pred = {}, {}
    for s, e in subjects.items():
        rec, k = next(iter(e["seeds"].values()))
        w = rec["windows"][k]
        att = np.mean([np.asarray(r["attention"][j]) for r, j in e["seeds"].values()], 0)
        desc[s] = G.subject_descriptors(
            store.raw_windows(s, w["condition"], w["index"], V.channels_of(t)), att)
        pred[s] = S.predicted_class(e)
    return desc, pred


def gsa_rows(product: Path, store: S.Store, transfers: Sequence) -> tuple[list, list]:
    rows, table = [], []
    for t in transfers:
        if t.arm != "device":
            continue
        rival = best_rival(product, t)
        methods = [m for m in (S.MIL_CREDA, An.PREDICT, rival) if m]
        recs = {m: S.records(product, t, m) for m in methods}
        if not recs[S.MIL_CREDA] or any(not v for v in recs.values()):
            continue
        sep = {}
        for m in methods:
            desc, pred = method_descriptors(store, t, recs[m])
            sep[m] = class_separation(desc, pred)
            table += [{"transfer": t.name, "method": m, "subject": s, "separation": v}
                      for s, v in sep[m].items()]
        for other in methods[1:]:
            common = sorted(set(sep[S.MIL_CREDA]) & set(sep[other]))
            diff = [sep[S.MIL_CREDA][s] - sep[other][s] for s in common]
            rows.append(S.wilcoxon_row(
                f"GSA class separation, MIL-CREDA vs {other}, {t.name}", diff,
                hypothesis=EXPERIMENT, experiment=EXPERIMENT, family="support: GSA separation",
                transfer=t.name, versus=other, bestRival=rival))
    return S.holm_family(rows, "GSA class separation comparisons"), table


def umap_figure(product: Path, t, out: Path, seed: int = 1) -> Path | None:
    import umap
    from matplotlib import pyplot as plt
    from MIL_CREDA_EEG.ps_figures import apply_style, save_figure
    panels = [("predict", "before phase 2", An.PREDICT),
              ("predict", "after phase 2 (none: unchanged)", An.PREDICT),
              ("MIL-CREDA", "before phase 2", "phase1 [MIL-CREDA]"),
              ("MIL-CREDA", "after phase 2", S.MIL_CREDA)]
    data = []
    for name, when, method in panels:
        recs = {k: v for k, v in S.records(product, t, method).items() if k[1] == seed}
        if not recs:
            return None
        Zs = np.concatenate([r["sourceRepresentations"] for r in recs.values()])
        Zt = np.concatenate([r["targetRepresentations"] for r in recs.values()])
        yt = np.concatenate([r["labels"] for r in recs.values()])
        data.append((name, when, Zs, Zt, yt))
    apply_style(body_pt=10, printed_width_fraction=1.0)
    fig, axes = plt.subplots(2, 2)
    for ax, (name, when, Zs, Zt, yt) in zip(axes.ravel(), data):
        n = len(Zs) + len(Zt)
        E = umap.UMAP(random_state=0, n_neighbors=min(15, n - 1),
                      init="spectral" if n >= 20 else "random").fit_transform(np.r_[Zs, Zt])
        ax.scatter(*E[:len(Zs)].T, s=8, c="tab:blue", marker="o", label="source")
        for cls, mk in ((0, "o"), (1, "^")):
            sel = yt == cls
            ax.scatter(*E[len(Zs):][sel].T, s=10, c="tab:orange", marker=mk,
                       label=f"target, class {cls}")
        ax.set_xlabel(f"{name}, {when}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(loc="best", frameon=False)
    path = out / f"umap_{model_store.transfer_slug(t)}.pdf"
    save_figure(fig, path, png_preview=True)
    plt.close(fig)
    return path


def analyze(product: str | Path, bags, *, figures: bool = True) -> dict:
    product = Path(product)
    root = product / "Results" / "v63" / EXPERIMENT
    store = S.Store(product, bags)
    rows, table = gsa_rows(product, store, V.TRANSFERS)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    with (root / "tables" / "gsa_separation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["transfer", "method", "subject", "separation"])
        w.writeheader(); w.writerows(table)
    (product / "Results" / "significance_v63_gsa.json").write_text(
        json.dumps(rows, indent=1, default=float), encoding="utf-8")
    figs = []
    if figures:
        for t in (V.transfer("L6->Robots"), V.transfer("Robots->L6")):
            p = umap_figure(product, t, root / "figures")
            if p:
                figs.append(str(p))
    return {"rows": len(rows), "figures": figs}
