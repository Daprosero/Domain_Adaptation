"""Plan v63, Secs. 6.2 [exp:K2-emparejamiento] and 6.4 part 3 [exp:K4-conectividad].

Inference on the saved models and the processed windows only; nothing is trained.

**K2 probe.** In each rotation the probe subjects are the test-fold subjects holding both
conditions of the transfer; their other condition is read only here (`assignment.probe_bag`).
With the saved MIL-CREDA model of each report seed, each probe subject's target bag (target
trunk) and every probe subject's source bag (source trunk) are compared in the bag-kernel
space at the frozen sigma (`v63_saved.kernel_distances`). A subject's separation is its mean
distance to the OTHER probe subjects' source bags minus its distance to its own, averaged
over the seeds. The same separation with the global-term-only variant (Sec. 6.3) isolates
the local term. The four criterion tests, on L6 -> Robots and Robots -> L6 (declared in
advance): separation of MIL-CREDA > 0, and MIL-CREDA minus global only, in each transfer;
Wilcoxon over the probe subjects, Holm over the four. The other transfers are reported as
a complement with the same tests.

**Diagnostics, before the verdict,** at phase 2's start (the bridged model, `Store.bridged`)
over the probe subjects: the entropy of each target bag's local correspondence over the
probe source bags of its class divided by log(their number); the fraction whose own source
bag is the nearest of its class, and the margin ratio (distance to its own source bag over
the median distance between source bags of the same class); and the push, the reduction of
the distance to its own source bag from start to end, with MIL-CREDA and with global only.
**Reading condition:** median entropy ratio below 0.9 or median margin ratio at least 1.0;
otherwise the transfer's verdict is "inconclusive (the mechanism had no margin)". The
comparators are reported as context, without criterion, with the Euclidean distance of
their bag representations (target from their records, source through their frozen
phase-1 source side), on the probe subjects assigned to the target side.

**K4 part 3.** On the two declared transfers, each target test subject's PLV descriptors
(MIL-CREDA's attention) against each source training subject of its predicted class
(MIL-CREDA's source-side attention), the distances weighted by the local correspondence pi
of the record versus by pi permuted within the class (20 permutations, mean); per subject
the difference, averaged over seeds; Wilcoxon over subjects, Holm over the two, support
without criterion. Control: on the probe subjects, the distance to the own source bag's
descriptors versus the mean distance to the other probe subjects'.

**Median-displacement figure.** The probe subject of median MIL-CREDA push (fixed rule): its
target bag's and its own source bag's alpha-band graphs and their descriptors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG import mil_creda_model as M
from MIL_CREDA_EEG import v63_connectivity as G
from MIL_CREDA_EEG import v63_records as R
from MIL_CREDA_EEG import v63_saved as S
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["6.2", "6.4"],
    "experiments": ["K2-emparejamiento", "K4-conectividad"],
    "invariants": [
        "the_separation_is_mean_other_minus_own_distance",
        "the_reading_condition_is_entropy_below_0_9_or_margin_at_least_1",
        "the_four_k2_tests_share_one_holm_family",
        "the_permuted_correspondence_stays_within_the_class",
        "the_displacement_figure_subject_is_the_median_push",
    ],
}

DECLARED = ("L6->Robots", "Robots->L6")
PERMUTATIONS = 20
PERM_SEED = 20261109


def separation(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    other = (D.sum(1) - np.diag(D)) / max(n - 1, 1)
    return other - np.diag(D)


@torch.no_grad()
def _source_distances(model, source: Sequence[two_phase.BagTensor]) -> np.ndarray:
    ps = model.bag_pairs(model.embed([b.windows for b in source], "source", two_phase.EVAL_CHUNK))
    K = M.bag_kernel_matrix(ps, ps, model.sigma)
    d = torch.diagonal(K)
    return torch.sqrt(torch.clamp(d[:, None] + d[None, :] - 2 * K, min=0)).double().numpy()


@torch.no_grad()
def entropy_ratios(model, tgt, src, labels: np.ndarray) -> np.ndarray:
    pt = model.bag_pairs(model.embed([b.windows for b in tgt], "target", two_phase.EVAL_CHUNK))
    ps = model.bag_pairs(model.embed([b.windows for b in src], "source", two_phase.EVAL_CHUNK))
    K = M.bag_kernel_matrix(ps, pt, model.sigma)
    G_t = model.classify(model.represent(pt))
    lab = torch.as_tensor(labels)
    out = []
    for i in range(len(tgt)):
        pi = V._corr(K[:, i], lab, G_t[i], V.row_tau(K[:, i])).double().numpy()
        same = labels == labels[i]
        p = pi[same] / max(pi[same].sum(), 1e-300)
        n = int(same.sum())
        out.append(float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(n)) if n > 1 else np.nan)
    return np.asarray(out)


def probe_measures(store: S.Store, t) -> dict:
    """subject -> seed-mean measures over the probe of every rotation and seed."""
    acc: dict = {}
    seeds = sorted({s for _, s in S.records(store.product, t, S.MIL_CREDA)})
    for r in range(4):
        probe = store.probe(t, r)
        if len(probe) < 2:
            continue
        subs = sorted(probe)
        tgt = [probe[s][0] for s in subs]
        src = [probe[s][1] for s in subs]
        labels = np.asarray([b.label for b in src])
        for seed in seeds:
            try:
                end = store.mil_creda(t, r, seed)
                start = store.bridged(t, r, seed)
            except FileNotFoundError:
                continue
            D, D0 = S.kernel_distances(end, tgt, src), S.kernel_distances(start, tgt, src)
            m = {"separation": separation(D), "pushMIL-CREDA": np.diag(D0) - np.diag(D)}
            try:
                go = store.mil_creda(t, r, seed, S.GLOBAL_ONLY)
                Dg = S.kernel_distances(go, tgt, src)
                m["separationGlobalOnly"] = separation(Dg)
                m["pushGlobalOnly"] = np.diag(D0) - np.diag(Dg)
            except FileNotFoundError:
                pass
            Ds = _source_distances(start, src)
            m["entropyRatio"] = entropy_ratios(start, tgt, src, labels)
            margin, nearest = [], []
            for i in range(len(subs)):
                same = np.flatnonzero(labels == labels[i])
                pairs = [Ds[a, b] for a in same for b in same if a < b]
                margin.append(D0[i, i] / np.median(pairs) if pairs else np.nan)
                nearest.append(float(same[np.argmin(D0[i, same])] == i))
            m["marginRatio"], m["ownNearest"] = np.asarray(margin), np.asarray(nearest)
            for k, v in m.items():
                for i, s in enumerate(subs):
                    acc.setdefault(s, {}).setdefault(k, []).append(float(v[i]))
    return {s: {k: float(np.nanmean(v)) for k, v in d.items()} for s, d in acc.items()}


def readable(measures: Mapping) -> dict:
    ent = [m["entropyRatio"] for m in measures.values() if np.isfinite(m.get("entropyRatio", np.nan))]
    mar = [m["marginRatio"] for m in measures.values() if np.isfinite(m.get("marginRatio", np.nan))]
    me = float(np.median(ent)) if ent else float("nan")
    mm = float(np.median(mar)) if mar else float("nan")
    return {"medianEntropyRatio": me, "medianMarginRatio": mm,
            "ownNearestFraction": float(np.mean([m["ownNearest"] for m in measures.values()]))
            if measures else float("nan"),
            "readable": bool((np.isfinite(me) and me < 0.9) or (np.isfinite(mm) and mm >= 1.0))}


def k2_rows(store: S.Store, transfers: Sequence) -> tuple[list, list, dict]:
    crit, comp, diag, measures_of = [], [], [], {}
    for t in transfers:
        if not S.records(store.product, t, S.MIL_CREDA):
            continue
        m = probe_measures(store, t)
        if not m:
            continue
        measures_of[t.name] = m
        cond = readable(m)
        pushes = {k: float(np.nanmedian([x.get(k, np.nan) for x in m.values()]))
                  for k in ("pushMIL-CREDA", "pushGlobalOnly")}
        diag.append({"transfer": t.name, "probeSubjects": len(m), **cond,
                     **{f"median_{k}": v for k, v in pushes.items()}})
        fam = "K2 criterion" if t.name in DECLARED else "K2 complement"
        target = crit if t.name in DECLARED else comp
        extra = {"transfer": t.name, "readable": cond["readable"],
                 "readingVerdict": None if cond["readable"]
                 else "inconclusive (the mechanism had no margin)"}
        target.append(S.wilcoxon_row(f"K2 separation of MIL-CREDA, {t.name}",
                                     [x["separation"] for x in m.values()],
                                     hypothesis="K2-emparejamiento", experiment="K2-emparejamiento",
                                     family=fam, **extra))
        paired = [x["separation"] - x["separationGlobalOnly"] for x in m.values()
                  if "separationGlobalOnly" in x]
        if paired:
            target.append(S.wilcoxon_row(f"K2 separation, MIL-CREDA minus global term only, {t.name}",
                                         paired, hypothesis="K2-emparejamiento",
                                         experiment="K2-emparejamiento", family=fam, **extra))
    return (S.holm_family(crit, "the four K2 tests") + S.holm_family(comp, "K2 complement"),
            diag, measures_of)


def comparator_rows(store: S.Store, transfers: Sequence, strategies: Sequence[str]) -> list:
    """Context without criterion: Euclidean separation of bag representations."""
    from MIL_CREDA_EEG import search
    rows = []
    for t in transfers:
        for s in strategies:
            recs = S.records(store.product, t, s)
            if not recs:
                continue
            g = search.PHASE_ONE_GROUP_OF[s]
            per: dict = {}
            for (r, seed), rec in recs.items():
                probe = {k: v for k, v in store.probe(t, r).items() if k in rec["subjects"]}
                if len(probe) < 2:
                    continue
                model, _ = store.phase_one(t, r, seed, g)
                subs = sorted(probe)
                Zt = np.asarray([rec["targetRepresentations"][rec["subjects"].index(x)] for x in subs])
                Zs = two_phase.represent(model, [probe[x][1] for x in subs], "source",
                                         torch.device("cpu")).numpy()
                D = np.linalg.norm(Zt[:, None] - Zs[None], axis=-1)
                for x, v in zip(subs, separation(D)):
                    per.setdefault(x, []).append(float(v))
            if per:
                rows.append(S.wilcoxon_row(f"K2 context, {s} Euclidean separation, {t.name}",
                                           [np.mean(v) for v in per.values()],
                                           hypothesis="K2-emparejamiento",
                                           experiment="K2-emparejamiento",
                                           family="K2 comparators, no criterion", transfer=t.name,
                                           method=s))
    return S.holm_family(rows, "K2 comparators, no criterion")


def permuted_within_class(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(pi)


def k4_part3_rows(store: S.Store) -> list:
    rows = []
    rng = np.random.default_rng(PERM_SEED)
    for name in DECLARED:
        t = V.transfer(name)
        recs = S.records(store.product, t, S.MIL_CREDA)
        per, control = {}, {}
        for (r, seed), rec in recs.items():
            if "correspondence" not in rec:
                continue
            model = store.mil_creda(t, r, seed)
            src_subjects = rec["correspondenceSourceSubjects"]
            src_labels = np.asarray(rec["correspondenceSourceLabels"])
            src_desc = {}
            for s in src_subjects:
                bag = store.prepared(t, s, t.source)
                att = R.attention(model, [bag], "source", "cpu")[0]
                idx = R.selected_indices(len(store.bags.get(s, t.source).marks()["window_index"]))
                src_desc[s] = G.subject_descriptors(store.raw_windows(s, t.source, idx,
                                                                      V.channels_of(t)), att)
            for k, s in enumerate(rec["subjects"]):
                w = rec["windows"][k]
                x = G.subject_descriptors(store.raw_windows(s, w["condition"], w["index"],
                                                            V.channels_of(t)), rec["attention"][k])
                cls = int(rec["probabilities"][k] > 0.5)
                same = np.flatnonzero(src_labels == cls)
                if len(same) < 2:
                    continue
                pi = np.asarray(rec["correspondence"][k])[same]
                pi = pi / pi.sum()
                d = np.asarray([np.linalg.norm(x - src_desc[src_subjects[j]]) for j in same])
                perm = np.mean([permuted_within_class(pi, rng) @ d for _ in range(PERMUTATIONS)])
                per.setdefault(s, []).append(float(pi @ d - perm))
            probe = store.probe(t, r)
            if len(probe) >= 2:
                subs = sorted(probe)
                Xt = [G.subject_descriptors(store.raw_windows(
                    s, probe[s][0].condition, R.selected_indices(len(store.bags.get(
                        s, probe[s][0].condition).marks()["window_index"])), V.channels_of(t)),
                    R.attention(model, [probe[s][0]], "target", "cpu")[0]) for s in subs]
                Xs = [G.subject_descriptors(store.raw_windows(
                    s, probe[s][1].condition, R.selected_indices(len(store.bags.get(
                        s, probe[s][1].condition).marks()["window_index"])), V.channels_of(t)),
                    R.attention(model, [probe[s][1]], "source", "cpu")[0]) for s in subs]
                D = np.linalg.norm(np.asarray(Xt)[:, None] - np.asarray(Xs)[None], axis=-1)
                for s, v in zip(subs, separation(D)):
                    control.setdefault(s, []).append(float(v))
        if per:
            rows.append(S.wilcoxon_row(
                f"K4 part 3, pi-weighted minus permuted-pi descriptor distance, {name}",
                [np.mean(v) for v in per.values()], hypothesis="K4-conectividad",
                experiment="K4-conectividad", family="K4 part 3, support", transfer=name))
        if control:
            rows.append(S.wilcoxon_row(
                f"K4 part 3 control, own source bag separation in descriptors, {name}",
                [np.mean(v) for v in control.values()], hypothesis="K4-conectividad",
                experiment="K4-conectividad", family="K4 part 3 control", transfer=name))
    part = S.holm_family([x for x in rows if x["family"] == "K4 part 3, support"],
                         "K4 part 3 over the two declared transfers")
    ctrl = S.holm_family([x for x in rows if x["family"] == "K4 part 3 control"],
                         "K4 part 3 control")
    return part + ctrl


def displacement_figure(store: S.Store, measures_of: Mapping, out: Path) -> Path | None:
    from matplotlib import pyplot as plt
    from MIL_CREDA_EEG.ps_figures import apply_style, save_figure
    pushes = {(n, s): m["pushMIL-CREDA"] for n in DECLARED for s, m in measures_of.get(n, {}).items()}
    pick = S.median_rule(pushes)
    if pick is None:
        return None
    name, subject = pick
    t = V.transfer(name)
    for r in range(4):
        probe = store.probe(t, r)
        if subject in probe:
            break
    seed = min(s for rr, s in S.records(store.product, t, S.MIL_CREDA) if rr == r)
    model = store.mil_creda(t, r, seed)
    graphs, descs = [], []
    for bag, domain in ((probe[subject][0], "target"), (probe[subject][1], "source")):
        idx = R.selected_indices(len(store.bags.get(subject, bag.condition).marks()["window_index"]))
        att = R.attention(model, [bag], domain, "cpu")[0]
        g = G.weighted_graph(G.window_plv(store.raw_windows(subject, bag.condition, idx,
                                                             V.channels_of(t))), att)
        graphs.append(g["alpha"]); descs.append(G.descriptors(g))
    apply_style(body_pt=10, printed_width_fraction=1.0)
    fig, axes = plt.subplots(1, 3)
    for ax, g, lab in zip(axes[:2], graphs, ("target bag", "own source bag")):
        ax.imshow(g, vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel(f"{lab}, alpha PLV"); ax.set_xticks([]); ax.set_yticks([])
    axes[2].plot(descs[0], lw=0.8, label="target bag")
    axes[2].plot(descs[1], lw=0.8, label="own source bag")
    axes[2].set_xlabel("descriptor index"); axes[2].legend(frameon=False)
    path = out / f"displacement_median_{subject}.pdf"
    save_figure(fig, path, png_preview=True)
    plt.close(fig)
    return path


def analyze(product: str | Path, bags, *, figures: bool = True, comparators: bool = True) -> dict:
    from MIL_CREDA_EEG import model_store
    product = Path(product)
    store = S.Store(product, bags)
    rows, diag, measures_of = k2_rows(store, V.TRANSFERS)
    if comparators:
        rows += comparator_rows(store, [V.transfer(n) for n in DECLARED],
                                [s for s in model_store.STRATEGIES if s != S.MIL_CREDA])
    rows += k4_part3_rows(store)
    (product / "Results").mkdir(parents=True, exist_ok=True)
    (product / "Results" / "significance_v63_K2_K4.json").write_text(
        json.dumps(rows, indent=1, default=float), encoding="utf-8")
    tdir = product / "Results" / "v63" / "K2-emparejamiento" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "mechanism_diagnostics.json").write_text(json.dumps(diag, indent=1, default=float))
    fig = None
    if figures:
        fig = displacement_figure(store, measures_of,
                                  product / "Results" / "v63" / "K2-emparejamiento" / "qualitative")
    return {"rows": len(rows), "diagnostics": diag, "figure": str(fig) if fig else None}
