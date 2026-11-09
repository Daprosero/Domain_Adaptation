"""Plan v63, Sec. 6.1 [exp:K1-atencion]: does the attention select something meaningful?

Read from the refit records (attention per window, window markers) and, for the deletion
test only, from the saved phase-2 models (inference); nothing is trained.

**Arms** (each analysed apart, every target subject once): task (Robots <-> CogniFit,
EMOTIV on both sides; marked = integrity mark (step 2.3) or bad contact (step 2.5)), device
as support (L6 -> Robots, the Robots side), level (L1 <-> L11; the behavioural event mark).
A subject enters a marked-window contrast only when its bag holds marked and clean windows
(the count is reported). Per subject the contrast is the mean attention weight of the
marked (or event) windows minus that of the others, computed in every report seed and
averaged over the seeds.

**Criterion tests:** T1, MIL-CREDA's marked-minus-clean contrast in the task arm; T2, that
contrast of MIL-CREDA minus the own variant without consensus (Sec. 6.3), paired on the
same subjects; T3, MIL-CREDA's event-minus-rest contrast in the level arm. Wilcoxon over
the subjects, Holm over the three, rank-biserial correlation and Hodges-Lehmann with its
95% interval. **References without criterion:** the same contrasts for the variant without
consensus, the common attention of the non-MIL comparators (predict's model) and AdaptMIL,
per arm, and the device arm's T1/T2 as support; Holm within that family.

**Deletion test (fidelity).** In each target bag, the q = 10% and 25% windows of highest
MIL-CREDA attention are removed (at least one, never all) and the drop of MIL-CREDA's
score of the subject's class is measured; it is compared per subject with removing as many
windows at random (seeded by seed and subject) and with removing the windows AdaptMIL's
attention ranks highest. Paired Wilcoxon, Holm over every (arm, q, comparison).

**TCAMIL-style figure.** The subject of median T1 contrast in the task arm and of median T3
contrast in the level arm (fixed rule): its windows in time order, shaded by MIL-CREDA's
seed-mean attention, marked windows and event windows drawn on top.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG import v63_saved as S
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["6.1"],
    "experiments": ["K1-atencion"],
    "invariants": [
        "a_marked_contrast_needs_marked_and_clean_windows",
        "the_contrast_is_averaged_over_the_seeds_per_subject",
        "t1_t2_t3_share_one_holm_family",
        "the_deletion_never_removes_every_window",
        "the_figure_subject_is_the_median_contrast",
    ],
}

EXPERIMENT = "K1-atencion"
ARMS = {"task": (("Robots->CogniFit", "CogniFit->Robots"), "marked"),
        "device": (("L6->Robots",), "marked"),
        "level": (("L1->L11", "L11->L1"), "event")}
REFERENCES = (S.NO_CONSENSUS, "predict", "AdaptMIL")
FRACTIONS = (0.10, 0.25)


def window_flags(w: Mapping, marker: str) -> np.ndarray | None:
    if marker == "event":
        return None if w.get("event") is None else np.asarray(w["event"], bool)
    parts = [np.asarray(w[k], bool) for k in ("integrity", "contactBad") if w.get(k) is not None]
    return np.logical_or.reduce(parts) if parts else None


def contrast(attention: Sequence[float], flags: np.ndarray | None) -> float:
    a = np.asarray(attention, float)
    if flags is None or not flags.any() or flags.all():
        return float("nan")
    return float(a[flags].mean() - a[~flags].mean())


def subject_contrasts(product: Path, method: str, transfers: Sequence[str], marker: str) -> dict:
    out = {}
    for name in transfers:
        for s, e in S.by_subject(S.records(product, V.transfer(name), method)).items():
            vals = [contrast(rec["attention"][k], window_flags(rec["windows"][k], marker))
                    for rec, k in e["seeds"].values()]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                out[(name, s)] = float(np.mean(vals))
    return out


def criterion_rows(product: Path) -> tuple[list, dict]:
    task, task_marker = ARMS["task"]
    level, level_marker = ARMS["level"]
    mc_task = subject_contrasts(product, S.MIL_CREDA, task, task_marker)
    nc_task = subject_contrasts(product, S.NO_CONSENSUS, task, task_marker)
    mc_level = subject_contrasts(product, S.MIL_CREDA, level, level_marker)
    common = sorted(set(mc_task) & set(nc_task))
    rows = [S.wilcoxon_row("T1: MIL-CREDA attention, marked minus clean windows, task arm",
                           list(mc_task.values()), hypothesis=EXPERIMENT, experiment=EXPERIMENT,
                           family="K1 criterion", test_id="T1", eligibleSubjects=len(mc_task)),
            S.wilcoxon_row("T2: marked-minus-clean contrast, MIL-CREDA minus own without "
                           "consensus, task arm", [mc_task[k] - nc_task[k] for k in common],
                           hypothesis=EXPERIMENT, experiment=EXPERIMENT, family="K1 criterion",
                           test_id="T2", eligibleSubjects=len(common)),
            S.wilcoxon_row("T3: MIL-CREDA attention, event minus other windows, level arm",
                           list(mc_level.values()), hypothesis=EXPERIMENT, experiment=EXPERIMENT,
                           family="K1 criterion", test_id="T3", eligibleSubjects=len(mc_level))]
    return S.holm_family(rows, "T1, T2 and T3"), {"T1": mc_task, "T3": mc_level}


def reference_rows(product: Path) -> list:
    rows = []
    for arm, (transfers, marker) in ARMS.items():
        methods = REFERENCES + ((S.MIL_CREDA,) if arm == "device" else ())
        for m in methods:
            c = subject_contrasts(product, m, transfers, marker)
            if c:
                rows.append(S.wilcoxon_row(f"{m} attention, {marker} minus other windows, {arm} arm",
                                           list(c.values()), hypothesis=EXPERIMENT,
                                           experiment=EXPERIMENT, family="K1 references",
                                           arm=arm, method=m, eligibleSubjects=len(c)))
        if arm == "device":
            a = subject_contrasts(product, S.MIL_CREDA, transfers, marker)
            b = subject_contrasts(product, S.NO_CONSENSUS, transfers, marker)
            k = sorted(set(a) & set(b))
            if k:
                rows.append(S.wilcoxon_row("device arm: MIL-CREDA minus own without consensus",
                                           [a[x] - b[x] for x in k], hypothesis=EXPERIMENT,
                                           experiment=EXPERIMENT, family="K1 references",
                                           arm=arm, eligibleSubjects=len(k)))
    return S.holm_family(rows, "K1 references, no criterion")


def removal_count(n: int, q: float) -> int:
    return int(min(max(1, round(q * n)), n - 1))


@torch.no_grad()
def class_score(model, bag: two_phase.BagTensor, keep: np.ndarray, label: int) -> float:
    sub = two_phase.BagTensor(bag.subject, bag.condition, bag.label, bag.windows[torch.as_tensor(keep)])
    return float(V.probabilities(model, [sub], "target", "cpu")[0] if label == 1
                 else 1 - V.probabilities(model, [sub], "target", "cpu")[0])


def _rng(seed: int, subject: str) -> np.random.Generator:
    return np.random.default_rng([seed, int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16)])


def deletion_drops(store: S.Store, t, mc: Mapping, adapt: Mapping) -> dict:
    """subject -> q -> {"top", "random", "adaptmil"} seed-mean drops of the class score."""
    out: dict = {}
    for (r, seed), rec in mc.items():
        other = adapt.get((r, seed))
        if other is None:
            continue
        model = store.mil_creda(t, r, seed)
        for k, s in enumerate(rec["subjects"]):
            if s not in other["subjects"]:
                continue
            bag = store.prepared(t, s, rec["conditions"][k])
            a = np.asarray(rec["attention"][k])
            b = np.asarray(other["attention"][other["subjects"].index(s)])
            n, y = len(a), int(rec["labels"][k])
            if n < 2 or len(b) != n or bag.windows.shape[0] != n:
                continue
            full = class_score(model, bag, np.arange(n), y)
            for q in FRACTIONS:
                m = removal_count(n, q)
                drops = {}
                for name, removed in (("top", np.argsort(-a, kind="stable")[:m]),
                                      ("random", _rng(seed, s).permutation(n)[:m]),
                                      ("adaptmil", np.argsort(-b, kind="stable")[:m])):
                    keep = np.setdiff1d(np.arange(n), removed)
                    drops[name] = full - class_score(model, bag, keep, y)
                cell = out.setdefault(s, {}).setdefault(q, {"top": [], "random": [], "adaptmil": []})
                for name, v in drops.items():
                    cell[name].append(v)
    return {s: {q: {k: float(np.mean(v)) for k, v in d.items()} for q, d in qs.items()}
            for s, qs in out.items()}


def deletion_rows(product: Path, store: S.Store) -> list:
    rows = []
    for arm, (transfers, _) in ARMS.items():
        drops: dict = {}
        for name in transfers:
            t = V.transfer(name)
            drops.update(deletion_drops(store, t, S.records(product, t, S.MIL_CREDA),
                                        S.records(product, t, "AdaptMIL")))
        for q in FRACTIONS:
            for other, label in (("random", "random windows"), ("adaptmil", "AdaptMIL's top windows")):
                diff = [d[q]["top"] - d[q][other] for d in drops.values() if q in d]
                if diff:
                    rows.append(S.wilcoxon_row(
                        f"deletion, {arm} arm, q {q:.2f}: MIL-CREDA top windows minus {label}",
                        diff, hypothesis=EXPERIMENT, experiment=EXPERIMENT,
                        family="K1 deletion", arm=arm, q=q, versus=other))
    return S.holm_family(rows, "deletion test comparisons")


def tcamil_figure(product: Path, store: S.Store, subject_key: tuple, marker: str, out: Path) -> Path:
    from matplotlib import pyplot as plt
    from MIL_CREDA_EEG.ps_figures import apply_style, save_figure
    name, subject = subject_key
    t = V.transfer(name)
    e = S.by_subject(S.records(product, t, S.MIL_CREDA))[subject]
    rec, k = next(iter(e["seeds"].values()))
    w = rec["windows"][k]
    att = np.mean([np.asarray(r["attention"][j]) for r, j in e["seeds"].values()], 0)
    x = store.raw_windows(subject, w["condition"], w["index"], V.channels_of(t)).mean(1)
    flags = window_flags(w, marker)
    apply_style(body_pt=10, printed_width_fraction=1.0)
    fig, ax = plt.subplots()
    span = 512 / 128
    order = np.argsort(w["startSeconds"])
    for pos, j in enumerate(order):
        t0 = pos * span
        ax.axvspan(t0, t0 + span, color="tab:green", alpha=float(0.85 * att[j] / att.max()), lw=0)
        ax.plot(t0 + np.arange(x.shape[1]) / 128, x[j], color="k", lw=0.4)
        if flags is not None and flags[j]:
            ax.axvspan(t0, t0 + span, ymin=0.92, ymax=1.0, color="tab:red", lw=0)
    ax.set_xlabel("window, in time order (s of concatenated windows); green: attention; "
                  f"red: {'bad contact or lost samples' if marker == 'marked' else 'event'}")
    ax.set_ylabel("channel mean (uV)")
    path = out / f"tcamil_{marker}_{subject}.pdf"
    save_figure(fig, path, png_preview=True)
    plt.close(fig)
    return path


def analyze(product: str | Path, bags, *, figures: bool = True) -> dict:
    product = Path(product)
    root = product / "Results" / "v63" / EXPERIMENT
    store = S.Store(product, bags)
    crit, contrasts = criterion_rows(product)
    rows = crit + reference_rows(product) + deletion_rows(product, store)
    (product / "Results").mkdir(parents=True, exist_ok=True)
    (product / "Results" / "significance_v63_K1.json").write_text(
        json.dumps(rows, indent=1, default=float), encoding="utf-8")
    figs = {}
    if figures:
        for test, marker in (("T1", "marked"), ("T3", "event")):
            pick = S.median_rule({key: v for key, v in contrasts[test].items()})
            if pick is not None:
                figs[test] = str(tcamil_figure(product, store, pick, marker,
                                               root / "qualitative"))
    return {"rows": len(rows), "figures": figs}
