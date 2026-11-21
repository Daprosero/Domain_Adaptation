"""The aggregation ablation of MIL-CREDA on the criteria pool's units: one configuration, six aggregations.

One notebook = one unit. Phase 1 is the unit's frozen seed-1 base (`criteria_pool.base_file`), set up
exactly as `criteria_pool._setup` sets it up. Phase 2 takes the configuration that the label-free
criterion cef picked for MIL-CREDA in the pool (`MIL-CREDA-EEG/Manifests/pool_aggregation/
milcreda_picks.json`), re-derived as `criteria_pool.draw("MIL-CREDA", 1)[trial]` and refused when it
differs from the manifest, and trains it with the pool's own training seed (`criteria_pool.train_seed(1,
trial)`), epochs and annealed schedule (`criteria_pool.train`) under six aggregations (`ALL_SIX`), the
aggregation being the only thing that changes:

- "own": relevance + consensus, MIL-CREDA as the pool trained it (its `BApool` must reproduce the pick's);
- "own without consensus": gamma = 0, everything else identical (`xpilot.VARIANTS["nocons"]`);
- "published attention" and "gated attention": Ilse et al. 2018 Eqs. 7-8 and 7, 9
  (`aggregation_variants.AggregationVariantModel`), whose attention is not in the frozen base: the
  base's trunks, head and sigma are kept and V, U, w start from Glorot draws seeded by the training
  seed, then train in phase 2 as the own attention's parameters do;
- "mean" and "sum": z = (1/K) sum_k h_k and z = sum_k h_k on the base's own model
  (`xpilot.VARIANTS["mean"]`, `["sum"]`, `mil_creda_model.pooled_weights`).

Each variant records the pool's own evaluation (BA at the base's cut and AUC on the evaluation pool,
BA on the 12 target-test subjects) and, for the four with attention, per subject the AUC of the
inverted weight against the artifact mask and, for class-1 subjects, of the weight against the
event mask (the K1 convention, `mask_aucs`), read from `eeg_sim.SIM_MASKS`. Mean and sum weight
every window equally (AUC 0.5 by construction) and record no attention.

The same notebook runs the adaptation-term ablation, a separate experiment never crossed with the
aggregations (`FAMILIES`, `check_design`): at the same pick and seed, the own aggregation and
one of Eq. 39's coefficients at 0 (`TERMS`): "global only" (lambda_local = 0, `xpilot.VARIANTS
["global"]`) and "local only" (lambda_global = 0, `["local"]`); "full" is "own", and "no adaptation"
is the frozen base's own prediction on the same subjects (`no_adaptation`), not trained.

Two read-outs need no training of their own. Every attention run's per-window weights, with each
evaluation subject's label and its artifact, artifact-type and event masks, go to the sidecar
`attention-<unit>.npz`. The own-pair block (`families.ownPair`) reads the bag representations of
the unit's own target subjects and of each one's paired source subject at the start of phase 2,
at the end of the reference run ("own" = "full") and at the end of "global only", with their
distances; the representations go to `ownpair-<unit>.npz`. On the figure unit (`FIGURE_UNIT`) the
reference run is also snapshotted mid-phase-2 (`mid_snapshot`) and the source side is kept, in
`umap-<unit>.npz`; there too the comparator `FIGURE_METHOD` trains once at its own cef pick, under
"figure", outside both families.
"""
from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.optimizer import register_optimizer_step_pre_hook

from MIL_CREDA_EEG import ablation_pilot as AP
from MIL_CREDA_EEG import aggregation_variants as AV
from MIL_CREDA_EEG import criteria_pool as C
from MIL_CREDA_EEG import eeg_sim as E
from MIL_CREDA_EEG import mechanisms, remote_units
from MIL_CREDA_EEG import oracle_screen as O
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

METHOD = "MIL-CREDA"
PICK_SEED = 1
PICKS = O.BASES.parent / "pool_aggregation" / "milcreda_picks.json"
#: the attention figure's comparator: trained once, on the figure unit only, at ITS OWN cef pick
#: (`COMPARATOR_PICKS`), as `criteria_pool` trains comparators; never in either family
FIGURE_UNIT, FIGURE_METHOD = "d1p5-s1p5-r0", "UDA-DDA"
COMPARATOR_PICKS = O.BASES.parent / "pool_aggregation" / "comparator_picks.json"
#: the table's order: the own aggregation, `aggregation_variants.VARIANTS`, then the sum
ALL_SIX = ("own", *AV.VARIANTS, "sum")
#: the adaptation-term ablation, own aggregation: Eq. 39's local (lambda_local) or global
#: (lambda_global) coefficient at 0; "full" is "own", and "no adaptation" is the frozen base's own
#: prediction (`NO_ADAPTATION`), never trained
TERMS = ("global only", "local only")
NO_ADAPTATION = "no adaptation"
ALL = ALL_SIX + TERMS
#: two separate experiments, never crossed: every aggregation keeps both terms at the pick, every
#: term variant keeps the own aggregation; "full" of the terms is the aggregations' "own" run
FAMILIES = {"aggregation": ALL_SIX, "terms": ("full", *TERMS, NO_ADAPTATION)}
FULL = {"reference": "aggregation/own"}
#: the coefficient each term variant sets to 0, and nothing else
ZEROED = {"global only": "lambda_local", "local only": "lambda_global"}
LAMBDAS = ("lambda_global", "lambda_local")
WITH_ATTENTION = (*ALL_SIX[:4], *TERMS)
#: the variants run on the base's own model through xpilot's phase-2 ablations
ON_BASE = {"own without consensus": "nocons", "mean": "mean", "sum": "sum",
           "global only": "global", "local only": "local"}
PUBLISHED = ("published attention", "gated attention")
SLOWNESS = {"gated attention": 3.0, "published attention": 2.5}
#: the draw's log-space settings differ across platforms in the last bits (libm exp/log; measured
#: <= 1e-15 relative between this machine and the pool's remote run): a pick within this relative
#: tolerance of the draw is the draw, and trains with the manifest's bytes, the ones the pool trained
DRAW_RTOL = 1e-12


def checked_pick(unit_key: str, picks=None, method: str = METHOD) -> dict:
    """The unit's pick of `method`, refused unless its params are the pool's draw at its trial.
    MIL-CREDA's manifest holds one pick per unit, the comparators' one per unit and method."""
    if picks is None or isinstance(picks, (str, Path)):
        picks = json.loads(Path(picks or (PICKS if method == METHOD else COMPARATOR_PICKS)).read_text())
    units = picks["units"]
    if unit_key not in units or (method != METHOD and method not in units[unit_key]):
        raise ValueError(f"no {method} pick for unit {unit_key!r}; the manifest holds {sorted(units)}")
    pick = dict(units[unit_key] if method == METHOD else units[unit_key][method])
    drawn = C.draw(method, PICK_SEED)[int(pick["trial"])]
    got = dict(pick["params"])
    rel = (max(abs(float(got[k]) - float(drawn[k])) / max(abs(float(drawn[k])), 1e-300) for k in drawn)
           if set(got) == set(drawn) else float("inf"))
    if rel > DRAW_RTOL:
        raise ValueError(f"{unit_key}: the pick's params differ from criteria_pool.draw({method!r}, "
                         f"{PICK_SEED})[{pick['trial']}]: {got} != {drawn}")
    pick["drawRelDiff"] = float(rel)
    return pick


def variant_params(variant: str, p: Mapping) -> dict:
    """The pick's settings as the variant trains them: xpilot's ablations fix gamma (aggregations) or
    one of Eq. 39's coefficients (terms); nothing else moves."""
    if variant in ON_BASE:
        agg, fixed = XP.VARIANTS[ON_BASE[variant]]
        return {**p, **{k: v for k, v in agg.items() if k == "gamma"}, **fixed}
    return dict(p)


def figure_task(unit_key: str, epochs: int, picks=None):
    """The attention figure's comparator run: only on `FIGURE_UNIT`, else None."""
    if unit_key != FIGURE_UNIT:
        return None
    pick = checked_pick(unit_key, picks, FIGURE_METHOD)
    return (FIGURE_METHOD, dict(pick["params"]), C.train_seed(PICK_SEED, int(pick["trial"])), int(epochs)), pick


def design(variant: str) -> dict:
    """The variant's family, aggregation and zeroed coefficient."""
    if variant == FIGURE_METHOD:
        return {"family": "figure", "aggregation": "common attention (mil_wrapper)", "zeroedTerm": None}
    if variant in ZEROED:
        return {"family": "terms", "aggregation": "own", "zeroedTerm": ZEROED[variant]}
    if variant in ALL_SIX:
        return {"family": "aggregation", "aggregation": variant, "zeroedTerm": None}
    raise ValueError(f"variants are among {list(ALL)}; got {variant!r}")


def check_design(variant: str, trained: Mapping, pick: Mapping) -> dict:
    """Refuses a run that crosses the families: an aggregation with a term weight off the pick, or a
    term variant off the own aggregation or with anything but its own coefficient changed."""
    d = design(variant)
    if d["family"] == "aggregation":
        moved = [k for k in LAMBDAS if trained[k] != pick[k]]
        if moved:
            raise ValueError(f"{variant}: an aggregation variant keeps both terms at the pick; {moved} moved")
    else:
        agg = XP.VARIANTS[ON_BASE[variant]][0]
        moved = sorted(k for k in pick if trained[k] != pick[k])
        if agg or variant in PUBLISHED or moved != [d["zeroedTerm"]] or trained[d["zeroedTerm"]] != 0.0:
            raise ValueError(f"{variant}: a term variant keeps the own aggregation and sets only "
                             f"{d['zeroedTerm']} to 0; aggregation {agg}, moved {moved}")
    return d


def variant_base(variant: str, base, seed: int):
    """The model phase 2 starts from: the frozen base, or for a published attention the base's
    trunks, head and sigma under the published aggregation, its parameters drawn from `seed`."""
    if variant not in PUBLISHED:
        return base
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = AV.build(variant, base.source_trunk.channels, base.target_trunk.channels,
                         AV.variant_hyper(variant, {}))
        d = base.source_trunk.output_dim
        if model.V.shape[1] != d:
            # the base's searched embedding is wider or narrower than the default trunk's: redraw
            # V (and U) over that width from the same stream, so default-width units keep their draw
            for name in ("V", "U"):
                if hasattr(model, name):
                    setattr(model, name, model._glorot(getattr(model, name).shape[0], d))
    for name in ("source_trunk", "target_trunk", "head"):
        setattr(model, name, copy.deepcopy(getattr(base, name)))
    if model.V.shape[1] != base.source_trunk.output_dim:
        raise ValueError(f"{variant}: attention width {model.V.shape[1]} does not match the base's "
                         f"embedding {base.source_trunk.output_dim}")
    model.fix_sigma(base.sigma)
    return model


@contextlib.contextmanager
def attention_moves(model):
    """Phase 2 trains the attention (phi) and checks that every other phase-1 part stayed frozen,
    reading the own attention's names from `ablation_pilot.ATTENTION_KEYS`; for the block, the
    published attention's own parameters (V, U, w) join them, so the trunks and head stay checked."""
    saved = AP.ATTENTION_KEYS
    AP.ATTENTION_KEYS = (*saved, *getattr(model, "_attention", ()))
    try:
        yield
    finally:
        AP.ATTENTION_KEYS = saved


def mask_aucs(weights, artifact, event, label: int) -> dict:
    """K1's convention: AUC of the inverted weight against the artifact mask (above 0.5 = less
    weight on artifacts), and of the weight against the event mask for class-1 subjects only;
    None when a mask has one side empty."""
    w = np.asarray(weights, dtype=float)
    art, ev = np.asarray(artifact, dtype=bool), np.asarray(event, dtype=bool)
    two = lambda m: 0 < int(m.sum()) < len(m)
    return {"artifactAUC": float(roc_auc_score(art, -w)) if two(art) else None,
            "eventAUC": float(roc_auc_score(ev, w)) if int(label) == 1 and two(ev) else None}


def attention_rows(model, bags) -> tuple[dict, np.ndarray]:
    """Per subject the two mask AUCs and their means, and every subject's window weights in bag
    order, concatenated (the sidecar's per-window data)."""
    rows, weights = [], []
    for b in bags:
        m = E.SIM_MASKS[b.subject]
        w = mechanisms.attention_weights(model, b.windows, "target")
        weights.append(np.asarray(w, dtype=np.float32))
        rows.append({"subject": b.subject, "label": int(b.label), **mask_aucs(w, m["artifact"], m["event"], b.label)})
    art = [r["artifactAUC"] for r in rows if r["artifactAUC"] is not None]
    ev = [r["eventAUC"] for r in rows if r["eventAUC"] is not None]
    return ({"subjects": rows, "meanArtifactAUC": float(np.mean(art)) if art else None,
             "meanEventAUC": float(np.mean(ev)) if ev else None, "nArtifact": len(art), "nEvent": len(ev)},
            np.concatenate(weights) if weights else np.zeros(0, np.float32))


def window_data(bags) -> dict:
    """The sidecar's per-window masks, one row per window of every subject in bag order, with each
    subject's slice at offsets[i]:offsets[i + 1]: `eeg_sim.SIM_MASKS`' artifact flag, artifact type
    ("blink", "muscle", "blink+muscle" or "none", `eeg_sim.window_masks`) and event flag."""
    masks = [E.SIM_MASKS[b.subject] for b in bags]
    sizes = [len(m["artifact"]) for m in masks]
    if any(len(m["event"]) != n or len(m["artifactType"]) != n for m, n in zip(masks, sizes)):
        raise ValueError("a subject's masks disagree on its number of windows")
    return {"subjects": np.array([b.subject for b in bags]), "labels": np.array([int(b.label) for b in bags]),
            "offsets": np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            "artifact": np.concatenate([np.asarray(m["artifact"], bool) for m in masks]),
            "event": np.concatenate([np.asarray(m["event"], bool) for m in masks]),
            "artifactType": np.concatenate([np.asarray(m["artifactType"], dtype=str) for m in masks])}


def sidecar_key(variant: str) -> str:
    return f"weights[{variant}]"


def _has_masks(bags) -> bool:
    return bool(bags) and all(b.subject in E.SIM_MASKS for b in bags)


#: the own-pair distances read the end of these two runs (and the frozen base for the start)
OWN_PAIR_RUNS = {"fullEnd": ("aggregation", "own"), "globalOnlyEnd": ("terms", "global only")}


def own_pair_bags(sides) -> dict:
    """The unit's own target subjects (adapt half, then target test) and each one's own source
    subject: with `paired` units `xpilot.build_sides` simulates SRC-Sxxx and TGT-Sxxx as one subject
    (`eeg_sim.domain`), so the suffix pairs them; the source subject is in the source training or
    source test side."""
    sources = {b.subject: b for k in ("source", "source_test") for b in sides[k]}
    targets = [(k, b) for k in ("target", "target_test") for b in sides[k]]
    pairs = [sources.get("SRC-" + b.subject.split("-", 1)[1]) for _, b in targets]
    missing = [b.subject for (_, b), q in zip(targets, pairs) if q is None]
    if missing:
        raise ValueError(f"no own source pair for {missing}: the own-pair distances need a paired unit")
    return {"split": [k for k, _ in targets], "target": [b for _, b in targets], "pair": pairs}


@torch.no_grad()
def representations(model, bags, domain: str) -> np.ndarray:
    """`mechanisms.bag_representations` (the head's input, the space of the own-pair distances),
    leaving every module's train/eval mode as it found it, so a snapshot mid-training moves nothing."""
    modes = {m: m.training for m in model.modules()}
    try:
        return mechanisms.bag_representations(model, [b.windows for b in bags], domain).to(
            torch.float64).cpu().numpy()
    finally:
        for m, mode in modes.items():
            m.train(mode)


def own_pair_state(model, op) -> dict:
    """Both sides' representations and each target subject's Euclidean distance to its own pair
    (the diagonal of `mechanisms.euclidean_distances` over the matched lists)."""
    zt, zs = representations(model, op["target"], "target"), representations(model, op["pair"], "source")
    return {"target": zt, "pair": zs, "distance": np.linalg.norm(zt - zs, axis=1)}


@contextlib.contextmanager
def mid_snapshot(take, epochs: int, mid: int):
    """Calls `take(model)` once, during the run that trains `epochs` epochs, after `mid` epochs of
    its phase-2 optimizer: `phase2_trace.traced_path` is wrapped for the block to know the model
    and each call's epochs, and a global optimizer step pre-hook counts the steps of the optimizer
    that holds the target trunk. Steps per epoch come from the earlier (dry) call. Nothing the
    training reads is changed: the snapshot restores every module's mode and takes no gradient."""
    cur, out, original = {}, {}, XP.P.traced_path

    def traced(model, *args, **kwargs):
        n_epochs = int(args[10])
        if cur.get("model") is not None and cur["n"]:
            out["perEpoch"] = cur["n"] / cur["epochs"]
        cur.update(model=model, epochs=n_epochs, n=0, opt=None,
                   trunk={id(p) for p in model.target_trunk.parameters()})
        return original(model, *args, **kwargs)

    def hook(opt, a, k):
        if cur.get("model") is None:
            return
        if cur["opt"] is None and any(id(p) in cur["trunk"] for g in opt.param_groups for p in g["params"]):
            cur["opt"] = opt
        if opt is not cur["opt"]:
            return
        if (cur["epochs"] == int(epochs) and "perEpoch" in out and "taken" not in out
                and cur["n"] == round(out["perEpoch"] * int(mid))):
            out["taken"] = take(cur["model"])
            out["step"] = cur["n"]
        cur["n"] += 1

    handle = register_optimizer_step_pre_hook(hook)
    XP.P.traced_path = traced
    try:
        yield out
    finally:
        XP.P.traced_path = original
        handle.remove()
        if cur.get("model") is not None:
            out["totalSteps"] = cur["n"]


_STATE: dict = {}


def _setup(unit_key, eval_subjects, bases, threads) -> dict:
    methods = (METHOD, FIGURE_METHOD) if unit_key == FIGURE_UNIT else (METHOD,)
    s = C._setup(unit_key, methods, eval_subjects, bases, threads)
    if _has_masks(s["pool"]):
        s["attentionSubjects"], s["attentionBags"] = "evaluation pool", s["pool"]
    elif _has_masks(s["sides"]["target_test"]):
        s["attentionSubjects"], s["attentionBags"] = "target test", s["sides"]["target_test"]
    else:
        s["attentionSubjects"], s["attentionBags"] = None, []
    s["unitKey"], s["ownPair"] = unit_key, own_pair_bags(s["sides"])
    s["sourceBags"] = [*s["sides"]["source"], *s["sides"]["source_test"]]
    return s


def _init_worker(unit_key, eval_subjects, bases, threads):
    _STATE.update(_setup(unit_key, eval_subjects, bases, threads))


def _variant(task) -> tuple:
    """One variant: trains it from the frozen base, returns its row. Depends on the task only."""
    variant, p, tseed, epochs = task
    s, t0 = _STATE, time.time()
    if variant == FIGURE_METHOD:
        return _figure(task)
    trained = variant_params(variant, p)
    d = check_design(variant, trained, p)
    b = {**s["b"], "base": variant_base(variant, s["b"]["base"], tseed)}
    ctx = XP.phase2_variant(ON_BASE[variant]) if variant in ON_BASE else attention_moves(b["base"])
    figure = s["unitKey"] == FIGURE_UNIT
    take = lambda m: {**own_pair_state(m, s["ownPair"]), "source": representations(m, s["sourceBags"], "source")}
    mid = (mid_snapshot(take, epochs, epochs // 2) if figure and variant == "own"
           else contextlib.nullcontext({}))
    try:
        with ctx, mid as snap:
            model, totals = C.train(METHOD, trained, b, s["sides"], s["cfg"], s["dev"],
                                    tseed, s["rot"], s["name"], epochs)
    except FloatingPointError:     # the trainer's own non-finite refusal
        return variant, {**d, "params": trained, "nonFinite": True, "seconds": round(time.time() - t0, 1)}
    dev, cut = s["dev"], s["cut"]
    with torch.no_grad():
        on_pool = XC.snapshot(model, s["pool"], cut, dev)
        on_test = XC.snapshot(model, s["sides"]["target_test"], cut, dev)
    attention, weights = (attention_rows(model, s["attentionBags"])
                          if variant in WITH_ATTENTION and s["attentionBags"] else (None, None))
    reps = None
    if any(variant == v for _, v in OWN_PAIR_RUNS.values()):
        reps = {"end": take(model) if figure else own_pair_state(model, s["ownPair"])}
        if snap:
            reps["mid"] = snap.get("taken")
            reps["midInfo"] = {"epoch": epochs // 2, "step": snap.get("step"),
                               "totalSteps": snap.get("totalSteps"), "stepsPerEpoch": snap.get("perEpoch")}
    return variant, {**d, "params": trained, "_weights": weights, "_reps": reps, "BApool": on_pool["BA"], "AUCpool": on_pool["AUC"],
                     "BAtest12": on_test["BA"], "AUCtest12": on_test["AUC"], "nonFinite": False,
                     "attention": attention, "stepTotals": totals, "seconds": round(time.time() - t0, 1)}


def _figure(task) -> tuple:
    """The figure's comparator, trained and evaluated exactly as `criteria_pool._trial` does (its
    common MIL wrapper at its own tau_att around training and evaluation)."""
    method, p, tseed, epochs = task
    s, t0 = _STATE, time.time()
    d = design(method)
    try:
        model, totals = C.train(method, p, s["b"], s["sides"], s["cfg"], s["dev"], tseed, s["rot"],
                                s["name"], epochs)
    except FloatingPointError:
        return method, {**d, "params": p, "nonFinite": True, "seconds": round(time.time() - t0, 1)}
    with C._wrapper(method, s["b"], p["tau_att"]):
        on_pool = XC.snapshot(model, s["pool"], s["cut"], s["dev"])
        on_test = XC.snapshot(model, s["sides"]["target_test"], s["cut"], s["dev"])
        attention, weights = (attention_rows(model, s["attentionBags"]) if s["attentionBags"]
                              else (None, None))
    return method, {**d, "params": p, "_weights": weights, "BApool": on_pool["BA"], "AUCpool": on_pool["AUC"],
                    "BAtest12": on_test["BA"], "AUCtest12": on_test["AUC"], "nonFinite": False,
                    "attention": attention, "stepTotals": totals, "seconds": round(time.time() - t0, 1)}


def phase_two_start(prepared, p: Mapping):
    """The model the reference run and the global-only run start from: the frozen base's bridged,
    target-initialised copy with the aggregation settings phase 2 runs at (`xpilot.phase2_trial`
    sets the trial's gamma and tau_att on its copy); the no-adaptation row keeps phase 1's."""
    model = copy.deepcopy(prepared)
    model.attention_gamma, model.attention_temperature = float(p["gamma"]), float(p["tau_att"])
    return model


def _own_pair_record(state, rep, block, pairs, umap, reference, info=None):
    """One state of the own-pair block: distances in the JSON, representations in the sidecars
    (`ownpair-<unit>.npz` for every unit; with the source side, `umap-<unit>.npz` on the figure unit)."""
    d = [float(x) for x in rep["distance"]]
    block["states"][state] = {"reference": reference, "distances": d, "mean": float(np.mean(d)),
                              **({"snapshot": info} if info else {})}
    if state != "fullMid":
        pairs[f"{state}:target"], pairs[f"{state}:pair"] = rep["target"], rep["pair"]
    if umap is not None:
        umap[f"{state}:target"], umap[f"{state}:pair"] = rep["target"], rep["pair"]
        if "source" in rep:
            umap[f"{state}:source"] = rep["source"]


def no_adaptation(s) -> dict:
    """The frozen base's own target prediction (the bridged, target-initialised copy
    `xcompare.shared_base` predicts with), evaluated as every variant is."""
    with torch.no_grad():
        on_pool = XC.snapshot(s["b"]["prepared"], s["pool"], s["cut"], s["dev"])
        on_test = XC.snapshot(s["b"]["prepared"], s["sides"]["target_test"], s["cut"], s["dev"])
    return {"family": "terms", "aggregation": "own", "zeroedTerm": None, "BApool": on_pool["BA"],
            "AUCpool": on_pool["AUC"], "BAtest12": on_test["BA"], "AUCtest12": on_test["AUC"], "trained": False,
            "matchesBasePredict": on_test["BA"] == s["b"]["predict"]["BA"]}


def run(unit_key: str, variants: Sequence[str] = ALL, procs: int = 4, threads: int | None = 1,
        workdir=None, bases=None, picks=None, epochs: int | None = None,
        eval_subjects: int = O.EVAL_SUBJECTS, figure: bool = True, comparator_picks=None) -> dict:
    """`procs` worker processes, each with `threads` torch threads, share the unit's variants, as
    `criteria_pool.run` shares its trials. Writes `<workdir>/aggregation-<unit_key>.json`, its rows
    under `families`: "aggregation" (the six aggregations) and "terms" ("full" referencing the
    aggregations' "own" run, the two term variants, and the untrained "no adaptation"). On the
    figure unit (`FIGURE_UNIT`, unless `figure` is False) it also trains `FIGURE_METHOD` once at its
    own pick, under "figure". Every attention run's per-window weights, with the subjects' masks,
    go to the sidecar `<workdir>/attention-<unit_key>.npz` (`window_data`, `sidecar_key`)."""
    unknown = [v for v in variants if v not in ALL]
    if unknown:
        raise ValueError(f"variants are among {list(ALL)}; got {unknown}")
    pick = checked_pick(unit_key, picks)
    trial, params = int(pick["trial"]), dict(pick["params"])
    epochs = int(epochs or C.EPOCHS2)
    fig = figure_task(unit_key, epochs, comparator_picks) if figure else None
    tseed = C.train_seed(PICK_SEED, trial)
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    threads = int(threads or os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)
    _STATE.clear()
    _STATE.update(_setup(unit_key, eval_subjects, bases, threads))
    b = _STATE["b"]
    out_path = workdir / f"aggregation-{unit_key}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "unitKey": unit_key, "unit": _STATE["name"], "trial": trial, "params": params,
        "trainSeed": tseed, "drawRelDiff": pick["drawRelDiff"], "pickBApool": pick["BApool"], "pickAUCpool": pick["AUCpool"],
        "pickBAtest12": pick["BAtest12"], "epochs2": epochs, "cut": _STATE["cut"],
        "baseCommit": b["commit"], "baseSeed": 1, "evalSubjects": len(_STATE["pool"]),
        "schedule": "lr2*(1+10p)^-0.75", "attentionSubjects": _STATE["attentionSubjects"],
        "attentionAUC": "artifact: AUC(artifact mask, -weight) per subject; event: AUC(event mask, "
                        "weight) per class-1 subject; mean and sum record none (0.5 by construction)",
        "forms": {**AV.FORMS, AV.WITHOUT_CONSENSUS: AV.WITHOUT_CONSENSUS_FORM,
                  "own": "r44 Eqs. 14-16: relevance + consensus", "sum": "z = sum_k h_k"},
        "families": {"aggregation": {}, "terms": {"full": dict(FULL)}}, "figure": {}}
    out["procs"], out["threads"], out["commit"] = int(procs), threads, remote_units.source_commit()
    out.setdefault("figure", {})
    if fig:
        out["figurePick"] = {k: fig[1][k] for k in ("trial", "BApool", "AUCpool", "BAtest12", "drawRelDiff")}
    fam = {**out["families"], "figure": out["figure"]}
    side_path = workdir / f"attention-{unit_key}.npz"
    side = dict(np.load(side_path)) if side_path.exists() else {
        **window_data(_STATE["attentionBags"]), "attentionSubjects": np.array(_STATE["attentionSubjects"])}
    if NO_ADAPTATION not in fam["terms"]:
        fam["terms"][NO_ADAPTATION] = no_adaptation(_STATE)
    op, figure_unit = _STATE["ownPair"], unit_key == FIGURE_UNIT
    pair_path, umap_path = workdir / f"ownpair-{unit_key}.npz", workdir / f"umap-{unit_key}.npz"
    meta = {"subjects": np.array([b.subject for b in op["target"]]),
            "pairs": np.array([b.subject for b in op["pair"]]),
            "labels": np.array([int(b.label) for b in op["target"]]), "split": np.array(op["split"])}
    pairs = dict(np.load(pair_path)) if pair_path.exists() else dict(meta)
    umap = (dict(np.load(umap_path)) if umap_path.exists() else {
        **meta, "source:subjects": np.array([b.subject for b in _STATE["sourceBags"]]),
        "source:labels": np.array([int(b.label) for b in _STATE["sourceBags"]]),
        "source:split": np.array(["source"] * len(_STATE["sides"]["source"])
                                 + ["source_test"] * len(_STATE["sides"]["source_test"]))}) if figure_unit else None
    if "ownPair" not in out["families"]:
        first = phase_two_start(_STATE["b"]["prepared"], params)
        start = own_pair_state(first, op)
        if figure_unit:
            start["source"] = representations(first, _STATE["sourceBags"], "source")
        out["families"]["ownPair"] = {
            "space": "bag representations (the head's input, mechanisms.bag_representations); "
                     "Euclidean distance from each target subject to its own source pair",
            "subjects": [{"subject": a.subject, "pair": q.subject, "label": int(a.label), "split": k}
                         for a, q, k in zip(op["target"], op["pair"], op["split"])],
            "states": {}}
        _own_pair_record("start", start, out["families"]["ownPair"], pairs, umap,
                         "start of phase 2: the frozen base (bridged, target-initialised) at the "
                         "pick's gamma and tau_att, as xpilot.phase2_trial sets them before the first step")
    done = lambda v: v in fam[design(v)["family"]]
    tasks = [(v, params, tseed, epochs) for v in variants if not done(v)]
    if fig and not done(FIGURE_METHOD):
        tasks.append(fig[0])
    tasks.sort(key=lambda t: -SLOWNESS.get(t[0], 1.0))

    def record(result):
        variant, row = result
        weights, reps = row.pop("_weights", None), row.pop("_reps", None)
        if weights is not None:
            side[sidecar_key(variant)] = weights
            np.savez_compressed(side_path, **side)
        fam[row["family"]][variant] = row
        if reps:
            state = next(k for k, (_, v) in OWN_PAIR_RUNS.items() if v == variant)
            ref = "/".join(OWN_PAIR_RUNS[state])
            _own_pair_record(state, reps["end"], out["families"]["ownPair"], pairs, umap, ref)
            if reps.get("mid") is not None:
                _own_pair_record("fullMid", reps["mid"], out["families"]["ownPair"], pairs, umap,
                                 f"{ref} at epoch {reps['midInfo']['epoch']}", reps["midInfo"])
            np.savez_compressed(pair_path, **pairs)
            if umap is not None:
                np.savez_compressed(umap_path, **umap)
        own = fam["aggregation"].get("own", {}).get("BApool")
        diff = None if own is None else float(own) - float(out["pickBApool"])
        out["BApoolDifference"], out["reproduces"] = diff, (None if diff is None else abs(diff) <= 1e-9)
        out_path.write_text(json.dumps(out))

    np.savez_compressed(pair_path, **pairs)
    if umap is not None:
        np.savez_compressed(umap_path, **umap)
    out_path.write_text(json.dumps(out))
    try:
        if int(procs) <= 1:
            for t in tasks:
                record(_variant(t))
        else:
            _STATE["stack"].close()
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(int(procs), initializer=_init_worker,
                          initargs=(unit_key, eval_subjects, bases, threads)) as workers:
                for result in workers.imap_unordered(_variant, tasks, chunksize=1):
                    record(result)
    finally:
        _STATE["stack"].close()
    out.setdefault("BApoolDifference", None)
    out.setdefault("reproduces", None)
    out_path.write_text(json.dumps(out))
    trained = [v for v in (*ALL, FIGURE_METHOD) if done(v)]
    print(unit_key, "variants done", trained, "reproduces", out["reproduces"], flush=True)
    return {"file": out_path.name, "variants": trained}
