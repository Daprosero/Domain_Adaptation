"""Quick pilots of MIL-CREDA phase 2 outside BALLADEER's device transfers.

Diagnostic outside the plan. Three settings, one shared recipe: the spectral log-power
encoder, a phase 1 searched once per unit (rotation, half) on SOURCE subjects only and refit
per seed (TPE, `spectral_search`'s
space, score (BA@0.5 + AUC)/2 pooled over three subject-wise inner validations), then
`phase2_trace.run_rotation` with the objective eq39+hardloc and the copy arm (both sides
read the same channels; xds uses the warm arm). Compared on the target test subjects: predict (phase 1, no
adaptation) against the end of phase 2.

* "giga": GiGA motor imagery (Cho et al. 2017), cross-subject. A bag is one trial; its
  instances are 1 s windows at a 0.5 s step over the stored 2.5-5 s segment (0.5-3 s after
  the cue), all with the trial's label. 1-45 Hz band-pass on the whole segment, then common
  average, then the cut. The stored 2.5-5 s segment is the event-locked 3 s imagery period (cue-based
  paradigm). GiGA is the control: TRIALS_PER_CLASS trials per class per subject, the first ones.
* "ieee": IEEE ADHD (Nasrabadi et al.), cross-subject. A bag is one subject: 4 s windows at
  50% overlap, 50 Hz notch, then the v63 instance rule (<= 60 windows, 1-45 Hz, average).
* "xds": cross-dataset ADHD between IEEE and one BALLADEER task (default Robots, the best
  source domain known), both directions, each dataset with ALL its channels (IEEE 19, Robots 14) and the
  warm arm (shared channel slots copied from the source trunk, the others zeroed). The source
  is the whole source dataset. Both sides use 4 s windows at 50% overlap:
  BALLADEER's stored contiguous 4 s windows gain the half-shifted window between neighbours.

Partition, always by subject (no subject on two sides, ever):
* cross-subject: subject i (sorted, per class) goes to fold i % 5. Rotation r: the target is
  fold r, split by alternating subjects into two disjoint halves; `half` h adapts on one and
  tests the other. The source is the other four folds: fold (r + 1) % 5 is its test, the
  remaining three train, with three subject-wise inner splits for the search and the cut.
* cross-dataset: the source dataset keeps 1 in 5 subjects (per class) as its test; the
  target dataset is split in 4 folds per class, rotation r tests fold r and adapts the rest.
Target labels are read only to score, and to group GiGA test subjects by their predict BA.

* "sim": simulated EEG (`eeg_sim`, SEREEGA-style: lead field of a spherical head, NeuroDSP
  1/f plus bursty oscillations). `setting` "subject" (one device, cross-subject), "device"
  (19-channel 10-20 source, 14-channel EMOTIV-like target grown by `shift`, warm arm) or
  "invert" (device plus the class effect flipped in the target); `effect` sizes the
  ADHD-like frontal theta/beta change, or `d_target` states it as the Cohen's d of the
  subject-level Fz theta/beta ratio (translated by `eeg_sim.effect_for_d` on the clean
  10-20 device, then used unchanged in both domains). Optional: `artifact_rate` (blinks and
  muscle bursts per minute, class-independent unless `artifact_class_corr`),
  `event_fraction` (windows where the class effect is concentrated) and `paired` (the
  target holds the source's subjects, same identity and class, under its own device). The
  output records per target-test and target-train bag its artifact, artifact-type and
  event masks, the measured TBR d of source and target, and the per-window attention
  weights of the target bags at predict and at the end of phase 2.
"""
from __future__ import annotations

import contextlib
import copy
import glob
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import optuna
import scipy.io
import torch
from scipy.signal import filtfilt, iirnotch, sosfiltfilt

from MIL_CREDA_EEG import campaign, eegnet, phase2_select, remote_units, spectral_trunk, two_phase
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import spectral_search as SS
from MIL_CREDA_EEG import progress as PG
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import xds_pilot as X

EXPERIMENTS = ("giga", "ieee", "xds", "sim")
FOLDS, INNER, XDS_FOLDS = 5, 3, 5
TRIALS_PER_CLASS = 25
GIGA_WINDOW, GIGA_STEP = 128, 64
IEEE_WINDOW = 512
OBJECTIVE = "eq39+hardloc"
SAMPLER_SEED = 20261112
#: the opt-in phase-2 search (smoke(search2=True)): TPE over this space, wd2 stays fixed
SPACE2 = {"lr2": ("log", (1e-4, 1e-2)), "epochs2": ("cat", (10, 20, 40)),
          "lambda_global": ("log", (0.1, 10.0)), "lambda_local": ("log", (0.1, 10.0)),
          "tau_local": ("log", (0.1, 10.0))}
TRIALS2 = 20
SEARCH2_SEED = 1
SEARCH2_RULE = "phase2_select.centroid_score on the target adapt half + held-out source test bags"


# ------------------------------------------------------------------ data

def find_giga(root=None):
    if root and glob.glob(f"{root}/subject_*.npz"):
        return root
    hits = sorted(glob.glob("/kaggle/input/**/MI/2.5-5/subject_1.npz", recursive=True))
    if not hits:
        raise FileNotFoundError("no GiGA subject_*.npz under /kaggle/input")
    return str(Path(hits[0]).parent)


def giga_bags(root) -> list[two_phase.BagTensor]:
    out = []
    for f in sorted(glob.glob(f"{root}/subject_*.npz")):
        with np.load(f) as d:
            sid, x, y = int(d["subject_id"]), d["X"].astype(np.float64), d["y"].astype(np.int64)
        x = sosfiltfilt(V.SOS, x - x.mean(-1, keepdims=True), axis=-1)
        x = x - x.mean(1, keepdims=True)                       # common average
        starts = range(0, x.shape[-1] - GIGA_WINDOW + 1, GIGA_STEP)
        # GiGA is the control dataset: a balanced TRIALS_PER_CLASS trials per class per subject,
        # the first ones by index; every window lies inside the event-locked imagery segment
        for c in (0, 1):
            for t in np.flatnonzero(y == c)[:TRIALS_PER_CLASS]:
                w = np.stack([x[t, :, s:s + GIGA_WINDOW] for s in starts])
                out.append(two_phase.BagTensor(f"G{sid:02d}-t{t:03d}", "GiGA", int(c),
                                               torch.from_numpy(np.ascontiguousarray(w)).float()))
    return out


def ieee_bags(root, channels=X.IEEE_CHANNELS) -> list[two_phase.BagTensor]:
    b, a = iirnotch(50.0, 30.0, fs=X.RATE)
    pick = [X.IEEE_CHANNELS.index(c) for c in channels]
    out = []
    for group, label in (("ADHD_group", 1), ("Control_group", 0)):
        for f in sorted(glob.glob(f"{root}/{group}/*.mat")):
            d = scipy.io.loadmat(f)
            x = np.asarray(d[list(d.keys())[-1]], dtype=np.float64).T
            x = filtfilt(b, a, x, axis=-1)[pick]
            starts = range(0, x.shape[1] - IEEE_WINDOW + 1, IEEE_WINDOW // 2)
            w = np.stack([x[:, s:s + IEEE_WINDOW] for s in starts])
            out.append(two_phase.BagTensor(f"IEEE-{Path(f).stem}", "IEEE", label, X.instances(w)))
    return out


def half_overlap(w, starts, seconds=4.0):
    """4 s windows at 50% overlap from stored contiguous 4 s windows: between two windows that
    start exactly `seconds` apart, the second half of one plus the first half of the next."""
    half, out = w.shape[-1] // 2, []
    for k in range(len(w)):
        out.append(w[k])
        if k + 1 < len(w) and abs(float(starts[k + 1]) - float(starts[k]) - seconds) < 1e-6:
            out.append(np.concatenate([w[k][:, half:], w[k + 1][:, :half]], axis=-1))
    return np.stack(out)


def balladeer_bags(cfg, condition, channels) -> list[two_phase.BagTensor]:
    bags = campaign.load_bags(cfg)
    out = []
    for s in sorted(bags.holders(condition)):
        bag = bags.get(s, condition)
        w = np.asarray(bag.windows(), dtype=np.float64)
        names = list(bag.channels)
        w = w[:, [names.index(c) for c in channels]]
        w = half_overlap(w, bag.marks()["window_start_s"])
        label = int(bag.label if isinstance(bag.label, (int, np.integer))
                    else two_phase.CLASSES.index(bag.label))
        out.append(two_phase.BagTensor(s, condition, label, X.instances(w)))
    return out


def subject_of(bag) -> str:
    return bag.subject.split("-t")[0] if bag.condition == "GiGA" else bag.subject


# ------------------------------------------------------------------ partitions

def _per_class_order(bags):
    """Subjects sorted within each class, so i % k stratifies by class."""
    lab = {}
    for b in bags:
        lab.setdefault(subject_of(b), int(b.label))
    return {c: sorted(s for s, l in lab.items() if l == c) for c in sorted(set(lab.values()))}


def folds_of(bags, k):
    return {s: i % k for subs in _per_class_order(bags).values() for i, s in enumerate(subs)}


def _pick(bags, subjects):
    subjects = set(subjects)
    return [b for b in bags if subject_of(b) in subjects]


def _inner(bags, subjects):
    order = {c: [s for s in subs if s in subjects] for c, subs in _per_class_order(bags).items()}
    split = {s: i % INNER for subs in order.values() for i, s in enumerate(subs)}
    return [(_pick(bags, [s for s in subjects if split[s] != k]),
             _pick(bags, [s for s in subjects if split[s] == k])) for k in range(INNER)]


def cross_subject_sides(bags, rotation, half):
    f = folds_of(bags, FOLDS)
    tgt = {c: [s for s in subs if f[s] == rotation] for c, subs in _per_class_order(bags).items()}
    test = [s for subs in tgt.values() for i, s in enumerate(subs) if i % 2 == half]
    adapt = [s for subs in tgt.values() for i, s in enumerate(subs) if i % 2 != half]
    s_test = [s for s in f if f[s] == (rotation + 1) % FOLDS]
    s_train = [s for s in f if f[s] not in (rotation, (rotation + 1) % FOLDS)]
    return _sides(bags, s_train, s_test, bags, adapt, test)


def cross_dataset_sides(source, target, rotation):
    fs = folds_of(source, 5)
    s_test = [s for s in fs if fs[s] == 0]
    s_train = [s for s in fs if fs[s] != 0]
    ft = folds_of(target, XDS_FOLDS)
    return _sides(source, s_train, s_test, target, [s for s in ft if ft[s] != rotation],
                  [s for s in ft if ft[s] == rotation])


def _sides(source, s_train, s_test, target, t_adapt, t_test):
    assert not set(s_train) & set(s_test) and not set(t_adapt) & set(t_test)
    inner = _inner(source, set(s_train))
    return {"source": _pick(source, s_train), "source_test": _pick(source, s_test),
            "target": _pick(target, t_adapt), "target_test": _pick(target, t_test),
            "inner": inner, "inner_train": inner[0][0], "inner_valid": inner[0][1],
            "subjects": {"sourceTrain": sorted(s_train), "sourceTest": sorted(s_test),
                         "targetAdapt": sorted(t_adapt), "targetTest": sorted(t_test)}}


# ------------------------------------------------------------------ phase-1 search

def search_phase1(sides, cfg, device, trials, rotation):
    cs, ct = int(sides["source"][0].windows.shape[1]), int(sides["target"][0].windows.shape[1])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED + int(rotation)))

    def objective(trial):
        params = SS.suggest(trial)
        y, p = [], []
        for tr, va in sides["inner"]:
            try:
                m = P.phase_one(tr, cs, ct, P._hyper(cfg, temperature=params["tauAtt1"], searched=params),
                                SS.SEARCH_SEED, device, searched=params)
            except V.V63Error:          # collapsed encoder: every bag at one point, sigma undefined
                trial.set_user_attr("collapsed", True)
                return 0.0
            vy, vp = P._subjects(va, P._probabilities(m, va, "source", device))
            y, p = y + vy, p + vp
        score = SS.selection_score(y, p)
        return 0.0 if not np.isfinite(score) else score

    study.optimize(objective, n_trials=int(trials),
                   callbacks=[PG.optuna_callback("p1", "phase-1 search", trials)])
    done = [t for t in study.trials if t.value is not None]
    LAST_PHASE1_RANKED[:] = [(dict(t.params), round(float(t.value), 4))         # best first, ties by order
                             for t in sorted(done, key=lambda t: (-t.value, t.number))]
    return study.best_trial.params, round(float(study.best_value), 4)


# the last phase-1 search's completed trials, best first (campaign_unit's non-degenerate rule)
LAST_PHASE1_RANKED: list = []


# ------------------------------------------------------------------ one unit

@contextlib.contextmanager
def window_samples(n):
    """Window length of the spectral trunk (its default is bound at definition time)."""
    saved, defaults = eegnet.SAMPLES, spectral_trunk.SpectralTrunk.__init__.__defaults__
    eegnet.SAMPLES = int(n)
    spectral_trunk.SpectralTrunk.__init__.__defaults__ = (int(n),)
    try:
        yield
    finally:
        eegnet.SAMPLES = saved
        spectral_trunk.SpectralTrunk.__init__.__defaults__ = defaults


@contextlib.contextmanager
def attention_export(on: bool):
    """Turn phase2_trace's read-only attention export on for the duration of a run."""
    saved = P.ATTENTION_EXPORT
    P.ATTENTION_EXPORT = bool(on)
    try:
        yield
    finally:
        P.ATTENTION_EXPORT = saved


@contextlib.contextmanager
def full_trace_mode(on: bool):
    """phase2_trace.FULL_TRACE for the duration of a run, restored after."""
    saved = P.FULL_TRACE
    P.FULL_TRACE = bool(on)
    try:
        yield
    finally:
        P.FULL_TRACE = saved


def _lift_attention(out):
    """Move the snapshots' attention weights to out["attention"]: predict and the last
    phase-2 epoch per objective, dropped from every other snapshot."""
    lifted = {}
    for name, rec in out["objectives"].items():
        epochs = rec["epochs"]
        last = max(epochs, key=int) if epochs else None
        lifted[name] = {"predict": rec["predict"].get("attention"),
                        "end": epochs[last].get("attention") if last is not None else None,
                        "endEpoch": int(last) if last is not None else None}
    for rec in out["objectives"].values():
        rec["predict"].pop("attention", None)
        for e in rec["epochs"].values():
            e.pop("attention", None)
    if any(v["predict"] is not None for v in lifted.values()):
        out["attention"] = lifted
    return out


def _arm_names(sides):
    """(arm, source names, target names) of a unit."""
    if "s_names" in sides:      # two montages: warm arm (shared slots copied, the others zeroed)
        return "warm", sides["s_names"], sides["t_names"]
    names = [f"ch{i}" for i in range(int(sides["source"][0].windows.shape[1]))]
    return "copy", names, names     # one montage: copy arm


def run_unit(sides, cfg, device, seed, rotation, searched, unit, log=print, given=None):
    """Refit phase 1 at `searched` = (params, score, trials) with `seed` (or take the `given`
    shared base and cut), then phase 2."""
    params, score, trials = searched
    P.PHASE_ONE_SEARCHED.clear()
    P.PHASE_ONE_SEARCHED[int(rotation)] = params
    arm, s_names, t_names = _arm_names(sides)
    out = P.run_rotation(unit, cfg, None, None, device, seed, int(rotation), arm, [OBJECTIVE],
                         log=log, sides={**sides, "s_names": s_names, "t_names": t_names}, given=given)
    out = _lift_attention(out)
    out["arm"] = arm
    out["phase1Search"] = {"params": params, "score": score, "trials": int(trials)}
    out["subjectsSplit"] = sides["subjects"]
    out["targetTestUnits"] = [b.subject for b in sides["target_test"]]   # row order of the target test
    return out


# ------------------------------------------------------------------ phase-2 search (opt-in)

@contextlib.contextmanager
def phase2_settings(*, lr2, epochs2, lambda_global, lambda_local, tau_local):
    """Phase 2 at these settings for the duration of the block, defaults restored after:
    phase2_trace.PHASE_TWO (lr2, epochs2, tauLocal) and Eq. 39's coefficients
    (two_phase.LAMBDA_GLOBAL/LAMBDA_LOCAL and the "both terms" selection the objective reads)."""
    saved = (dict(P.PHASE_TWO), two_phase.LAMBDA_GLOBAL, two_phase.LAMBDA_LOCAL,
             two_phase.TERM_SELECTIONS[two_phase.BOTH_TERMS])
    P.PHASE_TWO.update({"lr2": float(lr2), "epochs2": int(epochs2), "tauLocal": float(tau_local)})
    two_phase.LAMBDA_GLOBAL, two_phase.LAMBDA_LOCAL = float(lambda_global), float(lambda_local)
    two_phase.TERM_SELECTIONS[two_phase.BOTH_TERMS] = (float(lambda_global), float(lambda_local))
    try:
        yield
    finally:
        P.PHASE_TWO.clear()
        P.PHASE_TWO.update(saved[0])
        two_phase.LAMBDA_GLOBAL, two_phase.LAMBDA_LOCAL = saved[1], saved[2]
        two_phase.TERM_SELECTIONS[two_phase.BOTH_TERMS] = saved[3]


def phase2_base(sides, cfg, device, seed, rotation, params1, unit):
    """The phase-1 base model run_rotation fits for phase 2 (searched params, all the source
    training subjects, `seed`); phase 2's settings do not enter it, so a search fits it once."""
    cs, ct = int(sides["source"][0].windows.shape[1]), int(sides["target"][0].windows.shape[1])
    hyper = P._hyper(cfg, temperature=float(params1["tauAtt1"]), searched=params1)
    return P.phase_one(sides["source"], cs, ct, hyper, seed, device, transfer=unit, fold=int(rotation),
                       searched=params1)


def _blind_reps(model, bags, side, device):
    """Bag representations (the head's input) and the predicted class (Eq. 22); no label read."""
    model.eval()
    with torch.no_grad():
        Z = model.represent(model.bag_pairs(model.embed([b.windows.to(device) for b in bags], side)))
        pred = model.classify(Z).argmax(1)
    return Z.detach().cpu().numpy().astype(np.float64), pred.cpu().numpy()


def phase2_trial(base, sides, cfg, device, seed, params1, params2):
    """Phase 2 (OBJECTIVE) from a copy of `base` at `params2`, on the unlabelled target adapt
    half, then its label-free score. The end model equals run_rotation's at the same settings.
    Reads sides["source"], ["source_test"] and ["target"] (windows only); never the target test."""
    arm, s_names, t_names = _arm_names(sides)
    source, target = sides["source"], sides["target"]
    with phase2_settings(**params2):
        hyper = P._hyper(cfg, temperature=float(params1["tauAtt1"]), searched=params1)
        model = copy.deepcopy(base)
        model.attention_gamma = P.PHASE_TWO_AGGREGATION["gamma"]
        model.attention_temperature = P.PHASE_TWO_AGGREGATION["temperature"]
        P.traced_path(model, source, target, None, hyper, seed, device, OBJECTIVE, 0.0,
                      lambda m: P.initialise_target(m, arm, target, s_names, t_names),
                      {"sigma": base.sigma}, P.PHASE_TWO["epochs2"], P.EQUIVALENCE_EPOCHS,
                      log=lambda *a, **k: None, light=True)
    zt, pseudo = _blind_reps(model, target, "target", device)
    zs, _ = _blind_reps(model, sides["source_test"], "source", device)
    score = phase2_select.centroid_score(zt, pseudo, zs, [b.label for b in sides["source_test"]])
    return {"model": model, "score": float(score), "cef": phase2_select.cef_score(pseudo)}


def _suggest2(trial):
    out = {}
    for name, (kind, values) in SPACE2.items():
        out[name] = (trial.suggest_categorical(name, list(values)) if kind == "cat" else
                     trial.suggest_float(name, values[0], values[1], log=True))
    return out


def search_phase2(sides, cfg, device, trials2, rotation, params1, unit, pick="centroid"):
    """TPE over SPACE2, once per unit: phase 1 at `params1` and seed SEARCH2_SEED (fit once),
    each trial's phase 2 on the target adapt half, scored by phase2_select.centroid_score. The
    sampler always follows that score; `pick` "cef" takes, among the same trials, the first one
    with the highest phase2_select.cef_score (a failed trial is never picked)."""
    if pick not in phase2_select.PICKS:
        raise ValueError(f"pick in {phase2_select.PICKS}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED + int(rotation)))
    base = phase2_base(sides, cfg, device, SEARCH2_SEED, rotation, params1, unit)
    history = []

    def objective(trial):
        params = _suggest2(trial)
        cef = None
        try:
            out = phase2_trial(base, sides, cfg, device, SEARCH2_SEED, params1, params)
            score, cef = out["score"], out["cef"]
        except (V.V63Error, FloatingPointError):     # collapsed or diverged trial
            score = 0.0
        score = round(float(score), 4) if np.isfinite(score) else 0.0
        history.append({"params": params, "score": score, "cef": cef})
        return score

    study.optimize(objective, n_trials=int(trials2),
                   callbacks=[PG.optuna_callback("milcreda-search", "MIL-CREDA phase-2 search", trials2)])
    picked = (study.best_trial.number if pick == "centroid" else
              phase2_select.first_best([h["cef"] for h in history]))
    best = history[picked]
    return {"params": best["params"], "score": best["score"], "trials": history, "rule": SEARCH2_RULE,
            "pick": pick, "pickedTrial": int(picked)}


def build_sides(cfg, environ, *, experiment, rotation, half=0, direction="IEEE->Robots",
                condition="Robots", setting="subject", effect=1.0, shift=0.0, n_subjects=60, sim_seed=0,
                d_target=None, artifact_rate=0.0, artifact_class_corr=0.0, event_fraction=0.0,
                paired=False, spread=1.0) -> dict:
    """The bags and the subject-wise partition of one unit, shared by MIL-CREDA (smoke) and the
    comparators (xcompare), so every method sees the same subjects on the same sides."""
    sim_record = None
    if experiment == "giga":
        bags = giga_bags(find_giga(environ.get("GIGA_ROOT")))
        cfg = {**cfg, "schedule": {**cfg["schedule"], "bagsPerStep": 32}}
        sides, unit, samples = cross_subject_sides(bags, rotation, half), f"giga:{rotation}:{half}", GIGA_WINDOW
    elif experiment == "ieee":
        bags = ieee_bags(X.find_ieee(environ.get("IEEE_ROOT")))
        sides, unit, samples = cross_subject_sides(bags, rotation, half), f"ieee:{rotation}:{half}", IEEE_WINDOW
    elif experiment == "sim":
        from MIL_CREDA_EEG import eeg_sim as E
        if d_target is not None:            # the class effect stated as a target TBR Cohen's d
            effect = E.effect_for_d(float(d_target), event_fraction=float(event_fraction))
        opts = {"artifact_rate": float(artifact_rate), "artifact_class_corr": float(artifact_class_corr),
                "event_fraction": float(event_fraction), "spread": float(spread)}
        if setting == "subject":            # one device, subjects differ only
            bags = E.domain("SIM", n_subjects, E.device("1020", 0.0, 7 + sim_seed), effect, seed=sim_seed,
                            **opts)
            sides = cross_subject_sides(bags, rotation, half)
        else:                               # device shift, optionally with an inverted class effect
            identity = sim_seed if paired else None     # paired: SRC-Sxxx and TGT-Sxxx are one subject
            src = E.domain("SRC", n_subjects, E.device("1020", 0.0, 7 + sim_seed), effect, seed=sim_seed,
                           identity_seed=identity, **opts)
            tgt = E.domain("TGT", n_subjects, E.device("emotiv", shift, 11 + sim_seed), effect,
                           invert=(setting == "invert"), seed=sim_seed + 1, identity_seed=identity, **opts)
            sides = cross_dataset_sides(src, tgt, rotation)
            sides["s_names"], sides["t_names"] = list(E.TEN_TWENTY), list(E.EMOTIV)
        size = f"d{float(d_target):g}" if d_target is not None else f"e{effect:g}"
        extra = "".join([f"-a{artifact_rate:g}" if artifact_rate else "",
                         f"-c{artifact_class_corr:g}" if artifact_class_corr else "",
                         f"-v{event_fraction:g}" if event_fraction else "", "-p" if paired else "",
                         f"-u{spread:g}" if float(spread) != 1.0 else ""])
        unit = f"sim-{setting}-{size}-s{shift:g}{extra}:{rotation}:{half}"
        samples = E.WINDOW
        sim_record = {
            "dTarget": None if d_target is None else float(d_target),
            "effectRule": (None if d_target is None else
                           "eeg_sim.effect_for_d: paired calibration, 1000 subjects, seed 0, "
                           "device('1020', 0, 0), Fz TBR, no artifacts"),
            "artifactRate": float(artifact_rate), "artifactClassCorr": float(artifact_class_corr),
            "eventFraction": float(event_fraction), "paired": bool(paired), "subjectSpread": float(spread),
            "measuredD": {"source": round(E.tbr_effect_size(sides["source"] + sides["source_test"]), 4),
                          "target": round(E.tbr_effect_size(sides["target"] + sides["target_test"]), 4),
                          "rule": "eeg_sim.tbr_effect_size: Cohen's d of the per-subject TBR "
                                  "(Fz, or F3/F4 on EMOTIV), every subject of the side"},
            "masks": {"targetTest": E.masks_of(sides["target_test"]),
                      "targetTrain": E.masks_of(sides["target"])}}
    else:
        loaded = campaign.load_bags(cfg)
        ball_names = list(loaded.get(sorted(loaded.holders(condition))[0], condition).channels)
        ieee = ieee_bags(X.find_ieee(environ.get("IEEE_ROOT")))           # all 19 IEEE channels
        ball = balladeer_bags(cfg, condition, ball_names)                  # all the task's channels
        names = {"IEEE": list(X.IEEE_CHANNELS), "BALLADEER": ball_names}
        src_is_ieee = direction.startswith("IEEE")
        src, tgt = (ieee, ball) if src_is_ieee else (ball, ieee)
        sides = cross_dataset_sides(src, tgt, rotation)
        sides["s_names"] = names["IEEE" if src_is_ieee else "BALLADEER"]
        sides["t_names"] = names["BALLADEER" if src_is_ieee else "IEEE"]
        unit, samples = f"{direction}:{rotation}", IEEE_WINDOW
    return {"sides": sides, "unit": unit, "samples": samples, "cfg": cfg, "sim_record": sim_record,
            "effect": effect}


def smoke(config: Mapping, product=None, *, experiment: str, rotation: int, seeds: Sequence[int] = (1,),
          half: int = 0, direction: str = "IEEE->Robots", condition: str = "Robots", trials: int = 20,
          epochs2: int | None = None, setting: str = "subject", effect: float = 1.0, shift: float = 0.0,
          n_subjects: int = 60, sim_seed: int = 0, d_target: float | None = None,
          artifact_rate: float = 0.0, artifact_class_corr: float = 0.0, event_fraction: float = 0.0,
          paired: bool = False, spread: float = 1.0, workdir=None,
          environ: Mapping[str, str] | None = None, search2: bool = False, trials2: int = TRIALS2,
          shared: Mapping | None = None, full_trace: bool = True,
          phase2_params: Mapping | None = None, pick: str = "centroid") -> dict:
    """One pilot unit: phase-1 search, optional phase-2 search, then `seeds`. `phase2_params`
    (SPACE2 keys) skips the phase-2 search and runs the seeds at those settings, chosen elsewhere
    without target labels. `full_trace` False
    (grid units): phase 2 runs light and records only predict and the
    final epoch, with no attention export and no equivalence check; the end model is the same."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment in {EXPERIMENTS}")
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)))
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ), placement)
    device = torch.device(cfg.get("device", "cpu"))
    if epochs2 is not None:
        P.PHASE_TWO["epochs2"] = int(epochs2)
    started = time.time()
    built = shared["built"] if shared else build_sides(
        cfg, environ, experiment=experiment, rotation=rotation, half=half, direction=direction,
        condition=condition, setting=setting, effect=effect, shift=shift,
        n_subjects=n_subjects, sim_seed=sim_seed, d_target=d_target,
        artifact_rate=artifact_rate, artifact_class_corr=artifact_class_corr,
        event_fraction=event_fraction, paired=paired, spread=spread)
    sides, unit, samples, cfg = built["sides"], built["unit"], built["samples"], built["cfg"]
    sim_record, effect = built["sim_record"], built["effect"]
    runs = {}
    with spectral_trunk.installed(), window_samples(samples), \
            attention_export(experiment == "sim" and full_trace), full_trace_mode(full_trace):
        t0 = time.time()
        params, score = (shared["phase1"] if shared else
                         search_phase1(sides, cfg, device, trials, rotation))   # once per unit
        timing = {"loadSeconds": round(t0 - started, 1), "searchSeconds": round(time.time() - t0, 1),
                  "seedSeconds": {}, "cpus": os.cpu_count(), "torchThreads": torch.get_num_threads(),
                  "bags": {k: len(sides[k]) for k in ("source", "source_test", "target", "target_test")}}
        print(unit, "phase-1 search", params, score, timing["searchSeconds"], "s", flush=True)
        found, settings = None, contextlib.nullcontext()
        if phase2_params is not None:                                     # chosen elsewhere
            missing = sorted(set(SPACE2) - set(phase2_params))
            if missing:
                raise ValueError(f"phase2_params lacks {missing}")
            found = {"params": {k: phase2_params[k] for k in SPACE2}, "score": None, "trials": [],
                     "fixed": True, "rule": "given: chosen before this run, without target labels"}
            settings = phase2_settings(**found["params"])
        elif search2:                                                     # once per unit too
            t2 = time.time()
            found = search_phase2(sides, cfg, device, trials2, rotation, params, unit, pick=pick)
            timing["search2Seconds"] = round(time.time() - t2, 1)
            settings = phase2_settings(**found["params"])
            print(unit, "phase-2 search", found["params"], found["score"], timing["search2Seconds"], "s",
                  flush=True)
        with settings:
            for sd in seeds:
                t1 = time.time()
                given = shared["base_of"](int(sd)) if shared and "base_of" in shared else None
                try:
                    runs[str(sd)] = run_unit(sides, cfg, device, int(sd), rotation, (params, score, trials),
                                             unit, given=given)
                except FloatingPointError as e:      # this seed's phase 2 diverged: keep the others
                    runs[str(sd)] = {"failed": f"FloatingPointError: {e}"}
                    print(unit, "seed", sd, "diverged:", e, flush=True)
                timing["seedSeconds"][str(sd)] = round(time.time() - t1, 1)
                print(unit, "seed", sd, "done", round(time.time() - started), "s", flush=True)
                PG.tick("milcreda-seed", f"MIL-CREDA seed {sd}")
            phase_two = (P.PHASE_TWO if found is None else
                         {**P.PHASE_TWO, "lambdaGlobal": two_phase.LAMBDA_GLOBAL,
                          "lambdaLocal": two_phase.LAMBDA_LOCAL})
    name = f"xpilot-{unit.replace(':', '-').replace('>', '').replace('-', '_')}.json"
    sim = ({"setting": setting, "effect": effect, "shift": shift, "nSubjects": n_subjects, "simSeed": sim_seed,
            **sim_record} if experiment == "sim" else None)
    record = {"experiment": experiment, "unit": unit, "rotation": rotation, "sim": sim,
              "half": half, "objective": OBJECTIVE, "runs": runs,
              "phaseTwo": phase_two, "seconds": round(time.time() - started), "timing": timing,
              "environment": environment}
    if found is not None:
        record["phase2Search"] = found
    if not full_trace:
        record["fullTrace"] = False
    (workdir / name).write_text(json.dumps(record, default=str), encoding="utf-8")
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": unit, "units": [unit],
                                "complete": True, "gpu": None, "estimates": {}},
                   "environment": environment, "runs": {"ran": [unit], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"ran": [unit], "seconds": round(time.time() - started)}
