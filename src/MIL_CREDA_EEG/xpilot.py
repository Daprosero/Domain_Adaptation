"""Quick pilots of MIL-CREDA phase 2 outside BALLADEER's device transfers.

Diagnostic outside the plan. Three settings, one shared recipe: the spectral log-power
encoder, a phase 1 searched once per unit (rotation, half) on SOURCE subjects only and refit
per seed (TPE, `spectral_search`'s
space, score (BA@0.5 + AUC)/2 pooled over three subject-wise inner validations), then
`phase2_trace.run_rotation` with the objective eq39+hardloc and the copy arm (both sides
read the same channels). Compared on the target test subjects: predict (phase 1, no
adaptation) against the end of phase 2.

* "giga": GiGA motor imagery (Cho et al. 2017), cross-subject. A bag is one trial; its
  instances are 1 s windows at a 0.5 s step over the stored 2.5-5 s segment (0.5-3 s after
  the cue), all with the trial's label. 1-45 Hz band-pass on the whole segment, then common
  average, then the cut. At most TRIALS_PER_CLASS trials per class per subject (the first ones, by index).
* "ieee": IEEE ADHD (Nasrabadi et al.), cross-subject. A bag is one subject: 4 s windows at
  50% overlap, 50 Hz notch, then the v63 instance rule (<= 60 windows, 1-45 Hz, average).
* "xds": cross-dataset ADHD between IEEE and one BALLADEER task (default L1, the largest),
  both directions, on the 19 IEEE channels (all present in L1). The source is the whole
  source dataset; BALLADEER's bags are its stored 4 s windows.

Partition, always by subject (no subject on two sides, ever):
* cross-subject: subject i (sorted, per class) goes to fold i % 5. Rotation r: the target is
  fold r, split by alternating subjects into two disjoint halves; `half` h adapts on one and
  tests the other. The source is the other four folds: fold (r + 1) % 5 is its test, the
  remaining three train, with three subject-wise inner splits for the search and the cut.
* cross-dataset: the source dataset keeps 1 in 5 subjects (per class) as its test; the
  target dataset is split in 4 folds per class, rotation r tests fold r and adapts the rest.
Target labels are read only to score, and to group GiGA test subjects by their predict BA.
"""
from __future__ import annotations

import contextlib
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

from MIL_CREDA_EEG import campaign, eegnet, remote_units, spectral_trunk, two_phase
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import spectral_search as SS
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import xds_pilot as X

EXPERIMENTS = ("giga", "ieee", "xds")
FOLDS, INNER, XDS_FOLDS = 5, 3, 4
TRIALS_PER_CLASS = 25
GIGA_WINDOW, GIGA_STEP = 128, 64
IEEE_WINDOW = 512
OBJECTIVE = "eq39+hardloc"
SAMPLER_SEED = 20261112


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


def balladeer_bags(cfg, condition, channels) -> list[two_phase.BagTensor]:
    bags = campaign.load_bags(cfg)
    out = []
    for s in sorted(bags.holders(condition)):
        bag = bags.get(s, condition)
        w = np.asarray(bag.windows(), dtype=np.float64)
        names = list(bag.channels)
        w = w[:, [names.index(c) for c in channels]]
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
    cs = ct = int(sides["source"][0].windows.shape[1])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED + int(rotation)))

    def objective(trial):
        params = SS.suggest(trial)
        y, p = [], []
        for tr, va in sides["inner"]:
            m = P.phase_one(tr, cs, ct, P._hyper(cfg, temperature=params["tauAtt1"], searched=params),
                            SS.SEARCH_SEED, device, searched=params)
            vy, vp = P._subjects(va, P._probabilities(m, va, "source", device))
            y, p = y + vy, p + vp
        return SS.selection_score(y, p)

    study.optimize(objective, n_trials=int(trials))
    return study.best_trial.params, round(float(study.best_value), 4)


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


def run_unit(sides, cfg, device, seed, rotation, searched, unit, log=print):
    """Refit phase 1 at `searched` = (params, score, trials) with `seed`, then phase 2."""
    params, score, trials = searched
    P.PHASE_ONE_SEARCHED.clear()
    P.PHASE_ONE_SEARCHED[int(rotation)] = params
    names = [f"ch{i}" for i in range(int(sides["source"][0].windows.shape[1]))]   # same montage
    out = P.run_rotation(unit, cfg, None, None, device, seed, int(rotation), "copy", [OBJECTIVE],
                         log=log, sides={**sides, "s_names": names, "t_names": names})
    out["phase1Search"] = {"params": params, "score": score, "trials": int(trials)}
    out["subjectsSplit"] = sides["subjects"]
    out["targetTestUnits"] = [b.subject for b in sides["target_test"]]   # row order of the target test
    return out


def smoke(config: Mapping, product=None, *, experiment: str, rotation: int, seeds: Sequence[int] = (1,),
          half: int = 0, direction: str = "IEEE->L1", condition: str = "L1", trials: int = 20,
          epochs2: int | None = None, workdir=None, environ: Mapping[str, str] | None = None) -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment in {EXPERIMENTS}")
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ), placement)
    device = torch.device(cfg.get("device", "cpu"))
    if epochs2 is not None:
        P.PHASE_TWO["epochs2"] = int(epochs2)
    started = time.time()
    if experiment == "giga":
        bags = giga_bags(find_giga(environ.get("GIGA_ROOT")))
        cfg = {**cfg, "schedule": {**cfg["schedule"], "bagsPerStep": 32}}
        sides, unit, samples = cross_subject_sides(bags, rotation, half), f"giga:{rotation}:{half}", GIGA_WINDOW
    elif experiment == "ieee":
        bags = ieee_bags(X.find_ieee(environ.get("IEEE_ROOT")))
        sides, unit, samples = cross_subject_sides(bags, rotation, half), f"ieee:{rotation}:{half}", IEEE_WINDOW
    else:
        ieee = ieee_bags(X.find_ieee(environ.get("IEEE_ROOT")))
        ball = balladeer_bags(cfg, condition, X.IEEE_CHANNELS)
        src, tgt = (ieee, ball) if direction.startswith("IEEE") else (ball, ieee)
        sides, unit, samples = cross_dataset_sides(src, tgt, rotation), f"{direction}:{rotation}", IEEE_WINDOW
    runs = {}
    with spectral_trunk.installed(), window_samples(samples):
        params, score = search_phase1(sides, cfg, device, trials, rotation)   # once per unit
        print(unit, "phase-1 search", params, score, round(time.time() - started), "s", flush=True)
        for sd in seeds:
            runs[str(sd)] = run_unit(sides, cfg, device, int(sd), rotation, (params, score, trials), unit)
            print(unit, "seed", sd, "done", round(time.time() - started), "s", flush=True)
    name = f"xpilot-{unit.replace(':', '-').replace('>', '').replace('-', '_')}.json"
    (workdir / name).write_text(json.dumps({"experiment": experiment, "unit": unit, "rotation": rotation,
                                            "half": half, "objective": OBJECTIVE, "runs": runs,
                                            "phaseTwo": P.PHASE_TWO, "seconds": round(time.time() - started),
                                            "environment": environment}, default=str), encoding="utf-8")
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
