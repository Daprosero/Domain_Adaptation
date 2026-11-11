"""Phase-1 search of the spectral log-power encoder, with and without the 1 Hz high-pass.

Diagnostic outside the plan's protocol; writes nothing to the model store. General scheme
settled with the operator on 2026-11-11: whole bags (at most 60 evenly spaced windows per
subject), common average reference, no amplitude rejection, ocular activity kept. Two
pre-filters, each with its own study and the same budget: "0.2" uses the stored
0.2-45 Hz windows as they are (plan Sec. 2.7), "1" adds the 1-45 Hz band-pass.

Per outer rotation, a seeded TPE study of `trials` trials over lr1, the explicit L2, the
projection width, the phase-1 attention temperature (gamma 0) and the epoch count. Each trial
trains on the three inner splits of the rotation's non-test source folds; the score pools the
three validations' subject predictions once: (BA at 0.5 + AUC) / 2. The test fold is never
read by the search. The chosen setting is refitted per seed on every non-test source subject.
Its decision cut is the one that maximizes BA over the best trial's pooled inner-validation
predictions, and it is applied unchanged to the source test subjects and to the target test
subjects (predict, no adaptation). No target label chooses anything.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, model_store, remote_units, spectral_trunk, two_phase
from MIL_CREDA_EEG import learned_phase1 as L
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2

SAMPLER_SEED = 20261111
SEARCH_SEED = 0
N_INNER = 3
SPACE = {"lr1": ("log", 1e-4, 1e-2), "l2": ("log", 1e-4, 1e-1), "width": ("cat", (32, 64, 128)),
         "tauAtt1": ("log", 1.0, 10.0), "epochs": ("cat", (50, 100, 200))}


def suggest(trial) -> dict:
    out = {}
    for k, (kind, *a) in SPACE.items():
        out[k] = (trial.suggest_float(k, a[0], a[1], log=True) if kind == "log"
                  else trial.suggest_categorical(k, list(a[0])))
    return out


def windows(raw, highpass: float, channels=None) -> list:
    """Whole bags with CAR; "1" adds the 1-45 Hz band-pass, "0.2" keeps the stored band."""
    if float(highpass) >= 1.0:
        return L._filtered_whole(raw, car=True, reject=None, channels=channels)
    out = []
    for b in raw:
        w = np.asarray(b.windows() if callable(b.windows) else b.windows, dtype=np.float64)
        if channels is not None:
            names = list(b.channels)
            w = w[:, [names.index(c) for c in channels]]
        w = w - w.mean(1, keepdims=True)
        out.append(L._stored_whole([two_phase.BagTensor(
            b.subject, b.condition, two_phase.CLASSES.index(b.label),
            torch.from_numpy(np.ascontiguousarray(w)).float())])[0])
    return out


@contextlib.contextmanager
def width(w: int):
    saved = spectral_trunk.WIDTH
    spectral_trunk.WIDTH = int(w)
    try:
        yield
    finally:
        spectral_trunk.WIDTH = saved


def fit(train, params: Mapping, seed: int, cfg, device):
    hyper = two_phase.check_hyper({**L.HYPER, "lr1": float(params["lr1"]), "epochs1": int(params["epochs"]),
                                   "attentionTemperature": float(params["tauAtt1"]), "attentionGamma": 0.0,
                                   "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    with spectral_trunk.installed(), width(params["width"]):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        c = int(train[0].windows.shape[1])
        m = model_store.build_model("MIL-CREDA", c, c, hyper).to(device)
        m.source_trunk.norm = sc.standard_of(m.source_trunk, train).to(device)
        with explicit_l2(float(params["l2"])):
            two_phase.train_source_phase(m, train, [], hyper, g, device, seed=seed)
    return m


def scores(m, units, device) -> tuple[list, list]:
    q = np.nan_to_num(two_phase.evaluate(m, units, "source", device)[:, 1].detach().cpu().numpy(), nan=.5)
    return [int(u.label) for u in units], [float(v) for v in q]


def selection_score(y, p) -> float:
    if len(set(y)) < 2:
        return float("nan")
    return 0.5 * (balanced_accuracy_score(y, np.asarray(p) > .5) + roc_auc_score(y, p))


def best_cut(y, p) -> float:
    grid = sorted(set(p))
    cands = [(grid[i] + grid[i + 1]) / 2 for i in range(len(grid) - 1)] or [0.5]
    return float(max(cands, key=lambda t: (balanced_accuracy_score(y, np.asarray(p) > t), -abs(t - .5))))


def metrics(y, p, cut) -> dict:
    p = np.asarray(p)
    return {"n": len(y), "AUC": round(float(roc_auc_score(y, p)), 3),
            "BA": round(float(balanced_accuracy_score(y, p > .5)), 3),
            "BAatCut": round(float(balanced_accuracy_score(y, p > cut)), 3),
            "positiveAtCut": round(float(np.mean(p > cut)), 3)}


def run_rotation(transfer: str, highpass: float, r: int, trials: int, seeds, cfg, bags, device) -> dict:
    from MIL_CREDA_EEG import v63_setup
    t = v63_setup.transfer(transfer)
    a = A.assign_from_bags(t, bags)
    channels = v63_setup.channels_of(t)
    get = lambda side, role, split=None: windows(A.side_bags(a, bags, side, r, role, split=split), highpass, channels)
    inner = [(get("source", "inner_train", i), get("source", "inner_validation", i)) for i in range(N_INNER)]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED + r))
    pooled = {}

    def objective(trial):
        params = suggest(trial)
        y, p = [], []
        for tr, va in inner:
            vy, vp = scores(fit(tr, params, SEARCH_SEED, cfg, device), va, device)
            y += vy
            p += vp
        pooled[trial.number] = (y, p)
        return selection_score(y, p)

    study.optimize(objective, n_trials=int(trials))
    best = study.best_trial
    cut = best_cut(*pooled[best.number])
    source_test, target_test = get("source", "test"), get("target", "test")
    refit = get("source", A.REFIT_ROLE)
    out = {"rotation": r, "params": best.params, "selection": round(float(best.value), 4), "cut": round(cut, 4),
           "trials": [{"number": x.number, "value": x.value, "params": x.params} for x in study.trials],
           "seeds": {}}
    for seed in seeds:
        m = fit(refit, best.params, int(seed), cfg, device)
        out["seeds"][str(seed)] = {"source": scores(m, source_test, device), "target": scores(m, target_test, device),
                                   "sourceSubjects": [u.subject for u in source_test],
                                   "targetSubjects": [u.subject for u in target_test]}
    return out


def summary(rotations: Sequence[Mapping], seed: str) -> dict:
    """Pooled over the rotations: each subject once, scored at its rotation's cut."""
    res = {}
    for side in ("source", "target"):
        y, p, c = [], [], []
        for rr in rotations:
            sy, sp = rr["seeds"][seed][side]
            y += sy
            p += sp
            c += [rr["cut"]] * len(sy)
        pa, ca = np.asarray(p), np.asarray(c)
        res[side] = {"n": len(y), "AUC": round(float(roc_auc_score(y, pa)), 3),
                     "BA": round(float(balanced_accuracy_score(y, pa > .5)), 3),
                     "BAatCut": round(float(balanced_accuracy_score(y, pa > ca)), 3),
                     "positiveAtCut": round(float(np.mean(pa > ca)), 3)}
    return res


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, highpasses=(0.2, 1.0),
          trials: int = 20, rotations=(0, 1, 2, 3), seeds=(1, 2),
          workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ), placement)
    device = torch.device(cfg.get("device", "cpu"))
    bags = campaign.load_bags(cfg)
    mode = environ.get(remote_units.RUN_MODE_ENV)
    started = time.time()
    runs = []
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness:
        for hp in highpasses:
            rots = [run_rotation(transfer, float(hp), int(r), trials, seeds, cfg, bags, device) for r in rotations]
            runs.append({"transfer": transfer, "highpass": float(hp), "rotations": rots,
                         "pooled": {str(s): summary(rots, str(s)) for s in seeds}})
            print(transfer, hp, json.dumps(runs[-1]["pooled"]), flush=True)
    (workdir / "spectral-search.json").write_text(json.dumps(
        {"runs": runs, "seconds": round(time.time() - started), "environment": environment}, default=str),
        encoding="utf-8")
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": transfer, "units": [transfer],
                                "complete": True, "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": [transfer], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"ran": [transfer], "skipped": [], "environment": environment}
