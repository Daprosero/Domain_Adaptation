"""The eight comparators on xpilot's units, from MIL-CREDA's shared phase-1 base.

Diagnostic outside the plan, the comparator side of a fair comparison with `xpilot.smoke`.
Every method, MIL-CREDA included, sees:

* the same unit: `xpilot.build_sides` (same subjects on the same sides, same partition);
* the same encoder and the same searched phase 1: `xpilot.search_phase1` once per unit, then
  per seed the base model refit at the searched parameters (`phase2_trace.phase_one`, the call
  `phase2_trace.run_rotation` makes, same seed, transfer tag and fold, so the same bytes);
* the same decision cut: chosen per seed on the source inner-validation subjects (split 0) of
  a phase-1 candidate trained on inner_train split 0 (`phase2_trace.CUT_RULE`);
* the same predict: the base after `two_phase.bridge` and the arm's target initialisation
  (`phase2_trace.initialise_target`: copy for one montage, warm for two), at the phase-1
  aggregation (gamma 0, the searched temperature);
* the same phase-2 search budget and selection rule: one TPE study per unit and method on seed
  1 with `trials2` trials, over the method's own declared phase-2 space (`SPACE_NOTES`),
  scored only by `phase2_select.centroid_score`: the adapted model's representations and
  argmax pseudo-labels of the target adapt half, against the held-out SOURCE test bags and
  their labels. No target label is read and the target test is never passed to the search.

Each comparator runs through its own registered phase-2 trainer (`search.train_phase_two`,
the hook the campaign calls), unchanged, handed the shared base as a "common" phase-1 record.
Two runtime adapters make the record fit that hook without touching it (`common_wrapper`):
the comparators' common model is built on the base's trunk (the spectral trunk with its fixed
feature standardization, `spectral_check.Standard`, in place of the trunk's LayerNorm, at the
searched width), and its attention softmax runs at the base's phase-1 temperature instead of
`mil_wrapper.COMMON_TEMPERATURE` = 1. At gamma 0 the common attention is Eq. 15's relevance
term alone, exactly the base's, so the wrapped model reproduces the base's predictions
(`commonWrapperEquality`, recorded per seed). The record's identity fields (name "common",
`search.PHASE_ONE_SEED`) are the ones the hooks' guards expect; the base behind them is this
unit's per-seed base. Every trainer then runs exactly `epochs2` epochs and keeps the last
(its own rule), with no target validation bags (a refit: nothing evaluated during training).

AdaptMIL is skipped (`SKIPPED`): its phase 2 needs its own architecture.
Writes xcompare-<unit>.json.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import torch
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score

from MIL_CREDA_EEG import progress as PG
from MIL_CREDA_EEG import campaign, eegnet, mil_wrapper, model_store, phase2_select, remote_units
from MIL_CREDA_EEG import search, spectral_trunk, two_phase
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG import spectral_search as SS
from MIL_CREDA_EEG import xpilot as XP

#: the eight comparators of Sec. 7.1, in the store's order
ALL8 = tuple(s for s in model_store.STRATEGIES if s != "MIL-CREDA")
SEARCH_SEED = 1
SAMPLER_SEED = XP.SAMPLER_SEED
SKIPPED = {
    "AdaptMIL": (
        "AdaptMIL keeps its native ABMIL aggregator and its own phase-1 group (model_store.GROUPS "
        "'AdaptMIL', comparators/adaptmil/model.AdaptMILModel, gated attention with its own "
        "parameters); its phase 2 (adaptmil.training.adapt_target_phase / train_phase_two) "
        "requires an AdaptMILModel and its own phase-1 state, so it cannot start from the shared "
        "MIL-CREDA base: running it would need a separate phase 1 with another aggregator, which "
        "breaks the shared-base design"),
}
SELECTION_RULE = (
    "phase2_select.centroid_score: the adapted model's bag representations and argmax "
    "pseudo-labels of the target adapt half (target path), class centroids, then the held-out "
    "source test bags (source path) assigned to the nearest centroid and scored by balanced "
    "accuracy against their source labels; no target label, no target test bag; one TPE study "
    "per unit and method on seed 1, same trials2 for every method, the best trial's parameters "
    "used by every seed")
#: Why each method searches the space its registration declares (search.phase_two_space).
SPACE_NOTES = {
    "UDA-DDA": "shared lr2/wd2 + the paper's two loss weights (MMD alpha, CMMD), kernel scale and "
               "the two pseudo-label confidence thresholds (sorted into Eq. 7's middle/late stages): "
               "uda_dda.register, plan Sec. 7.7 generic rule",
    "CCA-LSC": "shared lr2/wd2 + centroid, discriminative and adversarial loss weights, the "
               "calibration hm and the prior threshold: cca_lsc.register (REPLICATION.md "
               "Secs. 3.3-3.4); at least search.MIN_TRAINED_EPOCHS['CCA-LSC'] epochs (3 warm-up + 1)",
    "CASUAL": "shared lr2/wd2 + alignment, entropy and VAT loss weights and the VAT radius as a "
              "fraction of the typical window norm (own axis): casual.register",
    "COT": "shared lr2/wd2 + cluster and OT loss weights, the dual learning rate and the dual "
           "step multiplier: cot.register (COT Sec. 4.1, Algorithm 1)",
    "CS+CCS": "shared lr2/wd2 + CS and CCS divergence weights and the Z kernel multiplier: "
              "cs_ccs.register",
    "CLUST — variante source-free": "shared lr2/wd2 + clustering, consistency and information "
                                    "weights, the confidence threshold and the teacher EMA "
                                    "1 - rho (own axis): clust.register",
    "CLUDA": "shared lr2/wd2 + the contrastive weight and its temperature (own axis, contains the "
             "official 0.07): cluda.register",
}


# ------------------------------------------------------------------ the common wrapper on the base


class StandardizedSpectralTrunk(spectral_trunk.SpectralTrunk):
    """The base's trunk shape: the spectral trunk whose LayerNorm phase 1 replaced by a fixed
    feature standardization (`phase2_trace.phase_one`, `initialise_target`). The placeholder
    statistics are overwritten by the record's state when a trainer loads it."""

    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(channels, *args, **kwargs)
        n = int(self.norm.normalized_shape[0])
        self.norm = sc.Standard(torch.zeros(n), torch.ones(n))


@contextlib.contextmanager
def common_wrapper(width: int, temperature: float):
    """Models built inside use the base's trunk at `width`, and the common attention runs at
    `temperature` (the base's phase-1 one); both restored on exit."""
    saved_trunk, saved_temperature = None, mil_wrapper.COMMON_TEMPERATURE
    with spectral_trunk.installed(), SS.width(int(width)):
        saved_trunk = eegnet.EEGNetTrunk
        eegnet.EEGNetTrunk = StandardizedSpectralTrunk
        mil_wrapper.COMMON_TEMPERATURE = float(temperature)
        try:
            yield
        finally:
            eegnet.EEGNetTrunk = saved_trunk
            mil_wrapper.COMMON_TEMPERATURE = saved_temperature


def _context(params1) -> dict:
    return {"temperature": float(params1["tauAtt1"]), "width": int(params1["width"]),
            "epochs1": int(params1["epochs"])}


def _digest(source, target) -> str:
    subjects = {"source": sorted({b.subject for b in source}), "target": sorted({b.subject for b in target})}
    return hashlib.sha256(json.dumps(subjects).encode()).hexdigest()


def _channels(bags) -> int:
    return int(bags[0].windows.shape[1])


def common_record(prepared, source, target, rotation, unit):
    """(SearchData, record) for a comparator hook: the shared base, bridged and arm-initialised,
    as the "common" phase-1 checkpoint of this unit; no validation bags (a refit)."""
    data = search.SearchData(
        transfer=SimpleNamespace(name=str(unit)), partition_sha256=_digest(source, target),
        rotations={int(rotation): {"source_train": list(source), "source_valid": [],
                                   "target_train": list(target), "target_valid": []}})
    meta = {"phase": 1, "name": mil_wrapper.GROUP, "transfer": str(unit), "rotation": int(rotation),
            "seed": search.PHASE_ONE_SEED, "partition_sha256": data.partition_sha256,
            "channels": {"source": _channels(source), "target": _channels(target)}, "hyper": {}}
    state = {k: v.detach().clone() for k, v in prepared.state_dict().items()}
    return data, (state, meta)


def adapt(method, prepared, context, *, source, target, params, seed, rotation, unit, epochs2, cfg,
          device):
    """The method's own phase-2 trainer from the shared base. Call inside `common_wrapper`."""
    data, record = common_record(prepared, source, target, rotation, unit)
    schedule = {"epochs1": context["epochs1"], "epochs2": epochs_of(method, epochs2),
                "bagsPerStep": int(cfg["schedule"]["bagsPerStep"])}
    return search.train_phase_two(method, data, int(rotation), int(seed), record, dict(params), schedule,
                                  device)["model"]


def epochs_of(method, epochs2) -> int:
    return max(int(epochs2), int(search.MIN_TRAINED_EPOCHS.get(method, 1)))


# ------------------------------------------------------------------ the shared base


def snapshot(model, bags, cut, device) -> dict:
    """Subject rows (label, probability) on the target path, BA at the cut, AUC, kappa."""
    y, p = P._subjects(bags, P._probabilities(model, bags, "target", device))
    y, p = np.asarray(y), np.asarray(p)
    two = len(set(y.tolist())) > 1
    hard = (p > cut).astype(int)
    return {"rows": [[int(a), P._s(b, 4)] for a, b in zip(y, p)],
            "BA": P._s(balanced_accuracy_score(y, hard)) if two else None,
            "AUC": P._s(roc_auc_score(y, p)) if two else None,
            "kappa": P._s(cohen_kappa_score(y, hard)) if two else None,
            "BA05": P._s(balanced_accuracy_score(y, p > .5)) if two else None,
            "posCut": P._s(float(np.mean(hard)))}


def shared_base(sides, cfg, device, seed, rotation, params1, unit) -> dict:
    """The cut and the base of `seed`, as `phase2_trace.run_rotation` computes them at the
    searched phase 1, and the predict model (bridge + arm init)."""
    t0 = time.time()
    source, target = sides["source"], sides["target"]
    cs, ct = _channels(source), _channels(target)
    hyper = P._hyper(cfg, temperature=float(params1["tauAtt1"]), searched=params1)
    cand = P.phase_one(sides["inner_train"], cs, ct, hyper, int(seed), device, transfer=unit, fold=rotation,
                       searched=params1)
    vy, vp = P._subjects(sides["inner_valid"], P._probabilities(cand, sides["inner_valid"], "source", device))
    cut = P.best_cut(vy, vp)
    del cand
    base = P.phase_one(source, cs, ct, hyper, int(seed), device, transfer=unit, fold=rotation,
                       searched=params1)
    if "s_names" in sides:
        arm, s_names, t_names = "warm", sides["s_names"], sides["t_names"]
    else:
        arm = "copy"
        s_names = t_names = [f"ch{i}" for i in range(cs)]
    prepared = copy.deepcopy(base)
    two_phase.bridge(prepared)
    init = P.initialise_target(prepared, arm, target, s_names, t_names)
    sy, sp = P._subjects(sides["source_test"], P._probabilities(base, sides["source_test"], "source", device))
    return {"base": base, "prepared": prepared, "cut": cut, "arm": arm, "init": init,
            "context": _context(params1), "sourceTest": P._metrics(sy, sp, cut),
            "predict": snapshot(prepared, sides["target_test"], cut, device),
            "seconds": round(time.time() - t0, 1)}


def wrapper_equality(prepared, context, bags, device) -> dict:
    """Max |p| difference on `bags` (target path) between the base and the comparators' common
    model built on it in `common_wrapper`: the wrapper starts from exactly the base."""
    q = P._probabilities(prepared, bags, "target", device)
    with common_wrapper(context["width"], context["temperature"]):
        m = mil_wrapper.CommonMILModel(int(prepared.source_trunk.channels), int(prepared.target_trunk.channels))
        m.load_state_dict(prepared.state_dict(), strict=True)
        m.to(device)
        r = P._probabilities(m, bags, "target", device)
    return {"maxAbsProbability": float(np.abs(q - r).max()),
            "temperature": context["temperature"], "gamma": mil_wrapper.COMMON_GAMMA}


# ------------------------------------------------------------------ phase-2 search


def _describe(space) -> dict:
    out = {}
    for k, d in space.items():
        if isinstance(d, FloatDistribution):
            out[k] = {"low": d.low, "high": d.high, "log": bool(d.log)}
        elif isinstance(d, CategoricalDistribution):
            out[k] = {"choices": list(d.choices)}
        else:
            out[k] = repr(d)
    return out


def selection_score(model, target, source_test, device) -> float:
    """The shared label-free rule: target adapt half (representations, argmax pseudo-labels)
    against the source test bags and their source labels."""
    Zt = two_phase.represent(model, target, "target", device).numpy()
    pseudo = two_phase.evaluate(model, target, "target", device).argmax(1).cpu().numpy()
    Zs = two_phase.represent(model, source_test, "source", device).numpy()
    return phase2_select.centroid_score(Zt, pseudo, Zs, [int(b.label) for b in source_test])


def search_phase2(method, prepared, context, *, source, source_test, target, cfg, device, trials2,
                  rotation, unit, epochs2, seed=SEARCH_SEED) -> dict:
    """One TPE study over the method's own phase-2 space, from the base of `seed`. Receives the
    source training bags, the source test bags and the target ADAPT half only; it never sees
    the target test bags, and reads no target label."""
    space = search.phase_two_space(method)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED + int(rotation)))

    def objective(trial):
        params = search.suggest(trial, space)
        # the phase-2 epoch count is searched over the same choices as MIL-CREDA's (xpilot.SPACE2)
        e2 = (int(trial.suggest_categorical("epochs2", list(XP.SPACE2["epochs2"][1])))
              if epochs2 is None else int(epochs2))
        with common_wrapper(context["width"], context["temperature"]):
            try:
                model = adapt(method, prepared, context, source=source, target=target, params=params,
                              seed=seed, rotation=rotation, unit=unit, epochs2=e2, cfg=cfg, device=device)
            except FloatingPointError as e:        # the trainer's own non-finite refusal
                trial.set_user_attr("nonFinite", str(e))
                return 0.0
            score = selection_score(model, target, source_test, device)
        return float(score) if np.isfinite(score) else 0.0

    study.optimize(objective, n_trials=int(trials2),
                   callbacks=[PG.optuna_callback(f"search:{method}", f"{method} phase-2 search", trials2)])
    best = dict(study.best_trial.params)
    best_epochs = int(best.pop("epochs2")) if "epochs2" in best else int(epochs2)
    return {"space": _describe(space), "params": best, "bestEpochs2": epochs_of(method, best_epochs),
            "score": round(float(study.best_value), 4), "trials": int(trials2), "seed": int(seed),
            "epochs2Searched": epochs2 is None, "rule": SELECTION_RULE, "why": SPACE_NOTES.get(method),
            "trialScores": [None if t.value is None else round(float(t.value), 4) for t in study.trials],
            "nonFinite": [t.number for t in study.trials if "nonFinite" in t.user_attrs]}


# ------------------------------------------------------------------ one unit


def smoke(config: Mapping, product=None, *, experiment: str, rotation: int, half: int = 0,
          seeds: Sequence[int] = (1,), trials: int = 20, trials2: int = 20, methods: Sequence[str] = ALL8,
          direction: str = "IEEE->Robots", condition: str = "Robots", epochs2: int | None = None,
          setting: str = "subject", effect: float = 1.0, shift: float = 0.0, n_subjects: int = 60,
          sim_seed: int = 0, d_target: float | None = None, artifact_rate: float = 0.0,
          artifact_class_corr: float = 0.0, event_fraction: float = 0.0, paired: bool = False,
          spread: float = 1.0,
          workdir=None, environ: Mapping[str, str] | None = None, shared: Mapping | None = None,
          phase2_fixed: Mapping[str, Mapping] | None = None) -> dict:
    """`phase2_fixed` {method: {its phase-2 params..., "epochs2": e}} skips that method's
    phase-2 search and runs its seeds at those settings, chosen elsewhere without target labels."""
    if experiment not in XP.EXPERIMENTS:
        raise ValueError(f"experiment in {XP.EXPERIMENTS}")
    unknown = [m for m in methods if m not in ALL8]
    if unknown:
        raise ValueError(f"methods are among {list(ALL8)}; got {unknown}")
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)))
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register([m for m in methods if m not in SKIPPED])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ), placement)
    device = torch.device(cfg.get("device", "cpu"))
    fixed_epochs2 = None if epochs2 is None else int(epochs2)    # None: searched, like MIL-CREDA's
    epochs2 = int(P.PHASE_TWO["epochs2"] if epochs2 is None else epochs2)
    started = time.time()
    built = shared["built"] if shared else XP.build_sides(cfg, environ, experiment=experiment, rotation=rotation, half=half,
                           direction=direction, condition=condition, setting=setting, effect=effect,
                           shift=shift, n_subjects=n_subjects, sim_seed=sim_seed, d_target=d_target,
                           artifact_rate=artifact_rate, artifact_class_corr=artifact_class_corr,
                           event_fraction=event_fraction, paired=paired, spread=spread)
    sides, unit, samples, cfg = built["sides"], built["unit"], built["samples"], built["cfg"]
    sim_record, effect = built["sim_record"], built["effect"]
    out_methods, bases = {}, {}
    with spectral_trunk.installed(), XP.window_samples(samples):
        t0 = time.time()
        params1, score1 = (shared["phase1"] if shared else
                           XP.search_phase1(sides, cfg, device, trials, rotation))   # once per unit
        timing = {"loadSeconds": round(t0 - started, 1), "searchSeconds": round(time.time() - t0, 1),
                  "baseSeconds": {}, "cpus": os.cpu_count(), "torchThreads": torch.get_num_threads(),
                  "bags": {k: len(sides[k]) for k in ("source", "source_test", "target", "target_test")}}
        print(unit, "phase-1 search", params1, score1, timing["searchSeconds"], "s", flush=True)

        def base_of(sd):
            if sd not in bases and shared and "base_of" in shared:     # the unit's one base per seed
                bases[sd] = shared["base_of"](int(sd))
                timing["baseSeconds"][str(sd)] = bases[sd]["seconds"]
            elif sd not in bases:
                bases[sd] = shared_base(sides, cfg, device, sd, rotation, params1, unit)
                timing["baseSeconds"][str(sd)] = bases[sd]["seconds"]
                PG.tick("base", f"shared base seed {sd}")
            return bases[sd]

        for method in methods:
            if method in SKIPPED:
                out_methods[method] = {"skipped": SKIPPED[method]}
                continue
            t1 = time.time()
            try:
                if phase2_fixed is not None and method in phase2_fixed:
                    given = dict(phase2_fixed[method])
                    e2 = int(given.pop("epochs2"))
                    found = {"params": given, "bestEpochs2": epochs_of(method, e2), "epochs2Given": e2,
                             "score": None, "fixed": True,
                             "rule": "given: chosen before this run, without target labels"}
                else:
                    b1 = base_of(SEARCH_SEED)
                    t1 = time.time()
                    found = search_phase2(method, b1["prepared"], b1["context"], source=sides["source"],
                                          source_test=sides["source_test"], target=sides["target"],
                                          cfg=cfg, device=device, trials2=trials2, rotation=rotation,
                                          unit=unit, epochs2=fixed_epochs2)
                rec = {"phase2Search": found, "runs": {},
                       "seconds": {"search": round(time.time() - t1, 1), "seeds": {}}}
                for sd in seeds:
                    b = base_of(int(sd))
                    t2 = time.time()
                    with common_wrapper(b["context"]["width"], b["context"]["temperature"]):
                        model = adapt(method, b["prepared"], b["context"], source=sides["source"],
                                      target=sides["target"], params=found["params"], seed=int(sd),
                                      rotation=rotation, unit=unit, epochs2=found["bestEpochs2"], cfg=cfg,
                                      device=device)
                        final = snapshot(model, sides["target_test"], b["cut"], device)
                    rec["runs"][str(sd)] = {"predict": b["predict"], "final": final, "cut": P._s(b["cut"])}
                    rec["seconds"]["seeds"][str(sd)] = round(time.time() - t2, 1)
                    PG.tick(f"seed:{method}", f"{method} seed {sd}")
                out_methods[method] = rec
                print(unit, method, "search", found["score"], "BA", {s: r["final"]["BA"]
                      for s, r in rec["runs"].items()}, round(time.time() - t1), "s", flush=True)
            except Exception as e:      # one comparator failing must not lose the others
                traceback.print_exc()
                out_methods[method] = {"failed": f"{type(e).__name__}: {e}",
                                       "seconds": round(time.time() - t1, 1)}
        base_out = {}
        for sd in seeds:
            b = base_of(int(sd))
            base_out[str(sd)] = {"cut": P._s(b["cut"]), "arm": b["arm"], "init": b["init"],
                                 "phase1SourceTest": b["sourceTest"], "predict": b["predict"],
                                 "commonWrapperEquality": wrapper_equality(b["prepared"], b["context"],
                                                                           sides["target_test"], device)}
    name = f"xcompare-{unit.replace(':', '-').replace('>', '').replace('-', '_')}.json"
    sim = ({"setting": setting, "effect": effect, "shift": shift, "nSubjects": n_subjects, "simSeed": sim_seed,
            **{k: v for k, v in sim_record.items() if k != "masks"}} if experiment == "sim" else None)
    (workdir / name).write_text(json.dumps({
        "experiment": experiment, "unit": unit, "rotation": rotation, "half": half, "sim": sim,
        "phase1Search": {"params": params1, "score": score1, "trials": int(trials)},
        "subjectsSplit": sides["subjects"], "targetTestUnits": [b.subject for b in sides["target_test"]],
        "cutRule": P.CUT_RULE, "selectionRule": SELECTION_RULE, "trials2": int(trials2), "epochs2": epochs2,
        "seeds": [int(s) for s in seeds], "base": base_out, "methods": out_methods,
        "seconds": round(time.time() - started), "timing": timing, "environment": environment},
        default=str), encoding="utf-8")
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": unit, "units": [unit],
                                "complete": True, "gpu": None, "estimates": {}},
                   "environment": environment, "runs": {"ran": [unit], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"ran": [unit], "seconds": round(time.time() - started), "file": str(workdir / name)}
