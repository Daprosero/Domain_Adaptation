"""Plan v63, stage 1 on CPU: the target filter, the shift filter and the performance runs.

Experiments plan v63, Secs. 0 (part 1), 4.5-4.8, 5.1-5.4, 7.3 and 7.7. Three kinds of unit,
each resumable and each a whole remote session or a chain of them:

- `S0:<domain>`: the target filter of [exp:S0-codificador] (`v63_filters`), four rotations
  of the domain's own-label phase-1 search (MIL-CREDA's group, `lr1`/`l2`, 20 trials) and
  its refit with each of the 10 report seeds, then the permutation verdict.
- `F:<transfer>`: the label-free shift filter of [exp:F-corrimiento], 1000 permutations.
- `P:<transfer>:<rotation>`: one outer fold of one transfer of [exp:A-nivel], [exp:B-tarea]
  or [exp:C-dispositivo]. Phase 1: one study per group (common, AdaptMIL, MIL-CREDA), each
  20 trials of `lr1` and the explicit L2 coefficient `l2` (wd1 = 0) on the three inner splits (best pooled epoch E1, patience 30, cap
  100), then the refit on every non-test fold for E1 epochs with each report seed; the
  common group's refit applied to the target is `predict` (Sec. 7.5). Phase 2: one study per
  strategy (the 8 comparators and MIL-CREDA), 20 trials each, starting from the phase-1
  checkpoint of its group at the first report seed, selected by the centroid rule (no
  target label, patience 10, cap 100); the selected values are frozen for the outer fold and
  refit for E2 epochs from each seed's own phase-1 checkpoint. MIL-CREDA's phase 2 reads the
  shift gate of this transfer and rotation (its domain-classifier AUC, Sec. 4.7). When EnsV
  chooses another (trial, epoch), that choice is refit too, as the secondary rule of Sec. 4.5.

Every refit leaves one record per (method, seed) under `Results/v63/<experiment>/runs/`:
the target test subjects with label, probability, bag representation, age and frontal
amplitude, and the source test subjects' representations through the same group's frozen
source side (the domain classifier of Sec. 4.4). Labels and ages only score: no decision of
any unit reads a target label or a test-fold label.

**Sessions (remote-execution, the session-chain rule).** Before each trial, refit and
permutation the unit reads `FORGE_SESSION_DEADLINE`; when the next step may not finish
(1.25 x the longest step of its kind measured in this session, plus `marginSeconds`), it
stops, writes `partial.json` as `{"remaining": [...]}` with the ids it was sent and returns.
Optuna studies are SQLite files continued to their declared count; refits and permutations
are files, never redone. The carry paths are `Results/v63` and `Models/v63` (product folder).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from MIL_CREDA_EEG import bag_data, campaign, model_store, partition, remote_units
from MIL_CREDA_EEG import search, two_phase
from MIL_CREDA_EEG import v63_filters as Fl
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["0", "4.5", "4.6", "4.7", "4.8", "5.1", "5.2", "5.3", "5.4", "7.3", "7.5", "7.7"],
    "experiments": ["S0-codificador", "F-corrimiento", "A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "the_unit_ids_are_s0_domain_f_transfer_and_p_transfer_rotation",
        "every_study_has_twenty_trials_and_the_same_budget",
        "the_phase_two_search_starts_from_the_first_report_seeds_checkpoint",
        "each_seed_refits_from_its_own_phase_one_checkpoint_with_the_frozen_values",
        "predict_is_the_common_phase_one_model_on_the_target",
        "a_cut_session_writes_partial_with_the_ids_it_was_sent",
        "a_finished_refit_is_never_trained_again",
        "no_target_label_enters_a_selection",
        "a_diverged_trial_fails_inside_the_same_budget",
        "a_study_whose_trials_all_diverged_leaves_failed_records",
    ],
}

DEFAULT_PRODUCT = bag_data.REPOSITORY / "MIL-CREDA-EEG"
RESULTS = Path("Results") / "v63"
MODELS = Path("Models") / "v63"
CARRY_PATHS = (str(RESULTS), str(MODELS))
REPORT_SEEDS = tuple(range(1, 11))
SEARCH_SEED = search.SEARCH_SEED
GROUPS = ("common", "AdaptMIL", "MIL-CREDA")
STRATEGIES = model_store.STRATEGIES
PREDICT = "predict"
DEFAULT_MARGIN = 300.0
PHASE_ONE_OBJECTIVE = ("plan v63, Secs. 4.5 and 7.7: the best epoch's source-validation "
                       "balanced accuracy pooled over the three inner splits; cap 100, patience 30")
PHASE_TWO_OBJECTIVE = ("plan v63, Secs. 4.5 and 7.7, no target label: the best epoch's "
                       "centroid-rule balanced accuracy pooled over the three inner splits; "
                       "cap 100, patience 10; ensvProbabilities recorded per epoch")


class DeadlineReached(RuntimeError):
    """The next step of a unit may not finish before the session deadline."""


# ------------------------------------------------------------------ config and units


def check_config(config: Mapping) -> dict:
    cfg = dict(config)
    cfg["schedule"] = search.check_schedule(cfg.get("schedule") or {})
    cfg["nTrials"] = int(cfg.get("nTrials", search.N_TRIALS))
    cfg["seeds"] = [int(s) for s in cfg.get("seeds", REPORT_SEEDS)]
    cfg["strategies"] = list(cfg.get("strategies", STRATEGIES))
    unknown = [s for s in cfg["strategies"] if s not in STRATEGIES]
    if unknown or "MIL-CREDA" not in cfg["strategies"]:
        raise V.V63Error(f"strategies are among {list(STRATEGIES)} and include MIL-CREDA; "
                         f"unknown {unknown}")
    cfg["permutations"] = int(cfg.get("permutations", Fl.PERMUTATIONS))
    cfg["ensvRefits"] = bool(cfg.get("ensvRefits", True))
    cfg["marginSeconds"] = float(cfg.get("marginSeconds", DEFAULT_MARGIN))
    cfg["threads"] = int(cfg.get("threads", 4))
    cfg["device"] = cfg.get("device", "cpu")
    return cfg


def all_unit_ids() -> list[str]:
    return ([f"S0:{d}" for d in V.DOMAINS] + [f"F:{t.name}" for t in V.TRANSFERS]
            + [f"P:{t.name}:{r}" for t in V.TRANSFERS for r in range(partition.N_FOLDS)])


def parse_unit(uid: str) -> tuple:
    kind, _, rest = uid.partition(":")
    if kind == "S0" and rest in V.DOMAINS:
        return ("S0", rest)
    if kind == "F":
        return ("F", V.transfer(rest))
    if kind == "P":
        name, _, r = rest.rpartition(":")
        if r.isdigit() and int(r) < partition.N_FOLDS:
            return ("P", V.transfer(name), int(r))
    raise V.V63Error(f"{uid!r} is not a v63 unit; the units are S0:<domain>, F:<transfer> "
                     f"and P:<transfer>:<rotation>")


def requested_units(units: Sequence[str] | None, environ: Mapping[str, str]) -> list[str]:
    if units is None:
        raw = environ.get(remote_units.UNITS_ENV)
        if raw is None:
            raise V.V63Error(f"name the units, or set {remote_units.UNITS_ENV}")
        units = json.loads(raw)
        if not isinstance(units, list) or not all(isinstance(u, str) for u in units):
            raise V.V63Error(f"{remote_units.UNITS_ENV} must be a JSON list of unit ids")
    for u in units:
        parse_unit(u)
    return list(dict.fromkeys(units))


def passing_targets(product: Path) -> dict[str, bool]:
    """Domain -> passes, from the target-filter verdicts on disk."""
    out = {}
    for d in V.DOMAINS:
        p = product / RESULTS / "S0-codificador" / f"target_filter_{d}.json"
        if p.is_file():
            out[d] = bool(json.loads(p.read_text())["passes"])
    return out


def performance_units(product: Path) -> list[str]:
    """Every P unit whose target passed the filter; refuses while a verdict is missing."""
    verdict = passing_targets(product)
    missing = [d for d in V.DOMAINS if d not in verdict]
    if missing:
        raise V.V63Error(f"the target filter has no verdict yet for {missing}")
    return [f"P:{t.name}:{r}" for t in V.TRANSFERS if verdict[t.target]
            for r in range(partition.N_FOLDS)]


# ------------------------------------------------------------------ the session guard


class Guard:
    """Seconds of the longest step of each kind, and the deadline it is held to."""

    def __init__(self, deadline: float, margin: float, clock: Callable[[], float] = time.time,
                 prior: Mapping[str, float] | None = None):
        self.deadline, self.margin, self.clock = float(deadline), float(margin), clock
        self.longest: dict[str, float] = dict(prior or {})

    def stop(self, kind: str) -> bool:
        return self.clock() + 1.25 * self.longest.get(kind, 0.0) + self.margin > self.deadline

    def check(self, kind: str) -> None:
        if self.stop(kind):
            raise DeadlineReached(kind)

    def timed(self, kind: str, fn: Callable, *args, **kwargs):
        self.check(kind)
        started = self.clock()
        out = fn(*args, **kwargs)
        self.longest[kind] = max(self.longest.get(kind, 0.0), self.clock() - started)
        return out


# ------------------------------------------------------------------ studies


def study_path(product: Path, t, phase: int, name: str, rotation: int) -> Path:
    return (product / RESULTS / "hpo" / f"{model_store.transfer_slug(t)}__phase{phase}__"
            f"{model_store.name_slug(name)}__outer{int(rotation)}.db")


def study_identity(data: search.SearchData, phase: int, name: str, rotation: int,
                   schedule: Mapping, n_trials: int, space: Mapping, *,
                   phase_one: Mapping | None = None, extra: Mapping | None = None) -> dict:
    folds = data.folds[rotation]
    return search._json({
        "transfer": data.transfer.name, "arm": data.transfer.arm, "phase": phase, "name": name,
        "outerFold": int(rotation), "testFold": int(folds["test_fold"]),
        "innerSplits": folds["inner_splits"], "sampler": "GPSampler",
        "samplerSeed": search.SAMPLER_SEED, "nTrials": int(n_trials), "pruner": "none",
        "searchSeed": SEARCH_SEED, "reportSeeds": list(REPORT_SEEDS), "schedule": dict(schedule),
        "objective": PHASE_ONE_OBJECTIVE if phase == 1 else PHASE_TWO_OBJECTIVE,
        "space": {k: repr(v) for k, v in space.items()},
        partition.HASH_FIELD: data.partition_sha256, "revision": V.REVISION,
        "code_version": V.code_version(),
        **({"phaseOneCheckpoint": phase_one} if phase_one is not None else {}),
        **(dict(extra) if extra else {})})


#: A trial whose training diverges (a non-finite objective or score) is recorded as an
#: Optuna FAIL with its reason and counts against the study's declared trial budget, the
#: same for every arm; the sampler ignores it. v63 does not declare this case.
DIVERGED = (FloatingPointError,)
FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)


def _optimize(study, objective, remaining: int, guard: Guard | None, kind: str) -> None:
    for _ in range(remaining):
        if guard is not None:
            guard.check(kind)
        started = time.time()
        study.optimize(objective, n_trials=1, catch=DIVERGED)
        if guard is not None:
            guard.longest[kind] = max(guard.longest.get(kind, 0.0), time.time() - started)


def _finished(study) -> int:
    return sum(t.state in FINISHED for t in study.trials)


def _open(path: Path, identity: Mapping):
    study, _ = search.open_study(path, identity)
    return study, _finished(study)


def _finite(series, what: str):
    if not all(np.isfinite(series)):
        raise FloatingPointError(f"a non-finite {what} score: {series}")


def phase_one_study(group: str, data: search.SearchData, rotation: int, cfg: Mapping,
                    path: Path, guard: Guard | None = None, device="cpu"):
    schedule, n = cfg["schedule"], cfg["nTrials"]
    cap = search._search_cap(schedule, 1)
    study, done = _open(path, study_identity(data, 1, group, rotation, schedule, n,
                                             V.PHASE_ONE_SPACE))

    def objective(trial):
        params = search.suggest(trial, V.PHASE_ONE_SPACE)
        hyper = V.phase_one_hyper(group, params, schedule)
        with V.phase_one_l2(params["l2"]):
            outs, stopped = search.train_inner_splits(
                lambda i: V.train_phase_one(group, data.inner_view(rotation, i), rotation,
                                            SEARCH_SEED, hyper, device, sigma=False)["result"],
                two_phase.pooled_balanced_accuracy, 1, cap, 1)
        series = search.pooled_balanced_accuracy_series([o["validation"] for o in outs])
        _finite(series, "pooled phase-1")
        best = int(np.argmax(series))
        search._set_attrs(trial, {"pooledBalancedAccuracy": series, "bestEpoch": best,
                                  "epochs": best + 1, "seed": SEARCH_SEED, **stopped,
                                  "trainLoss": search._mean_loss([o["history"] for o in outs])})
        return float(series[best])

    _optimize(study, objective, n - done, guard, f"trial1:{group}")
    if _finished(study) < n:
        raise DeadlineReached(f"phase-1 study {path.name}")
    return study


def source_pass(strategy: str, data, rotation: int, record, device="cpu") -> list[dict]:
    """The centroid rule's source side, through the frozen phase-1 parts; for MIL-CREDA
    with its phase-2 aggregation (Sec. 7.3: switched on for source and target)."""
    state, meta = record
    if strategy == "MIL-CREDA":
        meta = {**meta, "hyper": {**meta["hyper"], **V.PHASE_TWO_AGGREGATION}}
    with V.standardization(V.standard_of_state(state)):
        return search.source_representations(strategy, data, rotation, (state, meta), device)


def phase_two_trainer(strategy: str, gate: float) -> Callable:
    if strategy == "MIL-CREDA":
        return lambda data, r, seed, record, params, schedule, device: \
            V.train_mil_creda_phase_two(data, r, seed, record, params, schedule, device, gate=gate)

    def train(data, r, seed, record, params, schedule, device):
        with V.standardization(V.standard_of_state(record[0]), thread=True):
            return search.train_phase_two(strategy, data, r, seed, record, params, schedule,
                                          device)
    return train


def phase_two_space(strategy: str) -> dict:
    return (dict(V.MIL_CREDA_PHASE_TWO_SPACE) if strategy == "MIL-CREDA"
            else search.phase_two_space(strategy))


def phase_two_study(strategy: str, data, rotation: int, record, cfg: Mapping, path: Path,
                    gate: float, guard: Guard | None = None, device="cpu"):
    schedule, n = cfg["schedule"], cfg["nTrials"]
    cap = search._search_cap(schedule, 2)
    space = phase_two_space(strategy)
    extra = {"shiftGate": gate} if strategy == "MIL-CREDA" else None
    study, done = _open(path, study_identity(
        data, 2, strategy, rotation, schedule, n, space,
        phase_one=search.phase_one_checkpoint_identity(record[1]), extra=extra))
    trainer = phase_two_trainer(strategy, gate)
    sources: list[dict] = []

    def objective(trial):
        if not sources:
            sources.extend(source_pass(strategy, data, rotation, record, device))
        params = search.suggest(trial, space)
        outs, stopped = search.train_inner_splits(
            lambda i: trainer(data.inner_view(rotation, i), rotation, SEARCH_SEED, record,
                              params, schedule, device)["result"],
            lambda recs: two_phase.pooled_centroid_balanced_accuracy(recs, sources),
            2, cap, search.MIN_TRAINED_EPOCHS.get(strategy, 1))
        per_split = [o["validation"] for o in outs]
        primary = search.pooled_centroid_series(per_split, sources)
        _finite(primary, "centroid")
        best = int(np.argmax(primary))
        search._set_attrs(trial, {search.PRIMARY_SERIES[2]: primary, "bestEpoch": best,
                                  "epochs": best + 1, **search.ensv_record(per_split),
                                  "seed": SEARCH_SEED, **stopped,
                                  "trainLoss": search._mean_loss([o["history"] for o in outs])})
        return float(primary[best])

    _optimize(study, objective, n - done, guard, f"trial2:{strategy}")
    if _finished(study) < n:
        raise DeadlineReached(f"phase-2 study {path.name}")
    return study


# ------------------------------------------------------------------ models and records


def model_path(product: Path, t, rotation: int, seed: int, phase: int, name: str,
               rule: str = "primary") -> Path:
    suffix = "" if rule == "primary" else f"__{rule}"
    return (product / MODELS / model_store.transfer_slug(t) / f"rotation{int(rotation)}"
            / f"seed{int(seed)}" / f"phase{phase}__{model_store.name_slug(name)}{suffix}.pt")


def phase_one_meta(group: str, data, rotation: int, seed: int, trained: Mapping,
                   selection: Mapping) -> dict:
    ck = trained["result"]["checkpoint"]
    return {"phase": 1, "name": group, "transfer": data.transfer.name, "rotation": int(rotation),
            "seed": int(seed), partition.HASH_FIELD: data.partition_sha256,
            "channels": trained["channels"], "hyper": dict(trained["hyper"]),
            "sigma": ck.get("sigma") if group == "MIL-CREDA" else None,
            "epochs": int(selection["epochs"]), "selection": dict(selection)}


def phase_one_record(group: str, data, rotation: int, seed: int, selection: Mapping,
                     cfg: Mapping, product: Path, guard: Guard | None, device="cpu"):
    """(state, meta) of the refit at `seed`: loaded when stored, trained and stored else."""
    path = model_path(product, data.transfer, rotation, seed, 1, group)
    if path.is_file():
        state, meta = V.load_state(path, data.partition_sha256)
        meta["state_sha256"] = meta.get("state_sha256")
        return state, meta
    schedule = search.refit_schedule(cfg["schedule"], 1, selection["epochs"])
    hyper = V.phase_one_hyper(group, selection["params"], schedule)
    def run():
        with V.phase_one_l2(selection["params"]["l2"]):
            return V.train_phase_one(group, data, rotation, seed, hyper, device)
    trained = guard.timed("refit1", run) if guard else run()
    meta = phase_one_meta(group, data, rotation, seed, trained, selection)
    state = trained["result"]["checkpoint"]["state"]
    V.save_state(path, state, meta)
    return V.load_state(path, data.partition_sha256)


def rebuild_phase_one(group: str, record, device="cpu"):
    state, meta = record
    with V.standardization(V.standard_of_state(state)):
        model = model_store.build_model(group, meta["channels"]["source"],
                                        meta["channels"]["target"], meta["hyper"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise V.V63Error(f"the stored {group} state does not fit its architecture")
    if group == "MIL-CREDA":
        model.fix_sigma(meta["sigma"])
    return model.to(device).eval()


class Covariates:
    """Age per subject and frontal amplitude per (subject, condition), from the manifest."""

    def __init__(self, processed_root: str | Path | None):
        root = Path(processed_root or bag_data.REPOSITORY / bag_data.PROCESSED_RELATIVE)
        m = pd.read_csv(root / "manifest.csv", low_memory=False)
        self.age = m.groupby("subject").age.first().astype(float).to_dict()
        self.frontal = {(s, c): float(v) for s, c, v in
                        m[["subject", "condition", "frontal_ptp_median"]].itertuples(index=False)}

    def of(self, bags) -> dict:
        return {"ages": [self.age.get(b.subject) for b in bags],
                "frontalAmplitude": [self.frontal.get((b.subject, b.condition)) for b in bags]}


def _rows(model, test: Mapping, target_domain: str, source_model, cov: Covariates,
          device) -> dict:
    tgt, src = test["target"], test["source"]
    return {"subjects": [b.subject for b in tgt], "conditions": [b.condition for b in tgt],
            "labels": [int(b.label) for b in tgt],
            "probabilities": V.probabilities(model, tgt, target_domain, device).tolist(),
            "targetRepresentations": two_phase.represent(model, tgt, target_domain,
                                                         torch.device(device)).tolist(),
            "sourceSubjects": [b.subject for b in src],
            "sourceRepresentations": two_phase.represent(source_model, src, "source",
                                                         torch.device(device)).tolist(),
            **cov.of(tgt)}


def run_record_path(product: Path, t, rotation: int, method: str, seed: int) -> Path:
    return (product / RESULTS / V.experiment_of(t) / "runs" / model_store.transfer_slug(t)
            / f"outer{int(rotation)}" / f"{model_store.name_slug(method)}__seed{int(seed)}.json")


def write_record(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, default=float), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ the units


def run_performance(t, rotation: int, cfg: Mapping, product: Path, bags, prep, cov: Covariates,
                    guard: Guard | None, device="cpu") -> dict:
    data = V.search_data(t, bags, prep, rotations=[rotation])
    test = V.test_bags(t, bags, rotation, prep)
    gate_auc = Fl.shift_gate_auc(t, bags, rotation, prep)
    gate = V.gate_weight(gate_auc)
    base = {"transfer": t.name, "arm": t.arm, "experiment": V.experiment_of(t),
            "rotation": int(rotation), partition.HASH_FIELD: data.partition_sha256,
            "revision": V.REVISION}
    strategies = cfg["strategies"]
    groups = [g for g in GROUPS if g == "common" or any(
        search.PHASE_ONE_GROUP_OF[s] == g for s in strategies)]
    selections, records = {}, {}
    for g in groups:
        study = phase_one_study(g, data, rotation, cfg,
                                study_path(product, t, 1, g, rotation), guard, device)
        selections[g] = search.select(study)
        for seed in cfg["seeds"]:
            records[(g, seed)] = phase_one_record(g, data, rotation, seed, selections[g], cfg,
                                                  product, guard, device)
            method = PREDICT if g == "common" else f"phase1 [{g}]"
            path = run_record_path(product, t, rotation, method, seed)
            if not path.is_file():
                m = rebuild_phase_one(g, records[(g, seed)], device)
                write_record(path, {**base, "method": method, "seed": seed, "phase": 1,
                                    "group": g, "selection": selections[g],
                                    **_rows(m, test, "source", m, cov, device)})
    first = cfg["seeds"][0]
    for s in strategies:
        g = search.PHASE_ONE_GROUP_OF[s]
        study = phase_two_study(s, data, rotation, records[(g, first)], cfg,
                                study_path(product, t, 2, s, rotation), gate, guard, device)
        if not any(tr.state == optuna.trial.TrialState.COMPLETE for tr in study.trials):
            reason = f"every one of the {len(study.trials)} trials of {study.study_name} diverged"
            for seed in cfg["seeds"]:
                path = run_record_path(product, t, rotation, s, seed)
                if not path.is_file():
                    write_record(path, {**base, "method": s, "seed": seed, "phase": 2,
                                        "group": g, "failed": reason})
            continue
        chosen = {"primary": search.select(study, "primary")}
        ensv = search.select(study, "ensv")
        if cfg["ensvRefits"] and (ensv["trial"], ensv["epoch"]) != (
                chosen["primary"]["trial"], chosen["primary"]["epoch"]):
            chosen["ensv"] = ensv
        trainer = phase_two_trainer(s, gate)
        for rule, sel in chosen.items():
            method = s if rule == "primary" else f"{s} [EnsV]"
            for seed in cfg["seeds"]:
                path = run_record_path(product, t, rotation, method, seed)
                if path.is_file():
                    continue
                record = records[(g, seed)]
                schedule = search.refit_schedule(cfg["schedule"], 2, sel["epochs"])
                saved, search.PHASE_ONE_SEED = search.PHASE_ONE_SEED, seed
                try:
                    run = lambda: trainer(data, rotation, seed, record, sel["params"], schedule,
                                          device)
                    out = guard.timed("refit2", run) if guard else run()
                except DIVERGED as error:          # recorded, never hidden nor retried
                    write_record(path, {**base, "method": method, "seed": seed, "phase": 2,
                                        "group": g, "rule": rule, "selection": sel,
                                        "failed": f"{type(error).__name__}: {error}"})
                    continue
                finally:
                    search.PHASE_ONE_SEED = saved
                V.save_state(model_path(product, t, rotation, seed, 2, s, rule),
                             out["model"].state_dict(),
                             {**base, "phase": 2, "name": s, "seed": seed, "rule": rule,
                              "selection": sel, "hyper": out["result"]["checkpoint"]["hyper"],
                              "channels": out["channels"], "shiftGate": gate})
                src_model = rebuild_phase_one(g, record, device)
                write_record(path, {**base, "method": method, "seed": seed, "phase": 2,
                                    "group": g, "rule": rule, "selection": sel,
                                    "primarySelection": chosen["primary"],
                                    "ensvSelection": ensv, "shiftGateAuc": gate_auc,
                                    "shiftGate": gate if s == "MIL-CREDA" else None,
                                    **_rows(out["model"], test, "target", src_model, cov, device)})
    return {"unit": f"P:{t.name}:{rotation}", "shiftGateAuc": gate_auc, "shiftGate": gate}


def run_target_filter(domain: str, cfg: Mapping, product: Path, bags, prep,
                      guard: Guard | None, device="cpu") -> dict:
    out_dir = product / RESULTS / "S0-codificador"
    rows = []
    for r in range(partition.N_FOLDS):
        data, test = Fl.own_domain_data(domain, bags, r, prep)
        path = study_path(product, data.transfer, 1, Fl.FILTER_GROUP, r)
        study = phase_one_study(Fl.FILTER_GROUP, data, r, cfg, path, guard, device)
        sel = search.select(study)
        for seed in cfg["seeds"]:
            rec_path = out_dir / "runs" / domain / f"outer{r}__seed{seed}.json"
            if not rec_path.is_file():
                record = phase_one_record(Fl.FILTER_GROUP, data, r, seed, sel, cfg, product,
                                          guard, device)
                m = rebuild_phase_one(Fl.FILTER_GROUP, record, device)
                p = V.probabilities(m, test, "source", device).tolist()
                write_record(rec_path, {"domain": domain, "rotation": r, "seed": seed,
                                        "selection": sel, "subjects": [b.subject for b in test],
                                        "labels": [int(b.label) for b in test],
                                        "probabilities": p, "revision": V.REVISION,
                                        partition.HASH_FIELD: data.partition_sha256})
            rec = json.loads(rec_path.read_text())
            rows += [{"subject": s, "label": y, "seed": seed, "probability": q}
                     for s, y, q in zip(rec["subjects"], rec["labels"], rec["probabilities"])]
    verdict = {"domain": domain, "revision": V.REVISION, "experiment": "S0-codificador",
               **Fl.target_filter_verdict(rows, cfg["permutations"])}
    write_record(out_dir / f"target_filter_{domain}.json", verdict)
    return verdict


def run_shift_filter(t, cfg: Mapping, product: Path, bags, prep, guard: Guard | None) -> dict:
    out_dir = product / RESULTS / "F-corrimiento"
    state_path = out_dir / "runs" / f"{model_store.transfer_slug(t)}.state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    last = {"t": time.time()}

    def should_stop() -> bool:
        if time.time() - last["t"] > 60:            # persist the null drawn so far
            write_record(state_path, state)
            last["t"] = time.time()
        return guard is not None and guard.stop("permutation")

    started = time.time()
    out = Fl.shift_filter(t, bags, cfg["permutations"], prep=prep, should_stop=should_stop,
                          state=state)
    write_record(state_path, state)
    if guard is not None and state.get("null"):
        guard.longest["permutation"] = max(guard.longest.get("permutation", 0.0),
                                           (time.time() - started) / max(1, len(state["null"])))
    if not out["complete"]:
        raise DeadlineReached(f"shift filter of {t.name} at {out['drawn']} permutations")
    out = {**out, "experiment": "F-corrimiento", "revision": V.REVISION}
    write_record(out_dir / f"shift_{model_store.transfer_slug(t)}.json", out)
    return out


# ------------------------------------------------------------------ entry points


def _setup(config: Mapping, environ: Mapping[str, str]):
    cfg = check_config(remote_units.resolve_inputs(config, environ=environ))
    torch.set_num_threads(cfg["threads"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    campaign.register(cfg["strategies"])
    return cfg


def run(config: Mapping, product: str | Path | None = None, *,
        units: Sequence[str] | None = None, workdir: str | Path | None = None,
        environ: Mapping[str, str] | None = None, clock: Callable[[], float] = time.time,
        deadline: float | None = None) -> dict:
    """Run `units` (default `FORGE_RUN_UNITS`) in order until the session deadline."""
    environ = os.environ if environ is None else environ
    product = Path(product) if product else DEFAULT_PRODUCT
    workdir = Path(workdir) if workdir else Path.cwd()
    chosen = requested_units(units, environ)
    cfg = _setup(config, environ)
    if deadline is None:
        deadline = float(environ.get(remote_units.DEADLINE_ENV, "inf"))
    guard = Guard(deadline, cfg["marginSeconds"], clock, cfg.get("stepSeconds"))
    marker = workdir / remote_units.PARTIAL_MARKER
    marker.unlink(missing_ok=True)
    bags = campaign.load_bags(cfg)
    prep = V.Prepared(bags)
    cov = Covariates(cfg.get("processedRoot"))
    ran, results, started = [], {}, time.time()
    with V.installed():
        for k, uid in enumerate(chosen):
            unit = parse_unit(uid)
            try:
                if unit[0] == "S0":
                    results[uid] = run_target_filter(unit[1], cfg, product, bags, prep, guard,
                                                     cfg["device"])
                elif unit[0] == "F":
                    results[uid] = run_shift_filter(unit[1], cfg, product, bags, prep, guard)
                else:
                    results[uid] = run_performance(unit[1], unit[2], cfg, product, bags, prep,
                                                   cov, guard, cfg["device"])
            except DeadlineReached as reached:
                remaining = chosen[k:]
                workdir.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"remaining": remaining,
                                              "progress": {uid: str(reached)}}),
                                  encoding="utf-8")
                return {"ran": ran, "remaining": remaining, "partial": str(marker),
                        "stepSeconds": guard.longest, "seconds": time.time() - started}
            ran.append(uid)
    return {"ran": ran, "remaining": [], "partial": None, "results": results,
            "stepSeconds": guard.longest, "seconds": time.time() - started}


def smoke(config: Mapping, product: str | Path | None = None, *,
          units: Sequence[str] | None = None, workdir: str | Path | None = None,
          environ: Mapping[str, str] | None = None, budget_seconds: float | None = None) -> dict:
    """The rehearsal and timing callable: `run` under the runner, with `shard.json` (the
    evidence `smoke record` reads) and the measured step seconds."""
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    deadline = float(environ.get(remote_units.DEADLINE_ENV, "inf"))
    if budget_seconds is not None:
        deadline = min(deadline, time.time() + float(budget_seconds))
    cfg0 = check_config(remote_units.resolve_inputs(config, environ=environ))
    placement = remote_units.device_placement(cfg0, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "codeVersion": V.code_version(),
                "revision": V.REVISION}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    outcome = run(config, product, units=units, workdir=workdir, environ=environ,
                  deadline=deadline)
    chosen = requested_units(units, environ)
    timing = {"stepSeconds": outcome["stepSeconds"], "seconds": outcome["seconds"],
              "device": environment.get("device"), "threads": torch.get_num_threads(),
              "cpuCount": os.cpu_count()}
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "v63-timing.json").write_text(json.dumps(timing, indent=1, default=str),
                                             encoding="utf-8")
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment.get("commit"),
                                "sourcesDigest": identity["codeVersion"],
                                "codeVersion": identity["codeVersion"], "unit": chosen[0],
                                "units": chosen, "complete": not outcome["remaining"],
                                "estimates": {}},
                   "environment": environment, "timing": timing,
                   "runs": {k: v for k, v in outcome.items() if k != "results"}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {k: v for k, v in outcome.items() if k != "results"} | {"timing": timing}
