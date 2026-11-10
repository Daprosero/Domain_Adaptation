"""Pilot: does a hyperparameter search let MIL-CREDA beat predict on balanced accuracy?

Diagnostic outside any plan, on L11->Robots, where every fixed-value run on the stored
windows predicted one class for every subject (balanced accuracy 0.500). Conditions are the
ones settled in experiments v64 Sec. 10.4 plus the operator's decisions on its open points
1, 2 and 5: EEGNet as defined, stored windows (no 1-45 Hz / CAR), the "blocks" data arm,
none of the six v63 fixes, phase 2 training the target trunk and the attention under the
full Eq. 39 (theta_s and the head frozen), L2-SP mu 10, gate 1.

Per outer rotation, with one inner split and the same budget for both arms:
  phase 1, predict (common group) and MIL-CREDA: `N_TRIALS` trials over lr1, l2 and the
    attention temperature, scored by the inner source-validation balanced accuracy;
  phase 2, MIL-CREDA: `N_TRIALS` trials over lr2, wd2, the phase-2 attention temperature,
    tau_loc and gamma in [0.1, 1], scored by the label-free centroid rule.
Each phase picks its best trial and epoch; both arms are then refitted on the rotation's
training view for every seed and scored on its target test fold. A unit id is
"<transfer>:<rotation>".
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
from optuna.distributions import FloatDistribution

from MIL_CREDA_EEG import ablation_pilot as AP, remote_units, search, two_phase
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import v64_pilot as P

RESULTS = Path("Results") / "search-pilot"
N_TRIALS = 10
N_INNER = 1
SAMPLER_SEED = 20261110
SEARCH_SEED = search.SEARCH_SEED
PHASE_ONE_SPACE = {"lr1": FloatDistribution(1e-4, 1e-1, log=True),
                   "l2": FloatDistribution(1e-4, 1e-1, log=True),
                   "tauAtt": FloatDistribution(0.1, 10.0, log=True)}
PHASE_TWO_SPACE = {"lr2": FloatDistribution(1e-5, 1e-2, log=True),
                   "wd2": FloatDistribution(1e-7, 1e-2, log=True),
                   "tauAtt": FloatDistribution(0.1, 10.0, log=True),
                   "tauLoc": FloatDistribution(0.1, 10.0, log=True),
                   "gamma": FloatDistribution(0.1, 1.0)}
GROUPS = ("common", "MIL-CREDA")


@contextlib.contextmanager
def settled_rules():
    """One inner split; none of the v63 fixes (spectral standardization a no-op)."""
    saved = search.N_INNER
    search.N_INNER = N_INNER
    try:
        with AP.fix_rules(frozenset()):
            yield
    finally:
        search.N_INNER = saved


@contextlib.contextmanager
def phase_two_values(params: Mapping):
    """MIL-CREDA's phase-2 aggregation and tau_loc for one trial or refit."""
    saved = V.PHASE_TWO_AGGREGATION, V.UNUSED_TAU_LOCAL
    V.PHASE_TWO_AGGREGATION = {"attentionGamma": float(params["gamma"]),
                               "attentionTemperature": float(params["tauAtt"])}
    V.UNUSED_TAU_LOCAL = float(params["tauLoc"])
    try:
        with AP.attention_rules():
            yield
    finally:
        V.PHASE_TWO_AGGREGATION, V.UNUSED_TAU_LOCAL = saved


def phase_one_hyper(group: str, params: Mapping, sched: Mapping, epochs1: int | None = None):
    hyper = V.phase_one_hyper(group, {"lr1": params["lr1"], "l2": params["l2"]},
                              {**sched, "epochs1": epochs1 or sched["epochs1"]})
    hyper["attentionTemperature"] = float(params["tauAtt"])
    return hyper


def _study(seed: int) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=seed))


def _trials(study: optuna.Study) -> list[dict]:
    return [{"number": t.number, "value": t.value, "params": t.params,
             "bestEpoch": t.user_attrs.get("bestEpoch"), "state": t.state.name}
            for t in study.trials]


def search_phase_one(group: str, data, rotation: int, sched: Mapping, device) -> dict:
    cap = search._search_cap(sched, 1)
    study = _study(SAMPLER_SEED + rotation)

    def objective(trial):
        params = search.suggest(trial, PHASE_ONE_SPACE)
        hyper = phase_one_hyper(group, params, sched)
        with V.phase_one_l2(params["l2"]):
            outs, _ = search.train_inner_splits(
                lambda i: V.train_phase_one(group, data.inner_view(rotation, i), rotation,
                                            SEARCH_SEED, hyper, device, sigma=False)["result"],
                two_phase.pooled_balanced_accuracy, 1, cap, 1)
        series = search.pooled_balanced_accuracy_series([o["validation"] for o in outs])
        best = int(np.nanargmax(series))
        trial.set_user_attr("bestEpoch", best)
        return float(series[best])

    study.optimize(objective, n_trials=N_TRIALS, catch=(FloatingPointError, V.V63Error))
    best = study.best_trial
    return {"params": best.params, "epochs": best.user_attrs["bestEpoch"] + 1,
            "value": best.value, "trials": _trials(study)}


def refit_phase_one(group: str, data, rotation: int, seed: int, chosen: Mapping,
                    sched: Mapping, device):
    hyper = phase_one_hyper(group, chosen["params"], sched, chosen["epochs"])
    with V.phase_one_l2(chosen["params"]["l2"]):
        model = V.train_phase_one(group, data, rotation, seed, hyper, device)["model"]
    meta = {"hyper": dict(hyper)}
    if group == "MIL-CREDA":
        meta["sigma"] = model._sigma if getattr(model, "_sigma", None) else \
            V.between_bag_sigma(model, data.rotations[rotation]["source_train"], seed=seed)
        model._sigma = meta["sigma"]
    return model, (two_phase.phase_one_state(model), meta)


def _phase_two(data, rotation: int, seed: int, record, params: Mapping, sched: Mapping, device):
    saved, search.PHASE_ONE_SEED = search.PHASE_ONE_SEED, seed
    try:
        with phase_two_values(params):
            return V.train_mil_creda_phase_two(
                data, rotation, seed, record, {"lr2": params["lr2"], "wd2": params["wd2"]},
                sched, device, gate=1.0, gamma=float(params["gamma"]))
    finally:
        search.PHASE_ONE_SEED = saved


def search_phase_two(data, rotation: int, record, sched: Mapping, device) -> dict:
    cap = search._search_cap(sched, 2)
    study = _study(SAMPLER_SEED + 100 + rotation)

    def objective(trial):
        params = search.suggest(trial, PHASE_TWO_SPACE)
        with phase_two_values(params):
            sources = C.source_pass("MIL-CREDA", data, rotation, record, device)
        outs, _ = search.train_inner_splits(
            lambda i: _phase_two(data.inner_view(rotation, i), rotation, SEARCH_SEED, record,
                                 params, sched, device)["result"],
            lambda recs: two_phase.pooled_centroid_balanced_accuracy(recs, sources),
            2, cap, 1)
        primary = search.pooled_centroid_series([o["validation"] for o in outs], sources)
        best = int(np.nanargmax(primary))
        trial.set_user_attr("bestEpoch", best)
        return float(primary[best])

    study.optimize(objective, n_trials=N_TRIALS, catch=(FloatingPointError, V.V63Error))
    best = study.best_trial
    return {"params": best.params, "epochs": best.user_attrs["bestEpoch"] + 1,
            "value": best.value, "trials": _trials(study)}


def run_unit(t, rotation: int, seeds: Sequence[int], bags, prep, device, epochs1: int,
             epochs2: int) -> dict:
    started = time.time()
    sched = {"epochs1": epochs1, "epochs2": epochs2, "bagsPerStep": P.BAGS_PER_STEP}
    data = V.search_data(t, bags, prep, rotations=[rotation])
    test = V.test_bags(t, bags, rotation, prep)["target"]
    out = {"transfer": t.name, "rotation": rotation, "arm": "blocks", "stored": True,
           "nTrials": N_TRIALS, "nInner": N_INNER, "epochCaps": [epochs1, epochs2], "mu": V.MU,
           "nUnits": {"source": len(data.rotations[rotation]["source_train"]),
                      "target": len(data.rotations[rotation]["target_train"]),
                      "test": len(test)}, "seeds": {}}
    with settled_rules():
        chosen1 = {g: search_phase_one(g, data, rotation, sched, device) for g in GROUPS}
        out["phase1"] = chosen1
        print(rotation, "phase 1 chosen", {g: (c["params"], c["epochs"], round(c["value"], 3))
                                           for g, c in chosen1.items()}, flush=True)
        _, search_record = refit_phase_one("MIL-CREDA", data, rotation, seeds[0],
                                           chosen1["MIL-CREDA"], sched, device)
        chosen2 = search_phase_two(data, rotation, search_record, sched, device)
        out["phase2"] = chosen2
        print(rotation, "phase 2 chosen", chosen2["params"], chosen2["epochs"],
              round(chosen2["value"], 3), flush=True)
        sched2 = {**sched, "epochs2": chosen2["epochs"]}
        for seed in seeds:
            rows = {}
            predict, _ = refit_phase_one("common", data, rotation, seed, chosen1["common"],
                                         sched, device)
            rows["predict"] = P._rows(predict.eval(), test, "source", device)
            mc1, record = refit_phase_one("MIL-CREDA", data, rotation, seed,
                                          chosen1["MIL-CREDA"], sched, device)
            rows["phase1"] = P._rows(mc1.eval(), test, "source", device)
            o2 = _phase_two(data, rotation, seed, record, chosen2["params"], sched2, device)
            rows["milCreda"] = P._rows(o2["model"].eval(), test, "target", device)
            out["seeds"][str(seed)] = rows
            print(rotation, "seed", seed, "done", round(time.time() - started), "s", flush=True)
    out["seconds"] = round(time.time() - started, 1)
    return out


def run(config: Mapping, product: str | Path | None = None, *, units: Sequence[str],
        seeds: Sequence[int] = (1, 2, 3), epochs1: int = P.EPOCHS1, epochs2: int = P.EPOCHS2,
        workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    product = Path(product) if product else C.DEFAULT_PRODUCT
    workdir = Path(workdir) if workdir else Path.cwd()
    config = {"schedule": {"epochs1": epochs1, "epochs2": epochs2,
                           "bagsPerStep": P.BAGS_PER_STEP}, **config}
    cfg = C._setup(config, environ)
    placement = remote_units.device_placement(cfg, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "codeVersion": V.code_version(),
                "revision": "search-pilot"}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    bags = C.campaign.load_bags(cfg)
    prep = P.BlockPrepared(bags, filtered=False)
    ran, started = [], time.time()
    for u in units:
        name, rotation = u.rsplit(":", 1)
        t, rotation = V.transfer(name), int(rotation)
        path = product / RESULTS / t.name.replace("->", "-to-") / f"rotation{rotation}.json"
        if path.is_file():
            continue
        out = run_unit(t, rotation, seeds, bags, prep, cfg["device"], epochs1, epochs2)
        C.write_record(path, {**out, "environment": environment})
        ran.append(u)
    remaining = [u for u in units if u not in ran and not (
        product / RESULTS / V.transfer(u.rsplit(":", 1)[0]).name.replace("->", "-to-")
        / f"rotation{int(u.rsplit(':', 1)[1])}.json").is_file()]
    timing = {"seconds": round(time.time() - started, 1), "device": environment.get("device")}
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        workdir.mkdir(parents=True, exist_ok=True)
        payload = {"mode": mode,
                   "evidence": {"commit": environment.get("commit"),
                                "sourcesDigest": identity["codeVersion"],
                                "codeVersion": identity["codeVersion"], "unit": units[0],
                                "units": list(units), "complete": not remaining,
                                "estimates": {}},
                   "environment": environment, "timing": timing,
                   "runs": {"ran": ran, "skipped": [], "remaining": remaining}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {"ran": ran, "remaining": remaining, "timing": timing}
