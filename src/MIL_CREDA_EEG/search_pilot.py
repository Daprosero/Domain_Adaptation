"""Pilot: does a hyperparameter search let MIL-CREDA beat predict on balanced accuracy?

Diagnostic outside any plan, on L11->Robots, where every fixed-value run on the stored
windows predicted one class for every subject (balanced accuracy 0.500). Conditions are the
ones settled in experiments v64 Sec. 10.4 plus the operator's decisions on its open points
1, 2 and 5: EEGNet as defined, stored windows (no 1-45 Hz / CAR), the "blocks" data arm,
none of the six v63 fixes, phase 2 training the target trunk and the attention under the
full Eq. 39 (theta_s and the head frozen), L2-SP mu 10, gate 1.

Per outer rotation, with one inner split:
  phase 1, ONE base model shared by predict and MIL-CREDA (neutral aggregation, gamma 0 and
    attention temperature 10, fixed): `N_TRIALS_PHASE_ONE` trials over lr1, l2 and the EEGNet
    setting (F1, D with F2 = F1 x D, temporal kernel length, dropout), scored by the inner
    source-validation balanced accuracy; refitted per seed, it is predict (scored through
    the source trunk) and the start of MIL-CREDA's phase 2;
  phase 2, MIL-CREDA: `N_TRIALS` trials over lr2, wd2, the phase-2 attention temperature,
    tau_loc and gamma in [0.1, 1], scored by the label-free centroid rule, from the seed-1
    base model; then refitted per seed from that seed's base model and scored on the
    rotation's target test fold. A unit id is
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
from optuna.distributions import CategoricalDistribution, FloatDistribution

from MIL_CREDA_EEG import ablation_pilot as AP, eegnet, remote_units, search, two_phase
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_setup as V
from MIL_CREDA_EEG import v64_pilot as P

RESULTS = Path("Results") / "search-pilot"
N_TRIALS = 10                # phase 2
N_TRIALS_PHASE_ONE = 20      # phase 1 also searches the EEGNet configuration
N_INNER = 1
SAMPLER_SEED = 20261110
SEARCH_SEED = search.SEARCH_SEED
PHASE_ONE_SPACE = {"lr1": FloatDistribution(1e-4, 1e-2, log=True),
                   "l2": FloatDistribution(1e-4, 1e-1, log=True),
                   "f1": CategoricalDistribution((4, 8, 16)),
                   "d": CategoricalDistribution((1, 2, 4)),
                   "kernelLength": CategoricalDistribution((32, 64, 128)),
                   "dropout": CategoricalDistribution((0.25, 0.5))}
ARCHITECTURE = ("f1", "d", "kernelLength", "dropout")


def architecture(params: Mapping):
    """EEGNet with the searched setting (F2 = F1 x D); pooling and separable kernel fixed."""
    return eegnet.configured(int(params["f1"]), int(params["d"]), int(params["kernelLength"]),
                             float(params["dropout"]))
PHASE_TWO_SPACE = {"lr2": FloatDistribution(1e-5, 1e-2, log=True),
                   "wd2": FloatDistribution(1e-7, 1e-2, log=True),
                   "tauAtt": FloatDistribution(0.1, 10.0, log=True),
                   "tauLoc": FloatDistribution(0.1, 10.0, log=True),
                   "gamma": FloatDistribution(0.1, 1.0)}
BASE = "MIL-CREDA"          # the shared phase-1 model, at the neutral aggregation


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
    return V.phase_one_hyper(group, {"lr1": params["lr1"], "l2": params["l2"]},
                             {**sched, "epochs1": epochs1 or sched["epochs1"]})


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
        models = []

        def train(i):
            trained = V.train_phase_one(group, data.inner_view(rotation, i), rotation,
                                        SEARCH_SEED, hyper, device, sigma=False)
            models.append((i, trained["model"]))
            return trained["result"]

        with V.phase_one_l2(params["l2"]), architecture(params):
            outs, _ = search.train_inner_splits(train, two_phase.pooled_balanced_accuracy, 1,
                                                cap, 1)
            for i, model in models:      # an embedding collapse (sigma 0) fails the trial
                V.between_bag_sigma(model, data.inner_view(rotation, i).rotations[rotation]
                                    ["source_train"], seed=SEARCH_SEED)
        series = search.pooled_balanced_accuracy_series([o["validation"] for o in outs])
        best = int(np.nanargmax(series))
        trial.set_user_attr("bestEpoch", best)
        return float(series[best])

    study.optimize(objective, n_trials=N_TRIALS_PHASE_ONE,
                   catch=(FloatingPointError, V.V63Error))
    best = study.best_trial
    return {"params": best.params, "epochs": best.user_attrs["bestEpoch"] + 1,
            "value": best.value, "trials": _trials(study)}


def refit_phase_one(group: str, data, rotation: int, seed: int, chosen: Mapping,
                    sched: Mapping, device):
    """The refitted base model and its phase-2 record; the record is None when the trunk's
    embeddings collapsed (between-bag sigma 0), which fails that seed's phase 2 only."""
    hyper = phase_one_hyper(group, chosen["params"], sched, chosen["epochs"])
    with V.phase_one_l2(chosen["params"]["l2"]):
        model = V.train_phase_one(group, data, rotation, seed, hyper, device,
                                  sigma=False)["model"]
    try:
        sigma = V.between_bag_sigma(model, data.rotations[rotation]["source_train"], seed=seed)
    except V.V63Error:
        return model, None
    model._sigma = sigma
    return model, (two_phase.phase_one_state(model), {"hyper": dict(hyper), "sigma": sigma})


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
           "nTrials": [N_TRIALS_PHASE_ONE, N_TRIALS], "nInner": N_INNER, "epochCaps": [epochs1, epochs2], "mu": V.MU,
           "nUnits": {"source": len(data.rotations[rotation]["source_train"]),
                      "target": len(data.rotations[rotation]["target_train"]),
                      "test": len(test)}, "seeds": {}}
    with settled_rules():
        chosen1 = search_phase_one(BASE, data, rotation, sched, device)
    with settled_rules(), architecture(chosen1["params"]):
        out["phase1"] = chosen1
        print(rotation, "phase 1 chosen", chosen1["params"], chosen1["epochs"],
              round(chosen1["value"], 3), flush=True)
        base = {seed: refit_phase_one(BASE, data, rotation, seed, chosen1, sched, device)
                for seed in seeds}
        for seed, (model, _) in base.items():
            out["seeds"][str(seed)] = {"predict": P._rows(model.eval(), test, "source", device)}
        # Operator's gate: phase 2 runs only when some phase-1 network beats 0.5 balanced
        # accuracy on the inner source validation; otherwise the run stops here.
        out["phaseTwoGate"] = {"bestPhaseOneBA": chosen1["value"], "passes": chosen1["value"] > 0.5}
        if not out["phaseTwoGate"]["passes"]:
            out["seconds"] = round(time.time() - started, 1)
            return out
        usable = [seed for seed in seeds if base[seed][1] is not None]
        out["collapsedSeeds"] = [seed for seed in seeds if base[seed][1] is None]
        if not usable:
            out["seconds"] = round(time.time() - started, 1)
            return out
        chosen2 = search_phase_two(data, rotation, base[usable[0]][1], sched, device)
        out["phase2"] = chosen2
        print(rotation, "phase 2 chosen", chosen2["params"], chosen2["epochs"],
              round(chosen2["value"], 3), flush=True)
        sched2 = {**sched, "epochs2": chosen2["epochs"]}
        for seed in usable:
            o2 = _phase_two(data, rotation, seed, base[seed][1], chosen2["params"], sched2,
                            device)
            out["seeds"][str(seed)]["milCreda"] = P._rows(o2["model"].eval(), test, "target",
                                                          device)
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
