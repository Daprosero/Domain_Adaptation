"""Oracle screening of the phase-2 search spaces on the synthetic development units.

One notebook = one development unit, one seed and a group of methods. Phase 1 is NOT fitted here:
the notebook loads the precomputed base of that unit and seed (`Manifests/oracle_bases/`, built by
`xcompare.shared_base` at the unit's recorded phase-1 settings), so every method of every notebook
starts phase 2 from the same bytes. For each method one TPE study runs over its FULL phase-2 space,
epochs2 included (MIL-CREDA also searches wd2, gamma and tau_att; nothing is held fixed in advance),
and maximises the balanced accuracy at the seed's cut on a large evaluation pool of fresh target
subjects drawn from the unit's own target domain (same device, shift and class effect, another seed,
unpaired). That objective reads TRUE target labels, which only the simulator has: the screening
decides which parameters matter, not any reported result. Each trial also records the BA on the
unit's 12 target-test subjects, the AUC on the pool and the label-free centroid and cef scores.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import optuna
import torch

from MIL_CREDA_EEG import campaign, phase2_select, remote_units, search, spectral_trunk
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

BASES = Path(__file__).resolve().parents[2] / "MIL-CREDA-EEG" / "Manifests" / "oracle_bases"
#: the evaluation pool: fresh subjects of the unit's target domain, never seen by any trial's training
EVAL_SUBJECTS = 120
EVAL_SEED_OFFSET = 101
EPOCHS2 = (10, 20, 40)
#: MIL-CREDA's space (protocol v81): the comparators' shared lr2 / wd2 ranges; each loss weight in [0.1, 1],
#: since the three loss terms lie in [0, 1]; tau_local from a hard match to a near-uniform one on the measured
#: kappa spread of both datasets; tau_att over the attention's working range; gamma in [0, 1]
MIL_SPACE = {"lr2": ("log", (1e-4, 1e-2)), "wd2": ("log", (1e-7, 1e-2)),
             "lambda_global": ("log", (0.1, 1.0)), "lambda_local": ("log", (0.1, 1.0)),
             "tau_local": ("log", (1e-3, 0.3)), "gamma": ("lin", (0.0, 1.0)),
             "tau_att": ("log", (0.01, 1.0))}
METHODS = ("MIL-CREDA", *[m for m in XC.ALL8 if m not in XC.SKIPPED])


def eval_pool(rec: dict, effect: float, n: int = EVAL_SUBJECTS) -> list:
    """`n` new subjects of the unit's target domain (device setting only): the device of
    `xpilot.build_sides`, the same class effect and options, another seed and no source pair."""
    from MIL_CREDA_EEG import eeg_sim as E
    u = rec["unitKwargs"]
    if u.get("setting") != "device":
        raise ValueError("the oracle screening covers the device-shift development units only")
    sim_seed = int(u.get("sim_seed", 0))
    opts = {k: float(u.get(k, 0.0)) for k in ("artifact_rate", "artifact_class_corr", "event_fraction")}
    return E.domain("TGX", int(n), E.device("emotiv", float(u["shift"]), 11 + sim_seed), effect,
                    seed=sim_seed + EVAL_SEED_OFFSET, spread=float(u.get("spread", 1.0)), **opts)


def _suggest_mil(trial) -> dict:
    return {k: (trial.suggest_float(k, lo, hi, log=(kind == "log"))) for k, (kind, (lo, hi)) in MIL_SPACE.items()}


def _params(method, trial) -> tuple[dict, int]:
    """The trial's phase-2 settings and epoch count, suggested in a fixed order."""
    e2 = int(trial.suggest_categorical("epochs2", list(EPOCHS2)))
    return (_suggest_mil(trial) if method == "MIL-CREDA" else
            search.suggest(trial, search.phase_two_space(method))), e2


def _train(method, p2, e2, b, sides, cfg, dev, seed, rotation, name):
    if method == "MIL-CREDA":
        p2 = dict(p2)
        wd2, saved = p2.pop("wd2"), XP.P.PHASE_TWO["wd2"]
        XP.P.PHASE_TWO["wd2"] = float(wd2)             # phase2_settings leaves wd2 at its default
        try:
            return XP.phase2_trial(b["base"], sides, cfg, dev, seed, b["phase1"], {**p2, "epochs2": e2})["model"]
        finally:
            XP.P.PHASE_TWO["wd2"] = saved
    with XC.common_wrapper(b["context"]["width"], b["context"]["temperature"]):
        return XC.adapt(method, b["prepared"], b["context"], source=sides["source"], target=sides["target"],
                        params=p2, seed=seed, rotation=rotation, unit=name, epochs2=e2, cfg=cfg, device=dev)


def run(unit_key: str, seed: int, trials: int = 80, methods: Sequence[str] = METHODS,
        eval_subjects: int = EVAL_SUBJECTS, workdir=None, bases=None) -> dict:
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)))
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"methods are among {list(METHODS)}; got {unknown}")
    with spectral_trunk.installed():
        b = torch.load(Path(bases or BASES) / f"{unit_key}-s{int(seed)}.pt", weights_only=False)
    cfg = remote_units.resolve_inputs(campaign.check_config(b["config"]), environ={})
    built = XP.build_sides(cfg, {}, experiment="sim", rotation=b["rotation"], half=0, **b["unitKwargs"])
    sides, name, cfg = built["sides"], built["unit"], built["cfg"]
    if name != b["unit"]:
        raise ValueError(f"base {b['unit']} does not match the rebuilt unit {name}")
    pool = eval_pool(b, built["effect"], eval_subjects)
    campaign.register([m for m in methods if m != "MIL-CREDA"])
    dev, rot, cut = torch.device("cpu"), int(b["rotation"]), float(b["cut"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_path = workdir / f"oracle-{unit_key}-s{int(seed)}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "unitKey": unit_key, "unit": name, "seed": int(seed), "cut": cut, "baseCommit": b["commit"],
        "evalSubjects": len(pool), "objective": "BA at the seed's cut on the evaluation pool (true labels)",
        "methods": {}}
    with spectral_trunk.installed(), XP.window_samples(built["samples"]), XP.full_trace_mode(False):
        for method in methods:
            rows = out["methods"].setdefault(method, {"trials": []})["trials"]
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
                seed=XP.SAMPLER_SEED + rot + 1000 * int(seed)))
            for r in rows:                      # resume: replay the finished trials into the sampler
                study.enqueue_trial(r["params"])
            replay = len(rows)

            def objective(trial):
                p2, e2 = _params(method, trial)
                if trial.number < replay:       # an enqueued finished trial: same params, stored value
                    return rows[trial.number]["value"]
                t0 = time.time()
                try:
                    model = _train(method, p2, e2, b, sides, cfg, dev, int(seed), rot, name)
                except FloatingPointError:      # the trainer's own non-finite refusal
                    rows.append({"params": trial.params, "value": 0.0, "nonFinite": True})
                    return 0.0
                wrap = (XC.common_wrapper(b["context"]["width"], b["context"]["temperature"])
                        if method != "MIL-CREDA" else torch.no_grad())
                with wrap:
                    on_pool = XC.snapshot(model, pool, cut, dev)
                    on_test = XC.snapshot(model, sides["target_test"], cut, dev)
                    pseudo = XC.two_phase.evaluate(model, sides["target"], "target", dev).argmax(1).cpu().numpy()
                    centroid = XC.selection_score(model, sides["target"], sides["source_test"], dev)
                value = float(on_pool["BA"]) if on_pool["BA"] is not None else 0.0
                rows.append({"params": trial.params, "value": value, "AUCpool": on_pool["AUC"],
                             "BAtest12": on_test["BA"], "centroid": float(centroid),
                             "cef": float(phase2_select.cef_score(pseudo)), "seconds": round(time.time() - t0, 1)})
                out_path.write_text(json.dumps(out))
                return value

            study.optimize(objective, n_trials=int(trials))
            print(unit_key, seed, method, "best", round(study.best_value, 4), flush=True)
    out_path.write_text(json.dumps(out))
    return {"file": out_path.name, "methods": {m: len(v["trials"]) for m, v in out["methods"].items()}}
