"""Oracle diagnostic for the comparators (not a method step, never selects).

Replays each comparator's phase-2 search of one campaign unit exactly as `xcompare.search_phase2`
runs it (same shared base of seed SEARCH_SEED, same cut, same TPE sampler and seed, same score
returned to the sampler, so the same trials) and records per trial the label-free centroid score
and the target-test BA/AUC at the cut. Target labels are read ONLY to evaluate the selection
(the oracle picks the configuration; training never sees a target label).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from MIL_CREDA_EEG import campaign, remote_units, search, spectral_trunk
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP


def replay(method, base, sides, cfg, device, trials2, rotation, unit):
    prepared, context, cut = base["prepared"], base["context"], base["cut"]
    space = search.phase_two_space(method)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=XC.SAMPLER_SEED + int(rotation)))
    rows = []

    def objective(trial):
        params = search.suggest(trial, space)
        e2 = int(trial.suggest_categorical("epochs2", list(XP.SPACE2["epochs2"][1])))
        row = {"trial": trial.number, "params": {**params, "epochs2": e2}}
        with XC.common_wrapper(context["width"], context["temperature"]):
            try:
                model = XC.adapt(method, prepared, context, source=sides["source"], target=sides["target"],
                                 params=params, seed=XC.SEARCH_SEED, rotation=rotation, unit=unit, epochs2=e2,
                                 cfg=cfg, device=device)
            except FloatingPointError as e:
                row["error"] = str(e)
                rows.append(row)
                return 0.0
            score = XC.selection_score(model, sides["target"], sides["source_test"], device)
            snap = XC.snapshot(model, sides["target_test"], cut, device)
        score = float(score) if np.isfinite(score) else 0.0
        row.update(ruleScore=round(score, 4), targetBA=snap["BA"], targetAUC=snap["AUC"],
                   targetKappa=snap["kappa"], rows=snap["rows"])
        rows.append(row)
        return score

    study.optimize(objective, n_trials=int(trials2))
    return rows


def run(config, product=None, *, experiment="ieee", rotation=0, half=0, trials2=20, phase1_params=None,
        methods=XC.ALL8, workdir=None, environ=None, **unit) -> dict:
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    built = XP.build_sides(cfg, environ, experiment=experiment, rotation=rotation, half=half, **unit)
    sides, name, cfg = built["sides"], built["unit"], built["cfg"]
    device = torch.device(cfg.get("device", "cpu"))
    campaign.register([m for m in methods if m not in XC.SKIPPED])
    out = {"unit": name, "rotation": rotation, "half": half, "phase1": phase1_params, "methods": {}}
    with spectral_trunk.installed(), XP.window_samples(built["samples"]):
        base = XC.shared_base(sides, cfg, device, XC.SEARCH_SEED, rotation, phase1_params, name)
        out["cut"] = float(base["cut"])
        for m in methods:
            if m in XC.SKIPPED:
                continue
            t0 = time.time()
            try:
                out["methods"][m] = {"trials": replay(m, base, sides, cfg, device, trials2, rotation, name)}
            except Exception as e:               # one comparator failing must not lose the others
                out["methods"][m] = {"failed": f"{type(e).__name__}: {e}"}
            out["methods"][m]["seconds"] = round(time.time() - t0, 1)
            print(f"[{name}] {m} done in {time.time() - t0:.0f} s", flush=True)
    out["seconds"] = round(time.time() - started)
    (workdir / f"comparator-diag-{name.replace(':', '-')}.json").write_text(json.dumps(out, default=str))
    return {"ran": [name], "seconds": out["seconds"]}
