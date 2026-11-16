"""Shared random pool for testing label-free phase-2 selection criteria against the oracle.

One notebook = one unit, one phase-2 seed k and one shard of the trials. Phase 1 is frozen: every
method of every notebook starts from the unit's single base (`<unit>-s1.pt`, the nested-searched
phase-1 settings, seed 1). For each method, `draw` fixes TRIALS settings over the method's FULL
phase-2 space with a seeded RandomSampler (Musgrave et al.'s protocol: validators select from a
shared pool, they never guide it); the shard trains its range of them, each for EPOCHS2 epochs under
the annealed learning rate of `lr_schedule`, with a training seed that depends on k and the trial
index only, so the shards together equal one unsharded run. Each trial records its settings, the
balanced accuracy and AUC on a large pool of fresh target subjects (the oracle, true labels only
the simulator has) and on the unit's 12 target-test subjects, and its target-train and source-test
outputs (class probabilities and representations), from which every criterion is computed offline.
No label-free criterion is computed or read here.
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
from optuna.distributions import FloatDistribution

from MIL_CREDA_EEG import campaign, lr_schedule, remote_units, search, spectral_trunk
from MIL_CREDA_EEG import oracle_screen as O
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

BASES = O.BASES
METHODS = O.METHODS
EPOCHS2 = 100
#: the dry run whose optimizer steps, scaled to EPOCHS2, give each scheduled optimizer its total
DRY_EPOCHS = 4
TRIALS = 100
SHARDS = 4
SEEDS = (1, 2, 3, 4, 5)
SAMPLER_SEED = 20261115
TAU_ATT = (1.0, 10.0)
#: Eq. 34's softmax(kappa / tau_local) is uniform for tau_local >= 0.1 on the measured kappa spread
#: (median within-class range 0.05), so the range reaches down to a hard match
TAU_LOCAL = (1e-3, 1.0)
DEV_UNITS = tuple(f"d{d}-s{s}-r{r}" for d in ("1p5", "2p5") for s in ("0p5", "1p5", "3") for r in (0, 1))
BOUNDARY_UNITS = tuple(f"d1-s{s}-r{r}" for s in ("0p5", "1p5", "3") for r in (0, 1))
UNITS = DEV_UNITS + BOUNDARY_UNITS


def space(method: str) -> dict:
    """The method's phase-2 space: MIL-CREDA's seven settings, or a comparator's own space plus the
    attention temperature of its common MIL wrapper. epochs2 is fixed (EPOCHS2), never searched."""
    if method == "MIL-CREDA":
        out = {}
        for k, (kind, (lo, hi)) in O.MIL_SPACE.items():
            lo, hi = TAU_LOCAL if k == "tau_local" else (lo, hi)
            out[k] = FloatDistribution(lo, hi, log=(kind == "log"))
        return out
    campaign.register([method])
    return {**search.phase_two_space(method), "tau_att": FloatDistribution(*TAU_ATT, log=True)}


def draw(method: str, seed: int, trials: int = TRIALS) -> list[dict]:
    """The `trials` settings of `method` under phase-2 seed `seed`, in order."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(
        seed=SAMPLER_SEED + 1000 * int(seed) + METHODS.index(method)))
    dists = space(method)
    return [dict(study.ask(dists).params) for _ in range(int(trials))]


def shard_trials(shard: int, shards: int = SHARDS, trials: int = TRIALS) -> range:
    per = -(-int(trials) // int(shards))
    return range(int(shard) * per, min(int(trials), (int(shard) + 1) * per))


def train_seed(seed: int, trial: int) -> int:
    return 1000 * int(seed) + int(trial)


def base_file(unit_key: str, bases=None) -> Path:
    """Always the seed-1 base: phase 1 is fitted once per unit and frozen."""
    return Path(bases or BASES) / f"{unit_key}-s1.pt"


def _wrapper(method, b, tau_att):
    return (XC.common_wrapper(b["context"]["width"], float(tau_att)) if method != "MIL-CREDA"
            else torch.no_grad())


def _fit(method, p, epochs, b, sides, cfg, dev, tseed, rot, name):
    if method == "MIL-CREDA":
        return O._train(method, dict(p), int(epochs), b, sides, cfg, dev, tseed, rot, name)
    q = dict(p)
    tau_att = q.pop("tau_att")
    with XC.common_wrapper(b["context"]["width"], float(tau_att)):
        return XC.adapt(method, b["prepared"], b["context"], source=sides["source"], target=sides["target"],
                        params=q, seed=tseed, rotation=rot, unit=name, epochs2=int(epochs), cfg=cfg, device=dev)


def dry_epochs(method) -> int:
    return DRY_EPOCHS if method == "MIL-CREDA" else XC.epochs_of(method, DRY_EPOCHS)


def train(method, p, b, sides, cfg, dev, tseed, rot, name, epochs=EPOCHS2):
    """One trial: a dry run counts the steps of every optimizer at lr2, then the full run anneals."""
    lr2 = float(p["lr2"])
    with lr_schedule.count_steps(lr2) as counts:
        _fit(method, p, dry_epochs(method), b, sides, cfg, dev, tseed, rot, name)
    full = epochs if method == "MIL-CREDA" else XC.epochs_of(method, epochs)
    totals = [max(1, round(c * full / dry_epochs(method))) for c in counts] or [1]
    with lr_schedule.annealed(lr2, totals):
        model = _fit(method, p, epochs, b, sides, cfg, dev, tseed, rot, name)
    return model, totals


def _outputs(model, bags, side, dev):
    P = XC.two_phase.evaluate(model, bags, side, dev).detach().cpu().numpy().astype(np.float32)
    Z = XC.two_phase.represent(model, bags, side, dev).detach().cpu().numpy().astype(np.float32)
    return P, Z


def run(unit_key: str, seed: int, shard: int, shards: int = SHARDS, trials: int = TRIALS,
        methods: Sequence[str] = METHODS, eval_subjects: int = O.EVAL_SUBJECTS, workdir=None, bases=None,
        epochs: int = EPOCHS2) -> dict:
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get("MIL_CREDA_THREADS") or os.cpu_count() or 2)))
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"methods are among {list(METHODS)}; got {unknown}")
    with spectral_trunk.installed():
        b = torch.load(base_file(unit_key, bases), weights_only=False)
    cfg = remote_units.resolve_inputs(campaign.check_config(b["config"]), environ={})
    built = XP.build_sides(cfg, {}, experiment="sim", rotation=b["rotation"], half=0, **b["unitKwargs"])
    sides, name, cfg = built["sides"], built["unit"], built["cfg"]
    if name != b["unit"]:
        raise ValueError(f"base {b['unit']} does not match the rebuilt unit {name}")
    pool = O.eval_pool(b, built["effect"], eval_subjects)
    campaign.register([m for m in methods if m != "MIL-CREDA"])
    dev, rot, cut = torch.device("cpu"), int(b["rotation"]), float(b["cut"])
    stem = f"criteria-{unit_key}-k{int(seed)}-sh{int(shard)}of{int(shards)}"
    out_path, arr_dir = workdir / f"{stem}.json", workdir / stem
    arr_dir.mkdir(exist_ok=True)
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "unitKey": unit_key, "unit": name, "seed": int(seed), "shard": int(shard), "shards": int(shards),
        "trials": int(trials), "epochs2": int(epochs), "cut": cut, "baseCommit": b["commit"],
        "baseSeed": 1, "evalSubjects": len(pool), "schedule": "lr2*(1+10p)^-0.75",
        "oracle": "BA at the base's cut on the evaluation pool (true labels)",
        "sourceTestLabels": [int(x.label) for x in sides["source_test"]],
        "targetSubjects": [x.subject for x in sides["target"]], "methods": {}}
    with spectral_trunk.installed(), XP.window_samples(built["samples"]), XP.full_trace_mode(False):
        for method in methods:
            rows = out["methods"].setdefault(method, {"trials": {}})["trials"]
            settings = draw(method, seed, trials)
            for i in shard_trials(shard, shards, trials):
                if str(i) in rows:
                    continue
                p, t0 = settings[i], time.time()
                try:
                    model, totals = train(method, p, b, sides, cfg, dev, train_seed(seed, i), rot, name, epochs)
                except FloatingPointError:     # the trainer's own non-finite refusal
                    rows[str(i)] = {"params": p, "nonFinite": True}
                    out_path.write_text(json.dumps(out))
                    continue
                with _wrapper(method, b, p.get("tau_att", 1.0)):
                    on_pool = XC.snapshot(model, pool, cut, dev)
                    on_test = XC.snapshot(model, sides["target_test"], cut, dev)
                    Pt, Zt = _outputs(model, sides["target"], "target", dev)
                    Ps, Zs = _outputs(model, sides["source_test"], "source", dev)
                np.savez_compressed(arr_dir / f"{method.split(' ')[0]}-{i}.npz", Pt=Pt, Zt=Zt, Ps=Ps, Zs=Zs)
                rows[str(i)] = {"params": p, "BApool": on_pool["BA"], "AUCpool": on_pool["AUC"],
                                "BAtest12": on_test["BA"], "AUCtest12": on_test["AUC"],
                                "stepTotals": totals, "seconds": round(time.time() - t0, 1)}
                out_path.write_text(json.dumps(out))
            print(unit_key, seed, shard, method, "done", len(rows), flush=True)
    out_path.write_text(json.dumps(out))
    return {"file": out_path.name, "methods": {m: len(v["trials"]) for m, v in out["methods"].items()}}
