"""A random search of EEGNet phase 1 with the corrections, chosen on the source inner splits.

Diagnostic, outside plan v60: it reports no result of the plan and writes nothing to the
model store. Every trial keeps the corrections already found necessary (inputs standardized
with the source training statistics, the head free of its 0.25 max-norm, BatchNorm
recalibrated before scoring, explicit L2 instead of Adam weight decay) and samples learning
rate, L2, dropout, network size, epochs and attention. A trial is scored by the balanced
accuracy pooled over the three inner validation splits of the outer rotation; the best one
is refit on the rotation's training subjects and read on its held-out source and target
subjects. No target label and no held-out source label chooses anything.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, model_store, remote_units, search, shallow_trunk, two_phase
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import (_side, _summary, explicit_l2, patched, recalibrate,
                                           standardized)

SIZES = ((8, 2, 16), (4, 2, 8))
SHALLOW_FILTERS = (20, 40)
EPOCH_CHOICES = (20, 50, 100)


def sample(rng: np.random.RandomState, backbone: str = "eegnet") -> dict:
    log = lambda lo, hi: float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    shared = {"DROPOUT": float(rng.choice([0.25, 0.5])), "epochs": int(rng.choice(EPOCH_CHOICES)),
              "attentionGamma": float(rng.uniform(0, 1)), "attentionTemperature": log(0.1, 10.0),
              "sigmaMultiplier": float(rng.choice([0.5, 1.0, 1.5]))}
    if backbone == "shallow":
        return {"lr1": log(1e-4, 3e-3), "l2": log(1e-6, 1e-2),
                "FILTERS": int(rng.choice(SHALLOW_FILTERS)), **shared}
    F1, D, F2 = SIZES[rng.randint(len(SIZES))]
    return {"lr1": log(1e-4, 3e-2), "l2": log(1e-5, 3e-2), "F1": F1, "D": D, "F2": F2, **shared}


@contextlib.contextmanager
def network(backbone: str, trial: Mapping):
    """The trial's network: EEGNet constants, or ShallowConvNet with its own filters and dropout."""
    if backbone != "shallow":
        with patched({**{k: trial[k] for k in ("DROPOUT", "F1", "D", "F2")}, "HEAD_MAX_NORM": 100.0}):
            yield
        return
    saved = (shallow_trunk.FILTERS, shallow_trunk.DROPOUT)
    shallow_trunk.FILTERS, shallow_trunk.DROPOUT = trial["FILTERS"], trial["DROPOUT"]
    try:
        with shallow_trunk.installed(), patched({"HEAD_MAX_NORM": 100.0}):
            yield
    finally:
        shallow_trunk.FILTERS, shallow_trunk.DROPOUT = saved


def fit(trial: Mapping, train, groups, cfg, device, seed: int, backbone: str = "eegnet") -> list:
    """Train on `train` with `trial`; return the scored rows of each group."""
    train, *groups = standardized(train, *groups)
    params = {"lr1": trial["lr1"], "wd1": 0.0, "attentionGamma": trial["attentionGamma"],
              "attentionTemperature": trial["attentionTemperature"],
              "sigmaMultiplier": trial["sigmaMultiplier"]}
    with network(backbone, trial):
        hyper = search.phase_one_hyper("MIL-CREDA", params,
                                       search.refit_schedule(cfg["schedule"], 1, trial["epochs"]))
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = model_store.build_model("MIL-CREDA", int(train[0].windows.shape[1]),
                                        int(groups[-1][0].windows.shape[1]), hyper).to(device)
        with explicit_l2(trial["l2"]):
            two_phase.train_source_phase(model, train, [], hyper, generator, device, seed=seed)
        recalibrate(model, train, device)
        rows = [_side(model, g, device) for g in (train, *groups)]
    del model
    return rows


def run(name: str, rotation: int, trials: int, cfg: Mapping, bags, device, seed: int,
        backbone: str = "eegnet", criterion: str = "ba") -> dict:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    side = lambda s, role, split=None: two_phase.bag_tensors(
        A.side_bags(a, bags, s, rotation, role, split=split))
    rng = np.random.RandomState(1000 * seed + rotation)
    history = []
    for _ in range(trials):
        trial = sample(rng, backbone)
        pooled = []
        for split in range(search.N_INNER):
            _, valid, _ = fit(trial, side("source", "inner_train", split),
                              [side("source", "inner_validation", split),
                               side("target", "test")[:1]], cfg, device, seed, backbone)
            pooled += valid
        history.append({"trial": trial, "inner": _summary(pooled)})
        print(name, rotation, len(history), history[-1]["inner"], flush=True)
    other = "auc" if criterion == "ba" else "ba"
    best = max(history, key=lambda h: (h["inner"][criterion] or 0, h["inner"][other] or 0))
    train, held, target = fit(best["trial"], side("source", A.REFIT_ROLE),
                              [side("source", "test"), side("target", "test")], cfg, device, seed,
                              backbone)
    return {"best": best, "history": history,
            "rows": {"train": train, "source": held, "target": target}}


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, rotation: int,
          trials: int = 12, seed: int = 1, backbone: str = "eegnet", criterion: str = "ba", workdir: str | Path | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    """One outer rotation of `transfer`, with the evidence a single run's fetch reads."""
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ),
                                                 placement)
    device = torch.device(cfg.get("device", "cpu"))
    bags = campaign.load_bags(cfg)
    mode = environ.get(remote_units.RUN_MODE_ENV)
    started = time.time()
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness:
        out = run(transfer, rotation, trials, cfg, bags, device, seed, backbone, criterion)
    record = {"transfer": transfer, "rotation": rotation, "backbone": backbone, "criterion": criterion,
              "seconds": round(time.time() - started),
              **{k: _summary(v) for k, v in out["rows"].items()}, "best": out["best"],
              "history": out["history"], "rows": out["rows"], "environment": environment}
    (workdir / "phase-one-search.json").write_text(json.dumps(record, default=str), encoding="utf-8")
    print({k: record[k] for k in ("transfer", "rotation", "seconds", "train", "source", "target")},
          flush=True)
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"],
                                "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": transfer,
                                "units": [transfer], "complete": True,
                                "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": [transfer], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {"ran": [transfer], "skipped": [], "environment": environment}
