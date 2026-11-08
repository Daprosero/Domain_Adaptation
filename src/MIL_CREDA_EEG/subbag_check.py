"""Sub-bags in phase-1 training only: does a source model that cannot memorize transfer?

Diagnostic, outside plan v59: it reports no result of the plan and writes nothing to the
model store. EEGNet carries the corrections already found necessary (inputs standardized by
the source training statistics, the head free of its 0.25 max-norm, explicit L2, BatchNorm
recalibrated, sigma recomputed on the trained network, balanced pseudo-labels in phase 2).
Only phase-1 training splits each source subject into sub-bags of `subbag` windows, all in
that subject's fold, since the diagnosis is a trait of the whole recording. Phase 2 and every
evaluation read whole bags. Settings are fixed before the run; no target label chooses.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, input_norm, model_store, remote_units, search, shallow_trunk, two_phase
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import (_side, _summary, explicit_l2, patched, recalibrate,
                                           standardized)
from MIL_CREDA_EEG.phase_two_terms import balanced_pseudolabels

SPEC = {"HEAD_MAX_NORM": 100.0, "DROPOUT": 0.5}
FIXED = {"lr1": 1e-2, "wd1": 0.0, "attentionGamma": 0.5, "attentionTemperature": 1.0,
         "sigmaMultiplier": 1.0, "lr2": 1e-3, "wd2": 1e-5, "tauLocal": 1.0}
L2, EPOCHS1, EPOCHS2 = 1e-3, 50, 20
#: Per-backbone phase-1 learning rate and L2; ShallowConvNet did not train at EEGNet's values.
RECIPES = {"shallow": {"lr1": 1e-3, "l2": 1e-4}}


def subbags(bags, size: int, generator: torch.Generator) -> list:
    """Each bag cut into sub-bags of `size` windows in a seeded order; whole bags when 0."""
    if not size:
        return list(bags)
    out = []
    for b in bags:
        order = torch.randperm(b.windows.shape[0], generator=generator)
        for start in range(0, len(order) - size + 1, size):
            out.append(two_phase.BagTensor(b.subject, b.condition, b.label,
                                           b.windows[order[start:start + size]]))
    return out


def common_average(bags) -> list:
    """Re-reference every window to the mean over its channels (label-free, per window)."""
    return [two_phase.BagTensor(b.subject, b.condition, b.label,
                                b.windows - b.windows.mean(dim=1, keepdim=True)) for b in bags]


def run_rotation(name: str, rotation: int, size: int, cfg, a, bags, device, seed: int,
                 scale: bool = True, recipe: Mapping | None = None, car: bool = False) -> dict:
    recipe = recipe or {}
    reference = common_average if car else (lambda g: g)
    side = lambda s, role: reference(two_phase.bag_tensors(A.side_bags(a, bags, s, rotation, role)))
    source, held = side("source", A.REFIT_ROLE), side("source", "test")
    target, test = side("target", A.REFIT_ROLE), side("target", "test")
    same = source[0].windows.shape[1] == target[0].windows.shape[1]
    if not scale:            # the network's own input normalization replaces it
        pass
    elif same:
        source, held, target, test = standardized(source, held, target, test)
    else:                    # device arm: each montage scaled by its own training statistics
        source, held = standardized(source, held)
        target, test = standardized(target, test)
    hyper = two_phase.check_hyper({**FIXED, "lr1": recipe.get("lr1", FIXED["lr1"]),
                                   "epochs1": EPOCHS1, "epochs2": EPOCHS2,
                                   "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = model_store.build_model("MIL-CREDA", int(source[0].windows.shape[1]),
                                    int(target[0].windows.shape[1]), hyper).to(device)
    common = {"seed": seed, "fold": rotation, "transfer": name}
    with explicit_l2(recipe.get("l2", L2)):
        phase1 = two_phase.train_source_phase(model, subbags(source, size, generator), [], hyper,
                                              generator, device, **common)
    recalibrate(model, source, device)
    median = two_phase.within_bag_median_distance(model, source, "source", device)
    if median > 0:           # a collapsed encoder (every window alike) keeps the phase-1 sigma
        model._sigma = hyper["sigmaMultiplier"] * median
    phase1["checkpoint"]["sigma"] = model._sigma
    rows = {"train": _side(model, source, device), "source": _side(model, held, device)}
    if same:                 # source-only prediction needs the source montage on the target
        rows["predict"] = _side(model, test, device)
    joint = two_phase.bridge(model)
    with balanced_pseudolabels(model):
        two_phase.adapt_target_phase(model, phase1, joint["frozen_state"], source, target, [], hyper,
                                     generator, device, **common)
    p = two_phase.evaluate(model, test, "target", device)[:, 1].detach().cpu().numpy()
    rows["milCreda"] = [{"subject": b.subject, "label": b.label, "probability": float(q)}
                        for b, q in zip(test, p)]
    return rows


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, subbag: int,
          seed: int = 1, backbone: str = "eegnet", arch: Mapping | None = None, car: bool = False,
          workdir: str | Path | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    """Four rotations of `transfer`, with the evidence a single run's fetch reads."""
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
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[transfer], bags)
    mode = environ.get(remote_units.RUN_MODE_ENV)
    pooled: dict[str, list] = {k: [] for k in ("train", "source", "predict", "milCreda")}
    started = time.time()
    trunk = (shallow_trunk.installed() if backbone == "shallow"
             else input_norm.log_power() if backbone == "eegnet-logpower"
             else input_norm.fixed_bands() if backbone == "eegnet-fixedbands"
             else input_norm.installed(backbone.split("-")[1]) if backbone.startswith("eegnet-")
             else contextlib.nullcontext())
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness, trunk, \
            patched({**SPEC, **(arch or {})}):
        for rotation in range(4):
            rows = run_rotation(transfer, rotation, subbag, cfg, a, bags, device, seed,
                                scale=backbone in ("eegnet", "eegnet-logpower", "eegnet-fixedbands", "shallow"),
                                recipe=RECIPES.get(backbone), car=car)
            for k, v in rows.items():
                pooled[k] += v
            pooled = {k: v for k, v in pooled.items() if v or k in rows}
            print(transfer, subbag, seed, rotation, {k: _summary(v) for k, v in rows.items()}, flush=True)
    record = {"transfer": transfer, "subbag": subbag, "seed": seed, "backbone": backbone,
              "arch": dict(arch or {}), "car": car,
              "seconds": round(time.time() - started),
              **{k: _summary(v) for k, v in pooled.items()}, "rows": pooled, "environment": environment}
    (workdir / "subbag-check.json").write_text(json.dumps(record, default=str), encoding="utf-8")
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
