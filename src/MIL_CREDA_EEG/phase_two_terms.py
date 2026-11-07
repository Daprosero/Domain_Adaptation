"""What in phase 2 lowers a phase-1 model that transfers: CogniFit->Robots, stored phase 1.

Diagnostic, outside plan v59: it reports no result of the plan and writes nothing to the
model store. Target test labels are read per epoch to see the trajectory and choose
nothing. `lr0` trains nothing, so any change there comes from BatchNorm statistics moving
on target bags. One session runs one condition, four rotations, sent as a single run.
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
from MIL_CREDA_EEG import campaign, model_store, remote_units, two_phase
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import _summary, recalibrate

NAME, EPOCHS, SEED = "CogniFit->Robots", 20, 1
CONDITIONS = {
    "lr0": dict(lr2=0.0, terms=two_phase.BOTH_TERMS),
    "both-1e-4": dict(lr2=1e-4, terms=two_phase.BOTH_TERMS),
    "global-1e-4": dict(lr2=1e-4, terms="global term only"),
    "local-1e-4": dict(lr2=1e-4, terms="local term only"),
    "both-1e-3": dict(lr2=1e-3, terms=two_phase.BOTH_TERMS),
}


def _scored(model, bags, device) -> list[dict]:
    p = two_phase.evaluate(model, bags, "target", device)[:, 1].detach().cpu().numpy()
    return [{"subject": b.subject, "label": b.label, "probability": float(q)}
            for b, q in zip(bags, p)]


def run_rotation(spec: Mapping, rotation: int, root: Path, a, bags, device) -> dict:
    path = root / "Models/Balladeer/CogniFit-to-Robots" / f"rotation{rotation}/seed1/phase1__MIL-CREDA.pt"
    model, meta = model_store.load_model(path, allow_code_mismatch=True)
    model = model.to(device)
    hyper = two_phase.check_hyper({**{k: meta["hyper"][k] for k in two_phase.PHASE_ONE_KEYS},
                                   "lr2": spec["lr2"], "wd2": 1e-5, "tauLocal": 1.0,
                                   "epochs1": 1, "epochs2": EPOCHS, "bagsPerStep": 8})
    source = two_phase.bag_tensors(A.side_bags(a, bags, "source", rotation, A.REFIT_ROLE))
    target = two_phase.bag_tensors(A.side_bags(a, bags, "target", rotation, A.REFIT_ROLE))
    test = two_phase.bag_tensors(A.side_bags(a, bags, "target", rotation, "test"))
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    joint = two_phase.bridge(model)
    before = _scored(model, test, device)
    result = two_phase.adapt_target_phase(model, {"checkpoint": {"sigma": meta["sigma"]}},
                                          joint["frozen_state"], source, target, test, hyper,
                                          generator, device, seed=SEED, fold=rotation,
                                          transfer=NAME, term_selection=spec["terms"])
    curve = [{k: round(float(h[k]), 4) for k in ("val_balanced_accuracy", "global", "local")}
             for h in result["history"]]
    after = _scored(model, test, device)
    recalibrate(model, target, device, trunk="target_trunk")
    return {"before": before, "after": after, "curve": curve,
            "recalibrated": _scored(model, test, device)}


def smoke(config: Mapping, product: str | Path | None = None, *, condition: str,
          workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    """All four rotations of `condition`, with the evidence a single run's fetch reads."""
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
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[NAME], bags)
    root = Path(cfg["productRoot"])
    mode = environ.get(remote_units.RUN_MODE_ENV)
    before, after, fresh, curves, done = [], [], [], [], []
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness:
        for rotation in range(4):
            started = time.time()
            rows = run_rotation(CONDITIONS[condition], rotation, root, a, bags, device)
            before += rows["before"]
            after += rows["after"]
            fresh += rows["recalibrated"]
            curves.append(rows["curve"])
            done.append(rotation)
            record = {"condition": condition, "rotations": done, "before": _summary(before),
                      "after": _summary(after), "recalibrated": _summary(fresh), "curves": curves,
                      "environment": environment,
                      "predictions": {"before": before, "after": after, "recalibrated": fresh}}
            (workdir / "phase-two-terms.json").write_text(json.dumps(record, default=str),
                                                          encoding="utf-8")
            print(condition, rotation, round(time.time() - started), record["before"],
                  record["after"], flush=True)
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"],
                                "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": condition,
                                "units": [condition], "complete": len(done) == 4,
                                "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": [condition], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {"ran": [condition], "skipped": [], "environment": environment}
