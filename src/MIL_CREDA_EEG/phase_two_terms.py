"""Can phase 2 beat source-only prediction from the stored phase-1 model, and at which step?

Diagnostic, outside plan v59: it reports no result of the plan and writes nothing to the
model store. Target test labels are read per epoch to see the trajectory and choose
nothing. The source trunk's BatchNorm is recalibrated on the source bags before the bridge,
so `predict` and phase 2 start from the same model. `stored` reuses the phase-2 settings the
v59 search selected. One session runs one condition over the task and level transfers.
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
from MIL_CREDA_EEG.phase_one_sweep import _summary, balanced_accuracy, recalibrate

TRANSFERS = ("CogniFit->Robots", "Robots->CogniFit", "L1->L11", "L11->L1")
EPOCHS, SEED = 20, 1
CONDITIONS = {"lr3e-4": dict(lr2=3e-4), "lr1e-3": dict(lr2=1e-3), "lr3e-3": dict(lr2=3e-3),
              "lr1e-2": dict(lr2=1e-2), "stored": dict()}


def _scored(model, bags, device) -> list[dict]:
    p = two_phase.evaluate(model, bags, "target", device)[:, 1].detach().cpu().numpy()
    return [{"subject": b.subject, "label": b.label, "probability": float(q)}
            for b, q in zip(bags, p)]


def run_rotation(name: str, spec: Mapping, rotation: int, root: Path, a, bags, device) -> dict:
    folder = root / "Models/Balladeer" / name.replace("->", "-to-") / f"rotation{rotation}/seed1"
    model, meta = model_store.load_model(folder / "phase1__MIL-CREDA.pt", allow_code_mismatch=True)
    model = model.to(device)
    chosen = {"lr2": spec.get("lr2"), "wd2": 1e-5, "tauLocal": 1.0}
    if not spec:
        chosen = json.loads((folder / "phase2__MIL-CREDA.json").read_text())["selection"]["params"]
        chosen = {k: chosen[k] for k in two_phase.PHASE_TWO_KEYS}
    hyper = two_phase.check_hyper({**{k: meta["hyper"][k] for k in two_phase.PHASE_ONE_KEYS},
                                   **chosen, "epochs1": 1, "epochs2": EPOCHS, "bagsPerStep": 8})
    source = two_phase.bag_tensors(A.side_bags(a, bags, "source", rotation, A.REFIT_ROLE))
    target = two_phase.bag_tensors(A.side_bags(a, bags, "target", rotation, A.REFIT_ROLE))
    test = two_phase.bag_tensors(A.side_bags(a, bags, "target", rotation, "test"))
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    recalibrate(model, source, device)
    joint = two_phase.bridge(model)
    before = _scored(model, test, device)
    result = two_phase.adapt_target_phase(model, {"checkpoint": {"sigma": meta["sigma"]}},
                                          joint["frozen_state"], source, target, test, hyper,
                                          generator, device, seed=SEED, fold=rotation,
                                          transfer=name)
    epochs = [{"labels": r["labels"], "p": [q[1] for q in r["probabilities"]]}
              for r in result["validation"]]
    return {"before": before, "after": _scored(model, test, device), "epochs": epochs}


def smoke(config: Mapping, product: str | Path | None = None, *, condition: str,
          workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    """Every transfer of `condition`, with the evidence a single run's fetch reads."""
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
    root = Path(cfg["productRoot"])
    mode = environ.get(remote_units.RUN_MODE_ENV)
    done = []
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness:
        for name in TRANSFERS:
            a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
            before, after, labels, curves = [], [], [], [[] for _ in range(EPOCHS)]
            started = time.time()
            for rotation in range(4):
                rows = run_rotation(name, CONDITIONS[condition], rotation, root, a, bags, device)
                before += rows["before"]
                after += rows["after"]
                labels += rows["epochs"][0]["labels"]
                for e, epoch in enumerate(rows["epochs"]):
                    curves[e] += epoch["p"]
            pooled = [round(balanced_accuracy(labels, c), 4) for c in curves if len(c) == len(labels)]
            record = {"condition": condition, "transfer": name, "before": _summary(before),
                      "after": _summary(after), "curve": pooled, "environment": environment,
                      "predictions": {"before": before, "after": after}}
            (workdir / f"p2-{name.replace('->', '-to-')}.json").write_text(
                json.dumps(record, default=str), encoding="utf-8")
            done.append(name)
            print(condition, name, round(time.time() - started), record["before"], record["after"],
                  pooled[::5], flush=True)
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"],
                                "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": condition,
                                "units": [condition], "complete": len(done) == len(TRANSFERS),
                                "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": [condition], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return {"ran": [condition], "skipped": [], "environment": environment}
