"""Does phase 2 of MIL-CREDA beat source-only prediction from the same phase-1 model?

Diagnostic, outside plan v59: it reports no result of the plan and writes nothing to the
model store. BatchNorm momentum is 0.1 in both phases and the settings are fixed before
the run, never from target labels: lr1 1e-3 for 40 epochs; lr2 1e-4, wd2 1e-5, tauLocal 1
for 20 epochs. The attention settings and wd1 are the stored source-selected ones.
One session runs one transfer, four rotations, sent as a single run of its job.
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
from MIL_CREDA_EEG.phase_one_sweep import _summary, patched

SELECTIONS = Path(__file__).with_name("beat_predict_selections.json")
SEED, EPOCHS1, EPOCHS2 = 1, 40, 20
FIXED = {"lr1": 1e-3, "lr2": 1e-4, "wd2": 1e-5, "tauLocal": 1.0}
SPEC = {"BN_MOMENTUM": 0.1}


def hyper_of(transfer: str, rotation: int, schedule: Mapping) -> dict:
    stored = json.loads(SELECTIONS.read_text(encoding="utf-8"))[transfer][str(rotation)]
    return two_phase.check_hyper({**{k: v for k, v in stored.items() if k != "wd1"},
                                  "wd1": stored["wd1"], **FIXED, "epochs1": EPOCHS1,
                                  "epochs2": EPOCHS2, "bagsPerStep": schedule["bagsPerStep"]})


def _scored(model, bags, domain: str, device) -> list[dict]:
    p = two_phase.evaluate(model, bags, domain, device)[:, 1].detach().cpu().numpy()
    return [{"subject": b.subject, "label": b.label, "probability": float(q)}
            for b, q in zip(bags, p)]


def run_rotation(name: str, rotation: int, cfg: Mapping, a, bags, device) -> dict:
    hyper = hyper_of(name, rotation, cfg["schedule"])
    source = two_phase.bag_tensors(A.side_bags(a, bags, "source", rotation, A.REFIT_ROLE))
    held = two_phase.bag_tensors(A.side_bags(a, bags, "source", rotation, "test"))
    target = two_phase.bag_tensors(A.side_bags(a, bags, "target", rotation, A.REFIT_ROLE))
    test = two_phase.bag_tensors(A.side_bags(a, bags, "target", rotation, "test"))
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = model_store.build_model("MIL-CREDA", int(source[0].windows.shape[1]),
                                    int(target[0].windows.shape[1]), hyper).to(device)
    common = {"seed": SEED, "fold": rotation, "transfer": name}
    phase1 = two_phase.train_source_phase(model, source, held, hyper, generator, device, **common)
    rows = {"sourceHeldOut": _scored(model, held, "source", device),
            "predict": _scored(model, test, "source", device)}
    joint = two_phase.bridge(model)
    two_phase.adapt_target_phase(model, phase1, joint["frozen_state"], source, target, [], hyper,
                                 generator, device, **common)
    rows["milCreda"] = _scored(model, test, "target", device)
    return rows


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str,
          workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    """All four rotations of `transfer`, with the evidence a single run's fetch reads."""
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
    pooled: dict[str, list] = {"sourceHeldOut": [], "predict": [], "milCreda": []}
    done = []
    path = workdir / "beat-predict.json"
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness, \
            patched(SPEC):
        for rotation in range(4):
            started = time.time()
            rows = run_rotation(transfer, rotation, cfg, a, bags, device)
            for key, value in rows.items():
                pooled[key].extend(value)
            done.append(rotation)
            record = {"transfer": transfer, "rotations": done, "environment": environment,
                      **{k: _summary(v) for k, v in pooled.items()}, "predictions": pooled}
            path.write_text(json.dumps(record, default=str), encoding="utf-8")
            print(transfer, rotation, round(time.time() - started),
                  {k: _summary(v) for k, v in rows.items()}, flush=True)
    outcome = {"ran": [transfer], "skipped": [], "environment": environment}
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"],
                                "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": transfer,
                                "units": [transfer], "complete": len(done) == 4,
                                "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": [transfer], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8")
    return outcome
