"""Leave-one-subject-out check: does phase 2 beat source-only with more subjects per fit?

Diagnostic, outside plan v60: it reports no result of the plan and writes nothing to the
model store. Phase 1 trains once on every source subject, at the source-selected settings
of rotation 0 (`loso_selections.json`), and its BatchNorm statistics are recalibrated on
the source. Then, for each target subject, phase 2 adapts on every other target subject,
without labels, and predicts the one left out. `predict` is phase 1 on that subject. No
setting is chosen from target labels.
"""

from __future__ import annotations

import contextlib
import copy
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
from MIL_CREDA_EEG.phase_two_terms import balanced_pseudolabels

SELECTIONS = Path(__file__).with_name("loso_selections.json")
EPOCHS2 = 20
VARIANTS = {"plain": dict(lr2=3e-3, balanced=False), "balanced": dict(lr2=3e-3, balanced=True)}


def all_bags(a, bags, side: str) -> list:
    return [bags.get(s, a.conditions[s]) for s in sorted(a.subjects(side))]


def _p(model, bags, domain, device) -> float:
    return float(two_phase.evaluate(model, bags, domain, device)[0, 1])


def run_transfer(name: str, cfg: Mapping, bags, device, seed: int) -> dict:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[name], bags)
    chosen = json.loads(SELECTIONS.read_text(encoding="utf-8"))[name]
    source = two_phase.bag_tensors(all_bags(a, bags, "source"))
    target = two_phase.bag_tensors(all_bags(a, bags, "target"))
    hyper = two_phase.check_hyper({**{k: chosen[k] for k in two_phase.PHASE_ONE_KEYS},
                                   "lr2": 3e-3, "wd2": 1e-5, "tauLocal": 1.0,
                                   "epochs1": chosen["epochs1"], "epochs2": EPOCHS2,
                                   "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = model_store.build_model("MIL-CREDA", int(source[0].windows.shape[1]),
                                    int(target[0].windows.shape[1]), hyper).to(device)
    phase1 = two_phase.train_source_phase(model, source, [], hyper, generator, device,
                                          seed=seed, fold=0, transfer=name)
    recalibrate(model, source, device)
    rows = {"predict": [], **{k: [] for k in VARIANTS}}
    for i, held in enumerate(target):
        others = target[:i] + target[i + 1:]
        rows["predict"].append({"subject": held.subject, "label": held.label,
                                "probability": _p(model, [held], "source", device)})
        for key, spec in VARIANTS.items():
            adapted = copy.deepcopy(model)
            joint = two_phase.bridge(adapted)
            step = two_phase.check_hyper({**hyper, "lr2": spec["lr2"]})
            local = torch.Generator().manual_seed(seed)
            with balanced_pseudolabels(adapted) if spec["balanced"] else contextlib.nullcontext():
                two_phase.adapt_target_phase(adapted, phase1, joint["frozen_state"], source,
                                             others, [], step, local, device, seed=seed,
                                             fold=i, transfer=name)
            rows[key].append({"subject": held.subject, "label": held.label,
                              "probability": _p(adapted, [held], "target", device)})
            del adapted
    return rows


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, seed: int = 1,
          workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    """Every target subject of `transfer`, with the evidence a single run's fetch reads."""
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
        rows = run_transfer(transfer, cfg, bags, device, seed)
    record = {"transfer": transfer, "seed": seed, "seconds": round(time.time() - started),
              **{k: _summary(v) for k, v in rows.items()}, "environment": environment,
              "predictions": rows}
    (workdir / "loso.json").write_text(json.dumps(record, default=str), encoding="utf-8")
    print({k: record[k] for k in ("transfer", "seconds", "predict", *VARIANTS)}, flush=True)
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
