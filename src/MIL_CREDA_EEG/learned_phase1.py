"""Phase 1 with a learned instance encoder (TSMNet-style or T-GARNet), unseen-source generalization.

Diagnostic, outside the plan's protocol; writes nothing to the model store. Same windows as the
spectral check (1-45 Hz, common average reference, at most 60 per bag), inputs standardized per
channel by the source training windows, neutral MIL aggregation (no consensus, flat attention),
explicit L2 .01. Unseen source test AUC and BA, pooled over the four outer rotations.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, learned_trunks, model_store, remote_units, two_phase
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2

HYPER = {**sc.HYPER, "lr1": 1e-3, "epochs1": 100}


def run_source(transfer, cfg, bags, device, kind, seed, epochs):
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[transfer], bags)
    hyper = two_phase.check_hyper({**HYPER, "epochs1": epochs, "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    y, p = [], []
    for r in range(4):
        side = lambda role: sc.prepared(two_phase.bag_tensors(A.side_bags(a, bags, "source", r, role)))
        train, test = side(A.REFIT_ROLE), side("test")
        with learned_trunks.installed(kind):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            c = int(train[0].windows.shape[1])
            m = model_store.build_model("MIL-CREDA", c, c, hyper).to(device)
            m.source_trunk.norm = learned_trunks.input_standard_of(m.source_trunk, train)
            with explicit_l2(sc.L2):
                two_phase.train_source_phase(m, train, [], hyper, g, device, seed=seed, fold=r, transfer=transfer)
            y += [b.label for b in test]
            p += np.nan_to_num(two_phase.evaluate(m, test, "source", device)[:, 1].detach().cpu().numpy(), nan=.5).tolist()
        print(kind, transfer, seed, "rotation", r, "done", flush=True)
    return {"source": transfer.split("->")[0], "kind": kind, "seed": seed, "epochs": epochs,
            "AUC": round(float(roc_auc_score(y, p)), 3),
            "BA": round(float(balanced_accuracy_score(y, np.asarray(p) > .5)), 3), "n": len(y)}


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, kind: str, seed: int = 1,
          epochs: int = 100, workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    campaign.register(cfg["strategies"])
    environment = remote_units.environment_stamp(remote_units.set_deterministic(),
                                                 remote_units.campaign_identity(cfg, environ), placement)
    device = torch.device(cfg.get("device", "cpu"))
    bags = campaign.load_bags(cfg)
    mode = environ.get(remote_units.RUN_MODE_ENV)
    started = time.time()
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness:
        result = run_source(transfer, cfg, bags, device, kind, int(seed), int(epochs))
    print(json.dumps(result), flush=True)
    (workdir / "learned-phase1.json").write_text(json.dumps({**result, "seconds": round(time.time() - started),
                                                             "environment": environment}, default=str), encoding="utf-8")
    if mode:
        payload = {"mode": mode,
                   "evidence": {"commit": environment["commit"], "sourcesDigest": environment["codeVersion"],
                                "codeVersion": environment["codeVersion"], "unit": transfer, "units": [transfer],
                                "complete": True, "gpu": witness.evidence(), "estimates": {}},
                   "environment": environment, "runs": {"ran": [transfer], "skipped": []}}
        (workdir / remote_units.STAMP_FILE).write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return {"ran": [transfer], "skipped": [], "environment": environment}
