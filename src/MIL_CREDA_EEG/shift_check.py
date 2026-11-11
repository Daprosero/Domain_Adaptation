"""Diagnostic of the one-sided EEGNet output on unseen source subjects (outside the plan's protocol).

EEGNet phase 1 as learned_phase1 (stored 0.2-45 Hz whole bags, inner split 0 of one rotation,
neutral aggregation, explicit L2), then, for the training, validation and test subjects: the
fraction scored above 0.5, the mean score per class, BA at 0.5 and AUC. Each split is scored with
the running BatchNorm statistics, with each bag's own batch statistics, and after recalibrating
BatchNorm on the training bags. Writes shift-check.json; nothing goes to the model store.
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
from MIL_CREDA_EEG import campaign, model_store, remote_units, two_phase, v63_setup
from MIL_CREDA_EEG import learned_phase1 as L
from MIL_CREDA_EEG import spectral_check as sc
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2, recalibrate


def _summary(y, q):
    y, q = np.asarray(y), np.asarray(q)
    return {"n": int(len(y)), "positives": round(float(y.mean()), 3),
            "above": round(float(np.mean(q > .5)), 3),
            "meanClass0": round(float(q[y == 0].mean()), 4), "meanClass1": round(float(q[y == 1].mean()), 4),
            "min": round(float(q.min()), 4), "max": round(float(q.max()), 4),
            "BA": round(float(balanced_accuracy_score(y, q > .5)), 3),
            "AUC": round(float(roc_auc_score(y, q)), 3)}


def _own_batch(model, units, device):
    """Each bag scored with its own BatchNorm batch statistics; running buffers left untouched."""
    out = []
    model.eval()
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [(m.momentum, m.running_mean.clone(), m.running_var.clone()) for m in norms]
    for m in norms:
        m.train()
        m.momentum = 0.0
    with torch.no_grad():
        for u in units:
            out.append(model.predict([u.windows.to(device)], "source", chunk=two_phase.EVAL_CHUNK)[:, 1])
    for m, (mom, mean, var) in zip(norms, saved):
        m.momentum, m.running_mean, m.running_var = mom, mean, var
        m.eval()
    return torch.cat(out).cpu().numpy()


def check(transfer, cfg, bags, device, seed, epochs, rotation=0):
    a = A.assign_from_bags(v63_setup.transfer(transfer), bags)
    hyper = two_phase.check_hyper({**L.HYPER, "epochs1": epochs, "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    role = lambda r: L._stored_whole(two_phase.bag_tensors(
        A.side_bags(a, bags, "source", rotation, r, split=0) if r.startswith("inner")
        else A.side_bags(a, bags, "source", rotation, r)))
    splits = {"train": role("inner_train"), "valid": role("inner_validation"), "test": role("test")}
    with L._trunk("eegnet"):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        c = int(splits["train"][0].windows.shape[1])
        m = model_store.build_model("MIL-CREDA", c, c, hyper).to(device)
        with explicit_l2(sc.L2):
            two_phase.train_source_phase(m, splits["train"], [], hyper, g, device, seed=seed, fold=rotation,
                                         transfer=transfer)
        score = lambda us: two_phase.evaluate(m, us, "source", device)[:, 1].detach().cpu().numpy()
        labels = {k: [u.label for u in us] for k, us in splits.items()}
        out = {"running": {k: _summary(labels[k], score(us)) for k, us in splits.items()},
               "ownBatch": {k: _summary(labels[k], _own_batch(m, us, device)) for k, us in splits.items()}}
        recalibrate(m, splits["train"], device)
        out["recalibrated"] = {k: _summary(labels[k], score(us)) for k, us in splits.items()}
    return {"transfer": transfer, "seed": seed, "rotation": rotation, "epochs": epochs, **out}


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, seed: int = 1,
          epochs: int = 100, rotation: int = 0, workdir: str | Path | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
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
        result = check(transfer, cfg, bags, device, int(seed), int(epochs), int(rotation))
        print(json.dumps(result), flush=True)
    (workdir / "shift-check.json").write_text(json.dumps({**result, "seconds": round(time.time() - started),
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
