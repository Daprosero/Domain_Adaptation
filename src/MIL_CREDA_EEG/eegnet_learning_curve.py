"""Would EEGNet's phase 1 generalize with more subjects? A learning curve over subjects.

Diagnostic, outside the plan's protocol: it reports no result of the plan and writes nothing to
the model store. In each outer rotation EEGNet (the recipe of `subbag_check`, fixed epochs, no
selection) is trained on a class-stratified fraction of the source training subjects and scored
on the same unseen source test subjects. Fractions below 1 are drawn several times so one draw
does not decide. If the score still rises from 75% to 100%, more subjects would help.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, model_store, remote_units, two_phase
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2, patched, recalibrate, standardized
from MIL_CREDA_EEG.subbag_check import FIXED, L2, SPEC

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
EPOCHS = 100
DRAWS = 3


def stratified(bags: list, fraction: float, rng: random.Random) -> list:
    """A class-stratified subset of the subjects (one bag per subject), at least 2 per class."""
    if fraction >= 1.0:
        return list(bags)
    out = []
    for label in sorted({b.label for b in bags}):
        group = [b for b in bags if b.label == label]
        rng.shuffle(group)
        out += group[:max(2, round(fraction * len(group)))]
    return out


def trained_scores(train, held, hyper, device, seed: int, rotation: int, transfer: str) -> list:
    """Train on `train` for the fixed epochs; the probabilities on `held` after recalibration."""
    train, held = standardized(train, held)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    channels = int(train[0].windows.shape[1])
    model = model_store.build_model("MIL-CREDA", channels, channels, hyper).to(device)
    with explicit_l2(L2):
        two_phase.train_source_phase(model, train, [], hyper, generator, device,
                                     seed=seed, fold=rotation, transfer=transfer)
    recalibrate(model, train, device)
    p = two_phase.evaluate(model, held, "source", device)[:, 1]
    return np.nan_to_num(p.detach().cpu().numpy(), nan=0.5).tolist()


def run_source(transfer: str, cfg, bags, device, seed: int, epochs: int, draws: int) -> dict:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[transfer], bags)
    hyper = two_phase.check_hyper({**FIXED, "epochs1": epochs, "epochs2": 1,
                                   "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    labels: list = []
    pooled = {(f, d): [] for f in FRACTIONS for d in range(draws if f < 1 else 1)}
    sizes = {f: [] for f in FRACTIONS}
    for r in range(4):
        side = lambda role: two_phase.bag_tensors(A.side_bags(a, bags, "source", r, role))
        train, held = side(A.REFIT_ROLE), side("test")
        labels += [b.label for b in held]
        for (f, d) in pooled:
            subset = stratified(train, f, random.Random(1000 * seed + 100 * r + d))
            if d == 0:
                sizes[f].append(len(subset))
            pooled[(f, d)] += trained_scores(subset, held, hyper, device, seed, r, transfer)
        print(transfer, seed, "rotation", r, "done", flush=True)
    rows = []
    for f in FRACTIONS:
        draws_f = [pooled[k] for k in pooled if k[0] == f]
        auc = [float(roc_auc_score(labels, p)) for p in draws_f]
        ba = [float(balanced_accuracy_score(labels, [v > 0.5 for v in p])) for p in draws_f]
        rows.append({"fraction": f, "trainSubjects": sizes[f], "draws": len(draws_f),
                     "AUC": round(float(np.mean(auc)), 3), "AUCdraws": [round(x, 3) for x in auc],
                     "BA": round(float(np.mean(ba)), 3), "BAdraws": [round(x, 3) for x in ba]})
    return {"source": transfer.split("->")[0], "epochs": epochs, "testSubjects": len(labels),
            "curve": rows}


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, seed: int = 1,
          epochs: int = EPOCHS, draws: int = DRAWS, workdir: str | Path | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    """The learning curve of one source (the source of `transfer`), with the run evidence."""
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
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness, \
            patched(SPEC):
        result = run_source(transfer, cfg, bags, device, seed, int(epochs), int(draws))
    print(json.dumps(result), flush=True)
    record = {"transfer": transfer, "seed": seed, "seconds": round(time.time() - started),
              **result, "environment": environment}
    (workdir / "eegnet-learning-curve.json").write_text(json.dumps(record, default=str),
                                                        encoding="utf-8")
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
