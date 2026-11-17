"""Does EEGNet's phase 1 need more epochs to generalize to unseen source subjects?

Diagnostic, outside the plan's protocol: it reports no result of the plan and writes nothing to
the model store. The recipe is the one of `subbag_check` (inputs standardized by the source
training statistics, head max-norm lifted, explicit L2, lr 1e-2, BatchNorm recalibrated before
every reading); only the number of epochs changes. Every epoch is read twice:

- the pooled inner source validation (each outer rotation's inner splits), which is label-free
  with respect to the target and could therefore choose the epoch;
- the unseen source test of the refit on every non-test fold (diagnostic only, chooses nothing).
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
from MIL_CREDA_EEG import campaign, model_store, remote_units, search, two_phase
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import explicit_l2, patched, recalibrate, standardized
from MIL_CREDA_EEG.subbag_check import FIXED, L2, SPEC

EPOCHS = 500
READ_AT = (50, 100, 200, 500)


class _Reader:
    """Epoch control: after every epoch, recalibrate BatchNorm and score the held-out bags."""

    def __init__(self, model, train, held, device):
        self.model, self.train, self.held, self.device = model, train, held, device
        self.probabilities: list[list[float]] = []

    def boundary(self, epoch: int) -> bool:
        modes = {m: m.training for m in self.model.modules()}
        recalibrate(self.model, self.train, self.device)
        p = two_phase.evaluate(self.model, self.held, "source", self.device)[:, 1]
        self.probabilities.append(np.nan_to_num(p.detach().cpu().numpy(), nan=0.5).tolist())
        for m, mode in modes.items():
            m.train(mode)
        return True

    def collect(self, record) -> None:
        return None


def curve(train, held, hyper, device, seed: int, rotation: int, transfer: str) -> list:
    """One training of `hyper['epochs1']` epochs; per epoch the probabilities on `held`."""
    train, held = standardized(train, held)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    channels = int(train[0].windows.shape[1])
    model = model_store.build_model("MIL-CREDA", channels, channels, hyper).to(device)
    reader = _Reader(model, train, held, device)
    with explicit_l2(L2), two_phase.epoch_control(reader):
        two_phase.train_source_phase(model, train, [], hyper, generator, device,
                                     seed=seed, fold=rotation, transfer=transfer)
    return reader.probabilities


def scores(labels: list, per_epoch: list) -> list:
    """Per epoch (AUC, balanced accuracy at 0.5) of the pooled predictions."""
    out = []
    for p in per_epoch:
        out.append((float(roc_auc_score(labels, p)),
                    float(balanced_accuracy_score(labels, [v > 0.5 for v in p]))))
    return out


def run_source(transfer: str, cfg, bags, device, seed: int, epochs: int, inner: int) -> dict:
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[transfer], bags)
    hyper = two_phase.check_hyper({**FIXED, "epochs1": epochs, "epochs2": 1,
                                   "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    test_y, test_p, valid_y, valid_p = [], None, [], None
    for r in range(4):
        side = lambda role, **k: two_phase.bag_tensors(A.side_bags(a, bags, "source", r, role, **k))
        held = side("test")
        p = curve(side(A.REFIT_ROLE), held, hyper, device, seed, r, transfer)
        test_y += [b.label for b in held]
        test_p = p if test_p is None else [x + y for x, y in zip(test_p, p)]
        for s in range(inner):
            valid = side("inner_validation", split=s)
            q = curve(side("inner_train", split=s), valid, hyper, device, seed, r, transfer)
            valid_y += [b.label for b in valid]
            valid_p = q if valid_p is None else [x + y for x, y in zip(valid_p, q)]
        print(transfer, seed, "rotation", r, "done", flush=True)
    test, valid = scores(test_y, test_p), scores(valid_y, valid_p)
    pick = int(np.argmax([v[0] for v in valid]))
    best = int(np.argmax([t[0] for t in test]))
    at = lambda c, e: {"epoch": e + 1, "AUC": round(c[e][0], 3), "BA": round(c[e][1], 3)}
    summary = {"source": transfer.split("->")[0],
               "test": [at(test, e - 1) for e in READ_AT if e <= epochs],
               "testBest": at(test, best),
               "innerValid": [at(valid, e - 1) for e in READ_AT if e <= epochs],
               "innerValidBest": at(valid, pick),
               "testAtInnerChosenEpoch": at(test, pick)}
    return {"summary": summary, "testCurve": test, "innerValidCurve": valid}


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, seed: int = 1,
          epochs: int = EPOCHS, inner: int = search.N_INNER, workdir: str | Path | None = None,
          environ: Mapping[str, str] | None = None) -> dict:
    """The epoch curves of one source (the source of `transfer`), with the run evidence."""
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
        result = run_source(transfer, cfg, bags, device, seed, int(epochs), int(inner))
    print(json.dumps(result["summary"]), flush=True)
    record = {"transfer": transfer, "seed": seed, "epochs": int(epochs), "inner": int(inner),
              "seconds": round(time.time() - started), **result, "environment": environment}
    (workdir / "eegnet-epochs.json").write_text(json.dumps(record, default=str), encoding="utf-8")
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
