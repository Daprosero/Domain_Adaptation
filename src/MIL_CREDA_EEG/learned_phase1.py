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


def _blocks(raw, filtered=True):
    """The "blocks" arm: 50%-overlap windows cut into ordered blocks of 15
    (v64_pilot.ordered_blocks); each block is one training unit. `filtered` adds the 1-45 Hz
    band-pass and common average reference; False keeps the stored 0.2-45 Hz windows."""
    from MIL_CREDA_EEG import v64_pilot
    return [blk for b in raw for blk in v64_pilot.ordered_blocks(b, None, filtered=filtered)]


def _stored_whole(bags):
    """The "whole" arm on the stored 0.2-45 Hz windows: at most WINDOWS evenly spaced per bag."""
    return [two_phase.BagTensor(b.subject, b.condition, b.label,
                                b.windows[::max(1, len(b.windows) // sc.WINDOWS)].float())
            for b in bags]


def euclidean_aligned(units):
    """Euclidean alignment (He & Wu, IEEE TBME 2020) per real subject, label-free: every unit of a
    subject (its bag, or all its blocks) is re-centred by that subject's mean window covariance,
    X -> R^{-1/2} X, as tools/diagnostics/ea_check.py does; test subjects use their own windows."""
    by = {}
    for u in units:
        by.setdefault(u.subject.split("#")[0], []).append(u)
    inv = {}
    for subject, us in by.items():
        x = torch.cat([u.windows for u in us]).double()
        R = torch.einsum("nct,ndt->cd", x, x) / (x.shape[0] * x.shape[2])
        w, V = torch.linalg.eigh(R + 1e-6 * torch.eye(R.shape[0], dtype=R.dtype) * R.diagonal().mean())
        inv[subject] = V @ torch.diag(w.clamp_min(1e-12) ** -0.5) @ V.T
    return [two_phase.BagTensor(u.subject, u.condition, u.label,
                                torch.einsum("cd,ndt->nct", inv[u.subject.split("#")[0]],
                                             u.windows.double()).float()) for u in units]


@contextlib.contextmanager
def _trunk(kind):
    if kind == "spectral":
        from MIL_CREDA_EEG import spectral_trunk
        with spectral_trunk.installed():
            yield
    else:
        with learned_trunks.installed(kind):
            yield


def run_source(transfer, cfg, bags, device, kind, seed, epochs, arm="whole", ea=False,
               stored=False):
    a = A.assign_from_bags({t.name: t for t in A.TRANSFERS}[transfer], bags)
    hyper = two_phase.check_hyper({**HYPER, "epochs1": epochs, "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    y, p = [], []
    for r in range(4):
        raw = lambda role: A.side_bags(a, bags, "source", r, role)
        if arm == "blocks":
            prep = lambda role: _blocks(raw(role), filtered=not stored)
        elif stored:
            prep = lambda role: _stored_whole(two_phase.bag_tensors(raw(role)))
        else:
            prep = lambda role: sc.prepared(two_phase.bag_tensors(raw(role)))
        side = (lambda role: euclidean_aligned(prep(role))) if ea else prep
        train, test = side(A.REFIT_ROLE), side("test")
        with _trunk(kind):
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            c = int(train[0].windows.shape[1])
            m = model_store.build_model("MIL-CREDA", c, c, hyper).to(device)
            m.source_trunk.norm = (sc.standard_of(m.source_trunk, train) if kind == "spectral"
                                   else learned_trunks.input_standard_of(m.source_trunk, train))
            renyi = learned_trunks.renyi_loss() if kind == "tgarstf" else contextlib.nullcontext()
            with renyi, explicit_l2(sc.L2):
                two_phase.train_source_phase(m, train, [], hyper, g, device, seed=seed, fold=r, transfer=transfer)
            q = np.nan_to_num(two_phase.evaluate(m, test, "source", device)[:, 1].detach().cpu().numpy(), nan=.5)
            per = {}                          # a subject's score is the mean over its bags or blocks
            for b, v in zip(test, q):
                per.setdefault(b.subject.split("#")[0], [b.label, []])[1].append(float(v))
            y += [lab for lab, _ in per.values()]
            p += [float(np.mean(vs)) for _, vs in per.values()]
        print(kind, transfer, seed, "rotation", r, "done", flush=True)
    return {"source": transfer.split("->")[0], "kind": kind, "arm": arm, "ea": bool(ea),
            "filter": "stored 0.2-45 Hz" if stored else "1-45 Hz + CAR", "seed": seed, "epochs": epochs,
            "predictedPositive": round(float(np.mean(np.asarray(p) > .5)), 3),
            "AUC": round(float(roc_auc_score(y, p)), 3),
            "BA": round(float(balanced_accuracy_score(y, np.asarray(p) > .5)), 3), "n": len(y)}


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, kind: str, seed: int = 1,
          epochs: int = 100, arms=("whole",), eas=(False,), stored: bool = False,
          workdir: str | Path | None = None,
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
        results = []
        for arm in arms:
            for ea in eas:
                results.append(run_source(transfer, cfg, bags, device, kind, int(seed), int(epochs),
                                          arm, bool(ea), bool(stored)))
            print(json.dumps(results[-1]), flush=True)
    result = results[0] if len(results) == 1 else {"arms": results}
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
