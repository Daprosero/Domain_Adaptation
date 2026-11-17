"""The phase-1 recipe that generalized in every source, with phase 2 on top. Diagnostic.

Outside plan v60: it reports no result of the plan and writes nothing to the model store.
Windows are filtered to 1-45 Hz and re-referenced to their common average, as the band-power
check that generalized; at most `WINDOWS` evenly spaced windows per bag. The spectral encoder
(`spectral_trunk`) reads log absolute and relative band power per channel, standardized by
each domain's own training windows (label-free for the target), then a learned linear map.
Phase 1: near-uniform attention, explicit L2, lr 3e-3, 200 epochs. Phase 2: sigma recomputed
on the trained network, balanced pseudo-labels, lr 1e-3, 20 epochs. No target label chooses.
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
from scipy.signal import butter, sosfiltfilt
from torch import nn

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import campaign, model_store, remote_units, spectral_trunk, two_phase
from MIL_CREDA_EEG.patience_exercise import GpuWitness
from MIL_CREDA_EEG.phase_one_sweep import _side, _summary, explicit_l2
from MIL_CREDA_EEG.phase_two_terms import balanced_pseudolabels

WINDOWS, L2 = 60, 1e-2
HYPER = {"lr1": 3e-3, "wd1": 0.0, "attentionGamma": 0.0, "attentionTemperature": 10.0,
         "sigmaMultiplier": 1.0, "lr2": 1e-3, "wd2": 1e-5, "tauLocal": 1.0,
         "epochs1": 200, "epochs2": 20}
SOS = butter(4, [1, 45], "bandpass", fs=128, output="sos")


class Standard(nn.Module):
    """Fixed feature standardization from one domain's training windows."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def prepared(bags) -> list:
    """1-45 Hz, common average reference, at most WINDOWS evenly spaced windows per bag."""
    out = []
    for b in bags:
        w = b.windows.double().numpy()
        w = w[::max(1, len(w) // WINDOWS)]
        w = sosfiltfilt(SOS, w - w.mean(-1, keepdims=True), axis=-1)
        w = w - w.mean(1, keepdims=True)
        out.append(two_phase.BagTensor(b.subject, b.condition, b.label, torch.from_numpy(w.copy()).float()))
    return out


def standard_of(trunk, bags) -> Standard:
    with torch.no_grad():
        device = trunk.masks.device              # the windows live on the CPU until read
        f = torch.nan_to_num(torch.cat([trunk.features(b.windows.to(device)) for b in bags]))
    return Standard(f.mean(0), f.std(0) + 1e-6)


def run_rotation(name: str, rotation: int, cfg, a, bags, device, seed: int) -> dict:
    side = lambda s, role: prepared(two_phase.bag_tensors(A.side_bags(a, bags, s, rotation, role)))
    source, held = side("source", A.REFIT_ROLE), side("source", "test")
    target, test = side("target", A.REFIT_ROLE), side("target", "test")
    same = source[0].windows.shape[1] == target[0].windows.shape[1]
    hyper = two_phase.check_hyper({**HYPER, "bagsPerStep": cfg["schedule"]["bagsPerStep"]})
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = model_store.build_model("MIL-CREDA", int(source[0].windows.shape[1]),
                                    int(target[0].windows.shape[1]), hyper).to(device)
    model.source_trunk.norm = standard_of(model.source_trunk, source).to(device)
    model.target_trunk.norm = (model.source_trunk.norm if same
                               else standard_of(model.target_trunk, target).to(device))
    common = {"seed": seed, "fold": rotation, "transfer": name}
    with explicit_l2(L2):
        phase1 = two_phase.train_source_phase(model, source, [], hyper, generator, device, **common)
    median = two_phase.within_bag_median_distance(model, source, "source", device)
    if median > 0:
        model._sigma = hyper["sigmaMultiplier"] * median
    phase1["checkpoint"]["sigma"] = model._sigma
    rows = {"train": _side(model, source, device), "source": _side(model, held, device)}
    if same:
        rows["predict"] = _side(model, test, device)
    joint = two_phase.bridge(model)
    if same:                 # the target trunk reads its own domain's statistics
        model.target_trunk.norm = standard_of(model.target_trunk, target).to(device)
    with balanced_pseudolabels(model):
        two_phase.adapt_target_phase(model, phase1, joint["frozen_state"], source, target, [], hyper,
                                     generator, device, **common)
    p = two_phase.evaluate(model, test, "target", device)[:, 1].detach().cpu().numpy()
    rows["milCreda"] = [{"subject": b.subject, "label": b.label, "probability": float(q)}
                        for b, q in zip(test, np.nan_to_num(p, nan=0.5))]
    return rows


def smoke(config: Mapping, product: str | Path | None = None, *, transfer: str, seed: int = 1,
          workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
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
    with GpuWitness(two_phase, torch) if mode else contextlib.nullcontext() as witness, \
            spectral_trunk.installed():
        for rotation in range(4):
            rows = run_rotation(transfer, rotation, cfg, a, bags, device, seed)
            for k, v in rows.items():
                pooled[k] += v
            print(transfer, seed, rotation, {k: _summary(v) for k, v in rows.items()}, flush=True)
    pooled = {k: v for k, v in pooled.items() if v}
    record = {"transfer": transfer, "seed": seed, "seconds": round(time.time() - started),
              **{k: _summary(v) for k, v in pooled.items()}, "rows": pooled, "environment": environment}
    (workdir / "spectral-check.json").write_text(json.dumps(record, default=str), encoding="utf-8")
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
