"""Pilot: does trial-level CREDA beat source-only EEGNet across GiGA (Cho et al. 2017) subjects?

Diagnostic outside any plan, default values, no search. Cross-subject, 5 subject folds
(subject i in fold i % 5). Rotation r: target fold r (unlabelled), source = the other four
(labelled). Each target subject's trials split by index parity (no label read): one half adapts
(alignment term only), the other is tested, then the halves swap, so every target trial is
tested once and never by a model that adapted on it. Source-only never reads the target, so
one model tests both halves. Inputs standardized per channel with source statistics only.
Statistics are per subject (a subject's tested trials), paired over subjects.
"""

from __future__ import annotations

import glob
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from CREDA.sample_adaptation import SampleCREDA, train
from MIL_CREDA_EEG import remote_units
from MIL_CREDA_EEG.eegnet import EEGNetTrunk

RESULTS = Path("Results") / "giga-pilot"
FOLDS, EPOCHS, BATCH = 5, 30, 64


def find_data(root: str | None = None) -> str:
    if root and glob.glob(f"{root}/subject_*.npz"):
        return root
    hits = sorted(glob.glob("/kaggle/input/**/MI/2.5-5/subject_1.npz", recursive=True))
    if not hits:
        raise FileNotFoundError("no GiGA subject_*.npz under /kaggle/input")
    return str(Path(hits[0]).parent)


def load(root: str) -> dict:
    out = {}
    for f in sorted(glob.glob(f"{root}/subject_*.npz")):
        with np.load(f) as d:
            out[int(d["subject_id"])] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
    return out


def run(config: Mapping, product: str | Path | None = None, *, rotations: Sequence[int],
        seeds: Sequence[int] = (1,), epochs: int = EPOCHS, data: str | None = None,
        workdir: str | Path | None = None, environ: Mapping[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    product = Path(product) if product else Path.cwd()
    workdir = Path(workdir) if workdir else Path.cwd()
    placement = remote_units.device_placement(config, environ)
    deterministic = remote_units.set_deterministic()
    identity = {"commit": remote_units.source_commit(environ), "revision": "giga-pilot"}
    environment = remote_units.environment_stamp(deterministic, identity, placement)
    dev = torch.device(config.get("device", "cpu"))
    subj = load(find_data(data))
    ids = sorted(subj)
    group = {s: i % FOLDS for i, s in enumerate(ids)}
    started, done = time.time(), []
    for r in rotations:
        path = product / RESULTS / f"rotation{r}.json"
        if path.is_file():
            continue
        tgt = [s for s in ids if group[s] == r]
        src = [s for s in ids if group[s] != r]
        xs = np.concatenate([subj[s][0] for s in src]); ys = np.concatenate([subj[s][1] for s in src])
        m, sd = xs.mean(axis=(0, 2), keepdims=True), xs.std(axis=(0, 2), keepdims=True) + 1e-6
        norm = lambda x: ((x - m) / sd).astype(np.float32)
        xs_t, ys_t = torch.from_numpy(norm(xs)).to(dev), torch.from_numpy(ys).to(dev)
        C, T = xs.shape[1], xs.shape[2]
        rows, log = [], []
        for half in (0, 1):
            xa = np.concatenate([subj[s][0][half::2] for s in tgt])
            xa_t = torch.from_numpy(norm(xa)).to(dev)
            for seed in seeds:
                for arm in ("source-only", "CREDA"):
                    if arm == "source-only" and half == 1:
                        continue
                    torch.manual_seed(seed); np.random.seed(seed)
                    g = torch.Generator().manual_seed(seed)
                    model = SampleCREDA(lambda: EEGNetTrunk(C, T), 2,
                                        adaptation=(arm == "CREDA")).to(dev)
                    t0 = time.time()
                    curve = train(model, xs_t, ys_t, xa_t, epochs=epochs, batch_size=BATCH,
                                  generator=g)
                    model.eval()
                    for s in tgt:
                        for th in ((0, 1) if arm == "source-only" else (1 - half,)):
                            x = torch.from_numpy(norm(subj[s][0][th::2])).to(dev)
                            with torch.no_grad():
                                p = torch.cat([torch.softmax(model.head(model.target_embeddings(
                                    x[i:i + 256])), 1)[:, 1] for i in range(0, len(x), 256)])
                            rows += [{"subject": s, "arm": arm, "seed": seed, "half": th,
                                      "y": int(yy), "p": float(pp)}
                                     for yy, pp in zip(subj[s][1][th::2], p.cpu().numpy())]
                    log.append({"half": half, "seed": seed, "arm": arm,
                                "seconds": round(time.time() - t0, 1), "last": curve[-1]})
                    print(r, half, seed, arm, log[-1]["seconds"], flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rotation": r, "target": tgt, "epochs": epochs,
                                    "rows": rows, "log": log, "environment": environment},
                                   default=float))
        done.append(r)
    timing = {"seconds": round(time.time() - started, 1), "device": environment.get("device")}
    mode = environ.get(remote_units.RUN_MODE_ENV)
    if mode:
        workdir.mkdir(parents=True, exist_ok=True)
        payload = {"mode": mode, "evidence": {"commit": environment.get("commit"),
                                              "unit": f"giga:{list(rotations)}",
                                              "units": [f"giga:{r}" for r in rotations],
                                              "complete": True, "estimates": {}},
                   "environment": environment, "timing": timing, "runs": {"ran": done}}
        (workdir / remote_units.STAMP_FILE).write_text(json.dumps(payload, indent=1, default=str))
    return {"ran": done, "timing": timing}
