"""Phase-1 variants on one ADHD rotation (protocol v81 discussion): A = full TPE search with the v80 ranges for
lr1, l2, width, epochs and tauAtt1 log [0.01, 1]; B = tauAtt1 log [0.01, 1] only, the other settings fixed at the
v80 choice of that rotation. 20 trials (sampler seed SAMPLER_SEED + rotation), then the frozen seed-1 base of both
halves. Writes phase1-<variant>-r<r>.json and the two bases under `workdir`."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from MIL_CREDA_EEG import ieee_pool as I, spectral_search as SS, spectral_trunk, xpilot as XP

V80_SPACE = {"lr1": ("log", 1e-4, 1e-2), "l2": ("log", 1e-4, 1e-1), "width": ("cat", (32, 64, 128)),
             "tauAtt1": ("log", 0.01, 1.0), "epochs": ("cat", (50, 100, 200))}
CONFIG = {"device": "cpu", "nTrials": 20, "rehearsal": True,
          "remoteInputs": {"processed": "daprosero/balladeer-existing-local"},
          "schedule": {"bagsPerStep": 8, "epochs1": 100, "epochs2": 100},
          "strategies": ["AdaptMIL", "UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS",
                         "CLUST — variante source-free", "CLUDA", "MIL-CREDA"], "transfers": ["L1->L6"]}


def run(rotation: int, variant: str, workdir=None, trials: int = 20, threads: int = 4) -> dict:
    torch.set_num_threads(int(threads))
    out = Path(workdir) if workdir else Path.cwd(); out.mkdir(parents=True, exist_ok=True)
    r = int(rotation)
    with spectral_trunk.installed():
        fixed = dict(torch.load(I.base_file(f"ieee-r{r}h0"), weights_only=False)["phase1"])
    if variant == "A":
        SS.SPACE.clear(); SS.SPACE.update(V80_SPACE)
    elif variant == "B":
        SS.suggest = lambda trial: {**fixed, "tauAtt1": trial.suggest_float("tauAtt1", 0.01, 1.0, log=True)}
    else:
        raise ValueError(f"variant is A or B; got {variant!r}")
    t0 = time.time(); b = I._build(CONFIG, r, 0)
    with spectral_trunk.installed(), XP.window_samples(b["samples"]), XP.full_trace_mode(False):
        params, score = XP.search_phase1(b["sides"], b["cfg"], torch.device("cpu"), int(trials), r)
    params = {**fixed, **params} if variant == "B" else params
    rec = {"variant": variant, "rotation": r, "params": params, "score": score, "v80": fixed,
           "seconds": round(time.time() - t0, 1)}
    (out / f"phase1-{variant}-r{r}.json").write_text(json.dumps(rec, indent=1))
    for h in (0, 1):
        I.build_base(r, h, params, CONFIG, out / variant)
    return rec
