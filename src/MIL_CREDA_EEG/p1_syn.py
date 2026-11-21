"""Phase 1 of the synthetic shared pool under the final rule (TPE over spectral_search.SPACE: lr1, l2, width and
epochs with their v80 ranges, tauAtt1 log [0.01, 1]; 20 trials, sampler seed SAMPLER_SEED + rotation), for one
class effect d and rotation: the search runs on the shift-1.5 unit's sides (the source does not depend on the
shift), then the frozen seed-1 base of the three shifts, in make_bases.py's layout. Each unit's config and
simulator settings are read from its v80 base in Manifests/oracle_bases."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import torch

from MIL_CREDA_EEG import campaign, oracle_screen as O, remote_units, spectral_trunk, xcompare as XC, xpilot as XP


def _unit(key):
    with spectral_trunk.installed():
        b = torch.load(O.BASES / f"{key}-s1.pt", weights_only=False)
    return b["config"], b["unitKwargs"]


def _built(key, r):
    config, kw = _unit(key)
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ={})
    return config, kw, XP.build_sides(cfg, {}, experiment="sim", rotation=r, half=0, **kw)


def run(d: str, rotation: int, workdir=None, trials: int = 20, threads: int = 4) -> dict:
    torch.set_num_threads(int(threads))
    out = Path(workdir) if workdir else Path.cwd(); out.mkdir(parents=True, exist_ok=True)
    r = int(rotation); t0 = time.time()
    _, _, b0 = _built(f"d{d}-s1p5-r{r}", r)
    with spectral_trunk.installed(), XP.window_samples(b0["samples"]), XP.full_trace_mode(False):
        params, score = XP.search_phase1(b0["sides"], b0["cfg"], torch.device("cpu"), int(trials), r)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(XC.__file__)).stdout.strip()
    keys = [f"d{d}-s{s}-r{r}" for s in ("0p5", "1p5", "3")]
    for key in keys:
        config, kw, bk = _built(key, r)
        with spectral_trunk.installed(), XP.window_samples(bk["samples"]), XP.full_trace_mode(False):
            b = XC.shared_base(bk["sides"], bk["cfg"], torch.device("cpu"), 1, r, params, bk["unit"])
        torch.save({"unitKey": key, "unit": bk["unit"], "seed": 1, "rotation": r, "phase1": params, "unitKwargs": kw,
                    "config": config, "cut": float(b["cut"]), "arm": b["arm"], "context": b["context"],
                    "sourceTest": b["sourceTest"], "predict": b["predict"], "commit": commit, "torch": torch.__version__,
                    "base": b["base"], "prepared": b["prepared"]}, out / f"{key}-s1.pt")
    rec = {"d": d, "rotation": r, "units": keys, "params": params, "score": score, "seconds": round(time.time() - t0, 1)}
    (out / f"phase1-d{d}-r{r}.json").write_text(json.dumps(rec, indent=1))
    return rec
