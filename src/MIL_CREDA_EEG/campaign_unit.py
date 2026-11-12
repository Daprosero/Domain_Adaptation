"""One campaign unit in one notebook: MIL-CREDA and the seven comparators on the same bags.

The bags, the subject-wise partition and the phase-1 search are computed ONCE and handed to
`xpilot.smoke` (MIL-CREDA, phase-2 search on) and `xcompare.smoke` (the comparators), so every
method starts from the same searched phase 1. Each step prints a progress line and appends a row
to `progress.jsonl` (`progress.Progress`); the remaining-time estimate weights each step by its
seconds measured in the timing probe of 2026-11-12 (one sim unit, Kaggle CPU) and rescales by the
rate this notebook actually shows.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from MIL_CREDA_EEG import campaign, remote_units, spectral_trunk
from MIL_CREDA_EEG import progress as PG
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

# seconds per step measured in the timing probe (timing_phase2_diagnostics.json); search
# entries are per trial (study seconds / 20 trials), seed entries per report seed
PROBE_SECONDS = {
    "p1": 57.0, "milcreda-search": 38.0, "milcreda-seed": 158.0, "base": 31.0,
    "search:UDA-DDA": 20.0, "seed:UDA-DDA": 22.0,
    "search:CCA-LSC": 17.0, "seed:CCA-LSC": 20.0,
    "search:CASUAL": 29.0, "seed:CASUAL": 15.0,
    "search:COT": 14.0, "seed:COT": 20.0,
    "search:CS+CCS": 12.0, "seed:CS+CCS": 12.0,
    "search:CLUST — variante source-free": 12.0, "seed:CLUST — variante source-free": 18.0,
    "search:CLUDA": 13.0, "seed:CLUDA": 7.0,
}
DEFAULT_SECONDS = 15.0

BUILD_KEYS = ("experiment", "rotation", "half", "direction", "condition", "setting", "effect", "shift",
              "n_subjects", "sim_seed", "d_target", "artifact_rate", "artifact_class_corr",
              "event_fraction", "paired", "spread")


def planned_work(trials: int, trials2: int, seeds: Sequence, methods: Sequence[str]) -> float:
    """Expected seconds of the declared steps, the denominator of the progress estimate."""
    w = lambda k: PROBE_SECONDS.get(k, DEFAULT_SECONDS)
    n = len(seeds)
    total = trials * w("p1") + trials2 * w("milcreda-search") + n * w("milcreda-seed")
    run = [m for m in methods if m not in XC.SKIPPED]
    if run:
        total += n * w("base")
        total += sum(trials2 * w(f"search:{m}") + n * w(f"seed:{m}") for m in run)
    return total


def run(config: Mapping, product=None, *, seeds: Sequence[int] = (1,), trials: int = 20, trials2: int = 20,
        methods: Sequence[str] = XC.ALL8, workdir=None, environ: Mapping[str, str] | None = None,
        **unit) -> dict:
    unknown = [k for k in unit if k not in BUILD_KEYS]
    if unknown:
        raise ValueError(f"unit keys are among {BUILD_KEYS}; got {unknown}")
    environ = os.environ if environ is None else environ
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    started = time.time()
    cfg = remote_units.resolve_inputs(campaign.check_config(config), environ=environ)
    built = XP.build_sides(cfg, environ, **unit)
    name = built["unit"]
    progress = PG.Progress(name, planned_work(trials, trials2, seeds, methods), workdir)
    PG.activate(progress, PROBE_SECONDS)
    try:
        print(f"[{name}] bags built in {time.time() - started:.0f} s", flush=True)
        device = torch.device(built["cfg"].get("device", "cpu"))
        with spectral_trunk.installed(), XP.window_samples(built["samples"]):
            phase1 = XP.search_phase1(built["sides"], built["cfg"], device, trials, unit.get("rotation", 0))
        shared = {"built": built, "phase1": phase1}
        common = dict(seeds=seeds, trials=trials, trials2=trials2, workdir=workdir, environ=environ,
                      shared=shared, **unit)
        mil = XP.smoke(config, product, search2=True, **common)
        cmp = XC.smoke(config, product, methods=methods, **common)
    finally:
        PG.activate(None)
    print(f"[{name}] unit done in {(time.time() - started) / 60:.1f} min", flush=True)
    return {"ran": [name], "seconds": round(time.time() - started),
            "milcreda": mil, "comparators": cmp.get("file")}
