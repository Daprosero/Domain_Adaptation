"""One campaign unit in one notebook: MIL-CREDA and the seven comparators on the same bags.

The bags, the subject-wise partition and the phase-1 search are computed ONCE and handed to
`xpilot.smoke` (MIL-CREDA, phase-2 search on) and `xcompare.smoke` (the comparators), so every
method starts from the same searched phase 1. Each step prints a progress line and appends a row
to `progress.jsonl` (`progress.Progress`); the remaining-time estimate weights each step by its
seconds measured in the timing probe of 2026-11-12 (one sim unit, Kaggle CPU) and rescales by the
rate this notebook actually shows.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import torch

from MIL_CREDA_EEG import campaign, remote_units, spectral_trunk
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import progress as PG
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

# seconds per step measured in the timing probe (timing_phase2_diagnostics.json); search
# entries are per trial (study seconds / 20 trials), seed entries per report seed. MIL-CREDA's
# seed is light (full_trace False, qm-lite 2026-11-12: 108 s) minus the shared base (~55 s)
PROBE_SECONDS = {
    "p1": 57.0, "milcreda-search": 38.0, "milcreda-seed": 53.0, "base": 55.0,
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
    total = trials * w("p1") + trials2 * w("milcreda-search") + n * (w("milcreda-seed") + w("base"))
    run = [m for m in methods if m not in XC.SKIPPED]
    if run:
        total += sum(trials2 * w(f"search:{m}") + n * w(f"seed:{m}") for m in run)
    return total


def job_seconds(job: str, trials2: int, seeds: Sequence) -> float:
    """Expected seconds of one parallel job: MIL-CREDA's phase-2 search and seeds, or one
    comparator's search, seeds and its own per-seed bases (each job fits its own)."""
    w = lambda k: PROBE_SECONDS.get(k, DEFAULT_SECONDS)
    n = len(seeds)
    if job == "MIL-CREDA":
        return trials2 * w("milcreda-search") + n * w("milcreda-seed")
    return trials2 * w(f"search:{job}") + n * (w(f"seed:{job}") + w("base"))


def _phase1(sides, cfg, device, trials, rotation, samples):
    with spectral_trunk.installed(), XP.window_samples(samples):
        return XP.search_phase1(sides, cfg, device, trials, rotation)


def _phase1_child():
    """The shared phase-1 search in a spawned child (all cores)."""
    return _phase1(*_FORKED["phase1_args"])


_FORKED: dict = {}        # each spawned child's copy of the parent's payload (set by _init)


def _settings() -> dict:
    """The search spaces and phase-2 defaults a child must share with the parent."""
    return {"space1": dict(XP.SS.SPACE), "space2": dict(XP.SPACE2), "phase2": dict(P.PHASE_TWO)}


def _init(payload: dict) -> None:
    """Spawned-child initializer: the parent's payload, and the parent's exact settings."""
    _FORKED.clear()
    _FORKED.update(payload)
    st = payload["settings"]
    XP.SS.SPACE.clear(); XP.SS.SPACE.update(st["space1"])
    XP.SPACE2.clear(); XP.SPACE2.update(st["space2"])
    P.PHASE_TWO.clear(); P.PHASE_TWO.update(st["phase2"])
    torch.set_num_threads(int(payload.get("threads", 1)))


def _job(job: str) -> dict:
    """One parallel job in a spawned child, one torch thread, its own output folder."""
    f = _FORKED
    os.environ["MIL_CREDA_THREADS"] = "1"
    torch.set_num_threads(1)
    sub = Path(f["workdir"]) / "parallel" / job.split(" ")[0].replace("+", "_")
    sub.mkdir(parents=True, exist_ok=True)
    PG.activate(PG.Progress(f"{f['name']}/{job.split(' ')[0]}",
                            job_seconds(job, f["trials2"], f["seeds"]), f["workdir"]), PROBE_SECONDS)
    t0 = time.time()
    common = {**f["common"], "workdir": sub}
    try:
        if job == "MIL-CREDA":
            XP.smoke(f["config"], f["product"], search2=True, full_trace=f["full_trace"], **common)
        else:
            XC.smoke(f["config"], f["product"], methods=(job,), **common)
        error = None
    except Exception as e:                                  # one job failing must not lose the others
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"
    return {"job": job, "dir": str(sub), "seconds": round(time.time() - t0, 1), "error": error}


def _merge(workdir: Path, results: list, methods: Sequence[str], skipped: dict) -> dict:
    """Move MIL-CREDA's file up and fold the comparators' per-job files into one xcompare file."""
    merged, xp_file = None, None
    for r in results:
        sub = Path(r["dir"])
        for f in sub.glob("xpilot-*.json"):
            xp_file = workdir / f.name
            f.replace(xp_file)
        for f in sub.glob("xcompare-*.json"):
            rec = json.loads(f.read_text())
            if merged is None:
                merged, name = rec, f.name
            else:
                merged["methods"].update(rec["methods"])
                merged["timing"]["baseSeconds"].update(
                    {f"{r['job']}:{k}": v for k, v in rec["timing"]["baseSeconds"].items()})
        stamp = sub / remote_units.STAMP_FILE
        if stamp.exists():
            stamp.replace(workdir / remote_units.STAMP_FILE)
    out = None
    if merged is not None:
        for m in methods:
            if m in skipped:
                merged["methods"][m] = {"skipped": skipped[m]}
            elif m not in merged["methods"]:
                failed = next((r["error"] for r in results if r["job"] == m), None)
                merged["methods"][m] = {"failed": failed or "no output"}
        merged["parallel"] = {"processes": PARALLEL_PROCESSES, "jobs": results}
        out = workdir / name
        out.write_text(json.dumps(merged, default=str), encoding="utf-8")
    return {"xpilot": str(xp_file) if xp_file else None, "xcompare": str(out) if out else None}


PARALLEL_PROCESSES = 4


def run(config: Mapping, product=None, *, seeds: Sequence[int] = (1,), trials: int = 20, trials2: int = 20,
        methods: Sequence[str] = XC.ALL8, workdir=None, environ: Mapping[str, str] | None = None,
        parallel: int = 1, full_trace: bool = True, **unit) -> dict:
    """`parallel` > 1 runs MIL-CREDA and each comparator as separate jobs over that many forked
    processes of one torch thread each, after the shared phase-1 search; 1 runs them in sequence."""
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
        args = (built["sides"], built["cfg"], device, trials, unit.get("rotation", 0), built["samples"])
        if int(parallel) <= 1:
            phase1 = _phase1(*args)
        else:   # spawn, never fork: torch's autograd and OpenMP pools are not fork-safe
            payload = {"phase1_args": args, "settings": _settings(), "threads": os.cpu_count() or 1}
            with mp.get_context("spawn").Pool(1, initializer=_init, initargs=(payload,)) as pool:
                phase1 = pool.apply(_phase1_child)
        shared = {"built": built, "phase1": phase1}
        common = dict(seeds=seeds, trials=trials, trials2=trials2, workdir=workdir, environ=environ,
                      shared=shared, **unit)
        if int(parallel) <= 1:
            bases = {}

            def base_of(sd):            # one phase-1 base and cut per seed, shared by all 8 methods
                if sd not in bases:
                    bases[sd] = XC.shared_base(built["sides"], built["cfg"], device, sd,
                                               unit.get("rotation", 0), phase1[0], name)
                    PG.tick("base", f"shared base seed {sd}")
                return bases[sd]

            common["shared"] = {**shared, "base_of": base_of}
            mil = XP.smoke(config, product, search2=True, full_trace=full_trace, **common)
            cmp = XC.smoke(config, product, methods=methods, **common)
            files = {"xpilot": mil, "xcompare": cmp.get("file")}
        else:
            jobs = ["MIL-CREDA"] + [m for m in methods if m not in XC.SKIPPED]
            jobs.sort(key=lambda j: -job_seconds(j, trials2, seeds))          # longest first
            payload = {"config": config, "product": product, "common": {**common, "environ": dict(environ)},
                       "workdir": str(workdir), "name": name, "trials2": trials2, "seeds": list(seeds),
                       "settings": _settings(), "threads": 1, "full_trace": bool(full_trace)}
            print(f"[{name}] {len(jobs)} jobs over {int(parallel)} processes", flush=True)
            with mp.get_context("spawn").Pool(int(parallel), initializer=_init, initargs=(payload,),
                                              maxtasksperchild=1) as pool:
                results = []
                for r in pool.imap_unordered(_job, jobs):
                    results.append(r)
                    print(f"[{name}] job {r['job']} done in {r['seconds'] / 60:.1f} min"
                          f"{' FAILED ' + r['error'] if r['error'] else ''}", flush=True)
            files = _merge(workdir, results, methods, XC.SKIPPED)
    finally:
        PG.activate(None)
    print(f"[{name}] unit done in {(time.time() - started) / 60:.1f} min", flush=True)
    return {"ran": [name], "seconds": round(time.time() - started), "parallel": int(parallel), "files": files}
