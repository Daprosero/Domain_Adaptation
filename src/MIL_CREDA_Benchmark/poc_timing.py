"""Timing probe for the PoC battery: one job of each kind, timed where it runs.

    python -m MIL_CREDA_Benchmark.poc_timing [--out PATH] [--smoke]

Times one job of each of the battery's four kinds in a single process, then
the same four kinds side by side on one worker per CPU, and writes the record
as JSON. It never runs the battery: it only measures what one job costs on
the machine it runs on, so the placement plan can compare machines.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch

from MIL_CREDA_Benchmark import poc

SRC = str(Path(__file__).resolve().parents[1])
PREREGISTRATION = (Path(SRC).parent / "MIL-CREDA" / "Results" / "PoC_Conditions"
                   / "preregistration.json")
DEFAULT_OUT = Path(SRC).parent / "MIL-CREDA" / "Results" / "PoC_Timing" / "probe.json"
KINDS = ("ladder", "ladderArtifact", "k1Donor", "k1Artifact")


def probe_jobs(prereg: dict) -> dict[str, tuple]:
    """The first job of each kind in the battery, at the middle contamination rate."""
    battery = poc.battery_jobs(prereg)
    seed, rate = prereg["seeds"][0], prereg["contaminationRates"][len(prereg["contaminationRates"]) // 2]

    def first(function, contamination, with_rate):
        for job in battery:
            name, args, kwargs = job
            if name != function or kwargs["conditions"].contamination != contamination:
                continue
            if function == "run_ladder_seed" and ("rate" in kwargs) != with_rate:
                continue
            if args[0] == seed and (function == "run_ladder_seed" or args[1] == rate):
                return job
        raise LookupError(f"no {function} {contamination} job in the battery")

    return {"ladder": first("run_ladder_seed", "donor", False),
            "ladderArtifact": first("run_ladder_seed", "artifact", True),
            "k1Donor": first("run_contamination_seed", "donor", None),
            "k1Artifact": first("run_contamination_seed", "artifact", None)}


def _time_one(job: tuple) -> float:
    start = time.perf_counter()
    poc._call(job)
    return time.perf_counter() - start


def _time_parallel(jobs: list[tuple], workers: int) -> float:
    start = time.perf_counter()
    poc.run_many(jobs, workers, SRC)
    return time.perf_counter() - start


def run(out: str | None = None, smoke: bool = False) -> dict:
    """Time the probe jobs and write the record to `out`; returns the record."""
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    jobs = probe_jobs(prereg)
    kinds = KINDS[:1] if smoke else KINDS
    torch.set_num_threads(1)
    single = {kind: _time_one(jobs[kind]) for kind in kinds}
    workers = max(1, os.cpu_count() or 1)
    parallel = None
    if not smoke:
        batch = [jobs[kind] for kind in kinds]
        parallel = {"workers": workers, "jobs": len(batch),
                    "wallSeconds": _time_parallel(batch, workers)}
    record = {"schema": 1, "smoke": smoke, "cpuCount": os.cpu_count(),
              "platform": platform.platform(), "torch": torch.__version__,
              "cudaAvailable": bool(torch.cuda.is_available()),
              "singleProcessSeconds": single, "parallel": parallel}
    path = Path(out) if out else DEFAULT_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.out, args.smoke), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
