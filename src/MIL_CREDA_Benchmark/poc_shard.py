"""Run one shard of a PoC battery into a job cache the notebook can reuse.

    python -m MIL_CREDA_Benchmark.poc_shard --shard i --shards n --out DIR [--workers W]
        [--battery full|k2k3|k2k3v2|k2k3v3] [--prereg PATH]

Shard `i` of `n` runs jobs `i::n` of the chosen battery, read from the
pre-registration: `full` (default) is `poc.battery_jobs`, run 3's 240 jobs;
`k2k3` is `poc.k2k3_jobs`, run 4's 60 ladder jobs with the gating readings
(pass run 4's `--prereg`); `k2k3v2` is `poc.k2k3v2_jobs`, the same ladder
under run 5's subject signature (pass run 5's `--prereg`, with its `k3Rate`
set); `k2k3v3` is `poc.k2k3v3_jobs`, run 6's ladder under the room scenario
with the K2 push reading (pass run 6's `--prereg`). The slice runs through `poc.run_many` with
`MIL_CREDA_POC_CACHE=DIR`, so a finished job is never run twice and the
notebook's battery cell reads the union of every shard's `DIR` as cache hits. The cache key includes `src`: pass
`--src` with the notebook's own `SRC` string when the shard runs elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from MIL_CREDA_Benchmark import poc

SRC = str(Path(__file__).resolve().parents[1])
#: The batteries a shard may slice, by name.
BATTERIES = {"full": poc.battery_jobs, "k2k3": poc.k2k3_jobs, "k2k3v2": poc.k2k3v2_jobs,
             "k2k3v3": poc.k2k3v3_jobs}

PREREGISTRATION = (Path(SRC).parent / "MIL-CREDA" / "Results" / "PoC_Conditions"
                   / "preregistration.json")


def shard_indices(total: int, shard: int, shards: int) -> range:
    """The job positions shard `shard` of `shards` runs: `shard::shards`."""
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} is not in 0..{shards - 1}")
    return range(shard, total, shards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m MIL_CREDA_Benchmark.poc_shard",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prereg", type=Path, default=PREREGISTRATION)
    parser.add_argument("--src", default=SRC)
    parser.add_argument("--battery", choices=sorted(BATTERIES), default="full")
    args = parser.parse_args(argv)
    jobs = BATTERIES[args.battery](json.loads(args.prereg.read_text()))
    mine = [jobs[i] for i in shard_indices(len(jobs), args.shard, args.shards)]
    os.environ[poc.CACHE_ENV] = str(args.out)
    start = time.perf_counter()
    poc.run_many(mine, args.workers, args.src)
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(jobs)} jobs in "
          f"{time.perf_counter() - start:.1f} s -> {args.out}")
    return 0


def run(shard: int, shards: int, battery: str = "k2k3", workers: int = 4,
        prereg: str | None = None, out: str | None = None) -> int:
    """Callable entry for a remote job: run one shard into a cache folder inside
    the repository (`MIL-CREDA/Results/<prereg folder>/cache/`), so the job's
    returned outputs carry it."""
    prereg_path = Path(prereg) if prereg else PREREGISTRATION
    if not prereg_path.is_absolute():
        prereg_path = Path(SRC).parent / prereg_path
    target = Path(out) if out else prereg_path.parent / "cache"
    if not target.is_absolute():
        target = Path(SRC).parent / target
    target.mkdir(parents=True, exist_ok=True)
    return main(["--shard", str(shard), "--shards", str(shards), "--out", str(target),
                 "--workers", str(workers), "--battery", battery,
                 "--prereg", str(prereg_path)])


if __name__ == "__main__":
    raise SystemExit(main())
