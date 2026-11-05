"""Run one shard of run 3's battery into a job cache the notebook can reuse.

    python -m MIL_CREDA_Benchmark.poc_shard --shard i --shards n --out DIR [--workers W]

Shard `i` of `n` runs jobs `i::n` of `poc.battery_jobs` (read from the
pre-registration) through `poc.run_many` with `MIL_CREDA_POC_CACHE=DIR`, so a
finished job is never run twice and the notebook's battery cell reads the
union of every shard's `DIR` as cache hits. The cache key includes `src`: pass
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
    args = parser.parse_args(argv)
    jobs = poc.battery_jobs(json.loads(args.prereg.read_text()))
    mine = [jobs[i] for i in shard_indices(len(jobs), args.shard, args.shards)]
    os.environ[poc.CACHE_ENV] = str(args.out)
    start = time.perf_counter()
    poc.run_many(mine, args.workers, args.src)
    print(f"shard {args.shard}/{args.shards}: {len(mine)} of {len(jobs)} jobs in "
          f"{time.perf_counter() - start:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
