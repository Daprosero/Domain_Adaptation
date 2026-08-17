"""Putting several machines' runs back together without pretending they were one.

A campaign split across machines produces records that are identical in shape and
not interchangeable in meaning. Accuracy is a property of the method and pools
freely; wall time and peak memory are properties of the machine, and averaging
them across two of them yields a number that describes neither.

So the merge builds two grids rather than one, and that split is what lets every
aggregation function stay exactly as it was:

    grid              poolable dimensions only, every shard together
    gridByEnvironment all dimensions, one machine at a time

`paired_across_transfers` reads only the accuracy dimensions, so the pooled grid
is safe for it by construction. `ladder_rows` iterates every key of
`config.DIMENSIONS` and would raise on a cell with a machine-described dimension
left out — giving it one environment at a time means every dimension is
legitimately present, and with a single environment its output is what it always
was. Nothing in `harness` had to change to accommodate this.

What the merge refuses rather than reconciles: shards that disagree on anything
the declaration lists as having to be identical. A different epoch count is a
different experiment, not different hardware, and silently averaging the two
would produce a table nobody could attribute.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from MIL_CREDA_Benchmark import config, harness


class ShardsDisagree(SystemExit):
    """Raised instead of merging, and never instead of averaging.

    A `SystemExit` because it ends the merge the way the campaign's other
    refusals end a run: loudly, with what to do, and without leaving a record
    that reads as an answer.
    """


def declaration() -> dict:
    """The `distribution` block of the benchmark's own declaration."""
    from MIL_CREDA_Benchmark import __benchmark__
    return __benchmark__.get("distribution") or {}


def read_shards(root: Path | None = None) -> list[dict]:
    """Every shard that came back, with its stamp and its raw runs.

    Enumerated from the disk. A shard that never arrived is absent here rather
    than reported as empty, and the difference matters: `merge` counts what it
    expected against what it has, and one of those numbers has to come from
    somewhere other than the pile of files that did show up.
    """
    home = (root if root is not None
            else config.RESULTS / harness.SHARDS_DIR)
    if not home.is_dir():
        return []
    found = []
    for folder in sorted(p for p in home.iterdir() if p.is_dir()):
        stamp_path = folder / "shard.json"
        runs_path = folder / "runs.jsonl"
        if not stamp_path.exists():
            continue
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        runs = [json.loads(line) for line in
                runs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if runs_path.exists() else []
        found.append({"shard": folder.name, "stamp": stamp, "runs": runs})
    return found


def disagreements(shards: list[dict], fields: list[str]) -> list[dict]:
    """Where the shards do not match on what they were required to match on.

    Reported per field with every value seen, because "they disagree" is not
    actionable and "one shard ran twenty epochs and two ran three" is.
    """
    found = []
    for field in fields:
        seen = {}
        for entry in shards:
            seen.setdefault(json.dumps(entry["stamp"].get(field), sort_keys=True),
                            []).append(entry["shard"])
        if len(seen) > 1:
            found.append({"field": field,
                          "values": {value: names for value, names in seen.items()}})
    return found


def partition(dimensions: dict, dist: dict) -> tuple[list[str], list[str]]:
    """Which dimensions pool across machines and which are read one machine at a time.

    Read from the declaration and never decided here. This module does not know
    what any dimension measures, and a merge that guessed would be choosing which
    of somebody's results are comparable.
    """
    poolable = list(dist.get("poolable") or [])
    per_environment = list(dist.get("perEnvironment") or [])
    missing = [d for d in dimensions if d not in poolable + per_environment]
    if missing:
        raise ShardsDisagree(
            "refusing to merge: the distribution declaration does not say whether "
            f"these dimensions pool across machines: {', '.join(sorted(missing))}.\n"
            "  Every dimension belongs to exactly one of `poolable` or "
            "`perEnvironment`; a dimension in neither would be silently dropped."
        )
    return poolable, per_environment


def _cells(runs: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["transfer"], run["arm"])].append(run)
    return grouped


def _grid(runs: list[dict], dimensions: list[str]) -> dict:
    """`{transfer: {arm: {dimension: spread}}}` over the dimensions asked for."""
    built: dict[str, dict] = defaultdict(dict)
    for (transfer, arm), cell in _cells(runs).items():
        built[transfer][arm] = {
            d: harness.spread([float(r[d]) for r in cell]) for d in dimensions
        }
    return dict(built)


def merge(shards: list[dict], expected: int | None = None,
          dimensions: dict | None = None, dist: dict | None = None) -> dict:
    """Two grids, the shards that arrived, and a refusal when they disagree.

    The pooled grid carries only what pools. The per-environment grids carry
    every dimension, one machine at a time, which is what keeps `ladder_rows` and
    `judge` unchanged — they iterate all of `DIMENSIONS` and a redacted cell would
    raise rather than be skipped.

    Scale is recomputed from what arrived and never from what was intended. Three
    shards planned and two returned is a smaller campaign, not a failed one, and
    the record says which seeds it actually rests on so nobody reads the intended
    number off the configuration.
    """
    dimensions = dimensions if dimensions is not None else config.DIMENSIONS
    dist = dist if dist is not None else declaration()

    identical = list(dist.get("identicalAcrossShards") or [])
    clashes = disagreements(shards, identical)
    if clashes:
        detail = "; ".join(f"{c['field']}: {list(c['values'].values())}" for c in clashes)
        raise ShardsDisagree(
            f"refusing to merge shards that disagree on {detail}.\n"
            "  These were declared as having to be identical across shards, so a "
            "difference is a different experiment rather than different hardware, "
            "and averaging them would produce a table nobody can attribute."
        )

    poolable, per_environment = partition(dimensions, dist)
    runs = [run for entry in shards for run in entry["runs"]]

    by_environment: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_environment[run.get("env", "unknown")].append(run)

    seeds = sorted({run["seed"] for run in runs})
    arrived = [entry["shard"] for entry in shards]
    return {
        "shardsArrived": arrived,
        "shardsExpected": expected if expected is not None else len(arrived),
        "missing": max(0, (expected or len(arrived)) - len(arrived)),
        "environments": {entry["stamp"].get("env"): entry["stamp"].get("environment")
                         for entry in shards},
        "seeds": seeds,
        # Recomputed from what arrived. The configuration says what was asked
        # for; only the runs say what the verdict actually rests on.
        "verdictsMeaningful": len(seeds) >= 3,
        "grid": _grid(runs, poolable),
        "gridByEnvironment": {key: _grid(group, list(dimensions))
                              for key, group in by_environment.items()},
        "poolable": poolable,
        "perEnvironment": per_environment,
    }


def promote_candidates(runs: list[dict], arrived: set) -> dict:
    """Which checkpoints the centre keeps, chosen over every measurement.

    The median is computed across *all* the runs, because measurements were never
    pruned — only weights were. Only the choice is restricted to the checkpoints
    that actually came back.

    That ordering matters. Recomputing the median over the arrived candidates
    would pick the median of the survivors and call it the median of the
    campaign; this picks the campaign's median and then takes the nearest thing
    to it that exists. Where fewer arrived than `CHECKPOINTS` asks for, the
    shortfall is reported rather than quietly filled from further out.
    """
    kept: dict = {}
    for (transfer, arm), cell in _cells(runs).items():
        wanted = harness.median_seeds(cell, arm)
        here = sorted(seed for seed in wanted if (arm, transfer, seed) in arrived)
        kept[f"{arm}|{transfer}"] = {
            "chosen": here,
            "wanted": sorted(wanted),
            "shortfall": len(wanted) - len(here),
        }
    return kept
