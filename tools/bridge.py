"""Merged shard data into what the report notebook can actually read, in scratch.

`shards.merge()` refuses or agrees; it does not shape its answer for a reader.
Its own shape — `shardsArrived`, `gridByEnvironment`, `poolable`,
`perEnvironment`, … — shares no keys with what `Benchmark_Phase1_Report.ipynb`
reads: `summary["reduction"]`, `summary["grid"]`, `summary["perTransfer"]`, plus
`runs.jsonl` beside it. This module is the join between the two.

**Never `config.RESULTS`.** `harness.run_smoke()`'s own docstring is explicit:
skipping the ceiling search is not a scientific claim, it is a statement that a
smoke run is plumbing, not a result, and "a smoke run never writes
`summary.json` or `Probe_results.json`". A merge of shards that each came from
`run_smoke()` is exactly that, five times over — so this module writes
somewhere a real campaign's record never lives, and says so from inside every
file it writes, not only from its own name.

**The grid carries only what was actually classified — which, after the
replication, is everything.** `MIL_CREDA_Benchmark.__benchmark__["distribution"]`
used to leave five of `config.DIMENSIONS` unclassified; a replication
(`A`/`B`/`C`) settled all of them, so every one of the eight now sits in
`poolable`, `perEnvironment` or `perRun`. `build_summary` therefore hands
`shards.merge()` `config.DIMENSIONS` in full rather than a subset narrowed to
whatever the declaration happened to cover — the narrowing existed only to
paper over the gap, and the gap is gone. `unclassifiedDimensions` is still
computed and still written, but it is no longer a report of what was left
out: on any declaration that actually covers `config.DIMENSIONS`, `merge()`
refuses before `build_summary` ever gets this far (see `shards.partition()`),
so the field can only read empty on a summary that exists at all — a receipt
that nothing was silently dropped, not a list of gaps. `runs.jsonl` is
unaffected — it carries every field each shard wrote, untouched, because the
notebook's own tables read most dimensions straight off `runs`, never off
`summary["grid"]`.

**The reduction carries only what merging actually proved.** `identicalAcrossShards`
is the one thing `shards.merge()` mechanically checks equal across every shard
that arrived; the reduction it hands the report is built from exactly that set,
plus the union of the seeds that arrived — never from a field a smoke shard's
stamp does not carry, and never filled in with a full run's own defaults
pretending to have been measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from MIL_CREDA_Benchmark import config, harness, shards, verdict  # noqa: E402

#: Where a bridge's output lands — never under `MIL-CREDA/Results`, and named
#: so a reader three months from now cannot mistake it for that. Ignored by
#: git for the same reason `.benchmark-data/` is: it is reproducible from the
#: shards on disk, not part of this repository's record.
SCRATCH = REPOSITORY / ".scratch" / "distributed-smoke-merge"

PROVENANCE_NOTE = (
    "PLUMBING, NOT A RESULT.\n\n"
    "This directory holds the output of tools/bridge.py, merging shards each "
    "produced by MIL_CREDA_Benchmark.harness.run_smoke() -- one arm, one "
    "transfer, two epochs, no ceiling search. run_smoke()'s own docstring is "
    "explicit that skipping the search is not a scientific claim about where "
    "either family's ceiling sits: it is a statement that the run is "
    "plumbing. This merge exists to prove the report notebook can consume "
    "what a distributed run produces, and carries no scientific claim of its "
    "own.\n\n"
    "Never copy these files into MIL-CREDA/Results/Benchmark/, and never "
    "quote a number from them as a campaign result."
)


def build_summary(found: list[dict], dist: dict | None = None) -> tuple[dict, list[dict]]:
    """The campaign()-shaped `summary` and the flat `runs` list beside it.

    `dist` defaults to the module's own real `shards.declaration()`; a caller
    may pass a different one (tests do), but the shape this function builds
    stays exactly as data-driven as `shards.merge()` itself is.

    Raises exactly what `shards.merge()` raises — `ShardsDisagree` or
    `ShardIncomplete` — unweakened: this function reshapes a merge that
    succeeded, and refuses nothing of its own.
    """
    dist = dist if dist is not None else shards.declaration()
    classified = (list(dist.get("poolable") or []) + list(dist.get("perEnvironment") or [])
                  + list(dist.get("perRun") or []))
    # `config.DIMENSIONS` in full, unrestricted: with every dimension now
    # classified into `poolable`, `perEnvironment` or `perRun`, there is no
    # gap left to narrow around. A `dist` that still leaves one out is not
    # papered over here — `shards.merge()`'s own `partition()` refuses it,
    # exactly as it would refuse a shard that disagrees on
    # `identicalAcrossShards` or is not sealed.
    merged = shards.merge(found, dimensions=config.DIMENSIONS, dist=dist)

    identical = list(dist.get("identicalAcrossShards") or [])
    stamps = [entry["stamp"] for entry in found]
    reduction = {field: stamps[0][field] for field in identical if field in stamps[0]}
    reduction["seeds"] = merged["seeds"]

    grid = merged["grid"]
    per_transfer = {label: verdict.judge(harness.ladder_rows(cell, label))
                     for label, cell in grid.items()}

    summary = {
        "kind": "smokeMerge",
        "provenance": PROVENANCE_NOTE,
        "shardsArrived": merged["shardsArrived"],
        "shardsExpected": merged["shardsExpected"],
        "missing": merged["missing"],
        "environments": merged["environments"],
        "reduction": reduction,
        "verdictsMeaningful": merged["verdictsMeaningful"],
        "grid": grid,
        "perTransfer": per_transfer,
        "tally": {label: verdict.tally(rows) for label, rows in per_transfer.items()},
        "panorama": harness.paired_across_transfers(grid),
        # Computed rather than assumed empty: `merge()` above already refused
        # if `dist` left anything out, so on any summary that exists at all
        # this is `[]` — a receipt that every dimension `config.DIMENSIONS`
        # measures had a place, not a report of one that didn't.
        # `checkpoints` is left off `summary` entirely rather than written
        # empty — a smoke run keeps no weights, so there was never a
        # selection to report, empty or otherwise.
        "unclassifiedDimensions": sorted(set(config.DIMENSIONS) - set(classified)),
    }
    runs = [run for entry in found for run in entry["runs"]]
    return summary, runs


def write_scratch(summary: dict, runs: list[dict], root: Path | None = None) -> dict:
    """The summary, the runs, and a plain-text marker, all under `root`."""
    root = root if root is not None else SCRATCH
    root.mkdir(parents=True, exist_ok=True)

    summary_path = root / "summary.json"
    runs_path = root / "runs.jsonl"
    provenance_path = root / "PROVENANCE.txt"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with runs_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run) + "\n")
    provenance_path.write_text(PROVENANCE_NOTE + "\n", encoding="utf-8")

    return {"summary": summary_path, "runs": runs_path, "provenance": provenance_path}


def bridge(found: list[dict] | None = None, root: Path | None = None) -> dict:
    """Read what came back, build the report's shape, write it to scratch."""
    found = found if found is not None else shards.read_shards()
    if not found:
        raise SystemExit("no shards came back yet; nothing to bridge")
    summary, runs = build_summary(found)
    return write_scratch(summary, runs, root)


def main() -> int:
    written = bridge()
    print(f"written to {written['summary'].parent}:")
    for name, path in written.items():
        print(f"  {name}: {path.relative_to(REPOSITORY)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
