"""Merged shard data into what the report notebook can actually read, in scratch.

`shards.merge()` refuses or agrees; it does not shape its answer for a reader.
Its own shape — `shardsArrived`, `gridByEnvironment`, `poolable`,
`perEnvironment`, … — shares no keys with what `Benchmark_Phase1_Report.ipynb`
reads: `summary["reduction"]`, `summary["grid"]`, `summary["gridPerRun"]`,
`summary["perTransfer"]`, plus `runs.jsonl` beside it. This module is the join
between the two. `gridPerRun` travels through untouched — the merge already
refuses to average it, and reshaping it here would just move the overclaim
from one module to the other.

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

#: Kept verbatim for a merge of `run_smoke()` shards. Its prohibition is
#: justified entirely through `run_smoke()` never running a ceiling search —
#: which is exactly the fact that stops holding for a full-campaign merge
#: (see `FULL_CAMPAIGN_PROVENANCE` below and `_classify_provenance`). A
#: structure whose stated reason has expired is corrected, never silently
#: kept, so this note is never reused for shards it does not describe.
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

#: `distribute.py::run_shard` only ever calls `harness.campaign()` at
#: `FULL_EPOCHS` -- it never calls `run_smoke()` -- so a real distributed
#: campaign's shards reach this module too, and `PROVENANCE_NOTE` above
#: would be a factually wrong description of them: it blames a ceiling
#: search that a full campaign actually ran. This is the honest
#: replacement, never a silently dropped safety note: the surviving
#: invariant is that no merge output may be quoted as a campaign result on
#: the strength of the merge tool alone, and this text says so explicitly.
#: Exact text specified by this change's spec; do not alter it.
FULL_CAMPAIGN_PROVENANCE = (
    "This summary merges {shards_arrived} shards produced by full campaigns "
    "(`harness.campaign()` at `FULL_EPOCHS`), not smoke rehearsals. Arms: "
    "{arms}; transfers: {transfers}; epochs per shard: {epochs}; ceiling "
    "search: {ceiling_search}; seeds: {seed_count}. This merge combines real "
    "per-machine results but has not been promoted, scale-checked, or "
    "certified as a canonical benchmark result; it carries no scientific "
    "claim beyond what is stated here on the strength of the merge tool "
    "alone. Promotion into `MIL-CREDA/Results/Benchmark/` remains a manual, "
    "out-of-scope step."
)


def _classify_provenance(
    *, shards_arrived: int, grid: dict, reduction: dict, stamps: list[dict],
) -> tuple[str, str]:
    """`(kind, provenance)` for one merge, derived from the shards actually
    on hand -- never a caller flag.

    `identicalAcrossShards: ["epochs"]` (`shards.declaration()`) means
    `shards.merge()` already refused a shard set whose epochs disagree, so
    every merge that reaches this function agrees on one epoch count: there
    is no mixed smoke/full case to classify, and this needs no argument
    naming which one it is. `epochs >= config.FULL_EPOCHS` (20) is a full
    campaign; `run_smoke()` always runs exactly 2.
    """
    epochs = reduction["epochs"]
    if epochs < config.FULL_EPOCHS:
        return "smokeMerge", PROVENANCE_NOTE

    arms = sorted({arm for cell in grid.values() for arm in cell})
    transfers = sorted(grid)
    # Not part of `identicalAcrossShards`, so read directly off the first
    # shard's own stamp rather than off `reduction` -- an observed fact
    # about this merge, not a mechanically-verified agreement the way
    # `epochs` is.
    observed_ceilings = dict(stamps[0].get("ceilings") or {})
    ceiling_search_present = observed_ceilings != dict(harness.SMOKE_CEILINGS)

    provenance = FULL_CAMPAIGN_PROVENANCE.format(
        shards_arrived=shards_arrived,
        arms=", ".join(arms),
        transfers=", ".join(transfers),
        epochs=epochs,
        ceiling_search="present" if ceiling_search_present else "absent",
        seed_count=len(reduction["seeds"]),
    )
    return "campaignMerge", provenance


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
    # `grid` is the POOLED grid `shards.merge()` returns -- `perRun`
    # dimensions (`seconds`, `peakMiB`) are machine-described and were
    # never averaged into it (see this module's own docstring); reading
    # `ladder_rows` with the full, unfiltered `config.DIMENSIONS` against
    # this grid raises `KeyError` on the first perRun dimension it tries.
    # `ladder_rows` stays strict by default (`tests/test_shards.py`'s own
    # locked "reachable red" proves it), so the caller filters instead.
    poolable_and_per_environment = (
        list(dist.get("poolable") or []) + list(dist.get("perEnvironment") or [])
    )
    merge_dimensions = {
        dimension: better for dimension, better in config.DIMENSIONS.items()
        if dimension in poolable_and_per_environment
    }
    per_transfer = {
        label: verdict.judge(harness.ladder_rows(cell, label, dimensions=merge_dimensions))
        for label, cell in grid.items()
    }

    kind, provenance = _classify_provenance(
        shards_arrived=merged["shardsArrived"], grid=grid, reduction=reduction, stamps=stamps,
    )

    summary = {
        "kind": kind,
        "provenance": provenance,
        "shardsArrived": merged["shardsArrived"],
        "shardsExpected": merged["shardsExpected"],
        "missing": merged["missing"],
        "environments": merged["environments"],
        "reduction": reduction,
        "verdictsMeaningful": merged["verdictsMeaningful"],
        "grid": grid,
        # Every `perRun` reading, untouched and tagged with its own
        # environment — never averaged, and never dropped here the way it
        # would be if this module only forwarded `grid`. Absent from a
        # single-machine `campaign()` summary (there is no such key there),
        # so the report notebook must treat its absence as "nothing to show
        # per run", not as an empty table.
        "gridPerRun": merged["gridPerRun"],
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
    # The SAME text `summary["provenance"]` carries, never `PROVENANCE_NOTE`
    # unconditionally -- a full-campaign merge's marker file must read the
    # honest full-campaign note, not the smoke one.
    provenance_path.write_text(summary["provenance"] + "\n", encoding="utf-8")

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
