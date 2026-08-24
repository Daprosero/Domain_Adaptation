"""Putting several machines' runs back together without pretending they were one.

A campaign split across machines produces records that are identical in shape and
not interchangeable in meaning. Accuracy is a property of the method and pools
freely; wall time and peak memory turned out to be a property of neither — a
same-machine control run (`A` and `C`, both on Trayectoria51) produced three
different values for `seconds` and `peakMiB` across three runs of the identical
arm, transfer and seed. Averaging them across shards would claim a stable value
the method does not have; averaging them per machine would claim a stable value
the machine does not have either.

So the merge builds three shapes rather than one, and that split is what lets
every aggregation function stay exactly as it was for the dimensions it was
always correct for:

    grid              poolable dimensions only, every shard together, averaged
    gridByEnvironment poolable + perEnvironment dimensions, one machine at a
                       time, averaged within that machine
    gridPerRun         perRun dimensions, every run's own reading, never
                       averaged — see the field-by-field reasoning below

`paired_across_transfers` reads only the accuracy dimensions, so the pooled grid
is safe for it by construction. `ladder_rows` iterates every key of
`config.DIMENSIONS` and would raise on a cell with a machine-described dimension
left out; `gridByEnvironment` still carries every `poolable`/`perEnvironment`
dimension for exactly that reason. A `perRun` dimension is deliberately not
among them — see below.

Why `perRun` gets a shape of its own rather than reusing `gridByEnvironment`:
grouping by environment and taking a mean is precisely the framing the control
run disproved for `seconds` and `peakMiB` — it would still read as "this is
what this machine does," a claim about the machine that is not true. `merge()`
instead reports each run's own value, tagged with the shard that produced it:
not a property of the method (it is not pooled), not a property of the machine
(it is not grouped or averaged by environment), and not a bare mean standing in
for either. A caller that wants dispersion can compute a range from the raw
list; `merge()` does not compute one on the caller's behalf, because printing
one invites reading it as the thing itself.

What the merge refuses rather than reconciles: shards that disagree on anything
the declaration lists as having to be identical, or a dimension the declaration
does not classify into `poolable`, `perEnvironment` or `perRun` at all. A
different epoch count is a different experiment, not different hardware, and
silently averaging the two would produce a table nobody could attribute; an
unclassified dimension silently dropped would leave a column nobody notices is
gone.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from MIL_CREDA_Benchmark import config, harness


class ShardsDisagree(SystemExit):
    """Raised instead of merging, and never instead of averaging.

    A `SystemExit` because it ends the merge the way the campaign's other
    refusals end a run: loudly, with what to do, and without leaving a record
    that reads as an answer.
    """


class ShardIncomplete(SystemExit):
    """Raised instead of merging a shard whose stamp is not proven finished.

    A separate exception from `ShardsDisagree`: the shards here do not
    disagree with each other, one of them is simply unsealed — a run that
    died mid-way, or a stamp written before evidence stamping existed. Both
    are refused the same way and for the same reason: what merge produces is
    only as trustworthy as what it was allowed to include.
    """


class PlanConflict(SystemExit):
    """Raised instead of persisting or launching a shard id under seeds that
    disagree with what was already recorded for that same id.

    Neither `ShardsDisagree` nor `ShardIncomplete` fits: the shards here have
    not even run yet, or one of them already has and a relaunch is about to
    re-mint its id. `shard_paths()` keys a shard's entire on-disk home on its
    id string alone, and a relaunch that recomputes ids positionally over a
    failed subset lands back on `s00`. Without this refusal that collision is
    silent: `campaign()` opens `runs.jsonl` `"w"` and rewrites `shard.json`
    unconditionally, discarding a shard that already succeeded.
    """


def declaration() -> dict:
    """The `distribution` block of the benchmark's own declaration."""
    from MIL_CREDA_Benchmark import __benchmark__
    return __benchmark__.get("distribution") or {}


# The generic half of the shard reader — `read_shards`, `disagreements` and
# `completeness` — assumes only the `shard.json` + `runs.jsonl` contract and
# nothing about how runs group into a result cell. That grouping is this
# repository's own vocabulary and lives below. The generic half is vendored
# beside this file rather than reached for across the filesystem: a clone of
# this repository is a paper's implementation and has to stand on its own —
# run its tests, read its results — without a forge checkout above it.
# `tests/test_shard_io_vendoring.py` holds the copy equal to its origin
# whenever that origin is present, and says nothing when it is not.
from MIL_CREDA_Benchmark import shard_io as _shard_io  # noqa: E402

# `disagreements` needs no knowledge this repository would have to supply —
# it operates only on the stamps `read_shards` already collected — so it is
# re-exported as-is, with no wrapper and no logic duplicated.
disagreements = _shard_io.disagreements

# Same reason, same shape: `completeness()` names no field of its own, so it
# is re-exported as-is too. This repository supplies the vocabulary below,
# in `REQUIRED_EVIDENCE`, rather than pushing any field name into the forge.
completeness = _shard_io.completeness

#: What a shard's stamp must carry for `merge()` to accept it: the four
#: `evidence` paths a sealed stamp gains (`harness.write_shard_stamp` /
#: `seal_shard_stamp`), plus the four fields a shard's stamp already carried
#: before evidence stamping existed. An older stamp missing every one of
#: these is `incomplete`, never invalid — `read_shards`/`disagreements` list
#: it exactly as before; only `merge()` refuses it.
REQUIRED_EVIDENCE = [
    "evidence.commit",
    "evidence.codeDigest",
    "evidence.importsFrom",
    "evidence.outputs",
    "environment.device.kind",
    "environment.torch",
    "seeds",
    "epochs",
]


def read_shards(root: Path | None = None) -> list[dict]:
    """Every shard that came back, with its stamp and its raw runs.

    A thin wrapper, not a reimplementation: the only fact this repository
    adds is where its own shards live when the caller does not say —
    `config.RESULTS / harness.SHARDS_DIR` — a location the forge's generic
    reader has no way to know and no business assuming. Everything else,
    including "a shard that never arrived is absent here rather than
    reported as empty", is `shard_io.read_shards`'s behavior, unchanged.
    """
    home = root if root is not None else config.RESULTS / harness.SHARDS_DIR
    return _shard_io.read_shards(home)


def split_complete(found: list[dict],
                    required: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Partition `read_shards()` output into sealed-complete and incomplete,
    before anything is handed to `merge()`.

    Deliberately asymmetric: `complete` entries are the original
    `{"shard", "stamp", "runs"}` dicts, unmodified and directly passable to
    `merge()`; `incomplete` entries are reshaped to `{"id", "missing"}` —
    never the same shape. Same shape would let an incomplete entry reach
    `merge()` by a caller's mistake and be pooled as an unproven run; a
    different shape turns that mistake into a `KeyError` at the boundary
    instead. `merge()` itself is not called here — this is the same
    `completeness()` check it performs internally (line 224), lifted out so
    a caller can act on the split before `merge()` ever aborts on it.
    """
    required = required if required is not None else REQUIRED_EVIDENCE
    complete: list[dict] = []
    incomplete: list[dict] = []
    for entry in found:
        result = completeness(entry["stamp"], required)
        if result["complete"]:
            complete.append(entry)
        else:
            incomplete.append({"id": entry["shard"], "missing": result["missing"]})
    return complete, incomplete


def plan_path(root: Path | None = None) -> Path:
    """Where a campaign's plan lives: a sibling of every shard directory,
    never one itself — `shard_io.read_shards()` iterates `iterdir()` filtered
    to `is_dir()`, so this file is structurally invisible to the shard
    reader.
    """
    home = root if root is not None else config.RESULTS / harness.SHARDS_DIR
    return home / "plan.json"


def read_plan(root: Path | None = None) -> dict | None:
    """The recorded plan, or `None` when none was ever written.

    `None`, never `{}` and never a plan synthesized from what arrived: a
    missing plan is an honest "nobody recorded intent," and a caller that
    treated it as an empty plan or invented one from the shards that showed
    up would be unable to tell "nothing was ever launched" apart from
    "everything launched came back."
    """
    path = plan_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_plan(record: dict, root: Path | None = None) -> Path:
    """Record `record["shards"]` into the plan, amendable and union-by-id.

    Identical `id -> seeds` is idempotent — it only refreshes `writtenAt`.
    A new `id` is appended. A conflicting `id` — the same id recorded once
    already, now mapped to different seeds — refuses outright: this is the
    one guard standing between a naive relaunch and silently overwriting a
    shard that already succeeded, since `harness.shard_paths()` keys that
    shard's entire on-disk home on the id string alone. `epochs` and
    `accelerator` are last-write-wins — they describe the campaign, not any
    one shard's identity, and `merge()`'s own `identicalAcrossShards` check
    is what actually holds `epochs` to account, against the stamps
    themselves rather than against intent.
    """
    path = plan_path(root)
    existing = read_plan(root) or {}
    existing_shards = {entry["id"]: entry for entry in existing.get("shards", [])}

    for entry in record.get("shards", []):
        prior = existing_shards.get(entry["id"])
        if prior is not None and list(prior["seeds"]) != list(entry["seeds"]):
            raise PlanConflict(
                f"refusing to record shard {entry['id']!r} under seeds "
                f"{list(entry['seeds'])!r}: already recorded under "
                f"{list(prior['seeds'])!r}.\n"
                f"  {entry['id']!r} may already hold a succeeded shard on disk; "
                "relaunching it under a different seed list would silently "
                "overwrite that directory. Relaunch the recorded id under its "
                "own seeds, or mint a genuinely new id for the new seeds."
            )
        existing_shards[entry["id"]] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "shards": list(existing_shards.values()),
        "epochs": record.get("epochs", existing.get("epochs")),
        "accelerator": record.get("accelerator", existing.get("accelerator")),
        "writtenAt": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    return path


def partition(dimensions: dict, dist: dict) -> tuple[list[str], list[str], list[str]]:
    """Which dimensions pool across machines, which are read one machine at a
    time, and which belong to no wider scope than the run that produced them.

    Read from the declaration and never decided here. This module does not know
    what any dimension measures, and a merge that guessed would be choosing which
    of somebody's results are comparable.
    """
    poolable = list(dist.get("poolable") or [])
    per_environment = list(dist.get("perEnvironment") or [])
    per_run = list(dist.get("perRun") or [])
    missing = [d for d in dimensions if d not in poolable + per_environment + per_run]
    if missing:
        raise ShardsDisagree(
            "refusing to merge: the distribution declaration does not say whether "
            f"these dimensions pool across machines: {', '.join(sorted(missing))}.\n"
            "  Every dimension belongs to exactly one of `poolable`, "
            "`perEnvironment` or `perRun`; a dimension in none of them would be "
            "silently dropped."
        )
    return poolable, per_environment, per_run


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


def _grid_per_run(runs: list[dict], dimensions: list[str]) -> dict:
    """`{transfer: {arm: {dimension: [{"env", "seed", "value"}, ...]}}}`.

    Deliberately not `harness.spread`: a `perRun` dimension has no mean this
    module is willing to hand a caller, because the replication that created
    this category found none stable enough to report as one — not across
    shards, and not even across two runs of the same shard's own machine.
    Every reading keeps the shard's environment and seed, which is what a
    caller needs to tell "the method's own value" and "the machine's own
    value" apart from "one execution's own value" instead of losing that
    distinction to an average.
    """
    built: dict[str, dict] = defaultdict(dict)
    for (transfer, arm), cell in _cells(runs).items():
        built[transfer][arm] = {
            d: [{"env": run.get("env", "unknown"), "seed": run["seed"],
                 "value": float(run[d])} for run in cell]
            for d in dimensions
        }
    return dict(built)


def merge(shards: list[dict], expected: int | None = None,
          dimensions: dict | None = None, dist: dict | None = None) -> dict:
    """Three shapes, the shards that arrived, and a refusal when they disagree.

    The pooled grid carries only what pools. The per-environment grid carries
    every `poolable`/`perEnvironment` dimension, one machine at a time, which is
    what keeps `ladder_rows` and `judge` unchanged for those — they iterate all
    of `DIMENSIONS` and a redacted cell would raise rather than be skipped. A
    `perRun` dimension is never redacted into either shape silently — it goes to
    `gridPerRun` instead, as every run's own untouched reading, because grouping
    it by environment would repeat the same overclaim `perEnvironment` itself
    was rejected for.

    Scale is recomputed from what arrived and never from what was intended. Three
    shards planned and two returned is a smaller campaign, not a failed one, and
    the record says which seeds it actually rests on so nobody reads the intended
    number off the configuration.
    """
    dimensions = dimensions if dimensions is not None else config.DIMENSIONS
    dist = dist if dist is not None else declaration()

    # Refused before anything else is even read from the shard: an unsealed
    # or pre-evidence stamp is not proven finished, and a merge that pooled
    # it in anyway would be reporting on a run nobody can vouch for.
    for entry in shards:
        result = completeness(entry["stamp"], REQUIRED_EVIDENCE)
        if not result["complete"]:
            raise ShardIncomplete(
                f"refusing to merge shard {entry['shard']!r}: missing "
                f"{', '.join(result['missing'])}.\n"
                "  A shard whose run died mid-way is never sealed, and a "
                "stamp written before evidence stamping existed was never "
                "proven finished either way. Re-run the shard, or fetch its "
                "sealed stamp again."
            )

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

    poolable, per_environment, per_run = partition(dimensions, dist)
    runs = [run for entry in shards for run in entry["runs"]]

    by_environment: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_environment[run.get("env", "unknown")].append(run)

    # `gridByEnvironment` groups by machine, so only the two shapes that are
    # legitimately read that way — pooled across the whole campaign, or one
    # machine at a time — go into it. `perRun` dimensions are excluded on
    # purpose: presenting them here, even split by environment, would still
    # read as "this machine's value," which is exactly the claim the
    # replication's control run (same machine, three different readings)
    # disproved.
    by_environment_dims = list(poolable) + list(per_environment)

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
        "gridByEnvironment": {key: _grid(group, by_environment_dims)
                              for key, group in by_environment.items()},
        "gridPerRun": _grid_per_run(runs, per_run),
        "poolable": poolable,
        "perEnvironment": per_environment,
        "perRun": per_run,
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
