"""Planning a campaign across several remote sessions, and never running one blind.

This is the only file that knows a remote service exists. Nothing under `src/`
does, and neither does the skill: the benchmark declares that its runs may be
distributed and along which axis, and how that distribution actually happens is
this repository's business and this file's alone.

It lives in `tools/` for the reason `tools/` exists. It implements no equation,
so it cannot sit beside the method without declaring a provenance it has no
right to; it neither trains nor measures, so it does not belong in the benchmark
package either; and it cannot stay untracked, because then the configuration of
a run costing hours would live on one disk and no later session could reproduce
how it was launched.

**Credentials are a path here and never a value.** The store is never opened,
never printed, never parsed — not even indirectly: worker enumeration and the
capacity clamp are the forge's own `remote-execution` skill's job now.
`adapters/kaggle.py` runs the accounts CLI's own `list`, built to report
usernames and never keys; `packer.py` clamps what this file declares it wants
against what each worker states it allows. What stays genuinely this
repository's to say — the seed axis a shard is, the accelerator asked for, how
much of a worker it wants at once, and `run_shard()`'s own local execution
path — stays here, and only here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools"))

from MIL_CREDA_Benchmark import config, harness, shards  # noqa: E402
import bridge  # noqa: E402

#: The accelerator every session asks for. One class for every shard, because
#: `seconds` and `peakMiB` are dimensions of the verdict and two arms measured on
#: two GPU classes differ in something nobody declared.
#:
#: Asking is not receiving. A service allocates by availability, so the request
#: lives here and the fact lives in the stamp: `environment()` records what
#: arrived, and the merge groups by that rather than by this.
ACCELERATOR = "NvidiaTeslaT4"

#: The forge root, two levels above this file: `<root>/implementations/
#: Domain_Adaptation/tools/distribute.py`.
_FORGE_ROOT = REPOSITORY.parents[1]
_REMOTE_EXECUTION_SCRIPTS = (
    _FORGE_ROOT / ".claude" / "skills" / "remote-execution" / "scripts"
)


def _load_forge_module(module_name: str, relative_path: str):
    """Path-import one module from the forge's `remote-execution` skill,
    reusing an already-loaded copy under `module_name` when one exists.

    Same technique, and the same correctness reason, as that skill's own
    `packer.py`/`remote_cli.py` reusing each other's sibling loads:
    `packer.plan()`'s `isinstance(adapter, ADAPTER.Adapter)` check has to be
    made against the SAME `Adapter` class the loaded `adapters/kaggle.py`
    subclasses, not a second, separately exec'd copy of `adapter.py` that
    merely shares its name. Loading `packer.py` first, so it is the one that
    first puts `"remote_execution_adapter"` into `sys.modules`, is what makes
    the later Kaggle adapter load reuse that exact copy instead of its own.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    script = _REMOTE_EXECUTION_SCRIPTS / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PACKER = _load_forge_module("remote_execution_packer", "packer.py")
# Side effect only: executing adapters/kaggle.py registers "kaggle" into
# PACKER.ADAPTER's own registry. Nothing here keeps a direct reference to
# it — `capacity_plans()` below reaches the class back out through
# `PACKER.ADAPTER.resolve("kaggle")`, the same route `remote_cli.py`'s own
# `main()` uses to select a backend by name.
_load_forge_module("remote_execution_adapter_kaggle", "adapters/kaggle.py")

#: What this repository asks of every worker, concurrently — its own
#: declared want, and no opinion about what a worker will actually allow.
#: Two, mirroring the batch-session figure the forge's Kaggle adapter
#: itself observes for a single account's concurrent submissions, so
#: this repository now asks a worker for as much concurrency as that
#: account is known to offer, rather than under-asking against it. Each
#: concurrent submission still runs its own shard start to finish on its
#: own machine — `shard_seeds()` already guarantees that — so this
#: number only ever governs how many separate shards may be in flight
#: against the same worker at once, never a split within one shard.
#: Like the adapter's own figure, this is an observed property of the
#: service rather than one this repository derives independently, and
#: it is documented and expected to be revised as that allowance
#: changes — never asserted as a universal per-worker ceiling. The cap
#: this actually gets clamped against remains `adapter.workers()`'s own
#: fact, read fresh by `packer.plan()` below — never read or asserted
#: here.
REQUESTED_CAPACITY_PER_WORKER = 2


def capacity_plans() -> list["PACKER.Plan"] | None:
    """One clamped `Plan` per worker the Kaggle adapter currently reports.

    Replaces this file's own former `account_count()`, which re-derived a
    worker count through its own subprocess call and asked for no clamp at
    all. Worker enumeration now happens exactly once inside
    `adapter.workers()` — the same sanctioned `accounts_cli.py list --json`
    command `account_count()` used to run itself, moved to the one file in
    the forge's skill permitted to name a service — and the clamp happens
    exactly once inside `packer.plan()`. This function duplicates neither;
    it only asks each in turn, once per worker.

    Returns `None`, never a guess, when workers cannot be enumerated at
    all — everything `KaggleAdapterError` already refuses to fabricate past,
    surfaced here as the same `None` `account_count()` returned on the same
    kind of failure. A caller sizing a plan on this falls back to one shard,
    exactly as it always has.
    """
    adapter = PACKER.ADAPTER.resolve("kaggle")()
    try:
        workers = adapter.workers()
    except PACKER.ADAPTER.AdapterError:
        return None
    return [
        PACKER.plan(
            adapter=adapter,
            worker_id=worker.id,
            requested=REQUESTED_CAPACITY_PER_WORKER,
            ledger_lines=(),
            # No ledger lines means the fold never dereferences this value —
            # a plan here is a dry run before any notebook is even chosen,
            # so there is no source tree yet for a real digest to describe.
            live_digest="",
        )
        for worker in workers
    ]


def shard_seeds(seeds: list[int], parts: int) -> list[list[int]]:
    """Split the axis into shards, contiguously and without losing one.

    Contiguous rather than round-robin so a shard's identity reads off its seeds,
    and a missing shard leaves a gap somebody can name instead of a scatter.

    The axis is the seed because a seed is a whole repetition: every arm of every
    transfer within it runs on one machine, so no comparison is ever split across
    two. Sharding by arm would put the ladder's own subtraction across a hardware
    boundary, which is the one thing the distribution declaration forbids.
    """
    if parts < 1:
        raise SystemExit("a campaign cannot be split into fewer than one shard")
    parts = min(parts, len(seeds))
    size, extra = divmod(len(seeds), parts)
    out, start = [], 0
    for index in range(parts):
        span = size + (1 if index < extra else 0)
        out.append(seeds[start:start + span])
        start += span
    return out


def plan(parts: int | None = None, seeds: list[int] | None = None) -> dict:
    """What would be launched, said out loud before anything is.

    A plan and not a launch. The cost comes from what the pilot actually
    measured, so it is a projection from data rather than an estimate from
    memory, and it is reported without a threshold: whether it is worth
    distributing is not a question this file gets to answer.

    `requested`, `cap`, `inFlight` and `granted` are kept as four separate
    numbers per worker, exactly as `packer.plan()` itself reports them: a
    clamp collapsed down to one number would be indistinguishable from an
    unclamped grant that merely happens to equal it.
    """
    seeds = list(seeds if seeds is not None else config.FULL_SEEDS)
    worker_capacity = capacity_plans()
    available = (
        sum(one.granted for one in worker_capacity)
        if worker_capacity is not None else None
    )
    parts = parts or available or 1
    groups = shard_seeds(seeds, parts)
    return {
        "accelerator": ACCELERATOR,
        "capacity": {
            "requestedPerWorker": REQUESTED_CAPACITY_PER_WORKER,
            "granted": available,
            "workers": [
                {
                    "worker": one.worker,
                    "requested": one.requested,
                    "cap": one.cap,
                    "inFlight": one.in_flight,
                    "granted": one.granted,
                }
                for one in (worker_capacity or [])
            ],
        },
        "axis": (shards.declaration() or {}).get("axis"),
        "shards": [{"id": f"s{index:02d}", "seeds": group}
                   for index, group in enumerate(groups)],
        "epochs": config.FULL_EPOCHS,
        "note": ("The accelerator is requested, not guaranteed. Each shard stamps "
                 "what it received and the merge groups cost by that stamp, so a "
                 "session that landed elsewhere is visible rather than averaged in."),
    }


def run_shard(shard: str, seeds: list[int]) -> dict:
    """One shard's campaign, here, writing only into its own namespace."""
    device = harness.resolve_device()
    reduction = harness.Reduction(
        seeds=list(seeds), epochs=config.FULL_EPOCHS,
        device=str(device), environment=harness.environment())
    reduction = replace(reduction, ceilings=harness.ceilings_in_force(reduction, device))
    return harness.campaign(reduction, device, shard=shard)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="what would be launched, and nothing else")
    p_plan.add_argument("--shards", type=int, default=None)

    p_run = sub.add_parser("run", help="run one shard here, into its own namespace")
    p_run.add_argument("--shard", required=True)
    p_run.add_argument("--seeds", required=True,
                       help="comma-separated, as `plan` printed them")

    sub.add_parser("merge", help="put the shards that came back together")

    args = parser.parse_args()
    if args.command == "plan":
        drawn = plan(args.shards)
        # Persisted here, in `main()`, never inside `plan()` itself: `plan()`
        # stays a pure projection so it can be called directly (as the test
        # suite's own `test_the_plan_says_the_request_is_not_a_guarantee`
        # does) without dropping a file into a real, tracked results
        # directory on every call.
        shards.write_plan(drawn)
        print(json.dumps(drawn, indent=2))
        return 0
    if args.command == "run":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        # Refuse before touching this shard's directory at all: a recorded
        # id under different seeds is the exact collision that would
        # silently overwrite a shard that already succeeded, since
        # `harness.shard_paths()` keys that directory on the id string
        # alone. A recorded id under the SAME seeds, or an id never
        # recorded at all, proceeds unchanged.
        recorded = shards.read_plan()
        for entry in (recorded.get("shards", []) if recorded is not None else []):
            if entry["id"] == args.shard and list(entry["seeds"]) != seeds:
                raise shards.PlanConflict(
                    f"refusing to run shard {args.shard!r} under seeds {seeds!r}: "
                    f"already recorded under {list(entry['seeds'])!r}.\n"
                    f"  {args.shard!r} may already hold a succeeded shard on disk; "
                    "running it under a different seed list would silently "
                    "overwrite that directory. Run the recorded id under its own "
                    "seeds, or choose a different id."
                )
        run_shard(args.shard, seeds)
        return 0
    found = shards.read_shards()
    if not found:
        print("no shards came back yet; nothing to merge", file=sys.stderr)
        return 1
    # `bridge.py` is what turns `shards.merge()`'s own shape into what
    # `Benchmark_Phase1_Report.ipynb` reads, written to its own scratch
    # location and never into `config.RESULTS` — see its module docstring.
    # Called in two steps rather than through `bridge.bridge()` so the
    # in-memory `summary` (and its `relaunch` key) stays available here for
    # the copy-pasteable hint below; `bridge.bridge()` itself only returns
    # written file paths.
    recorded_plan = shards.read_plan()
    summary, runs = bridge.build_summary(found, plan=recorded_plan)
    written = bridge.write_scratch(summary, runs)
    print(f"written to {written['summary'].parent}:")
    for name, path in written.items():
        print(f"  {name}: {path.relative_to(REPOSITORY)}")

    # `shards.relaunch()` names which ids and seeds still need launching;
    # this file is the one that knows what a CLI invocation looks like, so
    # the hint is built here, from `relaunch["shards"]` alone — never a
    # command string returned by `shards.py` itself.
    to_relaunch = (summary.get("relaunch") or {}).get("shards") or []
    if to_relaunch:
        print("relaunch:")
        for entry in to_relaunch:
            seeds_csv = ",".join(str(seed) for seed in entry["seeds"])
            print(f"  python tools/distribute.py run --shard {entry['id']} "
                 f"--seeds {seeds_csv}  # {entry['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
