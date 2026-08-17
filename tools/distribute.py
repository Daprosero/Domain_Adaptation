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
never printed, never parsed. How many accounts exist comes from the accounts
CLI's own `list`, which is built to report usernames and never keys, and even
that is used only to size a plan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from MIL_CREDA_Benchmark import config, harness, shards  # noqa: E402

#: The accelerator every session asks for. One class for every shard, because
#: `seconds` and `peakMiB` are dimensions of the verdict and two arms measured on
#: two GPU classes differ in something nobody declared.
#:
#: Asking is not receiving. A service allocates by availability, so the request
#: lives here and the fact lives in the stamp: `environment()` records what
#: arrived, and the merge groups by that rather than by this.
ACCELERATOR = "T4"

#: Where the accounts skill keeps its store. A path, passed along, never read.
ACCOUNTS_CLI = (REPOSITORY.parents[1] / ".claude" / "skills" / "kaggle-accounts"
                / "scripts" / "accounts_cli.py")


def account_count() -> int | None:
    """How many accounts are available, via the command built to say so.

    `list` reports usernames and never keys, which is exactly why it is the only
    thing asked. Returns `None` when the question cannot be answered rather than
    guessing a number a plan would then be sized against.
    """
    if not ACCOUNTS_CLI.exists():
        return None
    try:
        done = subprocess.run([sys.executable, str(ACCOUNTS_CLI), "list", "--json"],
                              capture_output=True, text=True, timeout=30, shell=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or not done.stdout.strip().startswith(("{", "[")):
        return None
    listed = json.loads(done.stdout)
    accounts = listed.get("accounts") if isinstance(listed, dict) else listed
    return len(accounts) if isinstance(accounts, list) else None


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
    """
    seeds = list(seeds if seeds is not None else config.FULL_SEEDS)
    available = account_count()
    parts = parts or available or 1
    groups = shard_seeds(seeds, parts)
    return {
        "accelerator": ACCELERATOR,
        "accountsAvailable": available,
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
        print(json.dumps(plan(args.shards), indent=2))
        return 0
    if args.command == "run":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        run_shard(args.shard, seeds)
        return 0
    found = shards.read_shards()
    if not found:
        print("no shards came back yet; nothing to merge", file=sys.stderr)
        return 1
    print(json.dumps(shards.merge(found), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
