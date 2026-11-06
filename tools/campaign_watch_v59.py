"""Print the status of the v59 campaign's kernels. Read-only: submits, deletes, cancels nothing.

    .micromamba/envs/papersmith/bin/python \
        implementations/Domain_Adaptation/tools/campaign_watch_v59.py [--kernels kernels.json]

Reads `MIL-CREDA-EEG/Results/launch_v59/kernels.json`. For each kernel it prints account,
status and the time since its submission (a finished kernel's line keeps counting: Kaggle's
status call gives no end time), then one summary line. Credentials follow the adapter's
path: the account's token is materialized by `kaggle-accounts`, read once into the
`KAGGLE_API_TOKEN` of a child process, and the copy is deleted before the child runs.
A token is never printed. Run from anywhere; needs the interpreter holding `kagglesdk`.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1]
WORKSPACE = TARGET.parents[1]
ACCOUNTS = WORKSPACE / "skills" / "kaggle-accounts" / "scripts" / "accounts_cli.py"
KERNELS = TARGET / "MIL-CREDA-EEG" / "Results" / "launch_v59" / "kernels.json"

CHILD = r"""
import json, sys
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
client = KernelsApiClient(KaggleHttpClient())
out = {}
for ref in sys.argv[1:]:
    request = ApiGetKernelSessionStatusRequest()
    request.user_name, request.kernel_slug = ref.split("/", 1)
    try:
        response = client.get_kernel_session_status(request)
        out[ref] = {"status": response.status.name, "failure": response.failure_message or ""}
    except Exception as error:
        out[ref] = {"status": "UNREACHABLE", "failure": type(error).__name__}
print(json.dumps(out))
"""


def read_token(account: str) -> str:
    """The account's token; the materialized copy is gone when this returns."""
    done = subprocess.run([sys.executable, str(ACCOUNTS), "materialize", "--worker", account,
                           "--json"], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError("materialize refused")
    copy = Path(json.loads(done.stdout)["tokenPath"])
    try:
        return copy.read_text(encoding="utf-8").strip()
    finally:
        copy.unlink(missing_ok=True)


def statuses(account: str, refs: list[str]) -> dict[str, dict]:
    try:
        env = {"PATH": os.environ.get("PATH", ""), "KAGGLE_API_TOKEN": read_token(account)}
        done = subprocess.run([sys.executable, "-c", CHILD, *refs], capture_output=True,
                              text=True, env=env, timeout=180)
        if done.returncode != 0:
            raise RuntimeError("status child failed")
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (RuntimeError, subprocess.TimeoutExpired, ValueError, IndexError) as error:
        return {ref: {"status": "UNREACHABLE", "failure": str(error)[:80]} for ref in refs}


def elapsed(submitted: str, now: dt.datetime) -> str:
    start = dt.datetime.fromisoformat(submitted.replace("Z", "+00:00"))
    minutes = int((now - start).total_seconds() // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernels", default=str(KERNELS))
    args = parser.parse_args(argv)
    kernels = json.loads(Path(args.kernels).read_text(encoding="utf-8"))["kernels"]
    by_account: dict[str, list[str]] = collections.defaultdict(list)
    for kernel in kernels:
        by_account[kernel["account"]].append(kernel["ref"])
    found = {}
    for account, refs in by_account.items():
        found.update(statuses(account, refs))
    now = dt.datetime.now(dt.timezone.utc)
    counts = collections.Counter()
    for kernel in kernels:
        row = found[kernel["ref"]]
        counts[row["status"].lower()] += 1
        print(f"{kernel['account']:20s} {kernel['ref'].split('/', 1)[1]:32s} "
              f"{row['status']:12s} {elapsed(kernel['submittedAt'], now):>7s}"
              f"{'  ' + row['failure'][:80] if row['failure'] else ''}")
    known = ("running", "complete", "error")
    rest = {k: v for k, v in sorted(counts.items()) if k not in known}
    print(f"{now:%Y-%m-%dT%H:%MZ} total {len(kernels)}: "
          + ", ".join(f"{k} {counts[k]}" for k in known)
          + "".join(f", {k} {v}" for k, v in rest.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
