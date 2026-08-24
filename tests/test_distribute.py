"""The launcher: the only file here that knows a remote service exists.

Two obligations it must keep apart, because conflating them is how a comparison
quietly stops being one: requesting an accelerator, and having received it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]

#: `tools/distribute.py` loads the packer, the adapter registry and the service
#: adapter out of the forge's `remote-execution` skill, and it does so at import
#: time. That dependency is correct and is not vendored: splitting a campaign
#: across machines is orchestration, and orchestration is the forge's job. What
#: must not happen is this file erroring during collection when the forge is not
#: there, because a clone of this repository is a paper's implementation and has
#: to run its tests on its own. `src/` stands alone — see
#: `test_shard_io_vendoring.py`, which holds it to that — and this one module
#: says out loud that it cannot.
_FORGE_SCRIPTS = (
    REPOSITORY.parents[1] / ".claude" / "skills" / "remote-execution" / "scripts"
)
if not (_FORGE_SCRIPTS / "packer.py").is_file():
    pytest.skip(
        f"{_FORGE_SCRIPTS} is not there, so the launcher cannot be imported. "
        "Submitting a campaign to a remote service is the forge's job and this "
        "repository does not carry it; every other test here runs regardless.",
        allow_module_level=True,
    )

_spec = importlib.util.spec_from_file_location(
    "distribute", REPOSITORY / "tools" / "distribute.py")
distribute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distribute)


def test_the_axis_is_a_whole_repetition_so_no_comparison_is_split():
    """Every arm of every transfer within a seed runs on one machine.

    Sharding by arm would put the ladder's own subtraction across a hardware
    boundary, which is the one split the distribution declaration forbids.
    """
    groups = distribute.shard_seeds(list(range(30)), 6)
    assert [len(g) for g in groups] == [5] * 6
    assert sorted(s for g in groups for s in g) == list(range(30))


def test_no_seed_is_lost_when_the_split_is_uneven():
    groups = distribute.shard_seeds(list(range(10)), 3)
    assert [len(g) for g in groups] == [4, 3, 3]
    assert sorted(s for g in groups for s in g) == list(range(10))


def test_more_shards_than_seeds_collapses_rather_than_producing_empty_ones():
    """An empty shard would come back with nothing and read as a lost one."""
    groups = distribute.shard_seeds([0, 1], 8)
    assert groups == [[0], [1]]


def test_the_accelerator_is_requested_here_and_nowhere_else():
    """It is a service's configuration, so it lives in the launcher alone."""
    assert distribute.ACCELERATOR == "NvidiaTeslaT4"
    source = (REPOSITORY / "src").rglob("*.py")
    for path in source:
        assert "T4" not in path.read_text(encoding="utf-8"), path


def test_the_plan_says_the_request_is_not_a_guarantee():
    """A plan that promised the hardware would be the claim the stamp exists to
    check. Reachable red: drop the note and this fails."""
    drawn = distribute.plan(parts=3, seeds=list(range(9)))
    assert drawn["accelerator"] == "NvidiaTeslaT4"
    assert "requested, not guaranteed" in drawn["note"]
    assert [s["seeds"] for s in drawn["shards"]] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_the_store_is_never_opened(monkeypatch):
    """Credentials are a path here and never a value.

    Worker enumeration is now the forge's own Kaggle adapter's job, held to
    the same discipline this file used to enforce with its own subprocess
    call: usernames only, from the accounts CLI's own `list`, and a failure
    to answer returns None rather than a guess a plan would then be sized
    against.
    """
    import subprocess as sp

    opened = []
    real_open = Path.read_text

    def watched(self, *a, **k):
        opened.append(str(self))
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", watched)
    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))

    assert distribute.capacity_plans() is None
    assert not any("accounts.json" in p for p in opened)


def _fake_accounts_result(usernames):
    class _Result:
        returncode = 0
        stdout = json.dumps({"accounts": [{"username": name} for name in usernames]})
        stderr = ""

    return _Result()


def test_distribute_declares_capacity_and_calls_packer_plan(monkeypatch):
    """`capacity_plans()` delegates worker enumeration and the capacity
    clamp entirely to the forge's `adapters/kaggle.py` and `packer.plan()`;
    `plan()`'s own dict reports what came back, all four numbers included.

    Reachable red: before this task, neither `capacity_plans` nor a
    `capacity` key on `plan()`'s dict existed — this raised `AttributeError`
    / `KeyError` against pre-change code.
    """
    import subprocess as sp

    monkeypatch.setattr(
        sp, "run", lambda argv, **k: _fake_accounts_result(["alice", "bob"]))

    plans = distribute.capacity_plans()
    assert plans is not None
    assert sorted(p.worker for p in plans) == ["alice", "bob"]
    for one in plans:
        assert one.requested == distribute.REQUESTED_CAPACITY_PER_WORKER
        assert one.cap == 2
        assert one.in_flight == 0
        assert one.granted == 2

    drawn = distribute.plan(seeds=list(range(2)))
    assert drawn["capacity"]["requestedPerWorker"] == distribute.REQUESTED_CAPACITY_PER_WORKER
    assert drawn["capacity"]["granted"] == 4
    assert {w["worker"] for w in drawn["capacity"]["workers"]} == {"alice", "bob"}
    assert all(
        {"worker", "requested", "cap", "inFlight", "granted"} <= w.keys()
        for w in drawn["capacity"]["workers"]
    )
    assert len(drawn["shards"]) == 2


def test_a_clamped_plan_is_distinguishable_from_an_unclamped_one_with_the_same_granted(monkeypatch):
    """`requested`, `cap`, `inFlight` and `granted` stay four separate
    numbers: a request the worker fully honors and a request the worker
    clamps down can land on the identical `granted` count, and only the
    other three numbers tell those two plans apart.
    """
    import subprocess as sp

    monkeypatch.setattr(sp, "run", lambda argv, **k: _fake_accounts_result(["alice"]))

    monkeypatch.setattr(distribute, "REQUESTED_CAPACITY_PER_WORKER", 2)
    unclamped = distribute.capacity_plans()[0]

    monkeypatch.setattr(distribute, "REQUESTED_CAPACITY_PER_WORKER", 5)
    clamped = distribute.capacity_plans()[0]

    assert unclamped.granted == clamped.granted == 2
    assert unclamped.requested == unclamped.cap
    assert clamped.requested > clamped.cap


def test_run_shard_stamps_through_the_same_campaign_no_local_exemption():
    """A locally-run and a remotely-run shard are stamped by literally the
    same code: `run_shard()` drives `harness.campaign()` and does not
    reimplement any part of stamping itself. The earlier "local shards are
    exempt" assumption does not exist here to reintroduce."""
    import inspect

    source = inspect.getsource(distribute.run_shard)
    assert "harness.campaign(" in source
    assert "write_shard_stamp" not in source
    assert "seal_shard_stamp" not in source


def test_run_shard_obtains_the_ceilings_before_it_runs_the_campaign():
    """The same join `run_pilot()`, `run_search()` and `run_campaign_shard()`
    are already held to: `harness.campaign()` refuses outright without
    `reduction.ceilings` already populated, and `run_shard()` was copied from
    `run_campaign_shard()` before that join existed there either.

    Source-only, like its neighbour above: never execute `run_shard()` for
    real, or this would recreate the exact hermeticity landmine removed
    elsewhere in this change.
    """
    import inspect

    source = inspect.getsource(distribute.run_shard)
    assert source.index("ceilings_in_force(") < source.index("harness.campaign(")


def test_distribute_py_makes_no_direct_service_call_itself():
    """The forge's adapter is the one place allowed to shell out to a
    service or its accounts CLI; this launcher delegates entirely and
    invokes no subprocess of its own."""
    source = (REPOSITORY / "tools" / "distribute.py").read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess.run" not in source
    assert "kernels" not in source
