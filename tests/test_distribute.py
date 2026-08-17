"""The launcher: the only file here that knows a remote service exists.

Two obligations it must keep apart, because conflating them is how a comparison
quietly stops being one: requesting an accelerator, and having received it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
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
    assert distribute.ACCELERATOR == "T4"
    source = (REPOSITORY / "src").rglob("*.py")
    for path in source:
        assert "T4" not in path.read_text(encoding="utf-8"), path


def test_the_plan_says_the_request_is_not_a_guarantee():
    """A plan that promised the hardware would be the claim the stamp exists to
    check. Reachable red: drop the note and this fails."""
    drawn = distribute.plan(parts=3, seeds=list(range(9)))
    assert drawn["accelerator"] == "T4"
    assert "requested, not guaranteed" in drawn["note"]
    assert [s["seeds"] for s in drawn["shards"]] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_the_store_is_never_opened(monkeypatch):
    """Credentials are a path here and never a value.

    The count comes from the command built to report usernames and never keys,
    and a failure to answer returns None rather than a guess a plan would then
    be sized against.
    """
    import subprocess as sp

    opened = []
    real_open = Path.read_text

    def watched(self, *a, **k):
        opened.append(str(self))
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", watched)
    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(distribute.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError()))

    assert distribute.account_count() is None
    assert not any("accounts.json" in p for p in opened)
