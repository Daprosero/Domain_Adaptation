"""The vendored shard reader has to stay equal to the one it was copied from.

`src/MIL_CREDA_Benchmark/shard_io.py` is a copy of the forge's generic
shard-reading module. It is a copy rather than a path-import because a clone of
this repository is a paper's implementation: somebody should be able to clone
it, install it and run its tests without a forge checkout sitting above it. The
earlier arrangement reached four directories up into `.claude/skills/`, and a
standalone clone could not collect a single test.

A copy that nothing holds equal is a copy that drifts, so this file holds it —
but only when the origin is actually there. In the forge checkout it runs and
would fail on any divergence; anywhere else it skips and says why, because the
absence of the origin is exactly the situation the vendoring exists to serve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

VENDORED = Path(__file__).resolve().parents[1] / "src" / "MIL_CREDA_Benchmark" / "shard_io.py"

#: Where the origin lives when this repository sits inside a forge checkout.
#: Derived rather than configured: this file is at
#: <forge>/implementations/<repo>/tests/, so the forge root is three up.
ORIGIN = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "remote-execution" / "scripts" / "shard_io.py"
)


def test_the_vendored_reader_is_byte_identical_to_its_origin() -> None:
    if not ORIGIN.is_file():
        pytest.skip(
            f"{ORIGIN} is not there, so there is no origin to compare against. "
            "That is the standalone case this vendoring exists for, not a failure."
        )
    assert VENDORED.read_bytes() == ORIGIN.read_bytes(), (
        f"{VENDORED.name} has drifted from {ORIGIN}. The vendored copy is not a "
        "fork: re-copy it, or change the origin and re-copy, but do not let the "
        "two spell the same reader differently."
    )


def test_nothing_in_the_package_reaches_out_of_this_repository() -> None:
    """The vendoring is pointless if some other module still reaches for the forge.

    Named as a property of the package rather than of `shards.py`, because the
    next module to want a forge utility is the one this catches.
    """
    package = Path(__file__).resolve().parents[1] / "src"
    reaching = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if ".claude" in path.read_text(encoding="utf-8", errors="replace")
    )
    assert reaching == [], (
        "these modules name the forge's own directory, so a clone of this "
        f"repository cannot stand on its own: {reaching}"
    )
