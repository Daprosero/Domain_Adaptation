"""The admissibility ruling, read before any remedy is measured.

Order is the point. Whether a correction is sound and expressible inside the
proposal is decided first; whether it resolves the finding is measured only
afterwards. Running the sweep on a remedy that cites a missing equation, or
leans on notation the revision never defines, produces numbers that read as
evidence and lend rigour to something that should not have reached the bench.

The ruling is produced by `implementation_cli.py admit --revision <rNN>`, which
reads the revision in the forge and writes only its verdict here. Without it,
every remedy test fails immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

RULING_PATH = Path(__file__).with_name("admissibility.json")


def _ruling() -> dict:
    if not RULING_PATH.exists():
        raise AssertionError(
            "no admissibility ruling: run `implementation_cli.py admit --revision <rNN>` "
            "before measuring any remedy")
    return json.loads(RULING_PATH.read_text(encoding="utf-8"))


def require_admissible(finding_id: str) -> None:
    """Refuse to measure a remedy that was not ruled admissible."""
    verdicts = _ruling().get("findings", {})
    if finding_id not in verdicts:
        raise AssertionError(f"{finding_id} was never ruled on; efficacy cannot be measured")
    verdict = verdicts[finding_id]
    if not verdict.get("admissible"):
        raise AssertionError(
            f"{finding_id} is not admissible, so its efficacy is not measured: "
            + "; ".join(verdict.get("reasons", [])))


def ruled_revision() -> str:
    return _ruling().get("revision", "")
