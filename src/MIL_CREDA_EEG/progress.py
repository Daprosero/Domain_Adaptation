"""Progress lines for remote notebooks.

Each step prints one flushed line to the notebook log and appends one JSON row to
`progress.jsonl` in the working directory, so a run can be followed while it executes:

    [unit] method phase 12/40 · 41.0 min · ~58 min left

The estimate is the mean measured seconds per unit of work times the work still declared.
Steps may carry a weight (a search trial is cheaper or dearer than a seed), so the estimate
stays a measurement, never a guess about the host.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Progress:
    def __init__(self, unit: str, total: float, workdir=None, clock=time.time):
        self.unit, self.total, self.done = str(unit), float(total), 0.0
        self.clock, self.started = clock, clock()
        self.path = Path(workdir or ".") / "progress.jsonl"

    def add(self, work: float) -> None:
        """Declare more work (for example the seeds after the search chose its epochs)."""
        self.total += float(work)

    def step(self, label: str, weight: float = 1.0) -> dict:
        self.done += float(weight)
        elapsed = self.clock() - self.started
        left = (elapsed / self.done) * max(0.0, self.total - self.done) if self.done > 0 else None
        row = {"unit": self.unit, "label": str(label), "done": round(self.done, 3),
               "total": round(self.total, 3), "elapsedMin": round(elapsed / 60, 2),
               "leftMin": None if left is None else round(left / 60, 2), "at": round(self.clock(), 1)}
        eta = "?" if left is None else f"~{left / 60:.0f} min left"
        print(f"[{self.unit}] {label} {self.done:g}/{self.total:g} · {elapsed / 60:.1f} min · {eta}",
              flush=True)
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass
        return row


def probe(config=None, product=None, *, minutes: float = 5.0, every: float = 30.0,
          workdir=None, sleep=time.sleep) -> dict:
    """Log-visibility probe: one progress line every `every` seconds for `minutes` minutes."""
    n = max(1, int(round(minutes * 60 / every)))
    p = Progress("log-probe", n, workdir)
    for i in range(n):
        sleep(every)
        p.step(f"tick {i + 1}")
    return {"ran": ["log-probe"], "seconds": round(p.clock() - p.started)}
