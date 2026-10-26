"""Duration audit: how long each record really is, against its nominal length.

Experiments plan v02, Sec. 2.6. Three defects are declared: a duration below
the declared fraction of the nominal (a truncated session), one above it (a
restarted or concatenated session), and one that does not reach the bag
cardinality floor.

**None of the three numbers is instantiated in v02**: neither fraction nor the
floor. So this module measures and marks nothing: it reports the real duration,
its ratio to the nominal, and the number of complete four-second windows --
the instances a bag would get (Sec. 3.1). Sec. 10.1 asks for exactly these:
the real duration of every record, because the body derived bag cardinalities
from nominal durations, and the duration of the task with no event stream.

**Where each nominal comes from.**

* Slackline: the level's own flag schedule (`slackline_flags_info.json`): the
  last flag's spawn time plus the two seconds a flag stays on screen (data
  paper, Environment 1).
* Attention Robots: 14 rows of 20 seconds each (data paper, Environment 2).
* CogniFit: a third-party application; nothing in the release or the paper
  declares its length, so it has no nominal and only the real duration is
  reported.
"""

from __future__ import annotations

import numpy as np

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v02.md",
    "sections": ["2.6"],
    "experiments": ["2.6"],
    "invariants": [
        "nominal_duration_of_each_task_comes_from_its_own_source",
        "duration_reports_ratio_and_window_count_without_a_cut",
    ],
}

WINDOW_S = 4.0
#: Data paper, Environment 1: "remain on screen for a duration of 2 seconds".
FLAG_DISPLAY_S = 2.0
#: Data paper, Environment 2: 14 rows, "a 20-second timeframe per line".
ROBOTS_ROWS, ROBOTS_ROW_S = 14, 20.0


def nominal_duration(task: str, level: int | None, flags_info: dict) -> float | None:
    """Nominal length of a task, in seconds, or None where nothing declares it."""
    if task == "Slackline":
        for entry in flags_info.get("slackline_levels_flags_info", []):
            if entry.get("level") == f"Level{level}":
                last = max(float(f["flag_spawn_time"]) for f in entry["flags"])
                return last + FLAG_DISPLAY_S
        return None
    if task == "Robots":
        return ROBOTS_ROWS * ROBOTS_ROW_S
    return None


def audit_duration(timestamps: np.ndarray, rate: float, nominal: float | None,
                   window_s: float = WINDOW_S) -> dict:
    """Real duration, its ratio to the nominal and complete windows. No cut."""
    timestamps = np.asarray(timestamps, dtype=float)
    duration = float(timestamps[-1] - timestamps[0]) + 1.0 / rate
    return {
        "duration_s": duration,
        "nominal_s": float(nominal) if nominal is not None else None,
        "ratio_to_nominal": duration / nominal if nominal else None,
        "n_windows": int(np.floor(duration / window_s + 1e-9)),
        "marks": [],
        "pending": ["truncation_fraction", "excess_fraction", "bag_cardinality_floor"],
    }
