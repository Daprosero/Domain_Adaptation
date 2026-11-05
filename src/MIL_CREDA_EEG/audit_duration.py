"""Duration audit: how long each record really is, against its nominal length.

Experiments plan v54, Sec. 2.6. The duration is measured and excludes nothing
here. A truncated session only contributes fewer windows; whether they reach a
bag is decided by the bag cardinality floor, which belongs to the bag
definition of Sec. 3.2 and not to this step. A restarted or concatenated
session is marked and reported, never excluded. This module reports the real
duration, its ratio to the nominal, the number of complete four-second windows
-- the instances a bag would get (Sec. 3.1) -- and that one mark.

**The restart mark reads only unambiguous evidence.** v54 gives no numeric
rule for a restarted or concatenated session, so no gap or excess threshold is
invented here (assumption, recorded as such): a record is marked
`restarted_or_concatenated` when its timestamps step backwards at least once,
which a single continuous acquisition cannot produce. The sample counter is
not read for this mark: both devices count modulo a power of two
(`audit_integrity`), so a counter reset is indistinguishable from a wrap plus
lost samples. A duration far above the nominal is measured and reported in
`ratio_to_nominal`, not marked by itself.

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
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v54.md",
    "sections": ["2.6"],
    "experiments": ["2.6"],
    "invariants": [
        "nominal_duration_of_each_task_comes_from_its_own_source",
        "duration_reports_ratio_and_window_count_without_a_cut",
        "a_backward_timestamp_marks_a_restarted_or_concatenated_session",
        "the_mark_excludes_nothing_and_the_floor_is_not_pending_here",
    ],
}

WINDOW_S = 4.0
#: Data paper, Environment 1: "remain on screen for a duration of 2 seconds".
FLAG_DISPLAY_S = 2.0
#: Data paper, Environment 2: 14 rows, "a 20-second timeframe per line".
ROBOTS_ROWS, ROBOTS_ROW_S = 14, 20.0
#: Sec. 2.6: the one mark this step can set.
RESTART_MARK = "restarted_or_concatenated"


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


def backward_timestamp_steps(timestamps: np.ndarray) -> int:
    """Times the clock steps back: evidence of a restart or a concatenation."""
    timestamps = np.asarray(timestamps, dtype=float)
    return int(np.sum(np.diff(timestamps) < 0)) if len(timestamps) > 1 else 0


def audit_duration(timestamps: np.ndarray, rate: float, nominal: float | None,
                   window_s: float = WINDOW_S) -> dict:
    """Real duration, its ratio to the nominal, complete windows and the
    restart mark. No cut, no exclusion."""
    timestamps = np.asarray(timestamps, dtype=float)
    duration = float(timestamps[-1] - timestamps[0]) + 1.0 / rate
    backward = backward_timestamp_steps(timestamps)
    return {
        "duration_s": duration,
        "nominal_s": float(nominal) if nominal is not None else None,
        "ratio_to_nominal": duration / nominal if nominal else None,
        "n_windows": int(np.floor(duration / window_s + 1e-9)),
        "n_backward_timestamp_steps": backward,
        "marks": [RESTART_MARK] if backward else [],
        "pending": [],
    }
