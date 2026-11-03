"""K1's behavioural event, defined over parsed TAGS rows; preprocessing marks the windows.

Experiments plan v42, Sec. 6.1. Only the TAGS files of the high-density device's
Slackline levels log events with an instant; each logged row (one
`reactionOrOmission` entry, its `reacted`, `correct` and `duplicated` flags and its
`generalTime`) is one candidate. A K1 behavioural event is a non-duplicated row
that is

- a hit: reacted and correct;
- an error: reacted and incorrect;
- a correct no-press: not reacted and correct.

Duplicated rows, which repeat the same mark in the same window, are excluded. An
omission (not reacted and incorrect) cannot be used because the game does not log
omissions: the plan declares that no row has it, so one is refused
(`K1EventError`), never classified. The event instant is `generalTime`, seconds
from the level start (`event_instants`). A window holds an event when it contains
the instant: `start <= t < start + length`, no neighbouring window
(`event_windows`).

Pure functions only: the preprocessing task that writes each processed record's
`event_window` mark (`mechanisms.EVENT_MARK`) parses the TAGS file, aligns the
window starts to the level start and calls these; nothing here reads a file.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v42.md",
    "sections": ["6.1"],
    "experiments": ["K1-atencion"],
    "invariants": [
        "an_event_is_a_hit_an_error_or_a_correct_no_press",
        "duplicated_rows_are_excluded",
        "an_omission_row_is_refused_not_guessed",
        "the_event_instant_is_general_time",
        "a_window_holds_an_event_when_it_contains_its_instant",
    ],
}

EVENT_KINDS = ("hit", "error", "correct no-press")
INSTANT = "generalTime"
_FLAGS = {True: True, False: False, "True": True, "False": False}


class K1EventError(ValueError):
    """A TAGS row or window grid that Sec. 6.1's event definition does not cover."""


def _flag(row: Mapping, key: str) -> bool:
    if key not in row:
        raise K1EventError(f"a TAGS row has no {key!r}: {dict(row)}")
    value = row[key]
    if isinstance(value, str):
        value = value.strip()
    if value not in _FLAGS or isinstance(value, (int, float)) and not isinstance(value, bool):
        raise K1EventError(f"{key!r} is True or False, got {row[key]!r}")
    return _FLAGS[value]


def event_kind(row: Mapping) -> str | None:
    """The row's event kind, or None for a duplicated row."""
    reacted, correct = _flag(row, "reacted"), _flag(row, "correct")
    if _flag(row, "duplicated"):
        return None
    if reacted:
        return "hit" if correct else "error"
    if correct:
        return "correct no-press"
    raise K1EventError(f"an omission row (not reacted, incorrect) is not logged by the game "
                       f"and has no definition in Sec. 6.1: {dict(row)}")


def _instant(row: Mapping) -> float:
    value = row.get(INSTANT)
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise K1EventError(f"{INSTANT} is seconds from the level start, got {value!r}") from None
    if not math.isfinite(t) or t < 0:
        raise K1EventError(f"{INSTANT} is seconds from the level start, got {value!r}")
    return t


def event_instants(rows: Iterable[Mapping]) -> list[float]:
    """`generalTime` of every event row, in row order; duplicated rows dropped."""
    return [_instant(r) for r in rows if event_kind(r) is not None]


def event_windows(instants: Sequence[float], starts: Sequence[float],
                  length: float) -> np.ndarray:
    """Per window: it contains an event instant, `start <= t < start + length`.

    `starts` are the windows' start times in seconds from the level start and
    `length` their duration in seconds.
    """
    if not length > 0:
        raise K1EventError(f"the window length is positive seconds, got {length!r}")
    starts = np.asarray(starts, dtype=float).reshape(-1)
    t = np.asarray(instants, dtype=float).reshape(-1)
    if not len(t):
        return np.zeros(len(starts), dtype=bool)
    inside = (t[None, :] >= starts[:, None]) & (t[None, :] < starts[:, None] + length)
    return inside.any(axis=1)
