"""Experiments plan v42, Sec. 6.1: the K1 behavioural event, defined over parsed TAGS rows.

The rows are hand-written in the shape of a BALLADEER Slackline TAGS `reactionOrOmission`
entry; none is read from the dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from MIL_CREDA_EEG import k1_events as ev


def _row(reacted, correct, duplicated="False", t=1.0):
    return {"reacted": reacted, "correct": correct, "duplicated": duplicated,
            "generalTime": t, "reactionTime": 0.5, "focus": "Target"}


def test_an_event_is_a_hit_an_error_or_a_correct_no_press():
    assert ev.event_kind(_row("True", "True")) == "hit"
    assert ev.event_kind(_row("True", "False")) == "error"
    assert ev.event_kind(_row("False", "True")) == "correct no-press"
    assert ev.event_kind(_row(True, False)) == "error"       # already-parsed booleans
    assert ev.EVENT_KINDS == ("hit", "error", "correct no-press")


def test_duplicated_rows_are_excluded():
    assert ev.event_kind(_row("True", "True", duplicated="True")) is None
    rows = [_row("True", "True", t=2.0), _row("True", "True", duplicated="True", t=2.5),
            _row("False", "True", t=7.0)]
    assert ev.event_instants(rows) == [2.0, 7.0]


def test_an_omission_row_is_refused_not_guessed():
    # Sec. 6.1: the game logs no omission; no row has no reaction with a wrong answer.
    with pytest.raises(ev.K1EventError, match="omission"):
        ev.event_kind(_row("False", "False"))
    for bad in ({"reacted": "maybe", "correct": "True", "duplicated": "False",
                 "generalTime": 1.0},
                {"correct": "True", "duplicated": "False", "generalTime": 1.0}):
        with pytest.raises(ev.K1EventError):
            ev.event_kind(bad)


def test_the_event_instant_is_general_time():
    assert ev.event_instants([_row("True", "False", t=12.25)]) == [12.25]
    for bad in (None, "x", float("nan"), -1.0):
        with pytest.raises(ev.K1EventError, match="generalTime"):
            ev.event_instants([_row("True", "True", t=bad)])


def test_a_window_holds_an_event_when_it_contains_its_instant():
    starts = [0.0, 4.0, 8.0, 12.0]
    marks = ev.event_windows([0.0, 3.999, 4.0, 13.5, 16.0, 20.0], starts, 4.0)
    assert marks.dtype == np.bool_
    # [start, start + length): no neighbouring window is marked, an instant outside every
    # window marks nothing
    assert marks.tolist() == [True, True, False, True]
    assert ev.event_windows([], starts, 4.0).tolist() == [False] * 4
    with pytest.raises(ev.K1EventError, match="length"):
        ev.event_windows([1.0], starts, 0.0)
