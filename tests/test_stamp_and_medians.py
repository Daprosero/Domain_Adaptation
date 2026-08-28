"""Two things a table says about itself: what it averaged, and over how many.

Both were read from `config` instead of from the data. The stamp printed
`len(config.SEEDS)` — a constant compared against another constant, never
counting the readings underneath it. And a cell's average took every checkpoint
in the directory, including the ones a paired promotion added to a floor, which
were selected by *other* arms' orderings and are not a sample of that floor.
"""

from __future__ import annotations

import json

import pytest

from MIL_CREDA_Benchmark import config, latent, tables


def _reading(arm, transfer, seed, ratio, median=True):
    return {"arm": arm, "transfer": transfer, "seed": seed, "median": median,
            "unit": "bag", "targetAccuracy": 0.5, "domainSeparability": 0.5,
            "geometry": {"ratio": ratio, "crossDomainSameClass": ratio,
                         "betweenClasses": 1.0}}


# --- what a cell averages ---------------------------------------------------

def test_a_floors_paired_extras_never_enter_its_own_average():
    """They were chosen by a dependant arm's ordering, not by this cell's.

    The ratio here is 0.2 on the three medians and 0.9 on the extras: reading
    them together does not estimate the floor's row better, it estimates a cell
    that does not exist.
    """
    readings = ([_reading("B", "M->U", s, 0.2) for s in (10, 12, 24)]
                + [_reading("B", "M->U", s, 0.9, median=False) for s in (0, 4, 5)])
    averaged = tables._by_arm(readings, "geometry.ratio")
    assert averaged["B"]["M->U"] == pytest.approx(0.2)


def test_a_checkpoint_with_no_mark_is_kept():
    """A single-machine run has no promotion record at all.

    There everything on disk is its cell's median, because `keep_median()` is
    what put it there and it writes nothing else. Dropping unmarked entries would
    empty every table on exactly those runs.
    """
    readings = [{"arm": "B", "transfer": "M->U", "seed": 0}]
    assert tables._own_medians(readings) == readings


# --- how many repetitions the stamp claims ----------------------------------

def test_the_stamp_counts_the_data_and_not_the_configuration(monkeypatch):
    """`config.SEEDS` is what was asked for; the readings are what came back."""
    monkeypatch.setattr(config, "SEEDS", [0])
    readings = [_reading("B", "M->U", s, 0.2) for s in (10, 12, 24)]
    assert tables._pilot_note([], readings) == [
        "Piloto de 3 repetición(es): estimación puntual, todavía no un veredicto."]


def test_the_extras_do_not_inflate_the_count_either():
    """Six checkpoints, three repetitions: the stamp says three."""
    readings = ([_reading("B", "M->U", s, 0.2) for s in (10, 12, 24)]
                + [_reading("B", "M->U", s, 0.9, median=False) for s in (0, 4, 5)])
    assert "3 repetición" in tables._pilot_note([], readings)[0]


def test_when_cells_disagree_the_weakest_bounds_the_claim():
    """The thinnest cell is what the table can support, not the fattest."""
    readings = ([_reading("B", "M->U", s, 0.2) for s in (0, 1, 2, 3, 4)]
                + [_reading("G", "M->U", s, 0.2) for s in (7, 8)])
    assert "2 repetición" in tables._pilot_note([], readings)[0]


def test_a_full_scale_run_is_not_stamped_a_pilot():
    readings = [_reading("B", "M->U", s, 0.2) for s in range(len(config.FULL_SEEDS))]
    assert tables._pilot_note([], readings) == []


def test_being_unable_to_count_says_so_instead_of_going_quiet():
    """An absent stamp reads as a full-scale run, which is the opposite fact."""
    line = tables._pilot_note([], None)
    assert line and "no verificable" in line[0]


# --- the tag itself ---------------------------------------------------------

def test_available_marks_the_promoted_extras_as_not_median(tmp_path, monkeypatch):
    """Read from the promotion record, which is the only thing that knows."""
    monkeypatch.setattr(config, "MODELS", tmp_path)
    for seed in (10, 12, 24, 4):
        (tmp_path / f"B_M-U_seed{seed}.pt").write_bytes(b"w")
        (tmp_path / f"B_M-U_seed{seed}.manifest.json").write_text(json.dumps(
            {"arm": "B", "transfer": "M->U", "seed": seed}), encoding="utf-8")
    (tmp_path / latent.PROMOTION_RECORD).write_text(json.dumps(
        {"chosen": {"B|M->U": {"chosen": [10, 12, 24], "paired": [4]}}}),
        encoding="utf-8")

    marked = {r["seed"]: r["median"] for r in latent.available()}

    assert marked == {10: True, 12: True, 24: True, 4: False}


def test_without_a_promotion_record_everything_is_its_own_median(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS", tmp_path)
    (tmp_path / "B_M-U_seed0.pt").write_bytes(b"w")
    (tmp_path / "B_M-U_seed0.manifest.json").write_text(json.dumps(
        {"arm": "B", "transfer": "M->U", "seed": 0}), encoding="utf-8")

    assert [r["median"] for r in latent.available()] == [True]
