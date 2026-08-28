"""Promotion: the campaign's median checkpoints, moved where phase two reads.

`shards.promote_candidates()` held the rule and nothing called it. These tests
hold what makes the wiring honest rather than merely present: that it reads what
`harness.keep_median()` actually wrote instead of a second naming contract of its
own, that it never recomputes the median over the survivors, that it reports a
shortfall instead of filling it from further out, and that it refuses in the two
places where succeeding would be worse than failing — nothing arriving, and a
second promotion burying the first.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "promote", REPOSITORY / "tools" / "promote.py")
promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(promote)

from MIL_CREDA_Benchmark import config, harness  # noqa: E402


def _run(arm, transfer, seed, target):
    return {"arm": arm, "transfer": transfer, "seed": seed, "env": "e1",
            "targetAccuracy": target, "sourceAccuracy": 1.0, "seconds": 1.0,
            "contribution": 0.1, "supervised": 0.2, "adaptationShare": 0.3,
            "peakMiB": 50.0, "parameters": 11247434}


def _shard(root: Path, arm, transfer, seeds, monkeypatch, accuracy=0.5):
    """Lay a shard's checkpoints down the way `harness.keep_median()` lays them.

    Driven through `keep_median` rather than written by hand: the whole point of
    the reader is that it agrees with the writer, and a fixture this file
    authored would agree with whatever this file also reads.
    """
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "MODELS", root)
    # `keep_median` records each kept path relative to the repository root, so a
    # fixture laid down outside it has to say where its own root is.
    monkeypatch.setattr(config, "REPOSITORY", root)
    runs = [_run(arm, transfer, seed, accuracy) for seed in seeds]
    manifests = {(transfer, seed): {"source": "M", "target": "U"} for seed in seeds}
    for seed in seeds:
        (root / f"{arm}_{transfer.replace('->', '-')}_seed{seed}.pt").write_bytes(b"w")
    return harness.keep_median(runs, arm, transfer, manifests, harness.Reduction())


def test_it_reads_exactly_what_the_harness_wrote(tmp_path, monkeypatch):
    """The reader and the writer meet, and the join is what is checked.

    `keep_median` names a checkpoint and writes its manifest; `arrived` has to
    recover the same cell and seed from that. Testing each half against a
    fixture written here would check both and never the connection between them.
    """
    shard = tmp_path / "returned" / "s00" / "unpacked"
    _shard(shard, "G", "M->U", (0, 1, 2), monkeypatch)

    here = promote.arrived(tmp_path / "returned")
    assert set(here["found"]) == {("G", "M->U", 0), ("G", "M->U", 1), ("G", "M->U", 2)}
    assert here["unidentifiable"] == []
    assert here["duplicates"] == []


def test_the_median_is_taken_over_every_measurement_never_over_the_survivors(
        tmp_path, monkeypatch):
    """Five were measured; two came back. The median is of the five.

    Recomputing over the two that arrived would pick seed 0 or 4 — the median of
    the survivors — and call it the campaign's. The campaign's median of
    (0.10, 0.90, 0.50, 0.30, 0.70) is seeds {2, 3, 4}, so what is kept is the
    part of that which exists: 2 and 4, and never 0.
    """
    shard = tmp_path / "artifacts" / "s00"
    _shard(shard, "G", "M->U", (2, 4), monkeypatch)
    runs = [_run("G", "M->U", seed, accuracy)
            for seed, accuracy in enumerate((0.10, 0.90, 0.50, 0.30, 0.70))]

    report = promote.promote(runs=runs, root=tmp_path / "artifacts",
                             destination=tmp_path / "into",
                             superseded=tmp_path / "aside")

    assert report["chosen"]["G|M->U"]["wanted"] == [2, 3, 4]
    assert report["chosen"]["G|M->U"]["chosen"] == [2, 4]
    landed = {p.name for p in (tmp_path / "into").glob("*.pt")}
    assert landed == {"G_M-U_seed2.pt", "G_M-U_seed4.pt"}


def test_a_shortfall_is_reported_and_never_filled_from_further_out(
        tmp_path, monkeypatch):
    """One of the three the median wanted never came back.

    The cheap repair is to reach one place further out and keep three anyway.
    That would report a full cell and quietly describe a different one, so the
    count stays at two and the gap is named.
    """
    shard = tmp_path / "artifacts" / "s00"
    _shard(shard, "G", "M->U", (2, 4), monkeypatch)
    runs = [_run("G", "M->U", seed, accuracy)
            for seed, accuracy in enumerate((0.10, 0.90, 0.50, 0.30, 0.70))]

    report = promote.promote(runs=runs, root=tmp_path / "artifacts",
                             destination=tmp_path / "into",
                             superseded=tmp_path / "aside")

    assert report["shortfall"]["G|M->U"]["shortfall"] == 1
    assert len(list((tmp_path / "into").glob("*.pt"))) == 2


def test_what_was_already_there_is_moved_and_not_removed(tmp_path, monkeypatch):
    """The checkpoints being replaced are a real run's output, not ours to drop."""
    shard = tmp_path / "artifacts" / "s00"
    _shard(shard, "G", "M->U", (0,), monkeypatch)
    into = tmp_path / "into"
    into.mkdir()
    (into / "earlier.pt").write_bytes(b"earlier")

    report = promote.promote(runs=[_run("G", "M->U", 0, 0.5)],
                             root=tmp_path / "artifacts", destination=into,
                             superseded=tmp_path / "aside")

    assert report["superseded"] == str(tmp_path / "aside")
    assert (tmp_path / "aside" / "earlier.pt").read_bytes() == b"earlier"
    assert not (into / "earlier.pt").exists()


def test_a_second_promotion_refuses_rather_than_burying_the_first(
        tmp_path, monkeypatch):
    """Moving a second set on top would lose whatever the first move saved."""
    shard = tmp_path / "artifacts" / "s00"
    _shard(shard, "G", "M->U", (0,), monkeypatch)
    into = tmp_path / "into"
    into.mkdir()
    (into / "earlier.pt").write_bytes(b"earlier")
    aside = tmp_path / "aside"
    aside.mkdir()
    (aside / "older.pt").write_bytes(b"older")

    with pytest.raises(SystemExit):
        promote.promote(runs=[_run("G", "M->U", 0, 0.5)],
                        root=tmp_path / "artifacts", destination=into,
                        superseded=aside)

    assert (aside / "older.pt").read_bytes() == b"older"
    assert (into / "earlier.pt").read_bytes() == b"earlier"


def test_nothing_arriving_refuses_instead_of_emptying_the_destination(tmp_path):
    """An empty promotion moves the old set aside and copies none in.

    The destination ends up empty and phase two reports no checkpoints at all,
    which reads as "phase one never ran" rather than as "nothing came back".
    """
    empty = tmp_path / "artifacts"
    empty.mkdir()
    into = tmp_path / "into"
    into.mkdir()
    (into / "earlier.pt").write_bytes(b"earlier")

    with pytest.raises(SystemExit):
        promote.promote(runs=[_run("G", "M->U", 0, 0.5)], root=empty,
                        destination=into, superseded=tmp_path / "aside")

    assert (into / "earlier.pt").read_bytes() == b"earlier"


def test_a_checkpoint_with_no_manifest_is_named_never_parsed(tmp_path, monkeypatch):
    """Its name looks like a cell and a seed. That is not the same as saying so.

    Recovering the key from the filename would make this module a second naming
    contract, and the day a name and its manifest disagreed the guess would win
    in silence.
    """
    shard = tmp_path / "artifacts" / "s00"
    _shard(shard, "G", "M->U", (0,), monkeypatch)
    (shard / "G_M-U_seed7.pt").write_bytes(b"orphan")

    here = promote.arrived(tmp_path / "artifacts")

    assert set(here["found"]) == {("G", "M->U", 0)}
    assert [Path(p).name for p in here["unidentifiable"]] == ["G_M-U_seed7.pt"]


def test_the_record_says_which_seeds_were_wanted_and_which_existed(
        tmp_path, monkeypatch):
    """Without it the directory says what was chosen and never why."""
    shard = tmp_path / "artifacts" / "s00"
    _shard(shard, "G", "M->U", (2, 4), monkeypatch)
    runs = [_run("G", "M->U", seed, accuracy)
            for seed, accuracy in enumerate((0.10, 0.90, 0.50, 0.30, 0.70))]
    into = tmp_path / "into"

    promote.promote(runs=runs, root=tmp_path / "artifacts", destination=into,
                    superseded=tmp_path / "aside")

    record = json.loads((into / promote.RECORD).read_text(encoding="utf-8"))
    assert record["measurements"] == 5
    assert record["candidates"] == 2
    # `paired` is part of the record's contract: a cell that carries no extras
    # says so, rather than leaving a reader to infer it from the field's absence.
    assert record["chosen"]["G|M->U"] == {"chosen": [2, 4], "wanted": [2, 3, 4],
                                          "shortfall": 1, "paired": [], "display": []}


def test_two_shards_claiming_one_seed_are_reported_not_resolved(
        tmp_path, monkeypatch):
    """One seed of one cell runs on exactly one shard.

    The same key arriving twice means two shards disagree about who ran it.
    Taking either one hides the disagreement behind a plausible answer.
    """
    _shard(tmp_path / "artifacts" / "s00", "G", "M->U", (0,), monkeypatch)
    _shard(tmp_path / "artifacts" / "s01", "G", "M->U", (0,), monkeypatch)

    here = promote.arrived(tmp_path / "artifacts")

    assert here["duplicates"] == ["G|M->U seed 0"]


def test_promotion_never_touches_the_record_the_report_reads(tmp_path, monkeypatch):
    """Checkpoints are promoted; measurements are not re-written."""
    shard = tmp_path / "artifacts" / "s00"
    _shard(shard, "G", "M->U", (0,), monkeypatch)
    results = tmp_path / "results"
    results.mkdir()
    (results / "runs.jsonl").write_text("untouched\n", encoding="utf-8")
    (results / "summary.json").write_text("untouched\n", encoding="utf-8")
    monkeypatch.setattr(config, "RESULTS", results)

    promote.promote(runs=[_run("G", "M->U", 0, 0.5)], root=tmp_path / "artifacts",
                    destination=tmp_path / "into", superseded=tmp_path / "aside")

    assert (results / "runs.jsonl").read_text(encoding="utf-8") == "untouched\n"
    assert (results / "summary.json").read_text(encoding="utf-8") == "untouched\n"


# --- pairing: the precondition of `latent.against_floor()` ------------------
#
# The comparison is a paired difference — an arm at one seed against its own
# floor at that same seed. `promote_candidates()` takes each cell's median
# independently and never produces that pairing; these hold the extension that
# does, and hold that it only ever adds.

FLOORS = {"C": "A", "D": "A", "G": "B"}

#: Seven seeds, and two orderings whose medians are **disjoint**: C's are
#: {2, 3, 4} and A's are {0, 1, 5}. Chosen that way deliberately. An earlier
#: version of these tests used five seeds whose two orderings both landed on
#: {2, 3, 4}, so the floor already carried what its dependant needed and the
#: pairing was never exercised — removing the pairing entirely left every test
#: green. That accidental overlap is precisely the bug under test, so a fixture
#: that reproduces it cannot detect it.
_C = {5: 0.1, 6: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 0: 0.6, 1: 0.7}
_A = {2: 0.1, 3: 0.2, 0: 0.3, 1: 0.4, 5: 0.5, 4: 0.6, 6: 0.7}


def _every_seed(root, arm, transfer, accuracies, monkeypatch):
    """All of a cell's seeds on disk, laid down the way ten shards leave them.

    One `keep_median` call per seed, because that is what actually happened: each
    shard ran three seeds and kept its own, so nothing was pruned campaign-wide
    and all thirty survive. Calling it once over the whole cell would prune to
    three here and the fixture would not have the seeds the pairing needs.
    """
    for seed, accuracy in accuracies.items():
        _shard(root, arm, transfer, (seed,), monkeypatch, accuracy=accuracy)


def _disjoint(root, monkeypatch, with_floor=True):
    _every_seed(root, "C", "M->U", _C, monkeypatch)
    runs = [_run("C", "M->U", s, a) for s, a in _C.items()]
    if with_floor:
        _every_seed(root, "A", "M->U", _A, monkeypatch)
        runs += [_run("A", "M->U", s, a) for s, a in _A.items()]
    return runs


def _selection(runs, root):
    """What the selection chooses before any pairing touches it."""
    from MIL_CREDA_Benchmark import shards
    return shards.promote_candidates(runs, set(promote.arrived(root)["found"]))


def test_the_fixture_medians_really_are_disjoint(tmp_path, monkeypatch):
    """The premise every test below rests on, asserted rather than assumed.

    If these two ever coincide again the pairing tests stop testing anything and
    go on passing, which is how the first version of this file let the defect
    through.
    """
    root = tmp_path / "returned"
    runs = _disjoint(root, monkeypatch)
    picked = _selection(runs, root)
    assert picked["C|M->U"]["chosen"] == [2, 3, 4]
    assert picked["A|M->U"]["chosen"] == [0, 1, 5]


def test_every_promoted_arm_has_its_floor_at_the_same_seed(tmp_path, monkeypatch):
    """The invariant the whole comparison rests on, stated once.

    Without it `against_floor` finds no floor for that seed, skips the pair, and
    reports a comparison over whatever happened to overlap — silently.
    """
    root = tmp_path / "returned"
    runs = _disjoint(root, monkeypatch)
    into = tmp_path / "into"

    report = promote.promote(runs=runs, root=root, destination=into,
                             superseded=tmp_path / "aside", floor_of=FLOORS)

    landed = {p.name for p in into.glob("*.pt")}
    for seed in report["chosen"]["C|M->U"]["chosen"]:
        assert f"A_M-U_seed{seed}.pt" in landed, (
            f"C promoted seed {seed} and its floor A did not")


def test_an_arms_own_median_is_never_moved_to_match_its_floor(tmp_path, monkeypatch):
    """Matching by changing the arm would promote something that is not its median."""
    root = tmp_path / "returned"
    runs = _disjoint(root, monkeypatch)
    unpaired = _selection(runs, root)

    report = promote.promote(runs=runs, root=root, destination=tmp_path / "into",
                             superseded=tmp_path / "aside", floor_of=FLOORS)

    assert report["chosen"]["C|M->U"]["chosen"] == unpaired["C|M->U"]["chosen"] == [2, 3, 4]
    assert report["chosen"]["A|M->U"]["chosen"] == unpaired["A|M->U"]["chosen"] == [0, 1, 5]


def test_the_extras_a_floor_carries_are_kept_apart_from_its_median(tmp_path, monkeypatch):
    """A seed promoted for somebody else's comparison is not this cell's median.

    Merging the two would make a floor's own checkpoints read as its typical
    artefact when some of them were chosen by a different arm's ordering.
    """
    root = tmp_path / "returned"
    runs = _disjoint(root, monkeypatch)

    report = promote.promote(runs=runs, root=root, destination=tmp_path / "into",
                             superseded=tmp_path / "aside", floor_of=FLOORS)

    floor = report["chosen"]["A|M->U"]
    assert floor["chosen"] == [0, 1, 5]
    assert floor["paired"] == [2, 3, 4]
    assert not set(floor["paired"]) & set(floor["chosen"])
    assert report["chosen"]["C|M->U"]["paired"] == [], "an arm with a floor carries none"


def test_a_floor_that_never_arrived_is_named_not_dropped(tmp_path, monkeypatch):
    """The comparison it blocks would otherwise read as simply absent."""
    root = tmp_path / "returned"
    runs = _disjoint(root, monkeypatch, with_floor=False)

    report = promote.promote(runs=runs, root=root, destination=tmp_path / "into",
                             superseded=tmp_path / "aside", floor_of=FLOORS)

    assert len(report["unpairable"]) == 3
    assert all("its floor A never arrived" in u for u in report["unpairable"])


# --- the display seed: the one every comparative panel is drawn from ---------
#
# `display_seed` picks one seed for the whole grid and nothing arranged for that
# seed to exist in each cell. On the first paired campaign it was present in 1 of
# 21 cells and the figure came out one panel drawn, the rest blank canvas — with
# a PNG emitted, so no check downstream said a word.

def test_the_display_seed_lands_in_every_cell(tmp_path, monkeypatch):
    """Otherwise the panel is blank and the figure still looks like a figure."""
    root = tmp_path / "returned"
    runs = _disjoint(root, monkeypatch)
    into = tmp_path / "into"

    report = promote.promote(runs=runs, root=root, destination=into,
                             superseded=tmp_path / "aside", floor_of=FLOORS)

    seed = report["displaySeed"]
    landed = {p.name for p in into.glob("*.pt")}
    for cell in report["chosen"]:
        arm, transfer = cell.split("|", 1)
        stem = f"{arm}_{transfer.replace('->', '-')}_seed{seed}.pt"
        assert stem in landed, f"{cell} has no checkpoint at the display seed {seed}"


def test_the_display_extra_is_kept_apart_from_median_and_pairing():
    """Three reasons to promote, three fields: only the first is the median."""
    chosen = {"C|M->U": {"chosen": [2, 3, 4], "paired": []},
              "A|M->U": {"chosen": [0, 1, 5], "paired": [2, 3, 4]}}
    found = {("C", "M->U", 6): 1, ("A", "M->U", 6): 1}

    out = promote.displayed(chosen, found, runs=[], seed=6)

    assert out["displaySeed"] == 6
    assert out["chosen"]["C|M->U"]["display"] == [6]
    assert out["chosen"]["C|M->U"]["chosen"] == [2, 3, 4]
    assert out["chosen"]["A|M->U"]["paired"] == [2, 3, 4]


def test_a_cell_that_already_has_the_display_seed_gains_nothing():
    """Promoting it twice would copy one file over itself and inflate the record."""
    chosen = {"C|M->U": {"chosen": [2, 3, 6], "paired": []}}
    out = promote.displayed(chosen, {("C", "M->U", 6): 1}, runs=[], seed=6)
    assert out["chosen"]["C|M->U"]["display"] == []


def test_a_cell_with_no_checkpoint_at_the_display_seed_is_named():
    """Its panel will be blank; that is worth saying before the figure is read."""
    chosen = {"C|M->U": {"chosen": [2, 3, 4], "paired": []}}
    out = promote.displayed(chosen, {}, runs=[], seed=6)
    assert out["undisplayable"] == ["C|M->U seed 6"]
    assert out["chosen"]["C|M->U"]["display"] == []
