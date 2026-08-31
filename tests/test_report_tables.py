"""The phase-one report as it is actually printed, not as its parts could be.

`tables.render` is the table the whole report is built on, and until this file
existed no test called it once. Every claim about that table -- what its `±` is,
what its rows and columns are, that a pilot stamp does not suppress it -- rested
on functions the renderer happens to call, which is a different fact: `spread`
can be right while `render` prints something else entirely, and nothing would
have noticed.

The notebook is the other half. Which tables the report shows, in what order,
and what frames each one are properties of `Benchmark_Report_v1.ipynb` and of
nowhere else, so they are read from the notebook itself rather than asserted
about a module that cannot know them.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pytest

import MIL_CREDA_Benchmark
from MIL_CREDA_Benchmark import config, harness, tables

NOTEBOOKS = config.REPOSITORY / "MIL-CREDA" / "Notebooks"
REPORT = NOTEBOOKS / "Benchmark_Report_v1.ipynb"

LABELS = [f"{s}->{t}" for s, t in config.VERDICT_TRANSFERS]


def _runs(values: dict[tuple[str, str], list[float]],
          metric: str = "targetAccuracy") -> list[dict]:
    """One run per (arm, transfer, repetition), in the shape the campaign writes.

    `contribution` travels because `tables.table` reads it for every row; a
    fixture without it would exercise a narrower renderer than the real one.
    """
    runs = []
    for (arm, transfer), readings in values.items():
        for seed, value in enumerate(readings):
            runs.append({"arm": arm, "transfer": transfer, "seed": seed,
                         metric: value, "contribution": 0.25,
                         # A dispersion that is NOT the one the table prints,
                         # carried on the record so a renderer that read its `±`
                         # from the run instead of computing it across seeds
                         # would print this number and be caught.
                         "batchStdev": 0.30})
    return runs


def _reduction(seeds: int = 3) -> dict:
    return {"seeds": list(range(seeds)), "epochs": config.EPOCHS,
            "backbone": config.BACKBONE, "revision": config.REVISION}


# ------------------------------------------------------------------ the ± sign

def test_the_plus_minus_of_a_printed_cell_is_the_dispersion_across_seeds() -> None:
    """The `±` the report prints is the across-seed dispersion, and it is the
    printed one that matters: the reference paper's `±` is the batch-wise spread
    within one run, which stays plausible at any repetition count and is exactly
    the wrong quantity beside a pilot.

    Three seeds of one cell, and a second transfer that moves by ten times as
    much: the cell's `±` cannot see the other transfer, because the axis it
    pools over is the repetition and nothing else.

    Reachable red: divide by `n` instead of `n - 1` in `spread`, or read the
    `±` off the record's own `batchStdev`, and the printed cell moves.
    """
    seeds = [0.50, 0.60, 0.70]
    runs = _runs({("A", LABELS[0]): seeds,
                  ("A", LABELS[1]): [0.10, 0.60, 1.00]})
    printed = tables.render(runs, "targetAccuracy", _reduction())

    across_seeds = statistics.stdev(seeds)
    assert across_seeds == pytest.approx(0.1)
    assert f"{60.0:.1f} ± {across_seeds * 100:.1f}" in printed

    # and it is neither the population dispersion nor the record's own field
    assert f"± {statistics.pstdev(seeds) * 100:.1f}" not in printed
    assert "± 30.0" not in printed


def test_a_cell_of_one_repetition_prints_a_zero_dispersion_it_did_not_measure() -> None:
    """Zero by construction, which is why the stamp above the table says so.

    Held here because it is the state a pilot is read in, and a `±` that came out
    of nowhere would be indistinguishable from agreement.
    """
    printed = tables.render(_runs({("A", LABELS[0]): [0.42]}),
                            "targetAccuracy", _reduction(seeds=1))
    assert "42.0 ± 0.0" in printed


# ------------------------------------------------------- rows, columns, average

def test_the_table_is_arms_by_display_name_over_the_six_transfers_and_an_average() -> None:
    """Rows are arms by display name, columns the six transfers plus `Prom.`.

    An identifier is not a name: a reader of the report has no table mapping `G`
    to `MIL-CREDA`, and the row that says `G` says nothing to them.

    Reachable red: print `row['arm']`, drop the average column, or lose a
    transfer from the header.
    """
    runs = _runs({(arm, label): [0.5, 0.6]
                  for arm in ("A", "B", "G") for label in LABELS})
    printed = tables.render(runs, "targetAccuracy", _reduction(seeds=2),
                            markdown=True)
    header = [cell.strip() for cell in printed.splitlines()[0].strip("|").split("|")]

    assert header == ["Método", *LABELS, "Prom."]
    names = [line.split("|")[1].strip().strip("`") for line in printed.splitlines()[2:]]
    assert names == [config.NAME_OF[a] for a in ("A", "B", "G")]
    for identifier in ("A", "B", "G"):
        assert f"`{identifier}`" not in printed, "the table names an arm by its id"


def test_the_average_column_averages_the_transfers_and_not_the_repetitions() -> None:
    """One transfer counts once however many repetitions it ran.

    Reachable red: average the runs instead of the per-transfer means and the
    transfer with more seeds starts weighing more than the others.
    """
    runs = _runs({("A", LABELS[0]): [0.20, 0.20, 0.20, 0.20],
                  ("A", LABELS[1]): [1.00]})
    row = tables.table(runs, "targetAccuracy")[0]
    assert row["avg"] == pytest.approx(0.6)


# ------------------------------------------------------------ below the floor

def test_below_the_repetition_floor_the_reason_is_stamped_and_the_table_still_prints() -> None:
    """The pilot is the same program as the campaign, printed with a warning.

    Two halves, and neither is enough alone: the header has to say why no verdict
    is granted, and the table has to appear anyway -- a run that hid its numbers
    until it reached full scale would be a second program nobody had exercised.

    Reachable red: return `[]` from `_stamp` below the floor, or suppress the
    table when the stamp is not empty.
    """
    runs = _runs({(arm, LABELS[0]): [0.5] for arm in ("A", "G")})
    stamped = tables.stamp(_reduction(seeds=1))
    printed = tables.render(runs, "targetAccuracy", _reduction(seeds=1))

    assert "1 repetición(es)" in stamped
    assert "piloto" in stamped and str(len(config.FULL_SEEDS)) in stamped
    assert config.NAME_OF["G"] in printed and config.NAME_OF["A"] in printed
    assert printed.count("\n") >= 2, "the table was suppressed instead of stamped"


def test_at_the_declared_scale_nothing_is_stamped() -> None:
    """Otherwise the warning is decoration and stops being read."""
    assert tables._stamp({"seeds": list(config.FULL_SEEDS)}) == []
    assert "piloto" not in tables.stamp({"seeds": list(config.FULL_SEEDS)})


# ------------------------------------------------------------------- the rungs

def test_a_rung_is_named_by_display_names_and_never_by_identifiers() -> None:
    """`Baseline → MIL-Baseline`, not `A->B`.

    Reachable red: return `f"{left}->{right}"` and both halves of this fail --
    the name and the arrow that separates a reading from an expression.
    """
    assert tables.rung_name("A", "B") == "Baseline → MIL-Baseline"
    for left, right, _ in config.LADDER:
        named = tables.rung_name(left, right)
        assert named == f"{config.NAME_OF[left]} → {config.NAME_OF[right]}"
        assert "->" not in named


def test_the_rung_table_labels_its_rows_with_the_name_and_not_the_pair() -> None:
    """The renderer has to use it, or the rule lives in a function nobody calls."""
    cell = {arm: {"targetAccuracy": {"mean": 0.5 + index / 10, "stdev": 0.0, "n": 2}}
            for index, arm in enumerate(config.ARMS_BY_ID)}
    printed = tables.render_rungs({"grid": {LABELS[0]: cell}}, "targetAccuracy")
    assert tables.rung_name(*config.LADDER[0][:2]) in printed
    assert f"{config.LADDER[0][0]}->{config.LADDER[0][1]}" not in printed


# --------------------------------------------------- Spanish prose, English keys

#: Words a data contract must never contain. The convention is not "no accents":
#: `exactitudDestino` is ASCII and is still prose in the wrong file.
SPANISH_IN_A_KEY = ("exactitud", "tasa", "semilla", "techo", "brazo", "corrida",
                    "transferencia", "piso", "peldano", "peldaño", "bolsa",
                    "epoca", "época", "ruido", "mediana")


def _keys(node, found: set) -> set:
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            _keys(value, found)
    elif isinstance(node, list):
        for item in node:
            _keys(item, found)
    return found


def test_the_headings_are_spanish_and_the_keys_of_the_record_are_english() -> None:
    """One convention, two directions, and the pair is the whole rule.

    The report is read in Spanish, so its headings are Spanish. The record is a
    data contract every reader parses, so its keys stay English -- a key renamed
    into the report's language breaks every consumer for the sake of a reader who
    never sees it.

    Reachable red: rename `Método` to `Method`, or any declared key to a Spanish
    one -- `techo` for `ceiling` is the tempting case, and it is caught here.
    """
    from MIL_CREDA_Benchmark import __benchmark__

    printed = tables.render(_runs({("A", LABELS[0]): [0.5]}), "targetAccuracy",
                            _reduction(seeds=1), markdown=True)
    assert "Método" in printed and "Prom." in printed
    assert "Method" not in printed

    keys = _keys(__benchmark__, set()) | set(config.DIMENSIONS)
    for key in keys:
        assert key.isascii(), f"the key {key!r} is not a data contract, it is prose"
        lowered = key.lower()
        for word in SPANISH_IN_A_KEY:
            assert word not in lowered, f"the key {key!r} is written in the prose's language"

    # And the method names are English on both sides, because they are the join
    # between the two: the table prints them and the record keys by them.
    for name in config.NAME_OF.values():
        assert name.isascii(), f"{name!r} cannot be both a display name and a key"


# ------------------------------------------------- what the notebook actually shows

def _shown(notebook) -> list[tuple[str, str]]:
    """Every `show(tables.X(...))` of a notebook, in the order it runs.

    Only the shown ones: the cell that writes the report to disk repeats every
    call in a list, and counting those would report each table twice.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    calls = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for match in re.finditer(r"show\(\s*tables\.(\w+)\(([^)]*)", source):
            calls.append((match.group(1), match.group(2).replace("\n", " ")))
    return calls


def _metric_of(arguments: str) -> str | None:
    for metric in ("seconds", "sourceAccuracy", "targetAccuracy"):
        if f'"{metric}"' in arguments:
            return metric
    return None


def test_the_report_shows_the_target_table_and_its_source_complement() -> None:
    """Two tables, and the second is not optional.

    A method that wins on target by wrecking source is the degenerate case, and
    one table cannot tell it from a success. Target is the headline and source
    the complement, so source is read first and target last.

    Reachable red: delete either `render(runs, "…Accuracy", …)` call from the
    notebook, or show target before source.
    """
    levels = [_metric_of(args) for name, args in _shown(REPORT) if name == "render"]
    levels = [metric for metric in levels if metric]

    assert levels.count("sourceAccuracy") == 1, "the source complement is not shown once"
    assert levels.count("targetAccuracy") == 1, "the headline is not shown once"
    assert levels.index("seconds") < levels.index("sourceAccuracy") < \
        levels.index("targetAccuracy"), "the declared reading order is not the one printed"


def test_each_level_table_is_followed_by_the_ladder_of_that_same_metric() -> None:
    """Levels say who is ahead; only the rungs say which piece did the work, so
    the two are read together and in that order.

    What is asserted is what the notebook still holds: the rung table of a metric
    comes after that metric's level table and before the next metric's. It is NOT
    the literal `directly below` of the agreement -- the gains table sits between
    them -- and writing the stronger assertion would have made a test that fails
    for being right.

    Reachable red: move `render_rungs(summary, "sourceAccuracy")` after the
    target level table, or drop it.
    """
    shown = [(name, _metric_of(args)) for name, args in _shown(REPORT)]
    for metric in ("sourceAccuracy", "targetAccuracy"):
        level = shown.index(("render", metric))
        rungs = shown.index(("render_rungs", metric))
        assert level < rungs, f"the ladder of {metric} is printed before its levels"
        later = [index for index, entry in enumerate(shown)
                 if entry[0] == "render" and entry[1] and entry[1] != metric
                 and index > level]
        assert all(index > rungs for index in later), \
            f"another metric's table is printed between {metric} and its ladder"


def test_every_table_the_report_shows_declares_what_it_is_looking_for() -> None:
    """The three short lines above a table: what is measured, why, and which way
    is better. A number printed with no target is a number nobody can read.

    Reachable red: drop a key from `objective`'s `metas` -- the reading it frames
    still prints, and the fallback text is what this catches.
    """
    shown = _shown(REPORT)
    asked = [args.strip().strip('"') for name, args in shown if name == "objective"]
    assert asked, "the report frames nothing"
    for key in asked:
        stated = tables.objective(key)
        assert "sin objetivo declarado" not in stated, f"{key} is framed by a placeholder"
        assert len(stated) > 80, f"{key} is framed by a line too short to say three things"
        assert "Buscamos" in stated, f"{key} never says which way is better"

    renders = [index for index, (name, _) in enumerate(shown)
               if name.startswith("render")]
    for index in renders:
        before = [name for name, _ in shown[:index]]
        assert "objective" in before, "a table is printed before anything framed it"
        assert before[-1] in ("objective", *[n for n in before if n.startswith("render")]), \
            "a table is printed with a conclusion, not an objective, above it"


def test_the_lines_above_a_table_are_computed_from_the_protocol_and_never_typed(
        monkeypatch) -> None:
    """The milestone a reading is measured against comes out of `config`.

    Typed, it would age exactly like a typed measurement: the day the classes or
    the evaluation bags change, the sentence still names the old chance level and
    nothing contradicts it.

    Reachable red: write `0.100` into the target-accuracy line and it stops
    following the protocol it claims to describe.
    """
    monkeypatch.setattr(config, "CLASSES", 5)
    stated = tables.objective("targetAccuracy")
    assert "0.200" in stated and "0.100" not in stated

    monkeypatch.setattr(config, "VALID_BAGS", 4)
    monkeypatch.setattr(config, "SEARCH_RESOLUTION", 1.0 / 4)
    assert "0.25" in tables.objective("ceilings")


def test_each_level_table_is_followed_by_its_own_computed_conclusion() -> None:
    """And the conclusion belongs to the table it follows.

    Reachable red: point the conclusion under the target table at
    `"sourceAccuracy"` and it reads as the section's own reading while
    describing the section above.
    """
    shown = [(name, _metric_of(args)) for name, args in _shown(REPORT)]
    for metric in ("seconds", "sourceAccuracy", "targetAccuracy"):
        level = shown.index(("render", metric))
        concluded = next(((name, seen) for name, seen in shown[level + 1:]
                          if name.startswith("conclusion")), None)
        assert concluded is not None, f"the {metric} table concludes nothing"
        assert concluded[1] == metric, \
            f"the reading under the {metric} table is about {concluded[1]}"


def test_a_cell_shows_one_table_and_not_two() -> None:
    """One table per cell, so the framing above it belongs to one reading.

    A cell whose renders sit in the two branches of one `if` still shows one:
    that is the wall-time cell, which prints per-run or pooled and never both.

    Reachable red: add a second `show(tables.render(...))` to any cell.
    """
    cells = json.loads(REPORT.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        drawn = re.findall(r"show\(\s*tables\.(render\w*)\(", source)
        if len(drawn) <= 1:
            continue
        assert "if " in source and "else" in source, \
            f"one cell shows {len(drawn)} tables at once: {drawn}"
        assert len(drawn) == 2, f"one cell shows {len(drawn)} tables: {drawn}"


# ------------------------------------------ the conclusions under the figures

def _reading(arm: str, transfer: str, seed: int, *, ratio: float, cross: float,
             apart: float, separability: float, mass: float, spread: float) -> dict:
    return {"arm": arm, "transfer": transfer, "seed": seed, "median": True,
            "geometry": {"ratio": ratio, "crossDomainSameClass": cross,
                         "betweenClasses": apart},
            "domainSeparability": separability,
            "correspondence": {"massOnTrueClass": mass, "chance": 1.0 / config.CLASSES},
            "attentionSpread": spread}


def _record(swap: bool = False) -> dict:
    """One record with every phase the report concludes over, and its permutation.

    The permutation swaps what the floor measured with what the complete method
    measured. Nothing is added or removed, so a conclusion that comes out the
    same under it is not reading its own numbers.
    """
    strong = dict(ratio=0.40, cross=0.40, apart=1.00, separability=0.52,
                  mass=0.70, spread=0.55)
    weak = dict(ratio=0.90, cross=0.90, apart=1.00, separability=0.95,
                mass=0.20, spread=0.995)
    top, floor = (weak, strong) if swap else (strong, weak)
    readings = []
    for transfer in LABELS[:3]:
        readings.append(_reading("G", transfer, 0, **top))
        readings.append(_reading("B", transfer, 0, **floor))
    runs = _runs({(arm, label): ([0.8, 0.9] if arm == ("B" if swap else "G")
                                 else [0.2, 0.3])
                  for arm in ("B", "G") for label in LABELS[:3]})
    faster = "B" if swap else "G"
    for run in runs:
        run["seconds"] = 10.0 if run["arm"] == faster else 20.0
        run["sourceAccuracy"] = run["targetAccuracy"]
    scored = [{"arm": arm, "transfer": transfer,
               "hits": 9 if arm == ("B" if swap else "G") else 2,
               "classes": 10,
               "mass": 0.7 if arm == ("B" if swap else "G") else 0.2}
              for arm in config.BAG_PANELS for transfer in LABELS[:3]]
    return {"runs": runs, "reduction": _reduction(seeds=2),
            "readings": readings, "correspondence": scored}


def test_every_conclusion_the_report_produces_is_read_off_its_own_numbers() -> None:
    """The rule that makes a computed conclusion worth having: permute the record
    and every sentence has to move.

    A conclusion written by hand under a figure fixes itself -- the figure is
    regenerated from other data and the sentence stays, and it is believed
    exactly as much as before. This covers the six readings under the figures
    (geometry, distances, separability, mass, attention, correspondence) together
    with the phase-one levels, because `tables.conclusions` is the one entry
    point the report and the verification both go through.

    Reachable red: replace any one of those conclusions with a fixed sentence and
    it comes out identical under the permutation.
    """
    produced = tables.conclusions(_record())
    permuted = tables.conclusions(_record(swap=True))

    expected = {"niveles:seconds", "niveles:sourceAccuracy", "niveles:targetAccuracy",
                "geometría", "distancias", "separabilidad", "masa", "atención",
                "correspondencia"}
    assert expected <= set(produced), f"nothing concluded for {expected - set(produced)}"
    assert set(produced) == set(permuted)
    for key, said in produced.items():
        assert said.strip(), f"{key} concluded nothing at all"
        assert said != permuted[key], f"{key} says the same whatever the numbers say"


def test_the_figure_conclusions_name_the_method_the_measurement_favours() -> None:
    """Not merely different under a permutation: different in the right direction.

    A sentence that changed by naming the loser would pass the permutation and
    still be wrong, so the two readings whose subject is a single arm are pinned
    to the arm the numbers actually favour.

    Reachable red: order by `+_mean(...)` in `conclusion_mass`, or move the
    uniform-attention threshold past the value that triggers it.
    """
    readings = _record()["readings"]
    assert config.NAME_OF["G"] in tables.conclusion_mass(readings).split(".")[0]
    swapped = _record(swap=True)["readings"]
    assert config.NAME_OF["B"] in tables.conclusion_mass(swapped).split(".")[0]

    # attention: the floor's 0.995 is above the declared uniform threshold and
    # the method's 0.55 is not, so the two arms are described differently.
    said = tables.conclusion_attention(readings)
    assert "casi por igual" in said
    assert said.index(config.NAME_OF["B"]) < said.index("casi por igual")


# ---------------------------------------------------- what a decimal can mean

def test_the_printed_precision_is_the_granularity_of_the_instrument() -> None:
    """One decimal, and the number that justifies it computed rather than typed.

    Thirty-six evaluation bags make accuracy a count out of thirty-six: it moves
    in steps of `100 / 36 = 2.78` points and can take no value in between. A
    second decimal would print hundredths of a point on a scale whose smallest
    real step is nearly three points -- false precision, and the kind a reader
    believes because it is printed.

    Both halves are asserted together because either alone passes while the
    report is wrong: the stamp can state the granularity while the table prints
    four decimals, and the table can print one decimal while the stamp claims a
    resolution the run does not have.

    Reachable red: print a second decimal for an accuracy, or let the stamp
    state a granularity it did not compute from `EVAL_BAGS`.
    """
    assert config.EVAL_BAGS == 36
    granularity = 100 / config.EVAL_BAGS
    assert granularity == pytest.approx(2.7777, abs=1e-4)

    # the stamp says it, and says it from the constant
    stamped = tables.stamp(_reduction())
    assert f"{granularity:.2f}" in stamped
    assert f"{config.EVAL_BAGS} bolsas de evaluación" in stamped

    # the table prints one decimal for an accuracy: a value with more resolution
    # than the instrument has comes out rounded to the instrument
    runs = _runs({("G", label): [0.123456, 0.123456] for label in LABELS})
    printed = tables.render(runs, "targetAccuracy", _reduction(seeds=2),
                            markdown=True)
    assert "12.3" in printed
    assert "12.35" not in printed and "12.3456" not in printed

    # and the granularity is coarser than the decimal that was NOT printed, which
    # is the whole reason there is only one
    assert granularity > 0.1

    # a descriptive quantity is not an accuracy and keeps its two decimals: the
    # rule is about the instrument, not about the renderer's taste
    assert "targetAccuracy" in tables.PERCENT and "seconds" not in tables.PERCENT


def test_the_benchmark_declares_the_components_its_objective_is_made_of() -> None:
    """The two terms an arm's objective is made of, and the dimension carrying
    their ratio, named in the report contract.

    `contribution` alone is a numerator: a term that commanded nothing and a term
    that was scaled to nothing both print small, and only the share separates
    them. So the contract names both terms and the share, and every name it uses
    has to be a field the record actually carries -- a declared component that no
    run writes is a contract describing a different experiment.

    Two terms and not three: the harness applies one shared coefficient to the
    global and local terms together, so `supervised` and `contribution` are what
    the objective is made of here. Splitting them would need two coefficients,
    which is a change to the experiment and not a declaration.

    Reachable red: drop `components` from the contract, or declare a term the run
    record does not write.
    """
    import ast

    report = MIL_CREDA_Benchmark.__benchmark__["report"]
    components = report["components"]
    assert components["terms"] == ["supervised", "contribution"]
    assert components["share"] == "adaptationShare"

    # every declared name is a dimension the contract already knows how to read
    for name in [*components["terms"], components["share"]]:
        assert name in report["dimensions"]

    # and every one is a field the harness actually writes on a run, read from
    # the record's own literal rather than from a list kept by hand
    tree = ast.parse(Path(harness.__file__).read_text(encoding="utf-8"))
    written = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if {"targetAccuracy", "curve", "seed"} <= keys:
            written = keys
    assert written, "the run record was not found in the harness"
    for name in [*components["terms"], components["share"]]:
        assert name in written, f"{name} is declared and never written"
