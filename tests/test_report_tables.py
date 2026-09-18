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
import math
import re
import statistics
from pathlib import Path

import pytest

import MIL_CREDA_Benchmark
from MIL_CREDA_Benchmark import config, harness, tables

NOTEBOOKS = config.REPOSITORY / "MIL-CREDA" / "Notebooks"
REPORT = NOTEBOOKS / "Results_v1.ipynb"

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


def _filas(rendered: str) -> list[str]:
    """Las filas de datos de una tabla de dos bloques.

    Se reconocen por la primera columna, que es `Ruido` en las cinco tablas
    unificadas, y no por un backtick: el nombre del método dejó de ser lo
    primero de la fila el día que el material pasó a serlo.
    """
    marcas = (f"| {tables.NOISE_CLEAN} |", f"| {tables.NOISE_DIRTY} |")
    return [l for l in rendered.splitlines() if l.startswith(marcas)]


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
    runs = _runs({("B", LABELS[0]): seeds,
                  ("B", LABELS[1]): [0.10, 0.60, 1.00]})
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
    printed = tables.render(_runs({("B", LABELS[0]): [0.42]}),
                            "targetAccuracy", _reduction(seeds=1))
    assert "42.0 ± 0.0" in printed


# ------------------------------------------------------- rows, columns, average

def test_the_table_is_arms_by_display_name_over_the_six_transfers_and_an_average() -> None:
    """Rows are arms by display name, columns `Ruido`, the six transfers, `Prom.`.

    An identifier is not a name: a reader of the report has no table mapping `G`
    to `MIL-CREDA`, and the row that says `G` says nothing to them.

    `Ruido` leads and says `sin` on every row here, because no contaminated
    campaign was asked for: one shape, whether the section has one block or two.

    Reachable red: print `row['arm']`, drop the average column, lose a
    transfer from the header, or put the noise column anywhere but first.
    """
    runs = _runs({(arm, label): [0.5, 0.6]
                  for arm in ("B", "E", "G") for label in LABELS})
    printed = tables.render(runs, "targetAccuracy", _reduction(seeds=2),
                            markdown=True)
    header = [cell.strip() for cell in printed.splitlines()[0].strip("|").split("|")]

    assert header == [tables.NOISE_COLUMN, "Método", *LABELS, "Prom.", "Puesto"]
    rows = printed.splitlines()[2:]
    assert [line.split("|")[1].strip() for line in rows] == [tables.NOISE_CLEAN] * 3
    names = [line.split("|")[2].strip().strip("`") for line in rows]
    assert names == [config.NAME_OF[a] for a in ("B", "E", "G")]
    for identifier in ("B", "E", "G"):
        assert f"`{identifier}`" not in printed, "the table names an arm by its id"


def test_the_average_column_averages_the_transfers_and_not_the_repetitions() -> None:
    """One transfer counts once however many repetitions it ran.

    Reachable red: average the runs instead of the per-transfer means and the
    transfer with more seeds starts weighing more than the others.
    """
    runs = _runs({("B", LABELS[0]): [0.20, 0.20, 0.20, 0.20],
                  ("B", LABELS[1]): [1.00]})
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
    runs = _runs({(arm, LABELS[0]): [0.5] for arm in ("B", "G")})
    stamped = tables.stamp(_reduction(seeds=1))
    printed = tables.render(runs, "targetAccuracy", _reduction(seeds=1))

    assert "1 repetición(es)" in stamped
    assert "piloto" in stamped and str(len(config.FULL_SEEDS)) in stamped
    assert config.NAME_OF["G"] in printed and config.NAME_OF["B"] in printed
    assert printed.count("\n") >= 2, "the table was suppressed instead of stamped"


def test_at_the_declared_scale_nothing_is_stamped() -> None:
    """Otherwise the warning is decoration and stops being read."""
    assert tables._stamp({"seeds": list(config.FULL_SEEDS)}) == []
    assert "piloto" not in tables.stamp({"seeds": list(config.FULL_SEEDS)})


# `test_a_rung_is_named_by_display_names_and_never_by_identifiers` and
# `test_the_rung_table_labels_its_rows_with_the_name_and_not_the_pair` removed:
# `tables.rung_name` and `tables.render_rungs` are retired along with the
# rungs (gains) table and its conclusion -- none of the six required sections
# of `Results_v1.ipynb` shows a ladder of adjacent-arm differences any more.
# Section 2/3's own attribution reading (each arm against its own floor) is
# `tables.conclusion`, already covered by
# `test_every_conclusion_the_report_produces_is_read_off_its_own_numbers` and
# by `test_the_table_is_arms_by_display_name_over_the_six_transfers_and_an_average`.


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

    printed = tables.render(_runs({("B", LABELS[0]): [0.5]}), "targetAccuracy",
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
    for metric in ("sourceAccuracy", "targetAccuracy"):
        if f'"{metric}"' in arguments:
            return metric
    return None


#: Which renderer draws each metric's level table. `seconds`/`peakMiB` are gone
#: from the declaration along with every reader of them, so both remaining
#: metrics pool and are printed by `render`. The claims below -- the reading
#: order, and that every level table is followed by its own conclusion -- are
#: about the section, not about which function drew it, so they read the
#: renderer from here instead of assuming one.
LEVEL_RENDERER = {"sourceAccuracy": "render",
                  "targetAccuracy": "render"}


def _levels(notebook) -> list[str]:
    """The metrics whose level table the notebook shows, in the printed order."""
    shown = []
    for name, args in _shown(notebook):
        metric = _metric_of(args)
        if metric and name == LEVEL_RENDERER[metric]:
            shown.append(metric)
    return shown


def test_the_report_shows_the_target_table_and_its_source_complement() -> None:
    """Two tables, and the second is not optional.

    A method that wins on target by wrecking source is the degenerate case, and
    one table cannot tell it from a success. Target is the headline and source
    the complement, so source is read first and target last.

    No `seconds` table precedes them any more: `seconds`/`peakMiB` were
    retired from the declaration and from this report along with every reader
    of them, so the reading order this section owes is only source-before-
    target.

    Reachable red: delete either `render(runs, "…Accuracy", …)` call from the
    notebook, or show target before source.
    """
    levels = _levels(REPORT)

    assert levels.count("sourceAccuracy") == 1, "the source complement is not shown once"
    assert levels.count("targetAccuracy") == 1, "the headline is not shown once"
    assert levels.index("sourceAccuracy") < levels.index("targetAccuracy"), \
        "the declared reading order is not the one printed"


# `test_each_level_table_is_followed_by_the_ladder_of_that_same_metric` removed:
# `render_rungs` is retired along with the rungs (gains) table, so there is no
# ladder for a level table to be followed by any more. What replaces it is the
# rank column inside the level table itself, already covered by
# `test_the_table_is_arms_by_display_name_over_the_six_transfers_and_an_average`.


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
    for metric in ("sourceAccuracy", "targetAccuracy"):
        level = shown.index((LEVEL_RENDERER[metric], metric))
        concluded = next(((name, seen) for name, seen in shown[level + 1:]
                          if name.startswith("conclusion")), None)
        assert concluded is not None, f"the {metric} table concludes nothing"
        assert concluded[1] == metric, \
            f"the reading under the {metric} table is about {concluded[1]}"


def test_a_cell_shows_one_table_and_not_two() -> None:
    """One table per cell, so the framing above it belongs to one reading.

    A cell whose renders sit in the two branches of one `if` still shows one.
    The wall-time cell used to be that case -- per-run or pooled, never both --
    and stopped being it when `seconds` lost its pooled branch; the allowance
    stays because a cell is entitled to one table however it chooses it.

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

    expected = {"niveles:sourceAccuracy", "niveles:targetAccuracy",
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
    assert "extremo más concentrado" in said
    assert said.index(config.NAME_OF["G"]) < said.index("extremo más concentrado")


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


# ------------------------------------------- what the declaration forbids pooling

def test_render_refuses_a_dimension_the_declaration_calls_per_run(monkeypatch) -> None:
    """`render`'s guard, enforced instead of only written down.

    No reading of a `perRun` dimension is stable enough to stand for the
    method, or even for one machine across two of its own runs, so a
    `mean ± stdev` over it describes none of the runs behind it and reads
    exactly as rigorous as one that does. `render` checked nothing and
    pooled whatever it was handed; the guard is what changed that.

    `config.DIMENSIONS` declares no `perRun` dimension today --
    `seconds`/`peakMiB` are gone entirely (commit `60836d2`,
    "...drop timing"), not merely reclassified, along with the
    `render_per_run`/`conclusion_per_run` that used to print them
    (commit `177ce09`). The guard itself is not retired -- `pooling.refuse`
    stays generic over whatever the declaration names -- so a synthetic
    `perRun` dimension drives it here: "the class, not the instance."

    Read from the declaration and not listed here: which dimensions do not pool
    is the target's own statement, and a copy of the list in this file would be a
    second source of truth that ages in silence.

    Reachable red: delete the guard and `render` prints the pooled table again,
    ± and all.
    """
    monkeypatch.setattr(config, "DIMENSIONS", {**config.DIMENSIONS, "seconds": None})
    monkeypatch.setitem(MIL_CREDA_Benchmark.__benchmark__["distribution"],
                        "perRun", ["seconds"])
    declared = MIL_CREDA_Benchmark.__benchmark__["distribution"]["perRun"]
    assert tables.per_run_dimensions() == list(declared), \
        "the renderer's idea of what does not pool is not the declaration's"

    for metric in declared:
        runs = _runs({("G", LABELS[0]): [10.0, 30.0]}, metric=metric)
        with pytest.raises(ValueError) as raised:
            tables.render(runs, metric, _reduction(seeds=2))
        said = str(raised.value)
        assert metric in said and "perRun" in said, \
            f"the refusal does not say which dimension it is about: {said}"
        assert "gridPerRun" in said, "the refusal names no way forward"

    # And it still renders what the same declaration says does pool, so the
    # guard is a refusal and not an outage.
    pooled = MIL_CREDA_Benchmark.__benchmark__["distribution"]["poolable"]
    assert "targetAccuracy" in pooled
    assert tables.render(_runs({("G", LABELS[0]): [0.8, 0.9]}),
                         "targetAccuracy", _reduction(seeds=2))


#: Every way this module collapses a caller-supplied dimension into one number,
#: with a fixture in the shape that entry point actually reads. Written as a list
#: and driven in one loop on purpose: the defect this covers was a guard put on
#: `render` alone while seven siblings went on pooling the same numbers, and a
#: test that named one function per case would have been written the same way.
#:
#: `render_per_run` and `conclusion_per_run` are gone along with `seconds`/
#: `peakMiB` themselves -- there is no `perRun` dimension left that a form the
#: declaration permits would need to print. `render_rungs`/`conclusion_rungs`
#: (the rungs table) and `paired_gains`/`render_gains` (the gains table) are
#: gone for the same reason every reader of them is: neither has a place in
#: the six required sections of `Results_v1.ipynb`.
def _pooling_calls(metric: str):
    """One callable per aggregating entry point, ready to invoke with `metric`."""
    runs = _runs({(arm, label): [10.0 + index, 30.0 + index]
                  for index, arm in enumerate(("B", "E", "F", "G"))
                  for label in LABELS[:2]}, metric=metric)
    return {
        "cells": lambda: tables.cells(runs, metric),
        "table": lambda: tables.table(runs, metric),
        "render": lambda: tables.render(runs, metric, _reduction(seeds=2)),
        "conclusion": lambda: tables.conclusion(runs, metric, _reduction(seeds=2)),
        "ranking": lambda: tables.ranking(runs, metric),
        "best_transfers": lambda: tables.best_transfers(runs, metric=metric),
    }


def test_no_function_that_pools_a_dimension_accepts_one_the_declaration_forbids() -> None:
    """The class, not the instance: every aggregator refuses, not only `render`.

    `render` was guarded and `conclusion` was not, and `conclusion` sorts the
    same cells by their average and prints the extremes of it. A refusal that
    covers the table and leaves the sentence beneath it open has not removed the
    pooled number from the report, it has moved it one function to the right --
    and the sentence is the half a reader quotes.

    Which dimensions are refused is read from the declaration, never listed
    here. Which functions have to refuse is the list above, and it is the part
    that has to grow when a new aggregator is written.

    Reachable red: delete `_refuse_pooling`'s call from any one of `cells`,
    `render`, `conclusion`, `ranking` or `best_transfers`, and the entry
    points that reach the number through it stop refusing.

    `config.DIMENSIONS` declares no `perRun` dimension today (see the sibling
    test above, `test_render_refuses_a_dimension_the_declaration_calls_per_
    run`, for why): a synthetic one drives the mechanism here instead of
    relying on a live example that no longer exists.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(config, "DIMENSIONS", {**config.DIMENSIONS, "seconds": None})
        monkeypatch.setitem(MIL_CREDA_Benchmark.__benchmark__["distribution"],
                            "perRun", ["seconds"])
        declared = MIL_CREDA_Benchmark.__benchmark__["distribution"]["perRun"]

        for metric in declared:
            for name, call in _pooling_calls(metric).items():
                # `pytest.raises` on its own reports `DID NOT RAISE ValueError` and
                # names nothing: with ten entry points in one loop that failure tells
                # whoever broke a guard only that one of them is open. The point of
                # driving the family together is lost if the red does not say which.
                try:
                    call()
                except ValueError as refusal:
                    said = str(refusal)
                else:
                    pytest.fail(f"`{name}` pooled `{metric}` instead of refusing it")
                assert metric in said and "perRun" in said, \
                    f"`{name}` refused without saying which dimension: {said}"
                assert "gridPerRun" in said, \
                    f"`{name}` refuses and names no way forward: {said}"
    finally:
        monkeypatch.undo()


def test_the_same_functions_still_pool_what_the_declaration_says_pools() -> None:
    """The other half, and the one that makes the refusal mean something.

    A guard proved only by what it rejects is indistinguishable from a function
    that raises on everything, and it is also indistinguishable from one whose
    fixture was too thin to produce a number in the first place. So every entry
    point above is driven again on a `poolable` dimension and has to come back
    with a number: the refusal is a refusal, not an outage, and the path behind
    it really does collapse runs into one.

    A number and not merely a non-empty result, because four of these answer an
    unmeasurable fixture with a sentence -- `(sin peldaños medibles)`, `(sin
    pares para ...)`, `Sin corridas`, `Sin peldaños medibles` -- and every one
    of those is truthy. Matched at the start of the string and not anywhere in
    it: a rung's own reading says «sin ponderar», and looking for the word
    loose failed a table that had just printed twelve rows of numbers. A
    fixture that stopped producing rows would leave both halves of this file
    green: the refusal proved against nothing, and the pooling proved against a
    string that says nothing was pooled.

    Reachable red: make `_refuse_pooling` raise unconditionally, or thin the
    fixture until no arm has two transfers.
    """
    pooled = MIL_CREDA_Benchmark.__benchmark__["distribution"]["poolable"]
    assert "targetAccuracy" in pooled

    for name, call in _pooling_calls("targetAccuracy").items():
        produced = call()
        assert produced, f"`{name}` produced nothing on a dimension that pools"
        if isinstance(produced, str):
            assert re.search(r"\d", produced), \
                f"`{name}` came back with prose and no number: {produced!r}"
            assert not produced.lstrip().lower().startswith(("(sin", "sin ")), \
                f"`{name}` reported nothing measurable, so nothing was pooled: " \
                f"{produced!r}"


def test_the_reports_conclusions_ask_for_no_per_run_dimension() -> None:
    """`conclusions()` asked for `seconds` on every run, and got it.

    It is the single entry point the report and the verification both go
    through, and its metric list was written by hand: `("seconds",
    "sourceAccuracy", "targetAccuracy")`. So the one pooled reading the
    declaration forbids was not merely reachable, it was requested -- best
    average and worst average over numbers that do not describe one machine
    across two of its own runs.

    Read off `per_run_dimensions()` rather than checking for the word
    `seconds`: the list that put it there was a hand-written one, and a test
    with its own hand-written list ages the same way.

    Reachable red: put any per-run dimension back into that loop.
    """
    produced = tables.conclusions(_record())
    # Not merely "no forbidden key": a record that concluded no level at all
    # would satisfy that and prove nothing. The loop below has to be looking at
    # a `conclusions` that did produce levels.
    assert [key for key in produced if key.startswith("niveles:")], \
        "no level was concluded, so there was nothing for this to check"
    for metric in tables.per_run_dimensions():
        offending = [key for key in produced if key.endswith(f":{metric}")]
        assert not offending, \
            f"the report concludes {offending} over a dimension the declaration " \
            f"calls perRun"


def test_the_report_asks_for_no_pooled_table_of_a_per_run_dimension() -> None:
    """The other half: the guard refuses, and the notebook stopped asking.

    A refusal the report walks into on every run is a broken notebook, not a
    protected one -- the pooled `seconds` block was written three times across
    two cells, and `render_per_run` beside it was already the form the
    declaration permits.

    Scanned over every cell and not only the shown ones: the cell that writes
    the report to disk builds its blocks in a list, and two of the three calls
    lived there.

    Reachable red: put `tables.render(runs, "seconds", ...)` back into any cell.
    """
    cells = json.loads(REPORT.read_text(encoding="utf-8"))["cells"]
    for metric in tables.per_run_dimensions():
        asked = []
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            for match in re.finditer(r"tables\.(\w+)\(([^)]*)", source):
                if match.group(1) == "render" and f'"{metric}"' in match.group(2):
                    asked.append(index)
        assert not asked, \
            f"the report pools `{metric}` in cell(s) {asked}: the declaration " \
            f"calls it perRun and `render` refuses it"


# `_grid_per_run`, `test_the_inline_seconds_table_collapses_the_seed_axis_and_
# names_it`, `test_the_inline_seconds_table_never_puts_two_environments_in_one
# _row`, `test_the_inline_seconds_table_collapses_one_axis_and_not_two` and
# `test_the_written_record_still_carries_every_row` removed: all five
# exercised `tables.render_per_run`/`render_per_run_summary`, which read the
# `seconds`/`peakMiB` `perRun` dimensions. Both dimensions are retired from
# `config.DIMENSIONS` -- not renamed, removed -- and their two readers are
# retired from `tables.py` along with them. Nothing declared any more is
# `perRun`, so there is no inline-versus-written-record distinction left to
# hold.


# ---------------------------------- the phase-two readings, both rates at once

def _lectura(arm: str, transfer: str, seed: int, ratio: float) -> dict:
    """The one quantity these tests read, in the shape phase two writes."""
    return {"arm": arm, "transfer": transfer, "seed": seed, "median": True,
            "geometry": {"ratio": ratio}}


def _las_dos_tasas() -> tuple[list[dict], list[dict]]:
    limpias = [_lectura("B", LABELS[0], 0, 0.90), _lectura("B", LABELS[1], 0, 0.80),
               _lectura("G", LABELS[0], 0, 0.50), _lectura("G", LABELS[1], 0, 0.40)]
    sucias = [_lectura("B", LABELS[0], 0, 0.95), _lectura("B", LABELS[1], 0, 0.85),
              _lectura("G", LABELS[0], 0, 0.55), _lectura("G", LABELS[1], 0, 0.45)]
    return limpias, sucias


def test_the_clean_rows_come_first_as_a_block_and_the_contaminated_ones_after():
    """Two blocks, not one interleaved list.

    Every `sin` row, in arm order, and only then every `con` row in the same arm
    order. A reader comparing one arm across materials counts down a fixed number
    of rows and lands on the same method; a reader comparing two arms under the
    same material never has the other material's row between them.

    Reachable red: group by arm instead --- `sin`/`con` of `B` and then
    `sin`/`con` of `G` --- and the first column stops being two blocks.
    """
    limpias, sucias = _las_dos_tasas()
    filas = _filas(tables.render_readings(
        limpias, "geometry.ratio", "t", contaminated=sucias, rate=0.2,
        markdown=True))

    assert len(filas) == 4, filas
    assert [f.split("|")[1].strip() for f in filas] == [
        tables.NOISE_CLEAN, tables.NOISE_CLEAN,
        tables.NOISE_DIRTY, tables.NOISE_DIRTY]
    assert [f.split("|")[2].strip() for f in filas] == [
        f"`{config.NAME_OF['B']}`", f"`{config.NAME_OF['G']}`",
        f"`{config.NAME_OF['B']}`", f"`{config.NAME_OF['G']}`"]


def test_the_noise_is_the_first_column_and_the_rest_of_the_table_did_not_move():
    """One column was added, at the front. The six transfers and the average are
    where they were, because a reader who learned this table under two
    renderings must not have to learn it again."""
    limpias, sucias = _las_dos_tasas()
    encabezado = tables.render_readings(
        limpias, "geometry.ratio", "t", contaminated=sucias, rate=0.2,
        markdown=True).splitlines()[2]

    assert ([c.strip() for c in encabezado.split("|")[1:-1]]
            == [tables.NOISE_COLUMN, "Método", *LABELS, "Prom."])


def test_the_rate_is_stated_once_in_the_title_and_in_no_row():
    """`con` and `0.2` in the same row are one fact written twice.

    The rate is fixed for the whole campaign, so a column carrying it would
    repeat it once per row and the report's own contract counts a measurement
    rendered twice as a defect. It is stated in the table's title line ---
    computed from the `rate` the readings were loaded with, never typed --- and
    nowhere else.

    Reachable red: put the rate back in a column, or hard-code `0.2` into the
    title, and pass a different rate here.
    """
    limpias, sucias = _las_dos_tasas()
    rendered = tables.render_readings(
        limpias, "geometry.ratio", "t", contaminated=sucias, rate=0.35,
        markdown=True)

    assert rendered.count("0.35") == 1, rendered
    assert "0.35" in rendered.splitlines()[0], "the rate is not in the title line"
    assert all("0.35" not in f for f in _filas(rendered)), _filas(rendered)
    # And nothing says it when there is nothing contaminated to say it about.
    assert "ρ=" not in tables.render_readings(limpias, "geometry.ratio", "t",
                                              markdown=True)


def test_an_arm_without_a_reading_for_a_transfer_still_gets_an_empty_cell():
    """Not a zero: a zero is a measurement and this arm has none."""
    limpias, _ = _las_dos_tasas()
    fila = [l for l in tables.render_readings(limpias, "geometry.ratio", "t",
                                              markdown=True).splitlines()
            if config.NAME_OF["B"] in l][0]

    assert fila.split("|")[5].strip() == "—"


def test_without_a_contaminated_campaign_the_clean_rows_print_alone():
    """The notebook guards every call with `if readings_ruido else`, so this
    state is reachable and refusing here would take the table away from the run
    that has half the material rather than from the one that has none."""
    limpias, _ = _las_dos_tasas()
    filas = _filas(tables.render_readings(limpias, "geometry.ratio", "t",
                                          markdown=True))

    assert len(filas) == 2, filas
    assert all(f.split("|")[1].strip() == tables.NOISE_CLEAN for f in filas), filas


def test_contaminated_readings_with_no_rate_refuse_rather_than_printing_a_zero():
    """A `con` block whose title could not say which campaign it came from is
    the one failure this table exists to prevent, written by the table itself."""
    limpias, sucias = _las_dos_tasas()
    with pytest.raises(ValueError):
        tables.render_readings(limpias, "geometry.ratio", "t", contaminated=sucias)


# ------------------------------- the five tables that now carry both materials

def _campana_contaminada(tmp_path, monkeypatch, rate: float, *, runs=None,
                         grid=None, grid_per_run=None, stated=None) -> None:
    """Un árbol de resultados a `rate`, escrito como lo escribe una campaña.

    Hace falta de verdad y no se puede pasar en memoria: las cinco tablas piden
    el material contaminado **por su tasa**, que es como está guardado, y esa
    puerta ---`_level_or_note`--- es la que lleva la guarda que compara el
    `labelNoise` declarado contra el directorio donde apareció. Un fixture que
    entregara las corridas directamente saltearía justo lo que se quiere probar.
    """
    monkeypatch.setattr(config, "PRODUCT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    root = config.results_for(rate, "campaign")
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (runs or [])), encoding="utf-8")
    summary = {"reduction": {"labelNoise": rate if stated is None else stated},
               "grid": grid or {}}
    if grid_per_run is not None:
        summary["gridPerRun"] = grid_per_run
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _celda_de_grilla(values: dict[str, float], metric: str) -> dict:
    return {arm: {metric: {"mean": value}} for arm, value in values.items()}


def test_every_unified_table_prints_the_clean_block_and_then_the_contaminated(
        tmp_path, monkeypatch) -> None:
    """The four `_at` twins are gone and the clean renderers took the data.

    Each twin loaded the contaminated record and called its clean sibling again.
    It was declared apart for one reason only --- the duplication check looks at
    the call, and two calls to one renderer with two different records read as
    one measurement rendered twice --- so the resolution is the one
    `render_readings` already took: one call, no twin, nothing to misread.

    Asserted over both at once because the claim is about the class and not
    about any one member of it: the shape they share is `Ruido` first, every
    `sin` row, then every `con` row. `render` and `render_readings` are the two
    survivors of the family -- `render_rungs`/`render_gains`/`render_per_run*`
    are retired along with everything that read them.

    Reachable red: append the contaminated rows to the clean ones without the
    label, or interleave the two blocks by arm.
    """
    metric = "targetAccuracy"
    niveles = {arm: 0.50 + 0.05 * i for i, arm in enumerate(config.ARM_ORDER)}
    limpias = [{"arm": arm, "transfer": label, "seed": 0, metric: value,
                "contribution": 0.25}
               for arm, value in niveles.items() for label in LABELS]
    sucias = [dict(run, **{metric: run[metric] - 0.20}) for run in limpias]
    grid_sucia = {label: _celda_de_grilla(
        {arm: value - 0.20 for arm, value in niveles.items()}, metric)
        for label in LABELS}
    _campana_contaminada(tmp_path, monkeypatch, 0.2, runs=sucias, grid=grid_sucia)

    limpias_geom = [{"arm": arm, "transfer": label, "seed": 0, "median": True,
                     "geometry": {"ratio": value}}
                    for arm, value in niveles.items() for label in LABELS]
    sucias_geom = [dict(reading, geometry={"ratio": reading["geometry"]["ratio"] - 0.20})
                   for reading in limpias_geom]

    dibujadas = {
        "render": tables.render(limpias, metric, _reduction(seeds=1),
                                rate=0.2, markdown=True),
        "render_readings": tables.render_readings(
            limpias_geom, "geometry.ratio", "t", contaminated=sucias_geom,
            rate=0.2, markdown=True),
    }
    for name, rendered in dibujadas.items():
        etiquetas = [f.split("|")[1].strip() for f in _filas(rendered)]
        assert tables.NOISE_CLEAN in etiquetas and tables.NOISE_DIRTY in etiquetas, \
            f"{name} drew one block where the record has two"
        assert etiquetas == sorted(
            etiquetas, key=[tables.NOISE_CLEAN, tables.NOISE_DIRTY].index), \
            f"{name} interleaves the two materials instead of blocking them"
        assert [c.strip() for c in rendered.splitlines()
                if c.startswith("| ")][0].split("|")[1].strip() \
            == tables.NOISE_COLUMN, f"{name} does not lead with the noise column"

    # Y los números son los del registro contaminado, no una segunda copia de
    # los limpios: sin esto un `con` que volviera a dibujar el bloque `sin`
    # pasaría todo lo de arriba.
    primero = config.NAME_OF[config.ARM_ORDER[0]]
    filas_de_ese_brazo = [f for f in _filas(dibujadas["render"])
                          if f"`{primero}`" in f]
    assert len(filas_de_ese_brazo) == 2, filas_de_ese_brazo
    # `Prom.` es la anteúltima columna de datos: la última es `Puesto`.
    limpio, sucio = (float(f.split("|")[-3].strip().strip("*"))
                     for f in filas_de_ese_brazo)
    assert sucio == pytest.approx(limpio - 20.0), (limpio, sucio)

    filas_geom = [f for f in _filas(dibujadas["render_readings"])
                 if f"`{primero}`" in f]
    assert len(filas_geom) == 2, filas_geom
    limpio_g, sucio_g = (float(f.split("|")[-2].strip().strip("*"))
                        for f in filas_geom)
    assert sucio_g == pytest.approx(limpio_g - 0.20), (limpio_g, sucio_g)


def test_a_rate_that_never_ran_keeps_the_clean_block_and_says_why(
        tmp_path, monkeypatch) -> None:
    """Half the material is still material.

    The twins returned the note *instead of* a table, which was right when the
    note stood where a second table would have been. It is not right now: the
    `sin` rows belong to the run that has them, so the note goes under the table
    and not in place of it.

    Reachable red: return the note alone and the clean campaign loses its own
    numbers because the contaminated one has not run.
    """
    metric = "targetAccuracy"
    limpias = [{"arm": arm, "transfer": label, "seed": 0, metric: value,
                "contribution": 0.25}
               for arm, value in (("B", 0.80), ("G", 0.90)) for label in LABELS]
    monkeypatch.setattr(config, "PRODUCT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")

    rendered = tables.render(limpias, metric, _reduction(seeds=1), rate=0.2,
                             markdown=True)

    etiquetas = [f.split("|")[1].strip() for f in _filas(rendered)]
    assert etiquetas == [tables.NOISE_CLEAN, tables.NOISE_CLEAN], etiquetas
    assert "0.2" in rendered and "no existe" in rendered


def test_a_record_that_contradicts_its_own_directory_draws_no_contaminated_block(
        tmp_path, monkeypatch) -> None:
    """One guard, one place, five tables.

    Each twin carried its own copy of this refusal. Five copies are five texts
    that can drift, and a section drawing a block the one beside it refused
    would be the drift nobody could see.
    """
    metric = "targetAccuracy"
    limpias = [{"arm": "B", "transfer": LABELS[0], "seed": 0, metric: 0.8,
                "contribution": 0.25}]
    sucias = [{"arm": "B", "transfer": LABELS[0], "seed": 0, metric: 0.6,
               "contribution": 0.25}]
    _campana_contaminada(tmp_path, monkeypatch, 0.2, runs=sucias, stated=0.4)

    rendered = tables.render(limpias, metric, _reduction(seeds=1), rate=0.2,
                             markdown=True)

    assert not [f for f in _filas(rendered)
                if f.split("|")[1].strip() == tables.NOISE_DIRTY]
    assert "labelNoise=0.4" in rendered


def test_the_report_declares_no_renderer_that_no_longer_exists() -> None:
    """`verify` reads this declaration, and a name it cannot resolve is a
    contract that describes a document nobody can produce.

    Reachable red: delete a renderer and leave its name in `__benchmark__`, which
    is exactly what removing the four `_at` twins would have done.
    """
    declared = MIL_CREDA_Benchmark.__benchmark__["report"]
    for name in declared["renderers"] + declared["conclusions"]:
        module, attribute = name.split(".")
        assert module == "tables", name
        assert hasattr(tables, attribute), f"{name} is declared and does not exist"
    assert not [n for n in declared["renderers"] if n.endswith("_at")], \
        "a twin that loaded the contaminated record by rate is declared again"


def test_no_section_of_the_report_draws_one_quantity_twice() -> None:
    """The document half of the same claim.

    Each of the seven clean/contaminated sections used to spend a second framing
    cell and a second table cell on the same quantity. One table now carries both
    materials, so no renderer is called twice with the same reading in the
    notebook's shown cells.

    Reachable red: put `render(runs, "targetAccuracy", ...)` back into a second
    cell for the contaminated campaign.
    """
    shown = [(name, _metric_of(args)) for name, args in _shown(REPORT)
             if name.startswith("render")]
    repetidas = [pair for pair in set(shown)
                 if pair[1] is not None and shown.count(pair) > 1]
    assert not repetidas, f"one quantity is rendered twice: {repetidas}"
    assert not [name for name, _ in shown if name.endswith("_at")], \
        "the notebook still asks for a contaminated twin"


# ------------------------------------------------------ defect (e): attentionSpread

def _independent_min_normalized_entropy(m: int, spread: float) -> float:
    """The same claim, computed a different way than `tables.py`'s own search.

    `tables.py` sweeps a count `k` and builds the two-tier weights from
    `hi`/`lo` ratios; this instead builds the actual logit vector for every
    split, applies a plain `math.exp`/normalize softmax and Shannon entropy
    from first principles, and never imports `tables`. A shared mistake in
    the closed-form ratio algebra would pass both if they were the same
    computation wearing two names; this is not.
    """
    best = 1.0
    for k in range(1, m):
        logits = [spread / 2.0] * k + [-spread / 2.0] * (m - k)
        exps = [math.exp(x) for x in logits]
        total = sum(exps)
        weights = [e / total for e in exps]
        entropy = -sum(w * math.log(w) for w in weights)
        best = min(best, entropy / math.log(m))
    return best


def test_the_attention_spread_floor_is_computed_not_asserted_at_zero() -> None:
    """Defect (e): the floor `attentionSpread` can reach under today's neutral
    hyperparameters is far from zero, and the report text has to say so with a
    number this suite actually computed -- never a hand-typed one.

    Reachable red: hardcode `MIN_ATTENTION_SPREAD = 0.0`, or compute it with
    the wrong logit spread (e.g. `1 + gamma` instead of `2 + gamma`).
    """
    expected = _independent_min_normalized_entropy(
        config.INSTANCES_PER_BAG,
        2.0 + config.ATTENTION_GAMMA)
    assert tables.MIN_ATTENTION_SPREAD == pytest.approx(expected, abs=1e-9)
    # Far from the unreachable floor a prior version of this text implied.
    assert tables.MIN_ATTENTION_SPREAD > 0.5

    # The defect this fixes, stated as a fact about the numbers rather than
    # about the old prose: the previous "concentrated" branch fired at
    # `mean <= 1.0 - UNIFORM_ATTENTION`, and that threshold sits BELOW the
    # true floor -- no real measurement could ever have reached it.
    assert (1.0 - tables.UNIFORM_ATTENTION) < tables.MIN_ATTENTION_SPREAD


def test_the_floor_is_computed_per_arm_never_one_number_for_all() -> None:
    """A selecting arm reparte over its own declared budget, not over the full
    bag, so its floor would be a DIFFERENT, smaller number -- never
    `MIN_ATTENTION_SPREAD`, which is the full-bag floor every other attending
    arm shares.

    No arm id is written here: which arms select is read from
    `config.ARMS`'s own `selection` field, never assumed to be `SU`/`SA`/`SK`
    -- today's declaration selects none, which this test accepts as a valid,
    measured state rather than skipping over it, and every attending arm's
    floor is checked against the shared, full-bag one.

    Reachable red: key a selecting arm to `MIN_ATTENTION_SPREAD` (today's
    single-number defect) once the declaration adds one back, or compute a
    selecting arm's floor over `INSTANCES_PER_BAG` instead of its own budget.
    """
    attending = {arm["id"] for arm in config.ARMS if arm["attention"] == "learned"}
    assert attending == set(tables.MIN_ATTENTION_SPREAD_BY_ARM)

    selecting = {arm["id"] for arm in config.ARMS if arm.get("selection") is not None}
    full_bag = attending - selecting
    assert full_bag, "no non-selecting attending arm to compare against"

    for arm in full_bag:
        assert tables.MIN_ATTENTION_SPREAD_BY_ARM[arm] == pytest.approx(
            tables.MIN_ATTENTION_SPREAD, abs=1e-12)

    for arm in selecting:
        budget = config.ARMS_BY_ID[arm].get("budget")
        assert budget, f"{arm} selects and declares no budget to select over"
        expected = _independent_min_normalized_entropy(
            int(budget), 2.0 + config.ATTENTION_GAMMA)
        assert tables.MIN_ATTENTION_SPREAD_BY_ARM[arm] == pytest.approx(
            expected, abs=1e-9)
        # A selecting arm's floor is strictly LOWER than the full-bag floor: a
        # smaller bag reaches a narrower minimum entropy MORE easily, not
        # less. What is checked is the direction, not merely that they differ.
        assert tables.MIN_ATTENTION_SPREAD_BY_ARM[arm] < tables.MIN_ATTENTION_SPREAD - 1e-6


def _optimized_min_normalized_entropy(m: int, gamma: float, tau_att: float,
                                       seed: int) -> float:
    """The claimed floor, reached with a different algorithm than `tables.py`'s
    own closed-form vertex sweep: projected gradient descent that MINIMIZES
    the normalized entropy of a softmax over `m` logits, each clamped
    (projected) to r21's own stated bound `[-half_spread, half_spread]` after
    every step, from several random restarts.

    Never the same vertex algorithm `_min_reachable_attention_entropy` uses --
    a shared mistake in that closed form would pass both if they were the
    same computation wearing two names. And unlike a purely random sample of
    logit vectors, an optimizer that actually searches converges close to the
    true minimum, so it can tell a correct floor from one mutated too low:
    a floor set below the true minimum would read as `too far above the
    optimizer's own best`, which the second assertion below checks for.
    """
    import torch

    half_spread = (2.0 + gamma) / 2.0
    generator = torch.Generator().manual_seed(seed)
    best = 1.0
    # The landscape is not convex -- it has one local basin per possible split
    # `k` of instances between the logit's two bounds, and a handful of
    # restarts settles into whichever basin the random init happened to be
    # closest to rather than the global one. Enough restarts is what finds
    # the true minimum: measured, 40 restarts land within 1e-6 of the closed
    # form's answer for both bag sizes this suite checks; 6 restarts landed
    # a full 0.007 short of it.
    for _ in range(40):
        logits = (torch.rand(m, generator=generator) * 2 - 1) * half_spread
        logits.requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=0.2)
        for _ in range(300):
            optimizer.zero_grad()
            weights = torch.softmax(logits / tau_att, dim=0)
            entropy = -(weights * torch.log(weights + 1e-12)).sum()
            entropy.backward()
            optimizer.step()
            with torch.no_grad():
                logits.clamp_(-half_spread, half_spread)
        with torch.no_grad():
            weights = torch.softmax(logits / tau_att, dim=0)
            entropy = -(weights * torch.log(weights + 1e-12)).sum()
            best = min(best, float(entropy) / math.log(m))
    return best


def test_no_configuration_of_the_sweep_reaches_below_the_computed_floor() -> None:
    """An independent optimizer never beats the claimed minimum, for every
    bag size a declared arm actually attends over -- and it gets close
    enough to it that a floor mutated too low would be caught here: a purely
    random sample of logits, which the previous version of this test drew,
    can only ever prove the ">=" direction and passes just as well when the
    claimed floor is wrongly low, because "the sample never went below an
    artificially low floor" is true of every floor at least that low. Only a
    search that actually MINIMIZES catches that.
    """
    sizes = {tables._bag_size_for(arm) for arm in tables.MIN_ATTENTION_SPREAD_BY_ARM}
    assert sizes, "no attending arm declared, nothing to check the floor of"
    assert config.INSTANCES_PER_BAG in sizes

    for m in sizes:
        claimed = tables._min_reachable_attention_entropy(
            m, config.ATTENTION_GAMMA, config.ATTENTION_TEMPERATURE)
        found = _optimized_min_normalized_entropy(
            m, config.ATTENTION_GAMMA, config.ATTENTION_TEMPERATURE, seed=20260916)
        assert found >= claimed - 1e-6, (
            f"m={m}: the optimizer reached {found}, below the claimed floor "
            f"{claimed}")
        # The sweep is not vacuous: an optimizer that actually minimizes
        # lands close to the claimed floor. A floor set too low (e.g.
        # hardcoded at 0.0, or computed at the wrong m) would fail here even
        # though it trivially satisfies the ">=" check above.
        assert found <= claimed + 5e-3, (
            f"m={m}: the claimed floor {claimed} is far from the true "
            f"minimum {found} the optimizer actually found")


def test_attention_spread_text_names_the_computed_floor_and_the_r21_reading() -> None:
    """The objective and the conclusion both name r21 l.501's reading rather
    than calling a uniform weight a failure, and both quote the SAME computed
    number -- never a second, independently hand-typed one.

    Reachable red: revert either string to "buscamos un valor lejos de
    1.000" / "que venía a mejorar", or quote a different decimal in one of
    the two.
    """
    objective_text = tables.objective("attentionSpread")
    floor_str = f"{tables.MIN_ATTENTION_SPREAD:.3f}"
    assert floor_str in objective_text
    assert "dejó de elegir" not in objective_text
    assert "no es, por sí solo, un fallo" in objective_text
    assert "l.501" in objective_text
    # The r21 l.501 pairing is with the bag's own self-similarity, never with
    # the correspondence mass -- that was the defect: r21 pairs a near-uniform
    # weight with a LOW self-similarity (a diverse bag with no dominant
    # group), which has nothing to do with how much local-correspondence mass
    # landed on the true class.
    assert "autosimilitud" in objective_text
    assert "masa de correspondencia" not in objective_text

    said = tables.conclusion_attention([
        {"arm": "G", "transfer": "M->U", "seed": 0, "attentionSpread": 0.9},
    ])
    assert floor_str in said
    assert "l.501" in said
    assert "que venía a mejorar" not in said


def test_attention_spread_text_prints_the_per_arm_floor_not_only_the_global_one() -> None:
    """Defect (7): `objective` and `conclusion_attention` print the PER-ARM
    floor for a selecting arm -- its own, over its declared budget of
    instances -- beside the m=`INSTANCES_PER_BAG` floor a full-bag arm reads,
    never the global floor stamped onto a selecting arm's own row.

    No arm id is written here: the selecting arm, if any, is read from
    `config.ARMS`. Today's declaration selects none, so this reachable-red
    coverage is skipped rather than faked against an arm that does not exist
    -- a skip says plainly that the state is unmeasured, where silently
    reusing a full-bag arm would print a green result for a claim nothing here
    checked.

    Reachable red: read `MIN_ATTENTION_SPREAD` (the full-bag floor) for a
    selecting arm's row instead of its own `MIN_ATTENTION_SPREAD_BY_ARM`
    entry -- in either function.
    """
    selecting = [arm["id"] for arm in config.ARMS if arm.get("selection") is not None]
    if not selecting:
        pytest.skip("no selecting arm is declared: nothing to isolate a "
                    "per-arm floor against")
    arm_id = selecting[0]
    selecting_floor = tables.MIN_ATTENTION_SPREAD_BY_ARM[arm_id]
    full_bag_floor = tables.MIN_ATTENTION_SPREAD
    budget = config.ARMS_BY_ID[arm_id]["budget"]
    assert selecting_floor != pytest.approx(full_bag_floor, abs=1e-6), (
        "fixture assumption broken: the two floors coincide under this config"
    )

    objective_text = tables.objective("attentionSpread")
    assert f"{selecting_floor:.3f}" in objective_text
    assert f"{full_bag_floor:.3f}" in objective_text
    assert str(budget) in objective_text
    assert str(config.INSTANCES_PER_BAG) in objective_text

    # A selecting arm's own row names its own budget-sized floor.
    said_selecting = tables.conclusion_attention([
        {"arm": arm_id, "transfer": "M->U", "seed": 0, "attentionSpread": 0.9},
    ])
    assert f"piso {selecting_floor:.3f}" in said_selecting
    assert f"sobre {budget} instancias" in said_selecting
    assert f"piso {full_bag_floor:.3f}" not in said_selecting

    # A full-bag arm's own row names the full-bag floor instead.
    full_bag_arm = next(arm["id"] for arm in config.ARMS
                        if arm["attention"] == "learned" and arm["id"] != arm_id)
    said_full_bag = tables.conclusion_attention([
        {"arm": full_bag_arm, "transfer": "M->U", "seed": 0, "attentionSpread": 0.9},
    ])
    assert f"piso {full_bag_floor:.3f}" in said_full_bag
    assert f"sobre {config.INSTANCES_PER_BAG} instancias" in said_full_bag
    assert f"piso {selecting_floor:.3f}" not in said_full_bag


@pytest.mark.parametrize(
    "arm", [arm["id"] for arm in config.ARMS if arm["attention"] == "learned"])
def test_conclusion_attention_prints_this_exact_arms_own_floor(arm: str) -> None:
    """A mutation that special-cased one arm to read the global floor
    (`MIN_ATTENTION_SPREAD_BY_ARM.get(arm, MIN_ATTENTION_SPREAD)` rewritten
    as, say, `... if arm != "F" else MIN_ATTENTION_SPREAD`) would pass a
    test that only ever built a row for one or two arms. Driven over every
    attending arm, one row at a time -- read from `config.ARMS` and never
    written here -- so a mutation confined to any single arm's own lookup is
    caught by that arm's own row.

    Reachable red: any one arm's floor lookup replaced by the global
    `MIN_ATTENTION_SPREAD` while every other arm's stays correct.
    """
    floor = tables.MIN_ATTENTION_SPREAD_BY_ARM[arm]
    said = tables.conclusion_attention([
        {"arm": arm, "transfer": "M->U", "seed": 0, "attentionSpread": 0.9},
    ])
    assert f"piso {floor:.3f}" in said, (
        f"arm {arm}: expected its own floor {floor:.3f} in {said!r}"
    )


# `test_diagnostic_record_refuses_a_stamp_mismatch` and
# `test_diagnostic_record_accepts_a_correctly_stamped_measurement` removed:
# `tables._diagnostic_record`, `render_diagnostic`, `diagnostic_source_note`
# and `conclusion_diagnostic` are retired along with the noise-diagnostic
# apparatus. None of the six required sections of `Results_v1.ipynb` reads a
# re-searched-ceiling-under-noise diagnostic any more.
