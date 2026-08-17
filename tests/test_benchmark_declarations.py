"""The joins inside the benchmark package, checked from both ends.

Two files describe the arms and neither reads the other. `config.ARMS` says what
each arm computes; `MIL_CREDA_Benchmark.__benchmark__` says which sections of the
revision each arm exercises, which is what lets the drift report answer *does this
change oblige the bench to change*. Adding an arm to one and forgetting the other
leaves both files internally consistent and the pair wrong — the kind of omission
that arrives as a silence.

Everything here is config-level and imports no torch, so the suite still runs with
no network and no model weights.
"""

from __future__ import annotations

import MIL_CREDA_Benchmark
import json

import pytest
import torch

from MIL_CREDA_Benchmark import config


def test_every_arm_declares_which_sections_it_exercises() -> None:
    declared = set(MIL_CREDA_Benchmark.__benchmark__["arms"])
    configured = {arm["id"] for arm in config.ARMS}
    assert configured - declared == set(), (
        f"arms with no section declaration: {sorted(configured - declared)}")
    assert declared - configured == set(), (
        f"sections declared for arms that no longer exist: {sorted(declared - configured)}")


def test_the_benchmark_is_bound_to_the_same_revision_as_the_configuration() -> None:
    assert MIL_CREDA_Benchmark.__benchmark__["revision"] == config.REVISION


def test_the_benchmark_declares_what_its_protocol_assumes() -> None:
    """A change of reach leaves every arm intact and every dimension meaningless,
    so the premises sit beside the arms rather than in somebody's memory."""
    premises = MIL_CREDA_Benchmark.__benchmark__["premises"]
    for key in ("prediction", "unit", "metric", "direction"):
        assert premises.get(key), f"the protocol does not declare its {key}"


def test_every_rung_names_arms_that_exist() -> None:
    for left, right, reading in config.LADDER:
        assert left in config.ARMS_BY_ID, f"rung {left}->{right} names unknown {left}"
        assert right in config.ARMS_BY_ID, f"rung {left}->{right} names unknown {right}"
        assert reading.strip(), f"rung {left}->{right} says nothing about what it reads"


def test_every_arm_can_be_kept_and_every_kept_arm_exists() -> None:
    assert set(config.CHECKPOINTS) == set(config.ARM_ORDER)
    assert all(count >= 1 for count in config.CHECKPOINTS.values())


def test_each_arm_has_its_own_display_name() -> None:
    """Two arms sharing a name would collapse into one row of every table."""
    names = [arm["name"] for arm in config.ARMS]
    assert len(set(names)) == len(names), f"repeated display names in {names}"


def test_every_figure_names_arms_that_exist() -> None:
    for arm in [*config.LATENT_PANELS, *config.BAG_PANELS, *config.FLOOR_OF,
                *config.FLOOR_OF.values()]:
        assert arm in config.ARMS_BY_ID, f"a figure or floor names unknown arm {arm!r}"


def test_the_figures_pick_transfers_the_campaign_actually_ran() -> None:
    """The three shown are computed from the campaign, so the only way they can go
    wrong is by naming a transfer that produced no runs — which would draw from
    checkpoints that do not exist and come out silently empty."""
    from MIL_CREDA_Benchmark import tables

    ran = [f"{s}->{t}" for s, t in config.TRANSFERS]
    runs = [{"arm": "A", "transfer": label, "targetAccuracy": index / 10,
             "contribution": 0.0} for index, label in enumerate(ran)]
    chosen = tables.best_transfers(runs)
    assert len(chosen) == config.FIGURE_TRANSFER_COUNT
    assert all(label in ran for label in chosen), f"{chosen} was never run"


def test_the_figure_transfer_rule_ranks_by_the_outcome_it_declares() -> None:
    """The choice is made by the result, which is why it is declared in every
    caption. What has to hold is that it ranks the way it says it does — a rule
    that claimed one order and produced another would be worse than no rule."""
    from MIL_CREDA_Benchmark import tables

    ran = [f"{s}->{t}" for s, t in config.TRANSFERS]
    ascending = [{"arm": "A", "transfer": label, "targetAccuracy": index / 10,
                  "contribution": 0.0} for index, label in enumerate(ran)]
    assert tables.best_transfers(ascending, count=2) == [ran[-1], ran[-2]]
    assert config.FIGURE_TRANSFER_RULE.strip(), "the rule is not declared anywhere"


def test_both_floors_stay_in_the_grid_until_a_measurement_removes_one() -> None:
    """Whether the two floors are redundant is a measurement, not an assumption:
    they train the same encoder through different objectives, so their instance
    embeddings have no reason to agree. `latent.floors_agree` is what may retire
    one of these columns, and nothing else."""
    floors = [arm["id"] for arm in config.ARMS if arm["adaptation"] is None]
    assert set(floors).issubset(set(config.LATENT_PANELS)), (
        f"a floor was dropped from the grid without a measurement: "
        f"{sorted(set(floors) - set(config.LATENT_PANELS))}")


def test_the_bag_figure_shows_the_rung_the_local_term_lives_on() -> None:
    """Its panels are chosen by the mechanism, not by the ranking: a floor, an arm
    without the local term, and one with it. Without the middle one the figure
    cannot come out wrong."""
    specs = [config.ARMS_BY_ID[arm] for arm in config.BAG_PANELS]
    assert any(spec["local"] for spec in specs), "no panel carries the local term"
    assert any(not spec["local"] and spec["adaptation"] for spec in specs), (
        "no panel adapts without the local term, so the figure cannot isolate it")
    assert any(spec["adaptation"] is None for spec in specs), "no floor to read against"


def test_the_selecting_arms_hold_one_budget_and_differ_only_in_the_rule() -> None:
    """The rung between two of them is attributable only if this holds."""
    selecting = [arm for arm in config.ARMS if arm["selection"] is not None]
    assert len(selecting) >= 2, "a selection rung needs at least two selecting arms"
    assert len({arm["selection"] for arm in selecting}) == len(selecting), (
        "two selecting arms share a rule, so the rung between them differs in nothing")
    for arm in selecting:
        for axis in ("unit", "adaptation", "weighting", "local", "attention"):
            assert arm[axis] == config.ARMS_BY_ID["G"][axis], (
                f"{arm['id']} differs from the complete method in {axis} as well as "
                f"in its selection, so its rung is not attributable")


def test_the_budget_is_smaller_than_the_bag() -> None:
    """A selection that keeps everything is not a selection, and its rung against
    the arm that keeps everything would compare an arm with itself."""
    assert 0 < config.SELECT_K < config.INSTANCES_PER_BAG


def test_the_pilot_and_the_full_run_are_the_same_program() -> None:
    """Only the repetition count and the length may separate them."""
    assert config.EPOCHS <= config.FULL_EPOCHS
    assert len(config.SEEDS) <= len(config.FULL_SEEDS)
    assert set(config.SEEDS).issubset(set(config.FULL_SEEDS))


def _one_rung_grid(left_mean: float, right_mean: float) -> dict:
    """A grid holding one rung, both arms flat across every transfer."""
    left, right, _ = config.LADDER[0]

    def arm(mean: float) -> dict:
        entry = {"mean": mean, "stdev": 0.0, "max": mean, "n": 1}
        # Both metrics, because the panorama walks both and a cell missing one is
        # a fixture defect that would read as a defect in the code under test.
        return {"targetAccuracy": dict(entry), "sourceAccuracy": dict(entry)}

    return {f"{s}->{d}": {left: arm(left_mean), right: arm(right_mean)}
            for s, d in config.TRANSFERS}


def test_a_rung_subtracts_left_minus_right_everywhere_it_is_computed() -> None:
    """The sign is a reading convention, and a convention only holds if every
    place that computes it agrees. It is computed three times — the rung table,
    its conclusion, and the panorama that outlives both in the record — and the
    prose in the notebook is written against it. Nothing else pins it, so an
    inversion here comes back as a heading that says the opposite of its own
    table while every number stays correct.
    """
    from MIL_CREDA_Benchmark import harness, tables

    left, right, _ = config.LADDER[0]
    grid = _one_rung_grid(left_mean=0.40, right_mean=0.70)
    summary = {"grid": grid, "reduction": {"seeds": config.SEEDS},
               "panorama": harness.paired_across_transfers(grid)}

    # The right arm is 30 points above, so left - right is negative.
    printed = tables.render_rungs(summary, "targetAccuracy", markdown=True)
    assert "-30.0" in printed, printed

    panorama = [row for row in summary["panorama"]
                if row["rung"] == f"{left}->{right}"
                and row["metric"] == "targetAccuracy"][0]
    assert panorama["meanDifference"] < 0
    # The field counts transfers won by the right arm, which is now the negative
    # side. Flipping the subtraction without flipping this would report the right
    # arm losing all six of the transfers it won.
    assert panorama["favouringRight"] == len(config.TRANSFERS)


def test_the_rung_conclusion_names_who_is_ahead_not_how_far_it_moved() -> None:
    """A conclusion that reports a signed magnitude makes the reader reconstruct
    the direction against the row's own name. Naming the arm is the whole job."""
    from MIL_CREDA_Benchmark import tables

    left, right, _ = config.LADDER[0]
    summary = {"grid": _one_rung_grid(left_mean=0.40, right_mean=0.70),
               "reduction": {"seeds": config.SEEDS}}
    text = tables.conclusion_rungs(summary, "targetAccuracy")
    assert config.NAME_OF[right] in text, text
    assert "por encima de" in text, text
    assert f"**{config.NAME_OF[right]}**" in text, text


# --------------------------------------------------------------- the three roles

def test_the_three_roles_partition_the_bags_exactly() -> None:
    """Nothing is lost and nothing is counted twice.

    The selection role was funded with new material, not taken from anywhere. If
    somebody moves one of the three constants without moving the bags per class,
    the remainder lands silently in evaluation or the split blows up midway
    through a campaign. Reachable red: raise `VALID_BAGS` without raising
    `BAGS_PER_CLASS` and this fails.
    """
    assert config.TRAIN_BAGS + config.VALID_BAGS + config.EVAL_BAGS == \
        config.BAGS_PER_DOMAIN


def test_the_search_never_reads_the_role_the_verdict_rests_on() -> None:
    """The selection role exists because the search chooses by looking at outcomes."""
    assert config.SEARCH_ROLE == "valid"
    assert config.SEARCH_ROLE != "eval"


def test_every_role_covers_every_class_in_every_domain() -> None:
    """The local correspondence is undefined for a class with no source bag.

    A role that dropped one would produce a failure that reads as a defect of the
    method and is a defect of the split.
    """
    base, remainder = divmod(config.TRAIN_BAGS, config.CLASSES)
    assert base >= 1, "el rol de entrenamiento dejaría clases sin bolsa"
    base, remainder = divmod(config.VALID_BAGS, config.CLASSES)
    assert base >= 1, "el rol de selección dejaría clases sin bolsa"
    per_class_used = (config.TRAIN_BAGS + config.VALID_BAGS) // config.CLASSES
    assert config.BAGS_PER_CLASS - per_class_used >= 1, \
        "el rol de evaluación dejaría clases sin bolsa"


def test_the_search_grid_runs_between_the_two_declared_defaults() -> None:
    """The range is not arbitrary: its endpoints are the two declared defaults."""
    from CREDA.schedules import CREDA_CEILING
    from MIL_CREDA_Benchmark.schedules import MILCREDA_CEILING

    assert min(config.CEILING_GRID) == CREDA_CEILING
    assert max(config.CEILING_GRID) == MILCREDA_CEILING


def test_the_search_uses_the_complete_method_of_each_family() -> None:
    """Searching with an ablation would pick the ceiling of a method nobody compares."""
    for family, arm_id in config.SEARCH_ARMS.items():
        assert arm_id in config.ARMS_BY_ID
        assert config.ARMS_BY_ID[arm_id]["adaptation"] == family


# ---------------------------------------------------- the ceiling selection

def _rows(scores: dict[float, list[float]]) -> list[dict]:
    """Rows as the search builds them, from (seed, transfer) cells."""
    grid = sorted(scores)
    cells = [{c: scores[c][i] for c in grid} for i in range(len(scores[grid[0]]))]
    out = []
    for c in grid:
        centred = [cell[c] - sum(cell.values()) / len(cell) for cell in cells]
        out.append({"ceiling": c, "paired": sum(centred) / len(centred)})
    return out


def _pick(rows: list[dict]) -> float:
    top = max(r["paired"] for r in rows)
    tied = [r for r in rows if r["paired"] == top]
    return min(tied, key=lambda r: r["ceiling"])["ceiling"]


def test_the_pairing_survives_a_cell_that_is_simply_harder() -> None:
    """What pairing buys, in the case that motivates it.

    Two cells, one far harder than the other, and ceiling 0.1 wins both by two
    points. Comparing bare means lets the hard cell's difficulty dominate and
    drown the effect; centring each cell on its own mean lets the effect survive,
    because every ceiling was measured on the same material.

    Reachable red: picking by bare mean, 0.01 wins.
    """
    scores = {0.01: [0.90, 0.20], 0.1: [0.92, 0.22]}
    crudo = {c: sum(v) / len(v) for c, v in scores.items()}
    assert _pick(_rows(scores)) == 0.1
    # the bare mean happens to agree here too; what changes is the margin
    assert crudo[0.1] > crudo[0.01]

    # now a hard, noisy cell, where the bare mean loses its way
    scores = {0.01: [0.90, 0.05], 0.1: [0.92, 0.01]}
    crudo = {c: sum(v) / len(v) for c, v in scores.items()}
    assert crudo[0.01] > crudo[0.1], "the bare mean picks the wrong ceiling"
    assert _pick(_rows(scores)) == 0.01, "y el pareado coincide: aca no hay efecto"


def test_a_tie_goes_to_the_smallest_ceiling() -> None:
    """Below some point a term is inert and everything ties; on that stretch the
    tie-break is what actually chooses, so it is declared."""
    scores = {1e-4: [0.5, 0.5], 1e-3: [0.5, 0.5], 1e-2: [0.5, 0.5]}
    assert _pick(_rows(scores)) == 1e-4


def test_the_required_search_scale_is_declared_apart_from_the_running_one() -> None:
    """Without both, a ceiling found at pilot scale reads as finished."""
    assert config.FULL_SEARCH_EPOCHS == config.FULL_EPOCHS
    assert config.FULL_SEARCH_SEEDS == 3
    assert config.SEARCH_EPOCHS >= config.FULL_SEARCH_EPOCHS
    assert len(config.SEARCH_SEEDS) >= config.FULL_SEARCH_SEEDS


def test_the_campaign_refuses_ceilings_searched_below_scale(tmp_path, monkeypatch) -> None:
    """The quiet failure: somebody lowers the search to test it cheaply, the file
    is written from three epochs, and every later campaign consumes it."""
    import json as _json
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(_json.dumps({
        "creda": {"ceiling": 1e-4, "atRequiredScale": False},
        "milcreda": {"ceiling": 1.0, "atRequiredScale": True},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)

    reduction = harness.Reduction(ceilings={"creda": 1e-4, "milcreda": 1.0})
    with pytest.raises(SystemExit) as raised:
        harness.campaign(reduction, torch.device("cpu"), arms=["A"],
                         progress=lambda *a: None)
    assert "below scale" in str(raised.value)
    assert "creda" in str(raised.value)


# ------------------------------ the notebook actually obtains what it needs

def _phase_one() -> list[dict]:
    """The campaign notebook's code cells, in order."""
    path = (config.REPOSITORY / "MIL-CREDA" / "Notebooks"
            / "Benchmark_Phase1.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_the_notebook_obtains_the_ceilings_before_it_runs_the_campaign() -> None:
    """The join, checked from the notebook's end and not only from the harness'.

    `campaign` refuses without ceilings and `ceilings_in_force` supplies them, and
    both were green while the only caller that matters never called one from the
    other: the notebook built its `Reduction` with the empty mapping `config`
    imports and died on the refusal. Testing each half against a fixture written
    here verifies both halves and never the connection, which is the only thing
    this rule was ever about.
    """
    sources = ["".join(cell["source"]) for cell in _phase_one()]
    obtains = [i for i, s in enumerate(sources) if "ceilings_in_force" in s]
    runs = [i for i, s in enumerate(sources) if "harness.campaign(" in s]
    assert obtains, "no cell obtains the ceilings; the campaign will refuse"
    assert runs, "no cell runs the campaign"
    assert min(obtains) < min(runs), (
        f"the ceilings are obtained at cell {min(obtains)}, after the campaign at "
        f"{min(runs)}")


def test_the_notebook_forecasts_the_search_before_spending_it() -> None:
    """The search is the longest single wait and the one part with no pilot scale.

    A notebook forecasting only the grid puts it ahead of the estimate that exists
    to precede it.
    """
    sources = ["".join(cell["source"]) for cell in _phase_one()]
    forecasts = [i for i, s in enumerate(sources) if '["search"]' in s]
    obtains = [i for i, s in enumerate(sources) if "ceilings_in_force" in s]
    assert forecasts, "nothing forecasts what the ceiling search costs"
    assert min(forecasts) < min(obtains), (
        "the search is forecast after it is spent, which is not a forecast")


def test_the_ceilings_reach_the_written_report_and_not_only_the_screen() -> None:
    """The agreement is that the *report* says which ceiling each family found.

    A cell that prints it to the console satisfies a reader watching the run and
    nobody afterwards, and the record is what a later session reads.
    """
    written = [s for s in ("".join(c["source"]) for c in _phase_one())
               if "report.md" in s]
    assert written, "no cell writes the readable report"
    assert any("render_ceilings" in s and "conclusion_ceilings" in s
               for s in written), (
        "the written report carries no ceiling section")


def _searched(**overrides) -> dict:
    entry = {"arm": "G", "ceiling": 1e-4, "criterion": "targetAccuracy",
             "grid": [{"ceiling": 1e-4, "targetAccuracy": 0.8},
                      {"ceiling": 1.0, "targetAccuracy": 0.6}],
             "tied": [1e-4], "decidedByTieBreak": False, "seedsAgree": True,
             "role": "valid", "epochs": 20, "seeds": [0, 1, 2],
             "atRequiredScale": True, "neutral": 1.0}
    return {"milcreda": {**entry, **overrides}}


def test_the_ceiling_conclusion_separates_a_measurement_from_a_tie_break() -> None:
    """A ceiling chosen between four identical scores and one chosen by a real
    difference are the same number, so the number cannot be the conclusion."""
    from MIL_CREDA_Benchmark import tables

    measured = tables.conclusion_ceilings(_searched())
    tied = tables.conclusion_ceilings(
        _searched(decidedByTieBreak=True, tied=[1e-4, 1e-3, 1e-2, 1.0]))
    assert measured != tied, "the conclusion cannot come out different"
    assert "desempate" in tied and "desempate" not in measured


def test_the_ceiling_conclusion_says_what_the_record_does_not_say() -> None:
    """A record written by an earlier search carries no tie-break and no agreement.

    Defaulting to `False` would assert *chosen by a real difference* about a record
    that never said so — a reason invented to cover what was not found, which reads
    as a finding and gets acted on like one.
    """
    from MIL_CREDA_Benchmark import tables

    older = {"milcreda": {"ceiling": 1e-4, "grid": [], "criterion": "targetAccuracy"}}
    text = tables.conclusion_ceilings(older)
    assert "sin que el registro diga cómo se desempató" in text
    assert "el registro no dice si las semillas coincidieron" in text
    assert "por una diferencia en el criterio" not in text


def test_the_campaign_carries_the_whole_search_record_not_only_the_winner() -> None:
    """A ceiling chosen between four identical scores and one chosen by a real
    difference are the same number and not the same evidence.

    And filling the field at the caller's hand is how it ends up filled on the run
    somebody was paying attention to and empty on the next.
    """
    import inspect
    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.campaign)
    assert "ceilingSearch=searched" in source, (
        "the campaign does not copy the search record into its reduction")


def test_an_existing_ceiling_record_is_never_re_searched(tmp_path, monkeypatch) -> None:
    """Overwriting an answer because a later caller wanted a different one is the
    silent refunding the campaign's refusal exists to prevent."""
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "creda": {"ceiling": 0.5, "atRequiredScale": True},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)

    def refuse(*args, **kwargs):
        raise AssertionError("re-searched over an existing record")

    monkeypatch.setattr(harness, "search_ceilings", refuse)
    found = harness.ceilings_in_force(harness.Reduction(), torch.device("cpu"),
                                      progress=lambda *a: None)
    assert found == {"creda": 0.5}


def test_the_ceilings_are_read_back_from_disk_and_not_from_the_import(
        tmp_path, monkeypatch) -> None:
    """`config.CEILINGS` is filled once at import, from a file that may not exist.

    A caller that searches in the same process — a notebook, the only place that
    ever does — would otherwise hold the empty mapping it imported and hand it to
    a campaign that refuses it, with the answer sitting on disk beside it.
    """
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    monkeypatch.setattr(config, "CEILINGS", {})

    def write_it(*args, **kwargs):
        record.write_text(json.dumps({
            "milcreda": {"ceiling": 0.25, "atRequiredScale": True},
        }), encoding="utf-8")
        return {}

    monkeypatch.setattr(harness, "search_ceilings", write_it)
    found = harness.ceilings_in_force(harness.Reduction(), torch.device("cpu"),
                                      progress=lambda *a: None)
    assert found == {"milcreda": 0.25}, (
        f"read from the stale import instead of the record: {found}")


# ------------------------------------------- resuming a search cut in half

def test_a_measured_cell_survives_the_run_being_cut(tmp_path, monkeypatch) -> None:
    """Cutting a grid of hours on its last cell used to lose every one before it.

    Measured the hard way: 1h37 for nothing. Every cell is written as soon as it
    is measured, so an interruption costs one cell and not the run.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    cells = {(0, "M->U"): {1e-4: 0.51, 1.0: 0.62},
             (1, "S->M"): {1e-4: 0.40, 1.0: 0.44}}
    harness._write_partial("milcreda", "G", cells, 3.7, lambda *a: None)

    assert harness._partial_path().exists()
    assert harness._read_partial()["milcreda"] == cells


def test_the_partial_is_keyed_by_seed_and_transfer_not_by_position() -> None:
    """Resuming by counting would shift silently if the seed list moved.

    And it would then attribute one cell's measurements to another, which is worse
    than losing them: real numbers hanging off the wrong label.
    """
    import inspect
    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness._write_partial)
    assert '{seed}|{label}' in source


def test_the_scratch_file_is_not_the_finished_record(tmp_path, monkeypatch) -> None:
    """A half-filled file under the record's name would be read as an answer —
    including by the campaign's refusal, which only checks that it exists."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    harness._write_partial("creda", "D", {(0, "M->U"): {1e-4: 0.5}}, 1.0,
                           lambda *a: None)
    assert not config.CEILINGS_RECORD.exists()
    assert harness.search_record() is None


def test_the_power_state_is_stamped_and_never_fatal() -> None:
    """`seconds` and `peakMiB` are dimensions of the verdict, and a measurement
    describes whichever environment produced it. A throttled run must be labelled
    as such rather than attributed to the method.

    Never fatal: a stamp that crashed the run it was documenting would be worse
    than no stamp at all.
    """
    from MIL_CREDA_Benchmark import harness

    state = harness.power_state()
    assert state["source"] in ("mains", "battery", "unknown")
    assert "power" in harness.environment()


# ------------------------------------------------- the device actually received

def test_the_environment_stamps_the_accelerator_it_got() -> None:
    """Pinning an accelerator is an intention; the stamp is the fact.

    A remote service allocates by availability, so a shard can request one class
    and silently land on another. `seconds` and `peakMiB` describe whichever
    machine produced them, so grouping them by an environment that cannot tell
    two GPU classes apart is grouping by a label that lies.
    """
    from MIL_CREDA_Benchmark import harness

    device = harness.device_class()
    assert set(device) >= {"name", "kind"}
    assert isinstance(device["name"], str) and device["name"]
    assert device["kind"] in ("cuda", "mps", "cpu")
    assert harness.environment()["device"] == device


def test_two_stamps_differ_when_the_accelerator_differs() -> None:
    """Reachable red: a stamp that ignored the device would call these equal,
    which is exactly the case sharding across accounts produces."""
    from MIL_CREDA_Benchmark import harness

    base = {"python": "3.12.13", "platform": "linux", "torch": "2.13.0",
            "selfHosted": True, "power": {"source": "mains"}}
    t4 = harness.environment_key({**base, "device": {"name": "Tesla T4", "kind": "cuda"}})
    p100 = harness.environment_key({**base, "device": {"name": "Tesla P100", "kind": "cuda"}})
    assert t4 != p100
    assert t4 == harness.environment_key(
        {**base, "device": {"name": "Tesla T4", "kind": "cuda"}})


def test_the_charge_level_does_not_change_the_environment_key() -> None:
    """It moves during a run and is not a machine class; the source is, because
    throttling is a real between-arms difference."""
    from MIL_CREDA_Benchmark import harness

    base = {"python": "3.12.13", "platform": "linux", "torch": "2.13.0",
            "selfHosted": True, "device": {"name": "Tesla T4", "kind": "cuda"}}
    full = harness.environment_key({**base, "power": {"source": "mains", "charge": 100}})
    low = harness.environment_key({**base, "power": {"source": "mains", "charge": 12}})
    battery = harness.environment_key({**base, "power": {"source": "battery", "charge": 12}})
    assert full == low
    assert battery != low


def test_median_seeds_is_the_selection_rule_on_its_own() -> None:
    """Extracted so a shard and the centre share one rule.

    Acceptance criterion: bit-identical to what `keep_median` selected before, so
    the existing callers and tests do not move.
    """
    from MIL_CREDA_Benchmark import harness

    runs = [{"seed": s, "targetAccuracy": a}
            for s, a in ((0, 0.10), (1, 0.90), (2, 0.50), (3, 0.30), (4, 0.70))]
    # The historical body, reproduced here as the golden the extraction must match.
    ordered = sorted(runs, key=lambda r: r["targetAccuracy"])
    middle = len(ordered) // 2
    span = min(config.CHECKPOINTS["G"], len(ordered))
    start = max(0, min(middle - span // 2, len(ordered) - span))
    expected = {run["seed"] for run in ordered[start:start + span]}

    assert harness.median_seeds(runs, "G") == expected
    assert harness.median_seeds(runs, "G") == {2, 3, 4}


def test_every_run_carries_the_machine_that_produced_it() -> None:
    """The stamp travels on the run, not only on the campaign.

    A shard is a remote session, and one that times out and resumes can land on
    different hardware inside a single shard id. A per-campaign stamp cannot
    express that, and it is precisely what distributing produces.

    Reachable red: with the handle only on the Reduction, a merge could not tell
    two machines apart within one shard, and would pool their cost dimensions.
    """
    import inspect
    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_one)
    assert '"env": environment_key(reduction.environment)' in source


# ------------------------------------------------------- shard-safe persistence

def test_each_shard_writes_where_no_other_shard_writes(tmp_path, monkeypatch) -> None:
    """Today one hardcoded path, opened `"w"`, truncating on every campaign.

    Two shards running at once against it would clobber each other's records, and
    the loser would be a silent partial file rather than an error. Reachable red:
    with the paths fixed, both of these resolve to the same file.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    one = harness.shard_paths("alpha")
    two = harness.shard_paths("beta")
    assert one["runs"] != two["runs"]
    assert one["partial"] != two["partial"]
    assert one["stamp"] != two["stamp"]
    for name in ("runs", "partial", "stamp"):
        assert "alpha" in str(one[name]) and "beta" in str(two[name])


def test_no_shard_named_means_the_paths_stay_where_they_always_were(tmp_path, monkeypatch) -> None:
    """A single-machine run must not move its own files.

    Every notebook, every record already written and the whole `records`
    declaration name these paths. Sharding is an addition, not a relocation.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    here = harness.shard_paths(None)
    assert here["runs"] == tmp_path / "runs.jsonl"
    assert here["partial"] == tmp_path / "ceilings.partial.json"


def test_a_shard_records_its_own_stamp_beside_its_runs(tmp_path, monkeypatch) -> None:
    """The full environment lives once per shard; runs carry only the handle."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    stored = json.loads(harness.shard_paths("alpha")["stamp"].read_text())
    assert stored["shard"] == "alpha"
    assert stored["env"] == harness.environment_key(stored["environment"])
    assert "device" in stored["environment"]
