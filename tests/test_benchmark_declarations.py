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
