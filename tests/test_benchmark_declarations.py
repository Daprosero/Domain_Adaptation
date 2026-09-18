"""The joins inside the benchmark package, checked from both ends.

Two files describe the arms and neither reads the other. `config.ARMS` says what
each arm computes; `MIL_CREDA_Benchmark.__benchmark__` says which sections of the
revision each arm exercises, which is what lets the drift report answer *does this
change oblige the bench to change*. Adding an arm to one and forgetting the other
leaves both files internally consistent and the pair wrong — the kind of omission
that arrives as a silence.

Nothing here trains: what is not config-level drives the harness with a fake
`run_one`, so the suite still runs with no network and no model weights.
"""

from __future__ import annotations

import MIL_CREDA
import MIL_CREDA_Benchmark
import json

import pytest
import torch

from MIL_CREDA_Benchmark import config, harness


def _stamped(entry: dict) -> dict:
    """`entry`, with the CURRENT config's ceiling-record provenance stamped in.

    The fixtures below hand-write ceiling records the way `harness.
    sellar_techos` writes them once `ceiling_record.stamp` runs over each
    entry -- these tests are about re-search, disk-vs-import and scale, not
    about the stamp, so a fixture missing it would now refuse for an
    unrelated reason inside `ceilings_in_force`.
    """
    return {**entry, "revision": config.REVISION,
            "kernelSigma": config.KERNEL_SIGMA,
            "attentionGamma": config.ATTENTION_GAMMA,
            "attentionTemperature": config.ATTENTION_TEMPERATURE}


def test_every_arm_declares_which_sections_it_exercises() -> None:
    declared = set(MIL_CREDA_Benchmark.__benchmark__["arms"])
    configured = {arm["id"] for arm in config.ARMS}
    assert configured - declared == set(), (
        f"arms with no section declaration: {sorted(configured - declared)}")
    assert declared - configured == set(), (
        f"sections declared for arms that no longer exist: {sorted(declared - configured)}")


def test_the_declared_arms_and_rungs_are_exactly_the_operators_four_and_three() -> None:
    """Defect (11)'s guard, re-measured against the current identities.

    This pin has been re-measured twice, and both times because a decision
    about which arms exist was taken and the pin exists to catch the NEXT
    silent reversal of whichever decision is current -- never to hold one
    revision's decision against the next. `af796f8` had `GN` retired and the
    three selection arms (`SU`, `SA`, `SK`) present; `60836d2` ("retire
    selection arms, add GN, six-dim search...") reversed both halves, so the
    selection arms and their budget (`SELECT_K`/`SELECTION_SEED`) went from
    `wiring.Arm.select` and `GN` came back as an architecture variant
    (`normalization: "sourceBatch"`). `GN` is now retired again, deliberately
    and with the whole `"sourceBatch"` axis, so the ladder is `B, E, F, G`.

    Reachable red: dropping or renaming any of `B, E, F, G`, adding a fifth
    arm without a decision behind it, or repointing any rung of
    `config.LADDER`.
    """
    assert {arm["id"] for arm in config.ARMS} == {"B", "E", "F", "G"}
    assert set(config.ARM_ORDER) == {"B", "E", "F", "G"}
    assert not hasattr(config, "SELECT_K")
    assert not hasattr(config, "SELECTION_SEED")

    rungs = {(left, right) for left, right, _ in config.LADDER}
    assert rungs == {("B", "E"), ("E", "F"), ("F", "G")}


def test_the_benchmark_is_bound_to_the_same_revision_as_the_configuration() -> None:
    assert MIL_CREDA.__implementation__["revision"] == config.REVISION


# --------------------------------------------------------- distribution -------

def test_the_distribution_declares_exactly_what_was_approved() -> None:
    """`shards.declaration()` reads this block, and `merge()` refuses every
    merge until it exists. The replication (`A`/`B`/`C`: two machines, plus a
    same-machine control) settled every one of `config.DIMENSIONS`:
    `sourceAccuracy`, `targetAccuracy`, `contribution`, `supervised`,
    `adaptationShare` and `parameters` came back bit-identical across all
    three runs, so all six pool.

    `seconds` and `peakMiB` are not among them any more. They are what the
    replication's own `perRun` group existed for -- the two differed on every
    pair, including the control, same machine both times -- but commit
    `60836d2` ("...drop timing") removed time and memory from
    `config.DIMENSIONS` entirely, not merely reclassified them: `run_one`'s
    return, the campaign's progress line, `run_smoke`'s return and
    `shards.py`'s own per-run grid stopped naming either. `perRun` stays
    declared and empty rather than deleted: it is a measured fact (nothing the
    harness reports today needs it) and not an unfilled block.

    `ceilings` and `ceilingsByTransfer` joined `epochs` when the ceiling
    stopped being one number per family. They are the parameter the search
    most recently changed, so two shards straddling that search would merge
    into one table with adaptation inert on one half and not on the other,
    and nothing would object. Both are flat top-level fields of every stamp
    (`harness.write_shard_stamp`), which is exactly the property the next
    paragraph is about.

    `labelNoise` joined them with the noise axis, and it is the entry no
    averaging could ever repair: two shards contaminated at different rates are
    two experiments, not one experiment on two machines, and a table drawn over
    both is a table nobody can attribute. It is written flat and top-level onto
    every stamp by `harness.write_shard_stamp` precisely so `disagreements()`
    can see it, which is the property the paragraph below is about.

    `commit` and `codeDigest` are not named under `identicalAcrossShards`
    even though they were approved alongside `epochs` — both live at
    `evidence.commit` / `evidence.codeDigest` on a shard's stamp, and
    `shards.disagreements()` (the forge's own, re-exported unchanged) reads
    `identicalAcrossShards` entries with a flat `stamp.get(field)`, never a
    dotted path. Naming them here would silently check nothing rather than
    check what was asked for. `parameters` fails that same flat lookup —
    it is written once per *run*, in `runs.jsonl`, and never appears on a
    shard's stamp at all (see `harness.run_one`'s return dict versus
    `harness.write_shard_stamp`) — so it stays in `poolable`, where
    averaging a constant returns the constant, rather than being placed
    somewhere `disagreements()` could never actually check it.
    """
    from MIL_CREDA_Benchmark import harness, shards

    dist = dict(shards.declaration())

    # `shardsRoot` joined the block so that a command carrying no `--shards`
    # flag of its own can measure a `@shard` witness at all. Without it the
    # witness reads unmeasured FOREVER -- not false -- so the sequence item
    # behind it can never be marked and every later step refuses on order.
    # That is not an optional narrowing of a report; it is a dead flow.
    #
    # The value is not a new destination and is not chosen here: it is where
    # `shards.read_shards()` already looks when nobody tells it otherwise.
    # Asserted by composing that default rather than by repeating the string,
    # so moving the destination reddens this instead of leaving two spellings
    # to drift apart -- the failure this repository has already had twice.
    raiz = dist.pop("shardsRoot")
    assert (config.PRODUCT.parent / raiz) == (
        config.results_for(0.0, "campaign", False) / harness.SHARDS_DIR), (
        "the declared shards root must be the one read_shards() already "
        "defaults to; a second spelling is a second answer")

    assert dist == {
        "axis": "seed",
        "poolable": ["sourceAccuracy", "targetAccuracy", "contribution",
                     "supervised", "adaptationShare", "parameters"],
        "perEnvironment": [],
        "perRun": [],
        "identicalAcrossShards": ["epochs", "ceilings", "ceilingsByTransfer",
                                  "hyperByTransfer", "labelNoise"],
    }


def test_the_declared_dimensions_now_cover_every_dimension_the_harness_measures() -> None:
    """The replication closed the gap the earlier declaration left open: all
    eight of `config.DIMENSIONS` now sit in exactly one of `poolable`,
    `perEnvironment` or `perRun`, so `shards.partition()` no longer refuses
    on the harness's own full dimension set.
    """
    from MIL_CREDA_Benchmark import shards

    dist = shards.declaration()
    classified = (set(dist["poolable"]) | set(dist["perEnvironment"])
                  | set(dist["perRun"]))
    assert classified == set(config.DIMENSIONS)

    poolable, per_environment, per_run = shards.partition(config.DIMENSIONS, dist)
    assert set(poolable) | set(per_environment) | set(per_run) == set(config.DIMENSIONS)


def test_the_benchmark_declares_what_its_protocol_assumes() -> None:
    """A change of reach leaves every arm intact and every dimension meaningless,
    so the premises sit beside the method's own declaration rather than in
    somebody's memory."""
    premises = MIL_CREDA.__implementation__["premises"]
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
    runs = [{"arm": "B", "transfer": label, "targetAccuracy": index / 10,
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
    ascending = [{"arm": "B", "transfer": label, "targetAccuracy": index / 10,
                  "contribution": 0.0} for index, label in enumerate(ran)]
    assert tables.best_transfers(ascending, count=2) == [ran[-1], ran[-2]]
    assert config.FIGURE_TRANSFER_RULE.strip(), "the rule is not declared anywhere"


def test_every_declared_floor_stays_in_the_grid() -> None:
    """A floor is what makes an aligned column readable, so every declared one is
    drawn. Dropping one is a declaration change and never a side effect: with a
    single unit declared there is a single floor, and the grid keeps it."""
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


# `test_the_selecting_arms_hold_one_budget_and_differ_only_in_the_rule` and
# `test_the_budget_is_smaller_than_the_bag` are removed. Commit `60836d2`
# ("retire selection arms...") removed the selection arms (`SU`, `SA`, `SK`)
# and their budget (`config.SELECT_K`, `SELECTION_SEED`,
# `wiring.Arm.select`'s budgeted rules) entirely: every declared arm's
# `spec["selection"]` is `None` now, so the first test's own `selecting` list
# is permanently empty and the second names a constant that no longer exists.
# Both asserted a mechanism the operator's decision retired, not a live
# property of the current arms.


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
    place that computes it agrees.

    This used to check the claim three ways -- the rung (gains) table, its
    conclusion, and the panorama that outlives both in the record. Commit
    `177ce09` ("replace the report and latent notebooks with Results_v1")
    retired `tables.render_rungs`/`conclusion_rungs`: none of the six required
    sections of `Benchmark_Results.ipynb` reads a rung table any more, and no other
    notebook did either (see `__benchmark__["search"]`'s own retirement note
    in `src/MIL_CREDA_Benchmark/__init__.py`). What survives is
    `harness.paired_across_transfers` itself -- `campaign()` still calls it to
    write `"panorama"` into every `summary.json` (line ~1890), so the
    convention still has to hold for the one place it is still computed and
    read, even with no table printing it.

    Reachable red: flipping the subtraction inside `paired_across_transfers`
    without flipping `favouringRight`'s own count.
    """
    from MIL_CREDA_Benchmark import harness

    left, right, _ = config.LADDER[0]
    grid = _one_rung_grid(left_mean=0.40, right_mean=0.70)
    panorama = harness.paired_across_transfers(grid)

    row = [r for r in panorama
           if r["rung"] == f"{left}->{right}" and r["metric"] == "targetAccuracy"][0]
    # The right arm is 30 points above, so left - right is negative.
    assert row["meanDifference"] == pytest.approx(-0.30)
    assert row["meanDifference"] < 0
    # The field counts transfers won by the right arm, which is now the negative
    # side. Flipping the subtraction without flipping this would report the right
    # arm losing all six of the transfers it won.
    assert row["favouringRight"] == len(config.TRANSFERS)


# `test_the_rung_conclusion_names_who_is_ahead_not_how_far_it_moved` and the
# `perRun` table's three tests (`test_the_per_run_table_prints_one_row_per_
# run_tagged_with_its_environment`, `test_the_per_run_table_reports_nothing_
# measured_when_the_grid_is_empty`, `test_the_per_run_conclusion_declines_to_
# pool_and_points_back_at_the_table`) are removed. `tables.conclusion_rungs`,
# `render_per_run` and `conclusion_per_run` are retired along with
# `render_rungs` (see the note on the test above): the rungs table had no
# reader left to write a conclusion for, and `seconds`/`peakMiB` -- the only
# dimensions `perRun` ever held anything for -- are gone from
# `config.DIMENSIONS` entirely (commit `60836d2`, "...drop timing"), not
# merely reclassified, so there is no longer a `perRun` dimension for either
# function to render.


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

def _search_entry(scores: dict[float, list[float]], monkeypatch, tmp_path) -> dict:
    """One family's search entry, given one score per (cell, ceiling).

    Drives `harness.search_ceilings` itself with a fake `run_one`, in the shape
    `test_search_engine.py` uses for the trials engine: the centring and the
    tie-break under test are the ones the campaign runs. This file used to carry
    its own copy of both, and a copy proves nothing — the real rule could be
    deleted from `harness.py` with these two tests still green.

    Each list is read cell by cell: one cell per seed, over a single transfer,
    which is the same `(seed, transfer)` cell the search centres on.
    """
    grid = sorted(scores)
    seeds = list(range(len(scores[grid[0]])))
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    monkeypatch.setattr(config, "CEILING_GRID", grid)
    monkeypatch.setattr(config, "SEARCH_SEEDS", seeds)
    monkeypatch.setattr(config, "SEARCH_ARMS", {"milcreda": "G"})
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")

    class _Bags:
        pass

    def _build(code, cache, seed, noise=0.0):
        return _Bags()

    def _run_one(arm, transfer, seed, reduction, device, material, *, ceiling, role):
        return {config.SEARCH_CRITERION: scores[ceiling][seeds.index(seed)]}

    monkeypatch.setattr(harness.bags, "build", _build)
    monkeypatch.setattr(harness, "run_one", _run_one)
    found = harness.search_ceilings(
        harness.Reduction(seeds=seeds, epochs=config.SEARCH_EPOCHS),
        "cpu", progress=lambda *_: None,
        transfers=[config.SEARCH_TRANSFERS[0]])
    return found["milcreda"]


def _by_ceiling(entry: dict, key: str) -> dict[float, float]:
    """One reading of the searched grid, keyed by ceiling, as the record holds it."""
    return {row["ceiling"]: row[key] for row in entry["grid"]}


def test_the_pairing_survives_a_cell_that_is_simply_harder(monkeypatch, tmp_path) -> None:
    """What pairing buys, in the case that motivates it.

    Two searches over the same two ceilings, differing only in how hard the
    second cell is: ceiling 0.1 wins every cell of both by the same two points.
    The bare mean carries the cell's own difficulty into the reading and moves
    twenty-two points between the two; centring each cell on its own mean reports
    the effect and nothing else, identically in both, because every ceiling was
    measured on exactly the same material.

    The claim is about the reading and not about the winner, and that is not a
    weaker test but the honest one: the centring subtracts the same number from
    every ceiling of a cell, so on a complete grid it can never move the argmax.
    A pick-only assertion here would survive the pairing being deleted.

    Reachable red: with `paired` read off the bare values, the two searches stop
    agreeing and the effect is buried under the difficulty.
    """
    facil = _search_entry({0.01: [0.90, 0.50], 0.1: [0.92, 0.52]}, monkeypatch, tmp_path)
    dificil = _search_entry({0.01: [0.90, 0.05], 0.1: [0.92, 0.07]}, monkeypatch, tmp_path)

    assert facil["ceiling"] == dificil["ceiling"] == 0.1

    # the effect, centred on each cell's own mean: two points, and nothing of
    # where the cell sat
    apareado = _by_ceiling(dificil, "paired")
    assert apareado == pytest.approx({0.01: -0.01, 0.1: 0.01})
    assert _by_ceiling(facil, "paired") == pytest.approx(apareado)

    # the bare mean of those same two searches, which is what pairing avoids
    crudo_facil = _by_ceiling(facil, config.SEARCH_CRITERION)
    crudo_dificil = _by_ceiling(dificil, config.SEARCH_CRITERION)
    assert crudo_facil[0.1] - crudo_dificil[0.1] == pytest.approx(0.225), \
        "el crudo se mueve con la dificultad de la celda"


def test_a_tie_goes_to_the_smallest_ceiling(monkeypatch, tmp_path) -> None:
    """Below some point a term is inert and everything ties; on that stretch the
    tie-break is what actually chooses, so it is declared.

    Reachable red: the largest of the tied ceilings wins instead.
    """
    scores = {1e-4: [0.5, 0.5], 1e-3: [0.5, 0.5], 1e-2: [0.5, 0.5]}
    entry = _search_entry(scores, monkeypatch, tmp_path)
    # the tie is real and the record says so, so the pick below is the tie-break's
    assert entry["tied"] == sorted(scores) and entry["decidedByTieBreak"]
    assert entry["ceiling"] == 1e-4


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
        "creda": _stamped({"ceiling": 1e-4, "atRequiredScale": False}),
        "milcreda": _stamped({"ceiling": 1.0, "atRequiredScale": True}),
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    # `campaign()` hace `results_for(...).mkdir()` y `models_for(...).mkdir()`
    # ANTES de cualquiera de sus tres rechazos, así que un test que sólo
    # redirige el registro igual crea dos directorios en el árbol del dueño.
    # Dos raíces y no una: `MODELS` es hermana de `RESULTS` y no se deriva de
    # ella, así que redirigir una deja la otra apuntando a la corrida real.
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models" / "Benchmark")

    reduction = harness.Reduction(ceilings={"creda": 1e-4, "milcreda": 1.0})
    with pytest.raises(SystemExit) as raised:
        harness.campaign(reduction, torch.device("cpu"), arms=["B"],
                         progress=lambda *a: None)
    assert "below scale" in str(raised.value)
    assert "creda" in str(raised.value)


# `test_a_single_machine_campaign_records_its_per_run_readings` is removed.
# It asserted that `config.DIMENSIONS` names at least one `perRun` dimension
# ("the declaration has to name at least one for this to mean anything") --
# true when `seconds`/`peakMiB` were still measured, and false now that
# commit `60836d2` ("...drop timing") removed both from `config.DIMENSIONS`
# entirely rather than merely reclassifying them. `gridPerRun` itself is not
# retired -- `campaign()` still assembles it from whatever `shards.
# declaration()["perRun"]` names, which is `[]` today -- so a one-machine
# summary still carries the key and it is legitimately empty; there is no
# longer a per-run reading for this test to demonstrate keeps its own seed
# and environment.


def test_the_contaminated_correspondence_is_its_own_rendering() -> None:
    """Two halves calling one renderer with no dimension named read as one
    measurement rendered twice, and the contaminated correspondence was exactly
    that. It gets its own name because they are two different numbers, and
    saying so is not dodging the duplication check -- it leaves it intact for
    the case it exists to catch.

    The other five took the other exit: `render_readings`, `render`,
    `render_rungs`, `render_gains` and `render_per_run_summary` grew a `Ruido`
    column with a block per material, and their contaminated twins are gone, so
    one call renders both numbers. This one did not --- its rows are subjects and
    not arms, so a `con` block would have no `sin` row to pair with --- and the
    pair of names survives here and only here.

    And the empty case states the rate rather than a bare parenthesis, so a
    reader meets which campaign has not left checkpoints yet."""
    from MIL_CREDA_Benchmark import tables

    declared = MIL_CREDA_Benchmark.__benchmark__["report"]["renderers"]
    assert "tables.render_correspondence_contaminated" in declared

    empty = tables.render_correspondence_contaminated([], 0.2)
    assert "0.2" in empty and "no existe" in empty

    scored = [{"arm": arm, "transfer": "M->U", "hits": 2, "classes": 3,
               "mass": 0.5} for arm in config.BAG_PANELS]
    assert (tables.render_correspondence_contaminated(scored, 0.2, markdown=True)
            == tables.render_correspondence(scored, markdown=True))


def test_the_ceiling_record_says_which_machine_measured_it() -> None:
    """The ceiling record was the only record here without an environment stamp.

    Every shard.json carries interpreter, platform, torch version and device.
    The ceilings carried none of the four, and the ceiling is the scalar that
    governs the whole campaign -- so with the full search running on a worker,
    a later reader could not tell a record searched here from one searched
    there, which is exactly the distinction the stamp preserves everywhere
    else.

    Per family and not once at the top, because the record is a flat
    family -> entry map that every consumer iterates: a sibling key would be
    read as a third family and printed as a row. It is also the right
    granularity, since the two families' searches can be split across workers.
    """
    from MIL_CREDA_Benchmark import harness, tables

    reduction = harness.Reduction(environment={
        "interpreter": "/somewhere/.venv", "python": "3.12.13",
        "platform": "linux-x86_64", "torch": "2.13.0", "selfHosted": False,
        "power": {"source": "ac"}, "device": {"name": "T4", "kind": "cuda"}})
    found = {"creda": {"ceiling": 1e-4}, "milcreda": {"ceiling": 1.0}}

    stamped = harness.sellar_techos(found, reduction)

    for family, entry in stamped.items():
        assert entry["environment"] == reduction.environment, family
        # The short handle too, so a reader can compare and group without
        # carrying the whole stamp -- the same pair a run carries.
        assert entry["env"] == harness.environment_key(reduction.environment)

    # And the readers still see exactly two families: the stamp went inside the
    # entries, so nothing new appears as a row.
    rendered = tables.render_ceilings_by_transfer(stamped, ["M->U"], markdown=True)
    assert rendered.count("`creda`") == 1 and rendered.count("`milcreda`") == 1
    assert "environment" not in rendered


# --------------------------------------------------------- run_smoke checkpoints

def _fake_run_one(fake_state):
    def run_one(arm_id, transfer, seed, reduction, device, material, **kwargs):
        return {
            "arm": arm_id, "transfer": f"{transfer[0]}->{transfer[1]}", "seed": seed,
            "env": "test-env", "targetAccuracy": 0.5, "sourceAccuracy": 0.5,
            "seconds": 0.01, "peakMiB": 1.0, "parameters": 4, "contribution": 0.1,
            "supervised": 0.2, "adaptationShare": 0.3, "curve": [],
            "epochs": [{"epoch": 0}], "state": dict(fake_state),
        }
    return run_one


def _fake_build():
    from types import SimpleNamespace

    def build(code, cache, seed):
        return SimpleNamespace(manifest={"code": code, "seed": seed})
    return build


def test_run_smoke_writes_one_checkpoint_through_the_campaigns_own_path(
        tmp_path, monkeypatch) -> None:
    """Phase 2's `latent.available()`/`latent.load()` read `config.MODELS` in
    exactly `campaign()`'s own shape — a `.pt` saved with `torch.save({k: v.cpu()
    for k, v in state.items()}, ...)` beside a manifest `keep_median` writes
    through `bags.write_manifest`. A bespoke save in `run_smoke` would prove
    nothing about whether that real path works; this exercises
    `run_smoke(checkpoint=True)` over the same one, without paying for a real
    training loop — `run_one` and `bags.build` are faked, everything downstream
    of the state dict they return is real."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results")
    monkeypatch.setattr(harness, "run_one", _fake_run_one({"weight": torch.zeros(2, 2)}))
    monkeypatch.setattr(harness.bags, "build", _fake_build())

    harness.run_smoke(seed=0, checkpoint=True)

    left, right = config.VERDICT_TRANSFERS[0]
    stem = f"{harness.SMOKE_ARM}_{left}-{right}_seed0"
    weights = config.MODELS / f"{stem}.pt"
    manifest = config.MODELS / f"{stem}.manifest.json"
    assert weights.is_file()
    assert manifest.is_file()

    loaded_state = torch.load(weights)
    assert set(loaded_state) == {"weight"}

    written = json.loads(manifest.read_text())
    assert written["arm"] == harness.SMOKE_ARM
    assert written["seed"] == 0
    assert written["source"] == {"code": left, "seed": 0}
    # Exactly one, never the three per cell a campaign's own `CHECKPOINTS`
    # asks for: `median_seeds` degenerates to the only seed there is.
    assert len(list(config.MODELS.glob("*.pt"))) == 1


def test_run_smoke_writes_no_checkpoint_by_default(tmp_path, monkeypatch) -> None:
    """Off unless asked for: most rehearsals only need `runs.jsonl`, and a
    checkpoint costs a model's worth of disk on every call this doesn't ask
    for one. `config.MODELS` must not even be created."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "MODELS", tmp_path / "Models")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results")
    monkeypatch.setattr(harness, "run_one", _fake_run_one({"weight": torch.zeros(1)}))
    monkeypatch.setattr(harness.bags, "build", _fake_build())

    harness.run_smoke(seed=0)

    assert not config.MODELS.exists()


# ------------------------------ the notebook actually obtains what it needs

def _notebook_code_cells(name: str) -> list[dict]:
    """A notebook's code cells, in order."""
    path = config.REPOSITORY / "MIL-CREDA" / "Notebooks" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


# `_search_report_notebook()` is removed along with
# `Benchmark_Search_Report_v1.ipynb`, its `search-report` step and
# `steps.informe_de_busqueda`: the search's own report is not a result of this
# paper. The run that searches keeps its own notebook
# (`Benchmark_Ceiling_Search.ipynb`, driven by `search-pilot`); what went is
# the notebook that only presented the record it leaves.


# `test_the_notebook_obtains_the_ceilings_before_it_runs_the_campaign` and
# `test_the_notebook_forecasts_the_search_before_spending_it` are removed.
# Both read `Benchmark_Campaign_v1.ipynb`, deleted along with report, latent,
# campaign, noise-report and both diagnostic notebooks (commit `2f9bf32`,
# "delete every artefact the new structure will not overwrite"): the campaign
# is no longer notebook-driven at all -- `src/MIL_CREDA/__init__.py`'s own
# `resultados` docstring names the gap explicitly ("nada en `__steps__`
# invoca `harness.campaign()`/`run_campaign_shard` hoy"), which is what this
# stretch's own campaign step closes (see `__steps__["campaign"]` below).
# The harness-level equivalent of the join these two tested --
# `ceilings_in_force`/the record read before `campaign()` runs -- is already
# covered, unconditionally of any notebook, by
# `test_run_campaign_shard_obtains_ceilings_before_it_runs_the_campaign`
# further down this file, and stays green.


# --------------------- run_campaign_shard(): one shard's campaign, asked by name

def test_run_campaign_shard_is_callable_with_json_alone() -> None:
    """`tools/distribute.py`'s `run_shard()` already has the right shape, and a
    remote worker cannot reach it: `tools/` is outside every declared clone
    path, and that module path-imports the forge's own packer, which does not
    exist inside a kernel. So the fan-out the distribution declaration is built
    around had no way to be asked for remotely.

    Reachable red: before this, `harness` exposed `run_pilot`, `run_smoke`,
    `run_one` and `run_search`, and none of them runs one shard of the grid.

    `pilot` is the third parameter and is JSON-native like the other two,
    which is the whole property this asserts: a remote job names
    `{module, function, kwargs}` and hands over plain JSON, so a dial it
    cannot spell is a dial only a notebook has.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    parameters = inspect.signature(harness.run_campaign_shard).parameters
    assert list(parameters) == ["shard", "seeds", "pilot"]
    assert all(p.default is not inspect.Parameter.empty for p in parameters.values())


def test_run_campaign_shard_runs_at_full_scale_never_the_pilots(
        tmp_path, monkeypatch) -> None:
    """Called with no scale, a shard is a slice of the FULL campaign.

    This docstring used to say the epoch count is not a dial at all, "for the
    same reason it is not one on the search". That sentence outlived its
    mechanism twice over: the search took a `pilot` dial first, and this
    function has one now. What the sentence was right about is merger --- a
    shard measured at pilot scale would be pooled with the full ones as
    though it were one of them --- and that is still why the default is the
    full scale and why THIS test exists: the path a remote job takes when it
    names no scale must be byte-identical to what it always was. A pilot
    shard is never merged with anything, because it is not written where
    anything merges from; `test_run_campaign_shard_at_pilot_routes_to_the_
    pilot_tree` is the other half.

    `config.CEILINGS_RECORD` is pinned to an isolated `tmp_path` record so
    this test can never reach the real `Results/Benchmark/ceilings.json` —
    which does not exist on disk — and the tripwire below turns any future
    unpinning of that record into a named, millisecond-long failure instead
    of a real ceiling search running inside the unit suite.
    """
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "creda": _stamped({"ceiling": 1e-4, "atRequiredScale": True}),
        "milcreda": _stamped({"ceiling": 1.0, "atRequiredScale": True}),
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    monkeypatch.setattr(config, "CEILINGS", {})
    monkeypatch.setattr(harness, "search_ceilings", lambda *a, **k: pytest.fail(
        "the unit suite reached the real ceiling search"))

    seen = {}

    def _spy(reduction, device, arms=None, progress=print, shard=None):
        seen["reduction"], seen["shard"] = reduction, shard
        return {}

    monkeypatch.setattr(harness, "campaign", _spy)
    harness.run_campaign_shard(shard="a", seeds=[3, 4])

    assert seen["reduction"].epochs == config.FULL_EPOCHS
    assert seen["reduction"].seeds == [3, 4]
    assert seen["shard"] == "a"
    # A shard is one JSON-callable campaign entrypoint among four that must
    # each obtain ceilings in force before running — proven behaviourally,
    # against the record's own winners, not only against source text.
    assert seen["reduction"].ceilings == {"creda": 1e-4, "milcreda": 1.0}


def test_run_campaign_shard_at_pilot_routes_to_the_pilot_tree(
        tmp_path, monkeypatch) -> None:
    """The two calls, side by side, because either one alone is green.

    What this closes was measured and not argued: the declared flow was
    walked at pilot scale and this function trained the FULL grid --- five
    arms, six transfers, thirty seeds, twenty epochs --- for 56 minutes
    before a person killed it by hand. Nothing refused, because
    `run_campaign_shard` hardcoded `config.FULL_SEEDS`/`config.FULL_EPOCHS`
    and took no dial at all.

    Three things have to move together and are asserted together, because a
    `Reduction` that carries a pilot scale and a full destination is exactly
    the record that lies about itself. The DESTINATION is asserted through
    `results_for`/`models_for` --- the gates that turn `pilot` into a path
    --- rather than against a spelled path, so a change to how the `Pilot/`
    segment is composed moves the expectation with it.

    And the full call is asserted in the same test rather than left to the
    one above: `pilot=False` has to reach the same tree it always reached,
    and the two destinations have to DIFFER. Asserting only the pilot side
    would pass a `results_for` that ignored its own coordinate.

    Reachable red, proven by mutation and restore: hardcoding `pilot=False`
    on the `Reduction` (its previous behaviour) leaves the pilot call
    writing into `Results/Benchmark`; hardcoding `config.FULL_EPOCHS` leaves
    it training twenty epochs into the pilot tree.
    """
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    payload = json.dumps({
        "creda": _stamped({"ceiling": 1e-4, "atRequiredScale": True}),
        "milcreda": _stamped({"ceiling": 1.0, "atRequiredScale": True}),
    })
    record.write_text(payload, encoding="utf-8")
    pilot_record = tmp_path / "ceilings.pilot.json"
    pilot_record.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", pilot_record)
    monkeypatch.setattr(config, "CEILINGS", {})
    monkeypatch.setattr(harness, "search_ceilings", lambda *a, **k: pytest.fail(
        "the unit suite reached the real ceiling search"))

    seen: list = []

    def _spy(reduction, device, arms=None, progress=print, shard=None):
        seen.append(reduction)
        return {}

    monkeypatch.setattr(harness, "campaign", _spy)
    harness.run_campaign_shard(pilot=True)
    harness.run_campaign_shard(pilot=False)

    # Two calls per scale, not one: every accuracy table declares a clean block
    # and a contaminated one, and a shard that ran only the first leaves half of
    # every table with nothing behind it. Asserted as a count so a call silently
    # dropping back to one condition cannot pass by having the clean one right.
    assert len(seen) == 4, [r.labelNoise for r in seen]
    by_scale = {True: [r for r in seen if r.pilot], False: [r for r in seen if not r.pilot]}
    for pilot, pair in by_scale.items():
        assert [r.labelNoise for r in pair] == [0.0, config.NOISE_REPORTED], (
            f"pilot={pilot} did not run clean then contaminated: "
            f"{[r.labelNoise for r in pair]}")
        # the searched values are resolved ONCE and carried into both, which is
        # the agreement: the search runs clean and its values are used unchanged
        # under noise, so what mitigates the contamination is the method
        assert pair[0].ceilings == pair[1].ceilings
        assert pair[0].hyperByTransfer == pair[1].hyperByTransfer

    piloted, full = by_scale[True][0], by_scale[False][0]

    # the scale, from the two constants `is_pilot_scale` reads and no others
    assert piloted.epochs == config.EPOCHS
    assert piloted.seeds == list(config.SEEDS)
    assert full.epochs == config.FULL_EPOCHS
    assert full.seeds == list(config.FULL_SEEDS)

    # the destination, through the gates and never through a spelling written here
    for reduction, pilot in ((piloted, True), (full, False)):
        assert reduction.pilot is pilot
        assert (config.results_for(reduction.labelNoise, reduction.kind,
                                   reduction.pilot)
                == config.results_for(0.0, "campaign", pilot))
        assert (config.models_for(reduction.labelNoise, reduction.kind,
                                  reduction.pilot)
                == config.models_for(0.0, "campaign", pilot))

    # and they DIFFER: without this, a gate ignoring its own coordinate would pass
    assert (config.results_for(0.0, "campaign", True)
            != config.results_for(0.0, "campaign", False))
    assert (config.models_for(0.0, "campaign", True)
            != config.models_for(0.0, "campaign", False))


def test_run_mechanism_sweep_shard_at_pilot_routes_its_record_to_the_pilot_tree(
        tmp_path, monkeypatch) -> None:
    """Section 4's own half, and it is not the campaign's with a name changed.

    `run_mechanism_sweep` writes ONE json file, and its path was a fixed
    string (`config.PRODUCT / tables.MECHANISM_RECORD`) that no coordinate
    reached --- excused in `config.DESTINOS_SIN_COORDENADA` on the ground
    that "section 4 declares no pilot `Reduction` of its own", which
    described the absence of a dial rather than a property of the record.
    So a `pilot` dial alone would have written three-epoch numbers over the
    full record section 4 is read from: the exact defect this change closes,
    wearing the opposite mask.

    The record is therefore asserted on DISK --- which file exists after each
    call --- and not through the signature or the source, because the
    signature was never the thing that was wrong. The full-scale path is
    asserted to be byte-identical to the one the old fixed string composed,
    so "derived, not respelled" is a measurement rather than a claim.

    Reachable red, proven by mutation and restore: restoring the fixed
    `config.PRODUCT / tables.MECHANISM_RECORD` leaves the pilot call writing
    the full record.
    """
    from pathlib import Path

    from MIL_CREDA_Benchmark import harness, tables

    monkeypatch.setattr(config, "PRODUCT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models" / "Benchmark")
    monkeypatch.setattr(harness, "with_ceilings_in_force", lambda r, d, **k: r)
    monkeypatch.setattr(harness, "run_mechanism", lambda *a, **k: {
        "mechanism": "abmil", "transfer": "M->U", "targetAccuracy": 0.5})
    monkeypatch.setattr(harness.bags, "build", lambda *a, **k: {"stub": True})

    completo = config.results_for(0.0, "campaign", False) / Path(
        tables.MECHANISM_RECORD).name
    ensayo = config.results_for(0.0, "campaign", True) / Path(
        tables.MECHANISM_RECORD).name

    # the full-scale spelling is the one the fixed path composed, byte for byte
    assert completo == config.PRODUCT / tables.MECHANISM_RECORD
    assert ensayo != completo

    harness.run_mechanism_sweep_shard(pilot=True, seeds=[0])
    assert ensayo.is_file(), "the pilot call wrote nothing into the pilot tree"
    assert not completo.exists(), (
        "the pilot call wrote the full run's record, which is the one section 4 "
        "presents")

    harness.run_mechanism_sweep_shard(pilot=False, seeds=[0])
    assert completo.is_file(), "the full call wrote nothing into its own tree"


def test_run_mechanism_sweep_shard_is_callable_with_json_alone() -> None:
    """Its scale dial has to be spellable by a remote job too.

    Section 4 reaches a worker the same way the campaign does --- a
    `run-config.json` naming `{module, function, kwargs}` --- so a dial that
    only a notebook could pass would leave the two entrypoints asymmetric for
    no reason anybody declared.

    Reachable red: dropping `pilot` from the signature, or giving it no
    default, which would make the plain full-scale call a usage error.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    parameters = inspect.signature(harness.run_mechanism_sweep_shard).parameters
    assert list(parameters) == ["seeds", "pilot"]
    assert all(p.default is not inspect.Parameter.empty for p in parameters.values())


def test_run_campaign_shard_obtains_ceilings_before_it_runs_the_campaign() -> None:
    """The same join `run_pilot()` and `run_search()` are already held to
    (`test_run_pilot_obtains_ceilings_before_it_runs_the_campaign`): `campaign()`
    refuses outright without `reduction.ceilings` already populated, so this
    has to close that gap itself rather than hand a caller a refusal it
    cannot read.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_campaign_shard)
    # "campaign(replace(reduction" and not the bare "campaign(": the docstring
    # above already mentions "`campaign()`, which hands it to `shard_paths()`" —
    # matching on "campaign(" alone would find that prose, not the call. Both
    # conditions are asserted, and the LAST of them: the ceilings are resolved
    # once, before either runs, so a second call placed above the resolution
    # would run the contaminated half against nothing.
    calls = [i for i in range(len(source))
             if source.startswith("campaign(replace(reduction", i)]
    assert len(calls) == 2, f"expected both conditions, found {len(calls)}"
    assert source.index("ceilings_in_force(") < min(calls)


def test_run_campaign_shard_writes_into_its_own_namespace() -> None:
    """Two shards running at once would clobber each other's records if the
    name did not reach `campaign()`. `shard_paths()` is what separates them,
    and it only separates them if the caller passes the name through.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_campaign_shard)
    assert "shard=shard" in source


# ------------------------------ run_search(): the search alone, asked for by name

def test_run_search_is_callable_with_json_alone() -> None:
    """A remote job names a `module.function` and hands it JSON keyword
    arguments; it cannot build a `Reduction` or a `torch.device`. So every
    parameter of an entrypoint has to be JSON-native and optional, or the
    entrypoint cannot be reached from a job at all.

    Reachable red: before this, `harness` exposed `run_pilot`, `run_smoke`
    and `run_one`, and the search — the one experiment this repository is
    blocked on — had no launcher of its own.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    parameters = inspect.signature(harness.run_search).parameters
    assert "shard" in parameters and parameters["shard"].default is None
    # La intencion, y no una lista de nombres: cada parametro tiene que ser
    # JSON-nativo y opcional. Una lista exacta declaraba lo mismo mientras hubo
    # uno solo, y despues prohibia agregar un segundo que cumple la regla.
    for name, param in parameters.items():
        assert param.default is not inspect.Parameter.empty, name
        assert isinstance(param.default, (str, int, float, bool, type(None))), name


def test_run_search_offers_no_scale_dial_and_pilot_is_not_one() -> None:
    """La razon por la que no hay dial es el **destino**, no la escala.

    Un `epochs` suelto seria aceptado, ignorado por `search_ceilings()` y su
    respuesta escrita igual al archivo donde va la respuesta completa: un
    knob que solo parece funcionar. `pilot=True` no es ese knob — cambia el
    archivo. Corre a su propia escala declarada y escribe a
    `CEILINGS_PILOT_RECORD`, al que el registro completo le gana siempre.
    """
    import inspect

    from MIL_CREDA_Benchmark import config, harness

    assert "epochs" not in inspect.signature(harness.run_search).parameters
    assert "seeds" not in inspect.signature(harness.run_search).parameters
    assert "epochs" in inspect.signature(harness.run_pilot).parameters
    assert config.CEILINGS_PILOT_RECORD != config.CEILINGS_RECORD


def test_run_search_runs_at_the_declared_scale_and_never_the_pilots(monkeypatch) -> None:
    """What the search is handed has to be the scale it declares for itself,
    not whatever the pilot happens to be running at. Asserted on the
    `Reduction` that actually reaches `ceilings_in_force`, because the
    record written afterwards carries that scale and a later reader trusts
    it.
    """
    from MIL_CREDA_Benchmark import harness

    seen = {}

    def _spy(reduction, device, progress=print, shard=None, pilot=False):
        seen["reduction"] = reduction
        seen["pilot"] = pilot
        return {}

    monkeypatch.setattr(harness, "ceilings_in_force", _spy)
    monkeypatch.setattr(harness, "search_record", lambda pilot=False: {"creda": {}})

    harness.run_search()

    assert seen["reduction"].epochs == config.SEARCH_EPOCHS
    assert seen["reduction"].seeds == list(config.SEARCH_SEEDS)
    assert seen["reduction"].epochs != config.EPOCHS


def test_run_search_refuses_to_report_success_with_no_record_on_disk() -> None:
    """The record is the whole product. A run that returned normally while
    `ceilings.json` was never written would report an answer it cannot
    show, and everything downstream reads the file rather than the return
    value.
    """
    from MIL_CREDA_Benchmark import harness

    original_force = harness.ceilings_in_force
    original_record = harness.search_record
    harness.ceilings_in_force = lambda *a, **k: {}
    harness.search_record = lambda pilot=False: None
    try:
        with pytest.raises(SystemExit) as raised:
            harness.run_search()
    finally:
        harness.ceilings_in_force = original_force
        harness.search_record = original_record
    assert "left no record" in str(raised.value)


# ---------------------------- run_pilot(): the notebook, headless, cheapenable

def test_run_pilot_epochs_default_is_reductions_own_default_not_full_epochs() -> None:
    """`run_pilot()` is what a remote job's `--run-kwargs` actually reaches
    — distinct from `tools/distribute.py`'s `run_shard()`, whose epochs
    are pinned to `config.FULL_EPOCHS` for the fan-out path. With no
    override `run_pilot()` must reproduce exactly what a bare
    `Reduction()` already defaults to (`config.EPOCHS`), so calling it
    with nothing changes nothing about what the pilot cells already run.

    Reachable red: before this task, `harness` exposed no `run_pilot`
    attribute at all.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    default = inspect.signature(harness.run_pilot).parameters["epochs"].default
    assert default == config.EPOCHS
    assert default != config.FULL_EPOCHS


def test_run_pilot_epochs_is_overridable_as_a_plain_json_kwarg() -> None:
    """The whole point: a caller reaches a cheaper-than-pilot run by
    passing `{"epochs": 2}` through `--run-kwargs`, never by editing
    `config.py`. `seeds` stays optional the same way, defaulting to
    `config.SEEDS` exactly like a bare `Reduction()` would.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    params = inspect.signature(harness.run_pilot).parameters
    assert params["epochs"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    )
    assert params["seeds"].default is None


def test_run_pilot_obtains_ceilings_before_it_runs_the_campaign() -> None:
    """The same join the notebook itself is held to
    (`test_the_notebook_obtains_the_ceilings_before_it_runs_the_campaign`),
    now inside `run_pilot()`'s own source: `campaign()` refuses outright
    without `reduction.ceilings` already populated, so this has to close
    that gap itself rather than hand a caller a refusal it cannot read.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_pilot)
    assert source.index("ceilings_in_force(") < source.index("campaign(")


def test_run_pilot_is_the_single_machine_counterpart_never_a_shard() -> None:
    """The one thing that tells `run_pilot()` apart from
    `tools/distribute.py`'s `run_shard()`: it never passes `shard=` into
    `campaign()`, so the whole seed axis runs in one machine's own
    namespace rather than one shard of it."""
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_pilot)
    assert "shard=" not in source


def test_run_pilot_is_reachable_under_src_for_a_remote_run_module() -> None:
    """A remote job's generated runner only ever puts `<clone>/src` on
    `sys.path` (never `tools/`), and `--run-module` is resolved the same
    way at generation time: only what lives under `src/` is importable at
    all. `run_pilot()` has to live in `harness.py`, under `src/`, for
    `--run-module MIL_CREDA_Benchmark.harness --run-function run_pilot`
    to ever resolve — `run_shard()` staying in `tools/distribute.py` is
    what makes it the fan-out path's own, local-invocation-only entry
    point instead.
    """
    import inspect
    from pathlib import Path

    from MIL_CREDA_Benchmark import harness

    assert Path(inspect.getfile(harness.run_pilot)).is_relative_to(
        config.REPOSITORY / "src")


# `test_the_ceilings_reach_the_written_report_and_not_only_the_screen` is
# removed, and the agreement it stood for is NOT quietly dropped -- it is
# answered somewhere else, which is why this is a removal and not a hole.
#
# What it asserted: the ceilings reach something durable a later session can
# open, not only a `print` that scrolls off. The cell it read was
# `Benchmark_Search_Report_v1.ipynb`'s `show(render_ceilings)` /
# `show(conclusion_ceilings)` pair, and that notebook has been retired entire
# together with its `search-report` step -- the search's own report is not a
# result of this paper.
#
# What still satisfies the agreement, measured rather than assumed: the
# durable artefact is the RECORD, `Results/Benchmark/ceilings.json`, which the
# search writes through `config.ceilings_record_for(pilot)` and reads back
# through `harness.search_record(pilot=...)`. It carries each family's
# ceiling, its grid, its ties and `atRequiredScale`, and it is what the sweep
# and the campaign both consume -- so it outlives any session by construction,
# where a rendered table only outlived one by being executed in place.
# `__records__["ceilings"]` declares it, `tests/test_ceiling_record.py` and
# `tests/test_search_records.py` pin its contents, and
# `tests/test_ceiling_readers.py` pins `render_ceilings`/`conclusion_ceilings`
# themselves -- the renderers stay declared and stay tested; what went is the
# notebook that showed them.
#
# What genuinely went with it, said plainly rather than folded away: no
# notebook in this tree renders a ceiling table any more. Whether the results
# notebook should show one is a decision for whoever owns Section 0 of
# `Benchmark_Results.ipynb`, not something this removal may take by adding a cell.


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
    assert "ceilingSearch=search_record(pilot=reduction.pilot)" in source, (
        "the campaign does not copy the search record into its reduction")

    # Y de SU escala, que es la otra mitad y la que se rompe sola. El sello
    # compartía la variable con la lectura que GOBIERNA --- `searched`, que
    # nombra `pilot=False` porque abajo se le exige `atRequiredScale` --- y son
    # dos preguntas distintas: una campaña de ensayo se sellaba con la rejilla
    # de la búsqueda completa mientras corría bajo los techos del ensayo, así
    # que el informe explicaba cómo se eligió un número que esa corrida no usó.
    # Las dos lecturas se afirman juntas porque separarlas es justamente el
    # cambio, y volver a unirlas se lee igual de verde con una sola.
    assert "searched = search_record(pilot=False)" in source, (
        "la lectura que gobierna dejó de nombrar la escala completa")


def test_an_existing_ceiling_record_is_never_re_searched(tmp_path, monkeypatch) -> None:
    """Overwriting an answer because a later caller wanted a different one is the
    silent refunding the campaign's refusal exists to prevent."""
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "creda": _stamped({"ceiling": 0.5, "atRequiredScale": True}),
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
            "milcreda": _stamped({"ceiling": 0.25, "atRequiredScale": True}),
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
    # El de ensayo también, y no por prolijidad: sin esto el `search_record`
    # de abajo caía sobre el registro de ensayo REAL del disco de quien corre
    # la suite, así que el verde dependía de qué hubiera corrido esa persona.
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD",
                        tmp_path / "ceilings.pilot.json")
    harness._write_partial("creda", "D", {(0, "M->U"): {1e-4: 0.5}}, 1.0,
                           lambda *a: None)
    assert not config.CEILINGS_RECORD.exists()
    assert harness.search_record(pilot=False) is None


def test_a_resumed_ceiling_the_grid_no_longer_has_refuses_by_name(
        tmp_path, monkeypatch) -> None:
    """A partial is a file on disk and the grid is a line in config.py —
    nothing keeps them in step. Editing the grid after a partial was written
    used to surface as a bare `KeyError` at aggregation, hours in. It has to
    refuse by name instead, before anything else measures.
    """
    # Este test es del motor de rejilla, que sigue existiendo pero ya no es
    # el que `search_ceilings` despacha por defecto. Fijarlo acá dice qué
    # se está probando; sin esto el test pasaba a medir Optuna sin decirlo.
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(harness, "run_one", _fake_run_one({}))
    monkeypatch.setattr(harness.bags, "build", _fake_build())
    harness._write_partial("creda", "D", {(0, "M->U"): {0.05: 0.51}}, 1.0,
                           lambda *a: None)

    with pytest.raises(SystemExit) as raised:
        harness.search_ceilings(harness.Reduction(), torch.device("cpu"),
                                progress=lambda *a: None)

    message = str(raised.value)
    assert "creda" in message
    assert "0.05" in message
    assert str(config.CEILING_GRID) in message
    assert str(harness.shard_paths(None)["partial"]) in message
    assert "Delete" in message and "restore" in message


def test_the_stale_resume_refuses_before_it_measures_anything(
        tmp_path, monkeypatch) -> None:
    """Eagerness is the point: a stale key must refuse before a single
    further cell trains, not once some other family's grid has already run.
    """
    # Este test es del motor de rejilla, que sigue existiendo pero ya no es
    # el que `search_ceilings` despacha por defecto. Fijarlo acá dice qué
    # se está probando; sin esto el test pasaba a medir Optuna sin decirlo.
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")

    def _unreached(*args, **kwargs):
        pytest.fail("measured before refusing")

    monkeypatch.setattr(harness, "run_one", _unreached)
    monkeypatch.setattr(harness.bags, "build", _unreached)
    harness._write_partial("creda", "D", {(0, "M->U"): {0.05: 0.51}}, 1.0,
                           lambda *a: None)

    with pytest.raises(SystemExit):
        harness.search_ceilings(harness.Reduction(), torch.device("cpu"),
                                progress=lambda *a: None)


def test_write_partial_stamps_every_write(tmp_path, monkeypatch) -> None:
    """Defect (4): a partial is a ceiling record mid-measurement, and gets the
    same stamp `harness.sellar_techos` writes on a finished entry -- otherwise
    a partial predating this change (or written under a moved `KERNEL_SIGMA`)
    can be resumed and its cells silently spliced with ones measured under
    today's objective.
    """
    from MIL_CREDA_Benchmark import ceiling_record, harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    harness._write_partial("milcreda", "G", {(0, "M->U"): {1e-4: 0.5}}, 1.0,
                           lambda *a: None)
    stored = json.loads(harness._partial_path().read_text(encoding="utf-8"))
    assert ceiling_record.stamp_drift(stored) == {}


def test_search_ceilings_refuses_to_resume_a_stamp_drifted_partial(
        tmp_path, monkeypatch) -> None:
    """Defect (4): the grid engine's own resume path, checked before it reads
    a single cell out of the partial -- eagerness is the point, the same as
    the stale-grid-key refusal beside this test.

    Reachable red: `search_ceilings` calling `_read_partial` and resuming
    without checking `_partial_stamp_drift` first.
    """
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")

    def _unreached(*args, **kwargs):
        pytest.fail("measured before refusing on stamp drift")

    monkeypatch.setattr(harness, "run_one", _unreached)
    monkeypatch.setattr(harness.bags, "build", _unreached)

    # Write a partial the ordinary way, then move it out from under today's
    # config, the same shape a repository-committed leftover partial is in.
    harness._write_partial("creda", "D", {(0, "M->U"): {0.05: 0.51}}, 1.0,
                           lambda *a: None)
    path = harness._partial_path()
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["kernelSigma"] = config.KERNEL_SIGMA * 3
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        harness.search_ceilings(harness.Reduction(), torch.device("cpu"),
                                progress=lambda *a: None)
    assert str(path) in str(raised.value)


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


def test_two_shards_straddling_a_search_disagree_on_more_than_the_ceiling(
        tmp_path, monkeypatch) -> None:
    """The search moves six parameters, so the guard has to see all six.

    `shards.disagreements()` reads `identicalAcrossShards` entries with a flat
    `stamp.get(field)`, so the five dimensions beside the ceiling are only
    checkable if `write_shard_stamp` writes them flat and the declaration names
    them. Both halves are asserted here, because either alone passes while the
    guard checks nothing: a field written and not declared is never compared,
    and a field declared and not written compares `None` against `None` on every
    shard and agrees forever.

    Reachable red: drop `hyperByTransfer` from either side.
    """
    from MIL_CREDA_Benchmark import harness, shards

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    assert "hyperByTransfer" in shards.declaration()["identicalAcrossShards"]

    before = harness.Reduction()
    before.hyperByTransfer = {"milcreda": {"M-U": {"kernelSigma": 36.1}}}
    after = harness.Reduction()
    after.hyperByTransfer = {"milcreda": {"M-U": {"kernelSigma": 3.61}}}
    harness.write_shard_stamp("s00", before)
    harness.write_shard_stamp("s01", after)

    entries = [{"shard": name,
                "stamp": json.loads(harness.shard_paths(name)["stamp"].read_text()),
                "runs": []}
               for name in ("s00", "s01")]
    assert all("hyperByTransfer" in e["stamp"] for e in entries), (
        "the field is not a flat top-level key, so `disagreements()` cannot read it")
    fields = [d["field"] for d in shards.disagreements(
        entries, shards.declaration()["identicalAcrossShards"])]
    assert "hyperByTransfer" in fields, (
        "two shards trained under bandwidths an order of magnitude apart merged "
        f"without a word -> {fields}")


# --------------------------------------------------------------- shard evidence

def test_write_shard_stamp_carries_evidence_without_outputs(tmp_path, monkeypatch) -> None:
    """`outputs` cannot be known before the run finishes, so it is absent here.

    The other three fields can: the commit, the code digest, and where imports
    resolved from are all facts about the checkout at the moment the stamp is
    written, not about what the run produces.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    stored = json.loads(harness.shard_paths("alpha")["stamp"].read_text())

    evidence = stored["evidence"]
    assert evidence["commit"] and len(evidence["commit"]) == 40
    assert evidence["codeDigest"] and len(evidence["codeDigest"]) == 64
    assert evidence["importsFrom"].endswith("MIL_CREDA_Benchmark")
    assert "outputs" not in evidence


def test_seal_shard_stamp_adds_outputs_atomically(tmp_path, monkeypatch) -> None:
    """Sealing rewrites the stamp once the run's own files exist beside it."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    paths = harness.shard_paths("alpha")
    paths["runs"].parent.mkdir(parents=True, exist_ok=True)
    paths["runs"].write_text('{"seed": 0}\n', encoding="utf-8")

    sealed = harness.seal_shard_stamp("alpha")
    assert sealed == paths["stamp"]
    stored = json.loads(paths["stamp"].read_text())
    assert sorted(stored["evidence"]["outputs"]) == ["runs.jsonl", "shard.json"]
    # No leftover scratch file: the rename is the only trace of the rewrite.
    assert list(paths["stamp"].parent.glob("*.partial.json")) == []


def test_an_unsealed_shard_is_incomplete_and_a_sealed_one_is_not(
        tmp_path, monkeypatch) -> None:
    """The whole enforcement mechanism: a run that dies before sealing never
    lists `outputs`, so it is never complete, so it is never mergeable."""
    from MIL_CREDA_Benchmark import harness, shards

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    paths = harness.shard_paths("alpha")

    unsealed = json.loads(paths["stamp"].read_text())
    before = shards.completeness(unsealed, shards.REQUIRED_EVIDENCE)
    assert before["complete"] is False
    assert "evidence.outputs" in before["missing"]

    harness.seal_shard_stamp("alpha")
    sealed = json.loads(paths["stamp"].read_text())
    after = shards.completeness(sealed, shards.REQUIRED_EVIDENCE)
    assert after["complete"] is True
    assert after["missing"] == []


def test_a_concurrent_reader_during_seal_sees_the_pre_seal_stamp(
        tmp_path, monkeypatch) -> None:
    """Reachable red: sealing that mutated the stamp file in place, rather than
    rewriting a scratch copy and renaming it, would let a reader mid-write see a
    half-written JSON document instead of a whole, if older, one.

    `os.replace` is the boundary a concurrent reader can land either side of and
    never inside, so this test reads the file at the exact instant just before
    the rename and asserts it is still the complete, if incomplete-as-evidence,
    pre-seal stamp — correct, and worth this explicit test rather than an
    assumption, per the design's own open question.
    """
    import os as _os

    from MIL_CREDA_Benchmark import harness, shards

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    paths = harness.shard_paths("alpha")
    paths["runs"].write_text('{"seed": 0}\n', encoding="utf-8")

    seen_during_window = {}
    real_replace = _os.replace

    def watched_replace(src, dst):
        # The exact window: the scratch file is fully written, the real stamp
        # has not moved yet. A reader landing here sees only the old bytes.
        seen_during_window["stamp"] = json.loads(paths["stamp"].read_text())
        real_replace(src, dst)

    monkeypatch.setattr(_os, "replace", watched_replace)
    harness.seal_shard_stamp("alpha")

    mid_seal = shards.completeness(seen_during_window["stamp"], shards.REQUIRED_EVIDENCE)
    assert mid_seal["complete"] is False
    assert "evidence.outputs" in mid_seal["missing"]

    post_seal = shards.completeness(
        json.loads(paths["stamp"].read_text()), shards.REQUIRED_EVIDENCE)
    assert post_seal["complete"] is True


def test_a_checkout_with_no_git_history_never_produces_a_complete_shard(
        tmp_path, monkeypatch) -> None:
    """`evidence.commit` requires git in the runtime that stamps. A checkout
    with no git history — `.git` stripped, a source tarball — can never
    produce a complete shard. Accepted as correct; a behavior change for any
    non-git test fixture, hence flagged with its own explicit test."""
    from MIL_CREDA_Benchmark import harness, shards

    no_git = tmp_path / "no-git-checkout"
    no_git.mkdir()
    monkeypatch.setattr(config, "REPOSITORY", no_git)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")

    harness.write_shard_stamp("alpha", harness.Reduction())
    stored = json.loads(harness.shard_paths("alpha")["stamp"].read_text())

    assert "commit" not in stored["evidence"]
    result = shards.completeness(stored, shards.REQUIRED_EVIDENCE)
    assert result["complete"] is False
    assert "evidence.commit" in result["missing"]


# ---------------------------------------------------------------------------
# The ceiling in force on one transfer, which is two readings and not one.


def test_a_measured_transfer_keeps_its_own_pick_over_the_pooled_one() -> None:
    """The whole point of separating the two readings.

    Inverted, this is the failure the change exists to prevent: the pooled
    winner applied to a transfer the search actually measured and disagreed on.
    That failure is silent — the run proceeds and prints a number — so the only
    thing that catches it is asking the resolver directly.
    """
    from MIL_CREDA_Benchmark import harness

    reduction = harness.Reduction(
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
    )
    assert harness.ceiling_for(reduction, "milcreda", ("S", "M")) == 1e-4
    # and the pooled winner still governs everywhere it was never measured
    assert harness.ceiling_for(reduction, "milcreda", ("M", "U")) == 1e-2
    assert harness.ceiling_for(reduction, "milcreda", ("U", "S")) == 1e-2


def test_a_family_with_no_per_transfer_picks_falls_back_everywhere() -> None:
    """A record written before the key existed means what it meant then.

    An absent mapping is not a defect to refuse: it is a record whose every
    transfer ran at the pooled winner, and reading it as anything else would
    invent a distinction that run never made.
    """
    from MIL_CREDA_Benchmark import harness

    reduction = harness.Reduction(ceilings={"creda": 1e-4}, ceilingsByTransfer={})
    for transfer in config.TRANSFERS:
        assert harness.ceiling_for(reduction, "creda", transfer) == 1e-4


def test_a_floor_gets_the_neutral_and_never_a_families_ceiling() -> None:
    """An arm with no adaptation term has no ceiling to inherit."""
    from MIL_CREDA_Benchmark import harness

    reduction = harness.Reduction(
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
    )
    assert harness.ceiling_for(reduction, None, ("S", "M")) == config.RAMP_CEILING


def test_ceiling_for_refuses_when_the_attached_record_carries_stale_stamp() -> None:
    """`campaign()` always attaches the whole record as `ceilingSearch` before
    any run starts; `ceiling_for` is the backstop that reads it directly.

    Mutation this catches: `ceiling_for` accepting a stale-stamped ceiling
    would run a real campaign silently under a different `KERNEL_SIGMA`.
    """
    from MIL_CREDA_Benchmark import harness

    reduction = harness.Reduction(
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
        ceilingSearch={"milcreda": {"ceiling": 1e-2,
                                    "kernelSigma": config.KERNEL_SIGMA * 3}},
    )
    with pytest.raises(SystemExit):
        harness.ceiling_for(reduction, "milcreda", ("S", "M"))


def test_ceiling_for_refuses_on_an_entry_complete_except_a_revision_mismatch() -> None:
    """`kernelSigma`/`attentionGamma`/`attentionTemperature` are no longer a
    fixed backdrop the search runs under -- it explores all three itself
    (alongside `rampDelta`/`tauLocal`), per transfer, so an entry's own value
    is that search's winner and not a value `ceiling_for` compares against
    `config`. Only `revision` is still a genuine backdrop
    (`ceiling_record.STAMP_FIELDS`), so this is the one field left whose
    mismatch this refusal can still be measured against.

    Reachable red: `ceiling_for` (via `ceiling_record.stamp_drift`) ignoring
    a `revision` mismatch.
    """
    from MIL_CREDA_Benchmark import ceiling_record, harness

    entry = ceiling_record.stamp({"ceiling": 1e-2})
    entry["revision"] = "research-concept-r99.md"
    reduction = harness.Reduction(
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
        ceilingSearch={"milcreda": entry},
    )
    with pytest.raises(SystemExit):
        harness.ceiling_for(reduction, "milcreda", ("S", "M"))


def test_ceiling_for_accepts_a_correctly_stamped_attached_record() -> None:
    """The positive path beside the refusal above: a `ceilingSearch` entry
    stamped under the current config never refuses."""
    from MIL_CREDA_Benchmark import ceiling_record, harness

    reduction = harness.Reduction(
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
        ceilingSearch={"milcreda": ceiling_record.stamp({"ceiling": 1e-2})},
    )
    assert harness.ceiling_for(reduction, "milcreda", ("S", "M")) == 1e-4


def test_config_ceilings_on_record_refuses_a_stamp_mismatch_at_an_explicit_scale(
        tmp_path, monkeypatch) -> None:
    """The bypass a notebook actually reached: `Benchmark_Campaign_v1.ipynb`'s
    `ES_ENSAYO` branch and `Benchmark_Noise_Sweep.ipynb` both build
    `ceilings=`/`ceilingsByTransfer=` from `config.ceilings_on_record(pilot=...)`
    directly, never through `harness.with_ceilings_in_force` -- so
    `ceilings_in_force`'s own refusal never ran. The stamp check has to live
    where these two functions actually read the file.
    """
    # `kernelSigma`/`attentionGamma`/`attentionTemperature` no longer drift:
    # the search explores all three itself, so only `revision` -- the one
    # remaining member of `ceiling_record.STAMP_FIELDS` -- can still trigger
    # this refusal.
    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "milcreda": {"ceiling": 1e-2, "revision": "research-concept-r99.md"},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    with pytest.raises(SystemExit):
        config.ceilings_on_record(pilot=False)


def test_config_ceilings_by_transfer_on_record_refuses_a_stamp_mismatch(
        tmp_path, monkeypatch) -> None:
    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "milcreda": {"ceiling": 1e-2, "byTransfer": {"S->M": 1e-4},
                     "revision": "research-concept-r99.md"},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    with pytest.raises(SystemExit):
        config.ceilings_by_transfer_on_record(pilot=False)


def test_config_ceilings_on_record_accepts_a_correctly_stamped_entry(
        tmp_path, monkeypatch) -> None:
    """The positive path: a record stamped under the CURRENT config is read
    normally, at an explicit scale, and returns the plain `family -> ceiling`
    mapping this function has always returned."""
    from MIL_CREDA_Benchmark import ceiling_record

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "milcreda": ceiling_record.stamp({"ceiling": 1e-2}),
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    assert config.ceilings_on_record(pilot=False) == {"milcreda": 1e-2}


def test_config_ceilings_on_record_does_not_refuse_at_the_default_pilot_none(
        tmp_path, monkeypatch) -> None:
    """`pilot=None` -- the *vigente* resolver reading `CEILINGS.update(...)`
    uses at import time -- never refuses on a stale stamp: refusing there
    would make importing `config` itself depend on a ceiling record's
    freshness, which is a decision no repository should be able to make for
    every one of its readers by leaving a stale file on disk.
    `campaign()`'s own up-front check is what actually stands between a
    stale default and a run.
    """
    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "milcreda": {"ceiling": 1e-2},  # no stamp at all
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "nope.json")
    assert config.ceilings_on_record() == {"milcreda": 1e-2}


def _synthetic_bagset(domain: str) -> object:
    """A `BagSet` with the shape the harness reads and none of the material.

    Built rather than downloaded so this test stays offline like the rest of the
    file. Nothing here is trained on: the run is stopped at the moment the
    ceiling is resolved, which is the only thing being checked.
    """
    from MIL_CREDA_Benchmark import bags

    bagcount, per_bag = 12, 2
    return bags.BagSet(
        domain=domain,
        images=torch.zeros(bagcount * per_bag, 3, 32, 32),
        members=torch.arange(bagcount * per_bag).reshape(bagcount, per_bag),
        labels=torch.arange(bagcount) % config.CLASSES,
        train_idx=torch.arange(0, 6),
        valid_idx=torch.arange(6, 9),
        eval_idx=torch.arange(9, 12),
        manifest={},
    )


class _StopAtCeiling(Exception):
    """Raised from the `ramp` spy so nothing past the lookup ever runs."""


def test_run_one_resolves_the_ceiling_of_the_transfer_it_was_given(monkeypatch) -> None:
    """The resolver reaching the training loop, not merely existing beside it.

    Testing `ceiling_for` alone verifies the lookup and never the wiring, and the
    wiring is the half that decides what the campaign computes: a `run_one` still
    holding the old family-only expression would leave every test above green
    while the two measured transfers ran at the pooled winner.

    So this reads the coefficient out of `ramp` exactly as `run_one` calls it, and
    stops there. `wiring.build` is stubbed because constructing the real backbone
    would reach for pretrained weights, and what is being checked is which number
    arrives at the ramp — not what the model does with it afterwards.
    """
    from MIL_CREDA_Benchmark import harness, wiring

    seen: list[float] = []

    def spy(epoch, epochs, family, ceiling, delta=None):
        seen.append(ceiling)
        raise _StopAtCeiling

    monkeypatch.setattr(harness, "ramp", spy)
    monkeypatch.setattr(wiring, "build", lambda *a, **k: torch.nn.Linear(1, 1))

    reduction = harness.Reduction(
        epochs=1, seeds=[0],
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
    )
    material = {"source": _synthetic_bagset("S"), "target": _synthetic_bagset("M")}
    device = torch.device("cpu")

    for transfer, expected in ((("S", "M"), 1e-4), (("M", "U"), 1e-2)):
        seen.clear()
        with pytest.raises(_StopAtCeiling):
            harness.run_one("G", transfer, 0, reduction, device, material,
                            role="valid")
        assert seen == [expected], (transfer, seen)


def test_run_one_resolves_the_hyperparameters_of_the_transfer_it_was_given(
) -> None:
    """The other five searched dimensions, reaching `wiring.build` the same
    way the ceiling reaches `ramp` in the test above.

    Before this stretch's build, a record with non-default winners changed
    NOTHING: `search_ceilings_trials` explored `rampDelta`/`kernelSigma`/
    `attentionGamma`/`attentionTemperature`/`tauLocal` and recorded a winner
    per transfer, but `campaign()` never passed `hyper=` to `run_one`, so
    every arm trained at the bare `config` constant regardless of what the
    search found (`harness.hyper_for`'s own docstring names this
    explicitly). This is the wiring that closes it: two transfers with
    different winners in `reduction.hyperByTransfer` must reach
    `wiring.build` with two different `hyper` dicts.

    Reachable red (the mutation this test exists to catch): revert `run_one`
    to `if hyper is None: hyper = hyper` (a no-op) instead of
    `hyper = hyper_for(reduction, family, transfer)`, and both transfers
    below arrive at `wiring.build` with the bare `config`-default `hyper` --
    the two captured dicts stop differing and this test goes red.
    """
    from MIL_CREDA_Benchmark import harness, wiring

    captured: list[dict] = []

    def _spy(arm_id, classes, source, target, hyper=None):
        captured.append(hyper)
        raise _StopAtCeiling

    monkeypatch = pytest.MonkeyPatch()
    try:
        # `wiring.build` is `run_one`'s very first call after resolving
        # `hyper`, so raising from the spy itself captures the resolved
        # value and stops before anything trains -- no need to also stub
        # `ramp`, which `run_one` never reaches.
        monkeypatch.setattr(wiring, "build", _spy)

        winner_sm = {"rampDelta": 55.0, "kernelSigma": 12.5,
                    "attentionGamma": 0.4, "attentionTemperature": 3.3,
                    "tauLocal": 0.7}
        reduction = harness.Reduction(
            epochs=1, seeds=[0],
            ceilings={"milcreda": 1e-2},
            ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
            hyperByTransfer={"milcreda": {"S->M": winner_sm}},
        )
        material = {"source": _synthetic_bagset("S"), "target": _synthetic_bagset("M")}
        device = torch.device("cpu")

        # The searched transfer: `wiring.build` must receive exactly the
        # winner attached above, not the bare `config` defaults.
        with pytest.raises(_StopAtCeiling):
            harness.run_one("G", ("S", "M"), 0, reduction, device, material,
                            role="valid")
        assert captured[-1] == winner_sm, captured[-1]

        # A transfer the search never saw: falls back to `reduction`'s own
        # scalar fields, `config`'s bare declared defaults here since none
        # were overridden at construction -- never the OTHER transfer's
        # winner.
        with pytest.raises(_StopAtCeiling):
            harness.run_one("G", ("M", "U"), 0, reduction, device, material,
                            role="valid")
        assert captured[-1] != winner_sm, captured[-1]
        assert captured[-1] == {
            "rampDelta": config.RAMP_DELTA, "kernelSigma": config.KERNEL_SIGMA,
            "attentionGamma": config.ATTENTION_GAMMA,
            "attentionTemperature": config.ATTENTION_TEMPERATURE,
            "tauLocal": config.TAU_LOCAL,
        }
    finally:
        monkeypatch.undo()


def test_the_campaign_refuses_up_front_on_a_stale_ceiling_record_before_writing_anything(
        tmp_path, monkeypatch) -> None:
    """Defect (3): `campaign()` used to truncate `runs.jsonl` (opened `"w"`)
    and train arm B before `run_one(E)` reached `ceiling_for`'s own stamp
    check -- a stamp mismatch was discovered only after real work and a real
    write already happened. The check now runs before any mkdir, open,
    truncation or training.

    Proven by planting `runs.jsonl` with sentinel bytes from an unrelated
    earlier run and a `run_one` that fails the test if it is ever called:
    if the refusal fired late, the sentinel bytes would already be gone
    (`open("w")` truncates on open, before a single line is written) and
    `run_one` would have been called on arm B, the first declared arm.
    """
    from MIL_CREDA_Benchmark import ceiling_record, harness

    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models" / "Benchmark")
    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        # No stamp at all: predates `ceiling_record.stamp`, the exact shape
        # this repository's own `ceilings.pilot.json` is in today.
        "milcreda": {"ceiling": 1e-2, "byTransfer": {"S->M": 1e-4}},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "no-pilot-record.json")

    runs_path = config.results_for(0.0, "campaign", False) / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = b'{"arm": "sentinel-from-an-earlier-run"}\n'
    runs_path.write_bytes(sentinel)

    def run_one_must_not_be_called(*args, **kwargs):
        pytest.fail("run_one was called: the up-front stamp check did not "
                    "stop the campaign before training started")

    monkeypatch.setattr(harness, "run_one", run_one_must_not_be_called)

    reduction = harness.Reduction(ceilings={"milcreda": 1e-2},
                                  ceilingsByTransfer={"milcreda": {"S->M": 1e-4}})
    with pytest.raises(SystemExit):
        harness.campaign(reduction, torch.device("cpu"), progress=lambda *a: None)

    assert runs_path.read_bytes() == sentinel, (
        "runs.jsonl was truncated or written to before the stamp check refused"
    )
    assert not (tmp_path / "Models" / "Benchmark").exists(), (
        "models directory was created before the stamp check refused"
    )


def test_the_campaign_refuses_when_the_record_has_picks_and_the_run_does_not(
        tmp_path, monkeypatch) -> None:
    """The stale-field case, refused by name instead of running the old rule.

    A caller that sets `ceilings=` alone leaves `ceilingsByTransfer` holding
    whatever `config` was imported with. Every transfer then falls back to the
    pooled winner and the run looks entirely ordinary, which is why this has to
    refuse rather than warn.
    """
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "milcreda": _stamped({"ceiling": 1e-2, "byTransfer": {"S->M": 1e-4}}),
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    # `campaign()` hace `results_for(...).mkdir()` y `models_for(...).mkdir()`
    # ANTES de cualquiera de sus tres rechazos, así que un test que sólo
    # redirige el registro igual crea dos directorios en el árbol del dueño.
    # Dos raíces y no una: `MODELS` es hermana de `RESULTS` y no se deriva de
    # ella, así que redirigir una deja la otra apuntando a la corrida real.
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models" / "Benchmark")

    stale = harness.Reduction(ceilings={"milcreda": 1e-2}, ceilingsByTransfer={})
    with pytest.raises(SystemExit) as raised:
        harness.campaign(stale, torch.device("cpu"), arms=["G"])
    assert "per-transfer ceilings left behind" in str(raised.value)
    assert "milcreda" in str(raised.value)


def test_the_ceilings_are_what_has_to_agree_across_shards() -> None:
    """Both readings, and both reachable by a flat top-level lookup.

    `disagreements()` compares `stamp.get(field)`, so a declared name that is
    not a flat key on the stamp resolves to `None` on every shard and passes
    whatever the shards actually did — a guarantee that was never enforced.
    """
    declared = MIL_CREDA_Benchmark.__benchmark__["distribution"]["identicalAcrossShards"]
    assert "ceilings" in declared
    assert "ceilingsByTransfer" in declared


def test_the_per_transfer_conclusion_can_come_out_different() -> None:
    """A conclusion tied to nothing is a conclusion measuring nothing.

    The same reading the report contract applies to every other conclusion: run
    it over two records that differ in the one fact it is about, and the two
    texts must differ.

    And the two texts are not interchangeable prose: they are the two readings
    the family's average across transfers can be. Once a measured transfer picks
    a ceiling of its own the family stops running at one coefficient, so the row
    averaged over transfers mixes scalars that differ -- and the report has to say that
    is what it is giving, rather than leaving the reader to assume the single
    coefficient that no longer exists. When none of them departs, separating the
    two readings changed no number at all, and saying so is more honest than a
    mark suggesting it did.

    Differing texts alone would pass with the two swapped, which is the failure
    that matters here: the report would be announcing the mixture exactly where
    there is none.
    """
    from MIL_CREDA_Benchmark import tables

    transfers = ["M->U", "S->M", "U->S"]
    agrees = {"f": {"ceiling": 1e-2, "byTransfer": {"M->U": 1e-2, "S->M": 1e-2}}}
    differs = {"f": {"ceiling": 1e-2, "byTransfer": {"M->U": 1e-2, "S->M": 1e-4}}}

    said_agrees = tables.conclusion_ceilings_by_transfer(agrees, transfers)
    said_differs = tables.conclusion_ceilings_by_transfer(differs, transfers)
    assert said_agrees != said_differs
    assert "S->M" in said_differs and "S->M" not in said_agrees

    # which of the two readings the row is, named in the text and not inferred
    assert "mezcla escalares distintos" in said_differs
    assert "mezcla escalares distintos" not in said_agrees
    assert "no cambió ningún techo" in said_agrees
    assert "no cambió ningún techo" not in said_differs

    # and within a transfer every arm still shares the ceiling, which is what
    # keeps each rung attributable -- said in the same breath as the mixture
    assert "dentro de cada transferencia" in said_differs


def test_the_per_transfer_table_marks_measured_apart_from_inherited() -> None:
    """Without the mark all six cells read as if all six had been measured."""
    from MIL_CREDA_Benchmark import tables

    record = {"f": {"ceiling": 1e-2, "byTransfer": {"S->M": 1e-4}}}
    rendered = tables.render_ceilings_by_transfer(
        record, ["M->U", "S->M"], markdown=True)
    assert "**0.0001**" in rendered          # measured on S->M
    assert "| 0.01 |" in rendered            # inherited on M->U, unmarked


def test_the_per_transfer_ceilings_reach_the_written_report() -> None:
    """Declared in the contract, so the report is checked against it."""
    report = MIL_CREDA_Benchmark.__benchmark__["report"]
    assert "tables.render_ceilings_by_transfer" in report["renderers"]
    assert "tables.conclusion_ceilings_by_transfer" in report["conclusions"]
