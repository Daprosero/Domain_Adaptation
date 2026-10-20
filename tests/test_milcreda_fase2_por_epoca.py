"""MIL-CREDA's phase-2 criterion, in EPOCHS, and the oracle series beside it.

Two claims, one file, because they share a fixture and a loop: the stop reads
the watchman once per PASS of the training role instead of once per step, and
the run records what the target's test role read at every step so the oracle
column of the per-family table has something to be a maximum of.

The habits this file keeps, both of them this repository's own:

- **Never an expression against itself.** Every derivation is recomputed from
  the two ends it joins -- `CREDA.toy_experiment`'s own constants on one side
  and this module's declared programme on the other -- so a literal written in
  this file cannot satisfy it and neither side can move alone.
- **The join is crossed through the CALL SITE.** `steps_of_one_epoch` could
  return an impeccable number while the loop read the criterion somewhere else,
  so the pass property and the criterion's cadence are read off what
  `adapt_phase_two` RETURNS, never off the helper.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math

import pytest
import torch

from CREDA import toy_experiment as creda
from MIL_CREDA_Benchmark import bags, toy_run

DEVICE = torch.device("cpu")
ADAPTED = toy_run.TOY_ADAPTED


@pytest.fixture(scope="module")
def tiny() -> toy_run.Scale:
    """`PILOT` with its lengths and repetitions lowered, and its phase-2 budget
    kept at the pilot's own, because below that budget the criterion refuses.

    The budget is NOT lowered here, and that is the cost of putting the
    criterion in epochs: an epoch of this programme is
    `STEPS_PER_EPOCH_DECLARED` steps whatever the scale, since the bag
    programme is not a repetition field, so a budget holding fewer than
    `MIN_EPOCHS_2` of them is one in which the criterion can never choose at
    all. `adapt_phase_two` refuses it rather than running something that never
    restores.
    """
    return dataclasses.replace(
        toy_run.PILOT, name="test", epochs_1=2, patience_1=1,
        epochs_2=toy_run.MIN_EPOCHS_2 + 1, trials_1=1, trials_2=1,
        points_per_domain=810, instances_per_bag=6)


@pytest.fixture(scope="module")
def familia(tiny) -> object:
    return toy_run.build_family("lunas", 0, tiny)


def _prepared(familia, tiny, seed=0):
    """A phase-1-trained arm with theta_t copied: the state phase 2 starts at."""
    material = toy_run.material_for(familia, seed, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, source_valid, _ = bags.roles(material["source"])
        _, _, target_eval = bags.roles(material["target"])
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, seed)
        generator = torch.Generator().manual_seed(seed + 9973)
        toy_run.train_phase_one(model, source_train, source_valid, DEVICE,
                                tiny, hyper, generator)
        toy_run.copy_source_to_target(model)
    return {"model": model, "material": material, "hyper": hyper,
            "source_train": source_train, "source_valid": source_valid,
            "target_eval": target_eval}


def _run(payload, familia, tiny, **kwargs):
    with toy_run.material_bound(familia, tiny, "mlp"):
        generator = torch.Generator().manual_seed(1)
        return toy_run.adapt_phase_two(
            payload["model"], payload["source_train"], payload["source_valid"],
            DEVICE, tiny, payload["hyper"], generator, **kwargs)


# ======================================================== 1. what an epoch is


def test_the_epoch_is_the_pass_balanced_batches_actually_draws(familia, tiny):
    """`steps_of_one_epoch` is the step at which every training bag has come
    out exactly once, and that is MEASURED off the sampler rather than derived
    twice.

    The join this crosses is the only one worth crossing here: the number is
    computed from the role's class counts in one place and the draw is
    performed by `harness.balanced_batches` in another, and asserting each
    against an arithmetic written in this file would verify both halves and
    never the connection.
    """
    from MIL_CREDA_Benchmark import harness

    material = toy_run.material_for(familia, 0, tiny, "mlp")
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, _, _ = bags.roles(material["source"])
    declared = toy_run.steps_of_one_epoch(source_train.targets)

    generator = torch.Generator().manual_seed(0)
    drawn: set[int] = set()
    reached = None
    for step, batch in enumerate(
            harness.balanced_batches(source_train.targets, declared * 3,
                                     generator)):
        drawn.update(batch)
        if reached is None and len(drawn) == len(source_train.targets):
            reached = step + 1
    assert reached == declared, (reached, declared, len(source_train.targets))
    # The pole that the reading is about the PASS and not about the role's size:
    # one bag of every class leaves the step count at the PER-CLASS count, not
    # at the total, so a sampler drawing one bag per step would answer 48.
    assert declared < len(source_train.targets), (
        declared, len(source_train.targets))


def test_the_declared_epoch_is_the_programmes_own_and_equal_at_both_scales(
        familia, tiny):
    """`STEPS_PER_EPOCH_DECLARED` anchors the criterion on the DECLARED
    programme, which is CREDA's own choice of anchor, and it comes out the same
    at every scale because the bag programme is not a repetition field.

    CREDA anchors on `BLOQUES_AL_LOTE_DECLARADO` -- the module's declared lot
    over its declared train role -- precisely so the criterion's resolution
    cannot be decided by a searched hyperparameter. Here the lot is not searched
    at all (`harness.balanced_batches` takes none), so the same anchor is the
    declared bag programme: `realized_split()["bagsPerClass"]["train"]`.
    """
    for scale in (toy_run.FULL, toy_run.PILOT, tiny):
        per_class = scale.realized_split()["bagsPerClass"]["train"]
        assert per_class == toy_run.STEPS_PER_EPOCH_DECLARED, (
            scale.name, per_class, toy_run.STEPS_PER_EPOCH_DECLARED)
    # And measured against the sampler at the one scale a run is cheap at, so
    # the declared number is not merely consistent with itself.
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, _, _ = bags.roles(material["source"])
    assert (toy_run.steps_of_one_epoch(source_train.targets)
            == toy_run.STEPS_PER_EPOCH_DECLARED)


def test_the_epoch_length_refuses_when_the_classes_do_not_agree():
    """A role whose classes hold different numbers of bags has NO step at which
    every bag has come out exactly once, and this refuses rather than returning
    the first or the largest.

    CREDA's `pasos_de_una_epoca` refuses the same way when its two domains
    disagree in blocks, and for the same reason: wrapping the short queue would
    leave the property the whole change exists for -- every bag out exactly once
    per epoch -- false with nothing failing.

    The refusal is REACHABLE and that is what makes it a guard: `balanced_batches`
    refills each class's queue independently, so an unequal role runs happily
    and silently forever.
    """
    balanced = [0] * 4 + [1] * 4 + [2] * 4
    assert toy_run.steps_of_one_epoch(balanced) == 4
    with pytest.raises(ValueError, match="do not agree"):
        toy_run.steps_of_one_epoch([0] * 4 + [1] * 3 + [2] * 4)


# ============================================== 2. where each number comes from


def test_the_minimum_is_declared_and_its_input_is_gone():
    """`MIN_EPOCHS_2` was DERIVED from `min_steps_2` and is now DECLARED.

    It was `ceil(40 / 16) = 3` passes. That input was retired with the step
    ceiling on 2026-10-19, so the derivation would now read a number nobody
    maintains -- which reads as derived and is not. The VALUE did not move: 3
    is what the derivation gave, and changing it in the same step as declaring
    it would mix two things.

    The pole is the absence: `min_steps_2` must be GONE from the scale, on
    both sides. Without that half, reintroducing it on one side alone would
    pass -- and the two packages drifting apart on this field is exactly what
    the cross-package pin exists to prevent.
    """
    assert toy_run.MIN_EPOCHS_2 == 3, toy_run.MIN_EPOCHS_2
    assert not hasattr(toy_run.FULL, "min_steps_2")
    assert not hasattr(creda, "MIN_PASOS_2")

    criterion = toy_run.PHASE_TWO_CRITERION
    assert criterion["minEpochs2"] == toy_run.MIN_EPOCHS_2
    assert criterion["minEpochs2Is"] == "declared"
    assert "retired" in criterion["minEpochs2Was"]

    # CREDA's own minimum DIFFERS and the record PUBLISHES both, which is the
    # replacement for asserting they are equal. They came from the same
    # 40-step floor over different passes (16 steps here, 9 there); declaring
    # them did not reconcile the two numbers, because reconciling them is a
    # decision about the guard and not about the unit.
    assert criterion["minEpochs2InCreda"] == creda.MIN_EPOCAS_2
    assert toy_run.MIN_EPOCHS_2 != creda.MIN_EPOCAS_2, (
        toy_run.MIN_EPOCHS_2, creda.MIN_EPOCAS_2)
    assert "decision about the guard" in criterion["minEpochs2WhyTheyDiffer"]
    # And the record says the guard BITES, because a guard the criterion lands
    # on two times in five is deciding rather than guarding.
    assert "38" in criterion["minEpochs2Bites"]


def test_the_patience_is_retired_on_both_sides_and_phase_one_still_stops():
    """`PATIENCE_2_EPOCHS` is GONE, and so is CREDA's `PACIENCIA_2_EPOCAS`.

    This test asserted the opposite until 2026-10-19: that the patience was
    the operator's declared 5, identical on both sides, and NOT a derivation
    of a step count. The premise died -- phase 2 has no stopping rule, it runs
    its `epochs_2` ceiling whole and the criterion chooses the best epoch of
    the whole run.

    It is inverted rather than deleted. A file without it cannot tell "the
    patience was retired" from "nobody ever tested it", and what breaks for
    real is somebody reintroducing it on one side only.

    What the measurement said: over the 90 runs of `Results/Toyset/full`,
    every one died by patience, NONE reached the ceiling, and 38 chose exactly
    the first legal step.
    """
    for retired in ("PATIENCE_2_EPOCHS",):
        assert not hasattr(toy_run, retired), retired
    for retired in ("PACIENCIA_2_EPOCAS", "PACIENCIA_2"):
        assert not hasattr(creda, retired), retired
    assert not hasattr(toy_run.FULL, "patience_2")

    criterion = toy_run.PHASE_TWO_CRITERION
    assert criterion["patience"] is None
    assert "NO stopping rule" in criterion["patienceWhy"]

    # THE OTHER HALF, because the absences above read as "neither phase
    # stops" and that is false. Phase 1 DOES stop, its signal is supervised,
    # and the record says so rather than leaving one rule to cover both.
    assert toy_run.FULL.patience_1 == creda.PACIENCIA_1 == 30
    assert "phase 1 DOES stop" in criterion["phaseOneStillStops"]


def test_the_budget_is_an_epoch_ceiling_and_the_record_says_why_it_inverted():
    """`epochs_2` is an EPOCH ceiling at both scales, and so is CREDA's.

    This test asserted the opposite until 2026-10-19: that `steps_2` stayed a
    STEP ceiling "exactly as `PASOS_2` stayed one in CREDA", on the grounds
    that every frame schedule and checkpoint grid of this module is indexed in
    steps. The grounds were backwards. A step ceiling lets the LOT decide how
    many passes the experiment runs, and in CREDA the lot is SEARCHED --
    measured there, 200 steps bought 22.2 passes for `lunas` at `LOTE_2=128`
    and 11.8 for `toroide` and `rollo` at 64, so its global table compared
    families trained half as long as one another.

    The frames stay indexed in STEPS and that is not an oversight: a frame is
    a state of the weights, and the weights move once per step.
    """
    assert toy_run.FULL.epochs_2 == creda.EPOCAS_2 == 100
    assert not hasattr(toy_run.FULL, "steps_2")
    assert not hasattr(creda, "PASOS_2")

    criterion = toy_run.PHASE_TWO_CRITERION
    assert criterion["budgetUnit"] == "epoch"
    assert "INVERTS" in criterion["budgetWhy"]
    assert "SEARCHED" in criterion["budgetWhy"]
    # The measurement that forced it, carried in the record so a reader meets
    # it instead of having to find this test.
    medido = criterion["budgetChangeMeasured"]["inCredaAt200Steps"]
    assert "22.2" in medido["lunas"] and "11.8" in medido["toroide"]

    published = criterion["atTheFullScale"]
    assert published["epochs2"] == toy_run.FULL.epochs_2
    assert published["stepsItHolds"] == (toy_run.FULL.epochs_2
                                         * toy_run.STEPS_PER_EPOCH_DECLARED)
    # ZERO, and asserted as a number rather than as an absent key: it was 8 at
    # the step ceiling, and a reader of two artefacts has to see it go to zero
    # rather than vanish.
    assert published["stepsWithoutARead"] == 0
    assert "by construction" in published["note"]


def test_the_flags_of_every_run_publish_the_criterion_and_the_series(tiny):
    """Both declarations travel in `banderas.json`, because a table that does
    not say what it was measured under cannot be read.

    "Which step this run chose" means nothing without what a step is counted
    against, and the oracle column means nothing without the boundary that
    keeps a target label out of every criterion. Neither is derivable from the
    numbers beside them, so both are written where a reader of the artefact
    meets them.
    """
    flags = toy_run.flags(tiny, DEVICE, ("lunas",),
                          [{"secondsPerPhase1Epoch": 1.0,
                            "secondsPerPhase2Step": 1.0,
                            "instancesPerBag": tiny.instances_per_bag}], 1.0)
    assert flags["phaseTwoCriterion"] == toy_run.PHASE_TWO_CRITERION
    assert flags["targetSeries"] == toy_run.TARGET_SERIES
    # The two the reader actually acts on, named rather than taken on trust
    # that the whole dict arrived.
    assert flags["phaseTwoCriterion"]["unit"] == "epoch"
    assert flags["phaseTwoCriterion"]["budgetUnit"] == "epoch"
    assert flags["targetSeries"]["role"] == "test"


def test_the_record_publishes_the_asymmetry_between_the_two_epoch_lengths():
    """The two protocols' epochs differ, so one declared patience costs a
    different share of each budget -- and that comes from the material rather
    than from a choice, so the reader has to be able to see it.

    Both halves are recomputed here from each module's own constants: nothing
    in this file and nothing in the record is a transcribed number.
    """
    asymmetry = toy_run.PHASE_TWO_CRITERION["asymmetryAgainstCreda"]
    assert asymmetry["stepsPerEpochHere"] == toy_run.STEPS_PER_EPOCH_DECLARED
    assert asymmetry["stepsPerEpochInCreda"] == creda.BLOQUES_AL_LOTE_DECLARADO
    assert asymmetry["stepsPerEpochHere"] != asymmetry["stepsPerEpochInCreda"]
    # The two budgets are now the SAME NUMBER of passes, which is the whole
    # point of the unit. What a pass COSTS still differs, because the material
    # differs -- and that difference is material rather than protocol, which
    # is what the asymmetry block is for.
    assert asymmetry["epochsInTheBudgetHere"] == toy_run.FULL.epochs_2
    assert asymmetry["epochsInTheBudgetInCreda"] == creda.EPOCAS_2
    assert (asymmetry["epochsInTheBudgetHere"]
            == asymmetry["epochsInTheBudgetInCreda"])
    assert "material rather than protocol" in asymmetry["reading"]


def test_the_criterion_is_read_only_at_the_end_of_an_epoch(familia, tiny,
                                                           monkeypatch):
    """The step phase 2 keeps is an epoch end, and the number of times the
    criterion could have chosen is the number of complete passes.

    Scripted per EPOCH so the argmin, the argmax and the last legal epoch are
    three different passes -- asserted before the run, because a sequence whose
    argmin and argmax coincide cannot tell a minimization from a maximization.
    Read the criterion every step instead and `bestStep` stops being a multiple
    of the pass: red.
    """
    payload = _prepared(familia, tiny)
    per_epoch = [0.90, 0.80, 0.20, 0.95, 0.60, 0.70, 0.75, 0.78, 0.79]
    legal = per_epoch[toy_run.MIN_EPOCHS_2 - 1:]
    assert len(per_epoch) >= tiny.epochs_2
    argmin = per_epoch.index(min(legal))
    argmax = per_epoch.index(max(legal))
    assert argmin != argmax, "the scripted sequence cannot distinguish min from max"
    assert argmin != len(per_epoch) - 1, "the argmin is the last epoch; useless"

    calls = {"n": 0}

    def scripted(model_, dataset, device):
        epoch = min(calls["n"] // toy_run.STEPS_PER_EPOCH_DECLARED,
                    len(per_epoch) - 1)
        calls["n"] += 1
        return per_epoch[epoch]

    monkeypatch.setattr(toy_run, "watchman", scripted)
    phase_two = _run(payload, familia, tiny)

    assert calls["n"] > 0, "the scripted watchman was never called"
    assert phase_two["stepsPerEpoch"] == toy_run.STEPS_PER_EPOCH_DECLARED
    assert phase_two["bestWatchman"] == min(legal)
    assert phase_two["bestEpoch"] == argmin + 1
    assert phase_two["bestStep"] == (argmin + 1) * phase_two["stepsPerEpoch"]
    assert phase_two["bestStep"] % phase_two["stepsPerEpoch"] == 0
    assert phase_two["bestEpoch"] != argmax + 1
    assert phase_two["bestEpoch"] >= toy_run.MIN_EPOCHS_2


def test_nothing_ends_the_run_short_of_its_ceiling(familia, tiny, monkeypatch):
    """The run walks `epochs_2` WHOLE, whatever the criterion reads.

    **This test asserted the opposite until 2026-10-19.** It was
    `test_the_patience_ends_the_run_at_an_epoch_and_before_the_ceiling`, and
    it scripted a sequence that improves once inside the legal window and
    never again, so that the patience was the only thing that could end the
    run short of the budget. The patience is gone; the sequence stays, and
    what it now proves is that nothing ends the run early.

    It is inverted rather than deleted. Deleting it would leave "the ceiling
    runs whole" asserted nowhere, and a patience reintroduced by anyone would
    pass every other test in this file.

    The scripted signal is the WORST case for this claim: it peaks at the
    guard's first legal epoch and declines from there, which is precisely the
    shape a patience would cut. Measured on the real artefact, that shape is
    not hypothetical -- 38 of 90 runs chose exactly the first legal step.
    """
    payload = _prepared(familia, tiny)
    per_epoch = [0.90, 0.80, 0.20, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
    # Descending INSIDE the legal window, so the criterion never improves
    # after its first legal read and a patience of any length would fire.
    cef_per_epoch = [0.1, 0.2, 2.9, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    assert len(cef_per_epoch) >= tiny.epochs_2
    _ventana = cef_per_epoch[toy_run.MIN_EPOCHS_2 - 1:]
    assert _ventana[0] == max(_ventana), (
        "the scripted cef does not peak on the first legal epoch, so a "
        "patience would not have fired and this test proves nothing")

    calls = {"n": 0}
    cef_calls = {"n": 0}

    def scripted(model_, dataset, device):
        epoch = min(calls["n"] // toy_run.STEPS_PER_EPOCH_DECLARED,
                    len(per_epoch) - 1)
        calls["n"] += 1
        return per_epoch[epoch]

    def scripted_cef(model_, device):
        value = cef_per_epoch[min(cef_calls["n"], len(cef_per_epoch) - 1)]
        cef_calls["n"] += 1
        return value

    monkeypatch.setattr(toy_run, "watchman", scripted)
    monkeypatch.setattr(toy_run, "effective_classes", scripted_cef)
    phase_two = _run(payload, familia, tiny)

    # It CHOSE the guard's first legal epoch, as the script forces...
    assert phase_two["bestEpoch"] == toy_run.MIN_EPOCHS_2, phase_two["bestEpoch"]
    # ...and then kept running to the ceiling anyway, which is the claim.
    assert phase_two["epochs"] == tiny.epochs_2, (
        phase_two["epochs"], tiny.epochs_2)
    assert phase_two["steps"] == tiny.epochs_2 * phase_two["stepsPerEpoch"]
    assert phase_two["stepsWithoutARead"] == 0
    # And the criterion was READ on every one of those epochs, not merely
    # walked past: a loop that trained without reading would satisfy the
    # three assertions above.
    assert cef_calls["n"] == tiny.epochs_2, cef_calls["n"]
    assert len(phase_two["epochPass"]) == tiny.epochs_2


def test_every_epoch_draws_every_training_bag_exactly_once(familia, tiny):
    """The pass property, read through the CALL SITE and not off the sampler.

    Both halves are asserted and neither alone would do: `distinct == bags`
    admits one bag drawn twice and another dropped, and `total == bags` admits
    the same the other way round. `balanced_batches` could return an impeccable
    partition and the loop consume a queue twice -- that is the join this
    crosses.
    """
    payload = _prepared(familia, tiny)
    phase_two = _run(payload, familia, tiny)
    rows = phase_two["epochPass"]
    assert rows, "the run published no pass at all"
    assert len(rows) == phase_two["epochs"] == (
        phase_two["steps"] // phase_two["stepsPerEpoch"])
    bags_in_role = len(payload["source_train"].targets)
    for row in rows:
        assert row["distinct"] == bags_in_role, row
        assert row["total"] == bags_in_role, row
        assert row["steps"] == phase_two["stepsPerEpoch"], row
    assert [row["epoch"] for row in rows] == list(range(1, len(rows) + 1))
    assert phase_two["epochDefinedBy"] == "the source training role"


def test_phase_two_refuses_a_budget_that_holds_fewer_epochs_than_the_minimum(
        familia, tiny):
    """A budget below `MIN_EPOCHS_2` passes is one in which the criterion can
    never choose, so nothing is ever restored -- and that refuses rather than
    running.

    This is the cost of putting the criterion in epochs, and it is paid where it
    can be read: the pilot's twelve steps were below one whole pass of this
    programme, so a pilot kept at that budget would have rehearsed a criterion
    that cannot fire. Both declared scales are asserted to be above the floor,
    so the refusal is a guard and not a wall.
    """
    for scale in (toy_run.FULL, toy_run.PILOT, tiny):
        assert scale.epochs_2 >= toy_run.MIN_EPOCHS_2, (
            scale.name, scale.epochs_2, toy_run.MIN_EPOCHS_2)

    payload = _prepared(familia, tiny)
    starved = dataclasses.replace(tiny, epochs_2=toy_run.MIN_EPOCHS_2 - 1)
    with pytest.raises(ValueError, match="may not choose before"):
        _run(payload, familia, starved)


def test_the_cell_record_carries_the_epoch_quantities_the_criterion_reads(
        familia, tiny):
    """What the loop measured reaches the artefact, every field of it.

    A roster rather than one field: a dropped key leaves every other assertion
    green, which is this document's ninth recorded instance of a reading wired
    to nothing.
    """
    cell = toy_run.run_cell(familia, "batchnorm", 0, tiny,
                            toy_run.declared_hyper(tiny), DEVICE)
    row = cell["rows"][ADAPTED]
    for field in ("phase2StepsPerEpoch", "phase2Epochs", "phase2BestEpoch",
                  "phase2StepsWithoutARead", "phase2EpochPass"):
        assert field in row, sorted(row)
    assert row["phase2StepsPerEpoch"] == toy_run.STEPS_PER_EPOCH_DECLARED
    assert row["phase2BestStep"] % row["phase2StepsPerEpoch"] == 0
    assert row["phase2BestEpoch"] * row["phase2StepsPerEpoch"] == (
        row["phase2BestStep"])
    assert row["phase2Epochs"] == row["phase2Steps"] // row["phase2StepsPerEpoch"]
    # ZERO by construction with the ceiling in epochs, and asserted as the
    # number rather than as a derivation of a field that no longer exists.
    assert row["phase2StepsWithoutARead"] == 0
    assert row["phase2Steps"] == tiny.epochs_2 * row["phase2StepsPerEpoch"]
    assert len(row["phase2EpochPass"]) == row["phase2Epochs"]


# ======================================= 4. the oracle series over the test role


def test_the_run_records_the_target_series_over_the_test_role_at_every_step(
        familia, tiny):
    """One reading per step, over the TEST role of the target domain -- the same
    statistical unit the published accuracy is measured on.

    Measured and not asserted: the denominator is the eighteen test bags, so
    every value is a multiple of `1/18`. The trap this avoids is the only
    per-frame target reading that already existed -- `trajectoryPrediction`,
    which covers every bag of the target domain at the frame stride -- and a
    maximum over THAT population printed beside a final value over the test
    role would put two statistical units in one row.
    """
    payload = _prepared(familia, tiny)
    phase_two = _run(payload, familia, tiny, target_eval=payload["target_eval"])
    series = phase_two[toy_run.TARGET_SERIES_FIELD]
    assert series is not None, "the run was handed the test role and recorded nothing"
    assert len(series) == phase_two["steps"], (len(series), phase_two["steps"])

    test_bags = len(payload["target_eval"].targets)
    assert test_bags == 18, test_bags
    for value in series:
        scaled = value * test_bags
        assert abs(scaled - round(scaled)) < 1e-9, (value, test_bags)
    # And the pole that distinguishes the role from the whole domain: 81 bags
    # would admit values 18 bags cannot express.
    assert tiny.bags_total != test_bags, (tiny.bags_total, test_bags)
    assert toy_run.TARGET_SERIES["role"] == "test"
    assert toy_run.TARGET_SERIES["cadence"] == "every step"


def test_the_series_travels_under_the_name_the_oracle_column_reads(familia,
                                                                   tiny):
    """One spelling for both ends of the join.

    The renderer reads `toy_tables.SERIE_BLANCO` by name and the run writes
    under it; a literal on either side would be a second definition that
    nothing obliges to agree with the first, and the column would dash while
    the record carried the series.
    """
    from MIL_CREDA_Benchmark import toy_tables

    assert toy_run.TARGET_SERIES_FIELD == toy_tables.SERIE_BLANCO
    cell = toy_run.run_cell(familia, "batchnorm", 0, tiny,
                            toy_run.declared_hyper(tiny), DEVICE)
    row = cell["rows"][ADAPTED]
    assert toy_tables.SERIE_BLANCO in row, sorted(row)
    assert row[toy_tables.SERIE_BLANCO], "the cell carries an empty series"


def test_the_series_at_the_chosen_step_is_the_accuracy_the_cell_publishes(
        familia, tiny):
    """The join between the series and the one published number.

    `adapt_phase_two` restores the best step's weights before returning, so the
    accuracy `run_cell` then reads over the test role IS the series' own entry
    at that step. Asserting them equal crosses the restore: a run that restored
    the last step instead, or a series recorded over another role, breaks it.
    """
    cell = toy_run.run_cell(familia, "batchnorm", 0, tiny,
                            toy_run.declared_hyper(tiny), DEVICE)
    row = cell["rows"][ADAPTED]
    series = row[toy_run.TARGET_SERIES_FIELD]
    best = row["phase2BestStep"]
    assert best >= 1, best
    assert series[best - 1] == row["targetAccuracy"], (
        best, series[best - 1], row["targetAccuracy"])
    # The control that the equality is not trivially true of every entry: the
    # series has to vary, or this test would pass against any step.
    assert len(set(series)) > 1, series


def test_nothing_selects_on_the_target_series(familia, tiny, monkeypatch):
    """The oracle is for the column and for nothing else, and that is measured
    in two ways no single reading covers.

    First the WHOLE run is bit-identical with and without the series: same
    chosen step, same stop, same watchman history. Second the series is
    scripted to peak where the criterion must not go, and the choice does not
    move -- a run that merely never looked at it would pass the first pole and
    fail nothing, while a run that consulted it would pass neither.
    """
    # A FRESH preparation per run: `adapt_phase_two` moves theta_t and restores
    # the best step's weights, so a second run over the same model would start
    # where the first ended and the comparison below would be of two different
    # experiments. The control is right underneath -- the two starting states
    # are asserted bit-identical before either run.
    first, second = _prepared(familia, tiny), _prepared(familia, tiny)
    assert torch.equal(
        toy_run._parameter_vector(first["model"].target_encoder),
        toy_run._parameter_vector(second["model"].target_encoder)), (
        "the two preparations do not start from the same theta_t, so nothing "
        "below can be attributed to the series")
    without = _run(first, familia, tiny)
    with_series = _run(second, familia, tiny,
                       target_eval=second["target_eval"])
    for field in ("steps", "bestStep", "bestEpoch", "epochs", "bestWatchman",
                  "finalWatchman", "restoredWatchman"):
        assert without[field] == with_series[field], (field, without[field],
                                                      with_series[field])
    assert [row[4] for row in without["history"]] == [
        row[4] for row in with_series["history"]], (
        "recording the series moved the watchman the criterion reads")
    assert without[toy_run.TARGET_SERIES_FIELD] is None

    # The adversarial pole: an oracle that peaks at the first step, which the
    # minimum forbids the criterion from choosing.
    calls = {"n": 0}

    def peaked(model_, dataset, device):
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else 0.0

    third = _prepared(familia, tiny)
    monkeypatch.setattr(toy_run, "oracle_target_reading", peaked)
    scripted_run = _run(third, familia, tiny, target_eval=third["target_eval"])
    assert calls["n"] == scripted_run["steps"], (calls["n"],
                                                 scripted_run["steps"])
    assert scripted_run[toy_run.TARGET_SERIES_FIELD][0] == 1.0, (
        "the scripted oracle never reached the run")
    assert scripted_run["bestStep"] == with_series["bestStep"]
    assert scripted_run["steps"] == with_series["steps"]


def test_neither_search_is_handed_the_test_role(familia, tiny):
    """The two searches cannot see the series, and that is read off the syntax
    tree rather than off a docstring.

    `search_phase_two` calls `adapt_phase_two` and `search_phase_one` trains
    phase 1; neither passes `target_eval`, so no trial can record a series and
    no objective can read one. A later session that threads the role through
    reddens this instead of leaving a search quietly selecting on target labels.
    """
    tree = ast.parse(inspect.getsource(toy_run))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "adapt_phase_two"]
    assert len(calls) == 2, len(calls)
    with_role = [c for c in calls
                 if any(kw.arg == "target_eval" for kw in c.keywords)]
    assert len(with_role) == 1, (
        f"{len(with_role)} call sites hand phase 2 the test role; exactly one "
        "-- `run_cell`'s -- may")
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if inner in calls:
                    functions.setdefault(id(inner), node.name)
    owners = {functions[id(c)] for c in calls}
    assert owners == {"run_cell", "search_phase_two"}, owners
    assert functions[id(with_role[0])] == "run_cell", functions[id(with_role[0])]


def test_the_oracle_reading_declares_that_it_reads_target_labels():
    """The one reading in this module that touches `y_t`, said where it is made.

    Using target labels to MEASURE is not leakage; using them to CHOOSE would
    be. The declaration is what makes that distinction auditable instead of
    being a sentence in a docstring nobody crosses.
    """
    declaration = toy_run.TARGET_SERIES
    assert declaration["readsTargetLabels"] is True
    assert declaration["entersNoSelectionPath"] is True
    assert declaration["field"] == toy_run.TARGET_SERIES_FIELD
    assert "oracle" in declaration["why"].lower()
    # CREDA reads its own series at CHECKPOINTS; this reads at every step, so
    # its maximum is not a lower bound of the same kind. Recorded rather than
    # left for somebody to discover by comparing two tables.
    assert declaration["comparedWithCreda"]
