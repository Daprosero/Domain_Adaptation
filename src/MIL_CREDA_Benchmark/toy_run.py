"""MIL-CREDA over CREDA's own hard families, under CREDA's two-phase protocol.

What this module is for, in the operator's words: find out FIRST whether
MIL-CREDA works at all under the conditions CREDA uses. It runs attention beta
(Eqs. 16-17), the bag kernel, the global term (Eqs. 32-37) and the local term
with the correspondence pi (Eqs. 30-31) on the three families of
`CREDA.toy_experiment.DURAS`, cut into bags by `toy_bags`, and leaves an
artefact on disk shaped the way `CREDA.toy_experiment.guardar` shapes one.

**Its arms are its own, the confidence weighting is ON, and what is off is
Eq. (39)'s two balance coefficients.** This module declares `TOY_ARMS` -- ONE
arm -- and reads nothing of `config.ARMS`, `config.ARM_ORDER`,
`config.ARMS_BY_ID`, `config.LADDER` or `config.FLOOR_OF`, which are the DIGIT
benchmark's declaration of a different experiment. See `TOY_ARMS` for why that
independence survives the arm's three flags coming out equal to digit `G`'s,
and `WEIGHTING_GATES` for the three places the confidence now enters.

**One arm, and therefore no ladder, no rung, no floor and no margin.** The
operator retired the floor arm `TB`, and what that costs is stated here rather
than engineered around: this toyset now reports a BARE ACCURACY with nothing to
read it against. There is no number on this path that says what the two phase-2
terms bought, because buying is a difference and a difference needs two arms.
Every reading that existed only to compute one is gone rather than kept "for
reference" -- `TOY_FLOOR`, `TOY_LADDER`, `TOY_FLOOR_OF`, `ladder_rungs`,
`floor_comparison`, `VERDICT_REPETITION_FLOOR` and the `escalones`/`pisos`
blocks of the artefact -- because a retired reading that still computes is a
number somebody will quote. What a single arm answers is whether the method
does what its own premises say it does; what it cannot answer is whether
anything it does is better than not doing it.

**`sin sopesado` was mis-read once and the correction is recorded here rather
than silently applied.** The operator's instruction for phase 2 was `sin
sopesado los dos loss`, and an earlier stretch read `sopesado` as the
CONFIDENCE weighting (`w_j^t`, Eqs. 24/26/37) and declared this arm with
`weighting: False`. It meant the BALANCE COEFFICIENTS of Eq. (39): without
`lambda_glob` and without `lambda_loc`. So the confidence weighting is back on
-- Eq. (26)'s `D` and Eq. (37)'s `omega_c` both run, and Eq. (38)'s own
aggregation was never behind that flag and runs too. Nothing about the
confidence is switched off anywhere on this path now. What IS off is the pair
of coefficients, and `FIXED_COEFFICIENT` is where that lives.

**What turning the weighting on costs on this toyset is MEASURED BY THE RUN,
not quoted here.** Eq. (37)'s `omega_c` is SUB-convex, so where the routed
confidences are low the global term comes out divided -- and where they are not,
it comes out essentially unchanged. Which happens is PER CELL: measured on the
twelve-bag pilot the ratio runs 0.0398 to 1.0411 over six cells. With a searched
ceiling the division could be absorbed -- `CEILING_RANGE` spans four orders, so
a `lambda_glob` free to grow is free to grow by any of this. At a coefficient
fixed at 1 it cannot be, so the global term enters smaller than the local one by
whatever that factor is on the cell, and a reader of the tables has to meet that
before the numbers.

The factor is NOT written down as a constant, and that is a correction rather
than a preference. A previous stretch measured it as 1.5x to 37x (ratios
0.027-0.683) and that reading does NOT reproduce here: it was taken on an arm
trained with `weighting: False`, so the confidences it read belong to a theta_t
this programme no longer produces. `weighting_cost` measures it per cell, every
row carries `weightingCostRatio` and `weightingCostOmegaSum`, and
`WEIGHTING_COST` keeps the prior reading labelled as prior. A quoted factor is
exactly the kind of sentence that ages without anything failing.

**Why a new module and not `harness.run_one`.** `run_one` trains one phase and
reads `epochs_record[-1]`: the last epoch, never the best. CREDA's toy programme
does neither -- it trains the source alone and FREEZES it, then adapts a copy,
and on exit it RESTORES the weights of the step its own stopping rule chose.
Carrying that into `harness.py` would change every digit result on disk, so the
protocol lives here and `harness.py` is untouched. What is NOT re-implemented
here is a single equation: every term comes from `MIL_CREDA.*` through
`wiring.Arm`, exactly as every other arm of this benchmark reaches it.

**No `__provenance__`, deliberately**, for `toy_bags.py`'s reason: this is
benchmark-side plumbing and implements no equation of the formulation. It
declares a training protocol, which the revision does not state -- the same
boundary `CREDA.two_phase.__provenance__` draws with its own
`claims_no_equation`.

----------------------------------------------------------------------------
The apparatus, and what it was measured against
----------------------------------------------------------------------------

1. **Two phases.** `train_phase_one` fits the source encoder, the attention
   parameters of Eq. (15) and the head on the supervised term ALONE (Eq. 21,
   `MIL_CREDA.objective.source_loss`) over source bags; it early-stops on
   SOURCE VALID accuracy, restores the weights of its best epoch, and freezes
   them. `copy_source_to_target` then makes theta_t bit-identical to theta_s --
   once -- and `adapt_phase_two` lets the global and local terms, each entering
   Eq. (39) at a FIXED coefficient of 1, move theta_t and nothing else. No ramp
   is evaluated on this path: see `FIXED_COEFFICIENT`.

2. **One criterion for the phase-2 selection and for the phase-2 search:
   `cef`, MAXIMIZED.** `effective_classes` is CREDA's `cef` in this
   benchmark's own unit: `exp(H(p))` with `p` the histogram of the bag-level
   argmax over the target TRAIN role, in `[1, classes]`. Zero target labels --
   it is the histogram of the model's OWN predictions. Both the epoch selection
   and `search_phase_two` maximize it.

   It was the watchman, MINIMIZED, until 2026-10-19, and the watchman is still
   computed and recorded at every epoch -- it just chooses nothing. Keeping it
   is what makes "which of the two would have ordered this better" answerable
   from the artefact instead of being a question nobody can go back and ask.
   Measured in CREDA over 90 cells the watchman was the worst rule tried and
   ordered the trials BACKWARDS in two of three families.

3. **Phase 2 restores the best weights on exit, and the criterion is read once
   per PASS of the training role** -- `MIN_EPOCHS_2` passes of guard and then
   every pass to the ceiling. CREDA's own reason for restoring, quoted:
   "quedarse con los ultimos y solo reportar el paso seria el mismo defecto que
   la fase 1 ya tenia."

   **There is no patience and no stop: phase 2 runs `epochs_2` whole.** There
   was a stop at `PATIENCE_2_EPOCHS` passes without a new best, and it is what
   the artefact of `Results/Toyset/full` was actually measuring: all 90 runs
   died by patience, NONE reached the ceiling, and 38 chose EXACTLY the first
   legal step. Phase 1 still stops -- its signal is supervised -- and the two
   phases differing is declared rather than covered by one sentence.

   The criterion read every STEP until an earlier cut, which is a separate
   defect that closed: there was a pass -- `harness.balanced_batches` keeps a
   shuffled queue per class and refills it when it empties -- and the criterion
   did not look at it, so the stop was decided on whichever batch the step
   happened to draw. The SAMPLING needed no change; only the criterion.
   `PHASE_TWO_CRITERION` says where every number comes from, which of them are
   declared, that the budget is now in EPOCHS, and what the asymmetry against
   CREDA's own nine-step pass costs.

4. **`config.SOURCE_TO_TARGET_SYNC_STEPS` is retired on this path.** Two phases
   means the source freezes and nothing transmits after the one copy. The
   declared risk is recorded in `MIL-CREDA/AGREED.md` and is NOT re-litigated
   here: without the transmission the full arm fell below the floor on digits.
   **Whether it reproduces on these families is no longer measurable here**, and
   that is the one consequence of the single-arm shape worth naming twice: the
   risk is stated as a fall BELOW A FLOOR, the two readings that computed one
   are gone with the floor arm, and nothing on this path can answer it. The
   `syncRetired` block of `flags` says so beside the risk, so a reader meets the
   absence rather than assuming silence means the risk went away.

5. **Four of the eight declared search dimensions are retired, and passing one
   is REFUSED BY NAME.** `ceiling` and `ceilingLocal` ARE `lambda_glob` and
   `lambda_loc`; `rampDelta` and `rampDeltaLocal` are how each one grows. With
   both coefficients fixed at 1 none of the four decides anything, so they
   leave the search (`SEARCH_RANGES` now holds the four that remain and
   `RETIRED_RANGES` names the four that left) -- and `refuse_retired_hyper`
   raises on any of their names reaching a hyper dict. Silently ignoring the
   key is how a retired coefficient returns through a stale `barrido.json`:
   the artefact carries it, the caller who passes it believes it decides
   something, and nothing says otherwise. This is
   `CREDA.toy_experiment.corrida`'s own guard, which refuses `TECHO` and
   `DELTA` by name for the same reason, built here against this path's names.

----------------------------------------------------------------------------
Two scales, and what separates them
----------------------------------------------------------------------------

`FULL` declares the scale a verdict requires -- CREDA's own, measured off
`CREDA.toy_experiment`. `PILOT` is that configuration with the repetition counts
and the lengths lowered and NOTHING else different: `REPETITION_FIELDS` names
exactly the fields a reduction may touch, and `tests/test_toy_run.py` asserts
every other field is equal between the two. Every record this module writes
carries `Scale.stamp()`, so a pilot number can never travel as a result.

`epochs_1`, `epochs_2` and `patience_1` are the lengths in `REPETITION_FIELDS`,
and `patience_1` is the only criterion among them: it is phase 1's own stop,
over SOURCE valid accuracy. `patience_2` and `min_steps_2` left the scale on
2026-10-19 with the step ceiling -- what they were worth is in
`PHASE_TWO_CRITERION`.

**And `epochs_2` stopped being a reduction the pilot can take freely**, which is
the cost of the criterion being in passes. The pilot's floor is now
`MIN_EPOCHS_2 + 1` passes -- enough for the criterion to become legal and then
choose -- and below it the criterion could never
have chosen: `adapt_phase_two` refuses a budget below `MIN_EPOCHS_2` passes
rather than running something that silently never restores. `PILOT`'s own
docstring derives the 152 it carries instead, and says what the cheaper
alternative would give up.

----------------------------------------------------------------------------
Two globals are bent, and this is a bend rather than a parameter
----------------------------------------------------------------------------

`config.CLASSES` is 10 and these families have 3 classes, so that one still has
to be bent. `config.BAGS_PER_CLASS` is 12 and this toyset cuts 27, so that one
is bent too and the bend is no longer a no-op: the two numbers used to agree
because this toyset borrowed the digit rule, and `Scale.bags_per_class` records
why it stopped. `config.INSTANCES_PER_BAG` is 30 and the operator fixed the same
number here, so that one IS bent to the value it already holds -- the code does
not special-case that, because a binding that skipped a global whenever it
happened to agree would be a second code path nothing exercises.
`config.py` is not editable on this path, and `bags.build`/`bags._select` read
every one of them as module globals, so
`material_bound` overrides them -- and `TRAIN_BAGS`, `VALID_BAGS`, `EVAL_BAGS`,
`INSTANCES_PER_BAG`, `BAGS_PER_STEP`, `BACKBONE`, `DOMAINS` and
`bags._raw_dataset` with them -- and restores every one in a `finally`. This is
the pattern `harness.run_one` and `wiring._resolve_encoder` already rely on for
`config.BACKBONE`.

**Where it breaks:** a module global is process-wide. Two of these runs
overlapping in one interpreter -- two threads, or a digit campaign in the same
process -- would read each other's override, and the failure is silent: the
numbers come out of a draw nobody declared. Nothing here can prevent that;
`material_bound` asserts the globals it is about to bend still hold their
declared values on entry, so an overlap is at least caught rather than measured.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import math
import time
from dataclasses import asdict, dataclass, fields as _campos_de
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from CREDA.toy_experiment import (ASPECTO_MAX, COL_DURA, DURAS,
                                  sigma_de_la_fuente)
from MIL_CREDA.objective import source_loss                       # Eq. (21)
from MIL_CREDA_Benchmark import (bags, config, harness, latent, toy, toy_bags,
                                 toy_tables, wiring)

# --------------------------------------------------------------------- scale

#: The fields a reduction is allowed to lower, and the ONLY ones. Everything
#: else `Scale` carries has to be equal between `FULL` and `PILOT` or the pilot
#: is a different program and proves nothing about the one that matters.
REPETITION_FIELDS = frozenset({
    "epochs_1", "epochs_2", "patience_1",
    "search_seeds", "report_seeds", "trials_1", "trials_2",
    "points_per_domain",
})


@dataclass(frozen=True)
class Scale:
    """Every number that defines a run of this programme, in one place.

    `name` is the stamp every record carries. The two instances below differ in
    `REPETITION_FIELDS` and in nothing else, which is what makes the pilot a
    rehearsal of this programme rather than of another one.
    """

    name: str
    #: Lengths, BOTH IN EPOCHS since 2026-10-19. `epochs_1` is a CEILING on
    #: phase 1, not a length: the stopping rule ends it earlier, exactly as
    #: `EPOCAS_1` is a ceiling in CREDA. `epochs_2` IS the length -- phase 2
    #: has no stopping rule, so it runs its ceiling whole.
    #:
    #: `steps_2` was a STEP ceiling and it is gone, with `patience_2` and
    #: `min_steps_2`. What a step ceiling did is let the lot decide how many
    #: passes the experiment runs, which is measured in `PHASE_TWO_CRITERION`.
    epochs_1: int
    epochs_2: int
    patience_1: int
    #: Repetitions. The search seeds are DISJOINT from the report seeds: a
    #: hyperparameter chosen on the seeds that get published is a hyperparameter
    #: fitted to them.
    search_seeds: tuple[int, ...]
    report_seeds: tuple[int, ...]
    trials_1: int
    trials_2: int
    #: Material: how many points each domain's generator draws. The ONLY
    #: material field a reduction may lower, and what it has to stay large
    #: enough for is `instances_per_bag` below -- `_instances_per_bag` refuses
    #: a draw whose smallest bag cannot fund it, rather than quietly handing
    #: `bags._select` a bag it will refuse mid-draw.
    points_per_domain: int
    #: Everything below is the programme and is equal at both scales.
    lr_1: float = 1e-2
    lr_2: float = 1e-3
    classes: int = 3
    #: THIS toyset's own number, and no longer the digit benchmark's
    #: `config.BAGS_PER_CLASS`: 27 bags per class, so 81 bags per domain over
    #: CREDA's three classes. It was 4, then 12 (which was the digit rule), and
    #: what each change buys is resolution -- see `realized_split` for the
    #: roles that follow and `toy_bags.BAGS_PER_CLASS_TOY` for the measurement
    #: over several counts that makes 27 the efficient point rather than
    #: simply a larger one.
    #:
    #: **It is a programme field, so both scales carry it**, which is what
    #: makes the pilot a rehearsal of this instrument and not of a coarser one.
    bags_per_class: int = 27
    #: How many instances each bag contributes, DECLARED and not derived. The
    #: digit benchmark's own `config.INSTANCES_PER_BAG`, and a programme field
    #: rather than a repetition one: a bag is the same size at both scales, in
    #: every family, so the unit of work per step is fixed and declared instead
    #: of varying with whichever quantile band came out smallest.
    #:
    #: It used to be `points_per_domain // bags_total`, which was a THIRD
    #: number: the stamp carried that arithmetic (150 at the full scale) while
    #: `material_bound` bent `config.INSTANCES_PER_BAG` to the measured
    #: smallest bag (138), and the record said one and the run took the other.
    #: Declaring it collapses the two and makes the measurement a guard.
    instances_per_bag: int = 30
    #: CREDA's `FRAC_TRAIN`/`FRAC_VALID`, as REQUESTED. They are fractions of
    #: POINTS there and this benchmark splits BAGS, so what is realized is in
    #: `realized_split` -- reported beside the request rather than quietly in
    #: place of it.
    frac_train: float = 0.6
    frac_valid: float = 0.2

    def stamp(self) -> dict:
        """The scale a record was measured at, as the record carries it.

        Flat and complete: a reader of one record never has to find this module
        to learn whether the number in front of them came from a pilot.
        """
        payload = {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(self).items()}
        payload["isPilot"] = self.name != FULL.name
        payload["realizedSplit"] = self.realized_split()
        return payload

    @property
    def bags_total(self) -> int:
        return self.classes * self.bags_per_class

    def realized_split(self) -> dict:
        """How many bags per class each role actually gets, and the fractions
        that follow.

        `frac_train`/`frac_valid` cannot be honoured at bag granularity: with 27
        bags per class, 0.6/0.2/0.2 asks for 16.2/5.4/5.4 bags. The attainable
        split nearest the request with every role covering every class is
        16/5/6, and both numbers travel so nobody reads the request as the
        measurement.

        **What that realizes, as the number a reader compares two cells
        with: 18 test bags and a step of 5.56 pp.** The published accuracy is
        read over `test` in every class, so it takes one of 19 values and two
        cells cannot differ by less than 100/18 points. The previous cut of 12
        realized 7/2/3, so 9 test bags and 11.11 pp.

        **Which fractions these are matters and is not the digit benchmark's.**
        `frac_train`/`frac_valid` are CREDA's `FRAC_TRAIN`/`FRAC_VALID` and
        `tests/test_toy_run.py::test_the_full_scale_is_credas_own` pins them
        there. The digit campaign's own bag split is a different pair --
        `TRAIN_BAGS`/`VALID_BAGS`/`EVAL_BAGS` of 64/20/36 over 120 bags, so
        0.5333/0.1667/0.3000 -- and at 27 bags per class those would ask for
        14.4/4.5/8.1 and land on 14/5/8. The two readings differ by two bags
        per class in train and in test, and this path keeps CREDA's because
        that is what this scale declares and what the pin holds it to.

        **The third role is named `test`, which is CREDA's own word for it.**
        It is the role every published accuracy of the per-family table is read
        on, and `toy_tables.ROL_PUBLICADO` prints that name into each family
        header. The name is HERE because the header has to say the role the
        record carries: a header reading `test` over a record splitting on some
        other name would be a label nobody measured, which is exactly what
        `toy_tables.rol_publicado` refuses. The bagset this split feeds keeps
        `bags.BagSet.eval_idx` -- that attribute is the digit benchmark's as
        much as this one's and renaming it is not this toyset's to do -- so the
        record's role name and that attribute's name differ on purpose, and the
        join between them is asserted in
        `tests/test_toy_run.py::test_every_role_covers_every_class_at_the_new_counts`.

        **This role absorbs the rounding, and it is computed last for that
        reason.** `train` and `valid` round to the nearest whole bag and `test`
        is the complement, so the 0.4 of a bag `valid`'s 5.4 gave up and the
        0.2 `train`'s 16.2 gave up both land on `test` -- which is why `test`
        gets 6 where the request asked for 5.4. `bags._split_counts`' own docstring describes the same ordering
        and reads as though the complement were the role PROTECTED from the
        remainder; it is the role that receives it. Said plainly here because the
        verdict rests on that role and a reader is owed which direction the
        rounding went.

        `bags._share` never sees a remainder on this path at all: `material_bound`
        sets `config.TRAIN_BAGS`/`VALID_BAGS`/`EVAL_BAGS` to a PER-CLASS count
        times `classes`, so each total divides by the class count exactly and
        every class gets the identical 16/5/6. The per-class draw `_split_counts`
        performs -- the reason the local correspondence is defined at all, since
        it is undefined for a class with no source bag -- therefore cannot drop
        a class here even in principle.
        """
        train = max(1, round(self.bags_per_class * self.frac_train))
        valid = max(1, round(self.bags_per_class * self.frac_valid))
        evaluation = self.bags_per_class - train - valid
        if evaluation < 1:
            raise ValueError(
                f"the requested split {self.frac_train}/{self.frac_valid} "
                f"leaves {evaluation} evaluation bags per class out of "
                f"{self.bags_per_class}; every role must cover every class")
        return {
            "requested": {"train": self.frac_train, "valid": self.frac_valid,
                          "test": 1.0 - self.frac_train - self.frac_valid},
            "bagsPerClass": {"train": train, "valid": valid, "test": evaluation},
            "realized": {"train": train / self.bags_per_class,
                         "valid": valid / self.bags_per_class,
                         "test": evaluation / self.bags_per_class},
        }


#: CREDA's own scale, read off `CREDA.toy_experiment`: `EPOCAS_1`, `EPOCAS_2`,
#: `LR_1`, `LR_2`, `PACIENCIA_1`,
#: `SEMILLAS_BUSQUEDA`, `SEMILLAS_REPORTE`, `TRIALS_1`, `TRIALS_2`,
#: `FRAC_TRAIN`, `FRAC_VALID`. Declared, not run: nothing here releases it.
#:
#: **`points_per_domain` is the ONE field of this scale that is no longer
#: CREDA's, and that departure is forced rather than chosen.** It read
#: `toy_bags.PUNTOS_POR_DOMINIO` -- CREDA's own 1800, the number its
#: three-families figure draws -- and 27 bags per class of 30 declared
#: instances cannot be funded by it: measured over the thirteen seeds either
#: scale uses, 1800 leaves a smallest quantile band of 20 points and
#: `_instances_per_bag` refuses on all 39 (family, seed) pairs. The floor is
#: 2586, also measured over those thirteen seeds.
#:
#: What is written here instead is the DENSITY this scale has always drawn,
#: carried to the new bag count: 1800 over 36 bags is 50 points per bag, and
#: 81 bags at 50 is 4050. Measured at 4050, the smallest band over all 39
#: pairs is 46, against the 30 the programme declares. CREDA has no bags at
#: all -- it works on points -- so there is no bagged number of its own to
#: read here, and the density is the strongest tie that survives the cut.
FULL = Scale(
    name="full",
    epochs_1=100, epochs_2=100,
    patience_1=30,
    search_seeds=(100, 101, 102), report_seeds=tuple(range(10)),
    trials_1=25, trials_2=25,
    points_per_domain=4050,
)

#: The same programme, short enough to be wrong in minutes. What makes the
#: restore path run here is `epochs_2` holding enough passes for the criterion
#: to choose; the paragraph on it below derives the number.
#:
#: **`points_per_domain` is 2835 and the declared bag size is what forced it.**
#: 81 bags of 30 instances need 2430 instances per domain, and the bands are
#: quantiles of a draw rather than equal shares, so the arithmetic is not the
#: constraint -- the smallest band is. Measured over the thirteen seeds either
#: scale uses, with the same procedure that settled the previous 1260: the
#: floor is 2586 (2585 leaves a smallest bag of 29 on `rollo` and `toroide` at
#: seed 0 and refuses), and 2835 leaves 32 on the worst of all 39
#: (family, seed) pairs -- the identical margin the previous 1260 held at 36
#: bags. 2835 is this scale's own density carried to the new cut: 1260 over 36
#: bags is 35 points per bag, and 81 bags at 35 is 2835.
#:
#: So the cost of declaring the bag size is that `points_per_domain` stops
#: being a usable reduction: it may only fall as far as still funds the
#: programme's bag, and below that `_instances_per_bag` refuses instead of
#: running something smaller.
#:
#: What that buys is a pilot whose MATERIAL IS THE FULL RUN'S: a bag holds 30
#: instances at both scales and a domain holds 81 of them at both, so an epoch
#: here processes exactly what an epoch there processes and `cost_report`'s own
#: `instanceScale` is 1.0 by construction. The pilot is now a reduction in
#: repetitions and lengths alone, which is what it was always supposed to be and
#: was not: at 360 points its bags held 26 to 36 instances against the full
#: scale's 138 to 150, so the two differed in the one thing nothing declared.
#:
#: **`epochs_2` is 4 and the number is DERIVED rather than chosen.** The
#: phase-2 criterion reads once per PASS of the training role
#: (`PHASE_TWO_CRITERION`), so the pilot needs enough passes for the criterion
#: to become legal and then choose: `MIN_EPOCHS_2 + 1`. Below that the pilot
#: would rehearse a criterion that can never choose and a restore that never
#: happens, and `adapt_phase_two` refuses it rather than running something that
#: silently never restores.
#:
#: It used to be `MIN_EPOCHS_2 + PATIENCE_2_EPOCHS + 1` passes expressed as 152
#: steps, because the run had a THIRD branch -- stopped by patience -- that the
#: pilot had to rehearse. That branch no longer exists: phase 2 runs its
#: ceiling whole. So the pilot got cheaper by removing a rehearsal of something
#: that cannot happen, not by lowering a bar.
#:
#: The eight-step unread tail it also rehearsed is gone for the same reason:
#: with the ceiling in epochs there is no step of the budget that is trained
#: and never read.
PILOT = Scale(
    name="pilot",
    epochs_1=6, epochs_2=4,
    patience_1=3,
    search_seeds=(100,), report_seeds=(0,),
    trials_1=2, trials_2=2,
    points_per_domain=2835,
)

SCALES = {FULL.name: FULL, PILOT.name: PILOT}

# ------------------------------------------- the phase-2 criterion, in epochs
#
# Phase 2 read the watchman at EVERY STEP, so the early stop was decided on a
# model that had seen an arbitrary batch rather than a pass of the training
# role. Phase 1 (`train_phase_one`) already walks epochs. `CREDA.toy_experiment`
# moved its own phase-2 criterion to epochs first; this is the same move with
# this programme's own epoch, and the sampling needed no change at all --
# `harness.balanced_batches` already keeps a shuffled queue per class and
# refills it when it empties, so with 16 training bags per class 16 steps IS
# one pass.


def steps_of_one_epoch(targets) -> int:
    """How many steps ONE pass of a role has, as `balanced_batches` draws it.

    That sampler yields exactly one bag of every class per step and refills
    each class's queue independently, so a pass is as long as the PER-CLASS
    count -- not as long as the role.

    **It REFUSES when the classes do not agree**, rather than returning the
    first or the largest. CREDA's `pasos_de_una_epoca` refuses the same way
    when its two domains disagree in blocks, and for the same reason: an
    unequal role has no step at which every bag has come out exactly once, so
    wrapping the short queue would leave the property this whole change exists
    for false with nothing failing. The refusal is reachable -- an unequal role
    runs happily and silently forever through that sampler.
    """
    counts: dict[int, int] = {}
    for label in targets:
        counts[int(label)] = counts.get(int(label), 0) + 1
    if not counts:
        raise ValueError("the role holds no bags, so it has no pass")
    distinct = sorted(set(counts.values()))
    if len(distinct) != 1:
        raise ValueError(
            f"the classes of this role do not agree in bag count ({counts}), "
            "so there is no step at which every bag has come out exactly "
            "once; deciding which class defines the pass is a decision about "
            "the experiment and is not taken here")
    return distinct[0]


#: The steps one pass has at the DECLARED programme, and the anchor every
#: epoch quantity below is derived from.
#:
#: **A constant of the MODULE and not of the run, which is CREDA's own choice
#: of anchor.** There it exists because the lot is SEARCHED
#: (`LOTES_2_BUSCADOS`), so deriving the criterion's resolution per run would
#: hand it to a hyperparameter that does not know it is deciding it. Here the
#: lot is not searched at all -- `harness.balanced_batches` takes no lot
#: argument and yields one bag per class -- so the anchor is the declared bag
#: programme instead: the per-class TRAIN count `Scale.realized_split`
#: realizes. It comes out identical at both declared scales because
#: `bags_per_class`, `frac_train` and `classes` are programme fields, which is
#: what makes the pilot a rehearsal of this criterion rather than of a coarser
#: one.
STEPS_PER_EPOCH_DECLARED = FULL.realized_split()["bagsPerClass"]["train"]

#: THE MINIMUM, DECLARED. It was DERIVED from `min_steps_2 = 40` -- 40 steps of
#: this programme's pass is `ceil(40 / 16) = 3` of them -- and that input was
#: retired with the step ceiling, so the derivation would now read a number
#: nobody maintains. The VALUE did not move: 3 is what the derivation gave.
#:
#: **It does not equal CREDA's 5, and that is deliberate rather than an
#: oversight.** Both were derived from the same 40-step floor over different
#: passes (16 steps here, 9 there), and declaring them did not reconcile the
#: two numbers, because reconciling them is a decision about the guard and not
#: about the unit. The cross-package pin asserts both are declared in epochs
#: and publishes both values; it does not assert they are equal.
#:
#: Measured, this guard DECIDES rather than guards: over the 90 runs of
#: `Results/Toyset/full`, 38 chose EXACTLY the first legal step -- 42%.
MIN_EPOCHS_2 = 3


def _criterion_block() -> dict:
    """`PHASE_TWO_CRITERION`, computed rather than written down.

    Every number here is derived from this module's own declarations and from
    `CREDA.toy_experiment`'s, so no literal in either file can satisfy it alone
    and the asymmetry it reports cannot go stale while both sides move.
    """
    from CREDA import toy_experiment as creda

    creda_epoch = creda.BLOQUES_AL_LOTE_DECLARADO
    return {
        "unit": "epoch",
        "what": ("the MAXIMUM of `cef` -- the effective number of classes the "
                 "target routing uses -- read once at the END of each "
                 "complete pass of the source training role"),
        "signal": "cef",
        "signalIs": ("exp(H(histogram of argmax over the target TRAIN role)), "
                     "in [1, classes]. It reads no target label: it is the "
                     "histogram of the model's own predictions"),
        "direction": "maximum",
        "signalChange": (
            "2026-10-19: it was the MINIMUM of the watchman. Measured over "
            "CREDA's 90 cells the watchman was the worst rule tried and cef "
            "the best label-free one; here the watchman is kept as a RECORDED "
            "reading so the two orderings stay comparable"),
        "stepsPerEpochDeclared": STEPS_PER_EPOCH_DECLARED,
        "stepsPerEpochDeclaredDerivation": (
            "Scale.realized_split()['bagsPerClass']['train'], because "
            "harness.balanced_batches yields one bag of every class per step "
            "and refills each class's queue independently"),
        "epochDefinedBy": "the source training role",
        "minEpochs2": MIN_EPOCHS_2,
        "minEpochs2Is": "declared",
        "minEpochs2Was": (
            "derived as ceil(min_steps_2 / STEPS_PER_EPOCH_DECLARED) = "
            "ceil(40 / 16); that input was retired with the step ceiling and "
            "the value did not move"),
        "minEpochs2InCreda": creda.MIN_EPOCAS_2,
        "minEpochs2WhyTheyDiffer": (
            "both were derived from the same 40-step floor over different "
            f"passes ({STEPS_PER_EPOCH_DECLARED} steps here, {creda_epoch} "
            "there). Declaring them did not reconcile the numbers, because "
            "that is a decision about the guard and not about the unit"),
        "minEpochs2Bites": (
            "measured: over the 90 runs of Results/Toyset/full, 38 chose "
            "EXACTLY the first legal step (42%), and in CREDA mejor_paso == "
            "MIN_PASOS_2 in 5 of 9 cells. A guard the criterion lands on two "
            "times in five is deciding, not guarding"),
        "patience": None,
        "patienceWhy": (
            "phase 2 has NO stopping rule since 2026-10-19: it runs its "
            "ceiling whole and the criterion chooses the best epoch of the "
            "WHOLE run. PATIENCE_2_EPOCHS = 5 was retired rather than kept at "
            "its value, because a patience published beside a run that never "
            "stops reads as a mechanism"),
        "phaseOneStillStops": (
            "phase 1 DOES stop: patience_1 epochs without improving SOURCE "
            "valid accuracy, which is supervised. The two phases differ here "
            "and the record says so rather than leaving one rule to cover both"),
        "budgetUnit": "epoch",
        "budgetWhy": (
            "epochs_2 is an EPOCH ceiling since 2026-10-19, which INVERTS what "
            "this field said. It said `steps_2 stays a STEP ceiling, as "
            "PASOS_2 did in CREDA`, on the grounds that every frame schedule "
            "and checkpoint grid of this module is indexed in steps. The "
            "grounds were backwards: a step ceiling lets the lot decide how "
            "many passes the experiment runs, and in CREDA the lot is SEARCHED"),
        "budgetChangeMeasured": {
            "inCredaAt200Steps": {
                "lunas": "LOTE_2=128 -> 9 steps/epoch -> 22.2 passes",
                "toroide": "LOTE_2=64 -> 17 steps/epoch -> 11.8 passes",
                "rollo": "LOTE_2=64 -> 17 steps/epoch -> 11.8 passes",
            },
            "reading": ("CREDA's global table compared three families trained "
                        "half as long as one another and nothing said so"),
            "hereAt200Steps": FULL.epochs_2 and 200 / STEPS_PER_EPOCH_DECLARED,
            "hereNote": ("this programme's lot is NOT searched -- "
                         "harness.balanced_batches takes no lot argument -- so "
                         "the step ceiling never varied the pass count HERE. "
                         "The unit changed anyway, because the pin holds the "
                         "two protocols to one unit and a benchmark that "
                         "measured in another would not be the control"),
        },
        "atTheFullScale": {
            "epochs2": FULL.epochs_2,
            "stepsItHolds": FULL.epochs_2 * STEPS_PER_EPOCH_DECLARED,
            "stepsWithoutARead": 0,
            "note": ("zero by construction: with the ceiling in epochs there "
                     "is no step of the budget that is trained and never read. "
                     "At the step ceiling this was 8"),
        },
        "asymmetryAgainstCreda": {
            "stepsPerEpochHere": STEPS_PER_EPOCH_DECLARED,
            "stepsPerEpochInCreda": creda_epoch,
            "whyTheyDiffer": (
                f"this programme's pass is {STEPS_PER_EPOCH_DECLARED} steps "
                f"({FULL.realized_split()['bagsPerClass']['train'] * FULL.classes} "
                f"training bags at {FULL.classes} per lot) and CREDA's is "
                f"{creda_epoch} ({int(creda.FRAC_TRAIN * creda.N_POR_DOMINIO)} "
                f"training points at lot {creda.LOTE}). It comes from the "
                f"material, not from a choice"),
            "epochsInTheBudgetHere": FULL.epochs_2,
            "epochsInTheBudgetInCreda": creda.EPOCAS_2,
            "reading": (
                "the two budgets are now the SAME NUMBER of passes, which is "
                "the whole point of the unit: what a pass costs still differs "
                "because the material differs, and that difference is "
                "material rather than protocol"),
        },
    }


#: The criterion, as the record carries it. One dict, so a reader of
#: `barrido.json` meets the unit, both derivations and the asymmetry rather
#: than inferring them from two integers.
PHASE_TWO_CRITERION = _criterion_block()

#: The record field a per-step target reading travels in, taken from the
#: renderer that reads it rather than spelled again here. `toy_tables` names
#: `SERIE_BLANCO` and the oracle column of the per-family table reads that
#: name; a literal on this side would be a second definition nothing obliges
#: to agree with the first, and the column would dash while the record carried
#: the series.
TARGET_SERIES_FIELD = toy_tables.SERIE_BLANCO

#: What that series IS, declared where it is produced.
#:
#: **It reads target labels, and that is said out loud.** Using `y_t` to
#: MEASURE is not leakage; using it to CHOOSE would be. So this reading enters
#: no selection path at all -- not the stop, not the restore, not either
#: search -- and both halves of that are measured rather than asserted: the
#: whole run is bit-identical with and without it, and the two searches are
#: proved from the syntax tree never to hand phase 2 the test role.
TARGET_SERIES = {
    "field": TARGET_SERIES_FIELD,
    "role": "test",
    "cadence": "every step",
    "unit": "bag",
    "readsTargetLabels": True,
    "entersNoSelectionPath": True,
    "why": ("the oracle column of the per-family table -- CREDA's own `blanco "
            "pico · test (oráculo)` -- is the maximum of this series, and the "
            "gap between it and the published value is what separates `the "
            "term is the problem` from `the stopping rule is`. Without the "
            "series that column was a dash in every row"),
    "whyTheTestRole": (
        "the same statistical unit the published accuracy is read on, so the "
        "peak and the final value sit in one row legitimately. The only "
        "per-frame target reading that already existed is "
        "`trajectoryPrediction`, over every bag of the target domain at the "
        "frame stride; a maximum over THAT population printed beside a final "
        "value over the test role would put two statistical units in one line"),
    "comparedWithCreda": (
        "CREDA reads its own `acc_te_ser` at CHECKPOINTS (`CUADROS`, every "
        "tenth step), so its peak is a lower bound of the true one. This "
        "reads at EVERY step, so its maximum is the true maximum over the "
        "steps the run took -- a difference in the instrument, recorded "
        "rather than left for somebody to find by comparing two tables"),
}

# ----------------------------------------------------------------- the toyset

#: One single-character domain code per (family, role). Single characters are
#: not a style choice: `bags.build` seeds its generator with
#: `seed * 1000 + ord(code)`, and `ord` refuses anything longer.
FAMILY_CODES = {
    "lunas": ("l", "L"),
    "toroide": ("o", "O"),
    "rollo": ("w", "W"),
}

#: What each bent `config.DOMAINS` entry is called while a run is bound.
DOMAIN_NAMES = {
    "l": "CredaLunasSource", "L": "CredaLunasTarget",
    "o": "CredaToroideSource", "O": "CredaToroideTarget",
    "w": "CredaRolloSource", "W": "CredaRolloTarget",
}

#: CREDA's `POS_FILAS`, carried over WHOLE: `ninguna`, `batchnorm`,
#: `layernorm`, in that order, each mapped to the backbone `mlp.py` builds for
#: it. The order is the reading and not an accident of a dict -- every row
#: count, every figure row and every table block downstream is derived from
#: THIS, so a fourth backbone grows the grid without anybody editing a literal.
#:
#: **The third row is the control, which is why it is not optional.**
#: `batchnorm` does two things at once -- it normalizes, and it holds
#: `running_mean`/`running_var` that the target domain can re-estimate, which
#: is what AdaBN is. `layernorm` normalizes and holds nothing re-estimable, so
#: AdaBN has no rung on it: `adabn_reading` finds no buffer-carrying layer and
#: says so instead of reporting a number. Without this row a `batchnorm` row
#: reading above `ninguna` cannot be attributed -- normalizing and
#: re-estimating are confounded in it -- and the whole ladder stops being
#: attributable in exactly the cells where AdaBN reaches the method's rung.
#:
#: Was `{"ninguna": "mlp-nobn", "batchnorm": "mlp"}` with
#: `NORMALIZATION_ABSENT = ("layernorm",)`, and the reason given for the
#: absence -- that `mlp.py` was outside that stretch's authorized surface -- was
#: about a work boundary and never about the mathematics. The operator lifted
#: it; the backbone is `mlp-layernorm` and the declaration is complete.
NORMALIZATIONS = {"ninguna": "mlp-nobn", "batchnorm": "mlp",
                  "layernorm": "mlp-layernorm"}

#: The rows `CREDA.toy_experiment.POS_FILAS` draws that this benchmark has no
#: backbone for. EMPTY now, and it stays as a declaration rather than being
#: deleted: every reader downstream -- `toy_tables.ausentes`, the architecture
#: table, the cell that draws -- asks the record what is missing, and a record that
#: carried no answer at all would make "nothing is missing" indistinguishable
#: from "nobody looked". `tests/test_toy_run.py` holds it equal to the
#: difference against CREDA's own list, so it cannot go stale in either
#: direction.
NORMALIZATION_ABSENT = ()

#: ---------------------------------------------------------------------------
#: The toyset's OWN arm declaration, and why it has to be its own
#: ---------------------------------------------------------------------------
#:
#: `config.ARMS` (`B`, `E`, `F`, `G`) is the DIGIT benchmark's declaration -- a
#: different experiment, over a different material, answering a different
#: question. Until this stretch this module read `tuple(config.ARM_ORDER)` and
#: so borrowed that declaration whole, and the borrowing was the defect rather
#: than merely untidy. Two things make it structural:
#:
#: - `tests/test_benchmark_declarations.py::test_the_declared_arms_and_rungs_
#:   are_exactly_the_operators_four_and_three` PINS those four ids and those
#:   three rungs, and its own docstring says the pin exists "to catch the NEXT
#:   silent reversal of whichever decision is current". Adding an arm there to
#:   serve this toyset would change the digit campaign and trip a pin that is
#:   doing its job.
#: - The protocol is not theirs. `config.ARMS` is a declaration of ARMS, and
#:   the arms are only half of what separates these two runs: the digit
#:   benchmark trains one phase through `harness.run_one` and reads the LAST
#:   epoch, under `harness.ramp`'s warmup to a searched ceiling. This path
#:   trains two phases, restores the best step, and enters Eq. (39) at a FIXED
#:   coefficient of 1 with no schedule at all. Borrowing the declaration would
#:   have tied this protocol's arm list to the other one's.
#:
#: **ONE arm, and the absence is the declaration.** A floor arm `TB` stood here
#: -- the same unit, the same everything, with the adaptation term switched off
#: -- and the operator retired it. So this list has one entry and there is no
#: pair to subtract: what the run reports is this arm's accuracy and nothing
#: says what it is better or worse than. That absence is written down here
#: rather than compensated for, because a replacement comparison invented to
#: fill the hole would answer a question nobody asked, and a floor kept "for
#: reference" would be quoted as one.
#:
#: **The three flags came out equal to digit `G`'s, and the independence is
#: unchanged by that.** With the confidence weighting back on (see the module
#: docstring's note on the mis-read `sin sopesado`) the adapted arm declares
#: `adaptation="milcreda"`, `weighting=True`, `local=True` -- which is exactly
#: `config.ARMS_BY_ID["G"]`'s three. An earlier version of this comment argued
#: the independence from the OPPOSITE fact: that no digit arm carried
#: "local without weighting", so the shape this toyset needed could not be
#: borrowed. That sentence is now false and is NOT what holds the line. What
#: holds it is the two reasons above: the pin, and the protocol. `flags()`
#: reports the match rather than hiding it, and
#: `test_the_adapted_arms_three_flags_equal_digit_gs_and_the_schedule_is_what_
#: differs` measures both halves.
#:
#: The operator's own framing of what this toyset is for: "eso es de un
#: benchmark que no es el nuestro, despues los conectamos, pero es necesario
#: mirar primero si funciona o no bajo las condiciones que usamos en creda".
#: Connecting the two benchmarks is later work and is not declared here.
#:
#: What this independence does NOT mean: everything the toyset genuinely shares
#: with the digit benchmark it keeps sharing -- `wiring.Arm` itself, the Pools,
#: the material bridge, every equation under `MIL_CREDA.*`. See
#: `SHARED_WITH_DIGIT_BENCHMARK`. It is the ARM DECLARATION, and only that,
#: that is this module's own.
TOY_ARMS = [
    {
        "id": "TGL", "name": "Toy-MIL-CREDA (global + local, weighted)",
        "label": "global + local, confidence weighting on, both coefficients 1",
        "unit": "bag", "adaptation": "milcreda", "weighting": True,
        "local": True,
        "attention": "learned", "selection": None, "normalization": "shared",
        "why": ("what this toyset exists to find out: whether MIL-CREDA works "
                "at all under the conditions CREDA uses. Phase 2 computes BOTH "
                "declared terms -- the global one (Eqs. 32-37) and the local "
                "correspondence (Eqs. 30-31, aggregated by Eq. 38) -- with the "
                "confidence weighting ON: Eq. (26)'s `D`, Eq. (37)'s `omega_c` "
                "and Eq. (38)'s own aggregation all run, and nothing about the "
                "confidence is switched off anywhere. What IS off is Eq. (39)'s "
                "pair of balance coefficients, which the operator's `sin "
                "sopesado los dos loss de la fase 2` was about: both are "
                "exactly 1, fixed, with no ramp (`FIXED_COEFFICIENT`). The "
                "`weighting: False` an earlier stretch declared here came from "
                "reading `sopesado` as the confidence weighting; see the module "
                "docstring. What the confidence costs on this toyset is "
                "measured in `WEIGHTING_COST`, because at a fixed coefficient "
                "of 1 it cannot be absorbed."),
    },
]

TOY_ARMS_BY_ID = {arm["id"]: arm for arm in TOY_ARMS}

#: The arms this toyset runs, in declaration order. **Not `LADDER_ARMS`, which
#: is what this used to be called**: with the floor arm retired there is no
#: ladder for it to be the arms of, and a name describing a mechanism the module
#: no longer has is read and believed. `run_cell` trains phase 1 on
#: `TOY_ARM_IDS[-1]` and shares it, and `build_arm` resolves every id through
#: `TOY_ARMS_BY_ID` -- never through `config.ARMS_BY_ID`, which is what
#: `wiring.build` would have done.
TOY_ARM_IDS = tuple(arm["id"] for arm in TOY_ARMS)

#: Named rather than indexed at every call site, so the arm carrying the method
#: is read off the declaration above instead of spelled out again. With one arm
#: declared this IS that arm; `TOY_FLOOR`, which named the other end of the
#: retired rung, is gone rather than left pointing at this one.
TOY_ADAPTED = TOY_ARM_IDS[-1]

#: What `spec["weighting"] = True` actually switches ON, measured against
#: `wiring.Arm._milcreda_term` rather than assumed from the key's name. The
#: flag gates TWO distinct mechanisms, and a reader who took "sopesado" to mean
#: only the first would be wrong about what this arm computes -- so the reading
#: travels with every record this module writes. A third confidence reading was
#: never behind the flag at all and runs regardless; it is named here for the
#: same reason.
#:
#: This arm declares the flag TRUE, so all three run and nothing about the
#: confidence is switched off on this path. The "with it off" halves below are
#: kept because they are what the flag means, they are the branch
#: `tests/test_toy_run.py` flips a twin into to prove the mechanisms are live,
#: and `config.ARMS`' own `E` still reaches them.
WEIGHTING_GATES = {
    "eq26D": ("Eq. (26)'s `D` inside the kernel blocks: with the flag on -- "
              "which is this arm -- `conditional.weighted_blocks` forms "
              "`K~_c^t = D K_c^t D` and `K~_c^{st} = K_c^{st} D` from Eq. "
              "(25)'s diagonal of target confidences. With it off those two "
              "blocks would enter the mixture RAW, which is `D = I` -- "
              "`wiring.py`'s own `K_st_w, K_t_w = K_st_c, K_t_c` branch, which "
              "this arm does not take."),
    "eq37Omega": ("Eq. (37)'s aggregation does NOT read this axis, and has "
                  "not since r32. `global_loss` is given `n_c^s / N_s`, the "
                  "observed SOURCE proportions, on every arm -- the flag "
                  "cannot reach them. Until then the weights were "
                  "`omega_c = q_c / sum_c' n_c'^t`, built from the same "
                  "`w_j^t` as Eq. (26)'s `D`, so this call sat behind the flag "
                  "and the shortfall of a SUB-convex sum let the absolute "
                  "confidence level gate the term from outside the score. r32 "
                  "removed that: a weight the model itself determines can be "
                  "lowered instead of earned. So `WEIGHTING_COST` now measures "
                  "Eq. (26)'s `D` alone, which is what this axis has always "
                  "meant, and `weightingCostOmegaSum` is `nan` on every row "
                  "written from 2026-10-18 on."),
    "whyBothBehindOneFlag": ("deliberate, and not this module's decision: a "
                             "previous stretch hung Eq. (37)'s weights behind "
                             "the same flag so the digit ladder's E->F rung "
                             "would isolate ONE mechanism -- the confidence "
                             "reading, wherever it enters -- rather than half "
                             "of one. Both weights are built from the same "
                             "`w_j^t` of Eq. (24)."),
    "notGated": ("Eq. (38)'s own confidence weighting, in the LOCAL term's "
                 "aggregation: `local_loss(distances, w_t, EPSILON_LOCAL)` "
                 "weights each bag's `l_loc,j` by that bag's `w_j^t` and is "
                 "NOT behind this flag -- it is behind `spec[\"local\"]`. It "
                 "therefore ran even while this arm declared `weighting: "
                 "False`, which is why that declaration never meant `sin "
                 "sopesado` in any complete sense and is part of why the "
                 "mis-reading was worth recording. With the flag now true, all "
                 "three confidence readings are live together."),
}

#: What turning the confidence weighting ON costs the global term on THIS
#: toyset, measured rather than reasoned about -- and the reason it has to
#: travel with the tables rather than sit in a docstring.
#:
#: Eq. (37)'s `omega_c = q_c / sum_c' n_c'^t` is SUB-convex: the weights are
#: non-negative and sum to AT MOST one, and they reach one only where every
#: routed bag is maximally confident. On three-class families cut into
#: `Scale.bags_per_class` bags per class the confidences are low, so the sum is
#: far below one and the aggregation divides the term by `1 / sum(omega)`.
#:
#: **Why this is a finding and not a footnote.** A searched `lambda_glob` could
#: absorb it -- `CEILING_RANGE` spans four orders, so a coefficient free to
#: grow is free to grow by 37x. At a coefficient FIXED AT 1 it cannot be
#: absorbed, so the global term enters Eq. (39) smaller than the local one by
#: whatever that factor is on the cell, and a reader comparing the two
#: component shares in the tables is reading that factor as well as the
#: mathematics. `flags()` carries this into `banderas.json` and the generator
#: notebook's flag cell prints it before any number exists.
WEIGHTING_COST = {
    "what": ("Eq. (37)'s sub-convex `omega_c` divides the global term when the "
             "routed confidences are low, and leaves it essentially alone when "
             "they are not. Which of the two happens is per cell and not a "
             "property of the toyset: measured on the pilot of 2026-10 at "
             "TWELVE bags per class, the ratio ran 0.0398 to 1.0411 over the "
             "six cells, and the one above 1 was the cell whose sum(omega) "
             "came back at 0.9994. An earlier wording said the confidences are "
             "low 'here', full stop, which that twelve-bag pilot contradicts "
             "on one cell in six. The declared cut is now 27 bags per class, "
             "so those two bounds are the PRIOR reading and the pilot that "
             "re-measures them is the one that replaces them"),
    "whyItMatters": ("a searched ceiling could absorb a 37x division -- "
                     "config.CEILING_RANGE spans four orders -- but this path "
                     "fixes both coefficients at 1 (`FIXED_COEFFICIENT`), so "
                     "it cannot be absorbed and the global term's share in the "
                     "tables carries the factor"),
    # Where THIS run's own numbers come from, rather than a factor quoted here.
    "measuredBy": ("MIL_CREDA_Benchmark.toy_run.weighting_cost, called by "
                   "run_cell once per cell; every row carries "
                   "weightingCostRatio, weightingCostOmegaSum and "
                   "weightingCostGlobalUnweighted. From 2026-10-18 the ratio "
                   "is Eq. (26)'s `D` alone and omegaSum is nan: r32 took the "
                   "confidences out of Eq. (37)'s weights"),
    # The reading of the stretch that found this, kept and LABELLED as prior
    # because it DID NOT REPRODUCE. It was measured on a pilot whose arm
    # declared `weighting: False` and whose phase-2 search still chose a
    # ceiling and two deltas, so the theta_t it read confidences off is not the
    # one this programme produces. It is kept rather than deleted so a reader
    # who met the old factor somewhere can find out what happened to it.
    "priorMeasurement": {
        "ratioRange": (0.027, 0.683),
        "omegaSumRange": (0.0203, 0.7649),
        "divisionRange": "1.5x to 37x",
        "measuredOn": ("the six pilot cells of this toyset, three families x "
                       "two normalizations, comparing the global term under "
                       "`weighting: True` with the same arm's term under "
                       "`weighting: False` on the same batch and "
                       "bit-identical parameters -- but on an arm TRAINED "
                       "under `weighting: False`, through a phase-2 search "
                       "that still chose a ceiling and two deltas"),
        "reproduced": ("no. Read the live per-row fields instead: the ratio "
                       "depends on how confident this cell's target bags end "
                       "up, which the training this arm now runs moves"),
    },
}

#: What this toyset keeps sharing with the digit benchmark, and why each one is
#: safe to share. The test
#: `tests/test_toy_run.py::test_the_toyset_declares_its_own_arms_and_does_not_
#: derive_them_from_the_digit_benchmark` is what holds the line between this
#: list and the arm declaration above.
SHARED_WITH_DIGIT_BENCHMARK = {
    "wiring.Arm": ("the trainable module itself, constructed from a spec dict "
                   "this module hands it. Safe because the spec is the "
                   "argument: `Arm.__init__(spec, classes, source, target, "
                   "hyper=...)` reads `adaptation`, `weighting`, `local`, "
                   "`unit`, `attention` and `selection` off whatever dict it "
                   "was given and never consults `config.ARMS`. It is "
                   "`wiring.build` -- NOT `Arm` -- that resolves an id through "
                   "`config.ARMS_BY_ID`, and this module no longer calls it."),
    "MIL_CREDA.*": ("every equation. Not merely safe but required: SKILL.md's "
                    "rule is that an arm calls the method's modules and never "
                    "reimplements them, so a toyset with its own copy of a "
                    "term would have forked the formulation."),
    "wiring.Pool / harness.pool_of": ("how a bagset becomes the tensor an arm "
                                      "draws target bags from. Material "
                                      "plumbing with no arm identity in it."),
    "bags.build / bags.roles": ("the draw, the three disjoint roles and the "
                                "manifest. `material_bound` bends the globals "
                                "it reads and restores them; the arm "
                                "declaration is not among them."),
    "harness.accuracy / target_accuracy / balanced_batches / spread": (
        "the readings and the stratified draw. No arm id crosses that seam: "
        "each takes a model and a dataset."),
    "harness.ramp -- NOT shared, and this is the entry that says so": (
        "`ramp` is the digit benchmark's schedule and this path does not call "
        "it. It used to: `adapt_phase_two` evaluated it twice a step, with "
        "`ceiling`/`rampDelta` out of the hyper dict. Both coefficients of Eq. "
        "(39) are now fixed at 1 (`FIXED_COEFFICIENT`), and a fixed 1 is NOT a "
        "ramp evaluated at its ceiling -- `milcreda_ramp` is a warmup that "
        "starts near zero and approaches the ceiling, so routing a constant "
        "through it would leave a schedule nobody declared. The function is "
        "untouched and the digit path still uses it; what changed is that this "
        "path has no caller."),
    "verdict.decide -- NOT shared any more, and this is the entry that says so": (
        "`decide` refuses a winner inside the combined standard error, and this "
        "module used to call it from `ladder_rungs` and `floor_comparison`. "
        "Both are retired with the floor arm, so `verdict` is no longer "
        "imported here at all: one arm declares no pair and there is nothing "
        "for a rule about two dispersions to decide. The module is untouched "
        "and the digit path still uses it. Named rather than deleted from this "
        "list because an entry silently dropped reads as a dependency that was "
        "never there."),
    "config's constants and ranges": ("`KERNEL_SIGMA_RANGE`, `CEILING_RANGE`, "
                                      "`EPSILON`, `REVISION` and the rest. "
                                      "Shared and reported: `sigma_grid` says "
                                      "out loud where a declared constant was "
                                      "measured on digits and is meaningless "
                                      "here, rather than silently using it."),
}

#: The globals `material_bound` bends, with the value each must still hold on
#: entry. An override that found one already moved would be measuring somebody
#: else's draw, so entry asserts rather than assumes.
BENT_GLOBALS = (
    "CLASSES", "BAGS_PER_CLASS", "INSTANCES_PER_BAG",
    "TRAIN_BAGS", "VALID_BAGS", "EVAL_BAGS", "BAGS_PER_STEP", "BACKBONE",
    # `MLP_OUTPUT_DIM` joined this tuple when `DIM` became a searched
    # dimension. `mlp.MLP` reads the global at CONSTRUCTION, so a searched
    # value can only reach the encoder by being bent for the duration of a
    # binding -- and being in this tuple is what makes the bend asserted on
    # entry and restored on the way out, exactly like the other seven.
    "MLP_OUTPUT_DIM",
)


def family_generators() -> dict:
    """CREDA's three hard families by name, straight from `DURAS`."""
    return dict(DURAS)


def build_family(name: str, seed: int, scale: Scale) -> toy_bags.Familia:
    """One family's two domains, their labels and their bag indices.

    `toy_bags.construir` is the one call: it draws the source from `seed`, the
    target independently from `seed + DESPLAZAMIENTO_DE_SEMILLA`, cuts the bags
    off the UNPERTURBED target, and only then perturbs. It returns a frozen
    `Familia` dataclass -- not a 6-tuple -- and carries `Xt_crudo` as a seventh
    field.
    """
    generators = family_generators()
    if name not in generators:
        raise KeyError(f"no family {name!r}; CREDA declares {sorted(generators)}")
    return toy_bags.construir(name, generators[name], semilla=seed,
                              n=scale.points_per_domain,
                              bolsas_por_clase=scale.bags_per_class)


def datasets_of(familia: toy_bags.Familia) -> dict:
    """The bridge, and the whole of it: each domain as a `toy.ToyDataset` whose
    `.subjects` ARE the bag indices `toy_bags.bolsas` computed.

    `ToyDataset`'s own docstring states the contract `bags.build` needs:
    `.targets` is what `bags._labels` reads, `.subjects` is what `bags._select`
    draws from, and `__getitem__` returns `(tensor, label)`. `_select` then
    walks a class subject by subject in sorted order and hands subject `k` to
    bag `k` of that class, which is why the bag a point lands in downstream is
    the band `bolsas` put it in -- the join
    `tests/test_toy_run.py::test_the_bag_indices_that_reach_the_bagset_are_the_ones_bolsas_computed`
    crosses, by reading the built manifest rather than a fixture.
    """
    return {
        "source": toy.ToyDataset(familia.Xs, familia.ys, familia.bs),
        "target": toy.ToyDataset(familia.Xt, familia.yt, familia.bt),
    }


@contextlib.contextmanager
def material_bound(familia: toy_bags.Familia, scale: Scale, backbone: str,
                   hyper: dict | None = None):
    """Bend the globals this toyset needs, and restore every one on the way out.

    **`hyper` is read for the SEARCHED globals only.** Today that is `dim`,
    CREDA's `DIM`: `mlp.MLP` reads `config.MLP_OUTPUT_DIM` at construction, so
    the only way a trial's choice can reach the encoder is for the global to
    hold it while `build_arm` runs. Omitted, every global keeps the value
    `config` declares and this is byte-identical to the pre-search binding --
    which is why every caller that searches nothing passes nothing.

    See the module docstring for why this is a bend and where it breaks. Two
    things make it auditable rather than merely reversible: entry asserts every
    name in `BENT_GLOBALS` still holds the value `config` declares, and
    `bags._raw_dataset` is replaced at the ONE seam that decides which dataset a
    code resolves to -- never `toy.TOY_CODES` and never `toy.build_toy_dataset`,
    which would leave two patches to keep in agreement.
    """
    # The same guard every other entry taking a hyper dict runs, for the same
    # reason: this one now READS that dict (`dim`), so a stale `barrido.json`
    # reaching it carries a programme whose other values are stale too.
    refuse_retired_hyper(hyper)
    declared = {name: getattr(config, name) for name in BENT_GLOBALS}
    unexpected = {name: value for name, value in declared.items()
                  if value != _DECLARED_GLOBALS[name]}
    if unexpected:
        raise RuntimeError(
            f"these `config` globals are not at their declared values on entry: "
            f"{unexpected}; another run of this module is probably bound in this "
            f"same process, and its draw and this one would be each other's")

    split = scale.realized_split()["bagsPerClass"]
    prepared = datasets_of(familia)
    source_code, target_code = FAMILY_CODES[familia.nombre]
    by_code = {source_code: prepared["source"], target_code: prepared["target"]}

    raw_dataset = bags._raw_dataset
    names_added = []
    try:
        config.CLASSES = scale.classes
        config.BAGS_PER_CLASS = scale.bags_per_class
        config.INSTANCES_PER_BAG = _instances_per_bag(familia, scale)
        config.TRAIN_BAGS = split["train"] * scale.classes
        config.VALID_BAGS = split["valid"] * scale.classes
        # The KEY is the record's role name (`test`, CREDA's word, see
        # `realized_split`); the DESTINATION is the digit benchmark's own field
        # name and stays `EVAL_BAGS`. Two vocabularies meet on this line and
        # neither renames the other.
        config.EVAL_BAGS = split["test"] * scale.classes
        config.BAGS_PER_STEP = scale.classes
        config.BACKBONE = backbone
        config.MLP_OUTPUT_DIM = int(
            (hyper or {}).get("dim") or _DECLARED_GLOBALS["MLP_OUTPUT_DIM"])
        for code, dataset in by_code.items():
            if code not in config.DOMAINS:
                config.DOMAINS[code] = DOMAIN_NAMES[code]
                names_added.append(code)

        def _bound_raw_dataset(code: str, root: Path):
            """The declared families for the two bound codes, and a refusal for
            anything else: a code this binding does not name reaching here would
            otherwise fall through to a digit download."""
            del root
            if code not in by_code:
                raise ValueError(
                    f"domain {code!r} is not bound by this run; this binding "
                    f"carries {sorted(by_code)} for family "
                    f"{familia.nombre!r}")
            return by_code[code]

        bags._raw_dataset = _bound_raw_dataset
        yield by_code
    finally:
        bags._raw_dataset = raw_dataset
        for code in names_added:
            config.DOMAINS.pop(code, None)
        for name, value in declared.items():
            setattr(config, name, value)


#: The values `BENT_GLOBALS` must hold outside a binding, read once at import so
#: a nested or overlapping binding is caught instead of inherited.
_DECLARED_GLOBALS = {name: getattr(config, name) for name in BENT_GLOBALS}


def _instances_per_bag(familia: toy_bags.Familia, scale: Scale) -> int:
    """The scale's DECLARED instances per bag, once the draw is shown to fund it.

    It used to return the smallest bag it found, which made the material's unit
    a measurement: a bag held 150 points in `lunas` and 138 in `toroide` at the
    full scale, 30 at a 360-point pilot, and the number travelled nowhere a
    reader would meet it beside the accuracy it shaped. Two arms fed different
    amounts per step are being compared on their optimizer, and so are two
    families and two scales -- so the unit is declared (`Scale.instances_per_bag`)
    and this measures only whether the draw can pay for it.

    `bags._select` takes exactly `config.INSTANCES_PER_BAG` from every subject
    and refuses a subject that cannot fund it, mid-draw, naming the subject
    rather than the scale. The refusal below is that same fact asked EARLY and
    answered in the scale's own terms: quantile bands differ by several points
    when a class does not divide, so the smallest bag and not the arithmetic
    `points_per_domain // bags_total` is what decides, and a scale whose
    smallest bag falls short is a scale that cannot run rather than one that
    runs smaller.
    """
    counts = []
    for labels, indices in ((familia.ys, familia.bs), (familia.yt, familia.bt)):
        for class_id in np.unique(labels):
            for bag in range(scale.bags_per_class):
                counts.append(int(((labels == class_id) & (indices == bag)).sum()))
    smallest = min(counts)
    if smallest < scale.instances_per_bag:
        raise ValueError(
            f"the smallest bag of {familia.nombre!r} holds {smallest} points "
            f"and the {scale.name!r} scale declares "
            f"{scale.instances_per_bag} instances per bag over "
            f"{scale.bags_per_class} bags per class; raise "
            f"`points_per_domain` above {scale.points_per_domain} or lower the "
            f"declared bag, because `bags._select` would refuse this draw "
            f"subject by subject partway through it")
    return scale.instances_per_bag


def material_for(familia: toy_bags.Familia, seed: int, scale: Scale,
                 backbone: str) -> dict:
    """Both domains as `bags.BagSet`s, drawn through `bags.build` itself.

    Nothing about the draw is reimplemented: the permutation, the three disjoint
    roles, the manifest and the contamination seam are `bags.build`'s, and what
    this adds is only which dataset a code resolves to.
    """
    with material_bound(familia, scale, backbone):
        root = Path("/nonexistent-toy-root")   # never touched: see `_raw_dataset`
        source = bags.build(FAMILY_CODES[familia.nombre][0], root, seed)
        target = bags.build(FAMILY_CODES[familia.nombre][1], root, seed)
    return {"source": source, "target": target}


# ------------------------------------------- a measurement that trains nothing

def measuring(measurement):
    """A forward pass taken to MEASURE runs on a deep copy, so it leaves the
    model bit-identical.

    The defect this closes, measured: `run_cell` called `embedding_scales` --
    a diagnostic -- on the freshly built `reference` before handing that same
    object to `train_phase_one`. A fresh `nn.Module` is in `train` mode, so the
    diagnostic's forward re-estimated the encoder's running statistics. The
    doctrine is CREDA's own, at `toy_experiment.py`: a BatchNorm re-estimates
    `running_mean`/`running_var` in the FORWARD, without gradient, so
    `no_grad` is not enough -- it measured the CE over the full source drifting
    3.5e-02 with `||dtheta|| = 0` exact, an encoder frozen in weights and
    moving in behaviour.

    **And the channel is the early stop, not the training.** In `train` mode a
    BatchNorm normalizes by the BATCH statistics, so the running ones never
    enter the training forward and the training-loss column came back
    bit-identical. What they do enter is `harness.accuracy`, which reads in
    `eval` -- so phase 1's early stop chose its best epoch on a criterion a
    diagnostic had contributed to. Measured on `toroide`/`mlp`: `bestEpoch`
    1 -> 0, validation accuracy at epoch 0 0.8667 -> 0.9333, and the weights
    phase 1 then freezes and shares with phase 2 differ by
    `||dtheta|| = 2.548`. Absent under `mlp-nobn` and `mlp-layernorm`, which
    keep no such buffer at all.

    **A copy and NOT `eval`-and-restore, and the choice was measured rather
    than preferred.** Both were authorized. `eval` also leaves the model
    bit-identical, and it additionally RESTATES the reading: these measurements
    are called on a model in `train` mode, where a BatchNorm normalizes by the
    batch, and reading the same rows in `eval` on a freshly built encoder --
    whose running statistics are still `(0, 1)` -- is a different number.
    Measured on `toroide`/`mlp` at `embedding_scales`' own call site:
    `withinBag` 1.6038 -> 0.3570 (0.223x), `betweenBags` 0.8450 -> 0.3886
    (0.460x), `pooled` 1.6926 -> 0.5128 (0.303x). `sigma_grid` is centred on
    those, so the phase-1 search would range over a grid up to 4.5x away from
    every grid this module has ever searched. That is a decision about the
    experiment and not a tidy-up, exactly as `embedding_scales`' own docstring
    says of its `limit`, and it is not this work's to take. On a copy every one
    of the three comes back bit-identical, at both moments the two decorated
    functions are called at -- fresh and in `train`, and after phase 1 in
    `eval`, where the two routes agree anyway.

    **What the copy is NOT a fix for, said out loud.** Reading these in `train`
    mode makes a bag's embedding depend on which other bags shared its chunk --
    the very thing `bag_positions`' docstring refuses for the positions it
    draws. The copy makes the reading harmless; it does not make it
    chunk-independent. Moving it to `eval` is the exit and is the operator's.

    A decorator and not a `with` inside each body, so the claim is visible at
    the signature and the copy has one definition. The model has to be the
    first positional argument and that is CHECKED rather than assumed: a later
    edit that reordered the signature would otherwise leave a decorator that
    silently copies the wrong object and measures the live model anyway.
    """
    @functools.wraps(measurement)
    def wrapper(model, *args, **kwargs):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"`measuring` expects the model as the first positional "
                f"argument of {measurement.__name__} and received "
                f"{type(model).__name__}; a decorator that cannot find the "
                f"model copies nothing and measures the caller's own object")
        return measurement(copy.deepcopy(model), *args, **kwargs)
    return wrapper


# -------------------------------------------------------- the sigma calibration

#: The multipliers the grid is centred on, and the band a kernel value has to
#: land in to be worth searching. The multipliers are the deleted toy notebook's
#: own; the band is declared here so "in band" is a measurement and not a word.
SIGMA_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5)
SIGMA_BAND = (0.05, 0.95)


def _off_diagonal(d: torch.Tensor) -> torch.Tensor:
    return d[~torch.eye(d.shape[0], dtype=torch.bool, device=d.device)]


def _within_bag_median(embeddings: torch.Tensor) -> float:
    """The pairwise median over the set Eq. (14)'s instance kernel evaluates:
    within-bag pairs, pooled over the bags handed in.

    **One definition, two callers**, which is the whole reason this is a
    function rather than two inlined `median()` calls: `embedding_scales`
    reports this as the scale the bandwidth has to serve and
    `frozen_source_sigma` freezes it as the bandwidth itself, and two
    definitions of one statistic are two things nothing obliges to agree.

    **It is CREDA's own formula, restricted to CREDA's own scope.**
    `CREDALoss._median_pairwise` is `sqrt(median(d^2) + 1e-6)` over ONE set, and
    the median commutes with the monotone square root, so over a single bag --
    where "within-bag pairs" and "all pairs" are the same set -- this and
    CREDA's function agree to that epsilon. `test_the_frozen_sigma_reads_the_
    set_the_instance_kernel_evaluates` measures the agreement per bag rather
    than leaving the sentence to be believed. What differs is the SCOPE: the
    median is taken over the UNION of the per-bag pair sets, which is the set
    the kernel reads, and never over the pool of every instance against every
    other, which mixes pairs from different bags and different classes.
    """
    return float(torch.cat([_off_diagonal(torch.cdist(H, H))
                            for H in embeddings]).median())


def _weight_spread(model: wiring.Arm, embeddings: torch.Tensor,
                   sigma: float, device: torch.device) -> float:
    """How far Eq. (16)'s weights sit from uniform, at one bandwidth, averaged
    over the bags handed in.

    The mechanism-level reading the band exists to protect. A bandwidth that
    saturates the instance kernel drives Eq. (15)'s consensus term to the same
    value on every instance of a bag, and a softmax is invariant to a constant
    added to every logit -- so the attention becomes numerically mean pooling
    while every flag still says it is on.

    **It only moves when `attentionGamma` is non-zero**, because the consensus
    term is the only path the bandwidth takes into the logit.
    `config.ATTENTION_GAMMA` is `0.0`, so at the declared hyper this number is
    bit-identical at every bandwidth -- measured, across four orders of
    magnitude. The search ranges gamma over `(0.0, 1.0)` and chose 0.6028 on
    the pilot, which is where the reading has something to say; a reader
    finding this equal across two bandwidths should check gamma before
    concluding the bandwidth does not matter.
    """
    with torch.no_grad():
        return float(torch.stack([
            model.weights_for(H.to(device), sigma).std()
            for H in embeddings]).mean())


@measuring
def embedding_scales(model: wiring.Arm, bagset: bags.BagSet,
                     positions: torch.Tensor, device: torch.device,
                     limit: int = 16) -> dict:
    """The three distances Decision 1's ONE bandwidth has to serve, measured.

    This is where the first version of this function was WRONG, and the error is
    recorded rather than quietly fixed: it measured the median distance over all
    instances pooled together, and that is not the distance the instance kernel
    evaluates. Eq. (14)'s consensus half reads instances WITHIN one bag, and a
    bag is a band of one class, so its spread is much smaller than the pooled
    one -- measured on `lunas`, 0.88 against 1.59. A bandwidth centred on the
    pooled median leaves the within-bag kernel near one and so the attention
    weights of Eq. (16) at a fraction of a percent of uniform: the mechanism is
    numerically off while every flag says it is on.

    - `withinBag` is what Eq. (14) and so Eq. (15)'s consensus evaluate.
    - `betweenBags` is what the bag Gram of Eqs. (32-37) evaluates.
    - `pooled` is every instance against every instance, reported because it is
      what a naive calibration would have used.

    The two the bandwidth has to serve differ by a factor this returns as
    `scaleRatio`, and ONE constant cannot be in band for both. That is Decision
    1's consequence, not a defect to engineer around: it is reported, and the
    searched range is widened to span both so the search decides the trade-off.

    **It goes through `latent.represent` and never through
    `Arm.instance_embeddings` directly.** `tests/test_domain_mixup_guard.py`
    derives, from the syntax tree, that only `wiring.py` and `latent.py` may
    call that method at all -- a target-side reading routed through the SOURCE
    encoder is the one bug no shape check catches, so the call site roster is
    structural. `latent.represent` is the declared seam that states its domain
    and refuses without one, which is exactly what this measurement needs: it
    reads the source bags through theta_s and says so.

    The instance rows come back flattened, so the bag grouping is recovered by
    reshaping -- and that reshape is CHECKED rather than trusted, against
    `represent`'s own bag-unit reading of the same positions. If the flattening
    order were not bag-major the two would disagree, and a within-bag distance
    computed over the wrong grouping is exactly the silent defect this whole
    function exists to have found once already.

    **`limit` is a cap on how many bags are read, and at the declared cut it
    BITES HARD.** It was undocumented while the training role held 6 bags, so
    it never truncated anything and read as a formality. At 12 bags per class
    the role held 21 and the median was taken over the first 16. At the
    declared 27 the role holds 48, so the median is taken over the first 16 of
    48 -- a third of the role, in the role's own order. The value is left where
    it was on purpose -- raising it would move every bandwidth this module has
    ever measured, which is a decision about the experiment and not a tidy-up --
    and the truncation is REPORTED instead: `bagsOffered` is how many were
    handed in, `bagsRead` how many were used, and `bagsTruncated` says whether
    the two differ. A cap that silently starts biting is a measurement that
    quietly changed what it is a median of.
    """
    chosen = positions[:limit]
    rows, _ = latent.represent(model, bagset, chosen, device,
                               domain="source", unit="instance")
    count = int(len(chosen))
    if rows.shape[0] % count:
        raise ValueError(
            f"{rows.shape[0]} instance rows over {count} bags does not divide; "
            "the grouping cannot be recovered")
    per_bag = rows.shape[0] // count
    embeddings = rows.reshape(count, per_bag, -1)

    reference, _ = latent.represent(model, bagset, chosen, device,
                                    domain="source", unit="bag")
    with torch.no_grad():
        Z, _ = model.bag_representations(embeddings.to(device),
                                        config.KERNEL_SIGMA)
    gap = float((Z.float().cpu() - reference.float()).abs().max())
    if gap > 1e-5:
        raise ValueError(
            f"the reshaped grouping disagrees with `latent.represent`'s own "
            f"bag reading by {gap}; the instance rows are not bag-major")

    with torch.no_grad():
        pooled = _off_diagonal(torch.cdist(rows, rows))
        between = _off_diagonal(torch.cdist(reference, reference))
    within_median = _within_bag_median(embeddings)
    # The weights at the bandwidth the grid is about to be centred on, not
    # at the declared constant: what travels is how far Eq. (16) is from
    # uniform where this run will actually put it.
    spread = _weight_spread(model, embeddings, within_median, device)
    uniform = 1.0 / per_bag
    return {
        "withinBag": within_median,
        "betweenBags": float(between.median()),
        "pooled": float(pooled.median()),
        "bagsRead": count,
        # What the cap did, said out loud. `bagsRead` alone cannot tell a role
        # of sixteen bags from a role of twenty-one read sixteen deep.
        "bagsOffered": int(len(positions)),
        "bagsTruncated": bool(len(positions) > count),
        "bagLimit": int(limit),
        "instancesPerBag": per_bag,
        "groupingCheckedAgainstBagUnit": gap,
        # How far Eq. (16)'s weights are from uniform at this bandwidth. A
        # spread of 1e-4 against a uniform 1/m says the attention is numerically
        # mean pooling, whatever the flags say, so the number travels.
        "weightSpread": spread,
        "uniformWeight": uniform,
        "weightSpreadRelativeToUniform": spread / uniform if uniform else 0.0,
    }


#: What multiplies the frozen bandwidth when no search chose a multiplier.
#: Neutral and NOT `config.KERNEL_SIGMA`: that constant is 36.135, measured on
#: digits under resnet18, and the whole point of deriving the bandwidth is that
#: a number from another experiment's scale decides nothing here. A caller with
#: no search in hand runs at the statistic itself.
SIGMA_MULTIPLIER_NEUTRAL = 1.0

#: Where the frozen bandwidth's statistic comes from, as the record names it.
#: One string, so the record says which function produced the number instead of
#: leaving a reader to infer it from the value.
FROZEN_SIGMA_STATISTIC = "CREDA.toy_experiment.sigma_de_la_fuente"

#: The SET that statistic is taken over, and the half that was WRONG until this
#: stretch. CREDA takes its pairwise median over the set its own kernel sees --
#: points. Here Eq. (14)'s instance kernel evaluates pairs of instances WITHIN a
#: bag, and the first version of this function took the median over the whole
#: training role pooled, which mixes pairs from different bags and different
#: classes and so reads 17x to 77x the distance the kernel actually measures.
#: Measured on the pilot record, 4 of the 9 cells then fell outside
#: `SIGMA_BAND`, all four saturated, at the smallest multiplier the grid offers.
#:
#: The statistic is CREDA's and stays CREDA's; only the scope is corrected. Both
#: scopes travel in the record, with the ratio, so the correction is readable
#: rather than a number that silently moved.
FROZEN_SIGMA_SCOPE = ("within-bag instance pairs, pooled over the source "
                      "training role's bags")
FROZEN_SIGMA_POOLED_SCOPE = ("every instance against every instance over the "
                             "source training role, which is NOT the set "
                             "Eq. (14) evaluates")


@measuring
def frozen_source_sigma(model: wiring.Arm, bagset: bags.BagSet,
                        positions: torch.Tensor,
                        device: torch.device,
                        multiplier: float = SIGMA_MULTIPLIER_NEUTRAL) -> dict:
    """CREDA's own bandwidth, over the set CREDA's own kernel sees: the pairwise
    median of WITHIN-BAG instance pairs across the source's full train.

    **CREDA's statistic, with its scope corrected -- not a second statistic.**
    See `FROZEN_SIGMA_SCOPE` for what was wrong and what it cost. The formula is
    `CREDALoss._median_pairwise`'s, and `_within_bag_median` carries the one
    definition both this and `embedding_scales` read; over a single bag, where
    the two scopes are the same set, the two agree to the epsilon CREDA's own
    formula carries, and a test measures that per bag rather than asserting it.
    `sigma_de_la_fuente` is still CALLED here, on the pooled scope, because the
    reading this correction left has to stay visible beside the one that
    governs: `pooledSigma` is the number every pre-correction run ran at.

    **Source only, and that is what makes it worth deriving.** Nothing injected
    into the target -- an intruder, a relocation, contamination -- can move it,
    so damage entering through the class block stops being confused with damage
    entering through the unit of measurement. `readsTargetRows` is returned as a
    flat `False` rather than left implicit, because "the statistic is immune" is
    a claim and a reader is owed the field that carries it.

    **The FULL train, and the cap is refused rather than tolerated.**
    `embedding_scales` takes a `limit` of 16 bags, which at the declared cut is
    a third of the training role; this hands it the whole role and raises if the
    reading comes back truncated. A frozen bandwidth measured over a third of
    its own role, in the role's own order, is a bandwidth whose value depends on
    a cap nobody chose.

    **Measured AFTER phase 1 and frozen, which is where CREDA measures it.**
    `corrida` computes `sigma_fijo` from `enc(D['Xs_tr'])` the moment phase 1
    ends and the source encoder is frozen, with the comment that after phase 2
    it can no longer be computed at all. This is that moment in this
    benchmark's unit: `positions` is the source training role and `theta_s` is
    frozen, so the number cannot move again. The scope changed in this stretch;
    the moment and the freeze did not.

    **`multiplier` is read, never applied.** What it buys is that the two
    readings below -- the kernel value and Eq. (16)'s weight spread -- are
    reported at the bandwidth phase 2 will ACTUALLY run at rather than at the
    bare statistic. The returned `sigma` is the statistic itself, and
    `run_cell` is still what multiplies it; a function that applied its own
    multiplier would be a second place the phase-2 bandwidth is decided.

    It reads through `latent.represent` and never through
    `Arm.instance_embeddings` directly, for the reason `embedding_scales` gives:
    the call-site roster is structural, and `represent` is the declared seam
    that states its domain and refuses without one.
    """
    scales = embedding_scales(model, bagset, positions, device,
                              limit=int(len(positions)))
    if scales["bagsTruncated"]:
        raise ValueError(
            f"the frozen bandwidth was measured over {scales['bagsRead']} of "
            f"{scales['bagsOffered']} bags; a cap must not reach this reading")
    rows, _ = latent.represent(model, bagset, positions, device,
                               domain="source", unit="instance")
    detached = rows.detach()
    embeddings = detached.reshape(scales["bagsRead"],
                                  scales["instancesPerBag"], -1)

    within = scales["withinBag"]
    pooled = sigma_de_la_fuente(detached)
    at_within = multiplier * within
    at_pooled = multiplier * pooled
    uniform = scales["uniformWeight"]

    def kernel(sigma: float) -> float:
        return math.exp(-(within ** 2) / (2.0 * sigma ** 2))

    def relative_spread(sigma: float) -> float:
        spread = _weight_spread(model, embeddings, sigma, device)
        return spread / uniform if uniform else 0.0

    within_relative = relative_spread(at_within)
    pooled_relative = relative_spread(at_pooled)
    return {
        "sigma": within,
        "statistic": FROZEN_SIGMA_STATISTIC,
        "scope": FROZEN_SIGMA_SCOPE,
        # The reading this correction left, computed on the same rows at the
        # same moment, so what the scope bought is a subtraction a reader can
        # perform rather than a claim they have to take.
        "pooledSigma": pooled,
        "pooledScope": FROZEN_SIGMA_POOLED_SCOPE,
        "pooledToWithinRatio": pooled / within if within else float("nan"),
        "multiplierRead": multiplier,
        # What the instance kernel reads at each of the two bandwidths. The
        # first is `exp(-1/(2m^2))` and so in band for every multiplier the grid
        # offers -- an identity, and said so where it is recorded, because a
        # band satisfied by construction proves the scope and never the
        # mechanism.
        "kernelWithinBag": kernel(at_within),
        "inBand": SIGMA_BAND[0] <= kernel(at_within) <= SIGMA_BAND[1],
        "pooledKernelWithinBag": kernel(at_pooled),
        "pooledInBand": SIGMA_BAND[0] <= kernel(at_pooled) <= SIGMA_BAND[1],
        "bandIsAnIdentityNote": (
            "the within-bag kernel at m times the within-bag median is "
            "exp(-1/(2 m^2)) whatever the measured distance is, so `inBand` "
            "holds by construction once the scope is right. It reports that "
            "the SCOPE is right and says nothing about the mechanism; "
            "`weightSpreadGainOverPooled` is the reading with a reachable red"),
        # The mechanism the band exists to protect, measured by taking the
        # credit away: the same weights on the same embeddings at the scope
        # this correction left.
        "weightSpreadRelativeToUniform": within_relative,
        "pooledWeightSpreadRelativeToUniform": pooled_relative,
        "weightSpreadGainOverPooled": (within_relative / pooled_relative
                                       if pooled_relative else float("nan")),
        "weightSpreadNote": (
            "zero gain means `attentionGamma` is zero, not that the bandwidth "
            "does not matter: Eq. (15)'s consensus term is the only path the "
            "bandwidth takes into Eq. (16), and config.ATTENTION_GAMMA is 0.0"),
        "attentionGamma": float(model.attention_gamma),
        "bags": int(len(positions)),
        "instances": int(detached.shape[0]),
        "instancesPerBag": int(scales["instancesPerBag"]),
        "dims": int(detached.shape[1]),
        "bagsTruncated": bool(scales["bagsTruncated"]),
        # Not implicit. The immunity is the reason this statistic is worth
        # computing, so the record carries the fact rather than the argument.
        "readsTargetRows": False,
        "measuredAt": "after phase 1, with theta_s frozen",
        # The full within-bag reading this was taken from, on the TRAINED
        # encoder over the whole role -- distinct from `sigma`'s own `scales`
        # block in the record, which is the pre-phase-1 bootstrap reading
        # capped at 16 bags.
        "withinBagReading": scales,
    }


def sigma_grid(scales: dict) -> dict:
    """The bandwidth grid centred on the measured WITHIN-BAG distance, and what
    the kernel reads there and between bags.

    `config.KERNEL_SIGMA` is 36.135 and was measured on digits under resnet18;
    `config.KERNEL_SIGMA_RANGE` is that constant over a decade either way, so it
    drags the wrong scale with it. An ABSOLUTE grid cannot be in band for six
    cells at once -- the three families live on scales that differ by orders --
    so the grid is built from each cell's own measurement instead.

    The within-bag kernel value at the within-bag median depends only on the
    multiplier: `exp(-d^2 / (2 (m d)^2))` is `exp(-1 / (2 m^2))`, so the four
    points read 0.135/0.411/0.607/0.801 whatever the measured distance is. That
    identity is what makes the grid in band BY CONSTRUCTION, and it is asserted
    against the measured values rather than written down as four literals.

    `range` deliberately spans BOTH measured scales, and it takes the SMALLER
    median for its floor and the LARGER for its ceiling rather than assuming
    which is which. That is not defensive styling: measured over the six cells
    of this toyset, the between-bag median is larger than the within-bag one in
    three of them and SMALLER in the other three (`rollo`/`batchnorm` 1.00
    against 1.12, `toroide`/`batchnorm` 1.11 against 1.28). A range written as
    `(0.5 * within, 1.5 * between)` would therefore invert on a cell where the
    ratio fell below the multiplier ratio, and `optuna` raises on an inverted
    range -- a crash that would have waited for the full scale to arrive.
    """
    within = scales["withinBag"]
    between = scales["betweenBags"]
    points = []
    for multiplier in SIGMA_MULTIPLIERS:
        sigma = multiplier * within
        points.append({
            "multiplier": multiplier,
            "sigma": sigma,
            "kernelWithinBag": math.exp(-(within ** 2) / (2.0 * sigma ** 2)),
            "kernelBetweenBags": math.exp(-(between ** 2) / (2.0 * sigma ** 2)),
            "inBand": (SIGMA_BAND[0]
                       <= math.exp(-(within ** 2) / (2.0 * sigma ** 2))
                       <= SIGMA_BAND[1]),
        })
    return {
        "scales": scales,
        "centredOn": "withinBag",
        "multipliers": list(SIGMA_MULTIPLIERS),
        "band": list(SIGMA_BAND),
        "points": points,
        "range": (SIGMA_MULTIPLIERS[0] * min(within, between),
                  SIGMA_MULTIPLIERS[-1] * max(within, between)),
        "scaleRatio": between / within if within else float("nan"),
        "declaredRange": list(config.KERNEL_SIGMA_RANGE),
        "declaredConstant": config.KERNEL_SIGMA,
        "declaredKernelWithinBag": math.exp(
            -(within ** 2) / (2.0 * config.KERNEL_SIGMA ** 2)),
        "replaced": ("`config.KERNEL_SIGMA_RANGE` was measured on digits under "
                     "resnet18 and is meaningless on these families; the range "
                     "searched here spans both measured scales"),
        "oneBandwidthTwoJobs": ("the within-bag and between-bag medians differ "
                                "by `scaleRatio`, and Decision 1 declares ONE "
                                "bandwidth for both; no single value is in band "
                                "for both and the search chooses the trade-off. "
                                "Which of the two is larger is NOT fixed: it "
                                "varies by family and by backbone, so neither "
                                "the range nor any reader may assume a "
                                "direction"),
    }


# ------------------------------------------------------------- the eight ranges

#: Which phase searches each of the eight declared ranges, and why the split is
#: where it is. Phase 1's supervised term (Eq. 21 over Eq. 19's bag
#: representation) READS `kernelSigma`, `attentionGamma` and
#: `attentionTemperature` -- `wiring.Arm.weights_for`/`bag_representations` take
#: them -- so a phase-2 trial that varied them would be varying phase 1 too, and
#: phase 1 would stop being trained once and frozen. They therefore belong to
#: the phase-1 search, chosen on SOURCE VALID accuracy and zero target labels,
#: exactly as CREDA's `buscar_hp1` chooses its own. `tauLocal` reaches only the
#: local term and stays in phase 2 with the four schedule scalars.
PHASE_ONE_DIMENSIONS = ("kernelSigma", "attentionGamma", "attentionTemperature")
PHASE_TWO_DIMENSIONS = ("tauLocal",)

#: Where each SEARCHED dimension's range comes from. FOUR entries, which is
#: four of the eight declared ranges; `RETIRED_RANGES` names the other four and
#: says why each one left. Nothing is invented, and nothing is left unsearched
#: without being named: the two dicts partition `config`'s eight `*_RANGE`
#: constants, which `tests/test_toy_run.py` asserts rather than trusting.
#:
#: An earlier version of this comment read "Eight entries, which is every
#: declared range: nothing is left unsearched and nothing is invented." That
#: sentence was true when eight were searched and became false the moment four
#: of them stopped deciding anything; it is replaced rather than left to age.
SEARCH_RANGES = {
    "kernelSigma": "measured median, see `sigma_grid`",
    "attentionGamma": config.ATTENTION_GAMMA_RANGE,
    "attentionTemperature": config.ATTENTION_TEMPERATURE_RANGE,
    "tauLocal": config.TAU_LOCAL_RANGE,
}

#: The four declared ranges this path does NOT search, each with the thing it
#: parameterizes and the reason it decides nothing here.
#:
#: All four are Eq. (39)'s schedule: `ceiling` IS `lambda_glob` and
#: `ceilingLocal` IS `lambda_loc` -- `harness.ramp` multiplies its warmup curve
#: by the ceiling, so the ceiling is the coefficient the curve approaches --
#: and the two deltas are how fast each curve gets there. With both
#: coefficients fixed at exactly 1 and no curve, a value for any of the four
#: changes no number that this path computes.
#:
#: Naming them is the whole point. A key that decides nothing and is silently
#: accepted is indistinguishable from one that decides something, so
#: `refuse_retired_hyper` raises on each of these BY NAME.
RETIRED_RANGES = {
    "ceiling": {
        "range": config.CEILING_RANGE,
        "was": "lambda_glob of Eq. (39), as harness.ramp's ceiling",
        "why": ("the coefficient is fixed at 1 (`FIXED_COEFFICIENT`), so there "
                "is no ceiling for a curve to approach"),
    },
    "ceilingLocal": {
        "range": config.CEILING_LOCAL_RANGE,
        "was": "lambda_loc of Eq. (39), as harness.ramp's ceiling",
        "why": ("the same, for the local term's coefficient: also fixed at 1"),
    },
    "rampDelta": {
        "range": config.RAMP_DELTA_RANGE,
        "was": "how fast lambda_glob grows, as harness.ramp's delta",
        "why": ("nothing grows: no ramp is evaluated on this path, so the "
                "steepness of a curve that does not exist is not a dimension"),
    },
    "rampDeltaLocal": {
        "range": config.RAMP_DELTA_LOCAL_RANGE,
        "was": "how fast lambda_loc grows, as harness.ramp's delta",
        "why": ("the same, for the local term's curve"),
    },
}

#: Eq. (39)'s two coefficients on this path: `L = L_src + lambda_glob * L_glob
#: + lambda_loc * L_loc` with BOTH lambdas exactly 1, fixed, with no ramp.
#:
#: **A fixed 1 is not a ramp evaluated at its ceiling, which is why
#: `harness.ramp` is not called with this value instead of being dropped.**
#: `milcreda_ramp` is a WARMUP: `(1 - e^{-delta p}) / (1 + e^{-delta p})` times
#: the ceiling, which is ~0 at the first step and approaches the ceiling from
#: below without reaching it. Handing it `ceiling=1.0` would train under a
#: schedule from 0 to just under 1 -- a schedule nobody declared -- and would
#: report it as a fixed coefficient. So the two calls are gone and this
#: constant reaches `wiring.Arm.training_step` directly, which is what that
#: method wants: two plain floats, `ramp` and `ramp_local`.
FIXED_COEFFICIENT = 1.0

#: The phase-1 optimizer's own two scalars, CREDA's `buscar_hp1` range for each.
PHASE_ONE_OPTIMIZER_RANGES = {"lr1": (1e-4, 1e-1), "weightDecay1": (1e-7, 1e-2)}

#: CREDA's own `DIM` categorical, from `buscar_hp1`: the encoder's output
#: dimension, which is also the dimension every kernel block of Eqs. (32-37) is
#: evaluated in. Four choices and not a continuous range, because that is what
#: CREDA declares and because `mlp.MLP` builds a layer of exactly this width.
#:
#: **It was fixed at `config.MLP_OUTPUT_DIM = 16` and the stated reason for not
#: searching it was that bending a global per trial inside a search was out of
#: bounds.** `material_bound` already bends seven globals per binding with an
#: entry assertion and a `finally` restore, so the bend is the mechanism this
#: module already has rather than a new one; `MLP_OUTPUT_DIM` simply joined
#: `BENT_GLOBALS`.
PHASE_ONE_DIM_CHOICES = (8, 16, 32, 64)

#: Phase 2's own two optimizer scalars, CREDA's `buscar_hp2` range for each.
#: Before these existed, `scale.lr_2` ran at its `1e-3` default and was never
#: tuned, and `adapt_phase_two`'s Adam carried no `weight_decay` argument at
#: all -- which is not a searched zero, it is an absent dimension.
PHASE_TWO_OPTIMIZER_RANGES = {"lr2": (1e-5, 1e-2), "weightDecay2": (1e-7, 1e-2)}

#: What CREDA searches in phase 1 and this path cannot, with the measurement
#: that says why rather than an assertion.
PHASE_ONE_NOT_SEARCHED = {
    "LOTE_1": ("CREDA sweeps `[32, 64, 128, 256]` POINTS per step; the unit "
               "here is the BAG and there is no lot to choose. "
               "`harness.balanced_batches` yields exactly one bag per class "
               "per step -- it takes no lot argument and reads no "
               "`config.BAGS_PER_STEP` -- so the training lot in bags IS the "
               "class count. `config.BAGS_PER_STEP` divides the role length "
               "to get the STEP COUNT of an epoch, which is an epoch length "
               "and not a lot, so searching it would search a repetition "
               "field wearing another name. Making the lot searchable means "
               "giving `balanced_batches` a per-class count, which is the "
               "DIGIT benchmark's own batching and outside this toyset's "
               "surface. A range derived from the realized train count would "
               "be `(3, 6, 12, 24, 48)` -- one bag per class doubling to the "
               "whole 48-bag training role -- and it is written here rather "
               "than searched because nothing can consume it yet"),
}

#: The same, for phase 2. `LOTE_2` is blocked by the identical fact, in the
#: identical place: `adapt_phase_two` draws its steps from the same
#: `harness.balanced_batches`.
PHASE_TWO_NOT_SEARCHED = {
    "LOTE_2": ("the same block as `LOTE_1`: `harness.balanced_batches` fixes "
               "the lot at one bag per class and takes no lot argument, and "
               "`adapt_phase_two` draws its steps from that same generator. "
               "CREDA's `[32, 64, 128, 256]` are POINTS and do not transfer "
               "to a bag-unit axis"),
}


def searched_ranges(phase: str) -> dict:
    """Every dimension this path searches in `phase`, with where its range
    comes from.

    One place, so the record and the searches cannot name different sets. The
    record carries this verbatim: a record that named dimensions and not ranges
    would say a value was chosen and not from where, and a reader could not tell
    a decade-wide sweep from a two-point one.
    """
    if phase == "phase1":
        return {
            **{name: SEARCH_RANGES[name] for name in PHASE_ONE_DIMENSIONS},
            **{name: tuple(span) for name, span
               in PHASE_ONE_OPTIMIZER_RANGES.items()},
            "dim": tuple(PHASE_ONE_DIM_CHOICES),
        }
    if phase == "phase2":
        return {
            **{name: SEARCH_RANGES[name] for name in PHASE_TWO_DIMENSIONS},
            **{name: tuple(span) for name, span
               in PHASE_TWO_OPTIMIZER_RANGES.items()},
        }
    raise KeyError(f"no phase {phase!r}; this module declares phase1 and phase2")


#: How thin the per-dimension exploration got, as the artefact has to say it.
#: The trial budget did NOT move -- 25 per phase at the full scale, the same
#: `trials_1`/`trials_2` CREDA declares -- while phase 1 went from 5 searched
#: dimensions to 6 and phase 2 from 1 to 3. This repository has already
#: measured that the variance between two searches can swallow an effect, so a
#: reader comparing two cells is owed the budget per dimension and not only the
#: budget.
SEARCH_BUDGET_NOTE = {
    "trialsUnchanged": True,
    "phase1DimensionsBefore": 5,
    "phase1DimensionsAfter": 6,
    "phase2DimensionsBefore": 1,
    "phase2DimensionsAfter": 3,
    "expectedByTheBrief": {"phase1": 7, "phase2": 4},
    "shortfall": ("the bag lot of each phase -- CREDA's `LOTE_1`/`LOTE_2` -- "
                  "is not reachable from this module; see "
                  "`PHASE_ONE_NOT_SEARCHED` for the measurement"),
    "whyItMatters": ("the trial budget is fixed, so every dimension added "
                     "divides the same 25 trials further. Phase 1 now spends "
                     "25 trials over 6 dimensions and phase 2 over 3, where "
                     "they spent them over 5 and 1; a gap between two cells "
                     "smaller than the variance between two searches of this "
                     "budget is not a reading of the method"),
}


def refuse_retired_hyper(hyper: dict | None) -> None:
    """Raise if a hyper dict carries one of `RETIRED_RANGES`' four names.

    `CREDA.toy_experiment.corrida`'s guard against `TECHO` and `DELTA`, built
    against this path's names, and for the identical reason its own comment
    gives: "un `barrido.json` anterior los trae, y quien los manda cree que
    decide algo". A stale artefact of this toyset carries `ceiling`,
    `ceilingLocal`, `rampDelta` and `rampDeltaLocal` -- `MIL-CREDA/Notebooks/
    toy_search.json` on this disk still does, with `ceiling` at 0.01568 -- and
    a caller who reads one back and passes it in believes it is reproducing the
    run those values came from.

    **Refused, not ignored.** Ignoring is the failure mode: the key arrives,
    changes nothing, and nothing anywhere says so, so the run reads as the
    searched-ceiling run it is not. The name is in the message because the
    caller's next question is which key, and a message naming only "a retired
    key" sends them to read this source to find out.

    Called from every public entry on this path that takes a hyper dict --
    `build_arm`, `train_phase_one`, `adapt_phase_two`, `run_cell` -- rather
    than from one choke point, because a choke point is one early return away
    from being bypassed and each of those four is reachable on its own from a
    notebook and from the searches.
    """
    for name in RETIRED_RANGES:
        if name in (hyper or {}):
            retired = RETIRED_RANGES[name]
            raise ValueError(
                f"{name} is retired on this path: it was {retired['was']}, and "
                f"{retired['why']}. Both of Eq. (39)'s coefficients are fixed "
                f"at {FIXED_COEFFICIENT} with no ramp, so this value decides "
                f"nothing; received {(hyper or {})[name]!r}. A hyper dict "
                f"carrying it comes from a search made when it did decide "
                f"something, so the rest of that dict is from another "
                f"programme too. Drop the key rather than setting it to a "
                f"value that looks harmless.")


def declared_hyper(scale: Scale) -> dict:
    """Every searchable dimension at the value `config` declares for it.

    What a run with nothing searched trains at, and the fallback every search
    starts from. `scale` is taken so a caller cannot assemble this without
    having one in hand.

    **None of `RETIRED_RANGES`' four names appears here**, and that is load
    bearing rather than tidy: this dict is the base every search and every cell
    composes on top of, so a retired key emitted here would be handed straight
    into `refuse_retired_hyper` by the module's own callers and nothing would
    run at all. It used to emit all four, at `config.RAMP_CEILING` and
    `config.RAMP_DELTA`.
    """
    del scale
    return {
        # The PHASE-1 bootstrap only, and the last resort for it: `_resolved_
        # sigma` replaces this with a measured within-bag median on every
        # searched path, and `run_cell` replaces `kernelSigma` outright with
        # the frozen statistic the moment phase 1 ends. A run that trained
        # phase 2 at 36.135 would be running the pre-T2 path, and
        # `cell["sigma"]["declaredConstantUsed"]` is the field that says so.
        "kernelSigma": config.KERNEL_SIGMA,
        "attentionGamma": config.ATTENTION_GAMMA,
        "attentionTemperature": config.ATTENTION_TEMPERATURE,
        "tauLocal": config.TAU_LOCAL,
        "lr1": None,
        "weightDecay1": 0.0,
        # `None` for the two rates and `0.0` for the two decays, which is what
        # each one's absence meant before it was searchable: `scale.lr_1`/
        # `scale.lr_2` take over on `None`, and the decay really was zero.
        "dim": config.MLP_OUTPUT_DIM,
        "lr2": None,
        "weightDecay2": 0.0,
    }


# ------------------------------------------------------------------- the phases

def build_arm(arm_id: str, material: dict, scale: Scale, hyper: dict,
              device: torch.device, seed: int) -> wiring.Arm:
    """One arm of THIS toyset's own declaration, on this toyset's material.

    `wiring.Arm` is constructed directly, with a spec out of `TOY_ARMS_BY_ID`.
    `wiring.build` is deliberately not called: it resolves its `arm_id` through
    `config.ARMS_BY_ID` and so cannot build an arm the DIGIT benchmark does not
    declare -- which is the whole of what made this module borrow that
    declaration. `Arm.__init__` takes the spec dict as its first argument and
    reads every key it needs off it, so handing it this module's own is not a
    workaround: it is the seam the class already offers.

    Seeded immediately before construction so every arm of a cell draws
    identical shared parameters: `wiring.Arm.__init__` constructs theta_t LAST
    precisely so the encoder, the head and Eq. (15)'s parameters come out
    bit-identical between any two arms of one cell. With one arm declared that
    is a property nothing currently exercises and the seeding stays, for
    `run_cell`'s reason: it is the protocol's, not the arm count's.
    """
    refuse_retired_hyper(hyper)
    if arm_id not in TOY_ARMS_BY_ID:
        raise ValueError(
            f"unknown toyset arm {arm_id!r}; this module declares "
            f"{list(TOY_ARMS_BY_ID)}. The digit benchmark's own arms "
            f"({list(config.ARMS_BY_ID)}) are not this declaration and are not "
            f"reachable from here")
    torch.manual_seed(seed)
    source, target = material["source"], material["target"]
    return wiring.Arm(
        TOY_ARMS_BY_ID[arm_id], scale.classes,
        harness.pool_of(source, source.train_idx, device),
        harness.pool_of(target, target.train_idx, device),
        hyper=hyper,
    ).to(device)


def shared_phase_one_state(model: wiring.Arm) -> dict:
    """The parameters phase 1 fits: everything but theta_t.

    Snapshotted by name so the identical tensors can be loaded into every arm of
    the cell -- phase 1 is trained ONCE per (family, normalization, seed) and
    shared, because an arm that re-ran it would differ from its neighbour in
    phase 1 as well and the rung would stop being attributable.
    """
    return {name: value.detach().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("target_encoder.")}


def supervised_loss(model: wiring.Arm, x: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:
    """Eq. (21) over bag scores, which is phase 1's whole objective.

    `MIL_CREDA.objective.source_loss` is the equation and it is called, never
    rewritten. The three lines around it mirror `wiring.Arm.training_step`'s own
    bag branch; `training_step` itself is not usable here because it also builds
    the adaptation term, which would pass target bags through theta_t and let
    phase 1 train the encoder phase 2 is supposed to start from.
    """
    logits = model(x)
    return source_loss(
        F.softmax(logits, dim=1),
        F.one_hot(y, model.classes).to(logits.dtype),
        config.EPSILON,
    )


def _batch(dataset, positions, device):
    items = [dataset[i] for i in positions]
    x = torch.stack([item[0] for item in items]).to(device)
    y = torch.tensor([item[1] for item in items], device=device)
    return x, y


def train_phase_one(model: wiring.Arm, source_train, source_valid,
                    device: torch.device, scale: Scale, hyper: dict,
                    generator: torch.Generator) -> dict:
    """Phase 1: the supervised term alone, early-stopped on SOURCE VALID
    accuracy, best weights restored, then frozen.

    CREDA's `train_source_ce` in this benchmark's unit. Three things it keeps
    verbatim in shape, each because CREDA found the alternative wrong:

    - the stop reads VALID and never the role the table publishes;
    - the WEIGHTS of the best epoch are restored, not merely its number, and a
      `_restored` reading proves the load landed;
    - the freeze happens here, because every guarantee phase 2 offers rests on
      it.

    Phase 1 reads none of `RETIRED_RANGES`' four names and never did -- they
    were phase 2's schedule. It refuses them anyway: a dict carrying one came
    from a search of another programme, so every OTHER value in it (`lr1`,
    `weightDecay1`, `kernelSigma`) is from that programme too, and those this
    function does read.
    """
    refuse_retired_hyper(hyper)
    parameters = [p for name, p in model.named_parameters()
                  if not name.startswith("target_encoder.")]
    optimizer = torch.optim.Adam(
        parameters,
        lr=hyper.get("lr1") or scale.lr_1,
        weight_decay=hyper.get("weightDecay1", 0.0),
    )
    steps = max(1, -(-len(source_train) // config.BAGS_PER_STEP))

    history: list[tuple] = []
    best, best_epoch, best_state = -1.0, -1, None
    started = time.perf_counter()
    for epoch in range(scale.epochs_1):
        model.train()
        total, batches = 0.0, 0
        for positions in harness.balanced_batches(source_train.targets, steps,
                                                  generator):
            x, y = _batch(source_train, positions, device)
            loss = supervised_loss(model, x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # r31 Eq. (15): the l1 constraint on `v_R` is a max norm imposed
            # after each OPTIMIZER step. Here and not inside `training_step`
            # for the same reason `sync_target_from_source` is here -- the rule
            # counts optimizer steps, and projecting before the update would
            # constrain the value the step is about to move.
            model.project_constraints()
            total += float(loss.detach())
            batches += 1
        accuracy = harness.accuracy(model, source_valid, device)
        model.eval()
        with torch.no_grad():
            xv, yv = _batch(source_valid, range(len(source_valid)), device)
            valid_loss = float(supervised_loss(model, xv, yv))
        history.append((epoch, total / batches if batches else float("nan"),
                        valid_loss, accuracy))
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            best_state = shared_phase_one_state(model)
        if epoch - best_epoch >= scale.patience_1:
            break

    if best_state is None:
        raise RuntimeError("phase 1 completed no epoch; there is nothing to freeze")
    model.load_state_dict(best_state, strict=False)
    model.eval()
    restored = harness.accuracy(model, source_valid, device)
    model.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)
    return {
        "epochs": len(history),
        # The rate the forecast is built from, measured here because this is the
        # only place that knows how many epochs actually ran: both stops end a
        # phase before its declared length, so dividing the whole sweep by the
        # declared length would forecast from a length nothing ran.
        "secondsPerEpoch": ((time.perf_counter() - started) / len(history)
                            if history else float("nan")),
        "bestEpoch": best_epoch,
        "bestValidAccuracy": best,
        "restoredValidAccuracy": restored,
        "history": history,
        "state": shared_phase_one_state(model),
    }


def copy_source_to_target(model: wiring.Arm) -> None:
    """The one copy: theta_t starts bit-identical to theta_s.

    `Arm.sync_target_from_source` carries the PARAMETERS -- it deliberately
    leaves the buffers alone, because carrying normalization statistics over at
    every step of a leash measured worse. This is not that leash: it is CREDA's
    `clone_for_target`, which happens exactly once and does carry the buffers,
    and it has to, or a BatchNorm theta_t would begin at `running_mean = 0` and
    `running_var = 1` and so would NOT be bit-identical to the encoder it is
    supposed to start as.
    """
    if model.target_encoder is None:
        return
    model.sync_target_from_source()
    source = dict(model.encoder.named_buffers())
    with torch.no_grad():
        for name, buffer in model.target_encoder.named_buffers():
            buffer.copy_(source[name])


def set_phase_two_modes(model: wiring.Arm) -> None:
    """Source frozen and in `eval`, theta_t the only module in `train`.

    `eval` and not merely `no_grad`: `no_grad` stops the gradient and does not
    stop BatchNorm from re-estimating its running statistics in the forward. A
    source encoder left in `train` would have its buffers move during phase 2,
    and `sourceBuffersMoved` would report a freeze that did not hold -- which is
    exactly what CREDA's own buffer assertion exists to catch.
    """
    model.eval()
    if model.target_encoder is not None:
        model.target_encoder.train()


def watchman(model: wiring.Arm, source_valid, device: torch.device) -> float:
    """CREDA's `vigia`: the head through the TARGET encoder, over the SOURCE
    validation role, against the SOURCE labels.

    Literally `harness.target_accuracy` -- the TARGET-encoder reading -- handed
    the SOURCE's validation role. Crossing those two is the whole criterion, and
    reading the target's own split instead would be reading target labels, which
    this never does.

    It is MINIMIZED. Its argmax falls trivially at step 0 because theta_t starts
    as theta_s, so the maximum says nothing; the informative extreme is the
    argmin. What it measures literally is how much of the SOURCE's behaviour
    theta_t still has -- CREDA's own docstring is explicit that reading it as
    "how far it moved" is an inference and not the measurement.
    """
    value = harness.target_accuracy(model, source_valid, device)
    set_phase_two_modes(model)
    return value


def effective_classes(model: wiring.Arm, device: torch.device) -> float:
    """CREDA's `cef`: the EFFECTIVE number of classes in the target's routing.

    `exp(H(p))`, with `p` the normalized histogram of the bag-level `argmax`
    over the target TRAIN role and `H` natural. It is `1` when every target bag
    lands in ONE class -- the collapse -- and `classes` when it spreads evenly,
    so it lives in `[1, classes]`.

    **The histogram is taken over `classes` FIXED bins and not over the classes
    present.** Counting only the present ones would make a collapse to one
    class and a perfect split between the two that remain give numbers that
    cannot be compared across epochs, which is exactly the comparison the
    criterion makes.

    **It reads NO target label, and that is what qualifies it as a criterion.**
    It is the histogram of the model's OWN predictions. `oracle_target_reading`
    below reads `y_t` and is recorded for the oracle column alone.

    The role is `model.target` -- the pool `build_arm` fills from
    `target.train_idx` -- which is the same material phase 2 trains on, already
    on the device. The per-step draw is a SAMPLE of it
    (`config.BAGS_PER_STEP` bags with replacement across steps), so a histogram
    taken from a step's draw would not be comparable between epochs; this walks
    the whole role.

    Read in `eval` and restored with `set_phase_two_modes`, for the reason
    `watchman` and `oracle_target_reading` do it: `no_grad` stops the gradient
    and does NOT stop a BatchNorm from re-estimating its running statistics in
    the forward, so a reading taken in `train` would move the very buffers
    `sourceBuffersMoved` reports on.
    """
    positions = torch.arange(model.target.members.shape[0],
                             device=model.target.members.device)
    model.eval()
    counts = torch.zeros(model.classes, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, int(positions.numel()), config.BAGS_PER_STEP):
            chunk = positions[start:start + config.BAGS_PER_STEP]
            routed = model.score_target(model.target.take(chunk)).argmax(dim=1)
            counts += torch.bincount(routed.cpu(), minlength=model.classes)
    set_phase_two_modes(model)
    total = int(counts.sum())
    assert total == int(positions.numel()), (total, int(positions.numel()),
                                             "a bag was routed twice or not at all")
    assert total > 0, "an empty routing has no effective number of classes"
    assert len(counts) == model.classes, (
        len(counts), model.classes, "the routing named a class out of range")
    share = counts.double() / float(total)
    nonzero = share[share > 0.0]
    return float(torch.exp(-(nonzero * nonzero.log()).sum()))


def oracle_target_reading(model: wiring.Arm, target_eval,
                          device: torch.device) -> float:
    """The ORACLE: the head through the target encoder, over the target's own
    TEST role, against the TARGET labels. `TARGET_SERIES` declares what it is
    for.

    **The one reading in this module that touches `y_t` inside the loop, and it
    chooses nothing.** `watchman` above crosses the target encoder with the
    SOURCE's validation role precisely so no criterion ever sees a target
    label; this reads the target's own labels and is recorded for the oracle
    column alone. Using them to MEASURE is not leakage; using them to CHOOSE
    would be, and that boundary is measured rather than promised -- the run is
    bit-identical with and without this call.
    """
    value = harness.target_accuracy(model, target_eval, device)
    # The same restore `watchman` performs, for the same reason:
    # `harness.target_accuracy` reads in `eval` -- which is what keeps it from
    # moving a running statistic -- and ends on an unconditional
    # `model.train()`, so without this the per-module modes phase 2 declared
    # would be whatever that left behind from the first step onward.
    set_phase_two_modes(model)
    return value


#: ---------------------------------------------------------------------------
#: CREDA's intervention ladder, and which of its rungs this run can pay for
#: ---------------------------------------------------------------------------
#:
#: CREDA's per-dataset table reads its rows from least intervention to most:
#: `.predict` (the source model applied to the target, nothing touched), AdaBN
#: (the target's BatchNorm statistics re-estimated on target batches, no
#: optimizer), and then the method itself. The first two are not arms and cost
#: no second sweep, which is what makes the ladder worth having here even with
#: one arm declared:
#:
#: * `.predict` is the SAME RUN read at its step-0 checkpoint, where
#:   `copy_source_to_target` has just made `theta_t` bit-identical to
#:   `theta_s` -- CREDA says this in its own prose and the reason is pairing:
#:   the rung and the method share phase 1 bit for bit, so the difference
#:   between them does not carry the variance of training another source.
#: * AdaBN re-estimates buffers and nothing else. It exists only where there
#:   are buffers to re-estimate, and `_reestimable_layers` reads that off the
#:   MODULE TREE by type: `mlp` has two `nn.BatchNorm1d`, `mlp-nobn` has no
#:   normalization at all, and `mlp-layernorm` has two `nn.LayerNorm` that
#:   normalize and hold nothing. The last of those is the row that makes the
#:   ladder attributable -- see `NORMALIZATIONS` -- and what it is owed is the
#:   REASON and not a number: a number there would be `.predict` under another
#:   name, printed as if a second thing had been measured.
#:
#: Neither is the retired floor arm. The floor was a declared arm with its own
#: phase 2 and its own sweep; these two are two readings of a checkpoint this
#: run already produces, and recording them adds no arm to `TOY_ARMS`.

#: Why AdaBN has no rung, by what the module tree actually holds. Two reasons
#: and not one, because they are different facts about the backbone and
#: collapsing them into "no BatchNorm" would make the control row and the bare
#: row read alike -- which is exactly the confusion the control exists to end.
ADABN_ABSENT_NOTE = (
    "no hay AdaBN en esta fila: el backbone no lleva NINGUNA capa de "
    "normalización, así que no normaliza y no hay estadísticas que reestimar. "
    "Es una lectura del backbone y no una medición que faltó")

ADABN_NO_BUFFERS_NOTE = (
    "no hay AdaBN en esta fila, y por eso esta fila es el control: el backbone "
    "SÍ normaliza —{capas} capa(s) de {tipos}, sobre los features dentro de "
    "cada muestra— y no guarda ninguna estadística corrida, así que no hay "
    "nada que reestimar. Es lo que separa «normalizar ayuda» de «reestimar "
    "ayuda»: donde BatchNorm gana, esta fila dice si ganó por normalizar. Un "
    "número acá sería `.predict` con otro nombre")


def _batchnorm_layers(module: torch.nn.Module) -> list[torch.nn.Module]:
    """Every BatchNorm in a module, by type and never by the backbone's name.

    `NORMALIZATIONS` maps a row name to a backbone name, and reading AdaBN's
    reachability off that map would be reading a string: the row called
    `batchnorm` is reachable because `mlp` HOLDS `nn.BatchNorm1d` layers, and a
    backbone renamed or re-shaped would leave the string saying the old thing.
    """
    return [child for child in module.modules()
            if isinstance(child, torch.nn.modules.batchnorm._BatchNorm)]


def _normalization_layers(module: torch.nn.Module) -> list[torch.nn.Module]:
    """Every normalization layer in a module, re-estimable or not, by TYPE.

    What this buys over `_batchnorm_layers` alone is the distinction the
    control row rests on. "This backbone has no BatchNorm" is true of
    `mlp-nobn` and of `mlp-layernorm` and means two different things: the first
    does not normalize, the second normalizes and keeps nothing. Read off
    `torch.nn.modules.normalization` and the BatchNorm base class rather than
    off a list of names, so a layer type added later is seen.
    """
    kinds = (torch.nn.modules.batchnorm._BatchNorm,
             torch.nn.LayerNorm, torch.nn.GroupNorm,
             torch.nn.modules.instancenorm._InstanceNorm)
    return [child for child in module.modules() if isinstance(child, kinds)]


def _reestimable_layers(module: torch.nn.Module) -> list[torch.nn.Module]:
    """The normalization layers AdaBN can act on: the ones holding running state.

    Derived from the layer's own buffers and not from its class, because that
    is what AdaBN needs: `reset_running_stats` plus a forward pass in `train()`
    is what re-estimates, and both require the layer to be CARRYING statistics.
    A `nn.BatchNorm1d(track_running_stats=False)` holds none and would be as
    unreachable for AdaBN as a LayerNorm is -- reading the class would call it
    reachable and then re-estimate nothing.
    """
    return [layer for layer in _normalization_layers(module)
            if getattr(layer, "running_mean", None) is not None]


@torch.no_grad()
def adabn_reading(model: wiring.Arm, target_train, target_eval,
                  device: torch.device) -> dict:
    """AdaBN on the step-0 checkpoint: buffers re-estimated, nothing optimized.

    Call it with `theta_t` still bit-identical to `theta_s`, which is where
    `copy_source_to_target` leaves it: AdaBN's whole claim is that moving the
    normalization statistics alone already buys something, so reading it off a
    `theta_t` phase 2 had already moved would be reading the method and calling
    it the rung below.

    **It touches nothing.** A deep copy is what gets re-estimated, so the model
    the caller holds is bit-identical afterwards -- no optimizer, no `backward`,
    no step, and the arm's own generator is never drawn from. The target LABELS
    are never read: re-estimating a mean needs the inputs and nothing else, and
    the accuracy at the end is read on the evaluation role exactly as
    `targetAccuracy` is.

    `momentum = None` and not the layer's default. With a momentum the running
    statistics are an exponential average that still carries the source's,
    weighted by however many batches happened to go through; with `None`
    `nn.BatchNorm1d` keeps a cumulative average over the batches it saw, which
    is the target's own mean over this role. AdaBN is «replace the statistics»,
    not «drift toward them».

    `accuracy: None` where the backbone holds nothing re-estimable, with the
    reason in `note`. A number there would be `.predict` under another name, and
    printing it would publish the floor rung twice as if two things had been
    measured.

    **Two different reasons for that `None`, and they are reported apart.** A
    backbone with NO normalization layer does not normalize; a backbone whose
    normalization keeps no running statistics does normalize and has nothing to
    re-estimate. The second is the control row -- it is what makes a BatchNorm
    win attributable to normalizing rather than to re-estimating -- so its
    reading has to say that, in that cell, rather than share a sentence with the
    bare row. `normLayers` travels beside `layers` so the distinction is a
    number a reader can check and not only a sentence: `layers` is what AdaBN
    could act on and `normLayers` is what the backbone normalizes with.
    """
    encoder = model.target_encoder
    normalizers = _normalization_layers(encoder) if encoder is not None else []
    layers = _reestimable_layers(encoder) if encoder is not None else []
    if not layers:
        tipos = sorted({type(capa).__name__ for capa in normalizers})
        return {"accuracy": None, "layers": 0, "normLayers": len(normalizers),
                "batches": 0,
                "note": (ADABN_ABSENT_NOTE if not normalizers else
                         ADABN_NO_BUFFERS_NOTE.format(
                             capas=len(normalizers),
                             tipos=", ".join(f"`{t}`" for t in tipos)))}

    adapted = copy.deepcopy(model)
    for layer in _reestimable_layers(adapted.target_encoder):
        layer.reset_running_stats()
        layer.momentum = None
    adapted.eval()
    adapted.target_encoder.train()
    batches = 0
    for start in range(0, len(target_train), config.BAGS_PER_STEP):
        positions = range(start, min(start + config.BAGS_PER_STEP,
                                     len(target_train)))
        x, _ = _batch(target_train, positions, device)
        adapted.target_embeddings(x)
        batches += 1
    adapted.eval()
    return {"accuracy": harness.target_accuracy(adapted, target_eval, device),
            "layers": len(layers), "normLayers": len(normalizers),
            "batches": batches, "note": None}


def adapt_phase_two(model: wiring.Arm, source_train, source_valid,
                    device: torch.device, scale: Scale, hyper: dict,
                    generator: torch.Generator, trace=None,
                    target_eval=None) -> dict:
    """Phase 2: the global and local terms, under the ramp, move theta_t alone.

    The floor (`B`) has no theta_t, so it has no phase 2: it returns zero steps
    and its reading is the frozen phase-1 model, which is what makes it a floor.

    **The selection is `cef`'s MAXIMUM, read once at the END of each complete
    PASS of the source training role** -- `MIN_EPOCHS_2` passes of guard and
    then every pass to the ceiling. There is NO stop: phase 2 runs `epochs_2`
    whole and the criterion chooses the best epoch of the WHOLE run. The
    weights of that pass are restored on exit. `PHASE_TWO_CRITERION` says where
    every number comes from and which of them are declared.

    It used to read every STEP, which is the defect this closes: there was a
    pass (`harness.balanced_batches` keeps a shuffled queue per class and
    refills it when it empties) and the criterion did not look at it, so the
    stop was decided on whichever batch the step happened to draw. The
    SAMPLING needed no change at all -- only the criterion.

    **The budget is an EPOCH ceiling** (`scale.epochs_2`), which INVERTS what
    this paragraph said. It said the budget stays a STEP ceiling "as `PASOS_2`
    stayed one in CREDA", on the grounds that every frame schedule and
    checkpoint grid here is indexed in steps. The grounds were backwards: a
    step ceiling lets the LOT decide how many passes the experiment runs, and
    in CREDA the lot is searched -- measured there, 200 steps bought 22.2
    passes for `lunas` and 11.8 for `toroide` and `rollo`, so its global table
    compared families trained half as long as one another.

    The frames stay indexed in steps and that is deliberate: a frame is a state
    of the weights and the weights move once per step. What went to epochs is
    the CEILING. `stepsWithoutARead` is now zero by construction and is kept as
    a field rather than deleted, because a reader of two artefacts has to see
    it go to zero rather than vanish.

    `target_eval` is the target's own TEST role and is optional. Handed in, the
    run records `TARGET_SERIES_FIELD`: one oracle reading per step, for the
    per-family table's oracle column and for nothing else. `TARGET_SERIES`
    declares that boundary and `oracle_target_reading` is where it is read; the
    two searches never pass it, so no trial can see a target label.

    **`trace` is called INSIDE this loop, and that is the only place it could
    be.** It is a no-argument callable returning whatever a frame needs read off
    the model as it stands, and it is called at the step-0 checkpoint and then on
    the schedule `frame_steps` declares. Handing it back as `trajectory` rather
    than reconstructing anything afterwards is forced: what a quantity did on its
    way to its final value cannot be recovered from the restored weights, so a
    frame not recorded here is a frame that only another whole run can produce.

    The last frame is recorded BEFORE the best step is restored, so the
    trajectory is the states the run passed through and not the state it chose.
    Which step it chose is `bestStep`, and the figure says so rather than making
    the last frame stand in for it.
    """
    refuse_retired_hyper(hyper)
    set_phase_two_modes(model)
    trajectory: list[dict] = []

    def record_frame(applied: int) -> None:
        # `record_frame` and not `record`: the loop below binds `record` to the
        # step's own training record, so a helper called `record` is shadowed
        # from the first iteration and the call raises a `TypeError` inside the
        # loop rather than at definition.
        if trace is None:
            return
        trajectory.append({"step": int(applied), **trace()})

    # No adaptation family is resolved here any more. It was read off
    # `model.spec["adaptation"]` for one consumer only -- `harness.ramp`, which
    # dispatches on it -- and with the ramp gone there is no consumer left. The
    # floor is still recognized, one line below, by the thing that actually
    # distinguishes it: it has no theta_t.
    if model.target_encoder is None:
        # One frame, and it is the right number rather than an empty list: a
        # model with no theta_t stands at the step-0 checkpoint for the whole of
        # phase 2, so its animation is that one state held.
        record_frame(0)
        return {"steps": 0, "bestStep": -1, "secondsPerStep": float("nan"),
                "bestWatchman": watchman(model, source_valid, device),
                "finalWatchman": watchman(model, source_valid, device),
                "restoredWatchman": None, "history": [], "curve": [],
                "trajectory": trajectory,
                # The epoch quantities, present and empty rather than absent:
                # a payload whose shape changes with its answer makes every
                # reader test for a key, and the one that forgets reads `None`
                # and calls it zero.
                "stepsPerEpoch": steps_of_one_epoch(source_train.targets),
                "epochs": 0, "bestEpoch": -1, "stepsWithoutARead": 0,
                "epochPass": [], "epochDefinedBy": "the source training role",
                TARGET_SERIES_FIELD: None,
                "note": "the floor declares no adaptation term and no theta_t"}

    # CREDA's `LR_2` and `WD_2`, searched. `scale.lr_2` is the declared
    # fallback on `None`, the same shape `train_phase_one` uses for `lr1`;
    # before these two were threaded through, this line read
    # `lr=scale.lr_2` with no `weight_decay` argument at all.
    optimizer = torch.optim.Adam(
        model.target_encoder.parameters(),
        lr=hyper.get("lr2") or scale.lr_2,
        weight_decay=hyper.get("weightDecay2", 0.0),
    )
    # The pass, measured off the role this run was handed rather than taken
    # from the declared anchor, and then CROSSED against it: the anchor is what
    # `MIN_EPOCHS_2` was derived from, so a run whose own pass is a different
    # length would be walking one criterion while the minimum describes
    # another. Reachable: the bag programme is a declared field, and a scale
    # that lowered it would land here.
    steps_per_epoch = steps_of_one_epoch(source_train.targets)
    if steps_per_epoch != STEPS_PER_EPOCH_DECLARED:
        raise ValueError(
            f"this run's pass is {steps_per_epoch} steps and the declared "
            f"programme's is {STEPS_PER_EPOCH_DECLARED}, which is what "
            f"MIN_EPOCHS_2 = {MIN_EPOCHS_2} was derived from; re-deriving the "
            "criterion for a different bag programme is a decision about the "
            "experiment and is not taken here")
    # THE CEILING, DERIVED: `epochs_2` whole passes at this run's own pass
    # length. It was `scale.steps_2`, a step ceiling, and the inversion is in
    # `PHASE_TWO_CRITERION`: a step ceiling lets the lot decide how many passes
    # the experiment runs.
    whole_epochs = scale.epochs_2
    steps_total = whole_epochs * steps_per_epoch
    if whole_epochs < MIN_EPOCHS_2:
        raise ValueError(
            f"epochs_2 = {whole_epochs} and the criterion may not choose "
            f"before {MIN_EPOCHS_2}, so nothing would ever be restored and "
            "the run would report bestStep = -1 with no criterion read at "
            f"all; raise epochs_2 to at least {MIN_EPOCHS_2}")
    # Zero BY CONSTRUCTION, and kept as a field rather than deleted: it was a
    # real number at the step ceiling (8 at the full scale) and a reader of two
    # artefacts has to be able to see it go to zero rather than vanish.
    steps_without_a_read = 0
    started = time.perf_counter()
    history: list[tuple] = []
    curve: list[dict] = []
    epoch_pass: list[dict] = []
    oracle: list[float] | None = [] if target_eval is not None else None
    drawn_this_epoch: list[int] = []
    #: The steps of the epoch being walked, as (objective, adaptation,
    #: targetConfidenceMean, activeClasses), so the per-epoch row can average
    #: the readings `training_step` returns. Cleared with `drawn_this_epoch`.
    _this_epoch: list[tuple] = []
    # The criterion seeks a MAXIMUM of `cef`, so this starts at `-inf`.
    # Starting it at `inf` would let no epoch ever beat it: `best_step` would
    # stay at -1, no weights would be kept, and the run would publish the LAST
    # model as though it had been chosen.
    best, best_step, best_epoch, best_state = -math.inf, -1, -1, None
    stopped_at = steps_total

    stride = frame_stride(scale)
    # Frame zero: `theta_t` bit-identical to `theta_s`, before any update. It is
    # also the state `.predict` is read at, which is what makes the first frame
    # of the animation and the first rung of the table the same moment.
    record_frame(0)

    batches = harness.balanced_batches(source_train.targets, steps_total, generator)
    for step, positions in enumerate(batches):
        drawn_this_epoch.extend(int(p) for p in positions)
        # Eq. (39) at its two declared coefficients, both exactly 1 and both
        # the same float at every step. `harness.ramp` is NOT called with this
        # value: see `FIXED_COEFFICIENT` for why a constant routed through a
        # warmup is a schedule and not a constant. `training_step` wants two
        # plain floats here and that is all it ever wanted.
        coefficient = coefficient_local = FIXED_COEFFICIENT
        x, y = _batch(source_train, positions, device)
        optimizer.zero_grad()
        record = model.training_step(x, y, coefficient, generator,
                                     ramp_local=coefficient_local)
        # Eq. (21) leaves the loss and stays a reading: see
        # `PHASE_TWO_OBJECTIVE`. It carries no gradient into `theta_t`, so this
        # changes what the number MEANS and not what the step does -- which is
        # measured parameter by parameter rather than asserted here.
        objective = phase_two_objective(record)
        objective.backward()
        optimizer.step()
        # r31 Eq. (15)'s max-norm projection -- see the phase-one loop above.
        model.project_constraints()
        value = watchman(model, source_valid, device)
        history.append((step, record["supervised"], record["adaptation"],
                        float(objective.detach()), value))
        curve.append({
            # Named for what they are. They were `ramp`/`rampLocal` while a
            # ramp produced them; a reader of `barrido.json` finding a key
            # called `ramp` holding 1.0 at every step would reasonably conclude
            # the schedule had saturated rather than that there is none.
            "step": step, "coefficient": coefficient,
            "coefficientLocal": coefficient_local,
            # Eq. (21), as a READING. It is kept and never deleted -- the
            # figure draws it and it cannot be recovered from a checkpoint --
            # and `phaseTwoObjective` beside it is what was minimized.
            "supervised": record["supervised"],
            "phaseTwoObjective": float(objective.detach()),
            "adaptation": record["adaptation"],
            "contributionGlobal": record["contributionGlobal"],
            "contributionLocal": record["contributionLocal"],
            "contribution": record["contribution"],
            # |C_act| at this step. Added 2026-10-18, because r33's Eq. (37)
            # made it part of reading the global term rather than only a
            # degeneracy diagnostic: a class the routing abandoned contributes
            # the literal `l~_c = 1`, so a step that lost classes carries a
            # constant the target cannot move. Without this field the curve
            # cannot say WHY a step's global term did not respond -- which is
            # the control `test_the_frozen_sigma_is_immune_to_anything_injected_into_the_target`
            # needs. `training_step` already returns it; this only carries it.
            "activeClasses": record["activeClasses"],
            "watchman": value,
        })
        # The readings the per-epoch row averages, taken from `record` and NOT
        # from `curve[-1]`: the curve row carries what the FIGURE draws, and
        # `targetConfidenceMean` is a diagnostic that never entered it.
        _this_epoch.append((float(objective.detach()),
                            record["adaptation"],
                            record["targetConfidenceMean"],
                            record["activeClasses"]))
        if oracle is not None:
            # The oracle, every step, over the TEST role. Recorded here and
            # nowhere else: a reading taken afterwards could not exist, since
            # what a quantity did on its way to its final value cannot be
            # recovered from the restored weights.
            oracle.append(oracle_target_reading(model, target_eval, device))
        if (step + 1) % stride == 0:
            record_frame(step + 1)

        # THE CRITERION, ONCE PER PASS. `epoch` counts from zero and the read
        # happens on the pass's LAST step, so `epoch + 1` is how many complete
        # passes this run has walked.
        epoch, step_in_epoch = divmod(step, steps_per_epoch)
        if step_in_epoch != steps_per_epoch - 1:
            continue
        # The pass property, measured and fail-closed in the run as well as
        # asserted in the test: BOTH halves, because `distinct == n` admits one
        # bag drawn twice and another dropped, and `total == n` admits the same
        # the other way round.
        total, distinct = len(drawn_this_epoch), len(set(drawn_this_epoch))
        assert total == distinct == len(source_train.targets), (
            epoch + 1, total, distinct, len(source_train.targets),
            "a pass did not draw every training bag exactly once")
        # THE PER-EPOCH ROW, which is where the criterion is read and
        # therefore the only grain at which "why did it choose this epoch" is
        # answerable. Everything beyond the pass property is the EPOCH MEAN of
        # a reading `training_step` already returns -- nothing new is computed
        # here and nothing new reads a target label. CREDA's own row carries
        # the same shape for the same reason.
        _n = max(1, len(_this_epoch))
        _mean = lambda j: sum(r[j] for r in _this_epoch) / _n
        epoch_pass.append({"epoch": epoch + 1, "steps": steps_per_epoch,
                           "objective": _mean(0),
                           "adaptation": _mean(1),
                           # `u_t` in CREDA's row is this reading's complement:
                           # `targetConfidenceMean` is the mean of Eq. (24)'s
                           # own `w_t` over the bags the epoch drew.
                           "targetConfidenceMean": _mean(2),
                           "activeClasses": _mean(3),
                           "distinct": distinct, "total": total})
        drawn_this_epoch = []
        _this_epoch = []
        # THE CRITERION: the MAXIMUM of `cef`, read here and nowhere else.
        # `value` above is the watchman and it is RECORDED and never read by
        # the criterion, so which of the two would have ordered the epochs
        # better stays answerable from the artefact afterwards.
        cef = effective_classes(model, device)
        epoch_pass[-1]["cef"] = cef
        epoch_pass[-1]["watchman"] = value
        if epoch + 1 >= MIN_EPOCHS_2 and cef > best:
            best, best_step, best_epoch = cef, step + 1, epoch + 1
            best_state = {k: v.detach().clone()
                          for k, v in model.target_encoder.state_dict().items()}
        # **And there is NO `elif` that stops, since 2026-10-19.** There was a
        # stop at `PATIENCE_2_EPOCHS` passes without a new best, and it did the
        # same thing CREDA's did: measured over the 90 runs of
        # `Results/Toyset/full`, every one of them died by patience, NONE
        # reached the ceiling, and 38 chose EXACTLY the first legal step. What
        # is agreed is that the ceiling runs whole and the criterion chooses
        # the best epoch of the WHOLE run.

    # The last state the loop actually reached, if the stride did not already
    # land on it. Early stopping ends a run wherever it ends -- at the pilot's
    # stride of 1 this never fires and at the full scale's 10 it fires on nine
    # runs in ten -- and without it the animation would stop at the last
    # multiple of the stride and show a state the run moved past as its end.
    applied = history[-1][0] + 1 if history else 0
    if trace is not None and trajectory[-1]["step"] != applied:
        record_frame(applied)

    if best_state is not None:
        model.target_encoder.load_state_dict(best_state)
    set_phase_two_modes(model)
    restored = watchman(model, source_valid, device)
    # The claim this whole change is about, asserted where it can fail rather
    # than left to the test alone: the step the run chose is the END of a pass.
    assert best_step < 0 or best_step % steps_per_epoch == 0, (
        best_step, steps_per_epoch)
    assert best_step < 0 or best_step == best_epoch * steps_per_epoch, (
        best_step, best_epoch, steps_per_epoch)
    assert len(epoch_pass) == stopped_at // steps_per_epoch, (
        len(epoch_pass), stopped_at, steps_per_epoch)
    assert oracle is None or len(oracle) == len(history), (
        len(oracle or ()), len(history))
    return {"steps": stopped_at, "bestStep": best_step,
            "secondsPerStep": ((time.perf_counter() - started) / len(history)
                               if history else float("nan")),
            # `bestCef` is what the criterion CHOSE on; the three watchman
            # readings beside it are recorded and chose nothing. Before
            # 2026-10-19 `bestWatchman` was both at once, which is why it is
            # split rather than renamed.
            "bestCef": None if best == -math.inf else best,
            "bestWatchman": (epoch_pass[best_epoch - 1]["watchman"]
                             if best_epoch >= 1 else None),
            "finalWatchman": history[-1][4] if history else None,
            "restoredWatchman": restored,
            "history": history, "curve": curve, "trajectory": trajectory,
            # The criterion's own unit, beside the step quantities it is read
            # in. `epochs` is how many complete passes this run walked, which
            # is what `bestEpoch` and the patience are counted in.
            "stepsPerEpoch": steps_per_epoch,
            "epochs": len(epoch_pass),
            "bestEpoch": best_epoch,
            "stepsWithoutARead": steps_without_a_read,
            "epochPass": epoch_pass,
            "epochDefinedBy": "the source training role",
            TARGET_SERIES_FIELD: oracle,
            "note": None}


#: What phase 2 minimizes, and what Eq. (21) became. One dict, so the record
#: states the objective rather than leaving a reader to infer it from a column.
PHASE_TWO_OBJECTIVE = {
    "equation": "L = lambda_glob*L_glob(Eqs. 32-37) + lambda_loc*L_loc(Eqs. 30-31, 38)",
    "summands": ["contributionGlobal", "contributionLocal"],
    "readingOnly": ("Eq. 21 over the source, computed and recorded as "
                    "`supervised` on every step and drawn in the figure, and "
                    "summed into nothing"),
    "why": ("CREDA's phase 2 carries ONE adaptation term, the Renyi, and no "
            "supervised summand. Here the adaptation term is two -- global and "
            "local -- and that is the whole of the difference between the two "
            "objectives. A third summand CREDA's phase 2 does not have makes "
            "the two tables incomparable at the one place they are read side "
            "by side"),
    "whyItMovesNothing": ("phase 2 moves theta_t alone and Eq. (21) is "
                          "head o theta_s over the source, so it carries no "
                          "gradient into theta_t. The optimizer's step is "
                          "bit-identical with and without it -- measured, "
                          "parameter by parameter, in "
                          "`test_the_phase_two_objective_drops_eq_21_and_"
                          "moves_no_step`, not argued here"),
    "notAFlag": ("this is the objective, not an axis: there is no pole where "
                 "Eq. (21) is summed back in. `wiring.training_step` still "
                 "assembles Eq. (39) in full for every other caller -- the "
                 "digit benchmark reads it and is out of scope -- and nothing "
                 "in that file changed"),
}


def phase_two_objective(record: dict) -> torch.Tensor:
    """Eq. (39) minus its first summand: what phase 2 actually minimizes.

    **A subtraction and not a re-composition, because the terms come back
    detached.** `wiring.training_step` returns `loss` as the differentiable
    three-term total and `contributionGlobal`/`contributionLocal` as plain
    floats, so the two-term objective cannot be rebuilt from the dict -- there
    is no tensor in it for the two summands. Subtracting the supervised FLOAT
    from the total tensor gives the right value and, because the subtrahend is
    a constant, a gradient that is bit-identical to the total's. That identity
    is the whole reason this is sound, and it is measured rather than claimed.

    **Why it is done here and not in `wiring.py`.** That file assembles Eq. (39)
    for every arm of the digit benchmark too, which the operator deferred, so
    changing what it returns would change an experiment outside this stretch.
    The objective is a property of THIS protocol's phase 2, and this is the
    module that owns it.

    `supervised` is read off the same record, so a caller cannot pair the loss
    of one step with the reading of another.
    """
    return record["loss"] - record["supervised"]


def weighting_cost(model: wiring.Arm, source_train, device: torch.device,
                   seed: int) -> dict:
    """What the confidence weighting costs the global term, ON THIS CELL.

    One extra phase-2-shaped step, on a deep copy of `model` with `weighting`
    flipped to `False` and every parameter bit-identical, over the same batch
    and the same generator seed. Both global terms are read and the ratio is
    the measurement: `global(weighted) / global(unweighted)`.

    **Why the run measures this instead of a docstring stating it.** The
    operator's instruction was that the cost travel with the tables, and a
    sentence quoting a factor measured in some earlier session is a claim the
    reader cannot re-measure and the code cannot notice going stale. Eq. (37)'s
    `omega_c` is sub-convex, so the factor depends on how confident the target
    bags happen to be in THIS cell -- which is exactly the sort of number that
    stops being true quietly. `WEIGHTING_COST` keeps the explanation and the
    prior session's range, labelled as prior; this is the live reading.

    **It changes nothing it measures, and that sentence used to be false on
    one of its two sides.** The claim was written for the `unweighted` arm,
    which has always been a deep copy; the WEIGHTED half ran on the caller's
    own model. No optimizer, no `backward` and no step -- and a
    `training_step` is still a forward in phase-2 modes, which puts
    `theta_t` in `train`, so it re-estimated the target encoder's running
    statistics. Measured on `toroide`/`mlp`: `target_encoder.net.1.running_mean`
    moved 1.417e-01 and three more buffers with it. `run_cell` reads
    `targetAccuracy` -- the only published number of this table -- AFTER this
    call, so the published accuracy depended on a diagnostic having run first.
    `buffersMoved` could not see it either: it is measured on `model.encoder`,
    the SOURCE encoder, and this moves the target's.

    Both sides are deep copies now, which also makes them symmetric: one arm
    on a copy and one on the live model is a difference between the two readings
    that nobody declared. `eval` is deliberately NOT the remedy here, unlike
    `embedding_scales` and `frozen_source_sigma`: this has to compute what a
    phase-2 step computes, and a BatchNorm read in `eval` normalizes by the
    running statistics instead of the batch, so the ratio would stop being the
    ratio phase 2 pays.

    The generator is this function's own, seeded from `seed`, so the arm's own
    draw is not consumed -- two arms differing in how much of the generator
    they consume is the difference nobody declared.

    `None` for an arm that declares no weighting or no adaptation term: there
    is no cost to read, and `0.0` would be a reading.
    """
    if not model.spec.get("weighting") or model.spec["adaptation"] is None:
        return None
    weighted_model = copy.deepcopy(model)
    unweighted = copy.deepcopy(model)
    unweighted.spec = {**model.spec, "weighting": False}
    x, y = _batch(source_train, range(len(source_train)), device)
    # **The spy that read `omegaSum` is gone, and so is the quantity.** Until
    # 2026-10-18 this function monkeypatched `wiring.confidence_weights` to
    # capture `sum_c omega_c`, r29's sub-convex aggregation weights, because
    # their shortfall below one WAS the gate this cost measured. r32 replaced
    # them with `n_c^s / N_s`, which sum to one identically and come from the
    # observed source labels -- there is no shortfall to read and no gate on
    # the aggregation at all. `confidence_weights` was removed with the
    # formula.
    #
    # What this still measures is the cost of Eq. (26)'s `D`, which is what the
    # `weighting` axis has always meant: the same batch, the same parameters,
    # the conditional blocks weighted against raw.
    set_phase_two_modes(weighted_model)
    weighted_record = weighted_model.training_step(
        x, y, FIXED_COEFFICIENT, torch.Generator().manual_seed(seed + 51217),
        ramp_local=FIXED_COEFFICIENT)
    set_phase_two_modes(unweighted)
    unweighted_record = unweighted.training_step(
        x, y, FIXED_COEFFICIENT, torch.Generator().manual_seed(seed + 51217),
        ramp_local=FIXED_COEFFICIENT)
    weighted = weighted_record["contributionGlobal"]
    raw = unweighted_record["contributionGlobal"]
    return {
        # `nan` and never `0.0` where the ratio is unavailable: `0.0` is the
        # reading of a term the weighting annihilated, which is the thing this
        # exists to report.
        "ratio": (weighted / raw) if raw > 0.0 else float("nan"),
        # `nan` ALWAYS, since 2026-10-18, and the key is kept rather than
        # dropped: `barrido.json` carries `weightingCostOmegaSum` on every row
        # already written, and a reader meeting the field needs it to say
        # "unavailable" rather than to vanish between artefacts. The quantity
        # it reported -- `sum_c omega_c` under r29's sub-convex weights -- has
        # no counterpart in r33, whose weights sum to one by construction.
        "omegaSum": float("nan"),
        "globalWeighted": weighted,
        "globalUnweighted": raw,
        "activeClasses": weighted_record["activeClasses"],
    }


def _parameter_vector(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _buffer_vector(module: torch.nn.Module) -> torch.Tensor:
    buffers = [b.detach().reshape(-1).float() for b in module.buffers()]
    return torch.cat(buffers) if buffers else torch.zeros(1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ------------------------------------------------------- the latent positions

#: What the latent half of the artefact carries per cell, named once so the
#: writer, the loader and the figure cannot disagree about which member is
#: which. Each name is one `.npz` member per cell handle.
#:
#: **The EUCLIDEAN bag representation, Eq. (19), and the equation number is a
#: correction.** The brief this was built from named `Psi(B)` of Eq. (17) and
#: described "the vector the head actually classifies". In
#: `research-concept-r29.md` those are two different objects: Eq. (17) is
#: `Psi(B) = sum_a beta_a Phi_sigma(h_a)`, the bag's representation in the
#: kernel's own space `H_I`, and it is what the ALIGNMENT uses -- through
#: Eq. (18)'s inner product, never as coordinates, because a Gaussian kernel's
#: feature map has none to write down. Eq. (19) is `z = sum_a beta_a h_a` in
#: `R^d`, the convex combination the CLASSIFIER operates on, and it is what
#: `wiring.Arm.bag_representations` returns and `wiring.Arm.head` consumes.
#: Only the second is a position, so it is the one written here. A figure of
#: the first would be a kernel-PCA of Eq. (18)'s Gram matrix -- a different
#: object, drawn in coordinates nothing in the method ever computes -- and
#: that is a decision for whoever wants it, not a substitution to make quietly.
#:
#: **Bag representations and not instance embeddings, decided rather than
#: defaulted.** MIL-CREDA's unit is the bag, so "where did phase 2 put this
#: subject" is the question this space answers. The instance embeddings answer
#: a different one -- where it put the instances the attention pooled -- and
#: they are NOT written, for three reasons stated here so the omission is a
#: decision rather than an oversight: the figure draws bags, so nothing would
#: read them; they are `instances_per_bag` times the bytes per view, which is
#: thirty at both declared scales; and the in-bag weights `beta` that place a
#: bag inside its own instance cloud are not written either, so the cloud would
#: arrive without the thing that locates the bag in it, which is the only
#: reading that would have made it worth its size. Adding them is one call to
#: `instance_embeddings` here and a fourth view in this tuple.
#:
#: **No prediction and no confidence channel, for a measured reason.** The
#: figure already spends colour on the class and the marker on the domain, and
#: `toy_bags.BAGS_PER_CLASS_TOY`'s own docstring records what happened the last
#: time a third per-point channel was asked of a panel at this cardinality: at
#: twelve bags per class the label channel stopped reading -- 50 of 72 names
#: overlapping in `lunas`, 70 of 72 in `toroide` -- and the declared cut is now
#: 27, which is further past that limit. The operator also rejected
#: an earlier tint channel for the same reason. What the head got right is
#: already reported, per cell, by `toy_tables.render_familia`; drawing it again
#: would be the duplication rule as well as the unreadable one.
#:
#: **The five members after the seven are the animation's, and each one closes a
#: gap the static figure could live with and an animation cannot.** The seven
#: above carry two moments; an animation's frames are the steps, so the
#: trajectory travels as its own members: `trajectorySteps` is how many updates
#: each frame stands at, `trajectoryTarget` the target's positions at that
#: frame and `trajectoryPrediction` what the head decided for each bag there.
#: `predictionSource` travels ONCE and not per frame, which is a consequence of
#: the freeze rather than a saving: the head and `theta_s` are both outside the
#: optimizer, so the source's own decision cannot change between frames, and
#: writing it per frame would store the same row twenty times and invite a
#: reader to look for movement in it. `phaseTwoHistory` is the per-step trace
#: `_without_curves` strips out of `barrido.json` -- correctly, it is a trace
#: and not a scalar -- which is why column four of CREDA's grid had nothing to
#: draw until it travelled here.
LATENT_MEMBERS = ("source", "targetBefore", "targetAfter",
                  "classSource", "classTarget", "bagSource", "bagTarget",
                  "trajectorySteps", "trajectoryTarget",
                  "trajectoryPrediction", "predictionSource",
                  "phaseTwoHistory")

#: The three members above that are positions, in the order a panel draws them:
#: the frozen source, the target before phase 2, and the target after it.
#:
#: **Three views and not one, and the two extra are what the figure is FOR.**
#: The question a reader brings to this figure is what phase 2 did to the
#: target encoder, and a single cloud cannot answer it: `targetAfter` alone
#: says where the target ended and nothing about where it started, so any
#: movement a reader thinks they see is their own inference. `source` is the
#: frozen reference the movement is read against -- phase 2 moves `theta_t`
#: alone, which is what `sourceMoved`/`sourceBuffersMoved` already measure per
#: cell -- and `targetBefore` is `theta_t` at the one moment it was
#: bit-identical to `theta_s` (`copy_source_to_target`, parameters AND
#: buffers). Measured on this pilot the three cost about 7 KB per family on top
#: of a 6 KB artefact; the generator notebook prints the figure for the run in
#: front of it rather than quoting this one.
LATENT_VIEWS = ("source", "targetBefore", "targetAfter")

#: The columns of `phaseTwoHistory`, named once so the writer and the two loss
#: panels cannot disagree about which one is which. CREDA keeps the identical
#: habit (`CAMPOS_HT`) and for the identical reason: the arity of this trace has
#: already grown twice there, and a positional read that was right before the
#: growth is wrong after it without anything failing.
#: Every field is a key of `adapt_phase_two`'s own `curve`, read BY NAME. The
#: total the optimizer received is not among them and is not missing: with both
#: of Eq. (39)'s coefficients fixed at exactly 1 it is `supervised +
#: adaptation`, so storing it would be storing a sum of two columns already
#: here -- and the loss panel draws the two terms rather than their sum, since
#: `supervised` has no gradient path to anything trainable on this path.
TRAJECTORY_HISTORY_FIELDS = ("step", "supervised", "adaptation",
                             "contributionGlobal", "contributionLocal",
                             "contribution", "watchman")

#: How many frames the animation AIMS at, from which the stride follows. It is a
#: programme constant and equal at both scales -- the stride is not, because
#: `epochs_2` is a repetition field and lowering it is the whole of what makes a
#: pilot a pilot.
#:
#: **Twenty is CREDA's own, read off its declaration rather than chosen here.**
#: `CREDA.toy_experiment.CUADROS` is `range(0, EPOCAS_2 + 1, 5)` over 100
#: epochs: a stride of 5 epochs and twenty-one frames. It was cited here as
#: `range(0, PASOS_2 + 1, 10)` over 200 steps, which is the retired step
#: ceiling; the frame COUNT did not move with that change and the unit did.
#:
#: **At the full scale the count is honoured exactly and at the pilot it is
#: not, and that is the declaration.** The grid is a whole number of EPOCHS
#: (`frame_epochs`), so the count it can reach is bounded by `epochs_2`: at 100
#: epochs the stride is 5 and the twenty-one frames land; at the pilot's 4
#: epochs no whole-epoch stride reaches twenty, and the floor of one epoch
#: gives five. That is not a shortfall to repair -- it is what makes the pilot
#: fast, which is the whole of what a pilot is for. MEASURED at the pilot: five
#: frames a handle against the twenty-three a step grid gave, so 4.6x fewer
#: `bag_positions` calls, 4.6x fewer bytes (26,730 against 122,958 at 16 output
#: dimensions) and 4.6x fewer panels to draw -- 45 against 207 over its nine
#: handles, at about half a second each.
#:
#: What the count still protects is the floor: a literal stride copied from
#: CREDA would be 5 epochs, which at the pilot's 4 would leave TWO frames, and
#: two frames and twenty read as the same object until somebody says.
TRAJECTORY_FRAMES = 20


def frame_epochs(scale: Scale) -> int:
    """Every k-th phase-2 EPOCH is a frame, and this is k. The declared unit.

    The grid is declared here in epochs and converted to steps by
    `frame_stride`, which is the same direction `CREDA.toy_experiment.corrida`
    runs: `CUADROS` is epochs and `cuad = tuple(e * pasos_por_epoca for e in
    cuad_ep)`.

    The floor of one epoch is what the pilot runs at: at 4 epochs no
    whole-epoch stride reaches `TRAJECTORY_FRAMES`, so it takes every epoch and
    draws five frames.
    """
    return max(1, int(scale.epochs_2) // TRAJECTORY_FRAMES)


def frame_stride(scale: Scale) -> int:
    """Every k-th phase-2 step is a frame, and this is k. DERIVED, not declared.

    `frame_epochs` whole passes at the declared pass length. The loop walks
    steps -- the weights move once per step and a frame is a state of the
    weights -- so the schedule has to arrive in steps; what changed is which
    unit it is DECLARED in, and the step count is now the derived one.

    **Why this one is derived from a pass that cannot vary, when CREDA's
    `pasos_por_epoca` is computed per run.** Not because a frame is a state of
    the weights -- that is equally true in CREDA and distinguishes nothing. It
    is because this programme's pass is `STEPS_PER_EPOCH_DECLARED` and
    `adapt_phase_two` REFUSES a run whose own pass differs, so epoch and step
    are the same index divided by a constant and a grid in either unit measures
    the same moment. In CREDA the pass is derived from `LOTE_2`, which is
    SEARCHED -- 16, 8 and 4 steps for lots of 32, 64 and 128 -- so there a step
    grid measures three trials at three different points of their training, and
    that is what forced its grids into epochs.

    At the full scale this returns exactly what the step-declared version
    returned -- 5 epochs is 80 steps, so the twenty-one frames land on the same
    steps, byte for byte. At the pilot it does not: 23 frames become 5, which
    is measured in `TRAJECTORY_FRAMES` and is the point of the change.
    """
    return frame_epochs(scale) * STEPS_PER_EPOCH_DECLARED


def frame_steps(scale: Scale, applied: int) -> tuple[int, ...]:
    """Which update counts a run of `applied` steps leaves a frame at.

    **Zero and the last are always in, and that is the declaration rather than a
    convenience.** Zero is the one moment `theta_t` is bit-identical to
    `theta_s` -- the step-0 checkpoint `.predict` is read from -- so an animation
    that began at the first strided step would open on a target that had already
    moved and nothing in the picture would say so.

    **The last is now a no-op by construction, and it is kept rather than
    deleted.** Its reason was early stopping: with a stride of 10 and a run that
    stopped at 47, the final frame would otherwise be step 40 and the figure
    would animate up to a state the run moved past. Phase 2 has had no early
    stop since 2026-10-19 -- it runs its ceiling whole -- so `applied` is the
    ceiling, and the ceiling is `frame_epochs * TRAJECTORY_FRAMES` passes at
    most, which the strided range already lands on exactly: 1600 at the full
    scale with a stride of 80, 64 at the pilot with a stride of 16. The union is
    kept because it costs a set operation and it is what holds the schedule
    correct if a run is ever cut short again, the same reason
    `stepsWithoutARead` is kept at zero rather than removed.

    Derived here and not inside the loop that records, so the schedule a run
    FOLLOWED can be re-derived from the two numbers the record already carries
    (`scale.epochs_2` and `phase2Steps`) and compared with the steps the frames
    actually claim.
    """
    applied = int(applied)
    if applied < 0:
        raise ValueError(f"a run cannot have applied {applied} steps")
    stride = frame_stride(scale)
    steps = {0, applied}
    steps.update(range(stride, applied + 1, stride))
    return tuple(sorted(steps))


@torch.no_grad()
def bag_positions(model: wiring.Arm, bagset: bags.BagSet,
                  device: torch.device, domain: str) -> dict:
    """Eq. (19)'s `z` for EVERY bag of one domain, with its class and its index.

    Every bag of the domain and not a role: the figure is about where the two
    clouds sit, and a cloud drawn from one role would be the role's own
    sampling shown as the method's geometry. It also keeps the two domains
    equal in how many points they contribute, which is the one thing a reader
    reads as coverage without being told.

    `domain` states which encoder embeds `bagset` -- `"source"` for
    `model.instance_embeddings` (`theta_s`), `"target"` for
    `model.target_embeddings` (`theta_t`) -- and has no default anywhere in the
    call chain, the same refusal `latent.represent` already makes: a caller that
    cannot say which domain it holds gets an error rather than a plausible
    number from whichever encoder happened to be first.

    **`model.sigma` and never `config.KERNEL_SIGMA`, which is why this is not
    `latent.represent`.** That function does exactly this for the digit
    campaign and evaluates Eq. (19)'s weights at `config.KERNEL_SIGMA`. On this
    toyset that constant is measured OUT OF BAND -- see
    `tests/test_toy_run.py::test_the_declared_absolute_bandwidth_is_out_of_band_on_this_toyset`
    -- because the bandwidth here is re-measured per cell and resolved onto
    `Arm.sigma` by `_resolved_sigma`. Reusing it would draw a space at a
    bandwidth this run never computed in. `latent.py` is not on this work's
    authorized surface, so the duplication is reported rather than removed.

    In `eval` and under `no_grad`: a BatchNorm encoder read in `train` mode
    would make a bag's position depend on which other bags shared its batch,
    and then the same bag would land somewhere else at a different chunk size.
    The mode is restored to the one phase 2 runs under before returning, the
    same courtesy `watchman` already performs for the same reason.
    """
    if domain not in wiring.SIDES:
        raise ValueError(f"unknown domain {domain!r}; known: {wiring.SIDES}")
    embed = model.instance_embeddings if domain == "source" else model.target_embeddings
    total = int(bagset.members.shape[0])

    model.eval()
    rows, classes, decided = [], [], []
    for start in range(0, total, config.BAGS_PER_STEP):
        chunk = torch.arange(start, min(start + config.BAGS_PER_STEP, total))
        instances = bagset.images[bagset.members[chunk]].to(device)
        embeddings = embed(instances)
        Z, _ = model.bag_representations(embeddings, model.sigma)
        rows.append(Z.float().cpu())
        classes.append(bagset.labels[chunk].cpu())
        # The head's own decision for each bag, off the SAME `embeddings` the
        # positions came from, so column two of the grid is the same cloud
        # recoloured and never a second reading taken somewhere else.
        #
        # `model._scores` and not `model.head(Z)`, which is the obvious line and
        # is wrong for an arm that does not predict on the bag: `_scores` is
        # where the unit branch lives (a bag goes through Eq. (19), an
        # instance-unit arm averages its instances' class distributions and
        # takes the log), and writing one of its two halves here would fork that
        # rule. Promoting it out of `wiring.Arm`'s private surface is the
        # alternative and `wiring.py` is not on this work's authorized surface,
        # so the private call is used and said out loud rather than forked.
        decided.append(model._scores(embeddings).argmax(dim=1).cpu())
    set_phase_two_modes(model)

    return {
        "positions": torch.cat(rows).numpy().astype(np.float32),
        "classes": torch.cat(classes).numpy().astype(np.int16),
        # What the head decided, per bag, at the state the model is in RIGHT
        # NOW. On the source it cannot move -- neither `theta_s` nor the head is
        # in the optimizer -- and on the target it moves with every step, which
        # is the whole of what column two animates.
        "predictions": torch.cat(decided).numpy().astype(np.int16),
        # The bag's own index, written rather than left to row order. The
        # figure pairs each target bag's BEFORE position with its AFTER one to
        # draw the displacement, and pairing by row order is exactly the
        # implicit contract that breaks in silence: the arrows would still be
        # drawn, between the wrong pairs, and nothing in the picture would say
        # so. With the index on both views the pairing is a join the drawing
        # code asserts. It is not a drawing channel -- see `LATENT_MEMBERS` on
        # why no third per-point channel is spent.
        "bags": np.arange(total, dtype=np.int16),
    }


def _latent_moment(model: wiring.Arm, material: dict,
                   device: torch.device) -> dict:
    """Both domains' bag positions as the model stands RIGHT NOW.

    Called twice per arm -- once with `theta_t` still bit-identical to
    `theta_s`, once after phase 2 moved it -- so neither view is inferred from
    the other.
    """
    return {"source": bag_positions(model, material["source"], device, "source"),
            "target": bag_positions(model, material["target"], device, "target")}


def _trajectory_history(curve: list[dict]) -> np.ndarray:
    """The per-step phase-2 trace as one array, in `TRAJECTORY_HISTORY_FIELDS`
    order.

    This is the trace `_without_curves` keeps out of `barrido.json`, and keeping
    it out of there is right: it is a series and that file holds a cell's
    scalars. What was wrong is that it then reached nothing at all, so column
    four of CREDA's grid -- the phase-2 loss beside the embedding -- had nothing
    to draw from. It travels in the `.npz`, where the other series already do.
    """
    return np.asarray([[float(point[field])
                        for field in TRAJECTORY_HISTORY_FIELDS]
                       for point in curve], dtype=np.float32).reshape(
        len(curve), len(TRAJECTORY_HISTORY_FIELDS))


def _latent_members(before: dict, after: dict, phase_two: dict) -> dict:
    """The two moments and the trajectory, folded into `LATENT_MEMBERS`.

    The source positions written are the ones measured AFTER phase 2 -- the
    same weights `targetAccuracy` and `weightingCost` are read from, so the
    whole panel describes one moment. That they equal the before ones is the
    freeze, and the freeze is MEASURED rather than assumed: `latentSourceMoved`
    is the difference between the two, beside `latentTargetMoved` which is the
    same reading on the encoder phase 2 does move. One without the other would
    be a zero with nothing establishing that the reading can come out non-zero.

    **The frames are not derived from the two moments and could not be.** A
    before/after pair is two states; an animation's frames are the states in
    between, and nothing in a checkpoint recovers them -- which is the rule this
    repository already states about anything that exists only while training
    runs. `phase_two["trajectory"]` is what the loop recorded as it went, and
    frame zero of it is `targetBefore` by construction rather than by copy: the
    same reading, taken at the same moment, which is what makes the two
    comparable at all.
    """
    trajectory = phase_two["trajectory"]
    if not trajectory:
        raise ValueError(
            "phase 2 recorded no frame: an animation of zero frames is not an "
            "animation, and a run that applied no step still stands at the "
            "step-0 checkpoint, which is frame zero")
    return {
        "source": after["source"]["positions"],
        "targetBefore": before["target"]["positions"],
        "targetAfter": after["target"]["positions"],
        "classSource": after["source"]["classes"],
        "classTarget": after["target"]["classes"],
        "bagSource": after["source"]["bags"],
        "bagTarget": after["target"]["bags"],
        "trajectorySteps": np.asarray([frame["step"] for frame in trajectory],
                                      dtype=np.int16),
        "trajectoryTarget": np.stack([frame["positions"]
                                      for frame in trajectory]).astype(np.float32),
        "trajectoryPrediction": np.stack([frame["predictions"]
                                          for frame in trajectory]).astype(np.int16),
        # ONE row and not one per frame. The head and `theta_s` are both outside
        # the optimizer, so the source's decision is the same at every frame;
        # writing it per frame would store twenty copies of one row and offer a
        # reader a channel to look for movement in that cannot move.
        "predictionSource": after["source"]["predictions"],
        "phaseTwoHistory": _trajectory_history(phase_two["curve"]),
    }


def _moved(first: np.ndarray, second: np.ndarray) -> float:
    """How far the furthest row moved, in the units of the space itself."""
    return float(np.abs(second - first).max()) if first.size else float("nan")


# ------------------------------------------------------------------- one cell

def run_cell(familia: toy_bags.Familia, normalization: str, seed: int,
             scale: Scale, hyper: dict, device: torch.device,
             arms: tuple[str, ...] = TOY_ARM_IDS) -> dict:
    """Every declared arm on one (family, normalization, seed), sharing one
    phase 1.

    The unit of cost this module reports: `elapsedSeconds` is measured here, so
    the full scale can be forecast from data rather than guessed.

    **One arm is declared, and phase 1 is still trained once and shared.** With
    a single arm the sharing changes no number -- there is nobody to share with
    -- and the structure is kept rather than collapsed because what it holds is
    a property of the protocol and not of the arm count: phase 1 is fitted, its
    best epoch restored, and that state loaded into the arm phase 2 moves. A
    version that trained inside the loop would re-fit phase 1 per arm the moment
    a second one is ever declared again, which is the defect the sharing exists
    to prevent, and nothing would say so.
    """
    refuse_retired_hyper(hyper)
    started = time.perf_counter()
    backbone = NORMALIZATIONS[normalization]
    material = material_for(familia, seed, scale, backbone)

    with material_bound(familia, scale, backbone, hyper):
        source, target = material["source"], material["target"]
        # `_eval` and not `_test` on purpose: these three come out of
        # `bags.roles`, so they carry the BAGSET's names (`train_idx`,
        # `valid_idx`, `eval_idx`), which the digit benchmark owns jointly and
        # this toyset does not rename. The RECORD's third role is `test` --
        # CREDA's word, see `Scale.realized_split` -- and `toy_tables` prints
        # that one. Renaming these to follow the record would detach them from
        # the attribute they are read off, which is the opposite of coherent.
        source_train, source_valid, source_eval = bags.roles(source)
        # The target's TRAIN role is read here for AdaBN's rung, which needs
        # target inputs and no target label. The role is the one the method's own
        # phase 2 draws from, so the rung and the method see the same material.
        target_train, _, target_eval = bags.roles(target)

        reference = build_arm(arms[-1], material, scale, hyper, device, seed)
        measured_sigma = sigma_grid(embedding_scales(
            reference, source, source.train_idx, device))
        generator = torch.Generator().manual_seed(seed + 9973)
        phase_one = train_phase_one(reference, source_train, source_valid,
                                    device, scale, hyper, generator)
        shared = phase_one["state"]

        # The bandwidth phase 2 runs at, measured HERE and once: `reference`
        # has just been trained and frozen, so this is CREDA's own moment for
        # it (`corrida`'s `sigma_fijo`, which it computes the instant phase 1
        # ends because after phase 2 the weights are other weights). The
        # searched multiplier multiplies THIS and never `config.KERNEL_SIGMA`.
        #
        # Phase 1 above pooled at a different bandwidth and had to: the unit
        # here is the BAG, so Eq. (14)'s consensus needs a bandwidth before any
        # bag score exists, where CREDA's phase 1 is plain CE over points and
        # needs none. That is the one place this cannot be CREDA's shape, so
        # both numbers and their ratio go into the record rather than only the
        # one that governs phase 2.
        # The multiplier is read BEFORE the statistic, so the reading can report
        # the kernel value and Eq. (16)'s weight spread at the bandwidth phase 2
        # will actually run at rather than at the bare statistic. It is still
        # applied here and only here: `frozen_source_sigma` reads it and never
        # multiplies by it.
        multiplier = float(hyper.get("sigmaMultiplier",
                                     SIGMA_MULTIPLIER_NEUTRAL))
        frozen = frozen_source_sigma(reference, source, source.train_idx,
                                     device, multiplier=multiplier)
        bootstrap = float(hyper.get("kernelSigma", config.KERNEL_SIGMA))
        phase_two_sigma = multiplier * frozen["sigma"]
        measured_sigma = {
            **measured_sigma,
            "frozenSigma": frozen["sigma"],
            "frozenStatistic": frozen["statistic"],
            "frozenScope": frozen["scope"],
            # The scope this stretch corrected, beside the one that governs:
            # `frozenPooledSigma` is the number every pre-correction run ran at,
            # and the two kernel readings are what the correction bought. See
            # `FROZEN_SIGMA_SCOPE`.
            "frozenPooledSigma": frozen["pooledSigma"],
            "frozenPooledToWithinRatio": frozen["pooledToWithinRatio"],
            "frozenKernelWithinBag": frozen["kernelWithinBag"],
            "frozenKernelInBand": frozen["inBand"],
            "frozenPooledKernelWithinBag": frozen["pooledKernelWithinBag"],
            "frozenPooledKernelInBand": frozen["pooledInBand"],
            "frozenWeightSpreadRelativeToUniform":
                frozen["weightSpreadRelativeToUniform"],
            "frozenPooledWeightSpreadRelativeToUniform":
                frozen["pooledWeightSpreadRelativeToUniform"],
            "frozenWeightSpreadGainOverPooled":
                frozen["weightSpreadGainOverPooled"],
            "frozenReading": frozen,
            "frozenSigmaMultiplier": multiplier,
            "frozenSigmaMultiplierSource": ("search" if "sigmaMultiplier"
                                            in hyper else "neutral default"),
            "phaseTwoSigma": phase_two_sigma,
            "bootstrapSigma": bootstrap,
            "bootstrapNote": ("what phase 1 POOLED at, which CREDA has no "
                              "counterpart for: its phase 1 is CE over points "
                              "and needs no bandwidth, while a bag score "
                              "cannot be computed without one"),
            "frozenToBootstrapRatio": (frozen["sigma"] / bootstrap
                                       if bootstrap else float("nan")),
            "declaredConstantUsed": False,
        }
        # Everything below phase 1 runs at the frozen bandwidth. Written into
        # the hyper dict the arms are built from, because `wiring.Arm` reads
        # `kernelSigma` once at construction and every kernel block then reads
        # `self.sigma` -- one bandwidth for the three blocks, which is what
        # r25 line 732 declares.
        hyper = {**hyper, "kernelSigma": phase_two_sigma}

        rows = {}
        latent = {}
        for arm_id in arms:
            model = build_arm(arm_id, material, scale, hyper, device, seed)
            model.load_state_dict(shared, strict=False)
            for name, parameter in model.named_parameters():
                if not name.startswith("target_encoder."):
                    parameter.requires_grad_(False)
            copy_source_to_target(model)

            source_before = _parameter_vector(model.encoder).clone()
            buffers_before = _buffer_vector(model.encoder).clone()
            # The latent space BEFORE phase 2, read here and not reconstructed
            # later: this is the one moment `theta_t` is bit-identical to
            # `theta_s`, and a before-view computed afterwards through the
            # source encoder would be an inference standing in for a reading.
            latent_before = _latent_moment(model, material, device)
            # The intervention ladder's two lower rungs, both read HERE, at the
            # step-0 checkpoint, before anything moves. `.predict` is the frozen
            # source model applied to the target and AdaBN is that same model
            # with its target BatchNorm statistics re-estimated; neither is an
            # arm and neither costs a second sweep. Reading them after phase 2
            # would read the method and label it the rung below.
            predict_accuracy = harness.target_accuracy(model, target_eval, device)
            adabn = adabn_reading(model, target_train, target_eval, device)
            step_generator = torch.Generator().manual_seed(seed + 9973)
            phase_two = adapt_phase_two(
                model, source_train, source_valid, device, scale, hyper,
                step_generator,
                # The frame: where every target bag is and what the head decided
                # for it, read off the model as the loop stands. It has to be a
                # closure over `model` and not a precomputed anything -- the
                # whole point is that it is evaluated at the step, and a frame
                # that could be rebuilt afterwards would not need recording.
                trace=lambda: {
                    key: value for key, value in
                    bag_positions(model, material["target"], device,
                                  "target").items()
                    if key in ("positions", "predictions")},
                # The ORACLE's material, and the ONLY call site that hands it
                # over: `search_phase_two` calls this same function and passes
                # no `target_eval`, so no trial can record a series and no
                # objective can read one. `TARGET_SERIES` declares what the
                # reading is for and `tests/test_milcreda_fase2_por_epoca.py`
                # proves from the syntax tree that this is the one call site.
                target_eval=target_eval)
            # And AFTER, on the weights the run ended with -- `adapt_phase_two`
            # restores the best step's `theta_t` before returning, so this is
            # the same state every published reading of this cell comes from.
            latent_after = _latent_moment(model, material, device)
            latent[arm_id] = _latent_members(latent_before, latent_after,
                                             phase_two)
            moved = float((_parameter_vector(model.encoder) - source_before).norm())
            buffers_moved = float((_buffer_vector(model.encoder)
                                   - buffers_before).norm())

            # What the confidence weighting cost the global term on THIS cell,
            # measured after phase 2 on the weights the run ended with. Every
            # declared arm carries the weighting and an adaptation term, so
            # `weighting_cost`'s own `None` is unreachable from here and the
            # `nan` fallbacks that stood below it are gone with the floor arm
            # that reached them -- a branch nothing can enter reads as a guard.
            cost = weighting_cost(model, source_train, device, seed)
            curve = phase_two["curve"]
            supervised = _mean([abs(p["supervised"]) for p in curve])
            global_share = _mean([abs(p["contributionGlobal"]) for p in curve])
            local_share = _mean([abs(p["contributionLocal"]) for p in curve])
            contribution = global_share + local_share
            # Read off what was MINIMIZED, never recomposed from the shares
            # beside it. The first version of this line computed
            # `global_share + local_share`, which is a second definition of the
            # objective that nothing obliges to agree with the first: the
            # mutation that put Eq. (21) back into the loss left this field
            # unchanged and the test green. Now the field is the mean of the
            # per-step objective `adapt_phase_two` actually stepped on, and the
            # agreement with the two shares is what a test can cross.
            objective_mean = _mean([p["phaseTwoObjective"] for p in curve])
            rows[arm_id] = {
                "family": familia.nombre,
                "normalization": normalization,
                "arm": arm_id,
                "seed": seed,
                "backbone": backbone,
                # The published readings. `targetAccuracy` is the ONLY place a
                # target label is ever read, and it enters no criterion and no
                # search: the stop and both searches see the watchman and source
                # valid accuracy alone.
                "targetAccuracy": harness.target_accuracy(model, target_eval, device),
                "sourceAccuracy": harness.accuracy(model, source_eval, device),
                # CREDA's intervention ladder, read off this same run. The first
                # rung is the step-0 checkpoint -- paired with the method bit for
                # bit, since they share phase 1 -- and the second is that
                # checkpoint with the target's normalization statistics
                # re-estimated. `adabnTargetAccuracy` is `None` where the
                # backbone holds nothing re-estimable, with `adabnNote` saying
                # WHICH of the two reasons it is: no normalization at all, or a
                # normalization that keeps no running statistics. A number
                # there would be the first rung printed twice.
                #
                # `adabnLayers` is what AdaBN could act on; `adabnNormLayers`
                # is what the backbone normalizes with. The pair is what makes
                # the `layernorm` row the control rather than a second empty
                # row: two layers that normalize, zero that can be re-estimated.
                "predictTargetAccuracy": predict_accuracy,
                "adabnTargetAccuracy": adabn["accuracy"],
                "adabnLayers": adabn["layers"],
                "adabnNormLayers": adabn["normLayers"],
                "adabnBatches": adabn["batches"],
                "adabnNote": adabn["note"],
                # What the animation stands on, as scalars beside the arrays, so
                # a reader of `barrido.json` can check the frame schedule
                # against the steps the run applied without opening the `.npz`.
                "frameStride": frame_stride(scale),
                # The DECLARED unit beside the derived one. A reader of two
                # artefacts has to see the grid change unit rather than find
                # `frameStride` holding a different number for no stated reason.
                "frameEpochs": frame_epochs(scale),
                "trajectoryFrames": len(phase_two["trajectory"]),
                "phase1Epochs": phase_one["epochs"],
                "phase1BestEpoch": phase_one["bestEpoch"],
                "phase1ValidAccuracy": phase_one["bestValidAccuracy"],
                "phase1RestoredValidAccuracy": phase_one["restoredValidAccuracy"],
                "phase2Steps": phase_two["steps"],
                "phase2BestStep": phase_two["bestStep"],
                # The criterion's own unit. `phase2BestStep` is still a step
                # -- every checkpoint grid and the table's own `paso que
                # eligió` column read steps -- and it is now always the END of
                # a pass, which is the whole of what the change bought. The
                # pass count, the chosen pass and the unread tail travel
                # beside it so a reader can divide one by the other without
                # finding this module.
                "phase2StepsPerEpoch": phase_two["stepsPerEpoch"],
                "phase2Epochs": phase_two["epochs"],
                "phase2BestEpoch": phase_two["bestEpoch"],
                "phase2StepsWithoutARead": phase_two["stepsWithoutARead"],
                "phase2EpochPass": phase_two["epochPass"],
                "phase2EpochDefinedBy": phase_two["epochDefinedBy"],
                # The ORACLE: one reading of the target's TEST role per step.
                # It is the series the per-family table's oracle column is the
                # maximum of, and it enters no criterion -- see
                # `TARGET_SERIES`.
                TARGET_SERIES_FIELD: phase_two[TARGET_SERIES_FIELD],
                # THE CRITERION'S OWN VALUE, which the cell record did not
                # carry: the three watchman readings below were the only ones
                # here while the watchman WAS the criterion, and after
                # 2026-10-19 that left the artefact publishing three readings
                # of something that chooses nothing and none of what does.
                "cefBest": phase_two["bestCef"],
                "watchmanBest": phase_two["bestWatchman"],
                "watchmanFinal": phase_two["finalWatchman"],
                "watchmanRestored": phase_two["restoredWatchman"],
                # The share of the objective each declared component commanded.
                # Without it "the term did nothing" and "the term had no weight"
                # are the same number.
                "supervised": supervised,
                # What phase 2 MINIMIZED, read off the steps it minimized. The
                # two shares beside it are an independent reduction of the same
                # curve, so a reader can cross them -- and so can a test, which
                # is what makes "the supervised term left the loss" a
                # measurement instead of a sentence in a docstring.
                "phaseTwoObjective": objective_mean,
                "contribution": contribution,
                "contributionGlobal": global_share,
                "contributionLocal": local_share,
                "adaptationShare": (contribution / (supervised + contribution)
                                    if (supervised + contribution) > 0 else 0.0),
                # The weighting's cost on this cell, as three flat scalars so
                # a reader of `barrido.json` meets it beside the component
                # shares it explains, rather than having to find this module.
                "weightingCostRatio": cost["ratio"],
                "weightingCostOmegaSum": cost["omegaSum"],
                "weightingCostGlobalUnweighted": cost["globalUnweighted"],
                "parameters": sum(p.numel() for p in model.parameters()),
                # CREDA's control: phase 1 is only frozen if the source really
                # stayed still through phase 2, and both halves are read
                # because a BatchNorm whose buffers moved has not stayed still
                # either. It is the freezing this measures and never an arm
                # comparison, so retiring the floor arm takes nothing from it.
                "sourceMoved": moved,
                "sourceBuffersMoved": buffers_moved,
                # The same freeze, read in the space the FIGURE draws rather
                # than in the parameters. A frozen encoder whose positions
                # moved would mean the reading itself is not reproducible --
                # a mode left in `train`, a chunk-dependent normalization --
                # and the panel's whole composition rests on the source cloud
                # being one fixed frame. `latentTargetMoved` is the same
                # reading on the encoder phase 2 does move, so the zero above
                # is accompanied by the evidence that it can be non-zero.
                "latentSourceMoved": _moved(
                    latent_before["source"]["positions"],
                    latent_after["source"]["positions"]),
                "latentTargetMoved": _moved(
                    latent_before["target"]["positions"],
                    latent_after["target"]["positions"]),
                "phase2Note": phase_two["note"],
                "secondsPerPhase1Epoch": phase_one["secondsPerEpoch"],
                "secondsPerPhase2Step": phase_two["secondsPerStep"],
                "sigma": measured_sigma,
                "hyper": {k: v for k, v in hyper.items()},
                # Every record carries the scale it was measured at.
                "scale": scale.stamp(),
            }

    elapsed = time.perf_counter() - started
    for row in rows.values():
        row["cellElapsedSeconds"] = elapsed
    step_rates = [row["secondsPerPhase2Step"] for row in rows.values()
                  if not math.isnan(row["secondsPerPhase2Step"])]
    return {"rows": rows, "phase1": {k: v for k, v in phase_one.items()
                                     if k != "state"},
            # Keyed by arm because phase 2 is per arm: `phase1` above is shared
            # and this is not. It travels OUTSIDE `rows` on purpose -- these are
            # arrays, `barrido.json` is where `rows` go, and a `np.ndarray`
            # inside a row would raise at `json.dumps` rather than at the one
            # place that knows arrays belong in the `.npz`.
            "latent": latent,
            "sigma": measured_sigma, "elapsedSeconds": elapsed,
            "rates": {"secondsPerPhase1Epoch": phase_one["secondsPerEpoch"],
                      "secondsPerPhase2Step": _mean(step_rates),
                      "instancesPerBag": scale.instances_per_bag},
            "scale": scale.stamp()}


# ------------------------------------------------------------------ the searches

def _optuna():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def search_phase_one(familia_by_seed: dict, scale: Scale, device: torch.device,
                     report=print) -> tuple[dict, dict]:
    """Phase 1's hyperparameters, ONE configuration per family.

    Shared by both normalizations on purpose: per-row hyperparameters would
    confound the normalization axis with the hyperparameters and ruin the one
    axis this row varies on -- CREDA's own reason, in `buscar_hp1`.

    The objective is SOURCE VALID accuracy, maximized, and ties within
    `TIE_WIDTH` are broken by the valid loss, which still discriminates where
    the accuracy is against its ceiling. Three of the eight declared ranges are
    searched here because phase 1's own term reads them; see
    `PHASE_ONE_DIMENSIONS`.
    """
    optuna = _optuna()
    base = declared_hyper(scale)
    trials_done: list[dict] = []

    def objective(trial):
        hyper = dict(base)
        hyper["lr1"] = trial.suggest_float("lr1", *PHASE_ONE_OPTIMIZER_RANGES["lr1"],
                                           log=True)
        hyper["weightDecay1"] = trial.suggest_float(
            "weightDecay1", *PHASE_ONE_OPTIMIZER_RANGES["weightDecay1"], log=True)
        hyper["attentionGamma"] = trial.suggest_float(
            "attentionGamma", *config.ATTENTION_GAMMA_RANGE)
        hyper["attentionTemperature"] = trial.suggest_float(
            "attentionTemperature", *config.ATTENTION_TEMPERATURE_RANGE, log=True)
        hyper["dim"] = trial.suggest_categorical("dim",
                                                 list(PHASE_ONE_DIM_CHOICES))
        multiplier = trial.suggest_categorical("sigmaMultiplier",
                                               list(SIGMA_MULTIPLIERS))
        accuracies, losses, frozen_sigmas = [], [], []
        for normalization in NORMALIZATIONS:
            for seed in scale.search_seeds:
                familia = familia_by_seed[seed]
                backbone = NORMALIZATIONS[normalization]
                material = material_for(familia, seed, scale, backbone)
                # `hyper` goes INTO the binding: `dim` is a global this trial
                # chose, and `mlp.MLP` reads it when `build_arm` constructs.
                with material_bound(familia, scale, backbone, hyper):
                    source = material["source"]
                    source_train, source_valid, _ = bags.roles(source)
                    probe = build_arm(TOY_ADAPTED, material, scale, hyper, device, seed)
                    hyper["kernelSigma"] = multiplier * embedding_scales(
                        probe, source, source.train_idx, device)["withinBag"]
                    model = build_arm(TOY_ADAPTED, material, scale, hyper, device, seed)
                    generator = torch.Generator().manual_seed(seed + 9973)
                    phase_one = train_phase_one(model, source_train, source_valid,
                                                device, scale, hyper, generator)
                    # CREDA's own moment for the bandwidth, recorded per
                    # trial so the record says what the chosen multiplier will
                    # multiply. It enters no criterion -- the objective below
                    # is source valid accuracy and nothing else.
                    frozen_sigmas.append(frozen_source_sigma(
                        model, source, source.train_idx, device)["sigma"])
                accuracies.append(phase_one["bestValidAccuracy"])
                losses.append(phase_one["history"][phase_one["bestEpoch"]][2])
        trial.set_user_attr("validLoss", float(np.mean(losses)))
        trial.set_user_attr("kernelSigma", float(hyper["kernelSigma"]))
        trial.set_user_attr("frozenSigma", float(np.mean(frozen_sigmas)))
        return float(np.mean(accuracies))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=scale.trials_1)
    done = [t for t in study.trials if t.value is not None]
    if not done:
        raise RuntimeError("the phase-1 search completed no trial")
    top = max(t.value for t in done)
    tied = [t for t in done if t.value >= top - TIE_WIDTH]
    winner = min(tied, key=lambda t: t.user_attrs["validLoss"])
    for trial in done:
        trials_done.append({"params": dict(trial.params), "value": float(trial.value),
                            "validLoss": trial.user_attrs["validLoss"],
                            "kernelSigma": trial.user_attrs["kernelSigma"],
                            "frozenSigma": trial.user_attrs["frozenSigma"]})
    report(f"    {familia_by_seed[scale.search_seeds[0]].nombre:8s} phase 1: "
           f"{len(done)} trials, acc {top:.4f}, {len(tied)} within {TIE_WIDTH} "
           f"-> the valid loss breaks the tie "
           f"({winner.user_attrs['validLoss']:.4f})")
    chosen = dict(winner.params)
    chosen["kernelSigma"] = winner.user_attrs["kernelSigma"]
    ranges = searched_ranges("phase1")
    return chosen, {
        "trials": len(done), "bestAccuracy": float(top), "tied": len(tied),
        "validLoss": winner.user_attrs["validLoss"],
        "frozenSigma": winner.user_attrs["frozenSigma"],
        "frozenStatistic": FROZEN_SIGMA_STATISTIC,
        "criterion": "max(sourceValidAccuracy), ties by validLoss",
        "dimensions": sorted(ranges),
        # Every searched dimension WITH ITS RANGE, and the count beside it. A
        # record naming dimensions alone says a value was chosen and not from
        # where, so neither half travels without the other.
        "ranges": {name: (list(span) if isinstance(span, tuple) else span)
                   for name, span in ranges.items()},
        "dimensionCount": len(ranges),
        "trialsPerDimension": len(done) / len(ranges) if ranges else float("nan"),
        "budgetNote": SEARCH_BUDGET_NOTE,
        "notSearched": PHASE_ONE_NOT_SEARCHED,
        "perTrial": trials_done,
        "scale": scale.stamp(),
    }


#: How close two phase-1 trials have to be before the tie is broken by the loss
#: instead of by the accuracy. CREDA's own `1e-2`.
TIE_WIDTH = 1e-2


def _phase_two_range_block(trials: int) -> dict:
    """The dimensions/ranges/counts half of the phase-2 record.

    Lifted out so the two records compose the identical block from the identical
    `searched_ranges` call: the one failure a record of a search has that a
    reader cannot catch is two phases describing their own axes by two
    conventions.
    """
    ranges = searched_ranges("phase2")
    return {
        "dimensions": sorted(ranges),
        "ranges": {name: (list(span) if isinstance(span, tuple) else span)
                   for name, span in ranges.items()},
        "dimensionCount": len(ranges),
        "trialsPerDimension": trials / len(ranges) if ranges else float("nan"),
        "budgetNote": SEARCH_BUDGET_NOTE,
        "notSearched": PHASE_TWO_NOT_SEARCHED,
    }


def search_phase_two(familia_by_seed: dict, hyper1: dict, scale: Scale,
                     device: torch.device, report=print) -> tuple[dict, dict]:
    """Phase 2's ONE remaining hyperparameter, with phase 1 frozen and trained
    once per search seed.

    The objective is the WATCHMAN's minimum, minimized -- the same criterion the
    stop uses, which is the point: a search that minimized something else would
    be choosing a configuration for a stopping rule that reads another number.
    Zero target labels enter it.

    **One dimension, and it used to be five.** `ceiling`, `ceilingLocal`,
    `rampDelta` and `rampDeltaLocal` were four of those five, and all four are
    Eq. (39)'s schedule -- retired with it (`RETIRED_RANGES`). `tauLocal` is
    what is left: it reaches the local correspondence of Eqs. (30)-(31) through
    `total_correspondence`'s temperature and nothing else, which is why it was
    the one phase-2 dimension that was never part of the schedule.

    A one-dimensional TPE search is a weaker instrument than a five-dimensional
    one is normally assumed to be, and that reads the wrong way round: four of
    those five dimensions moved a coefficient that is now a declared constant,
    so searching them was spending trials on a knob this protocol does not
    have. `trials_2` is unchanged, so every trial now lands on the one
    dimension that does something.
    """
    optuna = _optuna()
    frozen = {}
    for normalization in NORMALIZATIONS:
        for seed in scale.search_seeds:
            familia = familia_by_seed[seed]
            backbone = NORMALIZATIONS[normalization]
            material = material_for(familia, seed, scale, backbone)
            # `hyper1` carries the `dim` phase 1 chose, and this cached phase 1
            # has to be built at THAT width or the state it freezes could not
            # be loaded into the arms every trial below constructs.
            cached_hyper = {**declared_hyper(scale), **hyper1}
            with material_bound(familia, scale, backbone, cached_hyper):
                source_train, source_valid, _ = bags.roles(material["source"])
                model = build_arm(TOY_ADAPTED, material, scale, cached_hyper,
                                  device, seed)
                generator = torch.Generator().manual_seed(seed + 9973)
                phase_one = train_phase_one(model, source_train, source_valid,
                                            device, scale, hyper1, generator)
                # The same moment `run_cell` freezes it at, for the same
                # reason: this search runs PHASE 2, so it has to run at the
                # bandwidth phase 2 runs at. Choosing `tauLocal` at a
                # different bandwidth from the cells would choose it in a
                # landscape no cell trains in.
                sigma = frozen_source_sigma(model, material["source"],
                                            material["source"].train_idx,
                                            device)["sigma"]
            frozen[(normalization, seed)] = (material, phase_one["state"], sigma)

    def objective(trial):
        hyper = {**declared_hyper(scale), **hyper1}
        hyper["tauLocal"] = trial.suggest_float("tauLocal", *config.TAU_LOCAL_RANGE,
                                                log=True)
        hyper["lr2"] = trial.suggest_float(
            "lr2", *PHASE_TWO_OPTIMIZER_RANGES["lr2"], log=True)
        hyper["weightDecay2"] = trial.suggest_float(
            "weightDecay2", *PHASE_TWO_OPTIMIZER_RANGES["weightDecay2"],
            log=True)
        cefs, watchmen, targets = [], [], []
        for (normalization, seed), (material, shared, sigma) in frozen.items():
            familia = familia_by_seed[seed]
            backbone = NORMALIZATIONS[normalization]
            multiplier = float(hyper.get("sigmaMultiplier",
                                         SIGMA_MULTIPLIER_NEUTRAL))
            cell_hyper = {**hyper, "kernelSigma": multiplier * sigma}
            with material_bound(familia, scale, backbone, cell_hyper):
                source_train, source_valid, _ = bags.roles(material["source"])
                _, _, target_eval = bags.roles(material["target"])
                model = build_arm(TOY_ADAPTED, material, scale, cell_hyper,
                                  device, seed)
                model.load_state_dict(shared, strict=False)
                for name, parameter in model.named_parameters():
                    if not name.startswith("target_encoder."):
                        parameter.requires_grad_(False)
                copy_source_to_target(model)
                generator = torch.Generator().manual_seed(seed + 9973)
                phase_two = adapt_phase_two(model, source_train, source_valid,
                                            device, scale, cell_hyper,
                                            generator)
                # Recorded and never read by the criterion, so the two readings
                # can be compared afterwards: which of them would have ordered
                # the trials better is a question the artefact has to be able to
                # answer, and it cannot if only the winner was kept.
                targets.append(harness.target_accuracy(model, target_eval, device))
            cefs.append(phase_two["bestCef"])
            watchmen.append(phase_two["bestWatchman"])
        trial.set_user_attr("targetAccuracy", float(np.mean(targets)))
        # Recorded and never read by the search, for the same reason the run
        # records it: which of the two readings would have ordered the TRIALS
        # better is a question the artefact has to be able to answer, and it
        # cannot if only the winner's was kept.
        trial.set_user_attr("watchman", float(np.mean(watchmen)))
        return float(np.mean(cefs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=scale.trials_2)
    done = [t for t in study.trials if t.value is not None]
    if not done:
        raise RuntimeError("the phase-2 search completed no trial")
    winner = max(done, key=lambda t: t.value)
    report(f"    {familia_by_seed[scale.search_seeds[0]].nombre:8s} phase 2: "
           f"{len(done)} trials, max(cef) {winner.value:.4f}")
    return dict(winner.params), {
        "trials": len(done), "bestCef": float(winner.value),
        "bestWatchman": float(winner.user_attrs["watchman"]),
        "criterion": "max(cef)",
        # The search and the EPOCH SELECTION used to read different numbers:
        # the search minimized the watchman while the selection maximized cef.
        # A search that optimized something the stop does not read is choosing
        # a configuration for a rule that reads another quantity.
        "criterionChange": (
            "2026-10-19: it was min(watchman), and the stop already read "
            "max(cef). Measured in CREDA, the watchman ordered the trials "
            "BACKWARDS in two of three families -- r(watchman, test) = -0.232 "
            "in lunas against r(cef, test) = +0.916 -- so the search was "
            "picking against the criterion"),
        # The bandwidth every trial of this search ran at, per
        # (normalization, seed): CREDA's frozen statistic times whatever
        # multiplier phase 1 chose. Recorded because a reader comparing this
        # search's `tauLocal` with a cell's has to be able to see that the two
        # ran in the same landscape.
        "frozenSigma": {f"{normalization}|{seed}": sigma
                        for (normalization, seed), (_, _, sigma)
                        in frozen.items()},
        "frozenStatistic": FROZEN_SIGMA_STATISTIC,
        "frozenSigmaMultiplier": float(hyper1.get(
            "sigmaMultiplier", SIGMA_MULTIPLIER_NEUTRAL)),
        "retired": {name: entry["was"] for name, entry in RETIRED_RANGES.items()},
        "fixedCoefficient": FIXED_COEFFICIENT,
        "perTrial": [{"params": dict(t.params), "cef": float(t.value),
                      "watchman": float(t.user_attrs["watchman"]),
                      "targetAccuracy": t.user_attrs["targetAccuracy"]}
                     for t in done],
        "scale": scale.stamp(),
        **_phase_two_range_block(len(done)),
    }


# -------------------------------------------------------------------- the sweep

def key(*parts) -> str:
    """One cell's key in `barrido.json`. CREDA uses its own `SEP`; this declares
    its own rather than importing a private name."""
    return SEP.join(str(p) for p in parts)


SEP = "|"


def sweep(scale: Scale, device: torch.device | None = None,
          families: tuple[str, ...] | None = None,
          report=print) -> dict:
    """The whole programme at one scale: two searches per family, then the one
    declared arm over the report seeds.

    Returns everything `guardar` writes. Nothing here releases the full run: the
    scale is an argument and `PILOT` is what a caller with no authorization has.

    **No `rungs` and no `floors` come back**, because with one arm there is no
    pair to subtract. The two keys are gone from this return and from the
    artefact rather than returned empty: an empty list reads as "measured, and
    there was nothing", and what is true is that nothing was measured because
    nothing can be. See the module docstring on what the single-arm shape costs.
    """
    device = harness.resolve_device() if device is None else device
    names = tuple(families) if families else tuple(n for n, _ in DURAS)
    started = time.perf_counter()

    cells: dict[str, dict] = {}
    rows: dict[str, dict] = {}
    hp1: dict[str, dict] = {}
    hp2: dict[str, dict] = {}
    records1: dict[str, dict] = {}
    records2: dict[str, dict] = {}
    per_family: dict[str, dict] = {}
    rates: list[dict] = []

    for name in names:
        report(f"  {name}")
        search_material = {seed: build_family(name, seed, scale)
                           for seed in scale.search_seeds}
        chosen1, record1 = search_phase_one(search_material, scale, device, report)
        chosen2, record2 = search_phase_two(search_material, chosen1, scale,
                                            device, report)
        hyper = {**declared_hyper(scale), **chosen1, **chosen2}
        hp1[name], hp2[name] = chosen1, chosen2
        records1[name], records2[name] = record1, record2

        family_rows: dict[str, dict[str, list]] = {}
        histories: dict[str, dict] = {}
        latent: dict[str, dict] = {}
        for normalization in NORMALIZATIONS:
            by_arm: dict[str, list] = {arm: [] for arm in TOY_ARM_IDS}
            for seed in scale.report_seeds:
                familia = build_family(name, seed, scale)
                # `kernelSigma` is re-measured per cell: the multiplier is what
                # the search chose, and the distance it multiplies is this
                # cell's own. A single absolute value cannot be in band for six
                # cells, which is the whole reason the grid is relative.
                cell = run_cell(familia, normalization, seed, scale,
                                _resolved_sigma(hyper, chosen1, familia,
                                                normalization, scale, device),
                                device)
                rates.append(cell["rates"])
                report(f"      {name}/{normalization}/seed {seed}: "
                       f"{cell['elapsedSeconds']:.1f}s")
                for arm_id, row in cell["rows"].items():
                    by_arm[arm_id].append(row)
                    cells[key(name, normalization, arm_id, seed)] = row
                    # The SAME handle `guardar` writes the flat scalars under,
                    # so a reader joins a cell's numbers to its positions
                    # without a second convention to learn.
                    latent[key(normalization, arm_id, seed)] = (
                        cell["latent"][arm_id])
                histories[key(normalization, seed)] = {
                    "phase1": cell["phase1"], "sigma": cell["sigma"]}
            family_rows[normalization] = by_arm
            for arm_id, arm_rows in by_arm.items():
                rows[key(name, normalization, arm_id)] = {
                    "runs": arm_rows,
                    "summary": {d: harness.spread([float(r[d]) for r in arm_rows])
                                for d in ("targetAccuracy", "sourceAccuracy",
                                          "contribution", "supervised",
                                          "adaptationShare")},
                    "scale": scale.stamp(),
                }
        per_family[name] = {"rows": family_rows, "histories": histories,
                            "latent": latent}

    elapsed = time.perf_counter() - started
    return {
        "flags": flags(scale, device, names, rates, elapsed),
        "cells": cells, "rows": rows,
        "hp1": hp1, "hp2": hp2, "records1": records1, "records2": records2,
        "perFamily": per_family,
        "cost": cost_report(rates, scale, elapsed, families=len(names)),
        "scale": scale.stamp(),
    }


def _resolved_sigma(hyper: dict, chosen1: dict, familia: toy_bags.Familia,
                    normalization: str, scale: Scale,
                    device: torch.device) -> dict:
    """`hyper` with the PHASE-1 bootstrap bandwidth re-measured on THIS cell.

    **This is no longer the bandwidth phase 2 runs at**, and the rename of what
    it answers is the whole of what changed. `run_cell` now freezes CREDA's own
    `sigma_de_la_fuente` the instant phase 1 ends and overwrites `kernelSigma`
    with the searched multiplier times THAT (see `frozen_source_sigma`). What
    this function still provides is the bandwidth Eq. (14)'s consensus pools
    with BEFORE any of that exists -- a bag score needs a bandwidth, and CREDA
    has no counterpart because its phase 1 is CE over points.

    It stays the capped WITHIN-BAG median and not the frozen pooled one on
    purpose: `embedding_scales`' own docstring records the measurement that a
    pooled median leaves Eq. (16)'s weights at a fraction of a percent of
    uniform, which would make phase 1 numerically mean-pool while every flag
    said otherwise. Both numbers and their ratio reach the record.

    The search chose a MULTIPLIER of the measured median; what it multiplies is
    per-cell. Passing the search's absolute value through would hand five of the
    six cells a bandwidth calibrated on a different family's scale.

    Measured ONCE per cell, at the first report seed, and held across every seed
    of that cell: a bandwidth re-measured per repetition would be a
    hyperparameter that varied with the repetition, which is not one. The cost
    is that the seeds after the first run at a bandwidth calibrated on the
    first one's draw, and that is the trade every chosen constant makes.
    """
    multiplier = chosen1.get("sigmaMultiplier")
    if multiplier is None:
        return dict(hyper)
    backbone = NORMALIZATIONS[normalization]
    material = material_for(familia, int(scale.report_seeds[0]), scale, backbone)
    with material_bound(familia, scale, backbone):
        probe = build_arm(TOY_ADAPTED, material, scale, hyper, device,
                          int(scale.report_seeds[0]))
        distance = embedding_scales(probe, material["source"],
                                    material["source"].train_idx,
                                    device)["withinBag"]
    return {**hyper, "kernelSigma": multiplier * distance}


def work_counts(scale: Scale, families: int) -> dict:
    """How many phase-1 trainings and phase-2 runs a sweep at this scale does.

    Counted from the loops that actually run, because the factors do NOT
    compose: the trial counts multiply the SEARCHES and the report seeds
    multiply the CELLS, so a forecast that multiplied all of them together --
    which an earlier version of this function did -- overstates the cost by
    orders of magnitude. The earlier reading forecast 7578 hours off a 2.1
    second cell, and the number is recorded here because it looked like a
    forecast and was a product of unrelated ratios.
    """
    normalizations = len(NORMALIZATIONS)
    search_seeds = len(scale.search_seeds)
    report_seeds = len(scale.report_seeds)
    arms = len(TOY_ARM_IDS)
    phase_one = families * (
        scale.trials_1 * normalizations * search_seeds       # the phase-1 search
        + normalizations * search_seeds                      # frozen for search 2
        + normalizations * report_seeds                      # one per cell, shared
    )
    # Every declared arm runs phase 2. It used to be `arms - 1`, because the
    # floor arm declared no adaptation term and `adapt_phase_two` returned
    # without stepping; with the floor retired the subtraction would count ZERO
    # phase-2 runs per cell for a one-arm declaration, and a forecast of zero
    # for work that happens is worse than no forecast.
    phase_two = families * (
        scale.trials_2 * normalizations * search_seeds       # the phase-2 search
        + normalizations * report_seeds * arms               # one per cell, per arm
    )
    return {"families": families, "normalizations": normalizations,
            "searchSeeds": search_seeds, "reportSeeds": report_seeds,
            "arms": arms,
            "phase1Trainings": phase_one, "phase2Runs": phase_two,
            "phase1EpochCeiling": phase_one * scale.epochs_1,
            "phase2EpochCeiling": phase_two * scale.epochs_2,
            # Kept beside the epoch figure because `secondsPerPhase2Step` is
            # what the forecast multiplies, and a forecast that multiplied
            # epochs by a per-step rate would be wrong by the pass length.
            "phase2StepCeiling": (phase_two * scale.epochs_2
                                  * STEPS_PER_EPOCH_DECLARED)}


def cost_report(rates: list[dict], scale: Scale, elapsed: float,
                families: int = 3) -> dict:
    """What a phase-1 epoch and a phase-2 step cost, and what the full scale
    would cost from them.

    Built from MEASURED RATES rather than from the sweep's own wall time divided
    by ratios, for the reason `work_counts` records: the repetition factors
    multiply different loops and multiplying them together is not a forecast.

    Two honest bounds remain and are stated rather than absorbed. Both early
    stops end a phase before its declared length, so the epoch and step counts
    are CEILINGS and the forecast is an upper bound. And the per-epoch cost
    scales with how many instances a bag holds, so the rates measured at a
    reduced `points_per_domain` are scaled by that ratio -- linearly, which is
    an assumption about the encoder's cost and is named as one.
    """
    epoch_rates = [r["secondsPerPhase1Epoch"] for r in rates
                   if not math.isnan(r["secondsPerPhase1Epoch"])]
    step_rates = [r["secondsPerPhase2Step"] for r in rates
                  if not math.isnan(r["secondsPerPhase2Step"])]
    per_epoch, per_step = _mean(epoch_rates), _mean(step_rates)
    instance_scale = (FULL.instances_per_bag / scale.instances_per_bag
                      if scale.instances_per_bag else float("nan"))
    here = work_counts(scale, families)
    there = work_counts(FULL, families)
    forecast = instance_scale * (
        there["phase1EpochCeiling"] * per_epoch
        + there["phase2StepCeiling"] * per_step)
    return {
        "cells": len(rates),
        "secondsPerPhase1Epoch": per_epoch,
        "secondsPerPhase2Step": per_step,
        "secondsPerCell": (elapsed / len(rates)) if rates else float("nan"),
        "sweepSeconds": elapsed,
        "measuredAt": scale.name,
        "workHere": here,
        "workAtFullScale": there,
        "instanceScale": instance_scale,
        "forecastFullSeconds": forecast,
        "forecastFullHours": forecast / 3600.0,
        "caveat": ("an upper bound, not a measurement: both early stops end a "
                   "phase before its declared length, so the epoch and step "
                   "counts are ceilings, and the per-epoch cost is scaled "
                   "linearly in instances per bag, which is an assumption "
                   "about the encoder and not a measurement of it"),
        "scale": scale.stamp(),
    }


#: The two method axes, as the record names them: the function that carries
#: each one, the parameter, and where its published value is declared. The
#: VALUE is deliberately absent here and derived in `method_axes()` -- a literal
#: beside the function would be a second declaration, and the one that drifts is
#: always the copy.
METHOD_AXES = {
    "weightCrossBlock": {
        "module": "MIL_CREDA.conditional",
        "function": "weighted_blocks",
        "parameter": "weight_cross_block",
        "declaredAs": "CRUZADO_SOPESADO",
        "what": ("whether Eq. (26)'s confidence diagonal multiplies the CROSS "
                 "block K_st as well as the target block"),
    },
    "detachTargetEntropy": {
        "module": "MIL_CREDA.global_term",
        "function": "balanced_dependency_score",
        "parameter": "detach_target_entropy",
        "declaredAs": "RHO_T_SIN_GRAD",
        "what": ("whether H_2(K~_t) enters the per-class score detached, "
                 "removing the dispersive force without changing the value"),
    },
}

#: Why the effective value is read off the signature and not threaded through.
#: The honest limit of this recording, stated where it is recorded.
METHOD_AXES_NOTE = (
    "each axis is keyword-only with CREDA's declared value as its default and "
    "the call sites pass neither, so the effective value IS the default and is "
    "read with inspect.signature rather than copied. Threading a searchable "
    "value through would mean editing the call sites, which live in wiring.py")


def method_axes() -> dict:
    """The two named axes with the value THIS run used, derived.

    Read from each function's own keyword-only default and crossed against
    CREDA's `HP_2`, so neither side can move alone and no literal here can
    satisfy it. `test_the_record_names_the_two_axes_and_the_value_this_run_used`
    also asserts, from the syntax tree, that the call sites override neither --
    which is the premise that makes "the default" and "what ran" the same
    sentence.
    """
    import importlib
    import inspect

    from CREDA.toy_experiment import HP_2

    out = {}
    for key, entry in METHOD_AXES.items():
        function = getattr(importlib.import_module(entry["module"]),
                           entry["function"])
        parameter = inspect.signature(function).parameters[entry["parameter"]]
        out[key] = {
            "value": parameter.default,
            "parameter": entry["parameter"],
            "function": f"{entry['module']}.{entry['function']}",
            "what": entry["what"],
            # Where the published value is declared, and where this run's copy
            # of it comes from. Two different sentences, and both are owed:
            # the first is CREDA's word, the second is how it reached the run.
            "declaredBy": "CREDA.toy_experiment.HP_2",
            "declaredAs": entry["declaredAs"],
            "declaredValue": HP_2[entry["declaredAs"]],
            "carriedBy": "the function's own keyword-only default",
            "agreesWithCreda": parameter.default is HP_2[entry["declaredAs"]],
            "searched": False,
        }
    return out


def conditional_by_construction() -> dict:
    """That the term is conditional BY CONSTRUCTION, with the absence measured.

    **Why this is recorded rather than left to the table.** CREDA has a knob:
    `class_mixing` is `0.0` for the pure marginal and `1.0` for the pure
    conditional, and `PASO_COND` switches between the two extremes at a declared
    step. A table printing `1.0` for this package beside CREDA's column would
    read as "this run chose the conditional end" -- a value somebody picked.
    Nobody picked it. There is no knob here at all: `conditional.py` builds the
    Gram blocks class by class (Eq. 23), so the conditional character is the
    construction and not a setting.

    **The absence is MEASURED, over the whole package.** "There is no such
    knob" is a claim about every module under `src/MIL_CREDA/`, and a reading
    that stopped at one file could not support it -- so the search is performed
    and the module count travels with the answer. A reader who doubts it can
    see where it looked.
    """
    package = Path(__file__).resolve().parent.parent / "MIL_CREDA"
    modules = sorted(package.rglob("*.py"))
    carrying = sorted(module.name for module in modules
                      if "class_mixing" in module.read_text(encoding="utf-8"))
    return {
        "isConditional": True,
        "byAKnob": False,
        "knobHere": None,
        "equation": "23",
        "builtBy": "MIL_CREDA.conditional.conditional_blocks",
        "how": ("the Gram blocks are built class by class, so every block the "
                "global term reads is already per class; there is no explicit "
                "conditional entropy and nothing mixes a marginal in"),
        "credasKnob": "CREDA.models.CREDALoss.class_mixing",
        "credasKnobNote": ("0.0 is the pure marginal and 1.0 the pure "
                           "conditional, switched at CREDA's own PASO_COND. A "
                           "1.0 printed for this package would read as that "
                           "end having been chosen, which no run here chose"),
        # Where the absence was looked for, and how wide. Named so the claim is
        # checkable rather than taken.
        "searchedUnder": "src/MIL_CREDA/",
        "modulesSearched": len(modules),
        "modulesCarryingTheKnob": carrying,
    }


def flags(scale: Scale, device: torch.device, families: tuple[str, ...],
          rates: list[dict], elapsed: float) -> dict:
    """Every flag of this run, so a reader can say what it was measured under.

    `CREDA.toy_experiment.guardar` writes `banderas()` for exactly this reason:
    a table that does not say what it was measured under cannot be read.
    """
    return {
        # The two axes that had no name, with the pole this run used, and the
        # objective Eq. (21) left. All three are facts about the METHOD rather
        # than about the reduction, so they travel in every record.
        "methodAxes": method_axes(),
        "methodAxesNote": METHOD_AXES_NOTE,
        "conditionalByConstruction": conditional_by_construction(),
        "phaseTwoObjective": PHASE_TWO_OBJECTIVE,
        # The criterion the stop reads, in its own unit, with both derivations
        # and the asymmetry against CREDA's pass. It travels here because
        # "which step a run chose" cannot be read without knowing what a step
        # is being counted against, and because the unread tail is a fact
        # somebody would otherwise discover by measuring.
        "phaseTwoCriterion": PHASE_TWO_CRITERION,
        # And the one reading in this flow that touches `y_t`, declared where a
        # reader of the record meets it rather than in a docstring.
        "targetSeries": TARGET_SERIES,
        "scale": scale.stamp(),
        "revision": config.REVISION,
        "device": str(device),
        "families": list(families),
        "familyCodes": {k: list(v) for k, v in FAMILY_CODES.items()},
        "normalizations": dict(NORMALIZATIONS),
        "normalizationsAbsent": list(NORMALIZATION_ABSENT),
        "arms": list(TOY_ARM_IDS),
        # No `ladder` and no `floorOf`. They are not empty here, they are
        # ABSENT: one arm declares no pair, so there is no rung to name and no
        # floor to name one against. An empty list would read as a ladder that
        # was computed and came out with nothing in it.
        "armsNote": ("one arm, so no ladder, no rung and no floor margin. This "
                     "run reports a bare accuracy and nothing on this path says "
                     "what it is better or worse than; the floor arm the rung "
                     "was read against was retired by the operator"),
        # The arm declaration this run measured under, in full, with each arm's
        # own reason beside it. It travels because it is this toyset's own and
        # not the digit benchmark's: a reader who assumed `config.ARMS` would be
        # reading the wrong four arms for these numbers.
        "armDeclaration": {
            "arms": [dict(arm) for arm in TOY_ARMS],
            "owner": ("MIL_CREDA_Benchmark.toy_run.TOY_ARMS -- this module's "
                      "own, declared here and derived from nothing"),
            "notDerivedFrom": {
                "declaration": "MIL_CREDA_Benchmark.config.ARMS",
                "ids": list(config.ARMS_BY_ID),
                "why": ("the digit benchmark's declaration, for a different "
                        "experiment. It is pinned by tests/test_benchmark_"
                        "declarations.py, so adding an arm there to serve this "
                        "toyset would change the digit campaign and trip a pin "
                        "doing its job; and the protocol is not theirs either "
                        "-- one phase read at the last epoch under a ramped "
                        "coefficient, against two phases read at the best step "
                        "under a fixed one"),
                # Reported rather than hidden: the adapted arm's three
                # mechanism flags are now EQUAL to digit `G`'s, because the
                # confidence weighting came back on. An earlier version of this
                # field was `shapeAbsentThere` and claimed the opposite -- that
                # no digit arm had this shape -- which was the stated reason for
                # the separation and is no longer true. What differs from `G`
                # is the schedule, named here beside the match.
                "sameFlagsAsDigitArm": next(
                    (arm["id"] for arm in config.ARMS
                     if (arm["adaptation"], arm["weighting"], arm["local"],
                         arm["unit"])
                     == (TOY_ARMS_BY_ID[TOY_ADAPTED]["adaptation"],
                         TOY_ARMS_BY_ID[TOY_ADAPTED]["weighting"],
                         TOY_ARMS_BY_ID[TOY_ADAPTED]["local"],
                         TOY_ARMS_BY_ID[TOY_ADAPTED]["unit"])), None),
                "whatDiffersInstead": ("the schedule: the digit arm's two "
                                       "coefficients come out of harness.ramp "
                                       "with a searched ceiling; this path "
                                       "fixes both at "
                                       f"{FIXED_COEFFICIENT} and calls no ramp"),
            },
            "weightingGates": WEIGHTING_GATES,
            "weightingCost": WEIGHTING_COST,
            "sharedWithDigitBenchmark": SHARED_WITH_DIGIT_BENCHMARK,
        },
        "bentGlobals": {name: _DECLARED_GLOBALS[name] for name in BENT_GLOBALS},
        "searchRanges": {k: (v if isinstance(v, str) else list(v))
                         for k, v in SEARCH_RANGES.items()},
        # The four declared ranges this run did NOT search, with what each one
        # was. Reported beside the four that were, because "four of eight" is
        # only readable if the other four are named -- and because a reader of
        # an older `banderas.json` will find eight here and has to be able to
        # tell which run they are holding.
        "retiredRanges": {name: {"range": list(entry["range"]),
                                 "was": entry["was"], "why": entry["why"]}
                          for name, entry in RETIRED_RANGES.items()},
        "fixedCoefficient": {
            "value": FIXED_COEFFICIENT,
            "equation": "Eq. (39): L = L_src + lambda_glob*L_glob + lambda_loc*L_loc",
            "what": ("both lambdas, at every step, with no ramp. harness.ramp "
                     "is not called on this path"),
        },
        "searchSplit": {"phase1": sorted(searched_ranges("phase1")),
                        "phase2": sorted(searched_ranges("phase2")),
                        "ranges": {"phase1": {k: (list(v) if isinstance(v, tuple)
                                                  else v) for k, v
                                              in searched_ranges("phase1").items()},
                                   "phase2": {k: (list(v) if isinstance(v, tuple)
                                                  else v) for k, v
                                              in searched_ranges("phase2").items()}},
                        "budget": SEARCH_BUDGET_NOTE,
                        "notSearched": {"phase1": PHASE_ONE_NOT_SEARCHED,
                                        "phase2": PHASE_TWO_NOT_SEARCHED},
                        "retired": sorted(RETIRED_RANGES),
                        "why": PHASE_ONE_SPLIT_REASON},
        "phase1NotSearched": PHASE_ONE_NOT_SEARCHED,
        "criteria": {"phase1Stop": "patience_1 epochs without improving "
                                   "max(sourceValidAccuracy)",
                     "phase1Search": "max(sourceValidAccuracy), ties by validLoss",
                     "phase2Stop": "none: phase 2 runs its epochs_2 ceiling "
                                   "whole",
                     "phase2Select": "max(cef) over the WHOLE run",
                     "phase2Search": "max(cef)"},
        "cef": ("exp(H(histogram of the bag argmax over the target TRAIN "
                "role)), in [1, classes]; zero target labels; maximized; it "
                "is what the epoch selection and the phase-2 search both read"),
        "watchman": ("head o target-encoder over the SOURCE valid role against "
                     "SOURCE labels; zero target labels; RECORDED and no "
                     "longer read by any criterion since 2026-10-19"),
        "syncRetired": {
            "constant": "config.SOURCE_TO_TARGET_SYNC_STEPS",
            "value": config.SOURCE_TO_TARGET_SYNC_STEPS,
            "why": ("two phases means the source freezes and nothing transmits "
                    "after the one copy"),
            "declaredRisk": ("MIL-CREDA/AGREED.md records the full arm falling "
                             "below the floor on digits without the "
                             "transmission (0.2500 vs 0.5833 on U->M; 0.0833 "
                             "vs 0.1389 on M->S), measured under resnet18 on "
                             "digits and not under an MLP on 3-D points"),
            # The consequence of the single-arm shape, stated where the risk is
            # stated rather than left for a reader to work out: the risk is a
            # fall BELOW A FLOOR, and this toyset declares no floor.
            "riskNotMeasurableHere": ("the risk names a fall below a floor and "
                                      "this toyset declares one arm, so "
                                      "nothing on this path can answer whether "
                                      "it reproduces on these families. The "
                                      "two readings that computed it -- a rung "
                                      "and a floor margin -- were retired with "
                                      "the floor arm, deliberately and not by "
                                      "oversight"),
        },
        "cost": cost_report(rates, scale, elapsed, families=len(families)),
    }


PHASE_ONE_SPLIT_REASON = (
    "phase 1's supervised term is Eq. (21) over Eq. (19)'s bag representation, "
    "which reads kernelSigma, attentionGamma and attentionTemperature through "
    "wiring.Arm.weights_for; a phase-2 trial varying them would vary phase 1 "
    "too, and phase 1 would stop being trained once and frozen"
)


# ---------------------------------------------------------------- the artefact

def guardar(destino, barrido: dict) -> list[Path]:
    """Write the artefact: `banderas.json`, `barrido.json`, one `<family>.npz`.

    The shape is `CREDA.toy_experiment.guardar`'s, with MIL-CREDA's own
    contents. The two habits that shape carries over are the ones that cost
    CREDA a whole sweep: a scalar dict travels as a 0-d array holding its JSON
    rather than as a second file that can be lost on its own, and what stays OUT
    of that dict is decided by the SHAPE of the value and not by a list of names
    -- a blacklist failed exactly as a blacklist fails.

    And it reads back what it just wrote: `np.savez` does not fail on a value it
    cannot serialize without pickle, it stores it and raises on load, in the
    other notebook and with no clue which key it was.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    written = []

    (destino / "banderas.json").write_text(
        json.dumps(barrido["flags"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    written.append(destino / "banderas.json")

    (destino / "barrido.json").write_text(json.dumps({
        "celdas": {k: _without_curves(v) for k, v in barrido["cells"].items()},
        "filas": {k: {"summary": v["summary"], "scale": v["scale"]}
                  for k, v in barrido["rows"].items()},
        "hp1": barrido["hp1"], "hp2": barrido["hp2"],
        "busqueda1": {k: _without_trials(v) for k, v in barrido["records1"].items()},
        "busqueda2": {k: _without_trials(v) for k, v in barrido["records2"].items()},
        "trials1": {k: v["perTrial"] for k, v in barrido["records1"].items()},
        "trials2": {k: v["perTrial"] for k, v in barrido["records2"].items()},
        "costo": barrido["cost"],
        "escala": barrido["scale"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written.append(destino / "barrido.json")

    for name, payload in barrido["perFamily"].items():
        arrays: dict[str, np.ndarray] = {}
        for normalization, by_arm in payload["rows"].items():
            for arm_id, runs in by_arm.items():
                for run in runs:
                    k = key(normalization, arm_id, run["seed"])
                    arrays[key(k, "info")] = np.array(json.dumps(
                        _flat_scalars(run), ensure_ascii=False))
        for history_key, history in payload["histories"].items():
            arrays[key(history_key, "hist_s")] = np.asarray(
                history["phase1"]["history"], dtype=np.float32)
            arrays[key(history_key, "sigma")] = np.asarray(
                [p["sigma"] for p in history["sigma"]["points"]], dtype=np.float32)
        # The latent positions, one member per name in `LATENT_MEMBERS` under
        # the same handle the flat scalars use. Derived from that tuple rather
        # than written out: a member added there and forgotten here would leave
        # the figure reading a key that never travelled, and `cargar` reads the
        # identical tuple on the way back.
        # `member` and not `name`: the family's own name is the loop variable
        # this block sits inside, and shadowing it wrote the `.npz` under the
        # last member's name. `test_the_artefact_round_trips_the_positions_
        # through_real_files` is what said so, at `guardar`'s own read-back.
        for handle, members in payload.get("latent", {}).items():
            missing = [m for m in LATENT_MEMBERS if m not in members]
            if missing:
                raise KeyError(
                    f"the latent payload of {handle!r} does not carry "
                    f"{missing}: a view that never reached disk is a panel the "
                    "figure draws empty, and an empty panel reads like a "
                    "measurement that came out flat")
            for member in LATENT_MEMBERS:
                arrays[key(handle, member)] = np.asarray(members[member])
        ruta = destino / f"{name}.npz"
        np.savez_compressed(ruta, **arrays)
        written.append(ruta)

    for name in barrido["perFamily"]:
        with np.load(destino / f"{name}.npz", allow_pickle=False) as z:
            if not z.files:
                raise ValueError(f"{name}.npz came back empty")
    return written


def _without_curves(row: dict) -> dict:
    """A cell record without the per-step curve: the curve is a trace and
    travels in the `.npz`, not in the JSON a reader opens."""
    return {k: v for k, v in row.items() if k not in ("curve", "history")}


def _without_trials(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "perTrial"}


def _flat_scalars(record: dict) -> dict:
    """The scalars of a record, decided by the SHAPE of each value.

    `bool` is tested before `int` because `bool` IS a subclass of `int`: without
    that branch `True` comes back as `1.0`, the write stays green, and the defect
    exists only on the way back.
    """
    def flat(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        return value

    out = {}
    for name, value in record.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (list, tuple)) and value and isinstance(value[0],
                                                                     np.ndarray):
            continue
        candidate = flat(value)
        try:
            json.dumps(candidate)
        except TypeError as error:
            raise TypeError(
                f"the field {name!r} of a cell record is not serializable "
                f"({type(value).__name__}): give it an array of its own, or say "
                f"why it is excluded") from error
        out[name] = candidate
    return out


#: What an artefact whose scale stamp is from another protocol raises, so the
#: callers that can do something about it (a test fixture that would rather
#: skip) can tell it from an ordinary read failure.
class ArtefactoDeOtroProtocolo(RuntimeError):
    """The stored artefact was measured under a `Scale` this module no longer
    declares, so its numbers describe a different experiment."""


def _refuse_a_foreign_stamp(origen: Path, stamp: dict) -> None:
    """Refuse an artefact whose scale stamp does not name this module's own
    `Scale` fields.

    **Derived from `Scale` itself and never from a list written here**, so the
    next time a field is added or retired this keeps working instead of
    checking for the previous change.

    It exists because of the shape of the failure it replaces. On 2026-10-19
    `steps_2` became `epochs_2`, and the pilot artefact on disk kept the old
    stamp: nothing refused it, `cargar` returned it, and nine renderers died
    later with `KeyError: 'epochs_2'` deep inside a `suptitle` -- a message
    that names neither the artefact nor the protocol nor what to do. The
    fixture that reads it already skips when the artefact is MISSING and told
    the reader to regenerate it; it had no way to tell that one was STALE.

    Refusing here and not in the fixture is the point: a renderer, a notebook
    and a test all read through this function, and a guard that lived in one of
    them would leave the other two dying the old way.
    """
    faltan = sorted({f.name for f in _campos_de(Scale)} - set(stamp))
    if not faltan:
        return
    sobran = sorted(set(stamp) - {f.name for f in _campos_de(Scale)}
                    - {"isPilot", "realizedSplit"})
    raise ArtefactoDeOtroProtocolo(
        f"the artefact in {origen} carries a scale stamp this module does not "
        f"declare: it is missing {faltan}"
        + (f" and carries {sobran}, which this Scale retired" if sobran else "")
        + ". Its numbers were measured under another protocol, so reading them "
        "as this one's would compare two experiments. Regenerate it with "
        "MIL-CREDA/Notebooks/Toyset_Generator.ipynb at this scale")


def cargar(origen) -> dict:
    """Read back what `guardar` wrote, from the real files.

    The pair round-trips or it does not: the keys this returns are the keys
    `guardar` wrote, and `tests/test_toy_run.py` crosses that join by running
    `guardar` to disk and asking `cargar` what it sees -- never by writing a
    fixture and reading it, which verifies both halves and never the connection.

    It REFUSES an artefact whose scale stamp is from another protocol --- see
    `_refuse_a_foreign_stamp`, which is where the reason lives.
    """
    origen = Path(origen)
    flags_ = json.loads((origen / "banderas.json").read_text(encoding="utf-8"))
    barrido = json.loads((origen / "barrido.json").read_text(encoding="utf-8"))
    # BEFORE anything is decoded: a foreign stamp means every number below it
    # describes another experiment, so there is nothing worth reading.
    _refuse_a_foreign_stamp(origen, barrido["escala"])
    families: dict[str, dict] = {}
    for name in flags_["families"]:
        ruta = origen / f"{name}.npz"
        if not ruta.exists():
            continue
        with np.load(ruta, allow_pickle=False) as z:
            info, histories, sigmas, latent = {}, {}, {}, {}
            for member in z.files:
                *rest, kind = member.split(SEP)
                handle = SEP.join(rest)
                if kind == "info":
                    info[handle] = json.loads(str(z[member]))
                elif kind == "hist_s":
                    histories[handle] = [tuple(r) for r in z[member].tolist()]
                elif kind == "sigma":
                    sigmas[handle] = z[member].tolist()
                elif kind in LATENT_MEMBERS:
                    # Kept as arrays, unlike everything above. The one consumer
                    # projects them with an SVD, so a round trip through lists
                    # would be a conversion for nobody -- and `np.load` without
                    # a memory map has already read the bytes by the time the
                    # file closes, so holding the array is not holding the file.
                    latent.setdefault(handle, {})[kind] = z[member]
        families[name] = {"info": info, "hist_s": histories, "sigma": sigmas,
                          "latent": latent}
    return {
        "flags": flags_,
        "cells": barrido["celdas"],
        "rows": barrido["filas"],
        "hp1": barrido["hp1"], "hp2": barrido["hp2"],
        "records1": barrido["busqueda1"], "records2": barrido["busqueda2"],
        "trials1": barrido["trials1"], "trials2": barrido["trials2"],
        "cost": barrido["costo"], "scale": barrido["escala"],
        "families": families,
    }


def destination(scale: Scale, repository: Path | None = None) -> Path:
    """Where a run of this scale writes, with the scale IN the path.

    A pilot and a full run sharing a destination is how the cheap one overwrites
    the expensive one -- `config.results_for`'s own `pilot` coordinate exists for
    that reason, and this mirrors it rather than reusing it: this is not a
    campaign over `config.DOMAINS` and carries none of a campaign's coordinates.
    """
    root = Path(__file__).resolve().parents[2] if repository is None else Path(repository)
    return root / "MIL-CREDA" / "Results" / "Toyset" / scale.name


# ------------------------- the latent space: the basis, the box, the readings

def _base_pca(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The 2-D basis this panel projects on, fitted on the SOURCE positions.

    **Fitted on the source alone, and the same basis then projects all three
    views.** The figure's question is where phase 2 moved the target relative
    to a frozen reference, and a basis refitted per view would move the frame
    along with the cloud -- the displacement would then be partly the
    projection's own, and nothing in the picture would separate the two. The
    source is the reference because it is the half that does not move, which
    `latentSourceMoved` measures per cell rather than assuming.

    PCA and not the UMAP `latent._embed` uses for the digit grid, for a reason
    that is about this figure rather than a preference: a before/after pair has
    to live in ONE frame, a linear basis is the only projection that can be
    fitted once and applied to a cloud it was not fitted on, and a non-linear
    embedding refitted over the union would place the two moments by their
    mutual neighbourhoods -- which is the very thing being measured. It is also
    CREDA's own choice for exactly this figure (`CREDA.toy_plots`' projection is
    an SVD of the source, "clavada a la fuente").
    """
    mu = positions.mean(0)
    _, singular, Vt = np.linalg.svd(positions - mu, full_matrices=False)
    total = float((singular ** 2).sum())
    explained = float((singular[:2] ** 2).sum() / total) if total > 0 else float("nan")
    return mu, Vt[:2].T, explained


def _limites(puntos: np.ndarray) -> tuple[list[tuple[float, float]], float, bool]:
    """A CENTRED box, square while the cloud is nearly isotropic.

    `CREDA.toy_plots.limites_fila`'s rule, carried over with its reasons rather
    than re-derived: centred because both clouds move and a min/max box pins its
    extreme on one of them, square because an alignment is judged by DISTANCES
    and two axes at different scales show distances that are not there. Past
    `ASPECTO_MAX` -- CREDA's own 3.0, imported rather than copied -- the box
    goes per axis and the panel's title SAYS so, because beyond that ratio a
    square box leaves most of the panel empty and the cloud reads as a line.
    """
    centro = (puntos.max(0) + puntos.min(0)) / 2
    rango = puntos.max(0) - puntos.min(0)
    razon = float(max(rango) / max(float(min(rango)), 1e-9))
    if razon <= ASPECTO_MAX:
        medio = float(np.max(rango)) / 2 * 1.06
        return ([(float(centro[d] - medio), float(centro[d] + medio))
                 for d in range(2)], razon, True)
    return ([(float(centro[d] - rango[d] / 2 * 1.06),
              float(centro[d] + rango[d] / 2 * 1.06))
             for d in range(2)], razon, False)


#: **The approach reading was RETIRED with the frame it printed in, and this
#: note stands where its defence stood.** `_lectura_de_cuadro` and
#: `trayectoria_de_acercamiento` measured the distance between the two centroids
#: and the target cloud's own dispersion, per frame, and the only thing that
#: ever read them was the first embedding column's second title line
#: («centroides 0.493 · dispersión del blanco 0.429 (en 16-D)»).
#:
#: **Settled by the operator on 2026-10-15: this figure carries CREDA's frame --
#: «mismos límites, mismas cosas» -- and CREDA's two embedding columns carry ONE
#: reading, repeated.** CREDA prints no approach and no dispersion anywhere in
#: `cuadro_config`, so the reading has no counterpart in the frame this figure
#: is now held to, and it goes rather than moving to a quieter place: a
#: measurement kept alive by its own tests after its last reader is gone is the
#: same defect as one nobody measured, and it reads as a live reading to whoever
#: finds it next. The two tests that drove it were deleted with it rather than
#: repointed. Do not re-raise it as an open gap, and do not reintroduce either
#: function to put the number back in a title.
#:
#: What a reader loses is the collapse complement: an alignment that falls
#: because the encoder sent every target bag to one point no longer says so in
#: the picture. The positions are still drawn, so a collapse is still visible as
#: a cloud contracting across the frames, and nothing that was MEASURED moved --
#: this was only ever printed, never recorded in the artefact.

def _angulo_entre_bases(primera: np.ndarray, segunda: np.ndarray) -> float:
    """The largest principal angle between two 2-D bases, in degrees.

    This is what makes "the limits are per architecture" a MEASUREMENT instead
    of a habit. Two panels could be drawn in one box only if they were drawn in
    one basis, and these are not: each architecture trains its own encoder, so
    its own principal directions. The angle says how far apart the two planes
    actually are, so the refusal to share is reported with its evidence rather
    than asserted in a comment.
    """
    cosenos = np.linalg.svd(primera.T @ segunda, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cosenos.min(), -1.0, 1.0))))


#: **The scale stamp was RETIRED from this figure, and this note stands where
#: its defence stood.** `_sello_plano` stripped the markdown off
#: `toy_tables.escala_sello` and the suptitle carried it as a SECOND line:
#: «escala «pilot» · ENSAYO · revisión research-concept-r29.md · dispositivo
#: mps — ningún número de esta lectura se cita como resultado».
#:
#: The operator retired that stamp from the tables on a previous pass, for a
#: reason that was never about the tables: CREDA's plot notebook emits no footer
#: under any of its ten blocks, and same structure as CREDA means same
#: structure. It survived here only because the figure sat outside that
#: stretch's surface. **Settled by the operator on 2026-10-15: it goes from the
#: figure too, and `toy_tables.escala_sello` goes with it** -- this was its last
#: reader, so the function is deleted rather than left standing for nobody.
#: CREDA's suptitle is ONE line; a second one is a line CREDA's frame does not
#: have. Do not re-raise it as an open gap, and do not reintroduce the stamp as
#: a figure text, a footer clause or a cell of prose.
#:
#: What a reader of the GIF alone loses is the sentence saying these numbers are
#: a trial. It is still stated where it was already stated: by the record
#: itself, and by the preamble cell of `Toyset_Plot.ipynb`, which prints the
#: scale and the revision before any figure is drawn.


def _brazo_unico(registro: dict) -> str:
    """The one arm this figure can draw, refusing rather than choosing.

    A panel grid of one row per architecture has no axis left for the arm, so
    with two declared arms there is no honest place to put the second: drawing
    the first would publish one arm's latent space under a title that names the
    run. This toyset declares one (`TOY_ARMS`), so the refusal is unreachable
    today and is here for the day a second one is declared.
    """
    brazos = list(registro["flags"]["arms"])
    if len(brazos) != 1:
        raise ValueError(
            f"el registro declara {len(brazos)} brazos ({brazos}) y esta figura "
            "tiene un panel por arquitectura y ninguna dimensión libre para el "
            "brazo: dibujar uno de los dos publicaría el espacio latente de un "
            "brazo bajo un título que nombra la corrida")
    return brazos[0]


def semilla_a_dibujar(registro: dict, familia: str, normalizacion: str) -> int:
    """The report seed this panel draws: the MEDIAN by target accuracy.

    The median and never the best, which is the rule this repository already
    applies to a kept artefact: the best of N repetitions is an extreme of N
    draws, so the panel somebody inspects would describe the luckiest run. With
    an even number of seeds the LOWER median is taken, declared here so the tie
    is a rule and not whichever order the record happened to hold.

    **Per architecture and not one seed for the whole figure.** Each panel is
    its own encoder with its own basis and its own box -- the figure never
    claims the two are commensurable, which is exactly why the limits are per
    panel -- and a seed chosen across architectures would have to collapse the
    architecture axis to pick. So each panel says which seed it draws, in its
    own title.
    """
    brazo = _brazo_unico(registro)
    semillas = [int(s) for s in registro["scale"]["report_seeds"]]
    lecturas = []
    for semilla in semillas:
        clave = key(familia, normalizacion, brazo, semilla)
        if clave not in registro["cells"]:
            raise KeyError(
                f"el registro no trae la celda {clave!r}: la figura de "
                f"{familia} elegiría su semilla sobre una grilla incompleta")
        lecturas.append((float(registro["cells"][clave]["targetAccuracy"]),
                         semilla))
    lecturas.sort()
    return lecturas[(len(lecturas) - 1) // 2][1]


# --------------------------------------------- the animated grid, CREDA's shape

#: The step of a frame and the pause on the LAST one, in MILLISECONDS. CREDA's
#: own two numbers and its own units: in `imageio` 2.37 `duration` is
#: milliseconds, and the seconds-shaped `0.30` that reads right rounded to 0 and
#: wrote GIFs with no time assigned at all.
MS_FRAME, MS_PAUSE = 140, 2500

#: The two embedding columns' labels, written ONCE. CREDA keeps the identical
#: constant (`COL_EMB_REAL`/`COL_EMB_PRED`) and records why: three copies of the
#: same literal is how the third one stays behind without anything failing, and
#: then two figures of one section label one column two ways.
COL_REAL = "embedding " + chr(183) + " color = clase REAL"
COL_PRED = "embedding " + chr(183) + " color = PREDICCIÓN"
#: The two loss columns carry CREDA's own two titles, word for word
#: (`cuadro_config` hands `panel_fase1`/`panel_fase2` exactly these). The second
#: one used to read «hasta este cuadro», which was this panel describing a
#: ceiling of its own; it now says what CREDA's says because the panel now stops
#: where CREDA's stops.
#:
#: And the second one said «hasta el PASO que eligió» until 2026-10-19, when
#: the criterion moved to epochs on both sides: what it chooses is an epoch, so
#: the title names an epoch.
#:
#: **That day this note argued the x axis could stay «paso de la fase 2» --
#: «not a contradiction, and CREDA's shape too, the criterion picks a pass and
#: the curve is drawn against the updates inside it». Retired by the operator
#: on 2026-10-19: everything is in epochs, which was already decided.** The
#: argument was wrong on its own terms, and the measurement says where: in
#: steps the three families' panels run 0-3400, 0-3400 and 0-900, because
#: `LOTE_2` is searched per family, so the same abscissa means three different
#: amounts of training and the three figures cannot be read side by side. In
#: epochs all three run 0-100. A title in one unit over an axis in another was
#: the smaller half of the defect.
COL_PHASE_ONE = "fase 1 " + chr(183) + " hasta la época que eligió"
COL_PHASE_TWO = "fase 2 " + chr(183) + " hasta la época que eligió"

#: The four columns of a frame, in the order CREDA declares them. A tuple and
#: not a literal `4` anywhere below: a column added or dropped moves the grid's
#: width and the label row together, which is the one pair that must not drift.
GRID_COLUMNS = (COL_REAL, COL_PRED, COL_PHASE_ONE, COL_PHASE_TWO)

#: What the trajectory costs on disk, MEASURED on this pilot and projected from
#: the same per-frame arithmetic rather than guessed.
#:
#: A frame stores one target position per bag and one decision per bag:
#: `bags * dims * 4` bytes for the positions and `bags * 2` for the decisions.
#: At the declared 81 bags that is `81 * dims * 4 + 162`, so 5,346 bytes per
#: frame at 16 output dimensions. `bags_per_class` is a programme field so the
#: bag count is the same at both scales; the output dimension is SEARCHED
#: (`PHASE_ONE_DIMENSIONS`' `dim`, over 8/16/32/64), so the per-frame cost runs
#: 2,754 to 20,898 bytes depending on what the search chose, and only the frame
#: COUNT moves with the scale.
#:
#: **Measured on the pilot of 2026-10-19, at the declared 81 bags with the
#: search choosing 64 output dimensions**: the three `.npz` files hold 1,470,858
#: bytes in total over NINE cell handles, at 5 frames each -- the epoch grid's
#: own count at this ceiling -- and 1,523,790 bytes of positions in memory
#: before compression. Both prior readings, kept as prior readings: on
#: 2026-10-15, under the retired 152-step ceiling and its stride of 7, the same
#: files held 2,219,547 bytes over the same nine handles at 7 to 13 frames
#: each; and before that, at twelve bags per class and a fixed 16 dimensions,
#: they went from 68,000 bytes to 166,771 over six handles at 7 frames each.
#:
#: The 13x growth is three things at once and none of them is a surprise: 81
#: bags against 36, 64 dimensions against 16, and nine handles against six.
#:
#: **Nothing reads this constant.** It is documentation: `flags()` carries
#: `WEIGHTING_COST` into `banderas.json` and carries no counterpart for this
#: one, so a figure here that went stale would go stale in silence. Said out
#: loud rather than left to be discovered.
#:
#: Projected to the full scale with the identical arithmetic: `frame_stride`
#: gives 80 steps over the 1,600 the ceiling holds -- 5 epochs, which is the
#: unit it is now declared in -- so 21 frames a handle, and the full scale
#: declares 10 report seeds over 3 normalizations and 3 families, so 90 handles
#: -- 10,103,940 bytes of trajectory uncompressed at 16 output dimensions,
#: before `savez_compressed` touches it.
#:
#: **All three inputs of that line were stale and are re-derived rather than
#: patched.** It read `frame_stride` as 10 over 200 steps, which is the retired
#: step ceiling; it counted 2 normalizations where `NORMALIZATIONS` declares 3,
#: so 60 handles where the measured pilot already showed NINE at one report
#: seed (3 families x 3 normalizations); and its byte figure was on the prior
#: 36-bag basis this same comment says was superseded by 81. The prior numbers
#: are kept above as the prior reading, which is the only place they belong.
#:
#: **That is the number, and it is why the stride is declared rather than left
#: at every step.** Without a stride the full scale would store 1,601 frames a
#: handle -- 770,305,140 bytes, seventy-six times as much -- and would render
#: 1,601 two-row, four-column frames per family, which is the cost that matters
#: more than the bytes: one frame of this grid takes about half a second to
#: draw.
#:
#: **At the pilot the epoch grid is what makes it a pilot, MEASURED**: a stride
#: of one epoch is 16 steps over a 64-step ceiling, so 5 frames a handle
#: against the 23 the step grid gave -- 4.6x fewer `bag_positions` calls, 4.6x
#: fewer bytes (26,730 against 122,958 at 16 dimensions) and 45 panels to draw
#: against 207 over its nine handles. The full scale is untouched by that
#: change: 5 epochs is 80 steps, so the same 21 frames land on the same steps.
#:
#: The generator notebook prints these for the run in front of the reader rather
#: than quoting this constant, which is what keeps the two from drifting apart.
TRAJECTORY_COST = {
    "bytesPerFrame": "bags * dims * 4 + bags * 2 = 5,346 at 81 bags and 16 dims",
    "measuredPilotArtefactBytes": 1470858,
    "measuredPilotLatentBytesInMemory": 1523790,
    "measuredPilotArtefactBytesUnderTheRetiredStepCeiling": 2219547,
    "measuredPilotArtefactBytesAtTheTwelveBagCut": 166771,
    "measuredPilotArtefactBytesBeforeTheTrajectory": 68000,
    "pilotFrames": [5],
    "pilotFramesUnderTheRetiredStepCeiling": [7, 9, 10, 12, 13],
    "pilotHandles": 9,
    "pilotOutputDim": 64,
    "projectedFullFrames": 21,
    "projectedFullHandles": 90,
    "projectedFullTrajectoryBytes": 10103940,
    "withoutStrideFullFrames": 1601,
    "withoutStrideFullTrajectoryBytes": 770305140,
    # What the epoch grid buys at the PILOT, which is the whole reason it is
    # declared in epochs: the full scale is untouched (5 epochs is 80 steps,
    # the same 21 frames on the same steps) and the pilot runs 4.6x lighter.
    "pilotFramesUnderTheStepGrid": 23,
    "pilotFramesUnderTheEpochGrid": 5,
    "pilotTrajectoryBytesUnderTheStepGrid": 122958,
    "pilotTrajectoryBytesUnderTheEpochGrid": 26730,
    "theStaleProjectedFullHandles": 60,
    "theStaleProjectedFullHandlesWhy":
        "it counted 2 normalizations where NORMALIZATIONS declares 3, and the "
        "measured pilot already showed 9 handles at one report seed",
}


@dataclass(frozen=True)
class AnimacionLatente:
    """What `animar_familia` hands back: the file, the facts, the absences.

    No live figure comes back. A GIF is not a `plt.Figure` -- it is
    many of them encoded and closed -- so there is nothing to hand back for an
    inline `display`, and the caller shows the file. The frame count and the
    stride come back because they are what a reader has to be told: an animation
    of two frames and one of twenty read as the same object until somebody says
    how many there were.
    """

    ruta: Path
    cuadros: int
    paso: int
    #: One entry per declared normalization, in the record's own order.
    filas: list[dict]
    #: The normalizations CREDA draws that this record has no backbone for.
    ausentes: tuple[str, ...]
    ejes: dict


def cuadros_comunes(largos) -> int:
    """How far an animation whose rows have different lengths can run.

    The MINIMUM, never the first. Early stopping ends each row at its own step,
    so taking the first row's count and indexing the others with it raises
    `IndexError` inside the shortest -- CREDA records that happening three times
    in its own notebook, with `nbconvert` exiting 0 anyway.

    Running to the LONGEST is the other exit and is worse: it would freeze the
    short rows, and then one frame would show rows standing at different steps
    under a title naming one step.

    Never padded, either: a length of 0 is a cell that left no frame at all,
    which is a defect of the artefact and not an animation of zero frames.
    """
    largos = [int(v) for v in largos]
    if not largos:
        raise ValueError("no hay ninguna serie que animar")
    if any(v <= 0 for v in largos):
        raise ValueError(
            f"alguna fila no dejó ningún cuadro ({largos}): un GIF de cero "
            "cuadros no es una animación, y una fila vacía se lee igual que una "
            "medición que salió plana")
    return min(largos)


def _fila_animada(registro: dict, familia: str, normalizacion: str) -> dict:
    """One row of the grid: its own basis, its own box, its whole trajectory.

    The basis is fitted on the SOURCE and applied to every frame, so the
    displacement a reader sees is the target's and not the projection's. A
    basis refitted per frame would move the frame with the cloud, which is the
    one thing an animation must not do.
    """
    brazo = _brazo_unico(registro)
    semilla = semilla_a_dibujar(registro, familia, normalizacion)
    handle = key(normalizacion, brazo, semilla)
    latentes = registro["families"][familia].get("latent") or {}
    if handle not in latentes:
        raise KeyError(
            f"{familia}.npz no trae las posiciones de {handle!r}: este "
            "artefacto lo escribió un generador anterior a `LATENT_MEMBERS`. "
            "Corré `MIL-CREDA/Notebooks/Toyset_Generator.ipynb` de nuevo; un "
            "cuadro vacío se lee igual que una medición que salió plana")
    miembros = latentes[handle]
    faltan = [m for m in LATENT_MEMBERS if m not in miembros]
    if faltan:
        raise KeyError(
            f"{handle} no trae {faltan}: este artefacto lo escribió un "
            "generador anterior a la trayectoria, así que no tiene cuadros que "
            "animar. Corré el generador de nuevo")

    celda = registro["cells"][key(familia, normalizacion, brazo, semilla)]
    mu, base, explicada = _base_pca(np.asarray(miembros["source"], dtype=float))
    proyectar = lambda Z: (np.asarray(Z, dtype=float) - mu) @ base
    fuente = proyectar(miembros["source"])
    trayectoria = np.stack([proyectar(cuadro)
                            for cuadro in np.asarray(miembros["trajectoryTarget"])])
    pasos = [int(p) for p in np.asarray(miembros["trajectorySteps"]).tolist()]

    # The box contains the SOURCE and every frame of the trajectory, so it is
    # fixed for the whole animation. A box recomputed per frame would move the
    # axes under a cloud that is what the figure is about, and the reader could
    # not tell the two apart.
    limites, razon, cuadrada = _limites(
        np.vstack([fuente, trayectoria.reshape(-1, trayectoria.shape[-1])]))

    hist_s = np.asarray(registro["families"][familia]["hist_s"][
        key(normalizacion, semilla)], dtype=float)
    return {
        "normalizacion": normalizacion,
        "semilla": int(semilla),
        "fuente": fuente,
        "trayectoria": trayectoria,
        "pasos": pasos,
        "claseFuente": np.asarray(miembros["classSource"]),
        "claseBlanco": np.asarray(miembros["classTarget"]),
        "prediccionFuente": np.asarray(miembros["predictionSource"]),
        "prediccionBlanco": np.asarray(miembros["trajectoryPrediction"]),
        "base": base,
        "varianzaExplicada": explicada,
        "limites": limites,
        "razonDeAspecto": razon,
        "cajaCuadrada": cuadrada,
        "histFase1": hist_s,
        "epocaElegida": int(celda["phase1BestEpoch"]),
        "histFase2": np.asarray(miembros["phaseTwoHistory"], dtype=float),
        "pasoElegido": int(celda["phase2BestStep"]),
        "pasosCorridos": int(celda["phase2Steps"]),
        # **The three the panel DRAWS with, and they are epochs.** The step
        # fields above stay because they index the history array -- a row of
        # `histFase2` is one step and nothing else -- but no step reaches the
        # reader: the axis, its limit, the chosen marker and the row label are
        # all epochs, which is what this package reports in.
        #
        # The epoch is READ, never divided here. `phase2BestEpoch` is the epoch
        # the criterion chose and `phase2Epochs` the ceiling it ran, both
        # written by the run that knows which pass it had. Dividing
        # `phase2BestStep` at the draw site would recompute, from a constant
        # declared somewhere else, a number the record already carries -- and
        # would keep agreeing with it right up until a run whose pass differs,
        # which is the day it would matter.
        "epocaElegidaFase2": int(celda["phase2BestEpoch"]),
        "epocasCorridas": int(celda["phase2Epochs"]),
        "pasosPorEpoca": int(celda["phase2StepsPerEpoch"]),
        "zancada": int(celda["frameStride"]),
        "fuenteQuieta": float(celda["latentSourceMoved"]),
        "aciertoBlanco": float(celda["targetAccuracy"]),
        "aciertoPredict": float(celda["predictTargetAccuracy"]),
        "aciertoAdabn": celda["adabnTargetAccuracy"],
        "bolsasPorDominio": int(fuente.shape[0]),
    }


def posiciones_del_cuadro(fila: dict, k: int) -> tuple:
    """The two clouds of frame `k`: the source and the target, projected.

    **Columns one and two of the grid call THIS, once each, and that is the
    whole reason it is a function.** The two columns are the same embedding
    drawn twice and differ only in which array colours it; if either of them
    computed its own positions, the two could drift apart and a reader would no
    longer be able to read the difference between the panels as «what the head
    got wrong».
    """
    return fila["fuente"], fila["trayectoria"][k]


def colores_del_cuadro(fila: dict, k: int, por_prediccion: bool) -> tuple:
    """The two colour channels of frame `k`: the true class, or the decision.

    The ONLY thing that differs between columns one and two. The source's
    decision has no frame index because it cannot have one: neither the head nor
    `theta_s` is in the optimizer, so the source's prediction is a single row
    that every frame reuses.
    """
    if por_prediccion:
        return fila["prediccionFuente"], fila["prediccionBlanco"][k]
    return fila["claseFuente"], fila["claseBlanco"]


def lectura_del_cuadro(fila: dict, k: int) -> str:
    """The ONE reading columns one and two both carry, under frame `k`.

    **One reading, repeated, which is CREDA's own shape and now this figure's.**
    `CREDA.toy_plots.panel_embedding` writes the identical second title line
    under both embedding columns -- «blanco 0.431 · fuente 0.989 · ruteo 45%» --
    so the only thing that differs between the two panels is the colouring, and
    a reader compares them without also reconciling two different sentences.
    This figure used to print a different line under each: an approach and a
    dispersion under the first, a count of misclassified bags under the second.
    **Settled by the operator on 2026-10-15: CREDA's frame, «mismos límites,
    mismas cosas».** The function exists so the two columns cannot drift apart,
    for the same reason `posiciones_del_cuadro` does.

    **The quantities are this experiment's own, and the unit is why.** CREDA's
    two numbers are accuracies over POINTS; here the statistical unit is the
    BAG, for every arm, so a bag-level accuracy is the honest equivalent of its
    point-level one and is said here rather than substituted silently. They are
    measured over the bags this frame DRAWS: the record carries one target
    decision per bag per frame (`trajectoryPrediction`) over the whole target
    domain, and no per-frame reading over the evaluation role at all, so the
    drawn population is the only one a per-frame number can honestly name.

    **CREDA's third quantity has no counterpart and is not invented one.** Its
    `ruteo` is `(rut == yt).mean()` -- the mixture head's routing of each target
    point, which its term reads to decide which class block a point falls in.
    What routes here is the bag's decided class, which is exactly the array
    `blanco` is already computed from, so a `ruteo` column would print the same
    number twice under two names. Two names for one measurement is worse than
    one name, so the line carries two readings.
    """
    blanco = float((np.asarray(fila["prediccionBlanco"][k])
                    == np.asarray(fila["claseBlanco"])).mean())
    fuente = float((np.asarray(fila["prediccionFuente"])
                    == np.asarray(fila["claseFuente"])).mean())
    return f"blanco {blanco:.3f} {chr(183)} fuente {fuente:.3f}"


def _panel_embebido(a, fila: dict, k: int, por_prediccion: bool) -> None:
    """One embedding panel of frame `k`, coloured by truth or by the decision."""
    fuente, blanco = posiciones_del_cuadro(fila, k)
    col_fuente, col_blanco = colores_del_cuadro(fila, k, por_prediccion)
    clases = sorted(set(fila["claseFuente"].tolist())
                    | set(fila["claseBlanco"].tolist()))
    for clase in clases:
        col = COL_DURA[int(clase)]
        ms = np.asarray(col_fuente) == clase
        mt = np.asarray(col_blanco) == clase
        vs = np.asarray(fila["claseFuente"]) == clase
        vt = np.asarray(fila["claseBlanco"]) == clase
        if ms.any():
            a.scatter(fuente[ms, 0], fuente[ms, 1], marker="x", c=[col], s=44,
                      linewidths=1.3, alpha=0.9, zorder=3)
        if mt.any():
            a.scatter(blanco[mt, 0], blanco[mt, 1], marker="o", c=[col], s=52,
                      edgecolors="0.15", linewidths=0.5, alpha=0.95, zorder=4)
        # The centroids are ALWAYS of the true class, in both columns, which is
        # what makes the second column readable: a point of one colour sitting
        # on another colour's diamond is a bag the head got wrong, and if the
        # diamonds moved with the colouring there would be nothing to see.
        a.scatter(*fuente[vs].mean(0), marker="D", s=150, facecolors="none",
                  edgecolors=col, linewidths=2.0, zorder=5)
        a.scatter(*blanco[vt].mean(0), marker="D", s=105, c=[col],
                  edgecolors="k", linewidths=1.0, zorder=6)
    a.set_xlim(*fila["limites"][0])
    a.set_ylim(*fila["limites"][1])
    # **Square limits are not a square BOX, and this line is the difference.**
    # `_limites` returns equal spans while the cloud is nearly isotropic, and
    # for a long time that was all: the panel was then drawn into whatever shape
    # the subplot happened to have, so the distances a reader measured by eye
    # were still anisotropic -- and the row label was advertising «caja cuadrada
    # (1.6:1)» for a box nothing had ever squared. `CREDA.toy_plots` pins it
    # (`panel_embedding`, `if cuadrada: a.set_aspect('equal')`) under exactly the
    # condition `_limites` already computes from CREDA's own imported
    # `ASPECTO_MAX`, and an alignment judged by DISTANCES is the whole reason.
    #
    # Conditional, for CREDA's reason and not out of caution: past that ratio
    # `aspect='equal'` would draw the panel as the strip the data is -- measured
    # on `rollo`, 4.7:1 -- and waste more of it than the square box did. There
    # the box goes per axis and the ROW LABEL says so, which is where CREDA says
    # it too.
    if fila["cajaCuadrada"]:
        a.set_aspect("equal")
    a.grid(alpha=0.25)
    a.tick_params(labelsize=8)
    # The column's own name, and under it the reading BOTH columns carry. The
    # second line is computed once, by `lectura_del_cuadro`, so the two panels
    # cannot say two different things about one frame.
    columna = COL_PRED if por_prediccion else COL_REAL
    a.set_title(f"{columna}\n{lectura_del_cuadro(fila, k)}", fontsize=9.5,
                linespacing=1.4)


def _panel_fase1(a, fila: dict) -> None:
    """Phase 1 up to the epoch it CHOSE, both cross-entropies and the accuracy.

    Up to `phase1BestEpoch` and not to the ceiling: phase 1 restores that
    epoch's weights and it is that encoder phase 2 transfers. The patience
    epochs after it exist to corroborate that it was the best, not to be shown.
    """
    hist = fila["histFase1"]
    if hist.size == 0:
        raise ValueError("la fase 1 siempre deja historia")
    elegida = fila["epocaElegida"]
    filas = hist[hist[:, 0] <= elegida]
    if filas.size == 0 or int(filas[-1, 0]) != elegida:
        raise ValueError(
            f"la época elegida ({elegida}) no es la última de la historia "
            f"dibujada ({None if filas.size == 0 else int(filas[-1, 0])}): el "
            "panel estaría diciendo que paró donde no paró")
    e, ce, cv, av = filas[:, 0], filas[:, 1], filas[:, 2], filas[:, 3]
    # With a marker, and not as a style choice: at the pilot's ceiling of six
    # epochs the stop can choose epoch 0, and then this panel has ONE point --
    # which a bare `plot` draws as nothing at all. An empty axis reads as a
    # measurement that came out flat, and what it would be hiding is that phase
    # 1 stopped immediately.
    a.plot(e, ce, color="#1f77b4", lw=1.4, alpha=0.7, marker="o", ms=3.5,
           label="CE de la FUENTE " + chr(183) + " train")
    a.plot(e, cv, color="#333333", lw=2.0, marker="o", ms=3.5,
           label="CE de la FUENTE " + chr(183) + " valid")
    a.set_ylim(0.0, 1.0)
    bajo, alto = float(min(ce.min(), cv.min())), float(max(ce.max(), cv.max()))
    # CREDA's own second line and nothing beside it: what of the fixed axis this
    # panel's curve actually used. The count of epochs drawn against epochs run
    # travelled here too and was retired on 2026-10-15 with the rest of this
    # figure's additions -- it has no counterpart in CREDA's title, and the same
    # reading is the row's `phase1Epochs`/`phase1BestEpoch` in the record.
    a.set_title(f"{COL_PHASE_ONE}\n(CE realizada: [{bajo:.4f}, {alto:.4f}] = "
                f"{100 * (alto - bajo):.1f}% del eje)", fontsize=9.5,
                linespacing=1.4)
    a.set_xlim(0, max(float(e.max()), 1.0))
    a.grid(alpha=0.25)
    a.tick_params(labelsize=8)
    a.set_xlabel("época de la fase 1", fontsize=9)
    a.legend(loc="upper right", fontsize=8, framealpha=0.85)
    b = a.twinx()
    b.plot(e, av, color="#8c564b", lw=1.2, alpha=0.75, ls="--")
    b.set_ylim(0, 1.02)
    b.tick_params(labelsize=7, colors="#8c564b")
    b.set_ylabel("acierto valid de la fuente", fontsize=8, color="#8c564b")
    b.plot([e[-1]], [av[-1]], "o", ms=6, color="#8c564b")


#: The headroom over the highest value a shared axis has to hold. CREDA's own
#: number, taken from the one axis in `panel_fase2` whose quantity is bounded
#: and whose limit is therefore not the bound itself: `b.set_ylim(0, 1.02)`.
HOLGURA_EJE = 1.02

#: The three series `_panel_fase2` draws on its LEFT axis, named once. The
#: watchman is not among them: it lives on the twin, bounded, with its own limit.
SERIES_FASE_DOS = ("adaptation", "contributionGlobal", "contributionLocal")


def limite_fase2(filas: list) -> float:
    """The ONE top limit the phase-2 panels of every row share.

    **Shared, which is the property CREDA buys, and derived, which is the only
    way to buy it here.** `CREDA.panel_fase2` pins `set_ylim(0.0, 1.0)` and says
    why: *«el eje compartido permite comparar los paneles de un vistazo»*. Its
    literal works there because its term is genuinely mapped to [0,1] by Eqs.
    35-36. **Here it is not**: measured over this record the adaptation term
    reaches **1.2457** (the global part 0.9720, the local 0.5510), so CREDA's
    literal would CLIP the very curve the panel exists to show, and a clipped
    curve is not a measurement -- it is a flat top that reads like a plateau.

    So the number is derived from the record instead: the maximum over every
    ROW, every step of each row's whole history and every series the left axis
    draws, with `HOLGURA_EJE` of headroom -- CREDA's own 1.02, read off the one
    bounded axis of that same panel. Taken over the full history and not over
    the frames drawn so far, because an axis that grew with the animation would
    move under the curves and nothing in the picture would separate «the term
    rose» from «the axis shrank».

    A per-row autoscale is what this replaces, and it was not a shared axis at
    all: three rows at three scales, side by side, read as one comparison.
    """
    if not filas:
        raise ValueError("no hay ninguna fila cuyo eje compartir")
    nombres = {nombre: i for i, nombre in enumerate(TRAJECTORY_HISTORY_FIELDS)}
    alto = 0.0
    for fila in filas:
        hist = fila["histFase2"]
        if hist.size == 0:
            continue
        for campo in SERIES_FASE_DOS:
            alto = max(alto, float(hist[:, nombres[campo]].max()))
    if alto <= 0.0:
        raise ValueError(
            f"ninguna serie de la fase 2 llega a un valor positivo ({alto}): un "
            "eje de altura cero no es un eje compartido, es un panel vacío")
    return alto * HOLGURA_EJE


def _panel_fase2(a, fila: dict, k: int, tope_y: float) -> None:
    """Phase 2 up to this frame's step and never past the one it CHOSE.

    On a y-axis handed in rather than computed here, because it is shared with
    every other row of the figure -- `limite_fase2` says where the number comes
    from and why it is not CREDA's literal.

    The supervised term is computed and NOT drawn, which is CREDA's own decision
    for its own reason: with the head and `theta_s` both outside the optimizer
    it has no gradient path to anything, so a curve in the loss panel that is
    not part of what moved would be read as if it were. What is drawn is the
    adaptation term and its two declared components, because summed they cannot
    be read apart -- «the term commanded nothing» and «the term was scaled to
    nothing» print the same small number.
    """
    hist = fila["histFase2"]
    nombres = {nombre: i for i, nombre in enumerate(TRAJECTORY_HISTORY_FIELDS)}
    # **Up to this frame's step and NEVER past the step the run chose**, which
    # is `CREDA.panel_fase2`'s own ceiling (`tope = min(k, mejor_paso)`) and its
    # own reason: there is where the early stop left `theta_t`, and that is the
    # encoder reported; the patience steps after it corroborated that it was the
    # best and are not drawn, *«igual que en la fase 1»*. This package already
    # draws phase 1 that way -- `_panel_fase1` cuts at `phase1BestEpoch` and
    # refuses when the chosen epoch is not the last one drawn -- so before this
    # the two loss panels of one figure obeyed two different rules.
    #
    # The ceiling comes WITH the title. CREDA's one-line «hasta el paso que
    # eligió» is true of a panel that stops there and false of one that runs on,
    # so adopting its wording without its ceiling would have hung a sentence
    # over a curve that contradicts it.
    tope = min(fila["pasos"][k], fila["pasoElegido"])
    filas = hist[hist[:, nombres["step"]] < tope] if hist.size else hist
    # **Frame zero draws the AXES, empty, and never a box of text instead.**
    # Standing before any update there is nothing to plot, and this panel used
    # to replace itself with a framed sentence saying so -- which left column
    # four of the first frame with no axes, no step axis and no legend, while
    # CREDA's `panel_fase2` draws its real axes at step 0 and lets them come out
    # empty. **Settled by the operator on 2026-10-15: CREDA's frame.** So there
    # is ONE path through this panel: the series are sliced to whatever the
    # frame stands at, empty included, and the axes, their limits, their label
    # and their legend are built the same way at every frame.
    serie = lambda campo: (filas[:, nombres[campo]] if filas.size
                           else np.empty(0, dtype=float))
    # **The abscissa is the EPOCH, and the step only gets it there.** A row of
    # `histFase2` is one optimizer step -- that is what the array is -- so the
    # epoch of a row is its step divided by the pass this cell ran, read from
    # the record and never assumed. `+ 1` first, as before: row zero is the
    # state AFTER the first step, not before it, so the curve starts inside
    # the first epoch rather than at a zero it never occupied.
    #
    # Dividing here and not at the source is deliberate: the slice above is by
    # STEP (`tope`), because that is the array's own index, and converting the
    # column before slicing would make the filter compare an epoch against a
    # step and silently keep the wrong rows.
    por_epoca = max(fila["pasosPorEpoca"], 1)
    s = ((filas[:, nombres["step"]] + 1) / por_epoca if filas.size
         else np.empty(0, dtype=float))
    a.plot(s, serie("adaptation"), color="#2ca02c", lw=2.0,
           label="adaptación (Ec. 39, coef. 1)")
    a.plot(s, serie("contributionGlobal"), color="#9467bd", lw=1.4,
           ls="--", label="parte global (Ecs. 32-37)")
    a.plot(s, serie("contributionLocal"), color="#d62728", lw=1.4,
           ls=":", label="parte local (Ecs. 30-31, 38)")
    a.set_xlim(0, max(fila["epocasCorridas"], 1))
    # The SAME limit in every row, handed in rather than computed here: a panel
    # that derived its own would derive it from its own row and the three would
    # stop being comparable, which is the whole point of the shared axis.
    a.set_ylim(0.0, tope_y)
    a.grid(alpha=0.25)
    a.tick_params(labelsize=8)
    a.set_xlabel("época de la fase 2", fontsize=9)
    a.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    # One line, as CREDA's is. The second line said which step this frame stands
    # at and which one the run chose; the step is in the suptitle, the chosen
    # step is in the row label, and the axvline below marks it in the panel.
    a.set_title(COL_PHASE_TWO, fontsize=10)
    b = a.twinx()
    b.plot(s, serie("watchman"), color="#8c564b", lw=1.1,
           alpha=0.75, ls="-.")
    b.set_ylim(0, 1.02)
    b.tick_params(labelsize=7, colors="#8c564b")
    b.set_ylabel("vigía: fuente conservada por theta_t (el que PARA)",
                 fontsize=7.5, color="#8c564b")
    # **The guard is in steps and the line is in epochs, on purpose.** `tope`
    # is this frame's own step ceiling, so asking whether the choice has
    # happened yet is a question about steps; where to draw it is a question
    # about the axis, which is epochs. Collapsing the two would compare an
    # epoch with a step and the marker would appear frames too early.
    elegido = fila["pasoElegido"]
    if 0 < elegido <= tope:
        b.axvline(fila["epocaElegidaFase2"], color="#8c564b", lw=0.9,
                  alpha=0.5)


def figura_del_cuadro(registro: dict, familia: str, filas: list, k: int):
    """Frame `k` as a LIVE figure: rows of normalizations, four columns.

    Split from the rasterizing half below for the reason CREDA splits its own
    (`figura_cuadro_contaminacion` against `cuadro_contaminacion`): a frame that
    only ever existed as an RGB array can be looked at and cannot be ASKED
    anything, so the one claim this grid rests on -- that columns one and two
    carry the same positions and differ only in colour -- would have to be
    checked by comparing pixels. Here a caller reads the artists.

    `fig.axes` holds the grid in row-major order first and the twin axes the
    loss panels create afterwards, so the first `rows * 4` are the grid.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(filas), len(GRID_COLUMNS), squeeze=False,
                           figsize=(6.0 * len(GRID_COLUMNS), 5.8 * len(filas)))
    # Derived ONCE for the figure and handed to every row: the point of the
    # shared axis is that the three panels are read against each other.
    tope_y = limite_fase2(filas)
    for r, fila in enumerate(filas):
        _panel_embebido(ax[r, 0], fila, k, False)
        _panel_embebido(ax[r, 1], fila, k, True)
        _panel_fase1(ax[r, 2], fila)
        _panel_fase2(ax[r, 3], fila, k, tope_y)
        # **The row label is CREDA's: the architecture's NAME, and the step this
        # row stopped at.** `CREDA.toy_plots.cuadro_config` writes exactly
        # `NOMBRE_FILA[q]` plus «paró en el paso N», with one conditional line
        # between them when the box stopped being square. This label used to
        # carry three lines -- the lowercase record key, the seed, the explained
        # variance and the box ratio on every row whether or not it mattered.
        # **Settled by the operator on 2026-10-15: CREDA's frame, «mismos
        # límites, mismas cosas».** The seed, the variance and the always-on box
        # ratio have no counterpart in that frame and are gone from the picture;
        # both are still in the row (`semilla`, `varianzaExplicada`) and in the
        # record, for anything that reads rather than looks.
        #
        # The name comes from `toy_tables.NOMBRE_FILA`, which is where this
        # package already maps «ninguna» to «sin normalización» for every table.
        # A second mapping written here is how the two start labelling one
        # architecture two ways.
        etiqueta = toy_tables.NOMBRE_FILA[fila["normalizacion"]]
        # CREDA's own conditional line, kept conditional: past `ASPECTO_MAX` the
        # box goes per axis and the distances a reader sees are no longer the
        # ones there, which has to be said WHERE it happens. Square is CREDA's
        # silent case and is silent here too.
        if not fila["cajaCuadrada"]:
            etiqueta += f"\n(ejes {fila['razonDeAspecto']:.0f}:1, no cuadrados)"
        # **«eligió», not «paró», and an EPOCH.** The sentence said «paró en el
        # paso N» because it was copied from a run that stopped: phase 2 had a
        # patience and the number was where it tripped. Since 2026-10-19 phase
        # 2 has no early stop -- it runs its ceiling whole and the criterion
        # PICKS a point inside it -- so «paró» names an event that no longer
        # happens, over a unit this package no longer reports in. Both halves
        # moved together, and CREDA's row label moved with them, because these
        # two figures are read side by side.
        etiqueta += f"\neligió la época {fila['epocaElegidaFase2']}"
        ax[r, 0].set_ylabel(etiqueta, fontsize=11.5, labelpad=10,
                            linespacing=1.4)
    quietas = max(fila["fuenteQuieta"] for fila in filas)
    # **One line, and the step said the way CREDA says it.** CREDA writes
    # «lunas · fuente y cabeza QUIETAS · paso 0 de 200 · azar 0.333»: the step
    # this frame stands at, out of the scale's DECLARED ceiling. This suptitle
    # used to say «cuadro 1 de 7 · paso 0 (cada 1)» and to carry the scale stamp
    # as a second line. **Settled by the operator on 2026-10-15: CREDA's frame.**
    # The frame index and the stride have no counterpart in it; the stride is
    # declared, printed by the generator notebook, and derivable from the record.
    #
    # The step comes from the FIRST row, as CREDA's does. Rows can only disagree
    # at their last frame, where each one's own `applied` step enters the
    # schedule; measured on this record the three rows stand at the same step at
    # every common frame.
    fig.suptitle(
        f"{familia} {chr(183)} fuente y cabeza QUIETAS {chr(183)} "
        f"época {filas[0]['pasos'][k] // STEPS_PER_EPOCH_DECLARED} de "
        f"{registro['scale']['epochs_2']} "
        f"{chr(183)} azar {toy_tables.azar(registro):.3f}",
        fontsize=13, y=0.985)
    # **One line, which is the shape of CREDA's.** Its footer names the markers,
    # the two kinds of centroid and the one term with a gradient, and stops.
    # This one used to append sentences about the per-axis box, about the rows
    # CREDA draws that this package has no backbone for, and about which rows
    # have no AdaBN rung. **Settled by the operator on 2026-10-15: CREDA's
    # frame.** The box note moved back to the row label it belongs to, which is
    # CREDA's own place for it; the absent-row and AdaBN sentences have no
    # counterpart in this frame and are gone from the picture -- both are still
    # read off the record, by `toy_tables.ausentes` and by the tables of the
    # same notebook, which is where a reader meets them.
    pie = (f"`x` fuente, `o` blanco {chr(183)} {chr(9671)} hueco = centroide de "
           f"la fuente, {chr(9671)} relleno = centroide del blanco, los dos por "
           f"clase REAL en las dos columnas {chr(183)} la fuente está quieta "
           f"(movimiento máximo medido {quietas:.2e}): el único término con "
           "camino de gradiente es el de la adaptación")
    fig.text(0.5, 0.012, pie, ha="center", fontsize=9.5)
    # The margins are measured rather than guessed: the row label on the left is
    # CREDA's two lines plus its conditional third, and the phase-2 panel on the
    # right carries a twin axis with its own label, so the defaults clip the
    # first line of one and the whole of the other.
    fig.subplots_adjust(left=0.085, right=0.955, top=0.880, bottom=0.085,
                        hspace=0.32, wspace=0.26)
    return fig


def _cuadro_animado(registro: dict, familia: str, filas: list, k: int):
    """Frame `k` rasterized and the figure closed: what the encoder is handed."""
    import matplotlib.pyplot as plt

    fig = figura_del_cuadro(registro, familia, filas, k)
    fig.canvas.draw()
    imagen = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return imagen


def _escribir_gif(ruta: Path, cuadros: list, al_paso=print) -> Path:
    """Encode the frames and READ BACK what was written.

    The final pause is the last frame's own duration and never a repeated frame:
    the encoder merges identical consecutive frames, so a duplicate disappears
    from the file with nothing failing. CREDA writes its GIFs through exactly
    this pair of steps, and the read-back is why: `mimsave` does not refuse a
    duration it cannot honour, it writes a file with no time assigned.
    """
    import imageio.v2 as imageio
    from PIL import Image as ImagenPIL

    if not cuadros:
        raise ValueError("no hay cuadros para escribir")
    duraciones = [MS_FRAME] * (len(cuadros) - 1) + [MS_PAUSE]
    imageio.mimsave(ruta, cuadros, duration=duraciones, loop=0)
    with ImagenPIL.open(ruta) as leido:
        if leido.n_frames != len(cuadros):
            raise ValueError(
                f"{ruta.name} quedó con {leido.n_frames} cuadros y se le "
                f"pasaron {len(cuadros)}")
        leido.seek(leido.n_frames - 1)
        if leido.info.get("duration") != MS_PAUSE:
            raise ValueError(
                f"{ruta.name} no lleva la pausa final declarada: "
                f"{leido.info.get('duration')} en vez de {MS_PAUSE} ms")
    al_paso(f"      {ruta.name}  {ruta.stat().st_size / 1e6:.2f} MB")
    return ruta


def animar_familia(registro: dict, familia: str, destino,
                   nombre_archivo: str | None = None,
                   al_paso=print) -> AnimacionLatente:
    """CREDA's per-dataset figure: one animated grid, rows of normalizations.

    Four columns, in CREDA's own order: the embedding coloured by the TRUE
    class, the same embedding coloured by what the head DECIDED, phase 1's loss
    and phase 2's. The frames are the phase-2 steps the generator left a frame
    at, so what moves between frames is `theta_t` and nothing else.

    **The rows are the normalizations the RECORD declares, counted from it.**
    CREDA draws three and this package has backbones for however many the record
    says, so a grid written as three would be a row drawn over nothing on the day
    one is missing, and a heading contradicting the picture under it. Which rows
    are absent is read from `normalizationsAbsent` and published by
    `toy_tables.ausentes`, by the architecture table and by the cell that calls
    this function -- and no longer by the GIF's own footer, which the operator
    held to CREDA's one line on 2026-10-15.

    Reads the artefact and runs nothing -- the split CREDA's own `animar`
    states the cost of getting wrong: *"Antes esto entrenaba las seis corridas
    y despues dibujaba, asi que retocar un color costaba el barrido entero."*
    Retouching a colour costs a redraw and not the sweep.
    """
    toy_tables._exigir_familia(registro, familia)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / (nombre_archivo or f"toy_paso_{familia}.gif")

    normas = toy_tables.normalizaciones(registro)
    filas = [_fila_animada(registro, familia, norma) for norma in normas]
    n = cuadros_comunes(len(fila["pasos"]) for fila in filas)
    cuadros = [_cuadro_animado(registro, familia, filas, k) for k in range(n)]
    _escribir_gif(ruta, cuadros, al_paso)

    angulos = [_angulo_entre_bases(filas[i]["base"], filas[i + 1]["base"])
               for i in range(len(filas) - 1)]
    rangos = np.array([[hi - lo for lo, hi in fila["limites"]]
                       for fila in filas], dtype=float)
    mayor, menor = rangos.max(0), rangos.min(0)
    ejes = {
        "compartidos": False,
        "discrepanciaRelativa": float(
            (mayor / np.clip(menor, 1e-12, None)).max() - 1.0),
        "anguloEntreBasesGrados": angulos,
        "porQue": ("cada arquitectura entrena su propio codificador, así que "
                   "tiene su propia base de PCA y su propia escala: una caja "
                   "común las aplastaría y haría leer como cercanas dos nubes "
                   "que viven en planos distintos"),
    }
    return AnimacionLatente(ruta=ruta, cuadros=n, paso=MS_FRAME, filas=filas,
                            ausentes=tuple(toy_tables.ausentes(registro)),
                            ejes=ejes)
