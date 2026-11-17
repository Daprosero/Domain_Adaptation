"""CREDA's adaptation schedule, with its coefficient as the ceiling it climbs to.

`training_pipeline.get_lambda` is untouched and still serves DANN, ADDA and
CDAN+E. Those three apply the schedule at full strength — CDAN+E feeds it to a
gradient-reversal hook — so a ceiling below one would not attenuate them, it
would switch their domain adversary off. Their call sites do not change, and
`git diff` says so without needing a test to prove it.

What changes is where CREDA's own coefficient lives. It used to be applied inside
`CREDALoss.lambda_creda` while the schedule ran to one outside, so the effective
coefficient was a product assembled in two files. Here it is one number in one
place: how fast the schedule grows, and how far.

`train_creda` still reads its value from the caller's configuration, because that
is where it always came from and it is not always the same number —
`Results_Generator.ipynb` carries at least two configurations, one with
`creda_lambda_default` at 1e-3 against a `creda_lambda_special` of 1e-4. The
default below is the published special value, for a caller that passes none.

**No equation in the managed revision, and the provenance below says so.** The
curve is prior work's `training_pipeline.get_lambda` — a warm-up schedule DANN,
ADDA and CDAN+E already ran — scaled by a published coefficient. The revision
states the coefficients of the objective as free non-negative constants
(Eq. 39) and declares no time dependence for them anywhere: there is no
schedule to be faithful to, so naming a revision here would claim a binding the
document does not offer. `CREDA_CEILING` is a published NUMBER, read off a
configuration, not a constant the mathematics fixes.

The invariants below are what this module can be held to instead, and
`tests/test_creda_schedule.py` holds each against
`tests/golden_creda_schedule.json` — numbers committed before the
reparameterization, so they are what prior work produced rather than a fixture
written by the same hand as the code.
"""

from __future__ import annotations

import math


__provenance__ = {
    "revision": None,
    "claims_no_equation": (
        "a warm-up schedule and a published coefficient, not a term; "
        "see the module docstring"),
    "invariants": [
        "the_untouched_loops_still_see_the_schedule_they_saw_before",
        "credas_default_ceiling_is_its_published_coefficient",
        "credas_coefficient_is_what_the_product_used_to_be",
        "moving_the_coefficient_to_the_ceiling_costs_at_most_two_ulp",
        "and_the_gradient_comes_out_bit_identical",
        "a_negative_ceiling_is_refused_rather_than_rewarding_misalignment",
    ],
}

#: CREDA's published coefficient for its special domains, `creda_lambda_special`.
CREDA_CEILING = 1e-4


def creda_ramp(epoch: int, epochs: int, delta: float,
               ceiling: float = CREDA_CEILING) -> float:
    """`get_lambda`'s curve scaled to `ceiling`: how fast it grows, and how far.

    Zero at the first epoch and approaching `ceiling` afterwards, at a rate set
    by `delta`. At `ceiling=1.0` this is bit-for-bit `get_lambda`, because
    multiplying a finite float by one returns it unchanged.

    The curve runs on the fraction of training elapsed rather than on the epoch,
    so the same `delta` warms up over a different number of epochs depending on
    how long the run is. Three epochs at delta 20 is already at 0.9975 by the
    second; twenty epochs take about five to get there. A short run is therefore
    not a slower version of a long one, it is a run with almost no warm-up.
    """
    if ceiling < 0:
        raise ValueError("the ceiling of an adaptation schedule cannot be negative")
    p = epoch / epochs
    return ceiling * float((1 - math.exp(-delta * p)) / (1 + math.exp(-delta * p)))
