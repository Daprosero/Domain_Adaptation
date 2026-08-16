"""The reparameterization of CREDA's schedule preserved every loop's coefficient.

`tests/golden_creda_schedule.json` was written with `get_lambda` untouched and
committed before the refactor, so these are not assertions against a fixture
written by the same hand that wrote the code being checked: they are the numbers
prior work actually produced.

Three things are established here, and they fail for different reasons:

  * the three loops nobody meant to touch — DANN, ADDA, CDAN+E — still get the
    schedule at full strength. A ceiling applied to them would not attenuate
    them; CDAN+E feeds `lambda_val` to a gradient-reversal hook, so scaling it
    down switches its domain adversary off rather than weakening it.
  * CREDA's loop, which did change, computes the same coefficient it always did.
  * and what that change cost, stated as a bound rather than denied. Moving
    `lambda_` out of `CREDALoss` and into the ceiling reassociates a product, so
    the result is the same number in arithmetic and up to two ULP away in
    float64. Bounding it is the honest form: the cost is there, it is measured,
    and a regression past it fails.

The duplication between `get_lambda` and `creda_ramp` is internal to prior work
and crosses no comparison, but it can still drift; the first test would catch it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from CREDA.schedules import creda_ramp as ramp

GOLDEN = json.loads(
    (Path(__file__).parent / "golden_creda_schedule.json").read_text()
)["rows"]


def test_the_untouched_loops_still_see_the_schedule_they_saw_before():
    """DANN, ADDA and CDAN+E read `get_lambda` directly, at ceiling one.

    Exact equality rather than a tolerance: `1.0 * y` is `y` for every finite
    float, so the default ceiling cannot perturb a single bit. A tolerance here
    would quietly accept a reassociation that a training loop compounds.
    """
    for row in GOLDEN:
        assert ramp(row["epoch"], row["epochs"], row["delta"], ceiling=1.0) == row["shared"]


def test_moving_the_coefficient_to_the_ceiling_costs_at_most_two_ulp():
    """What `train_creda` gave up by assembling its coefficient in one place.

    Published:  get_lambda(...) * (lambda_ * loss)   — lambda_ inside CREDALoss
    Now:       (lambda_ * get_lambda(...)) * loss    — lambda_ as the ceiling

    The same number in arithmetic, not the same float: multiplication does not
    associate. This bounds the difference rather than denying it, so the cost is
    a measurement somebody can weigh instead of a claim nobody checked.

    It is small against what the runs already carry. `set_seed` sets cudnn
    deterministic but never `torch.use_deterministic_algorithms`, so the
    reductions inside `CREDALoss` vary by more than this between two runs of the
    unmodified code.
    """
    losses = (1e-4, 0.37, 2.5, 9.99)
    worst = 0.0
    for row in GOLDEN:
        for lambda_ in (1e-4, 1e-3, 1e-2, 1.0):
            ceiling = ramp(row["epoch"], row["epochs"], row["delta"], ceiling=lambda_)
            for x in losses:
                published = row["shared"] * (lambda_ * x)
                now = ceiling * x
                if published:
                    worst = max(worst, abs(now - published) / abs(published))
    # Two ULP of float64. A regression past this is a change of arithmetic, not
    # of association, and deserves to fail.
    assert worst <= 2 ** -51, f"reassociation drifted to {worst}"


def test_credas_default_ceiling_is_its_published_coefficient():
    """A caller that passes no ceiling gets `creda_lambda_special`, not one.

    The value is still read from the caller's cfg in `train_creda`, because it is
    not always the same number — `Results_Generator.ipynb` carries a
    configuration whose `creda_lambda_default` is 1e-3. The default here serves
    whoever calls the schedule on its own.
    """
    from CREDA.schedules import CREDA_CEILING

    assert CREDA_CEILING == 1e-4
    for row in GOLDEN:
        assert ramp(row["epoch"], row["epochs"], row["delta"]) == pytest.approx(
            row["creda_1e-4"], rel=0, abs=1e-20
        )


@pytest.mark.parametrize("ceiling_key, ceiling", [
    ("creda_1e-4", 1e-4),          # the published value for the special domains
    ("creda_1e-2", 1e-2),
    ("creda_1", 1.0),              # what the bounded benchmark runs CREDA at
])
def test_credas_coefficient_is_what_the_product_used_to_be(ceiling_key, ceiling):
    """`ramp(ceiling=c)` replaces `get_lambda() * CREDALoss.lambda_creda`."""
    for row in GOLDEN:
        assert ramp(
            row["epoch"], row["epochs"], row["delta"], ceiling=ceiling
        ) == pytest.approx(row[ceiling_key], rel=0, abs=1e-18)


def test_the_shared_implementation_matches_the_one_the_other_methods_use():
    """The copy inside `training_pipeline` and this one have not drifted apart.

    Importing `training_pipeline` costs scikit-image, timm, pandas and
    matplotlib, which is why the harness cannot import it and why the schedule
    was duplicated in the first place. A test can afford it; a benchmark cannot.
    """
    training_pipeline = pytest.importorskip(
        "CREDA.training_pipeline",
        reason="requirements.txt declares scikit-image; install it to pin the copies",
    )
    for row in GOLDEN:
        assert training_pipeline.get_lambda(
            row["epoch"], row["epochs"], delta=row["delta"]
        ) == ramp(row["epoch"], row["epochs"], row["delta"], ceiling=1.0)


def test_a_negative_ceiling_is_refused_rather_than_rewarding_misalignment():
    with pytest.raises(ValueError):
        ramp(1, 20, 20.0, ceiling=-1.0)
