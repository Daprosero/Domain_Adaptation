"""MIL-CREDA's adaptation schedule: the mirror of CREDA's, with its own ceiling.

Each method names its own coefficient and carries its own default, so a run of
either one on its own gets the value that method was defined with — CREDA's
published 1e-4, MIL-CREDA's declared neutral of one. The experiment passes both
of them the same `delta` and the same `ceiling` explicitly, which is how "each
method keeps its own default" and "the two arms share one coefficient" hold at
the same time.

**The curve itself lives in exactly one place and this is a binding, not a
copy.** Two implementations of one formula across the two arms of a comparison
is the fork this benchmark exists not to have: they start identical, one of them
is corrected, and afterwards the arms differ in something nobody declared. So
the shape stays in `CREDA.schedules`, which owns it — the warm-up is CREDA's
`get_lambda`, applied to both sides precisely so the schedule is not one more
difference between them. `tests/test_creda_schedule.py` pins the two entry
points to the same numbers.

This lives in the benchmark package and not in `src/MIL_CREDA/`. A warm-up is a
training heuristic and implements no equation of the proposal, so it could only
sit beside the method's modules by declaring a `__provenance__` it has no right
to — and a falsified provenance empties the one check that keeps the code tied
to the mathematics.
"""

from __future__ import annotations

from CREDA.schedules import creda_ramp

#: MIL-CREDA's own ceiling. One is the neutral of Eq. (39): Eq. (36) normalizes
#: the global score by the conservative bounds, Eq. (38) bounds the local term,
#: and Eq. (18) is divided by its own supremum B_src, so all three terms live in
#: [0, 1) and a coefficient of one weighs them equally. It is not a value chosen
#: by looking at outcomes; it is the number the normalization makes meaningful.
MILCREDA_CEILING = 1.0


def milcreda_ramp(epoch: int, epochs: int, delta: float,
                  ceiling: float = MILCREDA_CEILING) -> float:
    """The coefficient of Eq. (39) as a schedule: how fast it grows, and how far.

    Identical to `creda_ramp` for identical arguments, and deliberately so — the
    two families must differ in the term and in nothing else. What differs is the
    default: this one starts from the neutral of a normalized objective, that one
    from CREDA's published `creda_lambda_special`.
    """
    return creda_ramp(epoch, epochs, delta=delta, ceiling=ceiling)
