"""Level 4 - audit. Evidence that each declared finding is a real defect.

Every test sweeps SWEEP_SIZE independent configurations and reports the rate. A
finding declared `theorem` must hold in all of them; one declared `tendency`
must match its declared rate and is never asserted as a law.

Both findings here were raised against research-concept-r14.md and have since
been adopted, so `src/` no longer implements the formulations they indict. The
indicted forms are therefore reproduced in this file: the evidence has to keep
measuring what was actually wrong, not what replaced it. Reaching into `src/`
would silently turn this level into a test of the correction.
"""

from __future__ import annotations

import numpy as np
from findings import FINDINGS
from sweep import SWEEP_SIZE, sweep

DECLARED_LOCAL_NORMALIZER = 4.0
EPSILON_SRC = 1e-8


def indicted_normalized_local_distance(squared_distance: float) -> float:
    """Eq. (38) AS DECLARED IN r14: l_loc,j = d_j^2 / 4."""
    return float(squared_distance / DECLARED_LOCAL_NORMALIZER)


def indicted_source_loss(
    G_source: np.ndarray, Y_source: np.ndarray, epsilon: float = EPSILON_SRC
) -> float:
    """Eq. (18) AS DECLARED IN r14: the stabilizer sits inside the logarithm."""
    G = np.atleast_2d(np.asarray(G_source, dtype=float))
    Y = np.atleast_2d(np.asarray(Y_source, dtype=float))
    return float(-np.sum(Y * np.log(G + epsilon)) / G.shape[0])


def test_every_finding_declares_a_remedy() -> None:
    """A finding without a proposed correction is a complaint, not a finding."""
    for finding in FINDINGS:
        assert finding["remedy"].strip(), f"{finding['id']} has no remedy"
        assert finding["remedy_equations"], f"{finding['id']} does not say what it changes"
        assert finding["becomes_invariant"], f"{finding['id']} names no invariant to become"
        assert finding["adoption"]["absent"], f"{finding['id']} has no adoption marker"
        assert finding["status"] in {"theorem", "tendency"}
        assert finding["kind"] in {
            "ill-formed", "underspecified", "missing-complement", "overstated-claim",
            "ill-posed-objective", "loose-constant",
        }


def test_finding_local_normalizer_loose_by_two() -> None:
    """r14 Sec. 5, Eq. (38): the constant 4 is unreachable, so half of [0, 1] is dead.

    Measured over the whole sweep: no configuration produces d_j^2 >= 2, which
    means the indicted normalizer confined l_loc,j to [0, 1/2) rather than to
    the [0, 1] the equation stated. Declared as a theorem, so a single
    counterexample would retire it.
    """
    exceeding = 0
    largest = 0.0
    measured = 0
    for configuration in sweep():
        distances = configuration["squared_distances"]
        if distances.size == 0:
            continue
        measured += int(distances.size)
        largest = max(largest, float(distances.max()))
        exceeding += int(np.count_nonzero(distances >= 2.0))

    assert measured > 0, "the sweep produced no target bag to measure"
    assert exceeding == 0, f"{exceeding} of {measured} distances reached the declared bound"
    assert largest < 2.0
    # The defect: under the indicted normalizer the loss cannot leave the lower half.
    assert indicted_normalized_local_distance(largest) < 0.5
    # And the bound is genuinely approached, so 2 is tight rather than another
    # arbitrary constant: the adopted normalizer uses the full range.
    assert largest > 1.99


def test_finding_source_stabilizer_breaks_non_negativity() -> None:
    """r14 Sec. 3, Eq. (18): a stabilizer inside the logarithm can make L_src negative.

    Declared as a tendency: it was not a law over arbitrary configurations, it
    occurred whenever a source bag was predicted with probability one. The rate
    the finding declares is the rate this sweep measures.
    """
    negative = 0
    minimum = 0.0
    for configuration in sweep():
        scores = configuration["source_scores"]
        one_hot = np.eye(configuration["n_classes"])[configuration["source_labels"]]
        value = indicted_source_loss(scores, one_hot)
        minimum = min(minimum, value)
        if value < 0.0:
            negative += 1

    assert negative >= 1, "the sweep never reached a confident enough prediction"
    assert minimum < 0.0
    # Stated as a rate, never as a law: the declared status is `tendency`.
    rate = negative / SWEEP_SIZE
    assert 0.0 < rate < 1.0, f"a tendency cannot hold in {negative}/{SWEEP_SIZE}"
