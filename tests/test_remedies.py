"""Level 5 - remedies. Each proposed correction, validated at the same rigour.

A remedy is accepted only when the sweep shows BOTH poles: the correction
satisfies the criterion, and the declared formulation fails it. With one pole
only, nothing distinguishes a real improvement from a measurement that would
have passed whatever it was handed — `verify` reports such a test as a remedy
without control.

Every test starts by refusing to measure what was not ruled admissible:
soundness and expressibility are decided first, efficacy second.

The proposed replacements live here, never in src/.
"""

from __future__ import annotations

import numpy as np
from admissibility import require_admissible
from sweep import SWEEP_SIZE, sweep

from MIL_CREDA.local_term import local_loss, normalized_local_distance
from MIL_CREDA.objective import source_loss

EPSILON_SRC = 1e-8


def remedy_normalized_local_distance(squared_distance: float) -> float:
    """PROPOSED Eq. (38): l_loc,j = d_j^2 / 2.

    The tight constant. Strict positivity of the Gaussian instance kernel makes
    <Psi(B_j^t), mu_j^s> > 0, hence d_j^2 < 2.
    """
    return float(squared_distance / 2.0)


def remedy_local_loss(
    squared_distances: np.ndarray, target_weights: np.ndarray, epsilon: float = 1e-8
) -> float:
    """PROPOSED Eq. (38): the same aggregation over the retightened per-bag loss."""
    losses = np.array(
        [remedy_normalized_local_distance(d2) for d2 in np.asarray(squared_distances, float)],
        dtype=float,
    )
    w = np.asarray(target_weights, dtype=float).ravel()
    if losses.size == 0:
        return 0.0
    return float((w @ losses) / (w.sum() + epsilon))


def remedy_source_loss(
    G_source: np.ndarray, Y_source: np.ndarray, epsilon: float = EPSILON_SRC
) -> float:
    """PROPOSED Eq. (18): ln((g + eps) / (1 + eps)), so the argument never exceeds one."""
    G = np.atleast_2d(np.asarray(G_source, dtype=float))
    Y = np.atleast_2d(np.asarray(Y_source, dtype=float))
    return float(-np.sum(Y * np.log((G + epsilon) / (1.0 + epsilon))) / G.shape[0])


def test_remedy_local_normalizer_loose_by_two() -> None:
    """Resolves it: dividing by 2 makes the normalizer tight, so l_loc,j approaches 1.
    Preserves: l_loc,j and the aggregate L_loc stay within [0, 1] in every
    configuration, and the loss remains monotone in the discrepancy.
    Control: the declared /4 form never leaves [0, 1/2) over the same sweep.
    """
    require_admissible("local_normalizer_loose_by_two")

    remedy_peak = 0.0
    declared_peak = 0.0
    measured = 0
    for configuration in sweep():
        distances = configuration["squared_distances"]
        if distances.size == 0:
            continue
        measured += int(distances.size)
        weights = configuration["target_weights"]
        for d2 in distances:
            remedied = remedy_normalized_local_distance(d2)
            declared = normalized_local_distance(d2)
            # Preserved: still a normalized loss, in the interval the equation declares.
            assert -1e-12 <= remedied <= 1.0 + 1e-12
            # Preserved: monotone in the discrepancy it measures.
            assert (remedied > 0.0) == (d2 > 0.0)
            remedy_peak = max(remedy_peak, remedied)
            declared_peak = max(declared_peak, declared)
        assert -1e-12 <= remedy_local_loss(distances, weights) <= 1.0 + 1e-12
        assert -1e-12 <= local_loss(distances, weights) <= 1.0 + 1e-12

    assert measured > 0, "the sweep produced no target bag to measure"
    # Resolves it: the remedy reaches the top of its declared interval.
    assert remedy_peak > 0.99
    # Control: the declared formulation cannot, over the very same sweep.
    assert declared_peak < 0.5
    assert remedy_peak > declared_peak


def test_remedy_source_stabilizer_breaks_non_negativity() -> None:
    """Resolves it: the renormalized argument never exceeds one, so L_src >= 0 always,
    and a perfect prediction costs exactly zero.
    Preserves: the two forms differ by the constant ln(1 + eps), so the ranking of
    configurations — and with it the direction of the gradient — is untouched, and
    the stabilizer still keeps g = 0 finite.
    Control: the declared form goes negative on the same sweep and at the same
    perfect prediction.
    """
    require_admissible("source_stabilizer_breaks_non_negativity")

    remedy_minimum = np.inf
    declared_negatives = 0
    shift = float(np.log1p(EPSILON_SRC))
    for configuration in sweep():
        scores = configuration["source_scores"]
        one_hot = np.eye(configuration["n_classes"])[configuration["source_labels"]]
        remedied = remedy_source_loss(scores, one_hot)
        declared = source_loss(scores, one_hot, EPSILON_SRC)
        # Resolves it: never negative.
        assert remedied >= -1e-15, f"remedy went negative: {remedied}"
        # Preserves: a constant offset, so no configuration is reordered.
        assert np.isclose(remedied - declared, shift, atol=1e-12)
        remedy_minimum = min(remedy_minimum, remedied)
        if declared < 0.0:
            declared_negatives += 1

    # Control: the declared formulation fails the criterion over the same sweep.
    assert declared_negatives >= 1, "the control never exercised the defect"
    assert declared_negatives < SWEEP_SIZE, "declared as a tendency, not as a law"
    assert remedy_minimum >= 0.0

    # The decisive case, stated directly: a bag predicted with certainty.
    perfect = np.array([[1.0, 0.0, 0.0]])
    label = np.array([[1.0, 0.0, 0.0]])
    assert perfect.max() == 1.0, "the fixture is not a certain prediction"
    assert np.isclose(remedy_source_loss(perfect, label), 0.0, atol=1e-15)
    assert source_loss(perfect, label, EPSILON_SRC) < 0.0

    # Preserved: the stabilizer still does its only job at the other extreme.
    hopeless = np.array([[0.0, 1.0, 0.0]])
    assert np.isfinite(remedy_source_loss(hopeless, label))
