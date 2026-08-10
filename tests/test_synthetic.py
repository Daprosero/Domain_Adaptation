"""Level 3 - synthetic data. Deterministic, fixed seed, ground truth by construction.

State each expectation in the docstring BEFORE the assertion, so a passing test
cannot be mistaken for a hypothesis fitted after seeing the output.
"""

from __future__ import annotations

import numpy as np
from conftest import make_bags, simplex_rows

from MIL_CREDA.bag_kernel import bag_kernel, bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import confidences
from MIL_CREDA.global_term import class_global_loss, conservative_bounds, dependency_score
from MIL_CREDA.local_term import class_correspondence, local_distance, local_loss

SEED = 20260810
SIGMA = 1.3


def test_aligned_domains_score_better_than_shifted_ones() -> None:
    """Expectation: with source and target drawn from the SAME distribution, the
    per-class global loss of Eq. (36) is lower than when the target is displaced.

    Stated before running: the score of Eq. (33) grows when the joint structure
    is more ordered relative to the marginals, and the decreasing affine map of
    Eq. (36) turns that into a smaller loss. A shift between domains breaks that
    joint order, so the shifted pair must score worse.
    """
    rng = np.random.default_rng(SEED)
    n_s = n_t = 6

    def class_loss(shift: float) -> float:
        source = make_bags(rng, n_s, 4, shift=0.0)
        target = make_bags(rng, n_t, 4, shift=shift)
        K_ss = bag_kernel_matrix(source, source, SIGMA)
        K_st = bag_kernel_matrix(source, target, SIGMA)
        K_tt = bag_kernel_matrix(target, target, SIGMA)
        w = np.ones(n_t)
        K_s_c, K_st_c, K_t_c = conditional_blocks(
            K_ss, K_st, K_tt, np.arange(n_s), np.arange(n_t)
        )
        K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w)
        score = dependency_score(K_s_c, K_t_w, mixed_matrix(K_s_c, K_st_w, K_t_w))
        return class_global_loss(score, *conservative_bounds(n_s, n_t))

    aligned = np.mean([class_loss(0.0) for _ in range(20)])
    shifted = np.mean([class_loss(3.0) for _ in range(20)])
    assert aligned < shifted


def test_confident_predictions_dominate_the_local_aggregate() -> None:
    """Expectation: in Eq. (38) a target bag with near-zero confidence contributes
    almost nothing, so replacing its distance by a much larger one barely moves
    the aggregate, while doing the same on a confident bag moves it clearly.

    Stated before running: w_j^t controls each subject's external participation.
    """
    rng = np.random.default_rng(SEED)
    C = 4
    scores = simplex_rows(rng, 2, C)
    confident = np.zeros(C)
    confident[0] = 1.0
    unconfident = np.full(C, 1.0 / C)
    w = confidences(np.vstack([confident, unconfident]))
    assert w[0] > 0.99 and w[1] < 0.01
    assert scores.shape == (2, C)

    baseline = local_loss(np.array([0.5, 0.5]), w)
    moved_unconfident = local_loss(np.array([0.5, 3.5]), w)
    moved_confident = local_loss(np.array([3.5, 0.5]), w)
    assert abs(moved_unconfident - baseline) < 0.01
    assert moved_confident - baseline > 0.5


def test_correspondence_concentrates_on_the_most_similar_subject() -> None:
    """Expectation: as tau_loc shrinks, pi_{ji|c} of Eq. (28) concentrates on the
    source subject whose bag kernel value with the target is largest.

    Stated before running: the temperature controls concentration; small values
    approach a point mass on the nearest subject, large values approach uniform.
    """
    rng = np.random.default_rng(SEED)
    source = make_bags(rng, 5, 4)
    H_t, w_t = make_bags(rng, 1, 4, shift=0.3)[0]
    similarities = np.array([bag_kernel(H_s, w_s, H_t, w_t, SIGMA) for H_s, w_s in source])
    nearest = int(np.argmax(similarities))

    cold = class_correspondence(similarities, np.arange(5), tau=0.01)
    warm = class_correspondence(similarities, np.arange(5), tau=100.0)
    assert cold[nearest] > 0.9
    assert np.allclose(warm, 0.2, atol=1e-3)


def test_target_matching_its_reference_has_no_local_penalty() -> None:
    """Expectation: when the correspondence puts all mass on a source subject whose
    bag is a copy of the target's, Eq. (31) gives d_j^2 = 0 and Eq. (38) gives no
    penalty for that subject.

    Stated before running: d_j^2 vanishes exactly when the target representation
    coincides with its personalized reference.
    """
    rng = np.random.default_rng(SEED)
    source = make_bags(rng, 3, 4)
    K_ss = bag_kernel_matrix(source, source, SIGMA)
    H_t, w_t = source[1]
    cross = np.array([bag_kernel(H_s, w_s, H_t, w_t, SIGMA) for H_s, w_s in source])
    pi = np.array([0.0, 1.0, 0.0])
    d2 = local_distance(bag_kernel(H_t, w_t, H_t, w_t, SIGMA), cross, K_ss, pi)
    assert abs(d2) < 1e-9
    assert local_loss(np.array([d2]), np.array([1.0])) < 1e-9
