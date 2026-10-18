"""Level 3 - synthetic data. Deterministic, fixed seed, ground truth by construction.

State each expectation in the docstring BEFORE the assertion, so a passing test
cannot be mistaken for a hypothesis fitted after seeing the output.
"""

from __future__ import annotations

import torch
from conftest import Sampler, allclose, make_bags, simplex_rows

from MIL_CREDA import DTYPE
from MIL_CREDA.bag_kernel import bag_kernel, bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import confidences
from MIL_CREDA.global_term import (
    balanced_dependency_score,
    balanced_mixture,
    class_global_loss_cosh,
)
from MIL_CREDA.renyi import quadratic_entropy
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
    rng = Sampler(SEED)
    n_s = n_t = 6

    def class_loss(shift: float) -> torch.Tensor:
        source = make_bags(rng, n_s, 4, shift=0.0)
        target = make_bags(rng, n_t, 4, shift=shift)
        K_ss = bag_kernel_matrix(source, source, SIGMA)
        K_st = bag_kernel_matrix(source, target, SIGMA)
        K_tt = bag_kernel_matrix(target, target, SIGMA)
        w = torch.ones(n_t, dtype=DTYPE)
        K_s_c, K_st_c, K_t_c = conditional_blocks(
            K_ss, K_st, K_tt, torch.arange(n_s), torch.arange(n_t)
        )
        K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w)
        # r33's path, which is the one the consumer runs. It was r25's
        # `dependency_score`/`conservative_bounds`/`class_global_loss` until
        # 2026-10-18; those three were removed with the ends they computed, and
        # the EXPECTATION above is about the formulation rather than about
        # either map, so it moved rather than being deleted.
        breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_w, K_t_w)
        score = balanced_dependency_score(K_s_c, K_st_w, K_t_w)
        return class_global_loss_cosh(
            score, quadratic_entropy(breve_s), quadratic_entropy(breve_t))

    aligned = torch.stack([class_loss(0.0) for _ in range(20)]).mean()
    shifted = torch.stack([class_loss(3.0) for _ in range(20)]).mean()
    assert float(aligned) < float(shifted)


def test_the_local_aggregate_ignores_the_confidences() -> None:
    """Expectation: in r33's Eq. (38) the aggregate is the UNIFORM mean, so
    moving one bag's distance moves it by the same amount whichever bag it is,
    and lowering a bag's confidence changes nothing at all.

    Stated before running: `w_j^t` left Eq. (38) at r32, because it is a
    differentiable function of the classifier's outputs -- a weight the model
    computes can be lowered on the worst-aligned subject instead of aligning
    it. This test replaces one that asserted the OPPOSITE claim (a near-zero
    confidence bag barely moved the aggregate), which was true of r31 and is
    the behaviour r33 removed.
    """
    rng = Sampler(SEED)
    C = 4
    confident = torch.zeros(C, dtype=DTYPE)
    confident[0] = 1.0
    unconfident = torch.full((C,), 1.0 / C, dtype=DTYPE)
    w = confidences(torch.stack([confident, unconfident]))
    assert float(w[0]) > 0.99 and float(w[1]) < 0.01

    baseline = local_loss(torch.tensor([0.5, 0.5], dtype=DTYPE))
    moved_first = local_loss(torch.tensor([3.5, 0.5], dtype=DTYPE))
    moved_second = local_loss(torch.tensor([0.5, 3.5], dtype=DTYPE))
    # Same move, same effect: under r31 the second of these was ~0.
    assert allclose(moved_first, moved_second)
    assert float(moved_first - baseline) > 0.5
    # And the confidences cannot enter: `local_loss` has nowhere to take them.
    assert 'target_weights' not in local_loss.__code__.co_varnames


def test_correspondence_concentrates_on_the_most_similar_subject() -> None:
    """Expectation: as tau_loc shrinks, pi_{ji|c} of Eq. (28) concentrates on the
    source subject whose bag kernel value with the target is largest.

    Stated before running: the temperature controls concentration; small values
    approach a point mass on the nearest subject, large values approach uniform.
    """
    rng = Sampler(SEED)
    source = make_bags(rng, 5, 4)
    H_t, w_t = make_bags(rng, 1, 4, shift=0.3)[0]
    similarities = torch.stack(
        [bag_kernel(H_s, w_s, H_t, w_t, SIGMA) for H_s, w_s in source]
    )
    nearest = int(torch.argmax(similarities))

    cold = class_correspondence(similarities, torch.arange(5), tau=0.01)
    warm = class_correspondence(similarities, torch.arange(5), tau=100.0)
    assert float(cold[nearest]) > 0.9
    assert allclose(warm, 0.2, atol=1e-3)


def test_target_matching_its_reference_has_no_local_penalty() -> None:
    """Expectation: when the correspondence puts all mass on a source subject whose
    bag is a copy of the target's, Eq. (31) gives d_j^2 = 0 and Eq. (38) gives no
    penalty for that subject.

    Stated before running: d_j^2 vanishes exactly when the target representation
    coincides with its personalized reference.
    """
    rng = Sampler(SEED)
    source = make_bags(rng, 3, 4)
    K_ss = bag_kernel_matrix(source, source, SIGMA)
    H_t, w_t = source[1]
    cross = torch.stack([bag_kernel(H_s, w_s, H_t, w_t, SIGMA) for H_s, w_s in source])
    pi = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    d2 = local_distance(bag_kernel(H_t, w_t, H_t, w_t, SIGMA), cross, K_ss, pi)
    assert abs(float(d2)) < 1e-9
    assert float(local_loss(d2.reshape(1))) < 1e-9
