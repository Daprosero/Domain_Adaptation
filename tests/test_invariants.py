"""Level 2 - properties and invariants. The bridge between proposal and code.

One test per mathematical claim the proposal makes. The name MUST be
`test_<invariant_id>`, matching an id declared in a module's
`__provenance__["invariants"]`; verification pairs them by that exact name.

Each docstring cites the passage it enforces. A test whose claim cannot be
traced back to the proposal does not belong here.

Every assertion is made over the tensors `src/` computes. The claims are the same
claims and the tolerances are the same tolerances: this level is what establishes
that changing the backend did not change the mathematics.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import Sampler, allclose, close, make_bag, make_bags, one_hot, simplex_rows

from MIL_CREDA import DTYPE
from MIL_CREDA.attention import bag_embedding, bag_weights, relevance_logits
from MIL_CREDA.bag_kernel import bag_gram, bag_kernel, bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import bag_confidence, confidences, pseudolabel
from MIL_CREDA.global_term import (
    class_global_loss,
    conservative_bounds,
    dependency_score,
    global_loss,
)
from MIL_CREDA.kernels import gaussian_kernel, gram
from MIL_CREDA.local_term import (
    class_correspondence,
    local_distance,
    local_loss,
    normalized_local_distance,
    total_correspondence,
)
from MIL_CREDA.objective import source_bound, source_loss, total_objective
from MIL_CREDA.renyi import quadratic_entropy, renyi_entropy, trace_normalize

TOL = 1e-10
SIGMA = 1.3


# --------------------------------------------------------------------------
# kernels.py - Section 1, Eqs. (2), (19)
# --------------------------------------------------------------------------


def test_kernel_unit_diagonal_and_bounded(rng: Sampler) -> None:
    """r16 Sec. 1, Eq. (2): 0 < kappa_sigma <= 1 with kappa_sigma(x, x) = 1.

    The unit diagonal is what gives every image unit norm in the RKHS, which
    Section 5 later uses to bound the bag representations and the local
    distance.
    """
    X = rng.normal(size=(9, 4))
    K = gram(X, SIGMA)
    assert bool((K > 0.0).all())
    assert bool((K <= 1.0 + TOL).all())
    assert allclose(torch.diagonal(K), 1.0, atol=TOL)


def test_kernel_psd(rng: Sampler) -> None:
    """r16 Sec. 1: the Gaussian kernel is positive semidefinite."""
    X = rng.normal(size=(11, 3))
    eigenvalues = torch.linalg.eigvalsh(gram(X, SIGMA))
    assert float(eigenvalues.min()) >= -1e-8


# --------------------------------------------------------------------------
# renyi.py - Section 2, Eqs. (9), (10)
# --------------------------------------------------------------------------


def test_trace_one_after_normalization(rng: Sampler) -> None:
    """r16 Sec. 2, Eq. (9): A = K / tr(K) satisfies tr(A) = 1.

    The constant Z_{d,sigma} cancels here, which is why no downstream matrix
    carries it.
    """
    A = trace_normalize(gram(rng.normal(size=(7, 5)), SIGMA))
    assert close(torch.trace(A), 1.0, atol=TOL)


def test_h2_frobenius_matches_spectral(rng: Sampler) -> None:
    """r16 Sec. 2, Eqs. (9)-(10): H_2 via ||A||_F^2 equals the spectral form at alpha=2.

    Two independent routes: the closed Frobenius expression and the eigenvalue
    sum of the general definition. Their agreement is the claim that lets the
    formulation skip the spectral decomposition.
    """
    A = trace_normalize(gram(rng.normal(size=(8, 4)), SIGMA))
    assert close(quadratic_entropy(A), renyi_entropy(A, 2.0), atol=1e-9)


def test_h2_bounded_by_log_n(rng: Sampler) -> None:
    """r16 Sec. 2: 0 <= H_2(A) <= ln N for a unit-trace positive semidefinite A.

    The rank-one spectrum reaches the minimum and the uniform spectrum the
    maximum. Section 5 builds the conservative bounds of Eq. (34) on this.
    """
    for size in (2, 5, 9):
        A = trace_normalize(gram(rng.normal(size=(size, 3)), SIGMA))
        assert -TOL <= float(quadratic_entropy(A)) <= math.log(size) + TOL
    concentrated = trace_normalize(torch.ones((6, 6), dtype=DTYPE))
    assert close(quadratic_entropy(concentrated), 0.0, atol=TOL)
    uniform = trace_normalize(torch.eye(6, dtype=DTYPE))
    assert close(quadratic_entropy(uniform), math.log(6), atol=TOL)


# --------------------------------------------------------------------------
# attention.py - Section 3, Eqs. (14)-(16)
# --------------------------------------------------------------------------


def test_bag_weights_on_simplex(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (15): beta > 0 and sum over the bag equals one.

    Normalization is strictly within one bag; the weights express relevance
    among the instances of a single subject.
    """
    dimension, hidden = 4, 6
    H = rng.normal(size=(5, dimension))
    logits = relevance_logits(
        H, rng.normal(size=(hidden, dimension)), rng.normal(size=hidden),
        rng.normal(size=hidden),
    )
    beta = bag_weights(logits)
    assert beta.shape == (5,)
    assert bool((beta > 0.0).all())
    assert close(beta.sum(), 1.0, atol=TOL)


def test_bag_embedding_permutation_invariant(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (16): z is invariant under permutations of the bag.

    Each weight is paired with its own embedding, so reordering the instances
    together with their weights cannot move the representation.
    """
    H, beta = make_bag(rng, 6, 4)
    order = rng.permutation(6)
    assert allclose(bag_embedding(H, beta), bag_embedding(H[order], beta[order]), atol=TOL)


# --------------------------------------------------------------------------
# bag_kernel.py - Section 3, Eqs. (20)-(21)
# --------------------------------------------------------------------------


def test_bag_kernel_symmetric(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (21): kappa^B is an inner product in H_I, hence symmetric."""
    (H_u, w_u), (H_v, w_v) = make_bags(rng, 2, 4)
    assert close(
        bag_kernel(H_u, w_u, H_v, w_v, SIGMA),
        bag_kernel(H_v, w_v, H_u, w_u, SIGMA),
        atol=TOL,
    )


def test_bag_kernel_psd(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (21): the bag kernel is positive semidefinite."""
    eigenvalues = torch.linalg.eigvalsh(bag_gram(make_bags(rng, 9, 4), SIGMA))
    assert float(eigenvalues.min()) >= -1e-8


def test_bag_self_similarity_at_most_one(rng: Sampler) -> None:
    """r16 Sec. 5: ||Psi(B)||_{H_I} <= 1, so kappa^B(B, B) <= 1.

    Psi is a convex combination of unit-norm images, so the triangle inequality
    caps its norm. This is the first half of the argument bounding d_j^2.
    """
    for H, w in make_bags(rng, 12, 4):
        assert 0.0 < float(bag_kernel(H, w, H, w, SIGMA)) <= 1.0 + TOL


def test_bag_kernel_reduces_to_instance_kernel_on_singletons(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (21): on one-instance bags the double sum is kappa^I itself.

    With m = 1 the only weight is one, so the bag kernel must collapse onto the
    instance kernel it is built from.
    """
    h, h_prime = rng.normal(size=(1, 4)), rng.normal(size=(1, 4))
    one = torch.ones(1, dtype=DTYPE)
    assert close(
        bag_kernel(h, one, h_prime, one, SIGMA),
        gaussian_kernel(h, h_prime, SIGMA)[0, 0],
        atol=TOL,
    )


# --------------------------------------------------------------------------
# confidence.py - Section 3, Eqs. (22), (24)
# --------------------------------------------------------------------------


def test_pseudolabel_deterministic_tie_break() -> None:
    """r16 Sec. 3, Eq. (22): a deterministic rule with a fixed tie-break.

    Ties must resolve the same way every time, otherwise the class routing of
    Eq. (23) would not be a function of the classifier's output.
    """
    tied = torch.full((4,), 0.25, dtype=DTYPE)
    assert pseudolabel(tied) == pseudolabel(tied) == 0
    assert pseudolabel(torch.tensor([0.1, 0.6, 0.3], dtype=DTYPE)) == 1


def test_confidence_in_unit_interval(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (24): w_j in [0, 1], because 0 <= H_2(g) <= ln C."""
    for C in (2, 3, 6):
        for row in simplex_rows(rng, 25, C):
            assert -TOL <= float(bag_confidence(row, C)) <= 1.0 + TOL


def test_confidence_one_at_onehot_zero_at_uniform() -> None:
    """r16 Sec. 3, Eq. (24): concentrated prediction -> 1, uniform prediction -> 0."""
    C = 5
    onehot = torch.zeros(C, dtype=DTYPE)
    onehot[2] = 1.0
    assert close(bag_confidence(onehot, C), 1.0, atol=TOL)
    assert close(bag_confidence(torch.full((C,), 1.0 / C, dtype=DTYPE), C), 0.0, atol=TOL)
    with pytest.raises(ValueError):
        bag_confidence(torch.ones(1, dtype=DTYPE), 1)


# --------------------------------------------------------------------------
# conditional.py - Section 3, Eqs. (23), (25)-(27)
# --------------------------------------------------------------------------


def _class_setup(rng: Sampler, n_source: int = 4, n_target: int = 3):
    """One class's blocks, built from a single bag Gram over both domains."""
    source = make_bags(rng, n_source, 4, shift=0.0)
    target = make_bags(rng, n_target, 4, shift=0.6)
    K_ss = bag_kernel_matrix(source, source, SIGMA)
    K_st = bag_kernel_matrix(source, target, SIGMA)
    K_tt = bag_kernel_matrix(target, target, SIGMA)
    w = confidences(simplex_rows(rng, n_target, 3))
    return K_ss, K_st, K_tt, torch.arange(n_source), torch.arange(n_target), w


def test_weighted_target_equals_diagonal_congruence(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (26): K_c^t . (w w^T) equals D_c^t K_c^t D_c^t.

    The proposal states the Hadamard product and the diagonal congruence as the
    same object; the test computes both routes and compares them.
    """
    K_ss, K_st, K_tt, S_c, T_c, w = _class_setup(rng)
    _, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, S_c, T_c)
    _, congruence = weighted_blocks(K_st_c, K_t_c, w)
    hadamard = K_t_c * torch.outer(w, w)
    assert allclose(hadamard, congruence, atol=TOL)


def test_source_block_never_weighted(rng: Sampler) -> None:
    """r16 Sec. 3, Eqs. (25)-(27): confidence touches only blocks with target bags.

    Source labels are observed, not estimated, so two different confidence
    vectors must leave the source block of the mixed matrix identical.
    """
    K_ss, K_st, K_tt, S_c, T_c, w = _class_setup(rng)
    K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, S_c, T_c)
    other = torch.clamp(w * 0.5 + 0.25, 0.0, 1.0)
    assert not allclose(w, other)
    first = mixed_matrix(K_s_c, *weighted_blocks(K_st_c, K_t_c, w))
    second = mixed_matrix(K_s_c, *weighted_blocks(K_st_c, K_t_c, other))
    n_s = K_s_c.shape[0]
    assert allclose(first[:n_s, :n_s], K_s_c, atol=TOL)
    assert allclose(first[:n_s, :n_s], second[:n_s, :n_s], atol=TOL)
    assert not allclose(first[n_s:, n_s:], second[n_s:, n_s:])


def test_mixed_matrix_symmetric_psd(rng: Sampler) -> None:
    """r16 Sec. 3, Eq. (27): the mixed matrix is symmetric positive semidefinite.

    It has to be: Eq. (32) trace-normalizes it and Eq. (9) needs a density with
    non-negative spectrum. It is the congruence M K M of the joint bag Gram with
    M = diag(I, D), which preserves positive semidefiniteness.
    """
    K_ss, K_st, K_tt, S_c, T_c, w = _class_setup(rng)
    K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, S_c, T_c)
    K_mix = mixed_matrix(K_s_c, *weighted_blocks(K_st_c, K_t_c, w))
    assert allclose(K_mix, K_mix.transpose(0, 1), atol=TOL)
    assert float(torch.linalg.eigvalsh(K_mix).min()) >= -1e-8


# --------------------------------------------------------------------------
# global_term.py - Section 5, Eqs. (32)-(37)
# --------------------------------------------------------------------------


def test_score_within_conservative_bounds(rng: Sampler) -> None:
    """r16 Sec. 5, Eq. (34): L_c <= J_c <= U_c.

    Each entropy lies in [0, ln n] for its own size; combining the permitted
    extremes separately gives the two conservative ends.
    """
    for n_s, n_t in ((4, 3), (2, 6), (5, 5)):
        K_ss, K_st, K_tt, S_c, T_c, w = _class_setup(rng, n_s, n_t)
        K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, S_c, T_c)
        K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w)
        score = dependency_score(K_s_c, K_t_w, mixed_matrix(K_s_c, K_st_w, K_t_w))
        lower, upper = conservative_bounds(n_s, n_t)
        assert lower - TOL <= float(score) <= upper + TOL


def test_class_global_loss_in_unit_interval(rng: Sampler) -> None:
    """r16 Sec. 5, Eq. (36): the affine map sends U_c to 0, L_c to 1, and stays in [0, 1]."""
    for n_s, n_t in ((3, 4), (6, 2)):
        K_ss, K_st, K_tt, S_c, T_c, w = _class_setup(rng, n_s, n_t)
        K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, S_c, T_c)
        K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w)
        score = dependency_score(K_s_c, K_t_w, mixed_matrix(K_s_c, K_st_w, K_t_w))
        lower, upper = conservative_bounds(n_s, n_t)
        assert -TOL <= float(class_global_loss(score, lower, upper)) <= 1.0 + TOL
        assert close(class_global_loss(upper, lower, upper), 0.0, atol=TOL)
        assert close(class_global_loss(lower, lower, upper), 1.0, atol=TOL)


def test_global_loss_is_convex_combination(rng: Sampler) -> None:
    """r16 Sec. 5, Eq. (37): a convex combination of values in [0, 1] stays in [0, 1].

    It also never leaves the range of the per-class losses it aggregates, and
    it rejects weights that do not form a distribution.
    """
    losses = rng.uniform(0.0, 1.0, size=5)
    weights = rng.dirichlet(5)
    aggregate = global_loss(losses, weights)
    assert float(losses.min()) - TOL <= float(aggregate) <= float(losses.max()) + TOL
    assert close(global_loss(losses), losses.mean(), atol=TOL)
    with pytest.raises(ValueError):
        global_loss(losses, torch.full((5,), 0.5, dtype=DTYPE))


def test_global_loss_zero_when_no_active_class() -> None:
    """r16 Sec. 5, Eq. (37): with C_act empty the global term imposes no penalty."""
    assert float(global_loss([])) == 0.0


# --------------------------------------------------------------------------
# local_term.py - Sections 4-5, Eqs. (28)-(31), (38)
# --------------------------------------------------------------------------


def test_correspondence_simplex_per_class(rng: Sampler) -> None:
    """r16 Sec. 4, Eq. (28): pi_{ji|c} > 0 and sums to one over S_c alone.

    A class with no source bag has no denominator, and the proposal states
    there is no default value that replaces it.
    """
    similarities = rng.uniform(0.0, 1.0, size=8)
    members = torch.tensor([1, 3, 4, 6])
    pi = class_correspondence(similarities, members, tau=0.7)
    assert bool((pi > 0.0).all())
    assert close(pi.sum(), 1.0, atol=TOL)
    with pytest.raises(ValueError):
        class_correspondence(similarities, torch.empty(0, dtype=torch.long), tau=0.7)


def test_correspondence_order_preserving(rng: Sampler) -> None:
    """r16 Sec. 4, Eq. (28): a more similar source subject receives a larger weight.

    The exponential normalization is monotone, so within one class the ordering
    of the kernel values is the ordering of the correspondence weights.
    """
    similarities = torch.tensor([0.1, 0.9, 0.4, 0.6], dtype=DTYPE)
    members = torch.arange(4)
    pi = class_correspondence(similarities, members, tau=0.5)
    assert torch.equal(torch.argsort(pi), torch.argsort(similarities[members]))
    uniform = class_correspondence(torch.full((4,), 0.3, dtype=DTYPE), members, tau=2.0)
    assert allclose(uniform, 0.25, atol=TOL)


def test_total_correspondence_sums_to_one(rng: Sampler) -> None:
    """r16 Sec. 4, Eq. (29): sum_i pi_{ji} = 1 over ALL source bags.

    The class scores fix each class's mass and the within-class distributions
    are already normalized, so no second global normalization is needed. This
    depends on the coverage assumption S_c != empty for every class with mass.
    """
    C, n_source = 3, 9
    source_labels = torch.tensor([c for c in range(C) for _ in range(3)])
    for g in simplex_rows(rng, 12, C):
        pi = total_correspondence(rng.uniform(size=n_source), source_labels, g, tau=0.8)
        assert bool((pi >= 0.0).all())
        assert close(pi.sum(), 1.0, atol=1e-9)


def test_local_distance_bounded_by_two(rng: Sampler) -> None:
    """r16 Sec. 5, Eq. (31): 0 <= d_j^2 < 2, the tight bound the revision derives.

    Non-negative because it is a squared norm. Bounded by two, not four: the
    triangle inequality alone would give four, but kappa_sigma^I > 0 makes every
    inner product in H_I strictly positive, so <Psi(B_j^t), mu_j^s> > 0 and the
    expansion of the squared norm stays strictly below two.
    """
    source = make_bags(rng, 6, 4)
    target = make_bags(rng, 5, 4, shift=1.2)
    K_ss = bag_kernel_matrix(source, source, SIGMA)
    for H_t, w_t in target:
        cross = torch.stack(
            [bag_kernel(H_s, w_s, H_t, w_t, SIGMA) for H_s, w_s in source]
        )
        pi = class_correspondence(cross, torch.arange(6), tau=0.5)
        d2 = local_distance(bag_kernel(H_t, w_t, H_t, w_t, SIGMA), cross, K_ss, pi)
        assert -1e-9 <= float(d2) < 2.0

    # The bound is not decoration: Eq. (38) normalizes by it, so the normalizer
    # must map exactly [0, 2) onto [0, 1). Two far-apart singleton bags drive the
    # kernel to zero and the distance to the bound, where the loss must read one.
    far_apart = local_distance(
        1.0, torch.zeros(1, dtype=DTYPE), torch.ones((1, 1), dtype=DTYPE),
        torch.ones(1, dtype=DTYPE),
    )
    assert close(far_apart, 2.0, atol=TOL), "the adversarial fixture did not reach the bound"
    assert close(normalized_local_distance(far_apart), 1.0, atol=TOL)
    assert close(normalized_local_distance(1.0), 0.5, atol=TOL)


def test_local_distance_zero_iff_reference_matches(rng: Sampler) -> None:
    """r16 Sec. 4, Eq. (31): d_j^2 = 0 exactly when Psi(B_j^t) equals mu_j^s.

    Concentrating the correspondence on a source bag identical to the target
    makes the reference coincide with the target representation; any other
    configuration keeps it strictly positive.
    """
    source = make_bags(rng, 4, 4)
    K_ss = bag_kernel_matrix(source, source, SIGMA)
    H_t, w_t = source[2]
    cross = torch.stack([bag_kernel(H_s, w_s, H_t, w_t, SIGMA) for H_s, w_s in source])
    concentrated = torch.zeros(4, dtype=DTYPE)
    concentrated[2] = 1.0
    self_similarity = bag_kernel(H_t, w_t, H_t, w_t, SIGMA)
    assert close(local_distance(self_similarity, cross, K_ss, concentrated), 0.0, atol=1e-9)
    spread = torch.full((4,), 0.25, dtype=DTYPE)
    assert float(local_distance(self_similarity, cross, K_ss, spread)) > 1e-9


def test_local_loss_in_unit_interval(rng: Sampler) -> None:
    """r16 Sec. 5, Eq. (38): l_loc,j in [0, 1] and the aggregate stays in [0, 1].

    With negligible total confidence the stabilizer drives it to zero: if no
    pseudo-label is reliable, local alignment imposes no penalty.
    """
    d2 = rng.uniform(0.0, 2.0, size=7)
    w = rng.uniform(0.0, 1.0, size=7)
    per_subject = normalized_local_distance(d2)
    assert bool(((per_subject >= 0.0) & (per_subject <= 1.0)).all())
    assert -TOL <= float(local_loss(d2, w)) <= 1.0 + TOL
    assert float(local_loss(d2, torch.zeros(7, dtype=DTYPE))) == 0.0


# --------------------------------------------------------------------------
# objective.py - Sections 3, 5, Eqs. (18), (39)
# --------------------------------------------------------------------------


def test_source_loss_matches_negative_log_likelihood_of_the_observed_class(
    rng: Sampler,
) -> None:
    """r17 Sec. 3, Eq. (18): the observed class alone, averaged per bag, over B_src.

    Three things at once, because they are one claim. The one-hot label selects
    the observed class, so the loss must equal what the observed class scores on
    its own. The term is the per-bag AVERAGE, checked by decomposition: the loss
    of a batch equals the mean of the single-bag losses, which the sum would
    violate. And the whole thing is divided by B_src, checked against the
    unnormalized likelihood computed here rather than against the code's own
    bound, so a wrong divisor cannot agree with itself.
    """
    C, n_bags = 4, 6
    G = simplex_rows(rng, n_bags, C)
    classes = rng.integers(0, C, size=n_bags)
    Y = one_hot(classes, C)
    epsilon = 1e-8

    observed = torch.stack([G[i, int(c)] for i, c in enumerate(classes)])
    unnormalized = -torch.log((observed + epsilon) / (1.0 + epsilon))
    expected = unnormalized.mean() / math.log(1.0 + 1.0 / epsilon)

    batch = source_loss(G, Y)
    assert close(batch, expected, atol=1e-12)

    singles = torch.stack([source_loss(G[i : i + 1], Y[i : i + 1]) for i in range(n_bags)])
    assert close(batch, singles.mean(), atol=1e-12)
    assert not close(batch, singles.sum(), atol=1e-6)


def test_source_loss_non_negative(rng: Sampler) -> None:
    """r17 Sec. 3, Eq. (18): L_src >= 0 always, and exactly 0 at a certain, correct bag.

    The stabilizer is normalized by its own maximum, so the argument of the
    logarithm never exceeds one. Adding it inside the logarithm alone would let
    a perfectly predicted bag score -ln(1 + eps) < 0, which no negative
    log-likelihood may do. Dividing by B_src > 0 preserves the sign.
    """
    C, n_bags = 4, 12
    for _ in range(30):
        G = simplex_rows(rng, n_bags, C)
        Y = one_hot(rng.integers(0, C, size=n_bags), C)
        assert float(source_loss(G, Y)) >= -1e-15

    certain = torch.zeros((1, C), dtype=DTYPE)
    certain[0, 1] = 1.0
    assert float(certain.max()) == 1.0, "the fixture is not a certain prediction"
    assert close(source_loss(certain, certain), 0.0, atol=1e-15)
    # The stabilizer still keeps the other extreme finite.
    wrong = torch.zeros((1, C), dtype=DTYPE)
    wrong[0, 0] = 1.0
    assert bool(torch.isfinite(source_loss(certain, wrong)))


def test_source_loss_in_unit_interval(rng: Sampler) -> None:
    """r17 Sec. 3, Eq. (18): L_src in [0, 1), the upper end a supremum not attained.

    B_src is the exact supremum of a single bag's loss, so dividing by it caps
    the term at one. Two poles, because the bound only means something if it is
    tight and only holds if it is reached from below.

    Strictly positive scores — the only ones the classifier can produce — stay
    strictly under one. The supremum is attained only at g = 0, which no softmax
    reaches, and the fixture that sets it there scores exactly one: that is what
    makes the divisor exact rather than merely safe. A loose bound would leave
    the worst case short of one, so the equality is the test.
    """
    C, n_bags = 5, 16
    for _ in range(30):
        G = simplex_rows(rng, n_bags, C)
        Y = one_hot(rng.integers(0, C, size=n_bags), C)
        assert float(G.min()) > 0.0, "the fixture is not strictly positive"
        value = float(source_loss(G, Y))
        assert 0.0 <= value < 1.0

    # The other pole: all the mass off the observed class is the supremum itself.
    worst = torch.zeros((1, C), dtype=DTYPE)
    worst[0, 0] = 1.0
    label = torch.zeros((1, C), dtype=DTYPE)
    label[0, 1] = 1.0
    assert float(worst[0, 1]) == 0.0, "the fixture does not place zero on the observed class"
    assert close(source_loss(worst, label), 1.0, atol=1e-12)

    # And the bound is what eps fixes, not a constant: a larger stabilizer is a
    # smaller ceiling, so the same bag costs more of it.
    assert source_bound(1e-6) < source_bound(1e-8)
    assert float(source_loss(G, Y, epsilon=1e-6)) > float(source_loss(G, Y, epsilon=1e-8))


def test_objective_reduces_to_source_only_when_coefficients_vanish() -> None:
    """r16 Sec. 5, Eq. (39): lambda_glob = lambda_loc = 0 leaves L_src alone.

    Source-only supervised training is contained as a particular case; a
    negative coefficient is refused because it would reward misalignment.
    """
    supervised, glob, loc = 1.7, 0.4, 0.6
    assert close(total_objective(supervised, glob, loc, 0.0, 0.0), supervised, atol=TOL)
    assert close(total_objective(supervised, glob, loc, 1.0, 0.0), supervised + glob, atol=TOL)
    assert close(total_objective(supervised, glob, loc, 0.0, 1.0), supervised + loc, atol=TOL)
    with pytest.raises(ValueError):
        total_objective(supervised, glob, loc, -0.1, 0.0)


def test_objective_monotone_in_each_adaptation_term() -> None:
    """r16 Sec. 5, Eq. (39): with a positive coefficient the objective grows with its term.

    Both coefficients are non-negative precisely so that reducing a discrepancy
    can never be penalized.
    """
    base = float(total_objective(1.0, 0.2, 0.2, 0.5, 0.5))
    assert float(total_objective(1.0, 0.9, 0.2, 0.5, 0.5)) > base
    assert float(total_objective(1.0, 0.2, 0.9, 0.5, 0.5)) > base
    assert close(total_objective(1.0, 0.9, 0.2, 0.0, 0.5), base - 0.5 * 0.2 + 0.0 * 0.9)
