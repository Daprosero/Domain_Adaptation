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
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import Sampler, allclose, close, make_bag, make_bags, one_hot, simplex_rows

from sweep import SWEEP_SIZE

from MIL_CREDA import DTYPE
from MIL_CREDA.attention import (
    bag_embedding,
    bag_weights,
    consensus_component,
    effective_bag_size,
    project_onto_l1_ball,
    relevance_component,
    relevance_logits,
    mean_consensus,
)
from MIL_CREDA.bag_kernel import bag_gram, bag_kernel, bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import bag_confidence, confidences, pseudolabel
from MIL_CREDA.global_term import (
    balanced_dependency_score,
    balanced_mixture,

    class_global_loss_cosh,

    entropy_cosh_bounds,
    GLOBAL_LOSS_SPAN,
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
from MIL_CREDA.renyi import (
    information_potential,
    joint_normalized,
    matrix_mutual_information,
    quadratic_entropy,
    renyi_entropy,
    trace_normalize,
)

TOL = 1e-10
SIGMA = 1.3


# --------------------------------------------------------------------------
# kernels.py - Section 1, Eqs. (2), (14)
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


def _numerically_integrate(f, low: float, high: float, points: int) -> float:
    """Trapezoidal quadrature of `f` over one axis, in plain torch.

    Used only to check Eq. (6) against a route that never assumes the
    closed-form Gaussian-convolution identity it exists to establish: a
    numerical integral is blind to whether the integrand happens to be
    Gaussian.
    """
    grid = torch.linspace(low, high, points, dtype=DTYPE)
    values = torch.stack([f(x) for x in grid])
    return float(torch.trapezoid(values, grid))


def test_gaussian_convolution_identity_eq6(rng: Sampler) -> None:
    """Sec. 2, Eq. (6): the un-numbered constant right before it,
    Z_{d,sigma} = (pi sigma^2)^{d/2}, and

        integral over R^d of kappa_sigma(x, x_i) kappa_sigma(x, x_j) dx
        = Z_{d,sigma} kappa_{sqrt(2) sigma}(x_i, x_j).

    Checked by NUMERICAL integration, in d=1 -- where the integral is a
    single quadrature this suite can actually perform -- rather than by
    trusting the closed form the identity itself asserts. `Eqs. (7)-(8)`
    (`information_potential`) rely on this identity to replace an integral
    with pairwise kernel evaluations; this is what establishes it rather
    than assuming it.
    """
    sigma = float(rng.uniform(0.4, 1.5))
    for _ in range(8):
        x_i = float(rng.uniform(-2.0, 2.0))
        x_j = float(rng.uniform(-2.0, 2.0))

        def integrand(x: torch.Tensor, x_i=x_i, x_j=x_j) -> torch.Tensor:
            k_i = torch.exp(-((x - x_i) ** 2) / (2.0 * sigma**2))
            k_j = torch.exp(-((x - x_j) ** 2) / (2.0 * sigma**2))
            return k_i * k_j

        # +/- 10 sigma from either center is far enough past the tails
        # (exp(-50) ~ 2e-22) that truncating the integral there is not what
        # limits the agreement below.
        low = min(x_i, x_j) - 10.0 * sigma
        high = max(x_i, x_j) + 10.0 * sigma
        numeric = _numerically_integrate(integrand, low, high, 20001)

        Z = (math.pi * sigma**2) ** 0.5
        closed_form = Z * float(gaussian_kernel(
            torch.tensor([[x_i]], dtype=DTYPE), torch.tensor([[x_j]], dtype=DTYPE),
            math.sqrt(2.0) * sigma,
        )[0, 0])
        assert numeric == pytest.approx(closed_form, rel=1e-4, abs=1e-6)


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


def test_information_potential_matches_eqs_7_8_definition(rng: Sampler) -> None:
    """Sec. 2, Eqs. (7)-(8): V_2(X) = Z_{d,sigma}/N^2 * sum_i sum_j K_ij =
    Z_{d,sigma}/N^2 * 1^T K 1, with K_ij = kappa_{sqrt(2) sigma}(x_i, x_j).

    Recomputed independently with an explicit double loop over the sample
    pairs and Python's own `math.exp`/`math.pi` -- never `torch.sum` or
    `gaussian_kernel` -- so a wrong power of pi, a wrong denominator (N vs
    N^2, or N(N-1) excluding self-pairs) or a dropped Z factor in
    `information_potential` disagrees with this rather than sharing the bug.
    """
    for _ in range(SWEEP_SIZE):
        n = int(rng.integers(2, 9))
        d = int(rng.integers(1, 4))
        sigma = float(rng.uniform(0.3, 2.0))
        X = rng.normal(size=(n, d))
        K = gaussian_kernel(X, X, math.sqrt(2.0) * sigma)

        produced = float(information_potential(K, d, sigma))

        Z = (math.pi * sigma**2) ** (d / 2.0)
        total = 0.0
        rows = X.tolist()
        for xi in rows:
            for xj in rows:
                squared = sum((a - b) ** 2 for a, b in zip(xi, xj))
                total += math.exp(-squared / (2.0 * (math.sqrt(2.0) * sigma) ** 2))
        expected = Z * total / (n**2)
        assert produced == pytest.approx(expected, rel=1e-9)


def test_joint_normalized_matches_eq11_definition(rng: Sampler) -> None:
    """Sec. 2, Eq. (11)'s joint matrix: K_{X,Y} = K_X . K_Y (Hadamard),
    trace-normalized -- `joint_normalized` against an independent
    element-by-element recomputation, never `torch.mul` or the module's own
    `trace_normalize` called on anything but the final comparison target.
    """
    for _ in range(SWEEP_SIZE):
        n = int(rng.integers(2, 8))
        K_x = gram(rng.normal(size=(n, 3)), SIGMA)
        K_y = gram(rng.normal(size=(n, 4)), SIGMA * 1.7)

        produced = joint_normalized(K_x, K_y)

        hadamard = torch.tensor(
            [[float(K_x[i, j]) * float(K_y[i, j]) for j in range(n)]
             for i in range(n)],
            dtype=DTYPE,
        )
        trace = sum(float(hadamard[i, i]) for i in range(n))
        expected = hadamard / trace
        assert allclose(produced, expected, atol=1e-9)


def test_matrix_mutual_information_matches_eq12_definition(rng: Sampler) -> None:
    """Sec. 2, Eq. (12): I_alpha(K_X; K_Y) = H(K_X) + H(K_Y) - H(K_X, K_Y),
    each term from its OWN trace-normalized matrix -- checked as the sum of
    three independently-called entropy evaluations, so a wrong sign or a
    term computed from the wrong matrix disagrees with `renyi_entropy`
    called by hand at each of the three sites.

    And symmetric in K_X/K_Y, since the Hadamard product Eq. (11) builds the
    joint matrix from commutes -- a property of the definition, not assumed.
    """
    alpha = 2.0
    for _ in range(SWEEP_SIZE):
        n = int(rng.integers(2, 8))
        K_x = gram(rng.normal(size=(n, 3)), SIGMA)
        K_y = gram(rng.normal(size=(n, 4)), SIGMA * 0.6)

        produced = matrix_mutual_information(K_x, K_y, alpha)
        expected = (
            renyi_entropy(trace_normalize(K_x), alpha)
            + renyi_entropy(trace_normalize(K_y), alpha)
            - renyi_entropy(joint_normalized(K_x, K_y), alpha)
        )
        assert close(produced, expected, atol=1e-9)

        swapped = matrix_mutual_information(K_y, K_x, alpha)
        assert close(produced, swapped, atol=1e-9)


# --------------------------------------------------------------------------
# attention.py - Section 3, Eqs. (15), (16), (19)
# --------------------------------------------------------------------------

#: `attention.py`'s own sweep base, disjoint from `sweep.SWEEP_BASE`: this
#: file's claims are about the attention pipeline alone (m, d, sigma, gamma,
#: tau_att, the scale of v_R), never about the cross-bag machinery `sweep.py`
#: draws its own 200 configurations for.
_ATTENTION_SWEEP_SEED = 20260916 * 10000


def _attention_configurations(count: int = SWEEP_SIZE):
    """`count` independent (H, V_R, b_R, v_R, gamma, sigma, tau_att) draws.

    Sweeps every axis the r21 attention claims quantify over: bag size m in
    [1, 12], embedding dimension d in [2, 8], the instance-kernel bandwidth
    sigma in [0.1, 4], the consensus weight gamma in [0, 3], the temperature
    tau_att in [0.05, 5], and the scale of the raw (unconstrained) relevance
    parameter v_R from 1e-3 to 50, mixed sign -- so the l1 reparametrization
    is exercised both inside and outside the unit ball.
    """
    for index in range(count):
        s = Sampler(_ATTENTION_SWEEP_SEED + index)
        m = int(s.integers(1, 13))
        d = int(s.integers(2, 9))
        hidden = int(s.integers(2, 7))
        H = s.normal(size=(m, d))
        V_R = s.normal(size=(hidden, d))
        b_R = s.normal(size=hidden)
        scale = float(s.uniform(1e-3, 50.0))
        v_R = s.normal(size=hidden) * scale
        gamma = float(s.uniform(0.0, 3.0))
        sigma = float(s.uniform(0.1, 4.0))
        tau_att = float(s.uniform(0.05, 5.0))
        yield {
            "H": H, "V_R": V_R, "b_R": b_R, "v_R": v_R,
            "gamma": gamma, "sigma": sigma, "tau_att": tau_att, "m": m,
        }


def _independent_consensus(H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """(1/(m-1)) sum_{a' != a} kappa_sigma^I(h_a, h_a'), Eq. (14)'s kernel
    evaluated directly with `torch.cdist` and `torch.exp` -- never through
    `MIL_CREDA.kernels.gaussian_kernel` or `attention.consensus_component` --
    EXCLUDING the self term, which is r34's own definition: the self
    evaluation kappa^I(h, h) = 1 added 1/m to every consensus of the bag and
    shrank their differences by (m-1)/m, so gamma's effective weight depended
    on the instance count. A single-instance bag has no other instance to
    average over, so the term is omitted and this returns zero, matching the
    module's own omission.

    Defect (b): comparing the module's own output to a second call of the
    module's own `consensus_component` is a self-comparison, and a mutant
    that scaled sigma by a constant factor survives it, because both sides
    of the "check" apply the identical wrong scale. This recomputes the
    kernel from its definition instead, so a wrong bandwidth in the module
    under test disagrees with an independent implementation of the same
    equation.
    """
    H = torch.as_tensor(H, dtype=torch.float64)
    m = H.shape[0]
    squared = torch.cdist(H, H) ** 2
    K = torch.exp(-squared / (2.0 * float(sigma) ** 2))
    if m == 1:
        return torch.zeros(1, dtype=K.dtype)
    K = K * (1.0 - torch.eye(m, dtype=K.dtype))
    return K.sum(dim=1) / (m - 1)


def test_bag_weights_on_simplex(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (16): beta > 0 and sum over the bag equals one.

    Normalization is strictly within one bag; the weights express relevance
    among the instances of a single subject. Confirmed over the sweep: holds
    for every m, d, sigma, gamma, tau_att and v_R scale it draws.
    """
    dimension, hidden = 4, 6
    H = rng.normal(size=(5, dimension))
    logits = relevance_logits(
        H, rng.normal(size=(hidden, dimension)), rng.normal(size=hidden),
        rng.normal(size=hidden), 0.7, SIGMA,
    )
    beta = bag_weights(logits, 1.3)
    assert beta.shape == (5,)
    assert bool((beta > 0.0).all())
    assert close(beta.sum(), 1.0, atol=TOL)

    for cfg in _attention_configurations():
        logits = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        beta = bag_weights(logits, cfg["tau_att"])
        assert bool((beta > 0.0).all())
        assert close(beta.sum(), 1.0, atol=1e-8)


def test_bag_embedding_permutation_invariant(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (19): z is invariant under permutations of the bag.

    Each weight is paired with its own embedding, so reordering the instances
    together with their weights cannot move the representation.
    """
    H, beta = make_bag(rng, 6, 4)
    order = rng.permutation(6)
    assert allclose(bag_embedding(H, beta), bag_embedding(H[order], beta[order]), atol=TOL)


def test_relevance_logit_matches_eq15_definition(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (15): nu_a = R_phi(h_a) + gamma * consensus_a, exactly --
    the composition itself, not only its gamma = 0 special case. A mutant that
    dropped the consensus term (returning R_phi alone for every gamma) would
    pass every OTHER test in this section undetected; this is the one that
    catches it, by recomputing both sides independently and comparing.

    Defect (b): the consensus half used to be recomputed by calling
    `consensus_component` a second time -- the same function `relevance_logits`
    calls internally -- so a mutation to THAT function's own bandwidth (3*sigma,
    sqrt(2)*sigma, dropping the self term) was invisible here: both "sides" of
    the comparison shared the identical bug. `_independent_consensus` recomputes
    Eq. (14)'s kernel from `torch.cdist`/`torch.exp` directly and never calls
    `consensus_component` or `MIL_CREDA.kernels.gaussian_kernel` at all.
    """
    gamma_positive = visible = invisible = 0
    for cfg in _attention_configurations():
        logits = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        expected = (
            relevance_component(cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"])
            + cfg["gamma"] * _independent_consensus(cfg["H"], cfg["sigma"])
        )
        assert allclose(logits, expected, atol=1e-9)
        if cfg["gamma"] <= 0.01:
            continue
        gamma_positive += 1
        component = relevance_component(cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"])
        # The precondition, measured on the REFERENCE rather than assumed: at
        # a small bandwidth r34's consensus is strictly positive and can still
        # be far below the logits' own magnitude -- the sweep's smallest is
        # 2.27e-100 -- so `gamma * consensus` moves no float64 bit and the
        # logit genuinely EQUALS the relevance. r33's form could not land
        # there, being floored at 1/m. Asserting a difference on those
        # configurations would assert something false; measured over this
        # sweep, 32 of the 199 with gamma > 0.01 are of that kind.
        if allclose(expected, component, atol=1e-9):
            invisible += 1
            continue
        visible += 1
        assert not allclose(logits, component, atol=1e-9), (
            "the consensus term made no difference at gamma > 0 where the "
            "independent reference says it makes one"
        )

    assert gamma_positive == visible + invisible
    assert visible > 0, (
        "no configuration with gamma > 0 had a numerically visible consensus "
        "term, so this sweep cannot catch a mutant that dropped it"
    )


def test_consensus_component_reads_the_declared_bandwidth_not_a_scaled_one(
        rng: Sampler) -> None:
    """Defect (b), stated as a direct mutation check rather than left to a
    manual audit pass: `consensus_component` at the bandwidth `sigma` this
    test independently computes must equal the SAME independent computation
    at that same `sigma` -- `assert allclose(produced, at_sigma, atol=1e-9)`,
    the one comparison in this loop that actually establishes
    `consensus_component` read the declared bandwidth and not a mutated one.

    What is dropped is the 90%-of-configurations rate threshold the mutation
    check used to run under: that rate measured how often
    `_independent_consensus` -- the REFERENCE helper -- gives a numerically
    different answer at `sigma` than at a scaled bandwidth, which is a fact
    about the reference implementation's own geometry (how much a
    3x/sqrt(2)x rescaling moves ITS output) and proves nothing about the
    module under test.

    `distinguishable_3sigma`/`distinguishable_sqrt2sigma` replace it with a
    precondition rather than a second mutation check: they count how many
    configurations the reference itself can tell `sigma` from `3*sigma` or
    `sqrt(2)*sigma` on, and the two `assert ... > 0` below the loop refuse a
    sweep that drew none. An inner `assert not allclose(produced, at_Xsigma)`
    inside that same `if` used to sit beside the counter; it is gone because
    it could never fire on its own -- once `produced == at_sigma` (atol=1e-9)
    and the `if` has already established `at_sigma != at_Xsigma` (atol=1e-6),
    `produced != at_Xsigma` follows for free and catches no mutation the
    first assertion had not already caught.
    """
    import math

    # The inner `assert not allclose(produced, at_Xsigma)` this loop used to
    # carry, gated behind `if not allclose(at_sigma, at_Xsigma)`, could never
    # fire on its own: the unconditional `assert allclose(produced, at_sigma,
    # atol=1e-9)` two lines above already pins `produced` to `at_sigma` at a
    # tolerance ten times tighter, so once the `if` establishes that
    # `at_sigma` and `at_Xsigma` disagree at atol=1e-6, `produced` disagreeing
    # with `at_Xsigma` follows from the first assertion and proves nothing new
    # -- any mutation it could catch was already caught two lines above. What
    # is kept is the independent comparison it was paired with: the
    # `distinguishable_*` counters below prove the reference implementation
    # itself can tell `sigma` from a scaled bandwidth on this sweep, which is
    # the precondition the whole check needs to mean anything.
    distinguishable_3sigma = distinguishable_sqrt2sigma = 0
    for cfg in _attention_configurations():
        if cfg["m"] < 2:
            continue  # r34 omits the term there, so it is 0 regardless of sigma
        produced = consensus_component(cfg["H"], cfg["sigma"])
        at_sigma = _independent_consensus(cfg["H"], cfg["sigma"])
        assert allclose(produced, at_sigma, atol=1e-9)

        at_3sigma = _independent_consensus(cfg["H"], 3.0 * cfg["sigma"])
        at_sqrt2sigma = _independent_consensus(cfg["H"], math.sqrt(2.0) * cfg["sigma"])

        if not allclose(at_sigma, at_3sigma, atol=1e-6):
            distinguishable_3sigma += 1
        if not allclose(at_sigma, at_sqrt2sigma, atol=1e-6):
            distinguishable_sqrt2sigma += 1

    assert distinguishable_3sigma > 0, (
        "the sweep drew no configuration where a 3*sigma bandwidth is "
        "numerically distinguishable from sigma"
    )
    assert distinguishable_sqrt2sigma > 0, (
        "the sweep drew no configuration where a sqrt(2)*sigma bandwidth is "
        "numerically distinguishable from sigma"
    )


def test_project_onto_l1_ball_imposes_the_max_norm_constraint(rng: Sampler) -> None:
    """r31 Eq. (15): "tras cada paso de optimizacion, si ||v_R||_1 > 1, se
    reescala v_R <- v_R / ||v_R||_1".

    The projection is the whole mechanism by which Eq. (15)'s bound holds under
    r31, so it is tested directly: it must land every infeasible parameter
    exactly on the ball's edge, leave every feasible one untouched, and report
    which of the two it did.
    """
    hidden = 8
    for scale in (1e6, -1e6, 3.0, -2.5):
        v_R = torch.full((hidden,), float(scale), dtype=DTYPE)
        assert float(torch.linalg.vector_norm(v_R, ord=1)) > 1.0
        assert project_onto_l1_ball(v_R) is True
        assert float(torch.linalg.vector_norm(v_R, ord=1)) == pytest.approx(
            1.0, abs=1e-12)

    for scale in (1e-9, -1e-9, 0.0):
        v_R = torch.full((hidden,), float(scale), dtype=DTYPE)
        before = v_R.clone()
        assert project_onto_l1_ball(v_R) is False
        assert torch.equal(v_R, before), "a feasible parameter must not move"

    # Random infeasible draws, and the direction has to survive the rescale.
    for _ in range(16):
        v_R = rng.normal(size=hidden) * 1e5
        direction = v_R / torch.linalg.vector_norm(v_R, ord=2)
        assert project_onto_l1_ball(v_R) is True
        assert float(torch.linalg.vector_norm(v_R, ord=1)) == pytest.approx(
            1.0, abs=1e-12)
        assert allclose(v_R / torch.linalg.vector_norm(v_R, ord=2),
                        direction, atol=TOL)


def test_relevance_component_bounded_by_the_l1_constraint(rng: Sampler) -> None:
    """r31 Eq. (15): |R_phi(h)| <= ||v_R||_1 <= 1 for every FEASIBLE v_R.

    "Como cada coordenada de tanh pertenece a [-1,1], la desigualdad de Hoelder
    da |R_phi(h)| <= ||v_R||_1 <= 1 para todo h."

    **Both poles, because r31 moved the reason the bound holds.** Through r30
    the selector divided by `max(1, ||v_R||_1)` inside the forward pass, so the
    bound held for ANY parameter at any scale and the earlier version of this
    test asserted exactly that. r31 constrains the parameter instead, so the
    bound is a statement about feasible parameters only -- and a test that
    still demanded it for an unconstrained `v_R` would be asserting r30.

    So the test asserts the bound on the feasible side AND asserts that it
    genuinely fails on the infeasible side. Without the second half it could
    not distinguish r31's raw selector from r30's dividing one, since both
    satisfy the first half.
    """
    dimension, hidden = 5, 8
    H = rng.normal(size=(9, dimension))
    V_R = rng.normal(size=(hidden, dimension))
    b_R = rng.normal(size=hidden)

    # The feasible pole: projected parameters respect the bound.
    for scale in (1e6, -1e6, 1e-9, -1e-9, 3.0):
        v_R = torch.full((hidden,), float(scale), dtype=DTYPE)
        project_onto_l1_ball(v_R)
        assert float(torch.linalg.vector_norm(v_R, ord=1)) <= 1.0 + TOL
        component = relevance_component(H, V_R, b_R, v_R)
        assert bool((component.abs() <= 1.0 + TOL).all())

    random_huge = rng.normal(size=hidden) * 1e5
    project_onto_l1_ball(random_huge)
    component = relevance_component(H, V_R, b_R, random_huge)
    assert bool((component.abs() <= 1.0 + TOL).all())

    zero = torch.zeros(hidden, dtype=DTYPE)
    assert allclose(relevance_component(H, V_R, b_R, zero),
                     torch.zeros(9, dtype=DTYPE), atol=TOL)

    # The infeasible pole: WITHOUT the projection the bound is breached, which
    # is what makes the projection the mechanism rather than a formality. The
    # hidden layer is driven into saturation so `tanh` reaches its own ends and
    # Hoelder's inequality is close to tight.
    saturating = rng.normal(size=(hidden, dimension)) * 50.0
    infeasible = torch.full((hidden,), 2.0, dtype=DTYPE)
    assert float(torch.linalg.vector_norm(infeasible, ord=1)) > 1.0
    breached = relevance_component(H, saturating, b_R, infeasible).abs().max()
    assert float(breached) > 1.0 + TOL, (
        "an unprojected v_R no longer breaches the bound, so this test can no "
        f"longer tell r31's raw selector from r30's dividing one: {float(breached)}")

    for cfg in _attention_configurations():
        v_R = cfg["v_R"].clone()
        project_onto_l1_ball(v_R)
        component = relevance_component(cfg["H"], cfg["V_R"], cfg["b_R"], v_R)
        assert bool((component.abs() <= 1.0 + TOL).all())


def test_bag_weights_permutation_equivariant(rng: Sampler) -> None:
    """r21 Sec. 3: 'Los pesos son equivariantes frente a permutaciones de las
    instancias' -- the learned component acts instance by instance and the
    consensus average is symmetric, so permuting the instances permutes the
    logits, and therefore the weights, the same way.
    """
    for cfg in _attention_configurations():
        if cfg["m"] < 2:
            continue
        order = rng.permutation(cfg["m"])
        logits = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        logits_perm = relevance_logits(
            cfg["H"][order], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        assert allclose(logits_perm, logits[order], atol=1e-8)
        beta = bag_weights(logits, cfg["tau_att"])
        beta_perm = bag_weights(logits_perm, cfg["tau_att"])
        assert allclose(beta_perm, beta[order], atol=1e-8)


def test_bag_weights_singleton_is_one(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (16): 'una bolsa con una sola instancia le asigna peso
    uno.' m = 1: the only weight is the softmax of a single value, which is
    exactly one for any logit and any temperature.
    """
    dimension, hidden = 3, 5
    for _ in range(SWEEP_SIZE):
        H = rng.normal(size=(1, dimension))
        V_R = rng.normal(size=(hidden, dimension))
        b_R = rng.normal(size=hidden)
        v_R = rng.normal(size=hidden) * float(rng.uniform(1e-3, 20.0))
        gamma = float(rng.uniform(0.0, 3.0))
        sigma = float(rng.uniform(0.1, 4.0))
        tau_att = float(rng.uniform(0.05, 5.0))
        logits = relevance_logits(H, V_R, b_R, v_R, gamma, sigma)
        beta = bag_weights(logits, tau_att)
        assert close(beta[0], 1.0, atol=TOL)


def test_attention_logits_depend_only_on_own_bag(rng: Sampler) -> None:
    """r21 Sec. 3: 'el logit es funcion de la instancia y de su bolsa, pero
    nunca de otras bolsas' -- the within-bag-only hypothesis. Every argument
    here is the bag's own and nothing carries state between calls, so
    computing an unrelated bag in between cannot move the result.
    """
    for cfg in _attention_configurations():
        before = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        distractor = rng.normal(size=(int(rng.integers(1, 9)), cfg["H"].shape[1]))
        relevance_logits(
            distractor, cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        after = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        assert allclose(before, after, atol=TOL)


def test_consensus_component_in_unit_interval(rng: Sampler) -> None:
    """r34 Sec. 3 (consensus term preceding Eq. 15): consensus in (0, 1].

    kappa^I is strictly positive and at most one everywhere (Eq. 14), so an
    average of its off-diagonal entries cannot reach zero and cannot exceed
    one. **The bound no longer rests on the self term**, which is exactly what
    r34 bought: under r33 the floor was 1/m and the self evaluation was what
    held it up, so the interval the logit's two components were declared
    comparable on depended on the bag's instance count. Here it does not --
    which is why a single-instance bag, whose term is OMITTED and therefore
    zero, is the one configuration this interval does not describe and is
    skipped.

    **The strict lower end is what found a defect, so it is asserted strictly.**
    r33's form was floored at 1/m by its own self term, so nothing about the
    bottom of this range was ever exercised; r34's floor is zero, and the first
    implementation of it computed `K.sum(dim=1) - K.diagonal()`. That is the
    same number in real arithmetic and a different one in float64: the row sum
    is at least the diagonal's 1.0, so an off-diagonal entry below the machine
    epsilon relative to 1.0 leaves the sum unchanged and the subtraction
    returns exactly 0.0. Measured over this sweep's own 183 configurations with
    two or more instances, that form read a spurious zero on 14 of them -- one
    with a largest off-diagonal entry of 3.46e-18, positive and lost. Masking
    the diagonal before the sum leaves 0 of 183, and the smallest consensus the
    sweep produces is 2.27e-100: strictly positive, as Eq. (14) says, and far
    from the kernel's own float64 underflow floor at `d^2/(2 sigma^2) > 744.44`.

    The cancellation is asserted here as well as the interval, because a sweep
    that only checked `0 <= c <= 1` passed the defective form.
    """
    checked = cancelled = 0
    for cfg in _attention_configurations():
        if cfg["m"] < 2:
            continue  # the term is omitted there; see the docstring
        consensus = consensus_component(cfg["H"], cfg["sigma"])
        assert bool((consensus > 0.0).all()), (
            f"a consensus reached exactly zero at m={cfg['m']}, "
            f"sigma={float(cfg['sigma']):.4f}: kappa^I is strictly positive, "
            f"so either the kernel underflowed or the diagonal was subtracted "
            f"after the sum instead of masked before it"
        )
        assert bool((consensus <= 1.0 + TOL).all())

        # The regression: the form this replaced, on the same input.
        K = gaussian_kernel(cfg["H"], cfg["H"], cfg["sigma"])
        subtracted = (K.sum(dim=1) - K.diagonal()) / (cfg["m"] - 1)
        if bool((subtracted == 0.0).any()):
            cancelled += 1
        checked += 1

    assert checked > 0, "the sweep drew no bag with two or more instances"
    assert cancelled > 0, (
        "the subtract-after-sum form produced no spurious zero on this sweep, "
        "so this configuration set no longer exercises the cancellation the "
        "masked implementation exists to avoid"
    )


def test_consensus_component_bandwidth_limits(rng: Sampler) -> None:
    """r34 Sec. 3: 'si sigma -> 0, el kernel se anula entre embeddings
    distintos y los consensos de la bolsa tienden a cero; si sigma ->
    infinito, el kernel tiende a uno y los consensos tambien.'

    **The lower limit is r34's change and not a restatement.** r33 read 1/m
    here, because at sigma -> 0 only the self terms survived and there were m
    of them over m; with the self term gone nothing survives, so the limit is
    zero for every bag size. A test still asserting 1/m would pass on a
    1-instance bag and on nothing else.

    `1e-4`, not an even smaller bandwidth: `gaussian_kernel`'s own squared
    distance is `sum(x^2) - 2 x.y + sum(y^2)`, exact in real arithmetic but a
    hair off zero in float64 on the diagonal; dividing that hair by `2 *
    sigma^2` is what the limit is about, and past `~1e-5` the division
    amplifies the float64 cancellation error itself rather than measuring the
    claim. `1e-4` is small enough to saturate the limit and large enough to
    stay clear of that floor.
    """
    for _ in range(SWEEP_SIZE):
        m = int(rng.integers(2, 10))
        d = int(rng.integers(2, 6))
        H = rng.normal(size=(m, d))
        tiny = consensus_component(H, 1e-4)
        assert allclose(tiny, torch.zeros(m, dtype=DTYPE), atol=1e-6)
        huge = consensus_component(H, 1e6)
        assert allclose(huge, torch.ones(m, dtype=DTYPE), atol=1e-6)


def test_consensus_component_excludes_the_instance_itself(rng: Sampler) -> None:
    """r34 Sec. 3, Eq. (15): the average runs over a' != a, divided by (m-1).

    Asserted as a VALUE against an independently built pair, not as a property
    that both forms happen to share: the leave-one-out average must equal
    `_independent_consensus` and must DIFFER from `_with_self_term_consensus`,
    which is r33's form. Without the second half a module that never changed
    would pass -- the two forms are related by
    `c_r33 = (1 + (m-1) c_r34) / m`, so they agree only where that fixed point
    sits, at c = 1.
    """
    differed = 0
    for _ in range(SWEEP_SIZE):
        m = int(rng.integers(2, 10))
        d = int(rng.integers(2, 6))
        H = rng.normal(size=(m, d))
        sigma = float(rng.uniform(0.3, 2.0))

        produced = consensus_component(H, sigma)
        assert allclose(produced, _independent_consensus(H, sigma), atol=1e-9)

        # The identity that ties the two revisions' forms together, checked
        # rather than asserted in prose.
        r33_form = _with_self_term_consensus(H, sigma)
        assert allclose(r33_form, (1.0 + (m - 1) * produced) / m, atol=1e-9)

        if not allclose(produced, r33_form, atol=1e-6):
            differed += 1

    assert differed == SWEEP_SIZE, (
        f"the two forms agreed on {SWEEP_SIZE - differed} of {SWEEP_SIZE} "
        f"configurations, where they coincide only at consensus 1: this sweep "
        f"cannot tell r34's definition from r33's"
    )


def test_consensus_component_omits_the_term_for_a_singleton_bag() -> None:
    """r34 Sec. 3: 'en una bolsa con una sola instancia el termino se omite, lo
    que no altera nada, porque esa bolsa recibe peso uno (Ec. 16).'

    Both halves are asserted, because the first alone would let the omission
    be any constant: the term is zero, AND the weight Eq. (16) gives that
    instance is one whatever gamma and sigma are. A division by (m-1) = 0
    would surface here as a nan rather than as a wrong number.
    """
    H = torch.tensor([[0.3, -1.2, 0.7]], dtype=DTYPE)
    for sigma in (0.05, 1.0, 50.0):
        omitted = consensus_component(H, sigma)
        assert omitted.shape == (1,)
        assert bool(torch.isfinite(omitted).all()), "a (m-1) = 0 division leaked"
        assert close(omitted[0], 0.0, atol=TOL)

    V_R = torch.full((2, 3), 0.4, dtype=DTYPE)
    b_R = torch.zeros(2, dtype=DTYPE)
    v_R = torch.tensor([0.5, 0.5], dtype=DTYPE)
    for gamma in (0.0, 0.7, 5.0):
        logits = relevance_logits(H, V_R, b_R, v_R, gamma, SIGMA)
        assert close(bag_weights(logits, 1.0)[0], 1.0, atol=TOL)


def test_relevance_logit_reduces_to_relevance_at_gamma_zero(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (15): 'Con gamma = 0 el logit se reduce a la componente
    aprendida' -- exactly, whatever sigma is, since the consensus term is
    multiplied by gamma.
    """
    for cfg in _attention_configurations():
        logits = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], 0.0, cfg["sigma"]
        )
        component = relevance_component(cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"])
        assert allclose(logits, component, atol=1e-12)


def test_attention_logit_ratio_bounded_by_temperature(rng: Sampler) -> None:
    """r21 Sec. 3: 'dos logits de una misma bolsa difieren a lo sumo en
    2 + gamma, de modo que la razon entre dos pesos de la bolsa es a lo sumo
    exp((2 + gamma) / tau_att).'

    **The `2` is `|R_phi| <= 1` twice, so under r31 it needs a FEASIBLE `v_R`.**
    Through r30 the selector divided by `max(1, ||v_R||_1)` inside the forward
    pass, so the spread bound held for any parameter and this test could draw
    `v_R` freely. r31 constrains the parameter instead, so an unprojected draw
    makes `|R_phi|` exceed 1 and the bound is genuinely false: measured without
    the projection, the ratio reaches 7.06 against a bound of 4.27.

    So the fixture's `v_R` is projected first, which is the state the training
    loop maintains it in. The sweep still ends by confirming the bound really
    does bind -- an unprojected draw must breach it -- so this cannot pass by
    having become vacuous.
    """
    breached_unprojected = 0
    for cfg in _attention_configurations():
        if cfg["m"] < 2:
            continue
        v_R = cfg["v_R"].clone()
        project_onto_l1_ball(v_R)
        assert float(torch.linalg.vector_norm(v_R, ord=1)) <= 1.0 + TOL

        logits = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], v_R, cfg["gamma"], cfg["sigma"]
        )
        spread = float(logits.max() - logits.min())
        assert spread <= 2.0 + cfg["gamma"] + 1e-8
        beta = bag_weights(logits, cfg["tau_att"])
        ratio = float(beta.max() / beta.min())
        bound = math.exp((2.0 + cfg["gamma"]) / cfg["tau_att"])
        assert ratio <= bound * (1.0 + 1e-6) + 1e-9

        raw_logits = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"],
            cfg["sigma"]
        )
        if float(raw_logits.max() - raw_logits.min()) > 2.0 + cfg["gamma"] + 1e-8:
            breached_unprojected += 1

    assert breached_unprojected > 0, (
        "no configuration breaches the spread bound without the projection, so "
        "this test can no longer tell r31's constrained parameter from r30's "
        "reparametrized forward pass")


def test_relevance_logit_reduces_to_the_raw_relevance_at_neutral_hyperparameters() -> None:
    """At gamma = 0 and tau_att = 1 (`config.ATTENTION_GAMMA`,
    `config.ATTENTION_TEMPERATURE`'s declared neutral) r31's Eqs. (15)-(16)
    reduce to a plain softmax of `v_R^T tanh(V_R h + b_R)` -- the learned
    relevance alone, with `v_R` read RAW.

    **This reduction is r31's and the opposite of r30's.** r30 reparametrized
    inside the forward pass, so the reduction was to `R_phi` with an
    l1-NORMALIZED `v_R`, and the previous version of this test asserted exactly
    that -- including a closing assertion that the bare, unconstrained pipeline
    gives a DIFFERENT number. r31 imposes the constraint on the parameter after
    each optimizer step instead, so the forward pass normalizes nothing and the
    bare pipeline is now the right-hand side rather than the refuted one.

    `v_R` is still drawn far outside the unit ball, and for the same reason it
    always was: inside the ball the two revisions agree exactly, so a small
    `v_R` would let this pass under either one. The closing assertion is
    therefore inverted rather than deleted -- it now pins that r30's dividing
    pipeline gives a different number, which is what makes this a test of which
    revision is implemented.
    """
    for index in range(SWEEP_SIZE):
        s = Sampler(_ATTENTION_SWEEP_SEED + 700_000 + index)
        m = int(s.integers(1, 12))
        dimension = int(s.integers(2, 9))
        hidden = int(s.integers(2, 7))
        H = s.normal(size=(m, dimension))
        V_R = s.normal(size=(hidden, dimension))
        b_R = s.normal(size=hidden)
        # ||v_R||_1 far above the unit ball, so r30 and r31 cannot agree here
        # and this cannot pass by accident of a small v_R.
        v_R = s.normal(size=hidden) * float(s.uniform(20.0, 80.0))
        assert float(torch.linalg.vector_norm(v_R, ord=1)) > 1.0

        logits = relevance_logits(H, V_R, b_R, v_R, 0.0, SIGMA)
        raw = torch.tanh(H @ V_R.transpose(0, 1) + b_R) @ v_R
        assert allclose(logits, raw, atol=1e-12)

        component = relevance_component(H, V_R, b_R, v_R)
        assert allclose(logits, component, atol=1e-12)

        beta = bag_weights(logits, 1.0)
        assert allclose(beta, torch.softmax(raw, dim=0), atol=1e-12)

        # And r30's pipeline really is a different number at this scale, so
        # the reduction above identifies the revision rather than restating an
        # algebraic identity both revisions would satisfy.
        norm = torch.linalg.vector_norm(v_R, ord=1)
        r30_logits = torch.tanh(H @ V_R.transpose(0, 1) + b_R) @ (
            v_R / torch.clamp(norm, min=1.0))
        assert not allclose(logits, r30_logits, atol=1e-6)


def test_effective_bag_size_in_range(rng: Sampler) -> None:
    """r21 Sec. 3 (diagnostic pair preceding Eq. 17): m_eff = 1 / sum(beta^2)
    in [1, m], equal to m only at uniform weights.
    """
    for cfg in _attention_configurations():
        logits = relevance_logits(
            cfg["H"], cfg["V_R"], cfg["b_R"], cfg["v_R"], cfg["gamma"], cfg["sigma"]
        )
        beta = bag_weights(logits, cfg["tau_att"])
        m_eff = float(effective_bag_size(beta))
        assert 1.0 - 1e-8 <= m_eff <= cfg["m"] + 1e-8

    uniform = torch.full((6,), 1.0 / 6, dtype=DTYPE)
    assert close(effective_bag_size(uniform), 6.0, atol=TOL)
    concentrated = torch.zeros(6, dtype=DTYPE)
    concentrated[0] = 1.0
    assert close(effective_bag_size(concentrated), 1.0, atol=TOL)


def test_mean_consensus_in_range(rng: Sampler) -> None:
    """r34 Sec. 3 (diagnostic pair following Eq. 16): the mean consensus
    c_bar(B) is in (0, 1], equal to 1 only when every embedding of the bag
    coincides, and tending to zero as they disperse.

    **The floor is the reading r34 fixed.** r33's quantity here was the uniform
    self-similarity, bounded below by 1/m, so "dispersed" was judged against a
    threshold that moved with the instance count and two bags of different
    sizes could not be compared on it. This one has floor zero for every size.
    """
    checked = 0
    for cfg in _attention_configurations():
        if cfg["m"] < 2:
            continue  # undefined there, and the function refuses -- see below
        value = float(mean_consensus(cfg["H"], cfg["sigma"]))
        # Closed at zero, and for the same reason the consensus itself is: the
        # Gaussian kernel underflows in float64. See
        # `test_consensus_component_in_unit_interval`.
        assert 0.0 <= value <= 1.0 + 1e-8
        checked += 1
    assert checked > 0, "the sweep drew no bag with two or more instances"

    identical = torch.ones((5, 3), dtype=DTYPE)
    assert close(mean_consensus(identical, SIGMA), 1.0, atol=TOL)

    # Dispersed: a tiny bandwidth against well-separated embeddings drives it
    # to the floor r33's form could not reach. The r33 quantity on the SAME
    # input is asserted alongside, because "reaches the floor" is a claim about
    # the difference between the two and not about this number alone.
    spread = torch.eye(4, dtype=DTYPE) * 50.0
    assert float(mean_consensus(spread, 0.5)) < 1e-6
    r33_form = float(_with_self_term_consensus(spread, 0.5).mean())
    assert close(r33_form, 1.0 / 4.0, atol=1e-6), (
        f"r33's form read {r33_form} on a maximally dispersed bag, where its "
        f"own floor 1/m = 0.25 is what it cannot go below"
    )

    try:
        mean_consensus(torch.ones((1, 3), dtype=DTYPE), SIGMA)
    except ValueError as refusal:
        assert "at least two instances" in str(refusal)
    else:
        raise AssertionError(
            "a single-instance bag has no consensus to average and the mean "
            "consensus must refuse rather than invent one")


def _with_self_term_consensus(H: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """The same average, but over ALL m instances -- the sum r34's definition
    does NOT take, and the one r33's did.

    **This helper changed poles at r34 and the inversion is the point.** Until
    r33 it was the correct sum and the leave-one-out form was the mutant; r34
    swapped them, so what is kept is the SAME pair of formulas with the labels
    exchanged, rather than a mutant quietly dropped. It exists so
    `test_separation_condition_implies_majority_consensus` can show it predicts
    something different from the real consensus -- which is what makes
    "included the self term" a mutation that sweep can actually catch, rather
    than one that happens to agree by coincidence.
    """
    H = torch.as_tensor(H, dtype=torch.float64)
    m = H.shape[0]
    squared = torch.cdist(H, H) ** 2
    K = torch.exp(-squared / (2.0 * float(sigma) ** 2))
    return K.sum(dim=1) / m


def test_separation_condition_implies_majority_consensus() -> None:
    """r21 Sec. 3: the sufficient condition for the majority group's
    consensus to exceed the minority group's, over disjoint index sets
    I1, I2 covering the bag:

        (|I1|-1) k_1_lower + |I2| k_12_lower
            > (|I2|-1) k_2_upper + |I1| k_12_upper
        =>  min consensus over I1  >  max consensus over I2

    Constructed rather than merely sampled, so the premise is genuinely
    exercised and not vacuous: two clusters at a controlled separation,
    checked against the bag's OWN actual pairwise kernel values (the true
    min/max over each block), never against assumed bounds.

    Defect (b)/(d): the consensus asserted against the premise is
    `_independent_consensus` -- Eq. (14)'s kernel recomputed from
    `torch.cdist`/`torch.exp` directly, never through
    `MIL_CREDA.attention.consensus_component` -- so this claim is checked
    against the mathematics rather than against whatever the module under
    test happens to compute. A second, WRONG consensus that INCLUDES the self
    term (`_with_self_term_consensus`) is computed for every premise-holding
    configuration too, and `consensus_component`'s own real output is
    checked against BOTH: it has to agree with the correct one and disagree
    with the wrong one, so the module under test is what this exercises,
    never only the two hand-built formulas comparing themselves.

    **The two poles swapped at r34 and the coefficients moved with them.** The
    bounds are taken over OFF-DIAGONAL pairs of each block, which is the other
    half of the same change: r33's text admitted the self evaluations into the
    bounds, "pues kappa^I(h, h) = 1", and r34 deletes that sentence because the
    sums no longer contain them. A `k_2_upper` left at 1.0 by a 1x1 diagonal
    would still be sound -- it only makes the premise harder -- but it would
    make this sweep hold the premise far less often, which is how a test goes
    quietly vacuous.
    """
    dimension = 4
    checked = held = disagreed = 0
    for index in range(SWEEP_SIZE):
        s = Sampler(_ATTENTION_SWEEP_SEED + 500_000 + index)
        n1 = int(s.integers(1, 6))
        n2 = int(s.integers(1, 6))
        spread1 = float(s.uniform(0.02, 0.6))
        spread2 = float(s.uniform(0.02, 0.6))
        separation = float(s.uniform(0.0, 8.0))
        center1 = s.normal(size=dimension)
        direction = s.normal(size=dimension)
        direction = direction / torch.linalg.vector_norm(direction)
        center2 = center1 + separation * direction
        H1 = center1 + s.normal(scale=spread1, size=(n1, dimension))
        H2 = center2 + s.normal(scale=spread2, size=(n2, dimension))
        H = torch.cat([H1, H2], dim=0)
        sigma = float(s.uniform(0.3, 2.0))

        K = gaussian_kernel(H, H, sigma)
        m = n1 + n2
        idx1, idx2 = torch.arange(n1), torch.arange(n1, m)
        K11, K22, K12 = K[idx1][:, idx1], K[idx2][:, idx2], K[idx1][:, idx2]
        # Off-diagonal only: the consensus of r34 never evaluates an instance
        # against itself, so a self evaluation has no business bounding it.
        # A singleton group has no within-group pair at all, and its bound is
        # multiplied by (|I| - 1) = 0 below, so the value is immaterial.
        off11 = K11[~torch.eye(n1, dtype=torch.bool)]
        off22 = K22[~torch.eye(n2, dtype=torch.bool)]
        k1_lower = float(off11.min()) if off11.numel() else 1.0
        k2_upper = float(off22.max()) if off22.numel() else 0.0
        k12_lower, k12_upper = float(K12.min()), float(K12.max())

        premise = ((n1 - 1) * k1_lower + n2 * k12_lower
                   > (n2 - 1) * k2_upper + n1 * k12_upper)
        checked += 1
        if not premise:
            continue
        held += 1
        consensus = _independent_consensus(H, sigma)
        min_group1 = float(consensus[idx1].min())
        max_group2 = float(consensus[idx2].max())
        assert min_group1 > max_group2, (
            f"premise held but group1's minimum consensus {min_group1} did "
            f"not exceed group2's maximum {max_group2}"
        )

        # The module under test agrees with the correct, independent
        # computation and disagrees with the one that omits the self term --
        # checked against `consensus_component` itself, not only against the
        # two hand-built reference formulas.
        produced = consensus_component(H, sigma)
        wrong = _with_self_term_consensus(H, sigma)
        assert allclose(produced, consensus, atol=1e-9)
        if not allclose(produced, wrong, atol=1e-6):
            disagreed += 1

    assert checked == SWEEP_SIZE
    assert held >= 1, "the sweep never constructed a configuration meeting the premise"
    assert disagreed == held, (
        f"the with-self-term consensus matched the module under test on "
        f"{held - disagreed} of {held} premise-holding configurations: an "
        f"included self term would go undetected there"
    )


# --------------------------------------------------------------------------
# bag_kernel.py - Section 3, Eqs. (17)-(18)
# --------------------------------------------------------------------------


def test_bag_kernel_symmetric(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (18): kappa^B is an inner product in H_I, hence symmetric."""
    (H_u, w_u), (H_v, w_v) = make_bags(rng, 2, 4)
    assert close(
        bag_kernel(H_u, w_u, H_v, w_v, SIGMA),
        bag_kernel(H_v, w_v, H_u, w_u, SIGMA),
        atol=TOL,
    )


def test_bag_kernel_psd(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (18): the bag kernel is positive semidefinite."""
    eigenvalues = torch.linalg.eigvalsh(bag_gram(make_bags(rng, 9, 4), SIGMA))
    assert float(eigenvalues.min()) >= -1e-8


def test_bag_self_similarity_at_most_one(rng: Sampler) -> None:
    """r21 Sec. 5: ||Psi(B)||_{H_I} <= 1, so kappa^B(B, B) <= 1.

    Psi is a convex combination of unit-norm images, so the triangle inequality
    caps its norm. This is the first half of the argument bounding d_j^2.
    """
    for H, w in make_bags(rng, 12, 4):
        assert 0.0 < float(bag_kernel(H, w, H, w, SIGMA)) <= 1.0 + TOL


def test_bag_kernel_reduces_to_instance_kernel_on_singletons(rng: Sampler) -> None:
    """r21 Sec. 3, Eq. (18): on one-instance bags the double sum is kappa^I itself.

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


def _class_bags(
    rng: Sampler,
    n_source: int = 4,
    n_target: int = 5,
    withdrawn: int | None = None,
):
    """One class's bags, with the target side on the gradient's path.

    The bags are real bags rather than a Gram matrix written down directly, so
    whatever reads `H.grad` reads it through Eq. (14)'s bag kernel. `withdrawn`
    names one target bag whose Eq. (24) confidence is driven to exactly zero;
    the rest are floored well above it so "this one got nothing" cannot be an
    accident of a confidence that was nearly zero anyway.
    """
    source = make_bags(rng, n_source, 4, shift=0.0)
    target = [
        (H.clone().requires_grad_(True), w)
        for H, w in make_bags(rng, n_target, 4, shift=0.6)
    ]
    confidence = confidences(simplex_rows(rng, n_target, 3)).clamp_min(0.2)
    if withdrawn is not None:
        confidence = confidence.clone()
        confidence[withdrawn] = 0.0
    return source, target, confidence, withdrawn


def _class_blocks_over_bags(source, target, confidence, *, weight_cross_block=True):
    """Eqs. (23) and (26) over one class's bags: `(K_s_c, K~_st_c, K~_t_c)`."""
    K_ss = bag_kernel_matrix(source, source, SIGMA)
    K_st = bag_kernel_matrix(source, target, SIGMA)
    K_tt = bag_kernel_matrix(target, target, SIGMA)
    K_s_c, K_st_c, K_t_c = conditional_blocks(
        K_ss, K_st, K_tt, torch.arange(len(source)), torch.arange(len(target))
    )
    K_st_w, K_t_w = weighted_blocks(
        K_st_c, K_t_c, confidence, weight_cross_block=weight_cross_block
    )
    return K_s_c, K_st_w, K_t_w


def _global_term_over_bags(
    source,
    target,
    confidence,
    *,
    weight_cross_block: bool = True,
    detach_target_entropy: bool = False,
):
    """One class's whole global term, Eq. (32) through Eq. (36), as a scalar."""
    K_s_c, K_st_w, K_t_w = _class_blocks_over_bags(
        source, target, confidence, weight_cross_block=weight_cross_block
    )
    breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_w, K_t_w)
    return class_global_loss_cosh(
        balanced_dependency_score(
            K_s_c, K_st_w, K_t_w, detach_target_entropy=detach_target_entropy
        ),
        quadratic_entropy(breve_s),
        quadratic_entropy(breve_t),
    )


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


def test_weight_cross_block_off_leaves_the_cross_block_unweighted(
    rng: Sampler,
) -> None:
    """r29 Sec. 3, Eq. (26): `weight_cross_block` decides whether `D` reaches it.

    Both poles of the axis, measured on the block itself. Under the default the
    cross block is `K_c^{st} D`, which is the Hadamard form `K_c^{st} * w^T`
    broadcast across ROWS -- weight on the target side only, since rows are
    source bags whose labels are observed. Under the axis off it is the block
    as it arrived, which is CREDA's original.

    The two have to actually differ, or naming the axis named nothing; and the
    target block has to be weighted under both, because Eq. (26)'s congruence
    on `K_c^t` is not what this axis is about.
    """
    K_ss, K_st, K_tt, S_c, T_c, w = _class_setup(rng)
    _, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, S_c, T_c)
    weighted_cross, weighted_target = weighted_blocks(K_st_c, K_t_c, w)
    raw_cross, raw_target = weighted_blocks(
        K_st_c, K_t_c, w, weight_cross_block=False
    )

    # The default pole, against the broadcast Hadamard form rather than
    # against this test's own copy of the matrix product.
    assert allclose(weighted_cross, K_st_c * w.reshape(1, -1), atol=TOL)
    # The other pole: the block as it arrived.
    assert allclose(raw_cross, K_st_c, atol=TOL)
    # The axis moves something. `w` is Eq. (24)'s confidence and is not all
    # ones, so this is reachable rather than an accident of the fixture.
    assert float(w.max()) < 1.0
    assert not allclose(weighted_cross, raw_cross)
    # And it moves only the cross block.
    assert allclose(weighted_target, raw_target, atol=TOL)


def test_weight_cross_block_freezes_a_zero_confidence_bag(rng: Sampler) -> None:
    """r29 Sec. 3, Eq. (26): which pole can still repair a withdrawn bag.

    The axis that separates *who defines the geometry* from *whom the term
    repairs*, measured as the gradient it is about rather than as a value.
    With the cross block weighted, a bag of confidence zero has its row and
    column emptied out of every block that reaches it, so the global term's
    gradient with respect to that bag's instance embeddings is exactly zero and
    the bag stays FROZEN. With the cross block raw, `K_c^{st}` still carries it
    and the term can still move it.

    The gradient travels the whole path a training step travels -- Eq. (14)'s
    bag kernel, Eq. (23)'s blocks, Eq. (26)'s weighting, Eq. (32)'s balance,
    Eq. (33)'s score, Eq. (36)'s map -- because a gradient only exists at the
    end of one, and the claim is about where it arrives.

    The bags the gate KEPT are the control: they receive gradient under both
    poles, so the zero is this bag's and not the whole term's.
    """
    source, target, w, withdrawn = _class_bags(rng, withdrawn=2)
    assert float(w[withdrawn]) == 0.0
    assert float(w.min()) == 0.0 and float(w[:withdrawn].min()) > 0.0

    norms = {}
    for weighted in (True, False):
        loss = _global_term_over_bags(
            source, target, w, weight_cross_block=weighted
        )
        grads = torch.autograd.grad(loss, [H for H, _ in target])
        norms[weighted] = [float(g.norm()) for g in grads]

    # The declared pole freezes it; the other one does not.
    assert norms[True][withdrawn] == 0.0
    assert norms[False][withdrawn] > 0.0
    # The control: every kept bag is reached under both.
    for index in range(len(target)):
        if index == withdrawn:
            continue
        assert norms[True][index] > 0.0, index
        assert norms[False][index] > 0.0, index


# --------------------------------------------------------------------------
# global_term.py - Section 5, Eqs. (32)-(37)
# --------------------------------------------------------------------------
# **r25's two retained claims lived here and were removed on 2026-10-18.**
# `test_score_within_conservative_bounds` and
# `test_class_global_loss_in_unit_interval` asserted Eq. (34)/(36) over r25's
# cardinality ends, and said so: they were kept "because `src/CREDA/bounds.py`
# copies the latter and `tests/test_creda_sample.py` pins the copy to it". r33
# states one pair of ends, `conservative_bounds` and `class_global_loss` were
# removed from both packages with the CREDA copies, and a test of a function
# that no longer exists is not a weaker check -- it is an import error. r33's
# own bound claims are in `tests/test_milcreda_r29_global_term.py`.


def test_detach_target_entropy_removes_the_dispersive_force_and_not_the_value(
    rng: Sampler,
) -> None:
    """r29 Sec. 5, Eq. (33): `detach_target_entropy` moves gradient, never value.

    **The value cannot see this axis, and that is why the test reads the
    gradient.** `x.detach()` and `x` are the same number, so `J_c` and every
    loss built on it come out bit-identical under both poles -- a test
    asserting on the score or on Eq. (36)'s loss would pass with the axis on
    AND with the axis broken. It is asserted here as an equality to the last
    bit, because it is the trap and not a tolerance.

    What the axis removes is the DISPERSIVE force the target block exerts on
    itself. Everything else is left intact: `H_mix` still carries the target
    inside, so the alignment is untouched, and Eq. (35)'s ends read
    `H_2(K~_t)` through `class_global_loss_cosh`'s own argument and not through
    the score, so the bounds are untouched too.

    The removal is exact rather than merely non-zero, which is what makes this
    a measurement of the axis and not of the fact that two gradients differ.
    r31's Eq. (36) is `(U_c - J_c) / ln 2` -- a CONSTANT denominator, where r29
    divided by the per-class span `U_c - L_c` and detached it -- and `H_2(K~_t)`
    enters `J_c` with coefficient `1/2`, so detaching it must move the gradient
    by exactly `-(1/2) grad H_2(K~_t) / ln 2` and by nothing else.

    **The denominator here follows Eq. (36) and is not a free constant.** It
    was `float((upper - lower).detach())` while that was Eq. (36)'s
    denominator; under r31 that value is `U_c + ln 2`, which differs from
    `ln 2` by `U_c` -- measured at this fixture, 0.694154 against 0.693147 --
    so leaving it would have pinned r29's map inside a test about a different
    axis entirely.
    """
    source, target, w, _ = _class_bags(rng)
    leaves = [H for H, _ in target]

    values, grads = {}, {}
    for detached in (False, True):
        loss = _global_term_over_bags(
            source, target, w, detach_target_entropy=detached
        )
        values[detached] = float(loss.detach())
        grads[detached] = torch.autograd.grad(loss, leaves)

    # The trap: identical to the last bit, so no value test can see the axis.
    assert values[False] == values[True]

    # The gradient can see it.
    moved = max(
        float((attached - held).abs().max())
        for attached, held in zip(grads[False], grads[True])
    )
    assert moved > 0.0

    # And what moved is exactly the force the detach removes.
    K_s_c, K_st_w, K_t_w = _class_blocks_over_bags(source, target, w)
    breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_w, K_t_w)
    h_source, h_target = quadratic_entropy(breve_s), quadratic_entropy(breve_t)
    # r31 Eq. (36)'s denominator, read from the module rather than recomputed,
    # so a later change to the equation cannot leave this reference behind.
    denominator = GLOBAL_LOSS_SPAN
    # And it really is the constant and not the old span: the two differ here.
    _lower, upper = entropy_cosh_bounds(h_source, h_target)
    assert abs(float((upper - _lower).detach()) - denominator) > 1e-6, (
        "U_c has collapsed to 0 at this fixture, so r29's span and r31's "
        "constant coincide and this test cannot tell the two maps apart")
    dispersive = torch.autograd.grad(h_target, leaves)
    for attached, held, force in zip(grads[False], grads[True], dispersive):
        assert allclose(
            attached - held, (-0.5 / denominator) * force, atol=1e-14, rtol=0.0
        )
def test_global_loss_is_convex_combination_of_source_proportions() -> None:
    """r33 Sec. 5, Eq. (37): the weights are `n_c^s / N_s`, they sum to exactly
    one, and the term is a convex combination of values in [0, 1].

    **This replaced a SUB-convex claim.** r31's weights were
    `omega_c = q_c / sum_c' n_c'^t`, built from Eq. (24)'s confidences, summing
    to at most one -- the shortfall was a gate on the absolute confidence
    level, and the test here asserted exactly that scaling every confidence
    scaled the term. r32 removed it, because a weight the model computes can be
    lowered instead of earned, and replaced it with the observed source
    proportions. There is no shortfall left to assert: `sum_c n_c^s / N_s` is
    one identically, so a caller cannot scale the term at all.
    """
    losses = [0.1, 0.3, 0.5, 0.7, 0.9]
    counts = [2, 2, 2, 2, 2]
    balanced = float(global_loss(losses, counts))
    assert close(balanced, sum(losses) / len(losses), atol=TOL)
    assert 0.0 - TOL <= balanced <= 1.0 + TOL

    # Unbalanced: each class weighs the source evidence it brought, so the term
    # moves toward the loss of the class with more source bags.
    heavy_low = float(global_loss(losses, [50, 1, 1, 1, 1]))
    heavy_high = float(global_loss(losses, [1, 1, 1, 1, 50]))
    assert heavy_low < balanced < heavy_high
    assert 0.0 - TOL <= heavy_low and heavy_high <= 1.0 + TOL

    # Counts, not weights: the caller has no way to hand a sum that is not one.
    with pytest.raises(ValueError):
        global_loss(losses, [2, 2, 2, 2])            # one count per class
    with pytest.raises(ValueError):
        global_loss(losses, [2, 2, 2, 2, -1])        # a count is non-negative
    with pytest.raises(ValueError):
        global_loss(losses, [0, 0, 0, 0, 0])         # N_s = 0, weights undefined
    with pytest.raises(ValueError):
        global_loss([], [])                          # Eq. (37) sums over C


def test_global_loss_charges_one_for_an_inactive_class() -> None:
    """r33 Sec. 5, Eq. (37): `l~_c = 1` for `c` not in `C_act`, so abandoning a
    class RAISES the term.

    **r31 answered 0 with `C_act` empty and excluded inactive classes from the
    sum, which is the defect this replaces.** A target encoder that collapsed
    until the hard pseudo-label named one class left a sum over one class --
    fewer terms, each in [0, 1] -- and the term went down while nothing was
    aligned. Charging the maximum makes the collapse read as a rise.
    """
    losses = [0.1, 0.2, 0.3]
    counts = [4, 4, 4]
    all_active = float(global_loss(losses, counts))

    # Abandon the third class. r31 would have averaged the surviving two and
    # reported LESS; r33 charges 1 for it and reports more.
    abandoned = float(global_loss([0.1, 0.2, None], counts))
    r31_would_have = (0.1 + 0.2) / 2
    assert abandoned > all_active
    assert abandoned > r31_would_have
    assert close(abandoned, (0.1 + 0.2 + 1.0) / 3, atol=TOL)

    # Every class abandoned: the weights still sum to one, so the term is 1.
    assert close(global_loss([None, None, None], counts), 1.0, atol=TOL)

    # A class with no SOURCE bag cannot be active and weighs zero, so it costs
    # nothing -- the charge is for a class the source has and the routing left.
    assert close(global_loss([0.1, 0.2, None], [4, 4, 0]),
                 (0.1 + 0.2) / 2, atol=TOL)


def test_global_loss_inactive_charge_carries_no_gradient() -> None:
    """r33 Sec. 5, Eq. (37): the charge is a literal and the weights are counts,
    so an inactive class contributes no gradient.

    Eq. (22)'s routing is an `argmax`, so there is no path to differentiate in
    the first place -- the charge's job is the VALUE. Asserted by differentiating
    a term in which one class is active and one is not, and checking that the
    gradient equals the active class's own weight.
    """
    active = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
    value = global_loss([active, None], [3, 1])
    value.backward()
    assert close(value, 0.4 * 0.75 + 1.0 * 0.25, atol=TOL)
    # d/d(l_glob,0) = n_0^s / N_s = 3/4, with nothing added by the charge.
    assert close(active.grad, 0.75, atol=TOL)


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
    assert -TOL <= float(local_loss(d2)) <= 1.0 + TOL
    # r33 Eq. (38) is the uniform mean, so zero confidence is not a zero term
    # any more: `w_j^t` left the aggregation at r32 and `local_loss` has no
    # argument to pass it through. The claim that replaces "no reliable
    # pseudo-label, no penalty" is that the confidences cannot enter at all.
    assert 'target_weights' not in local_loss.__code__.co_varnames


def test_local_loss_is_the_uniform_mean_and_ignores_confidence(
    rng: Sampler,
) -> None:
    """r33 Eq. (38): `L_loc = (1/N_t) sum_j l_loc,j`, every subject weighted
    alike.

    **The revision's own reason for dropping the weights is why "ignores" is
    half of this claim.** Until r31 the aggregate was
    `sum_j w_j^t l_loc,j / (sum_j w_j^t + eps_loc)`, and `w_j^t` is a
    DIFFERENTIABLE function of the classifier's outputs (Eq. 24): a weight the
    model computes can be lowered on the worst-aligned subjects instead of
    aligning them, which lowers `L_loc` without moving the geometry. A uniform
    mean has no such channel.

    Both halves are asserted and neither alone is enough. The positive half --
    equality with `np.mean` of the per-subject losses -- would pass on a
    weighted mean whose weights happened to be uniform on the fixture. The
    negative half -- that no spelling of a confidence vector reaches the
    result -- is checked by CALLING, not by reading the signature, because
    `local_loss` could still read a module-level default: the same distances
    under three deliberately lopsided weightings must give the identical
    value, bit for bit.

    The pseudo-label's uncertainty did not leave the term, it moved inside it:
    Eq. (29) keeps the whole `g_j^t`, so an uncertain subject is compared
    against a reference that mixes the classes in proportion to it. What this
    pins is that aggregation is where it no longer acts.
    """
    d2 = rng.uniform(0.0, 2.0, size=9)
    por_sujeto = normalized_local_distance(d2)

    # The uniform mean, against an independent computation rather than a second
    # call of the function under test.
    esperado = float(np.asarray(por_sujeto, dtype=float).mean())
    assert float(local_loss(d2)) == pytest.approx(esperado, abs=1e-12)
    # And it is the MEAN and not the sum, which a one-subject bag cannot tell
    # apart: with nine subjects the two differ by a factor of nine.
    assert float(local_loss(d2)) != pytest.approx(
        float(np.asarray(por_sujeto, dtype=float).sum()), rel=1e-6)

    # The confidences cannot enter. Three lopsided weightings, one of them
    # concentrating almost everything on the best-aligned subject and another
    # on the worst -- the exact move the weighted form rewarded.
    orden = np.argsort(np.asarray(d2, dtype=float))
    for destino in (orden[0], orden[-1]):
        w = np.full(len(d2), 1e-6)
        w[destino] = 1.0
        # Passed the only way a weighting could still reach it: positionally,
        # and by keyword under every name the retired form used.
        with pytest.raises(TypeError):
            local_loss(d2, w)
        for nombre in ('target_weights', 'weights', 'w_t'):
            with pytest.raises(TypeError):
                local_loss(d2, **{nombre: w})
    # Unchanged after all of that: the refusals left no state behind and the
    # value is the same one measured above.
    assert float(local_loss(d2)) == pytest.approx(esperado, abs=1e-12)

    # `eps_loc` went with the denominator it stabilized, so no stabilizer can
    # pull the aggregate toward zero any more: an all-zero confidence used to
    # drive this term to 0 and now it reads the geometry it was handed.
    assert float(local_loss(d2)) > 0.0, (
        "the fixture's distances are positive, so a zero aggregate would mean "
        "a stabilizer is still in the denominator")


def test_every_declared_invariant_of_the_bag_packages_has_a_test() -> None:
    """The pairing `tests/test_creda_provenance.py` holds for `src/CREDA`, held
    here for the two bag packages it does not reach.

    **This exists because the hole was found by hand, which is the thing it
    replaces.** `tests/test_creda_provenance.py::test_every_declared_invariant_
    has_a_test_somewhere_under_tests` scans `PACKAGE = src/CREDA` and nothing
    else, so `local_loss_is_the_uniform_mean_and_ignores_confidence` --
    declared in `src/MIL_CREDA/local_term.py` when r33 landed -- sat unbacked
    and every suite ran green over it. A declared invariant with no test and a
    tested property with no declaration look identical to a green run, and
    that is as true on this side of the repository as on the other.

    Read from the source without importing, so a module that fails to import
    is a different failure with a different message.
    """
    import ast

    repositorio = Path(__file__).resolve().parents[1]
    nombres = set()
    for archivo in sorted((repositorio / "tests").glob("test_*.py")):
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        nombres |= {n.name for n in ast.walk(arbol)
                    if isinstance(n, ast.FunctionDef)
                    and n.name.startswith("test_")}

    paquetes = [repositorio / "src" / "MIL_CREDA",
                repositorio / "src" / "MIL_CREDA_Benchmark"]
    assert all(p.is_dir() for p in paquetes), paquetes

    declaradas, sin_test = 0, []
    for paquete in paquetes:
        for modulo in sorted(paquete.rglob("*.py")):
            arbol = ast.parse(modulo.read_text(encoding="utf-8"),
                              filename=str(modulo))
            for nodo in arbol.body:
                objetivos = (nodo.targets if isinstance(nodo, ast.Assign)
                             else [nodo.target]
                             if isinstance(nodo, ast.AnnAssign) else [])
                if not any(isinstance(t, ast.Name)
                           and t.id == "__provenance__" for t in objetivos):
                    continue
                declarado = ast.literal_eval(nodo.value)
                for invariante in declarado.get("invariants", ()):
                    declaradas += 1
                    if f"test_{invariante}" not in nombres:
                        sin_test.append(
                            f"{modulo.relative_to(repositorio)}::{invariante}")

    # The sweep found something, so an empty result is a result and not a
    # reader that walked past every declaration.
    assert declaradas > 0, "no __provenance__ was read in either bag package"
    assert sorted(sin_test) == [], (
        f"declared invariants with no test: {sorted(sin_test)}")


# --------------------------------------------------------------------------
# objective.py - Sections 3, 5, Eqs. (21), (39)
# --------------------------------------------------------------------------


def test_source_loss_matches_negative_log_likelihood_of_the_observed_class(
    rng: Sampler,
) -> None:
    """r21 Sec. 3, Eq. (21): the observed class alone, averaged per bag, over B_src.

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
    """r21 Sec. 3, Eq. (21): L_src >= 0 always, and exactly 0 at a certain, correct bag.

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
    """r21 Sec. 3, Eq. (21): L_src in [0, 1), the upper end a supremum not attained.

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
