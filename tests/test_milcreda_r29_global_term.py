"""r29's four changes to the global term, and the two that carry the revision.

r25's global term trace-normalized each of the three structures separately and
normalized the score by ends derived from CARDINALITY alone. r29 replaces both,
and replaces the aggregation's weights with a confidence gate. Four changes land
in `src/MIL_CREDA/global_term.py`, all inside Section 5:

1. **Eq. 32 -- the trace balance becomes a congruence.** The joint is
   `K_mix = (1/2) Q K Q` with `Q = diag(tr(K_s)^-1/2 I, tr(K~_t)^-1/2 I)`, so
   positive semidefiniteness survives (a congruence by a positive diagonal) and
   the trace is exactly `1/2 + 1/2 = 1`. Each domain contributes half the mass
   whatever its bag count and whatever the COMMON level of the confidences; the
   RELATIVE confidences keep acting inside `K~_t` and `K~_st`, and a bag whose
   confidence vanishes is withdrawn.
2. **Eqs. 34/35 -- the bounds.** `L_c = -ln 2` and
   `U_c = 2 ln cosh((H_2(K~_t) - H_2(K_s)) / 4)`, both read off the same
   normalized matrices that build `J_c`.
3. **Eq. 36 -- the span carries no gradient.** The denominator `U_c - L_c` is a
   constant in each step; the numerator `U_c - J_c` keeps its graph.
4. **Eq. 37 -- the weights become a confidence gate.**
   `omega_c = q_c / sum_c' n_c'^t` with `q_c = sum_{j in T_c} w_j^t`, so the
   weights are SUB-convex: `sum omega_c <= 1`, with equality only when every
   routed confidence is one.

**Two tests here carry the revision and the rest support them.**
`test_entropy_cosh_bounds_order_alignment_above_the_collapse` reproduces r29's
own counterexample at `n_c^s = 43` and shows that r25's map prefers that
collapse to perfect alignment while r29's does not -- without it, change 2 is a
rewrite with no measured consequence.
`test_confidence_weights_gate_on_the_absolute_confidence_level` reproduces
r29's own worked figure (every confidence at 0.1 multiplies the term by 0.1) AND
measures the plausible wrong denominator, `sum_c q_c`, which would not gate at
all -- that second half is what distinguishes the right denominator from the one
that merely looks like a normalization.

**r25's pair was kept until 2026-10-18 and then removed.** `conservative_bounds`
and `class_global_loss` stayed byte-identical for as long as
`src/CREDA/bounds.py` held copies of both and `tests/test_creda_sample.py`
pinned them. r33 states ONE pair of ends, so the pair went from both packages
along with its copies and its two copy invariants. Where a test here needs
r25's map to show WHY it was replaced, it reconstructs it locally from
`proposals/research-concept-r25.md` -- see `_retired_r25_loss` -- rather than
keeping an implementation of a retired equation alive in `src/`.

**The cosh pair is implemented twice on purpose.** `src/CREDA/bounds.py`
already carries `entropy_cosh_bounds` and `class_global_loss_cosh` for its own
r28 fork, and `src/MIL_CREDA/` imports nothing from `CREDA` -- the dependency
may only ever point that way, never backwards. Rather than import across the
join, `test_cosh_bounds_agree_across_the_two_packages` asserts the two agree on
the same inputs, which crosses the join instead of assuming it.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from CREDA import bounds as creda_bounds
from MIL_CREDA import DTYPE
from MIL_CREDA.attention import bag_embedding
from MIL_CREDA.bag_kernel import bag_kernel_matrix
from MIL_CREDA.conditional import (
    conditional_blocks,
    mixed_matrix,
    weighted_blocks,
)
from MIL_CREDA.confidence import confidences, pseudolabel
from MIL_CREDA.global_term import (
    balanced_dependency_score,
    balanced_mixture,
    class_global_loss_cosh,
    entropy_cosh_bounds,
    global_loss,
    is_active,
)
from MIL_CREDA.renyi import quadratic_entropy, trace_normalize
from MIL_CREDA_Benchmark import config, wiring

#: Tight: every claim below is exact arithmetic over float64 densities, not a
#: statistical reading, so a tolerance wide enough to hide a wrong coefficient
#: would hide the only thing these tests measure.
TOL = 1e-12


def _joint_blocks(
    n_source: int, n_target: int, dimension: int = 5, seed: int = 29
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three conditional blocks that are jointly PSD, because they share a Gram.

    Drawing the three blocks independently would produce a `K_mix` that is not
    positive semidefinite, and Eq. 32's claim is that the congruence PRESERVES
    positive semidefiniteness -- which says nothing at all about an input that
    never had it. So the blocks are read off one Gram matrix over stacked
    source and target features, exactly as `conditional_blocks` reads them off
    the bag kernel in the real path.
    """
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(
        (n_source + n_target, dimension), generator=generator, dtype=DTYPE
    )
    features = features / features.norm(dim=1, keepdim=True)
    gram = features @ features.transpose(0, 1)
    return (
        gram[:n_source, :n_source],
        gram[:n_source, n_source:],
        gram[n_source:, n_source:],
    )


def _confidences(n_target: int, seed: int = 29) -> torch.Tensor:
    """Target confidences in (0, 1], the range Eq. 24 produces."""
    generator = torch.Generator().manual_seed(seed + 1)
    return 0.2 + 0.8 * torch.rand(n_target, generator=generator, dtype=DTYPE)


# ------------------------------------------------- 1. Eq. 32, the trace balance


class R29BalancedMixtureTests(unittest.TestCase):
    """Eq. 32: the balance is a congruence, and what that buys."""

    def setUp(self) -> None:
        self.n_source, self.n_target = 4, 7
        K_s, K_st, K_t = _joint_blocks(self.n_source, self.n_target)
        self.w = _confidences(self.n_target)
        self.K_s = K_s
        self.K_st_w, self.K_t_w = weighted_blocks(K_st, K_t, self.w)

    def test_balanced_mixture_is_a_unit_trace_congruence(self) -> None:
        """Eq. 32: `(1/2) Q K Q` -- positive semidefinite, and trace exactly 1.

        Three claims r29 states in one sentence, so they are measured together:
        the returned joint equals `(1/2) Q K Q` for the Q the equation writes,
        its trace is `1/2 + 1/2 = 1` so it is a valid density for Eq. 9, and its
        spectrum stays non-negative because a congruence by a positive diagonal
        preserves positive semidefiniteness.

        The leading `1/2` is asserted, not assumed: it is what makes the trace
        1 rather than 2, and r29's whole bound derivation reads
        `rho_mix = (1/4) rho_s + (1/4) rho_t + (1/2) ||K~_st||_F^2`, which is
        the Frobenius norm of a matrix carrying that factor.
        """
        breve_s, breve_t, breve_mix = balanced_mixture(
            self.K_s, self.K_st_w, self.K_t_w
        )

        # The trace is the claim: 1/2 from each domain, so 1 in total.
        self.assertAlmostEqual(float(torch.trace(breve_mix)), 1.0, places=12)
        self.assertAlmostEqual(float(torch.trace(breve_s)), 1.0, places=12)
        self.assertAlmostEqual(float(torch.trace(breve_t)), 1.0, places=12)

        # Against Eq. 32's own right-hand side, assembled here from Q.
        trace_s = torch.trace(self.K_s)
        trace_t = torch.trace(self.K_t_w)
        Q = torch.diag(
            torch.cat(
                (
                    torch.full((self.n_source,), float(trace_s) ** -0.5, dtype=DTYPE),
                    torch.full((self.n_target,), float(trace_t) ** -0.5, dtype=DTYPE),
                )
            )
        )
        K_mix = mixed_matrix(self.K_s, self.K_st_w, self.K_t_w)
        expected = 0.5 * Q @ K_mix @ Q
        self.assertTrue(torch.allclose(breve_mix, expected, atol=TOL, rtol=0.0))

        # A congruence by a positive diagonal preserves the spectrum's sign.
        self.assertGreaterEqual(
            float(torch.linalg.eigvalsh(K_mix).min()), -1e-10
        )
        self.assertGreaterEqual(
            float(torch.linalg.eigvalsh(breve_mix).min()), -1e-10
        )

    def test_balanced_mixture_halves_the_mass_of_each_domain(self) -> None:
        """Eq. 32: half the mass each, whatever the counts and the common level.

        Both poles. The balanced joint puts exactly `1/2` on each diagonal
        block for every pair of cardinalities and every uniform rescaling of
        the confidences. The UNBALANCED joint r25 normalized -- `K_mix` over
        its own trace -- does not: its source share is
        `tr(K_s) / tr(K~_mix)`, which r29 names as the quantity that depends on
        the batch's routing counts and on the confidences' common scale. The
        second half is what makes the first a correction rather than a
        restatement.
        """
        unbalanced_shares = []
        for n_source, n_target in ((2, 9), (5, 5), (9, 2)):
            K_s, K_st, K_t = _joint_blocks(n_source, n_target)
            w = _confidences(n_target)
            for scale in (1.0, 0.1, 0.01):
                K_st_w, K_t_w = weighted_blocks(K_st, K_t, w * scale)
                _, _, breve_mix = balanced_mixture(K_s, K_st_w, K_t_w)
                source_mass = float(torch.trace(breve_mix[:n_source, :n_source]))
                target_mass = float(torch.trace(breve_mix[n_source:, n_source:]))
                self.assertAlmostEqual(source_mass, 0.5, places=12)
                self.assertAlmostEqual(target_mass, 0.5, places=12)

                # The control: r25's normalization of the same blocks.
                K_mix = trace_normalize(mixed_matrix(K_s, K_st_w, K_t_w))
                unbalanced_shares.append(
                    float(torch.trace(K_mix[:n_source, :n_source]))
                )

        # The control has to actually vary, or "the balance fixes it" is a
        # claim about a quantity that was already constant.
        self.assertGreater(max(unbalanced_shares) - min(unbalanced_shares), 0.1)
        for share in unbalanced_shares:
            self.assertNotAlmostEqual(share, 0.5, places=2)

    def test_balanced_mixture_withdraws_a_bag_whose_confidence_vanishes(self) -> None:
        """Eq. 32: "una bolsa con `w_j^t -> 0` equivale a retirarla".

        Measured as the limit it is: the balanced joint built over all target
        bags with one confidence driven to zero converges to the balanced joint
        built over the remaining bags, with that bag's row and column empty.
        Asserting equality AT zero would be weaker -- a zero row is zero under
        any normalization -- so the approach is measured as the confidence
        shrinks and the residual is required to fall.
        """
        n_source, n_target = 4, 6
        K_s, K_st, K_t = _joint_blocks(n_source, n_target)
        w = _confidences(n_target)

        keep = torch.arange(n_target - 1)
        K_st_kept, K_t_kept = weighted_blocks(
            K_st[:, keep], K_t[keep][:, keep], w[keep]
        )
        _, _, withdrawn = balanced_mixture(K_s, K_st_kept, K_t_kept)

        residuals = []
        for epsilon in (1e-2, 1e-4, 1e-6):
            w_faded = w.clone()
            w_faded[-1] = epsilon
            K_st_w, K_t_w = weighted_blocks(K_st, K_t, w_faded)
            _, _, faded = balanced_mixture(K_s, K_st_w, K_t_w)
            # Drop the faded bag's own row and column to compare like with like.
            surviving = torch.cat(
                (torch.arange(n_source + n_target - 1),), dim=0
            )
            reduced = faded[surviving][:, surviving]
            residuals.append(float((reduced - withdrawn).abs().max()))

        self.assertLess(residuals[-1], 1e-10)
        for earlier, later in zip(residuals, residuals[1:]):
            self.assertLess(later, earlier)

    def test_balanced_score_is_invariant_to_the_common_confidence_scale(self) -> None:
        """Eq. 32/37: the normalization cancels the confidences' common scale.

        This is the premise Eq. 37's gate rests on -- "la normalizacion por
        traza (Ec. 32) cancela dentro de la puntuacion la escala comun de las
        confianzas, de modo que su nivel absoluto entra aqui, fuera de ella".
        If the score already moved with the common level, the gate outside it
        would be counting the same thing twice.

        Both poles again: the balanced score does not move, and r25's score
        over the same blocks does -- r29 states that without the balance "una
        caida uniforme de todas las confianzas puede elevar `J_c`".

        The control's margin is MEASURED rather than chosen. Across six
        cardinality pairs and three seeds, r25's score rose in all eighteen,
        with spans from 0.021 to 0.71; the smallest observed rise sets the
        margin asserted over the grid, and the `(2, 9)` fixture is singled out
        because there the rise carries the score from negative to positive --
        r29's claim in its strongest reachable form.
        """
        scales = (1.0, 0.5, 0.1, 0.01)

        def scores(K_s, K_st, K_t, w, scale):
            K_st_w, K_t_w = weighted_blocks(K_st, K_t, w * scale)
            breve_s, breve_t, breve_mix = balanced_mixture(K_s, K_st_w, K_t_w)
            balanced = float(
                0.5 * (quadratic_entropy(breve_s) + quadratic_entropy(breve_t))
                - quadratic_entropy(breve_mix)
            )
            unbalanced = float(
                0.5
                * (
                    quadratic_entropy(trace_normalize(K_s))
                    + quadratic_entropy(trace_normalize(K_t_w))
                )
                - quadratic_entropy(trace_normalize(mixed_matrix(K_s, K_st_w, K_t_w)))
            )
            return balanced, unbalanced

        for n_source, n_target in ((4, 7), (2, 9), (9, 2), (3, 3), (8, 3)):
            K_s, K_st, K_t = _joint_blocks(n_source, n_target)
            w = _confidences(n_target)
            measured = [scores(K_s, K_st, K_t, w, scale) for scale in scales]
            balanced = [pair[0] for pair in measured]
            unbalanced = [pair[1] for pair in measured]

            # r29: the balance makes the score blind to the common level.
            for value in balanced[1:]:
                self.assertAlmostEqual(value, balanced[0], places=12)
            # The control: r25's score rises as that level falls, every time,
            # and by more than the smallest rise measured over the grid.
            self.assertGreater(unbalanced[-1], unbalanced[0] + 0.02)

        # The sharpest instance: the same uniform fall takes r25's score across
        # zero, so the sign of the alignment reading itself is not preserved.
        K_s, K_st, K_t = _joint_blocks(2, 9)
        w = _confidences(9)
        full = scores(K_s, K_st, K_t, w, 1.0)
        faded = scores(K_s, K_st, K_t, w, 0.01)
        self.assertAlmostEqual(faded[0], full[0], places=12)
        self.assertLess(full[1], 0.0)
        self.assertGreater(faded[1], 0.0)


# ----------------------------------------------------- 2. Eqs. 34/35, the bounds


class _StubEncoder(nn.Module):
    """Something with a declared `output_dim`, standing in for
    `CREDA.models.FeatureExtractor`.

    **Restored on 2026-10-18.** It and `_pool` below were module-level helpers
    of this file and were destroyed by a scripted class replacement while
    `R29ConfidenceGateTests` was being rewritten as `R33AggregationTests`;
    `R29WiringTests` failed with a bare `NameError` naming this class. Rebuilt
    from the identical pair in `tests/test_local_dispersion.py`,
    `tests/test_arm_objectives.py` and `tests/test_kernel_normalization.py`,
    which exist for the same reason and use the same shape: no claim anywhere
    in this file is about the backbone.
    """

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


def _pool(seed: int) -> wiring.Pool:
    """A pool shaped for `_StubEncoder` above -- (3, 8, 8) images, the shape
    the sibling test files already use for the same reason.
    """
    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, 3, 8, 8,
                         generator=generator)
    members = torch.arange(images.shape[0]).reshape(
        bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % config.CLASSES
    return wiring.Pool(images, members, labels)


def _retired_r25_score(K_s_c, K_t_w, K_mix) -> float:
    """r25's Eq. (33) contrast, reconstructed HERE.

    `dependency_score` computed this until 2026-10-18 and was removed with the
    ends that bounded it: `H_2` of the trace-normalized source and target
    blocks, halved and summed, minus `H_2` of the trace-normalized mixture.
    Traceable to `proposals/research-concept-r25.md`; a reader for a retired
    score, never an assertion about the current one, which is
    `balanced_dependency_score` over Eq. (32)'s balanced congruence.
    """
    h_s = quadratic_entropy(trace_normalize(K_s_c))
    h_t = quadratic_entropy(trace_normalize(K_t_w))
    h_mix = quadratic_entropy(trace_normalize(K_mix))
    return float(0.5 * (h_s + h_t) - h_mix)


def _retired_r25_loss(score, n_source: int, n_target: int) -> float:
    """r25's Eq. (35)/(36) over one score, reconstructed HERE.

    `conservative_bounds` and `class_global_loss` computed this until
    2026-10-18 and were removed from both packages: r33 states one pair of
    ends and r25's cardinality pair is nowhere in the formulation.

    The comparison they served is the EVIDENCE for that removal -- that r25's
    constant ends ordered a collapsed target ABOVE perfect alignment -- so it
    is reconstructed here rather than left as a sentence. Three lines, all
    traceable to `proposals/research-concept-r25.md`:
    `L_c = -ln(n_c^s + n_c^t)`, `U_c = (1/2)(ln n_c^s + ln n_c^t)`, and
    `l_glob,c = (U_c - J_c) / (U_c - L_c)`. A reader for a retired map; never
    use it to assert anything about the current one.
    """
    lower = -math.log(n_source + n_target)
    upper = 0.5 * (math.log(n_source) + math.log(n_target))
    return (upper - float(score)) / (upper - lower)


def _density(matrix: torch.Tensor) -> torch.Tensor:
    """A unit-trace matrix, so the entropies below are Eq. 9's."""
    return matrix / torch.trace(matrix)


def _assemble(
    breve_s: torch.Tensor, breve_st: torch.Tensor, breve_t: torch.Tensor
) -> torch.Tensor:
    """Eq. 32's joint, from three blocks that are already densities.

    `balanced_mixture` takes UNNORMALIZED blocks and normalizes them; these
    tests need to place a chosen `K̆_s`, `K̆̃_t` and cross block directly, which
    is what r29's counterexample is stated in terms of. The `½` is the same one.
    """
    top = torch.cat((breve_s, breve_st), dim=1)
    bottom = torch.cat((breve_st.transpose(0, 1), breve_t), dim=1)
    return 0.5 * torch.cat((top, bottom), dim=0)


class R29EntropyCoshBoundsTests(unittest.TestCase):
    """Eqs. 34/35: the two ends, and the collapse they exist to outlaw."""

    def test_entropy_cosh_bounds_are_the_two_ends_of_equation_thirty_five(
        self,
    ) -> None:
        """Eq. 35: `L_c = -ln 2`, `U_c = 2 ln cosh((H_t - H_s)/4)`.

        Four claims r29 attaches to that pair: the lower end is the constant
        `-ln 2`; the upper end is the stated `cosh` expression; `U_c >= 0`
        because `cosh >= 1`; and `cosh` being even, the ORDER of the two
        entropies does not change the number -- so the subtraction is written
        the way the equation writes it, target minus source, and a test reading
        `U_c` alone could not tell the orders apart. Asserting the symmetry is
        what makes that explicit rather than accidental.
        """
        self.assertAlmostEqual(entropy_cosh_bounds(0.0, 0.0)[0], -math.log(2.0),
                               places=15)

        for h_source, h_target in ((0.0, 0.0), (1.5, 1.5), (0.0, 3.0),
                                   (3.0, 0.0), (2.25, 0.75), (0.1, 4.9)):
            h_s = torch.tensor(h_source, dtype=DTYPE)
            h_t = torch.tensor(h_target, dtype=DTYPE)
            lower, upper = entropy_cosh_bounds(h_s, h_t)

            self.assertAlmostEqual(lower, -math.log(2.0), places=15)
            expected = 2.0 * math.log(math.cosh((h_target - h_source) / 4.0))
            self.assertAlmostEqual(float(upper), expected, places=12)
            # cosh >= 1, so the upper end never goes below zero.
            self.assertGreaterEqual(float(upper), 0.0)
            # And it is zero exactly when the two entropies agree.
            if h_source == h_target:
                self.assertAlmostEqual(float(upper), 0.0, places=15)
            else:
                self.assertGreater(float(upper), 0.0)
            # cosh is even: swapping the order returns the same number.
            self.assertAlmostEqual(
                float(entropy_cosh_bounds(h_t, h_s)[1]), float(upper), places=15
            )

    def test_balanced_dependency_score_within_the_entropy_cosh_bounds(self) -> None:
        """Eq. 34: `-ln 2 <= J_c <= 2 ln cosh((H_t - H_s)/4)`, every time.

        Over the real path -- blocks read off one Gram, weighted by Eq. 26, and
        balanced by Eq. 32 -- across cardinality pairs on both sides of
        `n_s = n_t` and several confidence levels, because the upper end is a
        function of the two marginal entropies and nothing here should depend on
        the two domains contributing the same number of bags.
        """
        for n_source, n_target in ((2, 9), (4, 7), (5, 5), (9, 2)):
            K_s, K_st, K_t = _joint_blocks(n_source, n_target)
            w = _confidences(n_target)
            for scale in (1.0, 0.3):
                K_st_w, K_t_w = weighted_blocks(K_st, K_t, w * scale)
                score = balanced_dependency_score(K_s, K_st_w, K_t_w)
                breve_s, breve_t, _ = balanced_mixture(K_s, K_st_w, K_t_w)
                lower, upper = entropy_cosh_bounds(
                    quadratic_entropy(breve_s), quadratic_entropy(breve_t)
                )
                self.assertGreaterEqual(float(score), lower - TOL)
                self.assertLessEqual(float(score), float(upper) + TOL)

    def test_the_global_loss_is_zero_exactly_at_alignment(self) -> None:
        """Eq. 35/36: perfect alignment gives `U_c = 0 = J_c`, so the loss is 0.

        r29: equality in the cross block's Cauchy-Schwarz bound holds "solo
        cuando las dos estructuras normalizadas coinciden; en ese caso las
        entropias marginales son iguales, `U_c = 0 = J_c`, y la perdida por
        clase (Ec. 36) vale `l_glob,c = 0`". So the configuration is identical
        normalized blocks with the cross block saturating, which is
        `K̆_s = K̆̃_t = K̆̃_st = A`.

        And the other half of "if and only if": a configuration that is NOT
        aligned reads strictly above zero. Without it the test cannot tell the
        loss from one that returns zero always.
        """
        for n in (3, 8):
            A = _density(torch.eye(n, dtype=DTYPE))
            mix = _assemble(A, A, A)
            self.assertAlmostEqual(float(torch.trace(mix)), 1.0, places=12)
            # PSD, so this really is the saturating configuration and not an
            # algebraic fiction outside the feasible set.
            self.assertGreaterEqual(
                float(torch.linalg.eigvalsh(mix).min()), -1e-12
            )
            h = quadratic_entropy(A)
            score = 0.5 * (h + h) - quadratic_entropy(mix)
            self.assertAlmostEqual(float(score), 0.0, places=12)
            self.assertAlmostEqual(
                float(entropy_cosh_bounds(h, h)[1]), 0.0, places=15
            )
            self.assertAlmostEqual(
                float(class_global_loss_cosh(score, h, h)), 0.0, places=12
            )

        # The other pole: a misaligned class is strictly above zero.
        K_s, K_st, K_t = _joint_blocks(4, 7)
        K_st_w, K_t_w = weighted_blocks(K_st, K_t, _confidences(7))
        breve_s, breve_t, _ = balanced_mixture(K_s, K_st_w, K_t_w)
        misaligned = class_global_loss_cosh(
            balanced_dependency_score(K_s, K_st_w, K_t_w),
            quadratic_entropy(breve_s),
            quadratic_entropy(breve_t),
        )
        self.assertGreater(float(misaligned), 0.01)

    def test_entropy_cosh_bounds_order_alignment_above_the_collapse(self) -> None:
        """Eq. 35's reason for existing: r29's own counterexample, at n_s = 43.

        **This is the test that carries change 2.** r29 states that `J_c` alone
        does not reach its maximum at alignment when the marginal entropies
        differ, and gives the configuration: a source with `K̆_s = I / n_s` and
        a target collapsed far away, in the limit where the cross block
        vanishes. Then `rho_s = 1/n_s`, `rho_t = 1`,
        `rho_mix = (1/4)(1 + 1/n_s)` and

            J_c = ln((1 + 1/n_s)/4) + (1/2) ln n_s,

        "positiva para `n_c^s` grande (`0.52` para `n_c^s = 43`)" -- above the
        `0` that perfect alignment scores. An affine map with CONSTANT ends
        preserves that order, so r25's loss rewards the collapse.

        Both poles are measured here, which is the whole point:

        * r25's map, whose ends are constants once the counts are fixed, gives
          the collapse a LOWER loss than perfect alignment. It prefers it.
        * r29's map, whose upper end moves with the marginal entropies, gives
          perfect alignment the lower loss -- zero, in fact.

        Both configurations are built at the same cardinalities, because r25's
        ends are functions of exactly those; comparing them at different counts
        would change the bounds instead of the configuration.
        """
        n_source = n_target = 43

        breve_s = _density(torch.eye(n_source, dtype=DTYPE))
        # rho_t = 1: all the mass on one direction, which is the collapse.
        collapsed = torch.ones((n_target, n_target), dtype=DTYPE) / n_target
        vanished = torch.zeros((n_source, n_target), dtype=DTYPE)

        rho_s = float((breve_s * breve_s).sum())
        rho_t = float((collapsed * collapsed).sum())
        self.assertAlmostEqual(rho_s, 1.0 / n_source, places=12)
        self.assertAlmostEqual(rho_t, 1.0, places=12)

        collapse_mix = _assemble(breve_s, vanished, collapsed)
        rho_mix = float((collapse_mix * collapse_mix).sum())
        self.assertAlmostEqual(rho_mix, 0.25 * (1.0 + 1.0 / n_source), places=12)

        h_s = quadratic_entropy(breve_s)
        h_t = quadratic_entropy(collapsed)
        collapse_score = 0.5 * (h_s + h_t) - quadratic_entropy(collapse_mix)

        # r29's own closed form, and its own figure.
        closed_form = math.log((1.0 + 1.0 / n_source) / 4.0) + 0.5 * math.log(n_source)
        self.assertAlmostEqual(float(collapse_score), closed_form, places=12)
        self.assertAlmostEqual(float(collapse_score), 0.52, places=2)
        self.assertGreater(float(collapse_score), 0.0)

        # Perfect alignment at the same cardinalities scores exactly zero.
        aligned = _density(torch.eye(n_source, dtype=DTYPE))
        aligned_mix = _assemble(aligned, aligned, aligned)
        h_aligned = quadratic_entropy(aligned)
        aligned_score = 0.5 * (h_aligned + h_aligned) - quadratic_entropy(aligned_mix)
        self.assertAlmostEqual(float(aligned_score), 0.0, places=12)
        # So the SCORE itself already prefers the collapse -- that is the defect.
        self.assertGreater(float(collapse_score), float(aligned_score))

        # Pole one: r25's constant ends carry that order into the loss.
        r25_collapse = _retired_r25_loss(collapse_score, n_source, n_target)
        r25_aligned = _retired_r25_loss(aligned_score, n_source, n_target)
        self.assertLess(r25_collapse, r25_aligned)
        # And by a margin no tolerance could explain away.
        self.assertGreater(r25_aligned - r25_collapse, 0.05)

        # Pole two: r29's entropy-dependent upper end reverses it.
        r29_collapse = float(class_global_loss_cosh(collapse_score, h_s, h_t))
        r29_aligned = float(
            class_global_loss_cosh(aligned_score, h_aligned, h_aligned)
        )
        self.assertLess(r29_aligned, r29_collapse)
        self.assertAlmostEqual(r29_aligned, 0.0, places=12)
        self.assertGreater(r29_collapse - r29_aligned, 0.05)

        # The collapse stays inside r29's interval, which is why the reversal is
        # a correction and not merely a different number.
        _, upper_29 = entropy_cosh_bounds(h_s, h_t)
        self.assertLessEqual(float(collapse_score), float(upper_29) + TOL)

    def test_cardinality_bounds_the_upper_end_without_being_attained(self) -> None:
        """Eq. 35: `U_c <= 2 ln cosh((1/4) ln max(n_s, n_t))`, never reached.

        r29 keeps cardinality only as "una observacion a priori": since
        `0 <= H_2 <= ln N`, the entropy difference is bounded by
        `ln max(n_s, n_t)`, so `U_c` is bounded by that constant -- which "no se
        alcanza para `max(n_s, n_t) >= 2`", because attaining it would need the
        maximal entropy difference AND equality in the cross block's
        Cauchy-Schwarz bound at once, and the latter forces coinciding
        structures and therefore EQUAL entropies. That is why it is not used to
        normalize, and asserting the gap is what keeps the reason on record.
        """
        for n_source, n_target in ((2, 9), (4, 7), (5, 5), (9, 2)):
            K_s, K_st, K_t = _joint_blocks(n_source, n_target)
            K_st_w, K_t_w = weighted_blocks(K_st, K_t, _confidences(n_target))
            breve_s, breve_t, _ = balanced_mixture(K_s, K_st_w, K_t_w)
            _, upper = entropy_cosh_bounds(
                quadratic_entropy(breve_s), quadratic_entropy(breve_t)
            )
            ceiling = 2.0 * math.log(
                math.cosh(0.25 * math.log(max(n_source, n_target)))
            )
            self.assertLessEqual(float(upper), ceiling + TOL)
            # Strictly below, which is the half that makes it unusable as a
            # normalizer rather than merely a looser choice.
            self.assertLess(float(upper), ceiling)

    def test_cosh_bounds_agree_across_the_two_packages(self) -> None:
        """The join `src/MIL_CREDA/` deliberately does not import across.

        `src/CREDA/bounds.py` carries its own `entropy_cosh_bounds` and
        `class_global_loss_cosh` for its r28 fork, and the dependency may only
        point that way: CREDA is the baseline MIL-CREDA is measured against, so
        a CREDA module importing `MIL_CREDA` would contaminate the comparison at
        the import line. Implementing Eq. 35 here rather than importing it keeps
        `global_term.py`'s provenance truthful -- it is the one place
        MIL-CREDA's Section 5 global term lives -- at the cost of two
        implementations that could drift.

        This test is what pays that cost: it crosses the join instead of
        assuming it. Testing each side against a fixture written by the same
        hand would verify both halves and never the connection.
        """
        for h_source, h_target in ((0.0, 0.0), (0.0, 3.0), (3.0, 0.0),
                                   (2.25, 0.75), (1.1, 4.4), (0.5, 0.5)):
            h_s = torch.tensor(h_source, dtype=DTYPE)
            h_t = torch.tensor(h_target, dtype=DTYPE)

            mine = entropy_cosh_bounds(h_s, h_t)
            theirs = creda_bounds.entropy_cosh_bounds(h_s, h_t)
            self.assertEqual(mine[0], theirs[0])
            self.assertTrue(torch.equal(mine[1], theirs[1]))

            for score in (-0.6, 0.0, 0.25, 0.52):
                self.assertTrue(
                    torch.equal(
                        class_global_loss_cosh(score, h_s, h_t),
                        creda_bounds.class_global_loss_cosh(score, h_s, h_t),
                    )
                )


# ------------------------------------- 3. Eq. 36, and the span that left it


class R29DetachedSpanTests(unittest.TestCase):
    """Eq. 36's denominator, and the two tests that were retired from here.

    **r29 read: "el denominador `U_c - L_c` se trata como una constante".**
    Two tests in this class held exactly that -- that the gradient of
    `class_global_loss_cosh` matches a reference dividing by
    `(U_c - L_c).detach()` and disagrees with one that leaves it attached, and
    that the VALUE equals `(U_c - J_c) / (U_c - L_c)`.

    **Both were retired on 2026-10-17, because r31 removed the span they are
    about.** r30 proved `0 <= U_c - J_c <= ln 2`; r31 therefore divides by the
    literal `ln 2` and drops the gradient exception, since a literal carries no
    gradient to remove. So the two tests did not go stale in the ordinary way
    -- they pinned a superseded claim about a CURRENT function, and both had
    been failing against it: the value test measured
    `0.4995570110847078` where r31 computes `0.4506251350130099`. Keeping them
    red would have reported a defect the formulation does not have; repairing
    them in place would have made them say what r31 says.

    **Nothing was lost by retiring them, and that is measured rather than
    asserted.** `tests/test_r31_ec36.py` carries the same two claims under r31,
    and carries them more strongly: `test_nothing_is_detached_because_there_is`
    `_nothing_to_detach` compares the measured gradient against r31's reference
    AND against BOTH of r29's -- detached and attached -- with a margin on each
    disagreement, and `test_the_denominator_is_ln_two_and_not_the_span` pins
    the value. Both are parametrized over `MIL_CREDA.global_term` and
    `CREDA.bounds`, where the retired pair read this package alone.

    What stays here is `test_the_gradient_reaches_the_blocks_through_the_whole`
    `_term`, which `test_r31_ec36.py` does not cover: it crosses the
    Eq. 32 -> Eq. 33 -> Eq. 36 join rather than testing Eq. 36 alone, and the
    join is the only thing the wiring is about.
    """

    def test_the_gradient_reaches_the_blocks_through_the_whole_term(self) -> None:
        """The same rule crossing the join, not stated about the function alone.

        `class_global_loss_cosh` tested alone proves the function; this proves
        Eq. 32 -> Eq. 33 -> Eq. 36 composes with a graph attached all the way
        back to the kernel blocks, which is the only thing the wiring is about.
        """
        n_source, n_target = 4, 7
        K_s, K_st, K_t = _joint_blocks(n_source, n_target)
        K_t = K_t.clone().requires_grad_(True)
        K_st_w, K_t_w = weighted_blocks(K_st, K_t, _confidences(n_target))
        breve_s, breve_t, _ = balanced_mixture(K_s, K_st_w, K_t_w)
        loss = class_global_loss_cosh(
            balanced_dependency_score(K_s, K_st_w, K_t_w),
            quadratic_entropy(breve_s),
            quadratic_entropy(breve_t),
        )
        loss.backward()
        self.assertIsNotNone(K_t.grad)
        self.assertGreater(float(K_t.grad.abs().max()), 0.0)
        self.assertTrue(bool(torch.isfinite(K_t.grad).all()))


# --------------------------------------------- 4. Eq. 37, the confidence gate


class R33AggregationTests(unittest.TestCase):
    """Eq. (37)'s weights are the observed source proportions, and an inactive
    class costs the maximum.

    **This class replaced `R29ConfidenceGateTests` on 2026-10-18, and it
    asserts the opposite of what that one did.** r29 introduced
    `omega_c = q_c / sum_c' n_c'^t` with `q_c = sum_{j in T_c} w_j^t`, and the
    three tests here measured it: that the weights were SUB-convex, that the
    absolute confidence level gated the term (every confidence at 0.1
    multiplied it by 0.1), and that `global_loss` accepted a sum below one. All
    three were true of r31.

    r32 removed the gate. Its reason, in the revision's own words, is that a
    weight based on target routing or on the confidences "dependeria de
    cantidades que el propio modelo determina, mientras que `n_c^s` esta fijado
    por los datos" -- a weight the model computes can be lowered instead of
    earned. `confidence_weights` went with the formula, so there is nothing
    left to gate and the three claims above are now false by construction.
    """

    def test_the_confidence_gate_is_gone(self) -> None:
        """`confidence_weights` is not importable, and `global_loss` cannot be
        handed weights at all -- it takes counts.

        Asserted as an absence because that is what a deleted test cannot
        check: a gate that came back, or a `global_loss` that accepted a
        weight vector again, would silently reintroduce the channel r32 closed.
        """
        from MIL_CREDA import global_term
        self.assertFalse(hasattr(global_term, 'confidence_weights'))
        self.assertEqual(
            global_loss.__code__.co_varnames[:2],
            ('class_losses', 'source_counts'))
        self.assertNotIn('weights', global_loss.__code__.co_varnames)

    def test_the_weights_are_the_source_proportions_and_sum_to_one(self) -> None:
        """Convex, not sub-convex: `sum_c n_c^s / N_s` is one identically, so
        no caller can scale the term the way r29's shortfall did."""
        losses = [0.1, 0.3, 0.5]
        balanced = float(global_loss(losses, [5, 5, 5]))
        self.assertAlmostEqual(balanced, 0.3, places=12)

        # The term follows the SOURCE evidence, which is observed.
        self.assertLess(float(global_loss(losses, [50, 1, 1])), balanced)
        self.assertGreater(float(global_loss(losses, [1, 1, 50])), balanced)

        # r29's gate, measured on the same losses: every confidence at 0.1 took
        # the term to a tenth. Nothing here can do that any more, because the
        # caller hands counts and scaling them changes nothing.
        self.assertAlmostEqual(float(global_loss(losses, [50, 50, 50])),
                               balanced, places=12)

    def test_an_inactive_class_costs_the_maximum(self) -> None:
        """`l~_c = 1` for `c` not in `C_act`, so abandoning a class raises the
        term -- where r31 dropped it from the sum and the term FELL."""
        losses = [0.1, 0.2, 0.3]
        counts = [4, 4, 4]
        self.assertAlmostEqual(float(global_loss(losses, counts)), 0.2, places=12)

        abandoned = float(global_loss([0.1, 0.2, None], counts))
        self.assertAlmostEqual(abandoned, (0.1 + 0.2 + 1.0) / 3, places=12)
        self.assertGreater(abandoned, 0.2)
        # What r31 would have reported for the same routing, and why the
        # direction is the whole point: fewer terms, each in [0, 1].
        self.assertLess((0.1 + 0.2) / 2, 0.2)

        self.assertAlmostEqual(float(global_loss([None] * 3, counts)),
                               1.0, places=12)

    def test_the_charge_carries_no_gradient(self) -> None:
        """Eq. (22)'s routing is an `argmax`, so there is nothing to
        differentiate; the charge's job is the value."""
        active = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
        value = global_loss([active, None], [3, 1])
        value.backward()
        self.assertAlmostEqual(float(value), 0.4 * 0.75 + 0.25, places=12)
        self.assertAlmostEqual(float(active.grad), 0.75, places=12)


class R29WiringTests(unittest.TestCase):
    """`_milcreda_term` is the single consumer, and it must have moved too.

    Every claim above is about `global_term.py`. A module can hold r29's
    mathematics perfectly while the one place that runs the per-class loop keeps
    calling r25's functions, and nothing read separately would say so: the
    module declares Section 5, the arm declares Section 5, and both documents
    come back clean. So this crosses the join -- it asks the wiring what it
    actually computed and holds the answer against BOTH paths built by hand.
    """

    def setUp(self) -> None:
        self._real_extractor = wiring.FeatureExtractor
        wiring.FeatureExtractor = _StubEncoder
        torch.manual_seed(11)
        self.arm = wiring.build("G", config.CLASSES, _pool(1), _pool(2))

    def tearDown(self) -> None:
        wiring.FeatureExtractor = self._real_extractor

    def _run(self):
        x = self.arm.source.take(torch.arange(config.BAGS_PER_STEP))
        y = self.arm.source.labels[: config.BAGS_PER_STEP]
        embeddings = self.arm.instance_embeddings(x)
        target = self.arm.target.take(
            self.arm._draw_target(torch.Generator().manual_seed(3))
        )
        global_term, _ = self.arm._milcreda_term(embeddings, y, target)
        return embeddings, y, target, global_term

    def test_the_consumer_computes_the_r29_path_and_not_the_r25_one(self) -> None:
        """Both paths rebuilt from the arm's own blocks, and only one matches.

        The r25 path is rebuilt here from `_retired_r25_score` and
        `_retired_r25_loss` -- local reconstructions of what `dependency_score`,
        `conservative_bounds` and `class_global_loss` computed before they were
        removed from `src/` on 2026-10-18 -- and asserted NOT to be what the
        wiring produces. Without that half, a wiring that never moved would
        pass: the r33 reconstruction would simply be a second wrong number
        nobody compared against anything. Both paths now go through r33's
        Eq. (37) aggregation, so the comparison isolates the MAP, which is what
        it is about.
        """
        embeddings, y, target, measured = self._run()

        # Rebuild both paths from the same blocks the arm just used.
        H_t = self.arm.target_embeddings(target)
        sigma = self.arm.sigma
        bags_s = self.arm.bags_of(embeddings, sigma)
        bags_t = self.arm.bags_of(H_t, sigma)
        kernel_s = self.arm._kernel_pairs(bags_s)
        kernel_t = self.arm._kernel_pairs(bags_t)
        K_ss = bag_kernel_matrix(kernel_s, kernel_s, sigma)
        K_st = bag_kernel_matrix(kernel_s, kernel_t, sigma)
        K_tt = bag_kernel_matrix(kernel_t, kernel_t, sigma)
        Z_t = torch.stack([bag_embedding(H, w) for H, w in bags_t])
        G_t = torch.nn.functional.softmax(self.arm.head(Z_t), dim=1)
        w_t = confidences(G_t)
        pseudo = torch.tensor([pseudolabel(g) for g in G_t], device=K_ss.device)

        # r33 Eq. (37) sums over EVERY class, so both reconstructions carry one
        # entry per class with `None` for the inactive ones, and the source
        # counts come along as the weights.
        r33_losses, r25_losses, source_counts = [], [], []
        for class_id in range(self.arm.classes):
            s_idx = torch.nonzero(y == class_id, as_tuple=False).reshape(-1)
            t_idx = torch.nonzero(pseudo == class_id, as_tuple=False).reshape(-1)
            source_counts.append(int(s_idx.numel()))
            if s_idx.numel() == 0 or t_idx.numel() == 0:
                r33_losses.append(None)
                r25_losses.append(None)
                continue
            K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, s_idx, t_idx)
            K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, w_t[t_idx])
            K_mix = mixed_matrix(K_s_c, K_st_w, K_t_w)
            if not is_active(K_s_c, K_t_w, K_mix):
                r33_losses.append(None)
                r25_losses.append(None)
                continue
            breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_w, K_t_w)
            r33_losses.append(
                class_global_loss_cosh(
                    balanced_dependency_score(K_s_c, K_st_w, K_t_w),
                    quadratic_entropy(breve_s),
                    quadratic_entropy(breve_t),
                )
            )
            # r25's score and map, reconstructed locally -- both functions
            # were removed from `src/` on 2026-10-18. `_retired_r25_loss`
            # carries the two ends; the score is r25's own contrast over the
            # trace-normalized blocks, which is what `dependency_score`
            # computed.
            r25_losses.append(torch.tensor(
                _retired_r25_loss(
                    _retired_r25_score(K_s_c, K_t_w, K_mix),
                    int(s_idx.numel()), int(t_idx.numel())),
                dtype=DTYPE))

        # The loop has to have found something, or neither comparison means
        # anything and both reconstructions are zero.
        self.assertGreater(sum(1 for v in r33_losses if v is not None), 0)

        r33 = float(global_loss(r33_losses, source_counts).detach())
        r25 = float(global_loss(r25_losses, source_counts).detach())
        self.assertAlmostEqual(float(measured.detach()), r33, places=10)
        self.assertNotAlmostEqual(float(measured.detach()), r25, places=4)
        # And the two paths have to genuinely disagree here, or "it is not r25"
        # is a statement about rounding.
        self.assertGreater(abs(r33 - r25), 1e-3)

    def test_the_consumer_still_reports_its_diagnostics(self) -> None:
        """`activeClasses` counts the non-`None` entries, and the rest is
        untouched.

        The per-class loop is where the diagnostics are read off, so a rewrite
        of that loop is exactly the change that can drop one silently.
        """
        self._run()
        diagnostics = self.arm._diagnostics
        for key in (
            "targetBagEntropy",
            "targetConfidenceMean",
            "targetConfidenceMin",
            "activeClasses",
            "sourceKernelMean",
            "crossKernelMean",
            "targetKernelMean",
            "targetDispersion",
        ):
            self.assertIn(key, diagnostics)
        self.assertGreater(diagnostics["activeClasses"], 0)
        self.assertLessEqual(diagnostics["activeClasses"], config.CLASSES)

    def test_the_gate_only_reaches_the_arms_that_declare_the_weighting(self) -> None:
        """`weighting` is the ladder's confidence axis, and `E` declares it off.

        `config.ARMS` declares `E` (`MIL-CREDA**`) as the arm that lacks the
        local term AND the weighting, and Eq. 37's weights are built from the
        same `w_j^t` (Eq. 24) that Eq. 26's `D` is. So an arm that declares no
        weighting leaves Eq. (26)'s blocks RAW, and an arm that declares it
        weights them.

        **The axis stopped reaching the aggregation on 2026-10-18.** r29 built
        Eq. (37)'s weights from the same `w_j^t` as Eq. (26)'s `D`, so this call
        site sat behind `spec["weighting"]` too -- gating `E` there would have
        left it carrying confidence through the aggregation while declaring it
        does not. r32 replaced those weights with the observed source
        proportions, which have nothing to do with the confidences, so the
        aggregation is now ungated and the axis means exactly one thing again:
        Eq. (26)'s `D`. That is what this test now isolates.

        Measured rather than read off the source: the same step is run on `E`
        and on `F`, which differ in exactly that flag, and `E`'s term is
        asserted to equal its own reconstruction from UNWEIGHTED blocks -- put
        through the same Eq. (37) aggregation the weighted arms use.
        """
        self.assertFalse(config.ARMS_BY_ID["E"]["weighting"])
        self.assertTrue(config.ARMS_BY_ID["F"]["weighting"])

        torch.manual_seed(11)
        arm_e = wiring.build("E", config.CLASSES, _pool(1), _pool(2))
        x = arm_e.source.take(torch.arange(config.BAGS_PER_STEP))
        y = arm_e.source.labels[: config.BAGS_PER_STEP]
        embeddings = arm_e.instance_embeddings(x)
        target = arm_e.target.take(
            arm_e._draw_target(torch.Generator().manual_seed(3))
        )
        term, _ = arm_e._milcreda_term(embeddings, y, target)

        # `E` weights no block, so its per-class losses come off the unweighted
        # blocks and its aggregation is the plain mean over active classes.
        H_t = arm_e.target_embeddings(target)
        sigma = arm_e.sigma
        kernel_s = arm_e._kernel_pairs(arm_e.bags_of(embeddings, sigma))
        bags_t = arm_e.bags_of(H_t, sigma)
        kernel_t = arm_e._kernel_pairs(bags_t)
        K_ss = bag_kernel_matrix(kernel_s, kernel_s, sigma)
        K_st = bag_kernel_matrix(kernel_s, kernel_t, sigma)
        K_tt = bag_kernel_matrix(kernel_t, kernel_t, sigma)
        Z_t = torch.stack([bag_embedding(H, w) for H, w in bags_t])
        G_t = torch.nn.functional.softmax(arm_e.head(Z_t), dim=1)
        pseudo = torch.tensor([pseudolabel(g) for g in G_t], device=K_ss.device)

        losses, source_counts = [], []
        for class_id in range(arm_e.classes):
            s_idx = torch.nonzero(y == class_id, as_tuple=False).reshape(-1)
            t_idx = torch.nonzero(pseudo == class_id, as_tuple=False).reshape(-1)
            source_counts.append(int(s_idx.numel()))
            if s_idx.numel() == 0 or t_idx.numel() == 0:
                losses.append(None)
                continue
            K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, s_idx, t_idx)
            if not is_active(K_s_c, K_t_c, mixed_matrix(K_s_c, K_st_c, K_t_c)):
                losses.append(None)
                continue
            breve_s, breve_t, _ = balanced_mixture(K_s_c, K_st_c, K_t_c)
            losses.append(
                class_global_loss_cosh(
                    balanced_dependency_score(K_s_c, K_st_c, K_t_c),
                    quadratic_entropy(breve_s),
                    quadratic_entropy(breve_t),
                )
            )

        self.assertGreater(sum(1 for v in losses if v is not None), 0)
        self.assertAlmostEqual(
            float(term.detach()),
            float(global_loss(losses, source_counts).detach()), places=10)


if __name__ == "__main__":  # pragma: no cover - parity with the suite's runner
    unittest.main()
