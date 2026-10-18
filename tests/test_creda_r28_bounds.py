"""r28's three changes to the sample-level global term, and what each one buys.

r25 normalized each class's score by bounds that are functions of CARDINALITY
alone -- `L_c = -ln(n_s + n_t)`, `U_c = (ln n_s + ln n_t) / 2`. r28 replaces
both ends and the balance that feeds them. Three changes land here:

1. **One bandwidth for every class.** r28 line 732: a single sigma governs the
   three blocks, because the three derive from the same instance kernel. The
   toy's phase-2 search is no longer allowed to pick a per-class median.
2. **The mix balances by TRACE, not by cardinality** (Eq. 32). The Gaussian
   kernel has 1 on its diagonal, so `tr(K_s) = n_s` and the source side does
   not move; but `K~_t` arrives already weighted (`D K_t D`), so its diagonal
   is `w_j^2` and `tr(K~_t) = sum_j w_j^2`, which equals `n_t` only when every
   weight is exactly 1. Balancing by cardinality therefore drops the confidence
   weighting from the mass each domain contributes to the mixture. That is the
   defect `balance_mass='trace'` repairs.
3. **The normalization bounds are replaced** (Eq. 35): `L_c = -ln 2` and
   `U_c = 2 ln cosh((H_2(K~_t) - H_2(K_s)) / 4)`, both evaluated on the SAME
   normalized matrices that build `J_c`. `bounds_form='entropy-cosh'` selects
   them, and Eq. 36 keeps its form with the span detached.

Plus the active-class rule r28 §5 states: a class is active when both domains
contribute AND the three matrices that must be normalized have positive trace.
`forward` used to check only the counts.

**Both new keywords are opt-in VALUES, never changed defaults.** The
`CREDALoss` contract that `tests/test_creda_sample.py` holds -- every flag at
its default leaves the class bit-identical to what it was before the flag
existed -- is held here too, against values read off the unedited code.

The one place r28 changes the DEFAULT path is the active-class trace condition:
a class whose target trace has collapsed used to be divided by anyway, and the
1e-6 epsilon inside `_renyi_entropy_order_2` turned that into a finite number
rather than a NaN. r28 says the score is not defined there, so the class is now
skipped under every setting. Nothing else moves.

Every test here was observed RED against the unedited tree before the code was
written; the file is new precisely so that nothing in the 50k-line
`test_creda_sample.py` could lose a test to a name it already uses.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import math
import unittest

import torch

from CREDA import models
from CREDA.bounds import (
    class_global_loss_cosh,
    entropy_cosh_bounds,
)
from CREDA.models import CREDALoss

CLASSES = 3
DIM = 6
#: 43 per class, which is the cardinality every measurement already quoted for
#: this term was taken at -- the 8.2155-wide conservative span, the 5.1%
#: occupancy, and r28's own counterexample at `n_s = 43`.
PER_CLASS = 43
SEED = 20261009

#: `-ln 2`, Eq. 35's lower end. Written out rather than imported so that a
#: change to the module's constant does not silently change what is asserted.
LOWER_END = -math.log(2.0)
#: r31 Eq. (36)'s denominator. Numerically `-LOWER_END` and a different
#: claim: `LOWER_END` bounds `J_c` (Eq. 34), this bounds `U_c - J_c`.
DENOMINATOR = math.log(2.0)


# --------------------------------------------------------------- the fixtures


def _blobs(seed: int = SEED, per: int = PER_CLASS, spread: float = 12.0,
           dtype: torch.dtype = torch.float64):
    """Three Gaussian blobs, one per class, at a fixed separation."""
    generator = torch.Generator().manual_seed(seed)
    centres = torch.zeros(CLASSES, DIM, dtype=dtype)
    centres[1, 0] = spread
    centres[2, 1] = spread
    features = [centres[c] + torch.randn(per, DIM, generator=generator, dtype=dtype)
                for c in range(CLASSES)]
    labels = [torch.full((per,), c, dtype=torch.long) for c in range(CLASSES)]
    return torch.cat(features), torch.cat(labels)


def _one_hot(labels: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, CLASSES).to(dtype)


def _configurations() -> dict[str, tuple[torch.Tensor, ...]]:
    """The five configurations the two maps are read on, side by side.

    Each is `(f_source, f_target, y_source, g_target)`, every pseudo-label
    one-hot so the confidence weights are 1 and the only thing that differs
    between configurations is the geometry.
    """
    f_s, y_s = _blobs()
    total = CLASSES * PER_CLASS
    offset = torch.zeros(DIM, dtype=torch.float64)
    offset[2] = 40.0
    noise_generator = torch.Generator().manual_seed(SEED + 1)
    collapsed = f_s.mean(0).clone()
    collapsed[3] += 40.0
    # A cyclic relabelling of the target: every class's target block is another
    # class's source block, so each class is misaligned without being far away.
    permuted_order = torch.argsort((y_s + 1) % CLASSES, stable=True)
    return {
        'perfect': (f_s, f_s.clone(), y_s, _one_hot(y_s)),
        'separated': (f_s, f_s + offset, y_s, _one_hot(y_s)),
        'permuted': (f_s, f_s[permuted_order].clone(), y_s, _one_hot(y_s)),
        'noise': (f_s, torch.randn(total, DIM, generator=noise_generator,
                                   dtype=torch.float64) * 6.0, y_s, _one_hot(y_s)),
        'collapsed': (f_s, collapsed.expand(total, DIM).clone(), y_s, _one_hot(y_s)),
    }


def _loss_for(form: str, **overrides) -> CREDALoss:
    """The r28 wiring, with one keyword switched between the two maps.

    `batch-source` rather than a per-class median: the bandwidth has to be
    anchored to the frozen side, or displacing the target grows sigma with it
    and the kernel's own unit of measurement moves with the thing it measures.
    """
    keywords = dict(sigma='auto', sigma_from='batch-source',
                    weight_cross_block=True, balance_mix_blocks=True,
                    balance_mass='trace', normalize_by_bounds=True,
                    bounds_form=form)
    keywords.update(overrides)
    return CREDALoss(**keywords)


def _read(form: str, configuration, **overrides) -> float:
    f_s, f_t, y_s, g_t = configuration
    with torch.no_grad():
        return float(_loss_for(form, **overrides)(f_s, f_t, y_s, g_t))


def _read_retired_r25(configuration, **overrides) -> float:
    """r25's Eq. (35)/(36) over the same raw score, reconstructed HERE.

    `bounds_form='conservative'` read this until 2026-10-18, when
    `conservative_bounds` and `class_global_loss` were removed from both
    packages: r33 states one pair of ends and r25's cardinality pair is nowhere
    in the formulation, so code for it was a module declaring a revision it no
    longer held.

    The comparison those three functions served is the EVIDENCE for that
    removal -- that r25's map compressed every configuration into a narrow band
    and ordered a collapsed target above perfect alignment -- so it is
    reconstructed as a test-local reader rather than left as a sentence. Two
    lines, both traceable to `proposals/research-concept-r25.md`:
    `L_c = -ln(n_c^s + n_c^t)`, `U_c = (1/2)(ln n_c^s + ln n_c^t)`, and
    `l_glob,c = (U_c - J_c) / (U_c - L_c)`. It is a reader for a retired map
    and must never be used to assert anything about the current one.
    """
    f_s, f_t, y_s, g_t = configuration
    with torch.no_grad():
        raw = _loss_for('entropy-cosh', normalize_by_bounds=False,
                        **overrides)(f_s, f_t, y_s, g_t, reduction='none')
        pseudo = torch.argmax(g_t, dim=1)
        values = []
        for index, c in enumerate(torch.unique(y_s, sorted=True)):
            n_s = int((y_s == c).sum())
            n_t = int((pseudo == c).sum())
            if n_s == 0 or n_t == 0:
                continue
            lower = -math.log(n_s + n_t)
            upper = 0.5 * (math.log(n_s) + math.log(n_t))
            j_c = -float(raw[index])
            values.append((upper - j_c) / (upper - lower))
        return sum(values) / len(values)


def _counterexample(n_source: int = PER_CLASS):
    """r28 §5's own counterexample, built so that `K_s` really is near `I`.

    The construction needs `K^s_c = I / n_s` -- the source maximally spread
    against the bandwidth -- and a target collapsed far enough away that the
    cross block vanishes. A median-derived sigma cannot produce that by
    construction (half the source pairs sit at one sigma, so the off-diagonal
    is about `exp(-1/2)`), which is why the bandwidth here is handed in.

    Returns the aligned and the collapsed configuration over the same source,
    so the only difference between the two readings is the target.
    """
    generator = torch.Generator().manual_seed(SEED + 2)
    f_s = torch.randn(n_source, DIM, generator=generator, dtype=torch.float64) * 50.0
    y_s = torch.zeros(n_source, dtype=torch.long)
    far = torch.zeros(DIM, dtype=torch.float64)
    far[0] = 5000.0
    g_t = torch.zeros(n_source, CLASSES, dtype=torch.float64)
    g_t[:, 0] = 1.0
    aligned = (f_s, f_s.clone(), y_s, g_t)
    collapsed = (f_s, far.expand(n_source, DIM).clone(), y_s, g_t)
    return aligned, collapsed


def _blocks(loss: CREDALoss, f_s: torch.Tensor, f_t: torch.Tensor,
            weights: torch.Tensor, sigma: torch.Tensor | float):
    """The three blocks as `forward` builds them, weighting included."""
    K_s = loss._gaussian_kernel(f_s, f_s, sigma)
    K_t = loss._gaussian_kernel(f_t, f_t, sigma)
    K_st = loss._gaussian_kernel(f_s, f_t, sigma)
    K_t = K_t * torch.outer(weights, weights)
    K_st = K_st * weights.unsqueeze(0)
    return K_s, K_t, K_st


# ------------------------------------------------------- 1. Eq. 35's two ends


class R28EntropyCoshBoundsTests(unittest.TestCase):
    """`entropy_cosh_bounds` against Eq. 35, end by end."""

    def test_the_lower_end_is_exactly_minus_log_two(self) -> None:
        """`L_c = -ln 2`, and it does not depend on either entropy.

        r25's lower end was `-ln(n_s + n_t)`, which moved with the counts; this
        one is a constant, so the test that matters is that nothing it is
        handed can move it.
        """
        for h_source, h_target in ((0.0, 0.0), (3.76, 0.0), (0.0, 3.76),
                                   (1.5, 1.5), (12.0, 0.25)):
            lower, _ = entropy_cosh_bounds(torch.tensor(h_source, dtype=torch.float64),
                                           torch.tensor(h_target, dtype=torch.float64))
            self.assertEqual(float(lower), LOWER_END)

    def test_the_upper_end_is_exactly_zero_when_the_entropies_agree(self) -> None:
        """`cosh(0) = 1`, so `U_c = 2 ln 1 = 0` -- the perfect-alignment case."""
        for value in (0.0, 0.5, 1.0, 3.7611):
            entropy = torch.tensor(value, dtype=torch.float64)
            _, upper = entropy_cosh_bounds(entropy, entropy.clone())
            self.assertEqual(float(upper), 0.0)

    def test_the_upper_end_is_strictly_positive_whenever_they_differ(self) -> None:
        """`cosh >= 1` with equality only at zero, so `U_c > 0` off the diagonal."""
        for h_source, h_target in ((0.0, 0.1), (3.76, 0.0), (1.0, 2.0), (2.0, 1.0)):
            _, upper = entropy_cosh_bounds(torch.tensor(h_source, dtype=torch.float64),
                                           torch.tensor(h_target, dtype=torch.float64))
            self.assertGreater(float(upper), 0.0)

    def test_the_upper_end_is_symmetric_in_the_sign_of_the_difference(self) -> None:
        """`cosh` is even, so swapping the two entropies cannot move `U_c`.

        The ORDER still matters and is asserted elsewhere -- Eq. 35 writes
        target minus source -- but it cannot be read off `U_c`, which is why
        this test says only what it can see.
        """
        for h_source, h_target in ((0.0, 3.76), (1.0, 2.5), (0.25, 12.0)):
            a = torch.tensor(h_source, dtype=torch.float64)
            b = torch.tensor(h_target, dtype=torch.float64)
            _, forward_order = entropy_cosh_bounds(a, b)
            _, reversed_order = entropy_cosh_bounds(b, a)
            self.assertEqual(float(forward_order), float(reversed_order))

    def test_entropy_cosh_bounds_are_equation_thirty_fives_two_ends(self) -> None:
        """The invariant `CREDA.bounds` declares: both ends, written out by hand.

        Hand-written so a typo in the 4, the 2, or the order of the subtraction
        is caught -- r28 writes TARGET entropy minus source, and `cosh` being
        even means `U_c` alone cannot reveal the order, so the asymmetric case
        is checked against a closed form rather than against itself.
        """
        for h_s_value, h_t_value in ((3.761157, 0.0), (0.0, 3.761157),
                                     (1.25, 0.5), (0.0, 0.0)):
            with self.subTest(h_source=h_s_value, h_target=h_t_value):
                lower, upper = entropy_cosh_bounds(
                    torch.tensor(h_s_value, dtype=torch.float64),
                    torch.tensor(h_t_value, dtype=torch.float64))
                self.assertEqual(float(lower), -math.log(2.0))
                expected = 2.0 * math.log(
                    math.cosh((h_t_value - h_s_value) / 4.0))
                self.assertAlmostEqual(float(upper), expected, places=12)


# --------------------------------------- 2. zero at perfect alignment, Eq. 36


class R28NormalizedLossTests(unittest.TestCase):
    """What the two maps read, and r28's own claim about the new one."""

    def test_perfect_alignment_reads_zero_under_the_new_map(self) -> None:
        """r28 §5: the loss is zero if and only if the two structures coincide.

        Identical features and every weight 1, so `h_s = h_t`, `U_c = 0`, and
        the cross block saturates Cauchy-Schwarz: `J_c = 0` and the affine map
        sends it to zero. The tolerance is the 1e-6 epsilon that
        `_renyi_entropy_order_2` adds to its trace and its logarithm, which is
        the only thing between this and an exact zero.
        """
        value = _read('entropy-cosh', _configurations()['perfect'])
        self.assertAlmostEqual(value, 0.0, places=5)

    def test_two_separated_clouds_read_the_top_of_the_new_interval(self) -> None:
        """Both ends are reachable now, which is the whole point of the change.

        A target that is a rigid translate of the source has an identical
        kernel matrix, so `h_s = h_t` and `U_c = 0`; the cross block vanishes,
        so `J_c -> -ln 2 = L_c` and the loss goes to 1. Under the conservative
        map the same configuration reads about 0.54, barely above where perfect
        alignment reads, which is the defect being repaired -- so this test
        fails loudly on the old map rather than drifting.
        """
        configuration = _configurations()['separated']
        new = _read('entropy-cosh', configuration)
        old = _read_retired_r25(configuration)
        self.assertGreater(new, 0.9)
        self.assertLess(old, 0.65)

    def test_the_new_map_spans_what_the_old_one_compressed(self) -> None:
        """Across the five configurations, the range each map actually covers.

        The conservative span is 8.2155 wide at this cardinality while the raw
        score lives in about 0.69 of it, so every configuration lands in a
        narrow band around 0.5. That measurement is why r28 replaced the
        equation; it is asserted here rather than left in a comment.
        """
        configurations = _configurations()
        new = {name: _read('entropy-cosh', configuration)
               for name, configuration in configurations.items()}
        old = {name: _read_retired_r25(configuration)
               for name, configuration in configurations.items()}
        self.assertLess(max(old.values()) - min(old.values()), 0.15)
        self.assertGreater(max(new.values()) - min(new.values()), 0.9)
        for name, value in new.items():
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)

    def test_the_new_map_prefers_alignment_to_r28s_counterexample(self) -> None:
        """r28 §5's counterexample, as a regression test.

        With `K_s = I / n_s` and a target collapsed far away so the cross block
        vanishes, `J_c = ln((1 + 1/n_s)/4) + (1/2) ln n_s`, which is +0.52 at
        `n_s = 43` -- ABOVE the 0 of perfect alignment. An affine map with
        CONSTANT ends preserves that order, so it reads the collapse as better
        than the alignment. `U_c` depending on the two marginal entropies is
        what removes it.

        Both readings are recorded, not only the new one: the conservative
        number is the evidence that the old map really does invert the order,
        and a test that asserted only the new map could not tell a repair from
        a configuration that was never broken.
        """
        aligned, collapsed = _counterexample()
        new_aligned = _read('entropy-cosh', aligned, sigma=1.0)
        new_collapsed = _read('entropy-cosh', collapsed, sigma=1.0)
        old_aligned = _read_retired_r25(aligned, sigma=1.0)
        old_collapsed = _read_retired_r25(collapsed, sigma=1.0)
        # The closed form, as r28 writes it, against the score the code builds.
        expected_score = math.log((1 + 1 / PER_CLASS) / 4) + 0.5 * math.log(PER_CLASS)
        measured_score = -_read('entropy-cosh', collapsed, sigma=1.0,
                                normalize_by_bounds=False)
        self.assertAlmostEqual(measured_score, expected_score, places=3)
        self.assertGreater(expected_score, 0.0)
        # The new map punishes the collapse; the old one rewards it.
        self.assertGreater(new_collapsed, new_aligned + 0.1)
        self.assertLess(old_collapsed, old_aligned - 0.05)


# ------------------------------------------------------- 3. Eq. 32's balance


class R28TraceBalanceTests(unittest.TestCase):
    """`balance_mass`, and the one thing that separates the two values."""

    def _blocks_at(self, weights: torch.Tensor):
        loss = CREDALoss(balance_mix_blocks=True)
        generator = torch.Generator().manual_seed(SEED + 3)
        f_s = torch.randn(20, DIM, generator=generator, dtype=torch.float64)
        f_t = torch.randn(len(weights), DIM, generator=generator,
                          dtype=torch.float64) + 1.3
        sigma = loss._median_pairwise(f_s)
        return _blocks(loss, f_s, f_t, weights, sigma)

    def test_the_two_balances_agree_exactly_when_every_weight_is_one(self) -> None:
        """The equality half: the new path is the same construction.

        With every weight exactly 1 the Gaussian kernel keeps its unit
        diagonal, so `tr(K~_t) = n_t` exactly and the two divisors are the same
        number. `torch.equal`, not `allclose`: the claim is identity.
        """
        weights = torch.ones(14, dtype=torch.float64)
        K_s, K_t, K_st = self._blocks_at(weights)
        self.assertEqual(float(torch.trace(K_t)), float(K_t.shape[0]))
        by_cardinality = CREDALoss(balance_mix_blocks=True,
                                   balance_mass='cardinality')
        by_trace = CREDALoss(balance_mix_blocks=True, balance_mass='trace')
        self.assertTrue(torch.equal(by_cardinality._mix_kernel_concat(K_s, K_t, K_st),
                                    by_trace._mix_kernel_concat(K_s, K_t, K_st)))

    def test_the_two_balances_differ_by_a_real_margin_once_weights_bite(self) -> None:
        """The difference half: the weighting now reaches the mixture.

        `tr(K~_t) = sum_j w_j^2`, so with ordinary confidences the target block
        carries well under `n_t` of mass and dividing by `n_t` hands the
        mixture a block that was never rescaled to the weight it earned. The
        assertion is a magnitude on the score, not an inequality: a `!=` would
        pass on a typo that moved the number by 1e-12.
        """
        generator = torch.Generator().manual_seed(SEED + 4)
        weights = torch.rand(14, generator=generator, dtype=torch.float64) * 0.8 + 0.1
        K_s, K_t, K_st = self._blocks_at(weights)
        trace = float(torch.trace(K_t))
        self.assertLess(trace, 0.6 * K_t.shape[0])
        reference = CREDALoss(balance_mix_blocks=True)
        scores = {}
        for mode in ('cardinality', 'trace'):
            loss = CREDALoss(balance_mix_blocks=True, balance_mass=mode)
            K_mix = loss._mix_kernel_concat(K_s, K_t, K_st)
            scores[mode] = float(
                0.5 * (reference._renyi_entropy_order_2(K_s)
                       + reference._renyi_entropy_order_2(K_t))
                - reference._renyi_entropy_order_2(K_mix))
        self.assertGreater(abs(scores['trace'] - scores['cardinality']), 0.05)

    def test_the_balance_is_read_only_when_the_blocks_are_balanced_at_all(self) -> None:
        """`balance_mass` is read under `balance_mix_blocks`, and nowhere else."""
        K_s, K_t, K_st = self._blocks_at(torch.full((14,), 0.4, dtype=torch.float64))
        plain = CREDALoss(balance_mix_blocks=False, balance_mass='trace')
        expected = torch.cat([torch.cat([K_s, K_st], dim=1),
                              torch.cat([K_st.T, K_t], dim=1)], dim=0)
        self.assertTrue(torch.equal(plain._mix_kernel_concat(K_s, K_t, K_st), expected))


# ------------------------------------------------ 4. Eq. 36's detached span


class R28DetachedSpanTests(unittest.TestCase):
    """r28: the denominator is a constant in each step; the numerator is not."""

    @staticmethod
    def _pieces(leaf: torch.Tensor):
        """Three differentiable functions of one leaf: `h_s`, `h_t` and `J_c`."""
        h_source = (leaf ** 2).sum() * 0.3 + 0.5
        h_target = (leaf ** 3).sum() * 0.2 + 1.1
        score = leaf.sum() * 0.1
        return h_source, h_target, score

    @staticmethod
    def _gradient(leaf_value, build) -> torch.Tensor:
        leaf = leaf_value.clone().requires_grad_(True)
        build(leaf).backward()
        return leaf.grad.clone()

    def test_class_global_loss_cosh_divides_by_the_constant_span(self) -> None:
        """r31 Eq. (36): `(U_c - J_c) / ln 2`, against three references.

        **This test asserted the opposite through r29 and the change is the
        point.** It was `test_class_global_loss_cosh_detaches_only_the_span`,
        and it held this implementation equal to a reference dividing by
        `(U_c - L_c).detach()` and unequal to one dividing by the attached
        span. r30 proved `0 <= U_c - J_c <= ln 2`, so r31 divides by that
        constant and r31's backpropagation section drops the gradient
        exception r29 carried -- a literal has nothing to detach.

        Three references, because one cannot settle it. The gradient must
        equal the `/ ln 2` form and differ from BOTH r29 forms: asserting only
        against `/ ln 2` would pass on a tree that had merely dropped the
        `.detach()` while keeping the span in the denominator.

        Moved rather than deleted, and the reason is written down: a test that
        pins a superseded revision is the record of that revision, so the
        superseded assertions survive here as the two references this one is
        held UNEQUAL to.

        **The leaf value is chosen for discrimination, measured.** `_pieces` is
        a synthetic differentiable triple rather than a feasible `(U_c, J_c)`
        pair -- whether the quotient lands in [0, 1] over the configurations
        the theorem admits is `tests/test_r31_ec36.py`'s question, not this
        one's. What matters here is that `U_c` is large enough for the three
        maps to separate: at the previous `[0.3, -0.7, 1.1]`, `U_c = 0.0044`
        and the r31/r29-detached gradients differ by only 9.5e-4, under this
        test's own margin. At `[0.3, -0.7, 2.6]`, `U_c = 0.2067` and the gaps
        are 1.5e-1 and 1.4e-1.
        """
        leaf_value = torch.tensor([0.3, -0.7, 2.6], dtype=torch.float64)

        def under_test(leaf):
            h_source, h_target, score = self._pieces(leaf)
            return class_global_loss_cosh(score, h_source, h_target)

        def r31_reference(leaf):
            h_source, h_target, score = self._pieces(leaf)
            upper = 2.0 * torch.log(torch.cosh((h_target - h_source) / 4.0))
            return (upper - score) / DENOMINATOR

        def r29_detached(leaf):
            h_source, h_target, score = self._pieces(leaf)
            upper = 2.0 * torch.log(torch.cosh((h_target - h_source) / 4.0))
            return (upper - score) / (upper - LOWER_END).detach()

        def r29_attached(leaf):
            h_source, h_target, score = self._pieces(leaf)
            upper = 2.0 * torch.log(torch.cosh((h_target - h_source) / 4.0))
            return (upper - score) / (upper - LOWER_END)

        measured = self._gradient(leaf_value, under_test)
        r31 = self._gradient(leaf_value, r31_reference)
        detached = self._gradient(leaf_value, r29_detached)
        attached = self._gradient(leaf_value, r29_attached)

        self.assertTrue(torch.allclose(measured, r31, atol=1e-12, rtol=0.0))
        self.assertFalse(torch.allclose(measured, detached, atol=1e-8, rtol=0.0))
        self.assertFalse(torch.allclose(measured, attached, atol=1e-8, rtol=0.0))
        # Each disagreement has to be a margin and not a tolerance.
        self.assertGreater(float((r31 - detached).abs().max()), 1e-3)
        self.assertGreater(float((r31 - attached).abs().max()), 1e-3)

    def test_the_value_moved_by_exactly_the_ratio_of_the_denominators(self) -> None:
        """r31 changed the NUMBER too, which r29's change to Eq. (36) did not.

        This test was `test_the_value_is_unchanged_by_the_detach`, and under
        r29 it was right: detaching a denominator moves the gradient and never
        the value. r31 replaces the denominator itself, so the value moves --
        by exactly `(U_c + ln 2) / ln 2`, which is a measurement and not a
        tolerance. Asserting the old equality here would have pinned r29's map
        from inside a test whose name promised to be about the detach.
        """
        leaf = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64, requires_grad=True)
        h_source, h_target, score = self._pieces(leaf)
        upper = 2.0 * torch.log(torch.cosh((h_target - h_source) / 4.0))

        measured = class_global_loss_cosh(score, h_source, h_target)
        r31 = (upper - score) / DENOMINATOR
        self.assertAlmostEqual(float(measured.detach()), float(r31.detach()),
                               places=12)

        # And the r29 value is a DIFFERENT number, scaled by the two spans.
        r29 = (upper - score) / (upper - LOWER_END)
        ratio = float((upper - LOWER_END).detach()) / DENOMINATOR
        self.assertAlmostEqual(float(measured.detach()),
                               float(r29.detach()) * ratio, places=12)
        self.assertGreater(abs(float(measured.detach()) - float(r29.detach())),
                           1e-4)

    def test_the_gradient_reaches_the_target_features_through_the_loss(self) -> None:
        """The same rule, crossing the join rather than staying in `bounds`.

        `class_global_loss_cosh` tested alone proves the function; this proves
        `forward` actually routes through it with a graph attached, which is
        the only thing the wiring is about.
        """
        f_s, f_t, y_s, g_t = _configurations()['permuted']
        f_t = f_t.clone().requires_grad_(True)
        _loss_for('entropy-cosh')(f_s, f_t, y_s, g_t).backward()
        self.assertIsNotNone(f_t.grad)
        self.assertGreater(float(f_t.grad.abs().max()), 0.0)
        self.assertTrue(bool(torch.isfinite(f_t.grad).all()))


# ----------------------------------------------- 5. r28 §5's active-class rule


class R28ActiveClassTests(unittest.TestCase):
    """A class whose target trace collapsed is skipped, never divided by."""

    @staticmethod
    def _one_dead_class():
        """Class 0's target rows uniform (so `w -> 0`), 1 and 2 confident.

        An exactly uniform row would also make `argmax` send every target
        sample to class 0, which the COUNT condition would then catch for 1
        and 2 and leave nothing for the trace condition to do. The 1e-7 tilt
        keeps the routing intact while leaving the weight at about 2.7e-6.
        """
        f_s, y_s = _blobs(per=9)
        generator = torch.Generator().manual_seed(SEED + 5)
        f_t = f_s + torch.randn(f_s.shape, generator=generator,
                                dtype=torch.float64) * 0.4
        g_t = torch.zeros(f_t.shape[0], CLASSES, dtype=torch.float64)
        for row, label in enumerate(y_s.tolist()):
            if label == 0:
                g_t[row] = 1.0 / CLASSES
                g_t[row, 0] += 1e-7
                g_t[row, 1] -= 5e-8
                g_t[row, 2] -= 5e-8
            else:
                g_t[row, label] = 1.0
        return f_s, f_t, y_s, g_t

    def test_the_collapsed_class_does_not_reach_the_per_class_reduction(self) -> None:
        """Three classes contribute target samples; only two are active."""
        f_s, f_t, y_s, g_t = self._one_dead_class()
        self.assertEqual(len(torch.unique(torch.argmax(g_t, dim=1))), CLASSES)
        # One form, since 2026-10-18: `'conservative'` named a map r33 does not
        # state, and it was removed. The rule under test is about CLASSES and
        # never depended on which map normalized them.
        for form in ('entropy-cosh',):
            with self.subTest(form=form):
                with torch.no_grad():
                    per_class = _loss_for(form)(f_s, f_t, y_s, g_t, reduction='none')
                self.assertEqual(tuple(per_class.shape), (CLASSES - 1,))
                self.assertTrue(bool(torch.isfinite(per_class).all()))

    def test_the_collapsed_class_produces_no_nan_and_no_inf(self) -> None:
        """Dividing by a trace of about 1e-10 is what this rule exists to avoid."""
        f_s, f_t, y_s, g_t = self._one_dead_class()
        for form in ('entropy-cosh',):
            with self.subTest(form=form):
                value = _read(form, (f_s, f_t, y_s, g_t))
                self.assertFalse(math.isnan(value))
                self.assertFalse(math.isinf(value))

    def test_the_floor_is_a_declared_constant_tied_to_the_estimator(self) -> None:
        """Not a number buried in a comparison.

        It is the same 1e-6 that `_renyi_entropy_order_2` adds to the trace it
        divides by: at or below that, the estimator cannot tell the matrix from
        zero, so there is nothing to normalize.
        """
        self.assertTrue(hasattr(models, 'MIN_CLASS_TRACE'))
        self.assertEqual(models.MIN_CLASS_TRACE, 1e-6)
        # And `forward` reads the constant rather than repeating its value.
        source = inspect.getsource(CREDALoss.forward)
        self.assertIn('MIN_CLASS_TRACE', source)

    def test_an_ordinary_class_is_nowhere_near_the_floor(self) -> None:
        """Red has to be reachable from the other side too.

        A floor that excluded ordinary classes would pass every test above and
        silently empty the term, so the margin is measured rather than assumed.
        """
        f_s, f_t, y_s, g_t = _configurations()['permuted']
        with torch.no_grad():
            per_class = _loss_for('entropy-cosh')(f_s, f_t, y_s, g_t, reduction='none')
        self.assertEqual(tuple(per_class.shape), (CLASSES,))


# ----------------------------------------- 6. the defaults, and the keywords


class R28DefaultsUnchangedTests(unittest.TestCase):
    """The contract both new keywords are opt-in values in order to keep."""

    #: Read off the UNEDITED tree, before `bounds_form` or `balance_mass`
    #: existed, with `_default_fixture` below as the input. Hard-coded so that
    #: the reference cannot move with the code it is checking.
    PINNED_MEAN = -0.771175920963
    PINNED_PER_CLASS = (-0.865838766098, -0.726791977882, -0.720897138119)

    @staticmethod
    def _default_fixture():
        """A fixed seeded input, float32 and un-normalized, as the default path
        would actually be called."""
        f_s, y_s = _blobs(seed=777, per=9, dtype=torch.float32)
        generator = torch.Generator().manual_seed(778)
        f_t = f_s + torch.randn(f_s.shape, generator=generator) * 2.0
        logits = torch.randn(f_t.shape[0], CLASSES, generator=generator)
        return f_s, f_t, y_s, torch.softmax(logits, dim=1)

    def test_the_new_keywords_default_to_todays_behaviour(self) -> None:
        """**Both defaults moved, on two different dates, and for two reasons.**

        They were `'conservative'`/`'cardinality'` when this test was written,
        which is what "today's behaviour" meant then. On 2026-10-17 both
        defaults moved to r31's `'entropy-cosh'`/`'trace'`, so a caller that
        does not ask gets the formulation the repository states; the published
        path was pinned by name instead. On 2026-10-18 `'conservative'` stopped
        existing at all, because r33 states one pair of ends -- so this is no
        longer a default among alternatives, it is the only value there is.
        `'cardinality'` is still a real alternative and is still pinned by name
        at `CREDA.training_pipeline.train_creda`.
        """
        loss = CREDALoss()
        self.assertEqual(loss.bounds_form, 'entropy-cosh')
        self.assertEqual(loss.balance_mass, 'trace')
        self.assertEqual(models.BOUNDS_FORMS, ('entropy-cosh',))
        self.assertEqual(models.BALANCE_MASSES, ('cardinality', 'trace'))

    def test_the_default_path_still_reads_the_pinned_values(self) -> None:
        """Bit-for-bit against numbers captured before the edit."""
        f_s, f_t, y_s, g_t = self._default_fixture()
        mean = CREDALoss()(f_s, f_t, y_s, g_t)
        per_class = CREDALoss()(f_s, f_t, y_s, g_t, reduction='none')
        self.assertAlmostEqual(float(mean), self.PINNED_MEAN, places=10)
        self.assertEqual(tuple(per_class.shape), (CLASSES,))
        for measured, pinned in zip(per_class.tolist(), self.PINNED_PER_CLASS):
            self.assertAlmostEqual(measured, pinned, places=10)

    def test_the_conservative_map_is_gone_from_both_packages(self) -> None:
        """r25's ends and its loss are no longer importable, from either side.

        **This asserted the opposite until 2026-10-18**: that
        `conservative_bounds` and `class_global_loss` "still answer r25",
        because each was held function-by-function equal to its `MIL_CREDA`
        original. r33 states ONE pair of ends, the `cosh` pair, and r25's
        cardinality pair is nowhere in the formulation -- so keeping code for
        it meant two modules declaring a revision whose mathematics they no
        longer held. Both were removed, and the copy invariants with them.

        Asserted as an absence rather than left implicit: a reader who
        remembers the pin needs to be told it ended, and an `ImportError`
        somewhere else in the suite would not say which revision decided it.
        """
        from CREDA import bounds as creda_bounds
        from MIL_CREDA import global_term
        for module in (creda_bounds, global_term):
            for retired in ('conservative_bounds', 'class_global_loss'):
                with self.subTest(module=module.__name__, name=retired):
                    self.assertFalse(hasattr(module, retired))
        # The replacement is there and is what the revision states.
        lower, upper = creda_bounds.entropy_cosh_bounds(
            torch.tensor(0.5), torch.tensor(0.5))
        self.assertAlmostEqual(float(lower), -math.log(2.0), places=12)


class R28KeywordValidationTests(unittest.TestCase):
    """Both new keywords refuse in `__init__`, the way `sigma_from` already does."""

    def test_the_declared_values_are_published_as_tuples(self) -> None:
        self.assertEqual(models.BOUNDS_FORMS, ('entropy-cosh',))
        self.assertEqual(models.BALANCE_MASSES, ('cardinality', 'trace'))

    def test_an_unknown_bounds_form_raises_value_error(self) -> None:
        # `'conservative'` joins this list on 2026-10-18: the map it named was
        # removed, and the keyword survives precisely so that a run still
        # asking for it is told so instead of being handed the other one.
        for bad in ('entropy_cosh', 'cosh', '', None, 'Conservative',
                    'conservative'):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    CREDALoss(bounds_form=bad)

    def test_an_unknown_balance_mass_raises_value_error(self) -> None:
        for bad in ('traces', 'Trace', '', None, 'cardinalities'):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    CREDALoss(balance_mass=bad)

    def test_the_refusal_names_the_keyword_and_the_accepted_values(self) -> None:
        with self.assertRaises(ValueError) as raised:
            CREDALoss(bounds_form='cosh')
        self.assertIn('bounds_form', str(raised.exception))
        self.assertIn('entropy-cosh', str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            CREDALoss(balance_mass='traces')
        self.assertIn('balance_mass', str(raised.exception))
        self.assertIn('trace', str(raised.exception))


# ------------------------------------------- 7. the module's own declaration


class R28BoundsProvenanceTests(unittest.TestCase):
    """`src/CREDA/bounds.py` declares what it now holds, and no more."""

    def test_both_new_invariants_are_declared(self) -> None:
        from CREDA import bounds
        declared = bounds.__provenance__['invariants']
        self.assertIn('entropy_cosh_bounds_are_equation_thirty_fives_two_ends', declared)
        self.assertIn('class_global_loss_cosh_divides_by_the_constant_span', declared)

    def test_the_module_does_not_claim_a_revision_it_does_not_hold(self) -> None:
        """Every equation the module holds is r33's, so it declares r33.

        **It declared r25 until 2026-10-18, and the reason was real while it
        lasted**: `conservative_bounds` and `class_global_loss` implemented
        r25's Eq. 35/36 and sat in this same file, and a declaration is a claim
        about everything it covers, so the older half decided the stamp. Both
        were removed because r33 states one pair of ends. What is left --
        `entropy_cosh_bounds`, `class_global_loss_cosh`, `source_bound`,
        `source_loss` -- answers Eqs. 34, 35, 36 and 21 as r33 writes them,
        and Eq. 21 is byte-identical from r25 forward.
        """
        from CREDA import bounds
        self.assertEqual(bounds.__provenance__['revision'],
                         'research-concept-r33.md')
        declared = bounds.__provenance__['invariants']
        for retired in ('conservative_bounds_copy_agrees_with_the_original',
                        'class_global_loss_copy_agrees_with_the_original'):
            self.assertNotIn(retired, declared)
        self.assertIn('research-concept-r33.md', bounds.__provenance__['revision'])
        self.assertIn('entropy_cosh_bounds', bounds.__doc__)


# --------------------------------------------------- 8. the toy's own wiring


class R28ToyWiringTests(unittest.TestCase):
    """What `src/CREDA/toy_experiment.py` is now allowed to pick, and declares."""

    @classmethod
    def setUpClass(cls) -> None:
        from CREDA import toy_experiment
        cls.toy = toy_experiment

    def test_the_phase_two_search_cannot_pick_a_bandwidth_at_all(self) -> None:
        """r28 line 732 narrowed this to the batch scopes; a MEASUREMENT then
        retired the knob outright, on 2026-10-10.

        All three families had searched and all three PICKED `batch-pooled` --
        the scope whose own docstring in `models.SIGMA_SCOPES` records its
        defect: the pooled median reads the target features, which are exactly
        what the optimizer moves, so displacing a class makes its score FALL
        (0.505 at a displacement of 10, 0.437 at 30, 0.397 at 100). The search
        space is no longer narrowed; the bandwidth is not searched. It is
        `toy_experiment.sigma_de_la_fuente` over the whole source train, once,
        frozen -- see `tests/test_toy_sigma_fijo.py` for what it then has to
        satisfy.

        Read through the AST rather than as a substring on purpose: the search
        body EXPLAINS the retirement in a comment that names the old scopes, so
        a text search would read that prose as a live choice. What this asserts
        is the set of names actually handed to a `suggest_*` call.
        """
        arbol = ast.parse(textwrap.dedent(inspect.getsource(self.toy.buscar_hp2)))
        buscados = set()
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            fn = nodo.func
            if (isinstance(fn, ast.Attribute)
                    and fn.attr.startswith('suggest_')
                    and nodo.args
                    and isinstance(nodo.args[0], ast.Constant)):
                buscados.add(nodo.args[0].value)
        # `DELTA` salio el 2026-10-11 con el warmup: no esta en
        # `research-concept-r29.md`, y sobre un UNICO termino con gradiente un
        # coeficiente no balancea nada. Como `SIGMA_FROM`, dejo de ser una
        # dimension de esta busqueda.
        self.assertEqual(buscados, {'LR_2', 'WD_2', 'LOTE_2'}, buscados)
        for retirada in ('SIGMA_FROM', 'DELTA', 'TECHO'):
            self.assertNotIn(retirada, buscados)
            self.assertNotIn(retirada, self.toy.HP_2)

    def test_the_four_sigma_scopes_are_still_all_available(self) -> None:
        """Narrowing the search is not retiring a scope."""
        self.assertEqual(models.SIGMA_SCOPES,
                         ('class-pooled', 'class-source', 'batch-pooled', 'batch-source'))

    def test_the_toy_now_normalizes_by_bounds(self) -> None:
        self.assertIs(self.toy.NORMALIZAR_POR_COTAS, True)

    def test_the_toy_declares_the_two_new_values(self) -> None:
        self.assertEqual(self.toy.FORMA_DE_COTAS, 'entropy-cosh')
        self.assertEqual(self.toy.BALANCE_DE_MASA, 'trace')
        self.assertIn(self.toy.FORMA_DE_COTAS, models.BOUNDS_FORMS)
        self.assertIn(self.toy.BALANCE_DE_MASA, models.BALANCE_MASSES)

    def test_the_loss_is_built_from_those_declarations(self) -> None:
        """The construction site reads the constants rather than repeating them.

        A declaration nothing passes along is the defect this asserts away: the
        two constants above could both be right while `corrida` built the loss
        with neither.
        """
        source = inspect.getsource(self.toy.corrida)
        self.assertIn('bounds_form=FORMA_DE_COTAS', source)
        self.assertIn('balance_mass=BALANCE_DE_MASA', source)

    def test_the_prose_no_longer_argues_for_the_superseded_map(self) -> None:
        """The comment block above the flag is part of the change.

        It argued at length for keeping the old map off, quoting the 8.2155
        span and the 5.1% occupancy. Those measurements are evidence and stay;
        what cannot stay is the conclusion they were used to support, now that
        r28 replaced the equation they were measured on.
        """
        source = inspect.getsource(self.toy)
        start = source.index('#: Si el puntaje por clase')
        end = source.index('NORMALIZAR_POR_COTAS = True')
        self.assertGreater(end, start)
        block = source[start:end]
        # The measurements stay: they are the evidence r28 acted on.
        self.assertIn('8.2155', block)
        self.assertIn('5.1%', block)
        # The conclusion they supported does not.
        self.assertIn('r28', block)
        self.assertNotIn('ACA NO', block)
        self.assertNotIn('desviacion declarada', block)

    def test_no_prose_still_calls_the_mix_balance_a_cardinality_balance(self) -> None:
        """The same treatment wherever the superseded balance is still named.

        Two places name it: the flag's own block, and the comment that fixes
        the validation batch to the training batch's size because the bounds
        "depend on cardinality". Under Eq. 35 they depend on the two marginal
        entropies instead, so that reason has to be restated rather than
        deleted -- the batch sizes still have to match, for a different reason.
        """
        source = inspect.getsource(self.toy)
        self.assertNotIn('cotas que dependen de la cardinalidad', source)


# ------------------------------------- 10. Eq. 35's bounds are a THEOREM about
#                                           the balanced mix of Eq. 32


class R28BoundHypothesisTests(unittest.TestCase):
    """`entropy-cosh` may only be paired with the structure it was proved on.

    r28 line 1010 states the hypothesis in one sentence: "Las cotas de la
    puntuacion se derivan de la estructura por bloques de la conjunta
    **balanceada**." The derivation needs the whole of Eq. 32 -- the identity
    `rho_mix = rho_s/4 + rho_t/4 + ||K_st||_F^2 / 2` comes from the leading 1/2
    together with the trace-normalized diagonal blocks, and Cauchy-Schwarz on
    those blocks is what gives `||K_st||_F^2 <= sqrt(rho_s rho_t)`. From those
    two together come `L_c` and `U_c`.

    Pair that map with any other mixture and it is a theorem applied outside
    its hypothesis. Measured over 300 random configurations -- dimension,
    both cardinalities, both scales, the displacement, the confidences and the
    bandwidth all drawn -- the normalized loss leaves [0, 1]:

        trace          0 / 300
        cardinality  ~190 / 300   roughly [-0.83, +1.85]
        no balance   ~160 / 300   roughly [-0.83, +1.85]

    A negative loss is not a cosmetic range violation: it tells the optimizer
    to keep going in the direction that produced it.

    So the two keywords were never independent axes, and offering them as two
    was the defect. This is a refusal rather than a note, because a note leaves
    the pairing reachable.
    """

    @staticmethod
    def _random_configuration(index: int):
        generator = torch.Generator().manual_seed(SEED + index)

        def rand(*shape):
            return torch.rand(*shape, generator=generator, dtype=torch.float64)

        dim = int(torch.randint(2, 24, (1,), generator=generator))
        n_s = int(torch.randint(3, 60, (1,), generator=generator))
        n_t = int(torch.randint(3, 60, (1,), generator=generator))
        f_s = torch.randn(n_s, dim, generator=generator, dtype=torch.float64) * (float(rand(1)) * 4 + 0.2)
        f_t = torch.randn(n_t, dim, generator=generator, dtype=torch.float64) * (float(rand(1)) * 4 + 0.2)
        f_t = f_t + torch.randn(1, generator=generator, dtype=torch.float64) * 3
        if float(rand(1)) < 0.15:            # a collapsed target
            f_t = f_t * 0
        weights = rand(n_t) * float(rand(1)) + 1e-3
        sigma = float(rand(1)) * 3 + 0.1
        return f_s, f_t, weights, sigma

    def _normalized(self, loss: CREDALoss, configuration) -> float:
        """Eq. 33 then Eq. 36, by hand, so a REFUSED pairing can still be read."""
        f_s, f_t, weights, sigma = configuration
        with torch.no_grad():
            K_s = loss._gaussian_kernel(f_s, f_s, sigma)
            K_t = loss._gaussian_kernel(f_t, f_t, sigma) * torch.outer(weights, weights)
            K_st = loss._gaussian_kernel(f_s, f_t, sigma) * weights.unsqueeze(0)
            if float(torch.trace(K_t)) <= models.MIN_CLASS_TRACE:
                return 0.0
            h_s = loss._renyi_entropy_order_2(K_s)
            h_t = loss._renyi_entropy_order_2(K_t)
            mix = loss._mix_kernel_concat(K_s, K_t, K_st)
            score = loss._renyi_entropy_order_2(mix) - 0.5 * (h_s + h_t)
            return float(class_global_loss_cosh(-score, h_s, h_t))

    def test_the_trace_balance_never_leaves_the_unit_interval(self) -> None:
        """The positive half: the hypothesis holds, so the bound holds.

        The spread assertions are not decoration. A sweep whose every
        configuration short-circuited on `MIN_CLASS_TRACE` would return 300
        zeros and satisfy "none left [0, 1]" having measured nothing -- the
        shape of green this repository has shipped before. Requiring values on
        both sides of the interval's middle is what makes the emptiness of
        `outside` evidence.
        """
        loss = CREDALoss(balance_mix_blocks=True, balance_mass='trace')
        values = [self._normalized(loss, self._random_configuration(i))
                  for i in range(300)]
        outside = [v for v in values if v < -1e-6 or v > 1.0 + 1e-6]
        self.assertEqual(outside, [], f"{len(outside)} of 300 left [0, 1]")
        self.assertGreater(len([v for v in values if v > 0.5]), 30)
        self.assertGreater(len([v for v in values if 1e-6 < v < 0.5]), 30)
        self.assertGreater(max(values), 0.9)

    def test_the_other_mixtures_do_leave_it_which_is_why_this_is_refused(self) -> None:
        """The half that makes the refusal load-bearing, not decorative.

        Without this, a guard that refused a pairing which happened to be fine
        would look exactly as green.
        """
        for keywords in (dict(balance_mix_blocks=True, balance_mass='cardinality'),
                         dict(balance_mix_blocks=False)):
            with self.subTest(**keywords):
                loss = CREDALoss(**keywords)
                outside = [
                    value for value in (
                        self._normalized(loss, self._random_configuration(i))
                        for i in range(300)
                    ) if value < -1e-6 or value > 1.0 + 1e-6
                ]
                self.assertGreater(len(outside), 50)
                self.assertLess(min(outside), 0.0)       # negative, not merely out of range
                self.assertGreater(max(outside), 1.0)    # and broken both ways

    def test_entropy_cosh_refuses_an_unbalanced_mixture(self) -> None:
        with self.assertRaises(ValueError) as raised:
            CREDALoss(normalize_by_bounds=True, bounds_form='entropy-cosh',
                      balance_mix_blocks=False)
        self.assertIn('balance_mix_blocks', str(raised.exception))

    def test_entropy_cosh_refuses_a_raw_cross_block(self) -> None:
        """Eq. 26 weights BOTH blocks that touch target bags, so leaving the
        cross block raw breaks Eq. 35's hypothesis.

        **Found by a test, not by reading.** Moving
        `test_creda_sample.py`'s sign-chain measurement from r25's map to r33's
        on 2026-10-18 produced per-class losses of about -1.28 under a pairing
        the constructor accepted. Measured over 600 random configurations,
        1378 of 1733 per-class readings fall outside [0, 1] with the cross
        block raw, down to -20.26; with it weighted, 7 of 1733.
        """
        with self.assertRaises(ValueError) as raised:
            CREDALoss(normalize_by_bounds=True, bounds_form='entropy-cosh',
                      balance_mix_blocks=True, balance_mass='trace',
                      use_entropy_weighting=True, weight_cross_block=False)
        message = str(raised.exception)
        self.assertIn('weight_cross_block', message)
        self.assertIn('26', message)

        # Red from the other side: with the cross block weighted it
        # constructs, and so does the unweighted path, where there is no D at
        # all and therefore no asymmetry between the blocks to repair.
        CREDALoss(normalize_by_bounds=True, bounds_form='entropy-cosh',
                  balance_mix_blocks=True, balance_mass='trace',
                  use_entropy_weighting=True, weight_cross_block=True)
        CREDALoss(normalize_by_bounds=True, bounds_form='entropy-cosh',
                  balance_mix_blocks=True, balance_mass='trace',
                  use_entropy_weighting=False, weight_cross_block=False)

    def test_entropy_cosh_refuses_the_cardinality_balance(self) -> None:
        with self.assertRaises(ValueError) as raised:
            CREDALoss(normalize_by_bounds=True, bounds_form='entropy-cosh',
                      balance_mix_blocks=True, balance_mass='cardinality')
        self.assertIn('balance_mass', str(raised.exception))

    def test_the_refusal_names_the_hypothesis_and_not_just_the_keyword(self) -> None:
        """A reader has to learn WHY, or they will reach for the other value."""
        with self.assertRaises(ValueError) as raised:
            CREDALoss(normalize_by_bounds=True, bounds_form='entropy-cosh',
                      balance_mix_blocks=True, balance_mass='cardinality')
        message = str(raised.exception)
        for fragment in ('entropy-cosh', '32', 'trace'):
            self.assertIn(fragment, message)

    def test_the_r28_pairing_itself_constructs(self) -> None:
        """**`weight_cross_block=True` joined this pairing on 2026-10-18.**

        `use_entropy_weighting` defaults to `True`, so this construction used
        to leave Eq. 26 half applied -- `D K_c^t D` weighted, `K_c^st` raw --
        and `test_entropy_cosh_refuses_a_raw_cross_block` above has the
        measurement that makes that inadmissible: 1378 of 1733 per-class
        readings outside [0, 1]. Naming it here is not a workaround; it is
        what the admissible pairing is.
        """
        loss = CREDALoss(normalize_by_bounds=True, bounds_form='entropy-cosh',
                         balance_mix_blocks=True, balance_mass='trace',
                         weight_cross_block=True)
        self.assertEqual(loss.bounds_form, 'entropy-cosh')
        self.assertEqual(loss.balance_mass, 'trace')
        self.assertTrue(loss.weight_cross_block)

    def test_the_published_conservative_path_no_longer_constructs(self) -> None:
        """**The inverse of what this asserted until 2026-10-18.**

        It held that the `entropy-cosh` refusal was about ONE map's hypothesis
        and must not narrow r25's path, because r25's constant ends carry no
        hypothesis about the mixture and every published number was taken with
        them. That was right while the path existed. r33 states one pair of
        ends, the pair was removed, and the path it protected cannot be built
        any more -- under ANY pairing of the mixture flags, which is what the
        three cases below still check: the refusal is the keyword's, not a
        side effect of one combination.

        What it costs is recorded where it is owed, in `CREDA.bounds`' own
        docstring: those published numbers are reachable in the artefacts that
        hold them, not by re-running the function that produced them.
        """
        for keywords in (dict(balance_mix_blocks=True, balance_mass='cardinality'),
                         dict(balance_mix_blocks=True, balance_mass='trace'),
                         dict(balance_mix_blocks=False)):
            with self.subTest(**keywords):
                with self.assertRaises(ValueError) as raised:
                    CREDALoss(normalize_by_bounds=True,
                              bounds_form='conservative', **keywords)
                self.assertIn('bounds_form', str(raised.exception))
                self.assertIn('entropy-cosh', str(raised.exception))

    def test_an_unbalanced_mixture_without_the_bounds_is_untouched(self) -> None:
        """`normalize_by_bounds=False` reaches no map, so there is nothing to pair."""
        loss = CREDALoss(bounds_form='entropy-cosh', balance_mix_blocks=False)
        self.assertFalse(loss.normalize_by_bounds)


# ------------------------------- 11. The toy's own flags must agree, AT IMPORT


class R28ToyCoherenceTests(unittest.TestCase):
    """`toy_experiment.verificar_coherencia` catches the pairing at declaration.

    `CREDALoss` already refuses the pairing, but it refuses inside `corrida` --
    twelve minutes into a full run and after both Optuna searches have been
    spent. A sweep row with `balance=False` beside these flags is an incoherent
    DECLARATION, not an arm, so it is charged at import, where it still costs
    nothing.

    A function rather than a bare module-level `assert` precisely so these
    tests can hand it incoherent input: an assert at module scope can only be
    checked by reading the source, and a constant nobody reads looks exactly
    like green.
    """

    OK = (True, False, 'row', 'key')
    BAD = (False, False, 'row', 'key')

    def setUp(self) -> None:
        from CREDA import toy_experiment
        self.verify = toy_experiment.verificar_coherencia

    def test_a_row_asking_for_no_balance_is_refused(self) -> None:
        with self.assertRaises(ValueError) as raised:
            self.verify([self.BAD], [self.OK], True, 'entropy-cosh', 'trace')
        self.assertIn('32', str(raised.exception))

    def test_the_cardinality_mass_is_refused(self) -> None:
        with self.assertRaises(ValueError) as raised:
            self.verify([self.OK], [self.OK], True, 'entropy-cosh', 'cardinality')
        self.assertIn('trace', str(raised.exception))

    def test_the_table_is_checked_and_not_only_the_frames(self) -> None:
        """Both lists. Checking one would leave the other free to disagree."""
        with self.assertRaises(ValueError):
            self.verify([self.OK], [self.BAD], True, 'entropy-cosh', 'trace')

    def test_the_coherent_declaration_passes(self) -> None:
        self.verify([self.OK], [self.OK], True, 'entropy-cosh', 'trace')

    def test_nothing_is_checked_when_the_map_is_not_reached(self) -> None:
        """No bounds, or r25's constant ends, carry no hypothesis to violate."""
        self.verify([self.BAD], [self.BAD], False, 'entropy-cosh', 'cardinality')
        self.verify([self.BAD], [self.BAD], True, 'conservative', 'cardinality')

    def test_the_live_declaration_is_coherent(self) -> None:
        """And the module as it actually stands, which is what runs tonight."""
        from CREDA import toy_experiment as toy
        self.verify(toy.FILAS, toy.TABLA, toy.NORMALIZAR_POR_COTAS,
                    toy.FORMA_DE_COTAS, toy.BALANCE_DE_MASA)
        self.assertTrue(all(row[0] for row in list(toy.FILAS) + list(toy.TABLA)))


if __name__ == '__main__':
    unittest.main()
