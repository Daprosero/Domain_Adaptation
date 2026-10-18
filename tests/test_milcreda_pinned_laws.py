"""The settings MIL-CREDA pins with no flag, asserted to be CREDA's choices.

Four of CREDA's axes are LAW here. `normalize_by_bounds` is always on,
`bounds_form` is always Eq. 35's entropy-cosh, `balance_mix_blocks` is always
on and `balance_mass` is always the trace: `src/MIL_CREDA/global_term.py` has
no conservative branch, no unbalanced branch and no cardinality branch to
select. That is correct and it stays -- an axis nobody intends to sweep is a
dimension of configuration space that only costs runs.

**What was missing is that nothing asserted the pinned behaviour IS the one
CREDA chooses.** The evidence was the absence of an alternative, and an absence
is not evidence: a pin that drifts away from the choice it was copied from
reads exactly like a pin that still agrees with it. Both halves could move --
`src/CREDA/toy_experiment.py` could declare a different `FORMA_DE_COTAS`, or
`global_term.py` could be rewritten -- and every other test in this repository
would stay green, because each side is tested against itself.

So every expected value below is READ from `src/CREDA` rather than typed here:
the chosen settings from CREDA's own module-level declarations, and the numbers
from CREDA's own implementation of the same construction. Nothing in this file
asserts a literal MIL-CREDA could satisfy by itself.

The same file carries the two axes T4 named -- `weight_cross_block` and
`detach_target_entropy`. Those are genuine parameters here, not laws, so what
is asserted about them is narrower: that the DEFAULT each one takes is the
value CREDA's `HP_2` declares, and that the default is what the body actually
does rather than only what the signature says. Their mathematics is in
`tests/test_invariants.py`, where the claims about Eqs. (26) and (33) live.

**One difference is declared and is not a disagreement.** Eq. 32's leading
`1/2` is written in `balanced_mixture` and deliberately NOT written in CREDA's
`_mix_kernel_concat`: CREDA divides by the trace of whatever it is handed, so
the factor is a no-op there, while this package returns a matrix that is
already a density. Every comparison below therefore reads CREDA's matrix times
`1/2`, which is Eq. 32's own right-hand side.
"""

from __future__ import annotations

import inspect
import math
import unittest

import torch

from CREDA.models import CREDALoss
from CREDA.toy_experiment import (
    BALANCE_DE_MASA,
    FORMA_DE_COTAS,
    HP_2,
    NORMALIZAR_POR_COTAS,
)
from MIL_CREDA import DTYPE
from MIL_CREDA.conditional import weighted_blocks
from MIL_CREDA.global_term import (
    balanced_dependency_score,
    balanced_mixture,
    class_global_loss_cosh,
)
from MIL_CREDA.kernels import gaussian_kernel
from MIL_CREDA.renyi import quadratic_entropy

#: The bandwidth both packages are handed, so neither derives its own. CREDA's
#: `sigma='auto'` would make the two sides read different kernels and the
#: comparison would measure the bandwidth rule instead of the construction.
SIGMA = 1.3

#: Tight: the congruence identity below is the same arithmetic on both sides,
#: so a tolerance would only hide a wrong coefficient.
EXACT = 1e-13

#: Loose, and for one reason: CREDA's `_renyi_entropy_order_2` carries an
#: epsilon inside the logarithm and `MIL_CREDA.renyi.quadratic_entropy` does
#: not. Measured across the three cardinality pairs below, the two readings of
#: the same per-class loss differ by 2.5e-6 to 3.9e-6, while r25's pair reads
#: 0.24 to 0.33 away -- five orders of magnitude of separation, so this
#: tolerance distinguishes the two branches without being able to hide one.
EPSILON_GAP = 1e-4

#: The three cardinality pairs every numeric claim is measured over. More than
#: one because the cardinality balance this file rules out agrees with the
#: trace balance exactly when the counts and the weights conspire, and a single
#: pair cannot show that it does not.
PAIRS = ((4, 6, 32), (7, 3, 33), (3, 9, 34))


def _one_class(n_source: int, n_target: int, seed: int):
    """One class's blocks and confidences, in both packages' shared form.

    The features are what CREDA's own methods take, and the kernels MIL-CREDA
    builds from them are the Gaussian kernel of Eq. (2) at the same bandwidth,
    so the two sides are handed the same matrices rather than the same
    description of them.
    """
    generator = torch.Generator().manual_seed(seed)
    f_source = torch.randn((n_source, 5), generator=generator, dtype=DTYPE)
    f_target = torch.randn((n_target, 5), generator=generator, dtype=DTYPE) + 0.6
    weights = 0.2 + 0.8 * torch.rand(n_target, generator=generator, dtype=DTYPE)
    return f_source, f_target, weights


def _blocks(f_source, f_target, weights):
    """Eqs. (23) and (26): `(K_s, K~_st, K~_t)` at this file's bandwidth."""
    K_s = gaussian_kernel(f_source, f_source, SIGMA)
    K_st = gaussian_kernel(f_source, f_target, SIGMA)
    K_t = gaussian_kernel(f_target, f_target, SIGMA)
    K_st_w, K_t_w = weighted_blocks(K_st, K_t, weights)
    return K_s, K_st_w, K_t_w


def _creda(**overrides) -> CREDALoss:
    """A `CREDALoss` on this file's shared footing, with the axes overridden.

    `use_entropy_weighting` and `weight_cross_block` are on because they are
    what makes CREDA's blocks the same `K~_t` and `K~_st` Eq. (26) defines, and
    without them there is no `D` and nothing to compare.
    """
    settings = dict(
        sigma=SIGMA,
        use_entropy_weighting=True,
        weight_cross_block=True,
        balance_mix_blocks=True,
    )
    settings.update(overrides)
    return CREDALoss(**settings)


# ----------------------------------------- 1. the two axes T4 gave names to


class AxisDefaultsTests(unittest.TestCase):
    """Each named axis defaults to the value CREDA's `HP_2` declares."""

    def test_weight_cross_block_defaults_to_the_axis_value_creda_declares(
        self,
    ) -> None:
        """Eq. (26): `weighted_blocks`' default is CREDA's `CRUZADO_SOPESADO`.

        Both halves, because a signature default the body ignores would pass
        either one alone: the declared default equals CREDA's choice, AND
        calling with no keyword returns what calling with that choice returns.

        The control is the OTHER pole: it has to return something different, or
        "the default is this one" would be a claim about a parameter that
        decides nothing.
        """
        declared = inspect.signature(weighted_blocks).parameters[
            "weight_cross_block"
        ].default
        self.assertIs(declared, HP_2["CRUZADO_SOPESADO"])

        f_source, f_target, weights = _one_class(4, 6, 32)
        K_s = gaussian_kernel(f_source, f_source, SIGMA)
        K_st = gaussian_kernel(f_source, f_target, SIGMA)
        K_t = gaussian_kernel(f_target, f_target, SIGMA)

        silent = weighted_blocks(K_st, K_t, weights)
        chosen = weighted_blocks(
            K_st, K_t, weights, weight_cross_block=HP_2["CRUZADO_SOPESADO"]
        )
        other = weighted_blocks(
            K_st, K_t, weights, weight_cross_block=not HP_2["CRUZADO_SOPESADO"]
        )
        for quiet, named in zip(silent, chosen):
            self.assertTrue(torch.equal(quiet, named))
        self.assertFalse(torch.allclose(silent[0], other[0], atol=1e-8))

    def test_detach_target_entropy_defaults_to_the_axis_value_creda_declares(
        self,
    ) -> None:
        """Eq. (33): the default is CREDA's `RHO_T_SIN_GRAD`.

        The same two halves. The control cannot be a value, because this axis
        moves no value -- that is the whole of why its own invariant in
        `tests/test_invariants.py` reads the gradient. So the control here is
        the gradient: the other pole has to produce a different one.
        """
        declared = inspect.signature(balanced_dependency_score).parameters[
            "detach_target_entropy"
        ].default
        self.assertIs(declared, HP_2["RHO_T_SIN_GRAD"])

        f_source, f_target, weights = _one_class(4, 6, 32)

        def gradient(**kwargs) -> torch.Tensor:
            leaf = f_target.clone().requires_grad_(True)
            score = balanced_dependency_score(*_blocks(f_source, leaf, weights),
                                              **kwargs)
            return torch.autograd.grad(score, leaf)[0]

        silent = gradient()
        chosen = gradient(detach_target_entropy=HP_2["RHO_T_SIN_GRAD"])
        other = gradient(
            detach_target_entropy=not HP_2["RHO_T_SIN_GRAD"]
        )
        self.assertTrue(torch.equal(silent, chosen))
        self.assertFalse(torch.allclose(silent, other, atol=1e-8))


# --------------------------------------------- 2. the two laws T6 asserted


class PinnedLawTests(unittest.TestCase):
    """The settings this package pins, measured against CREDA's own choices."""

    def test_the_balance_is_credas_trace_mass_and_not_its_cardinality_one(
        self,
    ) -> None:
        """Eq. (32): the pin is `balance_mix_blocks` on with `balance_mass` trace.

        Three claims, none of them with an expected value typed into this file.

        First, the DECLARED one: CREDA's `BALANCE_DE_MASA` says `trace`, so the
        branch this package pins is the branch CREDA selects.

        Second, the CONSTRUCTION: `balanced_mixture`'s joint is exactly half of
        what CREDA's `_mix_kernel_concat` builds under that declaration, bit
        for bit, including the cross block rescaled by `sqrt(tr(K_s) tr(K~_t))`
        -- which is read off CREDA's matrix rather than recomputed here. The
        trace is then `1/2 + 1/2 = 1`, which is what makes it a density Eq. (9)
        can be taken of, and the `1/2` CREDA omits is exactly what buys it.

        Third, the pole that makes the first two a choice rather than a
        restatement: CREDA's `cardinality` balance divides by `n_s`, `n_t` and
        `sqrt(n_s n_t)` instead, and gives a DIFFERENT matrix, a different
        cross block, and a trace that is not 1 -- so had the pin landed on the
        other branch, something measurable here would say so.
        """
        self.assertEqual(BALANCE_DE_MASA, "trace")
        by_trace = _creda(normalize_by_bounds=False, balance_mass=BALANCE_DE_MASA)
        by_count = _creda(normalize_by_bounds=False, balance_mass="cardinality")
        self.assertNotEqual(by_count.balance_mass, BALANCE_DE_MASA)

        for n_source, n_target, seed in PAIRS:
            with self.subTest(n_source=n_source, n_target=n_target):
                K_s, K_st_w, K_t_w = _blocks(*_one_class(n_source, n_target, seed))
                _, _, breve_mix = balanced_mixture(K_s, K_st_w, K_t_w)
                chosen = 0.5 * by_trace._mix_kernel_concat(K_s, K_t_w, K_st_w)
                counted = 0.5 * by_count._mix_kernel_concat(K_s, K_t_w, K_st_w)

                # The construction, against CREDA's own.
                self.assertTrue(
                    torch.allclose(breve_mix, chosen, atol=EXACT, rtol=0.0)
                )
                # The trace Eq. 9 needs, and the `1/2` that buys it.
                self.assertAlmostEqual(float(torch.trace(breve_mix)), 1.0, places=12)

                # The cross block specifically: `sqrt(tr . tr)`, not `sqrt(n n)`.
                cross = 2.0 * breve_mix[:n_source, n_source:]
                self.assertTrue(
                    torch.allclose(
                        cross,
                        by_trace._mix_kernel_concat(K_s, K_t_w, K_st_w)[
                            :n_source, n_source:
                        ],
                        atol=EXACT,
                        rtol=0.0,
                    )
                )
                self.assertGreater(
                    float(
                        (
                            cross
                            - by_count._mix_kernel_concat(K_s, K_t_w, K_st_w)[
                                :n_source, n_source:
                            ]
                        )
                        .abs()
                        .max()
                    ),
                    1e-3,
                )

                # The pole: the other balance is neither this matrix nor a
                # density, so the choice is observable.
                self.assertGreater(
                    float((breve_mix - counted).abs().max()), 1e-3
                )
                self.assertGreater(
                    abs(float(torch.trace(counted)) - 1.0), 0.1
                )

    def test_the_bounds_are_credas_entropy_cosh_choice_and_not_r25s_pair(
        self,
    ) -> None:
        """Eqs. (35)-(36): the pin is `normalize_by_bounds` on, `entropy-cosh`.

        Three claims again, and again nothing typed here.

        The DECLARED one: CREDA's `NORMALIZAR_POR_COTAS` is on and its
        `FORMA_DE_COTAS` is `entropy-cosh`, which is the pair this package
        reaches with no flag.

        The READING: the per-class loss this package computes is the number
        CREDA's `_marginal_score` returns on the same blocks under that
        declaration. Agreement is asserted at `EPSILON_GAP` and not tighter,
        because CREDA's entropy carries an epsilon this one does not; see that
        constant for the measured separation.

        The POLE: r25's pair -- `L_c = -ln(n_s + n_t)` and
        `U_c = (ln n_s + ln n_t)/2` -- over the same score, two orders of
        magnitude further away than the epsilon. So if this package were
        normalizing by r25's pair, this test would say so rather than passing
        on a value that happened to be in [0, 1].

        **The pole was `CREDALoss(bounds_form='conservative')` until
        2026-10-18.** r33 states one pair of ends, so `conservative_bounds`
        and `class_global_loss` were removed from both packages and CREDA's
        selector has one branch left. The pole is reconstructed here instead,
        three lines traceable to `proposals/research-concept-r25.md`: a retired
        map's reader, used only to be the thing the current one is not.
        """
        self.assertIs(NORMALIZAR_POR_COTAS, True)
        self.assertEqual(FORMA_DE_COTAS, "entropy-cosh")
        by_cosh = _creda(
            normalize_by_bounds=NORMALIZAR_POR_COTAS,
            bounds_form=FORMA_DE_COTAS,
            balance_mass=BALANCE_DE_MASA,
        )
        # The retired selector: asking for r25's map is a refusal now, and
        # that refusal is itself part of the law this test pins.
        with self.assertRaises(ValueError):
            _creda(normalize_by_bounds=NORMALIZAR_POR_COTAS,
                   bounds_form="conservative",
                   balance_mass=BALANCE_DE_MASA)

        for n_source, n_target, seed in PAIRS:
            with self.subTest(n_source=n_source, n_target=n_target):
                f_source, f_target, _weights = _one_class(n_source, n_target, seed)
                # **Unit weights on BOTH sides since 2026-10-18.** This law is
                # about the MAP -- that MIL normalizes by r33's cosh ends and
                # not by r25's cardinality pair -- and a map can only be
                # compared on blocks both packages see the same way. r33 scopes
                # Eq. (26)'s `D` to the class-conditional blocks, so CREDA's
                # `_marginal_score` no longer weights anything: feeding MIL the
                # weighted blocks while CREDA reads raw ones measured the
                # SCOPING and not the map, and the gap grew to 0.135 where the
                # epsilon is 1e-4. With `D = I` the two see one matrix each.
                weights = torch.ones_like(_weights)
                K_s, K_st_w, K_t_w = _blocks(f_source, f_target, weights)
                breve_s, breve_t, _ = balanced_mixture(K_s, K_st_w, K_t_w)
                pinned = float(
                    class_global_loss_cosh(
                        balanced_dependency_score(K_s, K_st_w, K_t_w),
                        quadratic_entropy(breve_s),
                        quadratic_entropy(breve_t),
                    )
                )
                cosh = float(by_cosh._marginal_score(f_source, f_target))
                # r25's Eq. (35)/(36) over the SAME score the cosh map reads,
                # written out here because no code computes it any more.
                lower_25 = -math.log(n_source + n_target)
                upper_25 = 0.5 * (math.log(n_source) + math.log(n_target))
                j_c = float(balanced_dependency_score(K_s, K_st_w, K_t_w))
                r25 = (upper_25 - j_c) / (upper_25 - lower_25)

                self.assertLess(abs(pinned - cosh), EPSILON_GAP)
                self.assertGreater(abs(pinned - r25), 0.1)
                # The control on the control: the two branches of CREDA's own
                # selector have to disagree, or the pole measures nothing.
                self.assertGreater(abs(cosh - r25), 0.1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
