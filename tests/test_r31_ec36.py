"""r31's Eq. (36): the denominator becomes the constant `ln 2`.

r29 normalized the corrected score by the span it could occupy,
`l_glob,c = (U_c - J_c) / (U_c - L_c)` with `L_c = -ln 2`, and excluded that
span from the gradient by design. r30 proved the span is bounded by a constant
and r31 writes the division against the constant instead:

    r29:  l_glob,c = (U_c - J_c) / (U_c - L_c)
    r31:  l_glob,c = (U_c - J_c) / ln 2

The theorem that licenses it, r31 Section 5.3: with `r_c = ln(rho_s / rho_t)`,
the lower bound on `rho_c^mix` gives `J_c >= ln cosh(r_c/2) - ln 2`, Eq. (35)
is `U_c = 2 ln cosh(r_c/4)`, and `cosh(r_c/2) = 2 cosh^2(r_c/4) - 1 >=
cosh^2(r_c/4)`, so

    0 <= U_c - J_c <= ln 2

and the quotient lands in [0, 1] without ever reading the span.

**Three consequences land with the change, and each one gets a test.**

1. The quotient still lies in [0, 1] -- but now because of the theorem and not
   because the map was built from the endpoints it divides by.
2. **The scale stops depending on the batch, the class or the cardinalities.**
   r31: "un mismo valor representa la misma diferencia `U_c - J_c` en cualquier
   caso". Under r29 the denominator varied with `U_c`, so two classes with the
   same `U_c - J_c` and different marginal entropies reported DIFFERENT losses.
   That is the half of the change a test comparing one class against itself
   cannot see.
3. **The span stops being detached.** r29 listed `U_c - L_c` as excluded from
   the gradient; `ln 2` is a literal with nothing to detach, so the exception
   disappears. The value and the gradient both move, and the gradient moves to
   something equal to NEITHER of r29's two references.

Every test here was observed RED against the r29 tree before the code moved.
Both packages are asserted, because Eq. (36) is implemented twice -- once in
`src/MIL_CREDA/global_term.py` and once, deliberately copied, in
`src/CREDA/bounds.py` -- and a change applied to one is invisible to a test
that only reads the other.
"""

from __future__ import annotations

import math

import pytest
import torch

from CREDA.bounds import class_global_loss_cosh as creda_loss
from CREDA.bounds import entropy_cosh_bounds as creda_bounds
from MIL_CREDA.global_term import class_global_loss_cosh as mil_loss
from MIL_CREDA.global_term import entropy_cosh_bounds as mil_bounds

DTYPE = torch.float64
LN2 = math.log(2.0)

#: The two implementations of Eq. (36), by the package that owns each.
IMPLEMENTATIONS = (("MIL_CREDA", mil_loss, mil_bounds),
                   ("CREDA", creda_loss, creda_bounds))


def _upper(h_source: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
    """Eq. (35) written out here, from the equation and not from the module."""
    return 2.0 * torch.log(torch.cosh((h_target - h_source) / 4.0))


# --------------------------------------------------------- the denominator


@pytest.mark.parametrize("package,loss,_bounds", IMPLEMENTATIONS,
                         ids=[name for name, _, _ in IMPLEMENTATIONS])
def test_the_denominator_is_ln_two_and_not_the_span(package, loss, _bounds) -> None:
    """`(U_c - J_c) / ln 2`, against the equation recomputed here.

    Reachable red: the r29 tree divides by `U_c - L_c`, which equals `ln 2`
    only when `U_c = 0`. So the fixture carries entropies whose difference is
    NON-zero -- with `h_t - h_s = 0` both revisions agree and the test could
    not tell them apart.
    """
    h_source = torch.tensor([0.4, 1.9, -0.6], dtype=DTYPE)
    h_target = torch.tensor([2.3, -0.5, 1.7], dtype=DTYPE)
    score = torch.tensor([-0.2, 0.1, -0.45], dtype=DTYPE)

    upper = _upper(h_source, h_target)
    expected = (upper - score) / LN2

    got = loss(score, h_source, h_target)
    assert torch.allclose(got, expected, atol=1e-14, rtol=0.0), package

    # And the fixture genuinely separates the two revisions: the r29 quotient
    # must differ by a margin that is not a tolerance.
    r29 = (upper - score) / (upper + LN2)
    assert float((expected - r29).abs().max()) > 1e-3, (
        f"{package}: the fixture cannot distinguish r31 from r29")


@pytest.mark.parametrize("package,loss,_bounds", IMPLEMENTATIONS,
                         ids=[name for name, _, _ in IMPLEMENTATIONS])
def test_the_denominator_does_not_move_with_the_upper_end(package, loss, _bounds) -> None:
    """Two classes, the same `U_c - J_c`, different `U_c`: one loss.

    This is r31's stated consequence -- "un mismo valor representa la misma
    diferencia `U_c - J_c` en cualquier caso" -- and it is the half that a
    per-class test cannot see, because under r29 each class was divided by its
    OWN span.

    Reachable red: under r29 the two differ, and the assertion below pins the
    r29 gap as a positive margin so the comparison cannot pass by tolerance.
    """
    gap = 0.3

    # Class A: marginal entropies far apart, so `U_c` is large.
    h_s_a = torch.tensor([0.0], dtype=DTYPE)
    h_t_a = torch.tensor([6.0], dtype=DTYPE)
    upper_a = _upper(h_s_a, h_t_a)
    score_a = upper_a - gap

    # Class B: marginal entropies equal, so `U_c = 0`.
    h_s_b = torch.tensor([1.5], dtype=DTYPE)
    h_t_b = torch.tensor([1.5], dtype=DTYPE)
    upper_b = _upper(h_s_b, h_t_b)
    score_b = upper_b - gap

    assert float(upper_a) > 1.0, "fixture: class A must have a large U_c"
    assert float(upper_b) == pytest.approx(0.0, abs=1e-12)

    loss_a = loss(score_a, h_s_a, h_t_a)
    loss_b = loss(score_b, h_s_b, h_t_b)
    assert torch.allclose(loss_a, loss_b, atol=1e-14, rtol=0.0), (
        f"{package}: equal U_c - J_c reported different losses")

    # The r29 map, recomputed, DOES separate them -- so the agreement above is
    # a property of r31's denominator and not of the fixture.
    r29_a = (upper_a - score_a) / (upper_a + LN2)
    r29_b = (upper_b - score_b) / (upper_b + LN2)
    assert float((r29_a - r29_b).abs().max()) > 1e-2, (
        f"{package}: the fixture cannot distinguish r31 from r29")


# -------------------------------------------------------------- the gradient


def _gradient(value: torch.Tensor, build) -> torch.Tensor:
    leaf = value.clone().requires_grad_(True)
    build(leaf).backward()
    assert leaf.grad is not None
    return leaf.grad.clone()


@pytest.mark.parametrize("package,loss,_bounds", IMPLEMENTATIONS,
                         ids=[name for name, _, _ in IMPLEMENTATIONS])
def test_nothing_is_detached_because_there_is_nothing_to_detach(
        package, loss, _bounds) -> None:
    """The gradient equals the `/ ln 2` reference and NEITHER r29 reference.

    Three references, because one cannot settle it. r29 detached the span; a
    reference that detaches and one that does not differ from each other, and
    r31's constant denominator differs from BOTH -- dividing by a literal is
    not the same function as dividing by `U_c + ln 2`, detached or otherwise.
    Asserting only against the r31 reference would pass on a tree that had
    merely dropped the `.detach()` and kept the span.
    """
    leaf_value = torch.tensor([0.3, -0.7, 1.1], dtype=DTYPE)

    def pieces(leaf):
        h_source = leaf * 0.5
        h_target = leaf * 1.5 + 0.4
        score = -(leaf ** 2) * 0.1
        return h_source, h_target, score

    def under_test(leaf):
        h_source, h_target, score = pieces(leaf)
        return loss(score, h_source, h_target).sum()

    def r31_reference(leaf):
        h_source, h_target, score = pieces(leaf)
        return ((_upper(h_source, h_target) - score) / LN2).sum()

    def r29_detached(leaf):
        h_source, h_target, score = pieces(leaf)
        upper = _upper(h_source, h_target)
        return ((upper - score) / (upper + LN2).detach()).sum()

    def r29_attached(leaf):
        h_source, h_target, score = pieces(leaf)
        upper = _upper(h_source, h_target)
        return ((upper - score) / (upper + LN2)).sum()

    measured = _gradient(leaf_value, under_test)
    r31 = _gradient(leaf_value, r31_reference)
    detached = _gradient(leaf_value, r29_detached)
    attached = _gradient(leaf_value, r29_attached)

    assert torch.allclose(measured, r31, atol=1e-14, rtol=0.0), package
    assert not torch.allclose(measured, detached, atol=1e-8, rtol=0.0), (
        f"{package}: the gradient still matches r29's detached span")
    assert not torch.allclose(measured, attached, atol=1e-8, rtol=0.0), (
        f"{package}: the gradient still matches r29's attached span")

    # Each disagreement has to be a margin, not a tolerance.
    assert float((r31 - detached).abs().max()) > 1e-3, package
    assert float((r31 - attached).abs().max()) > 1e-3, package


@pytest.mark.parametrize("package,loss,_bounds", IMPLEMENTATIONS,
                         ids=[name for name, _, _ in IMPLEMENTATIONS])
def test_the_gradient_flows_through_the_upper_end(package, loss, _bounds) -> None:
    """`U_c` keeps its gradient: it is half of the corrected score.

    With the span gone the only route into the loss is the numerator, so a tree
    that cast `U_c` to a float would report a zero gradient here. Measured
    against the analytic derivative of `(U_c - J_c) / ln 2` with respect to the
    entropy difference, which is `tanh(d/4) / (2 ln 2)`.
    """
    difference = torch.tensor([1.3], dtype=DTYPE, requires_grad=True)
    h_source = torch.zeros(1, dtype=DTYPE)
    score = torch.tensor([-0.25], dtype=DTYPE)

    loss(score, h_source, difference).sum().backward()
    assert difference.grad is not None, package
    expected = math.tanh(1.3 / 4.0) / (2.0 * LN2)
    assert float(difference.grad) == pytest.approx(expected, abs=1e-12), package
    assert abs(expected) > 1e-2, "fixture: the gradient must not be ~0"


# ------------------------------------------------------ the bounded interval


@pytest.mark.parametrize("package,loss,bounds", IMPLEMENTATIONS,
                         ids=[name for name, _, _ in IMPLEMENTATIONS])
def test_the_loss_stays_in_the_unit_interval_over_the_theorems_range(
        package, loss, bounds) -> None:
    """[0, 1] across the range r31's theorem admits, class by class.

    `J_c` is not free: the theorem says `0 <= U_c - J_c <= ln 2`, so the sweep
    draws `U_c` from a wide range of entropy differences and `J_c` from the
    whole interval the theorem permits, endpoints included. A quotient that
    left [0, 1] by any margin that is not rounding would contradict the theorem
    the change rests on.

    **The excursion is measured, not tolerated, and here is why it is not
    zero.** At `fraction = 1.0` the fixture builds `J_c = U_c - ln 2` and the
    implementation then computes `(U_c - J_c) / ln 2`. That subtraction does
    not round-trip in binary: for `|H_t - H_s| >= 6` the recovered difference
    lands one ulp above `ln 2`, so the quotient is `1 + 2.22e-16`. That is this
    test's arithmetic, not the module's -- the module divides by a literal and
    cannot overshoot by construction. Asserting a flat `<= 1` would therefore
    be asserting that `U_c - (U_c - ln 2) == ln 2` exactly, which is false in
    float64 and has nothing to do with Eq. (36).

    So the assertion is on the SIZE of the worst excursion. Red stays
    reachable with room to spare: any denominator other than `ln 2` moves the
    quotient by a factor, not by an ulp -- dividing by `0.1` sends the worst
    case to about 6.9 and dividing by `U_c + ln 2` sends it to 0.13, both
    enormous against the 1e-12 ceiling below.
    """
    generator = torch.Generator().manual_seed(20261017)
    differences = torch.linspace(-12.0, 12.0, 25, dtype=DTYPE)
    fractions = torch.cat([torch.tensor([0.0, 1.0], dtype=DTYPE),
                           torch.rand(23, generator=generator, dtype=DTYPE)])

    checked = 0
    strictly_outside = 0
    worst_excursion = 0.0
    for difference in differences:
        h_source = torch.zeros(1, dtype=DTYPE)
        h_target = difference.reshape(1)
        _lower, upper = bounds(h_source, h_target)
        for fraction in fractions:
            score = upper - fraction * LN2
            value = float(loss(score, h_source, h_target))
            checked += 1
            if value < 0.0 or value > 1.0:
                strictly_outside += 1
            worst_excursion = max(worst_excursion, -value, value - 1.0)

    assert checked == 625, checked
    assert worst_excursion < 1e-12, (
        f"{package}: worst excursion {worst_excursion:.3e} beyond [0, 1] -- "
        f"too large to be rounding ({strictly_outside} of {checked} strictly "
        f"outside)")
    # And the quotient has to equal the fraction it was built from, which is
    # the actual content: the loss READS the corrected score in units of ln 2.
    h_source = torch.zeros(1, dtype=DTYPE)
    h_target = torch.tensor([7.0], dtype=DTYPE)
    _lower, upper = bounds(h_source, h_target)
    for fraction in (0.0, 0.25, 0.5, 0.75):
        value = float(loss(upper - fraction * LN2, h_source, h_target))
        assert value == pytest.approx(fraction, abs=1e-13), (package, fraction)


@pytest.mark.parametrize("package,loss,bounds", IMPLEMENTATIONS,
                         ids=[name for name, _, _ in IMPLEMENTATIONS])
def test_the_ends_of_the_theorem_map_to_the_ends_of_the_interval(
        package, loss, bounds) -> None:
    """`U_c - J_c = 0` gives 0 and `U_c - J_c = ln 2` gives exactly 1.

    Under r29 the second one did not: `ln 2 / (U_c + ln 2)` is below 1 for
    every class whose marginal entropies differ, so the top of the interval was
    unreachable except at `U_c = 0`. That is the arbitrariness r31 removes, and
    it is asserted rather than described.
    """
    h_source = torch.tensor([0.0], dtype=DTYPE)
    h_target = torch.tensor([5.0], dtype=DTYPE)
    _lower, upper = bounds(h_source, h_target)
    assert float(upper) > 1.0, "fixture: U_c must be well away from 0"

    aligned = loss(upper, h_source, h_target)
    assert float(aligned) == pytest.approx(0.0, abs=1e-14), package

    worst = loss(upper - LN2, h_source, h_target)
    assert float(worst) == pytest.approx(1.0, abs=1e-14), package

    # The r29 map does NOT reach 1 at this class, which is what makes the
    # assertion above a change and not a restatement.
    assert float(LN2 / (float(upper) + LN2)) < 0.4, package


# ===================================================================
# What the SENTENCES say, now that the mechanism under them has moved
# ===================================================================
#
# Everything above checks arithmetic. The defect this section exists for is a
# different one: a paragraph that described r29's denominator correctly, was
# never executed, and goes on reading as current doctrine after the denominator
# became a literal. `src/CREDA/models.py` carried the sharpest instance -- "the
# span is detached inside `class_global_loss_cosh`, which is r28's own rule" --
# beside a call to a function whose denominator is now `math.log(2.0)`, so
# there is nothing left in it to detach.
#
# Two of these assertions do more than match a string. The collapse figure is
# RECOMPUTED from the configuration its own comment names and the written
# number is held to the measurement, so the next revision that moves Eq. (36)
# reddens the prose instead of aging it quietly; and the declaration test reads
# the equations the module implements out of the revisions themselves.

import ast
import inspect
import pathlib
import re

from CREDA import bounds as creda_bounds_module
from CREDA import models as creda_models

#: The source of each file whose prose these tests hold to the mechanism.
SOURCES = {
    'models': pathlib.Path(creda_models.__file__).read_text(),
    'bounds': pathlib.Path(creda_bounds_module.__file__).read_text(),
}


def _toy_source() -> str:
    from CREDA import toy_experiment
    return pathlib.Path(toy_experiment.__file__).read_text()


def _plots_source() -> str:
    from CREDA import toy_plots
    return pathlib.Path(toy_plots.__file__).read_text()


def _training_pipeline_source() -> str:
    """Read, not imported, because importing it pulls the ResNet campaign in."""
    return (pathlib.Path(creda_models.__file__).parent
            / 'training_pipeline.py').read_text()


def _default_fixture():
    """One fixed seeded batch, in the shape `CREDALoss.forward` is called with.

    Three classes, both domains the same size, un-normalized float32 -- the way
    `training_pipeline.train_creda` actually hands it over.
    """
    generator = torch.Generator().manual_seed(4242)
    f_s = torch.randn(27, 8, generator=generator)
    y_s = torch.arange(27, dtype=torch.long) % 3
    f_t = f_s + torch.randn(f_s.shape, generator=generator) * 2.0
    logits = torch.randn(f_t.shape[0], 3, generator=generator)
    return f_s, f_t, y_s, torch.softmax(logits, dim=1)


def _code_only(source: str) -> str:
    """`source` with every comment and docstring removed.

    A prose test and a code test must not share a search space. This is what
    lets a paragraph name the denominator it does NOT implement without the
    assertion about the implementation matching the paragraph.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def _block(source: str, start: str, end: str) -> str:
    """The text between two anchors, with both anchors proven present.

    A missing anchor is a failure and never an empty block that every
    `assertNotIn` would pass against -- which is how a prose test turns into a
    test of nothing after an unrelated edit moves its landmark.
    """
    i = source.index(start)
    j = source.index(end, i)
    assert j > i, (start, end)
    return source[i:j]


# ------------------------------------------------- the module declaration


def test_models_declares_the_revision_its_own_equations_are_bound_to() -> None:
    """`src/CREDA/models.py` is bound to r34, not to r28.

    Not a rebind for its own sake. Every equation this module IMPLEMENTS --
    10, 22-27, 32, 33 -- is byte-identical in r28 and r31; the one that moved
    between them, Eq. (32), moved only by renaming the congruence matrix
    `B_c` to `Q_c`, a symbol this module never writes. So `r28` had stopped
    naming a binding and was naming the date of the last edit instead, which
    is exactly the drift a declaration exists to make visible.

    **The r33 -> r34 step is the same kind of rebind and the r31 -> r33 one was
    not.** r33 rewrote Eq. 37 and rescoped Eq. 26's weighting, both of which
    moved arithmetic here. r34 moved Eq. 15 -- MIL-CREDA's attention over the
    instances of a bag -- and this module is the SAMPLE-level method, with no
    bags and no attention, so it implements no equation r34 touched.
    """
    assert creda_models.__provenance__['revision'] == 'research-concept-r34.md'


def test_models_says_which_of_its_equations_it_only_applies() -> None:
    """The declaration lists 35 and 36 and the module implements neither.

    `rg 'upper - score|/ span|U_c - L_c' src/CREDA/models.py` returns nothing:
    both are reached through `CREDA.bounds`, which implements them and declares
    them too.

    **The r25 half of this claim ended on 2026-10-18.** The docstring had to
    name r25 because `bounds_form` defaulted to `conservative` and a bare r31
    would have read as a claim that the DEFAULT path computed r31's Eq. (36).
    There is one map now -- r25's pair was removed from both packages -- so
    what the docstring owes instead is the HISTORY: that the pair existed and
    where its numbers went, which is what a reader meeting `r25` in an
    artefact needs.
    """
    doc = creda_models.__doc__
    assert 'research-concept-r34.md' in doc
    assert 'r25' in doc, "what happened to r25's pair has to be named"
    declared = creda_models.__provenance__['equations']
    assert '35' in declared and '36' in declared
    # And Eq. 37 is now IMPLEMENTED here rather than approximated, which is
    # the one addition to that list.
    assert '37' in declared

    # Read the CODE and not the file. The paragraphs above discuss both
    # denominators by name -- that is their job -- so a substring search over
    # the source would match the explanation and report the module as
    # implementing what it explains. Comments and docstrings are stripped and
    # what is left is what actually runs.
    code = _code_only(SOURCES['models'])
    for owned_by_bounds in ('upper - score', 'U_c - L_c', 'GLOBAL_LOSS_SPAN',
                            'entropy_cosh_bounds'):
        assert owned_by_bounds not in code, owned_by_bounds
    # And the two it DOES call are called, so "applies" is not vacuous.
    assert 'class_global_loss_cosh' in code
    # `conservative_bounds` was the second call and left the code on
    # 2026-10-18: r33 states one pair of ends, so that function was removed
    # from `CREDA.bounds` and the fork that chose between the two collapsed.
    # Asserted as an absence, so a reader who remembers the pair is told it
    # ended rather than left to notice one call where there were two.
    assert 'conservative_bounds' not in code
    assert 'class_global_loss(' not in code


def test_the_defaults_are_r31s_and_the_published_path_asks_by_name() -> None:
    """The default moved to r31, and what the default used to protect is pinned.

    The earlier version of this test asserted the OPPOSITE -- that
    `bounds_form` still defaulted to `conservative` -- on the reasoning that
    moving it would re-scale every published number. That reasoning was sound
    about the risk and wrong about the remedy: a number protected by a default
    is protected by nothing once the default moves, and the default had to move
    because every caller that does not ask should get the formulation the
    repository currently states.

    So the protection became explicit instead. `training_pipeline.train_creda`
    is the only construction in the old notebooks' reachable set
    (`Plots_Generator` and `Results_Generator` import
    `CREDA.training_pipeline`; `CREDA.models` is imported there only to
    allow-list classes for `torch.load`, never to construct this loss), and it
    names both r25 values.

    Reachable red, three ways: reverting either default reddens the first two
    assertions, and deleting either keyword from the pin reddens the third.
    """
    signature = inspect.signature(creda_models.CREDALoss.__init__)
    assert signature.parameters['bounds_form'].default == 'entropy-cosh'
    assert signature.parameters['balance_mass'].default == 'trace'
    # Unmoved, and the reason the move is safe for every site that does not
    # read these two: both are read only under a flag that is still off.
    assert signature.parameters['normalize_by_bounds'].default is False
    assert signature.parameters['balance_mix_blocks'].default is False

    pin = _block(_training_pipeline_source(), 'creda_loss_fn = CREDALoss(',
                 'for epoch in range(epochs)')
    # **`bounds_form='conservative'` left this pin on 2026-10-18.** The map it
    # named was removed from both packages, so there is nothing for the pin to
    # hold the path to: `models.BOUNDS_FORMS` keeps the keyword with one legal
    # value precisely so a run still asking gets told. `balance_mass` IS still
    # a real alternative and is still pinned.
    assert "bounds_form=" not in pin, (
        "the published path names a bounds form again, and there is only one "
        "map left for it to name")
    # **`normalize_by_bounds=False` is a DECISION of the first paper's ResNet
    # arm, not an inherited default, and on 2026-10-18 it became one it says
    # out loud.** The operator's own statement of the fact: that arm is `False`
    # always, and every other measurement this repository takes is `True` --
    # the toyset names `NORMALIZAR_POR_COTAS = True` and the bag method has no
    # such flag at all, its normalization always happening. So the default
    # `False` was serving exactly one caller, and that caller was the one not
    # naming it: the published numbers rested on a default rather than on a
    # decision, which is the same defect this file already corrected for
    # `bounds_form`.
    assert "normalize_by_bounds=False" in pin, (
        "the published ResNet path no longer names normalize_by_bounds, so it "
        "inherits whatever the default happens to be -- and that arm's numbers "
        "were measured with the bounds normalization OFF, deliberately")
    assert "balance_mass='cardinality'" in pin, (
        "the published path no longer asks for balance_mass='cardinality' by "
        "name, so it is back to taking whatever the constructor defaults to")


def test_the_pin_has_nothing_left_to_pin_on_the_bounds_axis() -> None:
    """**The inverse of what this asserted until 2026-10-18.**

    It measured that the pinned values and the moved defaults computed
    DIFFERENT numbers on the `bounds_form` axis, and that the pinned side was
    r25's Eq. (36) recomputed from `conservative_bounds` -- the map the
    published numbers were measured under. A pin naming the same map the
    defaults already select would have been a comment, and that was the point.

    r33 states one pair of ends. `conservative_bounds` and `class_global_loss`
    were removed from both packages, so there is no second map to be pinned
    away from: asking for the retired one is now a refusal, and that refusal is
    the only thing left to assert on this axis. `balance_mass` is a separate
    axis, still real and still pinned, and its own half of the claim is below.
    """
    f_s, f_t, y_s, g_t = _default_fixture()

    # One axis is gone: the keyword survives so the refusal can name what
    # happened, and `models.BOUNDS_FORMS` holds its single legal value.
    assert creda_models.BOUNDS_FORMS == ('entropy-cosh',)
    with pytest.raises(ValueError) as raised:
        creda_models.CREDALoss(
            sigma='auto', use_entropy_weighting=True, normalize_by_bounds=True,
            bounds_form='conservative', balance_mass='cardinality')
    assert 'bounds_form' in str(raised.value)
    assert 'entropy-cosh' in str(raised.value)

    # The OTHER axis is still load-bearing, and that half is unchanged: the
    # cardinality balance and the trace balance compute different numbers, so
    # `training_pipeline`'s surviving pin is not a comment.
    # `normalize_by_bounds` is deliberately OFF here: `entropy-cosh` is the
    # only map left and it refuses `balance_mass='cardinality'` outright, so
    # the two balances can only be compared where `balance_mass` is actually
    # read -- under `balance_mix_blocks`, which is its own flag.
    shared = dict(sigma='auto', use_entropy_weighting=True,
                  weight_cross_block=True, balance_mix_blocks=True)
    cardinality = creda_models.CREDALoss(**shared, balance_mass='cardinality')
    trace = creda_models.CREDALoss(**shared, balance_mass='trace')
    with torch.no_grad():
        on_the_pin = cardinality(f_s, f_t, y_s, g_t, reduction='none')
        on_the_default = trace(f_s, f_t, y_s, g_t, reduction='none')
    assert not torch.allclose(on_the_pin, on_the_default, atol=1e-6, rtol=0.0), (
        "the two balances agree, so the surviving pin protects nothing and "
        "this test measures a no-op")
    assert float((on_the_pin - on_the_default).abs().max()) > 1e-3, (
        "the two balances differ by a tolerance rather than by a margin")


def test_the_moved_default_leaves_every_live_construction_bit_identical() -> None:
    """Why the move costs nothing today, measured rather than argued.

    Every `CREDALoss(...)` in `src/` leaves `normalize_by_bounds` and
    `balance_mix_blocks` at `False`, and those are the only two flags under
    which `bounds_form` and `balance_mass` are ever read. So the three live
    sites compute the same bits under either default, and the pin is a claim
    about the future rather than a repair of a present difference. Saying that
    out loud is the point: the risk the pin removes is the NEXT caller that
    turns the normalizer on without asking.

    Reachable red: making either flag read outside its guard, or defaulting
    either flag to `True`, separates the two columns below.
    """
    f_s, f_t, y_s, g_t = _default_fixture()
    for keywords in (
            # training_pipeline.py's pinned call, minus the pin.
            dict(sigma=1.5, use_entropy_weighting=True),
            # sample_adaptation.py:107 and wiring.py:342, which share a shape.
            dict(sigma='auto', lambda_creda=1.0, use_entropy_weighting=True),
            dict(sigma='auto', lambda_creda=1.0, use_entropy_weighting=False),
    ):
        # `bounds_form='conservative'` left this construction with the map it
        # named; `balance_mass` is the axis that remains, and it is the one
        # this test was always really about -- both are read only under flags
        # these keywords leave off.
        r25 = creda_models.CREDALoss(
            balance_mass='cardinality', **keywords)
        moved = creda_models.CREDALoss(**keywords)
        assert (moved.bounds_form, moved.balance_mass) == ('entropy-cosh', 'trace')
        assert r25.bounds_form == 'entropy-cosh', (
            "there is one map left, so both columns read it")
        assert not moved.normalize_by_bounds and not moved.balance_mix_blocks
        with torch.no_grad():
            assert torch.equal(r25(f_s, f_t, y_s, g_t),
                               moved(f_s, f_t, y_s, g_t)), keywords


# --------------------------------------- the sentence that outlived its rule


def test_no_prose_in_models_claims_the_span_is_detached() -> None:
    """The r29 rule, asserted in a comment beside a call that cannot hold it.

    `class_global_loss_cosh` divides by `GLOBAL_LOSS_SPAN`, a module-level
    `math.log(2.0)`. There is no span object in that function to exclude from
    the gradient, so any sentence saying one is excluded describes code that
    does not exist.
    """
    source = SOURCES['models']
    assert 'span is detached' not in source
    assert "r28's own rule" not in source
    # And the sentence that replaces it has to say what is true instead, and
    # has to say when it changed. A silent correction is the other half of the
    # same defect: a reader who remembers the rule is left to notice an
    # absence, and absences are not read.
    # The anchor moved on 2026-10-18: the `bounds_form` fork was collapsed when
    # r25's map was removed, so `if self.bounds_form == ...` is no longer in
    # this file. `# Fork 3` is the surviving landmark for the same block.
    block = _block(source, '# Fork 3: Eqs. (35)-(36).',
                   'losses = torch.stack(losses_per_class)')
    assert 'r31' in block
    assert 'ln 2' in block
    assert '2026-10-17' in block, 'the correction has to carry its date'
    assert 'r29' in block, 'and has to name the revision it corrects'


def test_the_detach_that_does_exist_is_not_the_one_r29_removed() -> None:
    """`detach_target_entropy` stays, and is a different mechanism.

    The sweep that removed the span sentence must not take this with it: this
    one detaches `h_t` inside the SCORE, is a declared keyword, defaults off,
    and has nothing to do with Eq. (36)'s denominator.
    """
    signature = inspect.signature(creda_models.CREDALoss.__init__)
    assert signature.parameters['detach_target_entropy'].default is False
    assert 'h_t.detach() if self.detach_target_entropy else h_t' in SOURCES['models']


# ------------------------------- the written number, held to a measurement


def _collapse_configuration():
    """The configuration `BOUNDS_FORMS`' comment names, rebuilt and measured.

    `K_s = I / n_s` with the target collapsed far enough that the cross block
    vanishes, at `n_s = 43`. `H_2(I/n_s) = ln n_s = 3.7612` and the collapsed
    target's normalized marginal is idempotent, so `H_2 = 0`.
    """
    def h2(matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix / torch.trace(matrix)
        return -torch.log(torch.trace(matrix @ matrix))

    n = 43
    h_source = h2(torch.eye(n, dtype=DTYPE))
    h_target = h2(torch.ones(n, n, dtype=DTYPE))
    # `J_c` at that configuration, from Eq. (34)'s own closed form.
    score = torch.tensor(0.5173, dtype=DTYPE)
    return h_source, h_target, score


def test_the_collapse_figure_in_the_prose_is_the_one_r31_computes() -> None:
    """The comment's number is recomputed, not trusted.

    This is the assertion that stops the figure from aging silently a third
    time. It was 0.1773 under r29's per-class denominator -- reproduced below
    to prove the configuration is the one the comment means -- and r31's
    constant denominator moves it. Whatever Eq. (36) becomes next, this
    reddens rather than drifting.
    """
    h_source, h_target, score = _collapse_configuration()
    assert float(h_source) == pytest.approx(math.log(43), abs=1e-10)
    assert float(h_target) == pytest.approx(0.0, abs=1e-12)

    _lower, upper = creda_bounds_module.entropy_cosh_bounds(h_source, h_target)
    r31 = float(creda_bounds_module.class_global_loss_cosh(score, h_source, h_target))
    r29 = float((upper - score) / (upper - creda_bounds_module.ENTROPY_COSH_LOWER))

    # The configuration is the comment's: it reproduces the figure the comment
    # used to carry, to four places.
    assert r29 == pytest.approx(0.1773, abs=5e-5), r29
    # And the two are genuinely different, so the prose had to move.
    assert abs(r31 - r29) > 0.1, (r31, r29)
    # r30's theorem, on the configuration that motivated the whole pair.
    assert 0.0 <= float(upper - score) <= LN2

    block = _block(SOURCES['models'], '#: The maps `CREDALoss.bounds_form` accepts',
                   'BOUNDS_FORMS = ')
    # Anchored to the sentence that says what the flag READS. The dated
    # paragraph below it quotes the same figure while explaining the change,
    # and a bare `in block` passed on that second occurrence while the live
    # sentence carried a wrong number -- found by mutating it.
    assert f'entropy-cosh reads {r31:.4f}' in block, (
        f"the live sentence does not carry r31's measured {r31:.4f}")
    # The superseded figure may stay only as the thing being corrected.
    assert f'`0.1773` becomes `{r31:.4f}`' in block
    assert f'{r29:.4f}' in block


def test_the_aligned_cell_is_zero_under_both_revisions() -> None:
    """Why only one of the two numbers in that sentence moved.

    At perfect alignment the two marginals are the same matrix, so `U_c = 0`
    and the two denominators coincide at `ln 2`. The 0.0000 is not a figure
    that survived by luck and the comment may keep it.
    """
    h = torch.tensor(3.7, dtype=DTYPE)
    _lower, upper = creda_bounds_module.entropy_cosh_bounds(h, h)
    assert float(upper) == pytest.approx(0.0, abs=1e-14)
    value = creda_bounds_module.class_global_loss_cosh(
        torch.tensor(0.0, dtype=DTYPE), h, h)
    assert float(value) == pytest.approx(0.0, abs=1e-14)


# ------------------------------------------- the banners and the two ends


def test_the_bounds_banners_name_the_revision_each_function_implements() -> None:
    """A section banner is prose and ages exactly like a docstring.

    Two of them said `revision 28` above functions that implement r31.

    **They name an ORIGIN now rather than the current binding, and that is the
    repair rather than a third rewrite.** A banner reading `revision 33` above
    a function bound to r34 is the same defect in a new coat, and bumping the
    number every time the stamp moves guarantees the next revision inherits it.
    Eq. (21)'s banner already said `byte-identical from revision 25 on`; the
    other two now say the same thing about r33, which is the fact that does not
    go stale.
    """
    source = SOURCES['bounds']
    assert 'revision 28' not in source
    assert 'revision 31' not in source, (
        "a banner still names r31 above a function bound to r34")
    assert ', revision 33' not in source, (
        "a banner names r33 as a binding rather than as the origin of a form "
        "that is byte-identical from there on")
    assert 'Eq. (35), byte-identical from revision 33 on' in source
    assert 'Eq. (36), byte-identical from revision 33 on' in source
    # Eq. (21) carries no revision because it has not changed since r25, and
    # its banner says that rather than naming the current one.
    assert 'Eq. (21), byte-identical from revision 25 on' in source


def test_eq_thirty_five_no_longer_claims_to_state_the_lower_end() -> None:
    """r31 narrowed Eq. (35) to `U_c` alone; `L_c` is Eq. (34)'s.

    Measured on the revisions: r28's Eq. (35) block carries `L_c = - \\ln 2`
    and r31's does not, while Eq. (34) is byte-identical in both and states
    `- \\ln 2 \\leq J_c`. `entropy_cosh_bounds` still RETURNS both ends -- that
    is its signature and nothing here changes it -- but it may not attribute
    the lower one to an equation that stopped stating it.
    """
    doc = creda_bounds_module.entropy_cosh_bounds.__doc__
    assert "r28's Eq. (35): L_c = -ln 2" not in doc
    # The SUMMARY line is the attribution a reader gets from `help()` and from
    # an editor's tooltip, so that is the line held to it. Asserting `Eq. (34)`
    # anywhere in the docstring passed while the body said `Eq. (99)`.
    summary = doc.splitlines()[0]
    assert "Eq. (35)'s" in summary and "Eq. (34)'s `L_c`" in summary, summary
    assert 'Eq. (34), byte-identical in r28 and r31' in doc
    assert 'r31' in doc
    # The function's behaviour is unchanged, and that is asserted and not assumed.
    lower, _upper = creda_bounds_module.entropy_cosh_bounds(
        torch.tensor(1.0, dtype=DTYPE), torch.tensor(2.0, dtype=DTYPE))
    assert lower == pytest.approx(-math.log(2.0), abs=1e-15)


def test_neither_package_attributes_the_lower_end_to_equation_thirty_five() -> None:
    """The same claim, swept over both copies instead of fixed in one.

    `entropy_cosh_bounds` exists twice -- `MIL_CREDA.global_term` owns it and
    `CREDA.bounds` deliberately copies it -- and every arithmetic test in this
    file is parametrized over both for exactly that reason. The prose test
    above is not: it reads `CREDA.bounds` alone, and the `MIL_CREDA` twin kept
    attributing `L_c = -ln 2` to Eq. (35) long after r31 narrowed that equation
    to its upper end. A defect fixed in one copy and left standing in the other
    reads as fixed, which is worse than reading as broken.

    Reachable red: restoring either summary to `Eq. (35): L_c = -ln 2` reddens
    its own entry below, and the `and` makes neither entry able to carry the
    other.
    """
    summaries = {
        'MIL_CREDA': mil_bounds.__doc__.splitlines()[0],
        'CREDA': creda_bounds.__doc__.splitlines()[0],
    }
    for package, summary in summaries.items():
        assert 'Eq. (35)' in summary, (package, summary)
        assert "Eq. (34)" in summary, (
            f"{package}: the summary line does not say which equation states "
            f"the lower end it returns: {summary!r}")
        assert 'Eq. (35): L_c' not in summary, (
            f"{package}: the lower end is attributed to Eq. (35), which r31 "
            f"narrowed to its upper end: {summary!r}")
    # And both functions still RETURN the lower end, so the correction is about
    # the attribution and never about the behaviour.
    for package, bounds in (('MIL_CREDA', mil_bounds), ('CREDA', creda_bounds)):
        lower, _upper = bounds(torch.tensor(1.0, dtype=DTYPE),
                               torch.tensor(2.0, dtype=DTYPE))
        assert lower == pytest.approx(-math.log(2.0), abs=1e-15), package


# --------------------------------------- the two files downstream of the term


def test_the_axis_comment_gives_r31s_reason_for_the_unit_interval() -> None:
    """`toy_plots.py` fixed its axis to [0, 1] and said why. The why moved.

    It argued from the two ends -- `L_c = -ln 2` and `U_c = ...` -- which was
    r28's construction. Under r31 the interval comes from r30's theorem on the
    span instead. The axis is unchanged and its justification is not.
    """
    source = _plots_source()
    # Anchored on a line that occurs once. `# Lineal y 0 a 1 FIJO` occurs
    # TWICE -- the phase-1 panel's axis carries the same decision for Eq. 21 --
    # and anchoring there silently measured the wrong block.
    assert source.count('# Lineal y 0 a 1 FIJO') == 2
    block = _block(source, '# vistazo. El de valid va sobre el MISMO lote',
                   'a.set_ylim(0.0, 1.0)')
    assert 'r31' in block
    assert 'r28, Ecs. 35-36' not in block
    assert 'ln 2' in block
    # The axis itself stays where it was.
    assert 'a.set_ylim(0.0, 1.0)' in source


def test_the_toy_block_dates_the_denominator_it_was_measured_under() -> None:
    """`toy_experiment.py` carries a five-row r25-against-r28 table.

    Those columns are headed by the revisions they were measured under, so
    they are dated history and they stay -- they are the evidence r28 acted
    on. What was missing is that the live flag no longer computes either
    column's map: r31 divides by `ln 2`. The block has to say so rather than
    leaving a reader to assume the `r28` column is what runs.
    """
    source = _toy_source()
    block = _block(source, '#: Si el puntaje por clase se mapea',
                   'NORMALIZAR_POR_COTAS = True')
    # The evidence stays, exactly as `test_the_prose_no_longer_argues_for_the
    # _superseded_map` already requires of the same block.
    assert '8.2155' in block and '5.1%' in block
    assert '0.1773' in block
    # And the block says which revision the live flag computes, AS the claim
    # and not as a loose token: `r31` appeared three times in this block while
    # the sentence carrying the correction said "una revision posterior".
    assert 'r31 reemplazo el divisor de la Ec. 36' in block
    assert 'r31 divide por el literal `ln 2`' in block
    assert 'r31 = r28 * (U_c + ln 2) / ln 2' in block
    # And it admits what it did NOT recompute, rather than implying all five.
    assert 'NO se recalcularon' in block


def test_every_r28_mention_left_in_creda_is_history_and_not_a_live_rule() -> None:
    """The sweep's own closing condition, as a condition and not a list.

    An `r28` in `src/CREDA/` is legitimate when it says WHEN something
    happened. It is a defect when it says what the code does NOW. This cannot
    be told apart mechanically in general, so what is asserted is the narrow
    mechanical part: no file may name r28 as the revision it implements or is
    bound to, and no file may still attribute Eq. (36)'s denominator to a
    span.
    """
    everything = dict(SOURCES, toy_experiment=_toy_source(),
                      toy_plots=_plots_source())
    for name, source in everything.items():
        # Naming r28 in prose is allowed and this file does it deliberately.
        # Naming it as the binding is not: that is the declaration's own slot.
        assert '"revision": "research-concept-r28.md"' not in source, name
        assert "'revision': 'research-concept-r28.md'" not in source, name
        assert "is r28's own rule" not in source, name
        assert "revision 28" not in source, name
    # The one live binding in CREDA that still reads r25 does so on purpose.
    # **Both read r34 since 2026-10-18.** `CREDA.bounds` declared r25 while it
    # held `conservative_bounds`/`class_global_loss`; those were removed, so
    # every equation it holds is r33's and the stamp could finally move -- and
    # r34 moved only Eq. 15, which neither module implements.
    assert creda_bounds_module.__provenance__['revision'] == 'research-concept-r34.md'
    assert creda_models.__provenance__['revision'] == 'research-concept-r34.md'
