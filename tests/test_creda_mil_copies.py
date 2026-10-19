"""Every function CREDA keeps a copy of, and what holds each copy to its original.

`src/CREDA/bounds.py` states the rule these tests enforce: CREDA is the baseline
MIL-CREDA is measured against, so a CREDA module that imported from `MIL_CREDA`
would point the dependency backwards and contaminate the comparison at the
import line. The copies are deliberate. What the same docstring also says is
that "what keeps a copy from drifting is a test, not a promise" -- and until
2026-10-18 that promise was kept for two of the five copies. Eq. (24)'s weight
was one of the three with no net, and it had already drifted: `CREDALoss.forward`
lost its epsilon that day and `toy_experiment.incertidumbre_declarada` kept
hers, so the two halves of one equation disagreed and nothing failed.

**The roster below DERIVES itself, which is the whole point.** A list of copies
maintained by hand is a list somebody has to remember to extend, and the copy
that drifts is precisely the one that was added without anybody remembering.
`test_the_roster_of_copies_derives_itself` walks both packages, intersects the
public names they define, and demands a ruling for each -- so a new copy turns
the suite red on the commit that introduces it, before it can drift. The
rulings are declared because a ruling is a judgement; the roster is derived
because a roster is a fact.

Two kinds of copy this cannot derive, and both are named rather than hidden. A
copy under a DIFFERENT name on each side is invisible to a name intersection:
Eq. (9)-(10)'s entropy is `CREDALoss._renyi_entropy_order_2` here and
`renyi.quadratic_entropy` there, so it is declared in `RENAMED` and paired by
its own test. And a copy of something that is not a function at all -- a
constant, a default -- is not reached by any of this.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGES = {"CREDA": ROOT / "src" / "CREDA", "MIL_CREDA": ROOT / "src" / "MIL_CREDA"}

#: Every public name defined in BOTH packages needs one of these rulings, and
#: the reason is part of the ruling: a bare "known duplicate" would let the next
#: reader assume somebody checked.
COPIES = {
    "source_bound": (
        "Eq. (21)'s supremum. Baseline arithmetic, so it stays a copy; held "
        "equal by test_creda_sample.test_source_bound_copy_agrees_with_the_original."
    ),
    "source_loss": (
        "Eqs. (21) and (39)'s supervised term. Baseline arithmetic, so it stays "
        "a copy; held equal by "
        "test_creda_sample.test_source_loss_copy_agrees_with_the_original."
    ),
    "entropy_cosh_bounds": (
        "Eq. (35)'s two ends. Baseline arithmetic, so it stays a copy; "
        "test_r31_ec36 parametrizes the SAME properties over both "
        "implementations (its IMPLEMENTATIONS tuple), which is weaker than an "
        "equality -- two implementations can satisfy every declared property "
        "and still differ on a value no property names."
    ),
    "class_global_loss_cosh": (
        "Eq. (36)'s per-class loss. Same ruling and same weaker net as "
        "entropy_cosh_bounds, through the same parametrization."
    ),
    "as_tensor": (
        "`MIL_CREDA.__init__`'s plumbing, copied so that importing plumbing "
        "does not re-establish the dependency the copies exist to avoid. "
        "Paired by test_the_plumbing_copies_agree_with_their_originals below."
    ),
    "as_matrix": (
        "Same ruling as as_tensor, and paired by the same test below."
    ),
}

#: Copies that carry a different name on each side, which a name intersection
#: cannot see. One entry per equation, with the pairing test that holds it.
RENAMED = {
    "Eq. (9)-(10)": (
        "CREDALoss._renyi_entropy_order_2 here, renyi.quadratic_entropy composed "
        "with renyi.trace_normalize there. Paired by "
        "test_the_renyi_entropy_copy_agrees_with_the_original below."
    ),
}

#: The one equation CREDA IMPORTS instead of copying, and why it is allowed to.
#: Eq. (24) is the method's treatment -- the toy's two arms differ in exactly
#: that weight -- and a treatment has no independence to lose from the method
#: that proposes it. Decided by the operator on 2026-10-18.
IMPORTED = ("confidences",)


#: Vendored upstream lives here, and its defects are not ours to repair. The
#: only one that reaches this file is a docstring that spells LaTeX in a
#: non-raw string (`comparators/daln/upstream_grl.py:41`), which makes
#: `ast.parse` emit a SyntaxWarning. Tolerated by path and never by warning
#: class, so the same mistake in our own sources still fails below.
VENDORED = "comparators"


def _public_functions(root: pathlib.Path) -> dict[str, list[str]]:
    """Every public function and method the package defines, by name."""
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            # `filename=` is what makes the warning nameable at all: without it
            # the message reads `<unknown>:41` and cannot be attributed.
            tree = ast.parse(path.read_text(), filename=str(path))
        assert not caught or VENDORED in path.parts, (
            f"{path.relative_to(ROOT)} raised {[str(w.message) for w in caught]} "
            "while being parsed, and it is not vendored, so it is ours to fix")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            found.setdefault(node.name, []).append(
                f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def test_the_roster_of_copies_derives_itself() -> None:
    """A copy nobody declared turns this red, and a ruling without a subject too.

    Both directions matter. The first is the defect this exists for: a function
    added to CREDA under a name `MIL_CREDA` already uses is a new copy, and it
    arrives with nothing holding it to its original. The second is the defect
    this file would otherwise grow: a ruling that outlived its copy reads like
    protection and protects nothing.
    """
    defined = {name: _public_functions(root) for name, root in PACKAGES.items()}
    derived = set(defined["CREDA"]) & set(defined["MIL_CREDA"])

    undeclared = sorted(derived - set(COPIES))
    assert not undeclared, (
        "these public names are defined in BOTH packages and no ruling covers "
        "them, so nothing holds each copy to its original: "
        + "; ".join(
            f"{n} ({', '.join(defined['CREDA'][n])} vs "
            f"{', '.join(defined['MIL_CREDA'][n])})" for n in undeclared)
        + ". Add it to COPIES with the test that pairs it, or import it when it "
          "belongs to the method rather than to the baseline.")

    orphaned = sorted(set(COPIES) - derived)
    assert not orphaned, (
        "these rulings no longer name a function defined in both packages, so "
        f"they protect nothing: {orphaned}")


def test_the_renamed_copy_is_still_there_under_both_names() -> None:
    """`RENAMED` is a hand-held list, so it gets the check a derived one gets.

    A renamed copy cannot be derived by name. What CAN be derived is whether
    each side still defines the symbol the entry claims, which is what turns a
    retired copy into a red test instead of a stale paragraph.
    """
    from CREDA.models import CREDALoss
    from MIL_CREDA import renyi

    assert "Eq. (9)-(10)" in RENAMED
    assert callable(CREDALoss._renyi_entropy_order_2)
    assert callable(renyi.quadratic_entropy)
    assert callable(renyi.trace_normalize)


def test_the_renyi_entropy_copy_agrees_with_the_original() -> None:
    """Eq. (9)-(10) on both sides, on ordinary input and on what each refuses.

    This copy had NO net until 2026-10-18, which is also the day the two sides
    came to agree: CREDA's carried `trace(K) + 1e-6` and a second epsilon inside
    the log, so before that commit this assertion would have failed by up to
    8.5e-06 nats on an ordinary block and by more than a nat on a small trace.
    """
    from CREDA.models import CREDALoss
    from MIL_CREDA.renyi import quadratic_entropy, trace_normalize

    loss = CREDALoss()
    torch.manual_seed(0)
    for n, d in ((7, 5), (12, 3), (40, 8), (129, 16)):
        X = torch.randn(n, d, dtype=torch.float64)
        K = torch.exp(-torch.cdist(X, X) ** 2 / 2.0)
        assert float(loss._renyi_entropy_order_2(K)) == float(
            quadratic_entropy(trace_normalize(K))), (n, d)

    # And the refusal: Eq. (9) does not define a normalization for a
    # non-positive trace, and BOTH sides say so rather than returning a number.
    dead = torch.zeros((4, 4), dtype=torch.float64)
    with pytest.raises(ValueError):
        loss._renyi_entropy_order_2(dead)
    with pytest.raises(ValueError):
        trace_normalize(dead)


def test_the_plumbing_copies_agree_with_their_originals() -> None:
    """`as_tensor` and `as_matrix`, which had no net either.

    **Neither one refuses anything, and that is measured rather than assumed.**
    A first draft of this test asserted that a 1-D input was refused; both
    sides reshape it to one row instead, so the assertion was testing a guard
    that does not exist. What is worth pinning is the shape each one PRODUCES,
    including the degenerate inputs -- a scalar, a flat list, a 3-D tensor that
    passes straight through -- because that silent reshape is exactly where two
    copies of a helper drift apart without any caller noticing.
    """
    from CREDA import bounds
    import MIL_CREDA

    for value in (3.0, [1.0, 2.0], torch.tensor([1.0, 2.0]), [[1.0, 2.0]]):
        assert torch.equal(bounds.as_tensor(value), MIL_CREDA.as_tensor(value)), value
    for value in (3.0, [1.0, 2.0], torch.tensor([1.0, 2.0]),
                  [[1.0, 2.0], [3.0, 4.0]], torch.eye(3), torch.zeros(2, 2, 2)):
        copy, original = bounds.as_matrix(value), MIL_CREDA.as_matrix(value)
        assert torch.equal(copy, original), value
        assert copy.shape == original.shape, (value, copy.shape, original.shape)


def test_equation_twenty_four_has_exactly_one_implementation() -> None:
    """The lock on the 2026-10-18 decision: CREDA imports it and computes it nowhere.

    The drift this replaces was not detectable from either side alone. Both
    copies were correct Eq. (24) when they were written; one of them stopped
    being so the day the other was repaired. What makes this checkable is that
    there is now a single implementation, so the only thing to assert is that
    nobody has written a second one.
    """
    import CREDA.models as models
    import CREDA.toy_experiment as toy
    from MIL_CREDA.confidence import confidences

    assert models.confidences is confidences
    assert toy.confidences is confidences

    # No second normalizer by `ln C` anywhere under CREDA: that division is the
    # signature of this equation, and a new one means a new copy.
    marks = ("log(CLASES)", "torch.log(torch.tensor(float(g_t.shape")
    offenders = []
    for path in sorted(PACKAGES["CREDA"].rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if any(mark in line for mark in marks):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, (
        "a second implementation of Eq. (24)'s normalizer appeared under "
        f"src/CREDA: {offenders}. Call MIL_CREDA.confidence.confidences instead.")

    # And the value the equation requires: the uniform distribution weighs
    # nothing. Asserted as a BOUND and not as an exact zero, because float32
    # cannot deliver one here and the reason is the input, not the estimator:
    # `1/3` and `1/10` are not representable, so `sum g^2` misses `1/C` by an
    # ulp before the estimator sees it. Measured: exactly 0.0 at C=2, where
    # `1/2` IS representable, and 1.19e-07 at C=3 and C=10 -- one float32
    # epsilon, the smallest a wrong answer could be. The double-precision
    # normalizer does not change that and was never going to.
    eps = float(torch.finfo(torch.float32).eps)
    for classes in (2, 3, 10):
        uniform = torch.full((1, classes), 1.0 / classes)
        assert abs(float(confidences(uniform)[0])) <= eps, classes
    exact = torch.full((1, 2), 0.5)
    assert float(confidences(exact)[0]) == 0.0
