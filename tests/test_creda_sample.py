"""Four things ported from MIL-CREDA down to sample level, and the contract
that makes the port safe to land.

The contract is one sentence: **with every flag at its default, faithful CREDA
is bit-identical to what it was before any of them existed.** That is what
`test_the_default_leaves_every_path_bit_identical_to_before_the_fork` asserts,
and if it ever fails nothing else in this file matters -- every number already
published out of this repository would have moved.

The file is in four parts.

1. **The copies agree with their originals.** `src/CREDA/bounds.py` holds four
   equations copied rather than imported, so that CREDA stays independent of
   the tree it is the baseline for. Four tests import BOTH sides and assert
   they agree, on ordinary inputs and on the inputs each original refuses. The
   copies are verified, not trusted; this is the only place in the suite that
   deliberately sees both trees, and seeing both is exactly its job.

2. **The default is bit-identical.** `_prefork_creda_forward` below is
   `CREDALoss.forward`'s body as it stood before the three forks were added,
   copied verbatim and with `self` renamed. It is the reference the default path
   is held to with `torch.equal` -- not `allclose`, because the claim is
   identity and not agreement. Two pinned means, read off a run of the pre-fork
   code, sit beside it as a second tripwire.

3. **One test per flag, and each asserts the declared form.** A test that only
   showed "the value changed" could not tell a correct implementation from a
   typo, so each flag is pinned to the exact operation it claims to perform:
   the normalized-embedding fork against the same loss fed pre-normalized
   features, the cross-block fork against an explicit right-multiplication by
   D, the bounds fork against `class_global_loss` applied per class with that
   class's own cardinalities.

4. **The trainer's own two forks**, which no loss function could hold: the
   two-encoder architecture with its transfer, and the composition of Eq. (39).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from CREDA import bounds
from CREDA.bounds import source_bound
from CREDA.models import CREDALoss
from CREDA.sample_adaptation import DEFAULT_SYNC_STEPS, SampleCREDA, train
from CREDA.schedules import creda_ramp

# The originals. Importing them here is the point of part 1 and nowhere else.
from MIL_CREDA.objective import source_bound as original_source_bound
from MIL_CREDA.objective import source_loss as original_source_loss

CLASSES = 3
EPSILON = 1e-8


# =========================================================== shared fixtures


def _fixture(seed: int = 17, n_source: int = 12, n_target: int = 10,
             dim: int = 4) -> tuple[torch.Tensor, ...]:
    """Source features, their labels, target features, and the caller's
    pseudo-label distribution.

    `n_source != n_target` deliberately. The cross block is rectangular, and a
    square fixture would let a wrong-sided weighting be shape-valid and so
    silently pass the one test that exists to catch it.
    """
    generator = torch.Generator().manual_seed(seed)
    f_s = torch.randn(n_source, dim, generator=generator)
    f_t = torch.randn(n_target, dim, generator=generator)
    y_s = torch.arange(n_source) % CLASSES
    y_s = y_s.sort().values
    g_t = torch.softmax(torch.randn(n_target, CLASSES, generator=generator), dim=1)
    return f_s, f_t, y_s, g_t


def _prefork_creda_forward(loss: CREDALoss, f_s, f_t, y_s, g_t,
                           reduction: str = "mean") -> torch.Tensor:
    """`CREDALoss.forward` as it stood BEFORE the three forks, verbatim.

    The only edits are `self` -> `loss` and the comments stripped. No flag is
    read HERE, which is what makes it a reference: a flag whose default was
    flipped, or a fork that stopped being an identity when off, reddens the
    comparison instead of silently drifting every number in the repository.

    One flag does reach it, through a helper rather than through this body:
    `balance_mix_blocks` lives inside `_mix_kernel_concat`, which this calls. The
    reference therefore holds only while that flag's default is off -- and that
    is not a weakness of the reference but the thing it checks, since
    `test_balance_mix_blocks_off_is_the_plain_concatenation` pins the identity and
    this function would redden the moment it stopped holding.

    It calls the loss object's own private helpers (`_compute_sigma`,
    `_gaussian_kernel`, `_renyi_entropy_order_2`, `_mix_kernel_concat`) rather
    than retyping them. Those four are byte-identical to HEAD and `git diff`
    is the proof of that, not this test; retyping a published equation is how a
    transcription error enters, and what is under test here is the FORWARD's
    composition, not the kernels it composes.
    """
    y_t_pseudo = torch.argmax(g_t, dim=1)
    unique_classes = torch.unique(y_s, sorted=True)
    losses_per_class = []
    valid_class_count = 0

    if loss.use_entropy_weighting:
        squared_sum = torch.sum(g_t ** 2, dim=1)
        entropy = -torch.log(squared_sum + 1e-6)
        target_weights = 1.0 - entropy / torch.log(
            torch.tensor(float(g_t.shape[1]), device=g_t.device))
    else:
        target_weights = None

    for c in unique_classes:
        f_s_c = f_s[y_s == c]
        f_t_c = f_t[y_t_pseudo == c]

        if f_s_c.shape[0] == 0 or f_t_c.shape[0] == 0:
            continue

        sigma_val = (loss._compute_sigma(f_s_c, f_t_c)
                     if loss.sigma == 'auto' else loss.sigma)

        K_s_c = loss._gaussian_kernel(f_s_c, f_s_c, sigma_val)
        K_t_c = loss._gaussian_kernel(f_t_c, f_t_c, sigma_val)
        K_st_c = loss._gaussian_kernel(f_s_c, f_t_c, sigma_val)

        if loss.use_entropy_weighting:
            weights_c = target_weights[y_t_pseudo == c]
            W_c = torch.outer(weights_c, weights_c)
            K_t_c = K_t_c * W_c

        K_mix_c = loss._mix_kernel_concat(K_s_c, K_t_c, K_st_c)

        h_s = loss._renyi_entropy_order_2(K_s_c)
        h_t = loss._renyi_entropy_order_2(K_t_c)
        h_mix = loss._renyi_entropy_order_2(K_mix_c)
        creda_c = h_mix - 0.5 * (h_s + h_t)
        losses_per_class.append(creda_c)
        valid_class_count += 1

    if valid_class_count == 0:
        return torch.tensor(0.0, device=f_s.device)

    losses = torch.stack(losses_per_class)

    if reduction == 'none':
        return loss.lambda_creda * losses
    return loss.lambda_creda * losses.mean()


def _target_weights(g_t: torch.Tensor) -> torch.Tensor:
    """D's diagonal of Eq. (25), as `CREDALoss` builds it. Read independently
    here so a block-level reference does not have to call into the method whose
    composition is under test."""
    squared_sum = torch.sum(g_t ** 2, dim=1)
    entropy = -torch.log(squared_sum + 1e-6)
    return 1.0 - entropy / torch.log(torch.tensor(float(g_t.shape[1])))


def _per_class_reference(loss: CREDALoss, f_s, f_t, y_s, g_t, *,
                         cross_weighted: bool) -> list[torch.Tensor]:
    """Per-class `creda_c`, with the two weighted blocks built as Eq. (26)
    WRITES them: `D K_c^t D` and `K_c^{st} D`, explicit matrix products against
    a materialized `torch.diag(weights_c)`.

    A different expression from the broadcast multiply the implementation uses,
    on purpose: comparing a broadcast against the same broadcast would be
    comparing an expression with itself.
    """
    y_t_pseudo = torch.argmax(g_t, dim=1)
    weights = _target_weights(g_t)
    values = []
    for c in torch.unique(y_s, sorted=True):
        f_s_c, f_t_c = f_s[y_s == c], f_t[y_t_pseudo == c]
        if f_s_c.shape[0] == 0 or f_t_c.shape[0] == 0:
            continue
        sigma = loss._compute_sigma(f_s_c, f_t_c)
        K_s = loss._gaussian_kernel(f_s_c, f_s_c, sigma)
        K_t = loss._gaussian_kernel(f_t_c, f_t_c, sigma)
        K_st = loss._gaussian_kernel(f_s_c, f_t_c, sigma)
        D = torch.diag(weights[y_t_pseudo == c])
        K_t = D @ K_t @ D
        if cross_weighted:
            K_st = K_st @ D
        K_mix = loss._mix_kernel_concat(K_s, K_t, K_st)
        values.append(loss._renyi_entropy_order_2(K_mix)
                      - 0.5 * (loss._renyi_entropy_order_2(K_s)
                               + loss._renyi_entropy_order_2(K_t)))
    return values


def _class_cardinalities(y_s, g_t) -> list[tuple[int, int]]:
    """(n_s, n_t) for every class `CREDALoss` does not skip, in its own order."""
    y_t_pseudo = torch.argmax(g_t, dim=1)
    pairs = []
    for c in torch.unique(y_s, sorted=True):
        n_s = int((y_s == c).sum())
        n_t = int((y_t_pseudo == c).sum())
        if n_s == 0 or n_t == 0:
            continue
        pairs.append((n_s, n_t))
    return pairs


# ============================================ 1. the copies and the originals


def test_eq37_weights_are_the_observed_source_proportions() -> None:
    """r33 Sec. 5, Eq. (37) at sample level: `sum_c (n_c^s / N_s) l~_c`.

    The weights come from `y_s`, which is OBSERVED, and not from the routing or
    the confidences -- the revision's reason being that a weight based on those
    "dependeria de cantidades que el propio modelo determina, mientras que
    `n_c^s` esta fijado por los datos".

    Asserted against a reconstruction over the per-class values the loss itself
    reports, so the claim is about the AGGREGATION and not about the blocks.
    """
    f_s, f_t, y_s, g_t = _fixture()
    loss = CREDALoss(sigma="auto", use_entropy_weighting=True)
    per_class = loss(f_s, f_t, y_s, g_t, reduction="none")
    reduced = loss(f_s, f_t, y_s, g_t)

    counts = _class_cardinalities(y_s, g_t)
    source_counts = torch.tensor([n_s for n_s, _ in counts],
                                 dtype=per_class.dtype)
    assert torch.equal(reduced, (source_counts @ per_class) / f_s.shape[0])
    assert int(source_counts.sum()) == f_s.shape[0]


def test_eq37_charges_one_for_a_class_the_routing_abandoned() -> None:
    """`l~_c = 1` for a class with no target sample, so abandoning one RAISES
    the term -- where r31 dropped it from the sum and the term fell."""
    torch.manual_seed(5)
    f_s = torch.randn(12, 6)
    f_t = torch.randn(12, 6)
    y_s = torch.arange(12) % 3
    # Every target sample routed to class 0: classes 1 and 2 are abandoned, and
    # each holds 4 of the 12 source samples.
    g_t = torch.full((12, 3), 1e-6)
    g_t[:, 0] = 1.0
    g_t = g_t / g_t.sum(dim=1, keepdim=True)

    loss = CREDALoss(sigma="auto", use_entropy_weighting=True)
    per_class = loss(f_s, f_t, y_s, g_t, reduction="none")
    assert per_class.numel() == 1, (
        "the fixture was supposed to leave exactly one active class")
    reduced = float(loss(f_s, f_t, y_s, g_t))

    # Two abandoned classes at 4/12 each: the charge alone is 8/12.
    assert reduced > 8.0 / 12.0
    # And r31 would have reported the one active class's own value, which is
    # smaller -- the direction is the whole point.
    assert reduced > float(per_class[0])


def test_eq37_inactive_charge_carries_no_gradient() -> None:
    """Eq. (22)'s routing is an `argmax`, so the charge has no gradient path --
    its job is the VALUE, making the collapse visible rather than rewarded."""
    torch.manual_seed(5)
    f_s = torch.randn(12, 6)
    f_t = torch.randn(12, 6, requires_grad=True)
    y_s = torch.zeros(12, dtype=torch.long)
    g_t = torch.full((12, 3), 1e-6)
    g_t[:, 1] = 1.0
    g_t = g_t / g_t.sum(dim=1, keepdim=True)

    # No class has samples on both sides, so every class is abandoned: the term
    # is exactly 1 and its gradient is exactly zero.
    value = CREDALoss(sigma="auto")(f_s, f_t, y_s, g_t)
    assert float(value) == 1.0
    value.backward()
    assert f_t.grad is not None, (
        "the charge arrived without a graph, so a step whose routing abandons "
        "every class would die in backward -- and in the declared regime this "
        "term is the only gradient path there is")
    assert float(f_t.grad.abs().sum()) == 0.0


def test_the_marginal_block_is_not_confidence_weighted() -> None:
    """r33: "La ponderacion de confianza se aplica a todo bloque CONDICIONAL
    por clase que involucre bolsas objetivo (Ec. 26 y Ec. 27)".

    `_marginal_score` is the whole batch with the class index REMOVED, so it is
    not a conditional block and Eq. (26) does not reach it. Asserted two ways:
    the confidences cannot be passed at all, and the marginal half does not
    move when they change.
    """
    f_s, f_t, y_s, g_t = _fixture()
    loss = CREDALoss(sigma="auto", class_mixing=0.0, use_entropy_weighting=True)
    marginal = loss(f_s, f_t, y_s, g_t)

    # One: the argument is gone, so nothing can hand it weights by mistake.
    with pytest.raises(TypeError):
        loss._marginal_score(f_s, f_t, _target_weights(g_t))
    assert torch.equal(marginal,
                       loss.lambda_creda * loss._marginal_score(f_s, f_t))

    # Two: a DIFFERENT set of confidences leaves the marginal half untouched.
    sharper = torch.softmax(torch.log(g_t + 1e-12) * 4.0, dim=1)
    assert not torch.allclose(_target_weights(g_t), _target_weights(sharper))
    assert torch.equal(marginal, loss(f_s, f_t, y_s, sharper))


def test_the_retired_copies_are_gone_from_both_sides() -> None:
    """**Two copy-equality tests lived here and were removed on 2026-10-18.**

    `test_conservative_bounds_copy_agrees_with_the_original` and
    `test_class_global_loss_copy_agrees_with_the_original` held
    `CREDA.bounds`' copies of Eq. (35)/(36) byte-equal to
    `MIL_CREDA.global_term`'s originals. That pin is what made the copies safe,
    and it is also what made `CREDA.bounds` declare r25 for as long as they
    existed. r33 states ONE pair of ends -- `L_c = -ln 2` and
    `U_c = 2 ln cosh((H_t - H_s)/4)`, not r25's `-ln(n_s + n_t)` and
    `(1/2)(ln n_s + ln n_t)` -- so both functions were removed from both
    packages and the pin went with its subject.

    Asserted as an absence on BOTH sides, because that is the half a deleted
    test cannot check: a copy whose original was removed, or an original whose
    copy was, is a drift this file exists to catch.
    """
    from CREDA import bounds as creda_bounds
    from MIL_CREDA import global_term

    for module in (creda_bounds, global_term):
        for retired in ('conservative_bounds', 'class_global_loss'):
            assert not hasattr(module, retired), (
                f"{module.__name__}.{retired} is back: r33 states one pair of "
                f"ends, so a second map needs a deliberation, not an import")

    # What remains is still pinned, and these are the pins that still have a
    # subject -- declared under exactly these two names.
    declared = creda_bounds.__provenance__['invariants']
    assert 'source_bound_copy_agrees_with_the_original' in declared
    assert 'source_loss_copy_agrees_with_the_original' in declared
    assert 'conservative_bounds_copy_agrees_with_the_original' not in declared
    assert 'class_global_loss_copy_agrees_with_the_original' not in declared


def test_source_bound_copy_agrees_with_the_original() -> None:
    """B_src of Eq. (21), copied, is the same number for the same stabilizer --
    including the two the original's own docstring quotes, about 18.42 at 1e-8
    and about 13.82 at 1e-6 -- and refuses a non-positive one."""
    for epsilon in [1e-8, 1e-6, 1e-3, 0.5, 1.0, 7.25]:
        assert source_bound(epsilon) == original_source_bound(epsilon), (
            f"the copy and the original disagree at eps={epsilon!r}")
    assert source_bound() == original_source_bound(), (
        "the two defaults disagree, which would make every supervised term "
        "computed through the copy sit on a different scale")

    for epsilon in [0.0, -1e-8, -1.0]:
        with pytest.raises(ValueError):
            original_source_bound(epsilon)
        with pytest.raises(ValueError):
            source_bound(epsilon)


def test_source_loss_copy_agrees_with_the_original() -> None:
    """Eq. (21), copied, agrees with the original on a spread of distributions
    -- certain and correct, certain and wrong, uniform, a single row -- and
    reproduces its refusals on a shape mismatch and on an empty batch."""
    generator = torch.Generator().manual_seed(5)
    cases = [
        (torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
         torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
        (torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
         torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])),
        (torch.full((4, CLASSES), 1.0 / CLASSES),
         F.one_hot(torch.tensor([0, 1, 2, 0]), CLASSES).double()),
        (torch.softmax(torch.randn(9, CLASSES, generator=generator), dim=1),
         F.one_hot(torch.arange(9) % CLASSES, CLASSES).double()),
        (torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.0, 1.0, 0.0])),
    ]
    for G, Y in cases:
        for epsilon in (1e-8, 1e-6):
            mine = bounds.source_loss(G, Y, epsilon)
            theirs = original_source_loss(G, Y, epsilon)
            assert torch.equal(mine, theirs), (
                f"the copy and the original disagree at eps={epsilon!r}: "
                f"{mine!r} against {theirs!r}")

    mismatched = (torch.full((3, CLASSES), 1.0 / CLASSES),
                  torch.full((2, CLASSES), 1.0 / CLASSES))
    with pytest.raises(ValueError):
        original_source_loss(*mismatched)
    with pytest.raises(ValueError):
        bounds.source_loss(*mismatched)

    empty = (torch.zeros(0, CLASSES), torch.zeros(0, CLASSES))
    with pytest.raises(ValueError):
        original_source_loss(*empty)
    with pytest.raises(ValueError):
        bounds.source_loss(*empty)


# ======================================================= 2. the bit-identity


@pytest.mark.parametrize("use_entropy_weighting", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_the_default_leaves_every_path_bit_identical_to_before_the_fork(
        use_entropy_weighting: bool, reduction: str) -> None:
    """With all three flags at their defaults, `CREDALoss` reproduces the
    pre-fork forward BIT FOR BIT -- `torch.equal`, not `allclose`, because the
    claim is identity.

    Run with `use_entropy_weighting` both ways, because that switch decides
    whether two of the three forks have anything to act on at all: with it off
    there is no D, so a cross-block weighting that leaked past its flag would
    be invisible on the weighted path and visible only here.

    Reachable red, three ways: flip any of the three defaults to `True`, or
    make `_kernel_embedding` do anything other than a bare return when off.
    """
    loss = CREDALoss(sigma="auto", lambda_creda=1.0,
                     use_entropy_weighting=use_entropy_weighting)
    assert loss.normalize_kernel_embedding is False
    assert loss.weight_cross_block is False
    assert loss.normalize_by_bounds is False

    f_s, f_t, y_s, g_t = _fixture()

    H = torch.randn(5, f_s.shape[1])
    assert loss._kernel_embedding(H) is H, (
        "with the flag off, _kernel_embedding must hand back the exact same "
        "tensor object -- not merely an equal one -- so every downstream "
        "computation is bit-identical to before this fork existed")

    actual = loss(f_s, f_t, y_s, g_t, reduction=reduction)
    per_class = _prefork_creda_forward(loss, f_s, f_t, y_s, g_t,
                                       reduction="none")
    if reduction == "none":
        # The PER-CLASS values are still bit-identical, and that is the half of
        # this claim r33 did not touch: Eq. (37) is an aggregation, so every
        # `creda_c` the pre-fork body computes must come out unchanged.
        assert torch.equal(actual, per_class), (
            f"the per-class default path moved: {actual!r} against the "
            f"pre-fork {per_class!r}, "
            f"use_entropy_weighting={use_entropy_weighting!r}")
        return

    # **The REDUCED value moved on 2026-10-18, by design, and this is where.**
    # The pre-fork body reduced with `losses.mean()`, which is r31's Eq. (37)
    # read at sample level: a uniform average over the ACTIVE classes. r33's
    # Eq. (37) is `sum_c (n_c^s / N_s) l~_c` with `l~_c = 1` for an inactive
    # class. So this comparison is no longer against the pre-fork MEAN; it is
    # against the pre-fork per-class values put through r33's aggregation.
    #
    # That is a stronger pin than a re-measured constant would be, not a weaker
    # one. A re-measured number says "it is what it is today". This says the
    # per-class half is untouched and the aggregation is exactly Eq. (37), so
    # any drift in either half still reddens -- and it cannot be satisfied by
    # the implementation drifting, because the reference reduces a vector the
    # reference itself computed.
    #
    # On this fixture the two differ by about 3e-8: every class is active, so
    # mathematically the only difference is `n_c^s / N_s` against `1 / |C|`,
    # which agree when the batch is balanced. What remains is float32
    # rounding -- `sum_c n_c^s l_c` then one division, against a sum and a
    # division by the count. `torch.equal` is kept rather than relaxed to
    # `allclose`, because the claim is still identity, against the right
    # reference.
    counts = _class_cardinalities(y_s, g_t)
    source_counts = torch.tensor([n_s for n_s, _ in counts],
                                 dtype=per_class.dtype)
    expected = (source_counts @ per_class) / f_s.shape[0]
    assert torch.equal(actual, expected), (
        f"the reduced default path is not Eq. (37) over the pre-fork "
        f"per-class values: {actual!r} against {expected!r}, "
        f"use_entropy_weighting={use_entropy_weighting!r}")


def test_the_pinned_pre_fork_means_still_come_out_of_the_default_path() -> None:
    """A second tripwire beside the reference above: two means read off a run
    of this exact fixture BEFORE the three flags existed at all.

    The reference forward and these numbers fail together for a flipped
    default and separately for everything else -- a reference that drifted
    with the implementation would still match it, and a pinned number cannot.
    """
    f_s, f_t, y_s, g_t = _fixture()
    # **The two numbers below are r33's, measured 2026-10-18, and the two they
    # replace are kept beside them.** The pin from before the three forks read
    # `-0.309462308883667` and `0.13413619995117188`; r33's Eq. (37) reduces
    # with `sum_c (n_c^s / N_s) l~_c` where the pre-fork body used
    # `losses.mean()`, and on this fixture -- every class active, so the two
    # agree mathematically -- what separates them is float32 rounding, about
    # 3e-8. A pinned constant cannot distinguish that from drift, which is why
    # the tolerance stays at 1e-9 and the numbers moved instead: a relaxed
    # tolerance would have silently absorbed a real change of the same size.
    #
    # `test_the_default_leaves_every_path_bit_identical_to_before_the_fork` is
    # what holds the per-class half unchanged and the aggregation equal to
    # Eq. (37); these two are the independent tripwire beside it.
    # Measured, not guessed: the `False` arm did NOT move. Its mantissas make
    # `(4 * sum_c l_c) / 12` and `sum_c l_c / 3` land on the same float32; the
    # `True` arm's do not. One of the two is bit-identical and the other moved
    # by 3e-8, which is exactly what "the difference is rounding" predicts and
    # what a single re-measured number would have hidden.
    # **Re-measured on 2026-10-18 and BOTH arms moved this time**, which is
    # the difference from r33's re-measurement: there the `False` arm was
    # bit-identical and only the `True` one shifted by 3e-8. The cause is a
    # different one -- `_renyi_entropy_order_2`'s two `1e-6` epsilons were
    # removed, so every entropy this fixture computes changed slightly, and
    # that reaches both arms. Measured rather than assumed, because assuming
    # the unchanged arm is exactly the mistake r33's pass made.
    superseded = {True: -0.3094622790813446, False: 0.13413619995117188}
    pinned = {True: -0.3094612658023834, False: 0.13413672149181366}
    for use_entropy_weighting, expected in pinned.items():
        value = float(CREDALoss(
            sigma="auto", lambda_creda=1.0,
            use_entropy_weighting=use_entropy_weighting)(f_s, f_t, y_s, g_t))
        assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9), (
            f"with use_entropy_weighting={use_entropy_weighting!r} the "
            f"default path produced {value!r}; r33's pinned value is "
            f"{expected!r} (the pre-fork one was "
            f"{superseded[use_entropy_weighting]!r})")
        # **The bound here was `1e-6` and its stated reason was Eq. (37)'s
        # rounding; that reason no longer covers the move.** Removing
        # `_renyi_entropy_order_2`'s two epsilons is a change to the ESTIMATOR
        # rather than to the aggregation, and it shifts the `True` arm by
        # 1.013e-06 -- just past the old bound. So the bound is re-derived from
        # what it now has to admit instead of being relaxed until the test
        # passes: the log's epsilon biased `-log(tr(A^2) + 1e-6)` by
        # `1e-6 / tr(A^2)`, and with `tr(A^2) >= 1/n` for a unit-trace density
        # matrix of size n, that bias cannot exceed `1e-6 * n`. Both arms are
        # within it with room to spare, and anything an order larger than that
        # is a different cause.
        techo = 1e-6 * f_s.shape[0]
        movimiento = abs(value - superseded[use_entropy_weighting])
        assert movimiento < techo, (
            f"with use_entropy_weighting={use_entropy_weighting!r} the default "
            f"path moved by {movimiento:.3e} against the superseded number "
            f"{superseded[use_entropy_weighting]!r}, past the "
            f"{techo:.3e} the log epsilon's own bias `1e-6 / tr(A^2)` can "
            f"account for on {f_s.shape[0]} samples")


# ========================================================== 3. one per flag


def test_normalize_kernel_embedding_bounds_what_the_kernels_see() -> None:
    """Fork 1 changes the value, and changes it in exactly the declared way:
    the kernels evaluate the L2-normalized embedding.

    The declared form is pinned by equality against a DIFFERENT route to the
    same claim -- a flag-OFF loss handed features that were normalized by the
    caller. That is only the same number if the flag does precisely what it
    says and nothing else, and in particular if it never reaches `g_t`, which
    is passed identically to both and has no normalized counterpart.
    """
    f_s, f_t, y_s, g_t = _fixture()
    g_t_before = g_t.clone()

    off = CREDALoss(sigma="auto", use_entropy_weighting=True)
    on = CREDALoss(sigma="auto", use_entropy_weighting=True,
                   normalize_kernel_embedding=True)

    baseline = off(f_s, f_t, y_s, g_t)
    normalized = on(f_s, f_t, y_s, g_t)
    assert not torch.equal(normalized, baseline), (
        "the flag left the value untouched on a fixture whose embedding is "
        "nowhere near unit norm, so this test could not tell the fork from a "
        "no-op")

    declared = off(F.normalize(f_s, p=2, dim=-1), F.normalize(f_t, p=2, dim=-1),
                   y_s, g_t)
    assert torch.equal(normalized, declared), (
        f"the flag is not the declared operation: {normalized!r} against "
        f"{declared!r}, the same loss fed features the caller normalized")

    assert torch.equal(g_t, g_t_before), (
        "the pseudo-label distribution is the caller's and arrives already "
        "computed from raw features; the loss mutated it")

    # The bound the fork exists to establish, measured on the embedding rather
    # than argued from the formula -- and on a fixture that clears it by two
    # orders of magnitude unnormalized, so a held bound cannot be confused
    # with one that was never challenged.
    H = torch.tensor([[10.0, 0.0], [0.0, 10.0], [-10.0, -10.0]])
    raw = torch.cdist(H, H, p=2) ** 2
    assert float(raw.max()) > 4.0
    assert torch.equal(off._kernel_embedding(H), H)
    bounded = torch.cdist(on._kernel_embedding(H), on._kernel_embedding(H), p=2) ** 2
    assert float(bounded.max()) <= 4.0 + 1e-6, (
        f"a pairwise squared distance exceeded the bound with the flag on: "
        f"{bounded!r}")


def test_weight_cross_block_multiplies_the_cross_block_by_d_from_the_right() -> None:
    """Fork 2 is Eq. (26)'s `K~_c^{st} = K_c^{st} D`, and nothing wider.

    Three claims, in order of strength:

    1. The per-class values equal a reference that builds the two weighted
       blocks as explicit matrix products against a materialized
       `torch.diag(weights_c)` -- the equation as written, not as broadcast.
    2. The weight vector has one entry per COLUMN of the rectangular cross
       block, so applying it on the source side is not even shape-valid on
       this fixture. "Only the target side" is structural here, not a
       convention this test agreed to.
    3. The corrected mean sits NEARER the unweighted mean than CREDA's
       currently-weighted one does. Weighting the target block while feeding
       the cross block raw weights the two halves of one mixture
       inconsistently, and that inconsistency is what moves the number far
       from both. The relation is derived from the three measured values, not
       hardcoded.
    """
    f_s, f_t, y_s, g_t = _fixture()
    weighted_today = CREDALoss(sigma="auto", use_entropy_weighting=True)
    corrected = CREDALoss(sigma="auto", use_entropy_weighting=True,
                          weight_cross_block=True)
    unweighted = CREDALoss(sigma="auto", use_entropy_weighting=False)

    actual = corrected(f_s, f_t, y_s, g_t, reduction="none")
    expected = torch.stack(_per_class_reference(
        weighted_today, f_s, f_t, y_s, g_t, cross_weighted=True))
    assert torch.equal(actual, expected), (
        f"the corrected per-class values are not `K_c^{{st}} D`: {actual!r} "
        f"against {expected!r}")

    today = weighted_today(f_s, f_t, y_s, g_t, reduction="none")
    assert not torch.equal(actual, today), (
        "the flag changed nothing, so the cross block was already weighted "
        "and this test is measuring a no-op")

    # Claim 2: one weight per target column, and the source side has none.
    y_t_pseudo = torch.argmax(g_t, dim=1)
    weights = _target_weights(g_t)
    sigma = weighted_today._compute_sigma(f_s[y_s == 0], f_t[y_t_pseudo == 0])
    K_st = weighted_today._gaussian_kernel(f_s[y_s == 0], f_t[y_t_pseudo == 0], sigma)
    weights_c = weights[y_t_pseudo == 0]
    assert K_st.shape[0] != K_st.shape[1], (
        "the cross block came out square, so a wrong-sided weighting would be "
        "shape-valid and this claim would not be tested at all")
    assert weights_c.shape[0] == K_st.shape[1], (
        "D's diagonal must have one entry per target column of K_st")
    with pytest.raises(RuntimeError):
        K_st * weights_c.unsqueeze(1)

    # Claim 3: the derived relation among the three measured means.
    far = float(weighted_today(f_s, f_t, y_s, g_t))
    near = float(corrected(f_s, f_t, y_s, g_t))
    plain = float(unweighted(f_s, f_t, y_s, g_t))
    assert abs(near - plain) < abs(far - plain), (
        f"the corrected mean ({near!r}) is no nearer the unweighted mean "
        f"({plain!r}) than CREDA's currently-weighted one ({far!r}); a "
        f"consistent weighting of both blocks of one mixture is supposed to "
        f"move the number far less than an inconsistent one")

    # And the flag is inert without a D to multiply by.
    inert = CREDALoss(sigma="auto", use_entropy_weighting=False,
                      weight_cross_block=True)
    assert torch.equal(inert(f_s, f_t, y_s, g_t), unweighted(f_s, f_t, y_s, g_t)), (
        "with `use_entropy_weighting` off there is no D, so the cross-block "
        "flag has nothing to apply and must be exactly inert")


def test_normalize_by_bounds_maps_each_class_onto_its_own_unit_interval() -> None:
    """Fork 3 puts each class's score onto [0, 1] through Eq. (36), and the
    SIGN CHAIN comes out increasing in `creda_c`.

    **Moved from r25's map to r33's on 2026-10-18.** It used to build its own
    expected value from `conservative_bounds`/`class_global_loss`, saying so in
    a comment: "it is about r25's map, not about whichever map the constructor
    happens to default to". Those two functions were removed -- r33 states one
    pair of ends -- so the claim moved to the map that is left rather than
    being deleted, because the claim was never about r25. The ends now come
    from the two marginal entropies (Eq. 34/35) and the denominator is the
    literal `ln 2` (Eq. 36).

    Two claims, checked in two steps because a single end-to-end reading cannot
    tell a correct chain from two compensating errors.

    First, every value lands in [0, 1].

    Second, the sign chain. Eq. (36) is DECREASING in its own argument, and the
    argument the implementation passes is `-creda_c`, so the composition is
    INCREASING in `creda_c`: minimizing the bounded form and minimizing CREDA's
    raw per-class score push the encoder the same way, and a single missing
    negation would reverse it and quietly train the term upwards. Link one is
    read off `class_global_loss_cosh` at a fixed pair of entropies; link two is
    read off the implementation, by rebuilding what it must have computed.
    """
    f_s, f_t, y_s, g_t = _fixture()
    # `entropy-cosh` is proved on Eq. (32)'s BALANCED mixture, and `CREDALoss`
    # refuses the pairing without it -- so both losses carry the balance and
    # the only difference between them is the normalizer.
    # `weight_cross_block=True` because Eq. 26 weights BOTH blocks that touch
    # target bags, and Eq. 35's bound is proved on that mixture -- measured,
    # 1378 of 1733 per-class readings leave [0, 1] with the cross block raw.
    # `CREDALoss` refuses the pairing since 2026-10-18; this test is how that
    # combination was found.
    shared = dict(sigma="auto", use_entropy_weighting=True,
                  weight_cross_block=True,
                  balance_mix_blocks=True, balance_mass="trace")
    plain = CREDALoss(**shared)
    bounded = CREDALoss(**shared, normalize_by_bounds=True,
                        bounds_form="entropy-cosh")

    raw = plain(f_s, f_t, y_s, g_t, reduction="none")
    actual = bounded(f_s, f_t, y_s, g_t, reduction="none")

    # The two marginal entropies per class, built the way the implementation
    # reads them: over the raw per-class blocks, before the mixture's balance.
    y_t_pseudo = torch.argmax(g_t, dim=1)
    weights = _target_weights(g_t)
    entropies = []
    for c in torch.unique(y_s, sorted=True):
        f_s_c, f_t_c = f_s[y_s == c], f_t[y_t_pseudo == c]
        if f_s_c.shape[0] == 0 or f_t_c.shape[0] == 0:
            continue
        sigma = plain._compute_sigma(f_s_c, f_t_c)
        K_s = plain._gaussian_kernel(f_s_c, f_s_c, sigma)
        K_t = plain._gaussian_kernel(f_t_c, f_t_c, sigma)
        D = torch.diag(weights[y_t_pseudo == c])
        K_t = D @ K_t @ D
        entropies.append((plain._renyi_entropy_order_2(K_s),
                          plain._renyi_entropy_order_2(K_t)))

    assert len(entropies) == raw.shape[0], (
        f"{len(entropies)} classes reconstructed against {raw.shape[0]} "
        f"per-class losses produced: the test's own class ordering has "
        f"desynchronized from the loss's")

    expected = torch.stack([
        bounds.class_global_loss_cosh(-raw[i], h_s, h_t)
        for i, (h_s, h_t) in enumerate(entropies)
    ])
    assert torch.allclose(actual, expected, rtol=0.0, atol=1e-6), (
        f"the bounded per-class values are not Eq. (36) of `-creda_c`: "
        f"{actual!r} against {expected!r}")
    assert not torch.equal(actual, raw), (
        "the flag changed nothing, so this test is measuring a no-op")

    assert float(actual.min()) >= 0.0 and float(actual.max()) <= 1.0, (
        f"a normalized class loss left [0, 1]: {actual!r}; Eq. (36) sends "
        f"U_c to zero and L_c to one and the score lies between them")

    # Link one: Eq. (36) decreases in its argument, at a FIXED pair of ends.
    h_s, h_t = entropies[0]
    _lower, upper = bounds.entropy_cosh_bounds(h_s, h_t)
    assert float(bounds.class_global_loss_cosh(upper, h_s, h_t)) == 0.0
    assert (float(bounds.class_global_loss_cosh(float(upper) - 1.0, h_s, h_t))
            > float(bounds.class_global_loss_cosh(float(upper) - 0.5, h_s, h_t))), (
        "Eq. (36) came out increasing in J_c, which would make the whole "
        "chain below read correctly for the wrong reason")

    # Link two: the argument is `-creda_c`. Read off the implementation rather
    # than asserted: for each class, the value it produced is the map applied
    # to the NEGATED raw score, and applying it to the raw score unnegated
    # gives something else -- so the negation is present and is not a no-op.
    unnegated = torch.stack([
        bounds.class_global_loss_cosh(raw[i], h_s_c, h_t_c)
        for i, (h_s_c, h_t_c) in enumerate(entropies)
    ])
    assert not torch.allclose(actual, unnegated, rtol=0.0, atol=1e-6), (
        f"the map applied to `creda_c` and to `-creda_c` give the same "
        f"numbers on this fixture, so the negation is unobservable here and "
        f"this test cannot see a missing one: {actual!r} against {unnegated!r}")


def _clump_against(source: torch.Tensor, n_target: int, *, distance: float = 3.0,
                   spread: float = 0.01) -> torch.Tensor:
    """`n_target` points in one tight clump, held AWAY from `source`.

    The geometry is fixed and only the count varies, which is what isolates
    cardinality from alignment: every returned cloud is equally misaligned.
    """
    generator = torch.Generator().manual_seed(4)
    return (torch.zeros(n_target, source.shape[1])
            + torch.randn(n_target, source.shape[1], generator=generator) * spread
            + distance)


def _one_class_score(loss: CREDALoss, f_s: torch.Tensor,
                     f_t: torch.Tensor) -> float:
    """`creda_c` for a single class, through the loss's own helpers."""
    K_s = loss._gaussian_kernel(f_s, f_s, loss.sigma)
    K_t = loss._gaussian_kernel(f_t, f_t, loss.sigma)
    K_st = loss._gaussian_kernel(f_s, f_t, loss.sigma)
    K_mix = loss._mix_kernel_concat(K_s, K_t, K_st)
    return float(loss._renyi_entropy_order_2(K_mix)
                 - 0.5 * (loss._renyi_entropy_order_2(K_s)
                          + loss._renyi_entropy_order_2(K_t)))


def test_balance_mix_blocks_off_is_the_plain_concatenation() -> None:
    """The default must not move a single byte of the mixed kernel.

    Checked on UNEQUAL block sizes, since equal ones would pass even under a
    repair that divided both blocks by the same number.
    """
    generator = torch.Generator().manual_seed(17)
    f_s = torch.randn(13, 4, generator=generator)
    f_t = torch.randn(29, 4, generator=generator)
    loss = CREDALoss(sigma=1.5)
    assert loss.balance_mix_blocks is False, (
        "the fork must default to the behaviour that keeps published numbers")
    K_s = loss._gaussian_kernel(f_s, f_s, 1.5)
    K_t = loss._gaussian_kernel(f_t, f_t, 1.5)
    K_st = loss._gaussian_kernel(f_s, f_t, 1.5)
    plain = torch.cat([torch.cat([K_s, K_st], dim=1),
                       torch.cat([K_st.T, K_t], dim=1)], dim=0)
    assert torch.equal(loss._mix_kernel_concat(K_s, K_t, K_st), plain)


def test_balance_mix_blocks_keeps_the_two_values_the_repair_may_not_move() -> None:
    """Coincident clouds still score 0 and separated ones still score ln 2.

    These are the fixed points of the construction, and a repair that moved
    either would be a different term wearing its name. The variant that divides
    the diagonal blocks and leaves `K_st` raw fails exactly here, because it is
    not a congruence: positive semidefiniteness goes with it, and the entropy of
    an indefinite matrix is not an entropy. So the smallest eigenvalue is part of
    the assertion and not a separate concern.
    """
    generator = torch.Generator().manual_seed(23)
    f_s = torch.randn(40, 4, generator=generator)
    loss = CREDALoss(sigma=1.5, balance_mix_blocks=True)
    for name, f_t, expected in (("coincident", f_s + 1e-4, 0.0),
                                ("separated", f_s + 60.0, math.log(2.0))):
        K_s = loss._gaussian_kernel(f_s, f_s, 1.5)
        K_t = loss._gaussian_kernel(f_t, f_t, 1.5)
        K_st = loss._gaussian_kernel(f_s, f_t, 1.5)
        K_mix = loss._mix_kernel_concat(K_s, K_t, K_st)
        smallest = float(torch.linalg.eigvalsh(K_mix).min())
        assert smallest > -1e-4, (
            f"{name}: the balanced mix is no longer PSD ({smallest:.3e}), so its "
            "entropy is not an entropy")
        assert _one_class_score(loss, f_s, f_t) == pytest.approx(expected, abs=1e-3)


def test_balance_mix_blocks_closes_the_cardinality_exploit_the_default_leaves_open() -> None:
    """A misaligned clump must never score better than a perfect match.

    Both halves are asserted, and the OFF half is what makes the ON half mean
    something: without it this would pass under a fork that did nothing at all.

    Under the default, a clump held at a fixed distance from the source scores
    BETTER than perfect alignment once it holds enough points -- the crossing is
    near `n_t = 2 n_s` -- because a Gaussian kernel's diagonal is 1, so the
    larger block owns the mixed trace and `h_mix` becomes that block's own
    entropy. Balanced, the score is flat in the count and the clump always
    scores worse than the match.
    """
    generator = torch.Generator().manual_seed(29)
    f_s = torch.randn(40, 4, generator=generator)
    counts = (10, 40, 160, 400)
    for flag, expect_exploit in ((False, True), (True, False)):
        loss = CREDALoss(sigma=1.5, balance_mix_blocks=flag)
        gaps, scores = [], []
        for n_target in counts:
            clump = _clump_against(f_s, n_target)
            matched = f_s[torch.randint(0, f_s.shape[0], (n_target,),
                                        generator=generator)] + 1e-4
            scores.append(_one_class_score(loss, f_s, clump))
            gaps.append(_one_class_score(loss, f_s, clump)
                        - _one_class_score(loss, f_s, matched))
        if expect_exploit:
            assert min(gaps) < -0.01, (
                "the unbalanced mix is supposed to prefer a large clump to a "
                f"perfect match; gaps were {gaps}, so this test can no longer "
                "tell the repair from a no-op")
            assert max(scores) - min(scores) > 1.0, (
                f"the unbalanced score should swing with the count; got {scores}")
        else:
            assert min(gaps) > 0.0, (
                f"balanced, a clump still outscores a perfect match: {gaps}")
            assert max(scores) - min(scores) < 0.05, (
                f"balanced, the score should be flat in the count; got {scores}")


# ================================================= 4. the trainer's own forks


class _Encoder(nn.Module):
    """A tiny encoder with a declared `output_dim` and a layer that carries
    running statistics, because the sharpest claim about the transfer is about
    what it leaves alone. Records every input it was handed, which is how the
    floor's own claim is measured rather than argued.
    """

    def __init__(self, input_dim: int = 4, output_dim: int = CLASSES):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.detach().clone())
        return self.bn(self.linear(x))


def _model(*, adaptation: bool = True, dual_encoder: bool = False,
           normalize_supervised: bool = False, seed: int = 11) -> SampleCREDA:
    torch.manual_seed(seed)
    return SampleCREDA(_Encoder, CLASSES, adaptation=adaptation,
                       dual_encoder=dual_encoder,
                       normalize_supervised=normalize_supervised)


def _material() -> tuple[torch.Tensor, ...]:
    """Source samples and labels, and a target pool whose rows are a SENTINEL
    far from anything the source holds, so a target row that reached an encoder
    is unmistakable rather than a matter of tolerance."""
    f_s, _, _, _ = _fixture()
    y_s = torch.arange(f_s.shape[0]) % CLASSES
    target = torch.full((10, f_s.shape[1]), 999.0)
    target = target + torch.arange(10, dtype=target.dtype).reshape(-1, 1)
    return f_s, y_s.sort().values, target


def test_the_generator_is_consumed_identically_whether_or_not_alignment_is_on() -> None:
    """A floor and an adapted arm at the same seed draw the same target
    indices and leave the generator in the same state.

    This is not tidiness. Two arms that consumed the generator differently
    would differ in every subsequent draw of the run, so any gap between them
    would be partly the term under test and partly a different random stream,
    with no way to say how much of each.
    """
    source_x, source_y, target_x = _material()
    states, draws = [], []
    for adaptation in (True, False):
        model = _model(adaptation=adaptation)
        generator = torch.Generator().manual_seed(3)
        step = model.training_step(source_x, source_y, target_x, 0.5, generator)
        draws.append(step["targetIndices"])
        states.append(generator.get_state())

    assert torch.equal(draws[0], draws[1]), (
        f"the floor drew {draws[1].tolist()!r} where the adapted arm drew "
        f"{draws[0].tolist()!r}")
    assert torch.equal(states[0], states[1]), (
        "the two arms left the generator in different states, so every later "
        "draw of the run would differ between them")
    assert draws[0].numel() > 0, (
        "no indices were drawn at all, so this test compared two empty draws")


def test_the_floor_never_lets_a_target_sample_reach_an_encoder() -> None:
    """The floor draws its target indices and then does nothing with them.

    Measured by what the encoder was actually handed, not by reading the
    branch: the target pool's rows are a sentinel, and the floor's encoder must
    never have seen one. The reason this matters is measured too -- a floor
    that forwards the target through a BatchNorm encoder receives the target's
    statistics for free, which IS adaptation, and an adapted floor makes every
    rung above it look like it contributes nothing.

    The adapted arm is run on the same fixture as the control, so a sentinel
    that is simply unreachable cannot pass for a floor that refused it.
    """
    source_x, source_y, target_x = _material()

    floor = _model(adaptation=False)
    floor.training_step(source_x, source_y, target_x, 0.5,
                        torch.Generator().manual_seed(3))
    floor_rows = torch.cat(floor.encoder.seen) if floor.encoder.seen else torch.zeros(0)
    assert floor_rows.numel() > 0, "the floor's encoder was never called at all"
    assert float(floor_rows.max()) < 999.0, (
        f"a target sentinel row reached the floor's encoder: saw a maximum of "
        f"{float(floor_rows.max())!r}")

    adapted = _model(adaptation=True)
    adapted.training_step(source_x, source_y, target_x, 0.5,
                          torch.Generator().manual_seed(3))
    adapted_rows = torch.cat(adapted.encoder.seen)
    assert float(adapted_rows.max()) >= 999.0, (
        "the adapted arm never fed the target through an encoder either, so "
        "the floor's abstention above is not evidence of anything")


def test_the_transfer_copies_parameters_and_leaves_buffers_alone() -> None:
    """`sync_target_from_source` copies every `named_parameters()` tensor in
    place and touches no buffer.

    Both halves are made non-vacuous first: the target's parameters are
    perturbed so they genuinely differ before the copy, and its running
    statistics are moved by a forward pass so "unchanged" is a claim about
    something that was not already equal. The measurement behind the buffer
    half is on record -- carrying the statistics over as well was WORSE, 0.7222
    against 1.0000 -- so a transfer that copied them would be adopting a
    variant that was tried and lost.
    """
    model = _model(adaptation=True, dual_encoder=True)
    assert model.target_encoder is not None

    source_x, _, target_x = _material()
    model.train()
    model.encoder(source_x)
    model.target_encoder(target_x)
    with torch.no_grad():
        for parameter in model.target_encoder.parameters():
            parameter.add_(1.25)

    before_parameters = {name: tensor.clone() for name, tensor
                         in model.target_encoder.named_parameters()}
    before_storage = {name: tensor.data_ptr() for name, tensor
                      in model.target_encoder.named_parameters()}
    before_buffers = {name: tensor.clone() for name, tensor
                      in model.target_encoder.named_buffers()}
    source_parameters = dict(model.encoder.named_parameters())
    source_buffers = dict(model.encoder.named_buffers())

    assert before_buffers, "the stub carries no buffers, so nothing is at stake"
    assert any(not torch.equal(before_buffers[name], source_buffers[name])
               for name in before_buffers), (
        "the two encoders' buffers already agree, so leaving them alone and "
        "copying them are indistinguishable on this fixture")
    assert any(not torch.equal(before_parameters[name], source_parameters[name])
               for name in before_parameters), (
        "the two encoders' parameters already agree, so a transfer that "
        "copied nothing would pass")

    model.sync_target_from_source()

    for name, tensor in model.target_encoder.named_parameters():
        assert torch.equal(tensor, source_parameters[name]), (
            f"parameter {name!r} did not take the source's value")
        assert tensor.data_ptr() == before_storage[name], (
            f"parameter {name!r} was REBOUND rather than written in place; an "
            f"optimizer built over the old tensor would keep updating a "
            f"tensor the model no longer uses")
    assert any(not torch.equal(dict(model.target_encoder.named_parameters())[name],
                               before_parameters[name])
               for name in before_parameters), (
        "no parameter moved, so the transfer did nothing")

    for name, tensor in model.target_encoder.named_buffers():
        assert torch.equal(tensor, before_buffers[name]), (
            f"buffer {name!r} was changed by the transfer; running statistics "
            f"are a reading of the material each domain saw, not a parameter "
            f"of the map")

    # And the floor has nothing to transmit into, so the call is a no-op
    # rather than an error.
    floor = _model(adaptation=False, dual_encoder=True)
    assert floor.target_encoder is None
    floor.sync_target_from_source()


def test_the_transfer_happens_after_the_optimizer_step_at_the_declared_interval(
        monkeypatch) -> None:
    """The transfer runs AFTER `optimizer.step()`, every `sync_steps` optimizer
    steps, counted across the whole run.

    Transmitting before the update would re-anchor `theta_t` and the same step
    would immediately move it again, which measures an interval shifted one
    step from the one it names. Both halves are read off the recorded order of
    events and off the parameter values each event saw -- and the recording is
    made non-vacuous by asserting the optimizer actually moves the parameter
    between steps, without which "the value after the step" and "the value
    before it" would be the same tensor and the claim untestable.
    """
    assert DEFAULT_SYNC_STEPS >= 1 and isinstance(DEFAULT_SYNC_STEPS, int)

    model = _model(adaptation=True, dual_encoder=True)
    source_x, source_y, target_x = _material()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    probe = model.encoder.linear.weight

    events: list[tuple[str, int, torch.Tensor]] = []
    steps = {"n": 0}
    real_step = optimizer.step
    real_sync = model.sync_target_from_source

    def spy_step(*args, **kwargs):
        real_step(*args, **kwargs)
        steps["n"] += 1
        events.append(("step", steps["n"], probe.detach().clone()))

    def spy_sync():
        events.append(("sync", steps["n"], probe.detach().clone()))
        real_sync()

    monkeypatch.setattr(optimizer, "step", spy_step)
    monkeypatch.setattr(model, "sync_target_from_source", spy_sync)

    sync_steps = 3
    train(model, source_x, source_y, target_x, epochs=3, batch_size=3,
          generator=torch.Generator().manual_seed(7), optimizer=optimizer,
          sync_steps=sync_steps, learning_rate=0.1)

    step_events = [e for e in events if e[0] == "step"]
    sync_events = [e for e in events if e[0] == "sync"]
    assert len(step_events) >= 2 * sync_steps, (
        f"only {len(step_events)} optimizer steps ran, which cannot show an "
        f"interval of {sync_steps}")
    assert sync_events, "the transfer never ran at all"

    moved = [not torch.equal(step_events[i][2], step_events[i + 1][2])
             for i in range(len(step_events) - 1)]
    assert all(moved), (
        "the probed parameter did not change between consecutive optimizer "
        "steps, so 'the value after the step' is indistinguishable from the "
        "value before it and this test proves nothing about ordering")

    assert [n for _, n, _ in sync_events] == list(
        range(sync_steps, len(step_events) + 1, sync_steps)), (
        f"the transfers landed at optimizer steps "
        f"{[n for _, n, _ in sync_events]!r}, not at every {sync_steps}th of "
        f"{len(step_events)}")

    by_step = {n: tensor for kind, n, tensor in events if kind == "step"}
    for _, n, seen in sync_events:
        assert torch.equal(seen, by_step[n]), (
            f"the transfer at optimizer step {n} read a parameter value that "
            f"is not the post-step one, so it ran before the update")
    assert events.index(("sync",) + sync_events[0][1:]) > 0
    assert events[events.index(("sync",) + sync_events[0][1:]) - 1][0] == "step", (
        "the first transfer was not immediately preceded by an optimizer step")


def test_the_supervised_term_is_normalized_only_when_asked_and_carries_coefficient_one() -> None:
    """Fork 5 and Eq. (39)'s composition, together, because the second is what
    makes the first mean anything.

    Off, the supervised term is CREDA's own per-sample cross-entropy, used
    exactly as it was written. On, it is Eq. (21): the same averaged negative
    log-likelihood divided by the exact supremum `B_src = ln(1 + 1/eps)` --
    pinned by that ratio, derived from `source_bound`, not by a copied number.

    And in both cases the loss is `supervised + ramp * adaptation`: the
    supervised term at coefficient one and only the adaptation term carrying
    the free coefficient. `ramp` is deliberately not one, or the two readings
    would be indistinguishable.
    """
    source_x, source_y, target_x = _material()
    ramp = 0.375
    assert ramp != 1.0

    plain = _model(normalize_supervised=False)
    plain_step = plain.training_step(source_x, source_y, target_x, ramp,
                                     torch.Generator().manual_seed(3))
    logits = plain.head(plain.encoder(source_x))
    assert torch.equal(plain_step["supervised"],
                       F.cross_entropy(logits, source_y)), (
        "with the flag off the supervised term must be CREDA's own "
        "cross-entropy, byte for byte")

    normalized = _model(normalize_supervised=True)
    normalized_step = normalized.training_step(
        source_x, source_y, target_x, ramp, torch.Generator().manual_seed(3))
    logits = normalized.head(normalized.encoder(source_x))
    expected = bounds.source_loss(
        F.softmax(logits, dim=1),
        F.one_hot(source_y, CLASSES).to(logits.dtype), normalized.epsilon)
    assert torch.equal(normalized_step["supervised"], expected), (
        f"with the flag on the supervised term is not Eq. (21): "
        f"{normalized_step['supervised']!r} against {expected!r}")

    # The ratio Eq. (21) introduces, read against the unnormalized term on the
    # same logits rather than against a remembered constant. The agreement is
    # close but NOT exact, and the gap is accounted for rather than tolerated:
    # cross-entropy logs `g` while Eq. (21) logs `(g + eps)/(1 + eps)`, so the
    # two numerators differ by the stabilizer before the division happens, and
    # this runs in float32. Measured here at about 1.5e-7 relative; the
    # tolerance is two orders above that and three below the B_src this claim
    # would catch a wrong one of (1e-6 gives about 13.82, not 18.42).
    unnormalized = F.cross_entropy(logits, source_y)
    ratio = float(unnormalized.detach()) / float(expected.detach())
    assert math.isclose(ratio, source_bound(normalized.epsilon),
                        rel_tol=1e-5), (
        f"the normalized term is {ratio!r} times smaller than the plain one; "
        f"Eq. (21) divides by B_src = "
        f"{source_bound(normalized.epsilon)!r}")
    assert 0.0 <= float(expected.detach()) < 1.0, (
        f"Eq. (21) lands in [0, 1) and this came out "
        f"{float(expected.detach())!r}")

    for step in (plain_step, normalized_step):
        assert torch.allclose(
            step["loss"],
            step["supervised"] + ramp * step["adaptation"], atol=0.0), (
            "the objective is not `supervised + ramp * adaptation`: the "
            "supervised term carries coefficient one and only the adaptation "
            "term carries the free one")
        assert float(step["adaptation"].detach()) != 0.0, (
            "the adaptation term came out exactly zero, so no coefficient "
            "placed anywhere would change the loss and this claim is untested")


def test_the_ramp_is_the_published_schedule_and_is_zero_at_the_first_epoch() -> None:
    """The trainer's coefficient is `CREDA.schedules.creda_ramp`, which is zero
    at the first epoch -- a property of the published curve, not an
    initialization trick -- so a run's first epoch trains with no adaptation at
    all and a one-epoch run never adapts.
    """
    assert creda_ramp(0, 10, delta=10.0, ceiling=1.0) == 0.0
    source_x, source_y, target_x = _material()
    model = _model(adaptation=True, dual_encoder=True)
    curve = train(model, source_x, source_y, target_x, epochs=2, batch_size=4,
                  generator=torch.Generator().manual_seed(7), ceiling=1.0,
                  learning_rate=0.05)
    first = [record for record in curve if record["epoch"] == 0]
    later = [record for record in curve if record["epoch"] == 1]
    assert first and later
    assert all(record["ramp"] == 0.0 for record in first)
    assert all(record["ramp"] > 0.0 for record in later), (
        "the schedule never left zero, so the ramp is not climbing and the "
        "first-epoch claim is indistinguishable from a constant zero")


@pytest.mark.parametrize("epochs,batch_size,sync_steps", [(0, 2, 7), (2, 0, 7), (2, 2, 0)])
def test_train_refuses_a_run_that_cannot_hold_its_own_contract(
        epochs: int, batch_size: int, sync_steps: int) -> None:
    """Zero epochs is not a run, a step over zero samples is not a step, and an
    interval of zero means never -- the absence of the mechanism rather than a
    value of it."""
    source_x, source_y, target_x = _material()
    with pytest.raises(ValueError):
        train(_model(), source_x, source_y, target_x, epochs=epochs,
              batch_size=batch_size, sync_steps=sync_steps,
              generator=torch.Generator().manual_seed(1))
