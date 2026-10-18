"""Two forks added to `CREDALoss`: the bandwidth's scope, and the class mixture.

Both default to the faithful value, so the first test here is the one that
matters: **with every flag at its default, the loss is bit-identical to what it
was before these forks existed.** If it ever fails, nothing else in this file
is worth reading.

`sigma_from` names which features the `auto` bandwidth's median reads.
`class-pooled` is today's behaviour: per class, over that class's source AND
target. The target half is exactly what the optimizer moves, so the bandwidth
is a function of the thing being aligned — measured over 800 steps of the free
two-phase regime, the pooled median drifts by 2.65x to 3.36x while the
source-only median drifts 1.03x to 1.06x. Which scope ends up higher depends on
the step read, so nothing here prefers one: the scope is a caller's choice.

`class_mixing` is the convex family `(1-a)*marginal + a*per_class`. `a=1` is the
declared term and the default; `a=0` is the marginal block alone. Both ends lie
in [0,1] under `normalize_by_bounds`, so the mixture inherits that bound with
nothing to prove, and `test_the_bounded_mixture_stays_in_the_unit_interval`
checks that rather than assuming it.
"""

from __future__ import annotations

import pytest
import torch

from CREDA.models import SIGMA_SCOPES, CREDALoss

CLASSES = 5
DIM = 8
N = 60


def _batch(seed: int = 0):
    torch.manual_seed(seed)
    f_s = torch.randn(N, DIM) * 3.0
    f_t = torch.randn(N, DIM) * 3.0 + 1.0
    y_s = torch.arange(N) % CLASSES
    g_t = torch.softmax(torch.randn(N, CLASSES) * 2.0, dim=1)
    return f_s, f_t, y_s, g_t


def _one_class_batch(seed: int = 0):
    """Every source label and every pseudo-label on one class.

    In this configuration the per-class loop has exactly one block over the
    whole batch, which is the only configuration where it must agree with
    `_marginal_score` exactly.
    """
    torch.manual_seed(seed)
    f_s = torch.randn(N, DIM) * 3.0
    f_t = torch.randn(N, DIM) * 3.0 + 1.0
    y_s = torch.zeros(N, dtype=torch.long)
    logits = torch.full((N, CLASSES), -20.0)
    logits[:, 0] = 20.0
    g_t = torch.softmax(logits, dim=1)
    assert torch.all(g_t.argmax(1) == 0), "the fixture does not route everything to class 0"
    return f_s, f_t, y_s, g_t


# **The two `normalize_by_bounds=True` rows gained keywords on 2026-10-18.**
# `bounds_form` now has one legal value, `entropy-cosh`, whose bound is proved
# on Eq. (32)'s balanced mixture built from Eq. (26)'s TWO weighted blocks --
# so turning the normalizer on requires `balance_mix_blocks=True`,
# `balance_mass='trace'` and, while `use_entropy_weighting` is on,
# `weight_cross_block=True`. r25's `conservative` map assumed nothing about
# the mixture and needed none of them; it was removed with the pair that
# computed it. These rows are about the sigma scope and the mixing default, so
# naming the admissible pairing is not a workaround, it is what the pairing is.
_BOUNDED = {"normalize_by_bounds": True, "balance_mix_blocks": True,
            "balance_mass": "trace", "weight_cross_block": True}
FLAG_SETS = [
    {},
    {"use_entropy_weighting": False},
    dict(_BOUNDED),
    dict(_BOUNDED, weight_cross_block=True),
    {"normalize_kernel_embedding": True},
]


def test_the_default_is_bit_identical_to_the_explicit_faithful_values():
    args = _batch()
    for flags in FLAG_SETS:
        plain = CREDALoss(sigma="auto", **flags)(*args)
        spelled = CREDALoss(sigma="auto", sigma_from="class-pooled",
                            class_mixing=1.0, **flags)(*args)
        assert torch.equal(plain, spelled), (
            f"the default moved once the forks existed, with {flags}")


def test_the_median_over_one_set_is_compute_sigma_s_own_heuristic():
    """`_median_pairwise` must be the same heuristic, not a second one."""
    loss = CREDALoss(sigma="auto")
    torch.manual_seed(1)
    x, y = torch.randn(13, DIM) * 2.0, torch.randn(17, DIM) * 2.0 - 1.0
    assert torch.equal(loss._median_pairwise(torch.cat([x, y], dim=0)),
                       loss._compute_sigma(x, y))


def test_each_sigma_scope_reads_a_different_bandwidth():
    args = _batch()
    values = {scope: float(CREDALoss(sigma="auto", sigma_from=scope)(*args))
              for scope in SIGMA_SCOPES}
    assert len(set(values.values())) == len(SIGMA_SCOPES), (
        f"two scopes produced the same loss, so one of them is not wired: {values}")


def test_a_fixed_sigma_ignores_every_scope():
    """`sigma_from` is about the `auto` heuristic and must not touch a given sigma."""
    args = _batch()
    values = {scope: CREDALoss(sigma=4.0, sigma_from=scope)(*args)
              for scope in SIGMA_SCOPES}
    first = values[SIGMA_SCOPES[0]]
    for scope, value in values.items():
        assert torch.equal(value, first), f"a fixed sigma changed under {scope}"


def test_class_mixing_one_is_the_declared_per_class_term():
    args = _batch()
    assert torch.equal(CREDALoss(sigma="auto", class_mixing=1.0)(*args),
                       CREDALoss(sigma="auto")(*args))


def test_class_mixing_zero_is_the_marginal_block_alone():
    f_s, f_t, y_s, g_t = _batch()
    loss = CREDALoss(sigma="auto", class_mixing=0.0)
    mixed = loss(f_s, f_t, y_s, g_t)
    squared_sum = torch.sum(g_t ** 2, dim=1)
    entropy = -torch.log(squared_sum + 1e-6)
    weights = 1.0 - entropy / torch.log(torch.tensor(float(CLASSES)))
    # `_marginal_score` stopped taking the confidences on 2026-10-18: r33
    # scopes Eq. (26)'s `D` to the class-conditional blocks, and this block is
    # the whole batch with the class index removed. `weights` is still built
    # above because the assertion below is that NOTHING here reads it -- the
    # marginal half cannot be reproduced by a call that passes them.
    assert torch.allclose(mixed, loss._marginal_score(f_s, f_t), atol=0, rtol=1e-6)
    with pytest.raises(TypeError):
        loss._marginal_score(f_s, f_t, weights)


def test_the_mixture_is_convex_between_its_two_ends():
    args = _batch()
    ends = [float(CREDALoss(sigma="auto", class_mixing=a)(*args)) for a in (0.0, 1.0)]
    half = float(CREDALoss(sigma="auto", class_mixing=0.5)(*args))
    assert half == pytest.approx(0.5 * ends[0] + 0.5 * ends[1], rel=1e-6)
    assert ends[0] != ends[1], "the two ends coincide, so convexity proves nothing here"


def test_the_marginal_block_equals_the_loop_when_one_class_holds_everything():
    """The copy is pinned to the original, the way `bounds.py`'s copies are.

    **`use_entropy_weighting=False` on both sides since 2026-10-18.** The two
    halves are the same computation only while they see the same blocks, and
    r33 scopes Eq. (26)'s `D` to the class-conditional ones: with the weighting
    on, the loop's single class is weighted and the marginal block is not, so
    they are two different matrices and the equality is false by construction
    rather than by a defect. With no `D` at all there is nothing to scope and
    the claim is the one this test was written for.
    """
    args = _one_class_batch()
    per_class = CREDALoss(sigma="auto", class_mixing=1.0,
                          use_entropy_weighting=False)(*args)
    marginal = CREDALoss(sigma="auto", class_mixing=0.0,
                         use_entropy_weighting=False)(*args)
    assert torch.allclose(per_class, marginal, atol=0, rtol=1e-6), (
        "with one class holding the whole batch the loop and the marginal "
        "block are the same computation, and they disagreed")


def test_mixing_refuses_a_per_class_reduction():
    args = _batch()
    with pytest.raises(ValueError, match="no per-class decomposition"):
        CREDALoss(sigma="auto", class_mixing=0.5)(*args, reduction="none")
    # And the faithful default still serves that reduction.
    out = CREDALoss(sigma="auto")(*args, reduction="none")
    assert out.ndim == 1 and out.numel() == CLASSES


def test_mixing_with_no_valid_class_returns_the_marginal_half():
    """No class valid leaves the per-class half undefined and the marginal half real."""
    torch.manual_seed(3)
    f_s, f_t = torch.randn(20, DIM), torch.randn(20, DIM) + 1.0
    y_s = torch.zeros(20, dtype=torch.long)          # source only in class 0
    logits = torch.full((20, CLASSES), -20.0)
    logits[:, 1] = 20.0                              # target only in class 1
    g_t = torch.softmax(logits, dim=1)
    # **`== 0.0` became `== 1.0` on 2026-10-18.** r33's Eq. (37) charges
    # `l~_c = 1` for every class the routing abandoned, and with none active
    # the weights still sum to one, so the per-class half is exactly 1 where
    # r31 answered 0 -- which told a collapsed encoder it had converged.
    assert float(CREDALoss(sigma="auto")(f_s, f_t, y_s, g_t)) == 1.0
    mixed = CREDALoss(sigma="auto", class_mixing=0.25)(f_s, f_t, y_s, g_t)
    per_class_alone = float(CREDALoss(sigma="auto")(f_s, f_t, y_s, g_t))
    assert float(mixed) != per_class_alone, (
        "with no overlapping class the marginal half was dropped silently")


def test_the_bounded_mixture_stays_in_the_unit_interval():
    args = _batch()
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = float(CREDALoss(sigma="auto", class_mixing=a, **_BOUNDED)(*args))
        assert 0.0 <= v <= 1.0, f"class_mixing={a} left [0, 1] at {v}"


@pytest.mark.parametrize("scope", ["pooled", "source", "", "class_pooled"])
def test_an_unknown_sigma_scope_is_refused(scope):
    with pytest.raises(ValueError, match="sigma_from must be one of"):
        CREDALoss(sigma_from=scope)


@pytest.mark.parametrize("a", [-0.1, 1.1, 2.0])
def test_a_mixing_outside_the_unit_interval_is_refused(a):
    with pytest.raises(ValueError, match="convex coefficient"):
        CREDALoss(class_mixing=a)


def test_the_declared_scope_makes_the_discrepancy_monotone_in_the_misalignment():
    """`class-pooled` is non-monotone; `batch-pooled`, which r25 declares, is not.

    Displacing one class's target features ought to make that class's
    discrepancy grow and then saturate. Under the per-class median it grows and
    then FALLS, because the median is taken over that class's source and target
    together, so the bandwidth grows with the displacement and the kernel's own
    unit of measurement moves with the thing it is measuring.
    """
    torch.manual_seed(0)
    n = 200
    f_s = torch.randn(n, DIM) * 3.0
    y_s = torch.arange(n) % CLASSES
    f_t_base = f_s + torch.randn(n, DIM) * 0.3
    g_t = torch.zeros(n, CLASSES)
    g_t[torch.arange(n), y_s] = 1.0
    g_t = g_t * 0.98 + 0.02 / CLASSES

    def sweep(scope):
        out = []
        for displacement in (0.0, 3.0, 10.0, 30.0, 100.0):
            f_t = f_t_base.clone()
            f_t[y_s == 0] += displacement
            per_class = CREDALoss(sigma="auto", sigma_from=scope,
                                  use_entropy_weighting=False)(
                f_s, f_t, y_s, g_t, reduction="none")
            out.append(float(per_class[0]))
        return out

    pooled_per_class = sweep("class-pooled")
    declared = sweep("batch-pooled")

    def worst_drop(values):
        """How far the reading FALLS after its peak. Zero is monotone."""
        peak = max(values)
        return peak - min(values[values.index(peak):])

    # Not a strict monotonicity check: every scope plateaus at ln 2, where float
    # noise moves the last digits, so `b >= a` fails on arithmetic rather than on
    # behaviour. What separates them is how far the reading falls.
    assert worst_drop(pooled_per_class) > 0.05, (
        "the per-class median is expected to FALL materially after its peak -- "
        f"a worse-aligned class reading less discrepant -- and it did not: {pooled_per_class}")
    assert worst_drop(declared) < 1e-4, (
        f"the declared scope should not fall after its peak: {declared}")
    assert declared[-1] == pytest.approx(0.69315, abs=1e-4), (
        "two separated clouds of equal size saturate at ln 2, and the declared "
        f"scope reached {declared[-1]}")
    # Four orders of magnitude apart, so the thresholds are not a squeeze.
    assert worst_drop(pooled_per_class) / max(worst_drop(declared), 1e-12) > 1e3


def test_a_per_class_bandwidth_measures_the_classes_at_different_resolutions():
    """The second defect, separate from the monotonicity one.

    A per-class median gives each class its own bandwidth, so the `creda_c`
    values that Eq. (37) then averages were measured at different resolutions.
    With one class displaced, `class-pooled` hands that class nearly double the
    bandwidth of the others -- the misaligned class is read at half the
    resolution, and then its score is averaged in as if comparable. The two
    `batch-*` scopes give every class the same bandwidth by construction, which
    is what r25 line 732 declares with "el mismo kernel de instancia".
    """
    torch.manual_seed(0)
    n = 200
    f_s = torch.randn(n, DIM) * 3.0
    y_s = torch.arange(n) % CLASSES
    f_t = f_s + torch.randn(n, DIM) * 0.3
    f_t[y_s == 0] += 10.0
    g_t = torch.zeros(n, CLASSES)
    g_t[torch.arange(n), y_s] = 1.0
    g_t = g_t * 0.98 + 0.02 / CLASSES
    y_t = g_t.argmax(1)

    def spread(scope):
        loss = CREDALoss(sigma="auto", sigma_from=scope)
        per = [float(loss._sigma_for(f_s[y_s == c], f_t[y_t == c], f_s, f_t))
               for c in range(CLASSES)]
        return max(per) / min(per), per

    pooled_ratio, pooled = spread("class-pooled")
    assert pooled_ratio > 1.5, (
        f"class-pooled is expected to spread the bandwidths widely: {pooled}")
    for scope in ("batch-pooled", "batch-source"):
        ratio, per = spread(scope)
        assert ratio == pytest.approx(1.0, abs=1e-12), (
            f"{scope} must give every class the same bandwidth: {per}")
