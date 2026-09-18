"""Section 4's attention-mechanism comparison: what each mechanism computes.

Comparison-only plumbing (`wiring.MECHANISMS`/`mechanism_weights`/
`mechanism_embedding`/`MechanismArm`) -- no declared arm reads any of it, and
`config.ARMS`'s own `G` still trains under Eq. (15)/(16) exactly as before this
file existed. Every test here recomputes a mechanism's own definition by hand,
never by calling the function under test a second time: a copy that called
itself back would pass on a value it never checked.
"""

from __future__ import annotations

import math

import pytest
import torch

from MIL_CREDA_Benchmark import wiring

torch.manual_seed(0)


def _bag(m: int = 5, d: int = 4) -> torch.Tensor:
    """One bag's instance embeddings, small and fixed."""
    return torch.randn(m, d)


# ------------------------------------------------------------------------ ours

def test_ours_is_eq15s_hybrid_with_the_l1_ball_reparametrization() -> None:
    """`"ours"` is exactly `MIL_CREDA.attention.relevance_logits` -- the
    hybrid of `R_phi` (l1-ball-constrained `v_R`) plus `gamma` times the
    consensus term -- recomputed here from the same primitives
    `relevance_logits` itself is built from, never by calling
    `mechanism_weights` and comparing it to itself.

    Reachable red: `mechanism_weights("ours", ...)` reading `v_R` raw
    (skipping the l1-ball reparametrization), or dropping the consensus term.
    """
    from MIL_CREDA.attention import _l1_ball_reparametrization, gaussian_kernel

    H = _bag()
    V_R = torch.randn(3, H.shape[1])
    b_R = torch.randn(3)
    v_R = torch.randn(3)
    gamma, sigma, tau = 0.7, 2.0, 1.3

    v_tilde = _l1_ball_reparametrization(v_R)
    hidden = torch.tanh(H @ V_R.transpose(0, 1) + b_R)
    relevance = hidden @ v_tilde
    K = gaussian_kernel(H, H, sigma)
    consensus = K.sum(dim=1) / H.shape[0]
    logits = relevance + gamma * consensus
    expected = torch.softmax(logits / tau, dim=0)

    got = wiring.mechanism_weights(
        "ours", H, {"V_R": V_R, "b_R": b_R, "v_R": v_R, "gamma": gamma,
                    "sigma": sigma, "tau_att": tau})
    assert torch.allclose(got, expected, atol=1e-6)
    # And a mutation actually changes the fixture: gamma=0 drops the
    # consensus term entirely, which must move the result.
    without_consensus = wiring.mechanism_weights(
        "ours", H, {"V_R": V_R, "b_R": b_R, "v_R": v_R, "gamma": 0.0,
                    "sigma": sigma, "tau_att": tau})
    assert not torch.allclose(got, without_consensus)


# ------------------------------------------------------------- abmil-published

def test_abmil_published_uses_v_r_raw_and_no_consensus() -> None:
    """The published mechanism: `R(h) = v_R^T tanh(V_R h + b_R)`, `v_R`
    UNCONSTRAINED -- never passed through the l1-ball reparametrization --
    and no `gamma * consensus` term added anywhere.

    Reachable red: `_abmil_published_logits` silently normalizing `v_R`, or
    a stray `+ gamma * consensus` term added to its logits.
    """
    H = _bag()
    V_R = torch.randn(3, H.shape[1])
    b_R = torch.randn(3)
    v_R = torch.randn(3) * 5.0  # deliberately outside the unit l1 ball
    tau = 0.8

    hidden = torch.tanh(H @ V_R.transpose(0, 1) + b_R)
    expected_logits = hidden @ v_R
    expected = torch.softmax(expected_logits / tau, dim=0)

    got = wiring.mechanism_weights(
        "abmil-published", H, {"V_R": V_R, "b_R": b_R, "v_R": v_R, "tau_att": tau})
    assert torch.allclose(got, expected, atol=1e-6)

    # It disagrees with "ours" on the SAME raw parameters, precisely because
    # "ours" applies the l1-ball reparametrization and adds a consensus term
    # this mechanism never sees.
    ours = wiring.mechanism_weights(
        "ours", H, {"V_R": V_R, "b_R": b_R, "v_R": v_R, "gamma": 0.5,
                    "sigma": 2.0, "tau_att": tau})
    assert not torch.allclose(got, ours)


# ----------------------------------------------------------------- abmil-gated

def test_abmil_gated_is_the_published_gated_form() -> None:
    """`a = w^T (tanh(Vh) (dot) sigm(Uh))`, recomputed with plain tensor ops
    and independent `V`/`U`/`w` -- never `V_R`/`b_R`/`v_R`.

    Reachable red: `_abmil_gated_logits` swapping the elementwise product for
    a sum, or reusing one projection for both `tanh` and `sigm`.
    """
    H = _bag()
    V = torch.randn(3, H.shape[1])
    U = torch.randn(3, H.shape[1])
    w = torch.randn(3)
    tau = 1.0

    expected_logits = (torch.tanh(H @ V.transpose(0, 1))
                       * torch.sigmoid(H @ U.transpose(0, 1))) @ w
    expected = torch.softmax(expected_logits / tau, dim=0)

    got = wiring.mechanism_weights(
        "abmil-gated", H, {"V": V, "U": U, "w": w, "tau_att": tau})
    assert torch.allclose(got, expected, atol=1e-6)

    # Swapping V and U changes the result -- tanh and sigm read different
    # projections and are not interchangeable.
    swapped = wiring.mechanism_weights(
        "abmil-gated", H, {"V": U, "U": V, "w": w, "tau_att": tau})
    assert not torch.allclose(got, swapped)


# ------------------------------------------------------------------------- max

def test_max_pooling_is_the_coordinatewise_maximum_and_has_no_weights() -> None:
    """Max pooling: one embedding per bag, the coordinatewise maximum over
    instances -- never a weighted sum, so `mechanism_weights` reports `None`
    rather than a vector a caller could average with.

    Reachable red: `mechanism_embedding("max", ...)` returning a weighted
    average that happens to equal the max on this fixture, or
    `mechanism_weights("max", ...)` returning a one-hot vector instead of
    `None`.
    """
    H = _bag()
    expected = torch.stack([H[:, j].max() for j in range(H.shape[1])])

    assert wiring.mechanism_weights("max", H, {}) is None
    got = wiring.mechanism_embedding("max", H, {})
    assert torch.allclose(got, expected)
    # Not a one-hot-weighted sum either: the maximum is taken PER FEATURE,
    # so unless one instance dominates every coordinate, no single row of
    # `H` equals the result.
    assert not any(torch.allclose(got, H[row]) for row in range(H.shape[0]))


# ------------------------------------------------------------------------ mean

def test_mean_pooling_equals_eq16_with_uniform_weights() -> None:
    """Mean pooling IS Eq. (16) with `beta_a = 1/m` for every instance --
    the exact claim this comparison's own report makes
    (`tables.render_mechanisms`'s docstring lists mean pooling as one of the
    five). Recomputed two ways and both must agree: the bare arithmetic mean
    of `H`'s rows, and `MIL_CREDA.attention.bag_weights`/`bag_embedding`
    fed a manually-built uniform-logit vector (so temperature and softmax
    normalization are exercised too, not sidestepped).

    Reachable red: `mechanism_weights("mean", ...)` returning any
    non-uniform vector, or `mechanism_embedding` computing the mean some
    other way than a weighted sum with those weights.
    """
    from MIL_CREDA.attention import bag_embedding, bag_weights

    H = _bag()
    m = H.shape[0]

    weights = wiring.mechanism_weights("mean", H, {})
    assert torch.allclose(weights, torch.full((m,), 1.0 / m))

    # Eq. (16) applied to a uniform logit vector reduces to the same uniform
    # weights, whatever the temperature -- the softmax of a constant is
    # constant.
    uniform_logits = torch.zeros(m)
    via_eq16 = bag_weights(uniform_logits, tau_att=3.7)
    assert torch.allclose(via_eq16, weights, atol=1e-6)

    got = wiring.mechanism_embedding("mean", H, {})
    assert torch.allclose(got, H.mean(dim=0), atol=1e-6)
    assert torch.allclose(got, bag_embedding(H, weights), atol=1e-6)


def test_the_five_mechanisms_give_five_different_embeddings_on_the_same_bag(
) -> None:
    """Not a rounding difference: on one fixed bag, the five mechanisms'
    embeddings are pairwise distinct. A comparison whose five branches
    silently collapsed to one function would still pass every test above in
    isolation; this is the one that would catch it."""
    H = _bag()
    width = H.shape[1]
    common = {"V_R": torch.randn(3, width), "b_R": torch.randn(3),
             "v_R": torch.randn(3), "gamma": 0.5, "sigma": 2.0, "tau_att": 1.0}
    gated = {"V": torch.randn(3, width), "U": torch.randn(3, width),
            "w": torch.randn(3), "tau_att": 1.0}
    params = {"ours": common, "abmil-published": common,
             "abmil-gated": gated, "max": {}, "mean": {}}

    embeddings = {m: wiring.mechanism_embedding(m, H, params[m])
                 for m in wiring.MECHANISMS}
    values = list(embeddings.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert not torch.allclose(values[i], values[j]), (
                list(embeddings)[i], list(embeddings)[j])


def test_mechanism_weights_refuses_an_unknown_mechanism() -> None:
    with pytest.raises(ValueError):
        wiring.mechanism_weights("not-a-mechanism", _bag(), {})
