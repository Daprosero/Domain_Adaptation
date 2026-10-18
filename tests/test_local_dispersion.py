"""`config.NORMALIZE_LOCAL_BY_DISPERSION`: a PROBE on the harness side,
never a change to `MIL_CREDA.local_term` -- see `config.py`'s own docstring
for the finding this probes and the cost it pays.

Eq. (38)'s `l_loc,j = d_j^2 / 2` bounds the local term into `[0, 1]` because
r25:1047 proves `<Psi, mu> > 0`, hence `d_j^2 < 2` -- a bound against
DIVERGENCE, never against DEGENERATION: minimizing `d_j^2` alone has a
degenerate global minimum at collapse (every target bag mapped to one point
in H_I), where `d_j^2 -> 0` for every bag and the local score reads perfect
while carrying zero information. Measured: without BatchNorm, only the arm
carrying the local term (`G`) lowers the target embedding's covariance
participation ratio against the floor, and on the rollo transfer it does so
while LOSING accuracy (0.4722 -> 0.2222).

The remedy this probes: normalize `d_j^2` by the target bags' own measured
dispersion in H_I instead of the constant 2, read off `K_tt` -- already
built by `_milcreda_term` -- via `wiring._target_dispersion` and applied by
`wiring._dispersion_normalized_local_term`.

Six invariants, each with its own test:

1. `test_the_flag_off_is_bit_identical` -- with the flag off (today's
   default), `training_step` reproduces the pinned values below byte for
   byte, and `targetDispersion` stays the nan sentinel: nothing computed one.
2. `test_the_flag_on_changes_only_the_local_half` -- with the flag on, for
   the same seed and inputs, the global half of `_milcreda_term`'s returned
   pair is bit-identical to the flag off; only the local half differs.
3. `test_the_dispersion_matches_the_one_K_tt_already_carries` -- the
   dispersion the implementation used is independently recomputed, in this
   test, from the exact `K_tt` the implementation built, by the plain
   pairwise formula -- never by calling `wiring._target_dispersion` again.
4. `test_collapse_stops_being_free_only_with_the_flag_on` -- two synthetic
   `K_tt` matrices, same `d_j^2`, different dispersions: the flag's own
   aggregation reads the collapsed one as a LARGER local term, while Eq.
   (38)'s own aggregation (the flag off) reads the two as identical.
5. `test_both_backbones_train_one_step_with_the_flag_on` -- `mlp` and
   `mlp-nobn`, arm `G`, one training step, flag on, finite loss.
6. `test_the_dispersion_denominator_carries_gradient_and_resists_collapse`
   -- the invariant that matters most: `_target_dispersion` must return a
   LIVE tensor, and a gradient step taken with `d_j^2` held fixed must
   INCREASE the measured dispersion. A detached denominator (an earlier,
   WRONG version of this probe) does the opposite: it amplifies collapse
   instead of resisting it, because `L = d_j^2 / D` with `D` detached gives
   a gradient on `d_j^2` alone scaled by `1/D`, which explodes as `D -> 0`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from MIL_CREDA.local_term import local_loss
from MIL_CREDA_Benchmark import config, wiring

CLASSES = config.CLASSES


class _Encoder(nn.Module):
    """Something with a declared `output_dim`, standing in for
    `CREDA.models.FeatureExtractor` -- no claim here is about the backbone.
    """

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)


def _pool(seed: int) -> wiring.Pool:
    """A pool shaped for the resnet/vit-sized `_Encoder` above -- (3, 8, 8)
    images, the identical shape `tests/test_arm_objectives.py` and
    `tests/test_kernel_normalization.py` already use for the same reason:
    nothing here is a claim about the backbone.
    """
    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, 3, 8, 8,
                         generator=generator)
    members = torch.arange(images.shape[0]).reshape(bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % CLASSES
    return wiring.Pool(images, members, labels)


def _toy_pool(seed: int, dim: int) -> wiring.Pool:
    """A pool of plain `(dim,)` vectors -- the shape `mlp`/`mlp-nobn`
    actually consume, mirroring `tests/test_kernel_normalization.py`'s own
    helper of the same name.
    """
    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, dim, generator=generator)
    members = torch.arange(images.shape[0]).reshape(bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % CLASSES
    return wiring.Pool(images, members, labels)


def _step(arm: wiring.Arm, generator: torch.Generator) -> dict:
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    return arm.training_step(x, y, 0.5, generator)


# --------------------------------------------------------------- invariant 1


def test_the_flag_off_is_bit_identical(encoder) -> None:
    """With the flag off -- `config.NORMALIZE_LOCAL_BY_DISPERSION`'s own
    default -- `training_step` must reproduce the exact values a full step
    produced before this flag existed (pinned by
    `tests/test_kernel_normalization.py`'s own
    `test_the_default_leaves_every_path_bit_identical_to_before_the_fork`,
    on the byte-identical fixture), and `targetDispersion` must stay the
    nan sentinel: nothing computed a dispersion this step.

    Pinned by value, not merely by shape, so a later edit to the DEFAULT
    path reddens this test rather than silently drifting every existing
    number in the repository.

    Four of these twelve are a WHOLE-PIPELINE witness, not a local one, and
    saying so is the point of this paragraph. `loss`, `adaptation`,
    `contributionGlobal` and `contribution` read the adaptation term, so they
    move whenever the formulation legitimately moves -- and then this test
    reddens with a message naming this flag, which is false. It happened: the
    r25 -> r29 re-binding of the global term (Eqs. 32, 34-36 and 37) moved all
    four, and NONE of the other eight. A red here is therefore read in two
    steps: if `supervised`, `contributionLocal`, the three kernel means and the
    two confidence readings are all still bit-identical, the default path of
    this flag is intact and what moved is the formulation. If any of those
    eight moved, the finding is this flag's.

    Reachable red: make the `if self.spec["local"]:` branch always take the
    dispersion-normalized path regardless of the flag -- every pinned value
    below changes, and `targetDispersion` stops being nan.
    """
    assert config.NORMALIZE_LOCAL_BY_DISPERSION is False, (
        "this invariant is about the DEFAULT; a test that flips it first "
        "would not be testing the default at all"
    )

    torch.manual_seed(11)
    arm = wiring.build("G", CLASSES, _pool(1), _pool(2))
    result = _step(arm, torch.Generator().manual_seed(3))

    # **Six of these twelve moved on 2026-10-18, and r33's Eq. (37) is why.**
    # This step has `activeClasses == 2` of 3 (asserted below, and it always
    # did): the target routing abandoned one class, and that class holds 4 of
    # the 10 source bags in the batch. r31's Eq. (37) dropped an inactive class
    # from the sum, so the global term read 8.4e-07 -- essentially nothing --
    # while a third of the classes were being abandoned. r33 charges
    # `l~_c = 1` for it, weighted by its own `n_c^s / N_s = 0.4`, so the term
    # reads 0.4000064: the 0.4 IS the charge and the 6.4e-06 is the two active
    # classes. That is the change working, not drifting.
    #
    # The superseded values are kept beside the new ones, because this is a
    # tripwire and a re-measured number with no record of what it replaced
    # cannot be told from one that drifted:
    #
    #   loss                0.12774033844470978  -> 0.5277459621429443
    #   adaptation          3.4521766792749986e-05 -> 0.8000458478927612
    #   contributionGlobal  8.428356181866548e-07  -> 0.4000064432621002
    #   contributionLocal   1.6418047380284406e-05 -> 1.646578311920166e-05
    #   contribution        1.7260883396374993e-05 -> 0.4000229239463806
    #
    # `supervised` did NOT move, and that is the half worth naming: Eq. (21)
    # reads the source alone, so no aggregation change can reach it. Nor did
    # the three kernel means, the two confidence readings or the bag entropy --
    # Eq. (37) is an aggregation and touches no block.
    #
    # `contributionLocal` moved by 4.8e-08, which is Eq. (38) going from the
    # confidence-weighted quotient to the uniform mean. It is small HERE
    # because this step's confidences are near-uniform and tiny
    # (`targetConfidenceMean` 0.0245), so the weighted quotient was already
    # close to the mean; it is not small in general.
    pinned = {
        "loss": 0.5277459621429443,
        "supervised": 0.1277230679988861,
        "adaptation": 0.8000458478927612,
        "contributionGlobal": 0.4000064432621002,
        "contributionLocal": 1.646578311920166e-05,
        "contribution": 0.4000229239463806,
        "targetBagEntropy": 7.510467548854649e-05,
        "targetConfidenceMean": 0.024530548602342606,
        "targetConfidenceMin": 0.021144390106201172,
        "sourceKernelMean": 0.9985478520393372,
        "crossKernelMean": 0.9985128045082092,
        "targetKernelMean": 0.9984973669052124,
    }
    for key, expected in pinned.items():
        value = result[key]
        value = float(value.detach()) if hasattr(value, "detach") else float(value)
        assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9), (
            f"{key} came out {value!r} with the flag off; the pinned value is "
            f"{expected!r}"
        )
    assert result["activeClasses"] == 2
    assert math.isnan(result["targetDispersion"]), (
        "the flag is off, so nothing computed a dispersion this step -- "
        "the reading must stay the nan sentinel, never a real number "
        "nobody measured"
    )


# --------------------------------------------------------------- invariant 2


def test_the_flag_on_changes_only_the_local_half(encoder, monkeypatch) -> None:
    """For the SAME seed and the SAME inputs, `_milcreda_term`'s global half
    must be bit-identical whether `config.NORMALIZE_LOCAL_BY_DISPERSION` is
    on or off; only the local half may differ.

    Reachable red: have the flag also gate something in the global score's
    own computation (e.g. route `K_ss`/`K_st` through a different
    normalization when the flag is on) -- `global_off` and `global_on`
    stop being `torch.equal`.
    """
    torch.manual_seed(11)
    arm = wiring.build("G", CLASSES, _pool(1), _pool(2))
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    embeddings = arm.instance_embeddings(x)
    target = arm.target.take(arm._draw_target(torch.Generator().manual_seed(3)))

    monkeypatch.setattr(config, "NORMALIZE_LOCAL_BY_DISPERSION", False)
    global_off, local_off = arm._milcreda_term(embeddings, y, target)

    monkeypatch.setattr(config, "NORMALIZE_LOCAL_BY_DISPERSION", True)
    global_on, local_on = arm._milcreda_term(embeddings, y, target)

    assert torch.equal(global_off, global_on), (
        "the global half must be bit-identical whether the probe flag is "
        "on or off -- the flag has no business touching Eqs. (36)/(37)'s "
        "own computation"
    )
    assert float(local_off.detach()) != pytest.approx(float(local_on.detach())), (
        "the flag must actually change the local half for this fixture, "
        "or this invariant was never really exercised"
    )


# --------------------------------------------------------------- invariant 3


def test_the_dispersion_matches_the_one_K_tt_already_carries(encoder, monkeypatch) -> None:
    """The dispersion the implementation used for THIS step is the mean
    pairwise squared distance among the target bag representations in
    `K_tt` -- recomputed here independently, by the plain formula
    `K[i,i] + K[j,j] - 2*K[i,j]` averaged over `i != j`, from the exact
    `K_tt` the implementation built, never by calling `wiring.
    _target_dispersion` a second time.

    Reachable red: make `wiring._target_dispersion` return a constant
    (e.g. `1.0`) instead of reading `K_tt` -- the independently-computed
    value stops matching `result["targetDispersion"]`.
    """
    monkeypatch.setattr(config, "NORMALIZE_LOCAL_BY_DISPERSION", True)

    captured: list[torch.Tensor] = []
    real_target_dispersion = wiring._target_dispersion

    def spy(K_tt):
        captured.append(K_tt.detach().clone())
        return real_target_dispersion(K_tt)

    monkeypatch.setattr(wiring, "_target_dispersion", spy)

    torch.manual_seed(11)
    arm = wiring.build("G", CLASSES, _pool(1), _pool(2))
    result = _step(arm, torch.Generator().manual_seed(3))

    assert len(captured) == 1, (
        f"expected exactly one call to _target_dispersion per step, saw "
        f"{len(captured)}"
    )
    K_tt = captured[0]
    n = K_tt.shape[0]
    assert n > 1, "a single target bag carries no off-diagonal pair to average"

    total = 0.0
    count = 0
    diag = K_tt.diagonal()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total += float(diag[i] + diag[j] - 2.0 * K_tt[i, j])
            count += 1
    independent = total / count

    assert not math.isnan(result["targetDispersion"]), (
        "the flag is on and the local term is on for arm G -- a dispersion "
        "must have been computed and reported"
    )
    assert math.isclose(result["targetDispersion"], independent, rel_tol=0.0, abs_tol=1e-5), (
        f"the implementation's targetDispersion "
        f"({result['targetDispersion']!r}) does not match the mean "
        f"pairwise squared distance computed independently from the exact "
        f"K_tt it built ({independent!r})"
    )


# --------------------------------------------------------------- invariant 4


def test_collapse_stops_being_free_only_with_the_flag_on() -> None:
    """Two synthetic target bag Grams, same `d_j^2` for every bag, different
    dispersions -- one spread (low off-diagonal similarity, i.e. the bags
    are far apart in H_I), one nearly collapsed (off-diagonal similarity
    close to the diagonal's own self-similarity of 1, i.e. the bags are
    nearly on top of each other).

    With `wiring._dispersion_normalized_local_term` (the flag's own
    aggregation), the collapsed case must read as a LARGER local term for
    the identical `d_j^2` values. With `MIL_CREDA.local_term.local_loss`
    (the flag off, Eq. (38) as declared), the two must come out identical --
    that aggregation never reads `K_tt` at all, which is exactly the
    degeneracy this probe exists to fix.

    Reachable red: make `wiring._target_dispersion` return the same
    constant for both fixtures -- `term_collapsed` stops exceeding
    `term_spread`.
    """
    distances = torch.tensor([0.5, 0.5, 0.5])
    weights = torch.tensor([1.0, 1.0, 1.0])

    spread_K_tt = torch.tensor([
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ])
    collapsed_K_tt = torch.tensor([
        [1.0, 0.999, 0.999],
        [0.999, 1.0, 0.999],
        [0.999, 0.999, 1.0],
    ])

    # `weights` left this signature with r33's Eq. (38): the aggregation is
    # the uniform mean, so the probe mirrors it and has nothing to pass.
    term_spread, dispersion_spread = wiring._dispersion_normalized_local_term(
        distances, spread_K_tt)
    term_collapsed, dispersion_collapsed = wiring._dispersion_normalized_local_term(
        distances, collapsed_K_tt)

    assert dispersion_collapsed < dispersion_spread, (
        "the fixture itself must be more collapsed than the spread one, or "
        "this test cannot tell a real difference from a mutation that "
        "never changed anything"
    )
    assert float(term_collapsed) > float(term_spread), (
        "with the SAME d_j^2 values, the flag's own aggregation must read "
        "the collapsed cloud as a LARGER local term than the spread one -- "
        "that is the whole claim this probe makes, and it must be measured "
        "here rather than asserted in a comment"
    )

    # r33 Eq. (38) is the uniform mean: `weights` and the stabilizer left the
    # signature at r32, so the two readings below are the same call twice --
    # which is the point here, the flag OFF ignores the dispersion entirely.
    off_spread = local_loss(distances)
    off_collapsed = local_loss(distances)
    assert math.isclose(float(off_spread), float(off_collapsed), rel_tol=0.0, abs_tol=1e-12), (
        "Eq. (38)'s own aggregation never reads K_tt at all, so the spread "
        "and collapsed fixtures must come out identical with the flag off "
        "-- this identical reading is exactly the degeneracy the probe "
        "exists to fix"
    )


# --------------------------------------------------------------- invariant 5


@pytest.mark.parametrize("backbone", ["mlp", "mlp-nobn"])
def test_both_backbones_train_one_step_with_the_flag_on(backbone, monkeypatch) -> None:
    """`mlp`/`mlp-nobn`, arm `G` (the only arm carrying the local term, so
    the only one that can ever reach the dispersion-normalized path), one
    training step, `config.NORMALIZE_LOCAL_BY_DISPERSION` on: no error, a
    finite loss, and a finite reported dispersion.

    Also prints the observed per-bag normalized value's range for this
    backbone -- plain output, never an assertion, per the task's own
    request to report how far outside `[0, 1]` it actually goes (run with
    `-s` to see it).
    """
    monkeypatch.setattr(config, "BACKBONE", backbone)
    monkeypatch.setattr(config, "NORMALIZE_LOCAL_BY_DISPERSION", True)

    captured: list[tuple[torch.Tensor, float]] = []
    real = wiring._dispersion_normalized_local_term

    def spy(distances, K_tt):
        result = real(distances, K_tt)
        captured.append((distances.detach().clone(), result[1]))
        return result

    monkeypatch.setattr(wiring, "_dispersion_normalized_local_term", spy)

    torch.manual_seed(11)
    arm = wiring.build("G", CLASSES, _toy_pool(1, dim=3), _toy_pool(2, dim=3))
    result = _step(arm, torch.Generator().manual_seed(3))

    assert torch.isfinite(result["loss"]), (
        f"backbone {backbone!r} produced a non-finite loss with the flag on"
    )
    assert math.isfinite(result["targetDispersion"]), (
        f"backbone {backbone!r} produced a non-finite targetDispersion "
        f"with the flag on"
    )
    assert captured, "the dispersion-normalized path was never reached"

    distances, dispersion = captured[0]
    normalized = distances / (dispersion + config.EPSILON_LOCAL)
    print(
        f"\n[invariant 5] backbone={backbone!r} dispersion={dispersion!r} "
        f"per-bag normalized value range: "
        f"[{float(normalized.min())!r}, {float(normalized.max())!r}]"
    )


# --------------------------------------------------------------- invariant 6


def test_the_dispersion_denominator_carries_gradient_and_resists_collapse() -> None:
    """The invariant that matters most, and the correction to an earlier,
    WRONG version of this probe that detached `_target_dispersion`'s
    result.

    With `L = d_j^2 / D` and `D` detached, `dL/dtheta = (1/D) *
    d(d_j^2)/dtheta`: descent lowers `d_j^2` with a step scaled by `1/D`,
    and as the target cloud collapses `D -> 0` so that scale EXPLODES --
    a detached denominator is a POSITIVE FEEDBACK LOOP into the exact
    collapse this probe exists to resist, not merely "no anti-collapse
    force". With `D` live, a second term `-(d_j^2/D^2) * dD/dtheta`
    appears and pushes to INCREASE `D`; that term is the entire remedy.

    Constructed so `d_j^2` (`distances`) is a plain constant with no
    `requires_grad` -- held fixed -- while `K_tt`, and therefore `D`, is a
    differentiable function of one scalar `alpha` (the off-diagonal
    similarity: the diagonal is 1, every off-diagonal entry is `alpha`, so
    `D = 2 * (1 - alpha)`). The only way a gradient step on `alpha` can
    lower the loss here is through `D`, so with the denominator live a
    step must INCREASE `D` (lower `alpha`).

    Reachable red: re-detach `_target_dispersion`'s return (append
    `.detach()` to it) -- `dispersion_before.requires_grad` turns `False`
    and this test fails on that assertion, naming the dispersion, before a
    gradient step is even attempted.
    """
    alpha = torch.tensor(0.9, requires_grad=True)
    eye = torch.eye(3)
    ones = torch.ones(3, 3)

    def K_tt_from(a: torch.Tensor) -> torch.Tensor:
        return eye * (1.0 - a) + ones * a

    distances = torch.tensor([0.5, 0.5, 0.5])  # d_j^2, held fixed: no grad
    weights = torch.tensor([1.0, 1.0, 1.0])

    K_tt = K_tt_from(alpha)
    dispersion_before = wiring._target_dispersion(K_tt)
    assert dispersion_before.requires_grad, (
        "_target_dispersion must return a LIVE tensor -- with a detached "
        "one, the loss's gradient never reaches the denominator at all, "
        "and the anti-collapse term this probe exists to add simply does "
        "not exist"
    )

    local_term, _ = wiring._dispersion_normalized_local_term(distances, K_tt)
    local_term.backward()
    assert alpha.grad is not None, (
        "no gradient reached alpha through K_tt and the dispersion -- the "
        "live path this correction exists to keep open is broken"
    )

    lr = 0.05
    with torch.no_grad():
        new_alpha = (alpha - lr * alpha.grad).clamp(0.0, 1.0)
    dispersion_after = wiring._target_dispersion(K_tt_from(new_alpha))

    assert float(dispersion_after) > float(dispersion_before.detach()), (
        f"a gradient step on alpha, with d_j^2 held fixed, must INCREASE "
        f"the measured dispersion -- that increase is the entire "
        f"anti-collapse remedy, and without it the flag is only a "
        f"rescaling. Saw {float(dispersion_before.detach())!r} -> "
        f"{float(dispersion_after)!r}"
    )
