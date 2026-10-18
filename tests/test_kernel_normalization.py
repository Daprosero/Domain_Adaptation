"""The instance kernel's scale, and the fork that bounds it without touching
the head.

A trained encoder's embedding scale drifts, and the kernel's regime is set by
the ratio of that scale to `config.KERNEL_SIGMA` -- a constant chosen before
training. Measured on toy datasets this session: the trained encoder's median
instance-embedding distance ended at 7.03 with BatchNorm and 38.97 without --
same architecture, same data, a 5.5x drift -- and two of six cells ended with
a dead or saturated kernel because of it.

L2-normalizing the instance embedding bounds ||h - h'||^2 to [0, 4] by
construction, which pins the scale the kernel operates in. But a GLOBAL
normalization would also change what the classification head eats, and the
head is the one consumer that wants the scale free -- so `config.
NORMALIZE_KERNEL_EMBEDDING` and `Arm._kernel_embedding` fork the two: every
kernel consumer reads the normalized view, and `self.head` never does.

Four invariants, each with its own test:

1. `test_the_default_leaves_every_path_bit_identical_to_before_the_fork` --
   with the flag off (today's default), nothing changes anywhere.
2. `test_every_kernel_consumer_sees_unit_norm_rows_and_the_head_does_not` --
   with the flag on, every call that reaches the instance kernel sees
   unit-norm rows, and `bag_embedding` (the head's own input) never does.
3. `test_the_pairwise_squared_distance_bound_actually_holds` -- the bound
   `_kernel_embedding` exists to establish is measured directly, both ways.
4. `test_both_backbones_and_both_named_arms_train_one_step_with_the_flag_on`
   -- `mlp`/`mlp-nobn` crossed with arms `B`/`G` train one step with the flag
   on, with a finite loss.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from MIL_CREDA_Benchmark import config, wiring

CLASSES = config.CLASSES


class _Encoder(nn.Module):
    """Something with a declared `output_dim`, standing in for
    `CREDA.models.FeatureExtractor` -- no claim here is about the backbone
    itself, only about what reaches the kernel and what reaches the head.
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
    images, the identical shape `tests/test_arm_objectives.py` already uses
    for the same reason: nothing here is a claim about the backbone.
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
    actually consume, since neither backbone has a convolutional front end.
    """
    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, dim, generator=generator)
    members = torch.arange(images.shape[0]).reshape(bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % CLASSES
    return wiring.Pool(images, members, labels)


def _pairwise_squared_distance(H: torch.Tensor) -> torch.Tensor:
    """||h_a - h_b||^2 for every pair of rows -- exactly the quantity
    `gaussian_kernel` (`MIL_CREDA.kernels`) puts inside its exponential, read
    directly here rather than through the kernel so this test is about the
    embedding `_kernel_embedding` hands the kernel, not about the kernel's
    own exponential.
    """
    return torch.cdist(H, H, p=2) ** 2


# --------------------------------------------------------------- invariant 1


def test_the_default_leaves_every_path_bit_identical_to_before_the_fork(encoder) -> None:
    """With the flag off -- `config.NORMALIZE_KERNEL_EMBEDDING`'s own default
    -- nothing anywhere changes: `_kernel_embedding` hands back the exact
    same tensor object it was given, `bags_of`'s pairs still carry the raw
    embedding, and a full `training_step` reproduces the pinned values below
    byte for byte.

    Pinned by value, not merely by shape: the floats below were read off one
    run of this exact fixed-seed arm BEFORE `config.NORMALIZE_KERNEL_
    EMBEDDING` existed at all -- the fork is additive by construction (an
    identity return when the flag is off), so a later edit to the DEFAULT
    path -- the flag flipping its own default, or `_kernel_embedding` doing
    something other than a bare return when off -- reddens this test rather
    than silently drifting every existing number in the repository.

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

    Reachable red: flip the identity branch in `_kernel_embedding` to always
    normalize (dropping the `if not config.NORMALIZE_KERNEL_EMBEDDING`
    guard) and every assertion below fails -- the object-identity one first,
    since a normalized tensor is never the same object as its input.
    """
    assert config.NORMALIZE_KERNEL_EMBEDDING is False, (
        "this invariant is about the DEFAULT; a test that flips it first "
        "would not be testing the default at all"
    )

    torch.manual_seed(11)
    arm = wiring.build("G", CLASSES, _pool(1), _pool(2))

    # The helper itself: identity, not merely equal -- the strongest form of
    # "unchanged" there is.
    H = torch.randn(5, arm.encoder.output_dim)
    assert arm._kernel_embedding(H) is H, (
        "with the flag off, _kernel_embedding must hand back the exact same "
        "tensor object -- not merely an equal one -- so every downstream "
        "computation is bit-identical to before this fork existed"
    )

    embeddings = arm.instance_embeddings(
        arm.source.take(torch.arange(config.BAGS_PER_STEP))
    )
    pairs = arm.bags_of(embeddings, arm.sigma)
    for i, (H_pair, _) in enumerate(pairs):
        assert torch.equal(H_pair, embeddings[i]), (
            "bags_of must still hand back the raw embedding with the flag "
            "off -- normalizing it here would be exactly the defect this "
            "design exists to avoid, silently reaching the head through it"
        )

    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    result = arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    pinned = {
        # **Five of these moved on 2026-10-18 and r33's Eq. (37) is why.** This
        # step has `activeClasses == 2` of 3, and the abandoned class holds 4 of
        # the 10 source bags: r31 dropped it from the sum, so the global term
        # read 8.4e-07 while a third of the classes were being abandoned, and
        # r33 charges `l~_c = 1` for it at weight 0.4. The same five values and
        # the same reason are recorded at
        # `tests/test_local_dispersion.py`'s own copy of this pin, which shares
        # this fixture; the superseded numbers are kept there in full.
        #
        #   loss                0.12774033844470978   -> 0.5277459621429443
        #   adaptation          3.4521766792749986e-05 -> 0.8000458478927612
        #   contributionGlobal  8.428356181866548e-07  -> 0.4000064432621002
        #   contributionLocal   1.6418047380284406e-05 -> 1.646578311920166e-05
        #   contribution        1.7260883396374993e-05 -> 0.4000229239463806
        #
        # `supervised` did NOT move: Eq. (21) reads the source alone, so no
        # aggregation change can reach it. Nor did the kernel means, the
        # confidence readings or the bag entropy -- which is what makes this a
        # pin and not a re-measurement.
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
            f"{key} came out {value!r} with the flag at its default; the "
            f"pinned value from before this fork existed is {expected!r}"
        )
    assert result["activeClasses"] == 2


# --------------------------------------------------------------- invariant 2


def test_every_kernel_consumer_sees_unit_norm_rows_and_the_head_does_not(
        encoder, monkeypatch) -> None:
    """With the flag on, every call that reaches the instance kernel --
    through `relevance_logits`'s consensus term, and through `bag_kernel` /
    `bag_kernel_matrix` -- sees unit-norm rows on both arguments, and
    `bag_embedding` -- the head's own input -- never does.

    Spies on the same two module-level names the instance kernel is actually
    imported under (`MIL_CREDA.attention.gaussian_kernel`,
    `MIL_CREDA.bag_kernel.gaussian_kernel`), the way `tests/
    test_arm_objectives.py::
    test_every_sigma_consumer_receives_the_same_one_whatever_it_is` already
    spies on `wiring.relevance_logits`/`bag_kernel_matrix`/`bag_kernel`.

    Reachable red, verified by actually doing it rather than by asserting it:
    commenting out the `self._kernel_embedding(H)` call in `_milcreda_term`'s
    `kernel_bags_t = self._kernel_pairs(bags_t)` line (routing `bags_t`'s raw
    H into `bag_kernel_matrix` for `K_tt` instead) reddens the kernel-side
    assertion below and the failure names `bag_kernel` as the consumer that
    saw a non-unit-norm row -- see the verification transcript.
    """
    import MIL_CREDA.attention as attention_module
    import MIL_CREDA.bag_kernel as bag_kernel_module

    kernel_calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    head_calls: list[torch.Tensor] = []

    real_attention_kernel = attention_module.gaussian_kernel

    def spy_attention_kernel(X, Y, sigma):
        kernel_calls.append(("attention.gaussian_kernel", X, Y))
        return real_attention_kernel(X, Y, sigma)

    real_bag_kernel_module_kernel = bag_kernel_module.gaussian_kernel

    def spy_bag_kernel_module_kernel(X, Y, sigma):
        kernel_calls.append(("bag_kernel.gaussian_kernel", X, Y))
        return real_bag_kernel_module_kernel(X, Y, sigma)

    real_bag_embedding = wiring.bag_embedding

    def spy_bag_embedding(H, w):
        head_calls.append(H)
        return real_bag_embedding(H, w)

    monkeypatch.setattr(attention_module, "gaussian_kernel", spy_attention_kernel)
    monkeypatch.setattr(bag_kernel_module, "gaussian_kernel", spy_bag_kernel_module_kernel)
    monkeypatch.setattr(wiring, "bag_embedding", spy_bag_embedding)
    monkeypatch.setattr(config, "NORMALIZE_KERNEL_EMBEDDING", True)

    torch.manual_seed(11)
    arm = wiring.build("G", CLASSES, _pool(1), _pool(2))
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    assert kernel_calls, "no kernel consumer was ever called"
    consumers = {name for name, _, _ in kernel_calls}
    assert consumers == {"attention.gaussian_kernel", "bag_kernel.gaussian_kernel"}, (
        f"expected both the attention consensus term and the bag kernel to "
        f"reach the instance kernel; saw only {consumers!r}"
    )
    assert head_calls, "bag_embedding -- the head's own input -- was never called"

    tol = 1e-5
    for name, X, Y in kernel_calls:
        norms_x = X.detach().norm(dim=-1)
        norms_y = Y.detach().norm(dim=-1)
        assert torch.allclose(norms_x, torch.ones_like(norms_x), atol=tol), (
            f"{name} saw a non-unit-norm row on its first argument with the "
            f"flag on: {norms_x.tolist()!r}"
        )
        assert torch.allclose(norms_y, torch.ones_like(norms_y), atol=tol), (
            f"{name} saw a non-unit-norm row on its second argument with "
            f"the flag on: {norms_y.tolist()!r}"
        )

    for H in head_calls:
        norms = H.detach().norm(dim=-1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=tol), (
            "bag_embedding -- the head's own input -- came out unit-norm "
            "with the flag on: the fork leaked past the head it must leave "
            "alone"
        )


# --------------------------------------------------------------- invariant 3


def test_the_pairwise_squared_distance_bound_actually_holds(encoder, monkeypatch) -> None:
    """The bound `_kernel_embedding` exists to establish, Decision: with the
    flag on, every pairwise squared distance the kernel sees lies in
    [0, 4]; with it off, on this same fixture, it does not.

    The fixture is deliberately scaled so the claim is exercised both ways
    on one input, rather than asserted from the normalization formula alone.
    """
    torch.manual_seed(11)
    arm = wiring.build("G", CLASSES, _pool(1), _pool(2))

    # Scaled so the UNNORMALIZED pairwise squared distance clears 4 by two
    # orders of magnitude -- a fixture that merely grazed the bound would
    # leave this test unable to tell "the bound held" from "the fixture
    # never challenged it".
    H = torch.tensor([[10.0, 0.0], [0.0, 10.0], [-10.0, -10.0]])

    raw_distances = _pairwise_squared_distance(H)
    assert float(raw_distances.max()) > 4.0, (
        "the fixture itself must exceed the bound unnormalized, or this "
        "test cannot tell a held bound from one that was never challenged"
    )

    monkeypatch.setattr(config, "NORMALIZE_KERNEL_EMBEDDING", False)
    off_distances = _pairwise_squared_distance(arm._kernel_embedding(H))
    assert torch.equal(off_distances, raw_distances), (
        "with the flag off, _kernel_embedding must not touch the distances "
        "at all"
    )
    assert float(off_distances.max()) > 4.0, (
        "with the flag off, the raw embedding's own distances must still "
        "exceed the bound -- that is the failure mode this flag exists to fix"
    )

    monkeypatch.setattr(config, "NORMALIZE_KERNEL_EMBEDDING", True)
    on_distances = _pairwise_squared_distance(arm._kernel_embedding(H))
    assert float(on_distances.min()) >= 0.0 - 1e-6, (
        f"a squared distance went negative with the flag on: {on_distances!r}"
    )
    assert float(on_distances.max()) <= 4.0 + 1e-6, (
        f"a pairwise squared distance exceeded the bound with the flag on: "
        f"{on_distances!r}"
    )


# --------------------------------------------------------------- invariant 4


@pytest.mark.parametrize("backbone", ["mlp", "mlp-nobn"])
@pytest.mark.parametrize("arm_id", ["B", "G"])
def test_both_backbones_and_both_named_arms_train_one_step_with_the_flag_on(
        arm_id: str, backbone: str, monkeypatch) -> None:
    """`mlp`/`mlp-nobn` crossed with the floor (`B`) and the complete method
    (`G`) train one step with `config.NORMALIZE_KERNEL_EMBEDDING` on, with no
    error and a finite loss -- the fork must not depend on which encoder
    produced the embedding it normalizes, nor on whether the arm carries an
    adaptation term at all.
    """
    monkeypatch.setattr(config, "BACKBONE", backbone)
    monkeypatch.setattr(config, "NORMALIZE_KERNEL_EMBEDDING", True)

    torch.manual_seed(11)
    arm = wiring.build(arm_id, CLASSES, _toy_pool(1, dim=3), _toy_pool(2, dim=3))
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    result = arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    assert torch.isfinite(result["loss"]), (
        f"arm {arm_id!r} on backbone {backbone!r} produced a non-finite "
        f"loss with the flag on: {result['loss']!r}"
    )
