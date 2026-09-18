"""What each arm computes, and what the campaign hands every arm.

Three claims live here that the declaration tests could only look at from the
outside. That the selecting arms spend a budget of ten is a fact about `select`,
not about the constant it reads. That prior work is used as it was written is a
fact about the objective `training_step` assembles for an instance-unit arm. And
that every declared arm sees one contamination draw is a fact about the loop in
`campaign`, where the material is built.

The encoder is stubbed -- it is a pretrained resnet18 and no claim here is about
it -- and everything the claims are about runs for real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from MIL_CREDA_Benchmark import config, harness, wiring

CLASSES = config.CLASSES


class _Encoder(nn.Module):
    """Something with an `output_dim` that maps an instance to a row."""

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)


class _BNEncoder(nn.Module):
    """Like `_Encoder`, but with a running-stats layer of its own.

    `_Encoder` carries no `BatchNorm`, deliberately: most claims here have
    nothing to do with it. The claim that an adapted arm's target forward
    updates the encoder's running statistics -- normalization is part of the
    architecture, not a separate mechanism -- is entirely ABOUT it, so it
    needs a stub that actually has running statistics to observe move.
    """

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)
        self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        return self.bn(self.linear(x.reshape(x.shape[0], -1)))


@pytest.fixture
def bn_encoder(monkeypatch):
    monkeypatch.setattr(wiring, "FeatureExtractor", _BNEncoder)


def _pool(seed: int) -> wiring.Pool:
    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, 3, 8, 8,
                         generator=generator)
    members = torch.arange(images.shape[0]).reshape(bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % CLASSES
    return wiring.Pool(images, members, labels)


def _arm(arm_id: str):
    torch.manual_seed(11)
    return wiring.build(arm_id, CLASSES, _pool(1), _pool(2))


# ------------------------------------------------------------- selection is gone

def test_no_declared_arm_selects_a_subset_of_its_bag(encoder) -> None:
    """The selection arms (`SU`, `SA`, `SK`) and their budget (`SELECT_K`,
    `SELECTION_SEED`) are removed: every declared arm's `spec["selection"]`
    is `None`, and `Arm.select` always returns every instance of a bag
    unchanged, for every one of them.

    Reachable red: a declared arm with a non-`None` `selection`, or `select`
    dropping an instance for an arm whose `selection` is `None`.
    """
    assert {arm["selection"] for arm in config.ARMS} == {None}
    assert not hasattr(config, "SELECT_K")
    assert not hasattr(config, "SELECTION_SEED")

    H = torch.randn(config.INSTANCES_PER_BAG, 6)
    for arm_id in config.ARMS_BY_ID:
        arm = _arm(arm_id)
        kept = arm.select(H, config.KERNEL_SIGMA)
        assert kept.shape[0] == config.INSTANCES_PER_BAG, (
            f"{arm_id}: select dropped an instance with no selection rule declared"
        )
        assert torch.equal(kept, H)


# --------------------------------------------------- attention stays within its bag

def test_a_bags_own_attention_weights_do_not_depend_on_another_bag_in_the_batch(
        encoder) -> None:
    """r21 Sec. 3: 'el logit es funcion de la instancia y de su bolsa, pero
    nunca de otras bolsas' -- checked at `Arm.bags_of`, the level a batch of
    several bags is actually built at, rather than at `relevance_logits` alone.

    Defect (c): `relevance_logits` takes one bag's `H` as its only tensor
    argument, so isolation there is a fact of the function's signature and
    cannot fail whatever a caller does with sigma -- the previous version of
    this claim tested exactly that and so could never turn red. The place
    cross-bag leakage could actually enter is the bandwidth: a batch-wide
    sigma (what `_milcreda_term` computed before Decision 1, from the whole
    source-and-target batch together) would move every bag's consensus term,
    and through it its attention weights, whenever any OTHER bag in the batch
    changed.

    Decision 1's constant `config.KERNEL_SIGMA` removes that path entirely;
    this is what proves it, by perturbing one bag of a multi-bag batch and
    checking every OTHER bag's weights are bit-identical.

    Reachable red: make sigma batch-dependent again -- recompute a median
    over the concatenated batch inside `bags_of` or a caller, instead of
    threading the one constant through.
    """
    arm = _arm("G")
    torch.manual_seed(5)
    embeddings = torch.randn(4, config.INSTANCES_PER_BAG, 6)

    baseline = arm.bags_of(embeddings, config.KERNEL_SIGMA)
    baseline_weights = [w.clone() for _, w in baseline]

    perturbed = embeddings.clone()
    perturbed[1] = perturbed[1] + 50.0  # large and unmissable
    after = arm.bags_of(perturbed, config.KERNEL_SIGMA)

    for index in (0, 2, 3):
        assert torch.equal(baseline_weights[index], after[index][1]), \
            f"bag {index}'s attention weights moved when only bag 1 changed"
    # and bag 1 itself did move, so the perturbation actually perturbed something
    assert not torch.equal(baseline_weights[1], after[1][1])


def test_weights_for_reads_the_arms_own_gamma_and_temperature(encoder) -> None:
    """Defect (h): `weights_for` passes its arm's own gamma/temperature
    through to `relevance_logits`/`bag_weights`, rather than a value a
    hardcoded 0.0/1.0 could silently stand in for at today's neutral
    hyperparameters.

    Built through `wiring.build(..., hyper=...)` rather than monkeypatching
    `config.ATTENTION_GAMMA` after construction: `Arm.__init__` resolves
    `self.attention_gamma`/`self.attention_temperature` ONCE, from `hyper` or
    `config`, precisely so the ceiling search's per-trial override is a
    property of the arm and not of whatever `config` happens to say at call
    time -- so a caller wanting a different value has to build a different
    arm, the same way the search itself does.

    Each parameter is patched ALONE, with the other pinned at its neutral --
    not both at once. Moving both together would let a mutant that hardcoded
    only ONE of the two (say, `bag_weights` always dividing by 1.0 while
    correctly reading gamma) pass unnoticed: the output would still differ
    from neutral because the OTHER parameter genuinely moved, and the
    hardcoded one would never be exercised on its own. Only a
    single-parameter patch can catch a single-parameter mutant.
    """
    def built(gamma, temperature):
        torch.manual_seed(11)
        return wiring.build("G", CLASSES, _pool(1), _pool(2),
                            hyper={"attentionGamma": gamma,
                                   "attentionTemperature": temperature})

    torch.manual_seed(7)
    H = torch.randn(config.INSTANCES_PER_BAG, 6)

    neutral_weights = built(0.0, 1.0).weights_for(H, config.KERNEL_SIGMA)

    # gamma alone, temperature pinned at its neutral
    gamma_only_weights = built(2.4, 1.0).weights_for(H, config.KERNEL_SIGMA)
    assert not torch.allclose(neutral_weights, gamma_only_weights), (
        "weights_for produced the same weights under a patched gamma alone "
        "as under the neutral ones -- it is not reading its own gamma"
    )

    # temperature alone, gamma pinned at its neutral
    temperature_only_weights = built(0.0, 0.2).weights_for(H, config.KERNEL_SIGMA)
    assert not torch.allclose(neutral_weights, temperature_only_weights), (
        "weights_for produced the same weights under a patched temperature "
        "alone as under the neutral ones -- it is not reading its own "
        "temperature"
    )


def test_local_loss_reads_the_declared_local_stabilizer(encoder, monkeypatch) -> None:
    """Defect (h): `local_loss`'s epsilon (Eq. 38's stabilizer) is passed
    explicitly from `config.EPSILON_LOCAL` in `_milcreda_term`, rather than
    silently falling through to the method's own `epsilon=1e-8` default --
    a value a caller could omit without anyone noticing which number
    governed the run.

    Patched to something far from the true default so a mutant that dropped
    the explicit argument (falling back to the function's own default) would
    read differently and this test would catch it.
    """
    arm = _arm("G")
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    generator = torch.Generator().manual_seed(3)
    embeddings = arm.instance_embeddings(x)
    target = arm.target.take(arm._draw_target(generator))

    monkeypatch.setattr(config, "EPSILON_LOCAL", 1e-8)
    _, local_default = arm._milcreda_term(embeddings, y, target)

    monkeypatch.setattr(config, "EPSILON_LOCAL", 0.5)
    _, local_patched = arm._milcreda_term(embeddings, y, target)

    assert local_default.detach().item() != pytest.approx(
        local_patched.detach().item()), (
        "the local term did not move when config.EPSILON_LOCAL changed -- "
        "local_loss is not reading it"
    )


def test_every_sigma_consumer_receives_the_one_declared_constant(
        encoder, monkeypatch) -> None:
    """Decision 1: `config.KERNEL_SIGMA` reaches the attention consensus, the
    top-k ranking, every kernel block `_milcreda_term` builds (K_ss, K_st,
    K_tt) and the local correspondence's own self-similarity evaluation --
    all as the SAME number, never a per-call recomputation.

    Spies on every function along the real training-step call path that
    takes a `sigma` argument and records what it was actually called with,
    named by function, so a consumer silently reading something else -- a
    stale per-batch median, a scaled bandwidth, anything but the one
    constant -- is caught by name rather than by a single aggregate check.

    Also drives `arm(x)` -- `Arm.forward`, the call `harness.accuracy` makes
    on every evaluation batch -- so the evaluation path's own sigma
    consumer (`forward` -> `bag_representations` -> `bags_of` ->
    `weights_for` -> `relevance_logits`) is checked too, not only the
    training path. Training and evaluation call the same method with the
    same declared constant, but nothing before this bound them together:
    `forward` could drift to a different value and every training-time spy
    above would stay green.

    Reachable red: pass a different value to any one consumer -- e.g. change
    `bag_kernel_matrix(bags_s, bags_s, sigma)`'s `sigma` to `sigma * 2` in
    `_milcreda_term`, or `config.KERNEL_SIGMA` to `config.KERNEL_SIGMA * 3`
    in `Arm.forward`.
    """
    import MIL_CREDA.attention as attention_module
    import MIL_CREDA.bag_kernel as bag_kernel_module

    seen: list[tuple[str, object]] = []

    real_relevance_logits = attention_module.relevance_logits

    def spy_relevance_logits(H, V_R, b_R, v_R, gamma, sigma):
        seen.append(("relevance_logits", sigma))
        return real_relevance_logits(H, V_R, b_R, v_R, gamma, sigma)

    real_bag_kernel_matrix = bag_kernel_module.bag_kernel_matrix

    def spy_bag_kernel_matrix(rows, cols, sigma):
        seen.append(("bag_kernel_matrix", sigma))
        return real_bag_kernel_matrix(rows, cols, sigma)

    real_bag_kernel = bag_kernel_module.bag_kernel

    def spy_bag_kernel(H_u, w_u, H_v, w_v, sigma):
        seen.append(("bag_kernel", sigma))
        return real_bag_kernel(H_u, w_u, H_v, w_v, sigma)

    monkeypatch.setattr(wiring, "relevance_logits", spy_relevance_logits)
    monkeypatch.setattr(wiring, "bag_kernel_matrix", spy_bag_kernel_matrix)
    monkeypatch.setattr(wiring, "bag_kernel", spy_bag_kernel)

    # G exercises weights_for and every kernel block plus local_distance's
    # self-similarity. `GN` drove the same path here with its own frozen-stats
    # target forward and is retired with the arm; every consumer this test
    # watches is reached by `G` alone.
    for arm_id in ("G",):
        arm = _arm(arm_id)
        x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
        y = arm.source.labels[:config.BAGS_PER_STEP]
        arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))
        # The evaluation path: `harness.accuracy` calls `model(x)` on every
        # batch, never `training_step` -- it has to be driven separately or
        # a drift confined to `forward` would never reach any spy above.
        arm(x)

    assert seen, "no sigma consumer was ever called"
    consumers = {name for name, _ in seen}
    assert consumers >= {"relevance_logits", "bag_kernel_matrix", "bag_kernel"}
    for name, sigma in seen:
        assert sigma == config.KERNEL_SIGMA, (
            f"{name} was called with sigma={sigma!r}, not the declared "
            f"constant {config.KERNEL_SIGMA!r}"
        )


class _FakeDataset:
    """The shape `harness.accuracy` iterates: `dataset[i] -> (instances, label)`."""

    def __init__(self, pool: wiring.Pool, n: int):
        self.pool = pool
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        instances = self.pool.take(torch.tensor([i]))[0]
        return instances, int(self.pool.labels[i])


def test_evaluation_sigma_matches_the_declared_constant_through_harness_accuracy(
        encoder, monkeypatch) -> None:
    """Decision 1, driven through the ACTUAL evaluation path and not a bare
    `arm(x)` call: `harness.accuracy` is `@torch.no_grad()` and explicitly
    calls `model.eval()` before scoring a batch and `model.train()` after,
    which `arm(x)` alone never does. Every declared arm is driven, not only
    two, so a drift confined to one arm's own path is caught by name --
    `bag_representations` is shared code, but nothing before this proved
    every `spec` reaches it with the same sigma in eval mode specifically.

    Reachable red: a sigma drift that only fires while `model.training` is
    `False`, or one confined to a single arm.
    """
    import MIL_CREDA.attention as attention_module

    real_relevance_logits = attention_module.relevance_logits
    seen: list[tuple[str, object]] = []

    def spy_relevance_logits(H, V_R, b_R, v_R, gamma, sigma):
        seen.append(("relevance_logits", sigma))
        return real_relevance_logits(H, V_R, b_R, v_R, gamma, sigma)

    monkeypatch.setattr(wiring, "relevance_logits", spy_relevance_logits)

    for arm_id in config.ARMS_BY_ID:
        seen.clear()
        arm = _arm(arm_id)
        dataset = _FakeDataset(arm.source, config.BAGS_PER_STEP)
        assert arm.training, f"arm {arm_id}: not in training mode before accuracy()"
        harness.accuracy(arm, dataset, torch.device("cpu"))
        assert arm.training, (
            f"arm {arm_id}: harness.accuracy did not restore training mode"
        )
        assert seen, f"arm {arm_id}: no sigma consumer was called through accuracy()"
        for name, sigma in seen:
            assert sigma == config.KERNEL_SIGMA, (
                f"arm {arm_id}: {name} was called in eval mode with "
                f"sigma={sigma!r}, not the declared constant {config.KERNEL_SIGMA!r}"
            )


# --------------------------------------------------------------- Decision 2: the floor

def test_a_floor_never_encodes_a_target_image_during_training(encoder, monkeypatch) -> None:
    """Decision 2: a floor's training step never lets a target image reach
    the encoder, ever.

    Hooked at `encoder.forward` itself, not at `instance_embeddings`.
    `instance_embeddings` is only the path the CURRENT code happens to
    reach the encoder through, so a spy placed there is blind to a mutation
    that called `self.encoder(target_bags...)` directly and bypassed it --
    a real risk for exactly the floor branch this test guards, where "call
    the encoder some other way" is the shape any reintroduced target
    exposure would take. Hooking the encoder's own `forward` catches every
    path, whatever `wiring.py` grows to call it through.

    Reachable red: restore the old unconditional
    `self.instance_embeddings(target_bags)` call in the floor branch, or add
    a direct `self.encoder(...)` call on a target-derived tensor anywhere in
    the floor's step.
    """
    arm = _arm("B")
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]

    seen_ids = []
    real_forward = arm.encoder.forward

    def spy(bags):
        seen_ids.append(bags.data_ptr())
        return real_forward(bags)

    monkeypatch.setattr(arm.encoder, "forward", spy)
    arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    # `x` is already contiguous (built by `Pool.take`'s own fancy indexing),
    # so the flattened view `instance_embeddings` reshapes it into before
    # handing it to the encoder shares `x`'s own storage and data_ptr.
    assert seen_ids == [x.data_ptr()], (
        f"the floor's training step called encoder.forward "
        f"{len(seen_ids)} time(s); only its own source batch may reach it"
    )


def test_a_floor_never_calls_take_on_the_target_pool(encoder, monkeypatch) -> None:
    """Decision 2's comment states it directly: "what a floor never does with
    the indices it draws is `take` or encode the images they name." The test
    above spies `encoder.forward` and proves no target image ever reaches
    it -- a stronger claim in one sense, but blind to a floor that called
    `self.target.take(target_indices)` and simply discarded the result
    without ever encoding it. That call has its own cost (indexing a
    potentially large `images` tensor) and its own meaning (Decision 2 is
    about not TOUCHING target material during training, not only about not
    encoding it), so it gets its own spy, directly on `Pool.take`.

    Reachable red: a floor branch that calls
    `self.target.take(target_indices)` and drops the result.
    """
    arm = _arm("B")
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]

    real_take = wiring.Pool.take
    calls = []

    def spy_take(self, positions):
        calls.append(self is arm.target)
        return real_take(self, positions)

    monkeypatch.setattr(wiring.Pool, "take", spy_take)
    arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    assert not any(calls), (
        "the floor's training step called Pool.take on its target pool "
        f"{sum(calls)} time(s); Decision 2 says it never touches it"
    )


@pytest.mark.parametrize("arm_id", ["E", "F", "G"])
def test_an_adapted_arms_target_forward_carries_gradient_to_the_encoder(
        encoder, arm_id) -> None:
    """The adaptation term trains the shared encoder through the TARGET
    forward pass, not only through the source one the supervised term
    already drives -- Eq. (39)'s whole point is one encoder both domains'
    inputs update. Isolated from the supervised term by capturing
    `_milcreda_term`'s own tensors before `training_step` detaches them into
    its report dict, and calling `.backward()` on their sum alone.

    Reachable red: wrap the target forward inside `_milcreda_term` (or the
    `target_bags = self.target.take(...)` -> `self.instance_embeddings` path
    that feeds it) in `torch.no_grad()`, or `.detach()` the target
    embeddings before they enter `_milcreda_term`.
    """
    arm = _arm(arm_id)
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]

    captured: dict = {}
    real_term = arm._milcreda_term

    def spy(embeddings, labels, target_bags):
        global_term, local_term = real_term(embeddings, labels, target_bags)
        captured["global_term"] = global_term
        captured["local_term"] = local_term
        return global_term, local_term

    arm._milcreda_term = spy
    arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    assert "global_term" in captured, "training_step never reached _milcreda_term"
    for p in arm.encoder.parameters():
        assert p.grad is None, "encoder already carried a gradient before backward"

    adaptation = captured["global_term"] + captured["local_term"]
    adaptation.backward()

    grads = [p.grad for p in arm.encoder.parameters() if p.grad is not None]
    assert grads, (
        f"arm {arm_id}: the adaptation term alone produced no gradient at all "
        "on the encoder"
    )
    assert any(torch.any(g != 0) for g in grads), (
        f"arm {arm_id}: the adaptation term's gradient on the encoder is all zero"
    )


def test_a_floor_consumes_the_generator_identically_to_an_adapted_arm(encoder) -> None:
    """Decision 2: the floor still draws the target indices `_draw_target`
    always draws -- SKILL.md: arms must not differ in how much of the
    generator they consume -- so the training generator advances by the
    same amount whichever arm is training. Only whether the images those
    indices name are ever taken or encoded differs (the test above).

    Reachable red: move the `_draw_target` call inside the
    `if adaptation == "milcreda"` branch, so the floor skips it entirely.
    """
    floor = _arm("B")
    adapted = _arm("G")
    x = floor.source.take(torch.arange(config.BAGS_PER_STEP))
    y = floor.source.labels[:config.BAGS_PER_STEP]

    gen_floor = torch.Generator().manual_seed(99)
    gen_adapted = torch.Generator().manual_seed(99)

    floor.training_step(x, y, 0.5, gen_floor)
    adapted.training_step(x, y, 0.5, gen_adapted)

    # Identical state after, from identical state before, is only possible if
    # the two arms consumed exactly the same amount of the generator's stream.
    draw_floor = torch.randn(4, generator=gen_floor)
    draw_adapted = torch.randn(4, generator=gen_adapted)
    assert torch.equal(draw_floor, draw_adapted), (
        "the floor and the adapted arm left the shared-shape generator in "
        "different states, so they consumed different amounts of it"
    )


# ------------------------------- normalization is part of the model, not a switch

def test_an_adapted_arms_target_forward_updates_running_statistics(
        bn_encoder) -> None:
    """Normalization is part of the architecture: an adapted arm's target
    forward runs through the encoder exactly like its source forward, so it
    updates the encoder's running statistics too -- checked directly against
    a stub encoder that actually carries a `BatchNorm1d`, at the exact
    method (`Arm._target_embeddings`) a future freeze would have to touch to
    reintroduce the mechanism this repository decided against.

    This is the guard against reintroducing a freeze silently: if
    `_target_embeddings` ever again special-cased the target forward (e.g.
    switching a running-stats layer to `eval()` for it), this would go red.

    Reachable red: wrap the call in `_target_embeddings` with anything that
    puts `self.encoder`, or one of its running-stats submodules, into
    `eval()` mode for the duration of the target forward.
    """
    arm = _arm("G")
    bn = arm.encoder.bn

    arm.encoder.train()
    arm.encoder(torch.randn(20, 3, 8, 8))  # give the stats something of their own
    before_mean, before_var = bn.running_mean.clone(), bn.running_var.clone()

    target_bags = arm.target.take(torch.arange(config.BAGS_PER_STEP))
    arm._target_embeddings(target_bags)

    assert not torch.equal(bn.running_mean, before_mean)
    assert not torch.equal(bn.running_var, before_var)


@pytest.mark.parametrize("arm_id", ["E", "F", "G"])
def test_every_adapted_arms_full_training_step_updates_running_statistics_a_source_only_twin_does_not(
        bn_encoder, arm_id) -> None:
    """The guard above covers `G` alone, and only at `_target_embeddings`
    directly. This drives every adapted arm through a full `training_step` and
    compares its encoder's running statistics against a `B` twin built from
    identical initial weights (`_arm` resets the seed before building each one,
    over the same deterministic pools) and fed the identical batch and
    generator. `B` never lets a target image reach the encoder (Decision 2), so
    its running statistics move only from the source forward; if the adapted
    arm's landed in the same place, its own target forward would have to have
    been skipped or frozen somewhere between `training_step` and
    `_target_embeddings`.

    Every adapted arm now, with no exception carved out of the list: `GN` was
    the one arm excluded here -- its running statistics were expected to MATCH
    its source-only twin's, because `normalization: "sourceBatch"` kept its
    target forward off them entirely -- and it is retired.

    Reachable red, two ways: freeze the target forward for `arm_id` (wrap it
    in `eval()`) anywhere from `training_step` down to `_target_embeddings`
    for every arm but `G`, or freeze it specifically inside
    `_milcreda_term` around the `_target_embeddings` call it makes.
    """
    floor = _arm("B")
    adapted = _arm(arm_id)

    floor.encoder.train()
    adapted.encoder.train()
    assert torch.equal(floor.encoder.bn.running_mean, adapted.encoder.bn.running_mean), (
        "the twins did not start with identical running statistics"
    )

    x = floor.source.take(torch.arange(config.BAGS_PER_STEP))
    y = floor.source.labels[:config.BAGS_PER_STEP]
    gen_floor = torch.Generator().manual_seed(41)
    gen_adapted = torch.Generator().manual_seed(41)

    floor.training_step(x, y, 0.5, gen_floor)
    adapted.training_step(x, y, 0.5, gen_adapted)

    assert not torch.equal(floor.encoder.bn.running_mean,
                           adapted.encoder.bn.running_mean), (
        f"{arm_id}'s running statistics matched its source-only twin's "
        "after a full training_step: its target forward never touched them"
    )
    assert not torch.equal(floor.encoder.bn.running_var,
                           adapted.encoder.bn.running_var)


# `test_gns_full_training_step_matches_a_source_only_twins_running_statistics`,
# `test_gns_objective_survives_its_own_backward` and
# `test_gns_target_embeddings_scale_comparably_to_the_source` removed: the arm
# `GN` is retired from `config.ARMS`, and with it the whole `normalization:
# "sourceBatch"` axis -- `wiring.Arm._source_embeddings`,
# `_encode_with_frozen_stats` and `_batchnorm_modules`, which existed to serve
# only that arm, are retired too. All three tests asserted a claim about that
# axis and nothing else: that the frozen-stats target forward left the running
# statistics exactly where the source forward set them, that its objective
# survived its own backward (the rebind-versus-`copy_` defect the pilot found
# on its fifth run), and that its target embeddings stayed on a scale
# comparable to the source's. No declared arm normalizes its target forward
# with anything but the encoder's own training-mode statistics any more, so
# there is no such claim left to hold. What replaces them for every arm that
# remains is the test directly above -- every adapted arm's running statistics
# MUST move away from its source-only twin's -- which `GN` was the one
# exception carved out of.


# ------------------------------------------------- the supervised term of a bag

def test_a_bag_unit_arm_uses_the_normalized_supervised_term_instead(encoder) -> None:
    """Which supervised term an arm gets is read from the unit it declares, and a
    bag-unit arm gets Eq. (21) normalized by `B_src` rather than a cross-entropy
    over instances."""
    from MIL_CREDA.objective import source_loss

    arm = _arm("B")
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    step = arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    Z, _ = arm.bag_representations(arm.instance_embeddings(x), config.KERNEL_SIGMA)
    scores = F.softmax(arm.head(Z), dim=1)
    expected = source_loss(scores, F.one_hot(y, CLASSES).to(scores.dtype),
                           config.EPSILON)
    assert step["supervised"] == pytest.approx(float(expected.detach()), abs=1e-6)


# ------------------------------------------------------------ the learning rate

def test_the_learning_rate_is_fixed_and_not_decayed() -> None:
    """One fixed, declared rate for every arm, at every epoch: `config.LR`.

    `LR_ALPHA`/`LR_BETA` and the decay of CREDA's own `get_eta` are removed --
    the dynamic schedule is replaced by a single constant, not searched.

    Reachable red: reintroducing `p = epoch / epochs; return config.LR * (1 +
    LR_ALPHA * p) ** (-LR_BETA)` (or any other epoch-dependent formula) in
    `harness.learning_rate`.
    """
    assert not hasattr(config, "LR_ALPHA")
    assert not hasattr(config, "LR_BETA")
    for epoch in range(5):
        assert harness.learning_rate(epoch, 20) == config.LR, (
            f"the rate moved at epoch {epoch}: it is supposed to be fixed"
        )


# ------------------------------------------------- what the campaign hands the arms

def _fake_run_one(seen):
    def run_one(arm_id, transfer, seed, reduction, device, material, **kwargs):
        seen.append({"arm": arm_id, "transfer": harness.transfer_label(transfer),
                     "seed": seed, "source": id(material["source"]),
                     "target": id(material["target"])})
        return {
            "arm": arm_id, "transfer": harness.transfer_label(transfer), "seed": seed,
            "env": "test-env", "targetAccuracy": 0.5 + seed / 100,
            "sourceAccuracy": 0.5, "seconds": 0.01, "peakMiB": 1.0, "parameters": 4,
            "contribution": 0.1, "supervised": 0.2, "adaptationShare": 0.3,
            "curve": [], "epochs": [{"epoch": 0}], "state": None,
        }
    return run_one


def _fake_build(built):
    def build(code, cache, seed, noise=0.0):
        built.append((code, seed, noise))
        return SimpleNamespace(manifest={"code": code, "seed": seed, "noise": noise})
    return build


@pytest.fixture
def campana(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    monkeypatch.setattr(config, "PRODUCT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results" / "Benchmark")
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models")
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")
    seen, built = [], []
    monkeypatch.setattr(harness, "run_one", _fake_run_one(seen))
    monkeypatch.setattr(harness.bags, "build", _fake_build(built))
    return {"runs": seen, "built": built}


def _run_campaign(seeds: list[int], noise: float = 0.0) -> dict:
    return harness.campaign(
        harness.Reduction(seeds=seeds, epochs=1, labelNoise=noise,
                          ceilings={"creda": 1e-4, "milcreda": 1.0},
                          ceilingsByTransfer={}),
        torch.device("cpu"), progress=lambda *a: None)


def test_one_draw_of_the_material_is_shared_by_every_arm(campana) -> None:
    """Arms that saw differently corrupted material differ in the draw as well as
    in what they compute, and no rung between them is attributable.

    Shared by construction and not by agreement: the material is built once per
    (seed, domain) outside the arm loop, so every arm of a cell is handed the
    same object. The count is what says so -- three domains for one seed, however
    many arms ran -- and the identity is what makes the count mean it.

    Reachable red: move `bags.build` inside the arm loop, and the declared
    arms of a cell get one draw each that agree in distribution and in
    nothing else.
    """
    harness_result = _run_campaign([7], noise=config.NOISE_LEVELS[2])
    built = campana["built"]

    assert len(built) == len(config.DOMAINS), \
        f"the material was drawn {len(built)} times for one seed"
    assert {code for code, _, _ in built} == set(config.DOMAINS)
    assert {rate for _, _, rate in built} == {config.NOISE_LEVELS[2]}

    runs = campana["runs"]
    arms = {run["arm"] for run in runs}
    assert len(arms) == len(config.ARMS) == 4
    for transfer in {run["transfer"] for run in runs}:
        of_cell = [run for run in runs if run["transfer"] == transfer]
        assert len({run["source"] for run in of_cell}) == 1, \
            "two arms of one cell were handed different source material"
        assert len({run["target"] for run in of_cell}) == 1
    assert harness_result["reduction"]["labelNoise"] == config.NOISE_LEVELS[2]


def test_the_campaign_runs_every_one_of_the_six_transfers_and_withholds_none(
        campana) -> None:
    """The verdict is read over all six.

    Withholding the two the search measured bought nothing -- the roles are
    already disjoint by bag -- and cost a third of the units the paired reading
    rests on.

    Reachable red: run `VERDICT_TRANSFERS` minus the searched ones, and both the
    count and the missing labels land here.
    """
    summary = _run_campaign([0, 1, 2])
    declared = {harness.transfer_label(t) for t in config.VERDICT_TRANSFERS}

    assert len(declared) == 6
    assert {run["transfer"] for run in campana["runs"]} == declared
    assert set(summary["grid"]) == declared
    assert set(summary["perTransfer"]) == declared
    searched = {harness.transfer_label(t) for t in config.SEARCH_TRANSFERS}
    assert searched <= set(summary["grid"]), "a searched transfer was withheld"


def test_three_checkpoints_are_kept_per_arm_per_cell_for_every_arm(campana) -> None:
    """Three per arm per cell, so the top three can be selected after the run
    instead of being guessed before it -- and for EVERY arm, because which arms
    the figures need is only known once the campaign has ranked them.

    Reachable red: drop any arm from `CHECKPOINTS`, or keep two instead of three.
    """
    assert set(config.CHECKPOINTS) == {arm["id"] for arm in config.ARMS}
    assert set(config.CHECKPOINTS.values()) == {3}

    summary = _run_campaign([0, 1, 2, 3])
    declared = {harness.transfer_label(t) for t in config.VERDICT_TRANSFERS}
    assert len(summary["checkpoints"]) == len(config.ARMS) * len(declared)
    for key, kept in summary["checkpoints"].items():
        assert len(kept) == 3, f"{key} kept {len(kept)} of four repetitions"

    written = list((config.MODELS).glob("*.manifest.json"))
    assert len(written) == len(config.ARMS) * len(declared) * 3
    one = json.loads(written[0].read_text(encoding="utf-8"))
    assert one["reduction"]["seeds"] == [0, 1, 2, 3]


def _run_one_no_timing(arm_id, transfer, seed, reduction, device, material, **kwargs):
    """`run_one`'s shape, with none of the timing/memory fields `harness.run_one`
    no longer returns -- time and memory (`seconds`/`peakMiB`) are removed from
    this comparison entirely.
    """
    label = harness.transfer_label(transfer)
    return {
        "arm": arm_id, "transfer": label, "seed": seed,
        "env": "test-env", "targetAccuracy": 0.5 + seed / 100,
        "sourceAccuracy": 0.5, "parameters": 4, "contribution": 0.1,
        "supervised": 0.2, "adaptationShare": 0.3,
        "curve": [], "epochs": [{"epoch": 0}], "state": None,
    }


def test_progress_prints_one_line_per_cell_and_names_no_timing(
        campana, monkeypatch) -> None:
    """One line per (seed, transfer), not one per run, and it names no timing.

    Six transfers over thirty seeds is 180 lines; the same call inside the arm
    loop prints 1800, and 1800 lines of a run measured in hours is a report
    nobody reads rather than the sign of life it is kept for.

    It used to name that cell's slowest arm and how long the cell took;
    both are gone along with `seconds`/`peakMiB` themselves, which
    `harness.run_one` no longer returns at all.

    Reachable red: move `progress` back inside the arm loop and the count
    lands on arms x transfers x seeds, or reintroduce a reference to
    `run["seconds"]` inside `campaign`'s progress line -- which would raise
    `KeyError` against this fixture's own run dict, which carries no such key.
    """
    arm_ids = [arm["id"] for arm in config.ARMS]
    monkeypatch.setattr(harness, "run_one", _run_one_no_timing)
    lines: list[str] = []
    seeds = [0, 1, 2]
    harness.campaign(
        harness.Reduction(seeds=seeds, epochs=1,
                          ceilings={"creda": 1e-4, "milcreda": 1.0},
                          ceilingsByTransfer={}),
        torch.device("cpu"), progress=lines.append)

    labels = [harness.transfer_label(t) for t in config.VERDICT_TRANSFERS]
    cells = [line for line in lines if " arms  " in line]
    assert len(cells) == len(seeds) * len(labels), (
        f"{len(cells)} progress lines for {len(seeds) * len(labels)} cells of "
        f"{len(arm_ids)} arms -- one per run would be "
        f"{len(seeds) * len(labels) * len(arm_ids)}")
    for line in cells:
        assert "slowest" not in line and "cell " not in line, (
            f"a progress line still names timing: {line!r}"
        )


# ------------------------------------------------------- what funds the third role

def test_the_selection_role_is_funded_by_new_material_and_takes_nothing(
) -> None:
    """Twelve bags per class instead of ten, and the two extra fund the search's
    own role outright: 64 / 20 / 36 where it used to be 64 / -- / 36.

    The relation is the claim. Asserting `VALID_BAGS == 20` beside
    `BAGS_PER_CLASS == 12` would pass just as well if the twenty had been taken
    out of training, which is the thing that was not done.

    Reachable red: raise `VALID_BAGS` without raising `BAGS_PER_CLASS`, or fund
    the role by lowering `TRAIN_BAGS` or `EVAL_BAGS`.
    """
    assert config.BAGS_PER_CLASS == 12
    grew_by = (config.BAGS_PER_CLASS - 10) * config.CLASSES
    assert grew_by == config.VALID_BAGS == 20
    # and neither of the other two paid for it
    assert config.TRAIN_BAGS == 64
    assert config.EVAL_BAGS == 36
    assert config.TRAIN_BAGS + config.EVAL_BAGS == 10 * config.CLASSES
    # the resolution the campaign was sized for is the evaluation role's, untouched
    assert 100 / config.EVAL_BAGS == pytest.approx(2.78, abs=0.01)


# ----------------------------------------------- the objective, assembled or written

def _unit_branch():
    """The two halves of `training_step`, read from the source tree.

    Which branch an arm takes is asserted by running it; which functions a branch
    is ALLOWED to call is a property of the code, and the only place it can be
    read is the code.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(wiring.__file__).read_text(encoding="utf-8"))
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "training_step")
    branch = next(node for node in step.body
                  if isinstance(node, ast.If) and "unit" in ast.dump(node.test))

    def called(nodes) -> set[str]:
        names = set()
        for statement in nodes:
            for node in ast.walk(statement):
                if isinstance(node, ast.Call):
                    function = node.func
                    names.add(function.attr if isinstance(function, ast.Attribute)
                              else getattr(function, "id", ""))
        return names

    return called(branch.body), called(branch.orelse)


def test_the_bag_unit_arms_assemble_the_objective_and_never_write_a_term_inline(
        encoder) -> None:
    """Eq. (21) and Eq. (39) as the revision states them, called and not restated.

    A supervised term written inline in the benchmark is a second copy of an
    equation the proposal already owns: it stops moving when the proposal moves,
    and nothing tells anyone. So the bag branch calls `source_loss` for its
    supervised term and `total_objective` for the sum, and the value each returns
    is asserted against the functions themselves.

    The inline cross-entropy at the other side of the branch is prior work's own,
    and it belongs to the instance unit alone. That is asserted too, because "no
    term written inline" is only meaningful beside the one place a term IS
    written inline on purpose.

    Reachable red: write Eq. (21) or Eq. (39) out by hand in the bag branch, or
    let the bag branch fall through to the instance one's cross-entropy.
    """
    from MIL_CREDA.objective import source_loss, total_objective

    arm = _arm("G")
    assert config.ARMS_BY_ID["G"]["unit"] == "bag"
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    ramp = 0.5
    step = arm.training_step(x, y, ramp, torch.Generator().manual_seed(3))

    embeddings = arm.instance_embeddings(x)
    Z, _ = arm.bag_representations(embeddings, config.KERNEL_SIGMA)
    scores = F.softmax(arm.head(Z), dim=1)
    supervised = source_loss(scores, F.one_hot(y, CLASSES).to(scores.dtype),
                             config.EPSILON)
    assert step["supervised"] == pytest.approx(float(supervised.detach()), abs=1e-6)

    target = arm.target.take(arm._draw_target(torch.Generator().manual_seed(3)))
    global_term, local_term = arm._milcreda_term(embeddings, y, target)
    assembled = total_objective(supervised, global_term, local_term, ramp, ramp)
    assert float(step["loss"].detach()) == pytest.approx(
        float(assembled.detach()), abs=1e-6)

    # and both equations are called rather than restated
    bag, instance = _unit_branch()
    assert {"source_loss", "one_hot", "softmax"} <= bag
    assert "cross_entropy" not in bag, "a supervised term written inline"
    assert "cross_entropy" in instance, "prior work's own term left the branch"
    assert "source_loss" not in instance, "Eq. (21) applied to prior work"
    assert "total_objective" in _milcreda_calls()


def _milcreda_calls() -> set[str]:
    """Everything the MIL-CREDA arm of `training_step` calls to build its total."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(wiring.__file__).read_text(encoding="utf-8"))
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "training_step")
    branch = next(node for node in ast.walk(step)
                  if isinstance(node, ast.If) and "milcreda" in ast.dump(node.test))
    return {node.func.attr if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", "")
            for statement in branch.body for node in ast.walk(statement)
            if isinstance(node, ast.Call)}


class _SpyF:
    """`torch.nn.functional` with one call recorded. Everything else passes through."""

    def __init__(self, seen):
        self._seen = seen

    def __getattr__(self, name):
        return getattr(F, name)

    def cross_entropy(self, scores, target, *args, **kwargs):
        self._seen.append(target)
        return F.cross_entropy(scores, target, *args, **kwargs)


def test_the_bag_label_never_leaves_its_bag(encoder, monkeypatch) -> None:
    """A contaminated instance arrives as a witness inside its bag and never as a
    wrong label -- asserted as the mechanism and not as its consequence.

    A bag-unit arm never expands the bag's label to its instances: the label stays
    at the subject, so a contaminant is one of thirty witnesses the confidence can
    downweight rather than a supervised target that is simply wrong.

    What is asserted is the absence of the broadcast itself -- how many supervised
    targets one bag's label becomes. Whether a wrong label and a downweightable
    witness differ in kind is the reading the report makes of this, and it stays
    an argument rather than becoming an assertion.

    The other half of this claim -- the instance unit broadcasting the label to
    all thirty -- no longer has a declared arm to run on. `wiring` still carries
    that branch, and `test_the_bag_unit_arms_assemble_the_objective_and_never_write
    _a_term_inline` still reads it out of the source tree; what is gone is the
    executed comparison between the two units.

    Reachable red: expand the label in the bag branch as well.
    """
    per_instance, per_bag = [], []
    monkeypatch.setattr(wiring, "F", _SpyF(per_instance))

    real_source_loss = wiring.source_loss

    def _spy_source_loss(scores, onehot, epsilon):
        per_bag.append(onehot)
        return real_source_loss(scores, onehot, epsilon)

    monkeypatch.setattr(wiring, "source_loss", _spy_source_loss)

    B = config.BAGS_PER_STEP
    m = config.INSTANCES_PER_BAG

    bag_arm = _arm("G")
    assert config.ARMS_BY_ID["G"]["unit"] == "bag"
    x = bag_arm.source.take(torch.arange(B))
    y = bag_arm.source.labels[:B]
    bag_arm.training_step(x, y, 0.5, torch.Generator().manual_seed(3))

    assert per_bag, "the bag unit computed no supervised term"
    # one target per bag, unexpanded: the label never leaves the subject
    assert per_bag[0].shape == (B, CLASSES)
    assert torch.equal(per_bag[0].argmax(dim=1), y)
    assert per_instance == [], "the bag unit broadcast the label anyway"

    # one supervised target against the thirty witnesses inside it, and the
    # factor between them is the bag's own cardinality
    assert m == config.INSTANCES_PER_BAG > 1


# `test_the_arbitrary_selection_draws_from_a_generator_of_its_own` (SA's own
# positions, once) is removed along with the selection arms: no declared arm
# has a `positions` buffer at all any more, since `spec["selection"]` is
# `None` everywhere and `Arm.__init__` only builds that buffer for
# `"regular"`/`"arbitrary"` selection.
