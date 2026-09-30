"""The source-to-target transfer, and what it must and must not touch.

Adopted 2026-09-29 as a TRAINING PROCEDURE, not as mathematics: Eq. (13)'s two
encoders stay distinct tensors that share none, and they diverge again after
every copy. The decision and its measurements live in `AGREED.md`; what lives
here is the contract the code must keep.

The encoder is stubbed with one that carries running statistics of its own,
because the sharpest claim below is about what the transfer leaves alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from MIL_CREDA_Benchmark import config, harness, wiring

CLASSES = config.CLASSES


class _BNEncoder(nn.Module):
    """A stub with an `output_dim` and a running-stats layer to observe."""

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
    bags = config.BAGS_PER_STEP + 2
    images = torch.randn(bags * config.INSTANCES_PER_BAG, 3, 8, 8, generator=generator)
    members = torch.arange(images.shape[0]).reshape(bags, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags) % CLASSES
    return wiring.Pool(images, members, labels)


def _arm(arm_id: str):
    torch.manual_seed(11)
    return wiring.build(arm_id, CLASSES, _pool(1), _pool(2))


def _adapted_arm_ids() -> list[str]:
    return [a["id"] for a in config.ARMS if a["adaptation"] is not None]


def _floor_arm_ids() -> list[str]:
    return [a["id"] for a in config.ARMS if a["adaptation"] is None]


# ------------------------------------------------------ the interval is declared

def test_the_transfer_interval_is_a_declared_positive_integer() -> None:
    """`N` is declared rather than searched, and the reason is measured: from
    every step to every seventieth the outcome was the same, so the mechanism's
    only real degree of freedom is that the re-anchoring exists at all. A
    searched knob would spend the search on a dimension that does not move.
    """
    steps = config.SOURCE_TO_TARGET_SYNC_STEPS
    assert isinstance(steps, int) and not isinstance(steps, bool), (
        "the interval is a count of optimizer steps, so it is an int")
    assert steps >= 1, (
        "an interval below one is not an interval; zero would mean never and "
        "that is the absence of the mechanism, not a value of it")


# --------------------------------------------------- what the transfer moves

@pytest.mark.parametrize("arm_id", _adapted_arm_ids())
def test_the_transfer_makes_every_parameter_of_theta_t_equal_theta_s(
        bn_encoder, arm_id: str) -> None:
    """After the transfer, every parameter tensor of the target encoder equals
    the source encoder's. This is the whole mechanism: theta_t is put back into
    the coordinate system the shared head -- trained on source labels alone --
    knows how to read.
    """
    arm = _arm(arm_id)
    with torch.no_grad():
        for p in arm.target_encoder.parameters():
            p.add_(1.0)          # drive them apart, so the copy has work to do
    source = dict(arm.encoder.named_parameters())
    moved = [n for n, p in arm.target_encoder.named_parameters()
             if not torch.equal(p, source[n])]
    assert moved, (
        "the fixture failed to separate the encoders, so a green result here "
        "would prove nothing about the copy")

    arm.sync_target_from_source()

    source = dict(arm.encoder.named_parameters())
    for name, p in arm.target_encoder.named_parameters():
        assert torch.equal(p, source[name]), (
            f"{name} did not follow the source encoder")


@pytest.mark.parametrize("arm_id", _adapted_arm_ids())
def test_the_transfer_leaves_the_target_running_statistics_alone(
        bn_encoder, arm_id: str) -> None:
    """Buffers do not travel. Measured: copying the normalization statistics
    too was WORSE at the shortest interval (0.7222 against 1.0000), and the
    statistics are what each domain's own material accumulates -- they are a
    reading of the target's material, not a parameter of the map.
    """
    arm = _arm(arm_id)
    with torch.no_grad():
        arm.target_encoder.bn.running_mean.add_(3.0)
        arm.target_encoder.bn.running_var.add_(5.0)
    before = {n: v.detach().clone()
              for n, v in arm.target_encoder.named_buffers()}
    assert not torch.equal(before["bn.running_mean"],
                           arm.encoder.bn.running_mean), (
        "the fixture failed to separate the buffers")

    arm.sync_target_from_source()

    for name, v in arm.target_encoder.named_buffers():
        assert torch.equal(v, before[name]), (
            f"the buffer {name} travelled with the parameters")


@pytest.mark.parametrize("arm_id", _floor_arm_ids())
def test_the_transfer_is_a_no_op_on_a_floor_that_has_no_target_encoder(
        bn_encoder, arm_id: str) -> None:
    """The floor carries one encoder, so there is nothing to transfer into.
    It must not raise: `run_one` calls this for every arm it trains, and a
    floor that refused would make the caller carry the arm's own shape.
    """
    arm = _arm(arm_id)
    assert arm.target_encoder is None, (
        "this test is about the floor and this arm has a target encoder")
    before = {n: p.detach().clone() for n, p in arm.named_parameters()}
    arm.sync_target_from_source()
    for name, p in arm.named_parameters():
        assert torch.equal(p, before[name]), (
            f"{name} moved on an arm with nothing to transfer into")


@pytest.mark.parametrize("arm_id", _adapted_arm_ids())
def test_the_transfer_carries_no_gradient(bn_encoder, arm_id: str) -> None:
    """An assignment, never a term. If the copy built a graph, the adaptation
    term's gradient would flow into theta_s through theta_t and the supervised
    half would be optimizing something nobody declared.
    """
    arm = _arm(arm_id)
    arm.sync_target_from_source()
    for name, p in arm.target_encoder.named_parameters():
        assert p.grad_fn is None, (
            f"{name} came out of the transfer carrying a graph")
        assert p.is_leaf, f"{name} stopped being a leaf of the graph"


# -------------------------------------------- and the loop actually calls it


class _Encoder(nn.Module):
    """A stub with an `output_dim`, no running statistics: `run_one` trains on
    one bag per step here and a `BatchNorm1d` over a single row is undefined.
    """

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


def _bagset(domain: str, seed: int):
    """`tests/test_target_degeneracy_diagnostic.py`'s own fixture shape, reused
    rather than reinvented: `total_correspondence` needs a source bag in every
    class the target pseudo-labels reach.
    """
    from MIL_CREDA_Benchmark import bags

    generator = torch.Generator().manual_seed(seed)
    bagcount, per_bag = config.CLASSES + 10, 3
    images = torch.randn(bagcount * per_bag, 3, 8, 8, generator=generator)
    members = torch.arange(bagcount * per_bag).reshape(bagcount, per_bag)
    labels = torch.arange(bagcount) % config.CLASSES
    return bags.BagSet(
        domain=domain, images=images, members=members, labels=labels,
        train_idx=torch.arange(0, config.CLASSES),
        valid_idx=torch.arange(config.CLASSES, config.CLASSES + 5),
        eval_idx=torch.arange(config.CLASSES + 5, config.CLASSES + 10),
        manifest={})


def _count_transfers(monkeypatch, arm_id: str, interval: int, epochs: int) -> int:
    """Transfers `run_one` actually performed, counted at the method itself."""
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)
    monkeypatch.setattr(config, "SOURCE_TO_TARGET_SYNC_STEPS", interval)
    seen = {"n": 0}
    original = wiring.Arm.sync_target_from_source

    def counting(self):
        seen["n"] += 1
        return original(self)

    monkeypatch.setattr(wiring.Arm, "sync_target_from_source", counting)
    torch.manual_seed(11)
    reduction = harness.Reduction(epochs=epochs, seeds=[0])
    material = {"source": _bagset("S", 1), "target": _bagset("M", 2)}
    harness.run_one(arm_id, ("S", "M"), 0, reduction, torch.device("cpu"),
                    material, role="valid")
    return seen["n"]


def _steps_per_epoch() -> int:
    """The loop's own arithmetic, read rather than assumed.

    `run_one` derives this from `TRAIN_BAGS`/`BAGS_PER_STEP` and NOT from how
    many training bags the fixture happens to carry, so a test that counted the
    fixture's bags would be off by a factor and would look like a defect in the
    interval.
    """
    return -(-config.TRAIN_BAGS // config.BAGS_PER_STEP)


@pytest.mark.parametrize("interval", [1, 2, 3])
def test_run_one_transfers_on_the_interval_config_declares(
        monkeypatch, interval: int) -> None:
    """Three intervals, three different counts, from a real `run_one`.

    One interval alone would pass over a loop that transferred once, or on a
    hardcoded number of its own; only counts that MOVE with the interval prove
    the loop reads `config.SOURCE_TO_TARGET_SYNC_STEPS`. The expected count is
    derived from the loop's own step arithmetic, never from a number typed here.
    """
    epochs = 2
    total = _steps_per_epoch() * epochs
    expected = total // interval
    assert expected >= 2, (
        "this cell is too small for the interval to be distinguishable from "
        "'transferred once', so a green result would prove nothing")
    assert _count_transfers(monkeypatch, "G", interval, epochs) == expected


def test_the_loop_asks_every_arm_and_the_floor_is_what_declines(
        monkeypatch) -> None:
    """The loop transfers uniformly; the FLOOR is what makes it a no-op.

    This is the design and not an accident: whether there is anything to
    transmit into is the arm's own shape, so `run_one` does not branch on it.
    What must hold is that a floor run reaches the same call and moves nothing
    -- the moving-nothing half is `test_the_transfer_is_a_no_op_on_a_floor...`
    above, at the arm; this half is that the floor run does not refuse.
    """
    epochs = 2
    total = _steps_per_epoch() * epochs
    assert _count_transfers(monkeypatch, "B", 1, epochs) == total, (
        "the floor must reach the same uniform call the adapted arms do")
