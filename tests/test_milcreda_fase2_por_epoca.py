"""MIL-CREDA's phase-2 epoch: one pass of the training role.

The tutorial's phase 2 reads its criterion once per epoch, and
`toy_run.steps_of_one_epoch` is how many steps an epoch has. That number is
measured here off `harness.balanced_batches` itself rather than derived twice,
so the join between the declared length and the sampler is crossed. That the
tutorial reads the criterion per epoch and runs every epoch is pinned by
`tests/test_milcreda_tutorial.py`, which executes the notebook.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from MIL_CREDA_Benchmark import bags, toy_run


@pytest.fixture(scope="module")
def tiny() -> toy_run.Scale:
    """`PILOT` with its lengths and its material lowered, so a draw costs
    seconds: 6 instances per bag over 810 points."""
    return dataclasses.replace(
        toy_run.PILOT, name="test", epochs_1=2, patience_1=1,
        epochs_2=toy_run.MIN_EPOCHS_2 + 1, points_per_domain=810,
        instances_per_bag=6)


@pytest.fixture(scope="module")
def familia(tiny) -> object:
    return toy_run.build_family("lunas", 0, tiny)


def test_the_epoch_is_the_pass_balanced_batches_actually_draws(familia, tiny):
    """`steps_of_one_epoch` is the step at which every training bag has come
    out exactly once, and that is MEASURED off the sampler rather than derived
    twice.

    The join this crosses is the only one worth crossing here: the number is
    computed from the role's class counts in one place and the draw is
    performed by `harness.balanced_batches` in another, and asserting each
    against an arithmetic written in this file would verify both halves and
    never the connection.
    """
    from MIL_CREDA_Benchmark import harness

    material = toy_run.material_for(familia, 0, tiny, "mlp")
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, _, _ = bags.roles(material["source"])
    declared = toy_run.steps_of_one_epoch(source_train.targets)

    generator = torch.Generator().manual_seed(0)
    drawn: set[int] = set()
    reached = None
    for step, batch in enumerate(
            harness.balanced_batches(source_train.targets, declared * 3,
                                     generator)):
        drawn.update(batch)
        if reached is None and len(drawn) == len(source_train.targets):
            reached = step + 1
    assert reached == declared, (reached, declared, len(source_train.targets))
    # The pole that the reading is about the PASS and not about the role's size:
    # one bag of every class leaves the step count at the PER-CLASS count, not
    # at the total, so a sampler drawing one bag per step would answer 48.
    assert declared < len(source_train.targets), (
        declared, len(source_train.targets))


def test_the_epoch_length_refuses_when_the_classes_do_not_agree():
    """A role whose classes hold different numbers of bags has NO step at which
    every bag has come out exactly once, and this refuses rather than returning
    the first or the largest.

    CREDA's `pasos_de_una_epoca` refuses the same way when its two domains
    disagree in blocks, and for the same reason: wrapping the short queue would
    leave the property the whole change exists for -- every bag out exactly once
    per epoch -- false with nothing failing.

    The refusal is REACHABLE and that is what makes it a guard: `balanced_batches`
    refills each class's queue independently, so an unequal role runs happily
    and silently forever.
    """
    balanced = [0] * 4 + [1] * 4 + [2] * 4
    assert toy_run.steps_of_one_epoch(balanced) == 4
    with pytest.raises(ValueError, match="do not agree"):
        toy_run.steps_of_one_epoch([0] * 4 + [1] * 3 + [2] * 4)
