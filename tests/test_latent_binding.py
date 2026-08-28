"""The checkpoints and the record they are read beside describe one run, or neither.

Phase two globs `config.MODELS` and measures whatever is there. A directory full
of valid checkpoints from an earlier, smaller run loads, measures and renders
exactly like the right ones — under the current record's stamp, with every other
check green. That is what these tests refuse.
"""

from __future__ import annotations

import pytest

from MIL_CREDA_Benchmark import latent


def _checkpoint(seed, **reduction):
    base = {"epochs": 20, "ceilings": {"creda": 1e-4}, "backbone": "resnet18"}
    base.update(reduction)
    return {"seed": seed, "manifest": f"/models/G_M-U_seed{seed}.manifest.json",
            "reduction": base}


def _summary(seeds, **reduction):
    base = {"epochs": 20, "ceilings": {"creda": 1e-4}, "seeds": list(seeds)}
    base.update(reduction)
    return {"reduction": base}


def test_checkpoints_from_the_run_the_record_describes_pass_through():
    found = [_checkpoint(3), _checkpoint(17)]
    assert latent.bound(found, _summary(range(30))) == found


def test_an_earlier_shorter_run_is_refused_under_the_current_stamp():
    """The failure this exists for: seed 0 is a seed the campaign ran too.

    Only `epochs` tells the two apart, which is why the comparison is over every
    shared field and not over the seed alone.
    """
    pilot = [_checkpoint(0, epochs=3)]
    with pytest.raises(latent.CheckpointsDisagree) as raised:
        latent.bound(pilot, _summary(range(30)))
    assert "epochs" in str(raised.value)


def test_a_seed_the_record_never_ran_is_refused():
    with pytest.raises(latent.CheckpointsDisagree) as raised:
        latent.bound([_checkpoint(41)], _summary(range(30)))
    assert "seed" in str(raised.value)


def test_the_shard_seed_list_is_not_compared_and_a_distributed_run_still_passes():
    """A manifest carries its shard's three seeds; the record carries all thirty.

    Requiring equality there would refuse every distributed campaign — which is
    every campaign this repository actually runs.
    """
    shard = _checkpoint(17, seeds=[15, 16, 17])
    assert latent.bound([shard], _summary(range(30))) == [shard]


def test_a_record_with_nothing_in_common_refuses_rather_than_passing_quietly():
    """An unprovable precondition is not a satisfied one.

    With no shared field the loop compares nothing and finds nothing, which reads
    exactly like agreement — in the one state where nothing at all is known.
    """
    with pytest.raises(latent.CheckpointsDisagree) as raised:
        latent.bound([{"seed": 3, "manifest": "m", "reduction": {"onlyMine": 1}}],
                     {"reduction": {"onlyTheirs": 2}})
    assert "cannot be established" in str(raised.value)


def test_no_checkpoints_at_all_refuses():
    with pytest.raises(latent.CheckpointsDisagree):
        latent.bound([], _summary(range(30)))


def test_the_disagreement_names_the_checkpoint_the_field_and_both_values():
    """A refusal that says only "they disagree" sends the reader to find out."""
    clashes = latent.disagreements([_checkpoint(0, epochs=3)], _summary(range(30)))
    assert clashes == [{"checkpoint": "G_M-U_seed0.manifest.json", "field": "epochs",
                        "checkpoint_says": 3, "record_says": 20}]
