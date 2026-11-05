"""Run-3 conditions of the proof-of-concept battery (`poc`, `harness`).

k source bags per class per phase-2 step, the tripled test role at the same
density, and the artifact contamination mode. Each is off by default and off
is the behaviour as it was.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from MIL_CREDA_Benchmark import config, harness, poc, toy_run

#: `harness.balanced_batches` over a fixed shuffled role and generator, taken
#: before the `per_class` option existed.
GOLDEN_BATCHES = "71b9134432166508fb6d7fb1a533fb4f8bfb348e4f5c217357fe4f6fb0a747b0"


def _targets():
    targets = [c for c in range(3) for _ in range(16)]
    random.Random(0).shuffle(targets)
    return targets


def test_one_bag_per_class_is_the_draw_as_it_was():
    for kwargs in ({}, {"per_class": 1}):
        seq = list(harness.balanced_batches(_targets(), 200,
                                            torch.Generator().manual_seed(7), **kwargs))
        assert hashlib.sha256(json.dumps(seq).encode()).hexdigest() == GOLDEN_BATCHES


def test_k_bags_per_class_per_step_without_repeats_inside_a_step():
    targets = _targets()
    for step in harness.balanced_batches(targets, 40, torch.Generator().manual_seed(7),
                                         per_class=4):
        labels = [targets[p] for p in step]
        assert [labels.count(c) for c in range(3)] == [4, 4, 4]
        assert len(set(step)) == len(step)


SMALL = dataclasses.replace(poc.CONDITIONS_SCALE, name="cond-test", epochs_1=1,
                            epochs_2=toy_run.MIN_EPOCHS_2, patience_1=1)


def test_the_conditions_scale_triples_the_test_role_at_the_same_density():
    split = poc.CONDITIONS_SCALE.realized_split()["bagsPerClass"]
    base = toy_run.FULL.realized_split()["bagsPerClass"]
    assert split == {"train": base["train"], "valid": base["valid"], "test": 3 * base["test"]}
    density = toy_run.FULL.points_per_domain / (3 * toy_run.FULL.bags_per_class)
    assert poc.CONDITIONS_SCALE.points_per_domain / (3 * poc.CONDITIONS_SCALE.bags_per_class) == density


def test_the_conditions_scale_funds_every_seed():
    for seed in range(10, 40):
        f = toy_run.build_family(poc.FAMILY, seed, poc.CONDITIONS_SCALE,
                                 **poc.RUN3.draw_options())
        assert toy_run._instances_per_bag(f, poc.CONDITIONS_SCALE) == 30


def test_the_off_conditions_draw_nothing_new():
    assert poc.OFF.draw_options() == {"per_pair_shift": 0.0, "rotation_degrees": None,
                                      "subject_field": 0.0}
    assert poc.OFF.source_bags_per_class == 1 and poc.OFF.contamination == "donor"


@pytest.fixture(scope="module")
def familia():
    return toy_run.build_family(poc.FAMILY, 3, SMALL, **poc.RUN3.draw_options())


def test_k_bags_per_class_reach_the_local_term_as_s_c_of_size_k(familia):
    material = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    hyper = {**poc.poc_hyper(), "kernelSigma": 0.5}
    device = torch.device("cpu")
    seen = {}
    with toy_run.material_bound(familia, SMALL, poc.BACKBONE, hyper):
        model = poc.build_poc_arm("TGL", material, SMALL, hyper, device, 7)
        toy_run.copy_source_to_target(model)
        original = model._milcreda_term

        def spy(H_s, labels, target_bags):
            seen["labels"] = labels.tolist()
            seen["targets"] = int(target_bags.shape[0])
            return original(H_s, labels, target_bags)

        model._milcreda_term = spy
        before = config.BAGS_PER_STEP
        source_train, _, _ = __import__("MIL_CREDA_Benchmark.bags", fromlist=["roles"]).roles(material["source"])
        record = poc.train_phase_two(model, source_train, SMALL, hyper,
                                     torch.Generator().manual_seed(1), device,
                                     source_bags_per_class=4)
        assert config.BAGS_PER_STEP == before
    assert [seen["labels"].count(c) for c in range(3)] == [4, 4, 4]
    assert seen["targets"] == 12
    # Same epochs, so the same exposure: 16 bags per class / 4 per step.
    assert record["stepsPerEpoch"] == 4
    assert record["steps"] == 4 * SMALL.epochs_2


def test_artifact_mode_is_off_manifold_in_both_domains_on_the_seams_slots(familia):
    clean = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    _, donor_masks = poc.contaminated_material(familia, 3, SMALL, poc.BACKBONE, 0.2)
    material, masks = poc.artifact_material(familia, 3, SMALL, poc.BACKBONE, 0.2,
                                            poc.ARTIFACT_SCALE)
    for side in ("source", "target"):
        b, c, mask = material[side], clean[side], masks[side]
        assert torch.equal(mask, donor_masks[side])
        assert torch.equal(b.labels, c.labels)
        X, X0 = b.images[b.members], c.images[c.members]
        assert torch.equal(X[~mask], X0[~mask])
        clean_points = X0[~mask].double()
        art = X[mask].double()
        sample = clean_points[torch.randperm(len(clean_points), generator=torch.Generator().manual_seed(0))[:2000]]
        nn_art = torch.cdist(art, clean_points).min(dim=1).values.median()
        d = torch.cdist(sample, clean_points)
        d[d == 0] = float("inf")
        nn_clean = d.min(dim=1).values.median()
        assert nn_art > 5 * nn_clean
    zero, zero_masks = poc.artifact_material(familia, 3, SMALL, poc.BACKBONE, 0.0,
                                             poc.ARTIFACT_SCALE)
    assert torch.equal(zero["target"].images, clean["target"].images)
    assert not zero_masks["target"].any()


def test_material_under_routes_each_mode(familia):
    clean, masks = poc.material_under(familia, 3, SMALL, poc.OFF, 0.0)
    assert not masks["source"].any()
    donor, _ = poc.material_under(familia, 3, SMALL, poc.OFF, 0.2)
    art, _ = poc.material_under(familia, 3, SMALL,
                                dataclasses.replace(poc.OFF, contamination="artifact"), 0.2)
    assert not torch.equal(donor["source"].images, art["source"].images)
    with pytest.raises(ValueError):
        poc.material_under(familia, 3, SMALL,
                           dataclasses.replace(poc.OFF, contamination="nope"), 0.2)


@pytest.mark.parametrize("runner, args", [("run_ladder_seed", (5,)),
                                          ("run_contamination_seed", (5, 0.2))])
def test_the_runners_hand_the_draw_options_of_their_conditions(monkeypatch, runner, args):
    seen = []

    class Stop(Exception):
        pass

    def fake(name, seed, scale, per_pair_shift=0.0, rotation_degrees=None, subject_field=0.0):
        seen.append((per_pair_shift, rotation_degrees, subject_field, scale.name))
        raise Stop

    monkeypatch.setattr(toy_run, "build_family", fake)
    with pytest.raises(Stop):
        getattr(poc, runner)(*args, conditions=poc.RUN3, scale=poc.CONDITIONS_SCALE)
    assert seen == [(0.0, poc.RUN3.rotation_degrees, poc.RUN3.subject_field,
                     poc.CONDITIONS_SCALE.name)]
