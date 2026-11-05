"""The proof-of-concept battery's own functions (`MIL_CREDA_Benchmark.poc`).

Each test pins one function the hypothesis notebook relies on: the declared
contamination injection on the toy path, the ladder arms and the
no-consensus arm, the attention-mass contrast, the partner separation, and
the paired statistics the verdicts are read from.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from MIL_CREDA_Benchmark import config, poc, toy_run


SMALL = dataclasses.replace(toy_run.FULL, name="poc-test", epochs_1=1,
                            epochs_2=toy_run.MIN_EPOCHS_2, patience_1=1)


@pytest.fixture(scope="module")
def familia():
    return toy_run.build_family(poc.FAMILY, 3, SMALL)


# --------------------------------------------------------- contamination

def test_the_noise_pairing_is_bent_for_three_classes_and_restored():
    before = list(config.NOISE_PAIRING)
    with poc.toy_noise_pairing():
        assert config.NOISE_PAIRING == poc.NOISE_PAIRING_TOY
        assert all(d != c for c, d in enumerate(config.NOISE_PAIRING))
    assert config.NOISE_PAIRING == before


def test_rate_zero_material_is_the_clean_material(familia):
    clean = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    material, masks = poc.contaminated_material(familia, 3, SMALL, poc.BACKBONE, 0.0)
    for side in ("source", "target"):
        assert torch.equal(material[side].images, clean[side].images)
        assert not masks[side].any()


def test_contamination_replaces_declared_slots_with_the_donor_class(familia):
    rate = 0.2
    clean = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    material, masks = poc.contaminated_material(familia, 3, SMALL, poc.BACKBONE, rate)
    per_bag = round(rate * SMALL.instances_per_bag)
    for side in ("source", "target"):
        bagset, mask = material[side], masks[side]
        assert mask.shape == tuple(bagset.members.shape)
        noised = torch.cat([bagset.train_idx, bagset.eval_idx])
        assert (mask[noised].sum(dim=1) == per_bag).all()
        assert not mask[bagset.valid_idx].any()
        # The label stays; only the evidence changes.
        assert torch.equal(bagset.labels, clean[side].labels)
        # Every contaminated slot holds a point of the declared donor class.
        dataset_labels = (familia.ys if side == "source" else familia.yt)
        for record in bagset.manifest["contamination"]["bags"]:
            donor = poc.NOISE_PAIRING_TOY[record["label"]]
            for image in record["imageIndices"]:
                assert int(dataset_labels[image]) == donor
        # Clean slots are untouched instances of the clean draw.
        same = (bagset.images[bagset.members] == clean[side].images[clean[side].members]
                ).all(dim=-1)
        assert same[~mask].all()


# ------------------------------------------------------------------ arms

def test_the_ladder_arms_differ_from_tgl_by_one_flag_each():
    tgl = poc.POC_ARMS["TGL"]
    assert tgl == toy_run.TOY_ARMS_BY_ID["TGL"]
    for arm, key in (("TG", "local"), ("TL", "global")):
        spec = poc.POC_ARMS[arm]
        changed = {k for k in set(spec) | set(tgl)
                   if spec.get(k) != tgl.get(k) and k not in ("id", "name", "label", "why")}
        assert changed == {key}
        assert spec[key] is False


def test_the_no_consensus_arm_is_tgl_with_gamma_zero_and_nothing_else():
    hyper = poc.poc_hyper()
    assert hyper["attentionGamma"] > 0
    nc = poc.hyper_for(poc.NO_CONSENSUS, hyper)
    assert nc["attentionGamma"] == 0.0
    assert {k for k in hyper if hyper[k] != nc[k]} == {"attentionGamma"}
    assert poc.spec_for(poc.NO_CONSENSUS) == poc.POC_ARMS["TGL"]


def test_own_and_no_consensus_share_their_initialization(familia):
    material = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    hyper = {**poc.poc_hyper(), "kernelSigma": 0.5}
    device = torch.device("cpu")
    with toy_run.material_bound(familia, SMALL, poc.BACKBONE, hyper):
        own = poc.build_poc_arm("TGL", material, SMALL, hyper, device, 7)
        nc = poc.build_poc_arm(poc.NO_CONSENSUS, material, SMALL, hyper, device, 7)
        assert own.attention_gamma > 0 and nc.attention_gamma == 0.0
        for (name, a), (_, b) in zip(own.state_dict().items(), nc.state_dict().items()):
            assert torch.equal(a, b), name
        x = material["source"].images[material["source"].members[:2]]
        H = own.instance_embeddings(x)[0].detach()
        assert not torch.allclose(own.weights_for(H, own.sigma), nc.weights_for(H, nc.sigma))


def test_a_reference_mechanism_builds_through_the_same_seam(familia):
    material = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    hyper = {**poc.poc_hyper(), "kernelSigma": 0.5}
    with toy_run.material_bound(familia, SMALL, poc.BACKBONE, hyper):
        arm = poc.build_poc_arm("abmil-published", material, SMALL, hyper,
                                torch.device("cpu"), 7)
    assert arm.mechanism == "abmil-published"


# ------------------------------------------------------------- attention

def test_uniform_weights_give_a_zero_attention_contrast():
    weights = torch.full((4, 10), 0.1)
    masks = torch.zeros(4, 10, dtype=torch.bool)
    masks[:, :3] = True
    assert torch.allclose(poc.attention_contrast(weights, masks).float(), torch.zeros(4))


def test_the_attention_contrast_is_clean_minus_contaminated_in_uniform_units():
    m = 10
    weights = torch.tensor([[0.05] * 2 + [0.1125] * 8])     # sums to one
    masks = torch.tensor([[True] * 2 + [False] * 8])
    expected = m * (0.1125 - 0.05)
    assert torch.allclose(poc.attention_contrast(weights, masks).float(),
                          torch.tensor([expected]), atol=1e-6)


def test_bags_with_no_contaminated_slot_are_left_out_of_the_contrast():
    weights = torch.full((2, 4), 0.25)
    masks = torch.tensor([[True, False, False, False], [False] * 4])
    assert poc.attention_contrast(weights, masks).shape == (1,)


def test_attention_weights_read_one_row_per_bag_summing_to_one(familia):
    material = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    hyper = {**poc.poc_hyper(), "kernelSigma": 0.5}
    device = torch.device("cpu")
    with toy_run.material_bound(familia, SMALL, poc.BACKBONE, hyper):
        arm = poc.build_poc_arm("TGL", material, SMALL, hyper, device, 7)
        toy_run.copy_source_to_target(arm)
        target = material["target"]
        weights = poc.attention_weights(arm, target, target.eval_idx, "target", device)
    assert weights.shape == (len(target.eval_idx), SMALL.instances_per_bag)
    assert torch.allclose(weights.sum(dim=1), torch.ones(len(target.eval_idx)), atol=1e-5)


# --------------------------------------------------------------- partners

def test_partner_separation_is_others_minus_own_within_class():
    # Two classes, two bags each; target bag j's partner is source bag j.
    D = torch.tensor([[1.0, 3.0, 9.0, 9.0],
                      [2.0, 1.0, 9.0, 9.0],
                      [9.0, 9.0, 0.5, 2.5],
                      [9.0, 9.0, 4.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    index = torch.tensor([0, 1, 0, 1])
    sep = poc.partner_separation(D, labels, index, labels, index, within_class=True)
    assert torch.allclose(sep, torch.tensor([2.0, 1.0, 2.0, 3.0]))
    across = poc.partner_separation(D, labels, index, labels, index, within_class=False)
    assert torch.allclose(across[0], torch.tensor((3.0 + 9.0 + 9.0) / 3 - 1.0))


def test_bag_distances_vanish_on_a_bag_against_itself(familia):
    material = toy_run.material_for(familia, 3, SMALL, poc.BACKBONE)
    hyper = {**poc.poc_hyper(), "kernelSigma": 0.5}
    device = torch.device("cpu")
    with toy_run.material_bound(familia, SMALL, poc.BACKBONE, hyper):
        arm = poc.build_poc_arm("TGL", material, SMALL, hyper, device, 7)
        toy_run.copy_source_to_target(arm)
        arm.eval()
        source = material["source"]
        D = poc.bag_distances(arm, source, source.eval_idx, "source",
                              source, source.eval_idx, "source", device)
    assert D.shape == (len(source.eval_idx),) * 2
    assert torch.allclose(torch.diagonal(D), torch.zeros(len(source.eval_idx)), atol=1e-5)
    assert (D >= -1e-6).all()


# ------------------------------------------------------------- statistics

def test_the_paired_wilcoxon_reads_direction_effect_and_hodges_lehmann():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    x = y + 1.0 + rng.normal(scale=0.1, size=20)
    result = poc.paired_wilcoxon(x, y)
    assert result["p"] < 1e-4
    assert result["rankBiserial"] == pytest.approx(1.0)
    assert 0.9 < result["hodgesLehmann"] < 1.1
    low, high = result["hodgesLehmannCI"]
    assert low < result["hodgesLehmann"] < high
    reverse = poc.paired_wilcoxon(y, x)
    assert reverse["p"] > 0.99 and reverse["rankBiserial"] == pytest.approx(-1.0)


def test_an_all_zero_difference_is_reported_untestable_not_significant():
    result = poc.paired_wilcoxon(np.ones(10), np.ones(10))
    assert result["p"] == 1.0 and result["nonZero"] == 0


def test_holm_adjusts_step_down_and_keeps_order():
    adjusted = poc.holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_summary_carries_mean_sd_and_a_t_interval():
    s = poc.summary([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == pytest.approx(2.5)
    assert s["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert s["ci95"][0] < 2.5 < s["ci95"][1]


def test_the_battery_records_are_declared_for_audits():
    from MIL_CREDA_Benchmark import __benchmark__
    assert "Results/PoC" in __benchmark__["report"]["records"]
