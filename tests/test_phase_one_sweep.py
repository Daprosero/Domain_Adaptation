from types import SimpleNamespace

import torch


from MIL_CREDA_EEG import eegnet, phase_one_sweep as sweep


def test_units_cover_every_config_transfer_and_seed_once():
    units = sweep.unit_ids()
    assert len(units) == len(set(units)) == len(sweep.CONFIGS) * 2 * len(sweep.SEEDS)
    assert sweep.parse("sweep.wide-k32.CR.seed2") == ("wide-k32", "CogniFit->Robots", 2)


def test_a_configuration_changes_the_network_and_leaves_the_plan_constants_intact():
    with sweep.patched(sweep.CONFIGS["wide-k32"]):
        trunk = eegnet.EEGNetTrunk(14)
        assert trunk.conv_temporal.out_channels == 16
        assert trunk.conv_temporal.kernel_size == (1, 32)
        assert trunk.output_dim == 32 * 16
    assert (eegnet.F1, eegnet.F2, eegnet.KERNEL_LENGTH) == (8, 16, 64)
    assert eegnet.EEGNetTrunk(14).output_dim == eegnet.EMBEDDING_DIM


def test_the_base_matches_the_plan_constants():
    assert all(getattr(eegnet, k) == v for k, v in sweep.BASE.items())


def test_a_diverged_side_is_flagged_and_never_reads_as_chance():
    rows = [{"label": 0, "probability": float("nan")}, {"label": 1, "probability": float("nan")}]
    assert sweep._summary(rows) == {"ba": None, "auc": None, "n": 2, "diverged": True}


def test_recalibration_replaces_stale_batchnorm_statistics_and_keeps_the_weights():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_trunk = eegnet.EEGNetTrunk(14)

    torch.manual_seed(0)
    model = Model()
    bag = SimpleNamespace(windows=100 * torch.randn(40, 14, 512))
    weights = model.source_trunk.conv_temporal.weight.clone()
    bn = model.source_trunk.bn1
    momentum, stale = bn.momentum, bn.running_var.clone()
    sweep.recalibrate(model, [bag], torch.device("cpu"))
    assert not torch.allclose(bn.running_var, stale)
    assert bn.momentum == momentum and not bn.training and not model.source_trunk.training
    assert torch.equal(model.source_trunk.conv_temporal.weight, weights)


def test_every_diagnostic_scores_with_recalibrated_batchnorm():
    from MIL_CREDA_EEG import beat_predict, phase_two_terms
    assert beat_predict.recalibrate is sweep.recalibrate
    assert phase_two_terms.recalibrate is sweep.recalibrate
    assert beat_predict.SIDES == ("sourceHeldOut", "predict", "milCreda")


def test_blending_interpolates_weights_and_keeps_integer_buffers():
    from MIL_CREDA_EEG.phase_two_terms import blended
    start = {"w": torch.zeros(2), "n": torch.tensor(1)}
    adapted = {"w": torch.ones(2), "n": torch.tensor(5)}
    mixed = blended(start, adapted, 0.25)
    assert torch.allclose(mixed["w"], torch.full((2,), 0.25)) and mixed["n"] == 5


def test_the_shallow_trunk_feeds_mil_creda_and_restores_the_plan_network():
    from MIL_CREDA_EEG import campaign, model_store, shallow_trunk, two_phase
    campaign.register(["MIL-CREDA"])
    hyper = {"attentionGamma": 0.5, "attentionTemperature": 1.0}
    with shallow_trunk.installed():
        model = model_store.build_model("MIL-CREDA", 14, 14, hyper)
        assert isinstance(model.source_trunk, shallow_trunk.ShallowTrunk)
        assert model.source_trunk(torch.randn(4, 14, 512)).shape == (4, 1160)
        assert model.head.linear.in_features == 1160
        model.fix_sigma(5.0)
        assert two_phase.bridge(model)["layer_copy"]["copied"] == list(shallow_trunk.LAYERS)
    assert eegnet.EEGNetTrunk is not shallow_trunk.ShallowTrunk and eegnet.HEAD_MAX_NORM == 0.25
