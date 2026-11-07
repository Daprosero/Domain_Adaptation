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
