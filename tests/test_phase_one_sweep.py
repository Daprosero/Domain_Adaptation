from types import SimpleNamespace

import torch


from MIL_CREDA_EEG import eegnet, phase_one_sweep as sweep


def test_units_cover_every_config_transfer_and_seed_once():
    units = sweep.unit_ids()
    assert len(units) == len(set(units)) == len(sweep.CONFIGS) * len(sweep.TRANSFERS) * len(sweep.SEEDS)
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


def test_the_fused_shallow_convolution_equals_the_two_layers():
    from MIL_CREDA_EEG import shallow_trunk
    torch.manual_seed(0)
    trunk = shallow_trunk.ShallowTrunk(14).eval()
    x = torch.randn(3, 14, 512)
    two = trunk.conv_spatial(trunk.conv_temporal(x.unsqueeze(1)))
    seen = {}
    trunk.bn.register_forward_hook(lambda m, i, o: seen.update(x=i[0]))
    trunk(x)
    assert torch.allclose(seen["x"], two, atol=1e-4)


def test_each_partial_condition_names_trainable_layers_of_the_trunk():
    from MIL_CREDA_EEG.phase_two_terms import CONDITIONS
    names = [k for k, _ in eegnet.EEGNetTrunk(14).named_parameters()]
    for spec in CONDITIONS.values():
        if "train" in spec:
            assert any(n.startswith(spec["train"]) for n in names)


def test_balanced_pseudolabels_give_both_classes_and_restore_the_rule():
    from MIL_CREDA_EEG import mil_creda_model, phase_two_terms
    rule = mil_creda_model.pseudolabel
    G = torch.tensor([[.1, .9], [.2, .8], [.3, .7], [.4, .6]])
    model = SimpleNamespace(classify=lambda _: G)
    with phase_two_terms.balanced_pseudolabels(model):
        scored = model.classify(None)
        labels = [mil_creda_model.pseudolabel(g) for g in scored]
    assert set(labels) == {0, 1}
    assert mil_creda_model.pseudolabel is rule


def test_loso_settings_cover_the_four_transfers_and_build_a_hyper():
    import json
    from MIL_CREDA_EEG import loso, two_phase
    chosen = json.loads(loso.SELECTIONS.read_text())
    assert set(chosen) == {"Robots->CogniFit", "CogniFit->Robots", "L1->L11", "L11->L1"}
    for values in chosen.values():
        two_phase.check_hyper({**{k: values[k] for k in two_phase.PHASE_ONE_KEYS}, "lr2": 1e-3,
                               "wd2": 1e-5, "tauLocal": 1.0, "epochs1": values["epochs1"],
                               "epochs2": 20, "bagsPerStep": 8})


def test_the_oracle_hands_true_labels_and_counts_the_rule_hits():
    from MIL_CREDA_EEG import mil_creda_model, phase_two_terms, two_phase
    step, rule = two_phase.phase_two_step, mil_creda_model.pseudolabel
    bags = [SimpleNamespace(label=1), SimpleNamespace(label=0)]
    G = torch.tensor([[.9, .1], [.8, .2]])
    two_phase_step = lambda model, source, target, *a, **k: [mil_creda_model.pseudolabel(g) for g in G]
    two_phase.phase_two_step = two_phase_step
    try:
        with phase_two_terms.observed_pseudolabels(oracle=True) as seen:
            assert two_phase.phase_two_step(None, [], bags) == [1, 0]
        assert seen["hits"] == 1 and seen["seen"] == 2
    finally:
        two_phase.phase_two_step = step
    assert mil_creda_model.pseudolabel is rule


def test_the_head_kernel_projects_embeddings_and_restores_the_plan_kernel():
    from MIL_CREDA_EEG import mil_creda_model, phase_two_terms
    plain = mil_creda_model.bag_kernel_matrix
    W = torch.zeros(2, 4, dtype=torch.float64); W[0, 0] = 1.0; W[1, 1] = 1.0
    model = SimpleNamespace(head=SimpleNamespace(linear=SimpleNamespace(weight=W)),
                            embed=lambda bags, domain, chunk: [torch.randn(5, 4, dtype=torch.float64)])
    pairs = [(torch.randn(5, 4, dtype=torch.float64), torch.full((5,), 0.2, dtype=torch.float64))]
    with phase_two_terms.head_kernel(model, [SimpleNamespace(windows=torch.zeros(5, 14, 512))], "cpu", 1.0) as scale:
        K = mil_creda_model.bag_kernel_matrix(pairs, pairs, 99.0)
        assert scale > 0 and K.shape == (1, 1)
    assert mil_creda_model.bag_kernel_matrix is plain


def test_prepared_caps_windows_and_filters_without_touching_labels():
    from MIL_CREDA_EEG import two_phase
    bag = two_phase.BagTensor("s", "c", 1, torch.randn(100, 14, 512))
    capped = sweep.prepared([bag], {"cap": 72})[0]
    assert capped.windows.shape == (72, 14, 512) and capped.label == 1
    filtered = sweep.prepared([bag], {"band": "filt"})[0]
    assert torch.allclose(filtered.windows.mean(1), torch.zeros(100, 512), atol=1e-4)
    assert sweep.prepared([bag], {}) == [bag]


def test_zbag_standardizes_each_channel_within_its_own_bag():
    from MIL_CREDA_EEG import two_phase
    bag = two_phase.BagTensor("s", "c", 0, 5 + 3 * torch.randn(20, 14, 512))
    z = sweep.prepared([bag], {"zbag": True})[0].windows
    assert torch.allclose(z.mean(dim=(0, 2)), torch.zeros(14), atol=1e-4)
    assert torch.allclose(z.std(dim=(0, 2)), torch.ones(14), atol=1e-3)


def test_the_spectral_trunk_reads_band_power_and_feeds_mil_creda():
    from MIL_CREDA_EEG import campaign, model_store, spectral_trunk, two_phase
    campaign.register(["MIL-CREDA"])
    with spectral_trunk.installed():
        model = model_store.build_model("MIL-CREDA", 14, 14, {"attentionGamma": 0.5, "attentionTemperature": 1.0})
        assert model.source_trunk(torch.randn(3, 14, 512)).shape == (3, 64)
        model.fix_sigma(1.0)
        assert two_phase.bridge(model)["layer_copy"]["copied"] == ["project"]
    t = 2 * torch.pi * 10 * torch.arange(512) / 128
    alpha = spectral_trunk.SpectralTrunk(1).features(torch.sin(t).reshape(1, 1, 512))[0]
    assert int(alpha[:5].argmax()) == 2           # the 10 Hz power lands in the alpha band


def test_explicit_l2_adds_the_weight_penalty_and_restores_the_term():
    from MIL_CREDA_EEG import two_phase
    plain = two_phase.source_term
    w = torch.ones(2, 2)
    model = SimpleNamespace(source_trunk=SimpleNamespace(named_parameters=lambda: [("a.weight", w)]),
                            head=SimpleNamespace(linear=SimpleNamespace(weight=w)))
    two_phase.source_term = lambda m, G, y: torch.tensor(0.0)
    try:
        with sweep.explicit_l2(0.5):
            assert float(two_phase.source_term(model, None, None)) == 4.0
    finally:
        two_phase.source_term = plain


def test_standardized_uses_the_training_statistics_for_every_group():
    from MIL_CREDA_EEG import two_phase
    tr = [two_phase.BagTensor("a", "c", 0, 3 + 2 * torch.randn(30, 4, 512))]
    te = [two_phase.BagTensor("b", "c", 1, 3 + 2 * torch.randn(10, 4, 512))]
    (s_tr,), (s_te,) = sweep.standardized(tr, te)
    assert torch.allclose(s_tr.windows.mean(dim=(0, 2)), torch.zeros(4), atol=1e-3)
    assert abs(float(s_te.windows.mean())) < 0.05 and s_te.label == 1


def test_euclidean_alignment_whitens_each_subject_covariance():
    from MIL_CREDA_EEG import two_phase
    mix = torch.randn(4, 4)
    bag = two_phase.BagTensor("s", "c", 0, torch.einsum("cd,ndt->nct", mix, torch.randn(30, 4, 512)))
    w = sweep.prepared([bag], {"ea": True})[0].windows.double()
    w = w - w.mean(-1, keepdim=True)
    cov = torch.einsum("nct,ndt->cd", w, w) / (w.shape[0] * w.shape[2])
    assert torch.allclose(cov, torch.eye(4, dtype=torch.float64), atol=1e-3)


def test_the_phase_one_search_samples_inside_its_space_and_is_seeded():
    import numpy as np
    from MIL_CREDA_EEG import phase_one_search as s
    a = s.sample(np.random.RandomState(3)); b = s.sample(np.random.RandomState(3))
    assert a == b
    assert 1e-4 <= a["lr1"] <= 3e-2 and 1e-5 <= a["l2"] <= 3e-2
    assert (a["F1"], a["D"], a["F2"]) in s.SIZES and a["epochs"] in s.EPOCH_CHOICES


def test_subbags_keep_the_subject_and_label_and_cut_full_windows():
    from MIL_CREDA_EEG import subbag_check, two_phase
    bag = two_phase.BagTensor("s1", "c", 1, torch.randn(70, 14, 512))
    parts = subbag_check.subbags([bag], 16, torch.Generator().manual_seed(0))
    assert len(parts) == 4 and all(p.windows.shape == (16, 14, 512) for p in parts)
    assert {p.subject for p in parts} == {"s1"} and {p.label for p in parts} == {1}
    assert subbag_check.subbags([bag], 0, torch.Generator()) == [bag]


def test_the_shallow_trunk_keeps_its_layers_under_the_sweep_constants():
    from MIL_CREDA_EEG import shallow_trunk
    with shallow_trunk.installed(), sweep.patched({"HEAD_MAX_NORM": 100.0, "DROPOUT": 0.5}):
        assert eegnet.EEGNetTrunk is shallow_trunk.ShallowTrunk and eegnet.HEAD_MAX_NORM == 100.0
    assert eegnet.EEGNetTrunk is not shallow_trunk.ShallowTrunk


def test_input_normalization_is_domain_specific_and_restores_eegnet():
    from MIL_CREDA_EEG import campaign, input_norm, model_store, two_phase
    campaign.register(["MIL-CREDA"])
    plain = eegnet.EEGNetTrunk
    for kind in ("batch", "layer"):
        with input_norm.installed(kind):
            model = model_store.build_model("MIL-CREDA", 14, 29, {"attentionGamma": 0.5, "attentionTemperature": 1.0})
            assert model.source_trunk(5 + 3 * torch.randn(4, 14, 512)).shape == (4, 256)
            model.fix_sigma(1.0)
            assert "input_norm" in two_phase.bridge(model)["layer_copy"]["kept_initialization"]
        assert eegnet.EEGNetTrunk is plain


def test_window_norm_removes_each_window_scale():
    from MIL_CREDA_EEG.input_norm import WindowNorm
    x = torch.randn(3, 4, 512) * torch.tensor([1.0, 10.0, 100.0])[:, None, None]
    y = WindowNorm(4)(x)
    assert torch.allclose(y.mean(dim=(1, 2)), torch.zeros(3), atol=1e-4)


def test_the_search_samples_shallow_settings_and_builds_its_filters():
    import numpy as np
    from MIL_CREDA_EEG import phase_one_search as s, shallow_trunk
    trial = s.sample(np.random.RandomState(0), "shallow")
    assert trial["FILTERS"] in s.SHALLOW_FILTERS and 1e-4 <= trial["lr1"] <= 3e-3
    with s.network("shallow", {**trial, "FILTERS": 20}):
        assert eegnet.EEGNetTrunk(14).conv_temporal.out_channels == 20
    assert shallow_trunk.FILTERS == 40 and eegnet.EEGNetTrunk is not shallow_trunk.ShallowTrunk


def test_the_log_power_block_keeps_eegnet_shapes_and_restores_it():
    from MIL_CREDA_EEG import input_norm
    plain = eegnet.EEGNetTrunk
    with input_norm.log_power():
        trunk = eegnet.EEGNetTrunk(14)
        out = trunk(torch.randn(3, 14, 512))
        assert out.shape == (3, 256) and torch.isfinite(out).all()
    assert eegnet.EEGNetTrunk is plain
