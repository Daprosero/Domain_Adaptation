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




def test_blending_interpolates_weights_and_keeps_integer_buffers():
    from MIL_CREDA_EEG.phase_two_terms import blended
    start = {"w": torch.zeros(2), "n": torch.tensor(1)}
    adapted = {"w": torch.ones(2), "n": torch.tensor(5)}
    mixed = blended(start, adapted, 0.25)
    assert torch.allclose(mixed["w"], torch.full((2,), 0.25)) and mixed["n"] == 5






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




















def test_spectral_check_prepares_windows_and_standardizes_each_domain():
    from MIL_CREDA_EEG import spectral_check, spectral_trunk, two_phase
    bag = two_phase.BagTensor("s", "c", 1, torch.randn(130, 14, 512) + 3)
    (p,) = spectral_check.prepared([bag])
    assert p.windows.shape[0] <= 66 and torch.allclose(p.windows.mean(dim=1), torch.zeros_like(p.windows[:, 0]), atol=1e-4)
    trunk = spectral_trunk.SpectralTrunk(14)
    std = spectral_check.standard_of(trunk, [p])
    z = std(trunk.features(p.windows))
    assert torch.allclose(z.mean(0), torch.zeros(z.shape[1]), atol=1e-4)
