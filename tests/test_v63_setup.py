"""Plan v63 setup: data views, encoder standardization and MIL-CREDA as it runs."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import mil_creda_model as M
from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG import v63_setup as V


class _Raw:
    def __init__(self, n, channels, label="yes", seed=0):
        rng = np.random.default_rng(seed)
        self.subject, self.condition, self.label = "S1", "L6", label
        self.channels = tuple(channels)
        self._w = rng.normal(size=(n, len(channels), 512)).astype(np.float32)

    def windows(self):
        return self._w


def test_a_bag_keeps_at_most_60_windows_at_step_ceil_n_over_60():
    assert V.window_step(130) == 3 and V.window_step(60) == 1 and V.window_step(61) == 2
    assert V.prepare(_Raw(130, ["a", "b"])).windows.shape[0] == len(range(0, 130, 3))
    assert V.prepare(_Raw(600, ["a", "b"])).windows.shape[0] == 60
    assert V.prepare(_Raw(20, ["a", "b"])).windows.shape[0] == 20


def test_device_transfers_read_the_12_shared_channels_before_the_average_reference():
    names = list(V.SHARED) + ["Cz", "Pz", "Fz"]
    raw = _Raw(10, names)
    b = V.prepare(raw, V.channels_of(A.Transfer("device", "L6", "Robots")))
    assert b.windows.shape[1] == 12
    assert torch.allclose(b.windows.mean(1), torch.zeros(10, 512), atol=1e-5)
    assert V.channels_of(A.Transfer("level", "L1", "L11")) is None
    with pytest.raises(V.V63Error):
        V.prepare(_Raw(5, ["a"]), V.SHARED)


def test_the_sixteen_transfers_of_sec_4_8():
    names = [t.name for t in V.TRANSFERS]
    assert len(names) == len(set(names)) == 16
    for core in ("L1->L11", "L11->L1", "Robots->CogniFit", "CogniFit->Robots",
                 "L6->Robots", "Robots->L6"):
        assert core in names
    device = [t for t in V.TRANSFERS if t.arm == "device"]
    assert len(device) == 12
    assert all({t.source in V.HIGH_DENSITY, t.target in V.HIGH_DENSITY} == {True, False}
               for t in device)


def test_the_trunk_standardization_is_fixed_and_reads_no_label():
    w = torch.randn(4, 3, 512)
    a = [two_phase.BagTensor("s", "c", 0, w)]
    b = [two_phase.BagTensor("s", "c", 1, w)]
    sa, sb = V.standard_of(a), V.standard_of(b)
    assert torch.equal(sa.mean, sb.mean) and torch.equal(sa.std, sb.std)
    assert not list(sa.parameters())
    with V.installed():
        with pytest.raises(V.V63Error):
            V.spectral_trunk.SpectralTrunk(3)
        with V.standardization(sa):
            trunk = V.spectral_trunk.SpectralTrunk(3)
        assert torch.equal(trunk.norm.mean, sa.mean)
        assert "norm.mean" in trunk.state_dict()


def test_the_gate_maps_auc_half_to_zero_and_one_to_one():
    assert V.gate_weight(0.5) == 0.0 and V.gate_weight(1.0) == 1.0
    assert V.gate_weight(0.75) == pytest.approx(0.5)
    assert V.gate_weight(0.3) == 0.0
    with pytest.raises(V.V63Error):
        V.gate_weight(None)


class _Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_trunk = nn.Linear(2, 2)


def test_mil_creda_phase_two_confidences_and_g_carry_no_gradient():
    G = torch.tensor([[0.3, 0.7]], requires_grad=True)
    with V.installed():
        assert V._detached_confidences(G).requires_grad
        with V.mil_creda_rules(_Fake(), threshold=0.5, gate=1.0):
            assert not V._detached_confidences(G).requires_grad
            seen = {}
            orig = V._corr

            def spy(cross, labels, g, tau):
                seen["g"], seen["tau"] = g, tau
                return torch.zeros(1)

            V._corr = spy
            try:
                V._correspondence(torch.tensor([1.0, 2.0, 4.0]), None, G[0], 99.0)
            finally:
                V._corr = orig
            assert not seen["g"].requires_grad


def test_tau_loc_is_the_kernel_row_standard_deviation():
    cross = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    assert V.row_tau(cross) == pytest.approx(float(cross.std()) + 1e-6)


def test_the_pseudolabel_cut_is_the_bridged_median_computed_once():
    with V.installed():
        with V.mil_creda_rules(_Fake(), threshold=0.8, gate=1.0):
            assert M.pseudolabel(torch.tensor([0.25, 0.75])) == 0
            assert M.pseudolabel(torch.tensor([0.15, 0.85])) == 1
        assert M.pseudolabel(torch.tensor([0.25, 0.75])) == V._pl(torch.tensor([0.25, 0.75]))


def test_the_l2_sp_term_is_mu_times_the_squared_distance_to_the_bridged_trunk():
    m = _Fake()
    terms = {"global": torch.tensor(2.0, dtype=torch.float64),
             "local": torch.tensor(3.0, dtype=torch.float64)}
    base = float(V._obj(terms))
    with V.installed():
        with V.mil_creda_rules(m, threshold=0.5, gate=0.25, mu=10.0):
            with torch.no_grad():
                m.target_trunk.weight.add_(0.1)
            out = float(two_phase.phase_two_objective(terms))
    prox = 0.1 ** 2 * m.target_trunk.weight.numel()
    assert out == pytest.approx(0.25 * base + 10.0 * prox, rel=1e-5)


def test_the_phase_one_space_is_lr1_and_l2_and_the_mil_creda_phase_two_space_is_lr2_and_wd2():
    assert set(V.PHASE_ONE_SPACE) == {"lr1", "l2"}
    d = V.PHASE_ONE_SPACE["l2"]
    assert (d.low, d.high, d.log) == (1e-4, 1e-1, True) and d.low < 1e-2 < d.high
    assert set(V.MIL_CREDA_PHASE_TWO_SPACE) == {"lr2", "wd2"}
    sched = {"epochs1": 3, "epochs2": 3, "bagsPerStep": 8}
    h = V.phase_one_hyper("MIL-CREDA", {"lr1": 1e-3, "l2": 1e-2}, sched)
    assert (h["attentionGamma"], h["attentionTemperature"], h["wd1"]) == (0.0, 10.0, 0.0)
    assert V.phase_one_hyper("common", {"lr1": 1e-3, "l2": 1e-2}, sched)["wd1"] == 0.0
    with pytest.raises(V.V63Error):
        V.phase_one_hyper("common", {"lr1": 1e-3, "wd1": 1e-5}, sched)
    h2 = V.mil_creda_hyper(h, {"lr2": 1e-3, "wd2": 1e-5}, sched)
    assert (h2["attentionGamma"], h2["attentionTemperature"]) == (0.5, 1.0)
    with pytest.raises(V.V63Error):
        V.mil_creda_hyper(h, {"lr2": 1e-3, "wd2": 1e-5, "tauLocal": 1.0}, sched)
    assert V.MU == 10.0


class _Trunks(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_trunk = nn.Flatten()
        self.p = nn.Parameter(torch.zeros(1))

    def trunk(self, domain):
        return self.source_trunk

    def parameters(self, recurse=True):
        return iter([self.p])


def test_the_between_bag_sigma_never_pairs_a_bag_with_itself():
    bags = [two_phase.BagTensor(f"s{i}", "c", 0, torch.full((5, 1, 2), float(i)))
            for i in range(3)]
    sigma = V.between_bag_sigma(_Trunks(), bags, pairs=200)
    assert sigma > 0 and math.isfinite(sigma)
    with pytest.raises(V.V63Error):
        V.between_bag_sigma(_Trunks(), bags[:1])


class _L2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_trunk = nn.Linear(2, 2)
        self.head = nn.Module()
        self.head.linear = nn.Linear(2, 2)
        self.classes = 2


def test_every_arms_phase_one_adds_the_searched_explicit_l2():
    from MIL_CREDA_EEG.comparators.adaptmil import training as adaptmil_training
    m = _L2Model()
    expected = 0.05 * ((m.source_trunk.weight ** 2).sum() + (m.head.linear.weight ** 2).sum())
    G, labels = torch.tensor([[0.3, 0.7]], dtype=torch.float64), torch.tensor([1])
    plain_source, plain_adapt = two_phase.source_term, adaptmil_training.stage_two_objective
    base = plain_source(m, G, labels)
    with V.phase_one_l2(0.05):
        assert float(two_phase.source_term(m, G, labels) - base) == pytest.approx(float(expected))
        assert adaptmil_training.stage_two_objective is not plain_adapt
    assert two_phase.source_term is plain_source
    assert adaptmil_training.stage_two_objective is plain_adapt
    assert float(V.l2_penalty(m, 0.05)) == pytest.approx(float(expected))
    with pytest.raises(V.V63Error):
        with V.phase_one_l2(float("nan")):
            pass


def test_the_code_version_reads_only_the_training_modules():
    assert set(V.TRAINING_MODULES) == {"v63_campaign.py", "v63_filters.py", "v63_records.py",
                                       "v63_setup.py"}
    assert not any("analysis" in m or "saved" in m or "connectivity" in m
                   for m in V.TRAINING_MODULES)
