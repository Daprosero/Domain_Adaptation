"""Experiments plan v40, Sec. 7.2: the common MIL wrapper of the non-MIL comparators."""

from __future__ import annotations

import pytest
import torch

from MIL_CREDA.attention import relevance_component, relevance_logits
from MIL_CREDA_EEG import eegnet, mil_wrapper, two_phase as tp
from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel

CPU = torch.device("cpu")


def _embeddings(seed: int, m: int) -> torch.Tensor:
    """A tight bag: within-bag distances near 1, so a kernel consensus would be live."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, eegnet.EMBEDDING_DIM, generator=g, dtype=torch.float64)
    scale = torch.linspace(0.02, 0.1, m, dtype=torch.float64).reshape(-1, 1)
    return base + scale * torch.randn(m, eegnet.EMBEDDING_DIM, generator=g, dtype=torch.float64)


def test_the_common_attention_is_eq_15_with_gamma_zero_and_no_consensus():
    torch.manual_seed(0)
    model = mil_wrapper.CommonMILModel(3, 2)
    H = _embeddings(1, 6)
    (_, weights), = model.bag_pairs([H])           # no sigma was ever fixed: none is read
    consensus_live = relevance_logits(H, model.V_R, model.b_R, model.v_R, 1.0,
                                      mil_wrapper.NEUTRAL_SIGMA)
    assert not torch.allclose(torch.softmax(consensus_live, dim=0), weights)
    relevance = relevance_component(H, model.V_R, model.b_R, model.v_R)
    assert torch.allclose(weights, torch.softmax(relevance, dim=0), atol=0, rtol=1e-12)
    for sigma in (0.01, 1.0, 100.0):                # Eq. 15 at gamma = 0 for any width
        logits = relevance_logits(H, model.V_R, model.b_R, model.v_R, 0.0, sigma)
        assert torch.allclose(weights, torch.softmax(logits, dim=0), atol=0, rtol=1e-12)
    # The same parameters in MIL-CREDA's own model at gamma 0, tau 1 give the same weights,
    # and a positive gamma does not: the consensus is what the wrapper leaves out.
    reference = MILCREDAModel(3, 2, attention_gamma=0.0, attention_temperature=1.0)
    reference.load_state_dict(model.state_dict())
    reference.fix_sigma(3.7)
    (_, ref_weights), = reference.bag_pairs([H])
    assert torch.allclose(weights, ref_weights, atol=0, rtol=1e-12)
    consensus = MILCREDAModel(3, 2, attention_gamma=0.5, attention_temperature=1.0)
    consensus.load_state_dict(model.state_dict())
    consensus.fix_sigma(float(torch.cdist(H, H).median()))  # a width the consensus reads
    (_, other), = consensus.bag_pairs([H])
    assert not torch.allclose(weights, other)


def test_the_bag_is_the_attention_weighted_sum_and_the_head_applies_once_per_bag():
    torch.manual_seed(0)
    model = mil_wrapper.CommonMILModel(3, 3).eval()
    g = torch.Generator().manual_seed(2)
    bags = [torch.randn(m, 3, eegnet.SAMPLES, generator=g) for m in (2, 4, 3)]
    G = model.predict(bags, "source")
    expected = []
    for b in bags:
        H = model.source_trunk(b).to(torch.float64)
        w = torch.softmax(relevance_component(H, model.V_R, model.b_R, model.v_R), dim=0)
        expected.append(w @ H)
    assert G.shape == (3, 2)
    assert torch.allclose(G, model.head(torch.stack(expected)), atol=1e-12)


def test_the_common_wrapper_fixes_gamma_and_temperature():
    model = mil_wrapper.CommonMILModel(3, 2)
    assert model.attention_gamma == mil_wrapper.COMMON_GAMMA == 0.0
    assert model.attention_temperature == mil_wrapper.COMMON_TEMPERATURE == 1.0
    with pytest.raises(TypeError):
        mil_wrapper.CommonMILModel(3, 2, attention_gamma=0.5)


def test_the_common_wrapper_is_the_phase_one_architecture_of_the_six_non_mil_comparators():
    assert mil_wrapper.GROUP == "common"
    assert mil_wrapper.COMPARATORS == ("UDA-DDA", "CCA-LSC", "CASUAL", "COT", "CS+CCS",
                                       "CLUST — variante source-free")
    torch.manual_seed(0)
    model = mil_wrapper.CommonMILModel(3, 2)
    g = torch.Generator().manual_seed(1)
    bags = [tp.BagTensor(f"s{i}", "c", i % 2, torch.randn(3, 3, eegnet.SAMPLES, generator=g))
            for i in range(6)]
    hyper = {**mil_wrapper.COMMON_PHASE_ONE_FIXED, "lr1": 1e-2, "wd1": 1e-4,
             "lr2": float("nan"), "wd2": float("nan"), "tauLocal": float("nan"),
             "epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 2}
    out = tp.train_source_phase(model, bags[:4], bags[4:], hyper,
                                torch.Generator().manual_seed(0), CPU)
    assert set(out["checkpoint"]["state"]) == set(tp.phase_one_state(model))
    assert not any(k.startswith("target_trunk.") for k in out["checkpoint"]["state"])
