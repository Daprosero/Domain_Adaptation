"""Proposal r38 Eqs. 13-39 on EEG: two trunks, one attention, one head."""

from __future__ import annotations

import pytest
import torch

from MIL_CREDA.attention import bag_embedding, bag_weights, relevance_logits
from MIL_CREDA_EEG import eegnet, mil_creda_model as mcm


def _bags(n: int, channels: int, sizes=(5, 7, 6, 4), seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(sizes[i % len(sizes)], channels, eegnet.SAMPLES, generator=g)
            for i in range(n)]


def _model(source=29, target=14, gamma=0.5, tau=1.0, sigma=2.0) -> mcm.MILCREDAModel:
    torch.manual_seed(0)
    model = mcm.MILCREDAModel(source, target, attention_gamma=gamma,
                              attention_temperature=tau)
    if sigma is not None:
        model.fix_sigma(sigma)
    return model.eval()


def test_source_and_target_trunks_are_independent_eegnets_with_their_own_channels():
    model = _model()
    assert model.source_trunk.channels == 29 and model.target_trunk.channels == 14
    s = {p.data_ptr() for p in model.source_trunk.parameters()}
    t = {p.data_ptr() for p in model.target_trunk.parameters()}
    assert s and t and not (s & t)


def test_one_shared_attention_serves_both_domains():
    model = _model(source=29, target=29)
    names = [n for n, _ in model.named_parameters()]
    attention = [n for n in names if not n.startswith(("source_trunk.", "target_trunk.",
                                                        "head."))]
    assert sorted(attention) == ["V_R", "b_R", "v_R"]
    H = torch.randn(6, eegnet.EMBEDDING_DIM, dtype=torch.float64)
    (_, w_s), = model.bag_pairs([H])
    (_, w_t), = model.bag_pairs([H.clone()])
    assert torch.equal(w_s, w_t)


def test_the_bag_representation_is_eq_19_over_eq_15_16_weights_at_the_fixed_sigma():
    model = _model(gamma=0.7, tau=0.5, sigma=3.0)
    bags = _bags(2, 29)
    with torch.no_grad():
        H = model.embed(bags, "source")
        pairs = model.bag_pairs(H)
        Z = model.represent(pairs)
    for k, H_k in enumerate(H):
        logits = relevance_logits(H_k, model.V_R, model.b_R, model.v_R, 0.7, 3.0)
        beta = bag_weights(logits, 0.5)
        assert torch.allclose(pairs[k][1], beta)
        assert torch.allclose(Z[k], bag_embedding(H_k, beta))


def test_the_head_is_applied_once_per_bag():
    model = _model()
    rows = []
    model.head.register_forward_hook(lambda m, i, o: rows.append(i[0].shape[0]))
    bags = _bags(3, 14)
    with torch.no_grad():
        G = model.predict(bags, "target")
    assert rows == [3]
    assert G.shape == (3, 2)
    assert torch.allclose(G.sum(dim=1), torch.ones(3, dtype=G.dtype))


def test_sigma_is_a_fixed_number_set_once():
    model = _model(sigma=None)
    with pytest.raises(mcm.SigmaNotFixed):
        model.predict(_bags(1, 29), "source")
    model.fix_sigma(1.25)
    assert model.sigma == 1.25 and isinstance(model.sigma, float)
    model.fix_sigma(1.25)                       # the same value again is not a change
    with pytest.raises(mcm.SigmaAlreadyFixed):
        model.fix_sigma(1.2500001)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            mcm.MILCREDAModel(29, 29, attention_gamma=0.0,
                              attention_temperature=1.0).fix_sigma(bad)


def test_the_l1_constraint_on_the_attention_holds_from_construction():
    model = _model()
    assert float(model.v_R.detach().abs().sum()) <= 1.0 + 1e-12


def _terms(model, labels=(0, 1, 0, 1)):
    source = _bags(len(labels), 29, seed=1)
    target = _bags(4, 14, seed=2)
    with torch.no_grad():
        pairs_s = model.bag_pairs(model.embed(source, "source"))
        pairs_t = model.bag_pairs(model.embed(target, "target"))
        return mcm.adaptation_terms(model, pairs_s, torch.tensor(labels), pairs_t,
                                    tau_local=1.0)


def test_the_adaptation_terms_lie_in_their_declared_bounds():
    terms = _terms(_model())
    lo, hi = mcm.TERM_BOUNDS["global"]
    assert lo <= float(terms["global"]) <= hi
    lo, hi = mcm.TERM_BOUNDS["local"]
    assert lo <= float(terms["local"]) <= hi
    assert set(terms["source_counts"]) == {2}


def test_a_class_no_target_bag_routes_to_is_charged_the_maximum():
    model = _model()
    with torch.no_grad():
        model.head.linear.bias.copy_(torch.tensor([50.0, -50.0], dtype=torch.float64))
    terms = _terms(model)
    assert terms["active_classes"] <= 1
    assert float(terms["global"]) >= 0.5 - 1e-12        # class 1: weight 2/4, loss 1


def test_the_local_correspondence_needs_every_class_in_the_source_batch():
    with pytest.raises(mcm.SourceBatchMissingClass):
        _terms(_model(), labels=(0, 0, 0, 0))
