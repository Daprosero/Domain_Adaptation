"""Experiments plan v42, Sec. 6.3 (K3): the ablation of the adaptation terms and the
aggregation variants of MIL-CREDA, trained on tiny seeded synthetic bags on CPU.

No number here is a measured result. The byte-for-byte witnesses compare the
default path against a reference computed inside the test: a verbatim copy of the
phase-2 step of 5395416 and a verbatim reconstruction of MILCREDAModel's
initialization, both run from the same fixed seed.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from MIL_CREDA.attention import (bag_embedding, bag_weights, project_onto_l1_ball,
                                 relevance_logits)
from MIL_CREDA.objective import total_objective
from MIL_CREDA_EEG import aggregation_variants as av
from MIL_CREDA_EEG import assignment, eegnet, mechanisms as mech, search
from MIL_CREDA_EEG import two_phase as tp
from MIL_CREDA_EEG.mil_creda_model import (ATTENTION_WIDTH, MILCREDAModel, adaptation_terms,
                                           source_term)

CPU = torch.device("cpu")
SCHEDULE = {"epochs1": 2, "patience1": 2, "epochs2": 2, "bagsPerStep": 4}
P1 = {"lr1": 1e-2, "wd1": 1e-4, "attentionGamma": 0.5, "attentionTemperature": 1.0,
      "sigmaMultiplier": 1.0}
P2 = {"lr2": 1e-2, "wd2": 1e-4, "tauLocal": 1.0}
HYPER = {**P1, **P2, **SCHEDULE}
TRANSFER = assignment.Transfer("level", "L1", "L11")


def _bags(n: int, channels: int, seed: int, prefix: str) -> list[tp.BagTensor]:
    g = torch.Generator().manual_seed(seed)
    return [tp.BagTensor(f"{prefix}{i}", "c", i % 2,
                         torch.randn(3 + i % 2, channels, eegnet.SAMPLES, generator=g))
            for i in range(n)]


def _data(source_channels: int = 3, target_channels: int = 2) -> search.SearchData:
    return search.SearchData(transfer=TRANSFER, partition_sha256="synthetic", rotations={0: {
        "source_train": _bags(6, source_channels, 1, "s"),
        "source_valid": _bags(4, source_channels, 2, "v"),
        "target_train": _bags(6, target_channels, 3, "t"),
        "target_valid": _bags(4, target_channels, 4, "w")}})


def _phase_one_record(data: search.SearchData, seed: int = 1):
    """Full MIL-CREDA's phase-1 checkpoint, as the store hands it back."""
    trained = search.train_phase_one("MIL-CREDA", data, 0, seed,
                                     search.phase_one_hyper("MIL-CREDA", P1, SCHEDULE))
    checkpoint = trained["result"]["checkpoint"]
    meta = {"phase": 1, "name": "MIL-CREDA", "transfer": TRANSFER.name, "rotation": 0,
            "seed": seed, "sigma": checkpoint["sigma"],
            "hyper": search._stored_hyper(trained["hyper"])}
    return checkpoint["state"], meta


# ------------------------------------------------------------------ the 5395416 reference


def _reference_objective(terms):
    """two_phase.phase_two_objective at 5395416, verbatim."""
    zero = torch.zeros((), dtype=terms["local"].dtype, device=terms["local"].device)
    return total_objective(zero, terms["global"], terms["local"], 1.0, 1.0)


def _reference_step(model, source, target, tau_local, device):
    """two_phase.phase_two_step at 5395416, verbatim."""
    labels = torch.tensor([b.label for b in source], device=device)
    with torch.no_grad():
        pairs_s = model.bag_pairs(model.embed([b.windows.to(device) for b in source], "source"))
        source_ce = source_term(model, model.classify(model.represent(pairs_s)), labels)
    pairs_t = model.bag_pairs(model.embed([b.windows.to(device) for b in target], "target"))
    terms = adaptation_terms(model, pairs_s, labels, pairs_t, tau_local)
    return {**terms, "objective": _reference_objective(terms), "source_ce": source_ce}


#: Every key a phase-2 history record carried at 5395416.
PHASE_TWO_KEYS_5395416 = {
    "epoch", "seed", "fold", "model", "transfer", "loss", "global", "local", "source_ce",
    "active_classes", "confidence_mean", "val_balanced_accuracy", "label_free_score",
    "lambda_global", "lambda_local", "global_min", "global_max", "local_min", "local_max",
    "global_bound_low", "global_bound_high", "local_bound_low", "local_bound_high",
    "terms_within_bounds", "sigma"}


def _through_bridge():
    torch.manual_seed(0)
    model = MILCREDAModel(3, 2, attention_gamma=0.5, attention_temperature=1.0)
    data = _data().rotations[0]
    generator = torch.Generator().manual_seed(0)
    phase1 = tp.train_source_phase(model, data["source_train"], data["source_valid"], HYPER,
                                   generator, CPU)
    joint = tp.bridge(model)
    return model, data, generator, phase1, joint


def _adapt(model, data, generator, phase1, joint, **kwargs):
    return tp.adapt_target_phase(model, phase1, joint["frozen_state"], data["source_train"],
                                 data["target_train"], data["target_valid"], HYPER, generator,
                                 CPU, **kwargs)


def _equal_states(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------ ablation (two_phase)


def test_the_term_selection_is_validated():
    assert tuple(tp.TERM_SELECTIONS) == mech.ABLATIONS
    assert tp.TERM_SELECTIONS["both terms"] == (tp.LAMBDA_GLOBAL, tp.LAMBDA_LOCAL)
    assert tp.TERM_SELECTIONS["global term only"] == (tp.LAMBDA_GLOBAL, 0.0)
    assert tp.TERM_SELECTIONS["local term only"] == (0.0, tp.LAMBDA_LOCAL)
    for wrong in ("global", "both", "no terms", "", None):
        with pytest.raises(tp.UnknownTermSelection):
            tp.term_coefficients(wrong)
    model, data, generator, phase1, joint = _through_bridge()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(tp.UnknownTermSelection):
        _adapt(model, data, generator, phase1, joint, term_selection="local")
    assert _equal_states(model.state_dict(), before)


def test_the_default_term_selection_leaves_every_caller_unchanged(monkeypatch):
    model, data, generator, phase1, joint = _through_bridge()
    # One step: the same objective and the same gradient as 5395416 (dropout draws from
    # the global generator, so both calls start from the same seed).
    source, target = data["source_train"][:4], data["target_train"][:4]
    torch.manual_seed(11)
    new = tp.phase_two_step(model, source, target, 1.0, CPU)
    torch.manual_seed(11)
    old = _reference_step(model, source, target, 1.0, CPU)
    assert torch.equal(new["objective"], old["objective"])
    params = list(model.target_trunk.parameters())
    assert all(torch.equal(a, b) for a, b in zip(
        torch.autograd.grad(new["objective"], params),
        torch.autograd.grad(old["objective"], params)))
    # A whole phase 2: the default against the 5395416 step, from the same state and seed.
    twin, twin_generator = copy.deepcopy(model), copy.deepcopy(generator)
    torch.manual_seed(12)
    current = _adapt(model, data, generator, phase1, joint)
    reference_step = tp.phase_two_step

    def old_step(model_, source_, target_, tau_, device_, term_selection=tp.BOTH_TERMS):
        assert term_selection == tp.BOTH_TERMS
        return _reference_step(model_, source_, target_, tau_, device_)

    monkeypatch.setattr(tp, "phase_two_step", old_step)
    torch.manual_seed(12)
    reference = _adapt(twin, data, twin_generator, phase1, joint)
    monkeypatch.setattr(tp, "phase_two_step", reference_step)
    assert current["history"] == reference["history"]
    assert set(current["history"][0]) == PHASE_TWO_KEYS_5395416
    assert [(r["lambda_global"], r["lambda_local"]) for r in current["history"]] == \
        [(1.0, 1.0)] * len(current["history"])
    assert _equal_states(current["checkpoint"]["target_state"],
                         reference["checkpoint"]["target_state"])
    assert {k: v for k, v in current["checkpoint"].items() if k != "target_state"} == \
        {k: v for k, v in reference["checkpoint"].items() if k != "target_state"}
    assert _equal_states(model.state_dict(), twin.state_dict())
    fresh = _through_bridge()
    torch.manual_seed(12)
    default = _adapt(*fresh)
    fresh = _through_bridge()
    torch.manual_seed(12)
    explicit = _adapt(*fresh, term_selection="both terms")
    assert explicit["history"] == default["history"]
    assert _equal_states(explicit["checkpoint"]["target_state"],
                         default["checkpoint"]["target_state"])


@pytest.mark.parametrize("selection,kept,dropped", [("global term only", "global", "local"),
                                                     ("local term only", "local", "global")])
def test_a_dropped_term_contributes_exactly_zero_to_the_gradient(selection, kept, dropped):
    model, data, _, _, _ = _through_bridge()
    step = tp.phase_two_step(model, data["source_train"][:4], data["target_train"][:4], 1.0,
                             CPU, term_selection=selection)
    params = list(model.target_trunk.parameters())
    assert torch.equal(step["objective"], step[kept])
    assert step[dropped].requires_grad and math.isfinite(float(step[dropped].detach()))
    alone = torch.autograd.grad(step[kept], params, retain_graph=True)
    total = torch.autograd.grad(step["objective"], params, retain_graph=True)
    assert all(torch.equal(a, b) for a, b in zip(alone, total))
    dropped_grad = torch.autograd.grad(step[dropped], params, allow_unused=True)
    assert any(g is not None and bool(g.abs().sum() > 0) for g in dropped_grad)


def test_the_dropped_terms_value_is_still_recorded_for_the_bound_control():
    model, data, generator, phase1, joint = _through_bridge()
    out = _adapt(model, data, generator, phase1, joint, term_selection="global term only")
    assert [(r["lambda_global"], r["lambda_local"]) for r in out["history"]] == \
        [(1.0, 0.0)] * len(out["history"])
    for r in out["history"]:
        assert set(r) == PHASE_TWO_KEYS_5395416
        assert 0.0 <= r["local_min"] <= r["local_max"] <= 1.0
        assert r["loss"] == pytest.approx(r["global"])


# ------------------------------------------------------------------ ablation (mechanisms)


def test_the_both_terms_ablation_is_full_mil_creda():
    data = _data()
    record = _phase_one_record(data)
    full = search.train_mil_creda_phase_two(data, 0, 1, record, P2, SCHEDULE)
    both = mech.train_ablation_variant(data, 0, 1, record, P2, SCHEDULE, "both terms")
    assert both["result"]["history"] == full["result"]["history"]
    assert _equal_states(both["model"].state_dict(), full["model"].state_dict())


def test_the_ablation_starts_from_full_mil_credas_phase_one_and_its_hyperparameters():
    data = _data()
    state, meta = _phase_one_record(data)
    out = mech.train_ablation_variant(data, 0, 1, (state, meta), P2, SCHEDULE,
                                      "local term only")
    frozen = tp.phase_one_state(out["model"])
    assert _equal_states(frozen, {k: v for k, v in state.items()
                                  if not k.startswith("target_trunk.")})
    assert out["model"].sigma == meta["sigma"]
    assert out["hyper"] == {**{k: meta["hyper"][k] for k in tp.PHASE_ONE_KEYS}, **P2,
                            **SCHEDULE}
    assert out["variant"] == "local term only" and out["term_selection"] == "local term only"
    assert {h["model"] for h in out["history"]} == {"MIL-CREDA [local term only]:phase2"}
    assert {(h["lambda_global"], h["lambda_local"]) for h in out["history"]} == {(0.0, 1.0)}
    with pytest.raises(mech.MechanismError, match="search of its own"):
        mech.train_ablation_variant(data, 0, 1, (state, meta), {**P2, "lr3": 1.0}, SCHEDULE,
                                    "local term only")
    with pytest.raises(mech.MechanismError, match="MIL-CREDA"):
        mech.train_ablation_variant(data, 0, 1, (state, {**meta, "name": "common"}), P2,
                                    SCHEDULE, "local term only")
    with pytest.raises(mech.MechanismError, match="rotation"):
        mech.train_ablation_variant(data, 0, 1, (state, {**meta, "rotation": 2}), P2,
                                    SCHEDULE, "local term only")
    with pytest.raises(mech.MechanismError, match="sigma"):
        mech.train_ablation_variant(data, 0, 1, (state, {**meta, "sigma": None}), P2,
                                    SCHEDULE, "local term only")
    with pytest.raises(tp.UnknownTermSelection):
        mech.train_ablation_variant(data, 0, 1, (state, meta), P2, SCHEDULE, "no terms")


# ------------------------------------------------------------------ aggregation


def _reference_mil_creda(source_channels: int, target_channels: int) -> dict:
    """MILCREDAModel.__init__ at 5395416, rebuilt step by step from the same seed."""
    torch.manual_seed(7)
    source, target = eegnet.EEGNetTrunk(source_channels), eegnet.EEGNetTrunk(target_channels)
    d = source.output_dim
    V_R = torch.empty(ATTENTION_WIDTH, d, dtype=torch.float64)
    b_R = torch.zeros(ATTENTION_WIDTH, dtype=torch.float64)
    v_R = torch.empty(ATTENTION_WIDTH, dtype=torch.float64)
    torch.nn.init.xavier_uniform_(V_R)
    torch.nn.init.normal_(v_R, std=0.1)
    project_onto_l1_ball(v_R)
    head = eegnet.EEGNetHead(d, 2)
    return {**{f"source_trunk.{k}": v for k, v in source.state_dict().items()},
            **{f"target_trunk.{k}": v for k, v in target.state_dict().items()},
            "V_R": V_R, "b_R": b_R, "v_R": v_R,
            **{f"head.{k}": v for k, v in head.state_dict().items()}}


def test_the_default_model_is_unchanged():
    reference = _reference_mil_creda(3, 2)
    torch.manual_seed(7)
    model = MILCREDAModel(3, 2, attention_gamma=0.5, attention_temperature=2.0)
    assert _equal_states(model.state_dict(), reference)
    assert not isinstance(model, av.AggregationVariantModel)
    model.fix_sigma(1.3)
    bag = _bags(1, 3, 9, "x")[0].windows
    model.eval()
    H = model.embed([bag], "source")[0]
    logits = relevance_logits(H, model.V_R, model.b_R, model.v_R, 0.5, 1.3)
    expected = model.head(bag_embedding(H, bag_weights(logits, 2.0))[None])
    assert torch.equal(model.predict([bag], "source"), expected)


def _variant(aggregation: str, seed: int = 7):
    torch.manual_seed(seed)
    model = av.build(aggregation, 3, 2, av.variant_hyper(aggregation, HYPER))
    model.fix_sigma(1.0)
    return model.eval()


def test_each_aggregation_is_its_published_form():
    assert av.VARIANTS == tuple(mech.AGGREGATIONS[1:]) == (
        "own without consensus", "published attention", "gated attention", "mean")
    assert tuple(av.FORMS) == ("published attention", "gated attention", "mean")
    published = _variant("published attention")
    D = published.source_trunk.output_dim
    H = torch.randn(5, D, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    a = torch.softmax(published.w @ torch.tanh(published.V @ H.T), dim=1).reshape(-1)  # Eq. 8
    assert torch.allclose(published.weights(H), a)
    assert torch.allclose(published.represent([(H, a)])[0], (a[:, None] * H).sum(0))   # Eq. 7
    gated = _variant("gated attention")
    g = torch.softmax(gated.w @ (torch.tanh(gated.V @ H.T) * torch.sigmoid(gated.U @ H.T)),
                      dim=1).reshape(-1)                                                 # Eq. 9
    assert torch.allclose(gated.weights(H), g)
    mean = _variant("mean")
    assert torch.allclose(mean.represent(mean.bag_pairs([H]))[0], H.mean(dim=0))         # 6
    for model in (published, gated, mean):
        w = model.weights(H)
        assert w.shape == (5,) and torch.all(w >= 0) and float(w.sum().detach()) == pytest.approx(1.0)
    for model in (published, gated):
        assert not any(name in dict(model.named_parameters()) for name in ("b_V", "b_U"))
    for name in ("own", "max", "own without consensus"):
        with pytest.raises(av.UnknownAggregation):
            av.AggregationVariantModel(3, 2, aggregation=name)
    assert not hasattr(av, "MAX_WEIGHTS_RULE")


def test_the_own_without_consensus_is_mil_creda_at_gamma_zero():
    hyper = av.variant_hyper("own without consensus", HYPER)
    assert hyper == {**HYPER, "attentionGamma": 0.0}
    assert av.not_applicable("own without consensus") == ()
    model = _variant("own without consensus")
    assert type(model) is MILCREDAModel
    assert model.attention_gamma == 0.0
    assert model.attention_temperature == P1["attentionTemperature"]
    torch.manual_seed(7)
    full = MILCREDAModel(3, 2, attention_gamma=P1["attentionGamma"],
                         attention_temperature=P1["attentionTemperature"])
    assert _equal_states(model.state_dict(), full.state_dict())   # bias, constraint, width
    H = torch.randn(5, model.source_trunk.output_dim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(3))
    logits = relevance_logits(H, model.V_R, model.b_R, model.v_R, 0.0, 1.0)
    assert torch.allclose(model.bag_pairs([H])[0][1],
                          bag_weights(logits, P1["attentionTemperature"]))


def test_the_variants_differ_only_in_the_aggregation():
    torch.manual_seed(7)
    own = MILCREDAModel(3, 2, attention_gamma=0.5, attention_temperature=1.0)
    own_shapes = {k: v.shape for k, v in own.state_dict().items()
                  if k not in ("V_R", "b_R", "v_R")}
    attention = {"own without consensus": {"V_R", "b_R", "v_R"},
                 "published attention": {"V", "w"}, "gated attention": {"V", "U", "w"},
                 "mean": set()}
    for name in av.VARIANTS:
        model = _variant(name)
        state = model.state_dict()
        assert {k: v.shape for k, v in state.items() if k not in attention[name]} == own_shapes
        for trunk in ("source_trunk", "target_trunk"):
            assert _equal_states({k: v for k, v in state.items() if k.startswith(trunk)},
                                 {k: v for k, v in own.state_dict().items()
                                  if k.startswith(trunk)})
        assert {id(p) for p in model.attention_parameters()} == \
            {id(getattr(model, k)) for k in attention[name]}
        assert model.head.__class__ is own.head.__class__


def test_the_parameters_a_variant_lacks_are_not_applicable():
    assert av.NOT_APPLICABLE == ("attentionGamma", "attentionTemperature")
    for name in av.FORMS:
        assert av.not_applicable(name) == av.NOT_APPLICABLE
        hyper = mech.aggregation_hyper(P1, P2, SCHEDULE, name)
        assert all(math.isnan(hyper[k]) for k in av.NOT_APPLICABLE)
        assert {k: hyper[k] for k in hyper if k not in av.NOT_APPLICABLE} == \
            {k: v for k, v in HYPER.items() if k not in av.NOT_APPLICABLE}
    assert mech.aggregation_hyper(P1, P2, SCHEDULE, "own without consensus") == \
        {**HYPER, "attentionGamma": 0.0}
    model = _variant("mean")
    assert model.attention_gamma is None and model.attention_temperature is None
    with pytest.raises(TypeError):
        av.AggregationVariantModel(3, 2, aggregation="mean", attention_gamma=0.5)
    with pytest.raises(mech.MechanismError, match="search of its own"):
        mech.aggregation_hyper({**P1, "extra": 1.0}, P2, SCHEDULE, "mean")


@pytest.mark.parametrize("aggregation", av.VARIANTS)
def test_an_aggregation_variant_trains_both_phases(aggregation):
    data = _data()
    out = mech.train_aggregation_variant(data, 0, 1, aggregation, P1, P2, SCHEDULE)
    assert out["variant"] == aggregation
    assert out["not_applicable"] == list(av.not_applicable(aggregation))
    if aggregation == "own without consensus":
        assert type(out["model"]) is MILCREDAModel and out["model"].attention_gamma == 0.0
        assert out["hyper"] == {**HYPER, "attentionGamma": 0.0}
    else:
        assert isinstance(out["model"], av.AggregationVariantModel)
    models = [h["model"] for h in out["history"]]
    assert set(models) == {f"MIL-CREDA [{aggregation}]:phase1",
                           f"MIL-CREDA [{aggregation}]:phase2"}
    assert models.count(f"MIL-CREDA [{aggregation}]:phase2") == SCHEDULE["epochs2"]
    sigma = out["phase1"]["sigma"]
    assert sigma["sigmaMultiplier"] == P1["sigmaMultiplier"]
    assert out["model"].sigma == sigma["sigma"] == out["result"]["checkpoint"]["sigma"]
    assert all(math.isnan(out["hyper"][k]) for k in av.not_applicable(aggregation))
    frozen = tp.phase_one_state(out["model"])
    assert _equal_states(frozen, {k: v for k, v in out["phase1"]["checkpoint"]["state"].items()})
    preds = tp.predictions(out["model"], data.rotations[0]["target_valid"], TRANSFER.name, CPU)
    assert all(0.0 <= p["probability"] <= 1.0 for p in preds)


def test_the_variant_trainers_refuse_an_unknown_variant():
    data = _data()
    with pytest.raises(av.UnknownAggregation):
        mech.train_aggregation_variant(data, 0, 1, "own", P1, P2, SCHEDULE)
    with pytest.raises(av.UnknownAggregation):
        mech.train_aggregation_variant(data, 0, 1, "max", P1, P2, SCHEDULE)
    with pytest.raises(av.UnknownAggregation):
        mech.train_aggregation_variant(data, 0, 1, "attention", P1, P2, SCHEDULE)


def test_an_aggregation_variant_resumes_from_its_stored_phase_one():
    data = _data()
    first = mech.train_aggregation_variant(data, 0, 1, "own without consensus", P1, P2,
                                           SCHEDULE)
    checkpoint = first["phase1"]["checkpoint"]
    record = (checkpoint["state"], {"phase": 1, "name": "MIL-CREDA [own without consensus]",
                                    "transfer": data.transfer.name, "rotation": 0, "seed": 1,
                                    "sigma": checkpoint["sigma"], "hyper": first["hyper"]})
    again = mech.train_aggregation_variant(data, 0, 1, "own without consensus", P1, P2,
                                           SCHEDULE, phase_one_record=record)
    assert again["phase1"] is None
    assert again["result"]["history"] == first["result"]["history"]
    assert _equal_states(again["model"].state_dict(), first["model"].state_dict())
    assert {h["model"] for h in again["history"]} == {
        "MIL-CREDA [own without consensus]:phase2"}
    with pytest.raises(mech.MechanismError, match="rotation"):
        mech.train_aggregation_variant(data, 0, 1, "own without consensus", P1, P2, SCHEDULE,
                                       phase_one_record=(record[0], {**record[1],
                                                                     "rotation": 3}))
    with pytest.raises(mech.MechanismError, match="name"):
        mech.train_aggregation_variant(data, 0, 1, "mean", P1, P2, SCHEDULE,
                                       phase_one_record=record)
