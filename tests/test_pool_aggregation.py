"""The pool aggregation ablation: its pick check, its aggregations and one short real run."""
import copy
import json

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import criteria_pool as C
from MIL_CREDA_EEG import pool_aggregation as PA
from MIL_CREDA_EEG import xpilot as XP
from MIL_CREDA_EEG.mil_creda_model import pooled_weights

UNIT = "d1p5-s1p5-r0"


def _picks(trial=4, **change):
    params = {**C.draw("MIL-CREDA", 1)[trial], **change}
    return {"source": "test", "units": {UNIT: {"trial": trial, "cef": 2.0, "BApool": 0.5, "AUCpool": 0.5,
                                                "BAtest12": 0.5, "params": params, "trialsSeen": 100}}}


def test_the_six_aggregations_in_the_table_order():
    assert PA.ALL_SIX == ("own", "own without consensus", "published attention", "gated attention",
                          "mean", "sum")
    assert PA.TERMS == ("global only", "local only")
    assert PA.ALL == PA.ALL_SIX + PA.TERMS
    assert PA.WITH_ATTENTION == (*PA.ALL_SIX[:4], *PA.TERMS)


def test_the_term_variants_zero_exactly_their_coefficient_and_nothing_else():
    p = C.draw("MIL-CREDA", 1)[4]
    assert set(C.space("MIL-CREDA")) >= {"lambda_global", "lambda_local"}
    g, l = PA.variant_params("global only", p), PA.variant_params("local only", p)
    assert g == {**p, "lambda_local": 0.0} and p["lambda_local"] != 0
    assert l == {**p, "lambda_global": 0.0} and p["lambda_global"] != 0


def test_refuses_a_pick_whose_params_differ_from_the_pool_draw(tmp_path):
    with pytest.raises(ValueError, match="differ from criteria_pool.draw"):
        PA.run(UNIT, picks=_picks(lr2=1e-3), workdir=tmp_path)
    with pytest.raises(ValueError, match="differ from criteria_pool.draw"):
        PA.run(UNIT, picks=_picks(trial=5, lr2=C.draw("MIL-CREDA", 1)[4]["lr2"]), workdir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_refuses_an_unknown_variant_or_unit(tmp_path):
    with pytest.raises(ValueError, match="variants are among"):
        PA.run(UNIT, variants=("max",), picks=_picks(), workdir=tmp_path)
    with pytest.raises(ValueError, match="no MIL-CREDA pick"):
        PA.run("d9-s9-r9", picks=_picks(), workdir=tmp_path)


def test_the_pick_check_accepts_the_pool_draw_up_to_its_last_bits():
    assert PA.checked_pick(UNIT, _picks())["params"] == C.draw("MIL-CREDA", 1)[4]
    lr2 = C.draw("MIL-CREDA", 1)[4]["lr2"]
    pick = PA.checked_pick(UNIT, _picks(lr2=lr2 * (1 + 1e-15)))
    assert pick["params"]["lr2"] == lr2 * (1 + 1e-15) and 0 < pick["drawRelDiff"] <= PA.DRAW_RTOL
    assert PA.checked_pick(UNIT)["trial"] == 34     # the shipped v82 manifest the paper ran


def test_sum_weights_every_window_one_so_the_bag_is_the_plain_sum():
    logits = torch.randn(7, dtype=torch.float64)
    w = pooled_weights(logits, 1.0, "sum")
    assert torch.equal(w, torch.ones(7, dtype=torch.float64))
    H = torch.randn(7, 5, dtype=torch.float64)
    from MIL_CREDA.attention import bag_embedding
    assert torch.allclose(bag_embedding(H, w), H.sum(0))
    assert torch.allclose(bag_embedding(H, pooled_weights(logits, 1.0, "mean")), H.mean(0))


def test_variant_params_change_only_the_aggregation():
    p = C.draw("MIL-CREDA", 1)[4]
    assert PA.variant_params("own", p) == p
    assert PA.variant_params("published attention", p) == p
    for v in ("own without consensus", "mean", "sum"):
        assert PA.variant_params(v, p) == {**p, "gamma": 0.0}


def test_the_two_families_are_never_crossed():
    p = C.draw("MIL-CREDA", 1)[4]
    assert PA.FAMILIES == {"aggregation": PA.ALL_SIX,
                           "terms": ("full", "global only", "local only", "no adaptation")}
    for v in PA.ALL_SIX:
        d = PA.check_design(v, PA.variant_params(v, p), p)
        assert d == {"family": "aggregation", "aggregation": v, "zeroedTerm": None}
        q = PA.variant_params(v, p)
        assert (q["lambda_global"], q["lambda_local"]) == (p["lambda_global"], p["lambda_local"])
    for v in PA.TERMS:
        d = PA.check_design(v, PA.variant_params(v, p), p)
        assert d["family"] == "terms" and d["aggregation"] == "own"
        assert XP.VARIANTS[PA.ON_BASE[v]][0] == {} and v not in PA.PUBLISHED
    with pytest.raises(ValueError, match="keeps both terms"):
        PA.check_design("mean", {**PA.variant_params("mean", p), "lambda_local": 0.0}, p)
    with pytest.raises(ValueError, match="sets only lambda_local"):
        PA.check_design("global only", {**PA.variant_params("global only", p), "gamma": 0.0}, p)


def test_a_published_attention_starts_from_the_base_with_seeded_attention_and_frees_only_it():
    from MIL_CREDA_EEG import ablation_pilot as AP
    from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel
    base = MILCREDAModel(4, 3, attention_gamma=0.5, attention_temperature=1.0)
    base.fix_sigma(1.5)
    a, b = (PA.variant_base("gated attention", base, 7) for _ in range(2))
    assert torch.equal(a.V, b.V) and torch.equal(a.U, b.U) and torch.equal(a.w, b.w)
    assert not torch.equal(a.V, PA.variant_base("gated attention", base, 8).V)
    for name in ("source_trunk", "target_trunk", "head"):
        sa, sb = getattr(a, name).state_dict(), getattr(base, name).state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sb)
    assert a.sigma == 1.5 and PA.variant_base("mean", base, 7) is base
    keys = AP.ATTENTION_KEYS
    with PA.attention_moves(a):
        assert AP.ATTENTION_KEYS == (*keys, "V", "U", "w")
    assert AP.ATTENTION_KEYS == keys


def test_attention_auc_convention_inverts_the_weight_for_artifacts():
    w = np.array([0.1, 0.1, 0.4, 0.4])
    art = np.array([True, True, False, False])
    ev = np.array([False, False, True, True])
    assert PA.mask_aucs(w, art, ev, label=1) == {"artifactAUC": 1.0, "eventAUC": 1.0}
    assert PA.mask_aucs(w, art, ev, label=0) == {"artifactAUC": 1.0, "eventAUC": None}
    assert PA.mask_aucs(w, np.zeros(4, bool), ev, label=1)["artifactAUC"] is None


@pytest.mark.skipif(not C.base_file(UNIT).exists(), reason=f"the frozen base {C.base_file(UNIT)} is "
                    "not in this checkout")
def test_a_short_real_run_writes_every_declared_key(tmp_path):
    out = PA.run(UNIT, variants=("own", "mean", "global only"), procs=1, threads=1, workdir=tmp_path,
                 epochs=1, eval_subjects=6)
    assert out == {"file": f"aggregation-{UNIT}.json", "variants": ["own", "mean", "global only", "UDA-DDA"]}
    rec = json.loads((tmp_path / out["file"]).read_text())
    for k in ("unitKey", "trial", "params", "drawRelDiff", "pickBApool", "reproduces", "BApoolDifference", "commit",
              "procs", "threads", "epochs2", "attentionSubjects", "families"):
        assert k in rec, k
    assert rec["unitKey"] == UNIT and rec["trial"] == 34 and rec["epochs2"] == 1
    assert rec["attentionSubjects"] == "evaluation pool"
    assert isinstance(rec["reproduces"], bool)
    agg, terms = rec["families"]["aggregation"], rec["families"]["terms"]
    assert set(agg) == {"own", "mean"} and set(terms) == {"full", "global only", "no adaptation"}
    assert set(rec["families"]) == {"aggregation", "terms", "ownPair"}
    fig = rec["figure"]["UDA-DDA"]
    assert fig["family"] == "figure" and fig["nonFinite"] is False and fig["attention"] is not None
    assert fig["params"] == PA.checked_pick(UNIT, method="UDA-DDA")["params"]
    assert rec["figurePick"]["trial"] == 43
    assert terms["full"] == {"reference": "aggregation/own"}
    own, mean, glob = agg["own"], agg["mean"], terms["global only"]
    assert (own["family"], mean["family"], glob["family"]) == ("aggregation", "aggregation", "terms")
    assert glob["aggregation"] == "own" and glob["zeroedTerm"] == "lambda_local"
    assert own["params"] == rec["params"] and glob["params"] == {**rec["params"], "lambda_local": 0.0}
    assert mean["params"] == {**rec["params"], "gamma": 0.0}
    none = terms["no adaptation"]
    assert none["trained"] is False and none["matchesBasePredict"] is True
    assert {"BApool", "AUCpool", "BAtest12"} <= set(none)
    assert glob["attention"] is not None
    for row in (own, mean, glob):
        for k in ("BApool", "AUCpool", "BAtest12", "seconds", "nonFinite", "attention"):
            assert k in row, k
        assert row["nonFinite"] is False
    assert mean["attention"] is None
    att = own["attention"]
    assert set(att) == {"subjects", "meanArtifactAUC", "meanEventAUC", "nArtifact", "nEvent"}
    assert len(att["subjects"]) == 6 and att["nArtifact"] >= 1
    assert all(0.0 <= s["artifactAUC"] <= 1.0 for s in att["subjects"] if s["artifactAUC"] is not None)


def _check_sidecars(tmp_path, rec, attention_runs):
    side = np.load(tmp_path / f"attention-{UNIT}.npz")
    n = len(side["subjects"])
    assert n == rec["evalSubjects"] and side["offsets"][-1] == len(side["artifact"]) == len(side["event"])
    assert set(side["artifactType"]) <= {"none", "blink", "muscle", "blink+muscle"}
    assert set(np.unique(side["labels"])) == {0, 1}
    for v in attention_runs:
        assert side[PA.sidecar_key(v)].shape == side["artifact"].shape
    assert PA.sidecar_key("mean") not in side.files


@pytest.mark.skipif(not C.base_file(UNIT).exists(), reason="the frozen base is not in this checkout")
def test_the_short_run_writes_the_sidecars_and_the_own_pair_states(tmp_path):
    PA.run(UNIT, variants=("own", "mean", "global only"), procs=1, threads=1, workdir=tmp_path, epochs=1,
           eval_subjects=6)
    rec = json.loads((tmp_path / f"aggregation-{UNIT}.json").read_text())
    _check_sidecars(tmp_path, rec, ("own", "global only", "UDA-DDA"))
    block = rec["families"]["ownPair"]
    assert len(block["subjects"]) == 60
    assert all(s["pair"] == "SRC-" + s["subject"].split("-", 1)[1] for s in block["subjects"])
    assert set(block["states"]) == {"start", "fullMid", "fullEnd", "globalOnlyEnd"}
    for st in block["states"].values():
        assert len(st["distances"]) == 60 and np.isclose(st["mean"], np.mean(st["distances"]))
    # at epochs=1 the mid snapshot is epoch 0, before the first step: the start model itself
    assert block["states"]["fullMid"]["snapshot"]["step"] == 0
    assert np.allclose(block["states"]["fullMid"]["distances"], block["states"]["start"]["distances"])
    pairs, umap = np.load(tmp_path / f"ownpair-{UNIT}.npz"), np.load(tmp_path / f"umap-{UNIT}.npz")
    for st in ("start", "fullEnd", "globalOnlyEnd"):
        assert pairs[f"{st}:target"].shape == pairs[f"{st}:pair"].shape == (60, pairs[f"{st}:target"].shape[1])
    for st in ("start", "fullMid", "fullEnd", "globalOnlyEnd"):
        assert umap[f"{st}:source"].shape[0] == len(umap["source:labels"]) == 60
        d = np.linalg.norm(umap[f"{st}:target"] - umap[f"{st}:pair"], axis=1)
        assert np.allclose(d, block["states"][st]["distances"])


@pytest.mark.skipif(not C.base_file("d1-s1p5-r0").exists(), reason="the frozen base is not in this checkout")
def test_the_figure_runs_only_on_the_figure_unit_and_stays_out_of_both_families(tmp_path):
    assert PA.figure_task("d1-s1p5-r0", 100) is None
    task, pick = PA.figure_task(UNIT, 100)
    assert task == ("UDA-DDA", pick["params"], C.train_seed(1, 43), 100) and pick["BApool"] == 0.9
    assert PA.design("UDA-DDA")["family"] == "figure"
    assert all("UDA-DDA" not in names for names in PA.FAMILIES.values()) and "UDA-DDA" not in PA.ALL
    out = PA.run("d1-s1p5-r0", variants=("mean",), procs=1, threads=1, workdir=tmp_path, epochs=1,
                 eval_subjects=6)
    rec = json.loads((tmp_path / out["file"]).read_text())
    assert out["variants"] == ["mean"] and rec["figure"] == {} and "figurePick" not in rec
    assert not (tmp_path / "umap-d1-s1p5-r0.npz").exists()
    assert (tmp_path / "ownpair-d1-s1p5-r0.npz").exists()
    assert set(rec["families"]["ownPair"]["states"]) == {"start"}


def test_the_figure_pick_is_refused_off_the_draw():
    picks = {"units": {UNIT: {"UDA-DDA": {"trial": 2, "params": {**C.draw("UDA-DDA", 1)[2], "lr2": 1.0},
                                          "BApool": 0.775}}}}
    with pytest.raises(ValueError, match="differ from criteria_pool.draw"):
        PA.figure_task(UNIT, 100, picks)


@pytest.mark.skipif(not C.base_file(UNIT).exists(), reason="the frozen base is not in this checkout")
def test_the_mid_snapshot_leaves_the_run_unchanged():
    s = PA._setup(UNIT, 6, None, 1)
    PA._STATE.clear(); PA._STATE.update(s)
    try:
        p = PA.checked_pick(UNIT)["params"]
        tseed = C.train_seed(1, 4)
        plain, _ = C.train("MIL-CREDA", p, s["b"], s["sides"], s["cfg"], s["dev"], tseed, s["rot"], s["name"], 2)
        with PA.mid_snapshot(lambda m: PA.own_pair_state(m, s["ownPair"]), 2, 1) as snap:
            hooked, _ = C.train("MIL-CREDA", p, s["b"], s["sides"], s["cfg"], s["dev"], tseed, s["rot"], s["name"], 2)
        assert snap["step"] == snap["perEpoch"] and snap["totalSteps"] == 2 * snap["perEpoch"]
        for a, b in zip(plain.state_dict().values(), hooked.state_dict().values()):
            assert torch.equal(a, b)
    finally:
        s["stack"].close()


def test_a_published_attention_follows_a_base_wider_than_the_default_trunk():
    from MIL_CREDA_EEG.mil_creda_model import MILCREDAModel
    narrow = MILCREDAModel(4, 3, attention_gamma=0.5, attention_temperature=1.0)
    narrow.fix_sigma(1.5)
    wide = copy.deepcopy(narrow)
    d = 2 * narrow.source_trunk.output_dim
    wide.source_trunk.output_dim = d
    for variant in ("published attention", "gated attention"):
        a = PA.variant_base(variant, wide, 7)
        assert a.V.shape[1] == d and torch.equal(a.V, PA.variant_base(variant, wide, 7).V)
        if variant == "gated attention":
            assert a.U.shape[1] == d
        n = PA.variant_base(variant, narrow, 7)
        assert n.V.shape[1] == narrow.source_trunk.output_dim
