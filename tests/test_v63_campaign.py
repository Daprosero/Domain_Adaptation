"""Plan v63 campaign units: ids, budgets, refits, resumption, deadline and label blindness."""

import dataclasses
import json
import time

import numpy as np
import pytest

from MIL_CREDA_EEG import bag_data, campaign, partition, search
from MIL_CREDA_EEG import v63_campaign as C
from MIL_CREDA_EEG import v63_setup as V

CFG = {"schedule": {"epochs1": 2, "epochs2": 2, "bagsPerStep": 4}, "nTrials": 1,
       "seeds": [1, 2], "strategies": ["MIL-CREDA"], "permutations": 10, "threads": 1,
       "marginSeconds": 0, "device": "cpu"}


def _bagset(root):
    rng = np.random.default_rng(0)
    bags = []
    for i in range(32):
        s, fold, label = f"UB{i:04d}", i % 4, ("yes", "no")[(i // 4) % 2]
        for c in {0: ("L6",), 1: ("Robots",), 2: ("L6", "Robots")}[i % 3]:
            path = root / f"{s}_{c}.npz"
            shift = 0.5 if c == "Robots" else 0.0
            marks = {"window_index": np.arange(4) * 2, "window_start_s": np.arange(4) * 4.0,
                     "integrity_mark": np.zeros(4, bool), "rail_window": np.zeros(4, bool)}
            if c == "Robots":
                marks["contact_bad"] = np.eye(4, 12, dtype=bool)
            else:
                marks["event_window"] = np.array([True, False, False, True])
            np.savez(path, signal=(rng.normal(size=(4, 12, 512)) + shift).astype(np.float32),
                     **marks)
            bags.append(bag_data.Bag(subject=s, condition=c, record_id=f"{s}-{c}", label=label,
                                     sex="M", fold=fold, device="CGX", channels=V.SHARED,
                                     n_windows=4, path=path))
    rot = {str(r): partition.rotation_roles(r) for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256="x" * 64, rotations=rot)


class _Cov:
    def of(self, bags):
        return {"ages": [10.0] * len(bags), "frontalAmplitude": [1.0] * len(bags)}


def _perf(tmp_path, product, bags, monkeypatch=None):
    cfg = C.check_config(CFG)
    campaign.register(cfg["strategies"])
    with V.installed():
        return C.run_performance(V.transfer("L6->Robots"), 0, cfg, product, bags,
                                 V.Prepared(bags), _Cov(), None)


def test_the_unit_ids_are_s0_domain_f_transfer_and_p_transfer_rotation():
    ids = C.all_unit_ids()
    assert len(ids) == 5 + 16 + 64
    assert C.parse_unit("S0:L1") == ("S0", "L1")
    assert C.parse_unit("P:L6->Robots:3")[2] == 3
    for bad in ("S0:L2", "P:L6->Robots:4", "X:L1", "F:L1->L6"):
        with pytest.raises(V.V63Error):
            C.parse_unit(bad)


def test_every_study_has_twenty_trials_and_the_same_budget():
    cfg = C.check_config({"schedule": {"epochs1": 100, "epochs2": 100, "bagsPerStep": 8}})
    assert cfg["nTrials"] == 20 and cfg["seeds"] == list(range(1, 11))
    assert cfg["strategies"] == list(C.STRATEGIES) and cfg["permutations"] == 1000


def test_the_phase_two_search_starts_from_the_first_report_seeds_checkpoint():
    assert C.REPORT_SEEDS[0] == search.PHASE_ONE_SEED == 1
    assert C.SEARCH_SEED not in C.REPORT_SEEDS


def test_each_seed_refits_from_its_own_phase_one_checkpoint_with_the_frozen_values(tmp_path):
    bags = _bagset(tmp_path)
    product = tmp_path / "P"
    _perf(tmp_path, product, bags)
    runs = product / C.RESULTS / "C-dispositivo" / "runs" / "L6-to-Robots" / "outer0"
    recs = {p.name: json.loads(p.read_text()) for p in runs.glob("*.json")}
    methods = {"predict", "phase1-MIL-CREDA", "MIL-CREDA", "MIL-CREDA-global-term-only",
               "MIL-CREDA-local-term-only", "MIL-CREDA-own-without-consensus",
               "ceiling-own-domain-model", "ceiling-phase-2-with-target-labels"}
    assert set(recs) == {f"{m}__seed{s}.json" for m in methods for s in (1, 2)}
    assert recs["MIL-CREDA__seed1.json"]["selection"] == recs["MIL-CREDA__seed2.json"]["selection"]
    m1 = C.model_path(product, V.transfer("L6->Robots"), 0, 1, 1, "MIL-CREDA")
    m2 = C.model_path(product, V.transfer("L6->Robots"), 0, 2, 1, "MIL-CREDA")
    s1 = json.loads(m1.with_suffix(".json").read_text())
    s2 = json.loads(m2.with_suffix(".json").read_text())
    assert s1["state_sha256"] != s2["state_sha256"] and s1["epochs"] == s2["epochs"]
    r = recs["MIL-CREDA__seed1.json"]
    assert len(r["probabilities"]) == len(r["subjects"]) == len(r["targetRepresentations"])
    assert 0.0 <= r["shiftGate"] <= 1.0


def test_predict_is_the_common_phase_one_model_on_the_target(tmp_path):
    bags = _bagset(tmp_path)
    product = tmp_path / "P"
    _perf(tmp_path, product, bags)
    rec = json.loads((product / C.RESULTS / "C-dispositivo" / "runs" / "L6-to-Robots" / "outer0"
                      / "predict__seed1.json").read_text())
    assert rec["phase"] == 1 and rec["group"] == "common"


def test_a_finished_refit_is_never_trained_again(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    product = tmp_path / "P"
    _perf(tmp_path, product, bags)

    def boom(*a, **k):
        raise AssertionError("trained again")

    monkeypatch.setattr(V, "train_phase_one", boom)
    monkeypatch.setattr(V, "train_mil_creda_phase_two", boom)
    _perf(tmp_path, product, bags)


def test_a_cut_session_writes_partial_with_the_ids_it_was_sent(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    monkeypatch.setattr(campaign, "load_bags", lambda cfg: bags)
    monkeypatch.setattr(C, "Covariates", lambda root: _Cov())
    sent = ["P:L6->Robots:0", "F:L6->Robots"]
    env = {"FORGE_RUN_UNITS": json.dumps(sent), "FORGE_SESSION_DEADLINE": str(time.time() - 1)}
    out = C.run(CFG, tmp_path / "P", workdir=tmp_path / "w", environ=env)
    marker = json.loads((tmp_path / "w" / "partial.json").read_text())
    assert marker["remaining"] == sent == out["remaining"] and out["ran"] == []


def test_no_target_label_enters_a_selection(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    a = _perf(tmp_path, tmp_path / "A", bags)
    real = V.search_data

    def flipped(t, b, prep=None, rotations=None):
        d = real(t, b, prep, rotations)
        flip = lambda xs: [dataclasses.replace(x, label=1 - x.label) for x in xs]
        rot = {r: {**v, "target_train": flip(v["target_train"]),
                   "target_valid": flip(v["target_valid"])} for r, v in d.rotations.items()}
        inner = {k: {**v, "target_train": flip(v["target_train"]),
                     "target_valid": flip(v["target_valid"])} for k, v in d.inner.items()}
        return dataclasses.replace(d, rotations=rot, inner=inner)

    monkeypatch.setattr(V, "search_data", flipped)
    b = _perf(tmp_path, tmp_path / "B", bags)
    assert a["shiftGateAuc"] == b["shiftGateAuc"]
    pick = lambda root, m: json.loads((root / C.RESULTS / "C-dispositivo" / "runs" / "L6-to-Robots"
                                       / "outer0" / f"{m}__seed1.json").read_text())["selection"]
    for m in ("predict", "MIL-CREDA"):
        sa, sb = pick(tmp_path / "A", m), pick(tmp_path / "B", m)
        assert (sa["params"], sa["epochs"]) == (sb["params"], sb["epochs"])


def test_a_diverged_trial_fails_inside_the_same_budget(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)
    cfg = C.check_config({**CFG, "nTrials": 2})
    data = V.search_data(V.transfer("L6->Robots"), bags, rotations=[0])
    real, calls = V.train_phase_one, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FloatingPointError("nonfinite")
        return real(*a, **k)

    monkeypatch.setattr(V, "train_phase_one", flaky)
    with V.installed():
        study = C.phase_one_study("common", data, 0, cfg, tmp_path / "s.db")
    states = [t.state.name for t in study.trials]
    assert len(states) == 2 and states.count("FAIL") == 1 and states.count("COMPLETE") == 1


def test_a_study_whose_trials_all_diverged_leaves_failed_records(tmp_path, monkeypatch):
    bags = _bagset(tmp_path)

    def diverge(*a, **k):
        raise FloatingPointError("nonfinite MIL-CREDA objective")

    monkeypatch.setattr(V, "train_mil_creda_phase_two", diverge)
    _perf(tmp_path, tmp_path / "P", bags)
    runs = tmp_path / "P" / C.RESULTS / "C-dispositivo" / "runs" / "L6-to-Robots" / "outer0"
    for seed in (1, 2):
        rec = json.loads((runs / f"MIL-CREDA__seed{seed}.json").read_text())
        assert "diverged" in rec["failed"] and "probabilities" not in rec
    assert (runs / "predict__seed1.json").is_file()



def _recs(product):
    runs = product / C.RESULTS / "C-dispositivo" / "runs" / "L6-to-Robots" / "outer0"
    return {p.name.rsplit("__", 1)[0]: json.loads(p.read_text()) for p in runs.glob("*__seed1.json")}


def test_the_k3_variants_refit_mil_credas_selection_on_the_core_transfers_only(tmp_path):
    bags = _bagset(tmp_path)
    _perf(tmp_path, tmp_path / "P", bags)
    recs = _recs(tmp_path / "P")
    for v in ("MIL-CREDA-global-term-only", "MIL-CREDA-local-term-only",
              "MIL-CREDA-own-without-consensus"):
        assert recs[v]["selection"] == recs["MIL-CREDA"]["selection"]
        assert recs[v]["experiment"] == "K3-ablacion"
    hpo = {p.name for p in (tmp_path / "P" / C.RESULTS / "hpo").glob("*.db")}
    assert not any("term-only" in h or "consensus" in h for h in hpo)
    assert all(t in V.CORE for t in V.TRANSFERS if t.name == "L6->Robots")
    assert V.transfer("L1->Robots") not in V.CORE


def test_the_ceilings_choose_nothing_and_use_the_folds_tuned_values(tmp_path):
    bags = _bagset(tmp_path)
    _perf(tmp_path, tmp_path / "P", bags)
    recs = _recs(tmp_path / "P")
    own, lab = recs["ceiling-own-domain-model"], recs["ceiling-phase-2-with-target-labels"]
    assert own["choosesNothing"] and lab["choosesNothing"]
    assert own["selection"] == recs["phase1-MIL-CREDA"]["selection"]
    assert lab["selection"] == recs["MIL-CREDA"]["selection"]
    hpo = sorted(p.name for p in (tmp_path / "P" / C.RESULTS / "hpo").glob("*.db"))
    assert len(hpo) == 2 + 1          # common and MIL-CREDA phase 1, MIL-CREDA phase 2


def test_every_refit_record_keeps_what_the_mechanism_analyses_read(tmp_path):
    bags = _bagset(tmp_path)
    _perf(tmp_path, tmp_path / "P", bags)
    recs = _recs(tmp_path / "P")
    for name, r in recs.items():
        assert len(r["windows"]) == len(r["attention"]) == len(r["subjects"])
        for w, a in zip(r["windows"], r["attention"]):
            assert len(w["index"]) == len(a) and abs(sum(a) - 1) < 1e-6
            assert w["contactBad"] is not None and w["event"] is None    # EMOTIV target
        assert len(r["sourceRepresentations"]) == len(r["sourceSubjects"])
    pi = recs["MIL-CREDA"]["correspondence"]
    assert len(pi) == len(recs["MIL-CREDA"]["subjects"])
    assert all(abs(sum(row) - 1) < 1e-6 for row in pi)
    assert len(pi[0]) == len(recs["MIL-CREDA"]["correspondenceSourceSubjects"])
    assert "correspondence" not in recs["predict"]
