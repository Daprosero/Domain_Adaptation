"""xcompare: the eight comparators on xpilot's units, from MIL-CREDA's shared phase-1 base.

Every method shares the subjects, the searched phase 1, the base model per seed and the cut;
each comparator's phase-2 search is scored by phase2_select.centroid_score only, without a
target label and without the target test bags.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import campaign, phase2_select, remote_units, spectral_trunk, two_phase
from MIL_CREDA_EEG import phase2_trace as P
from MIL_CREDA_EEG import xcompare as XC
from MIL_CREDA_EEG import xpilot as XP

RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"
DEVICE = torch.device("cpu")


def _raw_config():
    return json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]


class Forbidden(list):
    """A bag list the search must never open."""

    def _refuse(self, *a, **k):
        raise AssertionError("the phase-2 search touched the target test bags")

    __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = _refuse


class NoLabel:
    """A target bag whose label the search must never read."""

    def __init__(self, bag):
        object.__setattr__(self, "_bag", bag)

    def __getattr__(self, name):
        if name == "label":
            raise AssertionError("the phase-2 search read a target label")
        return getattr(object.__getattribute__(self, "_bag"), name)


def _run(tmp_path, monkeypatch, setting, spy=None):
    monkeypatch.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
    monkeypatch.setitem(P.PHASE_TWO, "epochs2", P.PHASE_TWO["epochs2"])     # restored after
    if spy is not None:
        monkeypatch.setattr(XC, "search_phase2", spy)
    XC.smoke(_raw_config(), experiment="sim", rotation=0, setting=setting, shift=1.0, n_subjects=20,
             trials=1, trials2=1, epochs2=1, workdir=tmp_path, environ={"FORGE_RUN_MODE": "smoke"})
    (path,) = glob.glob(str(tmp_path / "xcompare-*.json"))
    out = json.loads(Path(path).read_text())
    out["_stamp"] = json.loads((tmp_path / remote_units.STAMP_FILE).read_text())
    return out


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    calls = {}
    real = XC.search_phase2

    def spy(method, prepared, context, *, source, source_test, target, **kw):
        calls.setdefault(kw["unit"], []).append(
            {"method": method, "target": [b.subject for b in target],
             "sourceTest": [b.subject for b in source_test]})
        return real(method, prepared, context, source=source, source_test=source_test, target=target, **kw)

    out = {}
    with pytest.MonkeyPatch.context() as mp:
        for setting in ("subject", "device"):
            out[setting] = _run(tmp_path_factory.mktemp(setting), mp, setting, spy)
    out["_calls"] = calls
    return out


@pytest.mark.parametrize("setting", ["subject", "device"])
def test_every_method_shares_subjects_base_and_cut(outputs, setting):
    out = outputs[setting]
    assert set(out["methods"]) == set(XC.ALL8)
    base = out["base"]["1"]
    ran = {m: r for m, r in out["methods"].items() if "runs" in r}
    assert ran, "no comparator ran"
    for m, rec in ran.items():
        run = rec["runs"]["1"]
        assert run["cut"] == base["cut"]
        assert run["predict"] == base["predict"]                       # one base model, one predict
        assert [r[0] for r in run["final"]["rows"]] == [r[0] for r in base["predict"]["rows"]]
    split = out["subjectsSplit"]
    assert not set(split["targetTest"]) & set(split["targetAdapt"])
    assert not set(split["sourceTrain"]) & set(split["sourceTest"])
    assert out["phase1Search"]["trials"] == 1
    assert base["commonWrapperEquality"]["maxAbsProbability"] < 1e-9   # the wrapper is the base


@pytest.mark.parametrize("setting", ["subject", "device"])
def test_output_has_ba_auc_kappa_per_method_and_seed_and_reasons_for_skips(outputs, setting):
    out = outputs[setting]
    for m, rec in out["methods"].items():
        if "skipped" in rec or "failed" in rec:
            assert isinstance(rec.get("skipped") or rec.get("failed"), str) and len(
                rec.get("skipped") or rec.get("failed")) > 20
            continue
        s = rec["phase2Search"]
        assert s["trials"] == 1 and set(s["params"]) == set(s["space"]) and 0.0 <= s["score"] <= 1.0
        assert s["rule"] == XC.SELECTION_RULE
        for snap in ("predict", "final"):
            metrics = rec["runs"]["1"][snap]
            assert {"BA", "AUC", "kappa", "rows"} <= set(metrics)
            assert all(len(r) == 2 for r in metrics["rows"])
        assert rec["seconds"]["search"] >= 0 and "1" in rec["seconds"]["seeds"]
    assert out["methods"]["AdaptMIL"]["skipped"].startswith("AdaptMIL")
    failed = {m: r["failed"] for m, r in out["methods"].items() if "failed" in r}
    assert not failed, failed
    assert out["_stamp"]["mode"] == "smoke" and out["_stamp"]["runs"]["ran"] == [out["unit"]]


def test_search_receives_only_the_adapt_half_and_the_source_test(outputs):
    for setting in ("subject", "device"):
        out = outputs[setting]
        calls = outputs["_calls"][out["unit"]]
        split = out["subjectsSplit"]
        assert {c["method"] for c in calls} == {m for m, r in out["methods"].items() if "runs" in r}
        for c in calls:
            assert sorted(set(c["target"])) == split["targetAdapt"]
            assert not set(c["target"]) & set(split["targetTest"])
            assert sorted(set(c["sourceTest"])) == split["sourceTest"]


def test_search_never_reads_a_target_label_or_the_target_test(monkeypatch):
    """Direct call: target bags that refuse their label, a target test that refuses any access
    in the context, and the centroid rule as the only score."""
    monkeypatch.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
    cfg = remote_units.resolve_inputs(campaign.check_config(_raw_config()), environ={})
    campaign.register(list(XC.ALL8))
    remote_units.set_deterministic()
    built = XP.build_sides(cfg, {}, experiment="sim", rotation=0, setting="device", shift=1.0, n_subjects=20)
    sides, cfg = built["sides"], built["cfg"]
    scored = []
    real = phase2_select.centroid_score

    def score(t_reps, t_pseudo, s_reps, s_labels):
        scored.append((len(t_reps), list(map(int, s_labels))))
        return real(t_reps, t_pseudo, s_reps, s_labels)

    monkeypatch.setattr(XC.phase2_select, "centroid_score", score)
    with spectral_trunk.installed(), XP.window_samples(built["samples"]):
        params1, _ = XP.search_phase1(sides, cfg, DEVICE, 1, 0)
        shared = XC.shared_base(sides, cfg, DEVICE, 1, 0, params1, built["unit"])
        target = [NoLabel(b) for b in sides["target"]]
        sides["target_test"] = Forbidden()           # nothing after the base may open it
        for method in ("UDA-DDA", "CLUST — variante source-free", "COT"):
            res = XC.search_phase2(method, shared["prepared"], shared["context"], source=sides["source"],
                                   source_test=sides["source_test"], target=target, cfg=cfg, device=DEVICE,
                                   trials2=1, rotation=0, unit=built["unit"], epochs2=1)
            assert set(res["params"]) == set(res["space"])
    assert len(scored) == 3
    for n, labels in scored:
        assert n == len(sides["target"])
        assert labels == [int(b.label) for b in sides["source_test"]]


def test_common_wrapper_restores_trunk_width_and_temperature_even_on_error():
    from MIL_CREDA_EEG import eegnet, mil_wrapper
    before = (eegnet.EEGNetTrunk, mil_wrapper.COMMON_TEMPERATURE, spectral_trunk.WIDTH)
    with pytest.raises(RuntimeError):
        with XC.common_wrapper(32, 7.5):
            assert eegnet.EEGNetTrunk is XC.StandardizedSpectralTrunk
            assert mil_wrapper.COMMON_TEMPERATURE == 7.5 and spectral_trunk.WIDTH == 32
            raise RuntimeError("boom")
    assert (eegnet.EEGNetTrunk, mil_wrapper.COMMON_TEMPERATURE, spectral_trunk.WIDTH) == before
