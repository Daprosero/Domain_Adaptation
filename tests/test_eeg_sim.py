"""Simulated EEG (eeg_sim): calibrated class effect, artifact and event masks, paired
identities across devices, and xpilot's sim branch writing masks and attention weights."""
from __future__ import annotations

import contextlib
import glob
import json
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_EEG import eeg_sim as E
from MIL_CREDA_EEG import phase2_trace
from MIL_CREDA_EEG import xpilot as XP

CLEAN = E.device("1020", 0.0, 0)
RUN_CONFIG = Path(__file__).resolve().parents[1] / "tools/kaggle/sm-dev-e3-s15-r0/run-config.json"


# ------------------------------------------------------------------ calibrated effect


def test_calibration_engine_equals_tbr_effect_size_of_generated_bags():
    for fraction in (0.0, 0.3):
        cal = E._Calibration(CLEAN, 30, 5, "Fz", fraction, E.EVENT_OUTSIDE)
        for effect in (0.0, 0.2, 1.0):
            bags = E.domain("C", 30, CLEAN, effect, seed=5, event_fraction=fraction)
            assert cal.d(effect, paired=False) == pytest.approx(E.tbr_effect_size(bags), abs=1e-6)


def test_tbr_effect_size_falls_back_to_f3_f4_on_emotiv():
    assert E.tbr_channels("emotiv") == [E.EMOTIV.index("F3"), E.EMOTIV.index("F4")]
    assert E.tbr_channels("1020") == [E.TEN_TWENTY.index("Fz")]
    emo = E.device("emotiv", 1.0, 11)
    cal = E._Calibration(emo, 20, 2, "Fz", 0.0, E.EVENT_OUTSIDE)
    assert cal.d(0.5, paired=False) == pytest.approx(
        E.tbr_effect_size(E.domain("C", 20, emo, 0.5, seed=2)), abs=1e-6)


def test_generation_is_reproducible():
    a = E.domain("R", 4, CLEAN, 0.3, seed=9)
    b = E.domain("R", 4, CLEAN, 0.3, seed=9)
    assert all(np.array_equal(x.windows.numpy(), y.windows.numpy()) for x, y in zip(a, b))


@pytest.mark.parametrize("target", (0.3, 0.6))
def test_calibrated_d_holds_on_an_independent_sample(target):
    effect = E.effect_for_d(target)                       # default: 1000 subjects, seed 0
    assert E.calibration_d(effect) == pytest.approx(target, abs=1e-3)
    # an independent 1000-subject sample, labelled as `domain` labels it (alternating classes)
    fresh = E._calibration("1020", 0.0, 0, 1000, 1, "Fz", 0.0, E.EVENT_OUTSIDE)
    assert abs(fresh.d(effect, paired=False) - target) <= 0.15


def test_effect_for_d_is_monotone_and_zero_at_zero():
    effects = [E.effect_for_d(d) for d in (0.0, 0.3, 0.6, 0.75)]
    assert effects[0] == 0.0 and effects == sorted(effects) and len(set(effects)) == 4


# ------------------------------------------------------------------ artifacts


def test_artifact_masks_have_one_entry_per_window_and_follow_the_rate():
    rate = 4.0
    bags = E.domain("A", 60, CLEAN, 0.2, seed=3, artifact_rate=rate)
    masks = E.masks_of(bags)
    assert set(masks) == {b.subject for b in bags}
    for b in bags:
        m = masks[b.subject]
        assert len(m["artifact"]) == len(m["artifactType"]) == len(m["event"]) == b.windows.shape[0] == E.WINDOWS
        assert all(t == "none" for t, f in zip(m["artifactType"], m["artifact"]) if not f)
        assert all(t != "none" for t, f in zip(m["artifactType"], m["artifact"]) if f)
    counts = [len(E.SIM_MASKS[b.subject]["artifactEvents"]) for b in bags]
    per_minute = np.mean(counts) / (E.SPAN / E.RATE / 60.0)
    assert 0.5 * rate <= per_minute <= 1.5 * rate
    kinds = {e["type"] for b in bags for e in E.SIM_MASKS[b.subject]["artifactEvents"]}
    assert kinds == {"blink", "muscle"}


def test_blinks_are_frontal_and_muscle_is_temporal():
    w = E.blink_weights("1020")
    names = list(E.TEN_TWENTY)
    assert w[names.index("Fp1")] == pytest.approx(1.0) and w[names.index("Fp1")] > w[names.index("Fz")] > w[names.index("Pz")]
    rng = np.random.default_rng(0)
    x, events = E.artifacts(rng, "1020", 30.0)
    power = (x ** 2).sum(1)
    for ch in ("O1", "O2", "Pz", "P3", "P4"):              # amplitude under 5% of the peak channel
        assert np.sqrt(power[names.index(ch)] / power.max()) < 0.05
    assert min(power[names.index("T7")], power[names.index("T8")]) > power[names.index("C3")]
    assert {e["type"] for e in events} == {"blink", "muscle"}


def test_artifacts_do_not_depend_on_the_label_by_default():
    for seed in range(5):
        E.subject_bag("L0", 0, CLEAN, 0.5, False, seed, artifact_rate=6.0)
        E.subject_bag("L1", 1, CLEAN, 0.5, False, seed, artifact_rate=6.0)
        assert E.SIM_MASKS["L0"]["artifactEvents"] == E.SIM_MASKS["L1"]["artifactEvents"]
        assert E.SIM_MASKS["L0"]["artifact"] == E.SIM_MASKS["L1"]["artifact"]


def test_artifact_class_corr_raises_class_one_rate():
    bags = E.domain("K", 40, CLEAN, 0.2, seed=4, artifact_rate=4.0, artifact_class_corr=0.8)
    rate = {c: np.mean([E.SIM_MASKS[b.subject]["artifactRate"] for b in bags if b.label == c]) for c in (0, 1)}
    assert rate[1] > 3 * rate[0]


def test_no_artifacts_by_default():
    bags = E.domain("N", 4, CLEAN, 0.2, seed=1)
    assert all(not any(E.SIM_MASKS[b.subject]["artifact"]) for b in bags)
    assert all(not any(E.SIM_MASKS[b.subject]["event"]) for b in bags)


# ------------------------------------------------------------------ events


def test_event_mask_marks_where_the_class_effect_is_concentrated():
    bags = E.domain("V", 20, CLEAN, 0.5, seed=6, event_fraction=0.25)
    ratio = {0: [], 1: []}
    for b in bags:
        m = E.SIM_MASKS[b.subject]
        assert len(m["event"]) == E.WINDOWS and sum(m["event"]) >= round(0.25 * E.WINDOWS)
        ev = np.asarray(m["event"])
        idx = E.tbr_channels("1020")
        t_in = E.subject_tbr(b.windows.numpy()[ev], idx)
        t_out = E.subject_tbr(b.windows.numpy()[np.asarray(m["eventCover"]) == 0], idx)
        ratio[b.label].append(t_in / t_out)
    assert np.mean(ratio[1]) > 1.5 and abs(np.mean(ratio[0]) - 1.0) < 0.25


# ------------------------------------------------------------------ paired identity


def test_paired_identity_is_shared_across_devices():
    src = E.domain("SRC", 6, CLEAN, 0.3, seed=0, identity_seed=7)
    tgt = E.domain("TGT", 6, E.device("emotiv", 1.0, 11), 0.3, seed=1, identity_seed=7)
    for s, t in zip(src, tgt):
        assert s.subject.split("-")[-1] == t.subject.split("-")[-1] and s.label == t.label
        assert E.SIM_MASKS[s.subject]["identity"] == E.SIM_MASKS[t.subject]["identity"]
    unpaired = E.domain("TGT", 6, E.device("emotiv", 1.0, 11), 0.3, seed=1)
    assert all(E.SIM_MASKS[s.subject]["identity"] != E.SIM_MASKS[t.subject]["identity"]
               for s, t in zip(src, unpaired))


# ------------------------------------------------------------------ xpilot


def _tiny(tmp_path, monkeypatch, **kw):
    monkeypatch.setitem(XP.SS.SPACE, "epochs", ("cat", (2,)))
    monkeypatch.setitem(phase2_trace.PHASE_TWO, "epochs2", 1)
    cfg = json.load(open(RUN_CONFIG))["run"]["kwargs"]["config"]
    XP.smoke(cfg, experiment="sim", rotation=0, trials=1, epochs2=1, n_subjects=20, workdir=tmp_path,
             environ={}, **kw)
    (path,) = glob.glob(str(tmp_path / "xpilot-*.json"))
    return json.loads(open(path).read())


def _metrics(out):
    return {sd: {"predict": r["objectives"][XP.OBJECTIVE]["predict"], "epochs": r["objectives"][XP.OBJECTIVE]["epochs"]}
            for sd, r in out["runs"].items()}


def test_xpilot_sim_writes_masks_d_and_attention(tmp_path, monkeypatch):
    out = _tiny(tmp_path, monkeypatch, setting="device", shift=1.0, d_target=0.6, artifact_rate=3.0,
                paired=True)
    assert out["unit"] == "sim-device-d0.6-s1-a3-p:0:0"
    sim = out["sim"]
    assert sim["dTarget"] == 0.6 and sim["effect"] == E.effect_for_d(0.6)
    assert sim["paired"] and sim["artifactRate"] == 3.0 and sim["eventFraction"] == 0.0
    assert any(any(m["artifact"]) for m in sim["masks"]["targetTrain"].values())
    assert set(sim["measuredD"]) >= {"source", "target"}
    run = out["runs"]["1"]
    for side, key in (("targetTest", "targetTest"), ("targetTrain", "targetAdapt")):
        assert set(sim["masks"][side]) == set(run["subjectsSplit"][key])
        for m in sim["masks"][side].values():
            assert len(m["artifact"]) == len(m["artifactType"]) == len(m["event"]) == E.WINDOWS
    att = run["attention"][XP.OBJECTIVE]
    for when in ("predict", "end"):
        for side, key in (("targetTest", "targetTest"), ("targetTrain", "targetAdapt")):
            assert set(att[when][side]) == set(run["subjectsSplit"][key])
            for w in att[when][side].values():
                assert len(w) == E.WINDOWS and sum(w) == pytest.approx(1.0, abs=1e-2)
    assert not any("attention" in e for e in run["objectives"][XP.OBJECTIVE]["epochs"].values())
    assert phase2_trace.ATTENTION_EXPORT is False


def test_xpilot_sim_writes_event_masks(tmp_path, monkeypatch):
    out = _tiny(tmp_path, monkeypatch, setting="invert", shift=1.0, effect=0.2, event_fraction=0.2)
    assert out["unit"] == "sim-invert-e0.2-s1-v0.2:0:0" and out["sim"]["eventFraction"] == 0.2
    for side in ("targetTest", "targetTrain"):
        assert all(sum(m["event"]) >= round(0.2 * E.WINDOWS) for m in out["sim"]["masks"][side].values())


def test_xpilot_sim_default_path_unchanged_by_the_attention_hook(tmp_path, monkeypatch):
    with_hook = _tiny(tmp_path / "on", monkeypatch)
    assert with_hook["unit"] == "sim-subject-e1-s0:0:0"
    assert with_hook["sim"]["dTarget"] is None and with_hook["sim"]["effect"] == 1.0
    monkeypatch.setattr(XP, "attention_export", lambda on: contextlib.nullcontext())   # hook off
    without = _tiny(tmp_path / "off", monkeypatch)
    assert "attention" not in without["runs"]["1"]
    assert _metrics(with_hook) == _metrics(without)


def test_subject_spread_one_reproduces_identity_and_two_widens_it():
    import numpy as np
    from MIL_CREDA_EEG import eeg_sim as E
    a = E.subject_identity(np.random.default_rng(5))
    b = E.subject_identity(np.random.default_rng(5), 1.0)
    assert a == b
    iaf = lambda k: np.std([E.subject_identity(np.random.default_rng(i), k)["iaf"] for i in range(400)])
    assert 1.8 < iaf(2.0) / iaf(1.0) < 2.2
