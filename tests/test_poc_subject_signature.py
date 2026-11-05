"""Run 5's subject signature (`toy_bags.firmar_sujetos`, `poc.Conditions`):
every subject (bag index k within its class) carries one fixed seeded offset
along `rollo`'s class-free height, in BOTH domains, so the shared encoder can
tell one subject from another while the class information stays unchanged.

Off by default, and off is the draw, the run records and the cache keys
exactly as they were.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from MIL_CREDA_Benchmark import poc, poc_shard, toy_bags, toy_run

RUN4 = (Path(__file__).resolve().parents[1] / "MIL-CREDA" / "Results"
        / "PoC_K2K3" / "preregistration.json")
SIGNATURE = 0.13

#: `poc.job_key` of the first and last `k2k3_jobs` of run 4 under src "S",
#: taken before the option existed.
GOLDEN_KEYS = ("b0de56ca91e3d964948deea4063f269b6d543c1299bd12616ddab57c17f3fc9f",
               "57afa36e6ab872245bb3b72f6b0c1d3d8f8b329b44a41d54c53ffbbb752a6be4")


def _digest(f) -> str:
    h = hashlib.sha256()
    for a in (f.Xs, f.ys, f.bs, f.Xt, f.yt, f.bt, f.Xt_crudo):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def pair():
    off = toy_run.build_family("rollo", 10, poc.CONDITIONS_SCALE, **poc.RUN3.draw_options())
    on = poc.family_under(10, poc.CONDITIONS_SCALE,
                          dataclasses.replace(poc.RUN3, subject_signature=SIGNATURE))
    return off, on


# ---------------------------------------------------------------- off

def test_off_is_the_draw_exactly_as_it_was():
    base = toy_run.build_family("rollo", 10, poc.CONDITIONS_SCALE, **poc.RUN3.draw_options())
    assert _digest(poc.family_under(10, poc.CONDITIONS_SCALE, poc.RUN3)) == _digest(base)
    assert toy_bags.firmar_sujetos(base, 0.0, 10) is base
    assert poc.Conditions().subject_signature == 0.0


def test_off_keeps_run_four_cache_keys_and_records():
    jobs = poc.k2k3_jobs(json.loads(RUN4.read_text()))
    assert (poc.job_key(jobs[0], "S"), poc.job_key(jobs[-1], "S")) == GOLDEN_KEYS
    assert "subject_signature" not in poc.conditions_record(poc.RUN3)
    on = dataclasses.replace(poc.RUN3, subject_signature=SIGNATURE)
    assert poc.conditions_record(on)["subject_signature"] == SIGNATURE


# ----------------------------------------------------------------- on

def test_on_keeps_labels_bags_and_partners(pair):
    off, on = pair
    for name in ("ys", "bs", "yt", "bt", "Xt_crudo"):
        assert np.array_equal(getattr(off, name), getattr(on, name)), name


def test_on_moves_only_the_class_free_height(pair):
    off, on = pair
    for a, b in ((off.Xs, on.Xs), (off.Xt, on.Xt)):
        moved = b - a
        assert np.array_equal(np.delete(moved, toy_bags.EJE_DE_FIRMA, axis=1),
                              np.zeros((len(a), 2)))


def _offsets(off, on, domain):
    X0, X1 = getattr(off, f"X{domain}"), getattr(on, f"X{domain}")
    y, b = getattr(off, f"y{domain}"), getattr(off, f"b{domain}")
    out = {}
    for c in np.unique(y):
        for k in np.unique(b[y == c]):
            d = (X1 - X0)[(y == c) & (b == k), toy_bags.EJE_DE_FIRMA]
            assert np.allclose(d, d[0]), "one offset for every instance of a bag"
            out[(int(c), int(k))] = float(d[0])
    return out


def test_a_subject_has_one_signature_in_both_domains(pair):
    off, on = pair
    source, target = _offsets(off, on, "s"), _offsets(off, on, "t")
    assert source.keys() == target.keys()
    for key in source:
        assert source[key] == pytest.approx(target[key], abs=1e-12)


def test_signatures_are_distinct_across_subjects_and_free_of_class(pair):
    off, on = pair
    source = _offsets(off, on, "s")
    classes = sorted({c for c, _ in source})
    per_class = [np.array([source[(c, k)] for k in sorted(k for c2, k in source if c2 == c)])
                 for c in classes]
    for values in per_class:
        assert np.diff(np.sort(values)).min() > 1e-6
        assert values.mean() == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(values, per_class[0], atol=1e-12), "the same signatures in every class"


def test_the_amplitude_is_the_declared_multiple_of_the_class_separation(pair):
    off, on = pair
    firma = on.firma
    assert firma["amplitude"] == pytest.approx(SIGNATURE * firma["betweenClassSeparation"])
    assert firma["withinBagSpread"] > 0
    source = np.array(list(_offsets(off, on, "s").values()))
    assert np.sqrt((source ** 2).mean()) == pytest.approx(firma["amplitude"], rel=1e-9)
    assert firma["multipleOfClassSeparation"] == SIGNATURE


@pytest.mark.parametrize("fraction", [-0.1, float("nan"), float("inf")])
def test_a_negative_or_non_finite_multiple_is_refused(pair, fraction):
    off, _ = pair
    with pytest.raises(ValueError):
        toy_bags.firmar_sujetos(off, fraction, 10)


def test_the_signature_is_refused_off_the_roll():
    f = toy_run.build_family("toroide", 10, toy_run.FULL)
    with pytest.raises(ValueError):
        toy_bags.firmar_sujetos(f, SIGNATURE, 10)


def test_the_runners_build_their_family_under_the_signature(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def fake(seed, scale, conditions):
        seen.append((seed, conditions.subject_signature))
        raise Stop

    monkeypatch.setattr(poc, "family_under", fake)
    on = dataclasses.replace(poc.RUN3, subject_signature=SIGNATURE)
    for runner, args in (("run_ladder_seed", (5,)), ("run_contamination_seed", (5, 0.2))):
        with pytest.raises(Stop):
            getattr(poc, runner)(*args, conditions=on, scale=poc.CONDITIONS_SCALE)
    assert seen == [(5, SIGNATURE), (5, SIGNATURE)]


# ------------------------------------------------------------ battery

def _v2(rate):
    prereg = json.loads(RUN4.read_text())
    prereg["conditions"] = {**prereg["conditions"], "subject_signature": SIGNATURE}
    prereg["artifact"] = {**prereg["artifact"], "k3Rate": rate}
    return prereg


def test_k2k3v2_jobs_are_the_ladder_under_the_signature():
    jobs = poc.k2k3v2_jobs(_v2(0.3))
    assert len(jobs) == 60
    assert {j[2]["conditions"].subject_signature for j in jobs} == {SIGNATURE}
    assert {j[2].get("rate", 0.0) for j in jobs} == {0.0, 0.3}
    run4 = {poc.job_key(j, "S") for j in poc.k2k3_jobs(json.loads(RUN4.read_text()))}
    assert not run4 & {poc.job_key(j, "S") for j in jobs}


def test_k2k3v2_refuses_an_unset_rate_or_a_missing_signature():
    with pytest.raises(ValueError, match="k3Rate"):
        poc.k2k3v2_jobs(_v2("TO_BE_SET_BY_PARENT"))
    with pytest.raises(ValueError, match="subject_signature"):
        poc.k2k3v2_jobs(json.loads(RUN4.read_text()))


def test_the_shard_runs_its_slice_of_k2k3v2(monkeypatch, tmp_path):
    assert poc_shard.BATTERIES["k2k3v2"] is poc.k2k3v2_jobs
    prereg_path = tmp_path / "preregistration.json"
    prereg_path.write_text(json.dumps(_v2(0.3)))
    seen = []
    monkeypatch.setattr(poc, "run_many", lambda jobs, workers, src: seen.extend(jobs))
    assert poc_shard.main(["--shard", "1", "--shards", "5", "--out", str(tmp_path),
                           "--prereg", str(prereg_path), "--battery", "k2k3v2"]) == 0
    assert seen == poc.k2k3v2_jobs(_v2(0.3))[1::5]
