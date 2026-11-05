"""Run 6 (L129): the K2 push measurement and the room scenario.

- the push: each target bag's H_I distance to its own partner source bag
  (source through the frozen phase-1 encoder, target through theta_t) at the
  start and at the end of phase 2, for TGL and TG from the same start;
- the room: two data knobs on the toy, off by default -- a smooth per-subject
  target-domain shift along the class-free height, and instance noise.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from MIL_CREDA_Benchmark import poc, poc_shard, toy_bags, toy_run

RUN4 = (Path(__file__).resolve().parents[1] / "MIL-CREDA" / "Results"
        / "PoC_K2K3" / "preregistration.json")

SMALL = dataclasses.replace(poc.CONDITIONS_SCALE, name="k2k3v3-test", epochs_1=1,
                            epochs_2=toy_run.MIN_EPOCHS_2, patience_1=1)

ROOM = dataclasses.replace(poc.RUN3, subject_signature=0.15,
                           target_subject_shift=0.5, instance_noise=0.1)

PUSH_KEYS = ("pushMean", "ownDistanceStart", "ownDistanceEnd", "marginStart",
             "marginEnd", "roomRatioMedian", "sourceSpreadMedian", "bags")


def _digest(f) -> str:
    h = hashlib.sha256()
    for a in (f.Xs, f.ys, f.bs, f.Xt, f.yt, f.bt, f.Xt_crudo):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------ the data knobs

def test_the_new_knobs_are_off_by_default_and_leave_records_and_draws_alone():
    assert poc.Conditions().target_subject_shift == 0.0
    assert poc.Conditions().instance_noise == 0.0
    record = poc.conditions_record(poc.RUN3)
    assert "target_subject_shift" not in record and "instance_noise" not in record
    base = poc.family_under(10, poc.CONDITIONS_SCALE, poc.RUN3)
    assert toy_bags.desplazar_sujetos_en_target(base, 0.0, 10) is base
    assert toy_bags.ruido_de_instancia(base, 0.0, 10) is base


def test_run_four_cache_keys_are_unchanged():
    jobs = poc.k2k3_jobs(json.loads(RUN4.read_text()))
    assert poc.job_key(jobs[0], "S") == (
        "b0de56ca91e3d964948deea4063f269b6d543c1299bd12616ddab57c17f3fc9f")


@pytest.fixture(scope="module")
def room_pair():
    signed = dataclasses.replace(poc.RUN3, subject_signature=0.15)
    return (poc.family_under(10, poc.CONDITIONS_SCALE, signed),
            poc.family_under(10, poc.CONDITIONS_SCALE, ROOM))


def test_the_target_shift_moves_only_the_target_height_one_value_per_subject():
    signed = poc.family_under(10, poc.CONDITIONS_SCALE,
                              dataclasses.replace(poc.RUN3, subject_signature=0.15))
    shifted = toy_bags.desplazar_sujetos_en_target(signed, 0.5, 10)
    assert np.array_equal(shifted.Xs, signed.Xs)
    moved = shifted.Xt - signed.Xt
    assert np.array_equal(np.delete(moved, toy_bags.EJE_DE_FIRMA, axis=1),
                          np.zeros((len(moved), 2)))
    per_subject = []
    for c in np.unique(signed.yt):
        for k in np.unique(signed.bt[signed.yt == c]):
            m = (signed.yt == c) & (signed.bt == k)
            per_subject.append(moved[m, toy_bags.EJE_DE_FIRMA].mean())
    per_subject = np.asarray(per_subject)
    # distinct per subject, zero-mean over subjects, RMS the declared amplitude
    assert len(np.unique(np.round(per_subject, 6))) > 0.9 * len(per_subject)
    amplitude = shifted.desplazamientoDeSujeto["amplitude"]
    assert abs(per_subject.mean()) < 0.15 * amplitude
    assert np.sqrt((moved[:, toy_bags.EJE_DE_FIRMA] ** 2).mean()) == pytest.approx(
        amplitude, rel=0.05)


def test_the_target_shift_is_a_smooth_function_of_the_roll_position():
    signed = poc.family_under(10, poc.CONDITIONS_SCALE,
                              dataclasses.replace(poc.RUN3, subject_signature=0.15))
    shifted = toy_bags.desplazar_sujetos_en_target(signed, 0.5, 10)
    t = toy_bags.parametro("rollo", signed.Xt_crudo)
    moved = (shifted.Xt - signed.Xt)[:, toy_bags.EJE_DE_FIRMA]
    order = np.argsort(t)
    steps = np.abs(np.diff(moved[order]))
    assert steps.max() < 0.05 * shifted.desplazamientoDeSujeto["amplitude"] * 10


def test_instance_noise_touches_both_domains_and_keeps_labels(room_pair):
    signed, room = room_pair
    for name in ("ys", "bs", "yt", "bt", "Xt_crudo"):
        assert np.array_equal(getattr(signed, name), getattr(room, name)), name
    assert not np.array_equal(signed.Xs, room.Xs)
    assert room.ruido["sd"] > 0


def test_negative_knobs_are_refused():
    base = poc.family_under(10, poc.CONDITIONS_SCALE, poc.RUN3)
    with pytest.raises(ValueError):
        toy_bags.desplazar_sujetos_en_target(base, -0.1, 10)
    with pytest.raises(ValueError):
        toy_bags.ruido_de_instancia(base, -0.1, 10)


# ---------------------------------------------------------------- the push

def test_partner_push_reads_start_minus_end():
    start = {"own": torch.tensor([3.0, 2.0]), "othersMedian": torch.tensor([2.0, 2.0]),
             "spread": torch.tensor([2.0, 1.0])}
    end = {"own": torch.tensor([1.0, 1.5]), "othersMedian": torch.tensor([2.0, 2.5]),
           "spread": torch.tensor([2.0, 1.0])}
    push = poc.partner_push(start, end)
    assert push["pushMean"] == pytest.approx(((3 - 1) + (2 - 1.5)) / 2)
    assert push["marginEnd"] == pytest.approx(((1 - 2) + (1.5 - 2.5)) / 2)
    assert push["marginStart"] == pytest.approx(((3 - 2) + (2 - 2)) / 2)
    assert push["roomRatioMedian"] == pytest.approx(np.median([1.5, 2.0]))


@pytest.fixture(scope="module")
def ladder_pair():
    with_push = poc.run_ladder_seed(3, scale=SMALL, conditions=ROOM, push=True)
    without = poc.run_ladder_seed(3, scale=SMALL, conditions=ROOM)
    return with_push, without


def test_the_ladder_reports_the_push_for_tgl_and_tg_from_one_start(ladder_pair):
    with_push, _ = ladder_pair
    tgl, tg = with_push["arms"]["TGL"]["push"], with_push["arms"]["TG"]["push"]
    for record in (tgl, tg):
        for key in PUSH_KEYS:
            assert np.isfinite(record[key]), key
        assert record["bags"] > 0
    assert tgl["ownDistanceStart"] == pytest.approx(tg["ownDistanceStart"], abs=1e-6)
    assert tgl["roomRatioMedian"] == pytest.approx(tg["roomRatioMedian"], abs=1e-6)
    assert tgl["pushMean"] == pytest.approx(
        tgl["ownDistanceStart"] - tgl["ownDistanceEnd"], abs=1e-6)


def test_the_push_adds_keys_and_changes_no_other_value(ladder_pair):
    with_push, without = ladder_pair
    for arm, record in without["arms"].items():
        extra = {k: v for k, v in with_push["arms"][arm].items() if k != "push"}
        for key in ("targetAccuracy", "sourceAccuracy", "sourceOnlyTargetAccuracy"):
            assert extra[key] == record[key], (arm, key)
        assert "push" not in record


# --------------------------------------------------------------- the battery

def _prereg_v3():
    return {"seeds": list(range(10, 40)), "artifact": {"k3Rate": 0.4},
            "conditions": poc.conditions_record(ROOM)}


def test_k2k3v3_jobs_are_sixty_ladder_jobs_with_the_push():
    jobs = poc.k2k3v3_jobs(_prereg_v3())
    assert len(jobs) == 60
    assert all(j[0] == "run_ladder_seed" and j[2]["push"] for j in jobs)
    assert [j[2].get("rate", 0.0) for j in jobs] == [0.0] * 30 + [0.4] * 30
    assert jobs[30][2]["conditions"].contamination == "artifact"
    assert len({poc.job_key(j, "S") for j in jobs}) == 60


def test_k2k3v3_refuses_conditions_without_the_room_knobs():
    prereg = _prereg_v3()
    prereg["conditions"] = poc.conditions_record(
        dataclasses.replace(poc.RUN3, subject_signature=0.15))
    with pytest.raises(ValueError):
        poc.k2k3v3_jobs(prereg)


def test_the_shard_offers_k2k3v3_and_keeps_full_as_default():
    assert poc_shard.BATTERIES["k2k3v3"] is poc.k2k3v3_jobs
    assert set(poc_shard.BATTERIES) == {"full", "k2k3", "k2k3v2", "k2k3v3"}


def test_the_shard_runs_its_slice_of_k2k3v3(monkeypatch, tmp_path):
    prereg = tmp_path / "p.json"
    prereg.write_text(json.dumps(_prereg_v3()))
    seen = {}
    monkeypatch.setattr(poc, "run_many",
                        lambda jobs, workers, src: seen.setdefault("jobs", jobs))
    assert poc_shard.main(["--shard", "1", "--shards", "30", "--out", str(tmp_path / "c"),
                           "--battery", "k2k3v3", "--prereg", str(prereg)]) == 0
    assert [j[1][0] for j in seen["jobs"]] == [11, 11]
