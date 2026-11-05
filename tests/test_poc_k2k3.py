"""Run 4 (K2/K3 with gating measurements): the gating readings `run_ladder_seed`
adds, the 60-job ladder battery `poc.k2k3_jobs`, and its shard option."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from MIL_CREDA_Benchmark import poc, poc_shard, toy_run

PREREGISTRATION = (Path(__file__).resolve().parents[1] / "MIL-CREDA" / "Results"
                   / "PoC_K2K3" / "preregistration.json")
RUN3 = (Path(__file__).resolve().parents[1] / "MIL-CREDA" / "Results"
        / "PoC_Conditions" / "preregistration.json")

#: Run 3's scale and conditions with the shortest declared programme.
SMALL = dataclasses.replace(poc.CONDITIONS_SCALE, name="k2k3-test", epochs_1=1,
                            epochs_2=toy_run.MIN_EPOCHS_2, patience_1=1)

GATING_ARMS = ("TGL", "TG", "TL")
GATING_KEYS = ("piEntropyRatio", "partnerKernelRank", "partnerKernelRankEnd",
               "routingErrorRate")


@pytest.fixture(scope="module")
def prereg():
    return json.loads(PREREGISTRATION.read_text())


@pytest.fixture(scope="module")
def ladder_pair():
    with_gating = poc.run_ladder_seed(3, scale=SMALL, conditions=poc.RUN3, gating=True)
    without = poc.run_ladder_seed(3, scale=SMALL, conditions=poc.RUN3, gating=False)
    return with_gating, without


# ------------------------------------------------------------ the readings

def test_the_entropy_ratio_is_one_for_uniform_and_zero_for_one_hot():
    assert poc.entropy_ratio(torch.full((4,), 0.25)) == pytest.approx(1.0)
    assert poc.entropy_ratio(torch.tensor([0.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert 0.0 < poc.entropy_ratio(torch.tensor([0.7, 0.1, 0.1, 0.1])) < 1.0


def test_the_entropy_ratio_of_a_single_bag_class_is_undefined():
    assert math.isnan(poc.entropy_ratio(torch.tensor([1.0])))


def test_the_ladder_reports_every_gating_reading_in_the_unit_interval(ladder_pair):
    out, _ = ladder_pair
    for arm in GATING_ARMS:
        gating = out["arms"][arm]["gating"]
        assert set(GATING_KEYS) <= set(gating)
        for key in GATING_KEYS:
            assert 0.0 <= gating[key] <= 1.0, (arm, key, gating[key])
    assert "gating" not in out["arms"][poc.NO_CONSENSUS]


def test_the_start_readings_share_the_one_phase_one_checkpoint(ladder_pair):
    out, _ = ladder_pair
    starts = {(out["arms"][a]["gating"]["partnerKernelRank"],
               out["arms"][a]["gating"]["routingErrorRate"]) for a in GATING_ARMS}
    assert len(starts) == 1


def test_measuring_the_gates_changes_no_existing_key_or_value(ladder_pair):
    out, without = ladder_pair

    def strip(record):
        arms = {arm: {k: v for k, v in r.items() if k != "gating"}
                for arm, r in record["arms"].items()}
        return {**{k: v for k, v in record.items() if k not in ("seconds", "arms")},
                "arms": arms}

    assert all("gating" not in r for r in without["arms"].values())
    assert strip(out) == strip(without)


# ---------------------------------------------------------- the battery

def test_k2k3_jobs_are_the_sixty_ladder_jobs_in_a_fixed_order(prereg):
    jobs = poc.k2k3_jobs(prereg)
    assert len(jobs) == 60
    seeds = prereg["seeds"]
    assert seeds == list(range(10, 40))
    clean = poc.Conditions(**prereg["conditions"])
    artifact = dataclasses.replace(clean, contamination="artifact")
    rate = prereg["artifact"]["k3Rate"]
    expected = [("run_ladder_seed", (s,), {"scale": poc.CONDITIONS_SCALE,
                                           "conditions": clean, "gating": True})
                for s in seeds]
    expected += [("run_ladder_seed", (s,), {"scale": poc.CONDITIONS_SCALE,
                                            "conditions": artifact, "rate": rate,
                                            "gating": True}) for s in seeds]
    assert jobs == expected


def test_k2k3_jobs_never_share_a_cache_key_with_run_three(prereg):
    src = poc_shard.SRC
    run3 = {poc.job_key(j, src) for j in poc.battery_jobs(json.loads(RUN3.read_text()))}
    assert not run3 & {poc.job_key(j, src) for j in poc.k2k3_jobs(prereg)}


def test_five_shards_partition_the_sixty_jobs_exactly_once(prereg):
    jobs = poc.k2k3_jobs(prereg)
    seen = [i for shard in range(5) for i in poc_shard.shard_indices(len(jobs), shard, 5)]
    assert sorted(seen) == list(range(60))


@pytest.mark.parametrize("battery, expected", [(None, 240), ("full", 240), ("k2k3", 60)])
def test_the_shard_runs_its_slice_of_the_chosen_battery(monkeypatch, tmp_path, battery,
                                                        expected):
    seen = []
    monkeypatch.setattr(poc, "run_many", lambda jobs, workers, src: seen.extend(jobs))
    prereg_path = RUN3 if battery != "k2k3" else PREREGISTRATION
    argv = ["--shard", "1", "--shards", "5", "--out", str(tmp_path),
            "--prereg", str(prereg_path)]
    if battery is not None:
        argv += ["--battery", battery]
    assert poc_shard.main(argv) == 0
    prereg = json.loads(prereg_path.read_text())
    jobs = poc.k2k3_jobs(prereg) if battery == "k2k3" else poc.battery_jobs(prereg)
    assert len(jobs) == expected
    assert seen == jobs[1::5]


# -------------------------------------------------- the pre-registration

def test_the_preregistration_copies_run_three_and_declares_the_gates(prereg):
    run3 = json.loads(RUN3.read_text())
    for key in ("conditions", "seeds", "hyperparameters", "alpha", "artifact",
                "scale", "scenario", "sigmaMultiplier", "noisePairingToy"):
        assert prereg[key] == run3[key], key
    assert prereg["writtenAt"]
    families = prereg["families"]
    assert families["K2"]["tests"].keys() == {"K2a", "K2b"}
    assert families["K2"]["correction"] == "Holm over 2"
    assert families["K3-aggregation"]["tests"].keys() == {"K3a"}
    assert families["K3-ablation"]["tests"].keys() == {"K3b", "K3c"}
    assert families["K3-ablation"]["correction"] == "Holm over 2"
    gates = prereg["gates"]
    assert gates["K2b"]["piEntropyRatioMedianBelow"] == 0.9
    assert gates["K3b"]["partnerKernelRankMedianAtLeast"] == 0.5
    assert gates["K3c"]["routingErrorRateMedianAtLeast"] == 0.05
    assert prereg["gateFailureVerdict"] == "inconclusive (mechanism had no room)"


def test_shard_run_entry_writes_its_cache_inside_the_prereg_folder(tmp_path, monkeypatch):
    from MIL_CREDA_Benchmark import poc_shard
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(poc_shard, "main", fake_main)
    prereg = tmp_path / "PoC_K2K3" / "preregistration.json"
    prereg.parent.mkdir()
    prereg.write_text("{}")
    assert poc_shard.run(2, 5, prereg=str(prereg)) == 0
    argv = seen["argv"]
    assert argv[argv.index("--shard") + 1] == "2" and argv[argv.index("--shards") + 1] == "5"
    assert argv[argv.index("--battery") + 1] == "k2k3" and argv[argv.index("--workers") + 1] == "4"
    assert argv[argv.index("--out") + 1] == str(prereg.parent / "cache")
    assert (prereg.parent / "cache").is_dir()
