"""The battery's job list, the per-job cache of `poc.run_many`, and its shards."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from MIL_CREDA_Benchmark import poc, poc_shard, toy_run

SRC = str(Path(poc.__file__).resolve().parents[1])
PREREGISTRATION = (Path(__file__).resolve().parents[1] / "MIL-CREDA" / "Results"
                   / "PoC_Conditions" / "preregistration.json")

SMALL = dataclasses.replace(toy_run.FULL, name="poc-test", epochs_1=1,
                            epochs_2=toy_run.MIN_EPOCHS_2, patience_1=1)
TINY_JOB = ("run_contamination_seed", (3, 0.2), {"scale": SMALL, "arms": ("TGL",)})


@pytest.fixture(scope="module")
def prereg():
    return json.loads(PREREGISTRATION.read_text())


def _without_seconds(out: dict) -> dict:
    return {key: value for key, value in out.items() if key != "seconds"}


def _notebook_jobs(prereg):
    """The notebook's battery cell, verbatim but for its seed reduction."""
    scale, clean = poc.CONDITIONS_SCALE, poc.RUN3
    artifact = dataclasses.replace(poc.RUN3, contamination="artifact")
    rate = 0.2
    seeds, rates = prereg["seeds"], prereg["contaminationRates"]
    jobs = []
    for s in seeds:
        jobs.append(("run_ladder_seed", (s,), {"scale": scale, "conditions": clean}))
    for s in seeds:
        jobs.append(("run_ladder_seed", (s,), {"scale": scale, "conditions": artifact,
                                               "rate": rate}))
    for r in rates:
        for s in seeds:
            jobs.append(("run_contamination_seed", (s, r), {"scale": scale, "conditions": clean}))
        for s in seeds:
            jobs.append(("run_contamination_seed", (s, r), {"scale": scale, "conditions": artifact}))
    return jobs


def test_battery_jobs_is_the_notebook_battery_in_its_order(prereg):
    assert poc.battery_jobs(prereg) == _notebook_jobs(prereg)


def test_battery_jobs_holds_240_jobs_for_the_current_preregistration(prereg):
    assert len(poc.battery_jobs(prereg)) == 240


@pytest.mark.parametrize("shards", [1, 3, 7, 240])
def test_the_shards_partition_the_battery_exactly_once_each(prereg, shards):
    jobs = poc.battery_jobs(prereg)
    seen = [index for i in range(shards) for index in poc_shard.shard_indices(len(jobs), i, shards)]
    assert sorted(seen) == list(range(len(jobs)))


def test_without_the_cache_run_many_is_the_job_itself(monkeypatch, tmp_path):
    monkeypatch.delenv("MIL_CREDA_POC_CACHE", raising=False)
    [out] = poc.run_many([TINY_JOB], 1, SRC)
    assert _without_seconds(out) == _without_seconds(poc._call(TINY_JOB))
    assert list(tmp_path.iterdir()) == []


def test_a_cache_hit_returns_the_stored_output_without_recomputing(monkeypatch, tmp_path):
    monkeypatch.setenv("MIL_CREDA_POC_CACHE", str(tmp_path))
    [computed] = poc.run_many([TINY_JOB], 1, SRC)
    stored = tmp_path / f"{poc.job_key(TINY_JOB, SRC)}.json"
    assert json.loads(stored.read_text()) == computed

    calls = []
    monkeypatch.setattr(poc, "run_contamination_seed",
                        lambda *a, **k: calls.append(1) or {"recomputed": True})
    [again] = poc.run_many([TINY_JOB], 1, SRC)
    assert again == computed
    assert calls == []


def test_the_job_key_reads_function_args_kwargs_and_src():
    key = poc.job_key(TINY_JOB, SRC)
    reordered = (TINY_JOB[0], TINY_JOB[1], dict(reversed(list(TINY_JOB[2].items()))))
    assert poc.job_key(reordered, SRC) == key
    assert poc.job_key(("run_contamination_seed", (3, 0.2)), SRC) != key
    assert poc.job_key(("run_contamination_seed", (4, 0.2), TINY_JOB[2]), SRC) != key
    assert poc.job_key(TINY_JOB, SRC + "-elsewhere") != key
