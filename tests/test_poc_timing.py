"""Timing probe for the PoC battery: one job of each kind, timed where it runs."""

import json

from MIL_CREDA_Benchmark import poc, poc_timing


def test_probe_jobs_take_one_job_of_each_battery_kind():
    prereg = json.loads(poc_timing.PREREGISTRATION.read_text(encoding="utf-8"))
    battery = poc.battery_jobs(prereg)
    probe = poc_timing.probe_jobs(prereg)
    assert list(probe) == list(poc_timing.KINDS)
    assert all(job in battery for job in probe.values())
    assert probe["ladder"][0] == "run_ladder_seed" and "rate" not in probe["ladder"][2]
    assert probe["ladderArtifact"][0] == "run_ladder_seed" and probe["ladderArtifact"][2]["rate"] > 0
    assert probe["k1Donor"][0] == "run_contamination_seed"
    assert probe["k1Donor"][2]["conditions"].contamination == "donor"
    assert probe["k1Artifact"][2]["conditions"].contamination == "artifact"
    assert probe["k1Donor"][1] == probe["k1Artifact"][1]


def test_run_writes_one_record_per_kind_without_training(tmp_path, monkeypatch):
    calls = []

    def fake(job):
        calls.append(job[0])
        return {"seconds": 1.0}

    monkeypatch.setattr(poc_timing, "_time_one", lambda job: (fake(job), 1.0)[1])
    monkeypatch.setattr(poc_timing, "_time_parallel", lambda jobs, workers: 2.0)
    out = tmp_path / "probe.json"
    record = poc_timing.run(out=str(out), smoke=False)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved == record
    assert set(record["singleProcessSeconds"]) == set(poc_timing.KINDS)
    assert record["parallel"]["wallSeconds"] == 2.0 and record["parallel"]["workers"] >= 1
    assert len(calls) == len(poc_timing.KINDS)


def test_smoke_times_only_the_lightest_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(poc_timing, "_time_one", lambda job: 1.0)
    monkeypatch.setattr(poc_timing, "_time_parallel", lambda jobs, workers: 0.0)
    record = poc_timing.run(out=str(tmp_path / "p.json"), smoke=True)
    assert list(record["singleProcessSeconds"]) == ["ladder"]
    assert record["parallel"] is None
