"""Experiments plan v43, Secs. 4.2, 4.6, 7.3 and 7.7: the shard-aware remote units.

Everything runs locally on temporary folders and fake FORGE_* variables; nothing here
contacts a remote service. The synthetic set is seeded noise; no number it yields is a
measured result.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

import pytest
import torch

import numpy as np
import pandas as pd

from MIL_CREDA_EEG import campaign, model_store, preprocess, remote_units, search

SCHEDULE = {"epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 4}


@pytest.fixture
def registries():
    """Comparator registration mutates global search state; restore it afterwards."""
    saved = {(m, name): dict(getattr(m, name)) for m, name in (
        (search, "PHASE_TWO_TRAINERS"), (search, "PHASE_TWO_EXTRA"),
        (search, "PHASE_ONE_TRAINERS"), (model_store, "BUILDERS"))}
    yield
    for (m, name), value in saved.items():
        getattr(m, name).clear()
        getattr(m, name).update(value)


@pytest.fixture
def torch_flags():
    """`set_deterministic` changes process-wide torch state; restore it afterwards."""
    saved = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
             torch.backends.cudnn.deterministic, os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    yield
    torch.use_deterministic_algorithms(saved[0])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[1], saved[2]
    if saved[3] is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved[3]


def _config(tmp: Path, strategies=("MIL-CREDA", "COT", "HASD")) -> dict:
    cfg = campaign.write_synthetic_set(tmp / "synthetic", seed=0, subjects=24)
    cfg.update({"strategies": list(strategies), "nTrials": 1, "schedule": dict(SCHEDULE),
                "bootstrapResamples": 200, "device": "cpu"})
    return cfg


def _product(tmp: Path) -> Path:
    product = tmp / "product"
    product.mkdir(exist_ok=True)
    return product


# ------------------------------------------------------------------ enumeration


def test_units_are_enumerated_in_a_deterministic_order_with_stable_ids(tmp_path):
    cfg = _config(tmp_path)
    ids = [u.id for u in remote_units.enumerate_units(cfg, _product(tmp_path))]
    assert ids == [
        "search.phase1.L1-to-L6.common", "search.phase1.L1-to-L6.HASD",
        "search.phase1.L1-to-L6.MIL-CREDA",
        "search.phase1.L6-to-L1.common", "search.phase1.L6-to-L1.HASD",
        "search.phase1.L6-to-L1.MIL-CREDA",
        "search.phase2.L1-to-L6.COT", "search.phase2.L1-to-L6.HASD",
        "search.phase2.L1-to-L6.MIL-CREDA",
        "search.phase2.L6-to-L1.COT", "search.phase2.L6-to-L1.HASD",
        "search.phase2.L6-to-L1.MIL-CREDA",
        "final.L1-to-L6", "final.L6-to-L1", "k3.L1-to-L6", "k3.L6-to-L1"]
    shuffled = {**cfg, "strategies": ["HASD", "MIL-CREDA", "COT"],
                "transfers": list(reversed(cfg["transfers"]))}
    assert [u.id for u in remote_units.enumerate_units(shuffled, _product(tmp_path))] == ids


def test_units_name_their_dependencies(tmp_path):
    units = {u.id: u for u in remote_units.enumerate_units(_config(tmp_path),
                                                            _product(tmp_path))}
    assert units["search.phase1.L1-to-L6.common"].after == ()
    assert units["search.phase2.L1-to-L6.COT"].after == ("search.phase1.L1-to-L6.common",)
    assert units["search.phase2.L1-to-L6.HASD"].after == ("search.phase1.L1-to-L6.HASD",)
    assert set(units["final.L1-to-L6"].after) == {
        i for i in units if i.startswith("search.") and ".L1-to-L6." in i}
    assert units["k3.L6-to-L1"].after == ("final.L6-to-L1",)


def test_selected_units_come_from_forge_run_units(tmp_path):
    units = remote_units.enumerate_units(_config(tmp_path), _product(tmp_path))
    every = [u.id for u in units]
    assert [u.id for u in remote_units.select_units(units, environ={})] == every
    pick = ["final.L6-to-L1", "search.phase1.L1-to-L6.HASD"]
    env = {remote_units.UNITS_ENV: json.dumps(pick)}
    assert [u.id for u in remote_units.select_units(units, environ=env)] == pick
    assert remote_units.select_units(units, environ={remote_units.UNITS_ENV: "[]"}) == []
    assert [u.id for u in remote_units.select_units(units, pick[:1], environ=env)] == pick[:1]
    with pytest.raises(remote_units.UnitError, match="not a unit"):
        remote_units.select_units(units, environ={remote_units.UNITS_ENV: '["nope"]'})


# ------------------------------------------------------------------ inputs


def test_inputs_resolve_from_forge_input_datasets(tmp_path):
    cfg = _config(tmp_path)
    mount = tmp_path / "input" / "owner" / "balladeer-processed"
    shutil.copytree(Path(cfg["processedRoot"]), mount / "nested" / "processed")
    shutil.copy(cfg["partitionPath"], mount / "partition.json")
    cfg["remoteInputs"] = {"processed": "owner/balladeer-processed",
                           "partition": "owner/balladeer-processed"}
    env = {remote_units.INPUTS_ENV: json.dumps({"owner/balladeer-processed": str(mount)})}
    resolved = remote_units.resolve_inputs(cfg, environ=env)
    assert resolved["processedRoot"] == str(mount / "nested" / "processed")
    assert resolved["partitionPath"] == str(mount / "partition.json")
    assert remote_units.resolve_inputs(cfg, environ={}) == cfg     # local run: unchanged
    with pytest.raises(remote_units.InputUnavailable, match="owner/balladeer-processed"):
        remote_units.resolve_inputs(cfg, environ={
            remote_units.INPUTS_ENV: json.dumps({"owner/balladeer-processed": None})})


def test_mounted_data_must_match_the_repository_manifest(tmp_path):
    cfg = {**_config(tmp_path), "synthetic": False}
    with pytest.raises(remote_units.InputUnavailable, match="manifest.csv"):
        remote_units.data_identity(cfg)


# ------------------------------------------------------------------ running units


def _fake(calls: list):
    def execute(unit, cfg, product, bags_of, guard=None):
        calls.append((unit.id, torch.are_deterministic_algorithms_enabled()))
        models = product / "Models" / "Balladeer"
        models.mkdir(parents=True, exist_ok=True)
        (models / f"{unit.id}.pt").write_bytes(unit.id.encode())
        runs = product / "Results" / "A-nivel" / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        with (runs / "predictions.jsonl").open("a") as handle:
            handle.write(json.dumps({"unit": unit.id}) + "\n")
        derived = product / "Results" / "A-nivel" / "tables"
        derived.mkdir(parents=True, exist_ok=True)
        (derived / "table.csv").write_text(unit.id)
        return {"unit": unit.id}
    return execute


def _run(cfg, product, tmp, ids, calls, **kwargs):
    kwargs.setdefault("estimate", lambda unit: 0.0)
    return remote_units.run_units(cfg, product, units=ids, workdir=tmp / "work",
                                  execute=_fake(calls), environ={}, **kwargs)


def test_each_unit_writes_isolated_outputs_with_completion_and_environment_stamps(
        tmp_path, torch_flags):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    ids = ["search.phase1.L1-to-L6.common", "search.phase1.L1-to-L6.HASD"]
    outcome = _run(cfg, product, tmp_path, ids, calls)
    assert outcome["ran"] == ids and outcome["remaining"] == []
    assert not (product / "Models").exists(), "a unit never writes the product itself"
    shards = product / remote_units.SHARDS_RELATIVE
    stamp = json.loads((shards / ids[1] / "shard.json").read_text())
    env = stamp["environment"]
    assert set(remote_units.IDENTICAL_FIELDS) <= set(env)
    assert env["torch"] == torch.__version__ and env["commit"]
    assert env["partitionSha256"] == json.loads(
        Path(cfg["partitionPath"]).read_text())["partition_sha256"]
    assert env["seeds"] == {"search": search.SEARCH_SEED, "report": list(search.REPORT_SEEDS)}
    completion = stamp["completion"]
    assert completion["unit"] == ids[1] and completion["complete"] is True
    paths = {o["path"]: o for o in completion["outputs"]}
    assert paths[f"Models/Balladeer/{ids[1]}.pt"]["mode"] == "new"
    # The second unit was seeded with the first unit's line: only its own line is its output.
    tail = paths["Results/A-nivel/runs/predictions.jsonl"]
    assert tail["mode"] == "append"
    stored = (shards / ids[1] / "outputs" / "Results/A-nivel/runs/predictions.jsonl")
    assert stored.read_text() == json.dumps({"unit": ids[1]}) + "\n"
    assert completion["discarded"] == ["Results/A-nivel/tables/table.csv"]
    runs = (shards / ids[1] / "runs.jsonl").read_text().splitlines()
    assert [json.loads(x)["unit"] for x in runs] == [ids[1]]


def test_the_deadline_stops_before_a_unit_and_writes_partial(tmp_path, torch_flags):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    ids = ["search.phase1.L1-to-L6.common", "search.phase1.L1-to-L6.HASD",
           "search.phase1.L1-to-L6.MIL-CREDA"]
    outcome = _run(cfg, product, tmp_path, ids, calls, deadline=time.time() + 3600,
                   estimate=lambda unit: 0.0 if unit.id == ids[0] else 1e9)
    assert [c[0] for c in calls] == ids[:1]
    assert outcome["remaining"] == ids[1:]
    marker = json.loads((tmp_path / "work" / remote_units.PARTIAL_MARKER).read_text())
    assert marker == {"remaining": ids[1:]}
    calls.clear()
    env_deadline = {remote_units.DEADLINE_ENV: "0"}
    outcome = remote_units.run_units(cfg, product, units=ids[1:], workdir=tmp_path / "w2",
                                     execute=_fake(calls), environ=env_deadline,
                                     estimate=lambda unit: 0.0)
    assert calls == [] and outcome["remaining"] == ids[1:]


def test_completed_units_are_skipped_on_resume(tmp_path, torch_flags):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    ids = ["search.phase1.L1-to-L6.common", "search.phase1.L1-to-L6.HASD"]
    _run(cfg, product, tmp_path, ids[:1], calls)
    outcome = _run(cfg, product, tmp_path, ids, calls)
    assert [c[0] for c in calls] == ids
    assert outcome["skipped"] == ids[:1] and outcome["ran"] == ids[1:]


def test_a_unit_whose_dependencies_are_absent_is_refused_before_anything_runs(
        tmp_path, torch_flags):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    with pytest.raises(remote_units.UnitError, match="search.phase1.L1-to-L6.common"):
        _run(cfg, product, tmp_path, ["search.phase2.L1-to-L6.COT"], calls)
    assert calls == []


def test_deterministic_flags_are_set_before_any_unit(tmp_path, torch_flags):
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    state = remote_units.set_deterministic()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert state == {"useDeterministicAlgorithms": True, "cudnnBenchmark": False,
                     "cudnnDeterministic": True, "cublasWorkspaceConfig": ":4096:8"}
    torch.use_deterministic_algorithms(False)
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    _run(cfg, product, tmp_path, ["search.phase1.L1-to-L6.common"], calls)
    assert calls == [("search.phase1.L1-to-L6.common", True)]


# ------------------------------------------------------------------ merge


def _two_shards(tmp_path):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    ids = ["search.phase1.L1-to-L6.common", "search.phase1.L1-to-L6.HASD"]
    _run(cfg, product, tmp_path, ids, calls)
    return cfg, product, ids, product / remote_units.SHARDS_RELATIVE


def _edit_stamp(path: Path, **environment) -> None:
    stamp = json.loads(path.read_text())
    stamp["environment"].update(environment)
    path.write_text(json.dumps(stamp))


def test_merge_writes_the_product_layout_once(tmp_path, torch_flags):
    cfg, product, ids, _ = _two_shards(tmp_path)
    outcome = remote_units.merge(product, config=cfg)
    assert outcome["merged"] == ids
    for i in ids:
        assert (product / "Models" / "Balladeer" / f"{i}.pt").read_bytes() == i.encode()
    lines = (product / "Results/A-nivel/runs/predictions.jsonl").read_text().splitlines()
    assert [json.loads(x)["unit"] for x in lines] == ids
    assert not (product / "Results/A-nivel/tables").exists()
    with pytest.raises(remote_units.MergeError, match="already merged"):
        remote_units.merge(product, config=cfg)


def test_merge_refuses_an_environment_mismatch_and_writes_nothing(tmp_path, torch_flags):
    cfg, product, ids, shards = _two_shards(tmp_path)
    for field, value in (("commit", "0" * 40), ("partitionSha256", "x"),
                         ("dataChecksum", "y"), ("torch", "0.0")):
        path = shards / ids[1] / "shard.json"
        saved = path.read_text()
        _edit_stamp(path, **{field: value})
        with pytest.raises(remote_units.MergeError, match=field):
            remote_units.merge(product, config=cfg)
        assert not (product / "Models").exists()
        path.write_text(saved)


def test_merge_refuses_a_duplicate_unit(tmp_path, torch_flags):
    cfg, product, ids, shards = _two_shards(tmp_path)
    shutil.copytree(shards / ids[0], tmp_path / "fetched" / "again" / ids[0])
    with pytest.raises(remote_units.MergeError, match="duplicate"):
        remote_units.merge(product, roots=[shards, tmp_path / "fetched"], config=cfg)
    assert not (product / "Models").exists()


def test_merge_refuses_a_missing_field_or_a_tampered_output(tmp_path, torch_flags):
    cfg, product, ids, shards = _two_shards(tmp_path)
    _edit_stamp(shards / ids[0] / "shard.json", commit=None)
    with pytest.raises(remote_units.MergeError, match="commit"):
        remote_units.merge(product, config=cfg)
    _edit_stamp(shards / ids[0] / "shard.json",
                commit=json.loads((shards / ids[1] / "shard.json").read_text())
                ["environment"]["commit"])
    (shards / ids[1] / "outputs" / "Models" / "Balladeer" / f"{ids[1]}.pt").write_bytes(b"x")
    with pytest.raises(remote_units.MergeError, match="sha256"):
        remote_units.merge(product, config=cfg)
    assert not (product / "Models").exists()


# ------------------------------------------------------------------ the real campaign


def _predictions(product: Path) -> list[str]:
    path = product / "Results" / "A-nivel" / "runs" / "predictions.jsonl"
    return sorted(path.read_text().splitlines())


def test_units_run_the_campaign_functions_with_the_plans_seeds(tmp_path, registries,
                                                               torch_flags):
    """Units then merge give the campaign's own predictions: same functions, seeds 0 and 1."""
    cfg = _config(tmp_path, strategies=("MIL-CREDA",))
    direct = tmp_path / "direct"
    direct.mkdir()
    campaign.run("search", cfg, direct)
    campaign.run("final", cfg, direct)
    product = _product(tmp_path)
    ids = [u.id for u in remote_units.enumerate_units(cfg, product)
           if u.step in ("search", "final")]
    outcome = remote_units.run_units(cfg, product, units=ids, workdir=tmp_path / "work",
                                     environ={}, estimate=lambda unit: 0.0)
    assert outcome["ran"] == ids
    remote_units.merge(product, config=cfg)
    assert _predictions(product) == _predictions(direct)
    seeds = {json.loads(x)["seed"] for x in _predictions(product)}
    assert seeds == set(search.REPORT_SEEDS)
    def models(root: Path) -> list[str]:
        base = root / "Models" / "Balladeer"
        return sorted(str(p.relative_to(base)) for p in base.rglob("*.pt"))

    assert models(product) == models(direct) != []


# ------------------------------------------------------------------ pins


def test_requirements_pin_torch_optuna_and_scipy_to_the_installed_versions():
    lines = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines()
    for name in ("torch", "optuna", "scipy"):
        assert f"{name}=={importlib.metadata.version(name)}" in lines, name


# ------------------------------------------------------------------ R1: estimate and deadline


def test_a_unit_without_a_measured_duration_estimate_is_refused_naming_what_to_measure(
        tmp_path, torch_flags):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    ids = ["search.phase1.L1-to-L6.common"]
    with pytest.raises(remote_units.UnitError, match="timing.json") as refused:
        remote_units.run_units(cfg, product, units=ids, workdir=tmp_path / "work",
                               execute=_fake(calls), environ={})
    assert "timing probe" in str(refused.value) and "pilot" in str(refused.value)
    assert calls == []
    # the measured record, once it exists, is the estimate source
    record = {"schema": 1, "measurements": [
        {"measuredAt": "2026-11-03T00:00:00+00:00", "host": "local", "device": "cpu",
         "arm": "level", "unit": "epoch", "secondsPerUnit": 0.5, "unitsMeasured": 1}]}
    (product / "Results").mkdir(parents=True)
    (product / "Results" / "timing.json").write_text(json.dumps(record))
    outcome = remote_units.run_units(cfg, product, units=ids, workdir=tmp_path / "work",
                                     execute=_fake(calls), environ={})
    assert outcome["ran"] == ids
    estimate = remote_units.timing_estimate(product / "Results" / "timing.json", cfg, product)
    units = {u.id: u for u in remote_units.enumerate_units(cfg, product)}
    # one trial trains the four rotations for at most epochs1 epochs each
    assert estimate(units[ids[0]]) == 4 * SCHEDULE["epochs1"] * 0.5
    (product / "Results" / "timing.json").write_text(json.dumps({"schema": 1}))
    with pytest.raises(remote_units.UnitError, match="timing.json"):
        remote_units.timing_estimate(product / "Results" / "timing.json", cfg, product)


def _complete_trials(db: Path) -> int:
    if not db.exists():
        return 0
    with sqlite3.connect(db) as conn:
        return conn.execute("select count(*) from trials where state = 'COMPLETE'"
                            ).fetchone()[0]


def test_the_deadline_stops_a_study_between_trials_and_the_rerun_completes_it(
        tmp_path, registries, torch_flags):
    """Controlled clock: the deadline arrives once the first trial is stored."""
    cfg = {**_config(tmp_path, strategies=("MIL-CREDA",)), "nTrials": 3}
    product = _product(tmp_path)
    uid = "search.phase1.L1-to-L6.MIL-CREDA"
    shards = product / remote_units.SHARDS_RELATIVE
    working = shards / remote_units.PROGRESS_DIR / uid / "product"
    db = working / "Results" / "hpo" / "L1-to-L6__phase1__MIL-CREDA.db"

    def clock():
        return 1000.0 if _complete_trials(db) >= 1 else 0.0

    outcome = remote_units.run_units(cfg, product, units=[uid], workdir=tmp_path / "work",
                                     environ={}, deadline=100.0, estimate=lambda u: 10.0,
                                     clock=clock)
    assert outcome["ran"] == [] and outcome["remaining"] == [uid]
    marker = json.loads((tmp_path / "work" / remote_units.PARTIAL_MARKER).read_text())
    assert marker == {"remaining": [uid],
                      "progress": {uid: {"completedTrials": 1, "nTrials": 3}}}
    assert not (shards / uid).exists(), "an unfinished unit leaves no shard"
    assert _complete_trials(db) == 1, "the study's progress outlives the session"
    outcome = remote_units.run_units(cfg, product, units=[uid], workdir=tmp_path / "w2",
                                     environ={}, estimate=lambda u: 10.0)
    assert outcome["ran"] == [uid] and outcome["remaining"] == []
    assert not (tmp_path / "w2" / remote_units.PARTIAL_MARKER).exists()
    stored = shards / uid / "outputs" / "Results" / "hpo" / db.name
    assert _complete_trials(stored) == 3
    assert not (shards / remote_units.PROGRESS_DIR / uid).exists()
    import optuna
    study = optuna.load_study(study_name=stored.stem, storage=f"sqlite:///{stored}")
    assert study.user_attrs["resumedAfter"] == [1]


def _search_then(cfg, product, tmp_path):
    ids = [u.id for u in remote_units.enumerate_units(cfg, product) if u.step == "search"]
    remote_units.run_units(cfg, product, units=ids, workdir=tmp_path / "s",
                           environ={}, estimate=lambda u: 0.0)


def test_the_deadline_stops_final_between_rotations_and_the_rerun_completes_it(
        tmp_path, registries, torch_flags):
    """Controlled clock: the deadline arrives once rotation 0's predictions are stored;
    the continued unit gives the campaign's own predictions."""
    cfg = _config(tmp_path, strategies=("MIL-CREDA",))
    cfg["transfers"] = cfg["transfers"][:1]
    direct = tmp_path / "direct"
    direct.mkdir()
    campaign.run("search", cfg, direct)
    campaign.run("final", cfg, direct)
    product = _product(tmp_path)
    _search_then(cfg, product, tmp_path)
    uid = "final.L1-to-L6"
    working = (product / remote_units.SHARDS_RELATIVE / remote_units.PROGRESS_DIR / uid
               / "product")
    predictions = working / "Results" / "A-nivel" / "runs" / "predictions.jsonl"

    def clock():
        return 1000.0 if predictions.exists() else 0.0

    outcome = remote_units.run_units(cfg, product, units=[uid], workdir=tmp_path / "work",
                                     environ={}, deadline=100.0, estimate=lambda u: 10.0,
                                     clock=clock)
    assert outcome["remaining"] == [uid]
    marker = json.loads((tmp_path / "work" / remote_units.PARTIAL_MARKER).read_text())
    assert marker == {"remaining": [uid], "progress": {
        uid: {"completedRotations": [0], "rotations": [0, 1, 2, 3]}}}
    assert {json.loads(x)["rotation"] for x in predictions.read_text().splitlines()} == {0}
    outcome = remote_units.run_units(cfg, product, units=[uid], workdir=tmp_path / "w2",
                                     environ={}, estimate=lambda u: 10.0)
    assert outcome["ran"] == [uid]
    remote_units.merge(product, config=cfg)
    assert _predictions(product) == _predictions(direct)


def test_the_deadline_before_a_unit_with_no_time_left_writes_partial_and_runs_nothing(
        tmp_path, torch_flags):
    """The unit-level check still holds: an arrived deadline runs no unit."""
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    ids = ["search.phase1.L1-to-L6.common"]
    outcome = _run(cfg, product, tmp_path, ids, calls, deadline=0.0, clock=lambda: 1.0)
    assert calls == [] and outcome["remaining"] == ids
    marker = json.loads((tmp_path / "work" / remote_units.PARTIAL_MARKER).read_text())
    assert marker["remaining"] == ids


# ------------------------------------------------------------------ R2: currency


def test_merge_refuses_a_shard_whose_stamp_is_not_the_current_campaigns(
        tmp_path, torch_flags):
    """Shards that agree with each other but carry a stale commit are refused."""
    cfg, product, ids, shards = _two_shards(tmp_path)
    for field, value in (("commit", "0" * 40), ("codeVersion", "stale"),
                         ("requirementsDigest", "stale"), ("revision", "v1.md"),
                         ("configDigest", "stale"), ("dataPreparationVersion", "stale")):
        saved = {i: (shards / i / "shard.json").read_text() for i in ids}
        for i in ids:
            _edit_stamp(shards / i / "shard.json", **{field: value})
        with pytest.raises(remote_units.MergeError, match=field):
            remote_units.merge(product, config=cfg)
        assert not (product / "Models").exists()
        for i in ids:
            (shards / i / "shard.json").write_text(saved[i])
    # another configuration is another campaign
    with pytest.raises(remote_units.MergeError, match="configDigest"):
        remote_units.merge(product, config={**cfg, "nTrials": 2})
    assert remote_units.merge(product, config=cfg)["merged"] == ids


def test_resume_refuses_a_completed_shard_whose_stamp_is_not_the_current_campaigns(
        tmp_path, torch_flags):
    cfg, product, ids, shards = _two_shards(tmp_path)
    _edit_stamp(shards / ids[0] / "shard.json", commit="0" * 40)
    calls = []
    with pytest.raises(remote_units.UnitError, match="commit"):
        _run(cfg, product, tmp_path, ids + ["search.phase1.L1-to-L6.MIL-CREDA"], calls)
    assert calls == []


# ------------------------------------------------------------------ R3: data authentication


def _authenticated_set(tmp_path, monkeypatch):
    cfg = {**_config(tmp_path), "synthetic": False, "rehearsal": True}
    root = Path(cfg["processedRoot"])
    rows = [{"file": rel, "bytes": p.stat().st_size, "sha256": preprocess.file_sha256(p)}
            for rel, p in sorted((p.relative_to(root).as_posix(), p)
                                 for p in root.rglob("*") if p.is_file())]
    sha = tmp_path / "processed_sha256.csv"
    pd.DataFrame(rows).to_csv(sha, index=False)
    monkeypatch.setattr(remote_units, "SHA_MANIFEST", sha)
    remote_units._VERIFIED.clear()
    return cfg, root


def test_every_consumed_processed_file_is_verified_against_the_recorded_sha256(
        tmp_path, monkeypatch):
    cfg, root = _authenticated_set(tmp_path, monkeypatch)
    identity = remote_units.data_identity(cfg)
    assert identity["dataChecksum"] == preprocess.file_sha256(root / "manifest.csv")
    tampered = tmp_path / "tampered"
    shutil.copytree(root, tampered)
    npz = sorted((tampered / "signals").glob("*.npz"))[3]
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["signal"] = arrays["signal"] + np.float32(1.0)
    np.savez(npz, **arrays)
    with pytest.raises(remote_units.InputUnavailable, match=f"signals/{npz.name}"):
        remote_units.data_identity({**cfg, "processedRoot": str(tampered)})
    # a consumed file the record does not list is refused too
    pd.read_csv(remote_units.SHA_MANIFEST).iloc[1:].to_csv(remote_units.SHA_MANIFEST,
                                                           index=False)
    remote_units._VERIFIED.clear()
    with pytest.raises(remote_units.InputUnavailable, match="not recorded"):
        remote_units.data_identity(cfg)


def test_the_processed_files_are_verified_once_per_session(tmp_path, monkeypatch):
    cfg, _ = _authenticated_set(tmp_path, monkeypatch)
    remote_units.data_identity(cfg)
    hashed = []
    real = preprocess.file_sha256
    monkeypatch.setattr(preprocess, "file_sha256", lambda p, *a: hashed.append(p) or real(p))
    remote_units.data_identity(cfg)
    names = {Path(p).name for p in hashed}
    assert not any(n.endswith(".npz") for n in names), "no NPZ is hashed twice in a session"


def test_a_run_on_tampered_signals_is_refused_before_any_unit(tmp_path, monkeypatch,
                                                               torch_flags):
    cfg, root = _authenticated_set(tmp_path, monkeypatch)
    npz = sorted((root / "signals").glob("*.npz"))[0]
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["signal"] = arrays["signal"] * np.float32(2.0)
    np.savez(npz, **arrays)
    calls = []
    with pytest.raises(remote_units.InputUnavailable, match=npz.name):
        _run(cfg, _product(tmp_path), tmp_path, ["search.phase1.L1-to-L6.common"], calls)
    assert calls == []


# ------------------------------------------------------------------ R4: smoke evidence


def test_the_smoke_writes_the_rehearsal_shard_json_in_its_working_directory(
        tmp_path, registries, torch_flags, monkeypatch):
    cfg = _config(tmp_path, strategies=("MIL-CREDA",))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    product = _product(tmp_path)
    work = tmp_path / "runner-cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    outcome = remote_units.smoke(config=str(config_path), product=str(product),
                                 environ={"FORGE_RUN_MODE": "smoke"},
                                 estimate=lambda u: 0.0)
    uid = "search.phase1.L1-to-L6.MIL-CREDA"
    assert outcome["ran"] == [uid]
    stamp = json.loads((work / "shard.json").read_text())
    assert stamp["mode"] == "smoke"
    assert stamp["evidence"]["commit"] == remote_units.source_commit()
    assert stamp["evidence"]["sourcesDigest"] == model_store.code_version()
    assert stamp["evidence"]["unit"] == uid and stamp["evidence"]["complete"] is True
    assert stamp["environment"]["device"]["kind"] == "cpu"
    assert stamp["environment"]["device"]["name"]
    assert stamp["environment"]["accelerator"] == "cpu"
    unit_stamp = json.loads((product / remote_units.SHARDS_RELATIVE / uid / "shard.json")
                            .read_text())
    for field in remote_units.IDENTICAL_FIELDS:
        assert stamp["environment"][field] == unit_stamp["environment"][field], field
    assert unit_stamp["format"] == remote_units.FORMAT, "the per-unit stamp stays its own"
    # outside the runner the smoke leaves no rehearsal stamp
    (work / "shard.json").unlink()
    shutil.rmtree(product / remote_units.SHARDS_RELATIVE)
    remote_units.smoke(config=str(config_path), product=str(product), environ={},
                       estimate=lambda u: 0.0)
    assert not (work / "shard.json").exists()


# ------------------------------------------------------------------ device


def test_a_remote_run_must_declare_its_device(tmp_path, torch_flags):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    del cfg["device"]
    with pytest.raises(remote_units.UnitError, match="declare"):
        remote_units.run_units(cfg, product, units=["search.phase1.L1-to-L6.common"],
                               workdir=tmp_path / "work", execute=_fake(calls),
                               environ={remote_units.DEADLINE_ENV: "1e12"},
                               estimate=lambda u: 0.0)
    assert calls == []
    # a local run keeps the campaign's default
    _run(cfg, product, tmp_path, ["search.phase1.L1-to-L6.common"], calls)
    assert len(calls) == 1


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_a_declared_cuda_device_that_is_not_placed_on_cuda_is_refused_before_training(
        tmp_path, torch_flags):
    cfg, product, calls = {**_config(tmp_path), "device": "cuda"}, _product(tmp_path), []
    with pytest.raises(remote_units.UnitError, match="cuda"):
        _run(cfg, product, tmp_path, ["search.phase1.L1-to-L6.common"], calls)
    assert calls == []


def test_the_stamp_records_the_actual_device_placement(tmp_path, torch_flags):
    cfg, product, calls = _config(tmp_path), _product(tmp_path), []
    uid = "search.phase1.L1-to-L6.common"
    _run(cfg, product, tmp_path, [uid], calls)
    env = json.loads((product / remote_units.SHARDS_RELATIVE / uid / "shard.json")
                     .read_text())["environment"]
    assert env["device"]["declared"] == "cpu" and env["device"]["kind"] == "cpu"
    assert env["accelerator"] == "cpu"


# ------------------------------------------------------------------ the remaining pins


def test_requirements_pin_every_installed_runtime_dependency_to_its_installed_version():
    lines = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines()
    for name in ("torchvision", "timm", "tqdm", "matplotlib", "scikit-image", "scikit-learn",
                 "umap-learn", "ipykernel"):
        assert f"{name}=={importlib.metadata.version(name)}" in lines, name


def test_every_pinned_requirement_equals_its_installed_version():
    """A pin describes the environment the suite runs in: each `name==version` line
    names the version importlib.metadata reads in this venv."""
    lines = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines()
    pins = dict(line.split("==", 1) for line in lines
                if "==" in line and not line.lstrip().startswith("#"))
    assert {"torch", "optuna", "scipy", "numpy", "pandas"} <= set(pins)
    wrong = {n: (v, importlib.metadata.version(n)) for n, v in pins.items()
             if importlib.metadata.version(n) != v}
    assert wrong == {}
