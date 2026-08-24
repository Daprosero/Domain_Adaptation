"""Putting several machines' runs back together without pretending they were one.

Accuracy is a property of the method and pools freely. Wall time and peak memory
are properties of the machine, and averaging them across two of them yields a
number that describes neither. These tests hold that line, and hold the refusals
that keep a merge from quietly becoming an average.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from MIL_CREDA_Benchmark import config, harness, shards

DIST = {
    "axis": "seed",
    "poolable": ["targetAccuracy", "sourceAccuracy", "contribution",
                 "supervised", "adaptationShare", "parameters"],
    "perEnvironment": ["seconds", "peakMiB"],
    "identicalAcrossShards": ["revision", "epochs", "ceilings"],
}


def _run(arm, transfer, seed, env, accuracy, seconds):
    return {"arm": arm, "transfer": transfer, "seed": seed, "env": env,
            "targetAccuracy": accuracy, "sourceAccuracy": 0.8, "seconds": seconds,
            "contribution": 0.1, "supervised": 0.2, "adaptationShare": 0.3,
            "peakMiB": 50.0, "parameters": 11181642}


def _complete_evidence(**overrides):
    evidence = {"commit": "a" * 40, "codeDigest": "b" * 64,
                "importsFrom": "/repo/src/MIL_CREDA_Benchmark",
                "outputs": ["runs.jsonl", "shard.json"]}
    evidence.update(overrides)
    return evidence


def _shard(name, env, seeds, accuracy=0.5, seconds=10.0, **stamp):
    base = {"revision": "r17.md", "epochs": 20, "seeds": list(seeds),
            "ceilings": {"creda": 1e-4}, "env": env,
            "environment": {"device": {"name": name, "kind": "cuda"}, "torch": "2.13.0"},
            "evidence": _complete_evidence()}
    base.update(stamp)
    # Both arms of a rung, or `ladder_rows` skips the pair and never reaches a
    # dimension — which would make the redaction test pass for the wrong reason.
    return {"shard": name, "stamp": base,
            "runs": [_run(arm, "M->U", s, env, accuracy, seconds)
                     for s in seeds for arm in ("F", "G")]}


def test_the_pooled_grid_carries_only_what_pools():
    """A machine-described dimension pooled across machines describes neither."""
    merged = shards.merge([_shard("a", "e1", [0, 1]), _shard("b", "e2", [2, 3])],
                          dimensions=config.DIMENSIONS, dist=DIST)
    cell = merged["grid"]["M->U"]["G"]
    assert "targetAccuracy" in cell
    assert "seconds" not in cell and "peakMiB" not in cell


def test_each_environment_keeps_every_dimension():
    """`ladder_rows` iterates all of DIMENSIONS and would raise on a redacted cell.

    Reachable red: hand it the pooled grid and it raises; hand it one
    environment's grid and it is the function it always was.
    """
    merged = shards.merge([_shard("a", "e1", [0, 1], seconds=10.0),
                           _shard("b", "e2", [2, 3], seconds=90.0)],
                          dimensions=config.DIMENSIONS, dist=DIST)
    for key, grid in merged["gridByEnvironment"].items():
        assert set(grid["M->U"]["G"]) == set(config.DIMENSIONS)
        harness.ladder_rows(grid["M->U"], "M->U")

    with pytest.raises(KeyError):
        harness.ladder_rows(merged["grid"]["M->U"], "M->U")


def test_two_machines_are_never_averaged_into_one_cost():
    """Ten seconds and ninety are not fifty on any machine that ran."""
    merged = shards.merge([_shard("a", "e1", [0, 1], seconds=10.0),
                           _shard("b", "e2", [2, 3], seconds=90.0)],
                          dimensions=config.DIMENSIONS, dist=DIST)
    costs = sorted(g["M->U"]["G"]["seconds"]["mean"]
                   for g in merged["gridByEnvironment"].values())
    assert costs == [10.0, 90.0]


def test_shards_that_disagree_on_what_must_match_are_refused():
    """A different epoch count is a different experiment, not different hardware."""
    with pytest.raises(shards.ShardsDisagree) as raised:
        shards.merge([_shard("a", "e1", [0, 1]),
                      _shard("b", "e2", [2, 3], epochs=3)],
                     dimensions=config.DIMENSIONS, dist=DIST)
    assert "epochs" in str(raised.value)


def test_a_dimension_in_neither_half_is_refused_rather_than_dropped():
    """Silently dropping it would leave a column nobody notices is gone."""
    with pytest.raises(shards.ShardsDisagree) as raised:
        shards.merge([_shard("a", "e1", [0])], dimensions=config.DIMENSIONS,
                     dist={**DIST, "perEnvironment": ["seconds"], "perRun": []})
    assert "peakMiB" in str(raised.value)


def test_a_dimension_named_in_none_of_the_three_groups_is_still_refused():
    """Adding `perRun` as a third home must not loosen the refusal: a
    dimension absent from all three groups is exactly as unclassified as one
    absent from two, and silence is still not a classification."""
    with pytest.raises(shards.ShardsDisagree) as raised:
        shards.merge([_shard("a", "e1", [0])], dimensions=config.DIMENSIONS,
                     dist={**DIST, "perRun": ["seconds"], "perEnvironment": []})
    assert "peakMiB" in str(raised.value)


# ------------------------------------------------------------------- perRun

PER_RUN_DIST = {
    "axis": "seed",
    "poolable": ["targetAccuracy", "sourceAccuracy", "contribution",
                 "supervised", "adaptationShare", "parameters"],
    "perEnvironment": [],
    "perRun": ["seconds", "peakMiB"],
    "identicalAcrossShards": ["revision", "epochs", "ceilings"],
}


def test_partition_returns_the_three_groups_the_declaration_actually_holds():
    poolable, per_environment, per_run = shards.partition(config.DIMENSIONS, PER_RUN_DIST)
    assert set(poolable) == set(PER_RUN_DIST["poolable"])
    assert per_environment == []
    assert set(per_run) == {"seconds", "peakMiB"}


def test_a_perrun_dimension_never_reaches_the_pooled_grid():
    """Averaging `seconds` across shards would claim a stable central value
    the control run disproved — the same reason it is not `poolable`."""
    merged = shards.merge([_shard("a", "e1", [0, 1]), _shard("b", "e2", [2, 3])],
                          dimensions=config.DIMENSIONS, dist=PER_RUN_DIST)
    cell = merged["grid"]["M->U"]["G"]
    assert "seconds" not in cell and "peakMiB" not in cell


def test_a_perrun_dimension_never_reaches_the_per_environment_grid_either():
    """`gridByEnvironment` groups by machine, which is exactly the framing
    the replication's control run (same machine, three different `seconds`
    and `peakMiB` values) disproved. A `perRun` dimension is not a property
    of the machine either, so it stays out of this grid too."""
    merged = shards.merge([_shard("a", "e1", [0, 1]), _shard("b", "e2", [2, 3])],
                          dimensions=config.DIMENSIONS, dist=PER_RUN_DIST)
    for env_grid in merged["gridByEnvironment"].values():
        cell = env_grid["M->U"]["G"]
        assert "seconds" not in cell and "peakMiB" not in cell


def test_a_perrun_dimension_surfaces_as_each_runs_own_reading():
    """Neither a property of the method (pooled) nor of the machine
    (per-environment): `gridPerRun` carries every run's own value, tagged
    with the shard that produced it, and computes no mean over them — a mean
    would imply the stable central value these quantities do not have.
    """
    merged = shards.merge([_shard("a", "e1", [0, 1], seconds=10.0),
                           _shard("b", "e2", [2, 3], seconds=90.0)],
                          dimensions=config.DIMENSIONS, dist=PER_RUN_DIST)
    readings = merged["gridPerRun"]["M->U"]["G"]["seconds"]
    assert sorted(r["value"] for r in readings) == [10.0, 10.0, 90.0, 90.0]
    # Each reading keeps the environment that produced it, which a bare
    # average would erase.
    assert {r["env"] for r in readings} == {"e1", "e2"}
    # No aggregate key anywhere in the structure: `readings` is a plain list
    # of per-run dicts, never a `{"mean": ...}` shape.
    assert isinstance(readings, list)
    for reading in readings:
        assert set(reading) == {"env", "seed", "value"}


def test_the_perrun_group_is_echoed_on_the_merge_result():
    """Same pattern as `poolable`/`perEnvironment`: the classification the
    merge actually used is echoed back, not only usable indirectly."""
    merged = shards.merge([_shard("a", "e1", [0])], dimensions=config.DIMENSIONS,
                          dist=PER_RUN_DIST)
    assert merged["perRun"] == ["seconds", "peakMiB"]


def test_scale_is_recomputed_from_what_arrived():
    """Three shards planned and two back is a smaller campaign, not a failed one."""
    merged = shards.merge([_shard("a", "e1", [0]), _shard("b", "e2", [1])],
                          expected=3, dimensions=config.DIMENSIONS, dist=DIST)
    assert merged["missing"] == 1
    assert merged["seeds"] == [0, 1]
    assert merged["verdictsMeaningful"] is False

    enough = shards.merge([_shard("a", "e1", [0, 1, 2])], expected=1,
                          dimensions=config.DIMENSIONS, dist=DIST)
    assert enough["verdictsMeaningful"] is True


def test_the_median_is_taken_over_every_measurement_and_chosen_among_what_arrived():
    """Recomputing over the survivors would pick the median of the survivors.

    This picks the campaign's median and then takes the nearest thing to it that
    exists, and says how much it fell short.
    """
    runs = [_run("G", "M->U", seed, "e1", accuracy, 10.0)
            for seed, accuracy in enumerate((0.10, 0.90, 0.50, 0.30, 0.70))]
    wanted = harness.median_seeds(runs, "G")
    assert wanted == {2, 3, 4}

    arrived = {("G", "M->U", 2), ("G", "M->U", 4)}
    kept = shards.promote_candidates(runs, arrived)["G|M->U"]
    assert kept["wanted"] == [2, 3, 4]
    assert kept["chosen"] == [2, 4]
    assert kept["shortfall"] == 1


def test_shards_are_enumerated_from_the_disk(tmp_path, monkeypatch):
    """A shard that never arrived is absent, not reported empty."""
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    home = tmp_path / harness.SHARDS_DIR / "k01"
    home.mkdir(parents=True)
    (home / "shard.json").write_text(json.dumps({"shard": "k01", "epochs": 20}))
    (home / "runs.jsonl").write_text(json.dumps(_run("G", "M->U", 0, "e1", 0.5, 1.0)) + "\n")

    found = shards.read_shards()
    assert [e["shard"] for e in found] == ["k01"]
    assert len(found[0]["runs"]) == 1


# -------------------------------------------------------- merge refuses incomplete

def test_required_evidence_names_the_four_evidence_paths_and_the_shard_shape():
    assert shards.REQUIRED_EVIDENCE == [
        "evidence.commit", "evidence.codeDigest", "evidence.importsFrom",
        "evidence.outputs", "environment.device.kind", "environment.torch",
        "seeds", "epochs",
    ]


def test_completeness_is_the_forge_s_own_service_blind_predicate_reexported():
    """No field name is pushed into the forge: this repository's own
    `REQUIRED_EVIDENCE` is what supplies the vocabulary, and the predicate
    itself names nothing of its own — verified with foreign field names."""
    assert shards.completeness({"widget": {"gizmo": 1}}, ["widget.gizmo"]) == {
        "complete": True, "missing": []}
    assert shards.completeness({}, ["widget.gizmo"]) == {
        "complete": False, "missing": ["widget.gizmo"]}


def test_merge_accepts_a_fully_sealed_shard():
    """A complete stamp merges exactly as before evidence stamping existed."""
    merged = shards.merge([_shard("a", "e1", [0, 1])],
                          dimensions=config.DIMENSIONS, dist=DIST)
    assert merged["shardsArrived"] == ["a"]


def test_merge_refuses_an_incomplete_shard_naming_it_and_its_missing_paths():
    """A shard whose run died mid-way — or whose stamp predates evidence
    stamping entirely — is not merge-eligible. `ShardsDisagree` is the wrong
    exception for this: the shards here do not disagree with each other, one
    of them is simply not proven finished."""
    incomplete = _shard("b", "e2", [2, 3])
    incomplete["stamp"] = {k: v for k, v in incomplete["stamp"].items()
                           if k != "evidence"}

    with pytest.raises(shards.ShardIncomplete) as raised:
        shards.merge([_shard("a", "e1", [0, 1]), incomplete],
                     dimensions=config.DIMENSIONS, dist=DIST)
    assert "b" in str(raised.value)
    assert "evidence.commit" in str(raised.value)
    assert "evidence.outputs" in str(raised.value)


def test_an_old_stamp_with_no_evidence_at_all_reads_incomplete_never_invalid(
        tmp_path, monkeypatch):
    """Rollback strands nothing: an old `shard.json` from before evidence
    stamping existed still parses and lists normally through `read_shards`,
    and only refuses at `merge()`, never at read time."""
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    home = tmp_path / harness.SHARDS_DIR / "old"
    home.mkdir(parents=True)
    (home / "shard.json").write_text(json.dumps(
        {"shard": "old", "epochs": 20, "revision": "r17.md",
         "ceilings": {"creda": 1e-4}, "env": "e1",
         "environment": {"device": {"name": "old", "kind": "cuda"}}}))
    (home / "runs.jsonl").write_text(
        json.dumps(_run("G", "M->U", 0, "e1", 0.5, 1.0)) + "\n")

    found = shards.read_shards()
    assert [e["shard"] for e in found] == ["old"]

    result = shards.completeness(found[0]["stamp"], shards.REQUIRED_EVIDENCE)
    assert result["complete"] is False
    assert "evidence.commit" in result["missing"]

    with pytest.raises(shards.ShardIncomplete):
        shards.merge(found, dimensions=config.DIMENSIONS, dist=DIST)


# ---------------------------------------------------------- split_complete

def test_split_complete_passes_a_sealed_shard_through_merge_shaped():
    """A complete entry survives the split untouched — directly passable to
    `merge()`, not reshaped."""
    complete_shard = _shard("a", "e1", [0, 1])
    complete, incomplete = shards.split_complete([complete_shard])
    assert complete == [complete_shard]
    assert incomplete == []


def test_split_complete_reshapes_an_incomplete_entry_to_id_and_missing():
    """Deliberately a DIFFERENT shape from a complete entry: `{"id",
    "missing"}`, never `{"shard", "stamp", "runs"}`. Same shape would let an
    incomplete entry reach `merge()` by mistake and be pooled unproven; a
    different shape turns that mistake into a `KeyError` at the boundary."""
    incomplete_shard = _shard("b", "e2", [2, 3])
    incomplete_shard["stamp"] = {k: v for k, v in incomplete_shard["stamp"].items()
                                 if k != "evidence"}
    complete, incomplete = shards.split_complete([incomplete_shard])
    assert complete == []
    assert len(incomplete) == 1
    assert set(incomplete[0]) == {"id", "missing"}
    assert incomplete[0]["id"] == "b"
    assert "evidence.commit" in incomplete[0]["missing"]
    assert "evidence.outputs" in incomplete[0]["missing"]


def test_split_complete_separates_a_mixed_batch():
    complete_shard = _shard("a", "e1", [0])
    straggler = _shard("b", "e2", [1])
    straggler["stamp"] = {k: v for k, v in straggler["stamp"].items() if k != "evidence"}
    complete, incomplete = shards.split_complete([complete_shard, straggler])
    assert [e["shard"] for e in complete] == ["a"]
    assert [e["id"] for e in incomplete] == ["b"]


# ------------------------------------------------------- plan persistence

def test_plan_path_lives_beside_the_shard_directories_never_inside_one(tmp_path):
    home = tmp_path / harness.SHARDS_DIR
    assert shards.plan_path(home) == home / "plan.json"


def test_read_plan_is_none_when_nothing_was_ever_recorded(tmp_path):
    """Never `{}`, never a plan synthesized from what arrived — a caller
    must be able to tell "nothing was ever launched" apart from "everything
    launched came back."""
    assert shards.read_plan(tmp_path) is None


def test_write_plan_then_read_plan_round_trips(tmp_path):
    record = {"shards": [{"id": "s00", "seeds": [1, 2, 3]}],
             "epochs": 20, "accelerator": "NvidiaTeslaT4"}
    path = shards.write_plan(record, tmp_path)
    assert path == tmp_path / "plan.json"

    read_back = shards.read_plan(tmp_path)
    assert read_back["shards"] == [{"id": "s00", "seeds": [1, 2, 3]}]
    assert read_back["epochs"] == 20
    assert read_back["accelerator"] == "NvidiaTeslaT4"
    assert "writtenAt" in read_back


def test_write_plan_is_idempotent_on_the_identical_id_and_seeds(tmp_path):
    record = {"shards": [{"id": "s00", "seeds": [1, 2, 3]}], "epochs": 20}
    shards.write_plan(record, tmp_path)
    first_written_at = shards.read_plan(tmp_path)["writtenAt"]

    shards.write_plan(record, tmp_path)
    second = shards.read_plan(tmp_path)
    assert second["shards"] == [{"id": "s00", "seeds": [1, 2, 3]}]
    # Only writtenAt refreshes; the recorded shard entry itself is untouched.
    assert "writtenAt" in second and first_written_at is not None


def test_write_plan_appends_a_new_id_without_disturbing_the_recorded_one(tmp_path):
    shards.write_plan({"shards": [{"id": "s00", "seeds": [1, 2, 3]}]}, tmp_path)
    shards.write_plan({"shards": [{"id": "s01", "seeds": [4, 5, 6]}]}, tmp_path)

    record = shards.read_plan(tmp_path)
    assert record["shards"] == [
        {"id": "s00", "seeds": [1, 2, 3]},
        {"id": "s01", "seeds": [4, 5, 6]},
    ]


def test_write_plan_epochs_and_accelerator_are_last_write_wins(tmp_path):
    shards.write_plan({"shards": [{"id": "s00", "seeds": [1]}],
                       "epochs": 20, "accelerator": "NvidiaTeslaT4"}, tmp_path)
    shards.write_plan({"shards": [{"id": "s01", "seeds": [2]}],
                       "epochs": 20, "accelerator": "NvidiaTeslaP100"}, tmp_path)

    record = shards.read_plan(tmp_path)
    assert record["accelerator"] == "NvidiaTeslaP100"


def test_write_plan_refuses_a_recorded_id_under_different_seeds(tmp_path):
    """The load-bearing guard: `s00` already recorded under `[1, 2, 3]`. A
    second write mapping `s00` to `[7, 8, 9]` must refuse rather than
    silently rewrite the recorded seeds — `harness.shard_paths()` keys
    `s00`'s entire on-disk home on the id string alone."""
    shards.write_plan({"shards": [{"id": "s00", "seeds": [1, 2, 3]}]}, tmp_path)

    with pytest.raises(shards.PlanConflict) as raised:
        shards.write_plan({"shards": [{"id": "s00", "seeds": [7, 8, 9]}]}, tmp_path)
    message = str(raised.value)
    assert "s00" in message
    assert "[1, 2, 3]" in message
    assert "[7, 8, 9]" in message

    # Refused, so the record itself must be untouched.
    assert shards.read_plan(tmp_path)["shards"] == [{"id": "s00", "seeds": [1, 2, 3]}]


# ------------------------------------------------------- relaunch computation

def test_relaunch_names_an_incomplete_shard_with_its_missing_evidence_paths():
    plan = {"shards": [{"id": "s01", "seeds": [12, 13]}]}
    incomplete = [{"id": "s01", "missing": ["evidence.commit", "evidence.outputs"]}]

    result = shards.relaunch(plan, complete=[], incomplete=incomplete)
    assert result["shards"] == [{"id": "s01", "seeds": [12, 13], "reason": "incomplete",
                                 "missing": ["evidence.commit", "evidence.outputs"]}]


def test_relaunch_names_a_never_arrived_shard_as_missing():
    plan = {"shards": [{"id": "s02", "seeds": [20, 21]}]}

    result = shards.relaunch(plan, complete=[], incomplete=[])
    assert result["shards"] == [{"id": "s02", "seeds": [20, 21], "reason": "missing",
                                 "missing": []}]


def test_relaunch_entries_copy_id_and_seeds_verbatim_never_re_derived():
    """A9/A7: the entry's seeds must be sourced from the plan exactly as
    recorded, never recomputed via `shard_seeds()` over a subset."""
    plan = {"shards": [{"id": "s02", "seeds": [12, 13]}]}
    result = shards.relaunch(plan, complete=[], incomplete=[])
    assert result["shards"][0]["id"] == "s02"
    assert result["shards"][0]["seeds"] == [12, 13]


def test_relaunch_with_no_plan_recorded_is_an_honest_unknown():
    """Never `True` or `False` — both would claim a fact nobody measured."""
    result = shards.relaunch(None, complete=[], incomplete=[])
    assert result == {"planRecorded": False, "complete": None, "unplanned": [], "shards": []}


def test_relaunch_reports_an_unplanned_arrived_shard_separately():
    """A10: an arrived, sealed shard whose id is not in the plan merges
    normally -- reported under `unplanned`, never as a relaunch entry, and
    never able to make `complete` false."""
    plan = {"shards": [{"id": "s00", "seeds": [1, 2, 3]}]}
    complete = [{"shard": "s00", "stamp": {}, "runs": []},
               {"shard": "extra", "stamp": {}, "runs": []}]

    result = shards.relaunch(plan, complete=complete, incomplete=[])
    assert result["unplanned"] == ["extra"]
    assert result["shards"] == []
    assert result["complete"] is True


def test_relaunch_complete_is_true_only_when_every_planned_id_is_sealed():
    plan = {"shards": [{"id": "s00", "seeds": [1]}, {"id": "s01", "seeds": [2]}]}
    complete = [{"shard": "s00", "stamp": {}, "runs": []}]

    partial = shards.relaunch(plan, complete=complete, incomplete=[])
    assert partial["complete"] is False
    assert [entry["id"] for entry in partial["shards"]] == ["s01"]

    complete.append({"shard": "s01", "stamp": {}, "runs": []})
    full = shards.relaunch(plan, complete=complete, incomplete=[])
    assert full["complete"] is True
    assert full["shards"] == []


# -------------------------------------- the load-bearing inversion, both halves

def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_plan_s_refusal_is_load_bearing_a_bypassed_relaunch_overwrites_s00(
        tmp_path, monkeypatch):
    """Both halves, in one box, never against real results and never
    through `harness.campaign()` itself:

    Half one -- the guard: `s00` already recorded and sealed under seeds
    `[1, 2, 3]`. A second `write_plan()` mapping `s00` to `[7, 8, 9]` must
    refuse, naming both seed lists, and leave `s00`'s own files untouched
    (both digests unchanged).

    Half two -- what the guard is protecting against: a `campaign()`-shaped
    stand-in (`runs.jsonl` opened `\"w\"`, `shard.json` rewritten -- exactly
    what `harness.campaign()`'s own docstring says a real run does)
    executed directly against `harness.shard_paths(\"s00\")`, bypassing the
    guard entirely. Both digests change. This is what makes half one
    load-bearing: without this half, a refusal test alone would never show
    that the fact it refuses to lose was worth losing.
    """
    monkeypatch.setattr(config, "RESULTS", tmp_path)

    paths = harness.shard_paths("s00")
    paths["stamp"].parent.mkdir(parents=True, exist_ok=True)
    sealed_stamp = _shard("s00", "e1", [1, 2, 3])["stamp"]
    paths["stamp"].write_text(json.dumps(sealed_stamp, indent=2), encoding="utf-8")
    with paths["runs"].open("w", encoding="utf-8") as handle:
        for seed in (1, 2, 3):
            handle.write(json.dumps(_run("G", "M->U", seed, "e1", 0.9, 10.0)) + "\n")

    shards.write_plan({"shards": [{"id": "s00", "seeds": [1, 2, 3]}]})

    stamp_before = _digest(paths["stamp"])
    runs_before = _digest(paths["runs"])

    # Half one: the guard refuses, and s00's own files are untouched.
    with pytest.raises(shards.PlanConflict) as raised:
        shards.write_plan({"shards": [{"id": "s00", "seeds": [7, 8, 9]}]})
    message = str(raised.value)
    assert "s00" in message
    assert "[1, 2, 3]" in message
    assert "[7, 8, 9]" in message
    assert _digest(paths["stamp"]) == stamp_before
    assert _digest(paths["runs"]) == runs_before

    # Half two: what would have happened without the guard. No
    # `harness.campaign()` call, no training, nothing against Kaggle --
    # only the two writes its own docstring documents, executed directly
    # against the SAME paths a naive relaunch over seeds [7, 8, 9] would
    # resolve to, since `shard_paths()` keys them on the id string alone.
    with paths["runs"].open("w", encoding="utf-8") as handle:
        for seed in (7, 8, 9):
            handle.write(json.dumps(_run("G", "M->U", seed, "e1", 0.1, 99.0)) + "\n")
    relaunched_stamp = _shard("s00", "e1", [7, 8, 9])["stamp"]
    paths["stamp"].write_text(json.dumps(relaunched_stamp, indent=2), encoding="utf-8")

    assert _digest(paths["stamp"]) != stamp_before
    assert _digest(paths["runs"]) != runs_before
