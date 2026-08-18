"""Putting several machines' runs back together without pretending they were one.

Accuracy is a property of the method and pools freely. Wall time and peak memory
are properties of the machine, and averaging them across two of them yields a
number that describes neither. These tests hold that line, and hold the refusals
that keep a merge from quietly becoming an average.
"""

from __future__ import annotations

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
                     dist={**DIST, "perEnvironment": ["seconds"]})
    assert "peakMiB" in str(raised.value)


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
