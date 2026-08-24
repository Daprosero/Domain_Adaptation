"""The bridge: merged shard data into what the report notebook can actually read.

`shards.merge()`'s own shape — `shardsArrived`, `gridByEnvironment`, `poolable`,
`perEnvironment`, … — shares no keys with what `Benchmark_Phase1_Report.ipynb`
reads: `summary["reduction"]`, `summary["grid"]`, `summary["perTransfer"]`, plus
`runs.jsonl`. `tools/bridge.py` is what turns one into the other, and these tests
hold the two things that make its output honest rather than merely shaped right:
it never lands where a real campaign's record lives, and it never states more
than the shards actually proved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bridge", REPOSITORY / "tools" / "bridge.py")
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)

from MIL_CREDA_Benchmark import config, shards  # noqa: E402

REAL_DIST = shards.declaration()


def _evidence(**overrides):
    evidence = {"commit": "a" * 40, "codeDigest": "b" * 64,
                "importsFrom": "/repo/src/MIL_CREDA_Benchmark",
                "outputs": ["runs.jsonl", "shard.json"]}
    evidence.update(overrides)
    return evidence


def _run(arm, transfer, seed, env, target, source, seconds, **overrides):
    run = {"arm": arm, "transfer": transfer, "seed": seed, "env": env,
           "targetAccuracy": target, "sourceAccuracy": source, "seconds": seconds,
           "contribution": 0.1, "supervised": 0.2, "adaptationShare": 0.3,
           "peakMiB": 50.0, "parameters": 11247434}
    run.update(overrides)
    return run


def _shard(name, seed, target, source, seconds=10.0, epochs=2, env="e1", **stamp):
    base = {"env": env, "epochs": epochs, "seeds": [seed],
            "environment": {"device": {"name": "x86_64", "kind": "cpu"}, "torch": "2.10.0"},
            "evidence": _evidence()}
    base.update(stamp)
    return {"shard": name, "stamp": base,
            "runs": [_run("G", "M->U", seed, env, target, source, seconds)]}


def _rung_shard(name, seed, target, source, seconds=10.0, epochs=2, env="e1", **stamp):
    """Like `_shard`, but with BOTH arms of a real `config.LADDER` rung
    (`F`->`G`) -- `_shard`'s own single-arm fixture never reaches a rung
    at all (`ladder_rows` skips a pair unless both sides are present), so
    it cannot exercise `build_summary`'s own call into `ladder_rows` on
    the POOLED grid. This is the shape `tests/test_shards.py`'s own
    `_shard` already uses, for the same reason.
    """
    base = {"env": env, "epochs": epochs, "seeds": [seed],
            "environment": {"device": {"name": "x86_64", "kind": "cpu"}, "torch": "2.10.0"},
            "evidence": _evidence()}
    base.update(stamp)
    return {"shard": name, "stamp": base,
            "runs": [_run(arm, "M->U", seed, env, target, source, seconds)
                     for arm in ("F", "G")]}


def _straggler(name, seed, target, source, **stamp):
    """A shard whose run died mid-way: present, but never sealed — no
    `evidence` key at all, exactly what a crash before `seal_shard_stamp()`
    leaves behind."""
    entry = _rung_shard(name, seed, target, source, **stamp)
    entry["stamp"] = {k: v for k, v in entry["stamp"].items() if k != "evidence"}
    return entry


# --------------------------------------------------------------- scratch location

def test_the_scratch_location_never_lands_under_a_real_results_directory() -> None:
    """A smoke merge's output must never be mistaken for `campaign()`'s own
    record: `run_smoke`'s docstring is explicit that a smoke run never writes
    `summary.json` or `Probe_results.json`, and a merge of five smoke shards
    is exactly that, five times over."""
    assert not bridge.SCRATCH.is_relative_to(config.RESULTS)
    assert not bridge.SCRATCH.is_relative_to(config.PRODUCT / "Results")


def test_the_scratch_directory_name_says_what_it_is() -> None:
    lowered = str(bridge.SCRATCH).lower()
    assert "scratch" in lowered
    assert "smoke" in lowered or "merge" in lowered


# --------------------------------------------------------------------- reduction

def test_the_reduction_is_only_what_identical_across_shards_guarantees_plus_seeds() -> None:
    """Across five machines a `reduction` can only honestly be what the
    declaration actually verified equal — `identicalAcrossShards` — plus the
    union of the seeds that arrived. Nothing invented for the fields a smoke
    shard's stamp never carries at all."""
    found = [_shard("a", 0, 0.5, 0.8), _shard("b", 1, 0.6, 0.7)]
    summary, _ = bridge.build_summary(found)

    assert summary["reduction"] == {"epochs": 2, "seeds": [0, 1]}


def test_a_shard_disagreeing_on_epochs_is_still_refused_before_the_bridge_runs() -> None:
    """The bridge does not weaken `merge()`'s own refusal; it only reshapes
    what a merge that succeeded produces."""
    found = [_shard("a", 0, 0.5, 0.8, epochs=2), _shard("b", 1, 0.6, 0.7, epochs=3)]
    with pytest.raises(shards.ShardsDisagree):
        bridge.build_summary(found)


# -------------------------------------------------------------------------- grid

def test_the_grid_pools_every_dimension_the_replication_showed_was_poolable() -> None:
    """All six machine-independent dimensions — the replication found them
    bit-identical across three runs on two machines — reach the pooled grid,
    correctly averaged across every shard that arrived."""
    found = [_shard("a", 0, target=0.4, source=0.8),
             _shard("b", 1, target=0.6, source=0.6)]
    summary, _ = bridge.build_summary(found)

    cell = summary["grid"]["M->U"]["G"]
    assert set(cell) == {"sourceAccuracy", "targetAccuracy", "contribution",
                         "supervised", "adaptationShare", "parameters"}
    assert cell["targetAccuracy"]["mean"] == pytest.approx(0.5)
    assert cell["sourceAccuracy"]["mean"] == pytest.approx(0.7)
    # Both shards' fixtures agree on these four, so pooling a constant
    # returns the constant.
    assert cell["parameters"]["mean"] == pytest.approx(11247434)


def test_the_real_declaration_now_leaves_nothing_unclassified() -> None:
    """The replication settled every one of `config.DIMENSIONS`: six pool,
    two (`seconds`, `peakMiB`) are `perRun`. `unclassifiedDimensions` is
    still computed and still written — it is the receipt that nothing was
    left out, not a report of a gap."""
    found = [_shard("a", 0, 0.5, 0.8)]
    summary, _ = bridge.build_summary(found)

    assert summary["unclassifiedDimensions"] == []
    # seconds and peakMiB are real dimensions, just not pooled ones — they
    # must not silently vanish from the merge, only from this grid.
    assert "seconds" not in summary["grid"]["M->U"]["G"]
    assert "peakMiB" not in summary["grid"]["M->U"]["G"]


def test_merge_completes_without_keyerror() -> None:
    """`build_summary`'s own call into `harness.ladder_rows` reads the
    POOLED grid `shards.merge()` returns -- `perRun` dimensions
    (`seconds`, `peakMiB`) were never averaged into it and do not exist
    there, so an unfiltered `ladder_rows(cell, label)` call (every
    dimension in `config.DIMENSIONS`) used to raise `KeyError: 'seconds'`
    the instant a real rung (both arms present) reached a `perTransfer`
    computation -- exactly the shape `distribute.py merge` hits in
    practice. `_shard`'s own single-arm fixture used elsewhere in this
    file never reaches a rung at all, which is why this defect passed
    every OTHER test here; `_rung_shard` is what actually exercises it.
    """
    found = [_rung_shard("a", 0, target=0.4, source=0.8),
             _rung_shard("b", 1, target=0.6, source=0.6)]
    summary, _ = bridge.build_summary(found)

    per_transfer = summary["perTransfer"]["M->U"]
    assert per_transfer, "ladder_rows produced no rows for a real F->G rung"
    assert summary["tally"]["M->U"]
    # The pooled grid itself must still carry none of the perRun dims --
    # `ladder_rows` succeeding here is because it was handed the NARROWER
    # set, not because the pooled grid quietly grew `seconds`/`peakMiB`.
    cell = summary["grid"]["M->U"]["G"]
    assert "seconds" not in cell and "peakMiB" not in cell


def test_the_bridge_reads_the_distribution_it_is_given_not_a_hardcoded_one() -> None:
    """Data-driven, not a copy of the real declaration re-typed into the
    bridge: a caller-supplied `dist` with a different poolable set changes
    what the grid carries. Fully classified, unlike the old fixture, because
    the bridge no longer narrows `shards.merge()`'s dimensions to whatever a
    partial `dist` happens to cover — it hands over `config.DIMENSIONS` in
    full and lets the merge refuse if something is missing."""
    found = [_shard("a", 0, 0.5, 0.8)]
    custom = {"axis": "seed", "poolable": ["targetAccuracy"],
              "perEnvironment": ["sourceAccuracy"],
              "perRun": ["seconds", "peakMiB", "contribution", "supervised",
                        "adaptationShare", "parameters"],
              "identicalAcrossShards": ["epochs"]}
    summary, _ = bridge.build_summary(found, dist=custom)
    assert set(summary["grid"]["M->U"]["G"]) == {"targetAccuracy"}
    assert summary["unclassifiedDimensions"] == []


def test_a_dimension_the_given_distribution_leaves_out_still_refuses() -> None:
    """Removing the bridge's restriction must not turn a caller's gap into a
    silent drop: an incomplete `dist` still refuses, exactly as
    `shards.merge()` refuses it directly. Silence is not a classification,
    and that holds whether the caller is a test or a real declaration."""
    found = [_shard("a", 0, 0.5, 0.8)]
    incomplete = {"axis": "seed", "poolable": ["targetAccuracy"],
                  "perEnvironment": ["sourceAccuracy"],
                  "perRun": ["seconds", "contribution", "supervised",
                            "adaptationShare", "parameters"],
                  # peakMiB named nowhere.
                  "identicalAcrossShards": ["epochs"]}
    with pytest.raises(shards.ShardsDisagree) as raised:
        bridge.build_summary(found, dist=incomplete)
    assert "peakMiB" in str(raised.value)


def test_ladder_rows_default_dimensions_unchanged() -> None:
    """`harness.campaign()`'s own call
    (`ladder_rows(cell, label)`, two positional arguments, no `dimensions=`
    at all) must keep reading every dimension in `config.DIMENSIONS` by
    default -- the new optional parameter is additive, not a change to
    the single-machine path's own behaviour. A cell genuinely carrying
    every dimension (one environment's own grid, never the pooled one)
    still produces one row per rung-dimension pair; a cell missing one
    still raises `KeyError`, exactly as before this change.
    """
    full_cell = {
        dimension: {"mean": 1.0, "stdev": 0.0, "n": 1}
        for dimension in bridge.config.DIMENSIONS
    }
    rows = bridge.harness.ladder_rows({"F": full_cell, "G": full_cell}, "M->U")
    assert rows
    assert {row["metric"] for row in rows} == set(bridge.config.DIMENSIONS)

    redacted_cell = dict(full_cell)
    redacted_cell.pop("seconds", None)
    with pytest.raises(KeyError):
        bridge.harness.ladder_rows({"F": redacted_cell, "G": redacted_cell}, "M->U")


def test_the_summary_carries_gridperrun_for_the_report_to_read() -> None:
    """`shards.merge()` already computes `gridPerRun` — every `perRun` reading,
    tagged with its own environment, never averaged. `build_summary` must not
    let it die here: without this key the report notebook has no way to render
    `seconds`/`peakMiB` honestly, and falls back to pooling them across
    machines into a mean that describes none of them."""
    found = [_shard("a", 0, 0.5, 0.8, seconds=12.0, env="e1"),
             _shard("b", 1, 0.6, 0.7, seconds=34.0, env="e2")]
    summary, _ = bridge.build_summary(found)

    per_run = summary["gridPerRun"]["M->U"]["G"]["seconds"]
    assert {r["env"] for r in per_run} == {"e1", "e2"}
    assert {r["value"] for r in per_run} == {12.0, 34.0}


# --------------------------------------------------------------------------- runs

def test_runs_jsonl_is_the_untouched_concatenation_of_every_shard() -> None:
    """The notebook's own tables read every dimension straight off `runs`,
    never off `summary["grid"]` — `seconds`, `contribution` and the rest
    travel here exactly as each shard wrote them, not filtered by whatever
    the declaration did or did not classify."""
    found = [_shard("a", 0, 0.5, 0.8, seconds=12.0),
             _shard("b", 1, 0.6, 0.7, seconds=34.0)]
    _, runs = bridge.build_summary(found)

    assert runs == [found[0]["runs"][0], found[1]["runs"][0]]
    assert {r["seconds"] for r in runs} == {12.0, 34.0}
    assert all("contribution" in r for r in runs)


# ---------------------------------------------------------------------- provenance

def test_the_summary_declares_itself_plumbing_not_a_result() -> None:
    """A reader three months from now must not be able to mistake this for a
    real campaign's `summary.json`."""
    found = [_shard("a", 0, 0.5, 0.8)]
    summary, _ = bridge.build_summary(found)

    assert summary["kind"] != "bounded"
    assert "provenance" in summary
    assert "plumbing" in summary["provenance"].lower()
    assert "run_smoke" in summary["provenance"] or "smoke" in summary["provenance"].lower()


def test_smoke_shards_get_the_smoke_note_byte_identical_and_kind_smokemerge() -> None:
    """`PROVENANCE_NOTE` must be reused verbatim for shards that actually
    are smoke shards -- nothing about the classification may reword the
    note this codebase already ships for the case it was written for."""
    found = [_shard("a", 0, 0.5, 0.8, epochs=2)]
    summary, _ = bridge.build_summary(found)

    assert summary["kind"] == "smokeMerge"
    assert summary["provenance"] == bridge.PROVENANCE_NOTE


def test_full_campaign_shards_get_the_honest_full_campaign_note() -> None:
    """`distribute.py::run_shard` only ever calls `harness.campaign()` at
    `FULL_EPOCHS` -- a shard set agreeing on `epochs=20` (the only mixed
    case `identicalAcrossShards` structurally forbids is smoke-vs-full) must
    read as a full-campaign merge, with a note describing what actually
    ran, not the smoke note's now-false claim about a skipped search."""
    ceilings = {"creda": 0.42, "milcreda": 0.37}
    assert ceilings != bridge.harness.SMOKE_CEILINGS
    found = [
        _rung_shard("a", 0, 0.4, 0.8, epochs=20, ceilings=ceilings),
        _rung_shard("b", 1, 0.6, 0.6, epochs=20, ceilings=ceilings),
    ]
    summary, _ = bridge.build_summary(found)

    assert summary["kind"] == "campaignMerge"
    note = summary["provenance"]
    assert note != bridge.PROVENANCE_NOTE
    assert "full campaigns" in note
    assert "FULL_EPOCHS" in note
    assert "not smoke rehearsals" in note
    assert "Arms: F, G" in note
    assert "transfers: M->U" in note
    assert "epochs per shard: 20" in note
    assert "ceiling search: present" in note
    assert "seeds: 2" in note
    assert "not been promoted, scale-checked, or certified" in note
    assert "MIL-CREDA/Results/Benchmark/" in note


def test_full_campaign_note_says_ceiling_search_absent_when_ceilings_are_the_smoke_default() -> None:
    """The ceiling-search field is read from the shard's own stamp, not
    inferred from `epochs` alone: a full-epoch shard that never actually
    ran a search (still carrying the declared neutral) must say so."""
    found = [
        _rung_shard("a", 0, 0.4, 0.8, epochs=20, ceilings=dict(bridge.harness.SMOKE_CEILINGS)),
    ]
    summary, _ = bridge.build_summary(found)

    assert "ceiling search: absent" in summary["provenance"]


def test_the_provenance_note_renders_a_count_not_a_list() -> None:
    """`_classify_provenance`'s `shards_arrived` parameter must receive an
    integer count, never `merged["shardsArrived"]` itself -- passing the
    list renders as `"This summary merges ['a', 'b', 'c'] shards"`."""
    found = [
        _rung_shard("a", 0, 0.4, 0.8, epochs=20, ceilings={"creda": 0.5, "milcreda": 0.5}),
        _rung_shard("b", 1, 0.5, 0.7, epochs=20, ceilings={"creda": 0.5, "milcreda": 0.5}),
        _rung_shard("c", 2, 0.6, 0.6, epochs=20, ceilings={"creda": 0.5, "milcreda": 0.5}),
    ]
    summary, _ = bridge.build_summary(found)
    assert "This summary merges 3 shards" in summary["provenance"]


def test_full_campaign_note_never_claims_self_certification() -> None:
    found = [_rung_shard("a", 0, 0.4, 0.8, epochs=20, ceilings={"creda": 0.9, "milcreda": 0.9})]
    summary, _ = bridge.build_summary(found)

    assert "not been promoted" in summary["provenance"]
    assert "no scientific claim beyond what is stated here" in summary["provenance"]


# --------------------------------------------------------------- relaunch wiring

def test_a_straggler_no_longer_aborts_the_batch_and_is_reported_in_relaunch() -> None:
    """A1: `merge()` must receive only the sealed-complete shards -- the
    straggler is filtered out before `merge()` ever sees it, so the batch
    succeeds instead of raising `ShardIncomplete` for everything. Reported,
    not silently dropped: `summary["relaunch"]["shards"]` names it, with
    `reason: "incomplete"` and its own missing evidence paths."""
    plan = {"shards": [{"id": "a", "seeds": [0]}, {"id": "b", "seeds": [1]},
                       {"id": "c", "seeds": [2]}]}
    found = [_rung_shard("a", 0, 0.4, 0.8, epochs=20, ceilings={"creda": 0.5, "milcreda": 0.5}),
             _rung_shard("b", 1, 0.5, 0.7, epochs=20, ceilings={"creda": 0.5, "milcreda": 0.5}),
             _straggler("c", 2, 0.6, 0.6, epochs=20, ceilings={"creda": 0.5, "milcreda": 0.5})]

    summary, _ = bridge.build_summary(found, plan=plan)

    assert sorted(summary["shardsArrived"]) == ["a", "b"]
    relaunch_ids = {entry["id"]: entry for entry in summary["relaunch"]["shards"]}
    assert relaunch_ids["c"]["reason"] == "incomplete"
    assert relaunch_ids["c"]["seeds"] == [2]
    assert "evidence.commit" in relaunch_ids["c"]["missing"]


def test_all_incomplete_shards_raise_shardincomplete_not_indexerror() -> None:
    """A11: naming every incomplete id and its missing evidence, raised
    directly from `build_summary` before it ever indexes `stamps[0]` --
    which does not exist when every shard here is unsealed."""
    found = [_straggler("a", 0, 0.4, 0.8, epochs=20),
             _straggler("b", 1, 0.5, 0.7, epochs=20)]

    with pytest.raises(shards.ShardIncomplete) as raised:
        bridge.build_summary(found)
    message = str(raised.value)
    assert "a" in message and "b" in message
    assert "evidence.commit" in message


def test_an_absent_plan_reports_unknown_never_zero() -> None:
    """A6: `merge()` collapses `missing` to `0` when `expected is None` --
    `merge()` may not be edited, so `build_summary` must set
    `shardsExpected`/`missing` to `None` itself rather than copy `merge()`'s
    own values through, or the honest "not recorded" becomes a false
    "nothing missing"."""
    found = [_shard("a", 0, 0.5, 0.8)]
    summary, _ = bridge.build_summary(found)

    assert summary["shardsExpected"] is None
    assert summary["missing"] is None
    assert summary["relaunch"] == {
        "planRecorded": False, "complete": None, "unplanned": [], "shards": []}


def test_a_recorded_plan_yields_integer_shardsexpected_and_missing() -> None:
    plan = {"shards": [{"id": "a", "seeds": [0]}, {"id": "b", "seeds": [1]}]}
    found = [_shard("a", 0, 0.5, 0.8)]
    summary, _ = bridge.build_summary(found, plan=plan)

    assert summary["shardsExpected"] == 2
    assert summary["missing"] == 1
    assert summary["relaunch"]["planRecorded"] is True


def test_relaunch_key_carries_verbatim_ids_and_seeds_from_the_plan() -> None:
    """A7: never re-derived through `shard_seeds()` -- sourced from the
    plan exactly as recorded."""
    plan = {"shards": [{"id": "a", "seeds": [0]}, {"id": "s02", "seeds": [12, 13]}]}
    found = [_shard("a", 0, 0.5, 0.8)]
    summary, _ = bridge.build_summary(found, plan=plan)

    assert summary["relaunch"]["shards"] == [
        {"id": "s02", "seeds": [12, 13], "reason": "missing", "missing": []}]


def test_bridge_reads_the_plan_from_disk_and_threads_it_through(tmp_path, monkeypatch) -> None:
    """`bridge()` itself resolves `shards.read_plan()` -- `build_summary()`
    stays pure and never reads one on its own."""
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    shards.write_plan({"shards": [{"id": "a", "seeds": [0]}, {"id": "b", "seeds": [1]}]})

    found = [_shard("a", 0, 0.5, 0.8)]
    written = bridge.bridge(found=found, root=tmp_path / "scratch")
    summary = json.loads(written["summary"].read_text())

    assert summary["shardsExpected"] == 2
    assert summary["missing"] == 1
    assert summary["relaunch"]["shards"] == [
        {"id": "b", "seeds": [1], "reason": "missing", "missing": []}]


# -------------------------------------------------------------------------- write

def test_write_scratch_writes_summary_runs_and_a_readable_provenance_marker(
        tmp_path) -> None:
    found = [_shard("a", 0, 0.5, 0.8), _shard("b", 1, 0.6, 0.7)]
    summary, runs = bridge.build_summary(found)
    written = bridge.write_scratch(summary, runs, root=tmp_path)

    assert json.loads(written["summary"].read_text())["kind"] == summary["kind"]
    lines = written["runs"].read_text().splitlines()
    assert len(lines) == len(runs)
    assert "PLUMBING" in written["provenance"].read_text().upper()


def test_write_scratch_provenance_file_matches_a_full_campaign_summary_too(tmp_path) -> None:
    """`write_scratch` must never write `PROVENANCE_NOTE` unconditionally --
    the marker file has to carry the SAME text `summary["provenance"]` does,
    whichever branch produced it."""
    found = [_rung_shard("a", 0, 0.4, 0.8, epochs=20, ceilings={"creda": 0.5, "milcreda": 0.5})]
    summary, runs = bridge.build_summary(found)
    written = bridge.write_scratch(summary, runs, root=tmp_path)

    assert summary["kind"] == "campaignMerge"
    assert written["provenance"].read_text() == summary["provenance"] + "\n"
    assert "PLUMBING" not in written["provenance"].read_text()


def test_the_bridge_refuses_when_no_shards_came_back() -> None:
    with pytest.raises(SystemExit):
        bridge.bridge(found=[])


def test_the_bridge_end_to_end_uses_the_real_declaration(tmp_path) -> None:
    """The full path, against the module's own real `shards.declaration()` —
    not a test fixture standing in for it."""
    found = [_shard("a", 0, 0.5, 0.8), _shard("b", 1, 0.6, 0.7)]
    written = bridge.bridge(found=found, root=tmp_path)
    summary = json.loads(written["summary"].read_text())
    assert summary["reduction"] == {"epochs": 2, "seeds": [0, 1]}
    assert set(summary["grid"]["M->U"]["G"]) == set(REAL_DIST["poolable"])
