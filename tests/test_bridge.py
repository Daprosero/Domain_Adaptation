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
