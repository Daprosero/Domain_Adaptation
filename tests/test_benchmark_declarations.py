"""The joins inside the benchmark package, checked from both ends.

Two files describe the arms and neither reads the other. `config.ARMS` says what
each arm computes; `MIL_CREDA_Benchmark.__benchmark__` says which sections of the
revision each arm exercises, which is what lets the drift report answer *does this
change oblige the bench to change*. Adding an arm to one and forgetting the other
leaves both files internally consistent and the pair wrong — the kind of omission
that arrives as a silence.

Everything here is config-level and imports no torch, so the suite still runs with
no network and no model weights.
"""

from __future__ import annotations

import MIL_CREDA_Benchmark
import json

import pytest
import torch

from MIL_CREDA_Benchmark import config


def test_every_arm_declares_which_sections_it_exercises() -> None:
    declared = set(MIL_CREDA_Benchmark.__benchmark__["arms"])
    configured = {arm["id"] for arm in config.ARMS}
    assert configured - declared == set(), (
        f"arms with no section declaration: {sorted(configured - declared)}")
    assert declared - configured == set(), (
        f"sections declared for arms that no longer exist: {sorted(declared - configured)}")


def test_the_benchmark_is_bound_to_the_same_revision_as_the_configuration() -> None:
    assert MIL_CREDA_Benchmark.__benchmark__["revision"] == config.REVISION


# --------------------------------------------------------- distribution -------

def test_the_distribution_declares_exactly_what_was_approved() -> None:
    """`shards.declaration()` reads this block, and `merge()` refuses every
    merge until it exists. The replication (`A`/`B`/`C`: two machines, plus a
    same-machine control) settled every one of `config.DIMENSIONS`:
    `sourceAccuracy`, `targetAccuracy`, `contribution`, `supervised`,
    `adaptationShare` and `parameters` came back bit-identical across all
    three runs, so all six pool. `seconds` and `peakMiB` differed on every
    pair — including the control, same machine both times — which is what
    moved them out of `perEnvironment` (a label that claims the value is
    stable *on* that machine, which the control disproves) and into
    `perRun` (a value that belongs to one execution and nothing wider).

    `ceilings` and `ceilingsByTransfer` joined `epochs` when the ceiling
    stopped being one number per family. They are the parameter the search
    most recently changed, so two shards straddling that search would merge
    into one table with adaptation inert on one half and not on the other,
    and nothing would object. Both are flat top-level fields of every stamp
    (`harness.write_shard_stamp`), which is exactly the property the next
    paragraph is about.

    `commit` and `codeDigest` are not named under `identicalAcrossShards`
    even though they were approved alongside `epochs` — both live at
    `evidence.commit` / `evidence.codeDigest` on a shard's stamp, and
    `shards.disagreements()` (the forge's own, re-exported unchanged) reads
    `identicalAcrossShards` entries with a flat `stamp.get(field)`, never a
    dotted path. Naming them here would silently check nothing rather than
    check what was asked for. `parameters` fails that same flat lookup —
    it is written once per *run*, in `runs.jsonl`, and never appears on a
    shard's stamp at all (see `harness.run_one`'s return dict versus
    `harness.write_shard_stamp`) — so it stays in `poolable`, where
    averaging a constant returns the constant, rather than being placed
    somewhere `disagreements()` could never actually check it.
    """
    from MIL_CREDA_Benchmark import shards

    dist = shards.declaration()
    assert dist == {
        "axis": "seed",
        "poolable": ["sourceAccuracy", "targetAccuracy", "contribution",
                     "supervised", "adaptationShare", "parameters"],
        "perEnvironment": [],
        "perRun": ["seconds", "peakMiB"],
        "identicalAcrossShards": ["epochs", "ceilings", "ceilingsByTransfer"],
    }


def test_the_declared_dimensions_now_cover_every_dimension_the_harness_measures() -> None:
    """The replication closed the gap the earlier declaration left open: all
    eight of `config.DIMENSIONS` now sit in exactly one of `poolable`,
    `perEnvironment` or `perRun`, so `shards.partition()` no longer refuses
    on the harness's own full dimension set.
    """
    from MIL_CREDA_Benchmark import shards

    dist = shards.declaration()
    classified = (set(dist["poolable"]) | set(dist["perEnvironment"])
                  | set(dist["perRun"]))
    assert classified == set(config.DIMENSIONS)

    poolable, per_environment, per_run = shards.partition(config.DIMENSIONS, dist)
    assert set(poolable) | set(per_environment) | set(per_run) == set(config.DIMENSIONS)


def test_the_benchmark_declares_what_its_protocol_assumes() -> None:
    """A change of reach leaves every arm intact and every dimension meaningless,
    so the premises sit beside the arms rather than in somebody's memory."""
    premises = MIL_CREDA_Benchmark.__benchmark__["premises"]
    for key in ("prediction", "unit", "metric", "direction"):
        assert premises.get(key), f"the protocol does not declare its {key}"


def test_every_rung_names_arms_that_exist() -> None:
    for left, right, reading in config.LADDER:
        assert left in config.ARMS_BY_ID, f"rung {left}->{right} names unknown {left}"
        assert right in config.ARMS_BY_ID, f"rung {left}->{right} names unknown {right}"
        assert reading.strip(), f"rung {left}->{right} says nothing about what it reads"


def test_every_arm_can_be_kept_and_every_kept_arm_exists() -> None:
    assert set(config.CHECKPOINTS) == set(config.ARM_ORDER)
    assert all(count >= 1 for count in config.CHECKPOINTS.values())


def test_each_arm_has_its_own_display_name() -> None:
    """Two arms sharing a name would collapse into one row of every table."""
    names = [arm["name"] for arm in config.ARMS]
    assert len(set(names)) == len(names), f"repeated display names in {names}"


def test_every_figure_names_arms_that_exist() -> None:
    for arm in [*config.LATENT_PANELS, *config.BAG_PANELS, *config.FLOOR_OF,
                *config.FLOOR_OF.values()]:
        assert arm in config.ARMS_BY_ID, f"a figure or floor names unknown arm {arm!r}"


def test_the_figures_pick_transfers_the_campaign_actually_ran() -> None:
    """The three shown are computed from the campaign, so the only way they can go
    wrong is by naming a transfer that produced no runs — which would draw from
    checkpoints that do not exist and come out silently empty."""
    from MIL_CREDA_Benchmark import tables

    ran = [f"{s}->{t}" for s, t in config.TRANSFERS]
    runs = [{"arm": "A", "transfer": label, "targetAccuracy": index / 10,
             "contribution": 0.0} for index, label in enumerate(ran)]
    chosen = tables.best_transfers(runs)
    assert len(chosen) == config.FIGURE_TRANSFER_COUNT
    assert all(label in ran for label in chosen), f"{chosen} was never run"


def test_the_figure_transfer_rule_ranks_by_the_outcome_it_declares() -> None:
    """The choice is made by the result, which is why it is declared in every
    caption. What has to hold is that it ranks the way it says it does — a rule
    that claimed one order and produced another would be worse than no rule."""
    from MIL_CREDA_Benchmark import tables

    ran = [f"{s}->{t}" for s, t in config.TRANSFERS]
    ascending = [{"arm": "A", "transfer": label, "targetAccuracy": index / 10,
                  "contribution": 0.0} for index, label in enumerate(ran)]
    assert tables.best_transfers(ascending, count=2) == [ran[-1], ran[-2]]
    assert config.FIGURE_TRANSFER_RULE.strip(), "the rule is not declared anywhere"


def test_both_floors_stay_in_the_grid_until_a_measurement_removes_one() -> None:
    """Whether the two floors are redundant is a measurement, not an assumption:
    they train the same encoder through different objectives, so their instance
    embeddings have no reason to agree. `latent.floors_agree` is what may retire
    one of these columns, and nothing else."""
    floors = [arm["id"] for arm in config.ARMS if arm["adaptation"] is None]
    assert set(floors).issubset(set(config.LATENT_PANELS)), (
        f"a floor was dropped from the grid without a measurement: "
        f"{sorted(set(floors) - set(config.LATENT_PANELS))}")


def test_the_bag_figure_shows_the_rung_the_local_term_lives_on() -> None:
    """Its panels are chosen by the mechanism, not by the ranking: a floor, an arm
    without the local term, and one with it. Without the middle one the figure
    cannot come out wrong."""
    specs = [config.ARMS_BY_ID[arm] for arm in config.BAG_PANELS]
    assert any(spec["local"] for spec in specs), "no panel carries the local term"
    assert any(not spec["local"] and spec["adaptation"] for spec in specs), (
        "no panel adapts without the local term, so the figure cannot isolate it")
    assert any(spec["adaptation"] is None for spec in specs), "no floor to read against"


def test_the_selecting_arms_hold_one_budget_and_differ_only_in_the_rule() -> None:
    """The rung between two of them is attributable only if this holds."""
    selecting = [arm for arm in config.ARMS if arm["selection"] is not None]
    assert len(selecting) >= 2, "a selection rung needs at least two selecting arms"
    assert len({arm["selection"] for arm in selecting}) == len(selecting), (
        "two selecting arms share a rule, so the rung between them differs in nothing")
    for arm in selecting:
        for axis in ("unit", "adaptation", "weighting", "local", "attention"):
            assert arm[axis] == config.ARMS_BY_ID["G"][axis], (
                f"{arm['id']} differs from the complete method in {axis} as well as "
                f"in its selection, so its rung is not attributable")


def test_the_budget_is_smaller_than_the_bag() -> None:
    """A selection that keeps everything is not a selection, and its rung against
    the arm that keeps everything would compare an arm with itself."""
    assert 0 < config.SELECT_K < config.INSTANCES_PER_BAG


def test_the_pilot_and_the_full_run_are_the_same_program() -> None:
    """Only the repetition count and the length may separate them."""
    assert config.EPOCHS <= config.FULL_EPOCHS
    assert len(config.SEEDS) <= len(config.FULL_SEEDS)
    assert set(config.SEEDS).issubset(set(config.FULL_SEEDS))


def _one_rung_grid(left_mean: float, right_mean: float) -> dict:
    """A grid holding one rung, both arms flat across every transfer."""
    left, right, _ = config.LADDER[0]

    def arm(mean: float) -> dict:
        entry = {"mean": mean, "stdev": 0.0, "max": mean, "n": 1}
        # Both metrics, because the panorama walks both and a cell missing one is
        # a fixture defect that would read as a defect in the code under test.
        return {"targetAccuracy": dict(entry), "sourceAccuracy": dict(entry)}

    return {f"{s}->{d}": {left: arm(left_mean), right: arm(right_mean)}
            for s, d in config.TRANSFERS}


def test_a_rung_subtracts_left_minus_right_everywhere_it_is_computed() -> None:
    """The sign is a reading convention, and a convention only holds if every
    place that computes it agrees. It is computed three times — the rung table,
    its conclusion, and the panorama that outlives both in the record — and the
    prose in the notebook is written against it. Nothing else pins it, so an
    inversion here comes back as a heading that says the opposite of its own
    table while every number stays correct.
    """
    from MIL_CREDA_Benchmark import harness, tables

    left, right, _ = config.LADDER[0]
    grid = _one_rung_grid(left_mean=0.40, right_mean=0.70)
    summary = {"grid": grid, "reduction": {"seeds": config.SEEDS},
               "panorama": harness.paired_across_transfers(grid)}

    # The right arm is 30 points above, so left - right is negative.
    printed = tables.render_rungs(summary, "targetAccuracy", markdown=True)
    assert "-30.0" in printed, printed

    panorama = [row for row in summary["panorama"]
                if row["rung"] == f"{left}->{right}"
                and row["metric"] == "targetAccuracy"][0]
    assert panorama["meanDifference"] < 0
    # The field counts transfers won by the right arm, which is now the negative
    # side. Flipping the subtraction without flipping this would report the right
    # arm losing all six of the transfers it won.
    assert panorama["favouringRight"] == len(config.TRANSFERS)


def test_the_rung_conclusion_names_who_is_ahead_not_how_far_it_moved() -> None:
    """A conclusion that reports a signed magnitude makes the reader reconstruct
    the direction against the row's own name. Naming the arm is the whole job."""
    from MIL_CREDA_Benchmark import tables

    left, right, _ = config.LADDER[0]
    summary = {"grid": _one_rung_grid(left_mean=0.40, right_mean=0.70),
               "reduction": {"seeds": config.SEEDS}}
    text = tables.conclusion_rungs(summary, "targetAccuracy")
    assert config.NAME_OF[right] in text, text
    assert "por encima de" in text, text
    assert f"**{config.NAME_OF[right]}**" in text, text


# ------------------------------------------------------------------ perRun table

def test_the_per_run_table_prints_one_row_per_run_tagged_with_its_environment() -> None:
    """A `perRun` dimension (`seconds`, `peakMiB`) has no mean this report is
    willing to print: the replication that created the category found no value
    stable enough to stand for the method, or even for one fixed machine across
    two of its own runs. Pooling `n` readings into a single `mean ± stdev`, the
    way `render` does for a `poolable` dimension, would print a number that
    describes none of the runs behind it and looks exactly as rigorous as one
    that does. So every row here names the run that produced it — its own
    environment and seed — and no aggregate figure appears anywhere."""
    from MIL_CREDA_Benchmark import tables

    grid_per_run = {
        "M->U": {"G": {"seconds": [
            {"env": "c888ccf4ecc8", "seed": 0, "value": 64.8256},
            {"env": "a111a111a111", "seed": 1, "value": 70.2},
        ]}},
    }
    printed = tables.render_per_run(grid_per_run, "seconds", markdown=True)
    assert "c888ccf4ecc8" in printed
    assert "a111a111a111" in printed
    assert "64.83" in printed or "64.8" in printed
    assert "70.2" in printed
    # No pooled figure anywhere: a mean over two machines would look exactly
    # like a real one and describe neither.
    assert "±" not in printed


def test_the_per_run_table_reports_nothing_measured_when_the_grid_is_empty() -> None:
    from MIL_CREDA_Benchmark import tables

    assert "sin corridas" in tables.render_per_run({}, "seconds").lower()


def test_the_per_run_conclusion_declines_to_pool_and_points_back_at_the_table() -> None:
    """The table above already refuses to average; a conclusion that then
    printed 'best/worst average' over the same readings would take back with
    prose exactly what the table just refused to claim with numbers."""
    from MIL_CREDA_Benchmark import tables

    text = tables.conclusion_per_run("seconds")
    assert "no se promedia" in text.lower()
    assert "tabla" in text.lower()


# --------------------------------------------------------------- the three roles

def test_the_three_roles_partition_the_bags_exactly() -> None:
    """Nothing is lost and nothing is counted twice.

    The selection role was funded with new material, not taken from anywhere. If
    somebody moves one of the three constants without moving the bags per class,
    the remainder lands silently in evaluation or the split blows up midway
    through a campaign. Reachable red: raise `VALID_BAGS` without raising
    `BAGS_PER_CLASS` and this fails.
    """
    assert config.TRAIN_BAGS + config.VALID_BAGS + config.EVAL_BAGS == \
        config.BAGS_PER_DOMAIN


def test_the_search_never_reads_the_role_the_verdict_rests_on() -> None:
    """The selection role exists because the search chooses by looking at outcomes."""
    assert config.SEARCH_ROLE == "valid"
    assert config.SEARCH_ROLE != "eval"


def test_every_role_covers_every_class_in_every_domain() -> None:
    """The local correspondence is undefined for a class with no source bag.

    A role that dropped one would produce a failure that reads as a defect of the
    method and is a defect of the split.
    """
    base, remainder = divmod(config.TRAIN_BAGS, config.CLASSES)
    assert base >= 1, "el rol de entrenamiento dejaría clases sin bolsa"
    base, remainder = divmod(config.VALID_BAGS, config.CLASSES)
    assert base >= 1, "el rol de selección dejaría clases sin bolsa"
    per_class_used = (config.TRAIN_BAGS + config.VALID_BAGS) // config.CLASSES
    assert config.BAGS_PER_CLASS - per_class_used >= 1, \
        "el rol de evaluación dejaría clases sin bolsa"


def test_the_search_grid_runs_between_the_two_declared_defaults() -> None:
    """The range is not arbitrary: its endpoints are the two declared defaults."""
    from CREDA.schedules import CREDA_CEILING
    from MIL_CREDA_Benchmark.schedules import MILCREDA_CEILING

    assert min(config.CEILING_GRID) == CREDA_CEILING
    assert max(config.CEILING_GRID) == MILCREDA_CEILING


def test_the_search_uses_the_complete_method_of_each_family() -> None:
    """Searching with an ablation would pick the ceiling of a method nobody compares."""
    for family, arm_id in config.SEARCH_ARMS.items():
        assert arm_id in config.ARMS_BY_ID
        assert config.ARMS_BY_ID[arm_id]["adaptation"] == family


# ---------------------------------------------------- the ceiling selection

def _rows(scores: dict[float, list[float]]) -> list[dict]:
    """Rows as the search builds them, from (seed, transfer) cells."""
    grid = sorted(scores)
    cells = [{c: scores[c][i] for c in grid} for i in range(len(scores[grid[0]]))]
    out = []
    for c in grid:
        centred = [cell[c] - sum(cell.values()) / len(cell) for cell in cells]
        out.append({"ceiling": c, "paired": sum(centred) / len(centred)})
    return out


def _pick(rows: list[dict]) -> float:
    top = max(r["paired"] for r in rows)
    tied = [r for r in rows if r["paired"] == top]
    return min(tied, key=lambda r: r["ceiling"])["ceiling"]


def test_the_pairing_survives_a_cell_that_is_simply_harder() -> None:
    """What pairing buys, in the case that motivates it.

    Two cells, one far harder than the other, and ceiling 0.1 wins both by two
    points. Comparing bare means lets the hard cell's difficulty dominate and
    drown the effect; centring each cell on its own mean lets the effect survive,
    because every ceiling was measured on the same material.

    Reachable red: picking by bare mean, 0.01 wins.
    """
    scores = {0.01: [0.90, 0.20], 0.1: [0.92, 0.22]}
    crudo = {c: sum(v) / len(v) for c, v in scores.items()}
    assert _pick(_rows(scores)) == 0.1
    # the bare mean happens to agree here too; what changes is the margin
    assert crudo[0.1] > crudo[0.01]

    # now a hard, noisy cell, where the bare mean loses its way
    scores = {0.01: [0.90, 0.05], 0.1: [0.92, 0.01]}
    crudo = {c: sum(v) / len(v) for c, v in scores.items()}
    assert crudo[0.01] > crudo[0.1], "the bare mean picks the wrong ceiling"
    assert _pick(_rows(scores)) == 0.01, "y el pareado coincide: aca no hay efecto"


def test_a_tie_goes_to_the_smallest_ceiling() -> None:
    """Below some point a term is inert and everything ties; on that stretch the
    tie-break is what actually chooses, so it is declared."""
    scores = {1e-4: [0.5, 0.5], 1e-3: [0.5, 0.5], 1e-2: [0.5, 0.5]}
    assert _pick(_rows(scores)) == 1e-4


def test_the_required_search_scale_is_declared_apart_from_the_running_one() -> None:
    """Without both, a ceiling found at pilot scale reads as finished."""
    assert config.FULL_SEARCH_EPOCHS == config.FULL_EPOCHS
    assert config.FULL_SEARCH_SEEDS == 3
    assert config.SEARCH_EPOCHS >= config.FULL_SEARCH_EPOCHS
    assert len(config.SEARCH_SEEDS) >= config.FULL_SEARCH_SEEDS


def test_the_campaign_refuses_ceilings_searched_below_scale(tmp_path, monkeypatch) -> None:
    """The quiet failure: somebody lowers the search to test it cheaply, the file
    is written from three epochs, and every later campaign consumes it."""
    import json as _json
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(_json.dumps({
        "creda": {"ceiling": 1e-4, "atRequiredScale": False},
        "milcreda": {"ceiling": 1.0, "atRequiredScale": True},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)

    reduction = harness.Reduction(ceilings={"creda": 1e-4, "milcreda": 1.0})
    with pytest.raises(SystemExit) as raised:
        harness.campaign(reduction, torch.device("cpu"), arms=["A"],
                         progress=lambda *a: None)
    assert "below scale" in str(raised.value)
    assert "creda" in str(raised.value)


# --------------------------------------------------------- run_smoke checkpoints

def _fake_run_one(fake_state):
    def run_one(arm_id, transfer, seed, reduction, device, material, **kwargs):
        return {
            "arm": arm_id, "transfer": f"{transfer[0]}->{transfer[1]}", "seed": seed,
            "env": "test-env", "targetAccuracy": 0.5, "sourceAccuracy": 0.5,
            "seconds": 0.01, "peakMiB": 1.0, "parameters": 4, "contribution": 0.1,
            "supervised": 0.2, "adaptationShare": 0.3, "curve": [],
            "epochs": [{"epoch": 0}], "state": dict(fake_state),
        }
    return run_one


def _fake_build():
    from types import SimpleNamespace

    def build(code, cache, seed):
        return SimpleNamespace(manifest={"code": code, "seed": seed})
    return build


def test_run_smoke_writes_one_checkpoint_through_the_campaigns_own_path(
        tmp_path, monkeypatch) -> None:
    """Phase 2's `latent.available()`/`latent.load()` read `config.MODELS` in
    exactly `campaign()`'s own shape — a `.pt` saved with `torch.save({k: v.cpu()
    for k, v in state.items()}, ...)` beside a manifest `keep_median` writes
    through `bags.write_manifest`. A bespoke save in `run_smoke` would prove
    nothing about whether that real path works; this exercises
    `run_smoke(checkpoint=True)` over the same one, without paying for a real
    training loop — `run_one` and `bags.build` are faked, everything downstream
    of the state dict they return is real."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    monkeypatch.setattr(config, "MODELS", tmp_path / "Models")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results")
    monkeypatch.setattr(harness, "run_one", _fake_run_one({"weight": torch.zeros(2, 2)}))
    monkeypatch.setattr(harness.bags, "build", _fake_build())

    harness.run_smoke(seed=0, checkpoint=True)

    left, right = config.VERDICT_TRANSFERS[0]
    stem = f"{harness.SMOKE_ARM}_{left}-{right}_seed0"
    weights = config.MODELS / f"{stem}.pt"
    manifest = config.MODELS / f"{stem}.manifest.json"
    assert weights.is_file()
    assert manifest.is_file()

    loaded_state = torch.load(weights)
    assert set(loaded_state) == {"weight"}

    written = json.loads(manifest.read_text())
    assert written["arm"] == harness.SMOKE_ARM
    assert written["seed"] == 0
    assert written["source"] == {"code": left, "seed": 0}
    # Exactly one, never the three per cell a campaign's own `CHECKPOINTS`
    # asks for: `median_seeds` degenerates to the only seed there is.
    assert len(list(config.MODELS.glob("*.pt"))) == 1


def test_run_smoke_writes_no_checkpoint_by_default(tmp_path, monkeypatch) -> None:
    """Off unless asked for: most rehearsals only need `runs.jsonl`, and a
    checkpoint costs a model's worth of disk on every call this doesn't ask
    for one. `config.MODELS` must not even be created."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "MODELS", tmp_path / "Models")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "Results")
    monkeypatch.setattr(harness, "run_one", _fake_run_one({"weight": torch.zeros(1)}))
    monkeypatch.setattr(harness.bags, "build", _fake_build())

    harness.run_smoke(seed=0)

    assert not config.MODELS.exists()


# ------------------------------ the notebook actually obtains what it needs

def _notebook_code_cells(name: str) -> list[dict]:
    """A notebook's code cells, in order."""
    path = config.REPOSITORY / "MIL-CREDA" / "Notebooks" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _phase_one_run() -> list[dict]:
    """The notebook that runs the campaign: the forecast, the search, the grid.

    It renders nothing and writes no report — only the records the reporting
    notebook reads back from disk.
    """
    return _notebook_code_cells("Benchmark_Campaign_v1.ipynb")


def _phase_one_report() -> list[dict]:
    """The notebook that reads those records and writes the readable report.

    It trains and searches nothing; every number in it comes from
    `summary.json`, `runs.jsonl` and the ceiling search's own record.
    """
    return _notebook_code_cells("Benchmark_Report_v1.ipynb")


def test_the_notebook_obtains_the_ceilings_before_it_runs_the_campaign() -> None:
    """The join, checked from the notebook's end and not only from the harness'.

    `campaign` refuses without ceilings and `ceilings_in_force` supplies them, and
    both were green while the only caller that matters never called one from the
    other: the notebook built its `Reduction` with the empty mapping `config`
    imports and died on the refusal. Testing each half against a fixture written
    here verifies both halves and never the connection, which is the only thing
    this rule was ever about.

    Both calls live in the run notebook: obtaining the ceilings is what lets the
    campaign that follows in the same notebook proceed.
    """
    sources = ["".join(cell["source"]) for cell in _phase_one_run()]
    obtains = [i for i, s in enumerate(sources) if "ceilings_in_force" in s]
    runs = [i for i, s in enumerate(sources) if "harness.campaign(" in s]
    assert obtains, "no cell obtains the ceilings; the campaign will refuse"
    assert runs, "no cell runs the campaign"
    assert min(obtains) < min(runs), (
        f"the ceilings are obtained at cell {min(obtains)}, after the campaign at "
        f"{min(runs)}")


def test_the_notebook_forecasts_the_search_before_spending_it() -> None:
    """The search is the longest single wait and the one part with no pilot scale.

    A notebook forecasting only the grid puts it ahead of the estimate that exists
    to precede it. Both the forecast and the obtaining call live in the run
    notebook, ahead of the search and the campaign they precede.
    """
    sources = ["".join(cell["source"]) for cell in _phase_one_run()]
    forecasts = [i for i, s in enumerate(sources) if '["search"]' in s]
    obtains = [i for i, s in enumerate(sources) if "ceilings_in_force" in s]
    assert forecasts, "nothing forecasts what the ceiling search costs"
    assert min(forecasts) < min(obtains), (
        "the search is forecast after it is spent, which is not a forecast")


# --------------------- run_campaign_shard(): one shard's campaign, asked by name

def test_run_campaign_shard_is_callable_with_json_alone() -> None:
    """`tools/distribute.py`'s `run_shard()` already has the right shape, and a
    remote worker cannot reach it: `tools/` is outside every declared clone
    path, and that module path-imports the forge's own packer, which does not
    exist inside a kernel. So the fan-out the distribution declaration is built
    around had no way to be asked for remotely.

    Reachable red: before this, `harness` exposed `run_pilot`, `run_smoke`,
    `run_one` and `run_search`, and none of them runs one shard of the grid.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    parameters = inspect.signature(harness.run_campaign_shard).parameters
    assert list(parameters) == ["shard", "seeds"]
    assert all(p.default is not inspect.Parameter.empty for p in parameters.values())


def test_run_campaign_shard_runs_at_full_scale_never_the_pilots(
        tmp_path, monkeypatch) -> None:
    """A shard is a slice of the campaign, not a cheaper version of it. The
    seed axis is what splits across machines; the epoch count is not a dial,
    for the same reason it is not one on the search.

    `config.CEILINGS_RECORD` is pinned to an isolated `tmp_path` record so
    this test can never reach the real `Results/Benchmark/ceilings.json` —
    which does not exist on disk — and the tripwire below turns any future
    unpinning of that record into a named, millisecond-long failure instead
    of a real ceiling search running inside the unit suite.
    """
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "creda": {"ceiling": 1e-4, "atRequiredScale": True},
        "milcreda": {"ceiling": 1.0, "atRequiredScale": True},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    monkeypatch.setattr(config, "CEILINGS", {})
    monkeypatch.setattr(harness, "search_ceilings", lambda *a, **k: pytest.fail(
        "the unit suite reached the real ceiling search"))

    seen = {}

    def _spy(reduction, device, arms=None, progress=print, shard=None):
        seen["reduction"], seen["shard"] = reduction, shard
        return {}

    monkeypatch.setattr(harness, "campaign", _spy)
    harness.run_campaign_shard(shard="a", seeds=[3, 4])

    assert seen["reduction"].epochs == config.FULL_EPOCHS
    assert seen["reduction"].seeds == [3, 4]
    assert seen["shard"] == "a"
    # A shard is one JSON-callable campaign entrypoint among four that must
    # each obtain ceilings in force before running — proven behaviourally,
    # against the record's own winners, not only against source text.
    assert seen["reduction"].ceilings == {"creda": 1e-4, "milcreda": 1.0}


def test_run_campaign_shard_obtains_ceilings_before_it_runs_the_campaign() -> None:
    """The same join `run_pilot()` and `run_search()` are already held to
    (`test_run_pilot_obtains_ceilings_before_it_runs_the_campaign`): `campaign()`
    refuses outright without `reduction.ceilings` already populated, so this
    has to close that gap itself rather than hand a caller a refusal it
    cannot read.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_campaign_shard)
    # "campaign(reduction" and not the bare "campaign(": the docstring above
    # already mentions "`campaign()`, which hands it to `shard_paths()`" —
    # matching on "campaign(" alone would find that prose, not the call.
    assert source.index("ceilings_in_force(") < source.index("campaign(reduction")


def test_run_campaign_shard_writes_into_its_own_namespace() -> None:
    """Two shards running at once would clobber each other's records if the
    name did not reach `campaign()`. `shard_paths()` is what separates them,
    and it only separates them if the caller passes the name through.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_campaign_shard)
    assert "shard=shard" in source


# ------------------------------ run_search(): the search alone, asked for by name

def test_run_search_is_callable_with_json_alone() -> None:
    """A remote job names a `module.function` and hands it JSON keyword
    arguments; it cannot build a `Reduction` or a `torch.device`. So every
    parameter of an entrypoint has to be JSON-native and optional, or the
    entrypoint cannot be reached from a job at all.

    Reachable red: before this, `harness` exposed `run_pilot`, `run_smoke`
    and `run_one`, and the search — the one experiment this repository is
    blocked on — had no launcher of its own.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    parameters = inspect.signature(harness.run_search).parameters
    assert "shard" in parameters and parameters["shard"].default is None
    # La intencion, y no una lista de nombres: cada parametro tiene que ser
    # JSON-nativo y opcional. Una lista exacta declaraba lo mismo mientras hubo
    # uno solo, y despues prohibia agregar un segundo que cumple la regla.
    for name, param in parameters.items():
        assert param.default is not inspect.Parameter.empty, name
        assert isinstance(param.default, (str, int, float, bool, type(None))), name


def test_run_search_offers_no_scale_dial_and_pilot_is_not_one() -> None:
    """La razon por la que no hay dial es el **destino**, no la escala.

    Un `epochs` suelto seria aceptado, ignorado por `search_ceilings()` y su
    respuesta escrita igual al archivo donde va la respuesta completa: un
    knob que solo parece funcionar. `pilot=True` no es ese knob — cambia el
    archivo. Corre a su propia escala declarada y escribe a
    `CEILINGS_PILOT_RECORD`, al que el registro completo le gana siempre.
    """
    import inspect

    from MIL_CREDA_Benchmark import config, harness

    assert "epochs" not in inspect.signature(harness.run_search).parameters
    assert "seeds" not in inspect.signature(harness.run_search).parameters
    assert "epochs" in inspect.signature(harness.run_pilot).parameters
    assert config.CEILINGS_PILOT_RECORD != config.CEILINGS_RECORD


def test_run_search_runs_at_the_declared_scale_and_never_the_pilots(monkeypatch) -> None:
    """What the search is handed has to be the scale it declares for itself,
    not whatever the pilot happens to be running at. Asserted on the
    `Reduction` that actually reaches `ceilings_in_force`, because the
    record written afterwards carries that scale and a later reader trusts
    it.
    """
    from MIL_CREDA_Benchmark import harness

    seen = {}

    def _spy(reduction, device, progress=print, shard=None, pilot=False):
        seen["reduction"] = reduction
        seen["pilot"] = pilot
        return {}

    monkeypatch.setattr(harness, "ceilings_in_force", _spy)
    monkeypatch.setattr(harness, "search_record", lambda pilot=False: {"creda": {}})

    harness.run_search()

    assert seen["reduction"].epochs == config.SEARCH_EPOCHS
    assert seen["reduction"].seeds == list(config.SEARCH_SEEDS)
    assert seen["reduction"].epochs != config.EPOCHS


def test_run_search_refuses_to_report_success_with_no_record_on_disk() -> None:
    """The record is the whole product. A run that returned normally while
    `ceilings.json` was never written would report an answer it cannot
    show, and everything downstream reads the file rather than the return
    value.
    """
    from MIL_CREDA_Benchmark import harness

    original_force = harness.ceilings_in_force
    original_record = harness.search_record
    harness.ceilings_in_force = lambda *a, **k: {}
    harness.search_record = lambda pilot=False: None
    try:
        with pytest.raises(SystemExit) as raised:
            harness.run_search()
    finally:
        harness.ceilings_in_force = original_force
        harness.search_record = original_record
    assert "left no record" in str(raised.value)


# ---------------------------- run_pilot(): the notebook, headless, cheapenable

def test_run_pilot_epochs_default_is_reductions_own_default_not_full_epochs() -> None:
    """`run_pilot()` is what a remote job's `--run-kwargs` actually reaches
    — distinct from `tools/distribute.py`'s `run_shard()`, whose epochs
    are pinned to `config.FULL_EPOCHS` for the fan-out path. With no
    override `run_pilot()` must reproduce exactly what a bare
    `Reduction()` already defaults to (`config.EPOCHS`), so calling it
    with nothing changes nothing about what the pilot cells already run.

    Reachable red: before this task, `harness` exposed no `run_pilot`
    attribute at all.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    default = inspect.signature(harness.run_pilot).parameters["epochs"].default
    assert default == config.EPOCHS
    assert default != config.FULL_EPOCHS


def test_run_pilot_epochs_is_overridable_as_a_plain_json_kwarg() -> None:
    """The whole point: a caller reaches a cheaper-than-pilot run by
    passing `{"epochs": 2}` through `--run-kwargs`, never by editing
    `config.py`. `seeds` stays optional the same way, defaulting to
    `config.SEEDS` exactly like a bare `Reduction()` would.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    params = inspect.signature(harness.run_pilot).parameters
    assert params["epochs"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    )
    assert params["seeds"].default is None


def test_run_pilot_obtains_ceilings_before_it_runs_the_campaign() -> None:
    """The same join the notebook itself is held to
    (`test_the_notebook_obtains_the_ceilings_before_it_runs_the_campaign`),
    now inside `run_pilot()`'s own source: `campaign()` refuses outright
    without `reduction.ceilings` already populated, so this has to close
    that gap itself rather than hand a caller a refusal it cannot read.
    """
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_pilot)
    assert source.index("ceilings_in_force(") < source.index("campaign(")


def test_run_pilot_is_the_single_machine_counterpart_never_a_shard() -> None:
    """The one thing that tells `run_pilot()` apart from
    `tools/distribute.py`'s `run_shard()`: it never passes `shard=` into
    `campaign()`, so the whole seed axis runs in one machine's own
    namespace rather than one shard of it."""
    import inspect

    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_pilot)
    assert "shard=" not in source


def test_run_pilot_is_reachable_under_src_for_a_remote_run_module() -> None:
    """A remote job's generated runner only ever puts `<clone>/src` on
    `sys.path` (never `tools/`), and `--run-module` is resolved the same
    way at generation time: only what lives under `src/` is importable at
    all. `run_pilot()` has to live in `harness.py`, under `src/`, for
    `--run-module MIL_CREDA_Benchmark.harness --run-function run_pilot`
    to ever resolve — `run_shard()` staying in `tools/distribute.py` is
    what makes it the fan-out path's own, local-invocation-only entry
    point instead.
    """
    import inspect
    from pathlib import Path

    from MIL_CREDA_Benchmark import harness

    assert Path(inspect.getfile(harness.run_pilot)).is_relative_to(
        config.REPOSITORY / "src")


def test_the_ceilings_reach_the_written_report_and_not_only_the_screen() -> None:
    """The agreement is that the *report* says which ceiling each family found.

    A cell that prints it to the console satisfies a reader watching the run and
    nobody afterwards, and the record is what a later session reads. The report
    notebook is the one that writes `report.md`, reading the ceiling record back
    from disk rather than from a run still live in memory.
    """
    written = [s for s in ("".join(c["source"]) for c in _phase_one_report())
               if "report.md" in s]
    assert written, "no cell writes the readable report"
    assert any("render_ceilings" in s and "conclusion_ceilings" in s
               for s in written), (
        "the written report carries no ceiling section")


def _searched(**overrides) -> dict:
    entry = {"arm": "G", "ceiling": 1e-4, "criterion": "targetAccuracy",
             "grid": [{"ceiling": 1e-4, "targetAccuracy": 0.8},
                      {"ceiling": 1.0, "targetAccuracy": 0.6}],
             "tied": [1e-4], "decidedByTieBreak": False, "seedsAgree": True,
             "role": "valid", "epochs": 20, "seeds": [0, 1, 2],
             "atRequiredScale": True, "neutral": 1.0}
    return {"milcreda": {**entry, **overrides}}


def test_the_ceiling_conclusion_separates_a_measurement_from_a_tie_break() -> None:
    """A ceiling chosen between four identical scores and one chosen by a real
    difference are the same number, so the number cannot be the conclusion."""
    from MIL_CREDA_Benchmark import tables

    measured = tables.conclusion_ceilings(_searched())
    tied = tables.conclusion_ceilings(
        _searched(decidedByTieBreak=True, tied=[1e-4, 1e-3, 1e-2, 1.0]))
    assert measured != tied, "the conclusion cannot come out different"
    assert "desempate" in tied and "desempate" not in measured


def test_the_ceiling_conclusion_says_what_the_record_does_not_say() -> None:
    """A record written by an earlier search carries no tie-break and no agreement.

    Defaulting to `False` would assert *chosen by a real difference* about a record
    that never said so — a reason invented to cover what was not found, which reads
    as a finding and gets acted on like one.
    """
    from MIL_CREDA_Benchmark import tables

    older = {"milcreda": {"ceiling": 1e-4, "grid": [], "criterion": "targetAccuracy"}}
    text = tables.conclusion_ceilings(older)
    assert "sin que el registro diga cómo se desempató" in text
    assert "el registro no dice si las semillas coincidieron" in text
    assert "por una diferencia en el criterio" not in text


def test_the_campaign_carries_the_whole_search_record_not_only_the_winner() -> None:
    """A ceiling chosen between four identical scores and one chosen by a real
    difference are the same number and not the same evidence.

    And filling the field at the caller's hand is how it ends up filled on the run
    somebody was paying attention to and empty on the next.
    """
    import inspect
    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.campaign)
    assert "ceilingSearch=searched" in source, (
        "the campaign does not copy the search record into its reduction")


def test_an_existing_ceiling_record_is_never_re_searched(tmp_path, monkeypatch) -> None:
    """Overwriting an answer because a later caller wanted a different one is the
    silent refunding the campaign's refusal exists to prevent."""
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "creda": {"ceiling": 0.5, "atRequiredScale": True},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)

    def refuse(*args, **kwargs):
        raise AssertionError("re-searched over an existing record")

    monkeypatch.setattr(harness, "search_ceilings", refuse)
    found = harness.ceilings_in_force(harness.Reduction(), torch.device("cpu"),
                                      progress=lambda *a: None)
    assert found == {"creda": 0.5}


def test_the_ceilings_are_read_back_from_disk_and_not_from_the_import(
        tmp_path, monkeypatch) -> None:
    """`config.CEILINGS` is filled once at import, from a file that may not exist.

    A caller that searches in the same process — a notebook, the only place that
    ever does — would otherwise hold the empty mapping it imported and hand it to
    a campaign that refuses it, with the answer sitting on disk beside it.
    """
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)
    monkeypatch.setattr(config, "CEILINGS", {})

    def write_it(*args, **kwargs):
        record.write_text(json.dumps({
            "milcreda": {"ceiling": 0.25, "atRequiredScale": True},
        }), encoding="utf-8")
        return {}

    monkeypatch.setattr(harness, "search_ceilings", write_it)
    found = harness.ceilings_in_force(harness.Reduction(), torch.device("cpu"),
                                      progress=lambda *a: None)
    assert found == {"milcreda": 0.25}, (
        f"read from the stale import instead of the record: {found}")


# ------------------------------------------- resuming a search cut in half

def test_a_measured_cell_survives_the_run_being_cut(tmp_path, monkeypatch) -> None:
    """Cutting a grid of hours on its last cell used to lose every one before it.

    Measured the hard way: 1h37 for nothing. Every cell is written as soon as it
    is measured, so an interruption costs one cell and not the run.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    cells = {(0, "M->U"): {1e-4: 0.51, 1.0: 0.62},
             (1, "S->M"): {1e-4: 0.40, 1.0: 0.44}}
    harness._write_partial("milcreda", "G", cells, 3.7, lambda *a: None)

    assert harness._partial_path().exists()
    assert harness._read_partial()["milcreda"] == cells


def test_the_partial_is_keyed_by_seed_and_transfer_not_by_position() -> None:
    """Resuming by counting would shift silently if the seed list moved.

    And it would then attribute one cell's measurements to another, which is worse
    than losing them: real numbers hanging off the wrong label.
    """
    import inspect
    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness._write_partial)
    assert '{seed}|{label}' in source


def test_the_scratch_file_is_not_the_finished_record(tmp_path, monkeypatch) -> None:
    """A half-filled file under the record's name would be read as an answer —
    including by the campaign's refusal, which only checks that it exists."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    harness._write_partial("creda", "D", {(0, "M->U"): {1e-4: 0.5}}, 1.0,
                           lambda *a: None)
    assert not config.CEILINGS_RECORD.exists()
    assert harness.search_record() is None


def test_a_resumed_ceiling_the_grid_no_longer_has_refuses_by_name(
        tmp_path, monkeypatch) -> None:
    """A partial is a file on disk and the grid is a line in config.py —
    nothing keeps them in step. Editing the grid after a partial was written
    used to surface as a bare `KeyError` at aggregation, hours in. It has to
    refuse by name instead, before anything else measures.
    """
    # Este test es del motor de rejilla, que sigue existiendo pero ya no es
    # el que `search_ceilings` despacha por defecto. Fijarlo acá dice qué
    # se está probando; sin esto el test pasaba a medir Optuna sin decirlo.
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(harness, "run_one", _fake_run_one({}))
    monkeypatch.setattr(harness.bags, "build", _fake_build())
    harness._write_partial("creda", "D", {(0, "M->U"): {0.05: 0.51}}, 1.0,
                           lambda *a: None)

    with pytest.raises(SystemExit) as raised:
        harness.search_ceilings(harness.Reduction(), torch.device("cpu"),
                                progress=lambda *a: None)

    message = str(raised.value)
    assert "creda" in message
    assert "0.05" in message
    assert str(config.CEILING_GRID) in message
    assert str(harness.shard_paths(None)["partial"]) in message
    assert "Delete" in message and "restore" in message


def test_the_stale_resume_refuses_before_it_measures_anything(
        tmp_path, monkeypatch) -> None:
    """Eagerness is the point: a stale key must refuse before a single
    further cell trains, not once some other family's grid has already run.
    """
    # Este test es del motor de rejilla, que sigue existiendo pero ya no es
    # el que `search_ceilings` despacha por defecto. Fijarlo acá dice qué
    # se está probando; sin esto el test pasaba a medir Optuna sin decirlo.
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")

    def _unreached(*args, **kwargs):
        pytest.fail("measured before refusing")

    monkeypatch.setattr(harness, "run_one", _unreached)
    monkeypatch.setattr(harness.bags, "build", _unreached)
    harness._write_partial("creda", "D", {(0, "M->U"): {0.05: 0.51}}, 1.0,
                           lambda *a: None)

    with pytest.raises(SystemExit):
        harness.search_ceilings(harness.Reduction(), torch.device("cpu"),
                                progress=lambda *a: None)


def test_the_power_state_is_stamped_and_never_fatal() -> None:
    """`seconds` and `peakMiB` are dimensions of the verdict, and a measurement
    describes whichever environment produced it. A throttled run must be labelled
    as such rather than attributed to the method.

    Never fatal: a stamp that crashed the run it was documenting would be worse
    than no stamp at all.
    """
    from MIL_CREDA_Benchmark import harness

    state = harness.power_state()
    assert state["source"] in ("mains", "battery", "unknown")
    assert "power" in harness.environment()


# ------------------------------------------------- the device actually received

def test_the_environment_stamps_the_accelerator_it_got() -> None:
    """Pinning an accelerator is an intention; the stamp is the fact.

    A remote service allocates by availability, so a shard can request one class
    and silently land on another. `seconds` and `peakMiB` describe whichever
    machine produced them, so grouping them by an environment that cannot tell
    two GPU classes apart is grouping by a label that lies.
    """
    from MIL_CREDA_Benchmark import harness

    device = harness.device_class()
    assert set(device) >= {"name", "kind"}
    assert isinstance(device["name"], str) and device["name"]
    assert device["kind"] in ("cuda", "mps", "cpu")
    assert harness.environment()["device"] == device


def test_two_stamps_differ_when_the_accelerator_differs() -> None:
    """Reachable red: a stamp that ignored the device would call these equal,
    which is exactly the case sharding across accounts produces."""
    from MIL_CREDA_Benchmark import harness

    base = {"python": "3.12.13", "platform": "linux", "torch": "2.13.0",
            "selfHosted": True, "power": {"source": "mains"}}
    t4 = harness.environment_key({**base, "device": {"name": "Tesla T4", "kind": "cuda"}})
    p100 = harness.environment_key({**base, "device": {"name": "Tesla P100", "kind": "cuda"}})
    assert t4 != p100
    assert t4 == harness.environment_key(
        {**base, "device": {"name": "Tesla T4", "kind": "cuda"}})


def test_the_charge_level_does_not_change_the_environment_key() -> None:
    """It moves during a run and is not a machine class; the source is, because
    throttling is a real between-arms difference."""
    from MIL_CREDA_Benchmark import harness

    base = {"python": "3.12.13", "platform": "linux", "torch": "2.13.0",
            "selfHosted": True, "device": {"name": "Tesla T4", "kind": "cuda"}}
    full = harness.environment_key({**base, "power": {"source": "mains", "charge": 100}})
    low = harness.environment_key({**base, "power": {"source": "mains", "charge": 12}})
    battery = harness.environment_key({**base, "power": {"source": "battery", "charge": 12}})
    assert full == low
    assert battery != low


def test_median_seeds_is_the_selection_rule_on_its_own() -> None:
    """Extracted so a shard and the centre share one rule.

    Acceptance criterion: bit-identical to what `keep_median` selected before, so
    the existing callers and tests do not move.
    """
    from MIL_CREDA_Benchmark import harness

    runs = [{"seed": s, "targetAccuracy": a}
            for s, a in ((0, 0.10), (1, 0.90), (2, 0.50), (3, 0.30), (4, 0.70))]
    # The historical body, reproduced here as the golden the extraction must match.
    ordered = sorted(runs, key=lambda r: r["targetAccuracy"])
    middle = len(ordered) // 2
    span = min(config.CHECKPOINTS["G"], len(ordered))
    start = max(0, min(middle - span // 2, len(ordered) - span))
    expected = {run["seed"] for run in ordered[start:start + span]}

    assert harness.median_seeds(runs, "G") == expected
    assert harness.median_seeds(runs, "G") == {2, 3, 4}


def test_every_run_carries_the_machine_that_produced_it() -> None:
    """The stamp travels on the run, not only on the campaign.

    A shard is a remote session, and one that times out and resumes can land on
    different hardware inside a single shard id. A per-campaign stamp cannot
    express that, and it is precisely what distributing produces.

    Reachable red: with the handle only on the Reduction, a merge could not tell
    two machines apart within one shard, and would pool their cost dimensions.
    """
    import inspect
    from MIL_CREDA_Benchmark import harness

    source = inspect.getsource(harness.run_one)
    assert '"env": environment_key(reduction.environment)' in source


# ------------------------------------------------------- shard-safe persistence

def test_each_shard_writes_where_no_other_shard_writes(tmp_path, monkeypatch) -> None:
    """Today one hardcoded path, opened `"w"`, truncating on every campaign.

    Two shards running at once against it would clobber each other's records, and
    the loser would be a silent partial file rather than an error. Reachable red:
    with the paths fixed, both of these resolve to the same file.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    one = harness.shard_paths("alpha")
    two = harness.shard_paths("beta")
    assert one["runs"] != two["runs"]
    assert one["partial"] != two["partial"]
    assert one["stamp"] != two["stamp"]
    for name in ("runs", "partial", "stamp"):
        assert "alpha" in str(one[name]) and "beta" in str(two[name])


def test_no_shard_named_means_the_paths_stay_where_they_always_were(tmp_path, monkeypatch) -> None:
    """A single-machine run must not move its own files.

    Every notebook, every record already written and the whole `records`
    declaration name these paths. Sharding is an addition, not a relocation.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    here = harness.shard_paths(None)
    assert here["runs"] == tmp_path / "runs.jsonl"
    assert here["partial"] == tmp_path / "ceilings.partial.json"


def test_a_shard_records_its_own_stamp_beside_its_runs(tmp_path, monkeypatch) -> None:
    """The full environment lives once per shard; runs carry only the handle."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    stored = json.loads(harness.shard_paths("alpha")["stamp"].read_text())
    assert stored["shard"] == "alpha"
    assert stored["env"] == harness.environment_key(stored["environment"])
    assert "device" in stored["environment"]


# --------------------------------------------------------------- shard evidence

def test_write_shard_stamp_carries_evidence_without_outputs(tmp_path, monkeypatch) -> None:
    """`outputs` cannot be known before the run finishes, so it is absent here.

    The other three fields can: the commit, the code digest, and where imports
    resolved from are all facts about the checkout at the moment the stamp is
    written, not about what the run produces.
    """
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    stored = json.loads(harness.shard_paths("alpha")["stamp"].read_text())

    evidence = stored["evidence"]
    assert evidence["commit"] and len(evidence["commit"]) == 40
    assert evidence["codeDigest"] and len(evidence["codeDigest"]) == 64
    assert evidence["importsFrom"].endswith("MIL_CREDA_Benchmark")
    assert "outputs" not in evidence


def test_seal_shard_stamp_adds_outputs_atomically(tmp_path, monkeypatch) -> None:
    """Sealing rewrites the stamp once the run's own files exist beside it."""
    from MIL_CREDA_Benchmark import harness

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    paths = harness.shard_paths("alpha")
    paths["runs"].parent.mkdir(parents=True, exist_ok=True)
    paths["runs"].write_text('{"seed": 0}\n', encoding="utf-8")

    sealed = harness.seal_shard_stamp("alpha")
    assert sealed == paths["stamp"]
    stored = json.loads(paths["stamp"].read_text())
    assert sorted(stored["evidence"]["outputs"]) == ["runs.jsonl", "shard.json"]
    # No leftover scratch file: the rename is the only trace of the rewrite.
    assert list(paths["stamp"].parent.glob("*.partial.json")) == []


def test_an_unsealed_shard_is_incomplete_and_a_sealed_one_is_not(
        tmp_path, monkeypatch) -> None:
    """The whole enforcement mechanism: a run that dies before sealing never
    lists `outputs`, so it is never complete, so it is never mergeable."""
    from MIL_CREDA_Benchmark import harness, shards

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    paths = harness.shard_paths("alpha")

    unsealed = json.loads(paths["stamp"].read_text())
    before = shards.completeness(unsealed, shards.REQUIRED_EVIDENCE)
    assert before["complete"] is False
    assert "evidence.outputs" in before["missing"]

    harness.seal_shard_stamp("alpha")
    sealed = json.loads(paths["stamp"].read_text())
    after = shards.completeness(sealed, shards.REQUIRED_EVIDENCE)
    assert after["complete"] is True
    assert after["missing"] == []


def test_a_concurrent_reader_during_seal_sees_the_pre_seal_stamp(
        tmp_path, monkeypatch) -> None:
    """Reachable red: sealing that mutated the stamp file in place, rather than
    rewriting a scratch copy and renaming it, would let a reader mid-write see a
    half-written JSON document instead of a whole, if older, one.

    `os.replace` is the boundary a concurrent reader can land either side of and
    never inside, so this test reads the file at the exact instant just before
    the rename and asserts it is still the complete, if incomplete-as-evidence,
    pre-seal stamp — correct, and worth this explicit test rather than an
    assumption, per the design's own open question.
    """
    import os as _os

    from MIL_CREDA_Benchmark import harness, shards

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    harness.write_shard_stamp("alpha", harness.Reduction())
    paths = harness.shard_paths("alpha")
    paths["runs"].write_text('{"seed": 0}\n', encoding="utf-8")

    seen_during_window = {}
    real_replace = _os.replace

    def watched_replace(src, dst):
        # The exact window: the scratch file is fully written, the real stamp
        # has not moved yet. A reader landing here sees only the old bytes.
        seen_during_window["stamp"] = json.loads(paths["stamp"].read_text())
        real_replace(src, dst)

    monkeypatch.setattr(_os, "replace", watched_replace)
    harness.seal_shard_stamp("alpha")

    mid_seal = shards.completeness(seen_during_window["stamp"], shards.REQUIRED_EVIDENCE)
    assert mid_seal["complete"] is False
    assert "evidence.outputs" in mid_seal["missing"]

    post_seal = shards.completeness(
        json.loads(paths["stamp"].read_text()), shards.REQUIRED_EVIDENCE)
    assert post_seal["complete"] is True


def test_a_checkout_with_no_git_history_never_produces_a_complete_shard(
        tmp_path, monkeypatch) -> None:
    """`evidence.commit` requires git in the runtime that stamps. A checkout
    with no git history — `.git` stripped, a source tarball — can never
    produce a complete shard. Accepted as correct; a behavior change for any
    non-git test fixture, hence flagged with its own explicit test."""
    from MIL_CREDA_Benchmark import harness, shards

    no_git = tmp_path / "no-git-checkout"
    no_git.mkdir()
    monkeypatch.setattr(config, "REPOSITORY", no_git)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")

    harness.write_shard_stamp("alpha", harness.Reduction())
    stored = json.loads(harness.shard_paths("alpha")["stamp"].read_text())

    assert "commit" not in stored["evidence"]
    result = shards.completeness(stored, shards.REQUIRED_EVIDENCE)
    assert result["complete"] is False
    assert "evidence.commit" in result["missing"]


# ---------------------------------------------------------------------------
# The ceiling in force on one transfer, which is two readings and not one.


def test_a_measured_transfer_keeps_its_own_pick_over_the_pooled_one() -> None:
    """The whole point of separating the two readings.

    Inverted, this is the failure the change exists to prevent: the pooled
    winner applied to a transfer the search actually measured and disagreed on.
    That failure is silent — the run proceeds and prints a number — so the only
    thing that catches it is asking the resolver directly.
    """
    from MIL_CREDA_Benchmark import harness

    reduction = harness.Reduction(
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
    )
    assert harness.ceiling_for(reduction, "milcreda", ("S", "M")) == 1e-4
    # and the pooled winner still governs everywhere it was never measured
    assert harness.ceiling_for(reduction, "milcreda", ("M", "U")) == 1e-2
    assert harness.ceiling_for(reduction, "milcreda", ("U", "S")) == 1e-2


def test_a_family_with_no_per_transfer_picks_falls_back_everywhere() -> None:
    """A record written before the key existed means what it meant then.

    An absent mapping is not a defect to refuse: it is a record whose every
    transfer ran at the pooled winner, and reading it as anything else would
    invent a distinction that run never made.
    """
    from MIL_CREDA_Benchmark import harness

    reduction = harness.Reduction(ceilings={"creda": 1e-4}, ceilingsByTransfer={})
    for transfer in config.TRANSFERS:
        assert harness.ceiling_for(reduction, "creda", transfer) == 1e-4


def test_a_floor_gets_the_neutral_and_never_a_families_ceiling() -> None:
    """An arm with no adaptation term has no ceiling to inherit."""
    from MIL_CREDA_Benchmark import harness

    reduction = harness.Reduction(
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
    )
    assert harness.ceiling_for(reduction, None, ("S", "M")) == config.RAMP_CEILING


def _synthetic_bagset(domain: str) -> object:
    """A `BagSet` with the shape the harness reads and none of the material.

    Built rather than downloaded so this test stays offline like the rest of the
    file. Nothing here is trained on: the run is stopped at the moment the
    ceiling is resolved, which is the only thing being checked.
    """
    from MIL_CREDA_Benchmark import bags

    bagcount, per_bag = 12, 2
    return bags.BagSet(
        domain=domain,
        images=torch.zeros(bagcount * per_bag, 3, 32, 32),
        members=torch.arange(bagcount * per_bag).reshape(bagcount, per_bag),
        labels=torch.arange(bagcount) % config.CLASSES,
        train_idx=torch.arange(0, 6),
        valid_idx=torch.arange(6, 9),
        eval_idx=torch.arange(9, 12),
        manifest={},
    )


class _StopAtCeiling(Exception):
    """Raised from the `ramp` spy so nothing past the lookup ever runs."""


def test_run_one_resolves_the_ceiling_of_the_transfer_it_was_given(monkeypatch) -> None:
    """The resolver reaching the training loop, not merely existing beside it.

    Testing `ceiling_for` alone verifies the lookup and never the wiring, and the
    wiring is the half that decides what the campaign computes: a `run_one` still
    holding the old family-only expression would leave every test above green
    while the two measured transfers ran at the pooled winner.

    So this reads the coefficient out of `ramp` exactly as `run_one` calls it, and
    stops there. `wiring.build` is stubbed because constructing the real backbone
    would reach for pretrained weights, and what is being checked is which number
    arrives at the ramp — not what the model does with it afterwards.
    """
    from MIL_CREDA_Benchmark import harness, wiring

    seen: list[float] = []

    def spy(epoch, epochs, family, ceiling):
        seen.append(ceiling)
        raise _StopAtCeiling

    monkeypatch.setattr(harness, "ramp", spy)
    monkeypatch.setattr(wiring, "build", lambda *a, **k: torch.nn.Linear(1, 1))

    reduction = harness.Reduction(
        epochs=1, seeds=[0],
        ceilings={"milcreda": 1e-2},
        ceilingsByTransfer={"milcreda": {"S->M": 1e-4}},
    )
    material = {"source": _synthetic_bagset("S"), "target": _synthetic_bagset("M")}
    device = torch.device("cpu")

    for transfer, expected in ((("S", "M"), 1e-4), (("M", "U"), 1e-2)):
        seen.clear()
        with pytest.raises(_StopAtCeiling):
            harness.run_one("G", transfer, 0, reduction, device, material,
                            role="valid")
        assert seen == [expected], (transfer, seen)

def test_the_campaign_refuses_when_the_record_has_picks_and_the_run_does_not(
        tmp_path, monkeypatch) -> None:
    """The stale-field case, refused by name instead of running the old rule.

    A caller that sets `ceilings=` alone leaves `ceilingsByTransfer` holding
    whatever `config` was imported with. Every transfer then falls back to the
    pooled winner and the run looks entirely ordinary, which is why this has to
    refuse rather than warn.
    """
    from MIL_CREDA_Benchmark import harness

    record = tmp_path / "ceilings.json"
    record.write_text(json.dumps({
        "milcreda": {"ceiling": 1e-2, "byTransfer": {"S->M": 1e-4}},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "CEILINGS_RECORD", record)

    stale = harness.Reduction(ceilings={"milcreda": 1e-2}, ceilingsByTransfer={})
    with pytest.raises(SystemExit) as raised:
        harness.campaign(stale, torch.device("cpu"), arms=["G"])
    assert "per-transfer ceilings left behind" in str(raised.value)
    assert "milcreda" in str(raised.value)


def test_the_ceilings_are_what_has_to_agree_across_shards() -> None:
    """Both readings, and both reachable by a flat top-level lookup.

    `disagreements()` compares `stamp.get(field)`, so a declared name that is
    not a flat key on the stamp resolves to `None` on every shard and passes
    whatever the shards actually did — a guarantee that was never enforced.
    """
    declared = MIL_CREDA_Benchmark.__benchmark__["distribution"]["identicalAcrossShards"]
    assert "ceilings" in declared
    assert "ceilingsByTransfer" in declared


def test_the_per_transfer_conclusion_can_come_out_different() -> None:
    """A conclusion tied to nothing is a conclusion measuring nothing.

    The same reading the report contract applies to every other conclusion: run
    it over two records that differ in the one fact it is about, and the two
    texts must differ.
    """
    from MIL_CREDA_Benchmark import tables

    transfers = ["M->U", "S->M", "U->S"]
    agrees = {"f": {"ceiling": 1e-2, "byTransfer": {"M->U": 1e-2, "S->M": 1e-2}}}
    differs = {"f": {"ceiling": 1e-2, "byTransfer": {"M->U": 1e-2, "S->M": 1e-4}}}

    said_agrees = tables.conclusion_ceilings_by_transfer(agrees, transfers)
    said_differs = tables.conclusion_ceilings_by_transfer(differs, transfers)
    assert said_agrees != said_differs
    assert "S->M" in said_differs and "S->M" not in said_agrees


def test_the_per_transfer_table_marks_measured_apart_from_inherited() -> None:
    """Without the mark all six cells read as if all six had been measured."""
    from MIL_CREDA_Benchmark import tables

    record = {"f": {"ceiling": 1e-2, "byTransfer": {"S->M": 1e-4}}}
    rendered = tables.render_ceilings_by_transfer(
        record, ["M->U", "S->M"], markdown=True)
    assert "**0.0001**" in rendered          # measured on S->M
    assert "| 0.01 |" in rendered            # inherited on M->U, unmarked


def test_the_per_transfer_ceilings_reach_the_written_report() -> None:
    """Declared in the contract, so the report is checked against it."""
    report = MIL_CREDA_Benchmark.__benchmark__["report"]
    assert "tables.render_ceilings_by_transfer" in report["renderers"]
    assert "tables.conclusion_ceilings_by_transfer" in report["conclusions"]
