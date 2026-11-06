"""Experiments plan v50: the campaign entry point, rehearsed on a SYNTHETIC set on CPU.

The synthetic set is seeded noise with a weak class signal; no number it yields is a
measured result.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import shutil
from pathlib import Path

import optuna
import pytest
import torch
from optuna.distributions import FloatDistribution

from MIL_CREDA_EEG import aggregation_variants, campaign, controls, mechanisms, model_store
from MIL_CREDA_EEG import mil_wrapper
from MIL_CREDA_EEG import partition
from MIL_CREDA_EEG import results
from MIL_CREDA_EEG import search, two_phase
from MIL_CREDA_EEG import statistics as st

BASE = {"schedule": {"epochs1": 1, "epochs2": 1, "bagsPerStep": 4},
        "strategies": ["MIL-CREDA"], "transfers": ["L1->L6"], "rehearsal": True}
FULL = {"schedule": BASE["schedule"], "strategies": list(model_store.STRATEGIES),
        "transfers": [t.name for t in campaign.assignment_mod.TRANSFERS]}


@pytest.fixture(scope="module")
def registries():
    """Comparator registration mutates global search state; restore it afterwards."""
    saved = {(m, name): dict(getattr(m, name)) for m, name in (
        (search, "PHASE_TWO_TRAINERS"), (search, "PHASE_TWO_EXTRA"),
        (search, "PHASE_ONE_TRAINERS"), (model_store, "BUILDERS"))}
    yield
    for (m, name), value in saved.items():
        getattr(m, name).clear()
        getattr(m, name).update(value)


def _rehearsal(tmp, conditions=("L1", "L6"), arm="level",
               strategies=("MIL-CREDA", "COT", "AdaptMIL")):
    cfg = campaign.write_synthetic_set(tmp / "synthetic", seed=0, subjects=24,
                                       conditions=conditions, arm=arm)
    cfg.update({"strategies": list(strategies), "nTrials": 1, "reportSeeds": [1],
                "bootstrapResamples": 200,
                "schedule": {"epochs1": 1, "epochs2": 1, "bagsPerStep": 4}})
    product = tmp / "product"
    product.mkdir()
    return cfg, product, campaign.run("all", cfg, product)


@pytest.fixture(scope="module")
def level_run(tmp_path_factory, registries):
    tmp = tmp_path_factory.mktemp("level")
    return _rehearsal(tmp)


def test_the_steps_are_the_declared_ones_in_order():
    assert campaign.STEPS == ("search", "final", "k3", "statistics", "mechanisms", "results")
    assert list(campaign.RUNNERS) == list(campaign.STEPS)
    with pytest.raises(campaign.CampaignError, match="step"):
        campaign.run("launch", {**BASE}, "/nonexistent")


def test_required_inputs_have_no_default():
    for key in ("schedule", "strategies", "transfers"):
        with pytest.raises((campaign.CampaignError, ValueError)):
            campaign.check_config({k: v for k, v in BASE.items() if k != key})
    with pytest.raises(ValueError, match="schedule"):
        campaign.check_config({**BASE, "schedule": {"epochs1": 1}})
    with pytest.raises(campaign.CampaignError, match="MIL-CREDA"):
        campaign.check_config({**BASE, "strategies": ["COT"]})
    # Sec. 4.6: one report seed, seed 1; the search seed is never one
    with pytest.raises(campaign.CampaignError, match="report seed"):
        campaign.check_config({**BASE, "reportSeeds": [0, 1]})
    with pytest.raises(campaign.CampaignError, match="one report seed"):
        campaign.check_config({**BASE, "reportSeeds": [1, 2]})
    cfg = campaign.check_config(BASE)
    assert cfg["nTrials"] == search.N_TRIALS == 20
    assert cfg["reportSeeds"] == list(campaign.REPORT_SEEDS) == [1]
    assert search.SEARCH_SEED == 0
    assert cfg["bootstrapResamples"] == st.BOOTSTRAP_RESAMPLES
    # plan v42 retired these inputs: refused, never silently ignored
    for key, value in (("seedReduction", "mean-probability"),
                       ("k1", {"commonAttention": "COT", "bands": [[4, 8], [13, 30]],
                               "scope": "pooled"}),
                       ("pendingDecisions", {"clust-ema-range": "v42"})):
        with pytest.raises(campaign.CampaignError, match="retired"):
            campaign.check_config({**BASE, key: value})
    assert not hasattr(campaign, "check_mechanism_inputs")


def test_every_strategy_runs_without_an_open_decision_guard(registries, monkeypatch):
    assert not hasattr(campaign, "PENDING_DECISIONS")
    for strategy in model_store.STRATEGIES:
        cfg = campaign.check_config({**BASE, "strategies": sorted({"MIL-CREDA", strategy})})
        assert strategy in cfg["strategies"]
    # CASUAL's v42 register() takes no argument (the VAT radius axis is its own, Sec. 7.7)
    from MIL_CREDA_EEG.comparators.casual import training as casual
    calls = []
    monkeypatch.setattr(casual, "register", lambda *a, **k: calls.append((a, k)))
    monkeypatch.delitem(search.PHASE_TWO_TRAINERS, "CASUAL", raising=False)
    campaign.register(["MIL-CREDA", "CASUAL"])
    assert calls == [((), {})]


def test_the_campaign_never_launches_remote_work():
    source = Path(campaign.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {"subprocess", "socket", "urllib", "requests", "http", "kaggle", "kagglesdk"}
    assert not {m.split(".")[0] for m in imported} & forbidden
    assert "remote" not in " ".join(imported)
    assert "submit" not in campaign.STEPS
    parser_choices = inspect.getsource(campaign.main)
    assert "submit" not in parser_choices and "launch" not in parser_choices


def _identity(sha="x" * 64):
    return {"partition_sha256": sha, "revision": campaign.REVISION,
            "code_version": model_store.code_version()}


def test_an_incomplete_study_is_refused_not_extended(tmp_path):
    missing = tmp_path / "absent.db"
    with pytest.raises(campaign.CampaignError, match="search step"):
        campaign._study(missing, 2, "x" * 64)
    assert not missing.exists()
    path = tmp_path / "s.db"
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{path}")
    for key, value in _identity().items():
        study.set_user_attr(key, value)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
    with pytest.raises(campaign.CampaignError, match="1 completed trials of 2"):
        campaign._study(path, 2, "x" * 64)
    assert len(campaign._study(path, 1, "x" * 64).trials) == 1


def test_a_stale_study_is_refused_naming_what_differs(tmp_path, monkeypatch):
    path = tmp_path / "s.db"
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{path}")
    for key, value in _identity().items():
        study.set_user_attr(key, value)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
    with pytest.raises(campaign.CampaignError, match="partition_sha256"):
        campaign._study(path, 1, "y" * 64)
    monkeypatch.setattr(model_store, "code_version", lambda: "0" * 64)
    with pytest.raises(campaign.CampaignError, match="code_version"):
        campaign._study(path, 1, "x" * 64)


def _light_data(sha="x" * 64):
    """A SearchData that carries only what an identity reads: transfer, partition, folds."""
    return search.SearchData(transfer=campaign.assignment_mod.TRANSFERS[0],
                             partition_sha256=sha, rotations={},
                             folds=search.rotation_folds({str(r): partition.rotation_roles(r)
                                                          for r in range(4)}))


def _lowest(space):
    return {k: getattr(d, "low", None) if hasattr(d, "low") else d.choices[0]
            for k, d in space.items()}


def test_a_stored_study_is_reused_only_under_its_full_identity(tmp_path):
    """Partition, plan revision, code version, search space, schedule, sampler seed, trial
    budget, outer fold and inner splits: a stored study differing in any is refused naming
    the field."""
    data = _light_data()
    cfg = {"schedule": BASE["schedule"], "nTrials": 1}
    path = search.study_path(tmp_path, data.transfer, 1, "MIL-CREDA", 2)
    path.parent.mkdir(parents=True)
    full = search.study_identity(data, 1, "MIL-CREDA", 2, cfg["schedule"], 1, "objective")

    def stored(attrs):
        path.unlink(missing_ok=True)
        study = optuna.create_study(study_name=path.stem, storage=f"sqlite:///{path}")
        for key, value in attrs.items():
            study.set_user_attr(key, value)
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

    stored(full)
    assert len(campaign._stored_study(tmp_path, data, 1, "MIL-CREDA", 2, cfg).trials) == 1
    for field, value in (("outerFold", 3), ("testFold", 3),
                         ("innerSplits", full["innerSplits"][::-1]),
                         ("partition_sha256", "y" * 64),
                         ("revision", "experiments-plan-v43.md"),
                         ("code_version", "0" * 64),
                         ("space", {**full["space"], "lr1": "another distribution"}),
                         ("schedule", {**full["schedule"], "epochs2": 2}),
                         ("samplerSeed", full["samplerSeed"] + 1),
                         ("nTrials", 2)):
        stored({**full, field: value})
        with pytest.raises(campaign.CampaignError, match=f"stale.*{field}"):
            campaign._stored_study(tmp_path, data, 1, "MIL-CREDA", 2, cfg)


def _stored_sidecar(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def test_a_stored_phase_one_checkpoint_is_reused_only_under_its_hyperparameters(tmp_path):
    data = _light_data()
    cfg = {"schedule": BASE["schedule"], "nTrials": 1, "device": "cpu"}
    params = _lowest(search.PHASE_ONE_SPACES["MIL-CREDA"])
    # the selected configuration and its E1, refit on the non-test folds (Sec. 4.5)
    selection = {"params": params, "epochs": 1}
    hyper = search.phase_one_hyper("MIL-CREDA", params,
                                   search.refit_schedule(cfg["schedule"], 1, 1))
    hyper = {k: hyper[k] for k in (*two_phase.PHASE_ONE_KEYS, *two_phase.SCHEDULE_KEYS)}
    path = model_store.model_path(campaign._models(tmp_path), data.transfer, 0,
                                  search.PHASE_ONE_SEED, 1, "MIL-CREDA")
    meta = {**campaign.identity(data.partition_sha256), "hyper": hyper}
    _stored_sidecar(path, meta)
    assert campaign._phase_one_checkpoint("MIL-CREDA", data, 0, selection, cfg,
                                          tmp_path) == (path, [])
    for key in ("lr1", "epochs1"):              # a searched value, the selected E1
        _stored_sidecar(path, {**meta, "hyper": {**hyper, key: hyper[key] * 2}})
        with pytest.raises(campaign.CampaignError, match=f"stale.*hyper {key}"):
            campaign._phase_one_checkpoint("MIL-CREDA", data, 0, selection, cfg,
                                           tmp_path)
    _stored_sidecar(path, meta)
    with pytest.raises(campaign.CampaignError, match="stale.*hyper epochs1"):
        campaign._phase_one_checkpoint("MIL-CREDA", data, 0, {**selection, "epochs": 2},
                                       cfg, tmp_path)
    _stored_sidecar(path, {**meta, "revision": "experiments-plan-v43.md"})
    with pytest.raises(campaign.CampaignError, match="stale.*revision"):
        campaign._phase_one_checkpoint("MIL-CREDA", data, 0, selection, cfg, tmp_path)


def test_a_stored_phase_two_model_is_reused_only_under_the_requested_selection(
        tmp_path, monkeypatch):
    data = _light_data()
    cfg = {"schedule": BASE["schedule"], "nTrials": 1, "device": "cpu"}
    params = _lowest(search.phase_two_space("MIL-CREDA"))
    selection = {"rule": "primary", "trial": 0, "params": params, "epoch": 0, "epochs": 1,
                 "testFold": 0, "selectionRotations": [1, 2, 3], "excludedRotation": 0,
                 "score": 0.5}
    refit = search.refit_schedule(cfg["schedule"], 2, 1)
    root = campaign._models(tmp_path)
    path = model_store.model_path(root, data.transfer, 0, 1, 2, "MIL-CREDA")
    record = ({}, {"transfer": data.transfer.name, "rotation": 0,
                   "seed": search.PHASE_ONE_SEED, "phase": 1, "name": "MIL-CREDA",
                   "partition_sha256": data.partition_sha256, "state_sha256": "s" * 64})
    started = search.phase_one_checkpoint_identity(record[1])
    meta = {**campaign.identity(data.partition_sha256), "phaseOneCheckpoint": started,
            "hyper": {**params, **refit},
            "selection": {"rule": "primary", "trial": 0, "params": params, "epoch": 0,
                          "schedule": refit}}
    loaded = torch.nn.Identity()
    monkeypatch.setattr(mechanisms, "load_phase_two", lambda p, **k: (loaded, {}, []))

    def reuse():
        return campaign._phase_two_model(root, "MIL-CREDA", data, 0, 1, record, selection,
                                         cfg, torch.device("cpu"), None)

    _stored_sidecar(path, meta)
    model, history = reuse()
    assert model is loaded and history is None
    lr2 = {**params, "lr2": params["lr2"] * 2}
    schedule = {**refit, "epochs2": 2}
    for change, field in (
            ({"selection": {**meta["selection"], "epoch": 1}}, "selection epoch"),
            ({"selection": {**meta["selection"], "params": lr2}}, "selection params"),
            ({"selection": {**meta["selection"], "schedule": schedule}}, "selection schedule"),
            ({"hyper": {**meta["hyper"], "lr2": lr2["lr2"]}}, "hyper lr2"),
            ({"hyper": {**meta["hyper"], "epochs2": 2}}, "hyper epochs2"),
            ({"selection": {"rule": "primary", "trial": 0}}, "selection params"),
            ({"code_version": "0" * 64}, "code_version"),
            # refit from another phase-1 checkpoint than the one now in force (Sec. 4.5)
            ({"phaseOneCheckpoint": {**started, "seed": search.SEARCH_SEED}},
             "phaseOneCheckpoint")):
        _stored_sidecar(path, {**meta, **change})
        with pytest.raises(campaign.CampaignError, match=f"stale.*{field}"):
            reuse()


def test_stale_predictions_are_refused_naming_what_differs(tmp_path, monkeypatch):
    root = tmp_path / "Results"
    results.append_predictions(root, "A-nivel", [{"strategy": "MIL-CREDA", "transfer": "t",
                                                  "rotation": 0, "seed": 1}],
                               identity=_identity())
    assert campaign._predicted(tmp_path, "A-nivel", _identity()) == {
        ("MIL-CREDA", "t", 0, 1)}
    for current, field in ((_identity("y" * 64), "partition_sha256"),
                           ({**_identity(), "revision": "plan-v44.md"}, "revision"),
                           ({**_identity(), "code_version": "0" * 64}, "code_version")):
        with pytest.raises(campaign.CampaignError, match=field):
            campaign._predicted(tmp_path, "A-nivel", current)
        with pytest.raises(campaign.CampaignError, match=field):
            campaign._read_predictions(tmp_path, "A-nivel", current)


def test_every_changed_module_answers_to_one_plan_revision():
    assert campaign.REVISION.endswith("v57.md")
    for module in (search, results, model_store):
        assert module.__provenance__["revision"] == campaign.REVISION


def test_a_reduced_run_is_refused_outside_rehearsal_mode():
    cfg = campaign.check_config(FULL)
    assert cfg["rehearsal"] is False and cfg["nTrials"] == search.N_TRIALS
    for change, field in (({"strategies": ["MIL-CREDA", "COT"]}, "strategies"),
                          ({"transfers": ["L1->L6"]}, "transfers"),
                          ({"nTrials": 4}, "nTrials")):
        with pytest.raises(campaign.CampaignError, match=field) as refused:
            campaign.check_config({**FULL, **change})
        assert "rehearsal" in str(refused.value)
        assert campaign.check_config({**FULL, **change, "rehearsal": True})["rehearsal"]
    # a synthetic set is a rehearsal by construction
    assert campaign.check_config({**FULL, "nTrials": 4, "synthetic": True})["rehearsal"]


def test_the_rehearsal_summaries_say_so(level_run):
    _, product, _ = level_run
    for experiment in ("A-nivel", "K3-ablacion", "K1-atencion"):
        summary = json.loads((product / "Results" / experiment / "summary.json").read_text())
        assert summary["rehearsal"] is True


def test_the_final_models_are_trained_at_the_report_seeds_from_the_same_rotation(level_run):
    cfg, product, outcome = level_run
    models = product / "Models" / "Balladeer"
    for transfer in ("L1-to-L6", "L6-to-L1"):
        for r in range(4):
            for name in ("phase1__common", "phase1__AdaptMIL", "phase1__MIL-CREDA"):
                assert (models / transfer / f"rotation{r}" / "seed1" / f"{name}.pt").exists()
                assert not (models / transfer / f"rotation{r}" / "seed0" / f"{name}.pt").exists()
            assert (models / transfer / f"rotation{r}" / "seed1" / "phase2__COT.pt").exists()
            assert not (models / transfer / f"rotation{r}" / "seed0" / "phase2__COT.pt").exists()
    records = results.read_predictions(product / "Results", "A-nivel")
    assert {r["seed"] for r in records} == {1}
    strategies = {r["strategy"] for r in records}
    assert strategies == {"MIL-CREDA", "COT", "AdaptMIL", "source-only:common",
                          "source-only:AdaptMIL", "source-only:MIL-CREDA", "upper-bound"}
    st.check_once(records)
    for transfer in ("L1->L6", "L6->L1"):
        for r in range(4):
            mine = {x["subject"] for x in records if x["transfer"] == transfer
                    and x["rotation"] == r and x["strategy"] == "MIL-CREDA"}
            assert mine == {x["subject"] for x in records if x["transfer"] == transfer
                            and x["rotation"] == r and x["strategy"] == "upper-bound"}
    assert outcome["final"]["trained_phase2"] == 3 * 4 * 2


def test_a_stored_model_is_not_trained_again(level_run, registries):
    cfg, product, _ = level_run
    before = len(results.read_predictions(product / "Results", "A-nivel"))
    again = campaign.run_final(campaign.check_config(cfg), product)
    assert again == {"trained_phase1": 0, "trained_phase2": 0, "predictions": 0,
                     "trained_label_free": 0, "label_free_predictions": 0}
    assert len(results.read_predictions(product / "Results", "A-nivel")) == before


def test_the_device_arm_source_only_controls_are_never_numbers(tmp_path, registries):
    cfg, product, outcome = _rehearsal(tmp_path, conditions=("L6", "Robots"), arm="device",
                                       strategies=("MIL-CREDA",))
    records = results.read_predictions(product / "Results", "C-dispositivo")
    assert {r["strategy"] for r in records} == {"MIL-CREDA"}
    table = (product / "Results" / "C-dispositivo" / "tables" / "comparison.md").read_text()
    # v56 Sec. 5.3: three source-only rows (one per phase-1 group), three columns each
    assert table.count(controls.NA_RECORD) == 3 * 3


def test_the_rehearsal_writes_the_dashboard_layout_end_to_end(level_run):
    cfg, product, outcome = level_run
    root = product / "Results"
    for rel in ("significance_A-nivel.json", "significance_K1-atencion.json",
                "significance_K2-emparejamiento.json", "A-nivel/summary.json",
                "A-nivel/tables/comparison.csv", "A-nivel/tables/secondary.csv",
                "A-nivel/tables/transfer_detail.csv", "A-nivel/tables/per_fold.csv",
                "A-nivel/tables/ranking.csv", "A-nivel/tables/selection_rules.csv",
                "A-nivel/tables/hyperparameters.csv", "K1-atencion/summary.json",
                "K2-emparejamiento/tables/separation.csv", "K3-ablacion/tables/term_bounds.csv",
                "K3-ablacion/summary.json", "history_A-nivel.json"):
        assert (root / rel).exists(), rel
    summary = json.loads((root / "A-nivel" / "summary.json").read_text())
    assert summary["synthetic"] is True and summary["nTrials"] == 1
    assert summary["reportSeeds"] == [1] and "seedReduction" not in summary
    rows = json.loads((root / "significance_A-nivel.json").read_text())
    rivals = {r["comparison"].split(" vs ")[1] for r in rows
              if r["test"] == st.TEST_BOOTSTRAP}
    assert rivals == {"COT", "AdaptMIL", "Source only, common attention",
                      "Source only, AdaptMIL",
                      "Source only, MIL-CREDA"}
    assert all("holm" in r["correction"] for r in rows if r["test"] == st.TEST_BOOTSTRAP)
    # v44 Sec. 4.6: primary and secondary families, Holm within each metric; no row of the
    # arm is reported uncorrected, and the per-arm Friedman is gone (Sec. 5.4 is global).
    assert {r["test"] for r in rows} == set(st.CONTRAST_TESTS)
    assert all(r["correction"] != "none" and "holm" in r["correction"] for r in rows)
    for rel in ("A-nivel/tables/significance.csv", "K1-atencion/tables/tests.csv",
                "K2-emparejamiento/tables/tests.csv"):
        assert (root / rel).exists(), rel
    assert any("global critical-difference" in o for o in outcome["results"]["owed"])
    assert "mil_creda_top_three" in summary
    k1 = json.loads((root / "significance_K1-atencion.json").read_text())
    assert any("event_window" in (r.get("not_computed") or "") for r in k1)
    # v56 Sec. 6.1: the rehearsal runs only the level arm, which keeps only the event test;
    # the two task-arm criterion tests need a task arm, so the level event row is alone
    criterion = [r for r in k1 if r.get("criterion") == mechanisms.K1_CRITERION]
    assert [r["comparison"] for r in criterion] == ["K1-atencion:level:event vs rest, MIL-CREDA"]
    assert "event_window" in (criterion[0]["not_computed"] or "")
    assert not any("marked vs clean" in r["comparison"] for r in k1)
    references = {r["comparison"].split(", ", 1)[1] for r in k1
                  if r.get("criterion", "").startswith("none (reference")}
    assert references == {"MIL-CREDA [published attention]", "MIL-CREDA [gated attention]",
                          "common attention [COT]", "AdaptMIL"}
    assert not any("mean" in r["comparison"] or "band" in r["comparison"] for r in k1)
    k1_summary = json.loads((root / "K1-atencion" / "summary.json").read_text())
    assert k1_summary["excluded"] == mechanisms.K1_EXCLUDED
    assert "bands" not in k1_summary and "scope" not in k1_summary
    k2 = json.loads((root / "significance_K2-emparejamiento.json").read_text())
    no_local = next(r for r in k2 if r["comparison"] ==
                    "K2-emparejamiento:L1->L6 (smaller shift):MIL-CREDA vs "
                    "MIL-CREDA [global term only]")
    assert no_local["n"] > 0 and not no_local.get("not_computed")
    assert outcome["mechanisms"]["owed"] == []
    history = json.loads((root / "history_A-nivel.json").read_text())
    assert {"epoch", "seed", "fold", "model"} <= set(history[0])
    assert {h["model"] for h in history} >= {"MIL-CREDA:phase1", "MIL-CREDA:phase2",
                                              "common:phase1"}
    written = {Path(w).relative_to(root).as_posix() for w in outcome["results"]["written"]}
    assert {"A-nivel/figures/convergence.pdf", "A-nivel/figures/optimization_history.pdf"
            } <= written
    # v44 Sec. 5.4: the critical-difference diagram is global, over the ten transfers; a
    # rehearsal on one arm owes it instead of drawing a per-arm one.
    assert "A-nivel/figures/critical_difference.pdf" not in written
    for w in written:
        if w.endswith(".pdf"):
            assert (root / w).with_suffix(".png").exists()


def _csv(path: Path) -> list[dict]:
    import csv
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_k3_trains_the_variants_and_writes_both_tables_with_holm_per_table(level_run):
    cfg, product, outcome = level_run
    root = product / "Results"
    records = results.read_predictions(root, campaign.K3)
    st.check_once(records)
    variants = {f"aggregation:{v}" for v in aggregation_variants.VARIANTS}
    assert {r["strategy"] for r in records} == variants | {"ablation:global term only",
                                                           "ablation:local term only"}
    assert outcome["k3"]["trained_aggregation"] == 4 * 4 * 2
    assert outcome["k3"]["trained_ablation"] == 2 * 4 * 2
    assert outcome["k3"]["owed"] == []
    full = {(r["transfer"], r["rotation"], r["seed"], r["subject"])
            for r in results.read_predictions(root, "A-nivel") if r["strategy"] == "MIL-CREDA"}
    for name in {r["strategy"] for r in records}:
        assert {(r["transfer"], r["rotation"], r["seed"], r["subject"]) for r in records
                if r["strategy"] == name} == full
    for rel in ("K3-ablacion/tables/aggregation.csv", "K3-ablacion/tables/ablation.csv",
                "K3-ablacion/tables/variant_hyperparameters.csv",
                "K3-ablacion/tables/term_bounds.csv", "significance_K3-ablacion.json",
                "history_K3-ablacion.json", "K3-ablacion/summary.json"):
        assert (root / rel).exists(), rel
    aggregation = _csv(root / "K3-ablacion" / "tables" / "aggregation.csv")
    assert [r["variant"] for r in aggregation] == ["own", *aggregation_variants.VARIANTS]
    assert {r["pair"] for r in aggregation} == {"L1|L6"}
    ablation = _csv(root / "K3-ablacion" / "tables" / "ablation.csv")
    assert [r["variant"] for r in ablation] == ["global term only", "local term only",
                                                "both terms"]
    rows = json.loads((root / "significance_K3-ablacion.json").read_text())
    assert {r["experiment"] for r in rows} == {"K3-ablacion"}
    boot = [r for r in rows if r["test"] == st.TEST_BOOTSTRAP]
    agg = [r for r in boot if ":aggregation:" in r["comparison"]]
    abl = [r for r in boot if ":ablation:" in r["comparison"]]
    assert [r["comparison"] for r in agg] == ["K3-ablacion:aggregation:L1|L6:own vs "
                                              "own without consensus"]
    assert [r["comparison"] for r in abl] == [
        "K3-ablacion:ablation:L1|L6:both terms vs global term only",
        "K3-ablacion:ablation:L1|L6:both terms vs local term only"]
    assert all("aggregation" in r["correction"] and "m=1" in r["correction"] for r in agg)
    assert all("ablation" in r["correction"] and "m=2" in r["correction"] for r in abl)
    assert all(r["correction"] != "none" for r in rows)
    hyper = _csv(root / "K3-ablacion" / "tables" / "variant_hyperparameters.csv")
    for row in hyper:
        if row["variant"] == "own without consensus":
            assert float(row["attentionGamma"]) == 0.0
            assert row["attentionTemperature"] not in ("", "not applicable")
        elif row["table"] == "aggregation":
            assert row["attentionGamma"] == row["attentionTemperature"] == "not applicable"
        else:
            assert row["attentionGamma"] not in ("", "not applicable")
    # Sec. 6.3: every variant model is stored, as the final models are (Sec. 6.1 reads them)
    models = product / "Models" / "Balladeer"
    for transfer in ("L1-to-L6", "L6-to-L1"):
        for r in range(4):
            folder = models / transfer / f"rotation{r}" / "seed1"
            for v in aggregation_variants.VARIANTS:
                slug = model_store.name_slug(model_store.variant_name(v))
                assert (folder / f"phase1__{slug}.pt").exists()
                assert (folder / f"phase2__{slug}.pt").exists()
            for v in ("global term only", "local term only"):
                slug = model_store.name_slug(model_store.variant_name(v))
                assert (folder / f"phase2__{slug}.pt").exists()
                assert not (folder / f"phase1__{slug}.pt").exists()
    summary = json.loads((root / "K3-ablacion" / "summary.json").read_text())
    assert summary["term_bounds"] and summary["aggregation_variants"]["trained"] == list(
        aggregation_variants.VARIANTS)
    assert summary["ablation_variants"]["trained"] == ["global term only", "local term only"]
    history = json.loads((root / "history_K3-ablacion.json").read_text())
    assert {h["model"] for h in history} == {
        *(f"MIL-CREDA [{v}]:phase{n}" for v in aggregation_variants.VARIANTS for n in (1, 2)),
        "MIL-CREDA [global term only]:phase2", "MIL-CREDA [local term only]:phase2"}


def test_k3_does_not_train_a_variant_again(level_run, registries):
    cfg, product, _ = level_run
    before = len(results.read_predictions(product / "Results", campaign.K3))
    stored = sorted((product / "Models").rglob("*.pt"))
    again = campaign.run_k3(campaign.check_config(cfg), product)
    assert again["trained_aggregation"] == again["trained_ablation"] == 0
    assert len(results.read_predictions(product / "Results", campaign.K3)) == before
    assert sorted((product / "Models").rglob("*.pt")) == stored


def test_k2_trains_the_global_term_only_variant_on_its_declared_transfers():
    cfg = campaign.check_config({**BASE, "transfers": ["L1->L11", "L11->L1", "L1->L6",
                                                       "L6->L1", "L6->Robots"]})
    core = {"level": ["L1->L11", "L11->L1"], "device": ["L6->Robots", "Robots->L6"]}
    plan = campaign.k3_plan(cfg, core)
    full = {"aggregation": aggregation_variants.VARIANTS,
            "ablation": ("global term only", "local term only")}
    # core transfers train both tables; the declared smaller shift (Sec. 6.2), not a core
    # transfer, trains only the global-term-only variant; any other transfer, nothing
    assert plan == {"L1->L11": full, "L11->L1": full, "L6->Robots": full,
                    "L1->L6": {"aggregation": (), "ablation": ("global term only",)}}


def _sidecar(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text())


def test_both_phases_are_selected_per_outer_fold_and_refit_for_their_epochs(level_run):
    cfg, product, _ = level_run
    models = product / "Models" / "Balladeer"
    bags = campaign.load_bags(campaign.check_config(cfg))
    for transfer in campaign.transfers_of(campaign.check_config(cfg)):
        data = search.search_data(transfer, bags)
        slug = model_store.transfer_slug(transfer)
        for r in range(4):
            # phase 1: the configuration and E1 selected in outer fold r's own study
            one = search.select(campaign._study(
                search.study_path(product, transfer, 1, "MIL-CREDA", r), cfg["nTrials"],
                data.partition_sha256))
            meta = _sidecar(models / slug / f"rotation{r}" / f"seed{search.PHASE_ONE_SEED}"
                            / "phase1__MIL-CREDA.pt")
            assert {k: meta["hyper"][k] for k in one["params"]} == one["params"]
            assert meta["hyper"]["epochs1"] == one["epochs"]
            for strategy in ("MIL-CREDA", "COT"):
                study = campaign._study(search.study_path(product, transfer, 2, strategy, r),
                                        cfg["nTrials"], data.partition_sha256)
                chosen = search.select(study)
                meta = _sidecar(models / slug / f"rotation{r}" / "seed1"
                                / f"phase2__{strategy}.pt")
                assert (meta["selection"]["trial"], meta["selection"]["epoch"]) == \
                    (chosen["trial"], chosen["epoch"])
                assert meta["selection"]["testFold"] == r == meta["selection"]["excludedRotation"]
                assert meta["selection"]["selectionRotations"] == [f for f in range(4) if f != r]
                assert meta["selection"]["schedule"]["epochs2"] == chosen["epochs"]
                assert {k: meta["hyper"][k] for k in chosen["params"]} == chosen["params"]
    detail = _csv(product / "Results" / "A-nivel" / "tables" / "selected_configurations.csv")
    assert {(d["strategy"], d["transfer"], d["rule"], d["rotation"]) for d in detail} == {
        (s, t, rule, str(r)) for s in cfg["strategies"] for t in cfg["transfers"]
        for rule in ("primary", "label-free") for r in range(4)}
    k3 = _csv(product / "Results" / "K3-ablacion" / "tables" / "variant_hyperparameters.csv")
    assert {d["rotation"] for d in k3} == {"0", "1", "2", "3"}


def test_one_phase_one_checkpoint_per_group_and_outer_fold(level_run):
    """Sec. 4.5: the phase-1 refit leaves one checkpoint per group and outer fold, at the
    report seed, with no checkpoint per inner split or per seed; every phase-2 study of that
    fold and its phase-2 final refit start from it."""
    cfg, product, _ = level_run
    models = product / "Models" / "Balladeer"
    group_files = {f"phase1__{model_store.name_slug(g)}.pt" for g in model_store.GROUPS}
    stored = sorted(p.relative_to(models).as_posix() for p in models.rglob("phase1__*.pt")
                    if p.name in group_files)        # the K3 variants store their own
    groups = ("common", "AdaptMIL", "MIL-CREDA")      # the groups COT, AdaptMIL, MIL-CREDA use
    assert stored == sorted(f"{t}/rotation{r}/seed{s}/phase1__{g}.pt"
                            for t in ("L1-to-L6", "L6-to-L1") for r in range(4)
                            for s in (search.PHASE_ONE_SEED,) for g in groups)
    for path in (models / x for x in stored):
        meta = _sidecar(path)
        assert "validation" not in meta and meta["epoch"] == meta["hyper"]["epochs1"] - 1


def test_phase_two_search_and_final_refit_start_from_the_same_phase_one_checkpoint(
        level_run):
    """Secs. 4.5, 7.7: per group and outer fold, the phase-2 study's trials and the
    phase-2 final refit both start from the one phase-1 refit checkpoint ("desde el mismo
    punto de control"); each records that checkpoint's identity."""
    cfg, product, _ = level_run
    models = product / "Models" / "Balladeer"
    bags = campaign.load_bags(campaign.check_config(cfg))
    for transfer in campaign.transfers_of(campaign.check_config(cfg)):
        data = search.search_data(transfer, bags)
        slug = model_store.transfer_slug(transfer)
        for r in range(4):
            for strategy in cfg["strategies"]:
                group = search.PHASE_ONE_GROUP_OF[strategy]
                [checkpoint] = sorted((models / slug / f"rotation{r}").glob(
                    f"seed*/phase1__{model_store.name_slug(group)}.pt"))
                expected = search.phase_one_checkpoint_identity(_sidecar(checkpoint))
                study = campaign._study(search.study_path(product, transfer, 2, strategy, r),
                                        cfg["nTrials"], data.partition_sha256)
                final = _sidecar(models / slug / f"rotation{r}" / "seed1"
                                 / f"phase2__{model_store.name_slug(strategy)}.pt")
                assert study.user_attrs["phaseOneCheckpoint"] == expected, (strategy, r)
                assert final["phaseOneCheckpoint"] == expected, (strategy, r)


def _other_choice(real):
    """`search.select` with the label-free rule moved off the primary (trial, epoch) in the
    phase-2 studies of the outer folds `differ` names, and onto it elsewhere."""
    def choose(study, rule="primary", differ=(0, 1, 2, 3)):
        chosen = real(study, rule)
        if rule != "label-free" or study.user_attrs.get("phase") != 2:
            return chosen
        primary = real(study, "primary")
        if study.user_attrs["outerFold"] not in differ:
            return {**chosen, **{k: primary[k] for k in ("trial", "params", "epoch",
                                                         "epochs")}}
        n = 1 - primary["trial"]
        return {**chosen, "trial": n, "params": dict(study.trials[n].params),
                "epoch": 0, "epochs": 1}
    return choose


def test_the_label_free_rule_is_measured_on_the_test_fold(tmp_path, registries, monkeypatch):
    choose = _other_choice(search.select)
    monkeypatch.setattr(search, "select",
                        lambda study, rule="primary": choose(study, rule, differ=(1, 2, 3)))
    cfg = campaign.write_synthetic_set(tmp_path / "synthetic", seed=0, subjects=24)
    cfg.update({"strategies": ["MIL-CREDA"], "nTrials": 2, "bootstrapResamples": 200,
                "schedule": {"epochs1": 1, "epochs2": 1, "bagsPerStep": 4}})
    product = tmp_path / "product"
    product.mkdir()
    outcome = {s: campaign.run(s, cfg, product) for s in ("search", "final", "results")}
    # rotation 0 coincides (same trial and epoch): the primary model is reused, no refit
    assert outcome["final"]["final"]["trained_label_free"] == 3 * 2
    root = product / "Models" / "Balladeer" / campaign.LABEL_FREE_MODELS
    for slug in ("L1-to-L6", "L6-to-L1"):
        assert not (root / slug / "rotation0").exists()
        for r in (1, 2, 3):
            meta = _sidecar(root / slug / f"rotation{r}" / "seed1" / "phase2__MIL-CREDA.pt")
            assert meta["selection"]["rule"] == "label-free"
            assert meta["selection"]["epochRule"] == two_phase.LABEL_FREE_EPOCH_RULE
            primary = _sidecar(product / "Models" / "Balladeer" / slug / f"rotation{r}"
                               / "seed1" / "phase2__MIL-CREDA.pt")
            assert meta["selection"]["trial"] != primary["selection"]["trial"]
    res = product / "Results"
    primary = results.read_predictions(res, "A-nivel")
    free = results.read_predictions(res, "A-nivel", rule="label-free")
    key = lambda x: (x["transfer"], x["rotation"], x["subject"])  # noqa: E731
    p0 = sorted((key(x), x["probability"]) for x in primary
                if x["strategy"] == "MIL-CREDA" and x["rotation"] == 0)
    f0 = sorted((key(x), x["probability"]) for x in free if x["rotation"] == 0)
    assert p0 and f0 == p0          # where the rules coincide, the primary model is reused
    assert {x["rotation"] for x in free} == {0, 1, 2, 3}
    assert {x["strategy"] for x in free} == {"MIL-CREDA"}
    rows = _csv(res / "A-nivel" / "tables" / "selection_rules.csv")
    for row in rows:
        mine = [x for x in free if x["transfer"] == row["transfer"]]
        expected = st.pooled(mine)["balanced_accuracy"]
        assert float(row["test BA, label-free rule"]) == pytest.approx(expected, abs=1e-4)
        assert float(row["difference"]) == pytest.approx(
            float(row["test BA, primary rule"]) - expected, abs=2e-4)
        assert row["coinciding rotations"] == "1"
    # nothing is trained twice
    again = campaign.run_final(campaign.check_config(cfg), product)
    assert again["trained_label_free"] == again["label_free_predictions"] == 0


@pytest.fixture(scope="module")
def label_free_run(tmp_path_factory, registries):
    """Search, final, results and mechanisms with the label-free rule choosing another
    trial than the primary rule in every outer fold, so every label-free model is its own."""
    other_trial = _other_choice(search.select)

    tmp = tmp_path_factory.mktemp("label-free")
    cfg = campaign.write_synthetic_set(tmp / "synthetic", seed=0, subjects=24)
    cfg.update({"strategies": ["MIL-CREDA", "AdaptMIL"], "nTrials": 2, "bootstrapResamples": 200,
                "schedule": {"epochs1": 3, "epochs2": 4, "bagsPerStep": 4}})
    product = tmp / "product"
    product.mkdir()
    read_by_bounds, figures = [], []
    real_bounds = mechanisms.bound_control
    real_figure = results.convergence_figure

    def recording_figure(root, experiment, history, core, **kwargs):
        path = real_figure(root, experiment, history, core, **kwargs)
        figures.append((path.stem, kwargs.get("epoch_key", "val_balanced_accuracy")))
        return path

    def recording_bounds(history, *args, **kwargs):
        history = list(history)
        read_by_bounds.extend(history)
        return real_bounds(history, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(search, "select", other_trial)
        patch.setattr(mechanisms, "bound_control", recording_bounds)
        patch.setattr(results, "convergence_figure", recording_figure)
        for step in ("search", "final", "results", "mechanisms"):
            campaign.run(step, cfg, product)
    return cfg, product, read_by_bounds, figures


def test_the_label_free_final_models_are_refit_at_their_trial_and_epoch(label_free_run):
    """Sec. 4.5: a label-free final model is the label-free rule's (trial, epoch) refit on
    the non-test folds for exactly that many epochs, and its sidecar says which and by
    which rule."""
    cfg, product, _, _ = label_free_run
    history = campaign.read_history(product, "A-nivel", rule="label-free")
    assert history and {h["selectionRule"] for h in history} == {"label-free"}
    root = product / "Models" / "Balladeer" / campaign.LABEL_FREE_MODELS
    checked = 0
    for path in sorted(root.rglob("phase2__*.pt")):
        meta = _sidecar(path)
        mine = [h for h in history if h["model"] == f"{meta['name']}:phase2"
                and h["transfer"] == meta["transfer"] and h["fold"] == meta["rotation"]
                and h["seed"] == meta["seed"]]
        chosen = meta["selection"]
        assert chosen["rule"] == "label-free"
        assert chosen["epochRule"] == two_phase.LABEL_FREE_EPOCH_RULE
        assert len(mine) == chosen["epochs"] == chosen["epoch"] + 1 == meta["epoch"] + 1
        assert meta["hyper"]["epochs2"] == chosen["epochs"]
        assert {h["labelFreeEpoch"] for h in mine} == {chosen["epoch"]}
        assert not any("val_balanced_accuracy" in h for h in mine)   # a refit validates nothing
        primary = _sidecar(path.parents[4] / meta["transfer"].replace("->", "-to-")
                           / f"rotation{meta['rotation']}" / f"seed{meta['seed']}"
                           / path.name)
        assert chosen["trial"] != primary["selection"]["trial"]
        checked += 1
    assert checked == 2 * 2 * 4            # two strategies, two directions, four rotations


def test_the_label_free_histories_are_written_apart_from_the_strategies(label_free_run):
    """The label-free models' per-epoch records land in their own files, in the primary
    layout, read by the convergence output and the K3 bound control, never by the
    primary history the statistics step reads."""
    cfg, product, read_by_bounds, figures = label_free_run
    res = product / "Results"
    runs = res / "A-nivel" / "runs"
    assert (runs / "history_label_free.jsonl").exists()
    primary = campaign.read_history(product, "A-nivel")
    assert primary and all("selectionRule" not in h for h in primary)
    free = json.loads((res / "history_A-nivel_label_free.json").read_text())
    assert free == campaign.read_history(product, "A-nivel", rule="label-free")
    assert set(free[0]) - {"selectionRule", "labelFreeEpoch"} == set(
        next(h for h in primary if h["model"] == free[0]["model"]))
    assert (res / "A-nivel" / "figures" / "convergence_label_free.pdf").exists()
    # the label-free figure marks the label-free epoch; the primary one, max validation BA
    assert ("convergence_label_free", "label_free_score") in figures
    assert ("convergence", "val_balanced_accuracy") in figures
    summary = json.loads((res / "K3-ablacion" / "summary.json").read_text())
    bounds = {r["term"]: r for r in summary["term_bounds"]}
    assert bounds
    mil = [h for h in primary + free if h["model"] == "MIL-CREDA:phase2"]
    assert [h for h in read_by_bounds if h.get("selectionRule") == "label-free"
            and h["model"] == "MIL-CREDA:phase2"]
    for term, row in bounds.items():
        assert float(row["observed_max"]) == pytest.approx(
            max(h[f"{term}_max"] for h in mil), rel=1e-6)
        assert float(row["observed_min"]) == pytest.approx(
            min(h[f"{term}_min"] for h in mil), rel=1e-6)


def test_the_test_train_ratio_counts_the_refit_training_subjects():
    """Plan v50, Sec. 4.5: the model a rotation evaluates trains on every non-test fold."""
    folds = {f"UB{i:04d}": i % 4 for i in range(16)}
    labels = {s: ("yes", "no")[i % 2] for i, s in enumerate(folds)}
    holders = {"L1": {s for i, s in enumerate(folds) if i % 4 < 2},
               "L6": {s for i, s in enumerate(folds) if i % 4 >= 2}}
    rotations = partition.rotations(folds)
    a, b = (campaign.assignment_mod.assign(campaign.assignment_mod.Transfer("level", x, y),
                                           holders, folds, labels, rotations,
                                           partition_sha256="x" * 64)
            for x, y in (("L1", "L6"), ("L6", "L1")))
    expected = []
    for r in range(4):
        test = sum(1 for x in (a, b) for s in x.subjects("target") if folds[s] == r)
        expected.append(test / sum(1 for s in folds if folds[s] != r) / 2)
    assert campaign.test_train_ratio(a, b) == pytest.approx(sum(expected) / 4)


# ------------------------------------------------------------------ the v56 comparator set

#: Plan v56, Sec. 7.1: the eight comparators and MIL-CREDA (HASD left the comparison).
V54_STRATEGIES = {"AdaptMIL", "UDA-DDA", "CS+CCS", "CASUAL", "CCA-LSC", "COT",
                  "CLUST — variante source-free", "CLUDA", "MIL-CREDA"}


def test_the_comparison_is_the_v54_set_and_hasd_is_an_unknown_strategy():
    """Plan v56, Secs. 7.1, 7.3, 7.5: nine strategies, three phase-1 groups, CLUDA in the
    common group, HASD in no strategy list and refused as unknown."""
    assert set(model_store.STRATEGIES) == V54_STRATEGIES and len(model_store.STRATEGIES) == 9
    assert model_store.STRATEGIES[-1] == "MIL-CREDA"
    assert model_store.GROUPS == ("common", "AdaptMIL", "MIL-CREDA")
    assert set(search.PHASE_ONE_SPACES) == set(model_store.GROUPS)
    assert set(search.PHASE_ONE_GROUP_OF) == V54_STRATEGIES
    assert search.PHASE_ONE_GROUP_OF["CLUDA"] == "common"
    assert "CLUDA" in mil_wrapper.COMPARATORS and len(mil_wrapper.COMPARATORS) == 7
    assert controls.MATCHED_GROUPS == ("AdaptMIL", "MIL-CREDA")
    assert set(results.SOURCE_ONLY) == {"source-only:common", "source-only:AdaptMIL",
                                        "source-only:MIL-CREDA"}
    assert results.own_control("CLUDA") == "source-only:common"
    for listed in (model_store.STRATEGIES, model_store.GROUPS, search.PHASE_ONE_SPACES,
                   search.PHASE_ONE_GROUP_OF, mil_wrapper.COMPARATORS,
                   controls.MATCHED_GROUPS, results.METHOD_ROWS):
        assert "HASD" not in listed
    with pytest.raises(campaign.CampaignError, match="strategies are among"):
        campaign.check_config({**BASE, "strategies": ["MIL-CREDA", "HASD"]})
    assert importlib.util.find_spec("MIL_CREDA_EEG.comparators.hasd") is None


def test_cluda_registers_through_the_campaign_with_its_own_axis(registries, monkeypatch):
    """Plan v56, Sec. 7.7: CLUDA searches the generic contrastive weight and its own
    temperature axis, log-uniform in [0.01, 1]."""
    from MIL_CREDA_EEG.comparators.cluda import training as cluda
    assert search.OWN_AXES["CLUDA"] == (FloatDistribution(0.01, 1.0, log=True),)
    assert search.OWN_AXES["CLUDA"] == (cluda.TEMPERATURE,)
    monkeypatch.delitem(search.PHASE_TWO_TRAINERS, "CLUDA", raising=False)
    campaign.register(["MIL-CREDA", "CLUDA"])
    assert search.PHASE_TWO_TRAINERS["CLUDA"] is cluda.train_phase_two
    assert search.phase_two_space("CLUDA") == {**search.SHARED_PHASE_TWO,
                                               "contrastiveWeight": search.LOSS_WEIGHT,
                                               "temperature": cluda.TEMPERATURE}


def test_cluda_runs_through_the_campaign_and_loads_back_whole(tmp_path, registries):
    """Plan v56, Secs. 7.3, 7.4: CLUDA's phase 2 starts from the common group's checkpoint,
    its final models are stored, and loading one restores its projector and checks the
    comparator's source digest."""
    from MIL_CREDA_EEG.comparators.cluda.model import CLUDAModel
    cfg, product, outcome = _rehearsal(tmp_path, strategies=("MIL-CREDA", "CLUDA"))
    records = results.read_predictions(product / "Results", "A-nivel")
    assert {r["strategy"] for r in records} == {"MIL-CREDA", "CLUDA", "source-only:common",
                                                "source-only:MIL-CREDA", "upper-bound"}
    assert outcome["final"]["trained_phase2"] == 2 * 4 * 2
    path = (product / "Models" / "Balladeer" / "L1-to-L6" / "rotation0" / "seed1"
            / "phase2__CLUDA.pt")
    model, meta, extra = mechanisms.load_phase_two(path, partition_path=cfg["partitionPath"])
    assert isinstance(model, CLUDAModel) and meta["name"] == "CLUDA" and extra == []
    assert meta["phaseOneCheckpoint"]["name"] == "common"
    stored = torch.load(path, weights_only=True)
    assert all(torch.equal(v, stored[k]) for k, v in model.state_dict().items()
               if k.startswith("projector."))


def test_the_final_step_refuses_a_missing_phase_one_checkpoint(level_run, registries,
                                                               tmp_path):
    """Secs. 4.5, 7.3: the final step refits phase 2 from the stored phase-1 checkpoint its
    study started from; a missing one is refused, never trained again."""
    cfg, product, _ = level_run
    copy = tmp_path / "product"
    shutil.copytree(product, copy)
    gone = (copy / "Models" / "Balladeer" / "L1-to-L6" / "rotation0"
            / f"seed{search.PHASE_ONE_SEED}" / "phase1__common.pt")
    gone.unlink()
    gone.with_suffix(".json").unlink()
    before = results.read_predictions(copy / "Results", "A-nivel")
    with pytest.raises(campaign.CampaignError, match="phase-1 checkpoint") as refused:
        campaign.run_final(campaign.check_config(cfg), copy)
    assert "search step" in str(refused.value)
    assert not gone.exists() and not gone.with_suffix(".json").exists()
    assert results.read_predictions(copy / "Results", "A-nivel") == before


def test_the_final_step_refuses_before_any_transfer_writes(level_run, registries, tmp_path):
    """Secs. 4.5, 7.3: a missing phase-1 checkpoint of a later transfer is refused before
    an earlier transfer of the same step trains or records anything."""
    cfg, product, _ = level_run
    copy = tmp_path / "product"
    shutil.copytree(product, copy)
    first, *later = [t.name.replace("->", "-to-") for t in campaign.transfers_of(
        campaign.check_config(cfg))]
    assert later, "the rehearsal needs two transfers"
    models = copy / "Models" / "Balladeer"
    for stale in (models / first).rglob("phase2__*"):
        stale.unlink()
    for rule in ("primary", "label-free"):
        stale = results._predictions_path(copy / "Results", "A-nivel", rule)
        if stale.exists():
            stale.unlink()
    before = {e: results.read_predictions(copy / "Results", e) for e in ("A-nivel",)}
    assert before == {"A-nivel": []}
    gone = models / later[-1] / "rotation0" / f"seed{search.PHASE_ONE_SEED}" / "phase1__common.pt"
    gone.unlink()
    gone.with_suffix(".json").unlink()
    with pytest.raises(campaign.CampaignError, match="phase-1 checkpoint"):
        campaign.run_final(campaign.check_config(cfg), copy)
    assert not list((models / first).rglob("phase2__*"))
    assert {e: results.read_predictions(copy / "Results", e) for e in before} == before


def test_a_phase_two_study_from_another_checkpoint_names_the_study_to_remove(level_run):
    """Sec. 4.5: the refusal's remedy says the stale phase-2 study must be removed, since
    the search step would refuse to continue it under another identity."""
    cfg, product, _ = level_run
    cfg = campaign.check_config(cfg)
    transfer = campaign.transfers_of(cfg)[0]
    data = search.search_data(transfer, campaign.load_bags(cfg))
    state, meta = campaign._phase_one_record(product, data, "common", 0, cfg)
    other = (state, {**meta, "state_sha256": "0" * 64})
    with pytest.raises(campaign.CampaignError, match="remove") as refused:
        campaign._check_started_from(product, data, "COT", 0, other, cfg)
    assert str(search.study_path(product, transfer, 2, "COT", 0)) in str(refused.value)


def _k2_diagnostics(root: Path) -> dict[str, dict]:
    rows = _csv(root / "K2-emparejamiento" / "tables" / "diagnostics.csv")
    assert all(list(r) == list(mechanisms.K2_DIAGNOSTICS_COLUMNS) for r in rows)
    by_transfer = {r["Transferencia"]: r for r in rows}
    assert len(by_transfer) == len(rows), "one row per K2 transfer"
    return by_transfer


def _gate(row: dict) -> bool:
    return mechanisms.k2_has_room(float(row["Mediana de la razón de entropía"]),
                                  float(row["Mediana de la razón de margen"]))


def test_the_rehearsal_writes_the_k2_mechanism_diagnostics(level_run):
    """v56 Sec. 6.2: per K2 transfer, the diagnostics at the start of phase 2, the push
    with and without the local term, and the reading gate on the criterion rows."""
    cfg, product, outcome = level_run
    root = product / "Results"
    diagnostics = _k2_diagnostics(root)
    separation = _csv(root / "K2-emparejamiento" / "tables" / "separation.csv")
    assert set(diagnostics) == {r["transfer"] for r in separation
                                if r["model"] == "MIL-CREDA"}
    for name, row in diagnostics.items():
        assert int(row["Sujetos de la sonda"]) > 0, name
        assert 0.0 <= float(row["Mediana de la razón de entropía"]) <= 1.0, name
        assert 0.0 <= float(row["Fracción con la propia bolsa fuente como la más cercana"]) <= 1.0
        assert float(row["Mediana de la razón de margen"]) >= 0.0, name
        # every rehearsal transfer has the global-term-only variant: both pushes measured
        float(row["Empuje con término local"]), float(row["Empuje sin término local"])
        assert row["Condición de lectura"] == ("leída" if _gate(row)
                                               else mechanisms.K2_INCONCLUSIVE), name
    k2 = json.loads((root / "significance_K2-emparejamiento.json").read_text())
    declared = {t.name for t in mechanisms.K2_DECLARED.values()}
    for r in k2:
        if r["transfer"] in diagnostics:
            assert r["inconclusive_no_room"] is (not _gate(diagnostics[r["transfer"]]))
            assert (r.get("verdict") == mechanisms.K2_INCONCLUSIVE) is r["inconclusive_no_room"]
        else:
            assert r["transfer"] in declared and "inconclusive_no_room" not in r
    assert any(r["transfer"] in diagnostics for r in k2)
    summary = json.loads((root / "K2-emparejamiento" / "summary.json").read_text())
    assert summary["revision"].endswith("balladeer-v57.md")
    assert set(summary["diagnostics"]) == set(diagnostics)
    # v56 Sec. 6.3: both terms vs global only is inconclusive only when NEITHER direction
    # of the core pair meets the gate
    k3 = json.loads((root / "significance_K3-ablacion.json").read_text())
    rows = [r for r in k3 if "both terms vs global term only" in r["comparison"]]
    assert len(rows) > 1      # the primary row and its secondary metrics
    expected = not any(_gate(diagnostics[t]) for t in ("L1->L6", "L6->L1"))
    assert all(r["inconclusive_no_room"] is expected for r in rows)
    assert all((r.get("verdict") == mechanisms.K2_INCONCLUSIVE) is expected for r in rows)
    assert not any("inconclusive_no_room" in r for r in k3
                   if "both terms vs global term only" not in r["comparison"])


def test_a_missing_phase_one_checkpoint_leaves_the_k2_diagnostics_not_computed(
        level_run, registries, tmp_path):
    """v56 Sec. 6.2: a diagnostic without its stored model is recorded as not computed,
    with the reason, never fabricated; the gate is then not applied to that transfer."""
    cfg, product, _ = level_run
    copy = tmp_path / "product"
    shutil.copytree(product, copy)
    for gone in (copy / "Models" / "Balladeer" / "L6-to-L1").rglob("phase1__MIL-CREDA.*"):
        gone.unlink()
    out = campaign.run_mechanisms(campaign.check_config(cfg), copy)
    diagnostics = _k2_diagnostics(copy / "Results")
    missing = diagnostics["L6->L1"]
    assert missing["Condición de lectura"].startswith(mechanisms.K2_NOT_COMPUTED)
    assert "phase-1" in missing["Condición de lectura"]
    assert missing["Mediana de la razón de entropía"] == ""
    assert diagnostics["L1->L6"]["Condición de lectura"] in ("leída", mechanisms.K2_INCONCLUSIVE)
    assert any("L6->L1" in o and "phase-1" in o for o in out["owed"])
    # Sec. 6.3: with one direction not computed, the flag reads the computed one only
    k3 = json.loads((copy / "Results" / "significance_K3-ablacion.json").read_text())
    rows = [r for r in k3 if "both terms vs global term only" in r["comparison"]]
    assert rows and all(r["inconclusive_no_room"] is (not _gate(diagnostics["L1->L6"])) for r in rows)
