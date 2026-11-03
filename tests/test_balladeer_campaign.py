"""Experiments plan v43: the campaign entry point, rehearsed on a SYNTHETIC set on CPU.

The synthetic set is seeded noise with a weak class signal; no number it yields is a
measured result.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import optuna
import pytest

from MIL_CREDA_EEG import aggregation_variants, campaign, controls, mechanisms, model_store
from MIL_CREDA_EEG import results
from MIL_CREDA_EEG import search
from MIL_CREDA_EEG import statistics as st

BASE = {"schedule": {"epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 4},
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
               strategies=("MIL-CREDA", "COT", "HASD")):
    cfg = campaign.write_synthetic_set(tmp / "synthetic", seed=0, subjects=24,
                                       conditions=conditions, arm=arm)
    cfg.update({"strategies": list(strategies), "nTrials": 1, "reportSeeds": [1],
                "bootstrapResamples": 200,
                "schedule": {"epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 4}})
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
    assert campaign.REVISION.endswith("v43.md")
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
            for seed in (0, 1):
                for name in ("phase1__common", "phase1__HASD", "phase1__MIL-CREDA"):
                    assert (models / transfer / f"rotation{r}" / f"seed{seed}" /
                            f"{name}.pt").exists()
            assert (models / transfer / f"rotation{r}" / "seed1" / "phase2__COT.pt").exists()
            assert not (models / transfer / f"rotation{r}" / "seed0" / "phase2__COT.pt").exists()
    records = results.read_predictions(product / "Results", "A-nivel")
    assert {r["seed"] for r in records} == {1}
    strategies = {r["strategy"] for r in records}
    assert strategies == {"MIL-CREDA", "COT", "HASD", "source-only:common",
                          "source-only:HASD", "source-only:MIL-CREDA", "upper-bound"}
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
    assert table.count(controls.NA_RECORD) == 4 * 3


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
    assert rivals == {"COT", "HASD", "Source only, common attention", "Source only, HASD",
                      "Source only, MIL-CREDA"}
    assert all("holm" in r["correction"] for r in rows if r["test"] == st.TEST_BOOTSTRAP)
    k1 = json.loads((root / "significance_K1-atencion.json").read_text())
    assert any("event_window" in (r.get("not_computed") or "") for r in k1)
    criterion = [r for r in k1 if r.get("criterion") == mechanisms.K1_CRITERION]
    assert [r["comparison"] for r in criterion] == [
        "K1-atencion:level:marked vs clean, MIL-CREDA",
        "K1-atencion:level:marked vs clean, MIL-CREDA vs MIL-CREDA [own without consensus]",
        "K1-atencion:level:event vs rest, MIL-CREDA"]
    assert criterion[1]["n"] > 0 and criterion[1]["p_value"] is not None
    references = {r["comparison"].split(", ", 1)[1] for r in k1
                  if r.get("criterion", "").startswith("none (reference")}
    assert references == {"MIL-CREDA [published attention]", "MIL-CREDA [gated attention]",
                          "common attention [COT]", "HASD"}
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
    assert {"A-nivel/figures/convergence.pdf", "A-nivel/figures/optimization_history.pdf",
            "A-nivel/figures/critical_difference.pdf"} <= written
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
    assert all(r["correction"] == "none" for r in rows if r["test"] != st.TEST_BOOTSTRAP)
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


def test_phase_two_is_selected_per_test_rotation_and_phase_one_once(level_run):
    cfg, product, _ = level_run
    models = product / "Models" / "Balladeer"
    bags = campaign.load_bags(campaign.check_config(cfg))
    for transfer in campaign.transfers_of(campaign.check_config(cfg)):
        data = search.search_data(transfer, bags)
        # phase 1: one configuration over the four rotations, the study's best (Sec. 7.3)
        one = campaign._study(search.study_path(product, transfer, 1, "MIL-CREDA"),
                              cfg["nTrials"], data.partition_sha256)
        for r in range(4):
            meta = _sidecar(models / model_store.transfer_slug(transfer) / f"rotation{r}"
                            / "seed1" / "phase1__MIL-CREDA.pt")
            assert {k: meta["hyper"][k] for k in one.best_params} == one.best_params
        for strategy in ("MIL-CREDA", "COT"):
            study = campaign._study(search.study_path(product, transfer, 2, strategy),
                                    cfg["nTrials"], data.partition_sha256)
            chosen = search.select_per_rotation(study, data.folds)
            for r in range(4):
                meta = _sidecar(models / model_store.transfer_slug(transfer) / f"rotation{r}"
                                / "seed1" / f"phase2__{strategy}.pt")
                assert meta["selection"]["trial"] == chosen[r]["trial"]
                assert meta["selection"]["selectionRotations"] == \
                    search.selection_rotations(data.folds, r)
                assert meta["selection"]["excludedRotation"] == chosen[r]["excludedRotation"]
                assert {k: meta["hyper"][k] for k in chosen[r]["params"]} == \
                    chosen[r]["params"]
    detail = _csv(product / "Results" / "A-nivel" / "tables" / "selected_configurations.csv")
    assert {(d["strategy"], d["transfer"], d["rule"], d["rotation"]) for d in detail} == {
        (s, t, rule, str(r)) for s in cfg["strategies"] for t in cfg["transfers"]
        for rule in ("primary", "label-free") for r in range(4)}
    k3 = _csv(product / "Results" / "K3-ablacion" / "tables" / "variant_hyperparameters.csv")
    assert {d["rotation"] for d in k3} == {"0", "1", "2", "3"}


def test_the_label_free_rule_is_measured_on_the_test_fold(tmp_path, registries, monkeypatch):
    real = search.select_per_rotation

    def other_trial(study, rotations, rule="primary"):
        chosen = real(study, rotations, rule)
        if rule != "label-free":
            return chosen
        primary = real(study, rotations, "primary")
        for r in (1, 2, 3):         # rotation 0 coincides, the other three differ
            n = 1 - primary[r]["trial"]
            chosen[r] = {**chosen[r], "trial": n, "params": dict(study.trials[n].params)}
        chosen[0] = {**chosen[0], "trial": primary[0]["trial"],
                     "params": dict(primary[0]["params"])}
        return chosen

    monkeypatch.setattr(search, "select_per_rotation", other_trial)
    cfg = campaign.write_synthetic_set(tmp_path / "synthetic", seed=0, subjects=24)
    cfg.update({"strategies": ["MIL-CREDA"], "nTrials": 2, "bootstrapResamples": 200,
                "schedule": {"epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 4}})
    product = tmp_path / "product"
    product.mkdir()
    outcome = {s: campaign.run(s, cfg, product) for s in ("search", "final", "results")}
    assert outcome["final"]["final"]["trained_label_free"] == 3 * 2
    root = product / "Models" / "Balladeer" / campaign.LABEL_FREE_MODELS
    for slug in ("L1-to-L6", "L6-to-L1"):
        assert not (root / slug / "rotation0").exists()
        for r in (1, 2, 3):
            meta = _sidecar(root / slug / f"rotation{r}" / "seed1" / "phase2__MIL-CREDA.pt")
            assert meta["selection"]["rule"] == "label-free"
    res = product / "Results"
    primary = results.read_predictions(res, "A-nivel")
    free = results.read_predictions(res, "A-nivel", rule="label-free")
    key = lambda x: (x["transfer"], x["rotation"], x["subject"])  # noqa: E731
    p0 = sorted((key(x), x["probability"]) for x in primary
                if x["strategy"] == "MIL-CREDA" and x["rotation"] == 0)
    f0 = sorted((key(x), x["probability"]) for x in free if x["rotation"] == 0)
    assert p0 == f0 and p0          # where the rules coincide, the primary model is reused
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
