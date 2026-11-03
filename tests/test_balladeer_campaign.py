"""Experiments plan v40: the campaign entry point, rehearsed on a SYNTHETIC set on CPU.

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

from MIL_CREDA_EEG import campaign, controls, model_store, results, search
from MIL_CREDA_EEG import statistics as st

BASE = {"schedule": {"epochs1": 1, "patience1": 1, "epochs2": 1, "bagsPerStep": 4},
        "strategies": ["MIL-CREDA"], "transfers": ["L1->L11"],
        "seedReduction": "mean-probability"}


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


def _rehearsal(tmp, conditions=("L1", "L11"), arm="level",
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
    assert campaign.STEPS == ("search", "final", "statistics", "mechanisms", "results")
    assert list(campaign.RUNNERS) == list(campaign.STEPS)
    with pytest.raises(campaign.CampaignError, match="step"):
        campaign.run("launch", {**BASE}, "/nonexistent")


def test_required_inputs_have_no_default():
    for key in ("schedule", "strategies", "transfers", "seedReduction"):
        with pytest.raises((campaign.CampaignError, ValueError)):
            campaign.check_config({k: v for k, v in BASE.items() if k != key})
    with pytest.raises(ValueError, match="schedule"):
        campaign.check_config({**BASE, "schedule": {"epochs1": 1}})
    with pytest.raises(campaign.CampaignError, match="MIL-CREDA"):
        campaign.check_config({**BASE, "strategies": ["COT"]})
    with pytest.raises(campaign.CampaignError, match="seedReduction"):
        campaign.check_config({**BASE, "seedReduction": "majority"})
    with pytest.raises(campaign.CampaignError, match="report seed"):
        campaign.check_config({**BASE, "reportSeeds": [0, 1]})
    cfg = campaign.check_config(BASE)
    assert cfg["nTrials"] == search.N_TRIALS == 20
    assert cfg["reportSeeds"] == list(search.REPORT_SEEDS)
    assert cfg["bootstrapResamples"] == st.BOOTSTRAP_RESAMPLES
    with pytest.raises(campaign.CampaignError, match="k1"):
        campaign.check_mechanism_inputs(cfg)
    for missing in ("commonAttention", "bands", "scope"):
        k1 = {"commonAttention": "COT", "bands": [[4, 8], [13, 30]], "scope": "pooled"}
        k1.pop(missing)
        with pytest.raises(campaign.CampaignError, match=missing):
            campaign.check_mechanism_inputs({**cfg, "k1": k1})
    with pytest.raises(campaign.CampaignError, match="non-MIL"):
        campaign.check_mechanism_inputs({**cfg, "k1": {
            "commonAttention": "AdaptMIL", "bands": [[4, 8], [13, 30]], "scope": "pooled"}})


def test_a_strategy_with_an_open_plan_decision_refuses_without_its_resolution():
    assert campaign.PENDING_DECISIONS == {
        "CLUST — variante source-free": ("clust-ema-range",),
        "CASUAL": ("casual-vat-radius-range",),
        "UDA-DDA": ("uda-dda-conditional-gating", "uda-dda-threshold-ordering"),
        "CS+CCS": ("cs-ccs-prediction-kernel-width",)}
    for strategy, items in campaign.PENDING_DECISIONS.items():
        with pytest.raises(campaign.CampaignError, match=items[0]):
            campaign.check_config({**BASE, "strategies": ["MIL-CREDA", strategy]})
    partial = {"uda-dda-conditional-gating": "plan v41 Sec. 7.4"}
    with pytest.raises(campaign.CampaignError, match="threshold-ordering"):
        campaign.check_config({**BASE, "strategies": ["MIL-CREDA", "UDA-DDA"],
                               "pendingDecisions": partial})
    with pytest.raises(campaign.CampaignError, match="generic"):
        campaign.check_config({**BASE, "strategies": ["MIL-CREDA", "CASUAL"],
                               "pendingDecisions": {"casual-vat-radius-range": "[0.1, 5]"}})
    ok = campaign.check_config({**BASE, "strategies": ["MIL-CREDA", "CASUAL"],
                                "pendingDecisions": {"casual-vat-radius-range": {
                                    "distribution": "LOSS_WEIGHT"}}})
    assert ok["strategies"] == ["MIL-CREDA", "CASUAL"]
    for free in ("AdaptMIL", "HASD", "COT", "CCA-LSC"):
        campaign.check_config({**BASE, "strategies": ["MIL-CREDA", free]})


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


def test_an_incomplete_study_is_refused_not_extended(tmp_path):
    missing = tmp_path / "absent.db"
    with pytest.raises(campaign.CampaignError, match="search step"):
        campaign._study(missing, 2)
    assert not missing.exists()
    path = tmp_path / "s.db"
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{path}")
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
    with pytest.raises(campaign.CampaignError, match="1 completed trials of 2"):
        campaign._study(path, 2)
    assert len(campaign._study(path, 1).trials) == 1


def test_the_final_models_are_trained_at_the_report_seeds_from_the_same_rotation(level_run):
    cfg, product, outcome = level_run
    models = product / "Models" / "Balladeer"
    for transfer in ("L1-to-L11", "L11-to-L1"):
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
    for transfer in ("L1->L11", "L11->L1"):
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
    assert again == {"trained_phase1": 0, "trained_phase2": 0, "predictions": 0}
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
    rows = json.loads((root / "significance_A-nivel.json").read_text())
    rivals = {r["comparison"].split(" vs ")[1] for r in rows
              if r["test"] == st.TEST_BOOTSTRAP}
    assert rivals == {"COT", "HASD", "Source only, common attention", "Source only, HASD",
                      "Source only, MIL-CREDA"}
    assert all("holm" in r["correction"] for r in rows if r["test"] == st.TEST_BOOTSTRAP)
    k1 = json.loads((root / "significance_K1-atencion.json").read_text())
    assert any("event_window" in (r.get("not_computed") or "") for r in k1)
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
