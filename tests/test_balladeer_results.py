"""Experiments plan v42, Secs. 4.5, 5.1-5.4 and 6: the Results writers.

Every record here is synthetic (seeded), never a measured result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from MIL_CREDA_EEG import controls, model_store, ps_figures, results
from MIL_CREDA_EEG import statistics as st


def _records(strategy, transfer, *, n=12, seeds=(1,), shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in seeds:
        for i in range(n):
            label = i % 2
            out.append({"strategy": strategy, "transfer": transfer, "direction": transfer,
                        "rotation": i % 4, "seed": s, "subject": f"{transfer}:{i}",
                        "label": label, "probability": float(np.clip(
                            0.5 + (label - 0.5) * 0.4 + shift + rng.normal(0, 0.2), 0, 1))})
    return out


def _row(**extra):
    return st.significance_row(comparison="c", test="t", statistic=1.0, p_value=0.5,
                               effect_size=0.1, effect_size_name="e", ci_low=0.0, ci_high=0.2,
                               n=10, hypothesis="A-nivel", experiment="A-nivel", **extra)


def test_every_writer_lands_in_the_dashboard_layout(tmp_path):
    root = tmp_path / "Results"
    paths = [results.write_significance(root, "A-nivel", [_row()]),
             *results.write_table(root, "A-nivel", "comparison", [{"a": 1, "b": "x"}]),
             results.write_summary(root, "A-nivel", {"n": 3}, synthetic=True),
             results.write_history(root, "A-nivel", [{"epoch": 0, "loss": 1.0, "seed": 1,
                                                      "fold": 0, "model": "m"}]),
             results.append_predictions(root, "A-nivel", [{"subject": "s"}])]
    rel = sorted(p.relative_to(root).as_posix() for p in paths)
    assert rel == ["A-nivel/runs/predictions.jsonl", "A-nivel/summary.json",
                   "A-nivel/tables/comparison.csv", "A-nivel/tables/comparison.md",
                   "history_A-nivel.json", "significance_A-nivel.json"]
    summary = json.loads((root / "A-nivel" / "summary.json").read_text())
    assert summary["synthetic"] is True and summary["revision"].endswith("v42.md")
    with pytest.raises(results.ResultsError):
        results.write_table(root, "Z-unknown", "t", [])


def test_a_significance_file_missing_a_dashboard_key_is_refused(tmp_path):
    row = _row()
    for key in (*st.SIGNIFICANCE_KEYS, *st.OPTIONAL_KEYS):
        broken = {k: v for k, v in row.items() if k != key}
        with pytest.raises(results.ResultsError, match=key):
            results.write_significance(tmp_path, "A-nivel", [broken])
    with pytest.raises(results.ResultsError, match="belongs"):
        results.write_significance(tmp_path, "B-tarea", [row])


def test_json_never_carries_nan(tmp_path):
    path = results.write_json(tmp_path / "x.json", {"a": float("nan"), "b": [np.float64(
        np.inf), np.int64(3)], "c": np.array([1.0, np.nan])})
    assert json.loads(path.read_text()) == {"a": None, "b": [None, 3], "c": [1.0, None]}
    assert "NaN" not in path.read_text()


def test_the_comparison_table_has_every_method_row_and_every_transfer_column():
    records = _records("MIL-CREDA", "L1->L11") + _records("COT", "L1->L11", seed=1)
    table = results.comparison_table(records, "level", ["L1->L11", "L11->L1"], None)
    assert [r["method"] for r in table] == list(results.METHOD_ROWS.values())
    assert list(table[0]) == ["method", *results.ARM_TRANSFERS["level"], "average rank (core)"]
    mil = next(r for r in table if r["method"] == "MIL-CREDA")
    # Sec. 4.6: one report seed, one pooled value per transfer, no reduction between seeds
    assert mil["L1->L11"] == f"{st.pooled(records[:12])['balanced_accuracy']:.3f}"
    assert mil["L1->L6"] == ""
    with pytest.raises(st.StatisticsError, match="one report seed"):
        results.comparison_table(_records("MIL-CREDA", "L1->L11", seeds=(1, 2)), "level",
                                 ["L1->L11", "L11->L1"], None)


def test_the_device_arm_controls_print_the_na_record():
    table = results.comparison_table(_records("MIL-CREDA", "L6->Robots"), "device",
                                     ["L6->Robots", "Robots->L6"], None)
    for row in table:
        if row["method"].startswith("Source only"):
            assert set(row.values()) - {row["method"]} == {controls.NA_RECORD}
    secondary = results.secondary_table({}, "device")
    assert all(r["kappa"] == controls.NA_RECORD for r in secondary
               if r["method"].startswith("Source only"))


def test_the_rank_column_ranks_only_the_nine_adapted_strategies_on_core_transfers():
    records = []
    for k, s in enumerate(("MIL-CREDA", "COT", "HASD")):
        for t in ("L1->L11", "L11->L1", "L1->L6"):
            records += _records(s, t, shift=0.05 * k, seed=k)
    records += _records("source-only:common", "L1->L11") + _records("upper-bound", "L1->L11")
    scores = results.ranking_scores(records, ["L1->L11", "L11->L1"])
    assert sorted(scores) == ["COT", "HASD", "MIL-CREDA"]
    assert all(len(v) == 2 for v in scores.values())
    ranking = st.friedman_nemenyi(scores)
    table = results.comparison_table(records, "level", ["L1->L11", "L11->L1"], ranking)
    ranks = {r["method"]: r["average rank (core)"] for r in table}
    assert ranks["Source only, common attention"] == "" and ranks[
        "Target-supervised upper bound"] == ""
    assert ranks["MIL-CREDA"] == f"{ranking['average_rank']['MIL-CREDA']:.2f}"


def test_the_selection_table_reads_both_rules_from_the_same_trials():
    study = optuna.create_study(direction="maximize")
    for value, free in ((0.6, 0.1), (0.8, 0.05), (0.7, 0.3)):
        trial = study.ask()
        trial.suggest_float("lr2", 1e-5, 1e-2, log=True)
        trial.set_user_attr("label_free_score", free)
        study.tell(trial, value)
    row = results.selection_rows(study, "MIL-CREDA", "L1->L11", test_primary=0.7)
    assert row["primary trial"] == 1 and row["label-free trial"] == 2
    assert row["coincide"] is False and row["test BA, label-free rule"] is None
    assert row["difference"] is None


def _spy(monkeypatch):
    calls = []
    real_style, real_save = ps_figures.apply_style, ps_figures.save_figure
    monkeypatch.setattr(ps_figures, "apply_style",
                        lambda **kw: calls.append(("style", kw)) or real_style(**kw))
    monkeypatch.setattr(ps_figures, "save_figure",
                        lambda fig, path, **kw: calls.append(("save", Path(path), kw))
                        or real_save(fig, path, **kw))
    return calls


def _pdf_has_text(path):
    return b"/FontFile2" in path.read_bytes()       # TrueType embedded (pdf.fonttype 42)


def test_every_figure_goes_through_the_shipped_helper_with_a_png_preview(tmp_path, monkeypatch):
    kit = Path(results.__file__).with_name("ps_figures.py")
    assert kit.exists() and "def save_figure" in kit.read_text()
    calls = _spy(monkeypatch)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(0))
    study.optimize(lambda t: t.suggest_float("lr2", 1e-5, 1e-2, log=True)
                   - t.suggest_float("wd2", 1e-7, 1e-2, log=True), n_trials=6)
    paths = [
        results.optimization_history_figure(tmp_path, "A-nivel", {"MIL-CREDA": study}),
        results.contour_figure(tmp_path, "A-nivel", study, ["lr2", "wd2"]),
        results.critical_difference_figure(tmp_path, "A-nivel", st.friedman_nemenyi(
            {"MIL-CREDA": [0.8, 0.7], "COT": [0.6, 0.65], "HASD": [0.7, 0.6]})),
        results.ladder_figure(tmp_path, [{"strategy": "MIL-CREDA", "separability": 0.6,
                                          "separability_sd": 0.02, "gain": 0.05}], 0.95)]
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "A-nivel/figures/optimization_history.pdf", "A-nivel/figures/contour_mil_creda.pdf",
        "A-nivel/figures/critical_difference.pdf", "figures/domain_shift_ladder.pdf"]
    styles = [c for c in calls if c[0] == "style"]
    saves = [c for c in calls if c[0] == "save"]
    assert len(styles) == len(saves) == 4
    assert all(c[1]["body_pt"] == 10 and c[1]["printed_width_fraction"] == 0.8 for c in styles)
    assert all(c[2] == {"png_preview": True} for c in saves)
    for p in paths:
        assert p.exists() and p.with_suffix(".png").exists() and _pdf_has_text(p)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title("no")
    with pytest.raises(ps_figures.FigureExportError):
        ps_figures.save_figure(fig, tmp_path / "t.pdf")


def test_the_convergence_figure_reads_the_core_transfers_and_marks_the_selected_epoch(
        tmp_path, monkeypatch):
    history = []
    for transfer in ("L1->L11", "L1->L6"):
        for model in ("MIL-CREDA:phase1", "MIL-CREDA:phase2"):
            for fold in range(4):
                for e in range(4):
                    history.append({"epoch": e, "seed": 1, "fold": fold, "model": model,
                                    "transfer": transfer, "loss": 1.0 / (e + 1),
                                    "val_balanced_accuracy": [0.5, 0.7, 0.9, 0.9][e]
                                    if fold < 2 else [0.5, 0.9, 0.6, 0.6][e],
                                    "global": 0.5, "global_bound_low": 0.0,
                                    "global_bound_high": 1.0})
    assert results.selected_epoch([r for r in history if r["model"].endswith("phase2")
                                   and r["transfer"] == "L1->L11"]) == 1.5
    seen = []
    real = results._band
    monkeypatch.setattr(results, "_band", lambda ax, xs, rows, label, **k: (
        seen.append((label, np.asarray(rows).shape)), real(ax, xs, rows, label, **k)))
    path = results.convergence_figure(tmp_path, "A-nivel", history, ["L1->L11"])
    assert path == tmp_path / "A-nivel" / "figures" / "convergence.pdf" and path.exists()
    # Sec. 5.4: mean and sd over the four rotations, at the report seed
    assert ("global term", (4, 4)) in seen and ("validation BA", (4, 4)) in seen
    with pytest.raises(results.ResultsError, match="one report seed"):
        results.convergence_figure(tmp_path, "A-nivel",
                                   history + [{**history[0], "seed": 2}], ["L1->L11"])
    with pytest.raises(results.ResultsError, match="core"):
        results.convergence_figure(tmp_path, "A-nivel", history, ["L6->L1"])


def test_the_critical_difference_diagram_draws_every_rank_and_the_cd_bar(tmp_path,
                                                                         monkeypatch):
    texts = []
    import matplotlib.axes
    real = matplotlib.axes.Axes.annotate
    monkeypatch.setattr(matplotlib.axes.Axes, "annotate",
                        lambda self, text, *a, **k: texts.append(text) or real(self, text, *a,
                                                                                **k))
    ranking = st.friedman_nemenyi({"MIL-CREDA": [0.8, 0.7], "COT": [0.6, 0.65],
                                   "HASD": [0.7, 0.6]})
    results.critical_difference_figure(tmp_path, "B-tarea", ranking)
    assert f"CD = {ranking['critical_difference']:.2f}" in texts
    for s, r in ranking["average_rank"].items():
        assert f"{s} ({r:.2f})" in texts


def test_predictions_round_trip_through_the_runs_file(tmp_path):
    records = _records("MIL-CREDA", "L1->L11")
    results.append_predictions(tmp_path, "A-nivel", records[:5])
    results.append_predictions(tmp_path, "A-nivel", records[5:])
    assert results.read_predictions(tmp_path, "A-nivel") == records
    assert results.read_predictions(tmp_path, "B-tarea") == []


def test_the_ladder_reads_each_strategys_own_source_only_control():
    assert results.own_control("MIL-CREDA") == "source-only:MIL-CREDA"
    assert results.own_control("HASD") == "source-only:HASD"
    assert results.own_control("COT") == "source-only:common"
    records = _records("COT", "L1->L11", shift=0.1) + _records("source-only:common", "L1->L11")
    rows = [{"arm": "level", "transfer": "L1->L11", "reading": "all", "auc_rotation_mean": 0.6,
             "auc_rotation_sd": 0.03, "role": "core"},
            {"arm": "device", "transfer": "L6->Robots", "reading": "all",
             "auc_rotation_mean": 0.95, "auc_rotation_sd": 0.01, "role": "core"}]
    points = results.ladder_points(records, rows)
    assert [(p["strategy"], p["transfer"]) for p in points] == [("COT", "L1->L11")]
    mine = st.pooled(records[:12])["balanced_accuracy"]
    base = st.pooled(records[12:])["balanced_accuracy"]
    assert points[0]["gain"] == pytest.approx(mine - base)
    assert model_store.STRATEGIES[-1] == "MIL-CREDA"
