"""Experiments plan v50, Secs. 4.5, 5.1-5.4 and 6: the Results writers.

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
                        "rotation": (i // 2) % 4, "seed": s, "subject": f"{transfer}:{i}",
                        "label": label, "probability": float(np.clip(
                            0.5 + (label - 0.5) * 0.4 + shift + rng.normal(0, 0.2), 0, 1))})
    return out


IDENTITY = {"partition_sha256": "a" * 64, "code_version": "b" * 64,
            "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v43.md"}


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
             results.append_predictions(root, "A-nivel", [{"subject": "s"}],
                                        identity=IDENTITY)]
    rel = sorted(p.relative_to(root).as_posix() for p in paths)
    assert rel == ["A-nivel/runs/predictions.jsonl", "A-nivel/summary.json",
                   "A-nivel/tables/comparison.csv", "A-nivel/tables/comparison.md",
                   "history_A-nivel.json", "significance_A-nivel.json"]
    summary = json.loads((root / "A-nivel" / "summary.json").read_text())
    assert summary["synthetic"] is True and summary["revision"].endswith("v50.md")
    assert summary["rehearsal"] is False
    results.write_summary(root, "A-nivel", {"n": 3}, synthetic=False, rehearsal=True)
    assert json.loads((root / "A-nivel" / "summary.json").read_text())["rehearsal"] is True
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


def _selection(trials, rule):
    return {r: {"rotation": r, "rule": rule, "trial": n, "params": {"lr2": 1e-3 * (n + 1)},
                "score": 0.5, "testFold": r, "selectionRotations": [q for q in range(4)
                                                                    if q != (r + 3) % 4],
                "excludedRotation": (r + 3) % 4} for r, n in enumerate(trials)}


def test_the_selection_table_reads_both_rules_from_the_same_trials():
    primary, free = _selection([1, 1, 2, 0], "primary"), _selection([1, 2, 2, 3], "label-free")
    row = results.selection_rows("MIL-CREDA", "L1->L11", primary, free, test_primary=0.7,
                                 test_label_free=0.65)
    assert row["primary trial by rotation"] == "r0: 1; r1: 1; r2: 2; r3: 0"
    assert row["label-free trial by rotation"] == "r0: 1; r1: 2; r2: 2; r3: 3"
    assert row["coinciding rotations"] == 2
    assert row["test BA, primary rule"] == 0.7 and row["test BA, label-free rule"] == 0.65
    assert row["difference"] == pytest.approx(0.05)
    assert json.loads(row["primary configuration by rotation"])["3"] == {"lr2": 1e-3}
    detail = results.selected_configuration_rows("MIL-CREDA", "L1->L11",
                                                 {"primary": primary, "label-free": free})
    assert [(d["rule"], d["rotation"], d["trial"]) for d in detail][:2] == [
        ("primary", 0, 1), ("primary", 1, 1)]
    assert detail[0]["selection rotations"] == "0, 1, 2" and detail[0]["excluded rotation"] == 3
    assert detail[0]["score (pooled inner balanced accuracy)"] == 0.5
    assert not any("mean - sd" in k for k in detail[0])
    with pytest.raises(results.ResultsError, match="rotations"):
        results.selection_rows("MIL-CREDA", "L1->L11", primary, {0: free[0]})


def test_the_selection_score_is_labelled_the_pooled_inner_balanced_accuracy(tmp_path,
                                                                           monkeypatch):
    assert results.SELECTION_SCORE == "pooled inner balanced accuracy"
    primary = _selection([1, 0, 2, 1], "primary")
    detail = results.selected_configuration_rows("MIL-CREDA", "L1->L11", {"primary": primary})
    assert [d["score (pooled inner balanced accuracy)"] for d in detail] == [
        primary[r]["score"] for r in sorted(primary)]
    assert not any("mean" in k or "sd" in k.split() for k in detail[0])
    import matplotlib.axes
    import matplotlib.figure
    ylabels, colorbars = [], []
    real_ylabel, real_colorbar = matplotlib.axes.Axes.set_ylabel, matplotlib.figure.Figure.colorbar
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", lambda self, label, *a, **k: (
        ylabels.append(label), real_ylabel(self, label, *a, **k))[1])
    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", lambda self, *a, **k: (
        colorbars.append(k.get("label")), real_colorbar(self, *a, **k))[1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(0))
    study.optimize(lambda t: t.suggest_float("lr2", 1e-5, 1e-2, log=True)
                   - t.suggest_float("wd2", 1e-7, 1e-2, log=True), n_trials=6)
    results.optimization_history_figure(tmp_path, "A-nivel", {"MIL-CREDA": study})
    assert ylabels == ["best pooled inner balanced accuracy"]
    results.contour_figure(tmp_path, "A-nivel", study, ["lr2", "wd2"])
    assert colorbars == ["pooled inner balanced accuracy"]


def test_the_hyperparameter_table_carries_phase_two_per_test_rotation():
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("lr1", 1e-4, 1e-1, log=True), n_trials=2)
    rows = results.hyperparameter_rows({"MIL-CREDA": study},
                                       {"MIL-CREDA": _selection([1, 0, 1, 1], "primary")},
                                       "L1->L11")
    mine = next(r for r in rows if r["strategy"] == "MIL-CREDA")
    assert json.loads(mine["phase 1"]) == study.best_params
    assert json.loads(mine["phase 2"]) == {"0": {"lr2": 2e-3}, "1": {"lr2": 1e-3},
                                           "2": {"lr2": 2e-3}, "3": {"lr2": 2e-3}}


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


def test_the_label_free_convergence_figure_marks_the_label_free_epoch(tmp_path, monkeypatch):
    """Validation BA peaks at epoch 1, the label-free score at epoch 3: the primary figure
    marks 1, the label-free one 3 (Sec. 4.5's secondary rule reads no target label)."""
    history = [{"epoch": e, "seed": 1, "fold": fold, "model": "MIL-CREDA:phase2",
                "transfer": "L1->L11", "loss": 1.0 / (e + 1),
                "val_balanced_accuracy": [0.5, 0.9, 0.6, 0.7][e],
                "label_free_score": [0.1, 0.2, 0.3, 0.4][e]}
               for fold in range(4) for e in range(4)]
    marked = []
    import matplotlib.axes
    real = matplotlib.axes.Axes.axvline
    monkeypatch.setattr(matplotlib.axes.Axes, "axvline", lambda self, x=0, *a, **k: (
        marked.append(x), real(self, x, *a, **k))[1])
    results.convergence_figure(tmp_path, "A-nivel", history, ["L1->L11"])
    assert marked == [1.0]
    marked.clear()
    path = results.convergence_figure(tmp_path, "A-nivel", history, ["L1->L11"],
                                      name="convergence_label_free",
                                      epoch_key="label_free_score")
    assert marked == [3.0] and path.name == "convergence_label_free.pdf"


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
    results.append_predictions(tmp_path, "A-nivel", records[:5], identity=IDENTITY)
    results.append_predictions(tmp_path, "A-nivel", records[5:], identity=IDENTITY)
    assert results.read_predictions(tmp_path, "A-nivel") == [{**r, **IDENTITY}
                                                             for r in records]
    assert results.read_predictions(tmp_path, "B-tarea") == []


def test_every_prediction_carries_the_identity_it_was_written_under(tmp_path):
    records = _records("MIL-CREDA", "L1->L11")[:3]
    for broken in ({}, {k: v for k, v in IDENTITY.items() if k != "code_version"},
                   {**IDENTITY, "revision": ""}):
        with pytest.raises(results.ResultsError, match="identity"):
            results.append_predictions(tmp_path, "A-nivel", records, identity=broken)
    assert not (tmp_path / "A-nivel").exists()
    results.append_predictions(tmp_path, "A-nivel", records, identity=IDENTITY)
    assert results.stale_predictions(results.read_predictions(tmp_path, "A-nivel"),
                                     IDENTITY) == []
    newer = {**IDENTITY, "code_version": "c" * 64}
    stale = results.stale_predictions(results.read_predictions(tmp_path, "A-nivel"), newer)
    assert stale == [f"code_version: 3 records written under {'b' * 64!r}, current "
                     f"{'c' * 64!r}"]
    assert results.stale_predictions([{"strategy": "x"}], IDENTITY)[0].startswith(
        "partition_sha256: 1 records written under None")


def test_the_label_free_rule_predictions_live_in_their_own_file(tmp_path):
    records = _records("MIL-CREDA", "L1->L11")[:4]
    path = results.append_predictions(tmp_path, "A-nivel", records, identity=IDENTITY,
                                      rule="label-free")
    assert path.name == "predictions_label_free.jsonl" and path.parent.name == "runs"
    assert results.read_predictions(tmp_path, "A-nivel") == []
    assert len(results.read_predictions(tmp_path, "A-nivel", rule="label-free")) == 4
    with pytest.raises(results.ResultsError, match="rule"):
        results.read_predictions(tmp_path, "A-nivel", rule="oracle")


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


# ------------------------------------------------------------------ plan v50


def test_the_provenance_binds_plan_v50():
    assert results.__provenance__["revision"] == (
        "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md")


def _all_ten(strategies=model_store.STRATEGIES):
    records = []
    for k, s in enumerate(strategies):
        for t in results.GLOBAL_TRANSFERS:
            records += _records(s, t, shift=0.02 * k, seed=k)
    return records


def test_the_arm_ranking_is_descriptive_under_the_plans_headers():
    records = []
    for k, s in enumerate(("MIL-CREDA", "COT", "HASD")):
        for t in ("L1->L11", "L11->L1"):
            records += _records(s, t, shift=0.05 * k, seed=k)
    ranking = results.arm_ranking(records, ["L1->L11", "L11->L1"])
    assert ranking["test"] is None and "p_value" not in ranking
    table = results.ranking_table(ranking)
    assert table == [{"Ranking descriptivo del brazo":
                      "Nueve estrategias adaptadas, transferencias núcleo",
                      "Ranking promedio de MIL-CREDA": ranking["average_rank"]["MIL-CREDA"],
                      "Posición de MIL-CREDA entre las nueve": ranking["position"]["MIL-CREDA"]}]
    assert ranking["top_three"]["MIL-CREDA"] is (ranking["position"]["MIL-CREDA"] <= 3)
    # the comparison table still reads the rank of the nine from the same ranking
    comparison = results.comparison_table(records, "level", ["L1->L11", "L11->L1"], ranking)
    ranks = {r["method"]: r["average rank (core)"] for r in comparison}
    assert ranks["MIL-CREDA"] == f"{ranking['average_rank']['MIL-CREDA']:.2f}"
    empty = results.ranking_table(None)
    assert list(empty[0]) == list(table[0]) and empty[0][
        "Ranking promedio de MIL-CREDA"] is None


def test_the_arm_significance_table_carries_both_families_under_the_plans_headers():
    a = _records("MIL-CREDA", "L1->L11", n=24, shift=0.1)
    rows = []
    for k, rival in enumerate(("COT", "HASD")):
        rows += st.pair_contrast(a, _records(rival, "L1->L11", n=24, seed=3 + k),
                                 name_a="MIL-CREDA", name_b=rival, scope="L1|L11",
                                 test_train_ratio=0.3, hypothesis="A-nivel",
                                 experiment="A-nivel", resamples=200)
    st.apply_contrast_families(rows, "A-nivel MIL-CREDA vs rivals")
    table = results.significance_table(rows)
    assert list(results.SIGNIFICANCE_TABLE_COLUMNS) == [
        "Contraste", "Transferencia núcleo", "Diferencia de exactitud balanceada",
        "Intervalo bootstrap al 95 %", "Valor de p crudo", "Valor de p ajustado por Holm",
        "Diferencia de kappa", "Valor de p de la kappa ajustado por Holm",
        "Valor de p del AUC promedio ajustado por Holm",
        "Valor de p de Nadeau-Bengio ajustado por Holm", "Sujetos en que discrepan"]
    assert len(table) == 2 and all(list(r) == list(results.SIGNIFICANCE_TABLE_COLUMNS)
                                   for r in table)
    by_test = {(r["rival"], r["test"]): r for r in rows}
    first = table[0]
    assert first["Contraste"] == "MIL-CREDA vs COT" and first["Transferencia núcleo"] == "L1|L11"
    assert first["Valor de p de la kappa ajustado por Holm"] == by_test[
        ("COT", st.TEST_KAPPA)]["p_adjusted"]
    assert first["Diferencia de kappa"] == by_test[("COT", st.TEST_KAPPA)]["effect_size"]
    assert first["Valor de p del AUC promedio ajustado por Holm"] == by_test[
        ("COT", st.TEST_AUC_AVG)]["p_adjusted"]
    assert first["Valor de p ajustado por Holm"] == by_test[("COT", st.TEST_BOOTSTRAP)][
        "p_adjusted"]
    assert first["Sujetos en que discrepan"] == by_test[("COT", st.TEST_BOOTSTRAP)]["discordant"]


def test_the_global_ranking_is_one_friedman_over_the_ten_transfers_and_the_nine(tmp_path,
                                                                                  monkeypatch):
    assert len(results.GLOBAL_TRANSFERS) == 10
    assert set(results.GLOBAL_TRANSFERS) == {t for ts in results.ARM_TRANSFERS.values()
                                             for t in ts}
    records = _all_ten() + _records("source-only:common", "L1->L11") + _records(
        "upper-bound", "L1->L11")
    ranking = results.global_ranking(records)
    assert ranking["k"] == 9 and ranking["blocks"] == 10
    assert sorted(ranking["average_rank"]) == sorted(model_store.STRATEGIES)
    assert ranking["critical_difference"] == pytest.approx(3.102 * np.sqrt(1.5))
    assert ranking["groups"] == st.nemenyi_groups(ranking["average_rank"],
                                                  ranking["critical_difference"])
    assert ranking["limitation"] == st.GLOBAL_RANKING_LIMITATION
    calls = _spy(monkeypatch)
    titles = []
    import matplotlib.axes
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title",
                        lambda self, *a, **k: titles.append(a))
    paths = results.write_global_ranking(tmp_path, ranking)
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in paths)
    assert rel == ["figures/critical_difference.pdf", "significance_global_ranking.json",
                   "summary.json", "tables/global_ranking.csv", "tables/global_ranking.md",
                   "tables/global_ranking_strategies.csv",
                   "tables/global_ranking_strategies.md"]
    assert (tmp_path / "figures" / "critical_difference.png").exists() and not titles
    assert [c for c in calls if c[0] == "save"][0][2].get("png_preview") is True
    header = (tmp_path / "tables" / "global_ranking.csv").read_text(encoding="utf-8")
    assert header.splitlines()[0] == ("Ranking global,Estadístico de Friedman,Valor de p,"
                                      "Diferencia crítica de Nemenyi")
    assert "Nueve estrategias adaptadas, diez transferencias" in header
    strategies = (tmp_path / "tables" / "global_ranking_strategies.csv").read_text(
        encoding="utf-8").splitlines()
    assert strategies[0] == ("\"Estrategia, ranking global\","
                             "Ranking promedio sobre las diez transferencias")
    assert [line.split(",")[0] for line in strategies[1:]] == [
        f'"{s}"' if "," in s else s for s in model_store.STRATEGIES]
    sig = json.loads((tmp_path / "significance_global_ranking.json").read_text())
    assert len(sig) == 1 and sig[0]["test"] == st.TEST_FRIEDMAN_GLOBAL
    assert sig[0]["correction"] != "none" and sig[0]["n"] == 10
    assert set(st.SIGNIFICANCE_KEYS) <= set(sig[0])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["global_ranking"]["limitation"] == st.GLOBAL_RANKING_LIMITATION
    assert summary["global_ranking"]["critical_difference"] == pytest.approx(
        ranking["critical_difference"])


def test_the_global_ranking_refuses_a_strategy_missing_a_transfer():
    records = [r for r in _all_ten() if not (r["strategy"] == "COT"
                                             and r["transfer"] == "L6->Robots")]
    ranking = results.global_ranking(records)
    assert ranking is None          # the plan ranks the nine; eight is not its test
    assert results.global_ranking(_all_ten(model_store.STRATEGIES[:2])) is None


def test_the_auc_columns_read_auc_avg_and_the_per_fold_table_reports_each_cell():
    a = _records("MIL-CREDA", "L1->L11", n=24) + _records("MIL-CREDA", "L11->L1", n=24, seed=1)
    detail = results.transfer_detail(a, "level")
    one = [r for r in a if r["transfer"] == "L1->L11"]
    by = {d["transfer"]: d for d in detail}
    expected = st.auc_avg([r["label"] for r in one], [r["probability"] for r in one],
                          [st.cell_of(r) for r in one])
    assert by["L1->L11"]["auc"] == f"{expected['auc']:.3f}" and len(expected["cells"]) == 4
    secondary = {r["method"]: r for r in results.secondary_table({"MIL-CREDA": a}, "level")}
    assert secondary["MIL-CREDA"]["auc"] == f"{st.pooled(a)['auc']:.3f}"
    assert st.pooled(a)["auc"] == pytest.approx(np.mean(list(st.pooled(a)["auc_cells"].values())))
    per_fold = results.per_fold_table({"MIL-CREDA": a})
    assert per_fold[0]["auc per cell"].count("rotation 0") == 2      # both directions
