"""Every measurement the plan declares, written where the dashboard reads it.

Experiments plan v57, Secs. 4.5, 4.6, 5.1-5.4 and 6.1-6.3. The layout is the
`experimental-implementation` skill's Results table, under the product folder
`MIL-CREDA/Results/` (`DEFAULT_ROOT`), one folder per experiment
(`EXPERIMENTS`):

- `Results/significance_<experiment>.json`: a JSON list of rows with exactly
  the dashboard's keys plus `hypothesis` and `experiment`
  (`statistics.SIGNIFICANCE_KEYS`), refused otherwise (`write_significance`);
- `Results/<experiment>/tables/<name>.csv` and `.md` (`write_table`);
- `Results/<experiment>/summary.json`, one object (`write_summary`);
- `Results/history_<name>.json`, per-epoch records with `epoch`, `seed`,
  `fold` and `model` (`two_phase.write_history`, reused);
- `Results/<experiment>/figures/<name>.pdf` (the ladder of Sec. 5.4, which spans
  the arms, in `Results/figures/`) with a PNG preview, drawn through
  the skill's figure helper, copied byte for byte beside this module
  (`ps_figures.apply_style`, `ps_figures.save_figure`): printed size at 0.8 of
  the text width of a 10 pt body, no in-figure title, tight and transparent,
  vector PDF with embedded text. Declared figures: per arm the convergence
  curves, the optimization history, MIL-CREDA's contour over its two most
  important hyperparameters; across arms the domain-shift ladder and the global
  critical-difference diagram (`Results/figures/critical_difference.pdf`);
- `Results/<experiment>/runs/predictions.jsonl`: the per-run test predictions
  every table and contrast is computed from (`append_predictions`), each record
  stamped with the partition hash, plan revision and code version it was
  produced under (`IDENTITY_FIELDS`, `stale_predictions`); the EnsV secondary
  rule's predictions go to `runs/predictions_ensv.jsonl` and are read only by
  `selection_rules`. The panel does not show `runs/`.

**Tables (what each one is).** Per performance arm (Secs. 5.1-5.3):
`transfer_detail` (MIL-CREDA per transfer: balanced accuracy, AUC,
sensitivity, specificity at the one report seed of Sec. 4.6; balanced accuracy,
sensitivity and specificity on the pooled predictions, the AUC as AUC_avg, the
mean over the rotation-and-direction test cells, `statistics.auc_avg`),
`comparison` (one row per method, one column per transfer, each cell the
pooled balanced accuracy at the report seed, and the average rank of
the nine adapted strategies over the core transfers; the source-only controls
of the device arm print `controls.NA_RECORD`), `secondary` (kappa, AUC_avg,
sensitivity, specificity over the core pair), `per_fold` (the per-rotation
summary, Sec. 4.6, with each test cell's AUC), `ranking` (the descriptive arm ranking of
Secs. 5.1-5.3 under the plan's headers: MIL-CREDA's average rank and position
among the nine, no test, `arm_ranking`), `significance` (the arm's significance
table under the plan's headers, `significance_table`: per rival the
balanced-accuracy difference with its interval and raw and Holm p values, the
kappa difference with its Holm p value, the Holm p values of the AUC_avg
difference and of Nadeau-Bengio and the discordant subjects), `selection_rules` (Sec. 4.5's
table of the two selection rules under the plan's headers: one row per strategy over the
core transfers, the configuration and epoch each rule chose for each transfer and test
rotation, in how many of them they coincide, and the pooled test balanced accuracy of
each choice with their difference; under it, the outcome the rows show and the class
proportions of the source and target sides of each core transfer, also written as
`selection_class_proportions`, Sec. 8),
`selected_configurations` (per rule and test rotation, the trial, the
rotations it was selected over and its selection score, the balanced accuracy
from the subject-level counts pooled over the inner splits, Sec. 4.5) and `hyperparameters` (Sec. 5.4: phase 1's one
configuration, phase 2's configuration per test rotation). K1, K2 and K3 write the tables of Secs. 6.1-6.3.

**The global ranking (Sec. 5.4)** spans the arms, so it lands beside the
ladder at the top of `Results/` (`write_global_ranking`): one Friedman test
over the ten transfers as blocks and the nine adapted strategies, the Nemenyi
critical difference and its groups (`global_ranking`), in
`Results/figures/critical_difference.pdf`, `Results/tables/global_ranking`
and `Results/tables/global_ranking_strategies` (the plan's two tables and
headers), one row in `Results/significance_global_ranking.json` and the key
`global_ranking` of `Results/summary.json`, which carries the limitation the
plan declares (`statistics.GLOBAL_RANKING_LIMITATION`). It is computed only
when all nine strategies have a pooled value on all ten transfers.

Nothing here computes a model or a test: it lays out what `statistics` and
`mechanisms` returned. A result written from synthetic data says so in its
summary (`synthetic: true`).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from MIL_CREDA_EEG import bag_data, controls, model_store, ps_figures
from MIL_CREDA_EEG import statistics as st
from MIL_CREDA_EEG.two_phase import write_history as _write_history

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["4.5", "4.6", "5.1", "5.2", "5.3", "5.4", "6.1", "6.2", "6.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion",
                    "K2-emparejamiento", "K3-ablacion"],
    "invariants": [
        "every_writer_lands_in_the_dashboard_layout",
        "a_significance_file_missing_a_dashboard_key_is_refused",
        "json_never_carries_nan",
        "the_comparison_table_has_every_method_row_and_every_transfer_column",
        "the_device_arm_controls_print_the_na_record",
        "the_rank_column_ranks_only_the_nine_adapted_strategies_on_core_transfers",
        "the_selection_table_reads_both_rules_from_the_same_trials",
        "every_figure_goes_through_the_shipped_helper_with_a_png_preview",
        "the_convergence_figure_reads_the_core_transfers_and_marks_the_selected_epoch",
        "the_ensv_convergence_figure_marks_the_ensv_refit_epoch",
        "the_class_proportions_of_each_core_transfer_side_are_reported",
        "the_critical_difference_diagram_draws_every_rank_and_the_cd_bar",
        "the_arm_ranking_is_descriptive_under_the_plans_headers",
        "the_arm_significance_table_carries_both_families_under_the_plans_headers",
        "the_auc_columns_read_auc_avg_and_the_per_fold_table_reports_each_cell",
        "the_selection_score_is_labelled_the_pooled_inner_balanced_accuracy",
        "the_global_ranking_is_one_friedman_over_the_ten_transfers_and_the_nine",
        "predictions_round_trip_through_the_runs_file",
        "every_prediction_carries_the_identity_it_was_written_under",
        "the_ensv_rule_predictions_live_in_their_own_file",
        "the_hyperparameter_table_carries_phase_two_per_test_rotation",
        "the_ladder_reads_each_strategys_own_source_only_control",
    ],
}

DEFAULT_ROOT = bag_data.REPOSITORY / "MIL-CREDA" / "Results"
EXPERIMENT_OF_ARM = {"level": "A-nivel", "task": "B-tarea", "device": "C-dispositivo"}
EXPERIMENTS = (*EXPERIMENT_OF_ARM.values(), "K1-atencion", "K2-emparejamiento", "K3-ablacion")
#: Sec. 5.1-5.3: the transfer columns of each arm, in the plan's order.
ARM_TRANSFERS = {"level": ("L1->L6", "L1->L11", "L6->L1", "L6->L11", "L11->L1", "L11->L6"),
                 "task": ("Robots->CogniFit", "CogniFit->Robots"),
                 "device": ("L6->Robots", "Robots->L6")}
#: The method rows of the comparison tables (Secs. 5.1-5.3), keyed as predictions are.
SOURCE_ONLY = {"source-only:common": "Source only, common attention",
               "source-only:AdaptMIL": "Source only, AdaptMIL",
               "source-only:MIL-CREDA": "Source only, MIL-CREDA"}
UPPER_BOUND = "upper-bound"
#: What a stored prediction must have been produced under to be reused (Sec. 7.3).
IDENTITY_FIELDS = ("partition_sha256", "revision", "code_version")
#: Sec. 4.5's two selection rules: the primary rule's predictions feed every table and
#: contrast; the EnsV rule's are kept apart and read only by `selection_rules`.
PREDICTION_FILES = {"primary": "predictions.jsonl", "ensv": "predictions_ensv.jsonl"}
#: Sec. 4.5: the headers of the table of the two selection rules, the plan's own.
SELECTION_RULE_COLUMNS = ("Estrategia, reglas de selección",
                          "Configuración de la regla principal",
                          "Configuración de la regla secundaria", "Coinciden",
                          "Exactitud balanceada de prueba, regla principal",
                          "Exactitud balanceada de prueba, regla secundaria", "Diferencia")
#: Secs. 4.5 and 8: the class proportions of each core transfer's sides.
CLASS_PROPORTION_COLUMNS = ("Transferencia núcleo", "Lado", "Sujetos", "yes", "no",
                            "Proporción de yes")
METHOD_ROWS = {**SOURCE_ONLY, **{s: s for s in model_store.STRATEGIES},
               UPPER_BOUND: "Target-supervised upper bound"}
#: Sec. 5.4: the ten transfers of the three arms, the blocks of the global ranking.
GLOBAL_TRANSFERS = tuple(t for ts in ARM_TRANSFERS.values() for t in ts)
#: Secs. 5.1-5.3, the descriptive arm ranking table, headers verbatim.
RANKING_COLUMNS = ("Ranking descriptivo del brazo", "Ranking promedio de MIL-CREDA",
                   "Posición de MIL-CREDA entre las nueve")
RANKING_ROW = "Nueve estrategias adaptadas, transferencias núcleo"
#: Secs. 5.1-5.3, the arm significance table, headers verbatim.
SIGNIFICANCE_TABLE_COLUMNS = (
    "Contraste", "Transferencia núcleo", "Diferencia de exactitud balanceada",
    "Intervalo bootstrap al 95 %", "Valor de p crudo", "Valor de p ajustado por Holm",
    "Diferencia de kappa", "Valor de p de la kappa ajustado por Holm",
    "Valor de p del AUC promedio ajustado por Holm",
    "Valor de p de Nadeau-Bengio ajustado por Holm",
    "Sujetos en que discrepan")
#: Sec. 5.4, the global ranking's two tables, headers verbatim.
GLOBAL_RANKING_COLUMNS = ("Ranking global", "Estadístico de Friedman", "Valor de p",
                          "Diferencia crítica de Nemenyi")
GLOBAL_RANKING_ROW = "Nueve estrategias adaptadas, diez transferencias"
GLOBAL_STRATEGY_COLUMNS = ("Estrategia, ranking global",
                           "Ranking promedio sobre las diez transferencias")
#: Sec. 4.5: the selection score, the balanced accuracy from the subject-level confusion
#: counts pooled over the three inner splits.
SELECTION_SCORE = "pooled inner balanced accuracy"
FIGURE_BODY_PT = 10
FIGURE_WIDTH_FRACTION = 0.8


class ResultsError(RuntimeError):
    """A result cannot be written in the shape the dashboard reads."""


# ------------------------------------------------------------------ primitives


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(obj), indent=1, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    return path


def _experiment(experiment: str) -> str:
    if experiment not in EXPERIMENTS:
        raise ResultsError(f"{experiment!r} is not an experiment of the plan {EXPERIMENTS}")
    return experiment


def write_significance(root: str | Path, experiment: str, rows: Sequence[Mapping]) -> Path:
    """`Results/significance_<experiment>.json`, every row checked against the schema."""
    _experiment(experiment)
    for i, row in enumerate(rows):
        missing = [k for k in (*st.SIGNIFICANCE_KEYS, *st.OPTIONAL_KEYS) if k not in row]
        if missing:
            raise ResultsError(f"row {i} lacks the dashboard keys {missing}")
        if row["experiment"] != experiment:
            raise ResultsError(f"row {i} belongs to {row['experiment']}, not {experiment}")
    return write_json(Path(root) / f"significance_{experiment}.json", list(rows))


def _cell(value) -> str:
    value = _plain(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_table(root: str | Path, experiment: str, name: str, rows: Sequence[Mapping],
                columns: Sequence[str] | None = None, note: str | None = None) -> list[Path]:
    """`Results/<experiment>/tables/<name>.csv` and `.md`."""
    return _write_table_in(Path(root) / _experiment(experiment) / "tables", name, rows,
                           columns, note)


def _write_table_in(folder: Path, name: str, rows: Sequence[Mapping],
                    columns: Sequence[str] | None = None, note: str | None = None) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    columns = list(columns or (list(rows[0]) if rows else []))
    csv_path, md_path = folder / f"{name}.csv", folder / f"{name}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(r.get(k)) for k in columns) + " |" for r in rows]
    if note:
        lines += ["", note]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def write_summary(root: str | Path, experiment: str, summary: Mapping, *,
                  synthetic: bool, rehearsal: bool = False) -> Path:
    """`rehearsal` marks a result of a run that does not cover the plan's full comparison."""
    return write_json(Path(root) / _experiment(experiment) / "summary.json",
                      {"experiment": experiment, "revision": __provenance__["revision"],
                       "synthetic": bool(synthetic), "rehearsal": bool(rehearsal), **summary})


def write_history(root: str | Path, name: str, records: Sequence[Mapping]) -> Path:
    return _write_history(records, Path(root) / f"history_{name}.json")


def _predictions_path(root: str | Path, experiment: str, rule: str) -> Path:
    if rule not in PREDICTION_FILES:
        raise ResultsError(f"predictions are kept per selection rule {list(PREDICTION_FILES)}; "
                           f"got {rule!r}")
    return Path(root) / _experiment(experiment) / "runs" / PREDICTION_FILES[rule]


def append_predictions(root: str | Path, experiment: str, records: Iterable[Mapping], *,
                       identity: Mapping, rule: str = "primary") -> Path:
    """Append test predictions, each stamped with the partition hash, plan revision and
    code version they were produced under (`IDENTITY_FIELDS`)."""
    if sorted(identity) != sorted(IDENTITY_FIELDS) or not all(identity.values()):
        raise ResultsError(f"a prediction is written with its identity {list(IDENTITY_FIELDS)}, "
                           f"every one non-empty; got {dict(identity)}")
    path = _predictions_path(root, experiment, rule)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(_plain({**r, **identity}), allow_nan=False) + "\n")
    return path


def read_predictions(root: str | Path, experiment: str, rule: str = "primary") -> list[dict]:
    path = _predictions_path(root, experiment, rule)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stale_predictions(records: Sequence[Mapping], identity: Mapping) -> list[str]:
    """Per identity field, how many records were written under another value than now."""
    out = []
    for key in IDENTITY_FIELDS:
        stored: dict = {}
        for r in records:
            if r.get(key) != identity[key]:
                stored[r.get(key)] = stored.get(r.get(key), 0) + 1
        out += [f"{key}: {n} records written under {value!r}, current {identity[key]!r}"
                for value, n in stored.items()]
    return out


# ------------------------------------------------------------------ arm tables


def _select(records: Sequence[Mapping], method: str, transfers: Iterable[str]) -> list[Mapping]:
    transfers = set(transfers)
    return [r for r in records if r["strategy"] == method and r["transfer"] in transfers]


def _pooled(records: Sequence[Mapping]) -> dict:
    """The pooled metrics at the one report seed (Sec. 4.6); {} without records."""
    return st.pooled(records) if records else {}


def _value(summary: Mapping, metric: str) -> str:
    v = summary.get(metric) if summary else None
    return "" if v is None or not math.isfinite(v) else f"{v:.3f}"


def _mean_sd(summary: Mapping, metric: str) -> str:
    m = summary.get(metric) if summary else None
    if not m or m["mean"] is None or not math.isfinite(m["mean"]):
        return ""
    sd = m["sd"]
    return f"{m['mean']:.3f} ± {sd:.3f}" if sd is not None and math.isfinite(sd) \
        else f"{m['mean']:.3f}"


def transfer_detail(records: Sequence[Mapping], arm: str,
                    method: str = "MIL-CREDA") -> list[dict]:
    rows = []
    for t in ARM_TRANSFERS[arm]:
        s = _pooled(_select(records, method, [t]))
        rows.append({"transfer": t, **{m: _value(s, m) for m in
                                       ("balanced_accuracy", "auc", "sensitivity",
                                        "specificity")}})
    return rows


def comparison_table(records: Sequence[Mapping], arm: str, core: Sequence[str],
                     ranking: Mapping | None) -> list[dict]:
    """One row per method, one column per transfer; the rank column is the nine's."""
    rows = []
    for key, label in METHOD_ROWS.items():
        row = {"method": label}
        na = arm in controls.HETEROGENEOUS_ARMS and key in SOURCE_ONLY
        for t in ARM_TRANSFERS[arm]:
            row[t] = controls.NA_RECORD if na else _value(
                _pooled(_select(records, key, [t])), "balanced_accuracy")
        rank = (ranking or {}).get("average_rank", {}).get(key)
        row["average rank (core)"] = (controls.NA_RECORD if na else
                                      "" if rank is None else f"{rank:.2f}")
        rows.append(row)
    return rows


def ranking_scores(records: Sequence[Mapping], core: Sequence[str]) -> dict[str, list[float]]:
    """Per adapted strategy, the pooled BA at the report seed of each core transfer."""
    out = {}
    for s in model_store.STRATEGIES:
        values = []
        for t in core:
            summary = _pooled(_select(records, s, [t]))
            values.append(summary["balanced_accuracy"] if summary else float("nan"))
        if all(math.isfinite(v) for v in values):
            out[s] = values
    return out


def secondary_table(pooled_pair: Mapping[str, Sequence[Mapping]], arm: str) -> list[dict]:
    rows = []
    for key, label in METHOD_ROWS.items():
        na = arm in controls.HETEROGENEOUS_ARMS and key in SOURCE_ONLY
        s = {} if na else _pooled(pooled_pair.get(key, []))
        rows.append({"method": label, **{m: controls.NA_RECORD if na else _value(s, m)
                                         for m in ("kappa", "auc", "sensitivity",
                                                   "specificity")}})
    return rows


def per_fold_table(pooled_pair: Mapping[str, Sequence[Mapping]]) -> list[dict]:
    """Per rotation: BA and AUC_avg over the rotation's cells, and each cell's AUC."""
    rows = []
    for key, records in pooled_pair.items():
        if not records:
            continue
        f = st.per_fold(records)
        for r, m in sorted(f["rotations"].items()):
            cells = m.get("auc_cells", {})
            rows.append({"method": METHOD_ROWS.get(key, key), "rotation": r,
                         "balanced_accuracy": m["balanced_accuracy"], "auc": m["auc"],
                         "auc per cell": "; ".join(f"{c}: {_value(cells, c)}"
                                                   for c in cells)})
        rows.append({"method": METHOD_ROWS.get(key, key), "rotation": "mean ± sd",
                     "balanced_accuracy": _mean_sd(f, "balanced_accuracy"),
                     "auc": _mean_sd(f, "auc")})
    return rows


def arm_ranking(records: Sequence[Mapping], core: Sequence[str]) -> dict | None:
    """Secs. 5.1-5.3: the descriptive ranking of the nine on an arm's core transfers.

    None until at least two adapted strategies have a pooled value on every core
    transfer; no test is run (Sec. 5.4 holds the plan's one ranking test).
    """
    scores = ranking_scores(records, core)
    return st.descriptive_ranking(scores) if len(scores) >= 2 else None


def ranking_table(ranking: Mapping | None) -> list[dict]:
    """The arm's descriptive ranking table: MIL-CREDA's average rank and position."""
    rank = (ranking or {}).get("average_rank", {}).get("MIL-CREDA")
    position = (ranking or {}).get("position", {}).get("MIL-CREDA")
    return [dict(zip(RANKING_COLUMNS, (RANKING_ROW, rank, position)))]


def significance_table(rows: Sequence[Mapping]) -> list[dict]:
    """Secs. 5.1-5.3: one row per rival contrast, both families of Sec. 4.6 side by side."""
    by: dict = {}
    for r in rows:
        if r.get("contrast") and r["test"] in st.CONTRAST_TESTS:
            by.setdefault(r["contrast"], {})[r["test"]] = r
    out = []
    for parts in by.values():
        boot = parts.get(st.TEST_BOOTSTRAP)
        if boot is None:
            continue
        kappa = parts.get(st.TEST_KAPPA, {})
        interval = ("" if boot["ci_low"] is None or boot["ci_high"] is None
                    else f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
        name = boot["comparison"].rsplit(":", 1)[-1]
        out.append(dict(zip(SIGNIFICANCE_TABLE_COLUMNS, (
            name, boot.get("scope"), boot["effect_size"], interval, boot["p_value"],
            boot["p_adjusted"], kappa.get("effect_size"), kappa.get("p_adjusted"),
            parts.get(st.TEST_AUC_AVG, {}).get("p_adjusted"),
            parts.get(st.TEST_NADEAU_BENGIO, {}).get("p_adjusted"), boot.get("discordant")))))
    return out


def global_ranking(records: Sequence[Mapping]) -> dict | None:
    """Sec. 5.4: Friedman and Nemenyi over the ten transfers and the nine strategies.

    None unless every adapted strategy has a pooled value on all ten transfers.
    """
    scores = ranking_scores(records, GLOBAL_TRANSFERS)
    if sorted(scores) != sorted(model_store.STRATEGIES):
        return None
    ranking = st.friedman_nemenyi({s: scores[s] for s in model_store.STRATEGIES})
    return {**ranking, "transfers": list(GLOBAL_TRANSFERS), "test": st.TEST_FRIEDMAN_GLOBAL,
            "groups": st.nemenyi_groups(ranking["average_rank"],
                                        ranking["critical_difference"]),
            "limitation": st.GLOBAL_RANKING_LIMITATION}


def write_global_ranking(root: str | Path, ranking: Mapping) -> list[Path]:
    """Sec. 5.4's cross-arm outputs: the figure, the two tables, the test row, the summary."""
    root = Path(root)
    paths = [critical_difference_figure_at(root / "figures" / "critical_difference.pdf",
                                           ranking)]
    paths += _write_table_in(root / "tables", "global_ranking", [dict(zip(
        GLOBAL_RANKING_COLUMNS, (GLOBAL_RANKING_ROW, ranking["statistic"],
                                 ranking["p_value"], ranking["critical_difference"])))],
        note=ranking["limitation"])
    paths += _write_table_in(root / "tables", "global_ranking_strategies", [
        dict(zip(GLOBAL_STRATEGY_COLUMNS, (s, ranking["average_rank"][s])))
        for s in model_store.STRATEGIES if s in ranking["average_rank"]])
    row = st.significance_row(
        comparison="global ranking of the nine adapted strategies over the ten transfers",
        test=ranking["test"], statistic=ranking["statistic"], p_value=ranking["p_value"],
        p_adjusted=ranking["p_value"],
        correction="one global test (m=1); Nemenyi post hoc, critical difference",
        effect_size=ranking["average_rank"].get("MIL-CREDA"),
        effect_size_name="MIL-CREDA average rank over the ten transfers (1 = best)",
        ci_low=None, ci_high=None, n=ranking["blocks"],
        hypothesis="A-nivel, B-tarea, C-dispositivo", experiment="A-nivel, B-tarea, C-dispositivo",
        critical_difference=ranking["critical_difference"], methods=ranking["k"],
        groups=ranking["groups"], limitation=ranking["limitation"])
    paths.append(write_json(root / "significance_global_ranking.json", [row]))
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) \
        if summary_path.exists() else {}
    summary["global_ranking"] = {
        "revision": __provenance__["revision"], "transfers": ranking["transfers"],
        "statistic": ranking["statistic"], "p_value": ranking["p_value"],
        "critical_difference": ranking["critical_difference"], "alpha": ranking["alpha"],
        "average_rank": ranking["average_rank"], "position": ranking["position"],
        "groups": ranking["groups"], "limitation": ranking["limitation"]}
    paths.append(write_json(summary_path, summary))
    return paths


# ------------------------------------------------------------------ Sec. 4.5 and 5.4


def _choices(chosen: Mapping[str, Mapping[int, Mapping]]) -> str:
    """Per core transfer and test rotation, the trial, configuration and epoch count E2."""
    return json.dumps({f"{t} r{r}": {**chosen[t][r]["params"], "E2": chosen[t][r]["epochs"],
                                     "trial": chosen[t][r]["trial"]}
                       for t in sorted(chosen) for r in sorted(chosen[t])}, sort_keys=True)


def selection_rows(strategy: str, primary: Mapping[str, Mapping[int, Mapping]],
                   secondary: Mapping[str, Mapping[int, Mapping]],
                   test_primary: float | None = None,
                   test_secondary: float | None = None) -> dict:
    """Sec. 4.5's table of the two selection rules, one strategy's row over the core
    transfers (`primary[transfer][rotation]`, `search.select`): the configuration and epoch
    each rule chose for each transfer and test rotation, in how many of them they coincide
    (the same trial AND epoch), and the pooled test balanced accuracy of each choice with
    their difference (primary minus secondary), under the plan's headers."""
    def keys(chosen):
        return sorted((t, r) for t in chosen for r in chosen[t])

    if keys(primary) != keys(secondary) or not keys(primary):
        raise ResultsError(f"both rules select for the same transfers and test rotations; "
                           f"got {keys(primary)} and {keys(secondary)}")
    same = sum((primary[t][r]["trial"], primary[t][r]["epoch"])
               == (secondary[t][r]["trial"], secondary[t][r]["epoch"])
               for t, r in keys(primary))
    diff = (test_primary - test_secondary
            if test_primary is not None and test_secondary is not None else None)
    return dict(zip(SELECTION_RULE_COLUMNS, (
        strategy, _choices(primary), _choices(secondary), f"{same}/{len(keys(primary))}",
        test_primary, test_secondary, diff)))


def selection_outcome(row: Mapping) -> str:
    """Which of Sec. 4.5's three outcomes a row of the two rules shows, as far as the row
    decides it: the same choice everywhere, or different choices with the test difference
    the reader judges (the plan fixes no threshold for similar results)."""
    same, total = (int(x) for x in str(row["Coinciden"]).split("/"))
    name = row["Estrategia, reglas de selección"]
    if same == total:
        return (f"{name}: las dos reglas eligen lo mismo en las {total} rotaciones; la "
                f"selección no depende del validador.")
    diff = row["Diferencia"]
    measured = (f"diferencia de exactitud balanceada de prueba (principal menos "
                f"secundaria) {diff:.4f}" if diff is not None else
                "sin exactitud de prueba de alguna de las dos elecciones")
    return (f"{name}: eligen distinto en {total - same} de {total} rotaciones; {measured}. "
            f"El resultado reportado es el de la regla principal.")


def class_proportion_rows(assignment) -> list[dict]:
    """Secs. 4.5 and 8: the class proportions of the source and target sides of one core
    transfer, over the subjects each side holds (`assignment.Assignment`)."""
    rows = []
    for side, label in (("source", "fuente"), ("target", "objetivo")):
        subjects = sorted(assignment.subjects(side))
        yes = sum(assignment.labels[s] == "yes" for s in subjects)
        rows.append(dict(zip(CLASS_PROPORTION_COLUMNS, (
            assignment.transfer.name, label, len(subjects), yes, len(subjects) - yes,
            yes / len(subjects) if subjects else None))))
    return rows


def selected_configuration_rows(strategy: str, transfer: str,
                                selections: Mapping[str, Mapping[int, Mapping]]) -> list[dict]:
    """Per rule and test rotation: the trial chosen and the rotations it was chosen over."""
    return [{"strategy": strategy, "transfer": transfer, "rule": rule, "rotation": r,
             "test fold": sel["testFold"], "trial": sel["trial"],
             "configuration": json.dumps(sel["params"], sort_keys=True),
             "selection rotations": ", ".join(str(q) for q in sel["selectionRotations"]),
             "excluded rotation": sel["excludedRotation"],
             f"score ({SELECTION_SCORE})": sel["score"]}
            for rule, chosen in selections.items() for r, sel in sorted(chosen.items())]


def hyperparameter_rows(phase_one: Mapping[str, object],
                        phase_two: Mapping[str, Mapping[int, Mapping]],
                        transfer: str) -> list[dict]:
    """Sec. 5.4: chosen values per strategy and phase: phase 1 the one configuration of its
    group's study, phase 2 the configuration chosen for each test rotation (Sec. 4.5)."""
    from MIL_CREDA_EEG.search import PHASE_ONE_GROUP_OF
    rows = []
    for strategy in model_store.STRATEGIES:
        group = PHASE_ONE_GROUP_OF[strategy]
        one, two = phase_one.get(group), phase_two.get(strategy)
        rows.append({"strategy": strategy, "transfer": transfer, "phase-1 group": group,
                     "phase 1": json.dumps(one.best_params, sort_keys=True) if one else "",
                     "phase 2": json.dumps({str(r): two[r]["params"] for r in sorted(two)},
                                           sort_keys=True) if two else ""})
    return rows


def importance_rows(study, strategy: str, seed: int = 0) -> list[dict]:
    """fANOVA importance (Optuna) of each searched hyperparameter of one study."""
    import optuna
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=seed)
    importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
    return [{"strategy": strategy, "study": study.study_name, "parameter": k,
             "importance": v} for k, v in importances.items()]


# ------------------------------------------------------------------ figures


def _figure(rows: int = 1, cols: int = 1, aspect: float = ps_figures.GOLDEN, **kwargs):
    """A figure at its printed size through the skill's helper (no title, ever)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps_figures.apply_style(body_pt=FIGURE_BODY_PT, printed_width_fraction=FIGURE_WIDTH_FRACTION,
                           aspect=aspect)
    fig, axes = plt.subplots(rows, cols, squeeze=False, **kwargs)
    return plt, fig, axes


def _save(plt, fig, path: Path) -> Path:
    ps_figures.save_figure(fig, path, png_preview=True)
    plt.close(fig)
    return path


def _band(ax, xs, rows: Sequence[Sequence[float]], label: str, **style) -> None:
    arr = np.asarray(rows, dtype=float)
    mean = np.nanmean(arr, axis=0)
    line, = ax.plot(xs, mean, label=label, **style)
    if arr.shape[0] > 1:
        sd = np.nanstd(arr, axis=0, ddof=1)
        ax.fill_between(xs, mean - sd, mean + sd, alpha=0.2, color=line.get_color())


def selected_epoch(records: Sequence[Mapping], key: str = "val_balanced_accuracy",
                   field: str | None = None) -> float | None:
    """Median over runs (transfer, seed, fold) of the first epoch of maximal `key`; with
    `field`, of the epoch each run's records carry under it (an EnsV refit's `ensvEpoch`,
    the epoch the rule chose in the search, Sec. 4.5)."""
    if field is not None:
        picks = {(r.get("transfer"), r["seed"], r["fold"]): r[field]
                 for r in records if field in r}
        return float(np.median(list(picks.values()))) if picks else None
    runs: dict = {}
    for r in records:
        if key in r:
            runs.setdefault((r.get("transfer"), r["seed"], r["fold"]), []).append(r)
    picks = []
    for rs in runs.values():
        rs = sorted(rs, key=lambda r: r["epoch"])
        best = max(r[key] for r in rs)
        picks.append(next(r["epoch"] for r in rs if r[key] == best))
    return float(np.median(picks)) if picks else None


CONVERGENCE_SERIES = {"loss": "loss", "val_balanced_accuracy": "validation BA",
                      "global": "global term", "local": "local term"}


def convergence_figure(root: str | Path, experiment: str, history: Sequence[Mapping],
                       core: Sequence[str], name: str = "convergence",
                       epoch_key: str = "val_balanced_accuracy",
                       epoch_field: str | None = None) -> Path:
    """Sec. 5.4: one figure per arm on its core transfers, one panel per phase-1 group and
    per phase-2 strategy; each curve the mean +/- sd over the four rotations, at the one
    report seed, of the value averaged over the core transfers; the selected epoch marked;
    MIL-CREDA's global and local terms with their bounds. The selected epoch is the one of
    maximal `epoch_key`, or, with `epoch_field`, the epoch each run carries under it: the
    EnsV models' figure marks `ensvEpoch`, the epoch EnsV chose (Sec. 4.5)."""
    path = Path(root) / _experiment(experiment) / "figures" / f"{name}.pdf"
    records = [r for r in history if r.get("transfer") in set(core)]
    if not records:
        raise ResultsError(f"no history of the core transfers {list(core)}")
    seeds = sorted({r["seed"] for r in records})
    if len(seeds) > 1:
        raise ResultsError(f"Sec. 5.4 draws the one report seed of Sec. 4.6; the history "
                           f"holds seeds {seeds}")
    models = sorted({r["model"] for r in records}, key=lambda m: (not m.endswith("phase1"), m))
    plt, fig, axes = _figure(len(models), 1, aspect=0.42 * len(models), sharex=False)
    for ax, model in zip(axes[:, 0], models):
        rs = [r for r in records if r["model"] == model]
        epochs = sorted({r["epoch"] for r in rs})
        folds = sorted({r["fold"] for r in rs})
        for key, label in CONVERGENCE_SERIES.items():
            if not any(key in r for r in rs):
                continue
            per_fold = [[np.nanmean([r[key] for r in rs if r["fold"] == f
                                     and r["epoch"] == e and key in r] or [np.nan])
                         for e in epochs] for f in folds]
            _band(ax, epochs, per_fold, label)
        for term in ("global", "local"):
            if any(f"{term}_bound_high" in r for r in rs):
                for bound in (rs[0][f"{term}_bound_low"], rs[0][f"{term}_bound_high"]):
                    ax.axhline(bound, linestyle=":", linewidth=0.6, color="grey")
        chosen = selected_epoch(rs, epoch_key, epoch_field)
        if chosen is not None:
            ax.axvline(chosen, linestyle="--", linewidth=0.8, color="black",
                       label="selected epoch (median)" if epoch_field is None
                       else "EnsV epoch (median)")
        ax.set_ylabel(model.replace(":phase", ", phase "))
        ax.legend(loc="best")
    axes[-1, 0].set_xlabel("epoch")
    return _save(plt, fig, path)


def optimization_history_figure(root: str | Path, experiment: str,
                                studies: Mapping[str, object],
                                name: str = "optimization_history") -> Path:
    """Sec. 5.4: best selection score so far against the trial number, one line per
    strategy."""
    path = Path(root) / _experiment(experiment) / "figures" / f"{name}.pdf"
    plt, fig, axes = _figure()
    ax = axes[0, 0]
    for label, study in studies.items():
        values = [t.value for t in sorted(study.trials, key=lambda t: t.number)
                  if t.value is not None]
        if values:
            ax.plot(range(1, len(values) + 1), np.maximum.accumulate(values), label=label,
                    linewidth=2 if label.startswith("MIL-CREDA") else 1)
    ax.set_xlabel("trial")
    ax.set_ylabel(f"best {SELECTION_SCORE}")
    ax.legend(loc="best")
    return _save(plt, fig, path)


def contour_figure(root: str | Path, experiment: str, study, parameters: Sequence[str],
                   name: str = "contour_mil_creda") -> Path:
    """Sec. 5.4: MIL-CREDA's trials over its two most important hyperparameters."""
    if len(parameters) != 2:
        raise ResultsError("the contour reads exactly two hyperparameters")
    path = Path(root) / _experiment(experiment) / "figures" / f"{name}.pdf"
    trials = [t for t in study.trials if t.value is not None
              and all(p in t.params for p in parameters)]
    plt, fig, axes = _figure()
    ax = axes[0, 0]
    xs = [t.params[parameters[0]] for t in trials]
    ys = [t.params[parameters[1]] for t in trials]
    zs = [t.value for t in trials]
    if len(trials) >= 4:
        try:
            ax.tricontourf(xs, ys, zs, alpha=0.4)
        except (ValueError, RuntimeError):
            pass                                   # collinear trials: points alone
    points = ax.scatter(xs, ys, c=zs)
    fig.colorbar(points, ax=ax, label=SELECTION_SCORE)
    ax.set_xlabel(parameters[0])
    ax.set_ylabel(parameters[1])
    return _save(plt, fig, path)


def critical_difference_figure(root: str | Path, experiment: str, ranking: Mapping,
                               name: str = "critical_difference") -> Path:
    """Secs. 5.1-5.3: average ranks on one axis (1 = best) and the Nemenyi CD bar."""
    return critical_difference_figure_at(
        Path(root) / _experiment(experiment) / "figures" / f"{name}.pdf", ranking)


def critical_difference_figure_at(path: Path, ranking: Mapping) -> Path:
    """Average ranks (1 = best), the Nemenyi CD bar and a line under each group."""
    ranks = sorted(ranking["average_rank"].items(), key=lambda kv: kv[1])
    k, cd = ranking["k"], ranking["critical_difference"]
    plt, fig, axes = _figure(aspect=0.25 + 0.06 * k)
    ax = axes[0, 0]
    ax.set_xlim(k + 0.5, 0.5)
    ax.set_ylim(-1, len(ranks) + 1)
    for i, (strategy, rank) in enumerate(ranks):
        y = len(ranks) - i
        ax.plot([rank], [y], "o", markersize=4, color="black" if strategy == "MIL-CREDA"
                else "grey")
        ax.annotate(f"{strategy} ({rank:.2f})", (rank, y), xytext=(4, 0),
                    textcoords="offset points", va="center")
    ax.plot([1, 1 + cd], [0, 0], color="black", linewidth=2)
    ax.annotate(f"CD = {cd:.2f}", (1 + cd / 2, 0), xytext=(0, 3), textcoords="offset points",
                ha="center")
    for g, group in enumerate(ranking.get("groups") or []):
        values = [ranking["average_rank"][s] for s in group]
        ax.plot([min(values), max(values)], [-0.3 - 0.25 * g] * 2, color="black",
                linewidth=1.5)
    ax.set_xlabel("average rank (1 = best)")
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    return _save(plt, fig, path)


def own_control(strategy: str) -> str:
    """Sec. 5.4: the source-only control each strategy's gain is read against."""
    return f"source-only:{strategy}" if strategy in ("MIL-CREDA", "AdaptMIL") \
        else "source-only:common"


def ladder_points(records: Sequence[Mapping], separability_rows: Sequence[Mapping]) -> list[dict]:
    """Per strategy and level or task transfer: separability (reading `all`) and the gain
    of the pooled BA at the report seed over the strategy's own source-only control."""
    sep = {r["transfer"]: r for r in separability_rows if r["reading"] == "all"
           and r["arm"] in controls.SAME_CHANNEL_ARMS}
    points = []
    for strategy in model_store.STRATEGIES:
        for transfer, row in sep.items():
            mine = _pooled(_select(records, strategy, [transfer]))
            base = _pooled(_select(records, own_control(strategy), [transfer]))
            if not mine or not base:
                continue
            points.append({"strategy": strategy, "transfer": transfer, "role": row.get("role"),
                           "separability": float(row["auc_rotation_mean"]),
                           "separability_sd": float(row["auc_rotation_sd"]),
                           "gain": mine["balanced_accuracy"] - base["balanced_accuracy"],
                           "control": own_control(strategy)})
    return points


def ladder_figure(root: str | Path, points: Sequence[Mapping], device_separability: float | None,
                  name: str = "domain_shift_ladder") -> Path:
    """Sec. 5.4: gain over each strategy's own source-only control against separability."""
    path = Path(root) / "figures" / f"{name}.pdf"
    plt, fig, axes = _figure()
    ax = axes[0, 0]
    for strategy in sorted({p["strategy"] for p in points}):
        ps = [p for p in points if p["strategy"] == strategy]
        ax.errorbar([p["separability"] for p in ps], [p["gain"] for p in ps],
                    xerr=[p.get("separability_sd") or 0 for p in ps], fmt="o",
                    label=strategy, markersize=6 if strategy == "MIL-CREDA" else 3)
    if device_separability is not None:
        ax.axvline(device_separability, linestyle="--", label="device arm separability")
    ax.axhline(0.0, linewidth=0.5)
    ax.set_xlabel("domain separability (AUC)")
    ax.set_ylabel("BA gain over own source-only control")
    ax.legend(loc="best")
    return _save(plt, fig, path)
