"""The campaign: the plan's declared steps, wired to runnable local commands.

Experiments plan v42, Secs. 4.5, 4.6, 5.1-5.4, 6.1-6.3, 7.3, 7.5 and 7.7.
The steps, in order (`STEPS`), each one a subcommand of
`python -m MIL_CREDA_EEG.campaign <step> --config <file> [--product <dir>]`:

1. `search` -- per transfer, the four phase-1 studies (one per group a
   requested strategy starts from), each chosen configuration retrained at the
   search seed in every rotation and stored, then one phase-2 study per
   strategy from those checkpoints (`search.run_phase_one_study`,
   `search.run_phase_two_study`). An existing study is read back only when it
   holds exactly its budget of completed trials; an incomplete one is refused,
   never extended.
2. `final` -- per transfer and rotation, at the one report seed (Sec. 4.6, seed
   1): the chosen phase-1 configuration of each group trained and stored, its
   source-only control on the target test fold (`controls`; the device arm
   records `NA_RECORD`), the chosen phase-2 configuration of each strategy
   trained from the checkpoint of the same rotation and stored, its target test
   predictions, and the target-supervised upper bound with the common group's
   chosen `lr1`, `wd1`. A model already stored is not trained again (an
   interrupted run resumes).
3. `k3` -- Sec. 6.3 on the core transfers, per rotation at the report seed: the
   four aggregation variants (`aggregation_variants`: the own attention without
   consensus, the published and gated attentions and the mean) trained through
   both phases and the two ablation variants (global term only, local term only)
   trained in phase 2 from full MIL-CREDA's stored phase-1 checkpoint, all with
   full MIL-CREDA's selected hyperparameters of the transfer and no search of
   their own (`mechanisms.train_aggregation_variant`,
   `mechanisms.train_ablation_variant`). Sec. 6.2's "without local term" is the
   global-term-only variant, so it is also trained on a declared K2 transfer that
   is not a core transfer (`k3_plan`). Every variant model is stored as the final
   models are (`model_store.VARIANTS`), because K1 reads their attention and K2
   the global-term-only model. The own attention and "both terms" are full
   MIL-CREDA itself, read from the `final` step's predictions, never trained
   again. Per core pair, both directions pooled: the aggregation and ablation
   tables, and the Sec. 4.6 contrasts of own vs own without consensus and of
   both terms vs each single term, Holm over each table's bootstrap tests; the
   published and gated attentions and the mean are reported without criterion
   (`Results/K3-ablacion/`, `Results/significance_K3-ablacion.json`). A variant
   with its model stored and its predictions recorded is not trained again.
4. `statistics` -- per arm, the core pair pooled over its two directions, the
   tables of Secs. 5.1-5.3, the Sec. 4.6 contrasts of MIL-CREDA against every
   rival present with Holm over the arm's bootstrap family, and the Friedman
   ranking (`statistics`, `results`).
5. `mechanisms` -- K1 per arm over the stored models of its core transfers
   (`mechanisms.k1_tests`: the level arm's three criterion tests, support in the
   task and device arms, every other attention as a reference), K2 over every
   transfer's probe subjects with its criterion contrast against the
   global-term-only model, and the K3 implementation control of the term bounds
   from the recorded histories (`mechanisms`).
6. `results` -- the Sec. 5.4 outputs: the per-epoch histories
   (`Results/history_<experiment>.json`), the chosen hyperparameters, fANOVA
   importance, the two selection rules, and the declared figures through
   `ps_figures`: per arm the convergence curves, the optimization history,
   MIL-CREDA's contour and the critical-difference diagram; the domain-shift
   ladder across arms.

`all` runs them in order. `synthetic --out <dir>` writes a tiny seeded
synthetic processed set, its partition and a config (labelled synthetic), so
the whole flow can be rehearsed on CPU.

**Local only.** Nothing here submits, schedules or launches remote work: the
commands run in this process, on `config["device"]` (default `cpu`). Placing a
step on a remote worker belongs to `remote-execution`, behind the operator.

**Required inputs (no default, `CampaignError` when absent).** The schedule
(`epochs1`, `patience1`, `epochs2`, `bagsPerStep`: the pilot fixes them,
Sec. 7.6) and the strategies and transfers to run. Plan v42 resolved every
decision the earlier inputs stood in for -- the seed reduction (one report
seed), K1's common attention, band edges and scope (Sec. 6.1), and the open
comparator decisions (Sec. 7.7) -- so those inputs are retired and refused when
given (`RETIRED_INPUTS`), never silently ignored.

**Overridable for a rehearsal, recorded in every summary.** `nTrials`
(default `search.N_TRIALS`, 20) and `bootstrapResamples` (default
`statistics.BOOTSTRAP_RESAMPLES`). `reportSeeds` is not overridable: it is
`REPORT_SEEDS`, the single seed 1, and any other value is refused. The core
transfers come from the measured separability summary
(`Results/separability/summary.json`) unless `core` gives them; the
domain-shift ladder reads the measured separability table
(`Results/separability/tables/separability.csv`, or `separabilityTable`) and is
listed as owed when there is none.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from MIL_CREDA_EEG import aggregation_variants
from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, controls, mechanisms, model_store, partition, results
from MIL_CREDA_EEG import search
from MIL_CREDA_EEG import statistics as st
from MIL_CREDA_EEG import two_phase
from MIL_CREDA_EEG.mil_wrapper import COMPARATORS

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v42.md",
    "sections": ["4.5", "4.6", "5.1", "5.2", "5.3", "5.4", "6.1", "6.2", "6.3", "7.3",
                 "7.5", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion",
                    "K2-emparejamiento", "K3-ablacion"],
    "invariants": [
        "the_steps_are_the_declared_ones_in_order",
        "required_inputs_have_no_default",
        "every_strategy_runs_without_an_open_decision_guard",
        "the_campaign_never_launches_remote_work",
        "an_incomplete_study_is_refused_not_extended",
        "the_final_models_are_trained_at_the_report_seeds_from_the_same_rotation",
        "a_stored_model_is_not_trained_again",
        "the_device_arm_source_only_controls_are_never_numbers",
        "the_rehearsal_writes_the_dashboard_layout_end_to_end",
        "k3_trains_the_variants_and_writes_both_tables_with_holm_per_table",
        "k3_does_not_train_a_variant_again",
        "k2_trains_the_global_term_only_variant_on_its_declared_transfers",
    ],
}

STEPS = ("search", "final", "k3", "statistics", "mechanisms", "results")
K3 = "K3-ablacion"
#: Sec. 6.2's "without local term": the Sec. 6.3 ablation's global-term-only variant.
K2_GLOBAL_ONLY = "global term only"
#: Sec. 6.3's tables: the row that is full MIL-CREDA, the variants trained here, and the
#: rivals of the criterion contrasts (the published and gated attentions and the mean are
#: reported without a criterion).
K3_TABLES = {
    "aggregation": {"full": "own", "trained": aggregation_variants.VARIANTS,
                    "criterion": (aggregation_variants.WITHOUT_CONSENSUS,)},
    "ablation": {"full": "both terms",
                  "trained": tuple(a for a in mechanisms.ABLATIONS if a != "both terms"),
                  "criterion": tuple(a for a in mechanisms.ABLATIONS if a != "both terms")},
}
NOT_APPLICABLE = "not applicable"
DEFAULT_PRODUCT = bag_data.REPOSITORY / "MIL-CREDA"
SEPARABILITY_SUMMARY = Path("Results") / "separability" / "summary.json"
SEPARABILITY_TABLE = Path("Results") / "separability" / "tables" / "separability.csv"
#: Sec. 4.6: one report seed, seed 1 (the search seed is 0, Sec. 7.7); one source of truth.
REPORT_SEEDS = search.REPORT_SEEDS
#: Inputs plan v42 retired: each stood in for a decision v42 now takes.
RETIRED_INPUTS = {
    "seedReduction": "one report seed, no reduction between seeds (Sec. 4.6)",
    "k1": "K1's references, arms and contrasts are declared (Sec. 6.1); no band ratio",
    "pendingDecisions": "the comparator decisions are resolved (Sec. 7.7)",
}
HISTORY_FILE = Path("runs") / "history.jsonl"


class CampaignError(RuntimeError):
    """The campaign was asked for something the plan or its open decisions do not allow."""


# ------------------------------------------------------------------ configuration


def _require(config: Mapping, key: str):
    if key not in config or config[key] is None:
        raise CampaignError(f"the campaign config needs {key!r}; it has no default")
    return config[key]


def check_config(config: Mapping) -> dict:
    """The required inputs, the rehearsal overrides and the retired inputs."""
    cfg = dict(config)
    retired = [k for k in RETIRED_INPUTS if k in cfg]
    if retired:
        raise CampaignError("plan v42 retired " + "; ".join(
            f"{k!r} ({RETIRED_INPUTS[k]})" for k in retired) + ": remove it from the config")
    cfg["schedule"] = search.check_schedule(_require(cfg, "schedule"))
    strategies = list(_require(cfg, "strategies"))
    unknown = [s for s in strategies if s not in model_store.STRATEGIES]
    if unknown or "MIL-CREDA" not in strategies:
        raise CampaignError(f"strategies are among {list(model_store.STRATEGIES)} and include "
                            f"MIL-CREDA; unknown {unknown}")
    cfg["strategies"] = strategies
    names = {t.name: t for t in assignment_mod.TRANSFERS}
    transfers = list(_require(cfg, "transfers"))
    if any(t not in names for t in transfers):
        raise CampaignError(f"transfers are among {list(names)}; got {transfers}")
    cfg["transfers"] = transfers
    cfg.setdefault("nTrials", search.N_TRIALS)
    seeds = [int(s) for s in cfg.get("reportSeeds", REPORT_SEEDS)]
    if search.SEARCH_SEED in seeds:
        raise CampaignError("the search seed is never a report seed (Sec. 7.7)")
    if seeds != list(REPORT_SEEDS):
        raise CampaignError(f"Sec. 4.6 has one report seed, {list(REPORT_SEEDS)}; got {seeds}")
    cfg["reportSeeds"] = seeds
    cfg.setdefault("bootstrapResamples", st.BOOTSTRAP_RESAMPLES)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("synthetic", False)
    return cfg


def transfers_of(cfg: Mapping) -> list[assignment_mod.Transfer]:
    names = {t.name: t for t in assignment_mod.TRANSFERS}
    return [names[t] for t in cfg["transfers"]]


def core_transfers(cfg: Mapping, product: Path) -> dict[str, list[str]]:
    if cfg.get("core"):
        return {arm: list(v) for arm, v in cfg["core"].items()}
    path = product / SEPARABILITY_SUMMARY
    if not path.exists():
        raise CampaignError(f"no core transfers given and no measured separability at {path}")
    return {arm: list(v) for arm, v in json.loads(path.read_text())["core"].items()}


def register(strategies: Sequence[str]) -> None:
    """Each comparator's own explicit registration (it changes global search state)."""
    from MIL_CREDA_EEG.comparators import adaptmil, clust, hasd
    from MIL_CREDA_EEG.comparators.casual import training as casual
    from MIL_CREDA_EEG.comparators.cca_lsc import training as cca_lsc
    from MIL_CREDA_EEG.comparators.cot import training as cot
    from MIL_CREDA_EEG.comparators.cs_ccs import training as cs_ccs
    from MIL_CREDA_EEG.comparators.uda_dda import training as uda_dda
    registrars = {
        "AdaptMIL": adaptmil.register, "HASD": hasd.register, "COT": cot.register,
        "CCA-LSC": cca_lsc.register, "UDA-DDA": uda_dda.register, "CS+CCS": cs_ccs.register,
        "CLUST — variante source-free": clust.register,
        "CASUAL": casual.register,
    }
    for s in strategies:
        if s != "MIL-CREDA" and s not in search.PHASE_TWO_TRAINERS:
            registrars[s]()


def load_bags(cfg: Mapping) -> bag_data.BagSet:
    return bag_data.load_bags(cfg.get("processedRoot"), cfg.get("partitionPath"))


def _partition_path(cfg: Mapping) -> Path:
    return Path(cfg.get("partitionPath") or bag_data.default_partition_path())


def _models(product: Path) -> Path:
    return product / "Models" / "Balladeer"


def _results(product: Path) -> Path:
    return product / "Results"


# ------------------------------------------------------------------ studies


def _study(path: Path, n_trials: int):
    import optuna
    if not path.exists():
        raise CampaignError(f"{path} does not exist: run the search step first")
    study = optuna.load_study(study_name=path.stem, storage=f"sqlite:///{path}")
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) != n_trials:
        raise CampaignError(f"{path} holds {len(complete)} completed trials of {n_trials}: a "
                            f"study is never extended silently; remove it to search again")
    return study


def phase_one_study(group: str, data: search.SearchData, cfg: Mapping, product: Path):
    path = search.study_path(product, data.transfer, 1, group)
    if path.exists():
        return _study(path, cfg["nTrials"])
    return search.run_phase_one_study(group, data, cfg["schedule"], root=product,
                                      n_trials=cfg["nTrials"], device=cfg["device"])


def phase_two_study(strategy: str, data: search.SearchData, records: Mapping, cfg: Mapping,
                    product: Path):
    path = search.study_path(product, data.transfer, 2, strategy)
    if path.exists():
        return _study(path, cfg["nTrials"])
    return search.run_phase_two_study(strategy, data, records, cfg["schedule"], root=product,
                                      n_trials=cfg["nTrials"], device=cfg["device"])


def _groups(strategies: Sequence[str]) -> list[str]:
    return sorted({search.PHASE_ONE_GROUP_OF[s] for s in strategies},
                  key=model_store.GROUPS.index)


def _phase_one_checkpoint(group: str, data: search.SearchData, rotation: int, seed: int,
                          params: Mapping, cfg: Mapping, product: Path) -> tuple[Path, list]:
    """The stored phase-1 checkpoint of `group` at `seed`; trained once when absent."""
    path = model_store.model_path(_models(product), data.transfer, rotation, seed, 1, group)
    if path.exists():
        return path, []
    hyper = search.phase_one_hyper(group, params, cfg["schedule"])
    trained = search.train_phase_one(group, data, rotation, seed, hyper, cfg["device"])
    search.save_phase_one(_models(product), group, data, rotation, seed, trained)
    return path, trained["result"]["history"]


def _append_history(product: Path, experiment: str, records: Sequence[Mapping]) -> None:
    if not records:
        return
    path = _results(product) / experiment / HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(results._plain(r), allow_nan=False) + "\n")


def read_history(product: Path, experiment: str) -> list[dict]:
    path = _results(product) / experiment / HISTORY_FILE
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def run_search(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 1: phase-1 studies, their search-seed checkpoints, phase-2 studies."""
    bags = bags or load_bags(cfg)
    register(cfg["strategies"])
    done = {}
    for transfer in transfers_of(cfg):
        data = search.search_data(transfer, bags)
        for group in _groups(cfg["strategies"]):
            study = phase_one_study(group, data, cfg, product)
            for r in sorted(data.rotations):
                _phase_one_checkpoint(group, data, r, search.SEARCH_SEED, study.best_params,
                                      cfg, product)
            done[f"{transfer.name}:phase1:{group}"] = study.best_value
        for strategy in cfg["strategies"]:
            group = search.PHASE_ONE_GROUP_OF[strategy]
            records = {r: model_store.load(model_store.model_path(
                _models(product), transfer, r, search.SEARCH_SEED, 1, group),
                partition_path=_partition_path(cfg)) for r in sorted(data.rotations)}
            study = phase_two_study(strategy, data, records, cfg, product)
            done[f"{transfer.name}:phase2:{strategy}"] = study.best_value
    return done


# ------------------------------------------------------------------ final runs


def _tag(records: Sequence[Mapping], strategy: str, transfer: str, rotation: int,
         seed: int) -> list[dict]:
    return [{**r, "strategy": strategy, "transfer": transfer, "rotation": int(rotation),
             "seed": int(seed)} for r in records]


def _predicted(product: Path, experiment: str) -> set:
    return {(r["strategy"], r["transfer"], r["rotation"], r["seed"])
            for r in results.read_predictions(_results(product), experiment)}


def run_final(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 2: phase 1 and phase 2 at every report seed, with controls and upper bound."""
    bags = bags or load_bags(cfg)
    register(cfg["strategies"])
    device = torch.device(cfg["device"])
    counts = {"trained_phase1": 0, "trained_phase2": 0, "predictions": 0}
    for transfer in transfers_of(cfg):
        experiment = results.EXPERIMENT_OF_ARM[transfer.arm]
        data = search.search_data(transfer, bags)
        a = assignment_mod.assign_from_bags(transfer, bags)
        best_one = {g: phase_one_study(g, data, cfg, product).best_params
                    for g in _groups(cfg["strategies"])}
        best_two = {s: _study(search.study_path(product, transfer, 2, s), cfg["nTrials"])
                    .best_params for s in cfg["strategies"]}
        done = _predicted(product, experiment)
        for r in sorted(data.rotations):
            test = two_phase.bag_tensors(assignment_mod.side_bags(a, bags, "target", r, "test"))
            for seed in cfg["reportSeeds"]:
                out = []
                for group in best_one:
                    path, history = _phase_one_checkpoint(group, data, r, seed, best_one[group],
                                                          cfg, product)
                    counts["trained_phase1"] += bool(history)
                    _append_history(product, experiment, history)
                    key = f"source-only:{group}"
                    if (key, transfer.name, r, seed) in done:
                        continue
                    control = controls.source_only_from_store(
                        path, transfer, test, partition_path=_partition_path(cfg),
                        device=device)
                    if control.get("applicable"):
                        out += _tag(control["predictions"], key, transfer.name, r, seed)
                for strategy in cfg["strategies"]:
                    if (strategy, transfer.name, r, seed) in done:
                        continue
                    group = search.PHASE_ONE_GROUP_OF[strategy]
                    record = model_store.load(model_store.model_path(
                        _models(product), transfer, r, seed, 1, group),
                        partition_path=_partition_path(cfg))
                    path2 = model_store.model_path(_models(product), transfer, r, seed, 2,
                                                   strategy)
                    if path2.exists():
                        model = mechanisms.load_phase_two(
                            path2, partition_path=_partition_path(cfg))[0].to(device)
                    else:
                        trained = search.train_phase_two(strategy, data, r, seed, record,
                                                         best_two[strategy], cfg["schedule"],
                                                         device)
                        search.save_phase_two(_models(product), strategy, data, r, seed,
                                              trained)
                        _append_history(product, experiment, [
                            {**h, "model": h.get("model", f"{strategy}:phase2")}
                            for h in trained["result"]["history"]])
                        counts["trained_phase2"] += 1
                        model = trained["model"]
                    out += _tag(two_phase.predictions(model, test, transfer.name, device),
                                strategy, transfer.name, r, seed)
                if (results.UPPER_BOUND, transfer.name, r, seed) not in done and \
                        "common" in best_one:
                    upper = controls.train_upper_bound(a, bags, r, seed, best_one["common"],
                                                       cfg["schedule"], device)
                    _append_history(product, experiment, upper["history"])
                    out += _tag(controls.upper_bound_predictions(upper, test, device),
                                results.UPPER_BOUND, transfer.name, r, seed)
                results.append_predictions(_results(product), experiment, out)
                counts["predictions"] += len(out)
    return counts


# ------------------------------------------------------------------ statistics


def _assignments(cfg: Mapping, bags: bag_data.BagSet) -> dict[str, assignment_mod.Assignment]:
    return {t.name: assignment_mod.assign_from_bags(t, bags) for t in assignment_mod.TRANSFERS
            if all(bags.holders(c) for c in (t.source, t.target))}


def test_train_ratio(a: assignment_mod.Assignment, b: assignment_mod.Assignment) -> float:
    """Per rotation, target-test over training subjects summed over both directions; mean."""
    ratios = []
    for r in sorted(a.rotations, key=int):
        views = [x.rotation(int(r)) for x in (a, b)]
        test = sum(len(v["target_test"]) for v in views)
        train = sum(len(v["source_train"]) + len(v["target_train"]) for v in views)
        ratios.append(test / train)
    return float(np.mean(ratios))


def _pair_of(core: Sequence[str]) -> tuple[str, str]:
    if len(core) != 2:
        raise CampaignError(f"an arm's core is one pair, two directions; got {core}")
    return core[0], core[1]


def run_statistics(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 4: per arm, the tables of Secs. 5.1-5.3 and the contrasts of Sec. 4.6."""
    bags = bags or load_bags(cfg)
    core = core_transfers(cfg, product)
    assignments = _assignments(cfg, bags)
    root = _results(product)
    written = {}
    for arm in sorted({t.arm for t in transfers_of(cfg)}):
        experiment = results.EXPERIMENT_OF_ARM[arm]
        records = results.read_predictions(root, experiment)
        if not records:
            continue
        first, second = _pair_of(core[arm])
        pooled = assignment_mod.pair_pool(assignments[first], assignments[second])
        methods = sorted({r["strategy"] for r in records})
        pooled_pair = {m: st.pool_pair(results._select(records, m, (first, second)), pooled)
                       for m in methods if results._select(records, m, (first, second))}
        ratio = test_train_ratio(assignments[first], assignments[second])
        scope = "|".join(assignments[first].transfer.pair)
        rows = []
        for rival in [m for m in results.METHOD_ROWS if m in pooled_pair
                      and m not in ("MIL-CREDA", results.UPPER_BOUND)]:
            rows += st.pair_contrast(pooled_pair["MIL-CREDA"], pooled_pair[rival],
                                     name_a="MIL-CREDA", name_b=results.METHOD_ROWS[rival],
                                     scope=scope, test_train_ratio=ratio, hypothesis=experiment,
                                     experiment=experiment,
                                     resamples=cfg["bootstrapResamples"])
        boot = [r for r in rows if r["test"] == st.TEST_BOOTSTRAP]
        st.apply_holm(boot, f"{experiment} MIL-CREDA vs rivals, bootstrap")
        scores = results.ranking_scores(records, core[arm])
        ranking = st.friedman_nemenyi(scores) if len(scores) >= 3 else None
        if ranking:
            rows.append(st.significance_row(
                comparison=f"{experiment}:{scope}:ranking of the adapted strategies",
                test=st.TEST_FRIEDMAN, statistic=ranking["statistic"],
                p_value=ranking["p_value"], effect_size=ranking["average_rank"]["MIL-CREDA"],
                effect_size_name="MIL-CREDA average rank (1 = best)", ci_low=None,
                ci_high=None, n=ranking["blocks"], hypothesis=experiment,
                experiment=experiment, critical_difference=ranking["critical_difference"],
                methods=ranking["k"], mil_creda_position=ranking["position"]["MIL-CREDA"]))
        results.write_significance(root, experiment, rows)
        results.write_table(root, experiment, "transfer_detail",
                            results.transfer_detail(records, arm))
        results.write_table(root, experiment, "comparison",
                            results.comparison_table(records, arm, core[arm], ranking))
        results.write_table(root, experiment, "secondary",
                            results.secondary_table(pooled_pair, arm))
        results.write_table(root, experiment, "per_fold", results.per_fold_table(pooled_pair))
        results.write_table(root, experiment, "ranking", results.ranking_table(ranking))
        summary = {"pair": scope, "core": core[arm],
                   "bootstrapResamples": cfg["bootstrapResamples"], "nTrials": cfg["nTrials"],
                   "reportSeeds": cfg["reportSeeds"], "partition_sha256": bags.partition_sha256,
                   "pooled_pair_subjects": len(pooled), "test_train_ratio": ratio,
                   "pooled_balanced_accuracy": {
                       m: results._pooled(rs)["balanced_accuracy"]
                       for m, rs in pooled_pair.items()},
                   "mil_creda_position": ranking["position"]["MIL-CREDA"] if ranking else None,
                   "methods_present": methods}
        results.write_summary(root, experiment, summary, synthetic=cfg["synthetic"])
        written[experiment] = len(rows)
    return written


# ------------------------------------------------------------------ mechanisms


def _phase_two_path(product: Path, transfer, r: int, seed: int, strategy: str) -> Path:
    return model_store.model_path(_models(product), transfer, r, seed, 2, strategy)


def _merged_summary(root: Path, experiment: str) -> dict:
    """An experiment's summary on disk without the keys `write_summary` adds itself."""
    path = root / experiment / "summary.json"
    if not path.exists():
        return {}
    current = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in current.items() if k not in ("experiment", "revision",
                                                          "synthetic")}


def _k3_name(table: str, variant: str) -> str:
    return f"{table}:{variant}"


def _variant_hyper_row(table: str, variant: str, transfer: str, hyper: Mapping) -> dict:
    lacks = aggregation_variants.not_applicable(variant) if table == "aggregation" else ()
    row = {"table": table, "variant": variant, "transfer": transfer}
    for key in (*two_phase.PHASE_ONE_KEYS, *two_phase.PHASE_TWO_KEYS):
        row[key] = NOT_APPLICABLE if key in lacks else hyper[key]
    row["search"] = ("none: full MIL-CREDA's selection for this transfer, gamma = 0 "
                     "(Sec. 6.3)" if variant == aggregation_variants.WITHOUT_CONSENSUS else
                     "none: full MIL-CREDA's selection for this transfer (Sec. 6.3)")
    return row


def k3_plan(cfg: Mapping, core: Mapping[str, Sequence[str]]) -> dict[str, dict]:
    """Per transfer, the K3 variants the k3 step trains: both tables on a core transfer;
    only the global-term-only variant on a declared K2 transfer that is not core (Sec. 6.2
    reads it as MIL-CREDA without its local term)."""
    declared = {t.name for t in mechanisms.K2_DECLARED.values()}
    plan = {}
    for transfer in transfers_of(cfg):
        if transfer.name in core.get(transfer.arm, []):
            plan[transfer.name] = {"aggregation": K3_TABLES["aggregation"]["trained"],
                                   "ablation": K3_TABLES["ablation"]["trained"]}
        elif transfer.name in declared:
            plan[transfer.name] = {"aggregation": (), "ablation": (K2_GLOBAL_ONLY,)}
    return plan


def _save_variant(product: Path, data: search.SearchData, rotation: int, seed: int,
                  phase: int, variant: str, state: Mapping, hyper: Mapping, channels: Mapping,
                  sigma: float, checkpoint: Mapping) -> Path:
    extra = {"variant": variant, "epoch": checkpoint.get("epoch"),
             "valid_balanced_accuracy": checkpoint.get("valid_balanced_accuracy")}
    return model_store.save(_models(product), transfer=data.transfer, rotation=rotation,
                            seed=seed, phase=phase, name=model_store.variant_name(variant),
                            state=state, hyper=hyper, partition_sha256=data.partition_sha256,
                            channels=channels, sigma=sigma, extra=extra)


def _k3_variant_model(product: Path, data: search.SearchData, rotation: int, seed: int,
                      table: str, variant: str, phase_one: Mapping, phase_two: Mapping,
                      cfg: Mapping, device: torch.device, owed: list) -> tuple:
    """(model, history, trained): the stored variant, or trained and stored when absent."""
    ppath = _partition_path(cfg)
    name = model_store.variant_name(variant)
    path2 = model_store.model_path(_models(product), data.transfer, rotation, seed, 2, name)
    if path2.exists():
        return mechanisms.load_phase_two(path2, partition_path=ppath)[0].to(device), [], False
    if table == "aggregation":
        path1 = model_store.model_path(_models(product), data.transfer, rotation, seed, 1,
                                       name)
        record = model_store.load(path1, partition_path=ppath) if path1.exists() else None
        trained = mechanisms.train_aggregation_variant(
            data, rotation, seed, variant, phase_one, phase_two, cfg["schedule"], device,
            phase_one_record=record)
        if record is None:
            checkpoint = trained["phase1"]["checkpoint"]
            _save_variant(product, data, rotation, seed, 1, variant, checkpoint["state"],
                          trained["hyper"], trained["channels"], checkpoint["sigma"],
                          checkpoint)
    else:
        path1 = model_store.model_path(_models(product), data.transfer, rotation, seed, 1,
                                       two_phase.MODEL)
        if not path1.exists():
            owed.append(f"no stored MIL-CREDA phase-1 checkpoint for {data.transfer.name} "
                        f"r{rotation} s{seed}: run the final step")
            return None, [], False
        trained = mechanisms.train_ablation_variant(
            data, rotation, seed, model_store.load(path1, partition_path=ppath), phase_two,
            cfg["schedule"], variant, device)
    checkpoint = trained["result"]["checkpoint"]
    _save_variant(product, data, rotation, seed, 2, variant, trained["model"].state_dict(),
                  checkpoint["hyper"], trained["channels"], checkpoint["sigma"], checkpoint)
    return trained["model"], trained["history"], True


def run_k3(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 3: Sec. 6.3's variants on the core transfers, their tables and contrasts."""
    bags = bags or load_bags(cfg)
    device = torch.device(cfg["device"])
    core = core_transfers(cfg, product)
    root = _results(product)
    counts = {"trained_aggregation": 0, "trained_ablation": 0, "predictions": 0}
    owed, hyper_rows = [], []
    done = _predicted(product, K3)
    plan = k3_plan(cfg, core)
    for transfer in transfers_of(cfg):
        if transfer.name not in plan:
            continue
        data = search.search_data(transfer, bags)
        a = assignment_mod.assign_from_bags(transfer, bags)
        phase_one = _study(search.study_path(product, transfer, 1, "MIL-CREDA"),
                           cfg["nTrials"]).best_params
        phase_two = _study(search.study_path(product, transfer, 2, "MIL-CREDA"),
                           cfg["nTrials"]).best_params
        hyper_rows += [_variant_hyper_row("aggregation", v, transfer.name,
                                          mechanisms.aggregation_hyper(
                                              phase_one, phase_two, cfg["schedule"], v))
                       for v in plan[transfer.name]["aggregation"]]
        hyper_rows += [_variant_hyper_row("ablation", v, transfer.name,
                                          {**phase_one, **phase_two})
                       for v in plan[transfer.name]["ablation"]]
        for r in sorted(data.rotations):
            test = two_phase.bag_tensors(assignment_mod.side_bags(a, bags, "target", r, "test"))
            for seed in cfg["reportSeeds"]:
                out, history = [], []
                for table in K3_TABLES:
                    for variant in plan[transfer.name][table]:
                        name = _k3_name(table, variant)
                        stored = model_store.model_path(_models(product), transfer, r, seed, 2,
                                                        model_store.variant_name(variant))
                        if (name, transfer.name, r, seed) in done and stored.exists():
                            continue
                        model, records, trained = _k3_variant_model(
                            product, data, r, seed, table, variant, phase_one, phase_two, cfg,
                            device, owed)
                        if model is None:
                            continue
                        counts[f"trained_{table}"] += trained
                        history += _tag(records, name, transfer.name, r, seed)
                        if (name, transfer.name, r, seed) not in done:
                            out += _tag(two_phase.predictions(model, test, transfer.name,
                                                              device),
                                        name, transfer.name, r, seed)
                _append_history(product, K3, history)
                results.append_predictions(root, K3, out)
                counts["predictions"] += len(out)
    written = k3_tables(cfg, product, bags, core, owed)
    if hyper_rows:
        results.write_table(root, K3, "variant_hyperparameters", hyper_rows)
    history = read_history(product, K3)
    if history:
        results.write_history(root, K3, history)
    recorded = results.read_predictions(root, K3)
    trained = {t: [v for v in spec["trained"] if any(
        r["strategy"] == _k3_name(t, v) for r in recorded)] for t, spec in K3_TABLES.items()}
    summary = _merged_summary(root, K3)
    summary.update({
        "aggregation_variants": {"trained": trained["aggregation"],
                                 "full": "own = full MIL-CREDA (final step)",
                                 "forms": {**aggregation_variants.FORMS,
                                           aggregation_variants.WITHOUT_CONSENSUS:
                                               aggregation_variants.WITHOUT_CONSENSUS_FORM},
                                 "criterion": list(K3_TABLES["aggregation"]["criterion"]),
                                 "not_applicable": {
                                     v: list(aggregation_variants.not_applicable(v))
                                     for v in aggregation_variants.VARIANTS},
                                 "excluded": {"max": "no per-instance weights for the bag "
                                                     "kernel (Secs. 6.3 and 9)"}},
        "ablation_variants": {"trained": trained["ablation"],
                              "full": "both terms = full MIL-CREDA (final step)",
                              "phase_one": "full MIL-CREDA's stored checkpoint",
                              "k2_transfers": sorted(t for t, v in plan.items()
                                                     if not v["aggregation"])},
        "models": "stored per transfer, rotation and report seed (model_store.VARIANTS)",
        "hyperparameters": "full MIL-CREDA's selection per transfer; no search of their own",
        "pairs": written, "reportSeeds": cfg["reportSeeds"],
        "bootstrapResamples": cfg["bootstrapResamples"], "owed": sorted(set(owed))})
    results.write_summary(root, K3, summary, synthetic=cfg["synthetic"])
    return {**counts, "pairs": written, "owed": sorted(set(owed))}


def k3_tables(cfg: Mapping, product: Path, bags: bag_data.BagSet,
              core: Mapping[str, Sequence[str]], owed: list) -> list[str]:
    """Per core pair, both directions pooled: the two tables and their contrasts."""
    root = _results(product)
    records = results.read_predictions(root, K3)
    assignments = _assignments(cfg, bags)
    rows = {t: [] for t in K3_TABLES}
    contrasts = {t: [] for t in K3_TABLES}
    pairs = []
    for arm in sorted({t.arm for t in transfers_of(cfg)}):
        if arm not in core:
            continue
        first, second = _pair_of(core[arm])
        if first not in assignments or second not in assignments:
            continue
        full = results._select(results.read_predictions(root, results.EXPERIMENT_OF_ARM[arm]),
                               "MIL-CREDA", (first, second))
        if not full:
            owed.append(f"no full MIL-CREDA predictions for {first} and {second}: run the "
                        f"final step")
            continue
        pooled = assignment_mod.pair_pool(assignments[first], assignments[second])
        scope = "|".join(assignments[first].transfer.pair)
        ratio = test_train_ratio(assignments[first], assignments[second])
        for table, spec in K3_TABLES.items():
            predictions = {}
            for variant in (*spec["trained"], spec["full"]):
                mine = full if variant == spec["full"] else results._select(
                    records, _k3_name(table, variant), (first, second))
                if mine:
                    predictions[variant] = st.pool_pair(mine, pooled)
            order = [v for v in (mechanisms.AGGREGATIONS if table == "aggregation"
                                 else mechanisms.ABLATIONS) if v in predictions]
            rows[table] += [{"pair": scope, **row} for row in
                            mechanisms.variant_table({v: predictions[v] for v in order})]
            contrasts[table] += mechanisms.k3_contrasts(
                table, scope, spec["full"], predictions,
                [v for v in spec["criterion"] if v in predictions], test_train_ratio=ratio,
                resamples=cfg["bootstrapResamples"])
        pairs.append(scope)
    significance = []
    for table, spec in K3_TABLES.items():
        boot = [r for r in contrasts[table] if r["test"] == st.TEST_BOOTSTRAP]
        st.apply_holm(boot, f"{K3} {table}: {spec['full']} vs "
                            f"{', '.join(spec['criterion'])}, bootstrap")
        significance += contrasts[table]
        if rows[table]:
            results.write_table(root, K3, table, rows[table])
    if significance:
        results.write_significance(root, K3, significance)
    return pairs


def _k1_models(product: Path, transfer, r: int, seed: int,
               strategies: Sequence[str]) -> list[tuple[str, Path]]:
    """Sec. 6.1's attentions on one core transfer and rotation: (K1 name, stored path)."""
    models = [(mechanisms.K1_OWN, _phase_two_path(product, transfer, r, seed, "MIL-CREDA"))]
    for name in (mechanisms.K1_WITHOUT_CONSENSUS, *mechanisms.K1_REFERENCE_VARIANTS):
        models.append((name, _phase_two_path(product, transfer, r, seed, name)))
    for strategy in strategies:
        if strategy in COMPARATORS:
            models.append((mechanisms.common_attention_name(strategy),
                           _phase_two_path(product, transfer, r, seed, strategy)))
        elif strategy in ("AdaptMIL", "HASD"):
            models.append((strategy, _phase_two_path(product, transfer, r, seed, strategy)))
    return models


def run_mechanisms(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 5: K1, K2 and the K3 bound control, from stored models and histories."""
    bags = bags or load_bags(cfg)
    register(cfg["strategies"])
    core = core_transfers(cfg, product)
    root = _results(product)
    ppath = _partition_path(cfg)
    owed, out = [], {}
    k1_rows: dict[str, list] = {}
    k2_full, k2_no_local, k2_other = [], [], []
    for transfer in transfers_of(cfg):
        a = assignment_mod.assign_from_bags(transfer, bags)
        is_core = transfer.name in core.get(transfer.arm, [])
        for r in sorted(bags.rotations, key=int):
            r = int(r)
            for seed in cfg["reportSeeds"]:
                path = _phase_two_path(product, transfer, r, seed, "MIL-CREDA")
                if not path.exists():
                    owed.append(f"no stored MIL-CREDA model for {transfer.name} r{r} s{seed}")
                    continue
                model = mechanisms.load_phase_two(path, partition_path=ppath)[0]
                if is_core:
                    test = assignment_mod.side_bags(a, bags, "target", r, "test")
                    for name, mpath in _k1_models(product, transfer, r, seed,
                                                  cfg["strategies"]):
                        if not mpath.exists():
                            owed.append(f"no stored {name} model for K1 at {transfer.name} "
                                        f"r{r} s{seed}")
                            continue
                        attention = model if name == mechanisms.K1_OWN else \
                            mechanisms.load_phase_two(mpath, partition_path=ppath)[0]
                        k1_rows.setdefault(name, []).extend(mechanisms.k1_subject_rows(
                            attention, test, model_name=name, transfer=transfer, rotation=r,
                            seed=seed))
                k2_full += mechanisms.k2_subject_rows(model, a, bags, rotation=r, seed=seed,
                                                      model_name="MIL-CREDA", metric="kernel")
                npath = _phase_two_path(product, transfer, r, seed, mechanisms.K2_NO_LOCAL)
                if npath.exists():
                    k2_no_local += mechanisms.k2_subject_rows(
                        mechanisms.load_phase_two(npath, partition_path=ppath)[0], a, bags,
                        rotation=r, seed=seed, model_name=mechanisms.K2_NO_LOCAL,
                        metric="kernel")
                elif transfer.name in {t.name for t in mechanisms.K2_DECLARED.values()}:
                    owed.append(f"no stored {mechanisms.K2_NO_LOCAL} model for K2 at "
                                f"{transfer.name} r{r} s{seed}: run the k3 step")
                for strategy in cfg["strategies"]:
                    spath = _phase_two_path(product, transfer, r, seed, strategy)
                    if strategy == "MIL-CREDA" or not spath.exists():
                        continue
                    other = mechanisms.load_phase_two(spath, partition_path=ppath)[0]
                    k2_other += mechanisms.k2_subject_rows(other, a, bags, rotation=r,
                                                           seed=seed, model_name=strategy,
                                                           metric="euclidean")
    own = k1_rows.get(mechanisms.K1_OWN, [])
    if own:
        rows = mechanisms.k1_tests(k1_rows)
        results.write_significance(root, "K1-atencion", rows)
        out["K1-atencion"] = len(rows)
        results.write_table(root, "K1-atencion", "subjects",
                            [r for rs in k1_rows.values() for r in rs])
        results.write_summary(root, "K1-atencion", {
            "subjects": {arm: sum(r["arm"] == arm for r in own)
                         for arm in mechanisms.K1_CONTRASTS},
            "contrasts": {a: list(c) for a, c in mechanisms.K1_CONTRASTS.items()},
            "criterion": mechanisms.K1_CRITERION,
            "criterion_rival": mechanisms.K1_WITHOUT_CONSENSUS,
            "references": sorted(n for n in k1_rows
                                 if n not in (mechanisms.K1_OWN,
                                              mechanisms.K1_WITHOUT_CONSENSUS)),
            "common_attention": "one reference per stored non-MIL comparator "
                                "(mechanisms.common_attention_name)",
            "excluded": mechanisms.K1_EXCLUDED, "reportSeeds": cfg["reportSeeds"],
            "event_marks_recorded": any(r["event_minus_rest"] is not None for r in own),
            "subjects_without_marked_windows": sum(r["marked_minus_clean"] is None
                                                   for r in own)}, synthetic=cfg["synthetic"])
    if k2_full:
        rows = mechanisms.k2_tests(k2_full, k2_no_local)
        results.write_significance(root, "K2-emparejamiento", rows)
        results.write_table(root, "K2-emparejamiento", "separation",
                            k2_full + k2_no_local + k2_other)
        results.write_summary(root, "K2-emparejamiento", {
            "probe_subjects": len({(r["transfer"], r["subject"]) for r in k2_full}),
            "declared": {k: v.name for k, v in mechanisms.K2_DECLARED.items()},
            "transfers": sorted({r["transfer"] for r in k2_full}),
            "no_local_variant": mechanisms.K2_NO_LOCAL,
            "no_local_transfers": sorted({r["transfer"] for r in k2_no_local}),
            "reportSeeds": cfg["reportSeeds"]}, synthetic=cfg["synthetic"])
        out["K2-emparejamiento"] = len(rows)
    history = [h for arm in results.EXPERIMENT_OF_ARM.values()
               for h in read_history(product, arm)]
    try:
        bounds = mechanisms.bound_control(history)
        results.write_table(root, "K3-ablacion", "term_bounds", bounds)
        current = _merged_summary(root, "K3-ablacion")      # the k3 step's keys survive
        results.write_summary(root, "K3-ablacion", {
            "term_bounds": bounds,
            "aggregation_variants": current.pop("aggregation_variants", "not trained (owed)"),
            "ablation_variants": current.pop("ablation_variants", "not trained (owed)"),
            **{k: v for k, v in current.items() if k != "term_bounds"}},
            synthetic=cfg["synthetic"])
        out["K3-ablacion"] = len(bounds)
    except mechanisms.NotRecorded as error:
        owed.append(str(error))
    out["owed"] = sorted(set(owed))
    return out


# ------------------------------------------------------------------ Sec. 5.4


def run_results(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 6: Sec. 5.4's figures and tables, and Sec. 4.5's two selection rules."""
    root = _results(product)
    core = core_transfers(cfg, product)
    written, owed = [], []
    for arm in sorted({t.arm for t in transfers_of(cfg)}):
        experiment = results.EXPERIMENT_OF_ARM[arm]
        records = results.read_predictions(root, experiment)
        hp, importance, selection, studies = [], [], [], {}
        for transfer in [t for t in transfers_of(cfg) if t.arm == arm]:
            one = {g: _study(search.study_path(product, transfer, 1, g), cfg["nTrials"])
                   for g in _groups(cfg["strategies"])}
            two = {s: _study(search.study_path(product, transfer, 2, s), cfg["nTrials"])
                   for s in cfg["strategies"]}
            hp += results.hyperparameter_rows(one, two, transfer.name)
            for s, study in two.items():
                primary = results._pooled(results._select(records, s, [transfer.name]))
                selection.append(results.selection_rows(
                    study, s, transfer.name,
                    test_primary=primary["balanced_accuracy"] if primary else None))
                try:
                    importance += results.importance_rows(study, s)
                except (RuntimeError, ValueError) as error:
                    owed.append(f"fANOVA {study.study_name}: {error}")
            if transfer.name in core.get(arm, []):
                studies.update({f"{s} ({transfer.name})": st_ for s, st_ in two.items()})
        results.write_table(root, experiment, "hyperparameters", hp)
        results.write_table(root, experiment, "selection_rules", selection, note=(
            "The label-free rule's test balanced accuracy is empty until its own chosen "
            "configuration has final models; the plan says no extra run is needed, and the "
            "configurations it would need are listed here."))
        if importance:
            results.write_table(root, experiment, "importance", importance)
        history = read_history(product, experiment)
        if history:
            written.append(results.write_history(root, experiment, history))
        try:
            written.append(results.convergence_figure(root, experiment, history, core[arm]))
        except results.ResultsError as error:
            owed.append(f"{experiment} convergence: {error}")
        if studies:
            written.append(results.optimization_history_figure(root, experiment, studies))
        mil_name = next((k for k in studies if k.startswith("MIL-CREDA")), None)
        if mil_name is not None:
            study = studies[mil_name]
            ranked = [r["parameter"] for r in importance if r["study"] == study.study_name][:2]
            if len(ranked) == 2:
                written.append(results.contour_figure(root, experiment, study, ranked))
            else:
                owed.append(f"{experiment} contour: no fANOVA ranking of two hyperparameters "
                            f"for {study.study_name}")
        scores = results.ranking_scores(records, core.get(arm, []))
        if len(scores) >= 3:
            written.append(results.critical_difference_figure(
                root, experiment, st.friedman_nemenyi(scores)))
        else:
            owed.append(f"{experiment} critical-difference diagram: {len(scores)} adapted "
                        f"strategies with predictions on every core transfer; Friedman needs 3")
    table = Path(cfg.get("separabilityTable") or product / SEPARABILITY_TABLE)
    if table.exists():
        rows = pd.read_csv(table).to_dict("records")
        records = [r for e in ("A-nivel", "B-tarea") for r in results.read_predictions(root, e)]
        points = results.ladder_points(records, rows)
        results.write_table(root, "A-nivel", "domain_shift_ladder", points)
        device = [r["pair_auc"] for r in rows if r["arm"] == "device" and r["reading"] == "all"]
        written.append(results.ladder_figure(root, points, device[0] if device else None))
    else:
        owed.append(f"no separability table at {table}: the domain-shift ladder is not drawn")
    return {"written": [str(p) for p in written], "owed": sorted(set(owed))}


RUNNERS = {"search": run_search, "final": run_final, "k3": run_k3,
           "statistics": run_statistics, "mechanisms": run_mechanisms, "results": run_results}


def run(step: str, config: Mapping, product: str | Path = DEFAULT_PRODUCT) -> dict:
    cfg = check_config(config)
    product = Path(product)
    steps = STEPS if step == "all" else (step,)
    if any(s not in RUNNERS for s in steps):
        raise CampaignError(f"step is one of {list(STEPS)} or 'all'; got {step!r}")
    bags = load_bags(cfg)
    return {s: RUNNERS[s](cfg, product, bags) for s in steps}


# ------------------------------------------------------------------ synthetic rehearsal


def write_synthetic_set(out: str | Path, *, seed: int = 0, subjects: int = 32,
                        windows: int = 4, channels: int = 3,
                        conditions: tuple[str, str] = ("L1", "L6"),
                        arm: str = "level") -> dict:
    """A tiny seeded SYNTHETIC processed set of one pair, its partition and a config."""
    out = Path(out)
    signals = out / "processed" / "signals"
    signals.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(subjects):
        subject, label, sex = f"SYN{i:03d}", ("yes", "no")[i % 2], ("M", "F")[(i // 2) % 2]
        held = ((conditions[0],), (conditions[1],), conditions)[i % 3]
        for condition in held:
            record = f"{subject}-{condition}"
            signal = rng.normal(size=(windows, channels, 512)).astype(np.float32)
            signal += np.float32(0.25 if label == "yes" else 0.0) * np.sin(
                np.linspace(0, 12 * np.pi, 512, dtype=np.float32))
            integrity = np.zeros(windows, dtype=bool)
            integrity[rng.integers(0, windows)] = bool(rng.integers(0, 2))
            np.savez(signals / f"{record}.npz", signal=signal,
                     channels=np.array([f"c{k}" for k in range(channels)]),
                     integrity_mark=integrity,
                     contact_bad=np.zeros((windows, channels), dtype=bool))
            cond, device = bag_data.CONDITIONS[condition]
            rows.append({"record_id": record, "subject": subject, "condition": cond,
                         "device": device, "label": label, "sex": sex, "excluded": False,
                         "npz": f"signals/{record}.npz", "n_windows": windows})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "processed" / "manifest.csv", index=False)
    doc = partition.build_partition(partition.survivors(manifest))
    doc["synthetic"] = True
    doc[partition.HASH_FIELD] = partition.partition_hash(doc)
    ppath = out / "partition.json"
    ppath.write_bytes(partition.serialize(doc))
    pair = [f"{conditions[0]}->{conditions[1]}", f"{conditions[1]}->{conditions[0]}"]
    config = {"synthetic": True, "processedRoot": str(out / "processed"),
              "partitionPath": str(ppath),
              "schedule": {"epochs1": 2, "patience1": 1, "epochs2": 2, "bagsPerStep": 4},
              "strategies": ["MIL-CREDA", "COT", "HASD"], "transfers": pair,
              "core": {arm: pair}, "nTrials": 4, "reportSeeds": list(REPORT_SEEDS),
              "bootstrapResamples": 500}
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False))
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m MIL_CREDA_EEG.campaign",
                                     description="The BALLADEER campaign of plan v42, local only.")
    parser.add_argument("step", choices=(*STEPS, "all", "synthetic"))
    parser.add_argument("--config", help="campaign config (JSON)")
    parser.add_argument("--product", default=str(DEFAULT_PRODUCT),
                        help="product folder holding Results/ and Models/ (default MIL-CREDA/)")
    parser.add_argument("--out", help="synthetic: where to write the synthetic set")
    parser.add_argument("--seed", type=int, default=0, help="synthetic: generator seed")
    args = parser.parse_args(argv)
    if args.step == "synthetic":
        if not args.out:
            parser.error("synthetic needs --out")
        write_synthetic_set(args.out, seed=args.seed)
        print(json.dumps({"synthetic": str(Path(args.out) / "config.json")}))
        return 0
    if not args.config:
        parser.error("this step needs --config")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        outcome = run(args.step, config, args.product)
    except (CampaignError, search.SearchError, mechanisms.MechanismError) as error:
        print(json.dumps({"status": "refused", "error": type(error).__name__,
                          "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(results._plain(outcome), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
