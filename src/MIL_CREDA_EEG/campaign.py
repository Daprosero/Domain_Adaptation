"""The campaign: the plan's declared steps, wired to runnable local commands.

Experiments plan v60, Secs. 4.2, 4.5, 4.6, 5.1-5.4, 6.1-6.3, 7.3, 7.5 and 7.7.
The steps, in order (`STEPS`), each one a subcommand of
`python -m MIL_CREDA_EEG.campaign <step> --config <file> [--product <dir>]`:

1. `search` -- the nested cross-validation's inner level, per transfer and outer
   fold r (test fold k, non-test folds F): the phase-1 study of each group a
   requested strategy starts from, trained on the three inner splits of F
   (`search.run_phase_one_study`); its selected (configuration, E1)
   (`search.select`) refit on the source subjects of all of F for exactly E1
   epochs at the report seed (`search.PHASE_ONE_SEED`) and stored, the one phase-1
   checkpoint of that group and outer fold; then one phase-2 study per strategy,
   every trial starting from that checkpoint in each inner split, at the search
   seed, and recording its identity (`search.run_phase_two_study`). An
   existing study of the same identity is continued up to its declared count, or
   read back when complete; a study of another identity is refused naming what
   differs (`search.open_study`).
2. `final` -- per transfer and outer fold r, at the one report seed (Sec. 4.6,
   seed 1): each group's one phase-1 checkpoint of r, the one the search step
   stored and its phase-2 studies started from (refused when it is not), its
   source-only control on the target test fold (`controls`; the device arm records
   `NA_RECORD`), each strategy's selected phase-2 (configuration, E2) of r refit
   from that same checkpoint on the target bags of F for exactly E2 epochs, with no
   validation, and stored with the selection that chose it and the checkpoint's
   identity, its target test predictions, and
   the target-supervised upper bound with the common group's selected `lr1`,
   `wd1` of r (`controls.train_upper_bound`). Where the EnsV secondary rule
   selects another (trial, epoch) for r, that one is refit the same way, at the
   report seed, and stored apart (`Models/Balladeer/ensv-rule/`), its per-epoch
   records go to `runs/history_ensv.jsonl` and its test predictions are recorded
   apart (`runs/predictions_ensv.jsonl`); where both rules select the same
   (trial, epoch), the primary model is reused and its predictions are recorded
   for EnsV (Sec. 4.5). Neither rule reads a target label, and the test fold of r
   enters no decision. A
   model already stored is not trained again (an interrupted run resumes).
3. `k3` -- Sec. 6.3 on the core transfers, per rotation at the report seed: the
   four aggregation variants (`aggregation_variants`: the own attention without
   consensus, the published and gated attentions and the mean) trained through
   both phases and the two ablation variants (global term only, local term only)
   trained in phase 2 from full MIL-CREDA's stored phase-1 checkpoint, all with
   full MIL-CREDA's selected hyperparameters and epoch counts (E1, E2) of the
   transfer and outer fold, refit on the non-test folds with no validation, and no
   search of their own -- per outer fold, the selection of Sec. 4.5
   (`mechanisms.train_aggregation_variant`,
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
   (`mechanisms.k1_tests`, plan v60: three criterion tests, Holm over the three --
   the task arm's two marked-window tests on its EMOTIV bags and the level arm's
   event test --, the device arm's Robots side as support, every other attention
   as a reference), K2 over every
   transfer's probe subjects with its criterion contrast against the
   global-term-only model, and the K3 implementation control of the term bounds
   from the recorded histories (`mechanisms`).
6. `results` -- the Sec. 5.4 outputs: the per-epoch histories
   (`Results/history_<experiment>.json`), the chosen hyperparameters, fANOVA
   importance per study, the two selection rules per outer fold
   (`selection_rules`, `selected_configurations`), and the declared figures through
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
(`epochs1` and `epochs2`, the search's epoch caps, at most `search.EPOCH_CAP`, and
`bagsPerStep`: Sec. 7.6; each search trial stops on its pooled selection score's patience,
`search.PATIENCE`, plan v60, and a refit has no early stopping) and the strategies and
transfers to run. Plan v42 resolved every decision the earlier inputs stood in
for -- the seed reduction (one report seed), K1's common attention, band edges
and scope (Sec. 6.1), and the open comparator decisions (Sec. 7.7) -- so those
inputs are retired and refused when given (`RETIRED_INPUTS`), never silently
ignored.

**Reuse is checked.** A stored study, model or prediction is reused only when it
was produced under the partition in force, this plan revision (`REVISION`) and the
current code version (`identity`). A stored study must carry its full identity
(`_study_identity`: also its search space, schedule, sampler seed and trial
budget, outer fold and inner splits); a stored phase-1 checkpoint, the
hyperparameters and refit schedule (with its E1) now requested; a stored phase-2
model, the selection (trial and epoch) now in force with its hyperparameters and
refit schedule. A stale one is refused, naming what differs. The
model store itself refuses a model of another plan revision on every load
(`model_store.RevisionMismatch`).

**Full protocol unless rehearsing.** A config that is neither `synthetic` nor
`"rehearsal": true` must cover every strategy (Sec. 7.1), every transfer
(Secs. 5.1-5.3) and the declared trial budget (Sec. 7.7); a reduced run is
refused, so it cannot write tables that read as the full comparison. Every
summary records `rehearsal`.

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
import types
import sys
from collections.abc import Iterable, Mapping, Sequence
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
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v60.md",
    "sections": ["4.2", "4.5", "4.6", "5.1", "5.2", "5.3", "5.4", "6.1", "6.2", "6.3", "7.3",
                 "7.5", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion",
                    "K2-emparejamiento", "K3-ablacion"],
    "invariants": [
        "the_steps_are_the_declared_ones_in_order",
        "required_inputs_have_no_default",
        "every_strategy_runs_without_an_open_decision_guard",
        "the_campaign_never_launches_remote_work",
        "an_incomplete_study_is_refused_not_extended",
        "a_stale_study_is_refused_naming_what_differs",
        "a_stored_study_is_reused_only_under_its_full_identity",
        "a_stored_phase_one_checkpoint_is_reused_only_under_its_hyperparameters",
        "a_stored_phase_two_model_is_reused_only_under_the_requested_selection",
        "stale_predictions_are_refused_naming_what_differs",
        "every_changed_module_answers_to_one_plan_revision",
        "a_reduced_run_is_refused_outside_rehearsal_mode",
        "the_rehearsal_summaries_say_so",
        "both_phases_are_selected_per_outer_fold_and_refit_for_their_epochs",
        "one_phase_one_checkpoint_per_group_and_outer_fold",
        "the_ensv_rule_is_measured_on_the_test_fold",
        "the_ensv_final_models_are_refit_only_where_the_rules_differ",
        "the_ensv_histories_are_written_apart_from_the_strategies",
        "the_two_rules_table_reports_both_choices_and_the_class_proportions",
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
DEFAULT_PRODUCT = bag_data.REPOSITORY / "MIL-CREDA-EEG"
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
#: The EnsV final models' per-epoch records: the primary layout in their own file, so
#: nothing that reads the strategies' history takes them for a strategy.
ENSV_HISTORY_FILE = Path("runs") / "history_ensv.jsonl"
HISTORY_FILES = {"primary": HISTORY_FILE, "ensv": ENSV_HISTORY_FILE}
#: The one plan revision every result, study and model of this campaign answers to.
REVISION = __provenance__["revision"]
#: Sec. 4.5's secondary rule: where its configuration differs from the primary one, its
#: final model is stored apart, under `Models/Balladeer/<ENSV_MODELS>/`.
ENSV_MODELS = "ensv-rule"
#: The rule that chose a stored phase-2 model, as its sidecar records it.
EPOCH_RULES = {"primary": search.PRIMARY_RULE, "ensv": search.ENSV_RULE}


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
    cfg["rehearsal"] = bool(cfg.get("rehearsal", False)) or bool(cfg["synthetic"])
    if "rotations" in cfg:
        rotations = sorted({int(r) for r in cfg["rotations"]})
        if not rotations or not set(rotations) <= set(range(partition.N_FOLDS)):
            raise CampaignError(f"rotations are among {list(range(partition.N_FOLDS))}; "
                                f"got {cfg['rotations']}")
        cfg["rotations"] = rotations
    if not cfg["rehearsal"]:
        short = []
        if "rotations" in cfg:
            short.append(f"rotations: {cfg['rotations']}, the plan runs every outer fold "
                         f"(Sec. 4.5)")
        missing = [s for s in model_store.STRATEGIES if s not in strategies]
        if missing:
            short.append(f"strategies: missing {missing} (Sec. 7.1)")
        missing = [t for t in names if t not in transfers]
        if missing:
            short.append(f"transfers: missing {missing} (Secs. 5.1-5.3)")
        if cfg["nTrials"] != search.N_TRIALS:
            short.append(f"nTrials: {cfg['nTrials']}, the plan declares {search.N_TRIALS} "
                         f"(Sec. 7.7)")
        if short:
            raise CampaignError("a run that is not a rehearsal covers the whole comparison "
                                "the plan declares; " + "; ".join(short) + ". Set "
                                "\"rehearsal\": true for a reduced run: its results are "
                                "labelled rehearsal, never the full comparison")
    return cfg


def rotations_of(cfg: Mapping, rotations: Iterable) -> list[int]:
    """The outer folds of this run: all of `rotations`, or a rehearsal's `rotations`."""
    keep = cfg.get("rotations")
    return sorted(int(r) for r in rotations if keep is None or int(r) in keep)


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
    from MIL_CREDA_EEG.comparators import adaptmil, clust
    from MIL_CREDA_EEG.comparators.casual import training as casual
    from MIL_CREDA_EEG.comparators.cca_lsc import training as cca_lsc
    from MIL_CREDA_EEG.comparators.cluda import training as cluda
    from MIL_CREDA_EEG.comparators.cot import training as cot
    from MIL_CREDA_EEG.comparators.cs_ccs import training as cs_ccs
    from MIL_CREDA_EEG.comparators.uda_dda import training as uda_dda
    registrars = {
        "AdaptMIL": adaptmil.register, "COT": cot.register,
        "CCA-LSC": cca_lsc.register, "UDA-DDA": uda_dda.register, "CS+CCS": cs_ccs.register,
        "CLUST — variante source-free": clust.register,
        "CASUAL": casual.register, "CLUDA": cluda.register,
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


def identity(partition_sha256: str) -> dict:
    """What a stored study, model or prediction must have been produced under to be reused:
    the partition in force, this plan revision and the current code version."""
    return {partition.HASH_FIELD: partition_sha256, "revision": REVISION,
            "code_version": model_store.code_version()}


def _bags_identity(bags: bag_data.BagSet) -> dict:
    return identity(bags.partition_sha256)


def _study(path: Path, n_trials: int, partition_sha256: str, expect: Mapping | None = None):
    """A completed study of the current identity, read back; anything else is refused.
    `expect` is the full identity to match (`_study_identity`); without it, the partition,
    plan revision and code version."""
    import optuna
    if not path.exists():
        raise CampaignError(f"{path} does not exist: run the search step first")
    study = optuna.load_study(study_name=path.stem, storage=f"sqlite:///{path}")
    differ = search.identity_differences(study.user_attrs, expect or identity(partition_sha256))
    if differ:
        raise CampaignError(f"{path} is stale and is not reused: " + "; ".join(differ)
                            + "; remove it to search again")
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) != n_trials:
        raise CampaignError(f"{path} holds {len(complete)} completed trials of {n_trials}: "
                            f"run the search step to continue it up to its declared count")
    return study


def _study_identity(data: search.SearchData, phase: int, name: str, rotation: int,
                    cfg: Mapping) -> dict:
    """The identity a stored study must carry to be read back: every field the search step
    records (`search.study_identity`) -- transfer, outer fold and inner splits, partition,
    plan revision, code version, search space, schedule, sampler and its seed, trial
    budget -- except the objective's wording, which the search step alone states."""
    expected = search.study_identity(data, phase, name, rotation, cfg["schedule"],
                                     cfg["nTrials"], "")
    expected.pop("objective")
    return expected


def _stored_study(product: Path, data: search.SearchData, phase: int, name: str,
                  rotation: int, cfg: Mapping):
    """The completed study of `name` and outer fold `rotation`, refused unless its full
    identity is the current one."""
    return _study(search.study_path(product, data.transfer, phase, name, rotation),
                  cfg["nTrials"], data.partition_sha256,
                  _study_identity(data, phase, name, rotation, cfg))


def _same(a, b) -> bool:
    """Equal as stored JSON (NaN equals NaN, the way a sidecar records it)."""
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True,
                                                                   default=str)


def _reuse(path: Path, current: Mapping, expect: Mapping | None = None,
           hyper: Mapping | None = None) -> Path:
    """A stored model is reused only when its sidecar answers to the current identity, to
    the selection now in force (`expect`, read from the sidecar's `selection`) and to the
    hyperparameters and schedule now requested (`hyper`, read from its `hyper`)."""
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    differ = search.identity_differences(meta, current)
    for key, value in (expect or {}).items():
        stored = (meta.get("selection") or {}).get(key)
        if not _same(stored, value):
            differ.append(f"selection {key}: stored {stored!r}, current {value!r}")
    for key, value in (hyper or {}).items():
        stored = (meta.get("hyper") or {}).get(key)
        if not _same(stored, value):
            differ.append(f"hyper {key}: stored {stored!r}, current {value!r}")
    if differ:
        raise CampaignError(f"{path} is stale and is not reused: " + "; ".join(differ)
                            + "; remove it to train again")
    return path


def _selections(product: Path, data: search.SearchData, phase: int, name: str,
                cfg: Mapping, rule: str = "primary") -> dict[int, dict]:
    """Sec. 4.5: per outer fold, the (configuration, epoch) `rule` selects in its study."""
    return {r: search.select(_stored_study(product, data, phase, name, r, cfg), rule)
            for r in rotations_of(cfg, data.folds)}


def _phase_two_selection(product: Path, data: search.SearchData, strategy: str,
                         cfg: Mapping, rule: str = "primary") -> dict[int, dict]:
    """Sec. 4.5: the phase-2 (configuration, E2) chosen for each outer fold by `rule`."""
    return _selections(product, data, 2, strategy, cfg, rule)


def phase_one_study(group: str, data: search.SearchData, rotation: int, cfg: Mapping,
                    product: Path):
    """The search step: run, continue (same identity) or read back a phase-1 study."""
    return search.run_phase_one_study(group, data, rotation, cfg["schedule"], root=product,
                                      n_trials=cfg["nTrials"], device=cfg["device"])


def phase_two_study(strategy: str, data: search.SearchData, rotation: int, record,
                    cfg: Mapping, product: Path):
    """The search step: run, continue (same identity) or read back a phase-2 study."""
    return search.run_phase_two_study(strategy, data, rotation, record, cfg["schedule"],
                                      root=product, n_trials=cfg["nTrials"],
                                      device=cfg["device"])


def _groups(strategies: Sequence[str]) -> list[str]:
    return sorted({search.PHASE_ONE_GROUP_OF[s] for s in strategies},
                  key=model_store.GROUPS.index)


def _phase_one_checkpoint(group: str, data: search.SearchData, rotation: int,
                          selection: Mapping, cfg: Mapping, product: Path) -> tuple[Path, list]:
    """The one stored phase-1 checkpoint of `group` and outer fold `rotation`, at
    `search.PHASE_ONE_SEED`: the selected configuration refit on the source subjects of
    every non-test fold for exactly its E1 epochs (Sec. 4.5); trained once when absent, its
    per-epoch records appended to the arm's history."""
    seed = search.PHASE_ONE_SEED
    path = model_store.model_path(_models(product), data.transfer, rotation, seed, 1, group)
    schedule = search.refit_schedule(cfg["schedule"], 1, selection["epochs"])
    hyper = search.phase_one_hyper(group, selection["params"], schedule)
    if path.exists():
        stored = {k: hyper[k] for k in (*two_phase.PHASE_ONE_KEYS, *two_phase.SCHEDULE_KEYS)}
        return _reuse(path, identity(data.partition_sha256), hyper=stored), []
    trained = search.train_phase_one(group, data, rotation, seed, hyper, cfg["device"])
    search.save_phase_one(_models(product), group, data, rotation, seed, trained)
    _append_history(product, results.EXPERIMENT_OF_ARM[data.transfer.arm],
                    trained["result"]["history"])
    return path, trained["result"]["history"]


def _stored_phase_one_checkpoint(group: str, data: search.SearchData, rotation: int,
                                 selection: Mapping, cfg: Mapping, product: Path) -> Path:
    """The final step's phase-1 checkpoint of `group` and outer fold `rotation`: the one the
    search step stored and its phase-2 studies started from (Secs. 4.5, 7.3). It is reused
    under the selection in force and never trained here: a missing one is refused."""
    path = model_store.model_path(_models(product), data.transfer, rotation,
                                  search.PHASE_ONE_SEED, 1, group)
    if not path.exists():
        raise CampaignError(f"{data.transfer.name} r{rotation}: no stored {group} phase-1 "
                            f"checkpoint at {path}; the final step refits phase 2 from the "
                            f"checkpoint its study started from and never trains one: run "
                            f"the search step")
    schedule = search.refit_schedule(cfg["schedule"], 1, selection["epochs"])
    hyper = search.phase_one_hyper(group, selection["params"], schedule)
    stored = {k: hyper[k] for k in (*two_phase.PHASE_ONE_KEYS, *two_phase.SCHEDULE_KEYS)}
    return _reuse(path, identity(data.partition_sha256), hyper=stored)


def _phase_one_record(product: Path, data: search.SearchData, group: str, rotation: int,
                      cfg: Mapping):
    """The one stored phase-1 checkpoint (state, meta) of `group` and outer fold `rotation`."""
    return model_store.load(model_store.model_path(_models(product), data.transfer, rotation,
                                                   search.PHASE_ONE_SEED, 1, group),
                            partition_path=_partition_path(cfg))


def _check_started_from(product: Path, data: search.SearchData, strategy: str, rotation: int,
                        record, cfg: Mapping) -> None:
    """Sec. 4.5: the phase-2 final refit of outer fold `rotation` starts from the same
    phase-1 checkpoint its phase-2 study started from; anything else is refused."""
    study = _stored_study(product, data, 2, strategy, rotation, cfg)
    stored = study.user_attrs.get("phaseOneCheckpoint")
    current = search.phase_one_checkpoint_identity(record[1])
    if stored != current:
        path = search.study_path(product, data.transfer, 2, strategy, rotation)
        raise CampaignError(f"{data.transfer.name} r{rotation}: the {strategy} phase-2 study "
                            f"started from phase-1 checkpoint {stored!r} and the stored one is "
                            f"{current!r}; phase 2 is refit from the same checkpoint: remove "
                            f"the stale phase-2 study {path} and run the search step again")


def _measured(record: Mapping) -> dict:
    """A history record without the values that were not measured: a refit has no
    validation set (Sec. 4.5), so its NaN validation series are left out, never written
    as numbers."""
    return {k: v for k, v in record.items()
            if not (isinstance(v, float) and not np.isfinite(v))}


def _append_history(product: Path, experiment: str, records: Sequence[Mapping],
                    rule: str = "primary") -> None:
    if not records:
        return
    path = _results(product) / experiment / HISTORY_FILES[rule]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(results._plain(_measured(r)), allow_nan=False) + "\n")


def read_history(product: Path, experiment: str, rule: str = "primary") -> list[dict]:
    path = _results(product) / experiment / HISTORY_FILES[rule]
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def run_search(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 1: per outer fold, the phase-1 studies, each group's one refit checkpoint, and
    the phase-2 studies from it."""
    bags = bags or load_bags(cfg)
    register(cfg["strategies"])
    done = {}
    for transfer in transfers_of(cfg):
        data = search.search_data(transfer, bags)
        for r in rotations_of(cfg, data.rotations):
            for group in _groups(cfg["strategies"]):
                study = phase_one_study(group, data, r, cfg, product)
                _phase_one_checkpoint(group, data, r, search.select(study), cfg, product)
                done[f"{transfer.name}:phase1:{group}:outer{r}"] = study.best_value
            for strategy in cfg["strategies"]:
                record = _phase_one_record(product, data, search.PHASE_ONE_GROUP_OF[strategy],
                                           r, cfg)
                study = phase_two_study(strategy, data, r, record, cfg, product)
                done[f"{transfer.name}:phase2:{strategy}:outer{r}"] = study.best_value
    return done


# ------------------------------------------------------------------ final runs


def _tag(records: Sequence[Mapping], strategy: str, transfer: str, rotation: int,
         seed: int) -> list[dict]:
    return [{**r, "strategy": strategy, "transfer": transfer, "rotation": int(rotation),
             "seed": int(seed)} for r in records]


def _read_predictions(product: Path, experiment: str, current: Mapping,
                      rule: str = "primary") -> list[dict]:
    """The stored predictions, refused when any was written under another identity."""
    records = results.read_predictions(_results(product), experiment, rule)
    stale = results.stale_predictions(records, current)
    if stale:
        raise CampaignError(f"the stored {rule} predictions of {experiment} are stale and are "
                            f"not reused: " + "; ".join(stale) + "; remove them to run again")
    return records


def _predicted(product: Path, experiment: str, current: Mapping,
               rule: str = "primary") -> set:
    return {(r["strategy"], r["transfer"], r["rotation"], r["seed"])
            for r in _read_predictions(product, experiment, current, rule)}


#: What a stored phase-2 model records of the selection that chose it (Sec. 4.5).
SELECTION_FIELDS = ("rule", "trial", "params", "epoch", "epochs", "testFold",
                    "selectionRotations", "excludedRotation", "score")


def _selection_record(selection: Mapping, schedule: Mapping) -> dict:
    """What a stored phase-2 model records of the selection that chose it (Sec. 4.5), with
    the refit schedule it was trained at, which its reuse is checked against."""
    return {**{k: selection[k] for k in SELECTION_FIELDS}, "schedule": dict(schedule)}


def _ensv_history(history: Sequence[Mapping], strategy: str, epoch: int) -> list[dict]:
    """An EnsV model's per-epoch records: its refit's history, tagged with the rule and
    the epoch it was refit to."""
    return [{**h, "model": h.get("model", f"{strategy}:phase2"), "selectionRule": "ensv",
             "ensvEpoch": int(epoch)} for h in history]


def _same_choice(a: Mapping, b: Mapping) -> bool:
    """Sec. 4.5: the two rules coincide when they select the same trial AND epoch."""
    return (a["trial"], a["epoch"]) == (b["trial"], b["epoch"])


def _phase_two_model(root: Path, strategy: str, data: search.SearchData, r: int, seed: int,
                     record, selection: Mapping, cfg: Mapping, device: torch.device,
                     ppath: Path) -> tuple:
    """(model, history): the stored phase-2 model of this selection, or refit and stored.

    The refit starts from `record`, the one phase-1 checkpoint of this outer fold and group,
    and runs on the target bags of every non-test fold for exactly the selected epoch
    count, with no validation (Sec. 4.5); the model records that checkpoint's identity and
    is reused only under it. A reused model has no history (None)."""
    started = search.phase_one_checkpoint_identity(record[1])
    schedule = search.refit_schedule(cfg["schedule"], 2, selection["epochs"])
    expect = {"rule": selection["rule"], "trial": selection["trial"],
              "params": selection["params"], "epoch": selection["epoch"],
              "schedule": dict(schedule), "epochRule": EPOCH_RULES[selection["rule"]]}
    path = model_store.model_path(root, data.transfer, r, seed, 2, strategy)
    if path.exists():
        stored = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        # the stored hyperparameters must agree on every requested key they hold (a
        # comparator may store its own parameters only)
        requested = {**selection["params"], **schedule}
        _reuse(path, {**identity(data.partition_sha256), "phaseOneCheckpoint": started},
               expect,
               hyper={k: v for k, v in requested.items() if k in (stored.get("hyper") or {})})
        return mechanisms.load_phase_two(path, partition_path=ppath)[0].to(device), None
    trained = search.train_phase_two(strategy, data, r, seed, record, selection["params"],
                                     schedule, device)
    chosen = {**_selection_record(selection, schedule),
              "epochRule": EPOCH_RULES[selection["rule"]]}
    search.save_phase_two(root, strategy, data, r, seed, trained, selection=chosen,
                          phase_one=record[1])
    return trained["model"], trained["result"]["history"]


def run_final(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 2: per outer fold, the phase-2 refits at every report seed from each group's
    one phase-1 checkpoint, with the controls and the upper bound.

    Each strategy's phase-2 (configuration, E2) selected for outer fold r is refit on its
    non-test folds from the phase-1 checkpoint its study started from; where the EnsV
    rule selects another (trial, epoch) for r, that one is refit and stored too
    (`ENSV_MODELS`) and its test predictions recorded apart; where both rules
    coincide, the primary model is reused and its predictions are recorded for it.
    Nothing is selected with test outcomes.
    """
    bags = bags or load_bags(cfg)
    register(cfg["strategies"])
    device = torch.device(cfg["device"])
    ppath = _partition_path(cfg)
    current = _bags_identity(bags)
    counts = {"trained_phase1": 0, "trained_phase2": 0, "predictions": 0,
              "trained_ensv": 0, "ensv_predictions": 0}
    # every checkpoint of every transfer is checked before the step trains or records anything
    plans = {}
    for transfer in transfers_of(cfg):
        data = search.search_data(transfer, bags)
        best_one = {g: _selections(product, data, 1, g, cfg) for g in _groups(cfg["strategies"])}
        plans[transfer.name] = data, best_one, {
            r: {g: _stored_phase_one_checkpoint(g, data, r, best_one[g][r], cfg, product)
                for g in best_one}
            for r in rotations_of(cfg, data.rotations)}
    for transfer in transfers_of(cfg):
        experiment = results.EXPERIMENT_OF_ARM[transfer.arm]
        data, best_one, stored_one = plans[transfer.name]
        a = assignment_mod.assign_from_bags(transfer, bags)
        primary = {s: _phase_two_selection(product, data, s, cfg) for s in cfg["strategies"]}
        free = {s: _phase_two_selection(product, data, s, cfg, "ensv")
                for s in cfg["strategies"]}
        done = _predicted(product, experiment, current)
        done_free = _predicted(product, experiment, current, "ensv")
        free_root = _models(product) / ENSV_MODELS
        for r in rotations_of(cfg, data.rotations):
            test = two_phase.bag_tensors(assignment_mod.side_bags(a, bags, "target", r, "test"))
            checkpoints = stored_one[r]
            records = {}
            for seed in cfg["reportSeeds"]:
                out, out_free = [], []
                for group, path in checkpoints.items():
                    key = f"source-only:{group}"
                    if (key, transfer.name, r, seed) in done:
                        continue
                    control = controls.source_only_from_store(
                        path, transfer, test, partition_path=ppath, device=device)
                    if control.get("applicable"):
                        out += _tag(control["predictions"], key, transfer.name, r, seed)
                for strategy in cfg["strategies"]:
                    key = (strategy, transfer.name, r, seed)
                    coincide = _same_choice(free[strategy][r], primary[strategy][r])
                    need, need_free = key not in done, key not in done_free
                    if not need and not need_free:
                        continue
                    if strategy not in records:
                        records[strategy] = _phase_one_record(
                            product, data, search.PHASE_ONE_GROUP_OF[strategy], r, cfg)
                        _check_started_from(product, data, strategy, r, records[strategy], cfg)
                    record = records[strategy]
                    if need or coincide:
                        model, history = _phase_two_model(
                            _models(product), strategy, data, r, seed, record,
                            primary[strategy][r], cfg, device, ppath)
                        if history is not None:
                            _append_history(product, experiment, [
                                {**h, "model": h.get("model", f"{strategy}:phase2")}
                                for h in history])
                            counts["trained_phase2"] += 1
                        tagged = _tag(two_phase.predictions(model, test, transfer.name, device),
                                      strategy, transfer.name, r, seed)
                        out += tagged if need else []
                        if need_free and coincide:
                            # Sec. 4.5: the same (trial, epoch): the primary model is reused.
                            out_free += [{**x, "ensvModel": "the primary model (same trial "
                                                            "and epoch)"}
                                         for x in tagged]
                    if need_free and not coincide:
                        model, free_history = _phase_two_model(
                            free_root, strategy, data, r, seed, record, free[strategy][r],
                            cfg, device, ppath)
                        counts["trained_ensv"] += free_history is not None
                        _append_history(product, experiment, _ensv_history(
                            free_history or [], strategy, free[strategy][r]["epoch"]),
                            rule="ensv")
                        out_free += [{**x, "ensvModel": "its own"} for x in _tag(
                            two_phase.predictions(model, test, transfer.name, device),
                            strategy, transfer.name, r, seed)]
                if (results.UPPER_BOUND, transfer.name, r, seed) not in done and \
                        "common" in best_one:
                    upper = controls.train_upper_bound(a, bags, r, seed,
                                                       best_one["common"][r]["params"],
                                                       cfg["schedule"], device)
                    _append_history(product, experiment, upper["history"])
                    out += _tag(controls.upper_bound_predictions(upper, test, device),
                                results.UPPER_BOUND, transfer.name, r, seed)
                results.append_predictions(_results(product), experiment, out, identity=current)
                results.append_predictions(_results(product), experiment, out_free,
                                           identity=current, rule="ensv")
                counts["predictions"] += len(out)
                counts["ensv_predictions"] += len(out_free)
    return counts


# ------------------------------------------------------------------ statistics


def _assignments(cfg: Mapping, bags: bag_data.BagSet) -> dict[str, assignment_mod.Assignment]:
    return {t.name: assignment_mod.assign_from_bags(t, bags) for t in assignment_mod.TRANSFERS
            if all(bags.holders(c) for c in (t.source, t.target))}


def test_train_ratio(a: assignment_mod.Assignment, b: assignment_mod.Assignment) -> float:
    """Per rotation, target-test over training subjects summed over both directions; mean.

    The training subjects are the refit's (plan v60, Sec. 4.5): every non-test fold."""
    ratios = []
    for r in sorted(a.rotations, key=int):
        views = [x.rotation(int(r)) for x in (a, b)]
        test = sum(len(v["target_test"]) for v in views)
        train = sum(len(v[f"source_{assignment_mod.REFIT_ROLE}"])
                    + len(v[f"target_{assignment_mod.REFIT_ROLE}"]) for v in views)
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
    current = _bags_identity(bags)
    written = {}
    for arm in sorted({t.arm for t in transfers_of(cfg)}):
        experiment = results.EXPERIMENT_OF_ARM[arm]
        records = _read_predictions(product, experiment, current)
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
        # Sec. 4.6: primary and secondary families, Holm within each metric.
        st.apply_contrast_families(rows, f"{experiment} MIL-CREDA vs rivals")
        # Secs. 5.1-5.3: the arm's ranks are descriptive; the test is global (Sec. 5.4).
        ranking = results.arm_ranking(records, core[arm])
        results.write_significance(root, experiment, rows)
        results.write_table(root, experiment, "significance", results.significance_table(rows))
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
                   "mil_creda_top_three": ranking["top_three"]["MIL-CREDA"] if ranking else None,
                   "methods_present": methods}
        results.write_summary(root, experiment, summary, synthetic=cfg["synthetic"],
                              rehearsal=cfg["rehearsal"])
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
                                                          "synthetic", "rehearsal")}


def _k3_name(table: str, variant: str) -> str:
    return f"{table}:{variant}"


def _variant_hyper_row(table: str, variant: str, transfer: str, rotation: int,
                       hyper: Mapping) -> dict:
    lacks = aggregation_variants.not_applicable(variant) if table == "aggregation" else ()
    row = {"table": table, "variant": variant, "transfer": transfer, "rotation": rotation}
    for key in (*two_phase.PHASE_ONE_KEYS, *two_phase.PHASE_TWO_KEYS):
        row[key] = NOT_APPLICABLE if key in lacks else hyper[key]
    row["epochs1"], row["epochs2"] = hyper["epochs1"], hyper["epochs2"]
    row["search"] = ("none: full MIL-CREDA's selection for this transfer and outer fold, "
                     "gamma = 0 (Sec. 6.3)" if variant == aggregation_variants.WITHOUT_CONSENSUS
                     else "none: full MIL-CREDA's selection for this transfer and outer "
                          "fold (Sec. 6.3)")
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


def _k3_schedule(cfg: Mapping, phase_one: Mapping, phase_two: Mapping) -> dict:
    """The refit schedule of a K3 variant: full MIL-CREDA's E1 and E2 of the outer fold."""
    return {**search.refit_schedule(cfg["schedule"], 1, phase_one["epochs"]),
            "epochs2": search.refit_schedule(cfg["schedule"], 2, phase_two["epochs"])["epochs2"]}


def _k3_variant_model(product: Path, data: search.SearchData, rotation: int, seed: int,
                      table: str, variant: str, phase_one: Mapping, phase_two: Mapping,
                      schedule: Mapping, cfg: Mapping, device: torch.device,
                      owed: list) -> tuple:
    """(model, history, trained): the stored variant, or refit and stored when absent.
    `schedule` is the refit schedule (`_k3_schedule`); the variant trains on the non-test
    folds with no validation (`search.SearchData.rotations`)."""
    ppath = _partition_path(cfg)
    name = model_store.variant_name(variant)
    path2 = model_store.model_path(_models(product), data.transfer, rotation, seed, 2, name)
    current = identity(data.partition_sha256)
    if path2.exists():
        _reuse(path2, current)
        return mechanisms.load_phase_two(path2, partition_path=ppath)[0].to(device), [], False
    if table == "aggregation":
        path1 = model_store.model_path(_models(product), data.transfer, rotation,
                                       search.PHASE_ONE_SEED, 1, name)
        record = model_store.load(_reuse(path1, current), partition_path=ppath) \
            if path1.exists() else None
        trained = mechanisms.train_aggregation_variant(
            data, rotation, seed, variant, phase_one, phase_two, schedule, device,
            phase_one_record=record)
        if record is None:
            checkpoint = trained["phase1"]["checkpoint"]
            _save_variant(product, data, rotation, search.PHASE_ONE_SEED, 1, variant,
                          checkpoint["state"], trained["hyper"], trained["channels"],
                          checkpoint["sigma"], checkpoint)
    else:
        path1 = model_store.model_path(_models(product), data.transfer, rotation,
                                       search.PHASE_ONE_SEED, 1, two_phase.MODEL)
        if not path1.exists():
            owed.append(f"no stored MIL-CREDA phase-1 checkpoint for {data.transfer.name} "
                        f"r{rotation}: run the search step")
            return None, [], False
        trained = mechanisms.train_ablation_variant(
            data, rotation, seed, model_store.load(path1, partition_path=ppath), phase_two,
            schedule, variant, device)
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
    current = _bags_identity(bags)
    done = _predicted(product, K3, current)
    plan = k3_plan(cfg, core)
    for transfer in transfers_of(cfg):
        if transfer.name not in plan:
            continue
        data = search.search_data(transfer, bags)
        a = assignment_mod.assign_from_bags(transfer, bags)
        # Sec. 6.3: full MIL-CREDA's selection of both phases for each outer fold (Sec. 4.5)
        first = _selections(product, data, 1, "MIL-CREDA", cfg)
        selected = _phase_two_selection(product, data, "MIL-CREDA", cfg)
        schedules = {r: _k3_schedule(cfg, first[r], selected[r]) for r in rotations_of(cfg, data.rotations)}
        for r in rotations_of(cfg, data.rotations):
            phase_one, phase_two = first[r]["params"], selected[r]["params"]
            hyper_rows += [_variant_hyper_row("aggregation", v, transfer.name, r,
                                              mechanisms.aggregation_hyper(
                                                  phase_one, phase_two, schedules[r], v))
                           for v in plan[transfer.name]["aggregation"]]
            hyper_rows += [_variant_hyper_row("ablation", v, transfer.name, r,
                                              {**phase_one, **phase_two, **schedules[r]})
                           for v in plan[transfer.name]["ablation"]]
        for r in rotations_of(cfg, data.rotations):
            phase_one, phase_two = first[r]["params"], selected[r]["params"]
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
                            product, data, r, seed, table, variant, phase_one, phase_two,
                            schedules[r], cfg, device, owed)
                        if model is None:
                            continue
                        counts[f"trained_{table}"] += trained
                        history += _tag(records, name, transfer.name, r, seed)
                        if (name, transfer.name, r, seed) not in done:
                            out += _tag(two_phase.predictions(model, test, transfer.name,
                                                              device),
                                        name, transfer.name, r, seed)
                _append_history(product, K3, history)
                results.append_predictions(root, K3, out, identity=current)
                counts["predictions"] += len(out)
    written = k3_tables(cfg, product, bags, core, owed, current)
    if hyper_rows:
        results.write_table(root, K3, "variant_hyperparameters", hyper_rows)
    history = read_history(product, K3)
    if history:
        results.write_history(root, K3, history)
    recorded = _read_predictions(product, K3, current)
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
        "hyperparameters": "full MIL-CREDA's selection and epoch counts per transfer and "
                           "outer fold (Sec. 4.5); no search of their own",
        "pairs": written, "reportSeeds": cfg["reportSeeds"],
        "bootstrapResamples": cfg["bootstrapResamples"], "owed": sorted(set(owed))})
    results.write_summary(root, K3, summary, synthetic=cfg["synthetic"],
                          rehearsal=cfg["rehearsal"])
    return {**counts, "pairs": written, "owed": sorted(set(owed))}


def k3_tables(cfg: Mapping, product: Path, bags: bag_data.BagSet,
              core: Mapping[str, Sequence[str]], owed: list,
              current: Mapping | None = None) -> list[str]:
    """Per core pair, both directions pooled: the two tables and their contrasts."""
    root = _results(product)
    current = current or _bags_identity(bags)
    records = _read_predictions(product, K3, current)
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
        full = results._select(_read_predictions(product, results.EXPERIMENT_OF_ARM[arm],
                                                 current), "MIL-CREDA", (first, second))
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
        st.apply_contrast_families(contrasts[table], f"{K3} {table}: {spec['full']} vs "
                                                     f"{', '.join(spec['criterion'])}")
        significance += contrasts[table]
        if rows[table]:
            results.write_table(root, K3, table, rows[table])
    if significance:
        # the mechanisms step's K2 diagnostics, when already measured (Sec. 6.3, v56)
        _flag_global_only(cfg, significance, core,
                          _merged_summary(root, "K2-emparejamiento").get("diagnostics") or {})
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
        elif strategy == "AdaptMIL":
            models.append((strategy, _phase_two_path(product, transfer, r, seed, strategy)))
    return models


def _k2_diagnose(product: Path, transfer, a, bags: bag_data.BagSet, r: int, seed: int,
                 full, no_local, ppath, parts: dict, missing: dict, owed: list) -> None:
    """Sec. 6.2 (v56): one rotation's mechanism diagnostics, from full MIL-CREDA's stored
    phase-1 checkpoint (the start of phase 2) and the stored phase-2 models with and
    without the local term; a missing checkpoint is recorded, never replaced."""
    pairs = mechanisms.probe_pairs(a, bags, r)
    if len(pairs) < 2:
        return
    path1 = model_store.model_path(_models(product), transfer, r, search.PHASE_ONE_SEED, 1,
                                   two_phase.MODEL)
    if not path1.exists():
        reason = (f"no stored MIL-CREDA phase-1 checkpoint for {transfer.name} r{r}: run "
                  f"the search step")
        missing.setdefault(transfer.name, []).append(reason)
        owed.append(f"K2 diagnostics: {reason}")
        return
    start = mechanisms.k2_phase_two_start(model_store.load(path1, partition_path=ppath))
    tau_loc = model_store.load(_phase_two_path(product, transfer, r, seed, "MIL-CREDA"),
                               partition_path=ppath)[1]["hyper"]["tauLocal"]
    parts.setdefault(transfer.name, []).append(mechanisms.k2_rotation_diagnostics(
        start, full, no_local, pairs, tau_loc=tau_loc, transfer=transfer.name))


def _k2_transfer_diagnostics(full_rows, parts: Mapping, missing: Mapping) -> dict[str, dict]:
    """One diagnostics record per K2 transfer present: pooled over its rotations, or not
    computed with the reason when a rotation lacks its stored model."""
    out = {}
    for name in sorted({r["transfer"] for r in full_rows}):
        if missing.get(name):
            out[name] = {"transfer": name, "not_computed": "; ".join(sorted(set(missing[name])))}
        elif parts.get(name):
            out[name] = mechanisms.k2_pool_diagnostics(name, parts[name])
        else:
            out[name] = {"transfer": name,
                         "not_computed": "fewer than two probe subjects in every rotation"}
    return out


def _flag_global_only(cfg: Mapping, rows: list, core: Mapping,
                      diagnostics: Mapping[str, Mapping]) -> bool:
    """Sec. 6.3 (v56): per core pair, the both-terms vs global-only rows carry
    `inconclusive_no_room` -- and then the K2 verdict -- only when neither direction
    meets the Sec. 6.2 reading gate; a pair with no computed direction is left unflagged.
    True when a row was flagged."""
    by_name = {t.name: t for t in transfers_of(cfg)}
    changed = False
    for arm in sorted(core):
        first, second = _pair_of(core[arm])
        if first not in by_name or second not in by_name:
            continue
        flag = mechanisms.both_vs_global_only_inconclusive(
            [diagnostics[n] for n in (first, second) if n in diagnostics])
        if flag is None:
            continue
        comparison = (f"{K3}:ablation:{'|'.join(by_name[first].pair)}:"
                      f"{K3_TABLES['ablation']['full']} vs {K2_GLOBAL_ONLY}")
        for row in rows:     # the primary row and its secondary metrics, e.g. " (kappa)"
            if row["comparison"] == comparison or row["comparison"].startswith(
                    f"{comparison} ("):
                row["inconclusive_no_room"] = flag
                if flag:
                    row["verdict"] = mechanisms.K2_INCONCLUSIVE
                changed = True
    return changed


def _flag_global_only_rows(cfg: Mapping, root: Path, core: Mapping,
                           diagnostics: Mapping[str, Mapping]) -> None:
    """The k3 step's significance on disk, flagged by `_flag_global_only`."""
    path = root / f"significance_{K3}.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    if _flag_global_only(cfg, rows, core, diagnostics):
        results.write_significance(root, K3, rows)


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
    k2_parts: dict[str, list] = {}
    k2_missing: dict[str, list] = {}
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
                no_local = None
                if npath.exists():
                    no_local = mechanisms.load_phase_two(npath, partition_path=ppath)[0]
                    k2_no_local += mechanisms.k2_subject_rows(
                        no_local, a, bags, rotation=r, seed=seed,
                        model_name=mechanisms.K2_NO_LOCAL, metric="kernel")
                elif transfer.name in {t.name for t in mechanisms.K2_DECLARED.values()}:
                    owed.append(f"no stored {mechanisms.K2_NO_LOCAL} model for K2 at "
                                f"{transfer.name} r{r} s{seed}: run the k3 step")
                _k2_diagnose(product, transfer, a, bags, r, seed, model, no_local, ppath,
                             k2_parts, k2_missing, owed)
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
        results.write_table(root, "K1-atencion", "tests", mechanisms.k1_test_table(rows))
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
            "subjects_with_both_window_kinds": {
                arm: sum(r["arm"] == arm and r["marked_minus_clean"] is not None for r in own)
                for arm, c in mechanisms.K1_CONTRASTS.items() if "contact" in c},
            "subjects_not_emotiv": {
                arm: sum(r["arm"] == arm and not r["emotiv"] for r in own)
                for arm, c in mechanisms.K1_CONTRASTS.items() if "contact" in c}},
            synthetic=cfg["synthetic"],
            rehearsal=cfg["rehearsal"])
    if k2_full:
        diagnostics = _k2_transfer_diagnostics(k2_full, k2_parts, k2_missing)
        rows = mechanisms.k2_tests(k2_full, k2_no_local, diagnostics=diagnostics)
        results.write_significance(root, "K2-emparejamiento", rows)
        results.write_table(root, "K2-emparejamiento", "tests", mechanisms.k2_test_table(rows))
        results.write_table(root, "K2-emparejamiento", "diagnostics",
                            mechanisms.k2_diagnostics_table(list(diagnostics.values())),
                            columns=mechanisms.K2_DIAGNOSTICS_COLUMNS)
        results.write_table(root, "K2-emparejamiento", "separation",
                            k2_full + k2_no_local + k2_other)
        _flag_global_only_rows(cfg, root, core, diagnostics)
        results.write_summary(root, "K2-emparejamiento", {
            "probe_subjects": len({(r["transfer"], r["subject"]) for r in k2_full}),
            "declared": {k: v.name for k, v in mechanisms.K2_DECLARED.items()},
            "transfers": sorted({r["transfer"] for r in k2_full}),
            "no_local_variant": mechanisms.K2_NO_LOCAL,
            "no_local_transfers": sorted({r["transfer"] for r in k2_no_local}),
            "diagnostics": {t: {k: v for k, v in d.items() if k != "per_bag"}
                            for t, d in diagnostics.items()},
            "reading_gate": {"median_entropy_ratio_below": mechanisms.K2_ENTROPY_MAX,
                             "median_room_ratio_at_least": mechanisms.K2_ROOM_MIN,
                             "otherwise": mechanisms.K2_INCONCLUSIVE},
            "reportSeeds": cfg["reportSeeds"]}, synthetic=cfg["synthetic"],
            rehearsal=cfg["rehearsal"])
        out["K2-emparejamiento"] = len(rows)
    # The EnsV final models train MIL-CREDA's phase 2 too: their steps are read.
    history = [h for arm in results.EXPERIMENT_OF_ARM.values() for rule in HISTORY_FILES
               for h in read_history(product, arm, rule)]

    try:
        bounds = mechanisms.bound_control(history)
        results.write_table(root, "K3-ablacion", "term_bounds", bounds)
        current = _merged_summary(root, "K3-ablacion")      # the k3 step's keys survive
        results.write_summary(root, "K3-ablacion", {
            "term_bounds": bounds,
            "aggregation_variants": current.pop("aggregation_variants", "not trained (owed)"),
            "ablation_variants": current.pop("ablation_variants", "not trained (owed)"),
            **{k: v for k, v in current.items() if k != "term_bounds"}},
            synthetic=cfg["synthetic"], rehearsal=cfg["rehearsal"])
        out["K3-ablacion"] = len(bounds)
    except mechanisms.NotRecorded as error:
        owed.append(str(error))
    out["owed"] = sorted(set(owed))
    return out


# ------------------------------------------------------------------ Sec. 5.4


#: The declared diagnostic of the search's pooled early stopping (plan v60), per row.
EARLY_STOPPING_COLUMNS = ("phase", "name", "studies", "trials", "trialsReachedCap",
                          "fractionTrialsReachedCap", "primarySelectedAtLastEpoch",
                          "ensvSelectedAtLastEpoch")


def _stopping_row(diagnostic: Mapping) -> dict:
    at_last = diagnostic["selectedAtLastTrainedEpoch"]
    return {**{k: diagnostic[k] for k in EARLY_STOPPING_COLUMNS[1:6]},
            "primarySelectedAtLastEpoch": (at_last.get("primary") or {}).get("fraction"),
            "ensvSelectedAtLastEpoch": (at_last.get("ensv") or {}).get("fraction")}


def run_results(cfg: Mapping, product: Path, bags: bag_data.BagSet | None = None) -> dict:
    """Step 6: Sec. 5.4's figures and tables, and Sec. 4.5's two selection rules."""
    bags = bags or load_bags(cfg)
    root = _results(product)
    core = core_transfers(cfg, product)
    current = _bags_identity(bags)
    written, owed = [], []
    for arm in sorted({t.arm for t in transfers_of(cfg)}):
        experiment = results.EXPERIMENT_OF_ARM[arm]
        records = _read_predictions(product, experiment, current)
        free_records = _read_predictions(product, experiment, current, "ensv")
        hp, importance, detail, studies = [], [], [], {}
        stopping: dict[tuple[int, str], list] = {}
        core_choices: dict[str, dict[str, dict]] = {}
        proportions = []
        for transfer in [t for t in transfers_of(cfg) if t.arm == arm]:
            folds = search.rotation_folds(bags.rotations)
            # what a study's identity reads, without loading any bag tensor
            known = search.SearchData(transfer=transfer, partition_sha256=bags.partition_sha256,
                                      rotations={}, folds=folds)
            outer = sorted(folds)
            one = {g: {r: _stored_study(product, known, 1, g, r, cfg) for r in outer}
                   for g in _groups(cfg["strategies"])}
            two = {s: {r: _stored_study(product, known, 2, s, r, cfg) for r in outer}
                   for s in cfg["strategies"]}
            for phase, by_name in ((1, one), (2, two)):
                for name, by_r in by_name.items():
                    stopping.setdefault((phase, name), []).extend(by_r.values())
            chosen = {s: {rule: {r: search.select(study, rule) for r, study in by_r.items()}
                          for rule in search.RULES} for s, by_r in two.items()}
            # Sec. 5.4: phase 1's configuration and E1 per outer fold, in the one field the
            # hyperparameter table reads.
            first = {g: types.SimpleNamespace(best_params={
                str(r): {**sel["params"], "E1": sel["epochs"]}
                for r, sel in ((r, search.select(study)) for r, study in by_r.items())})
                for g, by_r in one.items()}
            hp += results.hyperparameter_rows(
                first, {s: c["primary"] for s, c in chosen.items()}, transfer.name)
            if transfer.name in core.get(arm, []):
                for s in two:
                    core_choices.setdefault(s, {})[transfer.name] = chosen[s]
                proportions += results.class_proportion_rows(
                    assignment_mod.assign_from_bags(transfer, bags))
            for s, by_r in two.items():
                detail += results.selected_configuration_rows(s, transfer.name, chosen[s])
                for study in by_r.values():
                    try:
                        importance += results.importance_rows(study, s)
                    except (RuntimeError, ValueError) as error:
                        owed.append(f"fANOVA {study.study_name}: {error}")
            if transfer.name in core.get(arm, []):
                studies.update({f"{s} ({transfer.name}, outer {r})": st_
                                for s, by_r in two.items() for r, st_ in by_r.items()})
        results.write_table(root, experiment, "hyperparameters", hp)
        selection = []
        for s, by_t in core_choices.items():
            primary = results._pooled(results._select(records, s, sorted(by_t)))
            free = results._pooled(results._select(free_records, s, sorted(by_t)))
            selection.append(results.selection_rows(
                s, {t: c["primary"] for t, c in by_t.items()},
                {t: c["ensv"] for t, c in by_t.items()},
                test_primary=primary["balanced_accuracy"] if primary else None,
                test_secondary=free["balanced_accuracy"] if free else None))
        results.write_table(
            root, experiment, "selection_rules", selection,
            columns=results.SELECTION_RULE_COLUMNS, note="\n".join([
                "Per core transfer and outer fold each rule selects a (trial, epoch) in that "
                "fold's study, without any target label (Sec. 4.5): the primary rule by the "
                "centroid score pooled over the three inner splits of the non-test folds, "
                "the secondary rule by EnsV over every (trial, epoch) of the study. The "
                "secondary rule's test balanced accuracy is pooled from its own refit models "
                "where its (trial, epoch) differs from the primary one, and from the primary "
                "models where they coincide; nothing is selected with test outcomes. "
                "Diferencia: primary minus secondary.", "",
                *(results.selection_outcome(row) for row in selection), "",
                "Class proportions of each core transfer's sides (Sec. 8):",
                *(f"- {p['Transferencia núcleo']}, {p['Lado']}: {p['yes']} yes, {p['no']} "
                  f"no of {p['Sujetos']} subjects" for p in proportions)]))
        results.write_table(
            root, experiment, "early_stopping",
            [{"phase": phase, **_stopping_row(search.stopping_diagnostic(name, sts))}
             for (phase, name), sts in sorted(stopping.items())],
            columns=EARLY_STOPPING_COLUMNS,
            note=("Plan v60, Secs. 4.5 and 7.6: each search trial trains at most "
                  f"{search.EPOCH_CAP} epochs and stops when its pooled selection score has "
                  f"not improved for {search.PATIENCE[1]} (phase 1) or {search.PATIENCE[2]} "
                  "(phase 2) epochs. Per group or strategy, over every transfer and outer "
                  "fold of this arm: the fraction of trials that reached the cap, and per "
                  "rule the fraction of selected epochs equal to the last epoch their trial "
                  "trained."))
        results.write_table(root, experiment, "selection_class_proportions", proportions,
                            columns=results.CLASS_PROPORTION_COLUMNS)
        results.write_table(root, experiment, "selected_configurations", detail)
        if importance:
            results.write_table(root, experiment, "importance", importance)
        history = read_history(product, experiment)
        if history:
            written.append(results.write_history(root, experiment, history))
        free_history = read_history(product, experiment, rule="ensv")
        if free_history:
            written.append(results.write_history(root, f"{experiment}_ensv",
                                                 free_history))
            try:
                written.append(results.convergence_figure(root, experiment, free_history,
                                                          core[arm],
                                                          name="convergence_ensv",
                                                          epoch_field="ensvEpoch"))
            except results.ResultsError as error:
                owed.append(f"{experiment} EnsV convergence: {error}")
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
    # Sec. 5.4: one Friedman test with Nemenyi over the ten transfers, nine strategies.
    every = [r for e in ("A-nivel", "B-tarea", "C-dispositivo")
             for r in _read_predictions(product, e, current)]
    ranking = results.global_ranking(every)
    if ranking:
        written += results.write_global_ranking(root, ranking)
    else:
        owed.append("global critical-difference diagram: not all nine strategies have "
                    "predictions on all ten transfers")
    table = Path(cfg.get("separabilityTable") or product / SEPARABILITY_TABLE)
    if table.exists():
        rows = pd.read_csv(table).to_dict("records")
        records = [r for e in ("A-nivel", "B-tarea")
                   for r in _read_predictions(product, e, current)]
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
              "schedule": {"epochs1": 2, "epochs2": 2, "bagsPerStep": 4},
              "strategies": ["MIL-CREDA", "COT", "AdaptMIL"], "transfers": pair,
              "core": {arm: pair}, "nTrials": 4, "reportSeeds": list(REPORT_SEEDS),
              "bootstrapResamples": 500}
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False))
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m MIL_CREDA_EEG.campaign",
                                     description="The BALLADEER campaign of plan v60, local only.")
    parser.add_argument("step", choices=(*STEPS, "all", "synthetic"))
    parser.add_argument("--config", help="campaign config (JSON)")
    parser.add_argument("--product", default=str(DEFAULT_PRODUCT),
                        help="product folder holding Results/ and Models/ (default MIL-CREDA-EEG/)")
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
