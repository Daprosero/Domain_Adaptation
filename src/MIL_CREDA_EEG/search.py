"""The Optuna searches of both phases, per transfer and outer fold.

Experiments plan v59, Secs. 4.2, 4.5, 7.3 and 7.7. The selection is a nested
cross-validation. The outer level is the frozen four-fold partition, rotated:
rotation r tests fold k and F is the other three folds. The inner level is the
three inner splits of F, read from the partition document (`assignment.inner_splits`,
the one source of the splits): each validates one fold of
F and trains on the other two, so every subject of F is validated exactly once
and the test fold is in none. There is one Optuna study per phase-1 group (the
common group of the seven non-MIL comparators, AdaptMIL and MIL-CREDA),
transfer and outer fold, and one per strategy, transfer and outer fold in
phase 2: 120 + 360 = 480 studies over the ten transfers, 120 per outer fold. Every study gets the
same budget: `N_TRIALS` trials with Optuna's Gaussian-process sampler
(`GPSampler`, experimental in Optuna; its first ten trials are random) seeded
with the fixed `SAMPLER_SEED` so the search can be repeated, and no pruner
(`NopPruner`): every trial runs complete. The sampler never observes a score of
the outer test fold's subjects.

**A trial (Secs. 4.5, 7.7).** Each configuration is trained on the three inner
splits of its outer fold with ONE training seed, `SEARCH_SEED`, distinct from
the single report seed (`REPORT_SEEDS`, Sec. 4.6), recording every trained epoch's
validation predictions. Plan v59 (Secs. 4.5, 7.6): a trial trains at most its epoch cap
(`epochs1` or `epochs2`, at most `EPOCH_CAP` = 100) and stops when the POOLED per-epoch
selection score of its phase has not improved for `PATIENCE[phase]` consecutive epochs
(30 in phase 1, 10 in phase 2), never before a strategy's `MIN_TRAINED_EPOCHS` (CCA-LSC:
3 warm-up + 1 calibrated). The three inner splits advance together for that
(`train_inner_splits`): each split trains in its own thread, one split computing at a
time in split order, each keeping its own RNG state, so every trainer's numbers are those
of a serial run; after every split's epoch e the pooled score of e decides whether all of
them train e + 1. The trial's selected epoch is still the best of its pooled series; the
refit trains exactly that many epochs with no early stopping; EnsV's candidates are the
(trial, epoch) pairs actually trained. Each trial records `trainedEpochs`, `reachedCap`,
`patience` and `epochCap`; `stopping_diagnostic` gives, per strategy, the fraction of
trials that reached the cap and of selected epochs equal to their trial's last trained
epoch. Per epoch, the subject-level confusion counts of the three inner
splits are pooled (each subject of F once) and the balanced accuracy is computed
once on them (Forman and Scholz, 2010): no mean per split and no deviation. The
trial's value, what the sampler observes, is the best epoch's pooled balanced
accuracy:

- phase 1: over the source subjects of the validating folds, with their labels
  (`two_phase.pooled_balanced_accuracy`, `pooled_balanced_accuracy_series`);
- phase 2: Sec. 4.5's primary rule, the centroid rule, which reads no target
  label (`pooledCentroidBalancedAccuracy`). In each inner split, the adapted
  model's target validation predictions are pseudo-labels (argmax) and their
  representations Z (the vector entering the predictive head) give one centroid
  per pseudo-class; the source bags of the validating fold, passed through the
  frozen source trunk and the same frozen attention or aggregator, are assigned
  to the nearest centroid and scored against their real source labels; the
  counts of the three splits are pooled and the balanced accuracy computed once
  (`two_phase.centroid_counts`, `two_phase.pooled_centroid_balanced_accuracy`).
  The source representations do not change between trials or epochs (the source
  side is frozen in phase 2, Sec. 7.3): they are computed once per outer fold
  and inner split from the phase-1 checkpoint, by inference only
  (`source_representations`), here, outside every trainer (CLUST's trainer reads
  no source data, Sec. 7.3). The EnsV secondary rule needs no score of its own
  per trial: each trial records, per epoch, its target validation probabilities
  pooled over the three inner splits (`ensvSubjects`, `ensvProbabilities`;
  probabilities only, never a representation or a label). Nothing is trained for
  either rule (Sec. 7.7).

**Selection (Sec. 4.5).** `select` picks, in one outer fold's study, a completed
trial and epoch: (trial, E1) in phase 1 and (trial, E2) in phase 2 by the best
pooled score of the primary rule, and by `rule="ensv"` the phase-2 (trial, epoch)
of EnsV (Hu et al., NeurIPS 2024): every (trial, epoch) of the study is a
candidate, their probabilities averaged per target bag of F are the role model
whose argmax is each bag's pseudo-label, and the candidate with the best accuracy
against those pseudo-labels wins. Ties go to the lowest trial and the earliest
epoch, for both rules. The selected configuration is refit on every fold of F for
exactly that many epochs, with no validation (`campaign`).

The search reads only the inner splits of F (`search_data`): no test-fold bag is
ever loaded.

**Phase 2 starts from the phase-1 refit (Secs. 4.5, 7.3, 7.7).** Every phase-2
trial of an outer fold, in each of its three inner splits, and that fold's phase-2
final refit start from the one phase-1 checkpoint of that outer fold and its group
("desde el mismo punto de control"): the selected phase-1 configuration refit on
the source subjects of all of F for E1 epochs at `PHASE_ONE_SEED`, the report
seed (Sec. 7.7 refits the chosen configuration with the report seed), read back
from the model store (`model_store.load`, which refuses a foreign partition). A
phase-2 study records that checkpoint's identity
(`phase_one_checkpoint_identity`) in its own identity, and a phase-2 model in its
sidecar. MIL-CREDA's sigma is that checkpoint's sigma
(`sigma2 = sigma1`) and is never searched: its phase-2 space is exactly `lr2`,
`wd2` and `tauLocal`, and its phase-1 choices are fixed.

**Spaces (Sec. 7.7).** Shared, log scale: `lr1` in [1e-4, 1e-1], `lr2` in
[1e-5, 1e-2], `wd1` and `wd2` in [1e-7, 1e-2]. MIL-CREDA: `attentionGamma` in
[0, 1], `attentionTemperature` and `tauLocal` in [0.1, 10] log,
`sigmaMultiplier` in {0.5, 0.75, 1, 1.5}. The comparators' phase-2 spaces
follow one generic rule (`LOSS_WEIGHT`, `KERNEL_MULTIPLIER`,
`CONFIDENCE_THRESHOLD`), registered per strategy by its own unit through
`register_phase_two`. Sec. 7.7 adds three comparator-specific axes, each admitted
only for its own strategy (`OWN_AXES`): CLUST's teacher EMA momentum rho in
[0.9, 0.999] with 1 - rho log-uniform, drawn as `EMA_MOMENTUM_COMPLEMENT`
(1 - rho in [1e-3, 1e-1], log) and converted by the CLUST trainer; and
CASUAL's VAT radius as a fraction of the typical window L2 norm, log-uniform
in [0.01, 1] (`VAT_RADIUS_FRACTION`), scaled by the CASUAL trainer; and
CLUDA's contrastive temperature, log-uniform in [0.01, 1]
(`CONTRASTIVE_TEMPERATURE`). None is wired here, and AdaptMIL has no phase-1
architecture registered until its unit registers it (`model_store.BUILDERS`),
so its studies refuse before any storage is created.

**Storage.** One SQLite file per study, `<root>/Results/hpo/
<source>-to-<target>__phase<p>__<group or strategy>__outer<r>.db`, with the study
named as the file's stem; `<root>` defaults to `MIL-CREDA-EEG/`. Each study records its
identity (`study_identity`: transfer, phase, outer fold and its test fold, the
inner splits, budget, space, sampler and its seed, schedule, the partition hash,
the plan revision and the code version). A study whose file exists is continued only when its identity
is the requested one, up to its declared trial count (an interrupted search
resumes), and returned as it is when complete; another identity is refused,
naming what differs (`open_study`). No budget is silently extended. A continued
study's sampler is seeded with `SAMPLER_SEED` plus the trials it already holds,
so it does not replay the seeded first draws; an interrupted-and-continued study
is therefore not the same sequence of trials as an uninterrupted one, and records
where it was continued (`resumedAfter`). Trials left running by a killed process
stay as they are: they are neither counted nor re-run. `should_stop(study)`, when
given, is asked before the first trial and after each one; once it answers true the
study stops between trials, holding what it completed, and the next call continues it
(`remote_units` asks it against the session deadline). Without it a study runs its
remaining count in one `optimize` call, as before.

**Implementation choices the plan does not fix.** The seeds: `SEARCH_SEED` = 0,
`REPORT_SEEDS` = (1,), `SAMPLER_SEED` = 20261102. The epoch caps (`epochs1`,
`epochs2`, at most `EPOCH_CAP`) and `bagsPerStep` are inputs with no default
(Sec. 7.6). Groups other than
MIL-CREDA fix `attentionGamma` = 0, `attentionTemperature` = 1 and
`sigmaMultiplier` = 1 in the phase-1 trainer's key set
(`mil_wrapper.COMMON_PHASE_ONE_FIXED`); none of them is searched. Phase-2
keys that phase 1 does not read are set to NaN in the phase-1 hyperparameters,
so a phase-1 trainer that read them would fail rather than run at a value
nobody chose.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.distributions import (BaseDistribution, CategoricalDistribution,
                                  FloatDistribution)

from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, mil_wrapper, model_store, partition, two_phase

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["4.2", "4.5", "5.4", "7.3", "7.6", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "the_sampler_is_gpsampler_with_the_fixed_seed_and_no_pruner",
        "the_search_seed_is_not_a_report_seed",
        "the_spaces_are_exactly_those_of_sec_7_7",
        "no_comparator_phase_two_is_wired_and_the_hook_takes_only_the_generic_rule",
        "the_comparator_specific_axes_are_admitted_only_for_their_own_strategy",
        "epoch_caps_have_no_default",
        "the_caps_and_patience_values_are_those_of_v59",
        "a_trial_stops_after_exactly_patience_epochs_without_pooled_improvement",
        "the_phase_one_stop_uses_the_pooled_score_not_one_split",
        "the_cap_of_100_is_respected",
        "cca_lsc_trains_its_minimum_epochs_before_it_can_stop",
        "the_coordinated_splits_keep_the_trainers_numerics",
        "the_refit_trains_exactly_the_selected_epochs_with_no_early_stopping",
        "the_stopping_diagnostic_reports_cap_and_last_epoch_fractions",
        "the_objective_is_the_best_epochs_pooled_balanced_accuracy_over_the_inner_splits",
        "the_phase_two_objective_is_the_centroid_rule_and_reads_no_target_label",
        "the_source_representations_are_computed_once_per_outer_fold_and_inner_split",
        "the_source_pass_runs_in_the_search_never_in_a_trainer",
        "the_search_never_loads_a_test_fold_bag",
        "the_search_data_holds_the_inner_splits_and_a_refit_view_without_validation",
        "the_phase_two_mil_creda_sigma_comes_from_the_phase_one_checkpoint",
        "phase_two_starts_only_from_the_outer_folds_one_phase_one_checkpoint",
        "the_ensv_probabilities_are_recorded_per_trial_and_epoch",
        "the_ensv_selection_reads_every_completed_trial_of_the_study",
        "the_selection_is_the_argmax_of_the_pooled_score_over_trial_and_epoch",
        "no_target_label_changes_either_phase_two_selection",
        "the_pooled_series_check_each_validated_subject_once_per_epoch",
        "the_labels_of_r_test_subjects_cannot_change_r_selection",
        "one_study_per_strategy_transfer_and_outer_fold",
        "the_study_records_its_outer_fold_and_inner_splits",
        "an_interrupted_study_of_the_same_identity_continues_to_its_declared_count",
        "a_study_of_another_identity_is_refused_naming_what_differs",
        "a_study_records_the_identity_it_is_checked_against",
        "unwired_groups_and_strategies_refuse_before_creating_storage",
        "a_two_trial_smoke_study_completes_and_writes_its_sqlite",
        "a_study_stops_between_trials_when_asked_and_the_rerun_completes_it",
    ],
}

#: Sec. 7.7: 20 trials per study, GPSampler with a fixed seed, no pruning.
N_TRIALS = 20
SAMPLER_SEED = 20261102
#: Sec. 7.7: one search training seed, distinct from the single report seed (Sec. 4.6).
SEARCH_SEED = 0
REPORT_SEEDS = (1,)
#: Secs. 4.5, 7.7: the seed of the one phase-1 refit checkpoint per group and outer fold,
#: the report seed; the phase-2 search trials and the phase-2 final refit both start from it.
PHASE_ONE_SEED = REPORT_SEEDS[0]
#: The fields of a phase-1 checkpoint's sidecar that name it (`phase_one_checkpoint_identity`).
PHASE_ONE_CHECKPOINT_FIELDS = ("transfer", "rotation", "seed", "phase", "name",
                               partition.HASH_FIELD, "state_sha256")
#: Plan v59, Secs. 4.5 and 7.6: every search trial trains at most `EPOCH_CAP` epochs and
#: stops when the POOLED per-epoch selection score of its phase (phase 1: source-validation
#: balanced accuracy; phase 2: the centroid score; both over the three inner splits) has
#: not improved for `PATIENCE[phase]` consecutive epochs. A refit has no early stopping.
EPOCH_CAP = 100
PATIENCE = {1: 30, 2: 10}
#: The epochs a strategy's trainer needs before its trial may stop: CCA-LSC's 3 warm-up
#: epochs plus one calibrated epoch (`comparators.cca_lsc.training.WARMUP_EPOCHS` + 1).
MIN_TRAINED_EPOCHS = {"CCA-LSC": 4}
#: Sec. 4.2: the inner splits of each outer fold, one per non-test fold.
N_INNER = partition.N_FOLDS - 1
DEFAULT_ROOT = bag_data.REPOSITORY / "MIL-CREDA-EEG"
#: Inputs with no default (Sec. 7.6: the pilot fixes them).
SCHEDULE_KEYS = two_phase.SCHEDULE_KEYS

# ------------------------------------------------------------------ spaces (Sec. 7.7)

LR1 = FloatDistribution(1e-4, 1e-1, log=True)
LR2 = FloatDistribution(1e-5, 1e-2, log=True)
WEIGHT_DECAY = FloatDistribution(1e-7, 1e-2, log=True)
#: The generic rule for the comparators' own phase-2 parameters.
LOSS_WEIGHT = FloatDistribution(1e-3, 10.0, log=True)
KERNEL_MULTIPLIER = CategoricalDistribution((0.5, 0.75, 1.0, 1.5))
CONFIDENCE_THRESHOLD = FloatDistribution(0.5, 0.95)
GENERIC_RULE = (LOSS_WEIGHT, KERNEL_MULTIPLIER, CONFIDENCE_THRESHOLD)
#: Sec. 7.7, CLUST: the teacher EMA momentum rho in [0.9, 0.999] with 1 - rho
#: log-uniform; the sampler draws 1 - rho and the CLUST trainer converts.
EMA_MOMENTUM_COMPLEMENT = FloatDistribution(1e-3, 1e-1, log=True)
#: Sec. 7.7, CASUAL: the VAT radius as a fraction of the typical window L2 norm.
VAT_RADIUS_FRACTION = FloatDistribution(0.01, 1.0, log=True)
#: Sec. 7.7, CLUDA: the temperature of its contrastive term, a range that contains the
#: official code's 0.07 (a plan assumption, not the paper's).
CONTRASTIVE_TEMPERATURE = FloatDistribution(0.01, 1.0, log=True)
#: Strategy -> the axes outside the generic rule that Sec. 7.7 assigns to it alone.
OWN_AXES: dict[str, tuple[BaseDistribution, ...]] = {
    "CLUST — variante source-free": (EMA_MOMENTUM_COMPLEMENT,),
    "CASUAL": (VAT_RADIUS_FRACTION,),
    "CLUDA": (CONTRASTIVE_TEMPERATURE,),
}

_SHARED_PHASE_ONE = {"lr1": LR1, "wd1": WEIGHT_DECAY}
PHASE_ONE_SPACES: dict[str, dict[str, BaseDistribution]] = {
    "common": dict(_SHARED_PHASE_ONE),
    "AdaptMIL": dict(_SHARED_PHASE_ONE),
    "MIL-CREDA": {**_SHARED_PHASE_ONE,
                  "attentionGamma": FloatDistribution(0.0, 1.0),
                  "attentionTemperature": FloatDistribution(0.1, 10.0, log=True),
                  "sigmaMultiplier": CategoricalDistribution((0.5, 0.75, 1.0, 1.5))},
}
SHARED_PHASE_TWO = {"lr2": LR2, "wd2": WEIGHT_DECAY}
#: Strategy -> its own phase-2 parameters beyond `SHARED_PHASE_TWO`.
PHASE_TWO_EXTRA: dict[str, dict[str, BaseDistribution]] = {
    "MIL-CREDA": {"tauLocal": FloatDistribution(0.1, 10.0, log=True)},
}
#: Sec. 7.3: the phase-1 group each strategy's phase 2 starts from.
PHASE_ONE_GROUP_OF = {s: (s if s in ("AdaptMIL", "MIL-CREDA") else mil_wrapper.GROUP)
                      for s in model_store.STRATEGIES}


class SearchError(RuntimeError):
    """A search was asked for something Secs. 4.5 and 7.7 do not allow."""


class NotWired(SearchError):
    """The strategy's phase-2 training is not implemented yet."""


# ------------------------------------------------------------------ data


@dataclass(frozen=True)
class SearchData:
    """Per outer rotation, the refit view (every non-test fold for training, no validation
    bags), and per (rotation, inner split) the inner split's training and validation bags.

    A trainer reads `rotations[rotation]`; `inner_view(rotation, split)` is a SearchData
    whose one rotation is that inner split, so every trainer runs on it unchanged."""

    transfer: assignment_mod.Transfer
    partition_sha256: str
    rotations: Mapping[int, Mapping[str, list[two_phase.BagTensor]]]
    #: Rotation -> its test fold and the validation folds of its inner splits.
    folds: Mapping[int, Mapping] = field(default_factory=dict)
    #: (rotation, inner split) -> that split's bags.
    inner: Mapping[tuple[int, int], Mapping[str, list[two_phase.BagTensor]]] = field(
        default_factory=dict)

    def channels(self) -> dict[str, int]:
        first = self.rotations[min(self.rotations)]
        return {"source": int(first["source_train"][0].windows.shape[1]),
                "target": int(first["target_train"][0].windows.shape[1])}

    def inner_view(self, rotation: int, split: int) -> "SearchData":
        try:
            bags = self.inner[(int(rotation), int(split))]
        except KeyError:
            raise SearchError(f"inner split {split} of rotation {rotation} is not in the "
                              f"search data") from None
        return SearchData(transfer=self.transfer, partition_sha256=self.partition_sha256,
                          rotations={int(rotation): bags},
                          folds={int(rotation): self.folds.get(int(rotation), {})})


def search_data(transfer: assignment_mod.Transfer, bags: bag_data.BagSet) -> SearchData:
    """Every rotation's refit view and inner splits; test-fold bags are never read."""
    a = assignment_mod.assign_from_bags(transfer, bags)
    rotations = sorted(int(r) for r in bags.rotations)
    if rotations != list(range(partition.N_FOLDS)):
        raise SearchError(f"the search runs over the {partition.N_FOLDS} rotations of the "
                          f"partition; got {rotations}")
    cache: dict[tuple[str, str], two_phase.BagTensor] = {}

    def tensors(side: str, r: int, role: str, split: int | None = None):
        out = []
        for b in assignment_mod.side_bags(a, bags, side, r, role, split=split):
            if (b.subject, b.condition) not in cache:
                cache[(b.subject, b.condition)] = two_phase.bag_tensors([b])[0]
            out.append(cache[(b.subject, b.condition)])
        return out

    refit = {r: {"source_train": tensors("source", r, assignment_mod.REFIT_ROLE),
                 "source_valid": [],
                 "target_train": tensors("target", r, assignment_mod.REFIT_ROLE),
                 "target_valid": []} for r in rotations}
    inner = {(r, i): {"source_train": tensors("source", r, "inner_train", i),
                      "source_valid": tensors("source", r, "inner_validation", i),
                      "target_train": tensors("target", r, "inner_train", i),
                      "target_valid": tensors("target", r, "inner_validation", i)}
             for r in rotations for i in range(N_INNER)}
    return SearchData(transfer=transfer, partition_sha256=a.partition_sha256,
                      folds=rotation_folds(bags.rotations), rotations=refit, inner=inner)


# ------------------------------------------------------------------ hyperparameters


def check_schedule(schedule: Mapping) -> dict:
    if set(schedule) != set(SCHEDULE_KEYS):
        raise ValueError(f"the schedule is exactly {list(SCHEDULE_KEYS)}, with no default "
                         f"(Sec. 7.6); got {sorted(schedule)}")
    return {k: schedule[k] for k in SCHEDULE_KEYS}


def phase_one_hyper(group: str, params: Mapping, schedule: Mapping) -> dict:
    """The phase-1 trainer's key set: searched, fixed, and NaN for what it never reads."""
    if group not in PHASE_ONE_SPACES:
        raise SearchError(f"{group!r} is not a phase-1 group")
    if set(params) != set(PHASE_ONE_SPACES[group]):
        raise SearchError(f"the {group} phase-1 parameters are exactly "
                          f"{list(PHASE_ONE_SPACES[group])}; got {sorted(params)}")
    fixed = {} if group == "MIL-CREDA" else mil_wrapper.COMMON_PHASE_ONE_FIXED
    nan = float("nan")
    return two_phase.check_hyper({**fixed, **params, "lr2": nan, "wd2": nan, "tauLocal": nan,
                                  **check_schedule(schedule)})


def refit_schedule(schedule: Mapping, phase: int, epochs: int) -> dict:
    """Sec. 4.5: a refit's schedule, the search's with the phase's epoch count set to the
    selected E1 (phase 1) or E2 (phase 2); the refit runs exactly that many epochs."""
    if phase not in (1, 2):
        raise SearchError(f"a refit is of phase 1 or 2, not {phase!r}")
    if int(epochs) != epochs or epochs < 1:
        raise SearchError(f"a refit runs a positive whole number of epochs; got {epochs!r}")
    return {**check_schedule(schedule), f"epochs{phase}": int(epochs)}


def _stored_hyper(hyper: Mapping) -> dict:
    return {k: hyper[k] for k in (*two_phase.PHASE_ONE_KEYS, *SCHEDULE_KEYS)}


def phase_two_space(strategy: str) -> dict[str, BaseDistribution]:
    if strategy not in model_store.STRATEGIES:
        raise SearchError(f"{strategy!r} is not a strategy")
    if strategy not in PHASE_TWO_TRAINERS:
        raise NotWired(f"the phase-2 training of {strategy} is not wired yet")
    return {**SHARED_PHASE_TWO, **PHASE_TWO_EXTRA.get(strategy, {})}


def register_phase_two(strategy: str, extra: Mapping[str, BaseDistribution],
                       trainer: Callable) -> None:
    """Wire a comparator's phase 2: its own parameters under the generic rule, plus
    only the Sec. 7.7 axes `OWN_AXES` assigns to this strategy.

    `trainer(data, rotation, seed, record, params, schedule, device)` returns
    `{"model", "result": {"checkpoint": {"valid_balanced_accuracy", "hyper", ...},
    "history": [...], "validation": [two_phase.validation_record(..., Z), ...]}}`, as
    `train_mil_creda_phase_two`: every validation record carries the bags'
    representations (`two_phase.represent`), which the centroid rule reads.
    """
    if strategy not in model_store.STRATEGIES or strategy == "MIL-CREDA":
        raise SearchError(f"{strategy!r} is not a comparator")
    clash = set(extra) & set(SHARED_PHASE_TWO)
    if clash:
        raise SearchError(f"{sorted(clash)} are the shared axes; a comparator adds its own")
    allowed = (*GENERIC_RULE, *OWN_AXES.get(strategy, ()))
    off_rule = [k for k, d in extra.items() if not any(d == g for g in allowed)]
    if off_rule:
        raise SearchError(f"{off_rule} do not follow the generic rule of Sec. 7.7 (loss "
                          f"weights [1e-3, 10] log, kernel multipliers {{0.5, 0.75, 1, 1.5}}, "
                          f"confidence thresholds [0.5, 0.95]) nor an axis Sec. 7.7 assigns "
                          f"to {strategy}")
    PHASE_TWO_EXTRA[strategy] = dict(extra)
    PHASE_TWO_TRAINERS[strategy] = trainer


def suggest(trial: optuna.Trial, space: Mapping[str, BaseDistribution]) -> dict:
    out = {}
    for name, d in space.items():
        if isinstance(d, FloatDistribution):
            out[name] = trial.suggest_float(name, d.low, d.high, log=d.log)
        elif isinstance(d, CategoricalDistribution):
            out[name] = trial.suggest_categorical(name, d.choices)
        else:
            raise SearchError(f"{name}: unsupported distribution {d!r}")
    return out


# ------------------------------------------------------------------ selection (Sec. 4.5)

#: Sec. 4.5's two phase-2 rules: the centroid rule (primary) and EnsV (secondary).
RULES = ("primary", "ensv")
#: Phase -> the per-epoch pooled series of a trial the primary rule reads: phase 1 the
#: source-validation balanced accuracy, phase 2 the centroid rule's.
PRIMARY_SERIES = {1: "pooledBalancedAccuracy", 2: "pooledCentroidBalancedAccuracy"}
#: What a phase-2 trial records for EnsV: the target subjects of F in pooled order, and
#: per epoch their predicted probabilities, in that order.
ENSV_SUBJECTS, ENSV_PROBABILITIES = "ensvSubjects", "ensvProbabilities"
#: The selection rules, for every record that names them (Sec. 4.5).
PRIMARY_RULE = ("phase 1: the best epoch's source-validation balanced accuracy of the "
                "subject-level confusion counts pooled over the three inner splits; phase 2: "
                "the centroid rule (source validation bags assigned to the nearest centroid "
                "of the target validation bags' pseudo-classes, counts pooled over the three "
                "inner splits, balanced accuracy once), no target label; the lowest trial "
                "and earliest epoch on ties (Sec. 4.5)")
#: What a study's sampler observes, part of its identity (`open_study` refuses a study
#: scored by another objective, a v58 one included).
PHASE_ONE_OBJECTIVE = (
    "plan v59, Secs. 4.5 and 7.6: the best epoch's source-validation balanced accuracy of "
    "the subject-level confusion counts pooled over the three inner splits; each trial "
    f"trains at most {EPOCH_CAP} epochs and stops when that pooled score has not improved "
    f"for {PATIENCE[1]} consecutive epochs")
PHASE_TWO_OBJECTIVE = (
    "plan v59, Secs. 4.5 and 7.6, no target label: the best epoch's centroid-rule balanced "
    "accuracy (the source validation bags of each inner split assigned to the nearest "
    "centroid of the adapted model's target validation pseudo-classes, scored against "
    "their source labels, counts pooled over the three inner splits), what the sampler "
    f"observes; each trial trains at most {EPOCH_CAP} epochs and stops when that pooled "
    f"score has not improved for {PATIENCE[2]} consecutive epochs (never before a "
    "strategy's MIN_TRAINED_EPOCHS); ensvProbabilities: per trained epoch, the pooled "
    "target validation probabilities the EnsV secondary rule reads")
ENSV_RULE = ("EnsV (Hu et al., NeurIPS 2024): every (trial, epoch) of the phase-2 study is "
             "a candidate; the mean of their predicted probabilities per target bag of F is "
             "the role model, whose argmax is the bag's pseudo-label; the candidate with "
             "the best accuracy against those pseudo-labels wins, the lowest trial and "
             "earliest epoch on ties; no target label (Sec. 4.5)")


def rotation_folds(rotations: Mapping) -> dict[int, dict]:
    """Rotation -> its test fold and the inner splits the partition stores (Sec. 4.2)."""
    out = {}
    for r, v in rotations.items():
        splits = assignment_mod.inner_splits(v)
        out[int(r)] = {"test_fold": int(v["test_fold"]), "inner_splits": splits,
                       "inner_validation_folds": [x["validation_fold"] for x in splits]}
    return out


def _by_epoch(per_split: Sequence[Sequence[Mapping]]) -> list[tuple]:
    """The validation records of the three inner splits, grouped by epoch."""
    if len(per_split) != N_INNER:
        raise SearchError(f"the score pools the {N_INNER} inner splits; got {len(per_split)}")
    epochs = [[int(r["epoch"]) for r in split] for split in per_split]
    if not epochs[0] or any(e != epochs[0] for e in epochs):
        raise SearchError(f"the inner splits recorded different epochs: {epochs}")
    return list(zip(*per_split))


def pooled_balanced_accuracy_series(per_split: Sequence[Sequence[Mapping]]) -> list[float]:
    """Per epoch, the balanced accuracy of the subject-level counts pooled over the inner
    splits, against the validated subjects' own labels: phase 1's selection score on the
    source subjects (Sec. 4.5), and the target-supervised upper bound's epoch
    (`controls`), which is a reference, not a phase-2 selection."""
    try:
        return [two_phase.pooled_balanced_accuracy(records) for records in _by_epoch(per_split)]
    except ValueError as error:
        raise SearchError(str(error)) from None


def pooled_centroid_series(per_split: Sequence[Sequence[Mapping]],
                           sources: Sequence[Mapping]) -> list[float]:
    """Per epoch, Sec. 4.5's primary phase-2 score: the centroid rule's counts of the three
    inner splits pooled, balanced accuracy once (`two_phase.pooled_centroid_balanced_
    accuracy`); `sources[i]` is split i's source pass (`source_representations`). It reads
    the target records' probabilities and representations, never their labels."""
    try:
        return [two_phase.pooled_centroid_balanced_accuracy(records, sources)
                for records in _by_epoch(per_split)]
    except ValueError as error:
        raise SearchError(str(error)) from None


def ensv_record(per_split: Sequence[Sequence[Mapping]]) -> dict:
    """What a phase-2 trial records for EnsV: the target subjects of F pooled over the
    three inner splits (each predicted once, from the split where it validates) and, per
    epoch, their predicted probabilities in that order. No label, no representation."""
    by_epoch = _by_epoch(per_split)
    subjects = [s for r in by_epoch[0] for s in r["subjects"]]
    if len(subjects) != len(set(subjects)):
        raise SearchError("the pooled validation must hold each subject exactly once")
    probabilities = []
    for records in by_epoch:
        if [s for r in records for s in r["subjects"]] != subjects:
            raise SearchError("the inner splits validated other subjects at another epoch")
        probabilities.append([[float(x) for x in p] for r in records
                              for p in r["probabilities"]])
    return {ENSV_SUBJECTS: subjects, ENSV_PROBABILITIES: probabilities}


def _completed(study: optuna.Study) -> list:
    trials = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                    key=lambda t: t.number)
    if not trials:
        raise SearchError(f"{study.study_name}: no completed trial to select from")
    return trials


def _study_phase(study: optuna.Study, trials: Sequence) -> int:
    phase = study.user_attrs.get("phase")
    if phase is not None:
        return int(phase)
    return 2 if any(PRIMARY_SERIES[2] in t.user_attrs for t in trials) else 1


def _primary_choice(study: optuna.Study, trials: Sequence, key: str):
    best, best_epoch, best_score = None, -1, -math.inf
    for t in trials:
        series = t.user_attrs.get(key)
        if not series:
            raise SearchError(f"{study.study_name} trial {t.number} records no per-epoch "
                              f"series under {key!r}")
        for epoch, value in enumerate(series):
            if value is not None and math.isfinite(value) and value > best_score:
                best, best_epoch, best_score = t, epoch, float(value)
    return best, best_epoch, best_score


def _ensv_choice(study: optuna.Study, trials: Sequence):
    """EnsV over every completed trial and epoch of the study (Sec. 4.5)."""
    subjects, candidates = None, []
    for t in trials:
        names, series = t.user_attrs.get(ENSV_SUBJECTS), t.user_attrs.get(ENSV_PROBABILITIES)
        if not names or not series:
            raise SearchError(f"{study.study_name} trial {t.number} records no EnsV "
                              f"probabilities ({ENSV_SUBJECTS}, {ENSV_PROBABILITIES})")
        if subjects is None:
            subjects = list(names)
        if sorted(names) != sorted(subjects):
            raise SearchError(f"{study.study_name} trial {t.number} predicted other target "
                              f"subjects than trial {trials[0].number}")
        order = [list(names).index(s) for s in subjects]
        for epoch, rows in enumerate(series):
            P = np.asarray([rows[k] for k in order], dtype=np.float64)
            if np.all(np.isfinite(P)):
                candidates.append((t, epoch, P))
    if not candidates:
        return None, -1, -math.inf
    pseudo = np.mean([P for _, _, P in candidates], axis=0).argmax(axis=1)
    best, best_epoch, best_score = None, -1, -math.inf
    for t, epoch, P in candidates:
        score = float(np.mean(P.argmax(axis=1) == pseudo))
        if score > best_score:
            best, best_epoch, best_score = t, epoch, score
    return best, best_epoch, best_score


def select(study: optuna.Study, rule: str = "primary") -> dict:
    """Sec. 4.5: the completed trial and epoch `rule` selects in one outer fold's study.

    `primary`: the best pooled score of the study's phase (`PRIMARY_SERIES`); `ensv`: the
    EnsV choice over every completed (trial, epoch) of a phase-2 study. Ties go to the
    lowest trial number, then to the earliest epoch; a non-finite score or probability is
    not a candidate. `epochs` is the refit's epoch count (the epoch index plus one)."""
    if rule not in RULES:
        raise SearchError(f"the selection rule is one of {list(RULES)}; got {rule!r}")
    trials = _completed(study)
    phase = _study_phase(study, trials)
    if rule == "ensv":
        if phase != 2:
            raise SearchError(f"{study.study_name}: EnsV selects in a phase-2 study only")
        best, best_epoch, best_score = _ensv_choice(study, trials)
    else:
        best, best_epoch, best_score = _primary_choice(study, trials, PRIMARY_SERIES[phase])
    if best is None:
        raise SearchError(f"{study.study_name}: no trial has a finite {rule} score")
    out = {"rule": rule, "trial": best.number, "params": dict(best.params),
           "epoch": best_epoch, "epochs": best_epoch + 1, "score": best_score}
    attrs = study.user_attrs
    if "outerFold" in attrs:
        # The fields the results tables read: the folds the inner splits validated on and
        # the outer test fold no decision reads.
        out.update(rotation=int(attrs["outerFold"]), testFold=int(attrs["testFold"]),
                   selectionRotations=[int(x["validation_fold"])
                                       for x in attrs["innerSplits"]],
                   excludedRotation=int(attrs["testFold"]))
    return out


# ------------------------------------------------------------------ pooled early stopping


@dataclass
class PooledStop:
    """Plan v59's stop of one search trial over its pooled per-epoch score: an epoch
    improves when its score is strictly greater than every earlier one (a non-finite score
    never improves); after `update`, the trial stops when it trained `cap` epochs, or at
    least `minimum` epochs and the last `patience` ones did not improve."""

    patience: int
    cap: int
    minimum: int = 1
    trained: int = 0
    best: float = -math.inf
    best_epoch: int = -1

    def update(self, score: float) -> bool:
        epoch, self.trained = self.trained, self.trained + 1
        if math.isfinite(score) and score > self.best:
            self.best, self.best_epoch = float(score), epoch
        return self.trained >= self.cap or (self.trained >= self.minimum
                                            and epoch - self.best_epoch >= self.patience)


def _rng_state() -> tuple:
    cuda = (torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized() else None)
    return torch.get_rng_state(), cuda, np.random.get_state(), random.getstate()


def _set_rng_state(state: tuple) -> None:
    cpu, cuda, numpy_state, python_state = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    np.random.set_state(numpy_state)
    random.setstate(python_state)


class _Lockstep:
    """The three inner splits of one trial, each trained in its own thread, advancing one
    epoch at a time in split order: only the split whose turn it is computes (so peak
    memory is one split's), each keeps its own global RNG state across its turns (so its
    numbers are those of a serial run), and after the last split's epoch e the pooled
    score of epoch e decides whether every split trains epoch e + 1."""

    def __init__(self, n: int, score: Callable[[Sequence[Mapping]], float], stop: PooledStop):
        self.n, self.score, self.stop = n, score, stop
        self.cond = threading.Condition()
        self.turn = 0
        self.alive = [True] * n
        self.pending: dict[int, dict[int, Mapping]] = {}
        self.decided = -1
        self.stopped = False
        self.error: BaseException | None = None
        self.series: list[float] = []
        self.rng: list[tuple | None] = [None] * n

    def _next(self, i: int) -> int | None:
        for k in range(1, self.n + 1):
            j = (i + k) % self.n
            if self.alive[j]:
                return j
        return None

    def _fail(self, error: BaseException) -> None:
        if self.error is None:
            self.error = error
        self.stopped = True

    def _decide(self, epoch: int) -> None:
        records = self.pending.pop(epoch, {})
        if not self.stopped:
            if sorted(records) != list(range(self.n)):
                self._fail(SearchError(f"epoch {epoch} was recorded by inner splits "
                                       f"{sorted(records)} only; the score pools all "
                                       f"{self.n}"))
            else:
                try:
                    value = float(self.score([records[k] for k in range(self.n)]))
                except Exception as error:      # failed the trial, never hidden
                    self._fail(error)
                else:
                    self.series.append(value)
                    if self.stop.update(value):
                        self.stopped = True
        self.decided = epoch

    def _hand_over(self, i: int, epoch: int | None) -> None:
        j = self._next(i)
        if epoch is not None and (j is None or j <= i):
            self._decide(epoch)        # split i was the last of this epoch's round
        self.turn = j if j is not None else i
        self.cond.notify_all()

    def run(self, train_split: Callable[[int], Mapping]) -> list[Mapping]:
        outs: list[Mapping | None] = [None] * self.n

        def work(i: int) -> None:
            control = _SplitControl(self, i)
            with self.cond:
                self.cond.wait_for(lambda: self.turn == i)
                failed = self.error is not None
            try:
                if failed:                      # an earlier split already failed the trial
                    return
                with two_phase.epoch_control(control):
                    outs[i] = train_split(i)
                if control.records:
                    raise SearchError(f"inner split {i}'s trainer recorded an epoch after "
                                      f"its last `two_phase.epoch_range` boundary")
            except BaseException as error:      # noqa: BLE001 -- re-raised below
                with self.cond:
                    self._fail(error)
            finally:
                with self.cond:
                    self.alive[i] = False
                    last = max(self.pending, default=None)
                    if last is not None and all(i_ in self.pending[last]
                                                for i_ in range(self.n) if self.alive[i_]):
                        self._decide(last)
                    self._hand_over(i, None)

        threads = [threading.Thread(target=work, args=(i,), name=f"inner-split-{i}",
                                    daemon=True) for i in range(self.n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if self.error is not None:
            raise self.error
        return outs


class _SplitControl:
    """One split's side of the lockstep (`two_phase.epoch_control`)."""

    def __init__(self, lockstep: _Lockstep, i: int):
        self.lockstep, self.i, self.records = lockstep, i, []

    def collect(self, record: Mapping) -> None:
        self.records.append(record)

    def boundary(self, epoch: int) -> bool:
        ls, i = self.lockstep, self.i
        with ls.cond:
            if [int(r["epoch"]) for r in self.records] != [epoch]:
                ls._fail(SearchError(f"inner split {i} recorded epochs "
                                     f"{[r['epoch'] for r in self.records]} in epoch {epoch}; "
                                     f"a trainer records exactly one validation per epoch"))
            else:
                ls.pending.setdefault(epoch, {})[i] = self.records[0]
            self.records = []
            ls.rng[i] = _rng_state()
            ls._hand_over(i, epoch)
            ls.cond.wait_for(lambda: ls.turn == i and (ls.decided >= epoch or ls.stopped))
            _set_rng_state(ls.rng[i])
            return not ls.stopped


def train_inner_splits(train_split: Callable[[int], Mapping],
                       score: Callable[[Sequence[Mapping]], float], phase: int,
                       cap: int, minimum: int = 1) -> tuple[list[Mapping], dict]:
    """Plan v59, Secs. 4.5 and 7.6: one search trial's trainings of its `N_INNER` inner
    splits (`train_split(i)` -> that split's trainer result), in lockstep, stopped by the
    pooled score `score(records of one epoch, one per split)` with `PATIENCE[phase]`, at
    most `cap` epochs and never before `minimum`. Returns the split results (each trainer
    returns normally, its history and validation holding the trained epochs) and the
    stop: trained epochs, whether the cap was reached and the pooled series."""
    stop = PooledStop(patience=int(PATIENCE[phase]), cap=int(cap), minimum=int(minimum))
    lockstep = _Lockstep(N_INNER, score, stop)
    outs = lockstep.run(train_split)
    trained = {len(o["validation"]) for o in outs}
    if len(trained) != 1:
        raise SearchError(f"the inner splits trained different epoch counts: {sorted(trained)}")
    epochs = trained.pop()
    replay = PooledStop(patience=stop.patience, cap=stop.cap, minimum=stop.minimum)
    expected = next((k + 1 for k, v in enumerate(lockstep.series) if replay.update(v)), None)
    if lockstep.series and (len(lockstep.series) != epochs or expected != epochs):
        raise SearchError(f"the trial trained {epochs} epochs; the pooled stop rule ends it "
                          f"at {expected}")
    return outs, {"trainedEpochs": epochs, "reachedCap": epochs >= int(cap),
                  "patience": stop.patience, "epochCap": int(cap)}


def stopping_diagnostic(name: str, studies: Sequence[optuna.Study]) -> dict:
    """Plan v59's declared diagnostic of the pooled early stopping, for one strategy (or
    phase-1 group) over its studies: the fraction of its trials that reached the epoch cap,
    and per selection rule the fraction of selected epochs equal to the last epoch their
    trial trained (a selection at the last trained epoch says the cap, not the patience,
    may have bound it)."""
    trials = [t for st_ in studies for t in _completed(st_)]
    capped = sum(bool(t.user_attrs.get("reachedCap")) for t in trials)
    at_last: dict[str, dict] = {}
    for st_ in studies:
        rules = RULES if _study_phase(st_, _completed(st_)) == 2 else ("primary",)
        for rule in rules:
            sel = select(st_, rule)
            trial = next(t for t in st_.trials if t.number == sel["trial"])
            row = at_last.setdefault(rule, {"selections": 0, "atLast": 0})
            row["selections"] += 1
            row["atLast"] += int(sel["epochs"] == trial.user_attrs.get("trainedEpochs"))
    for row in at_last.values():
        row["fraction"] = row["atLast"] / row["selections"] if row["selections"] else None
    return {"name": name, "studies": len(studies), "trials": len(trials),
            "trialsReachedCap": capped,
            "fractionTrialsReachedCap": capped / len(trials) if trials else None,
            "selectedAtLastTrainedEpoch": at_last}


def _search_cap(schedule: Mapping, phase: int) -> int:
    cap = int(schedule[f"epochs{phase}"])
    if cap > EPOCH_CAP:
        raise SearchError(f"epochs{phase} = {cap} exceeds the search's epoch cap of "
                          f"{EPOCH_CAP} (plan v59, Secs. 4.5 and 7.6)")
    return cap


# ------------------------------------------------------------------ training units


def _rotation(data: SearchData, rotation: int) -> Mapping[str, list[two_phase.BagTensor]]:
    try:
        return data.rotations[rotation]
    except KeyError:
        raise SearchError(f"rotation {rotation} is not in the search data") from None


#: Phase-1 group -> its own source trainer, for a group whose unit trains phase 1 with its
#: native loss. A group absent here trains with `two_phase.train_source_phase`.
PHASE_ONE_TRAINERS: dict[str, Callable] = {}


def register_phase_one(group: str, trainer: Callable) -> None:
    """Wire the native phase-1 trainer of a group registered by its own unit.

    `trainer` has the signature of `two_phase.train_source_phase`:
    `(model, source_train, source_valid, hyper, generator, device, *, seed, fold, transfer)`.
    The common group and MIL-CREDA keep the shared trainer.
    """
    if group not in model_store.GROUPS or group in ("common", "MIL-CREDA"):
        raise SearchError(f"{group!r} is not a phase-1 group registered by its own unit")
    PHASE_ONE_TRAINERS[group] = trainer


def train_phase_one(group: str, data: SearchData, rotation: int, seed: int, hyper: Mapping,
                    device: torch.device | str = "cpu") -> dict:
    """Phase 1 of `group` on one rotation (its registered trainer, else
    `two_phase.train_source_phase`)."""
    device = torch.device(device)
    d = _rotation(data, rotation)
    channels = data.channels()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = model_store.build_model(group, channels["source"], channels["target"],
                                    hyper).to(device)
    trainer = PHASE_ONE_TRAINERS.get(group, two_phase.train_source_phase)
    result = trainer(model, d["source_train"], d["source_valid"], hyper, generator, device,
                     seed=seed, fold=rotation, transfer=data.transfer.name)
    result["history"] = [{**r, "model": f"{group}:phase1"} for r in result["history"]]
    return {"group": group, "model": model, "result": result, "hyper": dict(hyper),
            "channels": channels}


def save_phase_one(root: str | Path, group: str, data: SearchData, rotation: int, seed: int,
                   trained: Mapping) -> Path:
    """Store a phase-1 checkpoint with its hyperparameters, partition and sigma."""
    checkpoint = trained["result"]["checkpoint"]
    extra = {"epoch": checkpoint["epoch"],
             "valid_balanced_accuracy": checkpoint["valid_balanced_accuracy"]}
    if group == "MIL-CREDA":
        extra["sigma_rule"] = trained["result"]["sigma"]
    return model_store.save(root, transfer=data.transfer, rotation=rotation, seed=seed,
                            phase=1, name=group, state=checkpoint["state"],
                            hyper=_stored_hyper(trained["hyper"]),
                            partition_sha256=data.partition_sha256,
                            channels=trained["channels"],
                            sigma=checkpoint["sigma"] if group == "MIL-CREDA" else None,
                            extra=extra)


def train_mil_creda_phase_two(data: SearchData, rotation: int, seed: int, record,
                              params: Mapping, schedule: Mapping,
                              device: torch.device | str = "cpu") -> dict:
    """MIL-CREDA's phase 2 from its stored phase-1 checkpoint, at that checkpoint's sigma."""
    state, meta = record
    sigma1 = meta.get("sigma")
    if sigma1 is None:
        raise SearchError("the MIL-CREDA phase-1 checkpoint carries no sigma; phase 2 runs at "
                          "exactly that value (sigma2 = sigma1) and never searches it")
    device = torch.device(device)
    hyper = two_phase.check_hyper({**{k: meta["hyper"][k] for k in two_phase.PHASE_ONE_KEYS},
                                   **params, **check_schedule(schedule)})
    d = _rotation(data, rotation)
    channels = data.channels()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = model_store.build_model("MIL-CREDA", channels["source"], channels["target"],
                                    hyper).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise SearchError(f"the phase-1 state does not fit (missing {missing}, unexpected "
                          f"{unexpected})")
    model.fix_sigma(sigma1)
    joint = two_phase.bridge(model)
    result = two_phase.adapt_target_phase(
        model, {"checkpoint": {"sigma": sigma1}}, joint["frozen_state"], d["source_train"],
        d["target_train"], d["target_valid"], hyper, generator, device, seed=seed,
        fold=rotation, transfer=data.transfer.name)
    return {"strategy": "MIL-CREDA", "model": model, "result": result, "hyper": hyper,
            "channels": channels, "layer_copy": joint["layer_copy"]}


#: Strategy -> its phase-2 trainer. Comparators register through `register_phase_two`.
PHASE_TWO_TRAINERS: dict[str, Callable] = {"MIL-CREDA": train_mil_creda_phase_two}


def train_phase_two(strategy: str, data: SearchData, rotation: int, seed: int, record,
                    params: Mapping, schedule: Mapping,
                    device: torch.device | str = "cpu") -> dict:
    """One phase-2 training of `strategy` (its registered trainer)."""
    space = phase_two_space(strategy)
    if set(params) != set(space):
        raise SearchError(f"the {strategy} phase-2 parameters are exactly {list(space)}; "
                          f"got {sorted(params)}")
    return PHASE_TWO_TRAINERS[strategy](data, rotation, seed, record, params, schedule,
                                        device)


def source_representations(strategy: str, data: SearchData, rotation: int, record,
                           device: torch.device | str = "cpu") -> list[dict]:
    """Sec. 4.5's source pass of the centroid rule, per inner split of outer fold
    `rotation`: the source validation bags through the frozen source trunk and the frozen
    attention or aggregator of `record`, the outer fold's one phase-1 checkpoint of the
    strategy's group, which every phase-2 trial keeps unchanged (Sec. 7.3; the trainers
    assert it, `two_phase.FrozenPartsMoved`). Inference only, never a training; it runs
    here, outside every trainer, so a source-free trainer (CLUST) reads no source data.
    Split i -> {"subjects", "labels", "representations"} (the labels are the SOURCE
    labels the rule scores against)."""
    group = PHASE_ONE_GROUP_OF[strategy]
    state, meta = record
    device = torch.device(device)
    channels = data.channels()
    model = model_store.build_model(group, channels["source"], channels["target"],
                                    meta.get("hyper") or {})
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("target_trunk.") for k in missing):
        raise SearchError(f"the phase-1 state does not fit {group} (missing {missing}, "
                          f"unexpected {unexpected})")
    if group == "MIL-CREDA":
        model.fix_sigma(meta["sigma"])
    model.to(device)
    out = []
    for i in range(N_INNER):
        bags = data.inner_view(rotation, i).rotations[rotation]["source_valid"]
        if not bags:
            raise SearchError(f"inner split {i} of rotation {rotation} validates no source bag")
        Z = two_phase.represent(model, bags, "source", device)
        out.append({"subjects": [b.subject for b in bags], "labels": [int(b.label) for b in bags],
                    "representations": Z.tolist()})
    return out


def save_phase_two(root: str | Path, strategy: str, data: SearchData, rotation: int,
                   seed: int, trained: Mapping, selection: Mapping | None = None,
                   phase_one: Mapping | None = None) -> Path:
    """Store a selected phase-2 model whole (frozen parts and the selected target trunk),
    with the Sec. 4.5 selection that chose its configuration and the identity of the
    phase-1 checkpoint it started from (`phase_one`, its sidecar), when given."""
    checkpoint = trained["result"]["checkpoint"]
    extra = {"selection": dict(selection)} if selection is not None else {}
    if phase_one is not None:
        extra["phaseOneCheckpoint"] = phase_one_checkpoint_identity(phase_one)
    return model_store.save(root, transfer=data.transfer, rotation=rotation, seed=seed,
                            phase=2, name=strategy, state=trained["model"].state_dict(),
                            hyper=dict(checkpoint["hyper"]),
                            partition_sha256=data.partition_sha256,
                            channels=trained["channels"], sigma=checkpoint.get("sigma"),
                            extra={"epoch": checkpoint["epoch"],
                                   "valid_balanced_accuracy":
                                       checkpoint["valid_balanced_accuracy"],
                                   **extra})


def phase_one_checkpoint_identity(meta: Mapping) -> dict:
    """What names a stored phase-1 checkpoint: its transfer, outer fold, seed, group,
    partition and the digest of its state (`model_store.save`)."""
    return {k: meta.get(k) for k in PHASE_ONE_CHECKPOINT_FIELDS}


def _check_phase_one_record(strategy: str, data: SearchData, rotation: int, record) -> None:
    """Phase 2 of outer fold `rotation` starts from that fold's one phase-1 checkpoint of
    the strategy's group, at `PHASE_ONE_SEED` and the partition in force (Sec. 4.5)."""
    group = PHASE_ONE_GROUP_OF[strategy]
    _, meta = record
    expected = {"phase": 1, "name": group, "transfer": data.transfer.name,
                "rotation": int(rotation), "seed": PHASE_ONE_SEED,
                partition.HASH_FIELD: data.partition_sha256}
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise SearchError(f"rotation {rotation}: the phase-1 checkpoint is not the one {group} "
                          f"checkpoint of {data.transfer.name} and this outer fold: {wrong}")


# ------------------------------------------------------------------ studies


def _outer(rotation: int) -> int:
    if not isinstance(rotation, (int, np.integer)) or not 0 <= int(rotation) < partition.N_FOLDS:
        raise SearchError(f"a study belongs to one outer fold of {list(range(partition.N_FOLDS))}"
                          f"; got {rotation!r}")
    return int(rotation)


def study_path(root: str | Path, transfer: assignment_mod.Transfer, phase: int,
               name: str, rotation: int) -> Path:
    """Sec. 4.5: one study per group or strategy, transfer and outer fold."""
    if phase == 1 and name not in PHASE_ONE_SPACES:
        raise SearchError(f"phase 1 studies a group {list(PHASE_ONE_SPACES)}, not {name!r}")
    if phase == 2 and name not in model_store.STRATEGIES:
        raise SearchError(f"phase 2 studies a strategy, not {name!r}")
    return (Path(root) / "Results" / "hpo"
            / f"{model_store.transfer_slug(transfer)}__phase{phase}__"
              f"{model_store.name_slug(name)}__outer{_outer(rotation)}.db")


def make_study(path: str | Path, name: str) -> optuna.Study:
    path = Path(path)
    if path.exists():
        raise SearchError(f"{path} exists: a study is never resumed or extended silently")
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name=name, storage=f"sqlite:///{path}",
                               direction="maximize",
                               sampler=optuna.samplers.GPSampler(seed=SAMPLER_SEED),
                               pruner=optuna.pruners.NopPruner())


#: The fields that make a stored study the one a run asks for. A stored study whose fields
#: differ is refused, naming them; one that matches and holds fewer completed trials than
#: its declared count is continued up to that count.
IDENTITY_FIELDS = ("transfer", "arm", "phase", "name", "outerFold", "testFold",
                   "innerSplits", "sampler", "samplerSeed", "nTrials", "pruner",
                   "searchSeed", "reportSeeds", "schedule", "objective", "space",
                   partition.HASH_FIELD, "revision", "code_version")
#: A phase-2 study is also the one started from its phase-1 checkpoint (Sec. 4.5).
PHASE_TWO_IDENTITY_FIELDS = ("phaseOneCheckpoint",)


def _json(value):
    import json
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def study_identity(data: SearchData, phase: int, name: str, rotation: int,
                   schedule: Mapping, n_trials: int, objective: str,
                   phase_one: Mapping | None = None) -> dict:
    """What a study is: its transfer, phase, outer fold with its test fold and inner
    splits, budget, space, sampler and the partition, plan revision and code version it
    was run against; a phase-2 study, also the phase-1 checkpoint it starts from
    (`phase_one`, its sidecar), when given."""
    space = PHASE_ONE_SPACES[name] if phase == 1 else phase_two_space(name)
    rotation = _outer(rotation)
    if rotation not in data.folds:
        raise SearchError(f"rotation {rotation} has no folds in the search data")
    test = int(data.folds[rotation]["test_fold"])
    splits = data.folds[rotation].get("inner_splits")
    if not splits:
        raise SearchError(f"rotation {rotation} stores no inner splits in the search data")
    return _json({
        "transfer": data.transfer.name, "arm": data.transfer.arm, "phase": phase,
        "name": name, "outerFold": rotation, "testFold": test,
        "innerSplits": splits,
        "sampler": "GPSampler", "samplerSeed": SAMPLER_SEED,
        "nTrials": n_trials, "pruner": "none", "searchSeed": SEARCH_SEED,
        "reportSeeds": list(REPORT_SEEDS), "schedule": dict(schedule),
        "objective": objective, "space": {k: repr(v) for k, v in space.items()},
        partition.HASH_FIELD: data.partition_sha256,
        "revision": __provenance__["revision"], "code_version": model_store.code_version(),
        **({"phaseOneCheckpoint": phase_one_checkpoint_identity(phase_one)}
           if phase_one is not None else {})})


def identity_differences(stored: Mapping, current: Mapping,
                         fields: Sequence[str] | None = None) -> list[str]:
    """`field: stored X, current Y` for every field of `current` the stored record differs in."""
    return [f"{k}: stored {_json(stored.get(k))!r}, current {current[k]!r}"
            for k in (fields or current) if _json(stored.get(k)) != current[k]]


def open_study(path: str | Path, identity: Mapping) -> tuple[optuna.Study, int]:
    """(study, completed trials): a new study, or the stored one of the same identity.

    A stored study of another identity is refused, naming what differs. A continued
    study's sampler is seeded with `SAMPLER_SEED` plus the number of trials it already
    holds, so the continuation does not replay the first draws of the seeded sampler; an
    uninterrupted study runs at `SAMPLER_SEED` exactly.
    """
    path = Path(path)
    if not path.exists():
        study = make_study(path, path.stem)
        for key, value in identity.items():
            study.set_user_attr(key, value)
        return study, 0
    study = optuna.load_study(study_name=path.stem, storage=f"sqlite:///{path}")
    differ = identity_differences(study.user_attrs, identity, [
        *IDENTITY_FIELDS, *(k for k in PHASE_TWO_IDENTITY_FIELDS if k in identity)])
    if differ:
        changed = (" (the search rule changed in plan v59, Secs. 4.5 and 7.6: each trial "
                   f"trains at most {EPOCH_CAP} epochs and stops on its pooled selection "
                   "score's patience, so a study searched under an earlier plan, v58 "
                   "included, is searched again)"
                   if any(d.startswith("objective:") for d in differ) else "")
        raise SearchError(f"{path} is a study of another identity and is not continued"
                          f"{changed}: " + "; ".join(differ) + "; remove it to search again")
    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if complete > identity["nTrials"]:
        raise SearchError(f"{path} holds {complete} completed trials, more than its declared "
                          f"{identity['nTrials']}")
    if complete < identity["nTrials"]:
        held = len(study.trials)
        study.sampler = optuna.samplers.GPSampler(seed=SAMPLER_SEED + held)
        study.set_user_attr("resumedAfter", [*study.user_attrs.get("resumedAfter", []), held])
    return study, complete


def _optimize(study: optuna.Study, objective: Callable, remaining: int,
              should_stop: Callable[[optuna.Study], bool] | None) -> None:
    """`remaining` trials, or fewer when `should_stop` answers true between two of them."""
    if should_stop is None:
        study.optimize(objective, n_trials=remaining)
        return
    if remaining <= 0 or should_stop(study):
        return

    def check(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if should_stop(study):
            study.stop()

    study.optimize(objective, n_trials=remaining, callbacks=[check])


def _set_attrs(trial: optuna.Trial, attrs: Mapping) -> None:
    for key, value in attrs.items():
        trial.set_user_attr(key, value)


def _mean_loss(histories: Sequence[Sequence[Mapping]]) -> list[float] | None:
    """Per epoch, the training loss averaged over the inner splits (Sec. 5.4's curves)."""
    if not all(h and all("loss" in r for r in h) for h in histories):
        return None
    return [float(np.mean([h[e]["loss"] for h in histories]))
            for e in range(min(len(h) for h in histories))]


def run_phase_one_study(group: str, data: SearchData, rotation: int, schedule: Mapping, *,
                        root: str | Path = DEFAULT_ROOT, n_trials: int = N_TRIALS,
                        device: torch.device | str = "cpu",
                        should_stop: Callable[[optuna.Study], bool] | None = None
                        ) -> optuna.Study:
    """One phase-1 study of outer fold `rotation`: `n_trials` configurations, each trained
    on its three inner splits for the whole `epochs1` cap and scored per epoch on the
    source subjects of the validating folds, pooled (Sec. 4.5)."""
    schedule = check_schedule(schedule)
    if group not in PHASE_ONE_SPACES:
        raise SearchError(f"{group!r} is not a phase-1 group")
    model_store.require_registered(group)
    path = study_path(root, data.transfer, 1, group, rotation)
    cap = _search_cap(schedule, 1)
    study, complete = open_study(path, study_identity(
        data, 1, group, rotation, schedule, n_trials, PHASE_ONE_OBJECTIVE))

    def objective(trial: optuna.Trial) -> float:
        hyper = phase_one_hyper(group, suggest(trial, PHASE_ONE_SPACES[group]), schedule)
        outs, stopped = train_inner_splits(
            lambda i: train_phase_one(group, data.inner_view(rotation, i), rotation,
                                      SEARCH_SEED, hyper, device)["result"],
            two_phase.pooled_balanced_accuracy, 1, cap, MIN_TRAINED_EPOCHS.get(group, 1))
        per_split = [out["validation"] for out in outs]
        histories = [out["history"] for out in outs]
        split_values = [[float(h["val_balanced_accuracy"]) for h in out["history"]]
                        for out in outs]
        series = pooled_balanced_accuracy_series(per_split)
        if not any(math.isfinite(v) for v in series):
            raise SearchError(f"no epoch has a finite pooled validation score: {series}")
        best = int(np.nanargmax(series))
        _set_attrs(trial, {"pooledBalancedAccuracy": series, "bestEpoch": best,
                           "epochs": best + 1, "splitBalancedAccuracy": split_values,
                           "trainLoss": _mean_loss(histories), "seed": SEARCH_SEED,
                           **stopped})
        return float(series[best])

    _optimize(study, objective, n_trials - complete, should_stop)
    return study


def run_phase_two_study(strategy: str, data: SearchData, rotation: int, record,
                        schedule: Mapping, *, root: str | Path = DEFAULT_ROOT,
                        n_trials: int = N_TRIALS,
                        device: torch.device | str = "cpu",
                        should_stop: Callable[[optuna.Study], bool] | None = None
                        ) -> optuna.Study:
    """One phase-2 study of outer fold `rotation`; `record` is that fold's one stored
    phase-1 checkpoint (state, meta) of the strategy's group, at `PHASE_ONE_SEED`. Every
    trial starts from it in each of the three inner splits, at the search seed; the
    study records its identity."""
    space = phase_two_space(strategy)
    schedule = check_schedule(schedule)
    _check_phase_one_record(strategy, data, rotation, record)
    path = study_path(root, data.transfer, 2, strategy, rotation)
    cap = _search_cap(schedule, 2)
    study, complete = open_study(path, study_identity(
        data, 2, strategy, rotation, schedule, n_trials, PHASE_TWO_OBJECTIVE,
        phase_one=record[1]))
    sources: list[dict] = []           # the source pass, once per inner split (Sec. 4.5)

    def objective(trial: optuna.Trial) -> float:
        if not sources:
            sources.extend(source_representations(strategy, data, rotation, record, device))
        params = suggest(trial, space)
        outs, stopped = train_inner_splits(
            lambda i: train_phase_two(strategy, data.inner_view(rotation, i), rotation,
                                      SEARCH_SEED, record, params, schedule,
                                      device)["result"],
            lambda records: two_phase.pooled_centroid_balanced_accuracy(records, sources),
            2, cap, MIN_TRAINED_EPOCHS.get(strategy, 1))
        per_split = [out["validation"] for out in outs]
        histories = [out["history"] for out in outs]
        sigmas = [out["checkpoint"].get("sigma") for out in outs]
        primary = pooled_centroid_series(per_split, sources)
        if not all(math.isfinite(v) for v in primary):
            raise SearchError(f"an epoch returned a non-finite pooled centroid score: "
                              f"{primary}")
        best = int(np.argmax(primary))
        _set_attrs(trial, {PRIMARY_SERIES[2]: primary, "bestEpoch": best, "epochs": best + 1,
                           **ensv_record(per_split),
                           "trainLoss": _mean_loss(histories), "sigmaBySplit": sigmas,
                           "seed": SEARCH_SEED, **stopped})
        return float(primary[best])

    _optimize(study, objective, n_trials - complete, should_stop)
    return study
