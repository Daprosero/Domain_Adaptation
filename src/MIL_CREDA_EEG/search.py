"""The Optuna searches of both phases, per transfer.

Experiments plan v43, Secs. 4.5, 7.3 and 7.7. Phase 1 has four studies per
transfer, one per pre-training group (`PHASE_ONE_SPACES`: the common group of
the six non-MIL comparators, AdaptMIL, HASD and MIL-CREDA); phase 2 has one
study per strategy per transfer. Every study gets the same budget: `N_TRIALS`
trials with Optuna's Gaussian-process sampler (`GPSampler`, experimental in
Optuna; its first ten trials are random) seeded with the fixed `SAMPLER_SEED`
so the search can be repeated, and no pruner (`NopPruner`): every trial runs
complete. The toy set's searches were the template of the MIL-CREDA rows of
Sec. 7.7; nothing of the retired toy-set runner is imported.

**A trial (Sec. 4.5).** Each configuration is trained from scratch in all four
rotations of the frozen partition with ONE training seed, `SEARCH_SEED`,
distinct from the single report seed (`REPORT_SEEDS`, plan v42 Sec. 4.6), so the search is not
tuned on the seeds that later measure it. The trial's value is the mean minus
the standard deviation over the four rotations (`robust`):

- phase 1: of the source-validation balanced accuracy of the restored best
  epoch (`two_phase.train_source_phase`, early stopping with `patience1`);
- phase 2: of the target-validation balanced accuracy of the selected epoch
  (`two_phase.adapt_target_phase`, which runs `epochs2` whole), Sec. 4.5's
  primary rule, which uses target labels of the validation fold for model
  selection only. The secondary label-free rule is recorded on the same trial
  as the user attribute `label_free_score`: per rotation the best per-epoch
  label-free score (entropy of the mean prediction minus mean per-bag entropy
  over the target validation bags, no labels), summarized by the same mean
  minus deviation. Nothing is trained for it in the search.

**Selection per test rotation (Secs. 4.5, 7.3, 7.7).** The value a phase-2 study
optimizes, what its sampler observes, is the mean minus sd over all four rotations.
The selection is not one for the four: each test fold is another rotation's
validation fold, so for the test fold of rotation r `select_per_rotation` picks
the completed trial with the highest mean minus sd over the three rotations whose
validation fold is not r's test fold (`selection_rotations`, derived from the
partition's rotations recorded in `SearchData.folds`, never from an offset). Both
rules (`RULES`) apply that exclusion. Each rotation may receive a different
configuration. The residual Sec. 4.5 declares stays: the sampler saw all four
rotations' scores, so r's test subjects may influence which configurations were
tried, never which one r gets. Phase 1 keeps one selection over the four rotations,
its study's best trial (Sec. 7.3: it reads source validation only).

The search reads only the training and validation roles of both sides
(`search_data`): no test-fold bag is ever loaded.

**Phase 2 starts from phase 1 (Secs. 7.3, 7.7).** A phase-2 trial of a
rotation starts again from the phase-1 checkpoint of that rotation, its group
and the search seed, read back from the model store (`model_store.load`, which
refuses a foreign partition). MIL-CREDA's sigma is that checkpoint's sigma
(`sigma2 = sigma1`) and is never searched: its phase-2 space is exactly
`lr2`, `wd2` and `tauLocal`, and its phase-1 choices are fixed.

**Spaces (Sec. 7.7).** Shared, log scale: `lr1` in [1e-4, 1e-1], `lr2` in
[1e-5, 1e-2], `wd1` and `wd2` in [1e-7, 1e-2]. MIL-CREDA: `attentionGamma` in
[0, 1], `attentionTemperature` and `tauLocal` in [0.1, 10] log,
`sigmaMultiplier` in {0.5, 0.75, 1, 1.5}. The comparators' phase-2 spaces
follow one generic rule (`LOSS_WEIGHT`, `KERNEL_MULTIPLIER`,
`CONFIDENCE_THRESHOLD`), registered per strategy by its own unit through
`register_phase_two`. Sec. 7.7 adds two comparator-specific axes, each admitted
only for its own strategy (`OWN_AXES`): CLUST's teacher EMA momentum rho in
[0.9, 0.999] with 1 - rho log-uniform, drawn as `EMA_MOMENTUM_COMPLEMENT`
(1 - rho in [1e-3, 1e-1], log) and converted by the CLUST trainer; and
CASUAL's VAT radius as a fraction of the typical window L2 norm, log-uniform
in [0.01, 1] (`VAT_RADIUS_FRACTION`), scaled by the CASUAL trainer. None is
wired here, and AdaptMIL and HASD have no
phase-1 architecture registered yet (`model_store.BUILDERS`), so their studies
refuse before any storage is created.

**Storage.** One SQLite file per study, `<root>/Results/hpo/
<source>-to-<target>__phase<p>__<group or strategy>.db`, with the study named
as the file's stem; `<root>` defaults to `MIL-CREDA/`. Each study records its
identity (`study_identity`: transfer, phase, budget, space, sampler and its seed,
schedule, the partition hash, the plan revision, the code version and the
rotations' folds). A study whose file exists is continued only when its identity
is the requested one, up to its declared trial count (an interrupted search
resumes), and returned as it is when complete; another identity is refused,
naming what differs (`open_study`). No budget is silently extended. A continued
study's sampler is seeded with `SAMPLER_SEED` plus the trials it already holds,
so it does not replay the seeded first draws; an interrupted-and-continued study
is therefore not the same sequence of trials as an uninterrupted one, and records
where it was continued (`resumedAfter`). Trials left running by a killed process
stay as they are: they are neither counted nor re-run.

**Implementation choices the plan does not fix.** The deviation is the sample
standard deviation (`SD_DDOF` = 1) over the four rotations. The seeds:
`SEARCH_SEED` = 0, `REPORT_SEEDS` = (1,), `SAMPLER_SEED` = 20261102. Epoch
caps and patience (`epochs1`, `patience1`, `epochs2`) and `bagsPerStep` are
inputs with no default: the pilot fixes them (Sec. 7.6). Groups other than
MIL-CREDA fix `attentionGamma` = 0, `attentionTemperature` = 1 and
`sigmaMultiplier` = 1 in the phase-1 trainer's key set
(`mil_wrapper.COMMON_PHASE_ONE_FIXED`); none of them is searched. Phase-2
keys that phase 1 does not read are set to NaN in the phase-1 hyperparameters,
so a phase-1 trainer that read them would fail rather than run at a value
nobody chose.
"""

from __future__ import annotations

import math
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
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v43.md",
    "sections": ["4.5", "7.3", "7.6", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": [
        "the_sampler_is_gpsampler_with_the_fixed_seed_and_no_pruner",
        "the_search_seed_is_not_a_report_seed",
        "the_spaces_are_exactly_those_of_sec_7_7",
        "no_comparator_phase_two_is_wired_and_the_hook_takes_only_the_generic_rule",
        "the_comparator_specific_axes_are_admitted_only_for_their_own_strategy",
        "epoch_caps_and_patience_have_no_default",
        "the_objective_is_mean_minus_sd_over_all_four_rotations",
        "the_search_never_loads_a_test_fold_bag",
        "the_phase_two_mil_creda_sigma_comes_from_the_phase_one_checkpoint",
        "phase_two_starts_only_from_the_phase_one_checkpoints_at_the_search_seed",
        "the_label_free_score_is_recorded_per_trial",
        "the_excluded_rotation_is_derived_from_the_frozen_partition",
        "the_selection_is_per_test_rotation_by_mean_minus_sd_over_three_rotations",
        "the_label_free_rule_applies_the_same_exclusion",
        "the_labels_of_r_test_subjects_cannot_change_r_selection",
        "the_study_records_each_rotations_folds_from_the_partition",
        "an_interrupted_study_of_the_same_identity_continues_to_its_declared_count",
        "a_study_of_another_identity_is_refused_naming_what_differs",
        "a_study_records_the_identity_it_is_checked_against",
        "unwired_groups_and_strategies_refuse_before_creating_storage",
        "a_two_trial_smoke_study_completes_and_writes_its_sqlite",
    ],
}

#: Sec. 7.7: 20 trials per study, GPSampler with a fixed seed, no pruning.
N_TRIALS = 20
SAMPLER_SEED = 20261102
#: Sec. 7.7: one search training seed, distinct from the single report seed (Sec. 4.6).
SEARCH_SEED = 0
REPORT_SEEDS = (1,)
#: Sec. 4.5: mean minus the (sample) standard deviation over the rotations.
SD_DDOF = 1
DEFAULT_ROOT = bag_data.REPOSITORY / "MIL-CREDA"
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
#: Strategy -> the axes outside the generic rule that Sec. 7.7 assigns to it alone.
OWN_AXES: dict[str, tuple[BaseDistribution, ...]] = {
    "CLUST — variante source-free": (EMA_MOMENTUM_COMPLEMENT,),
    "CASUAL": (VAT_RADIUS_FRACTION,),
}

_SHARED_PHASE_ONE = {"lr1": LR1, "wd1": WEIGHT_DECAY}
PHASE_ONE_SPACES: dict[str, dict[str, BaseDistribution]] = {
    "common": dict(_SHARED_PHASE_ONE),
    "AdaptMIL": dict(_SHARED_PHASE_ONE),
    "HASD": dict(_SHARED_PHASE_ONE),
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
PHASE_ONE_GROUP_OF = {s: (s if s in ("AdaptMIL", "HASD", "MIL-CREDA") else mil_wrapper.GROUP)
                      for s in model_store.STRATEGIES}


class SearchError(RuntimeError):
    """A search was asked for something Secs. 4.5 and 7.7 do not allow."""


class NotWired(SearchError):
    """The strategy's phase-2 training is not implemented yet."""


# ------------------------------------------------------------------ data


@dataclass(frozen=True)
class SearchData:
    """The training and validation bags of both sides, for every rotation."""

    transfer: assignment_mod.Transfer
    partition_sha256: str
    rotations: Mapping[int, Mapping[str, list[two_phase.BagTensor]]]
    #: Rotation -> its test and validation folds, read from the frozen partition.
    folds: Mapping[int, Mapping[str, int]] = field(default_factory=dict)

    def channels(self) -> dict[str, int]:
        first = self.rotations[min(self.rotations)]
        return {"source": int(first["source_train"][0].windows.shape[1]),
                "target": int(first["target_train"][0].windows.shape[1])}


def search_data(transfer: assignment_mod.Transfer, bags: bag_data.BagSet) -> SearchData:
    """Every rotation's train and validation bags; test-fold bags are never read."""
    a = assignment_mod.assign_from_bags(transfer, bags)
    rotations = sorted(int(r) for r in bags.rotations)
    if rotations != list(range(partition.N_FOLDS)):
        raise SearchError(f"the search runs over the {partition.N_FOLDS} rotations of the "
                          f"partition; got {rotations}")
    cache: dict[tuple[str, str], two_phase.BagTensor] = {}

    def tensors(side: str, r: int, role: str) -> list[two_phase.BagTensor]:
        out = []
        for b in assignment_mod.side_bags(a, bags, side, r, role):
            if (b.subject, b.condition) not in cache:
                cache[(b.subject, b.condition)] = two_phase.bag_tensors([b])[0]
            out.append(cache[(b.subject, b.condition)])
        return out

    return SearchData(transfer=transfer, partition_sha256=a.partition_sha256,
                      folds=rotation_folds(bags.rotations), rotations={
        r: {"source_train": tensors("source", r, "train"),
            "source_valid": tensors("source", r, "validation"),
            "target_train": tensors("target", r, "train"),
            "target_valid": tensors("target", r, "validation")} for r in rotations})


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
    `{"result": {"checkpoint": {"valid_balanced_accuracy", "label_free_score"},
    "history": [{"label_free_score"}, ...]}}`, as `train_mil_creda_phase_two`.
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


def robust(values: Sequence[float]) -> float:
    """Sec. 4.5: mean minus standard deviation over the four rotations."""
    if len(values) != partition.N_FOLDS:
        raise SearchError(f"the summary is over all {partition.N_FOLDS} rotations; got "
                          f"{len(values)} values")
    values = np.asarray(values, dtype=float)
    return float(values.mean() - values.std(ddof=SD_DDOF))


# ------------------------------------------------------------------ selection (Sec. 4.5)

#: Sec. 4.5's two rules -> the per-rotation trial attribute each one reads.
RULES = {"primary": "rotationValues", "label-free": "labelFreeRotationValues"}


def rotation_folds(rotations: Mapping) -> dict[int, dict[str, int]]:
    """Rotation -> its test and validation folds, from the frozen partition's rotations."""
    return {int(r): {"test_fold": int(v["test_fold"]),
                     "validation_fold": int(v["validation_fold"])}
            for r, v in rotations.items()}


def selection_rotations(rotations: Mapping, rotation: int) -> list[int]:
    """Sec. 4.5: the rotations whose validation fold is not `rotation`'s test fold.

    Derived from the partition (`rotation_folds`), never from an offset: the one rotation
    left out is the one that validates on the subjects `rotation` is tested on.
    """
    folds = rotation_folds(rotations)
    if sorted(folds) != list(range(partition.N_FOLDS)):
        raise SearchError(f"the selection reads the {partition.N_FOLDS} rotations of the "
                          f"partition; got {sorted(folds)}")
    test = folds[int(rotation)]["test_fold"]
    kept = [q for q in sorted(folds) if folds[q]["validation_fold"] != test]
    if len(kept) != partition.N_FOLDS - 1:
        raise SearchError(f"rotation {rotation}: Sec. 4.5 selects over three rotations, the "
                          f"ones whose validation fold is not its test fold {test}; the "
                          f"partition leaves {kept}")
    return kept


def robust_selection(values: Sequence[float]) -> float:
    """Sec. 4.5: mean minus standard deviation over the three selection rotations."""
    if len(values) != partition.N_FOLDS - 1:
        raise SearchError(f"a per-rotation selection is over {partition.N_FOLDS - 1} "
                          f"rotations; got {len(values)} values")
    values = np.asarray(values, dtype=float)
    return float(values.mean() - values.std(ddof=SD_DDOF))


def select_per_rotation(study: optuna.Study, rotations: Mapping,
                        rule: str = "primary") -> dict[int, dict]:
    """Sec. 4.5: per test rotation r, the completed trial with the highest mean minus sd
    of `rule`'s per-rotation values over `selection_rotations(rotations, r)`.

    Ties go to the lowest trial number. A trial whose summary is not finite is not a
    candidate. Phase 1 is not selected here: it keeps one selection over the four
    rotations, its study's best trial (Sec. 7.3).
    """
    if rule not in RULES:
        raise SearchError(f"the selection rule is one of {list(RULES)}; got {rule!r}")
    key = RULES[rule]
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        raise SearchError(f"{study.study_name}: no completed trial to select from")
    folds = rotation_folds(rotations)
    out = {}
    for r in sorted(folds):
        kept = selection_rotations(folds, r)
        best, best_score = None, -math.inf
        for t in trials:
            values = t.user_attrs.get(key)
            if values is None or len(values) != partition.N_FOLDS:
                raise SearchError(f"{study.study_name} trial {t.number} records no "
                                  f"{partition.N_FOLDS} per-rotation values under {key!r}")
            score = robust_selection([values[q] for q in kept])
            if math.isfinite(score) and score > best_score:
                best, best_score = t, score
        if best is None:
            raise SearchError(f"{study.study_name}: no trial has a finite {rule} score for "
                              f"rotation {r}")
        out[r] = {"rotation": r, "rule": rule, "trial": best.number,
                  "params": dict(best.params), "score": best_score,
                  "testFold": folds[r]["test_fold"], "selectionRotations": kept,
                  "excludedRotation": next(q for q in sorted(folds) if q not in kept)}
    return out


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
    space = phase_two_space(strategy)
    if set(params) != set(space):
        raise SearchError(f"the {strategy} phase-2 parameters are exactly {list(space)}; "
                          f"got {sorted(params)}")
    return PHASE_TWO_TRAINERS[strategy](data, rotation, seed, record, params, schedule,
                                        device)


def save_phase_two(root: str | Path, strategy: str, data: SearchData, rotation: int,
                   seed: int, trained: Mapping, selection: Mapping | None = None) -> Path:
    """Store a selected phase-2 model whole (frozen parts and the selected target trunk),
    with the Sec. 4.5 selection that chose its configuration when one is given."""
    checkpoint = trained["result"]["checkpoint"]
    extra = {"selection": dict(selection)} if selection is not None else {}
    return model_store.save(root, transfer=data.transfer, rotation=rotation, seed=seed,
                            phase=2, name=strategy, state=trained["model"].state_dict(),
                            hyper=dict(checkpoint["hyper"]),
                            partition_sha256=data.partition_sha256,
                            channels=trained["channels"], sigma=checkpoint.get("sigma"),
                            extra={"epoch": checkpoint["epoch"],
                                   "valid_balanced_accuracy":
                                       checkpoint["valid_balanced_accuracy"],
                                   "label_free_score": checkpoint["label_free_score"],
                                   **extra})


def _check_phase_one_records(strategy: str, data: SearchData, records: Mapping) -> None:
    group = PHASE_ONE_GROUP_OF[strategy]
    if sorted(records) != list(range(partition.N_FOLDS)):
        raise SearchError(f"phase 2 needs the phase-1 checkpoint of every rotation; got "
                          f"{sorted(records)}")
    for r, (_, meta) in records.items():
        expected = {"phase": 1, "name": group, "transfer": data.transfer.name,
                    "rotation": r, "seed": SEARCH_SEED,
                    partition.HASH_FIELD: data.partition_sha256}
        wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
        if wrong:
            raise SearchError(f"rotation {r}: the phase-1 checkpoint is not the {group} one "
                              f"of {data.transfer.name} at the search seed: {wrong}")


# ------------------------------------------------------------------ studies


def study_path(root: str | Path, transfer: assignment_mod.Transfer, phase: int,
               name: str) -> Path:
    if phase == 1 and name not in PHASE_ONE_SPACES:
        raise SearchError(f"phase 1 studies a group {list(PHASE_ONE_SPACES)}, not {name!r}")
    if phase == 2 and name not in model_store.STRATEGIES:
        raise SearchError(f"phase 2 studies a strategy, not {name!r}")
    return (Path(root) / "Results" / "hpo"
            / f"{model_store.transfer_slug(transfer)}__phase{phase}__"
              f"{model_store.name_slug(name)}.db")


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
IDENTITY_FIELDS = ("transfer", "arm", "phase", "name", "sampler", "samplerSeed", "nTrials",
                   "pruner", "searchSeed", "reportSeeds", "schedule", "objective", "space",
                   partition.HASH_FIELD, "revision", "code_version", "rotationFolds")


def _json(value):
    import json
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def study_identity(data: SearchData, phase: int, name: str, schedule: Mapping,
                   n_trials: int, objective: str) -> dict:
    """What a study is: its transfer, phase, budget, space, sampler and the partition,
    plan revision and code version it was run against."""
    space = PHASE_ONE_SPACES[name] if phase == 1 else phase_two_space(name)
    return _json({
        "transfer": data.transfer.name, "arm": data.transfer.arm, "phase": phase,
        "name": name, "sampler": "GPSampler", "samplerSeed": SAMPLER_SEED,
        "nTrials": n_trials, "pruner": "none", "searchSeed": SEARCH_SEED,
        "reportSeeds": list(REPORT_SEEDS), "schedule": dict(schedule),
        "objective": objective, "space": {k: repr(v) for k, v in space.items()},
        partition.HASH_FIELD: data.partition_sha256,
        "revision": __provenance__["revision"], "code_version": model_store.code_version(),
        "rotationFolds": {str(r): v for r, v in data.folds.items()}})


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
    differ = identity_differences(study.user_attrs, identity, IDENTITY_FIELDS)
    if differ:
        raise SearchError(f"{path} is a study of another identity and is not continued: "
                          + "; ".join(differ) + "; remove it to search again")
    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if complete > identity["nTrials"]:
        raise SearchError(f"{path} holds {complete} completed trials, more than its declared "
                          f"{identity['nTrials']}")
    if complete < identity["nTrials"]:
        held = len(study.trials)
        study.sampler = optuna.samplers.GPSampler(seed=SAMPLER_SEED + held)
        study.set_user_attr("resumedAfter", [*study.user_attrs.get("resumedAfter", []), held])
    return study, complete


def _summary(values: Sequence[float]) -> dict:
    return {"rotationValues": [float(v) for v in values], "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=SD_DDOF))}


def run_phase_one_study(group: str, data: SearchData, schedule: Mapping, *,
                        root: str | Path = DEFAULT_ROOT, n_trials: int = N_TRIALS,
                        device: torch.device | str = "cpu") -> optuna.Study:
    """One phase-1 study: `n_trials` configurations, each over the four rotations."""
    schedule = check_schedule(schedule)
    if group not in PHASE_ONE_SPACES:
        raise SearchError(f"{group!r} is not a phase-1 group")
    model_store.require_registered(group)
    path = study_path(root, data.transfer, 1, group)
    study, complete = open_study(path, study_identity(
        data, 1, group, schedule, n_trials,
        "mean - sd over the four rotations of the source-validation balanced accuracy"))

    def objective(trial: optuna.Trial) -> float:
        hyper = phase_one_hyper(group, suggest(trial, PHASE_ONE_SPACES[group]), schedule)
        values, epochs, best = [], [], []
        for r in sorted(data.rotations):
            out = train_phase_one(group, data, r, SEARCH_SEED, hyper, device)["result"]
            values.append(float(out["best_valid_balanced_accuracy"]))
            epochs.append(int(out["epochs"]))
            best.append(int(out["best_epoch"]))
        for key, value in {**_summary(values), "seed": SEARCH_SEED, "epochsRun": epochs,
                           "bestEpochs": best}.items():
            trial.set_user_attr(key, value)
        return robust(values)

    study.optimize(objective, n_trials=n_trials - complete)
    return study


def run_phase_two_study(strategy: str, data: SearchData, phase_one: Mapping,
                        schedule: Mapping, *, root: str | Path = DEFAULT_ROOT,
                        n_trials: int = N_TRIALS,
                        device: torch.device | str = "cpu") -> optuna.Study:
    """One phase-2 study; `phase_one` maps each rotation to its stored (state, meta)."""
    space = phase_two_space(strategy)
    schedule = check_schedule(schedule)
    _check_phase_one_records(strategy, data, phase_one)
    path = study_path(root, data.transfer, 2, strategy)
    study, complete = open_study(path, study_identity(
        data, 2, strategy, schedule, n_trials,
        "mean - sd over the four rotations of the target-validation balanced accuracy "
        "(primary rule, Sec. 4.5), what the sampler observes; the selection is per test "
        "rotation over the three rotations whose validation fold is not its test fold "
        "(select_per_rotation); label_free_score: the same summary of the label-free "
        "score (secondary rule)"))

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, space)
        primary, free, free_at_primary, sigmas, best = [], [], [], [], []
        for r in sorted(data.rotations):
            out = train_phase_two(strategy, data, r, SEARCH_SEED, phase_one[r], params,
                                  schedule, device)["result"]
            checkpoint = out["checkpoint"]
            primary.append(float(checkpoint["valid_balanced_accuracy"]))
            free.append(float(max(h["label_free_score"] for h in out["history"])))
            free_at_primary.append(float(checkpoint["label_free_score"]))
            sigmas.append(checkpoint.get("sigma"))
            best.append(int(out["best_epoch"]))
        for key, value in {**_summary(primary), "label_free_score": robust(free),
                           "labelFreeRotationValues": free,
                           "labelFreeAtPrimaryEpoch": free_at_primary,
                           "sigmaByRotation": sigmas, "bestEpochs": best,
                           "seed": SEARCH_SEED}.items():
            trial.set_user_attr(key, value)
        if not all(math.isfinite(v) for v in primary):
            raise SearchError(f"a rotation returned a non-finite validation score: {primary}")
        return robust(primary)

    study.optimize(objective, n_trials=n_trials - complete)
    return study
