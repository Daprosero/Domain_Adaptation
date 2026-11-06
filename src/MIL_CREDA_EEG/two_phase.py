"""MIL-CREDA's two-phase training on EEG, for one transfer, rotation and seed.

Experiments plan v57, Secs. 4.5, 7.3 and 7.7, over proposal r44 Eqs. 21 and
32-39. The phase mechanics follow the retired toy-set runner (git history,
`src/MIL_CREDA_Benchmark/toy_run.py` before 9b733d7) as a reference; nothing of
it is imported or restored, and the plan's rules govern where the two differ.

**Fixed epochs, no early stopping (Secs. 4.5, 7.7).** Every training here runs
the epoch count it is given, whole, and keeps the weights of its last epoch; no
phase uses patience and no epoch is restored. With validation bags (a search
training on one inner split) the validation predictions of every epoch are
recorded (`validation_record`: subject, label and predicted probabilities per
bag, and in phase 2 each bag's representation Z, the vector entering the
predictive head, `represent`), so the search pools them over the three inner
splits and picks the epoch: phase 1 by `pooled_balanced_accuracy` on the source
labels, phase 2 by Sec. 4.5's centroid rule (`centroid_counts`,
`pooled_centroid_balanced_accuracy`) and its EnsV secondary rule, neither of
which reads a target label. Without validation bags
(a refit on every non-test fold) nothing is evaluated: the refit runs exactly
the selected E1 or E2 epochs and its history carries NaN validation values.

**Phase 1, supervised in the source (Sec. 7.3).** Source trunk, attention and
head are trained on the source training bags with Eq. 21 normalized by B_src,
the supervised term of r44's objective (`mil_creda_model.source_term`), for
exactly `epochs1` epochs.

**sigma in phase 1: this implementation's rule.** Sec. 7.7 fixes that phase 2
uses exactly the numeric sigma of the phase-1 refit checkpoint and leaves
the phase-1 estimation rule open. The rule chosen here is: sigma =
`sigmaMultiplier` x the median pairwise Euclidean distance between instance
embeddings of the SAME bag (pairs pooled over every source training bag),
measured once at the start of phase 1 with the freshly initialized source
trunk, and held fixed for the whole of phase 1. The embeddings are read in the
geometry of the first training step: BatchNorm on the bag's own batch
statistics, dropout off, no gradient, and the running statistics restored
afterwards so the measurement leaves no trace in the model. Within-bag pairs,
not all pairs, because that is the set the instance kernel evaluates (the toy
set's `_within_bag_median` scope). Because sigma never changes in phase 1, the
sigma that produced the selected checkpoint is that one number, and it is
stored with the checkpoint.

**The bridge (Sec. 7.3).** `eegnet.copy_compatible_layers` initializes the
target trunk from the source trunk (all layers when channel counts match; all
but the depthwise spatial convolution otherwise). The source trunk, the
attention and the head are frozen: `requires_grad` off and the source trunk in
`eval`, so BatchNorm does not update its running statistics either.

**Phase 2, adaptation (Secs. 7.3, 7.7).** The optimizer holds the target trunk
only. The loss is `eta_glob * L_glob + eta_loc * L_loc` with both
coefficients 1 and no ramp (r44 Eqs. 28-33, 38; Eq. 39 with its supervised summand
dropped, `objective.total_objective` with a zero first argument). Eq. 21 over
the source batch is computed and recorded on every step and summed into
nothing: with theta_s, phi and psi frozen it carries no gradient to theta_t.
sigma is EXACTLY the stored phase-1 value (`sigma2 = sigma1`): no re-estimation,
no recalibration, no per-batch update, and `MILCREDAModel.fix_sigma` refuses
any other value. Phase 2 runs exactly `epochs2` epochs and keeps the last one;
with target validation bags it records, per epoch, their predictions and
representations for the pooled selection of Sec. 4.5, which the search makes
over the three inner splits, never here (their balanced accuracy is recorded
in the history as a reading; no selection reads it). After phase 2 the
source trunk, attention and head are asserted bit-identical to the end of the
bridge (`FrozenPartsMoved` otherwise).

**The K3 ablation (Sec. 6.3).** `term_selection` names which adaptation terms
enter the phase-2 objective: "both terms" (the default, every caller above),
"global term only" or "local term only" (`TERM_SELECTIONS`, the names of
`mechanisms.ABLATIONS`). A dropped term gets coefficient 0 and is detached
before it enters Eq. 39, so it contributes exactly zero to the gradient; its
value is still computed and recorded on every step, for the bound control.
With the default the objective is the same expression on the same tensors as
before the selection existed, so every existing caller is unchanged.

**Batches.** `bagsPerStep` bags per step, an implementation choice the plan
does not fix. Phase 1 takes one class-interleaved pass over the source training
bags per epoch. Phase 2 takes one shuffled pass over the target training bags
per epoch, each step paired with a source batch drawn with equal numbers per
class from per-class queues, so every class is in the source batch, as r44's
local correspondence requires (S_c non-empty). The source side of phase 2 is
encoded without gradient: it is frozen, so the target trunk's gradient is the
same and memory is not spent on a graph nobody differentiates.

**Records.** Per-epoch history in the dashboard's `Results/history*` shape
(`epoch`, numeric series, `seed`, `fold`, `model`); `fold` is the rotation and
`model` names the phase (`MIL-CREDA:phase1`, `MIL-CREDA:phase2`).
`run_transfer_rotation` returns the last-epoch checkpoints, sigma, the history and
the target test-fold predictions (subject, direction, probability of the
positive class, label), ready to pool over the two directions of a pair; it is a
single-rotation rehearsal on one of the rotation's stored inner splits (`split`,
0 by default: train on two non-test folds, validate on the third) and its test
fold, not the nested selection, which `search` and `campaign` run.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from MIL_CREDA.objective import total_objective
from MIL_CREDA_EEG import assignment as assignment_mod
from MIL_CREDA_EEG import bag_data, eegnet
from MIL_CREDA_EEG.mil_creda_model import (CLASSES, TERM_BOUNDS, MILCREDAModel,
                                           adaptation_terms, source_term)

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v57.md",
    "sections": ["4.5", "7.3", "7.7"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K3-ablacion"],
    "equations": ["21", "28", "29", "30", "31", "32", "33", "38", "39"],
    "invariants": [
        "the_hyperparameters_are_exactly_the_declared_ones",
        "phase_one_runs_its_epochs_and_keeps_the_last_without_restoring",
        "the_refit_runs_exactly_e_epochs_with_no_validation",
        "the_search_records_validation_every_epoch_and_never_stops_early",
        "the_pooled_count_balanced_accuracy_is_not_the_mean_of_the_split_ones",
        "the_phase_two_selection_reads_no_target_label",
        "the_representation_is_the_vector_entering_the_predictive_head",
        "the_centroid_rule_assigns_each_source_bag_to_the_nearest_pseudo_class_centroid",
        "a_collapse_to_one_pseudo_class_scores_one_half_with_two_classes",
        "sigma_is_fixed_during_phase_one_and_stored_with_the_checkpoint",
        "the_phase_one_sigma_is_the_multiplier_times_the_within_bag_median",
        "the_bridge_copies_compatible_layers_and_freezes_the_source_side",
        "phase_two_updates_only_the_target_trunk",
        "sigma_in_phase_two_is_exactly_the_phase_one_value",
        "the_adaptation_coefficients_are_one_with_no_ramp",
        "the_term_selection_is_validated",
        "the_default_term_selection_leaves_every_caller_unchanged",
        "a_dropped_term_contributes_exactly_zero_to_the_gradient",
        "the_dropped_terms_value_is_still_recorded_for_the_bound_control",
        "the_source_term_does_not_change_the_phase_two_gradient",
        "phase_two_keeps_its_last_epoch_and_records_validation_per_epoch",
        "history_records_carry_the_dashboard_keys",
        "test_predictions_are_only_for_target_test_fold_subjects",
    ],
}

MODEL = "MIL-CREDA"
#: Sec. 7.7: the adaptation coefficients of phase 2, fixed, with no ramp.
LAMBDA_GLOBAL = 1.0
LAMBDA_LOCAL = 1.0
#: Sec. 6.3: the ablation's term selections -> (eta_glob, eta_loc). A dropped term
#: has coefficient 0; the names are `mechanisms.ABLATIONS`, in that order.
TERM_SELECTIONS = {"global term only": (LAMBDA_GLOBAL, 0.0),
                   "local term only": (0.0, LAMBDA_LOCAL),
                   "both terms": (LAMBDA_GLOBAL, LAMBDA_LOCAL)}
BOTH_TERMS = "both terms"
#: Sec. 7.7: what each phase searches, and the schedule the pilot fixes (7.6). The
#: epochs are the search's caps, or the selected E1 and E2 of a refit; no patience.
PHASE_ONE_KEYS = ("lr1", "wd1", "attentionGamma", "attentionTemperature", "sigmaMultiplier")
PHASE_TWO_KEYS = ("lr2", "wd2", "tauLocal")
SCHEDULE_KEYS = ("epochs1", "epochs2", "bagsPerStep")
HYPER_KEYS = PHASE_ONE_KEYS + PHASE_TWO_KEYS + SCHEDULE_KEYS
SIGMA_RULE = ("sigmaMultiplier x median pairwise distance between instance embeddings "
              "of the same bag, pooled over the source training bags, measured once at "
              "the start of phase 1 (batch-statistics BatchNorm, dropout off) and held "
              "fixed through phase 1")
#: Windows per trunk pass when evaluating without gradient.
EVAL_CHUNK = 512
#: Sec. 4.5: how every phase-2 trainer's run is selected; the search chooses, never a run.
SELECTION = {"primary": "the centroid rule: source validation bags of each inner split "
                        "assigned to the nearest centroid of the target validation bags' "
                        "pseudo-classes, confusion counts pooled over the three inner "
                        "splits, chosen by the search (Sec. 4.5); no target label",
             "secondary": "EnsV: accuracy against the pseudo-labels of the ensemble of every "
                          "(trial, epoch) of the study, chosen by the search (Sec. 4.5); no "
                          "target label"}


class FrozenPartsMoved(RuntimeError):
    """Phase 2 changed the source trunk, the attention or the head."""


class SigmaChanged(RuntimeError):
    """Phase 2 would run at a sigma other than the stored phase-1 value."""


class UnknownTermSelection(ValueError):
    """A term selection that Sec. 6.3 does not declare."""


def term_coefficients(term_selection: str) -> tuple[float, float]:
    """(eta_glob, eta_loc) of a declared term selection; anything else refuses."""
    if not isinstance(term_selection, str) or term_selection not in TERM_SELECTIONS:
        raise UnknownTermSelection(f"the term selection is one of {list(TERM_SELECTIONS)}; "
                                   f"got {term_selection!r}")
    return TERM_SELECTIONS[term_selection]


@dataclass(frozen=True)
class BagTensor:
    """One bag as training reads it: windows in memory and an integer label."""

    subject: str
    condition: str
    label: int
    windows: torch.Tensor          # (m, channels, 512) float32


def check_hyper(hyper: Mapping) -> dict:
    """Exactly the declared keys: nothing missing, nothing else (no lambda, no ramp)."""
    keys = set(hyper)
    missing, extra = sorted(set(HYPER_KEYS) - keys), sorted(keys - set(HYPER_KEYS))
    if missing or extra:
        raise ValueError(f"hyperparameters must be exactly {list(HYPER_KEYS)}; "
                         f"missing {missing}, not declared {extra}")
    out = dict(hyper)
    for key in SCHEDULE_KEYS:
        if int(out[key]) != out[key] or out[key] < 1:
            raise ValueError(f"{key} must be a positive integer, got {out[key]!r}")
    if out["bagsPerStep"] < len(CLASSES):
        raise ValueError("bagsPerStep must hold at least one source bag per class")
    return out


def bag_tensors(bags: Sequence[bag_data.Bag]) -> list[BagTensor]:
    return [BagTensor(b.subject, b.condition, CLASSES.index(b.label),
                      torch.from_numpy(np.ascontiguousarray(b.windows())).float())
            for b in bags]


# ------------------------------------------------------------------ metrics


def labels_like(labels: Sequence[int], like: torch.Tensor) -> torch.Tensor:
    """Labels as a tensor on the device of the predictions they are scored against."""
    return torch.tensor(list(labels), device=like.device)


def balanced_accuracy(labels: Sequence[int], predicted: Sequence[int]) -> float:
    """Mean recall over the classes present in `labels`."""
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    recalls = [float(np.mean(predicted[labels == c] == c)) for c in np.unique(labels)]
    return float(np.mean(recalls)) if recalls else float("nan")


def validation_record(epoch: int, bags: Sequence[BagTensor], G: torch.Tensor,
                      Z: torch.Tensor | None = None) -> dict:
    """One epoch's validation predictions of one inner split, as the search pools them:
    per bag (one bag per subject) its subject, its label and its predicted probabilities,
    and, given `Z` (`represent`, phase 2), its representation. The search reads a target
    record's probabilities and representations, never its labels (Sec. 4.5)."""
    record = {"epoch": int(epoch), "subjects": [b.subject for b in bags],
              "labels": [int(b.label) for b in bags],
              "probabilities": G.detach().to(torch.float64).cpu().tolist()}
    if Z is not None:
        record["representations"] = Z.detach().to(torch.float64).cpu().tolist()
    return record


def _pooled(records: Sequence[Mapping]) -> list[Mapping]:
    """The records of one epoch over the inner splits, each validated subject once."""
    subjects = [s for r in records for s in r["subjects"]]
    if not subjects:
        raise ValueError("no validated subject to pool")
    if len(subjects) != len(set(subjects)):
        raise ValueError("the pooled validation must hold each subject exactly once; "
                         f"repeated {sorted({s for s in subjects if subjects.count(s) > 1})}")
    return list(records)


def pooled_balanced_accuracy(records: Sequence[Mapping]) -> float:
    """Sec. 4.5's selection score for one epoch: the balanced accuracy of the subject-level
    confusion counts pooled over the inner splits (`records`, one per split), computed once
    on the pooled counts, never averaged per split (Forman and Scholz, 2010)."""
    records = _pooled(records)
    labels = [y for r in records for y in r["labels"]]
    predicted = [int(np.argmax(p)) for r in records for p in r["probabilities"]]
    return balanced_accuracy(labels, predicted)


def centroid_counts(target_Z, target_probabilities, source_Z, source_labels,
                    classes: int | None = None) -> np.ndarray:
    """Sec. 4.5's primary rule on one inner split and epoch: the confusion counts
    (rows: real source label, columns: assigned class) of the source bags assigned to the
    nearest centroid of the target pseudo-classes.

    Each target bag's pseudo-label is the argmax of its predicted probabilities; one
    centroid per pseudo-class, the mean of its bags' representations (a class no bag is
    assigned to has no centroid); each source bag goes to the nearest centroid in Euclidean
    distance (the lowest class index on an exact tie). No model is trained, and no target
    label is an input."""
    Zt = np.asarray(target_Z, dtype=np.float64)
    P = np.asarray(target_probabilities, dtype=np.float64)
    Zs = np.asarray(source_Z, dtype=np.float64)
    y = np.asarray(source_labels, dtype=np.int64).reshape(-1)
    if Zt.ndim != 2 or P.ndim != 2 or Zt.shape[0] != P.shape[0] or Zt.shape[0] == 0:
        raise ValueError(f"one representation and one probability row per target bag; got "
                         f"{Zt.shape} and {P.shape}")
    if Zs.ndim != 2 or Zs.shape[0] != y.shape[0] or Zs.shape[0] == 0:
        raise ValueError(f"one representation and one label per source bag; got {Zs.shape} "
                         f"and {y.shape}")
    if Zs.shape[1] != Zt.shape[1]:
        raise ValueError(f"source and target representations live in different spaces: "
                         f"{Zs.shape[1]} vs {Zt.shape[1]}")
    classes = int(classes if classes is not None else P.shape[1])
    pseudo = P.argmax(axis=1)
    present = [c for c in range(classes) if np.any(pseudo == c)]
    centroids = np.stack([Zt[pseudo == c].mean(axis=0) for c in present])
    distances = ((Zs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    assigned = np.asarray(present)[distances.argmin(axis=1)]
    counts = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(counts, (y, assigned), 1)
    return counts


def balanced_accuracy_of_counts(counts) -> float:
    """Mean recall over the classes with at least one real member, from confusion counts
    (rows: real class, columns: assigned class), computed once on the counts."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1)
    recalls = [counts[c, c] / totals[c] for c in range(counts.shape[0]) if totals[c] > 0]
    return float(np.mean(recalls)) if recalls else float("nan")


def pooled_centroid_balanced_accuracy(target_records: Sequence[Mapping],
                                      sources: Sequence[Mapping]) -> float:
    """Sec. 4.5's primary rule for one epoch: the centroid counts of every inner split
    (`target_records[i]`, that split's target validation record, with `sources[i]`, its
    source validation bags: `representations` and `labels`) summed, and the balanced
    accuracy computed once on the pooled counts. It reads the target records'
    `probabilities` and `representations`, never their labels."""
    target_records = _pooled(target_records)
    _pooled(sources)
    if len(target_records) != len(sources):
        raise ValueError(f"one source pass per inner split; got {len(sources)} for "
                         f"{len(target_records)} splits")
    classes = max(len(r["probabilities"][0]) for r in target_records if r["probabilities"])
    total = None
    for record, source in zip(target_records, sources):
        if "representations" not in record:
            raise ValueError("the target validation record carries no representations")
        counts = centroid_counts(record["representations"], record["probabilities"],
                                 source["representations"], source["labels"], classes)
        total = counts if total is None else total + counts
    return balanced_accuracy_of_counts(total)


def _bag_representations(model, windows: Sequence[torch.Tensor], domain: str) -> torch.Tensor:
    """Z of `windows`' bags: the vector the predictive head reads. AdaptMIL names it
    `representations`; every MIL-CREDA-shaped model (MIL-CREDA, the common group, COT,
    CLUDA) `represent(bag_pairs(embed(...)))`."""
    if hasattr(model, "representations"):
        return model.representations(windows, domain, chunk=EVAL_CHUNK)
    return model.represent(model.bag_pairs(model.embed(windows, domain, EVAL_CHUNK)))


@torch.no_grad()
def represent(model, bags: Sequence[BagTensor], domain: str,
              device: torch.device) -> torch.Tensor:
    """Sec. 4.5: each bag's representation Z, the vector entering the predictive head, in
    eval mode and without gradient, as `evaluate` predicts (float64, on the CPU). Both
    domains write the same space (the shared attention and head), whatever their channel
    counts (the device arm)."""
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        Z = torch.cat([_bag_representations(model, [b.windows.to(device)], domain)
                       for b in bags]) if bags else torch.zeros(0, 0)
    finally:
        for m, mode in modes.items():
            m.train(mode)
    return Z.detach().to(torch.float64).cpu()


@torch.no_grad()
def evaluate(model: MILCREDAModel, bags: Sequence[BagTensor], domain: str,
             device: torch.device) -> torch.Tensor:
    modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        G = torch.cat([model.predict([b.windows.to(device)], domain, chunk=EVAL_CHUNK)
                       for b in bags]) if bags else torch.zeros(0, model.classes)
    finally:
        for m, mode in modes.items():          # per module: a frozen trunk stays in eval
            m.train(mode)
    return G


# ------------------------------------------------------------------ sigma


@torch.no_grad()
def within_bag_median_distance(model: MILCREDAModel, bags: Sequence[BagTensor],
                               domain: str, device: torch.device) -> float:
    """Median of within-bag pairwise embedding distances, pooled over `bags`."""
    trunk = model.trunk(domain)
    buffers = {k: v.clone() for k, v in trunk.state_dict().items()}
    modes = {m: m.training for m in trunk.modules()}
    trunk.train()
    for m in trunk.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    try:
        distances = []
        for b in bags:
            H = trunk(b.windows.to(device)).to(torch.float64)
            if H.shape[0] < 2:
                continue
            d = torch.cdist(H, H)
            distances.append(d[~torch.eye(d.shape[0], dtype=torch.bool, device=d.device)])
    finally:
        trunk.load_state_dict(buffers)
        for m, mode in modes.items():
            m.train(mode)
    if not distances:
        raise ValueError("no bag with two instances to measure sigma on")
    return float(torch.cat(distances).median())


def phase_one_sigma(model: MILCREDAModel, source_train: Sequence[BagTensor],
                    multiplier: float, device: torch.device) -> dict:
    median = within_bag_median_distance(model, source_train, "source", device)
    sigma = model.fix_sigma(float(multiplier) * median)
    return {"sigma": sigma, "withinBagMedian": median, "sigmaMultiplier": float(multiplier),
            "rule": SIGMA_RULE, "bags": len(source_train)}


# ------------------------------------------------------------------ batches


def interleaved_batches(bags: Sequence[BagTensor], size: int,
                        generator: torch.Generator) -> list[list[int]]:
    """One pass: each class shuffled, classes interleaved, cut into batches."""
    per_class = []
    for c in sorted({b.label for b in bags}):
        idx = [i for i, b in enumerate(bags) if b.label == c]
        order = torch.randperm(len(idx), generator=generator).tolist()
        per_class.append([idx[k] for k in order])
    sequence, k = [], 0
    while any(per_class):
        queue = per_class[k % len(per_class)]
        if queue:
            sequence.append(queue.pop(0))
        k += 1
    return [sequence[i:i + size] for i in range(0, len(sequence), size)]


def shuffled_batches(n: int, size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(n, generator=generator).tolist()
    return [order[i:i + size] for i in range(0, n, size)]


class ClassQueues:
    """Equal numbers per class per draw, each class cycling through a reshuffled queue."""

    def __init__(self, bags: Sequence[BagTensor], classes: int, generator: torch.Generator):
        self.members = {c: [i for i, b in enumerate(bags) if b.label == c]
                        for c in range(classes)}
        empty = [CLASSES[c] for c, m in self.members.items() if not m]
        if empty:
            raise ValueError(f"the source training bags hold no bag of class(es) {empty}")
        self.generator = generator
        self.queues = {c: [] for c in self.members}

    def draw(self, per_class: int) -> list[int]:
        out = []
        for c, members in self.members.items():
            for _ in range(per_class):
                if not self.queues[c]:
                    order = torch.randperm(len(members), generator=self.generator).tolist()
                    self.queues[c] = [members[k] for k in order]
                out.append(self.queues[c].pop(0))
        return out


# ------------------------------------------------------------------ state


def _clone_state(module: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def phase_one_state(model: MILCREDAModel) -> dict:
    """Everything phase 1 fits: all but the target trunk (buffers included)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if not k.startswith("target_trunk.")}


def _states_equal(a: Mapping, b: Mapping) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _labels(bags: Sequence[BagTensor]) -> list[int]:
    return [b.label for b in bags]


def _record(epoch: int, seed: int, fold: int, phase: int, transfer: str, values: dict) -> dict:
    return {"epoch": int(epoch), "seed": int(seed), "fold": int(fold),
            "model": f"{MODEL}:phase{phase}", "transfer": transfer,
            **{k: float(v) for k, v in values.items()}}


# ------------------------------------------------------------------ phase 1


def train_source_phase(model: MILCREDAModel, source_train: Sequence[BagTensor],
                       source_valid: Sequence[BagTensor], hyper: Mapping,
                       generator: torch.Generator, device: torch.device, *,
                       seed: int = 0, fold: int = 0, transfer: str = "") -> dict:
    """Phase 1: Eq. 21 on the source for exactly `epochs1` epochs, no early stopping.

    With `source_valid` (an inner split) every epoch's validation is recorded; without
    it (the refit on every non-test fold) nothing is evaluated. The checkpoint is the
    last epoch's."""
    hyper = check_hyper(hyper)
    sigma = phase_one_sigma(model, source_train, hyper["sigmaMultiplier"], device)
    parameters = [*model.source_trunk.parameters(), *model.attention_parameters(),
                  *model.head.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=hyper["lr1"], weight_decay=hyper["wd1"])
    valid_labels = _labels(source_valid)
    history, seconds, validation = [], [], []
    nan = float("nan")
    for epoch in range(hyper["epochs1"]):
        started = time.perf_counter()
        model.train()
        total, steps = 0.0, 0
        for batch in interleaved_batches(source_train, hyper["bagsPerStep"], generator):
            windows = [source_train[i].windows.to(device) for i in batch]
            labels = torch.tensor([source_train[i].label for i in batch], device=device)
            G = model.predict(windows, "source")
            loss = source_term(model, G, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.constrain("source_trunk", "head", "attention")
            if model.sigma != sigma["sigma"]:
                raise SigmaChanged("sigma moved during phase 1")
            total += float(loss.detach())
            steps += 1
        accuracy, valid_loss = nan, nan
        if source_valid:
            G_valid = evaluate(model, source_valid, "source", device)
            accuracy = balanced_accuracy(valid_labels, G_valid.argmax(dim=1).tolist())
            valid_loss = float(source_term(model, G_valid, labels_like(valid_labels,
                                                                        G_valid)))
            validation.append(validation_record(epoch, source_valid, G_valid))
        seconds.append(time.perf_counter() - started)
        history.append(_record(epoch, seed, fold, 1, transfer, {
            "loss": total / max(steps, 1), "val_loss": valid_loss,
            "val_balanced_accuracy": accuracy, "sigma": sigma["sigma"],
            "source_bound_low": TERM_BOUNDS["source"][0],
            "source_bound_high": TERM_BOUNDS["source"][1]}))
    model.eval()
    last = len(history) - 1
    final = history[last]["val_balanced_accuracy"]
    return {"checkpoint": {"state": phase_one_state(model), "sigma": model.sigma,
                           "epoch": last, "valid_balanced_accuracy": final,
                           "hyper": dict(hyper)},
            "sigma": sigma, "best_epoch": last, "epochs": len(history),
            "best_valid_balanced_accuracy": final, "validation": validation,
            "seconds_per_epoch": float(np.mean(seconds)), "seconds": seconds,
            "history": history}


# ------------------------------------------------------------------ bridge


def bridge(model: MILCREDAModel) -> dict:
    """Copy compatible layers source -> target; freeze theta_s, phi, psi."""
    record = eegnet.copy_compatible_layers(model.source_trunk, model.target_trunk)
    for p in [*model.source_trunk.parameters(), *model.attention_parameters(),
              *model.head.parameters()]:
        p.requires_grad_(False)
    set_phase_two_modes(model)
    return {"layer_copy": record, "frozen_state": phase_one_state(model)}


def set_phase_two_modes(model: MILCREDAModel) -> None:
    """Everything in eval (BatchNorm statistics frozen) but the target trunk."""
    model.eval()
    model.target_trunk.train()


# ------------------------------------------------------------------ phase 2


def phase_two_objective(terms: Mapping, term_selection: str = BOTH_TERMS) -> torch.Tensor:
    """Eq. 39 without its supervised summand: eta_glob L_glob + eta_loc L_loc.

    A term the selection drops enters detached with coefficient 0: no gradient.
    """
    lambda_global, lambda_local = term_coefficients(term_selection)
    global_term = terms["global"] if lambda_global else terms["global"].detach()
    local_term = terms["local"] if lambda_local else terms["local"].detach()
    zero = torch.zeros((), dtype=terms["local"].dtype, device=terms["local"].device)
    return total_objective(zero, global_term, local_term, lambda_global, lambda_local)


def phase_two_step(model: MILCREDAModel, source: Sequence[BagTensor],
                   target: Sequence[BagTensor], tau_local: float,
                   device: torch.device, term_selection: str = BOTH_TERMS) -> dict:
    """One step's terms: the objective, its two summands and Eq. 21 (read only)."""
    labels = torch.tensor([b.label for b in source], device=device)
    with torch.no_grad():
        pairs_s = model.bag_pairs(model.embed([b.windows.to(device) for b in source], "source"))
        source_ce = source_term(model, model.classify(model.represent(pairs_s)), labels)
    pairs_t = model.bag_pairs(model.embed([b.windows.to(device) for b in target], "target"))
    terms = adaptation_terms(model, pairs_s, labels, pairs_t, tau_local)
    return {**terms, "objective": phase_two_objective(terms, term_selection),
            "source_ce": source_ce}


def adapt_target_phase(model: MILCREDAModel, phase_one: Mapping, frozen_state: Mapping,
                       source_train: Sequence[BagTensor], target_train: Sequence[BagTensor],
                       target_valid: Sequence[BagTensor], hyper: Mapping,
                       generator: torch.Generator, device: torch.device, *,
                       seed: int = 0, fold: int = 0, transfer: str = "",
                       term_selection: str = BOTH_TERMS) -> dict:
    """Phase 2: the target trunk alone, at the phase-1 sigma, for exactly `epochs2` epochs.

    The checkpoint is the last epoch's (no early stopping). With `target_valid` (an inner
    split) every epoch's validation is recorded, with the bags' representations; without
    it (the refit) nothing is evaluated."""
    lambda_global, lambda_local = term_coefficients(term_selection)
    hyper = check_hyper(hyper)
    sigma1 = phase_one["checkpoint"]["sigma"]
    model.fix_sigma(sigma1)                       # refuses any other value
    if model.sigma != sigma1:
        raise SigmaChanged(f"phase 2 sigma {model.sigma!r} != phase 1 sigma {sigma1!r}")
    optimizer = torch.optim.Adam(model.target_trunk.parameters(), lr=hyper["lr2"],
                                 weight_decay=hyper["wd2"])
    queues = ClassQueues(source_train, model.classes, generator)
    per_class = hyper["bagsPerStep"] // model.classes
    valid_labels = _labels(target_valid)
    history, seconds, validation = [], [], []
    for epoch in range(hyper["epochs2"]):
        started = time.perf_counter()
        set_phase_two_modes(model)
        sums = {"loss": 0.0, "global": 0.0, "local": 0.0, "source_ce": 0.0,
                "active_classes": 0.0, "confidence_mean": 0.0}
        global_range = [math.inf, -math.inf]
        local_range = [math.inf, -math.inf]
        steps = 0
        for batch in shuffled_batches(len(target_train), hyper["bagsPerStep"], generator):
            source = [source_train[i] for i in queues.draw(per_class)]
            step = phase_two_step(model, source, [target_train[i] for i in batch],
                                  hyper["tauLocal"], device, term_selection=term_selection)
            optimizer.zero_grad()
            step["objective"].backward()
            optimizer.step()
            model.constrain("target_trunk")
            if model.sigma != sigma1:
                raise SigmaChanged("sigma moved during phase 2")
            g, l = float(step["global"].detach()), float(step["local"].detach())
            global_range = [min(global_range[0], g), max(global_range[1], g)]
            local_range = [min(local_range[0], l), max(local_range[1], l)]
            for key, value in (("loss", float(step["objective"].detach())), ("global", g),
                               ("local", l), ("source_ce", float(step["source_ce"])),
                               ("active_classes", step["active_classes"]),
                               ("confidence_mean", step["confidence_mean"])):
                sums[key] += value
            steps += 1
        accuracy = float("nan")
        if target_valid:
            G_valid = evaluate(model, target_valid, "target", device)
            accuracy = balanced_accuracy(valid_labels, G_valid.argmax(dim=1).tolist())
            validation.append(validation_record(
                epoch, target_valid, G_valid, represent(model, target_valid, "target", device)))
        seconds.append(time.perf_counter() - started)
        means = {k: v / max(steps, 1) for k, v in sums.items()}
        history.append(_record(epoch, seed, fold, 2, transfer, {
            **means, "val_balanced_accuracy": accuracy,
            "lambda_global": lambda_global, "lambda_local": lambda_local,
            "global_min": global_range[0], "global_max": global_range[1],
            "local_min": local_range[0], "local_max": local_range[1],
            "global_bound_low": TERM_BOUNDS["global"][0],
            "global_bound_high": TERM_BOUNDS["global"][1],
            "local_bound_low": TERM_BOUNDS["local"][0],
            "local_bound_high": TERM_BOUNDS["local"][1],
            "terms_within_bounds": float(
                TERM_BOUNDS["global"][0] <= global_range[0] <= global_range[1]
                <= TERM_BOUNDS["global"][1]
                and TERM_BOUNDS["local"][0] <= local_range[0] <= local_range[1]
                <= TERM_BOUNDS["local"][1]),
            "sigma": sigma1}))
    set_phase_two_modes(model)
    if not _states_equal(phase_one_state(model), frozen_state):
        raise FrozenPartsMoved("phase 2 changed the source trunk, the attention or the head")
    last = len(history) - 1
    return {"checkpoint": {"target_state": _clone_state(model.target_trunk),
                           "sigma": sigma1, "epoch": last,
                           "valid_balanced_accuracy": history[last]["val_balanced_accuracy"],
                           "hyper": dict(hyper)},
            "best_epoch": last, "epochs": len(history), "validation": validation,
            "selection": SELECTION,
            "seconds_per_epoch": float(np.mean(seconds)), "seconds": seconds,
            "history": history}


# ------------------------------------------------------------------ the run


def predictions(model: MILCREDAModel, bags: Sequence[BagTensor], direction: str,
                device: torch.device) -> list[dict]:
    """Target test-fold predictions, ready to pool over a pair's two directions."""
    G = evaluate(model, bags, "target", device)
    positive = CLASSES.index("yes")
    return [{"subject": b.subject, "direction": direction,
             "probability": float(G[k, positive]), "label": int(b.label)}
            for k, b in enumerate(bags)]


def run_transfer_rotation(transfer: assignment_mod.Transfer, rotation: int, seed: int,
                          hyper: Mapping, device: torch.device | str = "cpu", *,
                          bags: bag_data.BagSet | None = None, split: int = 0) -> dict:
    """Phase 1, bridge and phase 2 of MIL-CREDA for one transfer, rotation, inner split
    and seed."""
    hyper = check_hyper(hyper)
    device = torch.device(device)
    bags = bags if bags is not None else bag_data.load_bags()
    a = assignment_mod.assign_from_bags(transfer, bags)
    view = a.rotation(rotation)

    def side(name: str, role: str, inner: int | None = None) -> list[BagTensor]:
        return bag_tensors(assignment_mod.side_bags(a, bags, name, rotation, role, split=inner))

    source_train = side("source", "inner_train", split)
    source_valid = side("source", "inner_validation", split)
    target_train = side("target", "inner_train", split)
    target_valid = side("target", "inner_validation", split)
    target_test = side("target", "test")
    if set(b.subject for b in target_test) != view["target_test"]:
        raise RuntimeError("the target test bags are not the rotation's target test fold")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = MILCREDAModel(source_train[0].windows.shape[1], target_train[0].windows.shape[1],
                          attention_gamma=hyper["attentionGamma"],
                          attention_temperature=hyper["attentionTemperature"]).to(device)
    common = {"seed": seed, "fold": rotation, "transfer": transfer.name}
    phase1 = train_source_phase(model, source_train, source_valid, hyper, generator, device,
                                **common)
    joint = bridge(model)
    phase2 = adapt_target_phase(model, phase1, joint["frozen_state"], source_train,
                                target_train, target_valid, hyper, generator, device,
                                **common)
    counts = {"source_train": len(source_train), "source_valid": len(source_valid),
              "target_train": len(target_train), "target_valid": len(target_valid),
              "target_test": len(target_test)}
    windows = {k: sum(int(b.windows.shape[0]) for b in v) for k, v in
               (("source_train", source_train), ("source_valid", source_valid),
                ("target_train", target_train), ("target_valid", target_valid),
                ("target_test", target_test))}
    return {"transfer": transfer.name, "arm": transfer.arm, "rotation": int(rotation),
            "seed": int(seed), "device": str(device), "partition_sha256": a.partition_sha256,
            "revision": __provenance__["revision"], "hyper": dict(hyper),
            "sigma": phase1["checkpoint"]["sigma"],
            "checkpoints": {"phase1": phase1["checkpoint"], "phase2": phase2["checkpoint"]},
            "layer_copy": joint["layer_copy"], "bags": counts, "windows": windows,
            "seconds_per_epoch": {"phase1": phase1["seconds_per_epoch"],
                                  "phase2": phase2["seconds_per_epoch"]},
            "phase1": {k: v for k, v in phase1.items() if k not in ("checkpoint", "history")},
            "phase2": {k: v for k, v in phase2.items() if k not in ("checkpoint", "history")},
            "history": phase1["history"] + phase2["history"],
            "predictions": predictions(model, target_test, transfer.name, device)}


def write_history(records: Sequence[Mapping], path: str | Path) -> Path:
    """Write per-epoch records where the dashboard reads them (`Results/history*.json`)."""
    path = Path(path)
    if not path.name.startswith("history") or path.suffix != ".json":
        raise ValueError("the dashboard reads Results/history*.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=1), encoding="utf-8")
    return path
