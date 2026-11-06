"""The report and variance design: subject-level metrics, pooled summaries and contrasts.

Experiments plan v59, Secs. 4.4 and 4.6, with the significance tables of
Secs. 5.1-5.3, the global ranking of Sec. 5.4 and the tests of Secs. 6.1-6.3. Every function here reads
predictions that already exist (one record per target test subject, rotation,
seed and strategy, as `two_phase.predictions` writes them); nothing here trains.

**Metrics (Sec. 4.4).** Subject level only, because the bag is the subject:
balanced accuracy (primary, `two_phase.balanced_accuracy`), Cohen's kappa,
the area under the ROC curve (AUC_avg, below), sensitivity and specificity.
A bag is predicted
positive when its probability of the positive class exceeds 0.5, which is the
argmax of the two-class prediction that training and selection read (a tie
goes to class 0, as `torch.argmax` does).

**Two summaries (Sec. 4.6).** The pooled one joins the test predictions of
the four rotations, each subject predicted once, and computes one metric over
the target pool (`pooled`; the headline number). The per-fold mean and
deviation (`per_fold`) is the evidence that the pooled number does not hang on
one rotation. There is one report seed (Sec. 4.6, seed 1):
each transfer has a single pooled value, with no reduction between seeds, and
every summary and contrast refuses records of more than one seed
(`single_seed`).

**AUC_avg (Sec. 4.6).** The AUC is never computed on the pooled
predictions: pooling the probabilities of the four rotations' models, which may
be calibrated differently, penalizes a classifier whose calibration changes
between folds even when it orders the subjects well inside each one. It is
computed in each test cell, one per rotation and direction, and averaged over
the cells (`auc_avg`; Forman and Scholz 2010, AUC_avg); each cell's AUC is
reported too (`auc_cells`). Balanced accuracy, kappa, sensitivity and
specificity stay on the pooled hard predictions (pooled counts). A cell without
both classes has no AUC and leaves the average (`auc_cells_undefined` counts
it); the declared partition never produces one.

**The pair (Sec. 4.6).** The primary performance contrast is on each core pair
with its two directions pooled: every assigned subject is target exactly once
across the two directions and is predicted by the model of the direction where
it is target. `pool_pair` asserts it against `assignment.pair_pool`.

**The primary contrast (Sec. 4.6).** A paired, class-stratified subject
bootstrap (Koehn 2004 for the pairing, Robin et al. 2011 for the
stratification): each replicate resamples subjects with replacement inside
every (class, direction, rotation) stratum, scores both models on the same
replicate, and recomputes the whole balanced accuracy and kappa. Reported: the
difference, its 95% percentile interval, a p value, the kappa difference with
its own p value and the number of subjects on which the two models disagree (the
only subjects that carry information with binary responses). Every stratum lies
inside one rotation-and-direction cell, so the same replicates resample within
each cell, and on them the AUC of every cell and its average are recomputed for
both models: the AUC_avg difference with its percentile interval and p value
(`TEST_AUC_AVG`). The four rotations are compared with Nadeau and Bengio's
corrected resampled t test, one value per rotation at the report seed
(`nadeau_bengio`).

**Two families (Sec. 4.6).** No row is reported uncorrected. The primary
family (`FAMILY_PRIMARY`, confirmatory, the only one the success criteria
read) is the balanced-accuracy bootstrap, Holm over the rivals of each core
pair. The secondary family (`FAMILY_SECONDARY`, support, no criterion) is the
kappa difference through the same paired, stratified bootstrap with its own p
value, computed from the same replicates exactly as the balanced-accuracy one
(`TEST_KAPPA`), the AUC_avg difference from the same replicates
(`TEST_AUC_AVG`) and the Nadeau-Bengio t; Holm runs within
each metric over the same rivals. `pair_contrast` emits the four rows of one
rival, each carrying its `family` and the `contrast` it belongs to, and
`apply_contrast_families` adjusts all of them at once (`holm`, `apply_holm`).

**Expected precision (Sec. 4.6).** Before any run, the precision the core
pairs' contrasts can give, from the pooled target counts per class
(`expected_precision`): normal approximation, alpha 0.05 two-sided, power 0.8,
no Holm adjustment. The 95% half-width of balanced accuracy at a BA level takes
sensitivity = specificity = BA, so Var(BA) = (BA(1-BA)/n_cases +
BA(1-BA)/n_controls) / 4; the minimum detectable paired BA difference at a
discordance d (the fraction of subjects on which the two models disagree, an
assumption) takes the per-class variance of the recall difference as d / n_c,
so it is (z_{1-alpha/2} + z_power) sqrt((d/n_cases + d/n_controls) / 4).

**The ranking (Secs. 5.1-5.4).** The arm rankings are descriptive
(`descriptive_ranking`): the average rank of the nine adapted strategies over
the arm's core transfers (1 = best, ties share the average rank), the position
and whether it is among the top three, with no test, because two core
transfers per arm cannot separate any pair. The plan's one ranking test is the
global one of Sec. 5.4 (Demsar 2006): the ten transfers of the three arms are
the blocks, Friedman's test at alpha 0.05 and the Nemenyi critical difference
as its post hoc (`friedman_nemenyi`); `nemenyi_groups` gives the maximal sets
of strategies whose rank gap does not exceed the critical difference, the
lines of the diagram. `GLOBAL_RANKING_LIMITATION` is the limitation the plan
declares: the blocks share subjects.

**The mechanism tests (Secs. 6.1-6.3).** Wilcoxon's signed-rank test over
subjects, with the matched-pairs rank-biserial correlation as effect size and,
beside it, the Hodges-Lehmann estimator of the per-subject difference with its
Wilcoxon-based 95% confidence interval (`wilcoxon_signed_rank`,
`hodges_lehmann`). The estimator is the median of the n(n+1)/2 Walsh averages
(d_i + d_j) / 2, i <= j. The interval is the exact signed-rank interval on the
sorted Walsh averages (Hollander and Wolfe, Nonparametric Statistical Methods,
Sec. 3.3; cited from memory): with T+ the signed-rank statistic under the
null, whose exact distribution over n subjects is counted by dynamic
programming over the 2^n sign patterns, k is the largest integer with
P(T+ <= k - 1) <= alpha / 2, and the interval runs from the k-th to the
(M + 1 - k)-th Walsh average, M = n(n+1)/2; its achieved coverage
1 - 2 P(T+ <= k - 1) is reported, at least 0.95. Every finite difference
enters, zeros included, and the exact distribution assumes no ties, which
continuous attention weights and distances meet. Below six subjects no k
reaches 0.95 and the interval is not reported (`None`, with a note).

**Implementation choices the plan does not fix.** The bootstrap draws
`BOOTSTRAP_RESAMPLES` = 10000 replicates from `BOOTSTRAP_SEED`; its p value is
two-sided, twice the smaller tail of the replicate differences at zero with
the (1 + count) / (1 + replicates) convention, so it is never exactly zero.
The kappa and AUC_avg p values apply the same formula to the kappa and
AUC_avg differences of the same replicates; a replicate whose kappa is undefined (one predicted and true
class only, impossible while both classes are present in every stratum)
leaves the count, and the replicates are the finite ones. A Nemenyi group is
joined while the rank gap is at most the critical difference, the plan's
wording («no la supera»).
Every other test is two-sided as well; the direction a hypothesis states is
read from the sign of the effect, never from a one-sided p. The Nadeau-Bengio
test-to-training ratio is an input (`test_train_ratio`): the campaign passes
the target-test subjects over the subjects the models were fit on, summed over
the two directions and averaged over rotations.

Significance rows carry exactly the dashboard's keys (`SIGNIFICANCE_KEYS`)
plus `hypothesis` and `experiment` and any extra key a table needs.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from MIL_CREDA_EEG.two_phase import balanced_accuracy

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v59.md",
    "sections": ["4.4", "4.6", "5.1", "5.2", "5.3", "5.4", "6.1", "6.2", "6.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion",
                    "K2-emparejamiento", "K3-ablacion"],
    "invariants": [
        "metrics_are_subject_level_and_balanced_accuracy_is_the_mean_recall",
        "a_subject_predicted_twice_in_one_seed_is_refused",
        "the_pooled_summary_is_one_value_from_the_one_report_seed",
        "the_per_fold_summary_is_mean_and_sd_over_the_rotations",
        "a_pair_pools_each_subject_from_the_direction_where_it_is_target",
        "the_contrasts_read_one_report_seed_and_no_reduction",
        "the_bootstrap_is_paired_and_keeps_every_stratum_size",
        "the_bootstrap_reports_difference_interval_kappa_and_discordant_subjects",
        "the_kappa_p_value_comes_from_the_same_replicates_as_the_balanced_accuracy_one",
        "every_contrast_row_belongs_to_a_holm_family_within_its_metric",
        "identical_models_give_a_zero_difference_and_p_one",
        "auc_avg_is_the_mean_of_the_rotation_and_direction_cell_aucs",
        "balanced_accuracy_and_kappa_stay_on_pooled_hard_predictions",
        "the_balanced_accuracy_bootstrap_is_stratified_by_class_and_direction_only",
        "the_auc_avg_contrast_resamples_by_class_within_each_cell",
        "no_delong_test_remains",
        "expected_precision_reproduces_the_plans_core_pair_numbers",
        "nadeau_bengio_reduces_to_the_one_sample_t_at_zero_ratio",
        "holm_is_the_step_down_adjustment",
        "friedman_ranks_with_one_as_best_and_nemenyi_gives_the_critical_difference",
        "the_arm_ranking_is_descriptive_with_no_test",
        "a_nemenyi_group_joins_strategies_whose_rank_gap_does_not_exceed_the_cd",
        "wilcoxon_reports_the_rank_biserial_effect",
        "hodges_lehmann_is_the_walsh_median_with_the_exact_signed_rank_interval",
        "significance_rows_carry_exactly_the_dashboard_keys",
    ],
}

ALPHA = 0.05
POSITIVE_THRESHOLD = 0.5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261102
METRICS = ("balanced_accuracy", "kappa", "auc", "sensitivity", "specificity")
#: The dashboard's significance schema (SKILL.md, the Results table).
SIGNIFICANCE_KEYS = ("comparison", "test", "statistic", "p_value", "p_adjusted",
                     "correction", "effect_size", "effect_size_name", "ci_low", "ci_high",
                     "n", "alpha")
OPTIONAL_KEYS = ("hypothesis", "experiment")
#: Demsar 2006, Table 5(a): two-tailed Nemenyi q at alpha 0.05, by number of methods.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164}
TEST_BOOTSTRAP = "paired class-stratified subject bootstrap (balanced accuracy)"
TEST_KAPPA = "paired class-stratified subject bootstrap (Cohen's kappa)"
TEST_AUC_AVG = ("paired class-stratified subject bootstrap within rotation-and-direction "
                "cells (AUC_avg)")
TEST_NADEAU_BENGIO = "Nadeau-Bengio corrected resampled t (per-rotation balanced accuracy)"
TEST_WILCOXON = "Wilcoxon signed-rank over subjects"
TEST_FRIEDMAN = "Friedman over core transfers, Nemenyi critical difference"
TEST_FRIEDMAN_GLOBAL = "Friedman over the ten transfers, Nemenyi critical difference"
#: Sec. 4.6: the two declared families of the performance contrasts.
FAMILY_PRIMARY = "primary (confirmatory)"
FAMILY_SECONDARY = "secondary (support, no criterion)"
CONTRAST_TESTS = (TEST_BOOTSTRAP, TEST_KAPPA, TEST_AUC_AVG, TEST_NADEAU_BENGIO)
METRIC_OF_TEST = {TEST_BOOTSTRAP: "balanced accuracy", TEST_KAPPA: "kappa",
                  TEST_AUC_AVG: "AUC_avg", TEST_NADEAU_BENGIO: "Nadeau-Bengio"}
#: Sec. 4.6: the bootstrap strata; each lies inside one rotation-and-direction cell.
STRATA = "class x direction"
AUC_STRATA = "class x rotation-and-direction cell"
#: Sec. 4.6, the expected-precision grid of the core pairs.
PRECISION_BA_LEVELS = (0.6, 0.7, 0.8)
PRECISION_DISCORDANCES = (0.1, 0.2, 0.3)
PRECISION_POWER = 0.8
#: Sec. 5.4: the arm success criteria read the top three by average rank.
TOP_RANKS = 3
#: Sec. 5.4, the limitation the plan declares for the global ranking, verbatim.
GLOBAL_RANKING_LIMITATION = (
    "La limitación se declara: varias transferencias comparten sujetos, porque las dos "
    "direcciones de cada par usan los mismos sujetos con los lados invertidos y las "
    "transferencias de un mismo dispositivo salen de la misma cohorte, así que los bloques "
    "no son del todo independientes, como la prueba supone.")


class StatisticsError(RuntimeError):
    """A summary or contrast was asked for something Secs. 4.4-4.6 do not allow."""


# ------------------------------------------------------------------ metrics (Sec. 4.4)


def predicted_labels(probabilities: Sequence[float]) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float) > POSITIVE_THRESHOLD).astype(int)


def _auc_rows(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Row-wise AUC (Mann-Whitney with midranks for ties); NaN without both classes."""
    ranks = sps.rankdata(s, axis=-1)
    m = (y == 1).sum(axis=-1)
    n = (y == 0).sum(axis=-1)
    u = np.where(y == 1, ranks, 0.0).sum(axis=-1) - m * (m + 1) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((m > 0) & (n > 0), u / (m * n), np.nan)


def _cell_index(cells: Sequence) -> dict:
    groups: dict = defaultdict(list)
    for i, c in enumerate(cells):
        groups[c].append(i)
    return {c: np.asarray(v) for c, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def auc_avg(labels: Sequence[int], probabilities: Sequence[float],
            cells: Sequence | None = None) -> dict:
    """Sec. 4.6: the AUC of each test cell and their mean (Forman and Scholz 2010).

    `cells` names each subject's cell (one per rotation and direction); None is
    one cell, the plain AUC. A cell without both classes has no AUC and leaves
    the mean.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    cells = [None] * len(y) if cells is None else list(cells)
    if len(cells) != len(y):
        raise StatisticsError("one cell per subject")
    per_cell = {c: float(_auc_rows(y[i][None, :], p[i][None, :])[0])
                for c, i in _cell_index(cells).items()}
    finite = [v for v in per_cell.values() if math.isfinite(v)]
    return {"auc": float(np.mean(finite)) if finite else float("nan"), "cells": per_cell,
            "undefined": len(per_cell) - len(finite)}


def subject_metrics(labels: Sequence[int], probabilities: Sequence[float],
                    cells: Sequence | None = None) -> dict:
    """Balanced accuracy, kappa, sensitivity and specificity on the pooled hard
    predictions, and AUC_avg over `cells` (None: one cell, the plain AUC)."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise StatisticsError("labels and probabilities must be one value per subject")
    yhat = predicted_labels(p)
    auc = auc_avg(y, p, cells)
    sensitivity = float(np.mean(yhat[y == 1] == 1)) if (y == 1).any() else float("nan")
    specificity = float(np.mean(yhat[y == 0] == 0)) if (y == 0).any() else float("nan")
    kappa = float(_ba_kappa(y[None, :], yhat[None, :])[1][0]) if len(y) else float("nan")
    return {"balanced_accuracy": balanced_accuracy(y, yhat),
            "kappa": kappa if math.isfinite(kappa) else float("nan"),
            "auc": auc["auc"], "sensitivity": sensitivity, "specificity": specificity,
            "n": int(len(y)), "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            **({} if cells is None else {"auc_cells": auc["cells"],
                                         "auc_cells_undefined": auc["undefined"]})}


def _key(record: Mapping) -> tuple:
    return (record.get("strategy"), record.get("seed"), record["subject"])


def check_once(records: Iterable[Mapping]) -> list[Mapping]:
    """Each subject is predicted at most once per strategy and seed (Sec. 4.6)."""
    records = list(records)
    seen, twice = set(), []
    for r in records:
        k = _key(r)
        if k in seen:
            twice.append(k)
        seen.add(k)
    if twice:
        raise StatisticsError(f"subjects predicted more than once in one seed: {twice[:5]}")
    return records


def _by(records: Iterable[Mapping], field: str) -> dict:
    out: dict = defaultdict(list)
    for r in records:
        out[r[field]].append(r)
    return dict(sorted(out.items()))


def cell_of(record: Mapping) -> str:
    """The test cell of a prediction (Sec. 4.6): its rotation and direction."""
    return f"rotation {record.get('rotation')}, {record.get('direction')}"


def _metrics_of(records: Sequence[Mapping]) -> dict:
    return subject_metrics([r["label"] for r in records], [r["probability"] for r in records],
                           [cell_of(r) for r in records])


def single_seed(records: Iterable[Mapping]) -> list[Mapping]:
    """The records of the one report seed (Sec. 4.6); more than one seed is refused."""
    records = check_once(records)
    seeds = sorted({r.get("seed") for r in records}, key=str)
    if len(seeds) > 1:
        raise StatisticsError(f"Sec. 4.6 has one report seed and no reduction between seeds; "
                              f"got seeds {seeds}")
    return records


def pooled(records: Iterable[Mapping]) -> dict:
    """One metric over every target subject of the four rotations, at the report seed."""
    return _metrics_of(single_seed(records))


def _mean_sd(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}


def per_fold(records: Iterable[Mapping]) -> dict:
    """Per rotation the metric at the report seed; then mean and sd over the rotations."""
    records = single_seed(records)
    rotations = {}
    for rotation, rs in _by(records, "rotation").items():
        rotations[int(rotation)] = _metrics_of(rs)
    return {"rotations": rotations,
            **{m: _mean_sd([r[m] for r in rotations.values()]) for m in METRICS}}


def pool_pair(records: Iterable[Mapping], pooled: Mapping[str, str]) -> list[Mapping]:
    """The two directions' test records of a pair, each subject from its target direction.

    `pooled` is `assignment.pair_pool(a, b)`: subject -> the transfer where it is
    target. Raises when a subject is missing, comes from the other direction or
    appears twice in one seed.
    """
    records = check_once(records)
    wrong = [r["subject"] for r in records if pooled.get(r["subject"]) != r["direction"]]
    if wrong:
        raise StatisticsError(f"records not from the direction where the subject is target: "
                              f"{sorted(set(wrong))[:5]}")
    for seed, rs in _by(records, "seed").items():
        missing = set(pooled) - {r["subject"] for r in rs}
        if missing:
            raise StatisticsError(f"seed {seed}: pooled subjects without a prediction: "
                                  f"{sorted(missing)[:5]}")
    return records


# ------------------------------------------------------------------ the bootstrap


def _strata_index(strata: Sequence) -> list[np.ndarray]:
    groups: dict = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    return [np.asarray(v) for _, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]


def _ba_kappa(y: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise balanced accuracy and kappa of predictions `a` against labels `y`."""
    correct = (a == y)
    recalls = []
    for c in (0, 1):
        mask = y == c
        n_c = mask.sum(axis=-1)
        recalls.append(np.where(n_c > 0, (correct & mask).sum(axis=-1) / np.maximum(n_c, 1),
                                np.nan))
    ba = np.nanmean(np.stack(recalls), axis=0)
    n = y.shape[-1]
    po = correct.mean(axis=-1)
    pe = ((y == 1).sum(-1) * (a == 1).sum(-1) + (y == 0).sum(-1) * (a == 0).sum(-1)) / n ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(pe < 1, (po - pe) / (1 - pe), np.where(po == 1, 1.0, np.nan))
    return ba, kappa


def _bootstrap_p(replicates: np.ndarray) -> float | None:
    """Two-sided: twice the smaller tail at zero, (1 + count) / (1 + replicates)."""
    n = len(replicates)
    if n == 0:
        return None
    lower = (1 + np.sum(replicates <= 0)) / (n + 1)
    upper = (1 + np.sum(replicates >= 0)) / (n + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def _auc_avg_replicates(y: np.ndarray, sa: np.ndarray, sb: np.ndarray, index: np.ndarray,
                        cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per replicate row of `index`: AUC_avg of both models over the cells.

    A column of `index` is drawn inside one stratum, so inside one cell: the
    cell of a column is the cell of its first row.
    """
    column_cell = cell_ids[index[0]]
    aa, ab = [], []
    for c in np.unique(cell_ids):
        cols = index[:, column_cell == c]
        aa.append(_auc_rows(y[cols], sa[cols]))
        ab.append(_auc_rows(y[cols], sb[cols]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # a replicate with no cell AUC
        return np.nanmean(np.stack(aa), axis=0), np.nanmean(np.stack(ab), axis=0)


def paired_bootstrap(labels: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float],
                     strata: Sequence, *, cells: Sequence | None = None,
                     resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Sec. 4.6's contrasts: A minus B, on the same resampled subjects.

    Balanced accuracy and kappa on the pooled hard predictions, resampled within
    `strata` (Sec. 4.6: class and direction). AUC_avg over `cells` (None: one
    cell) on its own replicates, drawn with the same seed by class inside each
    cell, so every AUC stratum sits in one cell.
    """
    y = np.asarray(labels, dtype=int)
    sa, sb = np.asarray(prob_a, dtype=float), np.asarray(prob_b, dtype=float)
    a, b = predicted_labels(sa), predicted_labels(sb)
    cells = [None] * len(y) if cells is None else list(cells)
    if not (len(y) == len(a) == len(b) == len(strata) == len(cells)):
        raise StatisticsError("labels, both predictions, strata and cells must align per "
                              "subject")
    groups = _strata_index(strata)
    cell_names = {c: k for k, c in enumerate(_cell_index(cells))}
    cell_ids = np.asarray([cell_names[c] for c in cells])
    rng = np.random.default_rng(seed)
    index = np.concatenate([g[rng.integers(0, len(g), size=(resamples, len(g)))]
                            for g in groups], axis=1)
    auc_groups = _strata_index(list(zip(y.tolist(), cell_ids.tolist())))
    auc_rng = np.random.default_rng(seed)
    auc_index = np.concatenate([g[auc_rng.integers(0, len(g), size=(resamples, len(g)))]
                                for g in auc_groups], axis=1)
    ba_a, k_a = _ba_kappa(y[index], a[index])
    ba_b, k_b = _ba_kappa(y[index], b[index])
    diff, kdiff = ba_a - ba_b, k_a - k_b
    obs_a, obs_ka = _ba_kappa(y[None, :], a[None, :])
    obs_b, obs_kb = _ba_kappa(y[None, :], b[None, :])
    observed = float(obs_a[0] - obs_b[0])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    klo, khi = np.nanpercentile(kdiff, [2.5, 97.5])
    auc_a, auc_b = auc_avg(y, sa, cells), auc_avg(y, sb, cells)
    rep_a, rep_b = _auc_avg_replicates(y, sa, sb, auc_index, cell_ids)
    adiff = rep_a - rep_b
    afinite = adiff[np.isfinite(adiff)]
    alo, ahi = (np.percentile(afinite, [2.5, 97.5]) if len(afinite)
                else (float("nan"), float("nan")))
    return {"difference": observed, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": _bootstrap_p(diff),
            "kappa_difference": float(obs_ka[0] - obs_kb[0]),
            "kappa_ci_low": float(klo), "kappa_ci_high": float(khi),
            "kappa_p_value": _bootstrap_p(kdiff[np.isfinite(kdiff)]),
            "auc_a": auc_a["auc"], "auc_b": auc_b["auc"],
            "auc_difference": float(auc_a["auc"] - auc_b["auc"]),
            "auc_ci_low": float(alo), "auc_ci_high": float(ahi),
            "auc_p_value": _bootstrap_p(afinite), "auc_cells_a": auc_a["cells"],
            "auc_cells_b": auc_b["cells"], "cells": len(cell_names),
            "auc_replicates": adiff,
            "discordant": int(np.sum(a != b)), "n": int(len(y)),
            "strata_sizes": [int(len(g)) for g in groups], "resamples": int(resamples),
            "seed": int(seed), "replicates": diff}


# ------------------------------------------------------------------ expected precision


def expected_precision(n_controls: int, n_cases: int, *,
                       ba_levels: Sequence[float] = PRECISION_BA_LEVELS,
                       discordances: Sequence[float] = PRECISION_DISCORDANCES,
                       alpha: float = ALPHA, power: float = PRECISION_POWER) -> dict:
    """Sec. 4.6: BA 95% half-width per BA level and minimum detectable paired BA
    difference per discordance, normal approximation, two-sided, no Holm."""
    if n_controls < 1 or n_cases < 1:
        raise StatisticsError("expected precision needs at least one subject of each class")
    inverse = 1.0 / n_cases + 1.0 / n_controls
    z_alpha = float(sps.norm.ppf(1 - alpha / 2))
    z_power = float(sps.norm.ppf(power))
    return {"n_controls": int(n_controls), "n_cases": int(n_cases),
            "ba_levels": [float(v) for v in ba_levels],
            "ci_half_width": [z_alpha * math.sqrt(v * (1 - v) * inverse / 4)
                              for v in ba_levels],
            "discordances": [float(d) for d in discordances],
            "minimum_detectable_difference": [(z_alpha + z_power) * math.sqrt(d * inverse / 4)
                                              for d in discordances],
            "alpha": alpha, "power": power, "holm": False,
            "approximation": "normal; sensitivity = specificity = BA; per-class variance of "
                             "the paired recall difference = discordance / n"}


# ------------------------------------------------------------------ Nadeau-Bengio


def nadeau_bengio(differences: Sequence[float], test_train_ratio: float) -> dict:
    """Corrected resampled t over the rotations: var x (1/J + n_test/n_train), df J-1."""
    d = np.asarray(differences, dtype=float)
    J = len(d)
    if J < 2 or not np.all(np.isfinite(d)):
        raise StatisticsError(f"Nadeau-Bengio needs a finite difference per rotation; got {d}")
    if test_train_ratio < 0:
        raise StatisticsError("the test-to-training ratio is not negative")
    mean, var = float(d.mean()), float(d.var(ddof=1))
    se = math.sqrt((1.0 / J + test_train_ratio) * var)
    if se == 0:
        t, p = (0.0, 1.0) if mean == 0 else (math.copysign(math.inf, mean), 0.0)
    else:
        t = mean / se
        p = float(2 * sps.t.sf(abs(t), df=J - 1))
    half = float(sps.t.ppf(0.975, df=J - 1)) * se
    return {"t": float(t), "p_value": p, "mean_difference": mean, "se": se, "df": J - 1,
            "ci_low": mean - half, "ci_high": mean + half, "n": J,
            "test_train_ratio": float(test_train_ratio)}


# ------------------------------------------------------------------ Holm


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm's step-down adjustment over the finite p values of one family."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and math.isfinite(p)]
    out: list[float | None] = [None] * len(pvalues)
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: pvalues[k])):
        running = max(running, min(1.0, (m - rank) * float(pvalues[i])))
        out[i] = running
    return out


# ------------------------------------------------------------------ Friedman, Wilcoxon


def friedman_nemenyi(scores: Mapping[str, Sequence[float]], alpha: float = ALPHA) -> dict:
    """Friedman over blocks (columns), average ranks (1 = best) and the Nemenyi CD."""
    names = list(scores)
    table = np.asarray([scores[k] for k in names], dtype=float)      # (k, N)
    k, N = table.shape
    if k < 3 or N < 1 or not np.all(np.isfinite(table)):
        raise StatisticsError(f"Friedman needs >= 3 methods with a finite score per block; "
                              f"got {k} methods x {N} blocks")
    ranks = np.vstack([sps.rankdata(-table[:, j]) for j in range(N)]).T
    average = ranks.mean(axis=1)
    statistic, p = sps.friedmanchisquare(*table)
    if alpha != ALPHA or k not in NEMENYI_Q05:
        raise StatisticsError("the Nemenyi table is held for alpha 0.05 and 2-10 methods")
    cd = NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6.0 * N))
    position = {n: int(1 + np.sum(average < average[i])) for i, n in enumerate(names)}
    return {"statistic": float(statistic), "p_value": float(p), "k": k, "blocks": N,
            "average_rank": dict(zip(names, map(float, average))), "position": position,
            "critical_difference": cd, "q_alpha": NEMENYI_Q05[k], "alpha": alpha}


def _average_ranks(scores: Mapping[str, Sequence[float]]) -> tuple[list, np.ndarray]:
    names = list(scores)
    table = np.asarray([scores[k] for k in names], dtype=float)      # (k, N)
    if table.ndim != 2 or not np.all(np.isfinite(table)):
        raise StatisticsError("a ranking needs a finite score of every method in every block")
    ranks = np.vstack([sps.rankdata(-table[:, j]) for j in range(table.shape[1])]).T
    return names, ranks.mean(axis=1)


def descriptive_ranking(scores: Mapping[str, Sequence[float]]) -> dict:
    """Secs. 5.1-5.4: average rank (1 = best), position and top three; no test."""
    if len(scores) < 2:
        raise StatisticsError("a ranking needs at least two methods")
    names, average = _average_ranks(scores)
    position = {n: int(1 + np.sum(average < average[i])) for i, n in enumerate(names)}
    return {"average_rank": dict(zip(names, map(float, average))), "position": position,
            "top_three": {n: position[n] <= TOP_RANKS for n in names},
            "blocks": int(len(next(iter(scores.values())))), "k": len(names), "test": None}


def nemenyi_groups(average_rank: Mapping[str, float], critical_difference: float
                   ) -> list[list[str]]:
    """The maximal runs of strategies, by rank, whose rank gap is at most the CD."""
    ordered = sorted(average_rank.items(), key=lambda kv: (kv[1], kv[0]))
    groups: list[list[str]] = []
    for i in range(len(ordered)):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] - ordered[i][1] <= critical_difference:
            j += 1
        if j > i and not (groups and len(groups[-1]) and ordered[j][0] == groups[-1][-1]):
            groups.append([name for name, _ in ordered[i:j + 1]])
    return groups


def _signed_rank_cdf(n: int) -> np.ndarray:
    """P(T+ <= t), t = 0..n(n+1)/2, under the null: each rank's sign is a fair coin."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank].copy()
    return np.cumsum(counts) / 2.0 ** n


def hodges_lehmann(differences: Sequence[float], alpha: float = ALPHA) -> dict:
    """The Walsh-average median and its exact signed-rank (1 - alpha) interval."""
    d = np.asarray([x for x in differences if x is not None and math.isfinite(x)], float)
    n = len(d)
    if n == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "k": None, "walsh": 0,
                "coverage": None, "n": 0, "note": "no finite difference"}
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    M = len(walsh)
    cdf = _signed_rank_cdf(n)
    # k is the largest integer with P(T+ <= k - 1) <= alpha / 2
    admissible = np.nonzero(cdf <= alpha / 2)[0]
    out = {"estimate": float(np.median(walsh)), "walsh": int(M), "n": int(n)}
    if len(admissible) == 0:
        return {**out, "ci_low": None, "ci_high": None, "k": None, "coverage": None,
                "note": f"n = {n}: no exact signed-rank interval reaches {1 - alpha:.2f}"}
    k = int(admissible[-1]) + 1
    return {**out, "ci_low": float(walsh[k - 1]), "ci_high": float(walsh[M - k]), "k": k,
            "coverage": float(1 - 2 * cdf[k - 1]), "note": None}


def wilcoxon_signed_rank(differences: Sequence[float]) -> dict:
    """Two-sided signed-rank test of per-subject differences against zero."""
    d = np.asarray([x for x in differences if x is not None and math.isfinite(x)], float)
    nonzero = d[d != 0]
    hl = hodges_lehmann(d)
    interval = {"hodges_lehmann": hl["estimate"], "ci_low": hl["ci_low"],
                "ci_high": hl["ci_high"], "ci_coverage": hl["coverage"],
                "ci_note": hl["note"]}
    if len(nonzero) == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "rank_biserial": 0.0,
                "n": int(len(d)), "n_nonzero": 0, "median": float(np.median(d)) if len(d)
                else float("nan"), **interval}
    ranks = sps.rankdata(np.abs(nonzero))
    plus, minus = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    result = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
            "rank_biserial": (plus - minus) / (plus + minus), "n": int(len(d)),
            "n_nonzero": int(len(nonzero)), "median": float(np.median(d)), **interval}


# ------------------------------------------------------------------ rows


def _clean(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def significance_row(*, comparison: str, test: str, statistic, p_value, effect_size,
                     effect_size_name: str, ci_low, ci_high, n: int, hypothesis: str,
                     experiment: str, p_adjusted=None, correction: str = "none",
                     alpha: float = ALPHA, **extra) -> dict:
    """One row of `Results/significance*.json`: the dashboard keys, then extras."""
    row = {"comparison": comparison, "test": test, "statistic": statistic,
           "p_value": p_value, "p_adjusted": p_adjusted, "correction": correction,
           "effect_size": effect_size, "effect_size_name": effect_size_name,
           "ci_low": ci_low, "ci_high": ci_high, "n": int(n), "alpha": alpha,
           "hypothesis": hypothesis, "experiment": experiment}
    return {k: _clean(v) for k, v in {**row, **extra}.items()}


def apply_holm(rows: list[dict], family: str) -> list[dict]:
    """Holm over `rows` (one declared family); writes p_adjusted and correction."""
    adjusted = holm([r["p_value"] for r in rows])
    for r, p in zip(rows, adjusted):
        r["p_adjusted"] = _clean(p)
        r["correction"] = f"holm ({family}, m={sum(a is not None for a in adjusted)})"
    return rows


def apply_contrast_families(rows: list[dict], family: str) -> list[dict]:
    """Sec. 4.6: Holm within each metric over the rivals of `rows` (one core pair).

    The balanced-accuracy rows are the primary family; kappa, AUC_avg and
    Nadeau-Bengio are each a secondary family of their own. Rows of any other
    test are left as they are.
    """
    for test in CONTRAST_TESTS:
        members = [r for r in rows if r["test"] == test]
        if members:
            apply_holm(members, f"{family}, {METRIC_OF_TEST[test]}")
    return rows


def fold_differences(records_a: Sequence[Mapping], records_b: Sequence[Mapping]) -> list[float]:
    """Per rotation: BA(A) - BA(B) at the report seed."""
    fa, fb = per_fold(records_a)["rotations"], per_fold(records_b)["rotations"]
    if sorted(fa) != sorted(fb):
        raise StatisticsError(f"the two models cover different rotations: {sorted(fa)} vs "
                              f"{sorted(fb)}")
    return [fa[r]["balanced_accuracy"] - fb[r]["balanced_accuracy"] for r in sorted(fa)]


def pair_contrast(records_a: Sequence[Mapping], records_b: Sequence[Mapping], *,
                  name_a: str, name_b: str, scope: str, test_train_ratio: float,
                  hypothesis: str, experiment: str,
                  resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> list[dict]:
    """Sec. 4.6 for one rival on one pooled scope: bootstrap and Nadeau-Bengio rows.

    `records_*` hold the report seed's pooled test predictions (already
    `pool_pair`-ed when the scope is a pair). Balanced accuracy and kappa
    resample by class and direction (Sec. 4.6); the AUC_avg contrast resamples
    by class inside each rotation-and-direction cell, with the same seed.
    """
    records_a, records_b = single_seed(records_a), single_seed(records_b)
    seed_a = {r.get("seed") for r in records_a}
    if seed_a != {r.get("seed") for r in records_b}:
        raise StatisticsError("the two models are not at the same report seed")
    report_seed = next(iter(seed_a), None)
    comparison = f"{experiment}:{scope}:{name_a} vs {name_b}"
    rows = []
    ra = {r["subject"]: r for r in records_a}
    rb = {r["subject"]: r for r in records_b}
    if set(ra) != set(rb):
        raise StatisticsError(f"{comparison}: the two models predict different subjects")
    subjects = sorted(ra)
    labels = [ra[s]["label"] for s in subjects]
    if any(ra[s]["label"] != rb[s]["label"] for s in subjects):
        raise StatisticsError(f"{comparison}: labels disagree between the two records")
    pa, pb = [ra[s]["probability"] for s in subjects], [rb[s]["probability"] for s in subjects]
    if any(cell_of(ra[s]) != cell_of(rb[s]) for s in subjects):
        raise StatisticsError(f"{comparison}: a subject sits in different cells in the two "
                              f"records")
    strata = [(ra[s]["label"], ra[s].get("direction")) for s in subjects]
    cells = [cell_of(ra[s]) for s in subjects]
    boot = paired_bootstrap(labels, pa, pb, strata, cells=cells, resamples=resamples,
                            seed=seed)
    common = {"contrast": comparison, "rival": name_b, "scope": scope,
              "report_seed": report_seed}
    rows.append(significance_row(
        comparison=comparison, test=TEST_BOOTSTRAP, statistic=boot["difference"],
        p_value=boot["p_value"], effect_size=boot["difference"],
        effect_size_name=f"balanced accuracy difference ({name_a} - {name_b})",
        ci_low=boot["ci_low"], ci_high=boot["ci_high"], n=boot["n"],
        hypothesis=hypothesis, experiment=experiment,
        kappa_difference=boot["kappa_difference"], kappa_ci_low=boot["kappa_ci_low"],
        kappa_ci_high=boot["kappa_ci_high"], discordant=boot["discordant"],
        resamples=boot["resamples"], bootstrap_seed=boot["seed"],
        strata=STRATA, family=FAMILY_PRIMARY, **common,
        information_note="with binary responses the bootstrap's information lies only "
                         "in the discordant subjects"))
    rows.append(significance_row(
        comparison=f"{comparison} (kappa)", test=TEST_KAPPA,
        statistic=boot["kappa_difference"], p_value=boot["kappa_p_value"],
        effect_size=boot["kappa_difference"],
        effect_size_name=f"Cohen's kappa difference ({name_a} - {name_b})",
        ci_low=boot["kappa_ci_low"], ci_high=boot["kappa_ci_high"], n=boot["n"],
        hypothesis=hypothesis, experiment=experiment, discordant=boot["discordant"],
        resamples=boot["resamples"], bootstrap_seed=boot["seed"],
        strata=STRATA, family=FAMILY_SECONDARY, **common))
    rows.append(significance_row(
        comparison=f"{comparison} (AUC_avg)", test=TEST_AUC_AVG,
        statistic=boot["auc_difference"], p_value=boot["auc_p_value"],
        effect_size=boot["auc_difference"],
        effect_size_name=f"AUC_avg difference ({name_a} - {name_b})",
        ci_low=boot["auc_ci_low"], ci_high=boot["auc_ci_high"], n=boot["n"],
        hypothesis=hypothesis, experiment=experiment, auc_a=boot["auc_a"],
        auc_b=boot["auc_b"], auc_cells_a=boot["auc_cells_a"],
        auc_cells_b=boot["auc_cells_b"], cells=boot["cells"], discordant=boot["discordant"],
        resamples=boot["resamples"], bootstrap_seed=boot["seed"], strata=AUC_STRATA,
        family=FAMILY_SECONDARY, **common))
    nb = nadeau_bengio(fold_differences(records_a, records_b), test_train_ratio)
    rows.append(significance_row(
        comparison=f"{comparison} (per-rotation)", test=TEST_NADEAU_BENGIO, statistic=nb["t"],
        p_value=nb["p_value"], effect_size=nb["mean_difference"],
        effect_size_name=f"mean per-rotation balanced accuracy difference ({name_a} - {name_b})",
        ci_low=nb["ci_low"], ci_high=nb["ci_high"], n=nb["n"], hypothesis=hypothesis,
        experiment=experiment, df=nb["df"], test_train_ratio=nb["test_train_ratio"],
        family=FAMILY_SECONDARY, **common))
    return rows
