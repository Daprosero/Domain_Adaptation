"""The report and variance design: subject-level metrics, pooled summaries and contrasts.

Experiments plan v40, Secs. 4.4 and 4.6, with the significance tables of
Secs. 5.1-5.3 and the tests of Secs. 6.1-6.3. Every function here reads
predictions that already exist (one record per target test subject, rotation,
seed and strategy, as `two_phase.predictions` writes them); nothing here trains.

**Metrics (Sec. 4.4).** Subject level only, because the bag is the subject:
balanced accuracy (primary, `two_phase.balanced_accuracy`), Cohen's kappa,
the area under the ROC curve, sensitivity and specificity. A bag is predicted
positive when its probability of the positive class exceeds 0.5, which is the
argmax of the two-class prediction that training and selection read (a tie
goes to class 0, as `torch.argmax` does).

**Two summaries (Sec. 4.6).** The pooled one joins the test predictions of
the four rotations, each subject predicted once, and computes one metric over
the target pool (`pooled_by_seed`; the headline number). The per-fold mean and
deviation (`per_fold`) is the evidence that the pooled number does not hang on
one rotation, and the AUC is reported per fold too, because the pooled AUC
mixes four models' probabilities. With three training seeds, the pooled value
is computed per seed and reported as the mean and deviation across seeds
(`summarize_seeds`).

**The pair (Sec. 4.6).** The primary performance contrast is on each core pair
with its two directions pooled: every assigned subject is target exactly once
across the two directions and is predicted by the model of the direction where
it is target. `pool_pair` asserts it against `assignment.pair_pool`.

**The primary contrast (Sec. 4.6).** A paired, class-stratified subject
bootstrap (Koehn 2004 for the pairing, Robin et al. 2011 for the
stratification): each replicate resamples subjects with replacement inside
every (class, direction) stratum, scores both models on the same replicate,
and recomputes the whole balanced accuracy and kappa. Reported: the difference,
its 95% percentile interval, a p value, the kappa difference and the number of
subjects on which the two models disagree (the only subjects that carry
information with binary responses). Holm adjusts the p values of a declared
family (`holm`). The AUCs on the same subjects are compared with DeLong's test
(`delong`, the structural-components algorithm of Sun and Xu 2014). The
secondary contrast is Nadeau and Bengio's corrected resampled t test over the
four rotations, with the three seeds averaged per rotation (`nadeau_bengio`).

**The arm ranking (Secs. 5.1-5.3).** Friedman's test over the nine adapted
strategies, blocks the core transfers, with the Nemenyi critical difference
of the critical-difference diagram (Demsar 2006, alpha 0.05) as its post hoc
(`friedman_nemenyi`).

**The mechanism tests (Secs. 6.1-6.3).** Wilcoxon's signed-rank test over
subjects, with the matched-pairs rank-biserial correlation as effect size
(`wilcoxon_signed_rank`).

**Implementation choices the plan does not fix.** The bootstrap draws
`BOOTSTRAP_RESAMPLES` = 10000 replicates from `BOOTSTRAP_SEED`; its p value is
two-sided, twice the smaller tail of the replicate differences at zero with
the (1 + count) / (1 + replicates) convention, so it is never exactly zero.
Every other test is two-sided as well; the direction a hypothesis states is
read from the sign of the effect, never from a one-sided p. The Nadeau-Bengio
test-to-training ratio is an input (`test_train_ratio`): the campaign passes
the target-test subjects over the subjects the models were fit on, summed over
the two directions and averaged over rotations. The plan does not say how the
three seeds become one prediction per subject for the bootstrap and DeLong;
that is the required input `reduction` (`SEED_REDUCTIONS`), with no default.

Significance rows carry exactly the dashboard's keys (`SIGNIFICANCE_KEYS`)
plus `hypothesis` and `experiment` and any extra key a table needs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from MIL_CREDA_EEG.two_phase import balanced_accuracy

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v40.md",
    "sections": ["4.4", "4.6", "5.1", "5.2", "5.3", "6.1", "6.2", "6.3"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo", "K1-atencion",
                    "K2-emparejamiento", "K3-ablacion"],
    "invariants": [
        "metrics_are_subject_level_and_balanced_accuracy_is_the_mean_recall",
        "a_subject_predicted_twice_in_one_seed_is_refused",
        "the_pooled_summary_is_per_seed_then_mean_and_sd_across_seeds",
        "the_per_fold_summary_averages_seeds_within_each_rotation",
        "a_pair_pools_each_subject_from_the_direction_where_it_is_target",
        "the_seed_reduction_has_no_default",
        "the_bootstrap_is_paired_and_keeps_every_stratum_size",
        "the_bootstrap_reports_difference_interval_kappa_and_discordant_subjects",
        "identical_models_give_a_zero_difference_and_p_one",
        "delong_matches_its_structural_components_definition",
        "nadeau_bengio_reduces_to_the_one_sample_t_at_zero_ratio",
        "holm_is_the_step_down_adjustment",
        "friedman_ranks_with_one_as_best_and_nemenyi_gives_the_critical_difference",
        "wilcoxon_reports_the_rank_biserial_effect",
        "significance_rows_carry_exactly_the_dashboard_keys",
    ],
}

ALPHA = 0.05
POSITIVE_THRESHOLD = 0.5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261102
#: Required choice: how the three report seeds become one prediction per subject.
SEED_REDUCTIONS = ("mean-probability", "per-seed")
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
TEST_DELONG = "DeLong (paired AUC)"
TEST_NADEAU_BENGIO = "Nadeau-Bengio corrected resampled t (per-rotation balanced accuracy)"
TEST_WILCOXON = "Wilcoxon signed-rank over subjects"
TEST_FRIEDMAN = "Friedman over core transfers, Nemenyi critical difference"


class StatisticsError(RuntimeError):
    """A summary or contrast was asked for something Secs. 4.4-4.6 do not allow."""


# ------------------------------------------------------------------ metrics (Sec. 4.4)


def predicted_labels(probabilities: Sequence[float]) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float) > POSITIVE_THRESHOLD).astype(int)


def subject_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict:
    """Balanced accuracy, kappa, AUC, sensitivity and specificity over subjects."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise StatisticsError("labels and probabilities must be one value per subject")
    yhat = predicted_labels(p)
    both = len(np.unique(y)) == 2
    sensitivity = float(np.mean(yhat[y == 1] == 1)) if (y == 1).any() else float("nan")
    specificity = float(np.mean(yhat[y == 0] == 0)) if (y == 0).any() else float("nan")
    kappa = float(_ba_kappa(y[None, :], yhat[None, :])[1][0]) if len(y) else float("nan")
    return {"balanced_accuracy": balanced_accuracy(y, yhat),
            "kappa": kappa if math.isfinite(kappa) else float("nan"),
            "auc": float(roc_auc_score(y, p)) if both else float("nan"),
            "sensitivity": sensitivity, "specificity": specificity,
            "n": int(len(y)), "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum())}


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


def _metrics_of(records: Sequence[Mapping]) -> dict:
    return subject_metrics([r["label"] for r in records], [r["probability"] for r in records])


def pooled_by_seed(records: Iterable[Mapping]) -> dict[int, dict]:
    """Per seed: one metric over every target subject of the four rotations."""
    records = check_once(records)
    return {int(seed): _metrics_of(rs) for seed, rs in _by(records, "seed").items()}


def _mean_sd(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v))}


def summarize_seeds(by_seed: Mapping[int, Mapping]) -> dict:
    """Mean and sample deviation across seeds of each pooled metric."""
    return {m: _mean_sd([s[m] for s in by_seed.values()]) for m in METRICS}


def per_fold(records: Iterable[Mapping]) -> dict:
    """Per rotation the metric of each seed, averaged over seeds; then mean and sd."""
    records = check_once(records)
    rotations = {}
    for rotation, rs in _by(records, "rotation").items():
        seeds = [_metrics_of(x) for x in _by(rs, "seed").values()]
        rotations[int(rotation)] = {m: float(np.nanmean([s[m] for s in seeds]))
                                    if any(math.isfinite(s[m]) for s in seeds)
                                    else float("nan") for m in METRICS}
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


def reduce_seeds(records: Iterable[Mapping], reduction: str) -> dict:
    """Seeds -> predictions per subject: `{"all": [...]}` or `{seed: [...]}`."""
    if reduction not in SEED_REDUCTIONS:
        raise StatisticsError(f"the seed reduction is a required choice among "
                              f"{list(SEED_REDUCTIONS)}; got {reduction!r}")
    records = check_once(records)
    by_seed = _by(records, "seed")
    if reduction == "per-seed":
        return {int(s): sorted(rs, key=lambda r: r["subject"]) for s, rs in by_seed.items()}
    subjects = [{r["subject"] for r in rs} for rs in by_seed.values()]
    if any(s != subjects[0] for s in subjects):
        raise StatisticsError("mean-probability needs every seed to predict the same subjects")
    out = []
    for subject, rs in _by(records, "subject").items():
        labels = {r["label"] for r in rs}
        if len(labels) != 1:
            raise StatisticsError(f"{subject}: two labels across seeds")
        first = rs[0]
        out.append({**{k: first[k] for k in ("subject", "direction", "label") if k in first},
                    "probability": float(np.mean([r["probability"] for r in rs])),
                    "seeds": sorted(int(r["seed"]) for r in rs)})
    return {"all": out}


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


def paired_bootstrap(labels: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float],
                     strata: Sequence, *, resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Sec. 4.6's primary contrast: A minus B, on the same resampled subjects."""
    y = np.asarray(labels, dtype=int)
    a, b = predicted_labels(prob_a), predicted_labels(prob_b)
    if not (len(y) == len(a) == len(b) == len(strata)):
        raise StatisticsError("labels, both predictions and strata must align per subject")
    groups = _strata_index(strata)
    rng = np.random.default_rng(seed)
    index = np.concatenate([g[rng.integers(0, len(g), size=(resamples, len(g)))]
                            for g in groups], axis=1)
    ba_a, k_a = _ba_kappa(y[index], a[index])
    ba_b, k_b = _ba_kappa(y[index], b[index])
    diff, kdiff = ba_a - ba_b, k_a - k_b
    obs_a, obs_ka = _ba_kappa(y[None, :], a[None, :])
    obs_b, obs_kb = _ba_kappa(y[None, :], b[None, :])
    observed = float(obs_a[0] - obs_b[0])
    lower = (1 + np.sum(diff <= 0)) / (resamples + 1)
    upper = (1 + np.sum(diff >= 0)) / (resamples + 1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    klo, khi = np.nanpercentile(kdiff, [2.5, 97.5])
    return {"difference": observed, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, 2 * min(lower, upper))),
            "kappa_difference": float(obs_ka[0] - obs_kb[0]),
            "kappa_ci_low": float(klo), "kappa_ci_high": float(khi),
            "discordant": int(np.sum(a != b)), "n": int(len(y)),
            "strata_sizes": [int(len(g)) for g in groups], "resamples": int(resamples),
            "seed": int(seed), "replicates": diff}


# ------------------------------------------------------------------ DeLong


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(labels: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float]) -> dict:
    """DeLong's test of two correlated AUCs (fast structural components, Sun and Xu 2014)."""
    y = np.asarray(labels, dtype=int)
    scores = np.vstack([np.asarray(prob_a, float), np.asarray(prob_b, float)])
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise StatisticsError("DeLong needs at least two subjects of each class")
    tx = np.vstack([_midrank(r) for r in pos])
    ty = np.vstack([_midrank(r) for r in neg])
    tz = np.vstack([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v10) / m + np.cov(v01) / n
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        z, p = (0.0, 1.0) if diff == 0 else (math.copysign(math.inf, diff), 0.0)
    else:
        z = diff / math.sqrt(var)
        p = float(2 * sps.norm.sf(abs(z)))
    half = 1.959963984540054 * math.sqrt(max(var, 0.0))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": diff,
            "variance": var, "z": float(z), "p_value": p, "ci_low": diff - half,
            "ci_high": diff + half, "n": int(len(y)), "covariance": cov}


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


def wilcoxon_signed_rank(differences: Sequence[float]) -> dict:
    """Two-sided signed-rank test of per-subject differences against zero."""
    d = np.asarray([x for x in differences if x is not None and math.isfinite(x)], float)
    nonzero = d[d != 0]
    if len(nonzero) == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "rank_biserial": 0.0,
                "n": int(len(d)), "n_nonzero": 0, "median": float(np.median(d)) if len(d)
                else float("nan")}
    ranks = sps.rankdata(np.abs(nonzero))
    plus, minus = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    result = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
            "rank_biserial": (plus - minus) / (plus + minus), "n": int(len(d)),
            "n_nonzero": int(len(nonzero)), "median": float(np.median(d))}


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


def fold_differences(records_a: Sequence[Mapping], records_b: Sequence[Mapping]) -> list[float]:
    """Per rotation: BA(A) - BA(B), each the mean over seeds of that rotation's BA."""
    fa, fb = per_fold(records_a)["rotations"], per_fold(records_b)["rotations"]
    if sorted(fa) != sorted(fb):
        raise StatisticsError(f"the two models cover different rotations: {sorted(fa)} vs "
                              f"{sorted(fb)}")
    return [fa[r]["balanced_accuracy"] - fb[r]["balanced_accuracy"] for r in sorted(fa)]


def pair_contrast(records_a: Sequence[Mapping], records_b: Sequence[Mapping], *,
                  name_a: str, name_b: str, scope: str, reduction: str,
                  test_train_ratio: float, hypothesis: str, experiment: str,
                  resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> list[dict]:
    """Sec. 4.6 for one rival on one pooled scope: bootstrap, DeLong and Nadeau-Bengio rows.

    `records_*` hold every seed's pooled test predictions (already `pool_pair`-ed
    when the scope is a pair). The bootstrap strata are (class, direction).
    """
    reduced_a, reduced_b = reduce_seeds(records_a, reduction), reduce_seeds(records_b,
                                                                            reduction)
    if sorted(reduced_a) != sorted(reduced_b):
        raise StatisticsError("the two models do not share their seeds")
    comparison = f"{experiment}:{scope}:{name_a} vs {name_b}"
    rows = []
    for key in sorted(reduced_a, key=str):
        ra = {r["subject"]: r for r in reduced_a[key]}
        rb = {r["subject"]: r for r in reduced_b[key]}
        if set(ra) != set(rb):
            raise StatisticsError(f"{comparison}: the two models predict different subjects")
        subjects = sorted(ra)
        labels = [ra[s]["label"] for s in subjects]
        if any(ra[s]["label"] != rb[s]["label"] for s in subjects):
            raise StatisticsError(f"{comparison}: labels disagree between the two records")
        pa, pb = [ra[s]["probability"] for s in subjects], [rb[s]["probability"] for s in subjects]
        strata = [(ra[s]["label"], ra[s].get("direction")) for s in subjects]
        label = comparison if key == "all" else f"{comparison} [seed {key}]"
        boot = paired_bootstrap(labels, pa, pb, strata, resamples=resamples, seed=seed)
        rows.append(significance_row(
            comparison=label, test=TEST_BOOTSTRAP, statistic=boot["difference"],
            p_value=boot["p_value"], effect_size=boot["difference"],
            effect_size_name=f"balanced accuracy difference ({name_a} - {name_b})",
            ci_low=boot["ci_low"], ci_high=boot["ci_high"], n=boot["n"],
            hypothesis=hypothesis, experiment=experiment,
            kappa_difference=boot["kappa_difference"], kappa_ci_low=boot["kappa_ci_low"],
            kappa_ci_high=boot["kappa_ci_high"], discordant=boot["discordant"],
            resamples=boot["resamples"], bootstrap_seed=boot["seed"],
            strata="class x direction", seed_reduction=reduction,
            information_note="with binary responses the bootstrap's information lies only "
                             "in the discordant subjects"))
        try:
            dl = delong(labels, pa, pb)
            rows.append(significance_row(
                comparison=f"{label} (AUC)", test=TEST_DELONG, statistic=dl["z"],
                p_value=dl["p_value"], effect_size=dl["difference"],
                effect_size_name=f"AUC difference ({name_a} - {name_b})",
                ci_low=dl["ci_low"], ci_high=dl["ci_high"], n=dl["n"],
                hypothesis=hypothesis, experiment=experiment, auc_a=dl["auc_a"],
                auc_b=dl["auc_b"], seed_reduction=reduction))
        except StatisticsError as error:
            rows.append(significance_row(
                comparison=f"{label} (AUC)", test=TEST_DELONG, statistic=None, p_value=None,
                effect_size=None, effect_size_name=f"AUC difference ({name_a} - {name_b})",
                ci_low=None, ci_high=None, n=len(subjects), hypothesis=hypothesis,
                experiment=experiment, not_computed=str(error)))
    nb = nadeau_bengio(fold_differences(records_a, records_b), test_train_ratio)
    rows.append(significance_row(
        comparison=f"{comparison} (per-rotation)", test=TEST_NADEAU_BENGIO, statistic=nb["t"],
        p_value=nb["p_value"], effect_size=nb["mean_difference"],
        effect_size_name=f"mean per-rotation balanced accuracy difference ({name_a} - {name_b})",
        ci_low=nb["ci_low"], ci_high=nb["ci_high"], n=nb["n"], hypothesis=hypothesis,
        experiment=experiment, df=nb["df"], test_train_ratio=nb["test_train_ratio"],
        seeds="averaged per rotation"))
    return rows
