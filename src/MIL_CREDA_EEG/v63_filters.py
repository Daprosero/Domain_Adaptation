"""Plan v63: the target filter (Sec. 0, part 1) and the label-free shift filter (Sec. 4.7).

**Target filter [exp:S0-codificador], part 1.** A domain enters as target only if a model
trained with its OWN labels separates its unseen subjects from chance. Per domain and
rotation of the frozen partition, the model of Sec. 7.3 (the spectral encoder, MIL attention
and the common head; MIL-CREDA's phase-1 group with its neutral aggregation, gamma 0 and
temperature 10) is selected by the nested search of Secs. 4.5 and 7.7 on the three inner
splits of the non-test folds (`lr1` and the explicit L2 `l2`, wd1 = 0, 20 trials, the best pooled epoch E1) and refit on
every non-test fold for E1 epochs with each of the 10 report seeds; it predicts that
rotation's test fold. The statistic is the subject-level AUC of the seed-mean probability
pooled over the four rotations (each subject predicted once); its null is the AUC of the
same scores under `PERMUTATIONS` = 1000 permutations of the subjects' labels, and the domain
passes when p = (1 + #{null >= observed}) / (1 + 1000) < 0.05. Only the domain's own
labels are read, and the labels of a rotation's test fold choose nothing in that rotation:
they only score.

**Shift filter [exp:F-corrimiento].** Per transfer and rotation, a logistic regression
(standardized features, sklearn's default C = 1) separates the source from the target
TRAINING subjects of that rotation (every non-test fold, Sec. 4.3) on the fixed spectral
features of their windows (`v63_setup.feature_probe`), with a 5-fold cross-validation
stratified by domain and grouped by subject; a subject's score is the mean probability of
its windows and the AUC is subject-level. No class label is read. The transfer statistic is
the mean of the four rotations' AUC; its null permutes the domain labels of the subjects,
independently per rotation, `PERMUTATIONS` times; p < 0.05 classifies the transfer as
"shift", otherwise "no-shift". Each rotation's own AUC feeds the phase-2 shift gate of
MIL-CREDA in that rotation (`v63_setup.gate_weight`).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from MIL_CREDA_EEG import assignment as A
from MIL_CREDA_EEG import partition, search, statistics, two_phase
from MIL_CREDA_EEG import v63_setup as V

__provenance__ = {
    "revision": V.REVISION,
    "sections": ["0", "4.7"],
    "experiments": ["S0-codificador", "F-corrimiento"],
    "invariants": [
        "the_permutation_p_value_counts_the_observed_statistic",
        "the_target_filter_reads_only_the_domains_own_labels",
        "the_shift_filter_reads_no_class_label",
        "the_shift_folds_never_split_a_subject",
        "a_domain_passes_only_below_alpha",
    ],
}

PERMUTATIONS = 1000
ALPHA = 0.05
PERMUTATION_SEED = 20261108
SHIFT_FOLDS = 5
SHIFT_FOLD_SEED = 0
REPORT_SEEDS = tuple(range(1, 11))
FILTER_GROUP = "MIL-CREDA"


def permutation_p(observed: float, null: Sequence[float]) -> float:
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    return float(statistics._auc_rows(y[None, :], s[None, :])[0])


def label_permutation_null(labels: Sequence[int], scores: Sequence[float],
                           permutations: int = PERMUTATIONS,
                           seed: int = PERMUTATION_SEED) -> np.ndarray:
    """AUC of fixed `scores` under `permutations` shuffles of `labels`."""
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    Y = np.stack([rng.permutation(y) for _ in range(permutations)])
    return statistics._auc_rows(Y, np.broadcast_to(s, Y.shape))


# ------------------------------------------------------------------ the target filter


def own_domain_data(domain: str, bags, rotation: int,
                    prep: V.Prepared | None = None) -> tuple[search.SearchData, list]:
    """The domain's own subjects as a one-sided SearchData of `rotation` (the source side
    holds them, the target side mirrors them only so the trunk widths can be read), and
    its test-fold bags."""
    if domain not in V.DOMAINS:
        raise V.V63Error(f"{domain!r} is not a domain of Sec. 0")
    prep = prep or V.Prepared(bags)
    folds = bags.folds()
    held = sorted(s for s in bags.holders(domain) if s in folds)
    rot = bags.rotations[V.rotation_key(bags.rotations, rotation)]
    splits = A.inner_splits(rot)
    test_fold = int(rot["test_fold"])
    of = lambda fs: prep([bags.get(s, domain) for s in held if folds[s] in fs], None)
    non_test = [int(k) for k in rot["non_test_folds"]]
    refit = of(non_test)
    rotations = {rotation: {"source_train": refit, "source_valid": [],
                            "target_train": refit, "target_valid": []}}
    inner = {}
    for i, s in enumerate(splits):
        train, valid = of([int(k) for k in s["train_folds"]]), of([int(s["validation_fold"])])
        inner[(rotation, i)] = {"source_train": train, "source_valid": valid,
                                "target_train": train, "target_valid": valid}
    pseudo = A.Transfer("own", domain, domain)
    data = search.SearchData(transfer=pseudo, partition_sha256=bags.partition_sha256,
                             rotations=rotations,
                             folds={rotation: search.rotation_folds(bags.rotations)[rotation]},
                             inner=inner)
    return data, of([test_fold])


def target_filter_verdict(rows: Sequence[Mapping], permutations: int = PERMUTATIONS,
                          seed: int = PERMUTATION_SEED) -> dict:
    """Rows {subject, label, seed, probability} of one domain's four rotations -> the
    Sec. 0 inclusion record."""
    by_subject: dict[str, list] = {}
    labels: dict[str, int] = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(float(r["probability"]))
        labels[r["subject"]] = int(r["label"])
    subjects = sorted(by_subject)
    seeds = sorted({int(r["seed"]) for r in rows})
    counts = {s: len(v) for s, v in by_subject.items()}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) != len(seeds):
        raise V.V63Error("every subject is predicted once per seed (pooled test folds)")
    y = [labels[s] for s in subjects]
    score = [float(np.mean(by_subject[s])) for s in subjects]
    observed = auc(y, score)
    null = label_permutation_null(y, score, permutations, seed)
    p = permutation_p(observed, null)
    per_seed = {}
    for k in seeds:
        sub = {r["subject"]: float(r["probability"]) for r in rows if int(r["seed"]) == k}
        per_seed[str(k)] = auc(y, [sub[s] for s in subjects])
    return {"auc": observed, "p_value": p, "permutations": int(permutations),
            "permutationSeed": int(seed), "alpha": ALPHA, "passes": bool(p < ALPHA),
            "subjects": len(subjects), "positives": int(sum(y)),
            "negatives": int(len(y) - sum(y)), "seeds": seeds, "aucBySeed": per_seed,
            "statistic": "subject-level AUC of the seed-mean probability, test folds of the "
                         "four rotations pooled"}


# ------------------------------------------------------------------ the shift filter


def window_features(bags: Sequence[two_phase.BagTensor]) -> tuple[np.ndarray, np.ndarray]:
    """(windows, features) fixed spectral features and the subject of each row."""
    probe = V.feature_probe(int(bags[0].windows.shape[1]))
    with torch.no_grad():
        X = [torch.nan_to_num(probe.features(b.windows)).double().numpy() for b in bags]
    groups = np.concatenate([[b.subject] * len(x) for b, x in zip(bags, X)])
    return np.concatenate(X), groups


def domain_classifier_auc(X: np.ndarray, groups: np.ndarray, subject_domain: Mapping[str, int],
                          folds: int = SHIFT_FOLDS, fold_seed: int = SHIFT_FOLD_SEED) -> float:
    """Subject-level AUC of a grouped, stratified 5-fold logistic domain classifier."""
    subjects = np.asarray(sorted(subject_domain))
    d_rows = np.asarray([subject_domain[g] for g in groups])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
    p = np.full(len(groups), np.nan)
    for train, test in splitter.split(X, d_rows, groups):
        if len(set(groups[train]) & set(groups[test])):
            raise V.V63Error("a subject is on both sides of a shift-filter fold")
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[train], d_rows[train])
        p[test] = clf.predict_proba(X[test])[:, 1]
    score = {s: float(np.mean(p[groups == s])) for s in subjects}
    return auc([subject_domain[s] for s in subjects], [score[s] for s in subjects])


def shift_rotation(t: A.Transfer, bags, rotation: int,
                   prep: V.Prepared | None = None) -> dict:
    """The observed domain AUC of one rotation and the data its null needs."""
    data = V.search_data(t, bags, prep, rotations=[rotation])
    d = data.rotations[rotation]
    src, tgt = d["source_train"], d["target_train"]
    X, groups = window_features([*src, *tgt])
    domain = {**{b.subject: 0 for b in src}, **{b.subject: 1 for b in tgt}}
    return {"X": X, "groups": groups, "domain": domain,
            "auc": domain_classifier_auc(X, groups, domain)}


def shift_gate_auc(t: A.Transfer, bags, rotation: int, prep: V.Prepared | None = None) -> float:
    return shift_rotation(t, bags, rotation, prep)["auc"]


def shift_filter(t: A.Transfer, bags, permutations: int = PERMUTATIONS,
                 seed: int = PERMUTATION_SEED, prep: V.Prepared | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 state: dict | None = None) -> dict:
    """The Sec. 4.7 record of one transfer; `state` (a dict the caller persists) holds
    the null drawn so far, so a run cut at its deadline continues where it stopped."""
    prep = prep or V.Prepared(bags)
    rots = [shift_rotation(t, bags, r, prep) for r in range(partition.N_FOLDS)]
    observed = [r["auc"] for r in rots]
    state = {} if state is None else state
    null = state.setdefault("null", [])
    rng = np.random.default_rng([seed, len(null)])
    while len(null) < permutations:
        if should_stop is not None and should_stop():
            return {"complete": False, "drawn": len(null)}
        values = []
        for r in rots:
            subjects = sorted(r["domain"])
            shuffled = rng.permutation([r["domain"][s] for s in subjects])
            values.append(domain_classifier_auc(r["X"], r["groups"],
                                                dict(zip(subjects, map(int, shuffled)))))
        null.append(float(np.mean(values)))
    statistic = float(np.mean(observed))
    p = permutation_p(statistic, null)
    return {"complete": True, "transfer": t.name, "arm": t.arm, "aucByRotation": observed,
            "auc": statistic, "p_value": p, "permutations": int(permutations),
            "permutationSeed": int(seed), "alpha": ALPHA,
            "classification": "shift" if p < ALPHA else "no-shift",
            "gateByRotation": [V.gate_weight(a) for a in observed],
            "subjectsByRotation": [len(r["domain"]) for r in rots]}
