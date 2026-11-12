"""Label-free selection score for phase-2 hyperparameters (protocol v58 centroid rule).

Shared by MIL-CREDA and every comparator, so all of them are selected by the same rule.
No target label is read:

1. the adapted model pseudo-labels the target bags it adapted on (the adapt half, never
   the target test);
2. the class centroids of their bag representations (the head's input) are computed;
3. held-out labelled SOURCE bags, represented by the same model's source path, are assigned
   to the nearest centroid;
4. the score is the balanced accuracy of that assignment against the real source labels.

If the adaptation organised the target classes well, its centroids classify the source.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def centroid_score(target_reps, target_pseudo, source_reps, source_labels) -> float:
    """BA of nearest-target-centroid assignment of labelled source bags.

    target_reps (n, d) and target_pseudo (n,) come from the target adapt half;
    source_reps (m, d) and source_labels (m,) from held-out labelled source bags.
    A class with no pseudo-labelled target bag leaves the rule undefined: the score is 0.0,
    which can never be selected over a configuration that keeps both classes.
    """
    t = np.asarray(target_reps, dtype=np.float64)
    q = np.asarray(target_pseudo).astype(int).reshape(-1)
    s = np.asarray(source_reps, dtype=np.float64)
    y = np.asarray(source_labels).astype(int).reshape(-1)
    classes = np.unique(y)
    if any(not np.any(q == c) for c in classes):
        return 0.0
    centroids = np.stack([t[q == c].mean(0) for c in classes])
    dist = ((s[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return float(balanced_accuracy_score(y, classes[np.argmin(dist, axis=1)]))
