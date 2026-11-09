"""Plan v63: the target filter (Sec. 0, part 1) and the shift filter (Sec. 4.7)."""

import numpy as np
import pytest
import torch

from MIL_CREDA_EEG import bag_data, partition, two_phase
from MIL_CREDA_EEG import v63_filters as Fl


def test_the_permutation_p_value_counts_the_observed_statistic():
    assert Fl.permutation_p(0.9, [0.95, 0.5]) == pytest.approx(2 / 3)
    assert Fl.permutation_p(1.0, [0.5] * 999) == pytest.approx(1 / 1000)


def _rows(scores, labels, seeds=(1, 2)):
    return [{"subject": f"s{i}", "label": y, "seed": k, "probability": p + 0.01 * k}
            for k in seeds for i, (p, y) in enumerate(zip(scores, labels))]


def test_a_domain_passes_only_below_alpha():
    labels = [0] * 20 + [1] * 20
    good = Fl.target_filter_verdict(_rows(np.linspace(0, 1, 40), labels), permutations=200)
    assert good["auc"] == pytest.approx(1.0) and good["passes"] and good["p_value"] < 0.05
    rng = np.random.default_rng(0)
    flat = Fl.target_filter_verdict(_rows(rng.permutation(np.linspace(0, 1, 40)), labels),
                                    permutations=200)
    assert flat["passes"] == (flat["p_value"] < Fl.ALPHA)
    assert set(flat["aucBySeed"]) == {"1", "2"}
    with pytest.raises(Exception):
        Fl.target_filter_verdict(_rows([0.2, 0.8], [0, 1])[:-1])


def _bagset(root):
    rng = np.random.default_rng(0)
    bags = []
    for i in range(24):
        s, fold, label = f"UB{i:04d}", i % 4, ("yes", "no")[(i // 4) % 2]
        for c in {0: ("L6",), 1: ("Robots",), 2: ("L6", "Robots")}[i % 3]:
            path = root / f"{s}_{c}.npz"
            np.savez(path, signal=rng.normal(size=(4, 3, 512)).astype(np.float32))
            bags.append(bag_data.Bag(subject=s, condition=c, record_id=f"{s}-{c}", label=label,
                                     sex="M", fold=fold, device="CGX", channels=("a", "b", "c"),
                                     n_windows=4, path=path))
    rot = {str(r): partition.rotation_roles(r) for r in range(4)}
    return bag_data.BagSet(bags=tuple(bags), partition_sha256="x" * 64, rotations=rot)


def test_the_target_filter_reads_only_the_domains_own_labels(tmp_path):
    bags = _bagset(tmp_path)
    data, test = Fl.own_domain_data("Robots", bags, 0)
    seen = {b.condition for v in data.inner.values() for side in v.values() for b in side}
    assert seen == {"Robots"} and {b.condition for b in test} == {"Robots"}
    test_subjects = {b.subject for b in test}
    train_subjects = {b.subject for b in data.rotations[0]["source_train"]}
    assert not test_subjects & train_subjects


def test_the_shift_filter_reads_no_class_label():
    w = torch.randn(6, 3, 512)
    a = [two_phase.BagTensor("s", "c", 0, w)]
    b = [two_phase.BagTensor("s", "c", 1, w)]
    assert np.array_equal(Fl.window_features(a)[0], Fl.window_features(b)[0])


def test_the_shift_folds_never_split_a_subject():
    rng = np.random.default_rng(1)
    subjects = [f"s{i}" for i in range(20)]
    groups = np.repeat(subjects, 5)
    domain = {s: int(i >= 10) for i, s in enumerate(subjects)}
    X = rng.normal(size=(len(groups), 4)) + np.repeat([domain[s] for s in subjects], 5)[:, None]
    auc = Fl.domain_classifier_auc(X, groups, domain)
    assert 0.5 < auc <= 1.0
