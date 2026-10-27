"""Domain separability instrument (experiments plan v03, Secs. 2.7 and 5), on
synthetic records only: no BALLADEER file is read here."""

from __future__ import annotations

import numpy as np
import pytest

from MIL_CREDA_EEG import audit_domains as dom
from MIL_CREDA_EEG.modality import CGX_CHANNELS, EMOTIV_CHANNELS

RATE = 256.0
SECONDS = 40
N_CH = 6


def _record(rng, alpha_amp: float, rate: float = RATE) -> np.ndarray:
    t = np.arange(int(rate * SECONDS)) / rate
    noise = rng.standard_normal((len(t), N_CH))
    phase = rng.uniform(0, 2 * np.pi, N_CH)
    alpha = alpha_amp * np.sin(2 * np.pi * 10.0 * t[:, None] + phase[None, :])
    return 4000.0 + noise + alpha  # an Emotiv-like DC offset the band-pass removes


def _features(signal: np.ndarray, rate: float = RATE) -> np.ndarray:
    x, usable = dom.preprocess(signal, rate)
    return dom.record_features(dom.window_features(x, usable)).ravel()


def _design(alpha_a: float, alpha_b: float, n_subjects: int = 40, seed: int = 1):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in range(n_subjects):
        side = s % 2
        X.append(_features(_record(rng, alpha_a if side else alpha_b)))
        y.append(side)
        g.append(f"S{s:02d}")
    return np.vstack(X), np.asarray(y), np.asarray(g)


def test_domains_differing_in_alpha_power_are_separable():
    X, y, g = _design(alpha_a=3.0, alpha_b=1.0)
    res = dom.separability(X, y, g, n_permutations=20)
    assert res["auc_mean"] > 0.9
    assert res["perm_p"] <= 1 / 21 + 1e-12
    assert res["null_q975"] < res["auc_mean"]


def test_identical_domains_are_not_separable():
    aucs = [dom.separability(*_design(1.0, 1.0, seed=s), n_permutations=0)["auc_mean"]
            for s in range(5)]
    assert abs(np.mean(aucs) - 0.5) < 0.12


def test_grouped_folds_never_put_one_subject_in_train_and_test():
    groups = np.repeat([f"S{i}" for i in range(30)], 3)
    # subjects on both sides: two records of one domain and one of the other
    y = np.tile([1, 1, 0], 30)
    y[: 3 * 10] = 1
    folds = dom.grouped_folds(y, groups)
    assert len(folds) == dom.N_SPLITS
    seen = []
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        seen.extend(test)
    assert sorted(seen) == list(range(len(y)))


def test_features_never_alter_the_input_array():
    rng = np.random.default_rng(3)
    signal = _record(rng, 2.0)
    signal[100:140, 2] = np.nan
    before = signal.copy()
    x, usable = dom.preprocess(signal, RATE)
    windows = dom.window_features(x, usable)
    x_before = x.copy()
    dom.record_features(windows)
    np.testing.assert_array_equal(np.isnan(signal), np.isnan(before))
    np.testing.assert_array_equal(np.nan_to_num(signal), np.nan_to_num(before))
    np.testing.assert_array_equal(x, x_before)


def test_the_band_pass_reaches_the_features_and_windows_are_non_overlapping():
    rng = np.random.default_rng(4)
    signal = _record(rng, 1.0, rate=500.0)
    x, usable = dom.preprocess(signal, 500.0)
    assert len(x) == pytest.approx(len(signal) * 128 / 500, abs=1)
    windows = dom.window_features(x, usable)
    size = int(dom.COMMON_RATE * dom.WINDOW_S)
    assert len(windows) == len(x) // size - dom.TRANSIENT_WINDOWS
    # A large 0.05 Hz drift sits below the 0.2 Hz edge: once band-passed it
    # leaves delta power where it was; unfiltered it would leak into delta.
    t = np.arange(int(500 * SECONDS)) / 500.0
    drift = np.column_stack([np.sin(2 * np.pi * 0.05 * t)] * N_CH) * 500 + signal
    xd, ud = dom.preprocess(drift, 500.0)
    delta = list(dom.BANDS).index("delta")
    fd = dom.record_features(dom.window_features(xd, ud))[:, delta]
    f0 = dom.record_features(windows)[:, delta]
    np.testing.assert_allclose(fd, f0, atol=0.3)


def test_a_flat_channel_carries_no_feature():
    rng = np.random.default_rng(5)
    signal = _record(rng, 1.0)
    signal[:, 1] = 4100.0
    x, usable = dom.preprocess(signal, RATE)
    feats = dom.record_features(dom.window_features(x, usable))
    assert not usable[1]
    assert np.isnan(feats[1]).all()
    assert np.isfinite(np.delete(feats, 1, axis=0)).all()


def test_the_permutation_keeps_the_grouped_design():
    rng = np.random.default_rng(6)
    groups = np.array(["A", "A", "B", "B", "C", "D", "E", "F"])
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    for _ in range(50):
        z = dom.permute_domain(y, groups, rng)
        for s in ("A", "B"):  # two-sided subjects stay one record per side
            assert sorted(z[groups == s]) == [0, 1]
        single = np.isin(groups, ["C", "D", "E", "F"])
        assert sorted(z[single]) == sorted(y[single])


def test_the_common_channels_are_named_identically_in_both_montages():
    common = dom.common_channels()
    assert common and all(c in EMOTIV_CHANNELS and c in CGX_CHANNELS for c in common)
    assert set(common) == set(EMOTIV_CHANNELS) & set(CGX_CHANNELS)


def test_the_auc_figure_carries_title_axis_labels_caption_and_chance_line():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from MIL_CREDA_EEG import audit_plots

    table = pd.DataFrame({"pair": ["a vs b", "c vs d"], "arm": ["x", "y"],
                          "auc_mean": [0.9, 0.5], "auc_sd": [0.05, 0.1],
                          "null_q025": [0.4, 0.38], "null_q975": [0.6, 0.62],
                          "perm_p": [0.005, 0.5]})
    fig = audit_plots.auc_bars(table, "T", "AUC", "par", "Mirá la línea.")
    ax = fig.axes[0]
    assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    assert any(t.get_text() for t in fig.texts)
    assert any(list(line.get_xdata()) == [0.5, 0.5] for line in ax.lines)
    plt.close(fig)
