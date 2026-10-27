"""Experiments plan v03, Sec. 2.8 (ocular artifact, measurement only), on synthetic signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from MIL_CREDA_EEG import audit_ocular

RATE = 128.0
CHANNELS = ["AF3", "F7", "F3", "O1", "O2", "F8", "AF4"]
FRONT = ["AF3", "AF4", "F7", "F8"]
BACK = ["O1", "O2"]


def _eeg(seconds: float = 60.0, seed: int = 0) -> np.ndarray:
    """The same broadband background on every channel, around Emotiv's DC level."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    base = rng.normal(scale=5.0, size=(len(t), len(CHANNELS)))
    return 4200 + base + 4 * np.sin(2 * np.pi * 10 * t)[:, None]


def _blinks(n_samples: int, every_s: float = 3.0, amplitude: float = 150.0) -> np.ndarray:
    """A slow (~0.3 s wide) positive deflection every few seconds: a blink."""
    t = np.arange(n_samples) / RATE
    out = np.zeros(n_samples)
    for centre in np.arange(1.5, t[-1], every_s):
        out += amplitude * np.exp(-0.5 * ((t - centre) / 0.1) ** 2)
    return out


def _with(x: np.ndarray, names: list[str], wave: np.ndarray) -> np.ndarray:
    y = x.copy()
    for name in names:
        y[:, CHANNELS.index(name)] += wave
    return y


def _measure(x: np.ndarray) -> dict:
    return audit_ocular.window_measures(x, CHANNELS, RATE, FRONT, BACK)


def test_a_slow_frontal_blink_raises_the_frontal_posterior_ratio():
    x = _eeg()
    clean = audit_ocular.summarize(_measure(x))
    blinked = audit_ocular.summarize(_measure(_with(x, FRONT, _blinks(len(x)))))
    assert blinked["low_ratio_median"] > 10 * clean["low_ratio_median"]
    assert blinked["frontal_ptp_median"] > 3 * clean["frontal_ptp_median"]
    assert 0.5 < clean["low_ratio_median"] < 2.0


def test_posterior_only_slow_activity_does_not_raise_the_ratio():
    x = _eeg()
    clean = audit_ocular.summarize(_measure(x))
    back = audit_ocular.summarize(_measure(_with(x, BACK, _blinks(len(x)))))
    assert back["low_ratio_median"] < clean["low_ratio_median"]
    assert back["low_ratio_p90"] < clean["low_ratio_p90"]
    assert abs(back["frontal_ptp_median"] - clean["frontal_ptp_median"]) < 1e-9


def test_measures_never_alter_the_input_array():
    x = _with(_eeg(), FRONT, _blinks(len(_eeg())))
    x[100:140, 0] = np.nan
    x[500, 3] = 0.0
    before = x.copy()
    _measure(x)
    audit_ocular.summarize(_measure(x))
    np.testing.assert_array_equal(x, before)


def test_every_window_is_measured_and_none_is_dropped():
    x = _eeg(seconds=41.0)
    x[600:700, CHANNELS.index("AF3")] = 0.0          # Emotiv's lower rail
    m = _measure(x)
    assert len(m["frontal_ptp"]) == len(m["low_ratio"]) == 10   # floor(41 / 4)
    assert m["rail_windows"].sum() == 1
    summary = audit_ocular.summarize(m)
    assert summary["n_windows"] == 10 and summary["rail_window_fraction"] == 0.1


def test_a_flat_channel_carries_no_measure_and_a_flat_posterior_set_leaves_the_ratio_undefined():
    x = _eeg()
    for name in BACK:
        x[:, CHANNELS.index(name)] = -833333.3125      # CGX's rail, constant
    m = _measure(x)
    assert np.isnan(m["low_ratio"]).all()
    assert np.isfinite(m["frontal_ptp"]).all()
    one_flat = _eeg()
    one_flat[:, CHANNELS.index("O1")] = 4100.0
    ratio = _measure(one_flat)["low_ratio"]
    assert np.isfinite(ratio).all() and np.median(ratio) < 2.0


def test_the_channels_used_are_named_and_absent_ones_are_reported():
    used = audit_ocular.channels_used(["Fp1", "Fp2", "AF7", "O1", "Oz", "Cz"], "CGX")
    assert used["frontal"] == ["Fp1", "Fp2", "AF7"]
    assert used["posterior"] == ["O1", "Oz"]
    assert used["missing"] == ["Fpz", "AF8", "O2"]


def test_the_label_cross_keeps_undetermined_apart_and_tests_yes_against_no():
    rng = np.random.default_rng(1)
    subjects = [f"UB{k:04d}" for k in range(40)]
    label = ["yes"] * 15 + ["no"] * 20 + ["undetermined"] * 5
    demo = pd.DataFrame({"subject": subjects, "label": label,
                         "sex": ["M", "F"] * 20, "age": rng.integers(7, 13, 40),
                         "group": 1})
    value = np.where(np.array(label) == "yes", 3.0, 1.0) + rng.normal(scale=0.1, size=40)
    value[-5:] = 100.0                                # undetermined, far away
    row = audit_ocular.cross_measure(pd.Series(value, index=subjects), demo)
    assert row["n_yes"] == 15 and row["n_no"] == 20 and row["n_undetermined"] == 5
    assert row["median_undetermined"] > 50
    assert row["label_mannwhitney_p"] < 1e-4
    assert row["median_yes"] > row["median_no"]
    assert 0.0 <= row["sex_median_split_fisher_p"] <= 1.0
    assert row["age_spearman_rho_within_yes"] is not None


def test_an_age_effect_is_not_read_as_a_label_effect_after_adjustment():
    rng = np.random.default_rng(2)
    n = 120
    age = rng.uniform(7, 17, n)
    label = np.where(age + rng.normal(scale=1.5, size=n) < 12, "yes", "no")
    subjects = [f"UB{k:04d}" for k in range(n)]
    demo = pd.DataFrame({"subject": subjects, "label": label, "sex": "M",
                         "age": age, "group": 1})
    value = pd.Series(-age + rng.normal(scale=0.5, size=n), index=subjects)
    row = audit_ocular.cross_measure(value, demo)
    assert row["label_mannwhitney_p"] < 1e-4
    assert row["label_p_age_adjusted"] > 0.05
    shifted = value + np.where(label == "yes", 3.0, 0.0)
    assert audit_ocular.cross_measure(shifted, demo)["label_p_age_adjusted"] < 1e-4


def test_panel_figures_carry_title_axis_labels_and_caption():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from MIL_CREDA_EEG import audit_plots

    caption = "Mirá si las cajas se separan."
    boxes = audit_plots.box_panels(
        {"A": {"yes": [1, 2, 3], "no": [2, 3]}, "B": {"yes": [1.0], "no": []}},
        "T", "x", "y", caption, log_y=True)
    dots = audit_plots.scatter_panels(
        {"A": {"yes": ([7, 8], [1, 2]), "no": ([12], [3])}}, "T", "x", "y", caption)
    for fig, n_panels in ((boxes, 2), (dots, 1)):
        assert fig._suptitle is not None and fig._suptitle.get_text() == "T"
        drawn = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
        assert len(drawn) == n_panels
        for ax in drawn:
            assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
        assert any(t.get_text() == caption for t in fig.texts)
        plt.close(fig)


def test_holm_adjusts_upward_and_keeps_order():
    p = pd.Series([0.01, 0.04, 0.03, None])
    adjusted = audit_ocular.holm(p)
    assert adjusted.iloc[0] == 0.03 and adjusted.iloc[2] == 0.06
    assert adjusted.iloc[1] == 0.06 and pd.isna(adjusted.iloc[3])
