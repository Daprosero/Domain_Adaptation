"""Experiments plan v04, Sec. 2.5: is the Emotiv contact scale shared by both models?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from MIL_CREDA_EEG import audit_contact

RATE = 128.0


def _calibration(seed: int, shift: dict[str, int], n_records: int = 25) -> pd.DataFrame:
    """Record-level rows: the proxy depends on the TRUE level; a model reports it through `shift`."""
    rng = np.random.default_rng(seed)
    rows = []
    for device, offset in shift.items():
        for r in range(n_records):
            for true_level in (0, 2, 4):
                reported = true_level + offset
                if reported not in (0, 2, 4):
                    continue
                rows.append({"record_id": f"{device}-{r}", "device": device,
                             "level": reported,
                             "line_ratio": 10 ** (-1 - 0.5 * true_level + rng.normal(0, 0.2))})
    return pd.DataFrame(rows)


def test_two_models_with_the_same_proxy_per_level_read_as_the_same_scale():
    table = _calibration(0, {"A": 0, "B": 0})
    before = table.copy()
    result = audit_contact.scale_comparison(table, "line_ratio")
    assert result["verdict"] == "same"
    assert result["interaction_p"] >= 0.05
    assert all(row["p_holm"] >= 0.05 for row in result["per_level"])
    pd.testing.assert_frame_equal(table, before)


def test_a_model_reporting_a_rescaled_level_reads_as_a_different_scale():
    # B reports level 4 where A would report 2, and 2 where A would report 0.
    table = _calibration(1, {"A": 0, "B": 2})
    result = audit_contact.scale_comparison(table, "line_ratio")
    assert result["verdict"] == "different"
    assert any(row["p_holm"] is not None and row["p_holm"] < 0.05 for row in result["per_level"])


def test_value_counts_reads_distinct_levels_and_ignores_missing():
    values = np.array([4.0, 4.0, 2.0, np.nan, 0.0, 83.333336, 83.33334])
    counts = audit_contact.value_counts(values)
    assert counts == {0.0: 1, 2.0: 1, 4.0: 2, 83.3333: 2}
    assert np.isnan(values[3])


def _signal(seconds: float = 40.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    x = 4000 + rng.normal(size=(len(t), 2)) + 3 * np.sin(2 * np.pi * 10 * t)[:, None]
    x[:, 0] += 20 * np.sin(2 * np.pi * 50 * t)
    return x


def test_mains_on_the_poor_contact_channel_raises_its_proxy_and_nothing_is_altered():
    x = _signal()
    cq = np.column_stack([np.zeros(len(x)), np.full(len(x), 4.0)])
    x_before, cq_before = x.copy(), cq.copy()
    proxies = audit_contact.window_proxies(x, RATE)
    levels, pure = audit_contact.window_levels(cq, RATE)
    assert proxies["line_ratio"].shape == (10, 2) and levels.shape == (10, 2)
    assert pure.all()
    rows = {r["level"]: r for r in audit_contact.calibration_rows(proxies, levels, pure)}
    assert rows[0.0]["line_ratio_median"] > 10 * rows[4.0]["line_ratio_median"]
    assert rows[0.0]["n_windows"] == 10 and rows[4.0]["n_windows"] == 10
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(cq, cq_before)


def test_a_window_whose_level_changes_is_not_pure_and_stays_out_of_calibration():
    x = _signal(seconds=8.0)
    cq = np.full((len(x), 2), 4.0)
    cq[10, 1] = 2.0
    proxies = audit_contact.window_proxies(x, RATE)
    levels, pure = audit_contact.window_levels(cq, RATE)
    assert pure.tolist() == [[True, False], [True, True]]
    rows = {r["level"]: r for r in audit_contact.calibration_rows(proxies, levels, pure)}
    assert rows[4.0]["n_windows"] == 3 and 2.0 not in rows


def test_overall_is_cross_tabulated_against_the_channels_at_the_top_level():
    cq = np.array([[4, 4], [4, 2], [4, 2], [0, 0]], dtype=float)
    overall = np.array([100.0, 50.0, 50.0, 0.0])
    table = audit_contact.overall_vs_channels(cq, overall)
    assert table == {2: {100.0: 1}, 1: {50.0: 2}, 0: {0.0: 1}}


def test_the_paired_box_figure_carries_title_axis_labels_caption_and_legend():
    import matplotlib
    matplotlib.use("Agg")
    from MIL_CREDA_EEG import audit_plots
    panels = {"Robots": {"CQ = 0": {"EPOCX": [1.0, 2.0], "EPOCPLUS": [1.5]},
                         "CQ = 4": {"EPOCX": [0.1], "EPOCPLUS": []}}}
    fig = audit_plots.paired_box_panels(panels, "t", "x", "y", "caption")
    ax = fig.axes[0]
    assert fig._suptitle.get_text() == "t" and ax.get_xlabel() and ax.get_ylabel()
    assert any(t.get_text() for t in fig.texts)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["EPOCX", "EPOCPLUS"]
