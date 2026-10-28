"""Experiments plan v05, Sec. 2.7: the before/after spectrum figure of preprocessing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from MIL_CREDA_EEG import audit_plots  # noqa: E402


def test_the_spectrum_figure_shows_both_curves_the_band_edge_and_the_new_nyquist():
    raw = (np.linspace(0, 250, 501), np.exp(-np.linspace(0, 5, 501)) + 1e-3)
    processed = (np.linspace(0, 64, 129), np.exp(-np.linspace(0, 2, 129)))
    fig = audit_plots.psd_before_after(raw, processed, "T", "Mirá el corte.",
                                       lines={"45 Hz": 45.0, "Nyquist 64 Hz": 64.0})
    ax = fig.axes[0]
    assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    assert any(t.get_text() for t in fig.texts), "the caption is missing"
    assert ax.get_yscale() == "log"
    curves = [line for line in ax.get_lines() if len(line.get_xdata()) > 2]
    assert len(curves) == 2
    verticals = sorted(float(line.get_xdata()[0]) for line in ax.get_lines()
                       if len(line.get_xdata()) == 2)
    assert verticals == [45.0, 64.0]
    assert len(ax.get_legend().get_texts()) == 2
    plt.close(fig)
