"""The audit figures: no in-figure title, both axis labels and a caption, on every kind,
drawn at their printed size through `ps_figures` and saved as a PDF with a PNG preview."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from MIL_CREDA_EEG import audit_plots, ps_figures  # noqa: E402

CAPTION = "Mirá la barra más alta."


def _figures() -> list:
    return [
        audit_plots.bar({"a": 1, "b": 2}, "T", "x", "y", CAPTION),
        audit_plots.bar({"a": 1, "b": 2}, "T", "x", "y", CAPTION, horizontal=True),
        audit_plots.grouped_bar(pd.DataFrame({"yes": [1, 2], "no": [3, 4]},
                                             index=["u", "v"]), "T", "x", "y", CAPTION),
        audit_plots.hist(np.arange(10.0), "T", "x", "y", CAPTION, lines={"ref": 3}),
        audit_plots.hist(None, "T", "x", "y", CAPTION,
                         groups={"yes": [1, 2, 3], "no": [2, 3, 4]}, log_x=True),
        audit_plots.heatmap(pd.DataFrame([[1, 2], [3, 4]]), "T", "x", "y", CAPTION),
        audit_plots.box_by_group({"yes": [1, 2, 3], "no": [2, 3]}, "T", "x", "y", CAPTION),
    ]


def _check(fig) -> None:
    ax = fig.axes[0]
    assert ps_figures.figure_titles(fig) == []
    assert fig.get_label() == "T", "the title argument is kept as the figure's label"
    assert ax.get_xlabel() and ax.get_ylabel()
    assert any(t.get_text() == CAPTION for t in fig.texts), "the caption is missing"
    plt.close(fig)


def test_every_figure_carries_no_title_axis_labels_and_caption(tmp_path):
    figures = _figures()
    path = audit_plots.save(figures[0], "probe", tmp_path)
    assert path.is_file() and path.stat().st_size > 0
    for fig in figures:
        _check(fig)


def test_figures_print_at_the_declared_size_and_text_range():
    """ps_figures.apply_style: the printed width, and every text inside 0.70-0.90 x body."""
    width = ps_figures.rc_params(audit_plots.FIGURE_BODY_PT,
                                 audit_plots.FIGURE_WIDTH_FRACTION)["figure.figsize"][0]
    body = audit_plots.FIGURE_BODY_PT * ps_figures.TEX_PT_TO_PDF_PT
    low, high = body * ps_figures.MIN_RATIO, body * ps_figures.MAX_RATIO
    for fig in _figures():
        assert abs(fig.get_figwidth() - width) < 1e-6
        sizes = {t.get_fontsize() for t in fig.findobj(Text) if t.get_text()}
        assert sizes and all(low - 1e-9 <= s <= high + 1e-9 for s in sizes), sizes
        plt.close(fig)


def test_save_writes_the_pdf_and_a_png_preview_through_ps_figures(tmp_path):
    fig = audit_plots.bar({"a": 1}, "T", "x", "y", CAPTION)
    path = audit_plots.save(fig, "probe", tmp_path)
    assert path == tmp_path / "probe.pdf"
    assert path.read_bytes()[:5] == b"%PDF-"
    assert (tmp_path / "probe.png").is_file()
    plt.close(fig)


def test_the_same_arrays_reach_the_plot():
    """The module computes nothing: what it draws is exactly what it was given."""
    values = {"a": 3, "b": 1, "c": 2}
    fig = audit_plots.bar(values, "T", "x", "y", CAPTION)
    assert [p.get_height() for p in fig.axes[0].patches] == [3, 1, 2]
    plt.close(fig)

    table = pd.DataFrame({"yes": [1.0, 2.0], "no": [3.0, 4.0]}, index=["u", "v"])
    fig = audit_plots.grouped_bar(table, "T", "x", "y", CAPTION)
    assert [p.get_height() for p in fig.axes[0].patches] == [1.0, 2.0, 3.0, 4.0]
    plt.close(fig)

    data = np.array([0.5, 1.5, 1.7, 2.5, 3.0, np.nan])
    fig = audit_plots.hist(data, "T", "x", "y", CAPTION, bins=3)
    counts, _ = np.histogram(data[np.isfinite(data)], bins=audit_plots._edges(data, 3, False))
    assert [p.get_height() for p in fig.axes[0].patches] == list(counts)
    plt.close(fig)

    matrix = pd.DataFrame([[1.0, 2.0], [3.0, np.nan]])
    fig = audit_plots.heatmap(matrix, "T", "x", "y", CAPTION)
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array().data,
                                  matrix.to_numpy(dtype=float))
    plt.close(fig)

    groups = {"yes": [1.0, 2.0, 3.0], "no": [2.0, 5.0]}
    fig = audit_plots.box_by_group(groups, "T", "x", "y", CAPTION)
    drawn = [sorted(c.get_offsets()[:, 1]) for c in fig.axes[0].collections]
    assert drawn == [[1.0, 2.0, 3.0], [2.0, 5.0]]
    plt.close(fig)

    raw = (np.array([1.0, 2.0, 3.0]), np.array([0.3, 0.2, 0.1]))
    processed = (np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.1, 0.05]))
    fig = audit_plots.psd_before_after(raw, processed, "T", CAPTION)
    curves = [line for line in fig.axes[0].get_lines() if len(line.get_xdata()) > 2]
    for line, (freqs, psd) in zip(curves, (raw, processed)):
        np.testing.assert_array_equal(line.get_xdata(), freqs)
        np.testing.assert_array_equal(line.get_ydata(), psd)
    plt.close(fig)

    auc = pd.DataFrame({"pair": ["a vs b"], "arm": ["x"], "auc_mean": [0.8], "auc_sd": [0.1],
                        "null_q025": [0.4], "null_q975": [0.6], "perm_p": [0.01]})
    fig = audit_plots.auc_bars(auc, "T", "AUC", "par", CAPTION)
    bars = [p for p in fig.axes[0].patches if abs(p.get_height() - 0.55) < 1e-9]
    assert [b.get_width() for b in bars] == [0.8]
    plt.close(fig)


def test_panel_names_are_in_axes_labels_and_panel_data_is_unchanged():
    panels = {"A": {"yes": ([7.0, 8.0], [1.0, 2.0]), "no": ([12.0], [3.0])}}
    fig = audit_plots.scatter_panels(panels, "T", "x", "y", CAPTION)
    ax = fig.axes[0]
    assert ps_figures.figure_titles(fig) == []
    assert "A" in [t.get_text() for t in ax.texts]
    offsets = [c.get_offsets().tolist() for c in ax.collections]
    assert offsets == [[[7.0, 1.0], [8.0, 2.0]], [[12.0, 3.0]]]
    plt.close(fig)
