"""Plain figures for the BALLADEER audit: bars, histograms, heatmaps, boxes.

Experiments plan v55, Secs. 2.1-2.10 and 4.2. Every audit number is shown in a
figure a reader can take in at a glance: one message per figure, both axis
labels, and a one- or two-line caption under the plot saying what to look at.
The text on the figures is in Spanish because the notebooks are; this module
computes nothing -- it draws what the audit modules measured.

**Through `ps_figures`** (code review should-fix). Every figure is drawn at its
printed size by `ps_figures.apply_style` -- 0.8 of the text width of a 10 pt
body, the same as `results.py`'s figures -- so every text prints between 0.70
and 0.90 x the body; a `figsize` or `panel_size` argument keeps only its
aspect. No figure carries an in-figure title: the `title` argument is kept as
the figure's label (`fig.get_label()`), for the caption that carries it, and a
panel's name is written inside its axes. `save` writes the vector PDF through
`ps_figures.save_figure` with a PNG preview beside it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MIL_CREDA_EEG import ps_figures

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v55.md",
    "sections": ["2.10"],
    "experiments": ["2.10"],
    "invariants": [
        "every_figure_carries_no_title_axis_labels_and_caption",
        "figures_print_at_the_declared_size_and_text_range",
        "save_writes_the_pdf_and_a_png_preview_through_ps_figures",
        "the_same_arrays_reach_the_plot",
        "panel_names_are_in_axes_labels_and_panel_data_is_unchanged",
        "panel_figures_carry_title_axis_labels_and_caption",
        "the_auc_figure_carries_title_axis_labels_caption_and_chance_line",
        "the_spectrum_figure_shows_both_curves_the_band_edge_and_the_new_nyquist",
    ],
}

#: The printed size of `results.py`'s figures: 0.8 of the text width of a 10 pt body.
FIGURE_BODY_PT = 10
FIGURE_WIDTH_FRACTION = 0.8
#: Average glyph width as a fraction of the font size, to wrap a caption to the width.
_GLYPH_WIDTH = 0.5

PALETTE = {"yes": "#c0392b", "no": "#2471a3", "undetermined": "#7f8c8d",
           "M": "#5d6d7e", "F": "#d68910", "CGX": "#117a65", "EMOTIV": "#6c3483",
           "EPOCX": "#6c3483", "EPOCPLUS": "#a569bd"}
LABEL_ES = {"yes": "TDAH (sí)", "no": "control (no)", "undetermined": "indeterminado"}


def _subplots(figsize, nrows: int = 1, ncols: int = 1, **kwargs):
    """A figure at its printed size (`ps_figures.apply_style`), `figsize` giving the aspect."""
    ps_figures.apply_style(body_pt=FIGURE_BODY_PT, printed_width_fraction=FIGURE_WIDTH_FRACTION,
                           aspect=figsize[1] / figsize[0])
    return plt.subplots(nrows, ncols, **kwargs)


def _small() -> float:
    """The small text role `apply_style` set (ticks, legend, values written on marks)."""
    return matplotlib.rcParams["legend.fontsize"]


def _caption(fig, caption: str | None) -> None:
    if caption:
        size = _small()
        width = int(fig.get_figwidth() * 72 / (_GLYPH_WIDTH * size))
        fig.text(0.01, -0.01, textwrap.fill(caption, width), ha="left", va="top",
                 fontsize=size, style="italic", color="#333333")


def _finish(fig, ax, title: str, xlabel: str, ylabel: str, caption: str | None):
    fig.set_label(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    _caption(fig, caption)
    fig.tight_layout()
    return fig


def _panel_name(ax, panel) -> None:
    """A panel's name inside its axes: a label, not a title."""
    ax.text(0.02, 0.98, str(panel), transform=ax.transAxes, ha="left", va="top",
            fontsize=_small())


def bar(values: pd.Series | dict, title: str, xlabel: str, ylabel: str,
        caption: str | None = None, colors: list[str] | None = None,
        horizontal: bool = False, figsize=(8, 4.5)):
    """One bar per category, with its value written on it."""
    values = pd.Series(values)
    fig, ax = _subplots(figsize)
    positions = np.arange(len(values))
    if horizontal:
        bars = ax.barh(positions, values.values, color=colors)
        ax.set_yticks(positions, [str(i) for i in values.index])
        ax.invert_yaxis()
        for b, v in zip(bars, values.values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {_fmt(v)}",
                    va="center", fontsize=_small())
    else:
        bars = ax.bar(positions, values.values, color=colors)
        ax.set_xticks(positions, [str(i) for i in values.index],
                      rotation=30 if len(values) > 5 else 0, ha="right"
                      if len(values) > 5 else "center")
        for b, v in zip(bars, values.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _fmt(v),
                    ha="center", va="bottom", fontsize=_small())
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def grouped_bar(table: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                caption: str | None = None, figsize=(9, 4.5)):
    """Rows are categories on the x axis, columns are the bars of each group."""
    fig, ax = _subplots(figsize)
    n = len(table.columns)
    width = 0.8 / max(n, 1)
    x = np.arange(len(table.index))
    for k, column in enumerate(table.columns):
        bars = ax.bar(x + (k - (n - 1) / 2) * width, table[column].values, width,
                      label=LABEL_ES.get(str(column), str(column)),
                      color=PALETTE.get(str(column)))
        for b, v in zip(bars, table[column].values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _fmt(v),
                    ha="center", va="bottom", fontsize=_small())
    ax.set_xticks(x, [str(i) for i in table.index],
                  rotation=30 if len(x) > 4 else 0, ha="right" if len(x) > 4 else "center")
    ax.legend(frameon=False)
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def hist(values, title: str, xlabel: str, ylabel: str = "número de registros",
         caption: str | None = None, bins=30, lines: dict | None = None,
         groups: dict | None = None, figsize=(8, 4.5), log_x: bool = False):
    """Histogram; `groups` overlays one per group, `lines` marks reference values."""
    fig, ax = _subplots(figsize)
    if groups:
        everything = np.concatenate([np.asarray(v, dtype=float) for v in groups.values()])
        edges = _edges(everything, bins, log_x)
        for name, v in groups.items():
            ax.hist(np.asarray(v, dtype=float), bins=edges, alpha=0.55,
                    label=LABEL_ES.get(str(name), str(name)), color=PALETTE.get(str(name)))
        ax.legend(frameon=False)
    else:
        v = np.asarray(values, dtype=float)
        ax.hist(v[np.isfinite(v)], bins=_edges(v, bins, log_x), color="#5499c7")
    if log_x:
        ax.set_xscale("log")
    for label, x in (lines or {}).items():
        ax.axvline(x, color="#c0392b", linestyle="--", linewidth=1.5)
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {label}", color="#c0392b", fontsize=_small(),
                va="top")
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def heatmap(table: pd.DataFrame, title: str, xlabel: str, ylabel: str,
            caption: str | None = None, fmt: str = "{:.0f}", cmap: str = "Blues",
            figsize=(8, 6), vmin=None, vmax=None):
    """A matrix with every cell's value written in it."""
    fig, ax = _subplots(figsize)
    data = table.to_numpy(dtype=float)
    image = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    low, high = image.get_clim()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                dark = high > low and (data[i, j] - low) / (high - low) > 0.6
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        fontsize=_small(), color="white" if dark else "black")
    fig.colorbar(image, ax=ax, fraction=0.04)
    _finish(fig, ax, title, xlabel, ylabel, caption)
    ax.grid(False)
    return fig


def box_by_group(groups: dict, title: str, xlabel: str, ylabel: str,
                 caption: str | None = None, figsize=(7, 4.5), log_y: bool = False):
    """One box per group, with every point drawn on top of it."""
    fig, ax = _subplots(figsize)
    names = list(groups)
    data = [np.asarray(groups[n], dtype=float) for n in names]
    data = [d[np.isfinite(d)] for d in data]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(names) + 1),
                  [f"{LABEL_ES.get(str(n), str(n))}\n(n={len(d)})" for n, d in zip(names, data)])
    rng = np.random.default_rng(0)
    for k, (n, d) in enumerate(zip(names, data), start=1):
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(d)), d, s=12, alpha=0.6,
                   color=PALETTE.get(str(n), "#555555"))
    if log_y:
        ax.set_yscale("log")
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def _panel_grid(n: int, ncols: int, title: str, panel_size):
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = _subplots((panel_size[0] * ncols, panel_size[1] * nrows), nrows, ncols,
                          squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.set_label(title)
    return fig, list(axes.flat[:n])


def _finish_panels(fig, caption: str | None):
    fig.tight_layout()
    _caption(fig, caption)
    return fig


def box_panels(panels: dict, title: str, xlabel: str, ylabel: str,
               caption: str | None = None, log_y: bool = False, ncols: int = 3,
               panel_size=(4.2, 3.8)):
    """One small box-by-group plot per panel (a cohort, a task), every point drawn."""
    fig, axes = _panel_grid(len(panels), ncols, title, panel_size)
    rng = np.random.default_rng(0)
    for ax, (panel, groups) in zip(axes, panels.items()):
        names = list(groups)
        data = [np.asarray(groups[n], dtype=float) for n in names]
        data = [d[np.isfinite(d)] for d in data]
        ax.boxplot([d if len(d) else [np.nan] for d in data], showfliers=False)
        ax.set_xticks(range(1, len(names) + 1),
                      [f"{LABEL_ES.get(str(n), str(n))}\n(n={len(d)})" for n, d in zip(names, data)],
                      fontsize=_small())
        for k, (n, d) in enumerate(zip(names, data), start=1):
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(d)), d, s=10, alpha=0.6,
                       color=PALETTE.get(str(n), "#555555"))
        if log_y:
            ax.set_yscale("log")
        _panel_name(ax, panel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    return _finish_panels(fig, caption)


def scatter_panels(panels: dict, title: str, xlabel: str, ylabel: str,
                   caption: str | None = None, log_y: bool = False, ncols: int = 3,
                   panel_size=(4.2, 3.8)):
    """Per panel, one point cloud per group: `{panel: {group: (x, y)}}`."""
    fig, axes = _panel_grid(len(panels), ncols, title, panel_size)
    for ax, (panel, groups) in zip(axes, panels.items()):
        for name, (x, y) in groups.items():
            ax.scatter(np.asarray(x, dtype=float), np.asarray(y, dtype=float), s=12,
                       alpha=0.65, color=PALETTE.get(str(name), "#555555"),
                       label=LABEL_ES.get(str(name), str(name)))
        if log_y:
            ax.set_yscale("log")
        _panel_name(ax, panel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=_small())
    return _finish_panels(fig, caption)


def paired_box_panels(panels: dict, title: str, xlabel: str, ylabel: str,
                      caption: str | None = None, log_y: bool = False, ncols: int = 2,
                      panel_size=(6.0, 4.0)):
    """Per panel, side-by-side boxes per category, one colour per group:
    `{panel: {category: {group: values}}}`."""
    fig, axes = _panel_grid(len(panels), ncols, title, panel_size)
    rng = np.random.default_rng(0)
    for ax, (panel, categories) in zip(axes, panels.items()):
        groups = list(dict.fromkeys(g for c in categories.values() for g in c))
        width = 0.8 / max(len(groups), 1)
        for k, group in enumerate(groups):
            positions, data = [], []
            for i, cat in enumerate(categories):
                d = np.asarray(categories[cat].get(group, []), dtype=float)
                d = d[np.isfinite(d)]
                positions.append(i + (k - (len(groups) - 1) / 2) * width)
                data.append(d)
            color = PALETTE.get(str(group), "#555555")
            ax.boxplot([d if len(d) else [np.nan] for d in data], positions=positions,
                       widths=width * 0.9, showfliers=False,
                       medianprops={"color": color, "linewidth": 2})
            for p, d in zip(positions, data):
                ax.scatter(p + rng.uniform(-width / 4, width / 4, len(d)), d, s=9,
                           alpha=0.6, color=color)
            ax.scatter([], [], color=color, label=LABEL_ES.get(str(group), str(group)))
        ax.set_xticks(range(len(categories)), [str(c) for c in categories])
        if log_y:
            ax.set_yscale("log")
        _panel_name(ax, panel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=_small())
    return _finish_panels(fig, caption)


def auc_bars(table: pd.DataFrame, title: str, xlabel: str, ylabel: str,
             caption: str | None = None, figsize=(11, 5.5)):
    """One horizontal bar per pair: mean AUC with its fold sd, the 0.5 chance
    line, and the permutation null's 2.5-97.5 % band drawn behind each bar.

    `table` needs `pair`, `auc_mean`, `auc_sd`, `null_q025`, `null_q975`,
    `perm_p` and, for colour, `arm`.
    """
    fig, ax = _subplots(figsize)
    positions = np.arange(len(table))
    arms = list(dict.fromkeys(table["arm"]))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(arms.index(a) % 10) for a in table["arm"]]
    for k, row in enumerate(table.itertuples()):
        ax.add_patch(plt.Rectangle((row.null_q025, k - 0.42), row.null_q975 - row.null_q025,
                                   0.84, color="#bbbbbb", alpha=0.6, zorder=0))
    ax.barh(positions, table["auc_mean"], xerr=table["auc_sd"], color=colors,
            height=0.55, capsize=3, zorder=2)
    for k, row in enumerate(table.itertuples()):
        ax.text(min(row.auc_mean + row.auc_sd, 1.0) + 0.01, k,
                f"{row.auc_mean:.2f} (p={row.perm_p:.2g})", va="center", fontsize=_small())
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, zorder=3)
    ax.set_yticks(positions, list(table["pair"]), fontsize=_small())
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i % 10)) for i in range(len(arms))]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=0.6))
    ax.legend(handles, arms + ["nulo por permutación (2,5–97,5 %)"], fontsize=_small(),
              frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig = _finish(fig, ax, title, xlabel, ylabel, caption)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    return fig


def psd_before_after(raw: tuple, processed: tuple, title: str,
                     caption: str | None = None, lines: dict | None = None,
                     xlabel: str = "frecuencia (Hz)",
                     ylabel: str = "densidad espectral (µV²/Hz, escala log)",
                     labels: tuple[str, str] = ("crudo", "preprocesado"),
                     figsize=(9, 4.5)):
    """Two spectra on a log axis, `(freqs, psd)` each, with reference frequencies.

    Experiments plan v05, Sec. 2.7: the raw record against the processed one,
    so the 45 Hz edge and the 128 Hz rate (64 Hz Nyquist) are visible.
    """
    fig, ax = _subplots(figsize)
    for (freqs, psd), label, color in zip((raw, processed), labels, ("#7f8c8d", "#117a65")):
        freqs, psd = np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)
        keep = psd > 0
        ax.plot(freqs[keep], psd[keep], label=label, color=color, linewidth=1.3)
    ax.set_yscale("log")
    for k, (label, x) in enumerate((lines or {}).items()):
        ax.axvline(x, color="#c0392b", linestyle="--", linewidth=1.2)
        ax.text(x, 0.98 - 0.07 * (k % 2), f" {label}", color="#c0392b", fontsize=_small(),
                va="top", transform=ax.get_xaxis_transform())
    ax.legend(frameon=False)
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def save(fig, name: str, out_dir: str | Path) -> Path:
    """Write the figure as `<name>.pdf` with `<name>.png` beside it; return the PDF path."""
    return ps_figures.save_figure(fig, Path(out_dir) / f"{name}.pdf", png_preview=True)


def _edges(values, bins, log_x):
    if not np.isscalar(bins):
        return np.asarray(bins, dtype=float)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log_x:
        v = v[v > 0]
        if not len(v):
            return bins
        lo, hi = v.min(), v.max()
        if lo == hi:
            lo, hi = lo / 2, hi * 2
        return np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    if not len(v):
        return bins
    lo, hi = v.min(), v.max()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, bins + 1)


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v)}"
    return f"{v:.2f}"
