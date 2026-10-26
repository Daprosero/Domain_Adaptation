"""Plain figures for the BALLADEER audit: bars, histograms, heatmaps, boxes.

Experiments plan v02, Secs. 2.1-2.6 and 2.10. Every audit number is shown in a
figure a reader can take in at a glance: one message per figure, a title, both
axis labels, and a one- or two-line caption under the plot saying what to look
at. The text on the figures is in Spanish because the notebooks are; this
module computes nothing -- it draws what the audit modules measured.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v02.md",
    "sections": ["2.10"],
    "experiments": ["2.10"],
    "invariants": [
        "every_figure_carries_title_axis_labels_and_caption",
    ],
}

PALETTE = {"yes": "#c0392b", "no": "#2471a3", "undetermined": "#7f8c8d",
           "M": "#5d6d7e", "F": "#d68910", "CGX": "#117a65", "EMOTIV": "#6c3483",
           "EPOCX": "#6c3483", "EPOCPLUS": "#a569bd"}
LABEL_ES = {"yes": "TDAH (sí)", "no": "control (no)", "undetermined": "indeterminado"}


def _finish(fig, ax, title: str, xlabel: str, ylabel: str, caption: str | None):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if caption:
        fig.text(0.01, -0.01, textwrap.fill(caption, 110), ha="left", va="top",
                 fontsize=9, style="italic", color="#333333")
    fig.tight_layout()
    return fig


def bar(values: pd.Series | dict, title: str, xlabel: str, ylabel: str,
        caption: str | None = None, colors: list[str] | None = None,
        horizontal: bool = False, figsize=(8, 4.5)):
    """One bar per category, with its value written on it."""
    values = pd.Series(values)
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(values))
    if horizontal:
        bars = ax.barh(positions, values.values, color=colors)
        ax.set_yticks(positions, [str(i) for i in values.index])
        ax.invert_yaxis()
        for b, v in zip(bars, values.values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {_fmt(v)}",
                    va="center", fontsize=9)
    else:
        bars = ax.bar(positions, values.values, color=colors)
        ax.set_xticks(positions, [str(i) for i in values.index],
                      rotation=30 if len(values) > 5 else 0, ha="right"
                      if len(values) > 5 else "center")
        for b, v in zip(bars, values.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _fmt(v),
                    ha="center", va="bottom", fontsize=9)
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def grouped_bar(table: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                caption: str | None = None, figsize=(9, 4.5)):
    """Rows are categories on the x axis, columns are the bars of each group."""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(table.columns)
    width = 0.8 / max(n, 1)
    x = np.arange(len(table.index))
    for k, column in enumerate(table.columns):
        bars = ax.bar(x + (k - (n - 1) / 2) * width, table[column].values, width,
                      label=LABEL_ES.get(str(column), str(column)),
                      color=PALETTE.get(str(column)))
        for b, v in zip(bars, table[column].values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _fmt(v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(i) for i in table.index],
                  rotation=30 if len(x) > 4 else 0, ha="right" if len(x) > 4 else "center")
    ax.legend(frameon=False)
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def hist(values, title: str, xlabel: str, ylabel: str = "número de registros",
         caption: str | None = None, bins=30, lines: dict | None = None,
         groups: dict | None = None, figsize=(8, 4.5), log_x: bool = False):
    """Histogram; `groups` overlays one per group, `lines` marks reference values."""
    fig, ax = plt.subplots(figsize=figsize)
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
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {label}", color="#c0392b", fontsize=9,
                va="top")
    return _finish(fig, ax, title, xlabel, ylabel, caption)


def heatmap(table: pd.DataFrame, title: str, xlabel: str, ylabel: str,
            caption: str | None = None, fmt: str = "{:.0f}", cmap: str = "Blues",
            figsize=(8, 6), vmin=None, vmax=None):
    """A matrix with every cell's value written in it."""
    fig, ax = plt.subplots(figsize=figsize)
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
                        fontsize=8, color="white" if dark else "black")
    fig.colorbar(image, ax=ax, fraction=0.04)
    _finish(fig, ax, title, xlabel, ylabel, caption)
    ax.grid(False)
    return fig


def box_by_group(groups: dict, title: str, xlabel: str, ylabel: str,
                 caption: str | None = None, figsize=(7, 4.5), log_y: bool = False):
    """One box per group, with every point drawn on top of it."""
    fig, ax = plt.subplots(figsize=figsize)
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


def save(fig, name: str, out_dir: str | Path) -> Path:
    """Write the figure as PNG and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


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
