"""The audit figures: title, both axis labels and a caption, on every kind."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from MIL_CREDA_EEG import audit_plots  # noqa: E402


def _check(fig) -> None:
    ax = fig.axes[0]
    assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()
    assert any(t.get_text() for t in fig.texts), "the caption is missing"
    plt.close(fig)


def test_every_figure_carries_title_axis_labels_and_caption(tmp_path):
    caption = "Mirá la barra más alta."
    figures = [
        audit_plots.bar({"a": 1, "b": 2}, "T", "x", "y", caption),
        audit_plots.bar({"a": 1, "b": 2}, "T", "x", "y", caption, horizontal=True),
        audit_plots.grouped_bar(pd.DataFrame({"yes": [1, 2], "no": [3, 4]},
                                             index=["u", "v"]), "T", "x", "y", caption),
        audit_plots.hist(np.arange(10.0), "T", "x", "y", caption, lines={"ref": 3}),
        audit_plots.hist(None, "T", "x", "y", caption,
                         groups={"yes": [1, 2, 3], "no": [2, 3, 4]}, log_x=True),
        audit_plots.heatmap(pd.DataFrame([[1, 2], [3, 4]]), "T", "x", "y", caption),
        audit_plots.box_by_group({"yes": [1, 2, 3], "no": [2, 3]}, "T", "x", "y", caption),
    ]
    path = audit_plots.save(figures[0], "probe", tmp_path)
    assert path.is_file() and path.stat().st_size > 0
    for fig in figures:
        _check(fig)
