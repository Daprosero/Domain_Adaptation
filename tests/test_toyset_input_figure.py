"""Each toyset tutorial draws its input data as ONE figure with three panels.

One panel per family (lunas, toroide, rollo), the source and the target of that
family overlaid on the same 3D axes with distinct markers: the source coloured
by its class, the target in one neutral grey because training never sees its
labels. The points are the generated input-space data, never a latent.

These tests run the notebook's own code cells, in order, only up to the cell
that builds `INPUT_FIGURE` -- no training -- and interrogate the real matplotlib
figure against an independent call to the same data generator.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
FAMILIES = ("lunas", "toroide", "rollo")
NOTEBOOKS = {
    "CREDA": REPOSITORY / "CREDA" / "Notebooks" / "Toyset.ipynb",
    "MIL-CREDA": REPOSITORY / "MIL-CREDA" / "Notebooks" / "Toyset.ipynb",
}
#: What the figure cell must leave untouched, per notebook.
CONFIGURATION = {"CREDA": ("CONFIG", "SEED"),
                 "MIL-CREDA": ("ESCALA", "HIPER", "SEMILLA", "FAMILIAS")}


def _code_cells(method: str) -> list[dict]:
    document = json.loads(NOTEBOOKS[method].read_text(encoding="utf-8"))
    return [cell for cell in document["cells"] if cell["cell_type"] == "code"]


def _index(cells: list[dict], marker: str) -> int:
    for index, cell in enumerate(cells):
        if marker in "".join(cell["source"]):
            return index
    raise AssertionError(f"no code cell contains {marker!r}")


def _execute_through_input_figure(method: str) -> dict:
    """Run the code cells up to the input-figure cell, recording its effects."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = _code_cells(method)
    last = _index(cells, "INPUT_FIGURE =")
    namespace: dict = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(NOTEBOOKS[method].parent)
        patch.delenv("FORGE_CLONE_ROOT", raising=False)
        patch.delenv("FORGE_CLONE_COMMIT", raising=False)
        warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
        plt.close("all")
        for index, cell in enumerate(cells[:last + 1]):
            if index == last:
                namespace["_before"] = dict(
                    numpy=np.random.get_state(),
                    torch=torch.get_rng_state().clone(),
                    figures=set(plt.get_fignums()),
                    configuration={k: copy.deepcopy(namespace[k])
                                   for k in CONFIGURATION[method]})
            source = "".join(cell["source"])
            exec(compile(source, f"{method}/Toyset.ipynb[code {index}]", "exec"),
                 namespace)
        namespace["_after"] = dict(
            numpy=np.random.get_state(),
            torch=torch.get_rng_state().clone(),
            figures=set(plt.get_fignums()))
    namespace["_cells"] = cells
    namespace["_last"] = last
    return namespace


@pytest.fixture(scope="module", params=sorted(NOTEBOOKS))
def executed(request) -> tuple[str, dict]:
    return request.param, _execute_through_input_figure(request.param)


def _generated(method: str, namespace: dict) -> dict:
    """The same families, drawn again by the existing generators."""
    if method == "CREDA":
        from CREDA.toy_experiment import FAMILIAS, N_POR_DOMINIO, datos_semilla

        drawn = {}
        for family in FAMILIAS:
            if family["nombre"] not in FAMILIES:
                continue
            split = datos_semilla(family, namespace["SEED"])
            stack = lambda key: torch.cat(  # noqa: E731
                [split[f"{key}_{role}"] for role in ("tr", "va", "te")]).numpy()
            drawn[family["nombre"]] = dict(Xs=stack("Xs"), ys=stack("ys"),
                                          Xt=stack("Xt"))
            assert len(drawn[family["nombre"]]["Xs"]) == N_POR_DOMINIO
            assert len(drawn[family["nombre"]]["Xt"]) == N_POR_DOMINIO
        return drawn
    from MIL_CREDA_Benchmark import toy_run

    drawn = {}
    for name in FAMILIES:
        family = toy_run.build_family(name, namespace["SEMILLA"],
                                      namespace["ESCALA"])
        drawn[name] = dict(Xs=family.Xs, ys=family.ys, Xt=family.Xt)
    return drawn


def _offsets(collection) -> np.ndarray:
    return np.column_stack([np.asarray(c, dtype=float)
                            for c in collection._offsets3d])


def test_one_figure_with_one_3d_panel_per_family(executed) -> None:
    method, namespace = executed
    figure = namespace["INPUT_FIGURE"]
    assert len(figure.axes) == 3
    assert [ax.name for ax in figure.axes] == ["3d"] * 3
    for ax, name in zip(figure.axes, FAMILIES):
        assert name in ax.get_title(), (method, ax.get_title())
        assert len(ax.collections) == 2, (method, name)
    assert list(namespace["INPUT_DATA"]) == list(FAMILIES)


def test_each_panel_overlays_the_generated_source_and_target(executed) -> None:
    method, namespace = executed
    drawn = _generated(method, namespace)
    for ax, name in zip(namespace["INPUT_FIGURE"].axes, FAMILIES):
        data = namespace["INPUT_DATA"][name]
        source, target = ax.collections
        np.testing.assert_array_equal(data["Xs"], drawn[name]["Xs"])
        np.testing.assert_array_equal(data["Xt"], drawn[name]["Xt"])
        np.testing.assert_array_equal(data["ys"], drawn[name]["ys"])
        np.testing.assert_array_equal(_offsets(source), np.asarray(data["Xs"], float))
        np.testing.assert_array_equal(_offsets(target), np.asarray(data["Xt"], float))
        legend = ax.get_legend()
        assert legend is not None, (method, name)
        texts = [t.get_text() for t in legend.get_texts()]
        assert len(texts) == 2 and "fuente" in texts[0] and "blanco" in texts[1]


def test_source_and_target_use_distinct_markers(executed) -> None:
    from matplotlib.markers import MarkerStyle

    def path_of(marker):
        style = MarkerStyle(marker)
        return style.get_path().transformed(style.get_transform()).vertices

    method, namespace = executed
    for ax in namespace["INPUT_FIGURE"].axes:
        source, target = ax.collections
        np.testing.assert_array_equal(source.get_paths()[0].vertices, path_of("x"))
        np.testing.assert_array_equal(target.get_paths()[0].vertices, path_of("o"))


def test_source_is_coloured_by_class_and_target_is_one_neutral_grey(executed) -> None:
    from matplotlib.colors import to_rgb

    from CREDA.toy_experiment import COL_DURA

    method, namespace = executed
    for ax, name in zip(namespace["INPUT_FIGURE"].axes, FAMILIES):
        source, target = ax.collections
        expected = np.array([to_rgb(COL_DURA[int(c)])
                             for c in namespace["INPUT_DATA"][name]["ys"]])
        # `get_facecolors()` of a 3D scatter is the depth-sorted projection;
        # the per-point colours as given live in `_original_facecolor`.
        np.testing.assert_allclose(source._original_facecolor[:, :3], expected)
        grey = np.unique(np.atleast_2d(target._original_facecolor)[:, :3], axis=0)
        assert grey.shape == (1, 3), (method, name)
        assert grey[0, 0] == grey[0, 1] == grey[0, 2], (method, name)


def test_the_axes_contain_every_point_of_both_domains(executed) -> None:
    method, namespace = executed
    for ax, name in zip(namespace["INPUT_FIGURE"].axes, FAMILIES):
        data = namespace["INPUT_DATA"][name]
        points = np.concatenate([data["Xs"], data["Xt"]])
        limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
        for axis, (low, high) in enumerate(limits):
            assert low <= points[:, axis].min(), (method, name, axis)
            assert points[:, axis].max() <= high, (method, name, axis)


def test_the_figure_cell_draws_once_and_changes_no_state(executed) -> None:
    method, namespace = executed
    before, after = namespace["_before"], namespace["_after"]
    assert after["figures"] - before["figures"] <= {namespace["INPUT_FIGURE"].number}
    source = "".join(namespace["_cells"][namespace["_last"]]["source"])
    assert source.count("plt.show()") + source.count("display(") <= 1
    for left, right in zip(before["numpy"], after["numpy"]):
        assert np.array_equal(left, right)
    assert torch.equal(before["torch"], after["torch"])
    for key, value in before["configuration"].items():
        assert namespace[key] == value, (method, key)


def test_mil_creda_keeps_the_figure_under_its_old_name(executed) -> None:
    method, namespace = executed
    if method == "MIL-CREDA":
        assert namespace["FIGURAS"]["datos"] is namespace["INPUT_FIGURE"]


def test_creda_draws_the_inputs_before_training() -> None:
    cells = _code_cells("CREDA")
    figure = _index(cells, "INPUT_FIGURE =")
    assert _index(cells, "def load_data") < figure
    assert figure < _index(cells, "def train_source_phase")
