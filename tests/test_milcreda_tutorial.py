"""The MIL-CREDA toyset tutorial, executed cell by cell.

`MIL-CREDA/Notebooks/Toyset.ipynb` is the one toyset notebook of this method and
it exists to TEACH how MIL-CREDA is trained: labelled source bags, unlabelled
target bags, one BatchNorm backbone, phase 1 on the source, a frozen source
bandwidth, a cloned target encoder and phase 2 moving that encoder alone.

These tests run the notebook's own code -- every code cell, in order, in one
namespace -- with only the budget cell overridden to a short CPU smoke budget.
Nothing here re-implements a phase: each assertion reads what the notebook
computed or calls a function the notebook defined.
"""

from __future__ import annotations

import json
import math
import subprocess
import warnings
from pathlib import Path

import pytest
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPOSITORY / "MIL-CREDA" / "Notebooks"
NOTEBOOK = NOTEBOOKS / "Toyset.ipynb"
FAMILIES = ("lunas", "toroide", "rollo")

#: Short enough to run in seconds on a CPU; `EPOCAS_2` is the shortest phase 2
#: in which the per-epoch criterion is allowed to choose an epoch.
SMOKE_BUDGET = {"EPOCAS_1": 2, "PACIENCIA_1": 2, "EPOCAS_2": 3,
                "BARRIDOS_W": 6}

#: Full runners the tutorial must not hide its training behind.
OPAQUE_RUNNERS = ("run_cell", "adapt_phase_two", "train_phase_one", "sweep(",
                  "barrido_repartido", "search_phase_one", "search_phase_two",
                  "fusionar", "optuna")


def _document() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(document: dict) -> list[dict]:
    return [cell for cell in document["cells"] if cell["cell_type"] == "code"]


def _milcreda_tree_state() -> str:
    """Every tracked, untracked and ignored change under `MIL-CREDA/`."""
    return subprocess.run(
        ["git", "status", "--porcelain", "--ignored", "--untracked-files=all",
         "--", "MIL-CREDA"],
        cwd=REPOSITORY, capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def executed() -> dict:
    """The notebook's namespace after running every code cell once."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    document = _document()
    before = _milcreda_tree_state()
    namespace: dict = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patch:
        # The local layout: no runner handoff, working directory = notebook's.
        patch.delenv("FORGE_CLONE_ROOT", raising=False)
        patch.delenv("FORGE_CLONE_COMMIT", raising=False)
        patch.chdir(NOTEBOOKS)
        warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
        for index, cell in enumerate(_code_cells(document)):
            source = "".join(cell["source"])
            exec(compile(source, f"{NOTEBOOK.name}[code {index}]", "exec"),
                 namespace)
            if "parameters" in cell.get("metadata", {}).get("tags", []):
                namespace.update(SMOKE_BUDGET)
    namespace["_treeBefore"] = before
    namespace["_treeAfter"] = _milcreda_tree_state()
    yield namespace
    plt.close("all")


def test_the_notebook_is_a_python3_notebook_with_a_budget_cell() -> None:
    document = _document()
    assert document["metadata"]["kernelspec"]["name"] == "python3"
    tagged = [cell for cell in _code_cells(document)
              if "parameters" in cell.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1, "exactly one cell carries the `parameters` tag"
    for name in SMOKE_BUDGET:
        assert name in "".join(tagged[0]["source"])


def test_the_first_code_cell_is_the_owned_bootstrap_byte_for_byte() -> None:
    owned = (REPOSITORY.parents[1] / ".claude" / "skills" / "remote-execution"
             / "assets" / "notebook_repo_root.py")
    if not owned.is_file():
        pytest.skip(f"{owned} is not there to compare against")
    first = "".join(_code_cells(_document())[0]["source"])
    assert first == owned.read_text(encoding="utf-8")


def test_the_training_is_written_out_and_not_delegated_to_a_runner() -> None:
    code = "\n".join("".join(cell["source"])
                     for cell in _code_cells(_document()))
    found = [name for name in OPAQUE_RUNNERS if name in code]
    assert not found, f"the tutorial hides its training behind {found}"
    for visible in ("torch.optim.Adam", "optimizador.step()", ".backward()",
                    "training_step(", "supervised_loss(",
                    "frozen_source_sigma(", "copy_source_to_target(",
                    "set_phase_two_modes(", "effective_classes("):
        assert visible in code, f"{visible} is not visible in the notebook"


def test_one_batchnorm_architecture_and_all_three_families(executed) -> None:
    assert tuple(executed["FAMILIAS"]) == FAMILIES
    assert executed["NORMALIZACION"] == "batchnorm"
    assert executed["BACKBONE"] == "mlp"
    assert tuple(executed["RESULTADOS"]) == FAMILIES
    for name in FAMILIES:
        assert executed["RESULTADOS"][name]["capasBatchNorm"] > 0, name


def test_the_smoke_budget_reached_the_training(executed) -> None:
    assert executed["ESCALA"].epochs_2 == SMOKE_BUDGET["EPOCAS_2"]
    for name in FAMILIES:
        result = executed["RESULTADOS"][name]
        assert 1 <= len(result["faseUno"]["historia"]) <= SMOKE_BUDGET["EPOCAS_1"]
        steps = result["faseDos"]["pasosPorEpoca"] * SMOKE_BUDGET["EPOCAS_2"]
        assert len(result["faseDos"]["historia"]) == steps
        assert len(result["faseDos"]["epocas"]) == SMOKE_BUDGET["EPOCAS_2"]


def test_every_loss_is_finite(executed) -> None:
    for name in FAMILIES:
        result = executed["RESULTADOS"][name]
        for row in result["faseUno"]["historia"]:
            assert math.isfinite(row["perdida"]), (name, row)
        for row in result["faseDos"]["historia"]:
            for key in ("objetivo", "global", "local", "supervisadaLectura"):
                assert math.isfinite(row[key]), (name, key, row)


def test_parameters_move_exactly_where_each_phase_says(executed) -> None:
    for name in FAMILIES:
        checks = executed["RESULTADOS"][name]["chequeos"]
        # Phase 1 trains the source encoder, attention and head ...
        assert checks["fuenteMovidaFaseUno"] > 0.0, name
        # ... and never touches the target encoder, which is cloned afterwards.
        assert checks["blancoMovidoFaseUno"] == 0.0, name
        # The clone carries parameters AND BatchNorm buffers.
        assert checks["clonExacto"] is True, name
        # Phase 2 moves theta_t alone: source encoder, attention, head and the
        # source BatchNorm statistics stay bit-identical.
        assert checks["fuenteMovidaFaseDos"] == 0.0, name
        assert checks["buffersFuenteMovidosFaseDos"] == 0.0, name
        assert checks["blancoMovidoFaseDos"] > 0.0, name


def test_phase_two_runs_at_the_frozen_source_bandwidth(executed) -> None:
    for name in FAMILIES:
        result = executed["RESULTADOS"][name]
        assert result["sigmaFaseDos"] > 0.0
        assert result["sigmaFaseDos"] == result["modelo"].sigma
        assert result["sigmaFaseDos"] != result["sigmaFaseUno"]


def test_phase_two_selects_without_target_labels(executed) -> None:
    for name in FAMILIES:
        phase_two = executed["RESULTADOS"][name]["faseDos"]
        legal = [row for row in phase_two["epocas"]
                 if row["epoca"] >= executed["toy_run"].MIN_EPOCHS_2]
        best = max(legal, key=lambda row: row["cef"])
        assert phase_two["mejorEpoca"] == best["epoca"]


def test_target_labels_never_enter_the_training_objective(executed) -> None:
    """Poison the target labels the model holds and phase 2 is bit-identical.

    `-1` is not a class, so any read of those labels by the objective -- a
    one-hot, an index, a stratification -- would raise or change a number.
    """
    ns = executed
    toy_run = ns["toy_run"]
    family = toy_run.build_family("lunas", ns["SEMILLA"], ns["ESCALA"])
    material = toy_run.material_for(family, ns["SEMILLA"], ns["ESCALA"],
                                    ns["BACKBONE"])
    with toy_run.material_bound(family, ns["ESCALA"], ns["BACKBONE"],
                                ns["HIPER"]):
        source_train, source_valid, _ = ns["bags"].roles(material["source"])
        source_model, _ = ns["modelo_de_fase_uno"](material)
        ns["entrenar_fase_uno"](source_model, source_train, source_valid,
                                torch.Generator().manual_seed(1))
        runs = []
        for poison in (False, True):
            model, _, _ = ns["preparar_fase_dos"](source_model, material)
            if poison:
                model.target.labels = torch.full_like(model.target.labels, -1)
            phase_two = ns["entrenar_fase_dos"](
                model, source_train, torch.Generator().manual_seed(2))
            runs.append((phase_two["historia"],
                         {k: v.clone() for k, v in
                          model.target_encoder.state_dict().items()}))
    (clean, clean_state), (poisoned, poisoned_state) = runs
    assert clean == poisoned
    for key, value in clean_state.items():
        assert torch.equal(value, poisoned_state[key]), key


def test_each_family_is_evaluated_on_source_and_target(executed) -> None:
    for name in FAMILIES:
        evaluation = executed["RESULTADOS"][name]["evaluacion"]
        for key in ("aciertoFuente", "aciertoBlancoAntes", "aciertoBlanco"):
            assert 0.0 <= evaluation[key] <= 1.0, (name, key)
        for domain in ("latenteFuente", "latenteBlanco"):
            positions = evaluation[domain]["positions"]
            assert positions.shape[0] == 81, (name, domain)
            assert torch.isfinite(torch.as_tensor(positions)).all()


def test_the_plots_draw_every_family(executed) -> None:
    figures = executed["FIGURAS"]
    for kind in ("datos", "curvas", "latente"):
        assert kind in figures, kind
        titles = " ".join(ax.get_title() for ax in figures[kind].axes)
        for name in FAMILIES:
            assert name in titles, (kind, name)


def test_the_tutorial_writes_nothing_under_milcreda(executed) -> None:
    assert executed["_treeAfter"] == executed["_treeBefore"]


# ------------------------------------------- the restored teaching animations


def _gif_frames(data: bytes) -> list:
    """Every frame of an in-memory GIF, decoded to RGB arrays."""
    import io

    import numpy as np
    from PIL import Image

    assert isinstance(data, bytes) and data[:6] in (b"GIF87a", b"GIF89a")
    frames = []
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "GIF"
        for index in range(image.n_frames):
            image.seek(index)
            frames.append(np.asarray(image.convert("RGB")).copy())
    return frames


def test_the_animations_are_embedded_as_gif_images() -> None:
    code = "\n".join("".join(cell["source"])
                     for cell in _code_cells(_document()))
    assert 'format="gif"' in code
    assert "display(" in code


def test_each_family_has_an_adaptation_gif_from_its_own_training(executed) -> None:
    """One frame per captured epoch, 0 = the clone before any phase-2 step."""
    import numpy as np

    for name in FAMILIES:
        result = executed["RESULTADOS"][name]
        trajectory = result["trayectoria"]
        assert trajectory["epocas"] == list(range(SMOKE_BUDGET["EPOCAS_2"] + 1))
        frames = _gif_frames(executed["ANIMACIONES"][name]["adaptacion"])
        assert len(frames) == len(trajectory["epocas"]) > 1, name
        assert not np.array_equal(frames[0], frames[-1]), name
        # The captured positions are the model's own: the frame of the epoch
        # phase 2 restored is exactly what the evaluation drew afterwards.
        chosen = trajectory["epocas"].index(result["faseDos"]["mejorEpoca"])
        np.testing.assert_array_equal(
            trajectory["blanco"][chosen],
            result["evaluacion"]["latenteBlanco"]["positions"])
        # theta_t moved, so the target cloud of epoch 0 is not the last one.
        assert not np.array_equal(trajectory["blanco"][0],
                                  trajectory["blanco"][-1]), name
        # The source is frozen: one cloud, the one the evaluation drew.
        np.testing.assert_array_equal(
            trajectory["fuente"],
            result["evaluacion"]["latenteFuente"]["positions"])
        # Bag confidence of every target bag per frame, Eq. (24), in [0, 1].
        for w in trajectory["confianzaBlanco"]:
            assert w.shape == (81,)
            assert ((w >= 0.0) & (w <= 1.0)).all()


def test_capturing_the_trajectory_does_not_change_phase_two(executed) -> None:
    """Same seed with and without the observer: identical history and state."""
    ns = executed
    toy_run = ns["toy_run"]
    family = toy_run.build_family("lunas", ns["SEMILLA"], ns["ESCALA"])
    material = toy_run.material_for(family, ns["SEMILLA"], ns["ESCALA"],
                                    ns["BACKBONE"])
    with toy_run.material_bound(family, ns["ESCALA"], ns["BACKBONE"],
                                ns["HIPER"]):
        source_train, source_valid, _ = ns["bags"].roles(material["source"])
        source_model, _ = ns["modelo_de_fase_uno"](material)
        ns["entrenar_fase_uno"](source_model, source_train, source_valid,
                                torch.Generator().manual_seed(1))
        runs = []
        for observe in (False, True):
            model, _, _ = ns["preparar_fase_dos"](source_model, material)
            observer = (ns["tutorial_visuals"].Trayectoria(
                material, ns["DISPOSITIVO"], ns["ESCALA"].epochs_2)
                if observe else None)
            phase_two = ns["entrenar_fase_dos"](
                model, source_train, torch.Generator().manual_seed(2),
                observar=observer)
            runs.append((phase_two, model.state_dict(), observer,
                         [m.training for m in model.modules()]))
    (plain, plain_state, _, plain_modes), (seen, seen_state, observer,
                                            seen_modes) = runs
    assert observer.epocas == list(range(ns["ESCALA"].epochs_2 + 1))
    assert plain["historia"] == seen["historia"]
    assert plain["epocas"] == seen["epocas"]
    assert plain_modes == seen_modes
    for key, value in plain_state.items():
        assert torch.equal(value, seen_state[key]), key


def test_the_frame_epochs_are_bounded() -> None:
    from MIL_CREDA_Benchmark import tutorial_visuals

    assert tutorial_visuals.epocas_a_capturar(3, 20) == [0, 1, 2, 3]
    picked = tutorial_visuals.epocas_a_capturar(100, 20)
    assert picked[0] == 0 and picked[-1] == 100 and len(picked) <= 21
    assert picked == sorted(set(picked))


def test_each_family_has_a_bag_confidence_walk(executed) -> None:
    """Target bags walk from a boundary to a source centre, read by the head."""
    import numpy as np
    import torch.nn.functional as F
    from MIL_CREDA.confidence import confidences

    toy_run = executed["toy_run"]
    for name in FAMILIES:
        result = executed["RESULTADOS"][name]
        walk = result["caminataW"]
        model = result["modelo"]
        assert walk["barridos"] == SMOKE_BUDGET["BARRIDOS_W"]
        assert walk["arranques"] == list(toy_run.ARRANQUES_W)
        frames = _gif_frames(executed["ANIMACIONES"][name]["w"])
        assert len(frames) == SMOKE_BUDGET["BARRIDOS_W"], name
        assert not np.array_equal(frames[0], frames[-1]), name
        # The bag the path starts from, embedded by theta_t right now.
        with toy_run.material_bound(result["familia"], executed["ESCALA"],
                                    executed["BACKBONE"], executed["HIPER"]):
            target = executed["tutorial_visuals"].representaciones_del_blanco(
                model, result["material"], executed["DISPOSITIVO"])
        for start, destination in zip(walk["arranques"], walk["clasesDestino"]):
            path = walk["caminos"][start][str(destination)]
            np.testing.assert_allclose(path["camino"][-1],
                                       walk["centros"][destination],
                                       rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(
                path["camino"][0], target[walk["bolsaDelBlanco"][start]],
                rtol=1e-5, atol=1e-5)
            with torch.no_grad():
                g = F.softmax(model.head(torch.as_tensor(
                    path["camino"], dtype=torch.float32)), dim=1)
            np.testing.assert_allclose(path["w"], confidences(g).numpy(),
                                       rtol=1e-4, atol=1e-5)


def test_the_walk_is_labelled_as_a_sweep_and_not_an_epoch(executed) -> None:
    import matplotlib.pyplot as plt

    visuals = executed["tutorial_visuals"]
    walk = executed["RESULTADOS"]["lunas"]["caminataW"]
    figure = visuals.figura_de_caminata(walk, "lunas", 2)
    try:
        assert len(figure.axes) == 2
        curve = figure.axes[1]
        assert "barrido" in curve.get_xlabel()
        assert "época" not in curve.get_xlabel()
        # The curve is partial: up to sweep 2, three points per start.
        lengths = [len(line.get_xdata()) for line in curve.lines]
        assert lengths.count(3) == 3 and max(lengths) == 3, lengths
    finally:
        plt.close(figure)
