"""El cuaderno tutorial de CREDA sobre el toyset, ejecutado por sus propias celdas.

`CREDA/Notebooks/Toyset.ipynb` es el ejemplo de como se entrena CREDA: una sola
arquitectura (un Linear, la ReLU y BatchNorm), las tres familias del toyset, y
las dos versiones del metodo en un `if/else` de dos diccionarios, con la `v2`
por defecto.

Estos tests no reimplementan nada. Leen el cuaderno, ejecutan sus celdas de
codigo en orden en un espacio de nombres, y despues interrogan lo que esas
celdas dejaron definido. La unica intervencion es la de papermill: tras la celda
etiquetada `parameters` se fijan `VERSION` y `ADJUSTMENTS`, y `ADJUSTMENTS` achica los
presupuestos a un humo de pocas epocas en CPU.
"""

from __future__ import annotations

import copy
import io
import json
import math
import subprocess
from pathlib import Path

import pytest
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPOSITORY / "CREDA" / "Notebooks"
TUTORIAL = NOTEBOOKS / "Toyset.ipynb"
OWNED_CELL = (
    REPOSITORY.parents[1] / ".claude" / "skills" / "remote-execution"
    / "assets" / "notebook_repo_root.py"
)
FAMILIES = ("lunas", "toroide", "rollo")

#: Presupuestos de humo: unas pocas epocas para recorrer cada linea del flujo,
#: no para converger. Ningun numero de este archivo es un resultado.
SMOKE = {
    "v2": {"source_epochs": 3, "source_patience": 3,
           "adaptation_epochs": 2, "batch_size": 64},
    "v01": {"epochs": 3, "batch_size": 64},
}

#: Fewer GIF frames and a shorter confidence walk for the smoke runs. They are
#: still drawn and encoded: only how many frames, never whether.
SMOKE_VISUALS = {"FRAMES": 3, "WALK_STEPS": 6}


def _document() -> dict:
    return json.loads(TUTORIAL.read_text(encoding="utf-8"))


def _code_cells() -> list[dict]:
    return [cell for cell in _document()["cells"] if cell["cell_type"] == "code"]


def _execute(version: str | None, overrides: dict | None, *, stop_after=None,
             visuals: dict | None = None) -> dict:
    """Run the notebook's code cells in order, the way a kernel would.

    Right after the cell tagged `parameters`, `VERSION` and `ADJUSTMENTS` are set
    (papermill's injection), plus any of the visual parameters in `visuals`.
    `stop_after` ends the run after the code cell with that index, for tests
    that only need the configuration.
    """
    namespace: dict = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(NOTEBOOKS)
        patch.delenv("FORGE_CLONE_ROOT", raising=False)
        patch.delenv("FORGE_CLONE_COMMIT", raising=False)
        patch.setenv("MPLBACKEND", "Agg")
        for index, cell in enumerate(_code_cells()):
            source = "".join(cell["source"])
            exec(compile(source, f"Toyset.ipynb[code {index}]", "exec"), namespace)
            if "parameters" in cell.get("metadata", {}).get("tags", []):
                if version is not None:
                    namespace["VERSION"] = version
                if overrides is not None:
                    namespace["ADJUSTMENTS"] = dict(overrides)
                namespace.update(visuals or {})
            if stop_after is not None and index == stop_after:
                break
    return namespace


def _cell_index(marker: str) -> int:
    for index, cell in enumerate(_code_cells()):
        if marker in "".join(cell["source"]):
            return index
    raise AssertionError(f"no code cell contains {marker!r}")


def _config_cell_index() -> int:
    return _cell_index("if VERSION == ")


_SMOKE_RUNS: dict = {}


def _smoke_run(version: str) -> dict:
    """One smoke execution per version, shared by every test of the module."""
    if version not in _SMOKE_RUNS:
        _SMOKE_RUNS[version] = _execute(version, SMOKE[version], visuals=SMOKE_VISUALS)
    return _SMOKE_RUNS[version]


@pytest.fixture(params=["v2", "v01"])
def run(request) -> dict:
    return _smoke_run(request.param)


# --------------------------------------------------------------------------
# The notebook as a document


def test_the_tutorial_exists_with_the_python3_kernel() -> None:
    assert TUTORIAL.is_file(), f"{TUTORIAL} is missing"
    kernelspec = _document()["metadata"]["kernelspec"]
    assert kernelspec["name"] == "python3", kernelspec


def test_the_first_code_cell_is_the_owned_bootstrap_byte_for_byte() -> None:
    if not OWNED_CELL.is_file():
        pytest.skip(f"{OWNED_CELL} is not there to compare against")
    first = "".join(_code_cells()[0]["source"])
    owned = OWNED_CELL.read_text(encoding="utf-8")
    # nbformat stores a cell without its final newline; nothing else may differ.
    assert first == owned.removesuffix("\n"), "the bootstrap cell drifted"


def test_the_bootstrap_resolves_a_runner_pinned_root(tmp_path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True,
        capture_output=True, text=True).stdout.strip()
    toplevel = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=REPOSITORY, check=True,
        capture_output=True, text=True).stdout.strip())
    namespace: dict = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setenv("FORGE_CLONE_ROOT", str(toplevel))
        patch.setenv("FORGE_CLONE_COMMIT", commit)
        exec("".join(_code_cells()[0]["source"]), namespace)
    assert namespace["ROOT"] == toplevel


def test_version_defaults_to_v2_and_v01_is_the_other_branch() -> None:
    namespace = _execute(None, None, stop_after=_config_cell_index())
    assert namespace["VERSION"] == "v2"
    assert namespace["CONFIG"]["version"] == "v2"
    namespace = _execute("v01", None, stop_after=_config_cell_index())
    assert namespace["CONFIG"]["version"] == "v01"


def test_an_unknown_version_is_refused() -> None:
    with pytest.raises(ValueError, match="VERSION"):
        _execute("v3", None, stop_after=_config_cell_index())


def test_an_unknown_adjustment_is_refused_rather_than_ignored() -> None:
    with pytest.raises(KeyError, match="epochz"):
        _execute("v2", {"epochz": 1}, stop_after=_config_cell_index())


def test_the_version_dictionaries_carry_their_declared_losses() -> None:
    from CREDA.toy_experiment import (
        BALANCE_DE_MASA, FORMA_DE_COTAS, HP_2, NORMALIZAR_POR_COTAS)

    stop = _cell_index("def build_loss")
    namespace = _execute("v2", SMOKE["v2"], stop_after=stop)
    current = namespace["build_loss"](namespace["CONFIG"], sigma=1.5)
    assert current.normalize_by_bounds is NORMALIZAR_POR_COTAS
    assert current.bounds_form == FORMA_DE_COTAS
    assert current.balance_mass == BALANCE_DE_MASA
    assert current.weight_cross_block is HP_2["CRUZADO_SOPESADO"]
    assert current.detach_target_entropy is HP_2["RHO_T_SIN_GRAD"]
    assert current.use_entropy_weighting is True
    assert current.balance_mix_blocks is True
    assert current.sigma == 1.5
    with pytest.raises(ValueError, match="sigma"):
        namespace["build_loss"](namespace["CONFIG"])

    namespace = _execute("v01", SMOKE["v01"], stop_after=stop)
    published = namespace["build_loss"](namespace["CONFIG"])
    assert published.sigma == "auto"
    assert published.use_entropy_weighting is True
    assert published.normalize_by_bounds is False
    assert published.balance_mass == "cardinality"
    assert published.weight_cross_block is False
    assert published.balance_mix_blocks is False


# --------------------------------------------------------------------------
# Both versions, all three families


def test_every_family_runs_with_one_batchnorm_architecture(run) -> None:
    assert set(run["RESULTS"]) == set(FAMILIES)
    for name in FAMILIES:
        encoder = run["RESULTS"][name]["encoder_t"]
        norms = [m for m in encoder.modules()
                 if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm))]
        assert len(norms) == 1 and isinstance(norms[0], torch.nn.BatchNorm1d), norms


def test_every_family_reports_finite_source_and_target_accuracy(run) -> None:
    for name in FAMILIES:
        accuracy = run["RESULTS"][name]["acc"]
        for domain in ("source", "target"):
            assert 0.0 <= accuracy[domain] <= 1.0, (name, domain, accuracy)


def test_every_recorded_loss_is_finite(run) -> None:
    for name in FAMILIES:
        history = run["RESULTS"][name]["history"]
        assert history, name
        for phase_rows in history.values():
            for row in phase_rows:
                for key, value in row.items():
                    if isinstance(value, float):
                        assert math.isfinite(value), (name, key, row)


def test_the_figure_is_drawn_from_finite_results(run) -> None:
    figure = run["FIGURE"]
    assert len(figure.axes) >= 2 * len(FAMILIES)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    assert buffer.tell() > 0


def test_target_labels_never_reach_training(run) -> None:
    """Shuffling the target's train and valid labels changes no trained weight."""
    family = next(f for f in run["TUTORIAL_FAMILIES"] if f["nombre"] == "lunas")
    data = run["load_data"](family)
    shuffled = dict(data)
    generator = torch.Generator().manual_seed(123)
    for key in ("yt_tr", "yt_va"):
        shuffled[key] = data[key][torch.randperm(len(data[key]), generator=generator)]
    first = run["run_family"](family, run["CONFIG"], data=data)
    second = run["run_family"](family, run["CONFIG"], data=shuffled)
    for key in ("encoder_t", "head"):
        a, b = first[key].state_dict(), second[key].state_dict()
        for name in a:
            assert torch.equal(a[name], b[name]), (key, name)


# --------------------------------------------------------------------------
# v2: the current toy protocol


@pytest.fixture
def v2() -> dict:
    return _smoke_run("v2")


def _snapshot(module: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_v2_clones_the_target_including_batchnorm_buffers(v2) -> None:
    family = v2["TUTORIAL_FAMILIES"][0]
    data = v2["load_data"](family)
    encoder, head = v2["new_model"](v2["CONFIG"])
    v2["train_source_phase"](encoder, head, data["Xs_tr"], data["ys_tr"],
                          data["Xs_va"], data["ys_va"], v2["CONFIG"],
                          torch.Generator().manual_seed(0))
    target = v2["clone_for_target"](encoder)
    source_state, target_state = encoder.state_dict(), target.state_dict()
    assert any("running_mean" in name for name in source_state)
    for name in source_state:
        assert torch.equal(source_state[name], target_state[name]), name
    assert all(p.requires_grad for p in target.parameters())
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(not p.requires_grad for p in head.parameters())


def test_v2_adaptation_moves_only_the_target_without_source_ce(v2, monkeypatch) -> None:
    family = v2["TUTORIAL_FAMILIES"][1]
    data = v2["load_data"](family)
    config = v2["CONFIG"]
    encoder, head = v2["new_model"](config)
    v2["train_source_phase"](encoder, head, data["Xs_tr"], data["ys_tr"],
                          data["Xs_va"], data["ys_va"], config,
                          torch.Generator().manual_seed(0))
    target = v2["clone_for_target"](encoder)
    before_source, before_head = _snapshot(encoder), _snapshot(head)
    before_target = [p.detach().clone() for p in target.parameters()]

    def refuse(*args, **kwargs):
        raise AssertionError("the v2 adaptation objective must not contain source CE")

    monkeypatch.setitem(v2, "perdida_fuente", refuse)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", refuse)
    sigma = v2["sigma_de_la_fuente"](encoder(data["Xs_tr"]))
    history = v2["adapt_target"](encoder, head, target, data["Xs_tr"],
                                   data["ys_tr"], data["Xt_tr"], config,
                                   torch.Generator().manual_seed(0), sigma=sigma)
    assert len(history) == config["adaptation_epochs"]
    for name, value in _snapshot(encoder).items():
        assert torch.equal(value, before_source[name]), name
    for name, value in _snapshot(head).items():
        assert torch.equal(value, before_head[name]), name
    moved = [not torch.equal(a, b) for a, b in zip(before_target, target.parameters())]
    assert any(moved), "the target encoder did not move"


def test_v2_freezes_sigma_on_the_source_train(v2) -> None:
    for name in FAMILIES:
        result = v2["RESULTS"][name]
        assert math.isfinite(result["sigma"]) and result["sigma"] > 0
        assert result["encoder_s"] is not result["encoder_t"]


# --------------------------------------------------------------------------
# v01: the published joint objective


@pytest.fixture
def v01() -> dict:
    return _smoke_run("v01")


def test_v01_shares_one_encoder_and_moves_encoder_and_head(v01) -> None:
    family = v01["TUTORIAL_FAMILIES"][2]
    data = v01["load_data"](family)
    config = v01["CONFIG"]
    encoder, head = v01["new_model"](config)
    before = [p.detach().clone() for p in list(encoder.parameters()) + list(head.parameters())]
    v01["train_joint"](encoder, head, data["Xs_tr"], data["ys_tr"],
                             data["Xt_tr"], data["Xs_va"], data["ys_va"], config,
                             torch.Generator().manual_seed(0))
    after = list(encoder.parameters()) + list(head.parameters())
    assert all(not torch.equal(a, b) for a, b in zip(before, after))
    for name in FAMILIES:
        assert v01["RESULTS"][name]["encoder_s"] is v01["RESULTS"][name]["encoder_t"]


def test_v01_objective_is_ce_plus_the_ramped_creda_term(v01) -> None:
    from CREDA.schedules import creda_ramp

    config = v01["CONFIG"]
    for name in FAMILIES:
        rows = v01["RESULTS"][name]["history"]["joint"]
        assert len(rows) == config["epochs"]
        for epoch, row in enumerate(rows):
            expected = creda_ramp(epoch, config["epochs"], config["delta"],
                                  ceiling=config["lambda_ceiling"])
            assert row["lambda"] == pytest.approx(expected)
            assert row["total"] == pytest.approx(
                row["ce"] + row["lambda"] * row["creda"], rel=1e-5, abs=1e-7)
        assert rows[0]["lambda"] == 0.0


# --------------------------------------------------------------------------
# The adaptation GIF: snapshots of the real training trajectory


def _gif_frames(data: bytes) -> list:
    from PIL import Image

    frames = []
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "GIF"
        for index in range(image.n_frames):
            image.seek(index)
            frames.append(image.convert("RGB").copy())
    return frames


def _training_epochs(run: dict) -> int:
    config = run["CONFIG"]
    return config["adaptation_epochs"] if config["version"] == "v2" else config["epochs"]


def test_every_family_has_an_animated_adaptation_gif(run) -> None:
    from PIL import ImageChops

    assert set(run["ADAPTATION_GIFS"]) == set(FAMILIES)
    for name in FAMILIES:
        frames = _gif_frames(run["ADAPTATION_GIFS"][name])
        assert len(frames) > 1, name
        assert len(frames) == len(run["RESULTS"][name]["snapshots"]), name
        assert ImageChops.difference(frames[0], frames[-1]).getbbox() is not None, name


def test_snapshots_run_from_the_start_to_the_final_epoch(run) -> None:
    epochs = _training_epochs(run)
    for name in FAMILIES:
        snapshots = run["RESULTS"][name]["snapshots"]
        completed = [s["epoch"] for s in snapshots]
        assert completed[0] == 0 and completed[-1] == epochs, (name, completed)
        assert completed == sorted(set(completed)), (name, completed)
        assert len(snapshots) <= run["FRAMES"] + 1
        first, last = snapshots[0], snapshots[-1]
        assert not torch.equal(torch.as_tensor(first["f_t"]),
                               torch.as_tensor(last["f_t"])), name


def test_the_last_snapshot_is_the_final_model(run) -> None:
    from MIL_CREDA.confidence import confidences

    for name in FAMILIES:
        result = run["RESULTS"][name]
        last = result["snapshots"][-1]
        data = result["data"]
        with torch.no_grad():
            f_t = result["encoder_t"](data["Xt_te"])
            f_s = result["encoder_s"](data["Xs_te"])
            g_t = torch.softmax(result["head"](f_t), dim=1)
        assert torch.allclose(torch.as_tensor(last["f_t"]), f_t, atol=1e-6), name
        assert torch.allclose(torch.as_tensor(last["f_s"]), f_s, atol=1e-6), name
        assert torch.equal(torch.as_tensor(last["pred_t"]), g_t.argmax(1)), name
        assert torch.allclose(torch.as_tensor(last["w_t"]), confidences(g_t), atol=1e-6)


def test_capturing_snapshots_does_not_change_training(run) -> None:
    family = next(f for f in run["TUTORIAL_FAMILIES"] if f["nombre"] == "toroide")
    data = run["load_data"](family)
    plain = run["run_family"](family, run["CONFIG"], data=data, frames=0)
    recorded = run["run_family"](family, run["CONFIG"], data=data, frames=run["FRAMES"])
    assert plain["snapshots"] == [] and recorded["snapshots"]
    for key in ("encoder_s", "encoder_t", "head"):
        a, b = plain[key].state_dict(), recorded[key].state_dict()
        for name in a:
            assert torch.equal(a[name], b[name]), (key, name)
    assert plain["history"] == recorded["history"]
    assert plain["acc"] == recorded["acc"]


def test_a_snapshot_restores_modes_and_leaves_buffers_and_rng_alone(run) -> None:
    from CREDA.tutorial_visuals import capture_snapshot

    encoder, head = run["new_model"](run["CONFIG"])
    encoder.train()
    head.eval()
    data = run["load_data"](run["TUTORIAL_FAMILIES"][0])
    buffers = _snapshot(encoder)
    rng = torch.get_rng_state()
    snapshot = capture_snapshot(encoder, encoder, head, data["Xs_te"], data["Xt_te"], epoch=0)
    assert encoder.training and not head.training
    for name, value in _snapshot(encoder).items():
        assert torch.equal(value, buffers[name]), name
    assert torch.equal(torch.get_rng_state(), rng)
    assert set(snapshot) >= {"epoch", "f_s", "f_t", "pred_s", "pred_t", "w_s", "w_t"}
    assert "ys" not in snapshot and "yt" not in snapshot


# --------------------------------------------------------------------------
# The confidence walk: w along constructed paths, read by the trained head


def test_every_family_has_an_animated_confidence_walk(run) -> None:
    from PIL import ImageChops

    assert set(run["WALK_GIFS"]) == set(FAMILIES)
    for name in FAMILIES:
        frames = _gif_frames(run["WALK_GIFS"][name])
        assert len(frames) == run["WALK_STEPS"], name
        assert ImageChops.difference(frames[0], frames[-1]).getbbox() is not None, name


def test_the_walk_reads_w_through_the_trained_head_and_creda_confidence(run) -> None:
    from MIL_CREDA.confidence import confidences

    for name in FAMILIES:
        result = run["RESULTS"][name]
        data = result["data"]
        with torch.no_grad():
            latent = result["encoder_s"](data["Xs_tr"])
        paths = run["WALKS"][name]
        assert [p["classes"] for p in paths] == [(0, 1), (1, 2), (0, 1, 2)], name
        for path in paths:
            centre = latent[data["ys_tr"] == path["destination"]].mean(0)
            z = torch.as_tensor(path["Z"], dtype=torch.float32)
            assert len(z) == run["WALK_STEPS"]
            assert torch.allclose(z[-1], centre, atol=1e-4), (name, path["classes"])
            with torch.no_grad():
                g = torch.softmax(result["head"](z), dim=1)
            assert torch.allclose(torch.as_tensor(path["w"], dtype=torch.float32),
                                  confidences(g), atol=1e-5)
            # The start is a tie of the named classes that the others do not touch.
            start = g[0]
            named = list(path["classes"])
            assert float(start[named].min()) == pytest.approx(float(start[named].max()), abs=1e-4)
            expected = 1.0 - math.log(len(named)) / math.log(3)
            assert float(path["w"][0]) == pytest.approx(expected, abs=1e-3)
