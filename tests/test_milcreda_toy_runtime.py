"""The MIL-CREDA toyset runtime is the tutorial's runtime and nothing more.

`MIL-CREDA/Notebooks/Toyset.ipynb` writes its own training loops and
`MIL_CREDA_Benchmark.tutorial_visuals` draws its figures, so what
`toy_run`/`toy_bags` still owe is the material, the model and the readings
those two call. The retired benchmark -- phase runners, searches, sweeps,
section artefacts, ablation and mechanism arms, the stored-sweep tables and
the old renderers -- is asserted absent here, and every name the tutorial
calls is asserted present and working.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from MIL_CREDA_Benchmark import __benchmark__, bags, toy_bags, toy_run, tutorial_visuals

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "MIL-CREDA" / "Notebooks" / "Toyset.ipynb"

#: The retired experiment surface: runners, searches, sweeps, section
#: artefacts, ablation/mechanism/selection arms, the contamination and shift
#: experiments, the stored-sweep IO and the renderers `tutorial_visuals`
#: replaced.
RETIRED_RUN = (
    "train_phase_one", "adapt_phase_two", "run_cell", "watchman",
    "oracle_target_reading", "adabn_reading", "sweep", "search_phase_one",
    "search_phase_two", "barrido_repartido", "guardar", "cargar",
    "guardar_seccion", "artefacto_seccion_uno", "ABLACION", "MECANISMOS",
    "BrazoDeMecanismo", "brazos_de_mecanismo", "brazos_de_seleccion",
    "barrido_sopesado", "barrido_ablacion", "barrido_correspondencia",
    "desplazar_bolsas", "contaminar_el_train_del_blanco", "animar_familia",
    "animar_w", "figura_de_correspondencia", "figura_del_cuadro",
    "weighting_cost", "cost_report", "SEARCH_RANGES",
)

#: Names `toy_run` once imported for experiments `CREDA.toy_experiment` no
#: longer has to keep for it.
RETIRED_IMPORTS = ("contaminar", "MODO_REUBICADO", "MODOS_CONTAMINACION",
                   "toy_tables")

RETIRED_BAGS = ("figura_familias", "FiguraDeFamilias", "casco_de_bolsa",
                "etiqueta_de_bolsa", "correspondencia_de_bolsas", "lectura",
                "lecturas", "clase_reproducida", "separacion_de_lunas")


def _notebook_calls() -> set[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    names = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "toy_run"):
                names.add(node.attr)
    return names


def test_the_stored_sweep_tables_module_is_gone():
    assert importlib.util.find_spec("MIL_CREDA_Benchmark.toy_tables") is None


def test_historical_toy_records_remain_declared_for_audits():
    """Retiring the writers must not turn kept archives into undeclared data."""
    declarations = __benchmark__["report"]["records"]
    assert "Results/Toyset" in declarations
    package_root = ROOT / "MIL-CREDA"
    archived = [p for p in (package_root / "Results/Toyset").rglob("*")
                if p.is_file()]
    assert archived
    assert all(any(p.is_relative_to(package_root / declaration)
                   for declaration in declarations) for p in archived)


@pytest.mark.parametrize("name", RETIRED_RUN + RETIRED_IMPORTS)
def test_toy_run_no_longer_carries_the_retired_benchmark(name):
    assert not hasattr(toy_run, name), name


@pytest.mark.parametrize("name", RETIRED_BAGS)
def test_toy_bags_no_longer_draws_or_reads_bags(name):
    assert not hasattr(toy_bags, name), name


def test_every_toy_run_name_the_tutorial_calls_exists():
    called = _notebook_calls()
    assert called, "the tutorial stopped calling toy_run"
    assert sorted(n for n in called if not hasattr(toy_run, n)) == []


def test_the_toyset_declares_exactly_its_one_arm():
    assert list(toy_run.TOY_ARMS_BY_ID) == [toy_run.TOY_ADAPTED]
    with pytest.raises(ValueError, match="unknown toyset arm"):
        toy_run.build_arm("TG", {}, toy_run.PILOT, toy_run.declared_hyper(
            toy_run.PILOT), torch.device("cpu"), 0)


def test_the_retained_runtime_builds_trains_and_clones_the_source():
    scale = dataclasses.replace(toy_run.PILOT, name="tutorial")
    hyper = toy_run.declared_hyper(scale)
    backbone = toy_run.NORMALIZATIONS["batchnorm"]
    device = torch.device("cpu")
    familia = toy_run.build_family("lunas", 0, scale)
    material = toy_run.material_for(familia, 0, scale, backbone)
    split = scale.realized_split()["bagsPerClass"]
    source = material["source"]
    assert source.members.shape == (scale.classes * scale.bags_per_class,
                                    scale.instances_per_bag)
    assert len(source.train_idx) == scale.classes * split["train"]
    with toy_run.material_bound(familia, scale, backbone, hyper):
        model = toy_run.build_arm(toy_run.TOY_ADAPTED, material, scale, hyper,
                                  device, 0)
        train = bags.roles(source)[0]
        x = torch.stack([train[i][0] for i in range(3)])
        y = torch.tensor([train[i][1] for i in range(3)])
        loss = toy_run.supervised_loss(model, x, y)
        assert torch.isfinite(loss)
        sigma = toy_run.frozen_source_sigma(model, source, source.train_idx,
                                            device)["sigma"]
        assert sigma > 0
        toy_run.copy_source_to_target(model)
        a, b = model.encoder.state_dict(), model.target_encoder.state_dict()
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)
        toy_run.set_phase_two_modes(model)
        assert not model.encoder.training and model.target_encoder.training
        assert 1.0 <= toy_run.effective_classes(model, device) <= scale.classes


def test_tutorial_visuals_reads_only_retained_runtime_names():
    tree = ast.parse(Path(tutorial_visuals.__file__).read_text(encoding="utf-8"))
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "toy_run"}
    assert sorted(n for n in used if not hasattr(toy_run, n)) == []
