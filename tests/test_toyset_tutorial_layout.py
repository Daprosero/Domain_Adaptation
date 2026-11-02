"""The toyset is one teaching notebook per method, and the README says so.

Each method keeps exactly one toyset notebook, `Toyset.ipynb`, which trains in
memory and reads no stored artefact. The generator/plot/device-probe split it
replaced is gone; the publication notebooks and the BALLADEER notebooks are
unrelated and stay. The README must point at the tutorials with the parameters
they really carry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

TUTORIALS = {
    "CREDA": ROOT / "CREDA" / "Notebooks" / "Toyset.ipynb",
    "MIL-CREDA": ROOT / "MIL-CREDA" / "Notebooks" / "Toyset.ipynb",
}

RETIRED = (
    "CREDA/Notebooks/Toyset_Generator.ipynb",
    "CREDA/Notebooks/Toyset_Plot.ipynb",
    "MIL-CREDA/Notebooks/Toyset_Generator.ipynb",
    "MIL-CREDA/Notebooks/Toyset_Plot.ipynb",
    "MIL-CREDA/Notebooks/Toyset_Device_Probe.ipynb",
)

#: What the retired notebooks left next to themselves: the plot notebook's
#: archived figures and an old search record. Nothing reads either.
RETIRED_OUTPUTS = (
    "MIL-CREDA/Notebooks/toy_figures",
    "MIL-CREDA/Notebooks/toy_search.json",
)

RETAINED = (
    "CREDA/Notebooks/Plots_Generator.ipynb",
    "CREDA/Notebooks/Results_Generator.ipynb",
    "CREDA/Notebooks/Tables_Generator.ipynb",
    "MIL-CREDA/Notebooks/BALLADEER_01_Inventario.ipynb",
    "MIL-CREDA/Notebooks/BALLADEER_02_Auditoria.ipynb",
    "MIL-CREDA/Notebooks/BALLADEER_03_Preprocesamiento.ipynb",
)

#: What the retired split used to read or write: the stored sweep and its readers.
LEGACY_READERS = ("toy_tables", "toy_run.cargar", "toy_run.guardar",
                  "artefacto_seccion_uno", "guardar_seccion", "barrido.json",
                  "Toyset_Generator", "Toyset_Plot", "Toyset_Device_Probe")

#: Calls that would leave a file behind.
WRITES = ("savefig(", "json.dump", "write_text(", "write_bytes(", "to_csv(",
          "np.save", "torch.save", "mkdir(")


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")


@pytest.mark.parametrize("method", sorted(TUTORIALS))
def test_each_method_has_exactly_one_toyset_notebook(method):
    found = sorted(p.name for p in (ROOT / method / "Notebooks").glob("*.ipynb")
                   if "toyset" in p.name.lower())
    assert found == ["Toyset.ipynb"], found


@pytest.mark.parametrize("relative", RETIRED)
def test_the_retired_notebooks_are_gone(relative):
    assert not (ROOT / relative).exists(), relative


@pytest.mark.parametrize("relative", RETIRED_OUTPUTS)
def test_the_retired_notebooks_outputs_are_gone(relative):
    assert not (ROOT / relative).exists(), relative


@pytest.mark.parametrize("relative", RETAINED)
def test_publication_and_balladeer_notebooks_stay(relative):
    assert (ROOT / relative).is_file(), relative


@pytest.mark.parametrize("method", sorted(TUTORIALS))
def test_the_tutorial_reads_no_stored_sweep(method):
    code = _code(TUTORIALS[method])
    used = [name for name in LEGACY_READERS if name in code]
    assert used == [], used


@pytest.mark.parametrize("method", sorted(TUTORIALS))
def test_the_tutorial_writes_nothing_and_never_names_results(method):
    code = _code(TUTORIALS[method])
    assert "Results" not in code
    used = [call for call in WRITES if call in code]
    assert used == [], used


def test_the_readme_points_at_each_tutorial_and_no_retired_notebook():
    text = README.read_text(encoding="utf-8")
    for path in TUTORIALS.values():
        assert f"`{path.relative_to(ROOT).as_posix()}`" in text, path
    stale = [r for r in RETIRED if Path(r).name in text]
    assert stale == [], stale


def test_the_readme_states_the_creda_selector_and_batchnorm():
    text = README.read_text(encoding="utf-8")
    code = _code(TUTORIALS["CREDA"])
    assert 'VERSION = "v2"' in code
    assert '`VERSION = "v2"`' in text
    assert '`"v01"`' in text
    assert "BatchNorm" in text


@pytest.mark.parametrize("method, names", [
    ("CREDA", ("source_epochs", "source_patience", "adaptation_epochs",
               "epochs")),
    ("MIL-CREDA", ("EPOCAS_1", "PACIENCIA_1", "EPOCAS_2")),
])
def test_the_readme_quotes_the_budget_the_notebook_really_has(method, names):
    """A number in the README that the notebook does not carry misleads."""
    text = README.read_text(encoding="utf-8")
    code = _code(TUTORIALS[method])
    for name in names:
        values = re.findall(rf"^\s*{name}\s*=\s*(\d+)", code, re.M)
        assert len(values) == 1, (name, values)
        assert f"`{name}={values[0]}`" in text or \
            f"`{name} = {values[0]}`" in text, (name, values[0])
