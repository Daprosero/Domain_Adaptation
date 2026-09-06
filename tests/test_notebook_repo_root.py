"""Cada cuaderno recibe el clon; ninguno lo adivina.

La forja le exporta a un cuaderno el directorio del clon y el commit al que lo
fijó, y manda la celda que los lee como un asset del que es dueña. Este archivo
sostiene las tres cosas que el repositorio le debe a ese contrato: que la celda
esté adoptada byte por byte, que ningún cuaderno vuelva a resolver la raíz por
su cuenta, y que sin las variables el comportamiento sea el de siempre.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPOSITORY / "MIL-CREDA" / "Notebooks"

#: La celda es de la forja y viaja desde su skill. Se lee de ahí y nunca se
#: copia acá: una copia y el original se pueden desalinear, y entonces esta
#: comparación pasaría a afirmar que dos archivos de este repositorio coinciden
#: entre sí en vez de que el cuaderno lleva lo que la forja manda.
#:
#: El módulo se saltea entero cuando la forja no está, por la misma razón que
#: `test_distribute.py`: un clon de este repositorio es la implementación de un
#: paper y tiene que correr sus tests solo. Adoptar la celda es lo que se
#: verifica acá; ejecutar un cuaderno en un worker es trabajo de la forja.
_OWNED_CELL = (
    REPOSITORY.parents[1] / ".claude" / "skills" / "remote-execution"
    / "assets" / "notebook_repo_root.py"
)
if not _OWNED_CELL.is_file():
    pytest.skip(
        f"{_OWNED_CELL} is not there, so the cell this repository adopted "
        "cannot be read from its owner. Every other test here runs regardless.",
        allow_module_level=True,
    )

OWNED_TEXT = _OWNED_CELL.read_text(encoding="utf-8")


def _first_code_cell(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            return "".join(cell["source"])
    raise AssertionError(f"{path.name} has no code cell at all")


def _notebooks() -> list[Path]:
    found = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert found, f"no notebooks under {NOTEBOOKS}"
    return found


def test_every_notebook_opens_with_the_cell_the_forge_owns_byte_for_byte() -> None:
    """The kit's own notebooks open with this cell and so do these.

    Byte for byte and not "equivalent": the cell is the contract between a
    runner and the notebook it starts, and a copy that drifted by one line is a
    second implementation of the contract with nobody comparing them.

    Reachable red: reformat one character of the adopted cell in any notebook.
    """
    for path in _notebooks():
        assert _first_code_cell(path) == OWNED_TEXT, (
            f"{path.name} does not open with the forge's own cell")


def test_no_notebook_works_the_root_out_a_second_time() -> None:
    """The cell answers `WHERE IS THE REPOSITORY` once, and no later cell asks
    again.

    The old opening cell LOCATED a checkout -- cwd, its parents, an env var and
    two well-known directories -- and on a worker that resolves to a path which
    EXISTS and is wrong: the insert succeeds and the run dies several cells
    later naming a missing module, never the wrong root. A second guess left
    anywhere below would put that failure straight back.

    Reachable red: put `find_repository()` back into any cell, or compose a root
    from `Path.cwd()` outside the owned cell.
    """
    guesses = ("find_repository", "MIL_CREDA_REPO", "/kaggle/working",
               "/content", "Path.cwd()")
    for path in _notebooks():
        document = json.loads(path.read_text(encoding="utf-8"))
        code = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
        below = "\n".join("".join(cell["source"]) for cell in code[1:])
        found = sorted(guess for guess in guesses if guess in below)
        assert not found, (
            f"{path.name} works the repository root out again, below the cell "
            f"that already answered it: {found}")


def test_absent_variables_still_mean_local_and_land_on_this_repository() -> None:
    """A human opening a notebook on their own machine sees no change.

    This is the half the adoption could quietly break: with neither variable set
    the cell must resolve exactly the checkout the old one did, from the
    notebook's own directory -- which is also the directory `nbconvert` runs a
    cell in, so it is the local path AND the step's path.

    Reachable red: change `LOCAL_ROOT_DEPTH`, or make the absent-variable case
    refuse instead of falling back.
    """
    spec = importlib.util.spec_from_file_location("_owned_cell", _OWNED_CELL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.resolve_repository_root({}, cwd=NOTEBOOKS) == REPOSITORY
