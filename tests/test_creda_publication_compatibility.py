"""The CREDA publication notebooks still resolve everything they import.

The toyset cleanup removed the old toy benchmark from `src/CREDA`. The three
notebooks of the published paper -- `Results_Generator`, `Plots_Generator` and
`Tables_Generator` under `CREDA/Notebooks/` -- must not lose a single function
they reach, directly or through the modules those import. This file derives
that dependency set from the notebooks themselves rather than from a list
written here, so a notebook that starts importing something new is covered
the day it does:

1. every repository-local `import` in a notebook code cell is a ROOT;
2. the local modules those roots import, transitively, are the CLOSURE;
3. every name a notebook imports still resolves to the object it names, and
   no module in the closure reaches the toy code at all.

Notebook cells may carry IPython magics (`%...`, `!...`); those lines are
dropped before parsing, and any cell that still fails to parse fails the test
instead of being skipped. Nothing here downloads data or trains: importing a
module is the only side effect.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
NOTEBOOKS = tuple(REPOSITORY / "CREDA" / "Notebooks" / f"{name}.ipynb"
                  for name in ("Results_Generator", "Plots_Generator",
                               "Tables_Generator"))

#: Toy and tutorial modules: none of them may be reachable from the paper.
TOY_MODULES = ("CREDA.toy_experiment", "CREDA.toy_plots",
               "CREDA.tutorial_visuals", "MIL_CREDA_Benchmark")


def _local_path(module: str) -> Path | None:
    parts = module.split(".") if module else []
    if not parts:
        return None
    for candidate in (SOURCE.joinpath(*parts).with_suffix(".py"),
                      SOURCE.joinpath(*parts, "__init__.py")):
        if candidate.is_file():
            return candidate
    return None


def _code_cells(notebook: Path) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    out = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).split("\n")
        out.append("\n".join(line for line in lines
                             if not line.lstrip().startswith(("%", "!"))))
    return out


def _imports(tree: ast.AST, package: str | None = None):
    """`(module, names)` per import; relative imports resolved on `package`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = (package or "").split(".")
                base = base[:len(base) - node.level + 1]
                module = ".".join(base + ([module] if module else []))
            yield module, [alias.name for alias in node.names]


def _roots() -> dict[str, set[str]]:
    roots: dict[str, set[str]] = {}
    for notebook in NOTEBOOKS:
        for code in _code_cells(notebook):
            tree = ast.parse(code, filename=str(notebook))
            for module, names in _imports(tree):
                if _local_path(module):
                    roots.setdefault(module, set()).update(names or ())
    return roots


def _closure(roots) -> dict[str, Path]:
    seen: dict[str, Path] = {}
    todo = list(roots)
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        path = _local_path(module)
        seen[module] = path
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        parts = module.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported, names in _imports(tree, package):
            for candidate in [imported] + [f"{imported}.{n}" for n in names or ()]:
                if _local_path(candidate) and candidate not in seen:
                    todo.append(candidate)
    return seen


ROOTS = _roots()
CLOSURE = _closure(ROOTS)


def test_the_notebooks_are_there_and_their_roots_were_found():
    """A guard on the guards: an empty enumeration would pass everything."""
    for notebook in NOTEBOOKS:
        assert notebook.is_file(), notebook
    assert {"CREDA.artifacts", "CREDA.models",
            "CREDA.training_pipeline"} <= set(ROOTS), sorted(ROOTS)
    assert all(ROOTS[module] for module in ROOTS), ROOTS
    assert set(ROOTS) <= set(CLOSURE)


def test_no_module_the_paper_reaches_is_toy_code():
    reached = sorted(module for module in CLOSURE
                     if module.startswith(TOY_MODULES))
    assert reached == [], reached
    for module, path in CLOSURE.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {name for name, _ in _imports(tree, module)}
        assert not any(name.startswith(TOY_MODULES) for name in imported), (
            module, sorted(imported))


@pytest.mark.parametrize("module", sorted(CLOSURE))
def test_every_module_of_the_closure_imports(module):
    loaded = importlib.import_module(module)
    assert Path(loaded.__file__).resolve() == CLOSURE[module].resolve()


@pytest.mark.parametrize("qualified", sorted(
    f"{module}:{name}" for module, names in ROOTS.items() for name in names))
def test_every_name_a_notebook_imports_still_resolves(qualified):
    module, name = qualified.split(":")
    target = getattr(importlib.import_module(module), name)
    if callable(target):
        inspect.signature(target)
