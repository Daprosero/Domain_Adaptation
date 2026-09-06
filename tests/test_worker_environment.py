"""What a remote worker has to install before it can run one of these notebooks.

`__environment__` is the `environment` block a generated `run-config.json`
carries, and its whole job is to be true about two things this repository owns:
what the forge's own bootstrap probes for, and which kernel these notebooks
declare. Both halves are read here rather than restated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import MIL_CREDA_Benchmark as package

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPOSITORY / "MIL-CREDA" / "Notebooks"

#: The forge's own cell 0, where the probe lives. Read when it is there and
#: skipped when it is not, for the reason `test_distribute.py` states: a clone
#: of this repository is a paper's implementation and runs its tests alone.
_BOOTSTRAP = (
    REPOSITORY.parents[1] / ".claude" / "skills" / "remote-execution"
    / "assets" / "runner_bootstrap.py"
)

#: What `ipykernel` is in this declaration FOR: it is the distribution that
#: installs a kernelspec, and the one the notebooks name has to be its own.
_KERNELSPEC_PROVIDER = "ipykernel"


def _requirements() -> list[str]:
    return package.__environment__["install"]["requirements"]


def test_the_declaration_has_the_shape_run_config_json_accepts() -> None:
    """Every entry is a positional pip specifier and nothing else.

    The worker runs `python -m pip install <entries>` with `shell=False`, and
    it refuses any specifier beginning with `-` outright -- that one would be
    read by pip as an option rather than as a package. An empty list is refused
    too: a declared install with nothing in it is a block that costs a reading
    and answers nothing.

    Reachable red: put `--extra-index-url ...` or an empty string in the list.
    """
    install = package.__environment__["install"]
    assert set(package.__environment__) == {"install"}
    assert set(install) <= {"requirements", "indexUrl"}, (
        "run-config.json's environment.install takes these two keys and no "
        "others; anything else is read by nothing")
    requirements = install["requirements"]
    assert requirements, "an install that declares nothing is refused on the worker"
    for entry in requirements:
        assert isinstance(entry, str) and entry, entry
        assert not entry.startswith("-"), (
            f"{entry!r} would reach pip as an option, not as a package")
    assert len(set(requirements)) == len(requirements), "a name declared twice"


def test_the_declaration_provides_the_kernel_every_notebook_asks_for() -> None:
    """The half no refusal message names.

    Cell 0 resolves each notebook's own `metadata.kernelspec.name` against an
    installed kernelspec, and the refusal when it cannot says only that there is
    no kernelspec -- never which distribution would have installed one. These
    notebooks all ask for `python3`, and `ipykernel` is what puts that one on a
    worker. Nothing else in the chain pulls it: it is a test-time extra of
    `nbclient`, not a runtime dependency.

    Reachable red: drop `ipykernel` from the list, or change a notebook's kernel
    to one this declaration does not install.
    """
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert notebooks
    kernels = set()
    for path in notebooks:
        metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        kernels.add((metadata.get("kernelspec") or {}).get("name"))
    assert kernels == {"python3"}, (
        f"the notebooks ask for {sorted(kernels)}, and this declaration only "
        f"knows how to provide `python3` (through {_KERNELSPEC_PROVIDER})")
    assert _KERNELSPEC_PROVIDER in _requirements()


@pytest.mark.skipif(not _BOOTSTRAP.is_file(),
                    reason=f"{_BOOTSTRAP} is not there, so the probe this "
                           "declaration answers cannot be read from its owner")
def test_the_declaration_covers_every_module_the_bootstrap_probes_for() -> None:
    """Derived from the forge's own probe, never from a list copied beside it.

    `check_notebook_executor()` imports a fixed tuple of modules by name and
    refuses -- naming `environment.install` as the remedy -- when any of them is
    missing. That tuple is the requirement, so it is read out of the forge's
    cell 0 and compared, rather than restated here where the two could drift
    apart without a word.

    Reachable red: drop `jupyter-client` from the declaration, or add a fourth
    module to the forge's probe without declaring it.
    """
    import ast

    tree = ast.parse(_BOOTSTRAP.read_text(encoding="utf-8"))
    probed = None
    for node in tree.body:
        names = getattr(node, "targets", None) or [getattr(node, "target", None)]
        if any(getattr(name, "id", None) == "NOTEBOOK_EXECUTOR_MODULES"
               for name in names if name is not None):
            probed = ast.literal_eval(node.value)
    assert probed, "the bootstrap no longer declares NOTEBOOK_EXECUTOR_MODULES"

    # pip's own normalisation: `jupyter_client` the import and `jupyter-client`
    # the distribution are the same thing, and the declaration spells the
    # distribution because that is what reaches pip.
    declared = {entry.replace("-", "_") for entry in _requirements()}
    missing = sorted(module for module in probed if module not in declared)
    assert not missing, (
        f"cell 0 imports {missing} by name and refuses without them, and this "
        "declaration does not install them")
