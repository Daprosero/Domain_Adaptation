"""Every module under `src/CREDA/` declares what it implements, and backs it.

The repository's rule is blanket: each module carries a `__provenance__`, and
each id in its `invariants` has a matching `test_<id>` somewhere under
`tests/`. Two legitimate shapes answer it, and the difference between them is
the whole point of this file:

* **a mathematical object** -- `revision`, `sections`, `equations`,
  `invariants`. `sample_adaptation.py` and `bounds.py` are of this kind, and so
  is `models.py`, whose `CREDALoss` is the adaptation term itself.
* **a module that implements no equation** -- `"revision": None`, NO
  `equations` key, and a `claims_no_equation` string carrying the reason.
  `two_phase.py` established it (a training protocol is not a term) and
  `tests/test_creda_two_phase.py::test_the_provenance_claims_no_equation`
  holds that one module to it. The harnesses and the IO layer are of this kind
  too: a provenance says "this module implements these equations", and
  stamping one on a module that implements none empties the check for every
  module that does.

`test_a_module_that_claims_no_equation_declares_neither_a_revision_nor_equations`
and `test_a_module_that_declares_equations_names_the_revision_it_implements`
hold the two shapes apart, over the whole package, so a later module cannot be
added in a third shape that satisfies neither.

The cross-cutting tests read the sources with `ast` rather than importing
them, for the same reason the forge's own reader does: a module that cannot be
imported in this environment still has a declaration to check, and importing
`training_pipeline` costs scikit-image, timm, pandas and matplotlib.

The per-module tests below are here because their modules had no test of their
own anywhere -- `artifacts.py` and `toy_plots.py` were imported by nothing
under `tests/`, and `training_pipeline.py` only by the schedule's copy check.
An invariant declared with nothing to hold it reads as covered and is worse
than one left out, so each id those three declare is either an older test that
already existed or one of these.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "src" / "CREDA"
TESTS = Path(__file__).resolve().parent


COMPARATORS = PACKAGE / "comparators"


def _is_vendored(path: Path) -> bool:
    """A comparator file copied verbatim from a paper's official repository.

    Those files are third-party code kept byte-for-byte (local edits marked
    `# LOCAL EDIT:`); stamping a `__provenance__` on them would break that.
    Their own declaration is the citation header that
    `tests/test_comparators_official.py::test_vendored_files_carry_citation_headers`
    demands. The adapters around them are ours and stay under the rule.
    """
    return path.parent.parent == COMPARATORS and path.name.startswith("upstream_")


def _modules() -> list[Path]:
    return [path for path in sorted(PACKAGE.rglob("*.py"))
            if path.name != "__init__.py" and not _is_vendored(path)]


def _provenance(path: Path) -> dict | None:
    """The module's own `__provenance__`, read without importing it.

    Both assignment forms, because `__provenance__: dict = {...}` is an
    `ast.AnnAssign` and a reader walking only `ast.Assign` would answer `None`
    for it -- which every caller here would read as "declares nothing", the
    opposite of what the file says.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__provenance__"
                for target in node.targets):
            return ast.literal_eval(node.value)
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__provenance__"
                and node.value is not None):
            return ast.literal_eval(node.value)
    return None


def _test_function_names() -> set[str]:
    names: set[str] = set()
    for file in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


def _calls_in(path: Path, function: str) -> set[str]:
    """The names called inside one top-level function, read from source."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == function)
    return {getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            for call in ast.walk(target) if isinstance(call, ast.Call)}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    return imported


# ============================================ 1. the package-wide declaration


def test_the_package_holds_the_modules_this_file_speaks_for():
    """A guard on the guards: an empty enumeration passes every test below.

    If the package moved or `rglob` stopped matching, the three cross-cutting
    tests would iterate over nothing and report green while checking no
    module at all.
    """
    found = {path.name for path in _modules()}
    assert PACKAGE.is_dir(), f"{PACKAGE} is not a directory"
    assert found >= {
        "artifacts.py", "bounds.py", "models.py", "sample_adaptation.py",
        "schedules.py", "toy_experiment.py", "toy_plots.py",
        "training_pipeline.py", "two_phase.py",
    }, f"the package no longer holds the modules this file was written for: {found}"


def test_only_verbatim_comparator_files_are_exempt_from_the_rule():
    """The exemption reaches the vendored files and nothing else.

    Every exempt file must carry the citation header, and every comparator
    adapter must still be enumerated, so widening `_is_vendored` by mistake
    cannot silently drop our own code from the check.
    """
    exempt = [path for path in sorted(PACKAGE.rglob("*.py")) if _is_vendored(path)]
    assert exempt, "no vendored comparator file found: the exemption guards nothing"
    for path in exempt:
        head = path.read_text(encoding="utf-8")[:2500]
        assert "Repository:" in head and "Commit:" in head, path
    enumerated = set(_modules())
    adapters = sorted(COMPARATORS.glob("*/adapter.py"))
    assert adapters and all(path in enumerated for path in adapters)


def test_every_module_under_the_package_declares_a_provenance():
    missing = [str(path.relative_to(REPOSITORY)) for path in _modules()
               if _provenance(path) is None]
    assert missing == [], f"modules with no __provenance__: {missing}"


def test_every_declared_invariant_has_a_test_somewhere_under_tests():
    """The pairing the forge's `fidelity.invariantsWithoutTest` also reads.

    Checked here as well because `tests/` is the fail-closed source of truth:
    a declared invariant with no test and a tested property with no
    declaration look identical to a green run otherwise.
    """
    tests = _test_function_names()
    untested = sorted(
        f"{path.relative_to(REPOSITORY)}::{invariant}"
        for path in _modules()
        for invariant in (_provenance(path) or {}).get("invariants", [])
        if f"test_{invariant}" not in tests)
    assert untested == [], f"declared invariants with no test: {untested}"


def test_a_module_that_claims_no_equation_declares_neither_a_revision_nor_equations():
    """`two_phase.py`'s shape, held over the whole package.

    A provenance says "this module implements these equations". A module that
    implements none says so, once, and must not also name a revision or an
    equation list -- either one would make it read as a term.
    """
    offenders = []
    for path in _modules():
        provenance = _provenance(path) or {}
        if "claims_no_equation" not in provenance:
            continue
        reason = provenance["claims_no_equation"]
        if (provenance.get("revision") is not None
                or "equations" in provenance
                or not isinstance(reason, str) or not reason.strip()):
            offenders.append(str(path.relative_to(REPOSITORY)))
    assert offenders == [], (
        "modules claiming no equation while declaring a revision, an "
        f"equation list, or no reason: {offenders}")


def test_a_module_that_declares_equations_names_the_revision_it_implements():
    """And the other way round: a term says which document it is bound to.

    The revision is read as a filename and never checked against the latest
    one, deliberately: `bounds.py` declares `research-concept-r25.md` and its
    module docstring argues that choice. Which revision a module is bound to
    is the forge's `fidelity.drift` to report, not this test's to demand.
    """
    offenders = []
    for path in _modules():
        provenance = _provenance(path) or {}
        if "equations" not in provenance:
            continue
        revision = provenance.get("revision")
        if (not isinstance(revision, str) or not revision.endswith(".md")
                or not provenance.get("sections")):
            offenders.append(str(path.relative_to(REPOSITORY)))
    assert offenders == [], (
        "modules declaring equations without a revision filename and its "
        f"sections: {offenders}")


# ========================================== 2. `artifacts.py` -- the IO layer


def test_the_repository_root_is_only_a_directory_carrying_both_markers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One marker is not enough, and the configured root wins over the walk.

    `resolve_repository_root` walks upwards from where it is asked, so half a
    repository -- a `src/CREDA` with no `pyproject.toml`, which is what an
    extracted subdirectory or a stray copy looks like -- must not be accepted
    as the root every artifact path is then built from.

    `DOMAIN_ADAPTATION_ROOT` is removed first: it is read before the walk, so
    a value left in the environment would answer every call below and the
    test would pass without the walk ever running.
    """
    from CREDA import artifacts

    monkeypatch.delenv("DOMAIN_ADAPTATION_ROOT", raising=False)
    half = (tmp_path / "half").resolve()
    (half / "src" / "CREDA").mkdir(parents=True)
    elsewhere = (tmp_path / "elsewhere").resolve()
    elsewhere.mkdir()

    with pytest.raises(FileNotFoundError, match="DOMAIN_ADAPTATION_ROOT"):
        artifacts.resolve_repository_root(half)
    with pytest.raises(FileNotFoundError, match="DOMAIN_ADAPTATION_ROOT"):
        artifacts.resolve_repository_root(elsewhere)

    # The second marker is what makes it a root.
    (half / "pyproject.toml").write_text("", encoding="utf-8")
    assert artifacts.resolve_repository_root(half) == half

    # And the configured root is read before the walk starts.
    monkeypatch.setenv("DOMAIN_ADAPTATION_ROOT", str(half))
    assert artifacts.resolve_repository_root(elsewhere) == half


def test_an_unsupported_backbone_or_dataset_is_refused_rather_than_guessed():
    """Refused at the call, before a path is built from a name nobody declared.

    A guessed backbone directory resolves to a CSV that does not exist and the
    reader would then report a missing file, which sends whoever reads it
    looking for results instead of for the typo.
    """
    from CREDA import artifacts

    with pytest.raises(ValueError, match="Unsupported backbone"):
        artifacts.load_dataset_results("resnet99", artifacts.DATASETS[0])
    with pytest.raises(ValueError, match="Unsupported dataset"):
        artifacts.load_dataset_results(artifacts.BACKBONES[0], "NotADataset")
    with pytest.raises(ValueError, match="Unsupported backbone"):
        artifacts.checkpoint_path(Path("/models"), "resnet99", "creda", "a", "b")


def test_a_baseline_checkpoint_needs_a_component_and_the_others_carry_none():
    """The two filename shapes `run_all_models` actually wrote.

    A baseline is saved as two files, the extractor and the classifier, so a
    path with no component would name neither of them.
    """
    from CREDA import artifacts

    models_root = Path("/models")
    with pytest.raises(ValueError, match="component"):
        artifacts.checkpoint_path(models_root, "resnet18", "baseline", "MNIST", "USPS")
    with pytest.raises(ValueError, match="component"):
        artifacts.checkpoint_path(models_root, "resnet18", "baseline", "MNIST",
                                  "USPS", component="weights")

    extractor = artifacts.checkpoint_path(models_root, "resnet18", "baseline",
                                          "MNIST", "USPS", component="F")
    assert extractor == models_root / "resnet18" / "resnet18_Baseline_MNIST_USPS_F.pth"
    creda = artifacts.checkpoint_path(models_root, "resnet18", "creda", "MNIST", "USPS")
    assert creda == models_root / "resnet18" / "resnet18_CREDA_MNIST_USPS_weights.pth"


# ======================================= 3. `toy_plots.py` -- the drawing half


def test_the_drawing_module_reads_the_grid_it_draws_from_the_experiment():
    """The same objects, not equal copies. Two lists already diverged once.

    `toy_plots`' own docstring records the consequence: the figures showed an
    arm the table did not have, and the arm the table reported as best had no
    figure. Identity is what makes that unrepeatable -- a hand-written second
    copy would be `==` and would not be `is`.

    Only the mutable containers are checked. An `is` on a small integer or an
    interned string passes for a copy too, which would be an assertion that
    cannot fail.
    """
    from CREDA import toy_experiment, toy_plots

    shared = ("FILAS", "TABLA", "POS_FILAS", "FAMILIAS", "NOMBRE_FILA",
              "COLOR", "MARCA")
    for name in shared:
        drawn = getattr(toy_plots, name)
        computed = getattr(toy_experiment, name)
        assert isinstance(drawn, (list, dict)), (
            f"{name} is a {type(drawn).__name__}; identity would pass for a "
            "copy of it and this test would check nothing")
        assert drawn is computed, (
            f"toy_plots.{name} is not toy_experiment.{name} -- the grid has "
            "been written out a second time and the two can now disagree")


def test_nothing_in_the_drawing_module_imports_torch():
    """It reads the saved artifact, so its labels are numpy arrays.

    The module says so where it matters -- `panel_embedding` used to call
    `.numpy()` on them, which is what was needed while the same notebook both
    trained and drew. A `torch` import here is the signal that something began
    computing in the drawing half again.
    """
    from CREDA import toy_plots

    imported = _imported_modules(Path(toy_plots.__file__))
    offenders = sorted(name for name in imported
                       if name == "torch" or name.startswith("torch."))
    assert offenders == [], f"toy_plots imports {offenders}"
    assert "torch" not in vars(toy_plots)


# =============================== 4. `training_pipeline.py` -- the campaign loop


def test_the_campaign_builds_its_adaptation_term_from_the_method_module():
    """`train_creda` calls `CREDALoss`; it never writes the term inline.

    This is the claim that lets the module declare no equation at all. An
    inline copy of the term starts as a defensible simplification and ends as
    a silent fork: the module would keep declaring nothing while computing
    mathematics of its own.

    Reachable red, two ways: replace the `CREDALoss(...)` construction with an
    inline score, or copy one of the term's own helpers into this module.
    """
    source = REPOSITORY / "src" / "CREDA" / "training_pipeline.py"
    called = _calls_in(source, "train_creda")
    assert "CREDALoss" in called, (
        "train_creda no longer constructs CREDALoss -- if it computes the "
        "term itself, this module implements an equation and may not declare "
        "that it implements none")
    assert "creda_ramp" in called, "train_creda no longer applies the schedule"

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    forked = defined & {"_renyi_entropy_order_2", "_mix_kernel_concat",
                        "_gaussian_kernel", "_kernel_embedding",
                        "_marginal_score", "_compute_sigma"}
    assert forked == set(), (
        f"training_pipeline defines {sorted(forked)} -- the term's own "
        "helpers have been copied out of CREDA.models")
