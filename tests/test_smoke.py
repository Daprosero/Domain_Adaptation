"""Level 1 - smoke. Does it run at all, on the smallest input?

Smoke never asserts mathematics; it asserts the code is reachable and every
public entry point returns something of the right shape. Mathematical claims
belong in test_invariants.py.
"""

from __future__ import annotations

import importlib

import torch

from MIL_CREDA.objective import total_objective

MODULES = [
    "MIL_CREDA.kernels",
    "MIL_CREDA.renyi",
    "MIL_CREDA.attention",
    "MIL_CREDA.bag_kernel",
    "MIL_CREDA.confidence",
    "MIL_CREDA.conditional",
    "MIL_CREDA.global_term",
    "MIL_CREDA.local_term",
    "MIL_CREDA.objective",
]


def test_package_imports() -> None:
    for module in MODULES:
        assert importlib.import_module(module) is not None


def test_every_module_declares_provenance() -> None:
    """Every module names the same revision its own package is bound to.

    Read off `MIL_CREDA.__implementation__` rather than written here as a
    literal. The literal was `research-concept-r25.md`, and it went stale the
    moment the package moved to r29: a sentence ages without anything failing,
    and a hardcoded revision is that defect with an assertion wrapped around
    it -- it reddens on the legitimate re-bind and says nothing about the
    thing it was watching.

    It still crosses a join, so red stays reachable: the bound revision lives
    in `src/MIL_CREDA/__init__.py` and each module's in its own file, so a
    re-bind that reaches nine modules and misses the tenth fails here. WHICH
    revision is the current one is `verify`'s question (`fidelity`, against
    the newest published revision of the family), never this one's.
    """
    bound = importlib.import_module("MIL_CREDA").__implementation__["revision"]
    for module in MODULES:
        provenance = getattr(importlib.import_module(module), "__provenance__", None)
        assert provenance is not None, f"{module} has no __provenance__"
        assert provenance["revision"] == bound, (
            f"{module} is bound to {provenance['revision']!r} while its package "
            f"declares {bound!r}; a module and its own package cannot name two"
        )


def test_every_module_declares_at_least_one_invariant() -> None:
    for module in MODULES:
        provenance = importlib.import_module(module).__provenance__
        assert provenance["invariants"], f"{module} declares no invariant"
        assert provenance["equations"], f"{module} points at no equation"


def test_the_objective_carries_a_gradient_to_its_terms() -> None:
    """The one thing the backend conversion exists for: Eq. (39) is trainable.

    Smoke, not mathematics — it asserts nothing about the value of the objective,
    only that differentiating it reaches the three terms it is built from. That
    is the difference between an implementation that can be trained and one that
    can only be evaluated, and it is worth failing loudly on: a stray cast to a
    Python float anywhere upstream silently removes it, and every other test in
    this suite would still pass.
    """
    supervised, glob, loc = (
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (1.7, 0.4, 0.6)
    )
    objective = total_objective(supervised, glob, loc, 0.5, 0.25)
    assert objective.requires_grad
    objective.backward()
    assert supervised.grad is not None and float(supervised.grad) == 1.0
    assert glob.grad is not None and float(glob.grad) == 0.5
    assert loc.grad is not None and float(loc.grad) == 0.25
