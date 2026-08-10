"""Level 1 - smoke. Does it run at all, on the smallest input?

Smoke never asserts mathematics; it asserts the code is reachable and every
public entry point returns something of the right shape. Mathematical claims
belong in test_invariants.py.
"""

from __future__ import annotations

import importlib

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
    for module in MODULES:
        provenance = getattr(importlib.import_module(module), "__provenance__", None)
        assert provenance is not None, f"{module} has no __provenance__"
        assert provenance["revision"] == "research-concept-r16.md"


def test_every_module_declares_at_least_one_invariant() -> None:
    for module in MODULES:
        provenance = importlib.import_module(module).__provenance__
        assert provenance["invariants"], f"{module} declares no invariant"
        assert provenance["equations"], f"{module} points at no equation"
