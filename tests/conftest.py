"""Deterministic fixtures. Fixed seed, no data, no network.

Every value handed to a test is a torch tensor, because the implementation
computes with torch: a fixture returning arrays would let the suite assert over
objects the trained model never touches.

The numbers are still DRAWN by NumPy's generator, and that is deliberate.
`test_audit.py` measures rates over a fixed block of 200 configurations, and
those rates are the published evidence for two findings the deliberation has
already adopted. Reseeding with a different generator would draw a different
block, and any change in a measured rate would then be indistinguishable from an
effect of the backend — which is the one thing this conversion has to be able to
rule out. Sampling is not computation: nothing in this file evaluates a single
equation of the proposal. Every equation is evaluated in torch, by `src/`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from MIL_CREDA import DTYPE, as_tensor
from MIL_CREDA_Benchmark import config as _config

SEED = 20260810
TOL = 1e-10


class Sampler:
    """A seeded source of tensors, mirroring the generator API it draws from."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size=None) -> torch.Tensor:
        return _tensor(self._rng.normal(loc=loc, scale=scale, size=size))

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None) -> torch.Tensor:
        return _tensor(self._rng.uniform(low=low, high=high, size=size))

    def integers(self, low: int, high: int | None = None, size=None):
        drawn = self._rng.integers(low, high, size=size)
        return int(drawn) if size is None else torch.as_tensor(drawn, dtype=torch.long)

    def dirichlet(self, n_classes: int, size: int | None = None) -> torch.Tensor:
        return _tensor(self._rng.dirichlet(np.ones(n_classes), size=size))

    def permutation(self, n: int) -> torch.Tensor:
        return torch.as_tensor(self._rng.permutation(n), dtype=torch.long)


@pytest.fixture
def rng() -> Sampler:
    return Sampler(SEED)


def one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Bag labels as the one-hot rows Eq. (18) expects."""
    return torch.nn.functional.one_hot(labels.reshape(-1), n_classes).to(DTYPE)


def close(value, expected, *, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """`torch.isclose` over whatever the caller has, at the tolerances it names."""
    left = as_tensor(value)
    return bool(torch.isclose(left, _like(expected, left), atol=atol, rtol=rtol))


def allclose(value, expected, *, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """`torch.allclose`, same convenience, for whole tensors."""
    left = as_tensor(value)
    return bool(torch.allclose(left, _like(expected, left), atol=atol, rtol=rtol))


def make_bag(
    rng: Sampler, n_instances: int, dimension: int, shift: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """One bag: instance embeddings and in-bag relevance weights on the simplex.

    The weights come from Eq. (15) applied to arbitrary logits, so they are a
    legitimate in-bag distribution rather than a uniform placeholder.
    """
    from MIL_CREDA.attention import bag_weights

    H = rng.normal(loc=shift, scale=1.0, size=(n_instances, dimension))
    return H, bag_weights(rng.normal(size=n_instances))


def make_bags(
    rng: Sampler,
    n_bags: int,
    dimension: int,
    shift: float = 0.0,
    max_instances: int = 5,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A collection of bags of deliberately varying cardinality."""
    return [
        make_bag(rng, rng.integers(1, max_instances + 1), dimension, shift)
        for _ in range(n_bags)
    ]


def simplex_rows(rng: Sampler, n_rows: int, n_classes: int) -> torch.Tensor:
    """Class score vectors: one row per bag, each on the probability simplex."""
    return rng.dirichlet(n_classes, size=n_rows)


def _tensor(drawn) -> torch.Tensor:
    return torch.as_tensor(drawn, dtype=DTYPE)


def _like(expected, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(expected, torch.Tensor):
        return expected
    return torch.as_tensor(expected, dtype=reference.dtype, device=reference.device)


# --------------------------------------------------- the owner's own product tree

#: The real product folder, frozen at import time and never asked for again.
#:
#: `config.PRODUCT` is what almost every test here redirects, so asking `config`
#: for it inside the fixture would hand back whichever scratch the test under
#: observation installed --- and a guard that watches the scratch watches
#: nothing. This is the constant as `config` was imported with it, which is the
#: tree the owner's own runs live in.
_PRODUCT_TREE = _config.PRODUCT


def _product_fingerprint() -> dict[str, tuple[int, int] | None]:
    """`{path relative to the product tree: (size, mtime_ns)}`; directories `None`.

    **Directories are entries in their own right and not an implementation
    detail.** `campaign()` calls `results_for(...).mkdir()` and
    `models_for(...).mkdir()` BEFORE any of its three refusals, so the write a
    refused campaign actually performs is a directory and nothing else: a
    fingerprint over files alone walks straight past the defect this exists to
    catch.

    **Size AND `mtime_ns`, never size alone.** A file rewritten with the same
    number of bytes --- a marker file retouched, a `runs.jsonl` replaced by
    another run of the same shape --- is what a size-only reading cannot see,
    and that is the ordinary shape of an overwrite rather than an exotic one.

    No hashing, and `os.scandir` rather than `os.walk`: gigabytes of
    checkpoints live under here and this runs twice per test, so a guard that
    costs a full read --- or even a `Path` per entry --- is a guard somebody
    switches off.
    """
    fingerprint: dict[str, tuple[int, int] | None] = {}
    if not _PRODUCT_TREE.is_dir():
        return fingerprint
    pending = [("", str(_PRODUCT_TREE))]
    while pending:
        relative, directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            key = f"{relative}/{entry.name}" if relative else entry.name
            try:
                if entry.is_dir(follow_symlinks=False):
                    fingerprint[key] = None
                    pending.append((key, entry.path))
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                # Gone between the listing and the stat. Recorded as a reading
                # of its own rather than skipped: a file a test deleted is
                # exactly what this is looking for.
                fingerprint[key] = (-1, -1)
                continue
            fingerprint[key] = (stat.st_size, stat.st_mtime_ns)
    return fingerprint


@pytest.fixture(autouse=True)
def product_tree_is_read_only():
    """No test may leave a mark on the owner's own product tree. Autouse, so
    there is nothing for a test to remember to opt into.

    Every destination in this repository can be redirected and nearly every
    test redirects it. The ones that did not were safe only because a refusal
    fired before the write --- and safety that depends on another guard holding
    is not safety: the day one of those refusals was switched off to check it
    was reachable, two tests ran a real campaign into `MIL-CREDA/Results/` and
    overwrote `runs.jsonl`, `summary.json`, `shard.json`, `Probe_results.json`
    and six manifests. The suite stayed green throughout, because nothing was
    watching the tree.

    This watches the tree. It cannot be reasoned around, it asks no
    cooperation from the test it observes, and it names the paths that moved.
    """
    before = _product_fingerprint()
    yield
    after = _product_fingerprint()
    if after == before:
        return
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(key for key in set(before) & set(after)
                     if before[key] != after[key])
    raise AssertionError(
        f"this test wrote into the owner's own product tree at {_PRODUCT_TREE}.\n"
        f"  added:   {added}\n"
        f"  removed: {removed}\n"
        f"  changed: {changed}\n"
        "  Redirect every destination it reaches before the call: "
        "`config.PRODUCT`, `config.RESULTS`, `config.MODELS` and the two "
        "ceiling records are each their own root, and redirecting one does not "
        "redirect the others."
    )
