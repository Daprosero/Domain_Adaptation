"""Deterministic fixtures. Fixed seed, no data, no network.

The seed is chosen per run so that levels 2 and 3 vary between runs while
staying reproducible within one.
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 20260810
TOL = 1e-10


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def make_bag(
    rng: np.random.Generator, n_instances: int, dimension: int, shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """One bag: instance embeddings and in-bag relevance weights on the simplex.

    The weights come from Eq. (15) applied to arbitrary logits, so they are a
    legitimate in-bag distribution rather than a uniform placeholder.
    """
    from MIL_CREDA.attention import bag_weights

    H = rng.normal(loc=shift, scale=1.0, size=(n_instances, dimension))
    return H, bag_weights(rng.normal(size=n_instances))


def make_bags(
    rng: np.random.Generator,
    n_bags: int,
    dimension: int,
    shift: float = 0.0,
    max_instances: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A collection of bags of deliberately varying cardinality."""
    return [
        make_bag(rng, int(rng.integers(1, max_instances + 1)), dimension, shift)
        for _ in range(n_bags)
    ]


def simplex_rows(rng: np.random.Generator, n_rows: int, n_classes: int) -> np.ndarray:
    """Class score vectors: one row per bag, each on the probability simplex."""
    raw = rng.dirichlet(np.ones(n_classes), size=n_rows)
    return np.asarray(raw, dtype=float)
