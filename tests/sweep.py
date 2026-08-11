"""Randomized configuration sweep shared by the audit and the remedy checks.

One fixed size for every claim: a property surviving SWEEP_SIZE independent
configurations belongs to the formulation, not to a seed.

SWEEP_BASE offsets the block by the run's seed so separate runs audit disjoint
configurations. Without it, agreement between runs would confirm determinism
and nothing else.

`configuration` builds one full per-class MIL-CREDA instance: bags of varying
cardinality in both domains, a bandwidth, a local temperature, class scores of
varying sharpness, and everything Eqs. (23)-(38) need. What varies is what the
proposal allows to vary, so a claim holding only for one shape shows up as a
rate below the full sweep.

Every quantity here is computed by `src/` in torch; the configuration is drawn by
the seeded sampler of `conftest`, whose docstring says why the draws themselves
were left on NumPy's generator across the backend conversion.
"""

from __future__ import annotations

import torch
from conftest import Sampler

from MIL_CREDA import DTYPE
from MIL_CREDA.attention import bag_weights
from MIL_CREDA.bag_kernel import bag_kernel, bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix, weighted_blocks
from MIL_CREDA.confidence import confidences
from MIL_CREDA.local_term import class_correspondence, local_distance

SWEEP_SIZE = 200
SWEEP_BASE = 20260810 * 10000


def _bags(rng: Sampler, count: int, dimension: int, shift: float, spread: float):
    bags = []
    for _ in range(count):
        m = rng.integers(1, 7)
        H = rng.normal(loc=shift, scale=spread, size=(m, dimension))
        bags.append((H, bag_weights(rng.normal(size=m))))
    return bags


def _sharpened(rng: Sampler, rows: int, n_classes: int, sharpness: float):
    """Class scores of controlled concentration, from uniform to near one-hot."""
    return torch.softmax(rng.normal(size=(rows, n_classes)) * sharpness, dim=1)


def configuration(index: int) -> dict:
    """One reproducible configuration of the proposal's objects."""
    rng = Sampler(SWEEP_BASE + index)

    n_classes = rng.integers(2, 6)
    dimension = rng.integers(2, 8)
    n_source = rng.integers(1, 9)
    n_target = rng.integers(1, 9)
    sigma = float(rng.uniform(0.2, 3.0))
    tau = float(rng.uniform(0.05, 5.0))
    shift = float(rng.uniform(0.0, 4.0))
    spread = float(rng.uniform(0.2, 2.5))
    sharpness = float(rng.uniform(0.0, 12.0))

    source = _bags(rng, n_source, dimension, 0.0, spread)
    target = _bags(rng, n_target, dimension, shift, spread)

    K_ss = bag_kernel_matrix(source, source, sigma)
    K_st = bag_kernel_matrix(source, target, sigma)
    K_tt = bag_kernel_matrix(target, target, sigma)

    target_scores = _sharpened(rng, n_target, n_classes, sharpness)
    weights = confidences(target_scores)

    K_s_c, K_st_c, K_t_c = conditional_blocks(
        K_ss, K_st, K_tt, torch.arange(n_source), torch.arange(n_target)
    )
    K_st_w, K_t_w = weighted_blocks(K_st_c, K_t_c, weights)
    K_mix_w = mixed_matrix(K_s_c, K_st_w, K_t_w)

    squared_distances = []
    for column, (H_t, w_t) in enumerate(target):
        cross = K_st[:, column]
        pi = class_correspondence(cross, torch.arange(n_source), tau)
        squared_distances.append(
            local_distance(bag_kernel(H_t, w_t, H_t, w_t, sigma), cross, K_ss, pi)
        )

    source_scores = _sharpened(rng, max(n_source, 1), n_classes, sharpness)
    source_labels = rng.integers(0, n_classes, size=max(n_source, 1))

    return {
        "index": index,
        "n_classes": n_classes,
        "n_source": n_source,
        "n_target": n_target,
        "sigma": sigma,
        "tau": tau,
        "shift": shift,
        "sharpness": sharpness,
        "K_s_c": K_s_c,
        "K_st_w": K_st_w,
        "K_t_w": K_t_w,
        "K_mix_w": K_mix_w,
        "target_weights": weights,
        "squared_distances": (
            torch.stack(squared_distances)
            if squared_distances
            else torch.zeros(0, dtype=DTYPE)
        ),
        "source_scores": source_scores,
        "source_labels": source_labels,
    }


def sweep(size: int = SWEEP_SIZE):
    for index in range(size):
        yield configuration(index)
