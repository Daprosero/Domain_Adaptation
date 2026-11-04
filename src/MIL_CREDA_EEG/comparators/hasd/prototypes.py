"""HASD §2.3: label-free per-bag centroids with explicit correspondence."""
from __future__ import annotations

import torch

__provenance__ = {
    "revision": "experiments-plan-experimental-mil-creda-sobre-balladeer-v50.md",
    "sections": ["7.1", "7.4"],
    "experiments": ["A-nivel", "B-tarea", "C-dispositivo"],
    "invariants": ["hasd_prototypes_are_label_free_centroids_with_fixed_membership"],
}

PROTOTYPES = 10  # HASD Table2; cap at available windows.
LLOYD_ITERATIONS = 20  # Unspecified in HASD.


@torch.no_grad()
def cluster_assignments(h: torch.Tensor, k: int = PROTOTYPES) -> torch.Tensor:
    """Deterministic farthest-first Lloyd clustering of initial target features.

    Membership is fixed during adaptation: recomputing it on transformed features
    would make PACR compare different instances/centroids. Duplicate centroids
    may leave empty clusters, which are omitted by centroids().
    """
    if h.ndim != 2 or h.shape[0] == 0 or not torch.isfinite(h).all():
        raise ValueError("clustering requires finite nonempty (instances, features)")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    k = min(k, len(h))
    if k == len(h):
        return torch.arange(k, device=h.device)
    selected = [int(((h - h.mean(0)) ** 2).sum(1).argmax())]
    for _ in range(1, k):
        distances = torch.cdist(h, h[selected]).square().min(1).values
        distances[selected] = -1
        selected.append(int(distances.argmax()))
    c = h[selected].clone()
    previous = None
    for _ in range(LLOYD_ITERATIONS):
        ids = torch.cdist(h, c).argmin(1)
        if previous is not None and torch.equal(previous, ids):
            break
        for j in range(k):
            mask = ids == j
            if mask.any():
                c[j] = h[mask].mean(0)
        previous = ids
    return ids


def centroids(h: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    """HASD §2.3: means, gradients through every member; no label input."""
    if h.ndim != 2 or membership.shape != (len(h),) or len(h) == 0:
        raise ValueError("membership must give one cluster for every instance")
    return torch.stack([h[membership == j].mean(0) for j in membership.unique(sorted=True)])
