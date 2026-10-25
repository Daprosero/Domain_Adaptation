"""The dataset shape a toy domain is served in.

`ToyDataset` is the container `toy_bags` fills and `bags.build` indexes:
`.targets` is the per-point class `bags._labels` reads, `.subjects` the
per-point subject id `bags._select` draws from, and `__getitem__` returns a
`(tensor, label)` pair. The nine-code generator (swiss roll, tori, moons and
their translated and rotated targets) was retired with the image benchmark;
the toyset's own families live in `toy_bags`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    """One toy domain: a point cloud with a class label and a subject id per
    point.

    Shaped exactly like `bags._labels`/`bags._select` need: `.targets` is the
    array `_labels` reads, `.subjects` is the per-point subject id `_select`
    draws from when `bags.build` hands it over, and `__getitem__` returns a
    `(tensor, label)` pair, so `bags.build` needs nothing else to index this.
    """

    def __init__(self, points: np.ndarray, targets: np.ndarray, subjects: np.ndarray):
        self.points = points.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.subjects = subjects.astype(np.int64)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.points[index]), int(self.targets[index])
