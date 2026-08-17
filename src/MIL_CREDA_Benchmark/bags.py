"""Subjects built out of an image collection, and the manifest that pins them down.

MNIST, USPS and SVHN hold one label per image and no notion of a subject, so the
bags are constructed here. A bag is pure: 30 instances of one class, drawn at
random inside that class. The label belongs to the bag; no instance carries one.

The draw is stratified and balanced — 12 bags of every class in every domain —
rather than proportional. USPS and SVHN are imbalanced, and a proportional draw
can leave a class with too few source bags, which is not a matter of taste: the
local correspondence of Section 4 is undefined for a class with no source bag.

Nothing here is random at read time. Every selection is a function of the seed,
and the exact image indices are written into a manifest beside the bags, so a
later session can rebuild the identical material without depending on a library's
permutation staying the same across versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from MIL_CREDA_Benchmark import config

#: The transform CREDA's own generator applies to the digit domains: one shape and
#: one normalization for all three, so the backbone sees the same kind of input
#: whichever domain it came from.
TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


def _raw_dataset(code: str, root: Path):
    """The domain as torchvision serves it. Downloads on first use."""
    root.mkdir(parents=True, exist_ok=True)
    if code == "M":
        return datasets.MNIST(str(root), train=True, download=True, transform=TRANSFORM)
    if code == "U":
        return datasets.USPS(str(root), train=True, download=True, transform=TRANSFORM)
    if code == "S":
        return datasets.SVHN(str(root), split="train", download=True, transform=TRANSFORM)
    raise ValueError(f"unknown domain {code!r}; known: {sorted(config.DOMAINS)}")


def _labels(dataset) -> np.ndarray:
    """Every label as a flat array, however this particular class stores them."""
    for attribute in ("targets", "labels"):
        values = getattr(dataset, attribute, None)
        if values is not None:
            if torch.is_tensor(values):
                return values.numpy()
            return np.asarray(values)
    raise ValueError("the dataset exposes neither `targets` nor `labels`")


def _select(labels: np.ndarray, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Draw the images each class contributes, refusing rather than shrinking.

    A class that cannot fill its quota is an error. Returning a smaller slice
    would produce bags of uneven support and a failure downstream that reads as a
    defect of the method while being a defect of the draw.
    """
    needed = config.BAGS_PER_CLASS * config.INSTANCES_PER_BAG
    chosen: dict[int, np.ndarray] = {}
    for class_id in range(config.CLASSES):
        pool = np.flatnonzero(labels == class_id)
        if pool.size < needed:
            raise ValueError(
                f"class {class_id} offers {pool.size} images and the draw needs "
                f"{needed}; this domain cannot fund the agreed bag count"
            )
        chosen[class_id] = rng.permutation(pool)[:needed]
    return chosen


def _share(total: int, rng: np.random.Generator) -> dict[int, int]:
    """How many of each class's bags go to one role.

    A role's size need not divide by the class count, so the base share goes to
    every class and the remainder lands on classes picked by the seed. Which ones
    is recorded in the manifest, because a split nobody can reproduce is a split
    nobody can check.
    """
    base, remainder = divmod(total, config.CLASSES)
    counts = {class_id: base for class_id in range(config.CLASSES)}
    for class_id in rng.permutation(config.CLASSES)[:remainder]:
        counts[int(class_id)] += 1
    return counts


def _split_counts(rng: np.random.Generator) -> tuple[dict[int, int], dict[int, int]]:
    """How many of each class's bags go to training, and how many to selection.

    Three disjoint roles, drawn per class so every one of them covers every class:
    the local correspondence is undefined for a class with no source bag, so a
    role that dropped one would produce a failure that reads as a defect of the
    method and is a defect of the split.

    Selection is drawn second and evaluation is whatever is left, so the role the
    verdict rests on is never the one that absorbs a rounding remainder.
    """
    return _share(config.TRAIN_BAGS, rng), _share(config.VALID_BAGS, rng)


@dataclass
class BagSet:
    """One domain's bags, already materialized, split into its three roles.

    `images` holds the 3600 selected instances as one tensor. Decoding happens
    once, here, rather than inside the training loop: the run is timed, and time
    spent in PIL would be attributed to the method that happened to be running.
    """

    domain: str
    images: torch.Tensor          # (3600, 3, 32, 32)
    members: torch.Tensor         # (120, 30) rows of indices into `images`
    labels: torch.Tensor          # (120,) one class per bag
    train_idx: torch.Tensor       # (64,)  indices into `members`
    valid_idx: torch.Tensor       # (20,)  where the ceiling search looks
    eval_idx: torch.Tensor        # (36,)  never seen before the verdict
    manifest: dict

    def bag(self, position: int) -> tuple[torch.Tensor, int]:
        """The instances of one bag and its class."""
        return self.images[self.members[position]], int(self.labels[position])


class BagDataset(Dataset):
    """A role of a `BagSet`, shaped the way the harness feeds a model.

    Each item is one subject: a (30, 3, 32, 32) tensor and the bag's class. Every
    bag has the same cardinality, so the default collate stacks them without
    padding — the formulation allows cardinalities to differ, and the invariant
    sweep already exercises that; the trained comparison holds it fixed so no mask
    can quietly alter the weights inside a bag.

    `targets` is exposed because callers stratify over it.
    """

    def __init__(self, source: BagSet, positions: torch.Tensor):
        self.source = source
        self.positions = positions
        self.targets = [int(source.labels[p]) for p in positions]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int):
        position = int(self.positions[index])
        instances, label = self.source.bag(position)
        return instances, label


def build(code: str, root: Path, seed: int) -> BagSet:
    """Every bag of one domain, and the record of how they were drawn."""
    dataset = _raw_dataset(code, root)
    labels = _labels(dataset)
    rng = np.random.default_rng(seed * 1000 + ord(code))

    selected = _select(labels, rng)
    train_share, valid_share = _split_counts(rng)

    flat: list[int] = []
    members: list[list[int]] = []
    bag_labels: list[int] = []
    train_positions: list[int] = []
    valid_positions: list[int] = []
    eval_positions: list[int] = []

    for class_id in range(config.CLASSES):
        indices = selected[class_id]
        offset = len(flat)
        flat.extend(int(i) for i in indices)
        for k in range(config.BAGS_PER_CLASS):
            start = offset + k * config.INSTANCES_PER_BAG
            members.append(list(range(start, start + config.INSTANCES_PER_BAG)))
            bag_labels.append(class_id)
            position = len(members) - 1
            if k < train_share[class_id]:
                train_positions.append(position)
            elif k < train_share[class_id] + valid_share[class_id]:
                valid_positions.append(position)
            else:
                eval_positions.append(position)

    images = torch.stack([dataset[i][0] for i in flat])

    manifest = {
        "domain": code,
        "datasetName": config.DOMAINS[code],
        "seed": seed,
        "instancesPerBag": config.INSTANCES_PER_BAG,
        "bagsPerClass": config.BAGS_PER_CLASS,
        # The exact images, in order, so the material can be rebuilt without
        # trusting any library's permutation to be stable across versions.
        "imageIndices": flat,
        "bagLabels": bag_labels,
        "trainBags": train_positions,
        "validBags": valid_positions,
        "evalBags": eval_positions,
        "trainShareByClass": train_share,
        "validShareByClass": valid_share,
        "revision": config.REVISION,
    }

    return BagSet(
        domain=code,
        images=images,
        members=torch.tensor(members, dtype=torch.long),
        labels=torch.tensor(bag_labels, dtype=torch.long),
        train_idx=torch.tensor(train_positions, dtype=torch.long),
        valid_idx=torch.tensor(valid_positions, dtype=torch.long),
        eval_idx=torch.tensor(eval_positions, dtype=torch.long),
        manifest=manifest,
    )


def rebuild(manifest: dict, root: Path) -> BagSet:
    """The identical material, from the record rather than from the seed.

    This is what the manifest is for. Rebuilding by re-running the draw would
    depend on every library involved producing the same permutation it produced
    the first time, which nobody promises across versions. The indices were
    written down precisely so a later session does not have to trust that.
    """
    dataset = _raw_dataset(manifest["domain"], root)
    flat = manifest["imageIndices"]
    per_bag = manifest["instancesPerBag"]
    images = torch.stack([dataset[i][0] for i in flat])
    members = torch.arange(len(flat), dtype=torch.long).reshape(-1, per_bag)
    return BagSet(
        domain=manifest["domain"],
        images=images,
        members=members,
        labels=torch.tensor(manifest["bagLabels"], dtype=torch.long),
        train_idx=torch.tensor(manifest["trainBags"], dtype=torch.long),
        valid_idx=torch.tensor(manifest["validBags"], dtype=torch.long),
        eval_idx=torch.tensor(manifest["evalBags"], dtype=torch.long),
        manifest=manifest,
    )


def roles(bagset: BagSet) -> tuple[BagDataset, BagDataset, BagDataset]:
    """The three roles of a domain, disjoint: fitted on, selected on, judged on.

    Selection exists because the ceiling search chooses by outcome. Reading that
    outcome off the evaluation role would make the verdict report a decision it
    already made, and by an amount nobody can subtract afterwards.
    """
    return (BagDataset(bagset, bagset.train_idx),
            BagDataset(bagset, bagset.valid_idx),
            BagDataset(bagset, bagset.eval_idx))


def write_manifest(path: Path, **entries) -> Path:
    """Persist a manifest so a checkpoint never travels without its material."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path
