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
from MIL_CREDA_Benchmark import toy

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
    """The domain as torchvision serves it, or -- for a toy code -- as
    `toy.build_toy_dataset` generates it. Downloads on first use; a toy
    domain never touches `root` at all, since it has nothing to download.
    """
    if code in toy.TOY_CODES:
        return toy.build_toy_dataset(code)
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


def _subject_ids(dataset) -> np.ndarray | None:
    """Per-point subject id, when the dataset declares one -- `None` for a
    digit domain, which has no subject notion at all and is unaffected by
    anything this returns.
    """
    values = getattr(dataset, "subjects", None)
    if values is None:
        return None
    if torch.is_tensor(values):
        return values.numpy()
    return np.asarray(values)


#: How many of a domain's bags fall under one of the noised roles, per class.
#: `NOISE_ROLES` names which roles (`train` and `eval` today) the
#: contamination reaches; this is the total bag count across exactly those
#: roles, which is what `_reserve_size` and `_contaminate` both need to agree
#: on rather than each re-deriving from `NOISE_ROLES` its own way.
_ROLE_BAGS = {"train": config.TRAIN_BAGS, "valid": config.VALID_BAGS,
             "eval": config.EVAL_BAGS}


def _noised_bag_count() -> int:
    """Total bags (every class) subject to contamination, across `NOISE_ROLES`."""
    return sum(_ROLE_BAGS[role] for role in config.NOISE_ROLES)


def _reserve_size(rate: float) -> int:
    """How many spare images per class the contamination needs.

    Zero when nothing is contaminated, and that zero matters: the reserve is
    taken from the same permutation the bags are drawn from, so a non-zero
    reserve at rate 0 would move nothing and cost nothing but would still have to
    be justified. More to the point, `_select` must draw exactly what it drew
    before at rate 0 or every clean result stops being comparable to itself.

    Every bag draws its whole contamination from the ONE donor class
    `NOISE_PAIRING` declares for the bag's own label, so a class is never a
    donor to more than one other class -- its paired partner. The reserve a
    class holds is therefore sized against the noised bags labelled its
    partner, not against the whole noised population split over nine
    classes: `_noised_bag_count() * per_bag / CLASSES` rather than
    `/ (CLASSES - 1)`, because under a mutual pairing every class carries
    exactly one partner's worth of bags, not one ninth of everyone else's.
    The slack absorbs the remainder of that division.

    Checked against USPS, the domain that binds: at the top rung (k = 15)
    this asks class 8 -- the smallest class, 542 images -- for 180, and it
    has 182 spare once its own 360 are drawn (`BAGS_PER_CLASS *
    INSTANCES_PER_BAG`). Sized over every role `NOISE_ROLES` names (all
    three, `train`/`valid`/`eval`, since the noisy condition contaminates all
    three with one shared draw) rather than `TRAIN_BAGS` alone -- USPS is what
    binds, and a caller sizing the reserve over the wrong role count would
    refuse mid-draw instead of up front.
    """
    per_bag = config.noise_instances(rate)
    if per_bag == 0:
        return 0
    return (-(-_noised_bag_count() * per_bag // config.CLASSES)
            + config.INSTANCES_PER_BAG)


def _select(labels: np.ndarray, rng: np.random.Generator, reserve: int = 0,
            subjects: np.ndarray | None = None,
            ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Draw the images each class contributes, refusing rather than shrinking.

    A class that cannot fill its quota is an error. Returning a smaller slice
    would produce bags of uneven support and a failure downstream that reads as a
    defect of the method while being a defect of the draw.

    `reserve` is the spare each class holds back for contamination, drawn from the
    same permutation so a domain that cannot fund both is refused here rather than
    halfway through the replacement. At `reserve == 0` this draws exactly what it
    drew before the noise axis existed, image for image.

    `subjects` is `None` for every digit domain, and this function's behaviour
    at `subjects=None` is UNCHANGED, byte for byte, from before this parameter
    existed -- `test_toy_domains.py` pins that directly against a hand-replayed
    permutation, so a later edit to the branch below reddens rather than
    silently shifting the digit campaign by one image.

    When `subjects` is given (one id per entry of `labels`, meaningful only
    within a class), the draw goes subject by subject instead of permuting
    the whole class pool at once: for class `c`, subject `k`'s own pool is
    permuted on its own, its first `INSTANCES_PER_BAG` images become bag `k`
    of class `c` (`bags.build`'s own per-class loop assigns bags to classes in
    exactly that order), and the REST of that subject's pool -- never a share
    carved out of `reserve` by this function -- becomes that subject's own
    contamination reserve. The reserve stays subject-local on purpose: a
    contaminated bag's clean part has to come from its own subject, or the
    manifold patch that makes a bag a subject in the first place is blurred
    across subjects, which is the one thing this whole mechanism exists to
    avoid. `reserve` is still honoured, but as a single bound checked against
    the SUM of every subject's own leftover, not as a quota divided evenly
    across subjects beforehand.
    """
    needed = config.BAGS_PER_CLASS * config.INSTANCES_PER_BAG
    chosen: dict[int, np.ndarray] = {}
    spare: dict[int, np.ndarray] = {}
    for class_id in range(config.CLASSES):
        pool = np.flatnonzero(labels == class_id)
        if subjects is None:
            if pool.size < needed + reserve:
                raise ValueError(
                    f"class {class_id} offers {pool.size} images and the draw needs "
                    f"{needed}"
                    + (f" plus {reserve} held back for contamination" if reserve else "")
                    + "; this domain cannot fund the agreed bag count"
                )
            drawn = rng.permutation(pool)
            chosen[class_id] = drawn[:needed]
            spare[class_id] = drawn[needed:needed + reserve]
            continue

        class_subjects = subjects[pool]
        subject_ids = sorted(int(s) for s in np.unique(class_subjects))
        if len(subject_ids) != config.BAGS_PER_CLASS:
            raise ValueError(
                f"class {class_id} offers {len(subject_ids)} subjects and "
                f"the draw needs exactly {config.BAGS_PER_CLASS}, one per bag"
            )
        chosen_parts: list[np.ndarray] = []
        spare_parts: list[np.ndarray] = []
        for subject_id in subject_ids:
            subject_pool = pool[class_subjects == subject_id]
            if subject_pool.size < config.INSTANCES_PER_BAG:
                raise ValueError(
                    f"class {class_id} subject {subject_id} offers "
                    f"{subject_pool.size} images and its own bag needs "
                    f"{config.INSTANCES_PER_BAG}; this subject cannot fund its bag"
                )
            drawn = rng.permutation(subject_pool)
            chosen_parts.append(drawn[:config.INSTANCES_PER_BAG])
            spare_parts.append(drawn[config.INSTANCES_PER_BAG:])
        chosen[class_id] = np.concatenate(chosen_parts)
        spare_for_class = np.concatenate(spare_parts)
        if spare_for_class.size < reserve:
            raise ValueError(
                f"class {class_id} offers {spare_for_class.size} spare images "
                f"across its {len(subject_ids)} subjects and the draw needs "
                f"{reserve} held back for contamination; this domain cannot "
                f"fund the agreed bag count"
            )
        spare[class_id] = spare_for_class
    return chosen, spare


def _contaminate(flat: list[int], members: list[list[int]], bag_labels: list[int],
                 role_positions: dict[str, list[int]], spare: dict[int, np.ndarray],
                 rate: float, seed: int, code: str) -> dict:
    """Replace part of every noised-role bag with images of other classes.

    The bag's label is left exactly as it was. Bags are pure and no instance
    carries a label of its own, so there is nothing to flip: what this corrupts is
    the evidence a bag offers for its own label, and the two families read that
    corruption differently by construction -- an instance-unit arm receives the
    bag's label broadcast to all thirty instances and so is handed genuinely wrong
    labels, while a bag-unit arm keeps the label at the bag and sees witnesses its
    attention may learn to downweight.

    It edits `flat` in place, before the images are decoded, so `rebuild` needs to
    know nothing about noise: the manifest's `imageIndices` already names the
    contaminants. The record written beside it exists so a reader can tell a
    contaminated slot from a clean one without re-deriving the draw.

    Every role `config.NOISE_ROLES` names is touched -- `train` and `eval`
    today, `valid` deliberately not -- with ONE shared draw: `rng` below is
    seeded from `(seed, code)` alone, never from the arm or the role, so the
    identical instances are replaced regardless of which arm's material this
    call built for, and the walk below covers every named role's positions in
    one pass rather than one stream per role, which is what makes the draw a
    single shared one instead of independent ones that happened to use the same
    seed.

    Every bag's replacements come from exactly ONE donor class: the one
    `config.NOISE_PAIRING` declares for the bag's own label, never a rotation
    across the other nine. The decision a bag poses is its own class against
    the single strongest competitor, not against the summed contaminant count
    -- balanced over nine donors that competitor never exceeds `ceil(k / 9)`,
    one instance up to rho 0.30, which measured robustness to diluted evidence
    and could never reach ambiguous evidence at any admitted rate.
    `NOISE_PAIRING` is an involution, and that is what the top rung needs: a
    bag labelled `c` and a bag labelled `NOISE_PAIRING[c]` end up with
    IDENTICAL class composition and two different labels, so any composition
    reader gets exactly one of the pair right, which is the forced error the
    axis exists to reach rather than a side effect of raising the rate.
    """
    per_bag = config.noise_instances(rate)
    if per_bag == 0:
        return {"rate": rate, "instancesPerBag": 0, "roles": list(config.NOISE_ROLES),
                "bags": []}

    # A stream of its own, so the clean draw above is bit-identical whether or not
    # anything is contaminated afterwards. Sharing `rng` would make every seed's
    # bags depend on the rate and no rate would be comparable to another.
    rng = np.random.default_rng((seed * 1000 + ord(code)) * 7919 + 104729)

    cursor = {class_id: 0 for class_id in range(config.CLASSES)}
    record: list[dict] = []

    # One combined, ordered walk over every noised role's positions -- never one
    # loop per role -- so `rng`'s draws land in one deterministic sequence and the
    # draw is reproducible from `(seed, code)` alone, independent of how many
    # roles happen to be named.
    noised_positions = [position for role in config.NOISE_ROLES
                        for position in role_positions.get(role, [])]
    role_of = {position: role for role in config.NOISE_ROLES
              for position in role_positions.get(role, [])}
    for position in noised_positions:
        own = bag_labels[position]
        # ONE donor, concentrated rather than balanced: every replacement in
        # this bag comes from the single class `NOISE_PAIRING` pairs with
        # `own`, never a rotation across the other nine. That concentration
        # is the point -- it is what lets the top rung pose the bag's own
        # class against an equally-sized competitor instead of nine diluted
        # ones.
        donor = config.NOISE_PAIRING[own]
        donors = [donor] * per_bag
        slots = sorted(int(s) for s in
                       rng.permutation(config.INSTANCES_PER_BAG)[:per_bag])

        replaced: list[int] = []
        for slot, donor in zip(slots, donors):
            available = spare[donor]
            if cursor[donor] >= available.size:
                raise ValueError(
                    f"class {donor} ran out of spare images while contaminating "
                    f"at rate {rate}; the reserve is sized for the declared cap "
                    f"and this draw asked for more"
                )
            image = int(available[cursor[donor]])
            cursor[donor] += 1
            flat[members[position][slot]] = image
            replaced.append(image)

        record.append({"bag": position, "role": role_of[position], "label": own,
                       "slots": slots, "donorClasses": donors,
                       "imageIndices": replaced})

    return {"rate": rate, "instancesPerBag": per_bag,
            "roles": list(config.NOISE_ROLES), "bags": record}


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


def tie_ceiling(code: str, root: Path, seed: int) -> float:
    """The top rung's own ceiling for this (domain, seed), never a constant.

    AGREED.md, 'Label noise': for each mutual pair `NOISE_PAIRING` declares, a
    composition reader can get at most `max(n_a, n_b)` of that pair's
    `n_a + n_b` evaluation bags right -- guessing the larger side on every
    bag it cannot otherwise decide. Summed over the five pairs and divided by
    `EVAL_BAGS`, that is the best ANY composition reader can do at the tie,
    and it depends on how the evaluation split landed for this domain and
    this seed, so it travels with the result instead of being written down
    once as 1.0 or as a single shared number.

    Replays exactly the two calls `build()` makes before contamination ever
    enters the picture -- `_select` then `_split_counts`, on the SAME `rng`
    seeded from `(seed, code)` -- because the eval share per class is a
    function of that replay alone: `_select` always permutes each class's
    full pool regardless of how large a reserve it is asked to hold back, so
    the state `_split_counts` sees, and therefore the eval split it returns,
    does not depend on the noise rate. `reserve=0` here is exact for that
    reason, not an approximation of the real draw.

    Needs the real per-class pool sizes, which is why `root` has to point at
    actual cached material (`config.DATA_CACHE` today) rather than a
    synthetic fixture: a domain's own class sizes are what makes the eval
    split -- and therefore the ceiling -- differ from another domain's.
    """
    # The ceiling is only DEFINED for an involution, and the sum below is
    # where that assumption lives: it walks `a < NOISE_PAIRING[a]` expecting
    # that to enumerate each mutual pair exactly once. Refused rather than
    # computed, because the wrong answer is plausible instead of absurd --
    # measured against the long cycle `[1,2,...,9,0]`, which is a
    # fixed-point-free permutation like any other, the walk finds NINE pairs
    # instead of five, counts eight of the ten classes twice, and returns
    # 0.9444. That is inside (0, 1], reads like a ceiling, and nothing
    # downstream could tell it from one. A conclusion printing it would
    # report a bound nobody derived.
    #
    # `config.NOISE_PAIRING` being an involution is declared and asserted on
    # its own (`test_el_emparejamiento_es_una_involucion_sin_punto_fijo`), so
    # this cannot fire for the shipped declaration; it fires for a caller
    # that substituted one, which is exactly what a test exploring the
    # alternative does.
    pairing = config.NOISE_PAIRING
    if any(pairing[pairing[a]] != a or pairing[a] == a
           for a in range(config.CLASSES)):
        raise ValueError(
            f"the tie ceiling is defined only for an involution with no fixed "
            f"point -- five mutual pairs -- and {pairing} is not one: a bag's "
            f"own class and its partner's must show each other back, or the "
            f"pairs this sums over are neither five nor disjoint"
        )

    dataset = _raw_dataset(code, root)
    labels = _labels(dataset)
    rng = np.random.default_rng(seed * 1000 + ord(code))
    _select(labels, rng, 0)
    train_share, valid_share = _split_counts(rng)
    eval_share = {class_id: config.BAGS_PER_CLASS - train_share[class_id]
                  - valid_share[class_id] for class_id in range(config.CLASSES)}
    paired = sum(max(eval_share[a], eval_share[pairing[a]])
                for a in range(config.CLASSES) if a < pairing[a])
    return paired / config.EVAL_BAGS


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


def build(code: str, root: Path, seed: int, noise: float | None = None) -> BagSet:
    """Every bag of one domain, and the record of how they were drawn.

    `noise` is the fraction of each *training* bag replaced by images of other
    classes, defaulting to whatever `config.NOISE` currently declares. At 0 this
    draws exactly what it drew before the noise axis existed -- same permutation,
    same images, same split -- so the clean campaign stays comparable to itself.
    """
    rate = config.NOISE if noise is None else noise
    dataset = _raw_dataset(code, root)
    labels = _labels(dataset)
    subjects = _subject_ids(dataset)
    rng = np.random.default_rng(seed * 1000 + ord(code))

    selected, spare = _select(labels, rng, _reserve_size(rate), subjects=subjects)
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

    # Before the images are decoded, so `rebuild` needs to know nothing about
    # noise: `imageIndices` below already names whatever replaced what. All
    # three roles are handed over; `_contaminate` itself decides which of them
    # `config.NOISE_ROLES` actually reaches -- today `valid` is not among them,
    # and handing it over anyway is what keeps that a one-line decision.
    contamination = _contaminate(
        flat, members, bag_labels,
        {"train": train_positions, "valid": valid_positions, "eval": eval_positions},
        spare, rate, seed, code)

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
        # Which slots stopped supporting their bag's label, and what replaced
        # them. `imageIndices` already carries the contaminants, so this is not
        # what makes the material rebuildable -- it is what lets a reader tell a
        # contaminated slot from a clean one without re-deriving the draw.
        "contamination": contamination,
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
