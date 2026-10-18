"""Three manifolds with no pretrained backbone to hide behind.

Every measurement of this project's adaptation term so far ran under
`resnet18` -- 11M ImageNet-pretrained parameters on 32x32 digit images. The
objection that motivates this module: with a backbone that strong, a bare
normalization statistic (mean and variance) is already sufficient, so the
method's kernels are superfluous *by construction*, and nothing in the digit
campaign can tell that apart from the method genuinely mattering. These toy
domains remove the overbuilt backbone so the question becomes answerable at
low cost; `mlp.py` removes it on the model side.

Nine domain codes, three families (`config.DOMAINS` carries the full table):
swiss roll (`r`/`R`/`1`), tori (`t`/`T`/`2`) and moons (`n`/`N`/`3`).
Lowercase is a source, uppercase is a translation-only target, and the digit
is a translation+rotation+ambient-noise target. `config.TOY_TRANSFERS`
declares the six source-to-target pairs, one per family's two targets.

Every one of the ten classes `config.CLASSES` still names is a band of the
manifold's own parameter -- chance accuracy therefore stays exactly 0.10,
which is what makes these results directly comparable to the digit numbers
already measured. And a bag here is a SUBJECT, not an exchangeable random
slice: `config.BAGS_PER_CLASS` (12) disjoint sub-bands of each class's own
band, each generated as its own localized patch of the manifold the way an
EEG subject has its own distribution. `bags._select`'s `subjects` parameter
is what turns that generated structure into "bag k of class c is subject k of
class c" -- see its own docstring for the draw.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from MIL_CREDA_Benchmark import config

#: The three manifold families and the nine domain codes that name a draw from
#: one of them.
SWISS_ROLL_CODES = ("r", "R", "1")
TORI_CODES = ("t", "T", "2")
MOONS_CODES = ("n", "N", "3")
TOY_CODES = SWISS_ROLL_CODES + TORI_CODES + MOONS_CODES

#: Which perturbation a code's own draw gets, relative to the same family's
#: clean parametrization: `None` for a source, `"translate"` for a
#: translation-only target, `"translate_rotate_noise"` for the full
#: perturbation. Applied by `_apply_role` -- never baked into the manifold's
#: own parametrization, so the SAME (class, subject) sub-band is sampled
#: before the role's perturbation is applied to it.
_ROLE: dict[str, str | None] = {
    "r": None, "t": None, "n": None,
    "R": "translate", "T": "translate", "N": "translate",
    "1": "translate_rotate_noise", "2": "translate_rotate_noise",
    "3": "translate_rotate_noise",
}

_FAMILY: dict[str, str] = {}
for _code in SWISS_ROLL_CODES:
    _FAMILY[_code] = "swiss_roll"
for _code in TORI_CODES:
    _FAMILY[_code] = "tori"
for _code in MOONS_CODES:
    _FAMILY[_code] = "moons"

#: One fixed offset per family, so every code OF THE SAME FAMILY samples the
#: identical (class, subject) manifold parameter values -- a source and its
#: targets differ ONLY by the role's perturbation, never by which point of
#: the manifold they happened to draw. Noise (role-dependent, and therefore
#: necessarily code-dependent) is seeded separately, below.
_FAMILY_SEED_OFFSET = {"swiss_roll": 0, "tori": 1, "moons": 2}


def _subject_permutation(family: str, class_id: int) -> np.ndarray:
    """Which sub-band POSITION subject k of this class actually occupies.

    Measured defect this exists to fix: `bags.build` assigns train/valid/eval
    roles BY SUBJECT INDEX k (`if k < train_share[class_id]: train`, next:
    valid, rest: eval). Subject k occupying sub-band k outright makes patch
    position and subject index the SAME number, so the role split -- which
    is a split on k -- is ALSO a split on position: every class's band broke
    into disjoint train/eval halves, with eval sitting in manifold the model
    never trained on, adjacent to the next class's band. Measured on the
    swiss roll: class 0 train u in [+4.08, +5.21] vs eval u in [+1.92,
    +3.12]; source-against-source accuracy came out near chance as a direct
    consequence -- not an undertraining artefact (flat across epoch count and
    learning rate).

    So subject k's PATCH POSITION is `perm[k]`, a fixed permutation of
    `range(BAGS_PER_CLASS)`, rather than `k` itself -- decorrelating which
    patch a subject gets from the index the role split cuts on, while every
    subject still occupies its own disjoint, localized patch (nothing about
    the local-correspondence ground truth weakens). Derived from
    `config.TOY_SEED`, `family` and `class_id` alone -- NEVER from the domain
    code -- so every code of one family/class (a source and both of its
    targets) places subject k in the IDENTICAL sub-band, which is what keeps
    the matched-patch property `true_partners` relies on intact.
    """
    seed = config.TOY_SEED + 1000 * class_id + _FAMILY_SEED_OFFSET[family]
    return np.random.default_rng(seed).permutation(config.BAGS_PER_CLASS)


def subject_patch_position(code: str, class_id: int, subject: int) -> int:
    """The sub-band position subject `subject` of `class_id` occupies in
    `code`'s own family -- the same value `_subject_permutation` computes
    internally, exposed so `true_partners` and the tests read it rather than
    re-deriving it from generated data.
    """
    return int(_subject_permutation(_FAMILY[code], class_id)[subject])


class ToyDataset(Dataset):
    """One toy domain: a point cloud with a class label and a subject id per
    point.

    Shaped exactly like `bags._labels`/`bags._select` need: `.targets` is the
    array `_labels` reads (digit-dataset-compatible), `.subjects` is the new
    per-point subject id `_select` draws from when `bags.build` hands it
    over, and `__getitem__` returns a `(tensor, label)` pair -- the same
    contract the digit `torchvision` datasets satisfy, so `bags.build`/
    `bags.rebuild` need nothing new to index this.
    """

    def __init__(self, points: np.ndarray, targets: np.ndarray, subjects: np.ndarray):
        self.points = points.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.subjects = subjects.astype(np.int64)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.points[index]), int(self.targets[index])


# ------------------------------------------------------------- transforms
#
# Exposed as plain functions, apart from the generators below, so the
# invariant that matters most -- the rotation really mixes coordinates and is
# not a per-coordinate affine map in disguise -- is testable directly against
# the transform itself rather than only against whatever a full domain draw
# happens to produce.

def rotation_matrix(dim: int) -> np.ndarray:
    """The `config.TOY_ROTATION_ANGLE` rotation for an ambient dimension of
    2 or 3.

    2D (moons): the ordinary 2x2 rotation of the whole plane -- the only two
    coordinates there are. 3D (swiss roll, tori): a rotation of coordinates 0
    and 2, with coordinate 1 (the swiss roll's height, a torus's own minor-
    angle wobble -- the "free", class-independent dimension every family's
    generator below samples uniformly) left untouched. Either way exactly two
    coordinates are mixed, which is the whole point: a per-coordinate affine
    map (what `BatchNorm1d` can express) is diagonal, and this matrix is not.
    """
    c = np.cos(config.TOY_ROTATION_ANGLE)
    s = np.sin(config.TOY_ROTATION_ANGLE)
    if dim == 2:
        return np.array([[c, -s], [s, c]], dtype=np.float64)
    if dim == 3:
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0] = c
        matrix[0, 2] = -s
        matrix[2, 0] = s
        matrix[2, 2] = c
        return matrix
    raise ValueError(f"no declared rotation for ambient dimension {dim}; known: 2, 3")


def translation_vector(dim: int) -> np.ndarray:
    """`config.TOY_TRANSLATION_SCALE`, repeated across every ambient
    coordinate -- a pure per-coordinate constant offset, which is exactly
    what `BatchNorm1d`'s running mean can absorb. That is the property the
    translation-only domains (`R`, `T`, `N`) exist to exercise.
    """
    return np.full(dim, config.TOY_TRANSLATION_SCALE, dtype=np.float64)


def rotate_points(points: np.ndarray) -> np.ndarray:
    """`points` (N, D) rotated about the origin by `rotation_matrix`."""
    dim = points.shape[1]
    return points @ rotation_matrix(dim).T


def _apply_role(points: np.ndarray, role: str | None,
                rng_noise: np.random.Generator) -> np.ndarray:
    """The perturbation this role applies to an already-sampled clean patch.

    Rotation, when it applies, is about the origin -- BEFORE translation --
    so the rotation matrix stays the one `rotation_matrix` returns unchanged
    by the domain's own translated position, and `rotate_points` alone is
    what the pairwise-distance and non-affine invariants are checked against.
    """
    if role is None:
        return points
    if role == "translate":
        return points + translation_vector(points.shape[1])
    if role == "translate_rotate_noise":
        rotated = rotate_points(points)
        translated = rotated + translation_vector(points.shape[1])
        noise = rng_noise.normal(0.0, config.TOY_NOISE_SIGMA, size=translated.shape)
        return translated + noise
    raise ValueError(f"unknown toy domain role {role!r}")


# ------------------------------------------------------------- band helpers

def _band_index(values: np.ndarray, lo: float, hi: float, bands: int) -> np.ndarray:
    """Which of `bands` equal-width bins of `[lo, hi)` each value falls in."""
    edges = np.linspace(lo, hi, bands + 1)
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, bands - 1)


# ----------------------------------------------------------- family draws
#
# Each returns the full domain: every class, every subject, concatenated.
# `sample_rng` draws the manifold parameter (and the family's own free
# dimension) -- shared across a family's three codes, so a source and its
# targets sample the identical patches before any role's perturbation is
# applied. `rng_noise` is the role's own stream (seeded per CODE, below),
# since only the "translate_rotate_noise" role ever reads it.

def _swiss_roll_domain(role: str | None, rng_noise: np.random.Generator,
                       sample_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`sklearn.datasets.make_swiss_roll`'s own parametrization
    (`x = t*cos(t)`, `height ~ Uniform(0, 21)`, `z = t*sin(t)`, `t ~
    Uniform(1.5*pi, 4.5*pi)`), sampled directly rather than through the
    wrapper: the wrapper draws `t` from its own RNG with no way to request a
    point inside a specific sub-band, and the subject design needs exactly
    that. The formula is reproduced exactly, so the manifold this generates
    is the one that name denotes; only the SAMPLING is direct. 10 equal bands
    of `t` are the 10 classes, each cut into `config.BAGS_PER_CLASS` equal
    sub-bands -- subject k of class c occupies sub-band `perm[k]`, never
    sub-band k itself; see `_subject_permutation` for why.
    """
    t_lo, t_hi = 1.5 * np.pi, 4.5 * np.pi
    class_edges = np.linspace(t_lo, t_hi, config.CLASSES + 1)
    points, labels, subjects = [], [], []
    for class_id in range(config.CLASSES):
        subject_edges = np.linspace(class_edges[class_id], class_edges[class_id + 1],
                                    config.BAGS_PER_CLASS + 1)
        perm = _subject_permutation("swiss_roll", class_id)
        for subject in range(config.BAGS_PER_CLASS):
            position = perm[subject]
            n = config.TOY_POINTS_PER_SUBJECT
            t = sample_rng.uniform(subject_edges[position], subject_edges[position + 1], size=n)
            height = sample_rng.uniform(0.0, 21.0, size=n)
            patch = np.stack([t * np.cos(t), height, t * np.sin(t)], axis=1)
            points.append(_apply_role(patch, role, rng_noise))
            labels.append(np.full(n, class_id))
            subjects.append(np.full(n, subject))
    return np.concatenate(points), np.concatenate(labels), np.concatenate(subjects)


def _moons_domain(role: str | None, rng_noise: np.random.Generator,
                  sample_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`sklearn.datasets.make_moons`'s own two interleaving half-circles
    (outer: `x = cos(theta)`, `y = sin(theta)`; inner: `x = 1 - cos(theta)`,
    `y = 1 - sin(theta) - 0.5`; `theta ~ Uniform(0, pi)` either way), sampled
    directly for the same reason `_swiss_roll_domain` samples its own formula
    directly. Class = `branch * 5 + band`, 5 equal bands of `theta` along
    each branch; subject = the sub-band within that band -- `perm[k]`, never
    sub-band k itself; see `_subject_permutation` for why.
    """
    bands = 5
    theta_edges = np.linspace(0.0, np.pi, bands + 1)
    points, labels, subjects = [], [], []
    for branch in (0, 1):
        for band in range(bands):
            class_id = branch * bands + band
            subject_edges = np.linspace(theta_edges[band], theta_edges[band + 1],
                                        config.BAGS_PER_CLASS + 1)
            perm = _subject_permutation("moons", class_id)
            for subject in range(config.BAGS_PER_CLASS):
                position = perm[subject]
                n = config.TOY_POINTS_PER_SUBJECT
                theta = sample_rng.uniform(subject_edges[position], subject_edges[position + 1],
                                           size=n)
                if branch == 0:
                    x, y = np.cos(theta), np.sin(theta)
                else:
                    x, y = 1.0 - np.cos(theta), 1.0 - np.sin(theta) - 0.5
                patch = np.stack([x, y], axis=1)
                points.append(_apply_role(patch, role, rng_noise))
                labels.append(np.full(n, class_id))
                subjects.append(np.full(n, subject))
    return np.concatenate(points), np.concatenate(labels), np.concatenate(subjects)


#: The two linked tori's shared radii: `TOY_MAJOR_RADIUS` the loop radius,
#: `TOY_MINOR_RADIUS` the tube radius -- declared here, rather than in
#: `config.py`, because nothing outside this generator reads them (unlike
#: `TOY_TRANSLATION_SCALE`/`TOY_ROTATION_ANGLE`/`TOY_NOISE_SIGMA`, which
#: `_apply_role` shares across all three families). `TOY_MINOR_RADIUS` well
#: under `TOY_MAJOR_RADIUS` is what keeps a single torus from self-
#: intersecting.
_TORI_MAJOR_RADIUS = 2.0
_TORI_MINOR_RADIUS = 0.5


def _tori_domain(role: str | None, rng_noise: np.random.Generator,
                 sample_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two linked tori, not in `sklearn`, generated parametrically.

    Torus 0's core circle (radius `_TORI_MAJOR_RADIUS`) lies in the xy-plane,
    centred at the origin; torus 1's core circle, the same radius, lies in
    the xz-plane, centred at `(_TORI_MAJOR_RADIUS, 0, 0)`. Each core circle
    then passes through the other's centre (torus 0's circle crosses
    `(_TORI_MAJOR_RADIUS, 0, 0)` at `theta = 0`; torus 1's crosses the origin
    at `theta = pi`), which is the classic linked-pair construction: each
    ring threads through the other's hole. Class = `torus * 5 + band`, 5
    equal bands of the major angle `theta`; subject = the sub-band within
    that band -- `perm[k]`, never sub-band k itself; see
    `_subject_permutation` for why. The minor angle `phi` is the family's own
    free dimension (the tube's thickness), sampled uniformly and independent
    of class -- `_swiss_roll_domain`'s height, one torus over.
    """
    bands = 5
    theta_edges = np.linspace(0.0, 2 * np.pi, bands + 1)
    R, r = _TORI_MAJOR_RADIUS, _TORI_MINOR_RADIUS
    points, labels, subjects = [], [], []
    for torus_id in (0, 1):
        for band in range(bands):
            class_id = torus_id * bands + band
            subject_edges = np.linspace(theta_edges[band], theta_edges[band + 1],
                                        config.BAGS_PER_CLASS + 1)
            perm = _subject_permutation("tori", class_id)
            for subject in range(config.BAGS_PER_CLASS):
                position = perm[subject]
                n = config.TOY_POINTS_PER_SUBJECT
                theta = sample_rng.uniform(subject_edges[position], subject_edges[position + 1],
                                           size=n)
                phi = sample_rng.uniform(0.0, 2 * np.pi, size=n)
                tube = R + r * np.cos(phi)
                if torus_id == 0:
                    x, y, z = tube * np.cos(theta), tube * np.sin(theta), r * np.sin(phi)
                else:
                    x, y, z = R + tube * np.cos(theta), r * np.sin(phi), tube * np.sin(theta)
                patch = np.stack([x, y, z], axis=1)
                points.append(_apply_role(patch, role, rng_noise))
                labels.append(np.full(n, class_id))
                subjects.append(np.full(n, subject))
    return np.concatenate(points), np.concatenate(labels), np.concatenate(subjects)


_FAMILY_BUILDERS = {
    "swiss_roll": _swiss_roll_domain,
    "tori": _tori_domain,
    "moons": _moons_domain,
}


def build_toy_dataset(code: str) -> ToyDataset:
    """One toy domain, deterministic from `code` and `config.TOY_SEED` alone
    -- never from a campaign's own `seed` -- which is what lets
    `bags.rebuild` reconstruct it exactly from a manifest that only ever
    names the domain code.
    """
    if code not in TOY_CODES:
        raise ValueError(f"unknown toy domain {code!r}; known: {sorted(TOY_CODES)}")
    family = _FAMILY[code]
    role = _ROLE[code]
    sample_rng = np.random.default_rng(config.TOY_SEED + _FAMILY_SEED_OFFSET[family])
    noise_rng = np.random.default_rng(config.TOY_SEED * 1000 + ord(code))
    points, labels, subjects = _FAMILY_BUILDERS[family](role, noise_rng, sample_rng)
    return ToyDataset(points, labels, subjects)


# --------------------------------------------------------------- the payoff

def true_partners(source_bagset, target_bagset) -> np.ndarray:
    """For each of `target_bagset`'s EVAL bags, the index into
    `source_bagset.train_idx` of its nearest same-class subject.

    Subject k of class c occupies the SAME manifold sub-band in every code of
    a family, by construction of the generators above -- which is the whole
    payoff of making a bag a subject rather than an exchangeable random
    slice: the true source partner of a target bag is knowable, not assumed.

    Ranks by PATCH POSITION (`subject_patch_position`, i.e. `perm[k]`), never
    by the raw subject index k: since `_subject_permutation` exists
    precisely to decorrelate the two, ranking by index would rank by a number
    that no longer tracks where a subject actually sits on the manifold.
    Patch position is known at generation time, so it is read from there
    rather than re-estimated from the drawn material.

    It is a RANKED correspondence rather than an exact one, because the two
    domains' role splits genuinely differ: `bags.build`'s own `rng` is seeded
    from `(seed, code)`, and `code` differs between any source and target, so
    a target bag's exact-subject counterpart is often not among the source's
    own TRAIN bags at all (it may have landed in `valid` or `eval` there
    instead). So this ranks, among the source TRAIN bags of the target bag's
    own class, by how close their patch position is to the target bag's own
    patch position, and returns the closest. An exact match is returned
    whenever one is present.
    """
    source_code = source_bagset.domain
    target_code = target_bagset.domain

    def positions_and_classes(bagset_positions: np.ndarray, code: str):
        classes = bagset_positions // config.BAGS_PER_CLASS
        subjects = bagset_positions % config.BAGS_PER_CLASS
        patches = np.array([
            subject_patch_position(code, int(c), int(s))
            for c, s in zip(classes, subjects)
        ])
        return classes, patches

    train_positions = source_bagset.train_idx.numpy()
    train_classes, train_patches = positions_and_classes(train_positions, source_code)

    target_positions = target_bagset.eval_idx.numpy()
    target_classes, target_patches = positions_and_classes(target_positions, target_code)

    partners = np.empty(target_positions.shape[0], dtype=np.int64)
    for i in range(target_positions.shape[0]):
        candidates = np.flatnonzero(train_classes == target_classes[i])
        if candidates.size == 0:
            raise ValueError(
                f"class {target_classes[i]} has no source TRAIN bag at all; "
                "the correspondence is undefined for a target eval bag with "
                "no same-class source train bag to rank against"
            )
        distances = np.abs(train_patches[candidates] - target_patches[i])
        partners[i] = candidates[int(np.argmin(distances))]
    return partners
