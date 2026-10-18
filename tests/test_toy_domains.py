"""`toy.py`'s three manifolds, and the subject structure `bags.py` builds on.

Every measurement of this project's adaptation term so far ran under a
resnet18 backbone on 32x32 digit images -- strong enough to make the
objection "a normalization statistic is already sufficient" unanswerable at
that cost. These tests hold the toy domains to the properties that make them
answer it: the rotated target really mixes coordinates (row 2 of the
intended comparison), classes follow the manifold's own parameter, every
domain is funded at the top contamination rung, a bag is a subject -- not an
exchangeable random slice -- in both a source and its targets, and the
digit path (`subjects=None`) is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from MIL_CREDA_Benchmark import bags, config, toy

#: Ambient dimensionality per toy code, read once here rather than re-derived
#: in every test -- 3 for swiss roll/tori, 2 for moons.
_DIM = {code: (2 if code in toy.MOONS_CODES else 3) for code in toy.TOY_CODES}


# --------------------------------------------------------------- rotation

class TestTheRotationIsARealMixing:
    """Row 2 of the intended comparison exists only if the rotation is not
    something `BatchNorm1d` could already undo -- a per-coordinate affine
    map is diagonal, and this checks the rotation matrix is not.
    """

    @pytest.mark.parametrize("dim", [2, 3])
    def test_rotation_preserves_pairwise_distances(self, dim):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(40, dim))
        rotated = toy.rotate_points(points)

        def pairwise(p):
            diff = p[:, None, :] - p[None, :, :]
            return np.sqrt((diff ** 2).sum(-1))

        before = pairwise(points)
        after = pairwise(rotated)
        assert np.allclose(before, after, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_rotation_matrix_is_not_diagonal(self, dim):
        """A per-coordinate affine map is a diagonal linear part. If the
        rotation matrix were diagonal, it could be replicated by
        `BatchNorm1d`'s per-coordinate scale alone -- the whole reason this
        axis exists would be gone.
        """
        matrix = toy.rotation_matrix(dim)
        off_diagonal = matrix - np.diag(np.diag(matrix))
        assert not np.allclose(off_diagonal, 0.0)

    def test_rotation_angle_is_exactly_45_degrees(self):
        assert config.TOY_ROTATION_ANGLE == pytest.approx(np.pi / 4)


# ----------------------------------------------- real-domain mean/covariance
#
# The coordinator's own independently-verified premise: after per-coordinate
# standardization -- exactly what BatchNorm applies at inference, from each
# domain's own running statistics -- a pure translation is fully erased
# while a 45-degree rotation leaves a real covariance residue, PROVIDED the
# rotated plane is anisotropic/correlated. Computed here on the REAL
# generated domains, never on a synthetic cloud chosen to make the number
# come out a particular way.

def _standardize(points: np.ndarray) -> np.ndarray:
    return (points - points.mean(0)) / (points.std(0) + 1e-8)


class TestStandardizationErasesTranslationButNotRotation:
    """For each rotated code against its own source: standardizing both
    (the BatchNorm-at-inference operation) erases a pure translation's mean
    shift and its covariance difference, but leaves the rotated domain's
    covariance measurably different -- never both near zero, or row 2 of
    the comparison is silently uninformative despite looking like a hard
    case.
    """

    @pytest.mark.parametrize("source_code,translate_code,rotate_code", [
        ("r", "R", "1"), ("t", "T", "2"), ("n", "N", "3"),
    ])
    def test_translation_is_absorbed_and_rotation_is_not(
        self, source_code, translate_code, rotate_code
    ):
        source = toy.build_toy_dataset(source_code).points
        translated = toy.build_toy_dataset(translate_code).points
        rotated = toy.build_toy_dataset(rotate_code).points

        source_std = _standardize(source)
        translated_std = _standardize(translated)
        rotated_std = _standardize(rotated)

        translated_mean_diff = float(
            np.max(np.abs(source_std.mean(0) - translated_std.mean(0))))
        translated_cov_diff = float(
            np.linalg.norm(np.cov(source_std.T) - np.cov(translated_std.T)))
        rotated_mean_diff = float(
            np.max(np.abs(source_std.mean(0) - rotated_std.mean(0))))
        rotated_cov_diff = float(
            np.linalg.norm(np.cov(source_std.T) - np.cov(rotated_std.T)))

        print(
            f"[{source_code}->{translate_code}/{rotate_code}] "
            f"translatedMeanDiff={translated_mean_diff:.6g} "
            f"translatedCovDiff={translated_cov_diff:.6g} "
            f"rotatedMeanDiff={rotated_mean_diff:.6g} "
            f"rotatedCovDiff={rotated_cov_diff:.6g}"
        )

        # The translated pole: the statistic really erases the shift.
        assert translated_mean_diff < 0.05
        assert translated_cov_diff < 0.05
        # The rotated pole: covariance is NOT erased by the same statistic.
        # (Rotation leaves the mean roughly alone too -- it is a linear map
        # about a cloud's own centre -- so only the covariance pole is
        # required to be large; asserting a large mean diff here would be
        # asserting something rotation does not promise.)
        assert rotated_cov_diff > 0.05


# ----------------------------------------------------- classes follow t/theta

class TestClassesFollowTheManifoldParameter:
    """Every point's class is the band its own manifold parameter falls in
    -- checked on the three SOURCE domains, where the parameter is exactly
    invertible from the generated coordinates (no noise, no rotation).
    """

    def test_swiss_roll_source_classes_follow_t_band(self):
        dataset = toy.build_toy_dataset("r")
        t = np.sqrt(dataset.points[:, 0] ** 2 + dataset.points[:, 2] ** 2)
        edges = np.linspace(1.5 * np.pi, 4.5 * np.pi, config.CLASSES + 1)
        expected = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, config.CLASSES - 1)
        assert np.array_equal(expected, dataset.targets)

    def test_moons_source_classes_follow_branch_and_theta_band(self):
        dataset = toy.build_toy_dataset("n")
        branch = (dataset.targets // 5).astype(np.int64)
        x, y = dataset.points[:, 0], dataset.points[:, 1]
        theta_outer = np.arctan2(y, x)
        theta_inner = np.arctan2(0.5 - y, 1.0 - x)
        theta = np.where(branch == 0, theta_outer, theta_inner)
        edges = np.linspace(0.0, np.pi, 5 + 1)
        band = np.clip(np.searchsorted(edges, theta, side="right") - 1, 0, 4)
        expected = branch * 5 + band
        assert np.array_equal(expected, dataset.targets)

    def test_tori_source_classes_follow_torus_and_theta_band(self):
        dataset = toy.build_toy_dataset("t")
        torus_id = (dataset.targets // 5).astype(np.int64)
        x, y, z = dataset.points[:, 0], dataset.points[:, 1], dataset.points[:, 2]
        theta_0 = np.arctan2(y, x) % (2 * np.pi)
        theta_1 = np.arctan2(z, x - 2.0) % (2 * np.pi)
        theta = np.where(torus_id == 0, theta_0, theta_1)
        edges = np.linspace(0.0, 2 * np.pi, 5 + 1)
        band = np.clip(np.searchsorted(edges, theta, side="right") - 1, 0, 4)
        expected = torus_id * 5 + band
        assert np.array_equal(expected, dataset.targets)


# ---------------------------------------------------------------- funding

class TestEveryToyDomainFundsTheTopContaminationRung:
    """Every one of the nine codes offers enough points, per class, to fund
    `bags._reserve_size(config.NOISE_CAP)` on top of the agreed bag count --
    this is what stops a future subject-count change from producing a
    mid-draw refusal instead of a declared number.
    """

    @pytest.mark.parametrize("code", sorted(toy.TOY_CODES))
    def test_each_class_offers_enough_points(self, code):
        dataset = toy.build_toy_dataset(code)
        needed = bags._reserve_size(config.NOISE_CAP) + (
            config.BAGS_PER_CLASS * config.INSTANCES_PER_BAG)
        for class_id in range(config.CLASSES):
            count = int((dataset.targets == class_id).sum())
            assert count >= needed, (
                f"{code} class {class_id} offers {count} points, needs {needed}"
            )

    def test_all_ten_classes_are_present_in_every_domain(self):
        for code in sorted(toy.TOY_CODES):
            dataset = toy.build_toy_dataset(code)
            assert set(np.unique(dataset.targets)) == set(range(config.CLASSES))


# ---------------------------------------------------------- bag == subject

class TestBagKOfClassCIsSubjectKOfClassC:
    @pytest.mark.parametrize("code", sorted(toy.TOY_CODES))
    def test_bag_positions_decompose_into_class_and_subject(self, code, tmp_path):
        dataset = toy.build_toy_dataset(code)
        bagset = bags.build(code, tmp_path, seed=0, noise=0.0)
        manifest = bagset.manifest
        for position in range(config.CLASSES * config.BAGS_PER_CLASS):
            class_id = position // config.BAGS_PER_CLASS
            subject = position % config.BAGS_PER_CLASS
            start = position * config.INSTANCES_PER_BAG
            raw_indices = manifest["imageIndices"][start:start + config.INSTANCES_PER_BAG]
            assert manifest["bagLabels"][position] == class_id
            # The clean (non-contaminated) slot's subject id matches the bag's
            # own position-derived subject, for every instance of the bag.
            assert np.all(dataset.subjects[raw_indices] == subject)
            assert np.all(dataset.targets[raw_indices] == class_id)


class TestTruePartners:
    """Ranked, not exact: the two domains' role splits differ because
    `bags.build`'s own `rng` bakes in `ord(code)`, so a target eval bag's
    exact-subject counterpart is often not among the source's own train
    bags. `true_partners` ranks by nearest subject index instead.
    """

    @pytest.mark.parametrize("source_code,target_code", config.TOY_TRANSFERS)
    def test_every_partner_matches_class_and_prefers_an_exact_patch(
        self, source_code, target_code, tmp_path
    ):
        """Ranked by PATCH POSITION (`subject_patch_position`), never by the
        raw subject index -- the permutation that decorrelates index from
        position (`toy._subject_permutation`) makes ranking by index wrong.
        """
        source_bagset = bags.build(source_code, tmp_path, seed=0, noise=0.0)
        target_bagset = bags.build(target_code, tmp_path, seed=0, noise=0.0)

        partners = toy.true_partners(source_bagset, target_bagset)
        train_positions = source_bagset.train_idx.numpy()
        target_positions = target_bagset.eval_idx.numpy()
        assert partners.shape[0] == target_positions.shape[0]

        # Whether the exact-patch branch is reachable at all, counted rather
        # than assumed. Measured: it fires ZERO times in all six transfers,
        # because `bags.build` sends low subject indices to `train` and high
        # ones to `eval` in BOTH domains, so the source's train subject indices
        # and the target's eval subject indices are disjoint. The old `if
        # target_patch in same_class_patches: assert ...` was therefore dead
        # code that read like a guarantee. It is not structurally impossible --
        # the per-class shares are drawn per domain, so an unusual draw could
        # overlap -- which is exactly why this counts instead of asserting
        # either way, and why the ranking above is asserted unconditionally.
        exact_available: list[bool] = []

        for i, target_position in enumerate(target_positions):
            target_class = target_position // config.BAGS_PER_CLASS
            target_subject = target_position % config.BAGS_PER_CLASS
            target_patch = toy.subject_patch_position(
                target_code, int(target_class), int(target_subject))

            source_position = train_positions[partners[i]]
            source_class = source_position // config.BAGS_PER_CLASS
            source_subject = source_position % config.BAGS_PER_CLASS
            source_patch = toy.subject_patch_position(
                source_code, int(source_class), int(source_subject))

            assert source_class == target_class

            same_class_train = train_positions[
                train_positions // config.BAGS_PER_CLASS == target_class]
            same_class_patches = [
                toy.subject_patch_position(
                    source_code, int(target_class), int(p % config.BAGS_PER_CLASS))
                for p in same_class_train
            ]

            # THE claim the alignment figure rests on, asserted unconditionally:
            # the partner is the NEAREST same-class source train bag in patch
            # position. Nothing checked this before -- only that the classes
            # matched, which every same-class bag satisfies -- so a
            # `true_partners` that returned the FURTHEST same-class bag, or an
            # arbitrary one, passed. The figure draws a line to this bag and
            # colours it by whether pi agreed, so if this ranking is wrong every
            # green line in that figure is meaningless.
            best = min(abs(patch - target_patch) for patch in same_class_patches)
            assert abs(source_patch - target_patch) == best, (
                f"partner for target patch {target_patch} is at patch "
                f"{source_patch} (distance {abs(source_patch - target_patch)}), "
                f"but a same-class source train bag sits at distance {best}")
            exact_available.append(target_patch in same_class_patches)

        # Never silently zero: if a future split DOES make exact patches
        # available, this prints it instead of letting the change pass unseen.
        print(f"{source_code}->{target_code}: exact patch available for "
              f"{sum(exact_available)}/{len(exact_available)} target bags")


# ------------------------------------------------- the role-split defect's fix
#
# Measured defect (coordinator): `bags.build` assigns train/valid/eval BY
# SUBJECT INDEX, so with subject k occupying sub-band k outright, the role
# split was also a split of each class's band into disjoint source/eval
# halves -- source accuracy near chance, flat in epochs and learning rate,
# confirmed not an undertraining artefact. `_subject_permutation` fixes it
# by decorrelating index from patch position; these two invariants measure
# that the fix actually holds, on the real generated domains.

def _subject_centroids(dataset: toy.ToyDataset, class_id: int) -> np.ndarray:
    """One centroid per subject of `class_id`, in subject-index order
    (0..BAGS_PER_CLASS - 1)."""
    mask = dataset.targets == class_id
    points, subjects = dataset.points[mask], dataset.subjects[mask]
    return np.stack([points[subjects == k].mean(0) for k in range(config.BAGS_PER_CLASS)])


def _index_position_correlation(dataset: toy.ToyDataset, class_id: int) -> float:
    """abs(Pearson correlation) between subject index and the subject
    centroid's position along the class's own first principal direction --
    the same projection the coordinator measured the original defect with.
    """
    centroids = _subject_centroids(dataset, class_id)
    centered = centroids - centroids.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[0]
    index = np.arange(config.BAGS_PER_CLASS)
    return float(abs(np.corrcoef(index, projected)[0, 1]))


#: Measured over every toy code and class with the permutation in place: the
#: largest |correlation| observed is 0.6134 (swiss roll / class 5). Measured
#: with the permutation removed (identity, i.e. subject k back at sub-band k,
#: reproducing the original defect): 0.9997-1.0 for every one of `r`/`n`/`t`.
#: 0.75 sits well above the fixed measurement and well below the broken one,
#: so this threshold actually discriminates the two rather than passing
#: either way.
_DECORRELATION_THRESHOLD = 0.75


class TestSubjectIndexIsDecorrelatedFromPatchPosition:
    """Red is reached by removing `_subject_permutation` (identity instead
    of a real permutation): `test_removing_the_permutation_reproduces_the_
    original_defect` confirms that directly, so this is not an assertion
    that only ever happens to pass.
    """

    @pytest.mark.parametrize("code", sorted(toy.TOY_CODES))
    def test_correlation_is_small_for_every_class(self, code):
        dataset = toy.build_toy_dataset(code)
        correlations = [_index_position_correlation(dataset, c) for c in range(config.CLASSES)]
        print(f"[{code}] max |corr| = {max(correlations):.4f}")
        for class_id, corr in enumerate(correlations):
            assert corr < _DECORRELATION_THRESHOLD, (
                f"{code} class {class_id}: |corr| = {corr:.4f}"
            )

    def test_removing_the_permutation_reproduces_the_original_defect(self, monkeypatch):
        """Pins that this invariant is falsifiable: patching
        `_subject_permutation` back to the identity (subject k at sub-band k,
        the pre-fix layout) must push the correlation back up near 1 --
        confirming the measured 0.92-0.9997 the coordinator reported.
        """
        monkeypatch.setattr(
            toy, "_subject_permutation",
            lambda family, class_id: np.arange(config.BAGS_PER_CLASS))
        dataset = toy.build_toy_dataset("r")
        correlations = [_index_position_correlation(dataset, c) for c in range(config.CLASSES)]
        assert max(correlations) > 0.9


class TestTrainAndEvalSubjectsOverlapPerClass:
    """The invariant that was actually violated: with subjects assigned to
    roles by index and index tracking position, train and eval sat in
    disjoint stretches of every class's band. After the fix, their PATCH
    POSITION ranges must overlap, per class -- never only on average, since
    an average can hide one lethal class while every other looks fine.
    """

    @pytest.mark.parametrize("code", sorted(toy.TOY_CODES))
    def test_train_and_eval_patch_ranges_overlap(self, code, tmp_path):
        bagset = bags.build(code, tmp_path, seed=0, noise=0.0)
        train_positions = bagset.train_idx.numpy()
        eval_positions = bagset.eval_idx.numpy()

        for class_id in range(config.CLASSES):
            train_subjects = [
                int(p % config.BAGS_PER_CLASS) for p in train_positions
                if p // config.BAGS_PER_CLASS == class_id
            ]
            eval_subjects = [
                int(p % config.BAGS_PER_CLASS) for p in eval_positions
                if p // config.BAGS_PER_CLASS == class_id
            ]
            if not train_subjects or not eval_subjects:
                continue  # nothing to overlap; a role missing a class entirely
                          # is a separate, already-guaranteed-absent failure
                          # mode (every class gets >= 6 of the 64 train bags).

            train_patches = [
                toy.subject_patch_position(code, class_id, s) for s in train_subjects]
            eval_patches = [
                toy.subject_patch_position(code, class_id, s) for s in eval_subjects]

            train_lo, train_hi = min(train_patches), max(train_patches)
            eval_lo, eval_hi = min(eval_patches), max(eval_patches)
            overlaps = max(train_lo, eval_lo) <= min(train_hi, eval_hi)
            assert overlaps, (
                f"{code} class {class_id}: train patches {sorted(train_patches)}, "
                f"eval patches {sorted(eval_patches)} do not overlap"
            )


# -------------------------------------------------------- _select(subjects=None)

class TestSelectWithoutSubjectsIsUnchanged:
    """`_select`'s `subjects=None` path must stay byte-identical to the
    permutation-then-slice algorithm it always was. Pinned against a
    hand-replayed `rng.permutation`, independent of `_select`'s own source,
    so a later edit to the `subjects is not None` branch cannot silently
    shift this one.
    """

    def test_matches_a_hand_replayed_permutation(self):
        pool_per_class = 400
        labels = np.repeat(np.arange(config.CLASSES), pool_per_class)
        reserve = 5
        needed = config.BAGS_PER_CLASS * config.INSTANCES_PER_BAG

        rng = np.random.default_rng(123)
        chosen, spare = bags._select(labels, rng, reserve, subjects=None)

        rng_replay = np.random.default_rng(123)
        for class_id in range(config.CLASSES):
            pool = np.flatnonzero(labels == class_id)
            drawn = rng_replay.permutation(pool)
            assert np.array_equal(chosen[class_id], drawn[:needed])
            assert np.array_equal(spare[class_id], drawn[needed:needed + reserve])


# --------------------------------------------------------- end-to-end builds

class TestToyBagSetBuildsEndToEnd:
    @pytest.mark.parametrize("code", sorted(toy.TOY_CODES))
    def test_builds_with_the_right_shapes_and_rebuilds_exactly(self, code, tmp_path):
        bagset = bags.build(code, tmp_path, seed=0, noise=0.0)
        dim = _DIM[code]
        n_images = config.CLASSES * config.BAGS_PER_CLASS * config.INSTANCES_PER_BAG
        assert bagset.images.shape == (n_images, dim)
        assert bagset.members.shape == (config.BAGS_PER_DOMAIN, config.INSTANCES_PER_BAG)
        assert bagset.labels.shape == (config.BAGS_PER_DOMAIN,)
        assert bagset.train_idx.shape[0] == config.TRAIN_BAGS
        assert bagset.valid_idx.shape[0] == config.VALID_BAGS
        assert bagset.eval_idx.shape[0] == config.EVAL_BAGS

        rebuilt = bags.rebuild(bagset.manifest, tmp_path)
        assert torch.equal(rebuilt.images, bagset.images)
        assert torch.equal(rebuilt.members, bagset.members)
        assert torch.equal(rebuilt.labels, bagset.labels)
        assert torch.equal(rebuilt.train_idx, bagset.train_idx)
        assert torch.equal(rebuilt.valid_idx, bagset.valid_idx)
        assert torch.equal(rebuilt.eval_idx, bagset.eval_idx)


class TestToyBagSetBuildsWithContamination:
    def test_contamination_lands_in_the_declared_donor_class(self, tmp_path):
        bagset = bags.build("r", tmp_path, seed=0, noise=0.3)
        contamination = bagset.manifest["contamination"]
        assert contamination["bags"], "expected at least one contaminated bag"

        raw = toy.build_toy_dataset("r")
        for record in contamination["bags"]:
            own_label = record["label"]
            expected_donor = config.NOISE_PAIRING[own_label]
            assert all(d == expected_donor for d in record["donorClasses"])
            for image_index in record["imageIndices"]:
                assert int(raw.targets[image_index]) == expected_donor
