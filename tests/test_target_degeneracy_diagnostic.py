"""r25:1087's target-degeneracy diagnostic: quadratic entropy (Eq. 10) over
the trace-normalized UNCONDITIONAL target bag Gram, and the two readings it
must never be confused with.

r25 declares a failure mode left "al terreno empirico": `theta_t` receives
gradient only through Eqs. 37-38, so a target encoder that concentrates its
instances in an arbitrarily small region can drive Eq. (24)'s confidence
weights to zero (or to a confidently wrong single class) before Eq. 38 stops
penalizing -- a collapse invisible in `targetAccuracy` alone. The same
sentence that prescribes the instrument excludes a different one that looks
similar and does not detect this case (the intra-bag attention collapse), and
a third candidate elsewhere in this codebase (the per-class weighted
`H_target` inside the per-class score) is excluded for an
independent reason: it cannot see a collapse uniform across classes.

Level 2 tests (4.1.1-4.1.4) exercise `wiring._target_bag_entropy` directly,
over synthetic Gram matrices -- no encoder, no `Arm`, because the claim is
about the mathematics, not about training. 4.1.5-4.1.6 are scenario tests
against the two excluded instruments, built the same way. The wired
`training_step` tests (4.3 onward) live in this same file, added once the
instrument itself was proven correct in isolation.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn as nn

from MIL_CREDA import DTYPE
from MIL_CREDA.attention import effective_bag_size, uniform_self_similarity
from MIL_CREDA.bag_kernel import bag_kernel_matrix
from MIL_CREDA.conditional import conditional_blocks, mixed_matrix
from MIL_CREDA.global_term import balanced_dependency_score, is_active
from MIL_CREDA_Benchmark import config, harness, wiring

SIGMA = 1.5


# --------------------------------------------------------------------------
# 4.1.1-4.1.4 -- the diagnostic's own math and boundary, over synthetic Grams
# --------------------------------------------------------------------------


def test_a_rank_one_trace_normalized_target_gram_reads_the_diagnostics_minimum() -> None:
    """Spec `target-degeneracy-diagnostic`, "The reading decays toward its
    minimum...", scenario "A rank-one target Gram reads the diagnostic's
    minimum": a rank-one, trace-normalized Gram gives ||A||_F^2 = 1, hence
    H_2 = -ln(1) = 0, EXACTLY, not approximately.
    """
    K_tt = torch.ones((6, 6), dtype=DTYPE)  # rank one: every bag identical
    reading = wiring._target_bag_entropy(K_tt)
    assert float(reading) == pytest.approx(0.0, abs=1e-9)


def test_the_reading_falls_monotonically_as_effective_rank_decreases() -> None:
    """Same requirement, scenario "falls monotonically": a sequence of Grams
    moving from `I` (full effective rank, `H_2 = ln n`) to an all-ones matrix
    (rank one, `H_2 = 0`) produces a non-increasing sequence of readings.

    Measured, not assumed: `(1 - alpha) I + alpha * ones` was checked
    numerically before this test was written (alpha in [0, 1], six points) to
    confirm the sequence is in fact non-increasing before pinning it as a
    test, rather than trusting that an arbitrary interpolation must be
    monotonic.
    """
    n = 6
    identity = torch.eye(n, dtype=DTYPE)
    collapsed = torch.ones((n, n), dtype=DTYPE)
    alphas = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    readings = [
        float(wiring._target_bag_entropy((1.0 - a) * identity + a * collapsed))
        for a in alphas
    ]
    for earlier, later in zip(readings, readings[1:]):
        assert later <= earlier + 1e-9, (
            f"the reading rose from {earlier} to {later} as collapse increased"
        )
    assert readings[0] == pytest.approx(math.log(n), abs=1e-6)
    assert readings[-1] == pytest.approx(0.0, abs=1e-6)


def test_an_all_zero_target_gram_is_reported_not_raised() -> None:
    """Spec `target-degeneracy-diagnostic`, "The diagnostic reports rather
    than raises...", scenario "A non-positive-trace target Gram is reported,
    not raised": an all-zero Gram has trace 0, which `renyi.trace_normalize`
    refuses with `ValueError`. The diagnostic's own call site must catch it
    and report -- here, `float("nan")`, the sentinel `harness.accuracy` and
    `latent.geometry` already use for an absent reading -- rather than let
    the exception propagate.
    """
    K_tt = torch.zeros((5, 5), dtype=DTYPE)
    reading = wiring._target_bag_entropy(K_tt)  # must not raise
    assert math.isnan(float(reading))


def test_a_zero_frobenius_trace_normalized_gram_is_reported_not_raised(monkeypatch) -> None:
    """Same requirement, scenario "A zero-Frobenius-norm...is reported, not
    raised" -- `renyi.quadratic_entropy`'s OWN `ValueError`, distinct from
    `trace_normalize`'s.

    Measured, and named honestly rather than assumed reachable by ordinary
    composition: if `trace_normalize` succeeds, its output has trace exactly
    1 by construction (`K / trace(K)`), and a matrix of trace 1 cannot have
    zero Frobenius norm (`sum(A_ii) = 1` forces at least one nonzero diagonal
    entry, so `||A||_F^2 >= 1/n > 0`). So composing
    `quadratic_entropy(trace_normalize(K_tt))` on any real-valued `K_tt` can
    never reach `quadratic_entropy`'s own raise once `trace_normalize` has
    already succeeded -- the same shape of finding this repository has
    already named for an unreachable branch elsewhere (task 3.13). The
    boundary is still coded defensively (`_target_bag_entropy` guards both
    exceptions independently), and it is exercised here the same way this
    repository proves an unreachable-by-composition branch: by injection, not
    by pretending a natural trigger exists. `wiring.trace_normalize` is
    patched at the module level (the same mechanism Decision 4 already
    relies on for `FeatureExtractor`) to hand back a degenerate result
    `trace_normalize` itself could never actually produce.
    """
    monkeypatch.setattr(wiring, "trace_normalize", lambda K: torch.zeros_like(K))
    K_tt = torch.eye(4, dtype=DTYPE)  # positive trace; stubbed regardless
    reading = wiring._target_bag_entropy(K_tt)  # must not raise
    assert math.isnan(float(reading))


# --------------------------------------------------------------------------
# 4.1.5 -- the per-class weighted H_target is not a substitute
# --------------------------------------------------------------------------


def _bag(center: torch.Tensor, m: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`m` instances scattered around `center`, uniformly weighted."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(m, center.numel(), generator=generator, dtype=DTYPE) * 0.3
    H = center.unsqueeze(0) + noise
    weights = torch.full((m,), 1.0 / m, dtype=DTYPE)
    return H, weights


def test_a_class_uniform_pseudo_label_collapse_defeats_the_per_class_reading() -> None:
    """Spec `target-degeneracy-diagnostic`, "The per-class weighted `H_target`
    inside the per-class score MUST NOT substitute...", scenario "A
    class-uniform collapse defeats the per-class reading but not the
    required instrument": every target bag's hard pseudo-label collapses to
    the SAME class (here, by construction rather than through a trained
    head -- the claim is about `global_term`'s own per-class loop, not about
    how a pseudo-label is produced), while the unconditional target cloud
    (`K_tt`, built before any per-class conditioning) is ALSO collapsing.

    The required diagnostic reads its minimum (every target bag is the
    identical bag, so `K_tt` is exactly rank one). The per-class loop
    `_milcreda_term` runs (mirrored here over the same public functions,
    `conditional_blocks`/`mixed_matrix`/`is_active`/the score) finds
    target bags in exactly ONE class -- `C_act` has shrunk to a single class,
    the collapse the per-class reading cannot characterize
    on its own, demonstrating the required diagnostic is not redundant with
    it. Measured numerically before being pinned: with three declared
    classes and a target pseudo-label collapsed to one of them, exactly one
    class reaches `is_active`.
    """
    classes = 3
    centers = [
        torch.tensor([10.0, 0.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0, 10.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0, 0.0, 10.0], dtype=DTYPE),
    ]
    bags_s, source_labels = [], []
    seed = 1
    for class_id, center in enumerate(centers):
        for _ in range(2):
            bags_s.append(_bag(center, m=4, seed=seed))
            source_labels.append(class_id)
            seed += 1
    source_labels = torch.tensor(source_labels)

    # Every target bag is the identical bag -- the unconditional target cloud
    # collapses to a single point (Decision 9's rank-one boundary), and every
    # bag's pseudo-label collapses with it, to whichever class that one point
    # happens to sit closest to (class 1's center here; the choice is
    # arbitrary, the collapse is the point).
    collapsed_bag = _bag(centers[1], m=4, seed=99)
    bags_t = [collapsed_bag] * 5
    pseudo = torch.tensor([1] * 5)

    K_ss = bag_kernel_matrix(bags_s, bags_s, SIGMA)
    K_st = bag_kernel_matrix(bags_s, bags_t, SIGMA)
    K_tt = bag_kernel_matrix(bags_t, bags_t, SIGMA)

    diagnostic = wiring._target_bag_entropy(K_tt)
    assert float(diagnostic) == pytest.approx(0.0, abs=1e-6)

    losses = []
    for class_id in range(classes):
        s_idx = torch.nonzero(source_labels == class_id, as_tuple=False).reshape(-1)
        t_idx = torch.nonzero(pseudo == class_id, as_tuple=False).reshape(-1)
        if s_idx.numel() == 0 or t_idx.numel() == 0:
            continue
        K_s_c, K_st_c, K_t_c = conditional_blocks(K_ss, K_st, K_tt, s_idx, t_idx)
        K_mix = mixed_matrix(K_s_c, K_st_c, K_t_c)
        if not is_active(K_s_c, K_t_c, K_mix):
            continue
        # r33's score. It was r25's `dependency_score` until 2026-10-18, when
        # that function was removed with the ends it was bounded by; this
        # assertion is about how many classes stay ACTIVE, not about either
        # score's value, so the substitution leaves the claim untouched.
        losses.append(balanced_dependency_score(K_s_c, K_st_c, K_t_c))

    assert len(losses) == 1, (
        "the per-class loop should have at most one active class once every "
        "target pseudo-label collapses to it -- the required diagnostic "
        "still reads the whole cloud regardless of how many classes remain"
    )


# --------------------------------------------------------------------------
# 4.1.6 -- the excluded intra-bag instruments do not detect this collapse
# --------------------------------------------------------------------------


def test_the_excluded_intra_bag_instruments_do_not_detect_a_collapsed_target_cloud() -> None:
    """Spec `target-degeneracy-diagnostic`, "The excluded intra-bag
    instruments MUST NOT be used", scenario "A collapsed target bag cloud is
    invisible to the excluded instruments": `MIL_CREDA.attention.
    effective_bag_size` and `uniform_self_similarity` read WITHIN one bag;
    they say nothing about whether the CLOUD of bags has collapsed.

    Every target bag here is the identical bag -- five distinct instances
    (real intra-bag diversity) under a deliberately non-uniform weighting.
    Both excluded instruments read that single bag as unremarkable (`m_eff`
    close to its instance count, `uniform_self_similarity` well below its
    ceiling of 1 -- r25:516's own "expected behaviour in a dispersed bag").
    Meanwhile the required diagnostic, read on the SAME batch, is at its
    exact minimum: every bag is literally the same bag, so `K_tt` is
    rank-one by construction. Measured numerically before being pinned.
    """
    H = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=DTYPE,
    )
    weights = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10], dtype=DTYPE)

    m_eff = effective_bag_size(weights)
    self_similarity = uniform_self_similarity(H, SIGMA)
    assert float(m_eff) > H.shape[0] * 0.6, "m_eff read as collapsed, not dispersed"
    assert float(self_similarity) < 0.9, "self-similarity read as collapsed, not dispersed"

    bags_t = [(H, weights)] * 6
    K_tt = bag_kernel_matrix(bags_t, bags_t, SIGMA)
    diagnostic = wiring._target_bag_entropy(K_tt)
    assert float(diagnostic) == pytest.approx(0.0, abs=1e-6), (
        "the required diagnostic must read the collapse the excluded "
        "instruments above could not"
    )


# --------------------------------------------------------------------------
# Fixtures for the wired `training_step` tests below -- a local stub, not an
# import from `tests/test_arm_objectives.py` (Phase 3's own precedent,
# `tests/test_domain_mixup_guard.py`, defines its own rather than
# cross-importing a sibling test module's private helpers).
# --------------------------------------------------------------------------


class _Encoder(nn.Module):
    """Something with an `output_dim` that maps an instance to a row."""

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


def _pool(seed: int) -> wiring.Pool:
    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, 3, 8, 8,
                         generator=generator)
    members = torch.arange(images.shape[0]).reshape(bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % config.CLASSES
    return wiring.Pool(images, members, labels)


def _arm(monkeypatch, arm_id: str):
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)
    torch.manual_seed(11)
    return wiring.build(arm_id, config.CLASSES, _pool(1), _pool(2))


def _step(arm: wiring.Arm, generator: torch.Generator) -> dict:
    x = arm.source.take(torch.arange(config.BAGS_PER_STEP))
    y = arm.source.labels[:config.BAGS_PER_STEP]
    return arm.training_step(x, y, 0.5, generator)


_DIAGNOSTIC_FIELDS = (
    "targetBagEntropy", "targetConfidenceMean", "targetConfidenceMin",
    "activeClasses", "sourceKernelMean", "crossKernelMean", "targetKernelMean",
)


# --------------------------------------------------------------------------
# 4.3 -- the wired readings, on a real adapted arm's real `training_step`
# --------------------------------------------------------------------------


def test_an_adapted_arms_training_step_reports_all_seven_readings(monkeypatch) -> None:
    """Task 4.3: `training_step` reports `targetBagEntropy` (4.2) alongside
    `targetConfidenceMean`/`targetConfidenceMin`/`activeClasses`/
    `sourceKernelMean`/`crossKernelMean`/`targetKernelMean`, read off the
    real matrices an adapted arm's own `_milcreda_term` builds -- not a
    fabricated or hand-assembled dict.
    """
    arm = _arm(monkeypatch, "G")
    step = _step(arm, torch.Generator().manual_seed(3))

    assert not math.isnan(step["targetBagEntropy"])
    assert 0.0 <= step["targetConfidenceMean"] <= 1.0
    assert 0.0 <= step["targetConfidenceMin"] <= 1.0
    assert step["targetConfidenceMin"] <= step["targetConfidenceMean"]
    assert 1 <= step["activeClasses"] <= config.CLASSES
    for name in ("sourceKernelMean", "crossKernelMean", "targetKernelMean"):
        assert not math.isnan(step[name])


# --------------------------------------------------------------------------
# 4.4 -- the floor reports NaN, never 0.0, and the guard can fail
# --------------------------------------------------------------------------


def test_the_floor_reports_nan_for_every_degeneracy_reading_never_zero(monkeypatch) -> None:
    """Design Decision 8: `0.0` is exactly the collapsed value `targetBagEntropy`
    reports at its minimum, so a floor defaulting to `0.0` would read as
    maximally degenerate on every run forever. The floor never lets a target
    image reach an encoder (Decision 2), so it structurally cannot build
    `K_tt` at all -- every one of these seven fields must be `nan`.

    Mutation-verified reachable: this assertion was observed to RED under the
    mutation "report 0.0 instead of nan on the floor" (`wiring.py`'s
    `diagnostics` nan-default dict, temporarily replaced with `0.0` for every
    field), then reverted, then re-observed GREEN -- `git diff --numstat`
    confirmed the mutation actually landed and was actually reverted, per
    this repository's own discipline against a mutation that silently
    patches nothing.
    """
    floor = _arm(monkeypatch, "B")
    step = _step(floor, torch.Generator().manual_seed(3))

    for name in _DIAGNOSTIC_FIELDS:
        assert math.isnan(step[name]), (
            f"{name}: the floor must read nan, never a fabricated number"
        )


# --------------------------------------------------------------------------
# 4.5 -- per-epoch aggregation and the one per-run summary field
# --------------------------------------------------------------------------


def _bagset(domain: str, seed: int):
    """Ten training bags, one per class -- `tests/test_independent_adaptation_
    schedules.py`'s own fixture shape, reused rather than reinvented for the
    identical constraint: `total_correspondence` needs a source bag in every
    class the target pseudo-labels reach.
    """
    from MIL_CREDA_Benchmark import bags

    generator = torch.Generator().manual_seed(seed)
    bagcount, per_bag = config.CLASSES + 10, 3
    images = torch.randn(bagcount * per_bag, 3, 8, 8, generator=generator)
    members = torch.arange(bagcount * per_bag).reshape(bagcount, per_bag)
    labels = torch.arange(bagcount) % config.CLASSES
    return bags.BagSet(
        domain=domain, images=images, members=members, labels=labels,
        train_idx=torch.arange(0, config.CLASSES),
        valid_idx=torch.arange(config.CLASSES, config.CLASSES + 5),
        eval_idx=torch.arange(config.CLASSES + 5, config.CLASSES + 10),
        manifest={})


def test_run_ones_record_carries_the_per_epoch_trace_and_its_per_run_minimum(
        monkeypatch) -> None:
    """Design Decision 8's chosen shape: `training_step`'s per-step readings
    are aggregated into `epochs_record` beside `sourceAccuracy`/
    `targetAccuracy` (min for `targetBagEntropy` -- "the one reading the
    epoch trace makes awkward", per the design -- mean for the rest), plus
    ONE per-run field, `targetBagEntropyMin`: the minimum over epochs, never
    a new `config.DIMENSIONS` entry.

    A real (stub-encoder) two-epoch run, end to end through `harness.run_one`
    -- not a hand-assembled dict -- so this is proof the wiring actually
    reaches the returned record, not merely that the formula is right in
    isolation.
    """
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder)
    torch.manual_seed(11)

    reduction = harness.Reduction(epochs=2, seeds=[0])
    material = {"source": _bagset("S", 1), "target": _bagset("M", 2)}

    result = harness.run_one("G", ("S", "M"), 0, reduction, torch.device("cpu"),
                             material, role="valid")

    for point in result["epochs"]:
        for name in _DIAGNOSTIC_FIELDS:
            assert name in point, f"{name} missing from an epoch record"
            assert not math.isnan(point[name]), (
                f"{name}: an adapted arm's epoch record must not read nan"
            )

    assert "targetBagEntropyMin" in result
    epoch_minima = [point["targetBagEntropy"] for point in result["epochs"]]
    assert result["targetBagEntropyMin"] == pytest.approx(min(epoch_minima))

    # Decision 8: never a declared dimension, on the real returned record.
    assert "targetBagEntropy" not in config.DIMENSIONS
    assert "targetBagEntropyMin" not in config.DIMENSIONS


# --------------------------------------------------------------------------
# 4.6 -- absent from every declared dimension set (design Decision 8's guard)
# --------------------------------------------------------------------------


def test_the_degeneracy_diagnostic_fields_are_absent_from_every_declared_dimension_set() -> None:
    """Design Decision 8 rejects a seventh `config.DIMENSIONS` entry (the
    floor structurally cannot produce it, `ladder_rows` would compare a
    fabricated floor value against a real one, and `poolable` is a measured
    claim with no replication behind this reading) and rejects declaring it
    in `__benchmark__["report"]["dimensions"]` (membership there IS the
    render declaration). This guard exists so a later PR does not silently
    re-add either.
    """
    from MIL_CREDA_Benchmark import __benchmark__

    names = {
        "targetBagEntropy", "targetBagEntropyMin", "targetConfidenceMean",
        "targetConfidenceMin", "activeClasses", "sourceKernelMean",
        "crossKernelMean", "targetKernelMean",
    }
    assert names.isdisjoint(config.DIMENSIONS), (
        "a degeneracy-diagnostic field was declared in config.DIMENSIONS"
    )
    assert names.isdisjoint(__benchmark__["report"]["dimensions"]), (
        "a degeneracy-diagnostic field was declared in the render surface"
    )


# --------------------------------------------------------------------------
# 4.7 -- readable during training, not only at the end
# --------------------------------------------------------------------------


def test_the_diagnostic_is_readable_at_more_than_one_point_over_training(monkeypatch) -> None:
    """Spec `target-degeneracy-diagnostic`, "The diagnostic is readable
    during training, not only after", scenario "available at more than one
    point across a training run": three successive `training_step` calls,
    sharing one generator (so each draws a genuinely different target batch,
    exactly as `run_one`'s own epoch loop does), yield distinct readings --
    a trajectory, not a single frozen number.
    """
    arm = _arm(monkeypatch, "G")
    generator = torch.Generator().manual_seed(3)
    readings = [_step(arm, generator)["targetBagEntropy"] for _ in range(3)]

    assert len(set(readings)) > 1, (
        "the diagnostic read the identical value at every step, hiding its "
        "trajectory across training"
    )


# --------------------------------------------------------------------------
# 4.8 -- a NaN-carrying run dict survives its own on-disk round trip
# --------------------------------------------------------------------------


def test_a_run_dict_with_nan_degeneracy_readings_round_trips_through_json() -> None:
    """Precedent: `harness.accuracy`/`latent.geometry` already write
    `float("nan")` into a run dict, and `runs.jsonl` is written with plain
    `json.dumps` (`harness.py`), which accepts NaN by default. The floor's
    own reading (task 4.4) is the concrete case this guards.
    """
    run = {name: float("nan") for name in _DIAGNOSTIC_FIELDS}
    run["targetBagEntropyMin"] = float("nan")

    dumped = json.dumps(run)
    loaded = json.loads(dumped)

    for name, value in run.items():
        assert math.isnan(loaded[name]) == math.isnan(value)
