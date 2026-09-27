"""Eq. (39)'s two coefficients, each with its own schedule, at the harness level.

`wiring.Arm.training_step` (see `tests/test_arm_objectives.py`) is one half:
it accepts two independent coefficients and assembles `total_objective` with
them apart. This file is the other half -- that `harness.run_one` actually
RESOLVES two different numbers (its own ceiling and its own growth rate for
each term) and hands them to `training_step` in the right slots, that its
returned record carries the two resulting contributions apart with the old
sum still derivable, and that a reader built for the single old ceiling
(`config.ceilings_on_record`) keeps reading the GLOBAL one rather than
silently reading whichever of the two happened to land in that key.

Nothing here trains for real except the one full-pipeline test, which stubs
the encoder the same way `tests/test_arm_objectives.py` does, so the suite
stays offline and fast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from MIL_CREDA_Benchmark import config, harness, wiring


# --------------------------------------------------------- resolution alone

def test_hyper_for_resolves_the_local_ceiling_and_delta_independently_of_the_global_ones(
) -> None:
    """`ceiling_for` (unchanged by this stretch) still answers the GLOBAL
    ceiling alone. `ceilingLocal`/`rampDeltaLocal` are the two new dimensions
    `hyper_for` resolves through the identical per-transfer/pooled-fallback
    rule the other five (now: `rampDelta`, `kernelSigma`, `attentionGamma`,
    `attentionTemperature`, `tauLocal`) already had.

    Reachable red: drop `ceilingLocal`/`rampDeltaLocal` from `hyper_for`'s
    returned dict, or resolve them from the SAME `reduction.ceilings`/
    `rampDelta` fields the global ceiling reads instead of their own.
    """
    reduction = harness.Reduction(
        ceilings={"milcreda": 0.9}, ceilingsByTransfer={},
        ceilingLocal=0.05, rampDelta=40.0, rampDeltaLocal=3.0,
    )
    transfer = ("S", "M")

    assert harness.ceiling_for(reduction, "milcreda", transfer) == 0.9

    resolved = harness.hyper_for(reduction, "milcreda", transfer)
    assert resolved["ceilingLocal"] == 0.05
    assert resolved["rampDelta"] == 40.0
    assert resolved["rampDeltaLocal"] == 3.0
    # the pair is genuinely independent, not one value read twice under two names
    assert resolved["ceilingLocal"] != harness.ceiling_for(reduction, "milcreda", transfer)
    assert resolved["rampDeltaLocal"] != resolved["rampDelta"]


def test_hyper_for_reads_a_transfers_own_searched_local_ceiling_before_the_pooled_default(
) -> None:
    """The same two-reading rule `ceiling_for` already applies to the global
    ceiling, carried to the local one via `reduction.hyperByTransfer` --
    a transfer the search measured keeps its own winner rather than the
    scalar fallback.

    Reachable red: `hyper_for` ignoring `hyperByTransfer["ceilingLocal"]`/
    `["rampDeltaLocal"]` and always returning the pooled scalar fields.
    """
    reduction = harness.Reduction(
        ceilingLocal=1.0, rampDeltaLocal=20.0,
        hyperByTransfer={"milcreda": {"S->M": {"ceilingLocal": 0.001,
                                               "rampDeltaLocal": 77.0}}},
    )
    measured = harness.hyper_for(reduction, "milcreda", ("S", "M"))
    assert measured["ceilingLocal"] == 0.001
    assert measured["rampDeltaLocal"] == 77.0

    unmeasured = harness.hyper_for(reduction, "milcreda", ("M", "U"))
    assert unmeasured["ceilingLocal"] == 1.0
    assert unmeasured["rampDeltaLocal"] == 20.0


# ------------------------------------------------------- reaching run_one

def _synthetic_bagset(domain: str) -> object:
    """The same minimal `BagSet` shape `tests/test_benchmark_declarations.py`
    builds for a `run_one` call that never needs real material."""
    from MIL_CREDA_Benchmark import bags

    bagcount, per_bag = 12, 2
    return bags.BagSet(
        domain=domain,
        images=torch.zeros(bagcount * per_bag, 3, 32, 32),
        members=torch.arange(bagcount * per_bag).reshape(bagcount, per_bag),
        labels=torch.arange(bagcount) % config.CLASSES,
        train_idx=torch.arange(0, 6),
        valid_idx=torch.arange(6, 9),
        eval_idx=torch.arange(9, 12),
        manifest={},
    )


class _StopAtSecondRamp(Exception):
    """Raised once both the global and the local ramp have been resolved."""


def test_run_one_resolves_both_ramps_from_independent_bounds_before_training(
        monkeypatch) -> None:
    """`run_one` reaching `ramp()` TWICE per epoch -- once with the GLOBAL
    ceiling/delta (`ceiling_for`/`reduction.rampDelta`), once with the LOCAL
    ones (`hyper_for`'s `ceilingLocal`/`rampDeltaLocal`) -- before a single
    batch is drawn. `wiring.build` is stubbed to a bare `nn.Linear` exactly as
    `test_run_one_resolves_the_ceiling_of_the_transfer_it_was_given`
    (`tests/test_benchmark_declarations.py`) already does, because what is
    being checked is which two numbers reach the schedule, never what a real
    model does with them.

    Reachable red: `run_one` still calling `ramp()` once per epoch (the old
    shape), which would leave `seen` with only one pair and the second
    `assert` unreachable in a way that reads as this test being wrong rather
    than the production code.
    """
    seen: list[tuple[float, float]] = []

    def spy(epoch, epochs, family, ceiling, delta=None):
        seen.append((ceiling, delta))
        if len(seen) == 2:
            raise _StopAtSecondRamp
        return 0.0

    monkeypatch.setattr(harness, "ramp", spy)
    monkeypatch.setattr(wiring, "build", lambda *a, **k: torch.nn.Linear(1, 1))

    reduction = harness.Reduction(
        epochs=1, seeds=[0],
        ceilings={"milcreda": 0.9}, ceilingsByTransfer={},
        rampDelta=30.0,
        ceilingLocal=0.1, rampDeltaLocal=5.0,
    )
    material = {"source": _synthetic_bagset("S"), "target": _synthetic_bagset("M")}

    with pytest.raises(_StopAtSecondRamp):
        harness.run_one("G", ("S", "M"), 0, reduction, torch.device("cpu"),
                        material, role="valid")

    assert seen == [(0.9, 30.0), (0.1, 5.0)], seen


def test_run_mechanism_also_resolves_both_ramps_independently(monkeypatch) -> None:
    """`run_mechanism` (Section 4's attention-mechanism comparison) mirrors
    `run_one` deliberately closely -- its own docstring says so -- and shared
    `run_one`'s pre-fix bug (`total_objective(..., coefficient, coefficient)`,
    reached via `wiring.MechanismArm.training_step`, which it inherits
    unchanged from `Arm`) until this same stretch of work fixed both. This is
    not a named item of the ceiling-search task this stretch was scoped to,
    but leaving it unfixed would reintroduce the identical defect in a
    sibling function that reads the exact same `ceiling_for`/`hyper_for`.

    Reachable red: `run_mechanism` still calling `ramp()` once per epoch.
    """
    seen: list[tuple[float, float]] = []

    def spy(epoch, epochs, family, ceiling, delta=None):
        seen.append((ceiling, delta))
        if len(seen) == 2:
            raise _StopAtSecondRamp
        return 0.0

    monkeypatch.setattr(harness, "ramp", spy)
    monkeypatch.setattr(wiring, "build_mechanism", lambda *a, **k: torch.nn.Linear(1, 1))

    reduction = harness.Reduction(
        epochs=1, seeds=[0],
        ceilings={"milcreda": 0.7}, ceilingsByTransfer={},
        rampDelta=11.0,
        ceilingLocal=0.02, rampDeltaLocal=8.0,
    )
    material = {"source": _synthetic_bagset("S"), "target": _synthetic_bagset("M")}

    with pytest.raises(_StopAtSecondRamp):
        harness.run_mechanism("ours", ("S", "M"), 0, reduction, torch.device("cpu"),
                              material, role="valid")

    assert seen == [(0.7, 11.0), (0.02, 8.0)], seen


# ------------------------------------------------------------ the full record

class _Encoder8(nn.Module):
    """A stub encoder for 8x8 instances, the same shape
    `tests/test_arm_objectives.py`'s own `_Encoder` uses, duplicated here so
    this file stays self-contained the way `tests/test_search_declarations.py`
    and its siblings already are."""

    def __init__(self, backbone=None, pretrained=False):
        super().__init__()
        self.output_dim = 6
        self.linear = nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


def _bagset(domain: str, seed: int):
    """Ten training bags, one per class: Eq. (29)'s local correspondence is
    undefined for a target class with no source bag (`MIL_CREDA.local_term.
    total_correspondence`'s own guard), so a synthetic material smaller than
    `config.CLASSES` in its training role raises there instead of training.
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


def test_run_ones_record_carries_the_two_contributions_apart_and_their_derivable_sum(
        monkeypatch) -> None:
    """The end-to-end claim: a REAL (stub-encoder) `Arm` trained for two epochs
    under a GLOBAL ceiling of 1.0 and a LOCAL ceiling of 1e-4 -- four orders of
    magnitude apart, so an accidental coupling back to one shared coefficient
    would be visible rather than lost in floating-point noise -- reports
    `contributionGlobal` and `contributionLocal` that differ, and whose sum is
    exactly the reported `contribution`.

    Reachable red: `run_one` computing one coefficient (the old shape) and
    handing it to `training_step` as both `ramp` and `ramp_local`, which
    would make `contributionGlobal`/`contributionLocal` scale together
    instead of apart -- caught by the ratio assertion below, not merely by
    inequality, since two independently-initialized terms could differ by
    chance even under one shared coefficient.
    """
    monkeypatch.setattr(wiring, "FeatureExtractor", _Encoder8)
    torch.manual_seed(11)

    reduction = harness.Reduction(
        epochs=2, seeds=[0],
        ceilings={"milcreda": 1.0}, ceilingsByTransfer={},
        rampDelta=20.0,
        ceilingLocal=1e-4, rampDeltaLocal=20.0,
    )
    material = {"source": _bagset("S", 1), "target": _bagset("M", 2)}

    result = harness.run_one("G", ("S", "M"), 0, reduction, torch.device("cpu"),
                             material, role="valid")

    assert result["contribution"] == pytest.approx(
        result["contributionGlobal"] + result["contributionLocal"], abs=1e-9)
    assert result["contributionGlobal"] != pytest.approx(result["contributionLocal"])
    # the global ceiling outruns the local one by 1e4: its contribution should
    # not merely differ, it should dominate -- the signature of two genuinely
    # independent coefficients rather than two terms that happened to differ
    assert abs(result["contributionGlobal"]) > 10 * abs(result["contributionLocal"])


# --------------------------------------------------- a reader of the old field

def test_config_ceilings_on_record_still_reads_only_the_global_ceiling(
        tmp_path, monkeypatch) -> None:
    """`config.ceilings_on_record` -- the reader `tables.py` and `campaign()`
    build every declared-arm run on -- is a reader of the record's flat
    `"ceiling"` key, and that key was never repointed: it names the GLOBAL
    ceiling before this stretch of work and it names the GLOBAL ceiling after
    it, deliberately, because `tables.py` (outside this stretch's authorized
    scope) already reads `entrada["ceiling"]` as if one number governed the
    whole term. `"ceilingLocal"` lives beside it, inside `perTransfer`, and
    this reader does not surface it -- the asymmetry is declared here rather
    than left to be discovered as a silent narrowing.

    Reachable red: `search_ceilings_trials` stopping writing a flat
    `"ceiling"` key at all (renaming it, say, to `"ceilingGlobal"`), which
    would make this reader -- and every out-of-scope caller of it -- silently
    return nothing for every family.
    """
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "CEILINGS_RECORD", tmp_path / "ceilings.json")
    monkeypatch.setattr(config, "CEILINGS_PILOT_RECORD", tmp_path / "ceilings.pilot.json")
    monkeypatch.setattr(config, "SEARCH_ENGINE", "grid")
    monkeypatch.setattr(config, "CEILING_GRID", [0.5, 1.0])
    monkeypatch.setattr(config, "SEARCH_SEEDS", [0])
    monkeypatch.setattr(config, "SEARCH_ARMS", {"milcreda": "G"})

    class _Bags:
        pass

    def _build(code, cache, seed, noise=0.0):
        return _Bags()

    def _run_one(arm, transfer, seed, reduction, device, material, *, ceiling, role):
        return {config.SEARCH_CRITERION: 1.0 if ceiling == 1.0 else 0.1}

    monkeypatch.setattr(harness.bags, "build", _build)
    monkeypatch.setattr(harness, "run_one", _run_one)
    harness.search_ceilings(
        harness.Reduction(seeds=[0], epochs=config.SEARCH_EPOCHS),
        "cpu", progress=lambda *_: None, transfers=list(config.SEARCH_TRANSFERS))

    assert config.ceilings_on_record() == {"milcreda": 1.0}
