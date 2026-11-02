"""What `toy_run.py` claims, and the red each claim can still reach.

`toy_run` is the runtime the MIL-CREDA toyset tutorial trains with. Where a
claim needs a trained model, phase 1 and phase 2 run through the tutorial's OWN
functions (`entrenar_fase_uno`, `entrenar_fase_dos`, `lote`), loaded from
`MIL-CREDA/Notebooks/Toyset.ipynb` without its training loops -- never through
a runner of this module and never through a loop written here.

Every test here is written so that breaking the thing it names turns it red. The
two habits that make that true rather than hoped for:

- **Never an expression against itself.** A test that compared one call with
  the same call would measure nothing. So each test first establishes
  that the two readings it is about to distinguish ACTUALLY DIFFER on the
  fixture, and only then asserts which one the code took. Without that
  precondition the assertion passes on a fixture where both answers are equal,
  and the test is green while measuring nothing.
- **Mutate, then prove the mutation landed.** Where a claim is about a branch,
  the test perturbs the thing the branch reads and asserts the perturbation
  moved what it was supposed to move, before believing the result.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import math
import pathlib

import numpy as np
import pytest
import torch

from MIL_CREDA_Benchmark import (bags, config, harness, toy_bags, toy_run,
                                 wiring)
from MIL_CREDA_Benchmark import latent as latent_module

DEVICE = torch.device("cpu")

#: This toyset's own single arm, read off its own declaration and never spelled
#: out here: a test that hard-coded `"G"` would be asserting the digit
#: benchmark's identifier against a declaration that does not carry it.
#:
#: `FLOOR` stood beside it and is gone with the floor arm. Nothing here renames
#: it to point at the surviving arm: a name that resolved would make every
#: assertion that used to distinguish the two pass by comparing one arm with
#: itself, which is the one shape this file's own docstring forbids.
ADAPTED = toy_run.TOY_ADAPTED

#: A spec that is NOT a declared arm: no adaptation term, no weighting, no
#: local correspondence. It exists so the assertions about what the declared
#: arm computes keep BOTH poles after the floor arm's retirement -- "the term
#: did nothing" and "the term was never called" print the same `0.0`, and
#: without something that genuinely computes neither term the zero side of that
#: discrimination has nothing standing on it.
#:
#: It is built through `wiring.Arm` directly and never through
#: `toy_run.build_arm`, which refuses an id the module does not declare. That
#: refusal is the point: this is a test control and must not be reachable as an
#: arm of the run, so it carries no `id` the module could resolve.
_UNADAPTED_CONTROL_SPEC = {
    "id": "not-an-arm", "name": "test control", "label": "not this experiment",
    "unit": "bag", "adaptation": None, "weighting": False, "local": False,
    "attention": "learned", "selection": None, "normalization": "shared",
}


def _unadapted_control(material, scale, hyper, device, seed=0) -> wiring.Arm:
    """`_UNADAPTED_CONTROL_SPEC` as a built `wiring.Arm`, on this material.

    The same construction `toy_run.build_arm` performs, minus its id lookup --
    seeded identically, so the shared parameters come out bit-identical to a
    declared arm built on the same material and the control differs from it in
    the spec and in nothing else.
    """
    torch.manual_seed(seed)
    source, target = material["source"], material["target"]
    return wiring.Arm(
        _UNADAPTED_CONTROL_SPEC, scale.classes,
        harness.pool_of(source, source.train_idx, device),
        harness.pool_of(target, target.train_idx, device),
        hyper=hyper,
    ).to(device)


@pytest.fixture(scope="module")
def tiny() -> toy_run.Scale:
    """A scale smaller than `PILOT` and otherwise identical to it, so a test
    that needs a real run pays seconds rather than minutes.

    **It lowers `instances_per_bag`, which no real run does.** A bag holds 30
    instances at `PILOT` and at `FULL`, and at 30 instances over 81 bags the
    draw would need `points_per_domain` at 2586 or above -- minutes per test.
    So it declares 6 instances over 810 points: 81 bags at 10 points per bag,
    whose smallest quantile band is 8 over the thirteen seeds either scale
    draws, so `_instances_per_bag` funds it.

    `epochs_2` is `MIN_EPOCHS_2 + 1`: enough passes for the phase-2 criterion
    to become legal and then choose.
    """
    return dataclasses.replace(toy_run.PILOT, name="test", epochs_1=2,
                               epochs_2=toy_run.MIN_EPOCHS_2 + 1, patience_1=1,
                               points_per_domain=810, instances_per_bag=6)


@pytest.fixture(scope="module")
def familia(tiny) -> toy_bags.Familia:
    return toy_run.build_family("lunas", 0, tiny)


NOTEBOOK = (pathlib.Path(__file__).resolve().parents[1] / "MIL-CREDA"
            / "Notebooks" / "Toyset.ipynb")


def _tutorial_functions(scale: toy_run.Scale, hyper: dict) -> dict:
    """The tutorial's setup and its function definitions, without its loops.

    The setup cells (bootstrap, imports, budget, model choice) run whole; every
    later cell contributes only its `def`s, so nothing trains a family here.
    `ESCALA`, `HIPER` and `DISPOSITIVO` are then rebound to this file's scale,
    hyperparameters and CPU, which is what the tutorial's functions read.
    """
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    namespace: dict = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("FORGE_CLONE_ROOT", raising=False)
        patch.delenv("FORGE_CLONE_COMMIT", raising=False)
        patch.chdir(NOTEBOOK.parent)
        setup = True
        for index, cell in enumerate(cells):
            tree = ast.parse("".join(cell["source"]))
            defines = any(isinstance(node, ast.FunctionDef) for node in tree.body)
            if setup and defines and index > 0:
                setup = False
            if not setup:
                tree.body = [node for node in tree.body
                             if isinstance(node, ast.FunctionDef)]
            exec(compile(tree, f"{NOTEBOOK.name}[code {index}]", "exec"),
                 namespace)
    namespace.update(ESCALA=scale, HIPER=hyper, DISPOSITIVO=DEVICE)
    return namespace


@pytest.fixture(scope="module")
def notebook(tiny) -> dict:
    return _tutorial_functions(tiny, toy_run.declared_hyper(tiny))


def _parameter_vector(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _buffer_vector(module: torch.nn.Module) -> torch.Tensor:
    buffers = [b.detach().reshape(-1).float() for b in module.buffers()]
    return torch.cat(buffers) if buffers else torch.zeros(1)


def _train_phase_one(notebook, model, source_train, source_valid, seed=0,
                     hyper=None):
    """Phase 1 exactly as the tutorial trains it, at `hyper` when given."""
    functions = (notebook if hyper is None
                 else {**notebook, "HIPER": hyper})
    train = functions["entrenar_fase_uno"]
    if hyper is not None:
        train = type(train)(train.__code__, functions, train.__name__)
    return train(model, source_train, source_valid,
                 torch.Generator().manual_seed(seed + 9973))


def _batch(notebook, bagset, positions):
    """A lot of bags as the tutorial builds one: `(bags, instances, 3)`."""
    return notebook["lote"](bagset, positions)


def _prepared(notebook, familia, tiny, arm_id=ADAPTED, seed=0):
    """A phase-1-trained arm of `arm_id`, with theta_t copied and the source
    frozen: the state every phase-2 claim is about.
    """
    material = toy_run.material_for(familia, seed, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, source_valid, _ = bags.roles(material["source"])
        _, target_valid, _ = bags.roles(material["target"])
        model = toy_run.build_arm(arm_id, material, tiny, hyper, DEVICE, seed)
        phase_one = _train_phase_one(notebook, model, source_train,
                                     source_valid, seed)
        toy_run.copy_source_to_target(model)
        yield_payload = {
            "model": model, "material": material, "hyper": hyper,
            "source_train": source_train, "source_valid": source_valid,
            "target_valid": target_valid, "phase_one": phase_one,
        }
    return yield_payload


# ----------------------------------------------------------------- the freeze


def test_phase_one_leaves_every_shared_parameter_unrequiring_grad(
        notebook, familia, tiny):
    """The freeze is on the parameters themselves, not only on which ones the
    optimizer was handed."""
    payload = _prepared(notebook, familia, tiny)
    model = payload["model"]
    frozen = {name: p.requires_grad for name, p in model.named_parameters()
              if not name.startswith("target_encoder.")}
    assert frozen, "there are no shared parameters to freeze"
    assert not any(frozen.values()), [n for n, v in frozen.items() if v]
    trainable = [p.requires_grad for p in model.target_encoder.parameters()]
    assert trainable and all(trainable), "theta_t is not trainable"


def test_the_eval_mode_in_phase_two_is_what_keeps_the_source_buffers_still(
        notebook, familia, tiny, monkeypatch):
    """`set_phase_two_modes`'s `eval()` is load-bearing, not decoration.

    Disable it and the source encoder's BatchNorm buffers move during phase 2 --
    which is precisely the failure `no_grad` alone would not catch. This test is
    the reachable red for the buffer half of the freeze above.
    """
    payload = _prepared(notebook, familia, tiny, arm_id=ADAPTED)
    model = payload["model"]
    before = _buffer_vector(model.encoder).clone()
    monkeypatch.setattr(toy_run, "set_phase_two_modes", lambda m: m.train())
    with toy_run.material_bound(familia, tiny, "mlp"):
        generator = torch.Generator().manual_seed(1)
        notebook["entrenar_fase_dos"](model, payload["source_train"], generator)
    after = _buffer_vector(model.encoder)
    assert float((after - before).norm()) > 0.0, (
        "the source buffers stayed still even with the mode guard disabled; "
        "this fixture cannot show what the guard buys")


def test_theta_t_starts_bit_identical_to_theta_s(notebook, familia, tiny):
    """The one copy carries the buffers too.

    `Arm.sync_target_from_source` deliberately carries parameters only -- that is
    right for a repeated leash and wrong for CREDA's single
    `clone_for_target`, because a fresh BatchNorm starts at `running_mean = 0`
    and would not be the encoder phase 1 produced.
    """
    payload = _prepared(notebook, familia, tiny)
    model = payload["model"]
    assert float((_parameter_vector(model.encoder)
                  - _parameter_vector(model.target_encoder)).norm()) == 0.0
    assert float((_buffer_vector(model.encoder)
                  - _buffer_vector(model.target_encoder)).norm()) == 0.0


def test_the_buffer_copy_is_what_makes_the_start_identical(notebook, familia,
                                                          tiny):
    """Without the buffer half of `copy_source_to_target`, theta_t does not
    start as theta_s -- the reachable red for the test above."""
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, source_valid, _ = bags.roles(material["source"])
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
        _train_phase_one(notebook, model, source_train, source_valid)
        model.sync_target_from_source()          # parameters only, on purpose
        gap = float((_buffer_vector(model.encoder)
                     - _buffer_vector(model.target_encoder)).norm())
    assert gap > 0.0, (
        "the buffers already agreed after a parameter-only sync; this backbone "
        "carries no buffers and the claim is untestable here")


# ------------------------------------------------------------------- the scales


def test_the_requested_split_and_the_realized_split_both_travel():
    """CREDA's 0.6/0.2 are fractions of POINTS; this splits BAGS, and 27 bags per
    class cannot express them. Both numbers are carried so the request is never
    read as the measurement.

    The counts are DERIVED from the declared fractions here rather than copied
    from the module, so a fraction that moved without the roles moving with it
    is red: 27 x 0.6 = 16.2 rounds to 16 and 27 x 0.2 = 5.4 rounds to 5, which
    leaves 6.
    """
    scale = toy_run.FULL
    realized = scale.realized_split()
    assert realized["requested"]["train"] == 0.6
    assert realized["requested"]["valid"] == 0.2
    expected_train = round(scale.bags_per_class * scale.frac_train)
    expected_valid = round(scale.bags_per_class * scale.frac_valid)
    assert (expected_train, expected_valid) == (16, 5), (expected_train,
                                                         expected_valid)
    assert realized["bagsPerClass"] == {
        "train": expected_train, "valid": expected_valid,
        "test": scale.bags_per_class - expected_train - expected_valid}
    assert (realized["realized"]["train"]
            == expected_train / scale.bags_per_class)
    assert realized["realized"]["train"] != realized["requested"]["train"]


def test_the_instrument_has_eighteen_evaluation_bags_and_a_step_of_5_56_pp():
    """What the cut buys, as the number a reader compares two cells with.

    The accuracy every published cell carries is read over the `test` role of
    every class, so the finest step it can take is one bag out of that many.
    Asserted on BOTH scales because the bag programme is not a repetition
    field: a reduction that lowered the resolution would be measuring a
    different instrument from the one the verdict is sized for.

    Derived from the split rather than typed, and the reachable red is the
    arithmetic itself: the count is `test` per class times the classes, and the
    step is its reciprocal in percentage points. The previous cut of 12 bags
    per class gave 9 test bags and 11.11 pp, so both assertions move with the
    cut instead of standing as decoration.
    """
    for scale in (toy_run.FULL, toy_run.PILOT):
        per_class = scale.realized_split()["bagsPerClass"]
        assert per_class == {"train": 16, "valid": 5, "test": 6}, scale.name
        test_bags = per_class["test"] * scale.classes
        assert test_bags == 18, (scale.name, test_bags)
        assert 100.0 / test_bags == pytest.approx(5.5556, abs=1e-4), scale.name
        # The pole that makes the two assertions above a measurement: the cut
        # this replaces read 9 bags and 11.11 pp, so neither number survives a
        # silent return to it.
        assert test_bags != 9 and 100.0 / test_bags != pytest.approx(11.1111,
                                                                     abs=1e-4)


def test_evaluation_is_the_role_that_absorbs_the_rounding_and_is_taken_last():
    """`train` and `valid` round; `test` is the complement, so `test` is where
    the remainder lands.

    Both poles, because the direction is the whole claim. On CREDA's declared
    0.6/0.2 at 27 bags the exact ask is 16.2/5.4/5.4 and the realized split is
    16/5/6: `train` gave up 0.2 of a bag, `valid` gave up 0.4, and `test`
    received the 0.6 both of them shed. And a fraction pair whose asks are whole
    numbers leaves `test` exactly what it asked for -- so the assertion is about
    rounding and not about `test` always being the largest residue.

    **That second pole draws its own bag count**, because 27 is not divisible by
    four and no 0.x/0.y pair divides it into whole bags the way 0.5/0.25 divides
    12. What is under test is `realized_split`'s arithmetic and not the declared
    cut, so the pole declares the count it needs and says so here.

    `bags._split_counts`' own docstring describes this ordering as protecting
    the role the verdict rests on from the remainder. It is the role that
    RECEIVES it. Measured here rather than argued, because the verdict is read
    over `test` and a reader is owed which way the rounding went.
    """
    scale = toy_run.FULL
    bags_per_class = scale.bags_per_class
    exact_test = bags_per_class * (1.0 - scale.frac_train - scale.frac_valid)
    given_test = scale.realized_split()["bagsPerClass"]["test"]
    assert exact_test == pytest.approx(5.4)
    assert given_test == 6
    assert given_test > exact_test, (
        "evaluation did not absorb the remainder; it gave one up")

    # The other pole: a pair that divides exactly leaves nothing to absorb.
    whole = dataclasses.replace(scale, bags_per_class=12, frac_train=0.5,
                                frac_valid=0.25)
    exact = whole.bags_per_class * (1.0 - whole.frac_train - whole.frac_valid)
    assert exact == pytest.approx(3.0)
    assert whole.realized_split()["bagsPerClass"] == {"train": 6, "valid": 3,
                                                      "test": 3}

    # And a split that would leave evaluation empty is refused, not rounded
    # into a role that covers no class.
    starved = dataclasses.replace(scale, frac_train=0.9, frac_valid=0.1)
    with pytest.raises(ValueError, match="every role must cover every class"):
        starved.realized_split()


# ------------------------------------------------------- the frozen bandwidth

def test_the_frozen_sigma_reads_the_set_the_instance_kernel_evaluates(
        familia, tiny):
    """CREDA's statistic, over CREDA's own scope: the pairs its kernel sees.

    **The correction this test records, and why the previous version of it
    passed while the mechanism was off.** It used to assert the identity against
    `CREDA.toy_experiment.sigma_de_la_fuente` over the source's whole training
    role POOLED -- every instance against every instance, across bags and across
    classes. That is not the set Eq. (14) evaluates. The instance kernel reads
    instances WITHIN one bag, and a bag is a band of one class, so the pooled
    median reads a distance the kernel never measures: measured on the pilot
    record, 4 of the 9 cells came back outside `SIGMA_BAND`, all four saturated,
    which is the condition `embedding_scales`' own docstring names -- the
    mechanism numerically off while every flag says it is on.

    So the statistic is kept and its SCOPE is corrected. What is asserted:

    - the FORMULA is still CREDA's. `_median_pairwise` is
      `sqrt(median(d^2) + 1e-6)` and the median commutes with the monotone
      square root, so CREDA's own function applied to ONE bag has to agree with
      this module's within-bag median over that same bag -- to the epsilon
      CREDA's own formula carries and no further. That is the identity, and it
      is measured per bag rather than argued.
    - the SCOPE is within-bag pairs over the FULL training role, so neither the
      `limit` cap nor a shortened role can creep in.
    - the POOLED reading travels beside it, is a DIFFERENT number, and the
      record carries the ratio -- which is what makes the correction readable
      rather than a value that silently moved.
    """
    from CREDA.toy_experiment import sigma_de_la_fuente

    material = toy_run.material_for(familia, 0, tiny, "mlp")
    source = material["source"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        model = toy_run.build_arm(ADAPTED, material, tiny,
                                  toy_run.declared_hyper(tiny), DEVICE, 0)
        reading = toy_run.frozen_source_sigma(model, source, source.train_idx,
                                              DEVICE)
        rows, _ = latent_module.represent(model, source, source.train_idx,
                                          DEVICE, domain="source",
                                          unit="instance")
    detached = rows.detach()

    # The scope, named in the record and asserted against the set.
    assert reading["scope"] == toy_run.FROZEN_SIGMA_SCOPE
    assert reading["bags"] == len(source.train_idx) == 48
    assert reading["instances"] == 48 * tiny.instances_per_bag
    assert detached.shape[0] == reading["instances"]
    assert reading["bagsTruncated"] is False, (
        "the frozen statistic was taken over a capped sample of the role; the "
        "cap is `embedding_scales`' and must not reach this reading")

    # The FORMULA is CREDA's, measured where the two scopes coincide: over ONE
    # bag, "within-bag pairs" and "all pairs" are the same set, so CREDA's own
    # function and this module's within-bag median must agree there. Every bag
    # is checked, not one.
    embeddings = detached.reshape(reading["bags"], reading["instancesPerBag"], -1)
    per_bag_creda = [sigma_de_la_fuente(H) for H in embeddings]
    per_bag_here = [float(toy_run._off_diagonal(torch.cdist(H, H)).median())
                    for H in embeddings]
    worst = max(abs(a - b) for a, b in zip(per_bag_creda, per_bag_here))
    assert worst < 1e-5, (
        f"CREDA's own statistic and this module's within-bag median disagree by "
        f"{worst} on a single bag, where the two scopes are the same set: the "
        "formula is no longer CREDA's")

    # And the frozen value is the median over the UNION of those per-bag pair
    # sets, so it lies inside their range rather than beside it.
    assert min(per_bag_here) <= reading["sigma"] <= max(per_bag_here), (
        reading["sigma"], min(per_bag_here), max(per_bag_here))
    assert min(per_bag_here) < max(per_bag_here), (
        "every bag reads the same median on this fixture, so the bound above "
        "distinguishes nothing")

    # The pole: the POOLED reading -- the scope this correction left -- is a
    # different number, and the record carries it with the ratio.
    assert reading["pooledSigma"] == sigma_de_la_fuente(detached)
    assert reading["pooledSigma"] != reading["sigma"]
    assert reading["pooledToWithinRatio"] == pytest.approx(
        reading["pooledSigma"] / reading["sigma"])
    assert reading["pooledToWithinRatio"] > 1.0, (
        "the pooled median is not the larger of the two on this fixture; a bag "
        "is a band of one class and the pool is every class, so this is the "
        "one direction that holds by construction")


def test_the_frozen_bandwidth_leaves_the_instance_kernel_inside_the_band(
        notebook, familia, tiny):
    """What the corrected scope BUYS, measured on the mechanism and not on the
    bandwidth.

    Two readings, and the second is the one that matters:

    - **In band.** At the frozen bandwidth the within-bag kernel reads
      `exp(-1/(2m^2))` for the multiplier `m`, which lands inside `SIGMA_BAND`
      at the neutral multiplier the tutorial runs at. That is an identity rather than a
      measurement and it is said so: the number is checked against the measured
      value and against the identity, so the record cannot drift from either,
      but a green here proves the SCOPE and never the mechanism.
    - **And the mechanism.** The band exists to protect Eq. (16): a bandwidth
      that saturates the instance kernel makes the consensus term of Eq. (15) a
      constant added to every logit, and a softmax is invariant to a constant
      shift, so the attention becomes numerically mean pooling. That is the
      reading with a reachable red, and it is measured by taking the credit
      away -- the same weights, on the same embeddings, at the POOLED bandwidth
      this correction left.

    **`attentionGamma` has to be non-zero or this test measures nothing.**
    `config.ATTENTION_GAMMA` is `0.0`, and the consensus term is the ONLY path
    the bandwidth takes into Eq. (16), so under the declared hyper the weights
    are bit-identical at every bandwidth across four orders of magnitude --
    measured. This test supplies a non-zero gamma and asserts it reached the arm,
    because a fixture at the declared default would pass having compared one
    number with itself.

    **And phase 1 is TRAINED first, which is the moment the statistic declares
    and the first version of this test did not reach.** On the freshly built
    encoder the pooled and within-bag medians are nearly the same number
    (measured on this fixture: 1.17x, and the pooled bandwidth reads 0.234 --
    comfortably in band), so the two scopes are indistinguishable there and the
    pole below asserted something the fixture contradicts. Training is what
    separates them: `frozen_source_sigma` is called after phase 1 with
    `theta_s` frozen, which is CREDA's own moment, and there the ratio reaches
    13x on this family. The red this test could reach was a wrong claim of
    mine, not a defect in the code, and the fixture moved to the moment the
    code documents.
    """
    from CREDA.toy_experiment import sigma_de_la_fuente

    hyper = {**toy_run.declared_hyper(tiny), "attentionGamma": 0.6}
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    source = material["source"]
    multiplier = toy_run.SIGMA_MULTIPLIER_NEUTRAL
    with toy_run.material_bound(familia, tiny, "mlp", hyper):
        source_train, source_valid, _ = bags.roles(source)
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
        assert model.attention_gamma > 0.0, (
            "the consensus term is the only path the bandwidth takes into "
            "Eq. (16); at gamma = 0 this test compares one number with itself")
        _train_phase_one(notebook, model, source_train, source_valid,
                         hyper=hyper)
        reading = toy_run.frozen_source_sigma(model, source, source.train_idx,
                                              DEVICE, multiplier=multiplier)
        rows, _ = latent_module.represent(model, source, source.train_idx,
                                          DEVICE, domain="source",
                                          unit="instance")

    # In band, and the value is the identity the grid is built on.
    low, high = toy_run.SIGMA_BAND
    assert low <= reading["kernelWithinBag"] <= high, reading
    assert reading["kernelWithinBag"] == pytest.approx(
        math.exp(-1.0 / (2.0 * multiplier ** 2)))
    assert reading["inBand"] is True

    # The scope this left, on the same reading: what the pooled bandwidth would
    # have read, and that it is OUT of band on this fixture.
    assert reading["pooledKernelWithinBag"] > high, (
        f"the pooled bandwidth reads {reading['pooledKernelWithinBag']} inside "
        "the band on this fixture, so nothing here distinguishes the two scopes")
    assert reading["pooledInBand"] is False

    # And the mechanism, measured by taking the credit away: the same weights,
    # the same embeddings, the two bandwidths.
    assert reading["weightSpreadRelativeToUniform"] > (
        reading["pooledWeightSpreadRelativeToUniform"]), (
        reading["weightSpreadRelativeToUniform"],
        reading["pooledWeightSpreadRelativeToUniform"])
    assert reading["weightSpreadGainOverPooled"] == pytest.approx(
        reading["weightSpreadRelativeToUniform"]
        / reading["pooledWeightSpreadRelativeToUniform"])
    assert reading["weightSpreadGainOverPooled"] > 1.0

    # The control that the two bandwidths really are different numbers, so the
    # inequality above is about the scope and not about float noise.
    detached = rows.detach()
    assert sigma_de_la_fuente(detached) != reading["sigma"]


# ------------------------------------- the supervised term leaves the phase-2 loss

def test_the_phase_two_objective_drops_eq_21_and_moves_no_step(notebook,
                                                               familia, tiny):
    """Phase 2's loss is `lambda_glob*L_glob + lambda_loc*L_loc`, and Eq. (21)
    is a READING.

    **The parity this buys.** In CREDA the adaptation term is ONE -- the Renyi
    -- and nothing else is summed into phase 2. Here it is two, global and
    local, and that is the whole of the difference; a third summand that CREDA's
    phase 2 does not carry makes the two objectives incomparable at the one
    place the tables are read side by side.

    **And the reason it changes no number: with the source frozen, Eq. (21)
    carries no gradient.** Phase 2 moves `theta_t` alone, and the supervised
    term is `head o theta_s` over the source -- nothing it touches requires
    grad. So dropping it changes what the objective MEANS and not what the run
    DOES, and this test measures exactly that split rather than asserting it:

    - the two VALUES differ, by exactly Eq. (21), which is the pole. Without it
      "the step did not move" would be satisfied by an objective that never
      changed at all.
    - the two GRADIENTS are bit-identical, with the control that they are not
      all zero -- otherwise "identical" would hold on two empty tensors.
    - and the STEP, end to end: one Adam step from the same state on each
      objective leaves `theta_t` bit-identical, parameter by parameter.

    If any of the three moves, the premise that Eq. (21) is gradient-free in
    phase 2 is wrong and the change is not the bookkeeping it is written as.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp", hyper):
        source_train, source_valid, _ = bags.roles(material["source"])
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
        _train_phase_one(notebook, model, source_train, source_valid)
        toy_run.copy_source_to_target(model)
        for name, parameter in model.named_parameters():
            if not name.startswith("target_encoder."):
                parameter.requires_grad_(False)
        toy_run.set_phase_two_modes(model)

        x, y = _batch(notebook, source_train, range(len(source_train)))
        record = model.training_step(
            x, y, toy_run.FIXED_COEFFICIENT,
            torch.Generator().manual_seed(4242),
            ramp_local=toy_run.FIXED_COEFFICIENT)

        full = record["loss"]
        two_term = toy_run.phase_two_objective(record)

        # The pole: the two values differ, and the difference IS Eq. (21).
        assert record["supervised"] > 0.0, (
            "the supervised term is zero on this fixture, so dropping it "
            "changes no value and this test distinguishes nothing")
        full_value = float(full.detach())
        two_value = float(two_term.detach())
        assert two_value != full_value, (two_value, full_value)
        assert full_value - two_value == pytest.approx(record["supervised"],
                                                       rel=1e-6)
        # And what is left is the two declared summands and nothing else.
        assert two_value == pytest.approx(
            record["contributionGlobal"] + record["contributionLocal"],
            rel=1e-6)

        # The gradients, bit-identical, with the control that they can be
        # non-zero at all.
        parameters = [p for p in model.target_encoder.parameters()
                      if p.requires_grad]
        assert parameters, "phase 2 has no trainable parameter to measure"
        full_grads = torch.autograd.grad(full, parameters, retain_graph=True)
        two_grads = torch.autograd.grad(two_term, parameters)
        assert any(float(g.abs().max()) > 0.0 for g in full_grads), (
            "every phase-2 gradient is zero on this fixture, so 'bit-identical' "
            "would hold between two empty tensors")
        for one, other in zip(full_grads, two_grads):
            assert torch.equal(one, other), (
                float((one - other).abs().max()),
                "the supervised term carries gradient into theta_t: the premise "
                "this change rests on is wrong and it is not bookkeeping")

    # The step itself, end to end: the same state, one Adam step on each
    # objective, compared parameter by parameter. This is the check the brief
    # names, and the two above cannot stand in for it -- an optimizer is a
    # function of its gradients AND its state, and only running it says so.
    def stepped(objective_of) -> dict:
        material_i = toy_run.material_for(familia, 0, tiny, "mlp")
        with toy_run.material_bound(familia, tiny, "mlp", hyper):
            train_i, valid_i, _ = bags.roles(material_i["source"])
            arm = toy_run.build_arm(ADAPTED, material_i, tiny, hyper, DEVICE, 0)
            _train_phase_one(notebook, arm, train_i, valid_i)
            toy_run.copy_source_to_target(arm)
            for name, parameter in arm.named_parameters():
                if not name.startswith("target_encoder."):
                    parameter.requires_grad_(False)
            toy_run.set_phase_two_modes(arm)
            optimizer = torch.optim.Adam(arm.target_encoder.parameters(),
                                         lr=tiny.lr_2)
            xi, yi = _batch(notebook, train_i, range(len(train_i)))
            optimizer.zero_grad()
            row = arm.training_step(xi, yi, toy_run.FIXED_COEFFICIENT,
                                    torch.Generator().manual_seed(4242),
                                    ramp_local=toy_run.FIXED_COEFFICIENT)
            objective_of(row).backward()
            optimizer.step()
            return {k: v.detach().clone()
                    for k, v in arm.target_encoder.state_dict().items()}

    with_eq21 = stepped(lambda row: row["loss"])
    without = stepped(toy_run.phase_two_objective)
    assert set(with_eq21) == set(without)
    # The control that one Adam step MOVES anything at all on this fixture: two
    # runs that both went nowhere would be bit-identical for a reason that has
    # nothing to do with Eq. (21).
    untrained = stepped(lambda row: row["loss"] * 0.0)
    assert any(not torch.equal(with_eq21[k], untrained[k]) for k in with_eq21), (
        "a step on this fixture leaves theta_t unchanged even against a zero "
        "objective, so 'bit-identical' below distinguishes nothing")
    for key in with_eq21:
        assert torch.equal(with_eq21[key], without[key]), (
            key, float((with_eq21[key] - without[key]).abs().max()),
            "the optimizer's step moved when Eq. (21) left the loss")


# ------------------------------------------- the material the roles are drawn from

def test_twentyseven_bags_per_class_reach_the_material_the_run_actually_draws(
        tiny):
    """The bag count is read off the DRAWN BagSets, never off the constant.

    Two numbers could disagree and used to be able to: `toy_bags`' own
    `BAGS_PER_CLASS_TOY` is what `construir` defaults to, and
    `Scale.bags_per_class` is what `build_family` actually passes. Asserting the
    constant against 27 would pass while the run drew twelve bags a class, so
    what is asserted here is the shape of `bagset.members` -- one row per bag --
    and the per-class count recovered from it.

    The reachable red is the disagreement itself: the two declarations are
    crossed, so moving one without the other fails here.
    """
    assert toy_bags.BAGS_PER_CLASS_TOY == tiny.bags_per_class, (
        "the module constant and the scale disagree on bags per class; the "
        "figure and the run would be drawing different experiments")
    familia = toy_run.build_family("toroide", 0, tiny)
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    for role in ("source", "target"):
        bagset = material[role]
        assert bagset.members.shape[0] == tiny.classes * tiny.bags_per_class
        assert bagset.members.shape[0] == 81, bagset.members.shape
        per_class = {}
        for position in range(bagset.members.shape[0]):
            per_class.setdefault(int(bagset.labels[position]), 0)
            per_class[int(bagset.labels[position])] += 1
        assert per_class == {0: 27, 1: 27, 2: 27}, (role, per_class)
    # And the bands the figure cuts carry the same count, measured on the draw
    # rather than on the argument handed in.
    for labels, indices in ((familia.ys, familia.bs), (familia.yt, familia.bt)):
        assert sorted(set(indices.tolist())) == list(range(27))
        for class_id in np.unique(labels):
            drawn = sorted(set(indices[labels == class_id].tolist()))
            assert drawn == list(range(27)), (int(class_id), drawn)


def test_every_role_covers_every_class_at_the_new_counts(tiny):
    """Sixteen, five and six bags of every class, read off the built roles.

    The local correspondence is undefined for a class with no source bag, so a
    role that dropped one would produce a failure that reads as a defect of the
    method and is a defect of the split. `bags._split_counts` draws per class for
    that reason, and `material_bound` hands it per-class counts times the class
    count so no remainder can land anywhere.

    Read off `bags.roles` and not off `realized_split`: the split's arithmetic is
    asserted in the scales section, and this is the other end of the same join.
    """
    familia = toy_run.build_family("rollo", 0, tiny)
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    expected = tiny.realized_split()["bagsPerClass"]
    assert expected == {"train": 16, "valid": 5, "test": 6}, expected
    for role_name in ("source", "target"):
        bagset = material[role_name]
        for name, idx in (("train", bagset.train_idx),
                          ("valid", bagset.valid_idx),
                          # The record's role is `test` (CREDA's word, see
                          # `Scale.realized_split`) and the bagset's attribute
                          # is `eval_idx` (the digit benchmark owns it jointly,
                          # so this toyset does not rename it). THIS zip is the
                          # join between the two names, and it is written out
                          # here so the pair is asserted rather than assumed.
                          ("test", bagset.eval_idx)):
            classes = [int(bagset.labels[int(p)]) for p in idx]
            counted = {c: classes.count(c) for c in range(tiny.classes)}
            assert counted == {c: expected[name] for c in range(tiny.classes)}, (
                role_name, name, counted)
        # The three roles are disjoint and together are every bag: a role that
        # covered every class by overlapping another would pass the counts above.
        union = sorted(int(p) for p in torch.cat(
            [bagset.train_idx, bagset.valid_idx, bagset.eval_idx]))
        assert union == list(range(tiny.classes * tiny.bags_per_class)), role_name


# ------------------------------------------- the roles are three samples, not
# ------------------------------------------- three regions

#: How many standard errors of the difference two roles' accuracies may sit
#: apart and still be called one domain sampled twice.
#:
#: Three, so the bound is the ordinary ~99.7% two-sided normal one, and it is
#: declared here rather than inlined because the SAME number has to govern both
#: poles below: a bound chosen per pole could call the fix a pass and the
#: control a failure by moving between them.
SIGMAS_DE_INTERCAMBIABILIDAD = 3.0


def _cota_de_ruido(correctos_a, totales_a, correctos_b, totales_b,
                   sigmas=SIGMAS_DE_INTERCAMBIABILIDAD):
    """The standard error of the difference between two role accuracies.

    The pooled-rate binomial form, which is what "within sampling noise" means
    when the two roles hold different numbers of bags: 48 train bags and 18
    test bags do not carry the same precision, and a single tolerance applied to
    both would be too tight on one side and too loose on the other.
    """
    pooled = (correctos_a + correctos_b) / (totales_a + totales_b)
    return sigmas * math.sqrt(
        pooled * (1 - pooled) * (1 / totales_a + 1 / totales_b))


def _tres_roles_de_la_fuente(notebook, familia, scale, seed=0, backbone="mlp"):
    """One phase 1, its best epoch restored, read on all three SOURCE roles.

    The SAME restored encoder on the three, which is the whole point: a reading
    that re-trained per role would measure three models and could not say
    anything about the three roles.
    """
    material = toy_run.material_for(familia, seed, scale, backbone)
    hyper = toy_run.declared_hyper(scale)
    with toy_run.material_bound(familia, scale, backbone, hyper):
        source = material["source"]
        source_train, source_valid, source_eval = bags.roles(source)
        model = toy_run.build_arm(ADAPTED, material, scale, hyper, DEVICE, seed)
        # The tutorial's phase 1 restores its best epoch before returning.
        _train_phase_one(notebook, model, source_train, source_valid, seed)
        return {
            name: (harness.accuracy(model, role, DEVICE), len(role))
            for name, role in (("train", source_train),
                               ("valid", source_valid),
                               ("test", source_eval))}


def _familia_con_bandas_contiguas(familia, nombre, n):
    """The declared alternative: the band column WITHOUT the role permutation.

    `toy_bags.bolsas` is still the pure band cutter, so this is exactly the
    column every run drew before the permutation existed -- bag `k` is the
    `k`-th band along the parameter, and a role split by bag index is therefore
    a split by position on the manifold.

    It lives in this file and not in `src/` for the same reason
    `test_toy_bags._bandas_equiancho` does: it is the alternative the current
    decision is measured against, and nothing in the implementation may call
    it.
    """
    contigua = copy.deepcopy(familia)
    object.__setattr__(contigua, "bs", toy_bags.bolsas(
        toy_bags.parametro(nombre, familia.Xs), familia.ys, n))
    object.__setattr__(contigua, "bt", toy_bags.bolsas(
        toy_bags.parametro(nombre, familia.Xt_crudo), familia.yt, n))
    return contigua


@pytest.mark.parametrize("nombre", ["lunas", "toroide", "rollo"])
def test_the_three_roles_of_the_source_are_one_domain_sampled_three_times(
        nombre, notebook, tiny):
    """The measurement this whole task exists for, with both poles.

    The three roles are supposed to be three samples of ONE domain, so the same
    restored encoder read on each of them has to come back within sampling
    noise. They were not: the role split assigns by bag index, the bag index IS
    the position along the parameter, so `train` held the first stretch of every
    manifold and `test` the last -- and measured on `toroide` the encoder read
    1.0000 on train and below chance on test, in its own domain.

    Both poles, because one alone cannot distinguish a fix from a measurement
    that would have passed anything:

    - the permuted column puts every role within `_cota_de_ruido` of the
      others;
    - the contiguous column -- the declared alternative, built in this file --
      does NOT, on the same family, the same seed and the same phase 1.

    The control is what makes this a measurement. A bound wide enough to
    forgive a region gap would pass both, and this test asserts that it does
    not.
    """
    familia = toy_run.build_family(nombre, 0, tiny)
    medido = _tres_roles_de_la_fuente(notebook, familia, tiny)

    for primero, segundo in (("train", "test"), ("valid", "test"),
                             ("train", "valid")):
        (a, na), (b, nb) = medido[primero], medido[segundo]
        cota = _cota_de_ruido(round(a * na), na, round(b * nb), nb)
        assert abs(a - b) <= cota, (
            f"{nombre}: {primero}={a:.4f} and {segundo}={b:.4f} differ by "
            f"{abs(a - b):.4f}, past the {cota:.4f} sampling noise allows; the "
            f"two roles are not samples of one domain")

    # The pole: the same reading on the contiguous column fails the same bound.
    contigua = _familia_con_bandas_contiguas(familia, nombre,
                                             tiny.bags_per_class)
    assert not np.array_equal(contigua.bs, familia.bs), (
        f"{nombre}: the control's band column equals the real one, so the "
        "permutation never landed and this test compares one thing with itself")
    alternativa = _tres_roles_de_la_fuente(notebook, contigua, tiny)
    (a, na), (b, nb) = alternativa["train"], alternativa["test"]
    cota = _cota_de_ruido(round(a * na), na, round(b * nb), nb)
    assert abs(a - b) > cota, (
        f"{nombre}: the CONTIGUOUS split put train={a:.4f} and test={b:.4f} "
        f"within {cota:.4f} of each other, so this bound does not distinguish "
        "a split by geometry from a split by chance and the pole above proves "
        "nothing")


def _posiciones_en_el_parametro(fam, nombre, scale, backbone="mlp"):
    """Each role's bands, as RANKS ALONG THE PARAMETER, read off the manifest.

    **The rank is measured and never taken from the bag index, and that
    distinction is the whole reading.** A first version of this helper derived
    the position with `divmod(bag_position, bags_per_class)`, which is the bag
    INDEX -- the very number the permutation renumbers. It reported every role
    as the unbroken run `21..26` after the fix had landed and was green before
    it: a wrong claim of this file's, not a defect in the code, and the kind
    that reads as evidence. The band's position on the manifold has to come
    from the manifold.

    So: each bag's points are looked up in the family's own cloud, their
    parameter is read with `toy_bags.parametro`, and the bags of a class are
    ranked by it. Rank 0 is the band lowest on the parameter. The target is read
    on `Xt_crudo`, because that is the draw its bands were cut off.
    """
    material = toy_run.material_for(fam, 0, scale, backbone)
    n = scale.bags_per_class
    leido = {}
    for dominio, X, y in (("source", fam.Xs, fam.ys),
                          ("target", fam.Xt_crudo, fam.yt)):
        p = toy_bags.parametro(nombre, X)
        bagset = material[dominio]
        flat = bagset.manifest["imageIndices"]
        # The parameter each bag sits at, and the rank of that value within the
        # bag's own class: `np.unique` is not used, the bags of a class are
        # sorted among themselves so the rank is a position out of `n`.
        centro = {}
        for posicion in range(bagset.members.shape[0]):
            puntos = [int(flat[int(i)]) for i in bagset.members[posicion]]
            clase = int(bagset.labels[posicion])
            centro[posicion] = (clase, float(np.mean(p[puntos])))
        for clase in range(scale.classes):
            de_la_clase = sorted((v, k) for k, (c, v) in centro.items()
                                 if c == clase)
            rango = {posicion: r for r, (_, posicion) in enumerate(de_la_clase)}
            assert len(rango) == n, (dominio, clase, len(rango))
            for rol, idx in (("train", bagset.train_idx),
                             ("valid", bagset.valid_idx),
                             ("test", bagset.eval_idx)):
                tomado = sorted(rango[int(pp)] for pp in (int(q) for q in idx)
                                if int(pp) in rango)
                if tomado:
                    leido[(dominio, rol, clase)] = tomado
    return leido


@pytest.mark.parametrize("nombre", ["lunas", "toroide", "rollo"])
def test_every_role_receives_bands_from_across_the_parameter_range(nombre,
                                                                   tiny):
    """Which stretch of the manifold each role holds, measured not assumed.

    The permutation's whole purpose: a role must no longer be one contiguous
    stretch of the manifold. Measured before it existed, on 27 bands per class,
    `train` held the lowest 16 positions of every class, `valid` the next 5 and
    `test` the top 6 -- in both domains, in all three families.

    Read through `_posiciones_en_el_parametro`, which goes to the BUILT manifest
    and back to the cloud, so this crosses the same join
    `test_the_bag_indices_that_reach_the_bagset_are_the_ones_bolsas_computed`
    crosses rather than trusting a label.

    Both poles: no role is an unbroken run of consecutive positions, and the
    contiguous column -- the declared alternative -- gives exactly such a run
    for every role and class.
    """
    familia = toy_run.build_family(nombre, 0, tiny)
    n = tiny.bags_per_class

    def es_una_corrida(posiciones):
        return posiciones == list(range(posiciones[0], posiciones[-1] + 1))

    medido = _posiciones_en_el_parametro(familia, nombre, tiny)
    assert len(medido) == 2 * 3 * tiny.classes, sorted(medido)
    for clave, posiciones in sorted(medido.items()):
        assert not es_una_corrida(posiciones), (
            f"{nombre} {clave}: holds the unbroken run "
            f"{posiciones[0]}..{posiciones[-1]}, so this role is still one "
            "contiguous stretch of the manifold")
        # And it is spread, not merely broken: a role of `m` bands drawn from
        # `n` positions has to reach much further than the `m - 1` a contiguous
        # run spans.
        assert posiciones[-1] - posiciones[0] > len(posiciones), (
            f"{nombre} {clave}: {posiciones} spans "
            f"{posiciones[-1] - posiciones[0]} of {n - 1} positions, no wider "
            "than the contiguous run it replaced")

    alternativa = _posiciones_en_el_parametro(
        _familia_con_bandas_contiguas(familia, nombre, n), nombre, tiny)
    for clave, posiciones in sorted(alternativa.items()):
        assert es_una_corrida(posiciones), (
            f"{nombre} {clave}: the contiguous control gave {posiciones}, "
            "which is not a run, so the assertion above distinguishes nothing")


def test_the_declared_instances_per_bag_is_what_the_drawn_bags_hold(tiny):
    """Thirty at both declared scales, and what the fixture lowered it to here.

    The number is DECLARED (`Scale.instances_per_bag`) and the bags are measured:
    `bagset.members` is one row of instance indices per bag, so its width is what
    each bag actually contributed. It used to be `points_per_domain //
    bags_total` in the stamp and the measured smallest bag in the binding, and
    those two disagreed -- 150 against 138 at the full scale.

    The declared bag is 30 at `PILOT` and at `FULL`, which is the digit
    benchmark's own `config.INSTANCES_PER_BAG`, so an epoch processes the same
    material at both and the pilot is a reduction in repetitions and lengths
    alone. Asserted against `config.INSTANCES_PER_BAG` rather than against a 30
    typed here, so a digit-side change shows up as a disagreement.
    """
    assert toy_run.PILOT.instances_per_bag == config.INSTANCES_PER_BAG
    assert toy_run.FULL.instances_per_bag == toy_run.PILOT.instances_per_bag

    familia = toy_run.build_family("toroide", 0, tiny)
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    for role in ("source", "target"):
        members = material[role].members
        assert members.shape[1] == tiny.instances_per_bag, members.shape
        assert material[role].manifest["instancesPerBag"] == tiny.instances_per_bag
    # The control that this is a CAP and not the whole bag: the draw holds more
    # than the declaration takes, so equality above is the declaration being
    # honoured and not the material running out.
    smallest = min(
        int(((labels == c) & (indices == k)).sum())
        for labels, indices in ((familia.ys, familia.bs), (familia.yt, familia.bt))
        for c in np.unique(labels) for k in range(tiny.bags_per_class))
    assert smallest > tiny.instances_per_bag, (smallest,
                                               tiny.instances_per_bag)


def test_a_draw_that_cannot_fund_the_declared_bag_is_refused_before_it_runs(
        tiny):
    """The guard, and the reachable red for it.

    `bags._select` takes exactly `config.INSTANCES_PER_BAG` from every subject
    and refuses a subject that cannot fund it, mid-draw and naming the subject.
    `_instances_per_bag` asks the same question early, in the scale's own terms.
    Both poles: the fixture's own scale passes, and a scale whose material is
    too thin for its declared bag refuses rather than running something smaller.
    """
    familia = toy_run.build_family("toroide", 0, tiny)
    assert toy_run._instances_per_bag(familia, tiny) == tiny.instances_per_bag

    starved = dataclasses.replace(tiny, instances_per_bag=tiny.instances_per_bag
                                  + 1000)
    with pytest.raises(ValueError, match="declares"):
        toy_run._instances_per_bag(familia, starved)
    # And the refusal really is about the funding and not about the family: the
    # same family at the same draw funds the real scale, asserted above.
    with pytest.raises(ValueError, match="instances per bag"):
        toy_run.material_for(familia, 0, starved, "mlp")


# --------------------------------------------------------------------- the bridge

def test_the_bag_indices_that_reach_the_bagset_are_the_ones_bolsas_computed(
        familia, tiny):
    """The join, crossed by asking the tool what it sees.

    `toy_bags.bolsas` writes a bag index per point, `ToyDataset.subjects` carries
    it, `bags._select` draws subject by subject and `bags.build` assigns subject
    `k` to bag `k` of its class. Nothing short of reading the BUILT manifest
    proves that chain: testing each half against a fixture written here would
    verify both halves and never the connection.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    for role, labels, indices in (("source", familia.ys, familia.bs),
                                  ("target", familia.yt, familia.bt)):
        bagset = material[role]
        flat = bagset.manifest["imageIndices"]
        assert len(flat) == len(set(flat)), "a point was drawn into two bags"
        positions = bagset.members.shape[0]
        assert positions == tiny.classes * tiny.bags_per_class
        for position in range(positions):
            points = [flat[int(i)] for i in bagset.members[position]]
            expected_class, expected_bag = divmod(position, tiny.bags_per_class)
            assert {int(labels[i]) for i in points} == {expected_class}, (
                role, position)
            assert {int(indices[i]) for i in points} == {expected_bag}, (
                role, position)
            assert int(bagset.labels[position]) == expected_class


def test_a_shuffled_subject_column_moves_which_points_land_in_a_bag(familia,
                                                                    tiny):
    """The reachable red for the join above: if `.subjects` were ignored,
    permuting it would change nothing downstream."""
    shuffled = copy.deepcopy(familia)
    rng = np.random.default_rng(7)
    rolled = familia.bs.copy()
    for class_id in np.unique(familia.ys):
        mask = familia.ys == class_id
        rolled[mask] = rng.permutation(rolled[mask])
    assert not np.array_equal(rolled, familia.bs), "the permutation did not land"
    object.__setattr__(shuffled, "bs", rolled)

    original = toy_run.material_for(familia, 0, tiny, "mlp")["source"]
    moved = toy_run.material_for(shuffled, 0, tiny, "mlp")["source"]
    assert original.manifest["imageIndices"] != moved.manifest["imageIndices"], (
        "permuting the subject column changed nothing; `.subjects` is not read")


def test_material_bound_restores_every_global_it_bends(familia, tiny):
    before = {name: getattr(config, name) for name in toy_run.BENT_GLOBALS}
    codes_before = set(config.DOMAINS)
    with toy_run.material_bound(familia, tiny, "mlp"):
        inside = {name: getattr(config, name) for name in toy_run.BENT_GLOBALS}
        assert inside["CLASSES"] == tiny.classes
        assert inside["BAGS_PER_CLASS"] == tiny.bags_per_class
        assert inside != before, "nothing was bent; the test proves nothing"
        assert set(config.DOMAINS) > codes_before
    assert {name: getattr(config, name) for name in toy_run.BENT_GLOBALS} == before
    assert set(config.DOMAINS) == codes_before


def test_material_bound_refuses_a_second_overlapping_binding(familia, tiny):
    """A module global is process-wide, and two overlapping runs would be each
    other's draw. It cannot be prevented here; it is at least caught."""
    with toy_run.material_bound(familia, tiny, "mlp"):
        with pytest.raises(RuntimeError, match="not at their declared values"):
            with toy_run.material_bound(familia, tiny, "mlp"):
                pass


def test_a_code_the_binding_does_not_name_is_refused_rather_than_downloaded(
        familia, tiny):
    with toy_run.material_bound(familia, tiny, "mlp"):
        with pytest.raises(ValueError, match="not bound by this run"):
            bags._raw_dataset("M", pathlib.Path("/nonexistent-toy-root"))


# ---------------------------------------------------------------------- the sigma

def test_the_scales_are_measured_through_the_declared_domain_seam(familia, tiny):
    """`embedding_scales` goes through `latent.represent`, which refuses without
    a declared domain, and its reshape of the flattened instance rows is checked
    against `represent`'s own bag-unit reading of the same positions.

    The check has to be able to fail: the gap it reports is a real comparison of
    two computations, so a reshape over the wrong grouping would not agree.

    **And the bag cap is asserted as biting rather than assumed not to.** This
    used to read `bagsRead == len(train_idx)`, which held because the training
    role had 6 bags and the cap is 16. At twelve bags per class it had 21 and
    the cap truncated; at the declared 27 it has 48, so the median is taken
    over a third of the role. That is a fact about the measurement, so it is
    read here instead of being discovered as a failed equality, and the
    assertion names the count rather than the ratio so a cut that moved the
    role without moving the cap is red here.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    offered = len(material["source"].train_idx)
    with toy_run.material_bound(familia, tiny, "mlp"):
        probe = toy_run.build_arm(ADAPTED, material, tiny,
                                  toy_run.declared_hyper(tiny), DEVICE, 0)
        scales = toy_run.embedding_scales(probe, material["source"],
                                         material["source"].train_idx, DEVICE)
    assert scales["groupingCheckedAgainstBagUnit"] <= 1e-5
    assert scales["instancesPerBag"] == tiny.instances_per_bag
    assert scales["bagsOffered"] == offered == 48
    assert scales["bagsRead"] == min(scales["bagLimit"], offered)
    assert scales["bagsTruncated"] is True, (
        "the cap no longer bites; `bagsRead` would then be the whole role and "
        "this assertion would stop saying anything")
    # The other pole: handed fewer bags than the cap, nothing is truncated and
    # the whole role is read.
    with toy_run.material_bound(familia, tiny, "mlp"):
        few = toy_run.embedding_scales(
            probe, material["source"], material["source"].train_idx[:4], DEVICE)
    assert few["bagsOffered"] == few["bagsRead"] == 4
    assert few["bagsTruncated"] is False


#: ---------------------------------------------------------------------------
#: A measurement must not train the model it measures
#: ---------------------------------------------------------------------------
#:
#: Every forward pass this module takes to MEASURE rather than to train, named
#: as a ROSTER so the claim below is about the class and not about the one
#: instance that was found first. A retired runner called `embedding_scales`
#: -- a measurement -- before phase 1, and that forward moved BatchNorm's
#: running statistics, so the training that followed started from a different
#: model than the one the seeds describe. Every function `toy_run` decorates
#: with `measuring` is here, and the roster test below reads the decorator off
#: the source so a new measurement cannot stay out of the sweep.
#:
#: Keyed by the attribute name so a member whose name changes reddens here
#: rather than silently dropping out of the sweep.
MEASUREMENTS = ("embedding_scales", "frozen_source_sigma", "lectura_de_bolsas",
                "caminata_de_w")


def _measurement_calls(material, tiny):
    return {
        "embedding_scales": lambda m: toy_run.embedding_scales(
            m, material["source"], material["source"].train_idx, DEVICE),
        "frozen_source_sigma": lambda m: toy_run.frozen_source_sigma(
            m, material["source"], material["source"].train_idx, DEVICE),
        "lectura_de_bolsas": lambda m: toy_run.lectura_de_bolsas(
            m, material["target"], DEVICE, "target"),
        "caminata_de_w": lambda m: toy_run.caminata_de_w(
            m, material["source"], material["target"], DEVICE, barridos=3),
    }


def _float_buffers(model):
    return {name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if buffer.is_floating_point()}


def _modes(model):
    return {name: module.training for name, module in model.named_modules()}


def test_the_measurement_roster_names_every_measurement_this_module_takes(
        familia, tiny):
    """The roster above is the set the sweep below parametrizes over, so both
    halves of it have to agree and every name has to resolve.

    `pytest.mark.parametrize` reads the tuple at collection time, so a name
    that stopped resolving, or a call the table dropped, would shrink the class
    the sweep covers while every remaining case stayed green.
    """
    missing = [name for name in MEASUREMENTS if not hasattr(toy_run, name)]
    assert missing == [], missing
    tree = ast.parse(pathlib.Path(toy_run.__file__).read_text(encoding="utf-8"))
    decorated = {node.name for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and any(isinstance(d, ast.Name) and d.id == "measuring"
                         for d in node.decorator_list)}
    assert decorated == set(MEASUREMENTS), sorted(decorated)
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    assert sorted(_measurement_calls(material, tiny)) == sorted(MEASUREMENTS)


@pytest.mark.parametrize("measurement", MEASUREMENTS)
def test_a_measurement_leaves_every_buffer_of_the_model_bit_identical(
        measurement, notebook, familia, tiny):
    """A forward pass taken to measure leaves the model bit-identical, and
    leaves it in the mode the caller held it in.

    **Asserted on the BUFFERS and never on the loss.** The loss history was
    identical THROUGH this defect -- recorded and recomputed phase 1 agreed on
    every epoch's training loss, validation loss and validation accuracy, and
    differed only in which epoch won -- which is exactly why nothing caught it.
    A test that read a number out of the measurement would have been green on
    the broken code.

    Three poles, because the equality alone would pass on a model with no
    buffers and on a fixture where nothing could have moved them:

    - the backbone HAS floating-point buffers to move (`mlp` carries the
      running statistics; the two controls carry none, which is the next test);
    - a TRAINING forward in the same mode on the same fixture DOES move them,
      so the instrument can register movement at all;
    - and the mode is restored per module, because phase 2 runs the source in
      `eval` with `theta_t` in `train`.
    """
    preparado = _prepared(notebook, familia, tiny)
    model, material = preparado["model"], preparado["material"]
    call = _measurement_calls(material, tiny)[measurement]

    with toy_run.material_bound(familia, tiny, "mlp"):
        # The caller held it in `train()`: that is the condition the fix has to
        # hold under -- a freshly built `nn.Module` is in `train` mode, and the
        # tutorial hands `embedding_scales` one.
        model.train()
        before = _float_buffers(model)
        assert before, (
            "this backbone carries no floating-point buffer, so the equality "
            "below would hold over an empty set and say nothing")

        # The control: a training forward in this very mode moves them.
        control = copy.deepcopy(model)
        control.train()
        moving = _float_buffers(control)
        x, y = _batch(notebook, preparado["source_train"],
                      range(len(preparado["source_train"])))
        toy_run.supervised_loss(control, x, y)
        after_control = _float_buffers(control)
        assert any(not torch.equal(moving[name], after_control[name])
                   for name in moving), (
            "a training forward moved no buffer on this fixture, so the "
            "equality below cannot distinguish a harmless measurement from a "
            "measurement nothing could have registered")

        modes_before = _modes(model)
        call(model)
        after = _float_buffers(model)

    moved = {name: float((after[name] - before[name]).abs().max())
             for name in before
             if not torch.equal(before[name], after[name])}
    assert moved == {}, (measurement, moved)
    assert _modes(model) == modes_before, (
        measurement,
        sorted(name for name in modes_before
               if _modes(model)[name] != modes_before[name]))


def test_making_the_measurement_harmless_did_not_restate_what_it_measures(
        familia, tiny):
    """The measurement is taken in the mode the CALLER held, not forced into
    `eval`.

    The other half of the fix, and the half a copy buys that
    `eval`-and-restore does not. Both routes leave the model bit-identical;
    only the copy leaves the READING where it was. Measured on `toroide`/`mlp`
    at `embedding_scales`' own call site, a freshly built encoder in `train`
    mode: in `eval` the three distances read 0.223x, 0.460x and 0.303x of the
    `train`-mode values, because a fresh BatchNorm's running statistics are
    still `(0, 1)` while `train` mode normalizes by the batch. The tutorial's
    phase-1 bandwidth is that `withinBag` reading, so forcing `eval` would move
    it -- a decision about the experiment, and not this fix's to take.

    So the claim is that the two modes still give DIFFERENT readings, which is
    what says the caller's mode was respected. Its control is the backbone that
    has no mode-dependent layer, where the two must agree -- without it, a
    difference could be noise rather than the normalization.
    """
    readings = {}
    for backbone in ("mlp", "mlp-nobn"):
        material = toy_run.material_for(familia, 0, tiny, backbone)
        with toy_run.material_bound(familia, tiny, backbone):
            model = toy_run.build_arm(ADAPTED, material, tiny,
                                      toy_run.declared_hyper(tiny), DEVICE, 0)
            both = {}
            for mode in ("train", "eval"):
                getattr(model, mode)()
                both[mode] = toy_run.embedding_scales(
                    model, material["source"], material["source"].train_idx,
                    DEVICE)["withinBag"]
        readings[backbone] = both

    assert readings["mlp"]["train"] != readings["mlp"]["eval"], (
        readings["mlp"],
        "`embedding_scales` returns the same number in both modes under a "
        "BatchNorm backbone, so it is no longer reading in the mode the "
        "caller holds -- which silently restates the tutorial's phase-1 "
        "bandwidth")
    assert readings["mlp-nobn"]["train"] == readings["mlp-nobn"]["eval"], (
        readings["mlp-nobn"],
        "the two modes disagree on a backbone with no mode-dependent layer, "
        "so the difference above is not the normalization")


def test_no_module_this_work_added_calls_instance_embeddings_directly():
    """The structural guard `tests/test_domain_mixup_guard.py` derives, asserted
    here too for the one module this work added -- so a later edit that reaches
    for the convenient call reddens in the file it was made in."""
    import ast
    from pathlib import Path as _Path

    tree = ast.parse(_Path(toy_run.__file__).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "instance_embeddings"]
    assert calls == [], [node.lineno for node in calls]
    # The control: the module DOES call the declared seam, so the assertion
    # above is about the route taken and not about the module doing nothing.
    seam = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "represent"]
    assert seam, "the module reaches no embedding seam at all"


# ------------------------------------------------- the one arm, and no ladder

def test_the_toyset_declares_exactly_one_arm_and_no_pair_to_subtract():
    """One arm, and the names that made a pair are gone rather than retargeted.

    The floor arm `TB` was retired by the operator, so there is no rung, no
    floor and no margin. Red reachable in both directions: declaring a second
    arm fails the count, and reintroducing any of the retired names fails the
    absence -- which is the half that matters, because a floor kept "for
    reference" is a number somebody quotes.
    """
    assert len(toy_run.TOY_ARMS) == 1, [arm["id"] for arm in toy_run.TOY_ARMS]
    assert len(toy_run.TOY_ARM_IDS) == 1, toy_run.TOY_ARM_IDS
    assert toy_run.TOY_ARM_IDS == (toy_run.TOY_ADAPTED,)
    # The one declared arm carries the method; an arm declaring no adaptation
    # term would BE a floor whatever it was called.
    only = toy_run.TOY_ARMS_BY_ID[toy_run.TOY_ADAPTED]
    assert only["adaptation"] == "milcreda"
    assert only["weighting"] is True and only["local"] is True

    retired = ("TOY_FLOOR", "TOY_FLOOR_OF", "TOY_LADDER", "LADDER_ARMS",
               "ladder_rungs", "floor_comparison", "VERDICT_REPETITION_FLOOR")
    present = [name for name in retired if hasattr(toy_run, name)]
    assert not present, present
    # The control that `hasattr` discriminates at all on this module: a name
    # that IS there answers true, so the empty list above is about these seven.
    assert hasattr(toy_run, "TOY_ARMS_BY_ID")


def test_nothing_in_the_module_computes_a_floor_margin_or_a_rung():
    """The static half: no surviving code path produces a difference of arms.

    `test_...declares_exactly_one_arm...` reads the declaration; this reads the
    SOURCE, because a function could be reintroduced under another name and the
    declaration would stay a single arm. Two things are asserted: `verdict` --
    the module whose whole job is to decide between two dispersions -- is not
    imported here at all, and no top-level name in this module mentions a floor
    or a rung.

    The control is that the same walk finds the imports that ARE there, so an
    empty intersection is about these names and not about a walk seeing nothing.
    """
    import ast
    from pathlib import Path as _Path

    tree = ast.parse(_Path(toy_run.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "harness" in imported and "wiring" in imported, sorted(imported)
    assert "verdict" not in imported, (
        "`verdict` is imported again; its only callers here were the rung and "
        "the floor comparison, and both are retired")

    defined = set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
            defined.add(statement.name)
        elif isinstance(statement, ast.Assign):
            defined.update(t.id for t in statement.targets
                           if isinstance(t, ast.Name))
    assert "build_arm" in defined, sorted(defined)
    offenders = sorted(name for name in defined
                       if "FLOOR" in name or "floor" in name
                       or "LADDER" in name or "rung" in name)
    assert not offenders, offenders


# ------------------------------------------------------- the latent positions

def _perturbar_theta_t(model, paso: float = 0.05) -> float:
    """Move `theta_t` and return how far it actually moved.

    The return value is the point: a test that perturbs a model and then
    believes the result without checking the perturbation landed is reporting
    success while measuring nothing.
    """
    firma = _parameter_vector(model.target_encoder).clone()
    with torch.no_grad():
        for parametro in model.target_encoder.parameters():
            parametro.add_(torch.full_like(parametro, paso))
    return float((_parameter_vector(model.target_encoder) - firma).norm())


def test_the_positions_are_read_off_the_model_and_move_when_the_model_does(
        familia, tiny):
    """A position is a READING of the encoder as it stands, not a snapshot taken
    once and carried.

    Both poles, in one test: the same model read twice gives the same positions
    -- which is what `eval` and `no_grad` are for, since a BatchNorm encoder in
    `train` mode would make a bag's position depend on which bags shared its
    batch -- and the model mutated between the two readings gives different
    ones. Without the first half a function that returned noise would pass; without
    the second, one that returned a constant would.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp"):
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
        toy_run.copy_source_to_target(model)
        primera = toy_run.bag_positions(model, material["target"], DEVICE, "target")
        segunda = toy_run.bag_positions(model, material["target"], DEVICE, "target")
        movido = _perturbar_theta_t(model)
        tercera = toy_run.bag_positions(model, material["target"], DEVICE, "target")

    assert np.array_equal(primera["positions"], segunda["positions"]), (
        "dos lecturas del MISMO modelo dieron posiciones distintas: la lectura "
        "depende de algo que no es el modelo --- un modo de normalización, el "
        "tamaño del trozo --- y entonces la figura dibuja un artefacto del "
        "recorrido")
    assert movido > 0, (
        "la perturbación de theta_t no llegó a los parámetros: lo que sigue no "
        "probaría nada")
    assert not np.allclose(primera["positions"], tercera["positions"]), (
        f"theta_t se movió {movido:.4f} y las posiciones salieron iguales: no "
        "se están leyendo del modelo")


def test_the_domain_channel_is_not_crossed(familia, tiny):
    """Each view comes from ITS bagset through ITS encoder, and the three
    crossings that would look identical in the tutorial's figures do not
    reproduce it.

    Four pairings of (material, encoder) exist and the tutorial reads one per
    view. A crossed pairing writes an array of exactly the right shape and
    dtype, under the right member name, so nothing downstream could tell: the
    figure would draw the source cloud twice and call one of them the target.
    So the three wrong pairings are computed here and required to DIFFER.

    `theta_t` is perturbed first and the perturbation is proven to have landed,
    because right after `copy_source_to_target` the two encoders are
    bit-identical by construction -- and on that state the encoder crossing is
    undetectable, which would make this test green while measuring nothing.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp"):
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
        toy_run.copy_source_to_target(model)
        movido = _perturbar_theta_t(model)
        declarado = {
            "source": toy_run.bag_positions(model, material["source"], DEVICE,
                                            "source"),
            "target": toy_run.bag_positions(model, material["target"], DEVICE,
                                            "target"),
        }
        cruzado_encoder = toy_run.bag_positions(model, material["source"],
                                                DEVICE, "target")
        cruzado_material = toy_run.bag_positions(model, material["target"],
                                                 DEVICE, "source")
        with pytest.raises(ValueError, match="unknown domain"):
            toy_run.bag_positions(model, material["source"], DEVICE, "fuente")

    assert movido > 0, "theta_t no se movió: el cruce de codificador no se vería"
    assert not np.allclose(declarado["source"]["positions"],
                           cruzado_encoder["positions"]), (
        "la fuente leída por theta_s y por theta_t da lo mismo: el canal de "
        "dominio no decide nada")
    assert not np.allclose(declarado["target"]["positions"],
                           cruzado_material["positions"]), (
        "el material del blanco por theta_t y por theta_s da lo mismo")
    assert not np.allclose(declarado["source"]["positions"],
                           declarado["target"]["positions"]), (
        "las dos vistas declaradas son la misma nube: la figura dibujaría un "
        "dominio dos veces")


def test_every_normalization_creda_draws_has_a_backbone_here():
    """CREDA's MLP declares three normalizations and this benchmark declares
    three backbones, one per normalization, in CREDA's order.

    The order is CREDA's `NORMAS`, the variants its own `MLP` builds; the
    second half drops one row and watches the difference appear, so the
    equality is not true of a comparison that looks at nothing.
    """
    from CREDA import toy_experiment as creda
    from MIL_CREDA_Benchmark import mlp

    assert set(toy_run.NORMALIZATIONS.values()) == set(mlp.MLP_BACKBONES)
    assert list(toy_run.NORMALIZATIONS) == list(creda.NORMAS), (
        f"las filas de este paquete son {list(toy_run.NORMALIZATIONS)} y el MLP "
        f"de CREDA declara {list(creda.NORMAS)}")

    recortadas = {k: v for k, v in toy_run.NORMALIZATIONS.items()
                  if k != list(toy_run.NORMALIZATIONS)[-1]}
    assert set(creda.NORMAS) - set(recortadas) != set()


# ------------------------------------------- the toyset's own arm declaration

def _load_module_from(path, name):
    """A module loaded out of a file, NOW, and deliberately not registered.

    `importlib.reload` is not what any of this wants: it mutates the installed
    module in place, and every other test in this file holds references into
    `toy_run` and monkeypatches names on it. This builds a separate module
    object from the same source, so whatever that source computes AT IMPORT is
    computed against `config` as it stands at this instant -- which is the only
    way to catch a declaration copied through at import time, since a module
    constant assigned once never moves again no matter what `config` does
    afterwards.

    It is registered in `sys.modules` under its own throwaway name for the
    duration of the load and removed afterwards, because `@dataclass` resolves
    a field's annotation through `sys.modules[cls.__module__]` and raises on a
    class whose module is not there -- and left registered it would shadow
    nothing but would still be a second `toy_run` for anything importing later.
    """
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    _sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        _sys.modules.pop(name, None)
    return module


#: A declaration that is NOT this toyset's, for the digit benchmark's names to
#: be moved to. Its ids share no character with `TGL`, so an assertion cannot
#: pass by coincidence of naming.
_FOREIGN_ARMS = [
    {"id": "Z1", "name": "foreign-floor", "label": "not this experiment",
     "unit": "bag", "adaptation": None, "weighting": False, "local": False,
     "attention": "learned", "selection": None, "normalization": "shared"},
    {"id": "Z2", "name": "foreign-full", "label": "not this experiment",
     "unit": "bag", "adaptation": "milcreda", "weighting": True, "local": True,
     "attention": "learned", "selection": None, "normalization": "shared"},
]


def test_the_toyset_arms_do_not_move_when_the_digit_benchmark_is_mutated(
        monkeypatch, tmp_path):
    """`config.ARMS` is moved, and this toyset's arm declaration stays put.

    The mutation is applied to `ARMS` and to every name derived from it
    (`ARMS_BY_ID`, `ARM_ORDER`, `NAME_OF`) plus `LADDER` and `FLOOR_OF`,
    because those are derived once at `config` import and mutating `ARMS` alone
    would leave the others holding the real four -- a test that moved only
    `ARMS` would pass against code deriving from `ARM_ORDER`, which is exactly
    what this module used to do.

    **Why a second module instance, and why the control below is the whole
    test.** `LADDER_ARMS = tuple(config.ARM_ORDER)` -- the line this replaced,
    whose name is now `TOY_ARM_IDS` -- runs at import, so mutating `config`
    afterwards never moves it either. A test that mutated `config` and read the
    installed `toy_run` would be green against both the defect and the fix: it
    would be measuring nothing. So the source is loaded again inside the mutated
    window, and the CONTROL is that exact old line loaded in the same window: it
    moves to `("Z1", "Z2")`. That proves the window is live for any importer,
    which is what makes the freshly-loaded `toy_run` staying put a fact about
    `toy_run`.

    `TOY_LADDER` and `TOY_FLOOR_OF` were read here too and are retired with the
    floor arm; what is left to pin is the arm list, the specs and the one name
    derived from them. The retired names' absence is asserted in
    `test_the_toyset_declares_exactly_one_arm_and_no_pair_to_subtract`, and
    reading them here would have made this test red for the wrong reason.
    """
    installed_arms = tuple(toy_run.TOY_ARM_IDS)
    installed_specs = [dict(arm) for arm in toy_run.TOY_ARMS]
    real_order = tuple(config.ARM_ORDER)
    assert installed_arms and not (set(installed_arms) & {"Z1", "Z2"})

    monkeypatch.setattr(config, "ARMS", _FOREIGN_ARMS)
    monkeypatch.setattr(config, "ARMS_BY_ID",
                        {arm["id"]: arm for arm in _FOREIGN_ARMS})
    monkeypatch.setattr(config, "ARM_ORDER",
                        [arm["id"] for arm in _FOREIGN_ARMS])
    monkeypatch.setattr(config, "NAME_OF",
                        {arm["id"]: arm["name"] for arm in _FOREIGN_ARMS})
    monkeypatch.setattr(config, "LADDER",
                        [("Z1", "Z2", "not this experiment")])
    monkeypatch.setattr(config, "FLOOR_OF", {"Z2": "Z1"})

    # The mutation landed, before anything is believed about it.
    assert tuple(config.ARM_ORDER) != real_order
    assert {arm["id"] for arm in config.ARMS} == {"Z1", "Z2"}
    assert [rung[:2] for rung in config.LADDER] == [("Z1", "Z2")]

    # THE CONTROL: the derivation this module used to carry, loaded now.
    borrowed = tmp_path / "borrowed_declaration.py"
    borrowed.write_text("from MIL_CREDA_Benchmark import config\n"
                        "LADDER_ARMS = tuple(config.ARM_ORDER)\n",
                        encoding="utf-8")
    old = _load_module_from(borrowed, "borrowed_declaration")
    assert tuple(old.LADDER_ARMS) == ("Z1", "Z2"), (
        "the old derivation did not pick the mutation up; this window is not "
        "live and nothing below would be measuring anything")
    assert tuple(old.LADDER_ARMS) != installed_arms

    # And this module, loaded in that same live window.
    fresh = _load_module_from(toy_run.__file__, "toy_run_second_instance")
    assert tuple(fresh.TOY_ARM_IDS) == installed_arms, fresh.TOY_ARM_IDS
    assert [dict(arm) for arm in fresh.TOY_ARMS] == installed_specs
    assert fresh.TOY_ADAPTED == toy_run.TOY_ADAPTED
    # The retired names do not come back from a fresh import either.
    for name in ("TOY_FLOOR", "TOY_FLOOR_OF", "TOY_LADDER", "LADDER_ARMS"):
        assert not hasattr(fresh, name), name


def test_no_module_level_statement_reads_the_digit_benchmarks_arm_declaration():
    """The static half: nothing at this module's top level reads an arm
    declaration off `config`.

    Module level is where it matters -- a constant assigned at import is what
    the behavioural test above cannot see a second time, and is what the defect
    was. Inside a function, `config.ARMS_BY_ID` is legitimate: `build_arm`'s
    refusal names the digit arms to say they are not reachable.

    `ARM_ORDER`, `NAME_OF`, `LADDER`, `FLOOR_OF` and `CHECKPOINTS` are refused
    ANYWHERE in the file: no honest use for them exists on this path, and each
    one is a complete derivation of the ladder on its own.

    **What this cannot see, stated rather than implied:** a dynamic read -- the
    module does `getattr(config, name)` over `BENT_GLOBALS` -- so literal
    `getattr(config, "...")` calls are collected too, and a name computed at
    runtime would still slip past. That hole is covered behaviourally by
    `test_the_toyset_ladder_does_not_move_when_the_digit_benchmark_is_mutated`,
    which cares about the result and not the route.
    """
    import ast
    from pathlib import Path as _Path

    tree = ast.parse(_Path(toy_run.__file__).read_text(encoding="utf-8"))

    def config_reads(nodes):
        found = set()
        for root in nodes:
            for node in ast.walk(root):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "config"):
                    found.add(node.attr)
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "config"
                        and isinstance(node.args[1], ast.Constant)):
                    found.add(node.args[1].value)
        return found

    functions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    top_level = [s for s in tree.body if not isinstance(s, functions)]
    at_module_level = config_reads(top_level)
    everywhere = config_reads(tree.body)

    # The control: the walk finds real config reads at module level, so an
    # empty intersection below is about these names and not about a walk that
    # sees nothing.
    assert at_module_level, "no config read was found at all; the walk is blind"
    assert "KERNEL_SIGMA" in everywhere, sorted(everywhere)

    declaration = {"ARMS", "ARMS_BY_ID", "ARM_ORDER", "NAME_OF", "LADDER",
                   "FLOOR_OF", "CHECKPOINTS"}
    assert not (at_module_level & declaration), sorted(
        at_module_level & declaration)
    never = {"ARM_ORDER", "NAME_OF", "LADDER", "FLOOR_OF", "CHECKPOINTS"}
    assert not (everywhere & never), sorted(everywhere & never)


def test_build_arm_refuses_a_digit_benchmark_arm_rather_than_building_it(
        familia, tiny):
    """`wiring.build` would have resolved `"G"` through `config.ARMS_BY_ID`.
    This module builds `wiring.Arm` directly, so an id of the other benchmark
    is refused by name instead of silently constructing the other experiment's
    arm."""
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    with toy_run.material_bound(familia, tiny, "mlp"):
        for foreign in sorted(config.ARMS_BY_ID):
            assert foreign not in toy_run.TOY_ARMS_BY_ID
            with pytest.raises(ValueError, match="unknown toyset arm"):
                toy_run.build_arm(foreign, material, tiny,
                                  toy_run.declared_hyper(tiny), DEVICE, 0)
        # The control: a declared id of THIS ladder builds.
        built = toy_run.build_arm(ADAPTED, material, tiny,
                                  toy_run.declared_hyper(tiny), DEVICE, 0)
        assert built.spec is toy_run.TOY_ARMS_BY_ID[ADAPTED]


# --------------------------------------------- what phase 2 actually computes

def _one_step(notebook, familia, tiny, model, payload,
              coefficient=toy_run.FIXED_COEFFICIENT):
    """One phase-2 training step at Eq. (39)'s declared coefficient, so a
    contribution IS its term: `training_step` books `contributionGlobal =
    coefficient * global_term`.

    The default is read off `toy_run.FIXED_COEFFICIENT` and not written as
    `1.0` here: a test helper spelling the constant again would keep passing a
    one into a module that had moved off it.
    """
    with toy_run.material_bound(familia, tiny, "mlp"):
        x, y = _batch(notebook, payload["source_train"],
                      range(len(payload["source_train"])))
        generator = torch.Generator().manual_seed(4242)
        toy_run.set_phase_two_modes(model)
        return model.training_step(x, y, coefficient, generator,
                                   ramp_local=coefficient)


def test_the_adapted_arm_computes_a_nonzero_global_term_and_a_nonzero_local_one(
        notebook, familia, tiny):
    """BOTH phase-2 losses, and both measured away from zero.

    Two assertions and not one, in both directions: "the term did nothing" and
    "the term was never called" print the same `0.0`, so each term is read on
    its own. The step runs at coefficient one so the contribution the record
    carries IS the term, with no schedule between the assertion and the
    mathematics.
    """
    payload = _prepared(notebook, familia, tiny, arm_id=ADAPTED)
    record = _one_step(notebook, familia, tiny, payload["model"], payload)
    assert record["contributionGlobal"] > 0.0, record
    assert record["contributionLocal"] > 0.0, record
    assert record["contribution"] == pytest.approx(
        record["contributionGlobal"] + record["contributionLocal"])
    # The two are different computations, so a fixture where they came out
    # equal would not distinguish one term reported twice from two terms.
    assert record["contributionGlobal"] != record["contributionLocal"]
    # And the diagnostics prove the global block was really built: the floor
    # reports `nan` for every one of these by construction.
    assert not math.isnan(record["targetBagEntropy"])
    assert record["activeClasses"] >= 1


def test_an_unadapted_spec_computes_neither_term(notebook, familia, tiny):
    """A spec with no adaptation term runs none, and the declared arm on the
    very same batch runs both -- which is what makes the zeros a reading and not
    a default.

    The spec is `_UNADAPTED_CONTROL_SPEC` and NOT an arm of this toyset: the
    floor arm this used to be written against was retired, and reintroducing it
    as a declared arm to keep a test's control would put a floor back in the
    run. What a control needs is to be built and stepped, which `wiring.Arm`
    offers on any spec dict; what it must not be is reachable as an arm of the
    tutorial, and `build_arm` refuses it by name.
    """
    payload = _prepared(notebook, familia, tiny, arm_id=ADAPTED)
    with toy_run.material_bound(familia, tiny, "mlp"):
        control = _unadapted_control(payload["material"], tiny,
                                     payload["hyper"], DEVICE)
        assert control.target_encoder is None, (
            "the control built a theta_t; its spec declares no adaptation term")
        with pytest.raises(ValueError, match="unknown toyset arm"):
            toy_run.build_arm(_UNADAPTED_CONTROL_SPEC["id"],
                              payload["material"], tiny, payload["hyper"],
                              DEVICE, 0)
    record = _one_step(notebook, familia, tiny, control, payload)
    assert record["adaptation"] == 0.0
    assert record["contributionGlobal"] == 0.0
    assert record["contributionLocal"] == 0.0
    assert record["contribution"] == 0.0
    # Every diagnostic is `nan` and never `0.0`: `0.0` is the collapsed reading
    # the diagnostic exists to report, so an unavailable one must not look like
    # a degenerate one.
    assert math.isnan(record["targetBagEntropy"])
    assert math.isnan(record["targetConfidenceMean"])
    assert math.isnan(record["activeClasses"])

    # The control, on the same material and the same batch.
    adapted_payload = _prepared(notebook, familia, tiny, arm_id=ADAPTED)
    adapted = _one_step(notebook, familia, tiny, adapted_payload["model"],
                        adapted_payload)
    assert adapted["contributionGlobal"] > 0.0
    assert adapted["contributionLocal"] > 0.0
    assert not math.isnan(adapted["targetBagEntropy"])


def test_the_adapted_arm_declares_the_weighting_and_all_three_mechanisms_run(
        notebook, familia, tiny, monkeypatch):
    """`weighting: True` switches ON both mechanisms the flag gates, and the
    third confidence reading was never behind it and runs too. All three are
    MEASURED, never read off the flag.

    - **Eq. (26)'s `D`:** `conditional.weighted_blocks` IS reached, and what it
      built is read: the `D` is not the identity and it moves both blocks. A
      call alone would not distinguish "applied" from "applied as `I`".
    - **Eq. (37)'s weights:** `global_loss` is given SOURCE COUNTS, and those
      counts are read: non-negative, summing to the source batch, and they
      arrive on the twin too, because r33's weights come from the observed
      labels and not from the confidences.
    - **Eq. (38)'s aggregation:** `local_loss` is reached, and is reached with
      ONE argument: the confidences have no way in.

    **Only the first of the three is gated by this flag, since 2026-10-18.**
    This test used to declare all three: r29 built Eq. (37)'s weights from the
    same `w_j^t` as Eq. (26)'s `D`, and Eq. (38) aggregated by those
    confidences too, so the axis reached every one. r32 replaced Eq. (37)'s
    weights with `n_c^s / N_s` and made Eq. (38) the uniform mean -- a weight
    the model computes can be lowered instead of earned -- so two of the three
    stopped being confidence mechanisms at all and the axis means exactly one
    thing again.

    **A spy that fires proves it was reached, not that the branch is a branch**,
    so a twin -- the same object, deep-copied, with `weighting` flipped to
    `False` and every parameter bit-identical -- runs the same batch. On the
    twin the first two spies are SILENT and `local_loss` still fires, which is
    what makes "all three run here" a reading of this arm's declaration rather
    than of the code having one path.

    This test replaces `test_with_the_weighting_off_...`, which asserted the
    mirror image: that the first two never fired and the twin was the one that
    fired them. The arm's declaration moved, so the poles moved with it.
    """
    payload = _prepared(notebook, familia, tiny, arm_id=ADAPTED)
    model = payload["model"]
    # The FLAG is read at the END of this test, deliberately. Asserted here it
    # would be the first thing to fail when the declaration moves, and a
    # mutation of the declaration would then never reach the spy assertions
    # below -- so the half of this test that measures the mechanisms would
    # never be shown to have a reachable red. Measured in that order: flipping
    # `weighting` to `False` in `TOY_ARMS` fails on the Eq. (26) assertion,
    # not on a flag.
    twin = copy.deepcopy(model)
    # The twin's flag is the NEGATION of the arm's and is not spelled out: a
    # literal `False` here would be a second reading of the declaration, and
    # the aliasing control below would then fail before any mechanism was
    # measured. What this control is actually about is that the two specs are
    # separate objects, which `!=` says without pinning either value.
    twin.spec = {**model.spec, "weighting": not model.spec["weighting"]}
    assert twin.spec["weighting"] != model.spec["weighting"], (
        "the twin's spec aliased the arm's")
    assert float((_parameter_vector(model.encoder)
                  - _parameter_vector(twin.encoder)).norm()) == 0.0
    assert float((_parameter_vector(model.target_encoder)
                  - _parameter_vector(twin.target_encoder)).norm()) == 0.0

    calls = {"weighted_blocks": [], "global_loss": [], "local_loss": []}
    real_weighted_blocks = wiring.weighted_blocks
    real_global_loss = wiring.global_loss
    real_local_loss = wiring.local_loss

    def spy_weighted_blocks(K_st_c, K_t_c, target_weights):
        out = real_weighted_blocks(K_st_c, K_t_c, target_weights)
        calls["weighted_blocks"].append(
            (K_st_c.detach().clone(), K_t_c.detach().clone(),
             target_weights.detach().clone(),
             out[0].detach().clone(), out[1].detach().clone()))
        return out

    def spy_global_loss(class_losses, source_counts):
        calls["global_loss"].append((list(class_losses), list(source_counts)))
        return real_global_loss(class_losses, source_counts)

    def spy_local_loss(squared_distances):
        calls["local_loss"].append(squared_distances.detach().clone())
        return real_local_loss(squared_distances)

    monkeypatch.setattr(wiring, "weighted_blocks", spy_weighted_blocks)
    monkeypatch.setattr(wiring, "global_loss", spy_global_loss)
    monkeypatch.setattr(wiring, "local_loss", spy_local_loss)

    on = _one_step(notebook, familia, tiny, model, payload)

    # Eq. (26): applied, and what it applied is not the identity. Both halves,
    # because `weighted_blocks(K, K, ones)` is a call that does nothing.
    assert calls["weighted_blocks"], (
        "Eq. (26)'s D was never applied by an arm declaring the weighting")
    K_st_c, K_t_c, w, K_st_w, K_t_w = calls["weighted_blocks"][0]
    D = torch.diag(w.reshape(-1))
    assert float((D - torch.eye(D.shape[0], dtype=D.dtype)).abs().max()) > 0.0, (
        "D came out the identity, so this fixture cannot distinguish the "
        "weighted blocks from the raw ones")
    assert float((K_t_w - K_t_c).abs().max()) > 0.0
    assert float((K_st_w - K_st_c).abs().max()) > 0.0

    # Eq. (37): called with SOURCE COUNTS, one per class, summing to the source
    # batch -- the weights come from the observed labels, so `w_t` is nowhere
    # in them. One entry per class in C, with `None` for every class the
    # routing abandoned, which is what makes the charge of 1 reachable.
    assert calls["global_loss"], "the global aggregation was never reached"
    class_losses, source_counts = calls["global_loss"][0]
    assert len(class_losses) == len(source_counts) == model.classes
    assert all(n >= 0 for n in source_counts), source_counts
    assert sum(source_counts) > 0, source_counts
    weighted_counts = list(source_counts)

    # Eq. (38): reached, and reached with ONE argument -- the aggregation has
    # no parameter a confidence could arrive through.
    assert calls["local_loss"], "the local aggregation was never reached"
    weighted_local_d = calls["local_loss"][0]
    assert weighted_local_d.numel() > 1
    assert 'target_weights' not in real_local_loss.__code__.co_varnames

    for key in calls:
        calls[key] = []
    off = _one_step(notebook, familia, tiny, twin, payload)

    # The twin: the ONE gated mechanism goes silent, which is what makes its
    # firing above a reading of the declaration and not of the only path.
    assert calls["weighted_blocks"] == [], (
        "Eq. (26)'s D was applied by an arm declaring no weighting")
    # And the two that are NOT gated fire on the twin too, with the same source
    # counts: r33's Eq. (37) weights are the observed labels' and Eq. (38) is
    # behind `spec["local"]`, so switching this flag reaches neither.
    assert calls["global_loss"], "the global aggregation was never reached"
    assert calls["global_loss"][0][1] == weighted_counts, (
        "the source counts moved with the weighting flag; they come from the "
        "observed labels and have nothing to do with the confidences")
    assert calls["local_loss"], "the local aggregation was never reached"
    assert float((calls["local_loss"][0] - weighted_local_d).abs().max()) == 0.0
    assert off["contributionLocal"] == on["contributionLocal"], (
        "the local term moved with the flag; it is supposed to be behind "
        "spec['local'] and not behind this one")

    # The two arms therefore compute different global terms, which is the
    # end-to-end reading the spies predict.
    assert on["contributionGlobal"] != off["contributionGlobal"]

    # And only now the declaration itself, so the record of what this arm says
    # sits beside the measurement of what it does.
    assert model.spec["weighting"] is True
    assert model.spec["local"] is True
    assert toy_run.TOY_ARMS_BY_ID[ADAPTED]["weighting"] is True


# ----------------------------------- the two coefficients, fixed at one

def test_the_coefficient_reaching_training_step_is_exactly_one_at_every_step(
        notebook, familia, tiny, monkeypatch):
    """Eq. (39)'s two coefficients, read where they ARRIVE rather than where
    they are declared: `wiring.Arm.training_step`'s own `ramp` and `ramp_local`
    arguments, captured on every step of the tutorial's own phase 2.

    Four things asserted and each one can fail on its own:

    - every captured pair is EXACTLY `1.0` -- `==`, not `approx`, because a
      declared constant that arrived as 0.9999999 came through an arithmetic
      nobody declared;
    - the two are equal to each other, so one is not quietly the other's twin
      through a different path;
    - there is one capture PER STEP and at least two steps ran, so "exactly
      one at every step" is a statement about several steps;
    - the captured value is the module's `FIXED_COEFFICIENT` and not a literal
      written here.
    """
    payload = _prepared(notebook, familia, tiny, arm_id=ADAPTED)
    model = payload["model"]
    captured: list[tuple] = []
    real_step = wiring.Arm.training_step

    def spy(self, bags_, labels, ramp, generator, ramp_local=None):
        captured.append((ramp, ramp_local))
        return real_step(self, bags_, labels, ramp, generator,
                         ramp_local=ramp_local)

    monkeypatch.setattr(wiring.Arm, "training_step", spy)
    with toy_run.material_bound(familia, tiny, "mlp"):
        phase_two = notebook["entrenar_fase_dos"](
            model, payload["source_train"], torch.Generator().manual_seed(7))

    assert len(captured) >= 2, captured
    assert len(captured) == len(phase_two["historia"]), (
        len(captured), len(phase_two["historia"]))
    for coefficient, coefficient_local in captured:
        assert coefficient == toy_run.FIXED_COEFFICIENT, captured
        assert coefficient_local == toy_run.FIXED_COEFFICIENT, captured
        assert coefficient == coefficient_local
    assert toy_run.FIXED_COEFFICIENT == 1.0


# --------------------------- the four retired keys, refused one test per key

def _refusal_fixture(familia, tiny, key, value=0.5):
    """A hyper dict with exactly one retired key added to the declared base.

    Returned rather than asserted on here: each key gets its OWN test below,
    never a loop over the four, because a loop that skipped an entry -- a typo
    in the tuple, a `continue` that fires -- passes identically to one that
    checked all four.
    """
    base = toy_run.declared_hyper(tiny)
    assert key not in base, (
        f"declared_hyper emits {key}, so every call on this path would refuse")
    return {**base, key: value}


def test_ceiling_is_refused_by_name(familia, tiny):
    """`ceiling` IS Eq. (39)'s lambda_glob. A stale `toy_search.json` carries it
    and the caller who passes it believes it chooses the coefficient."""
    hyper = _refusal_fixture(familia, tiny, "ceiling")
    for call in _every_entry_taking_hyper(familia, tiny):
        with pytest.raises(ValueError, match="ceiling is retired"):
            call(hyper)
    # THE CONTROL: the same calls accept the base dict without the key.
    for call in _every_entry_taking_hyper(familia, tiny):
        call(toy_run.declared_hyper(tiny))


def test_ceiling_local_is_refused_by_name(familia, tiny):
    """`ceilingLocal` IS Eq. (39)'s lambda_loc."""
    hyper = _refusal_fixture(familia, tiny, "ceilingLocal")
    for call in _every_entry_taking_hyper(familia, tiny):
        with pytest.raises(ValueError, match="ceilingLocal is retired"):
            call(hyper)


def test_ramp_delta_is_refused_by_name(familia, tiny):
    """`rampDelta` is how lambda_glob grew. Nothing grows."""
    hyper = _refusal_fixture(familia, tiny, "rampDelta", value=20.0)
    for call in _every_entry_taking_hyper(familia, tiny):
        with pytest.raises(ValueError, match="rampDelta is retired"):
            call(hyper)


def test_ramp_delta_local_is_refused_by_name(familia, tiny):
    """`rampDeltaLocal` is how lambda_loc grew."""
    hyper = _refusal_fixture(familia, tiny, "rampDeltaLocal", value=20.0)
    for call in _every_entry_taking_hyper(familia, tiny):
        with pytest.raises(ValueError, match="rampDeltaLocal is retired"):
            call(hyper)


def test_the_refusal_is_not_one_choke_point_but_every_entry_taking_a_hyper_dict():
    """Each public entry on this path that takes a hyper dict calls the guard.

    The four tests above run every one of them, so this is the static half: it
    names them, so an entry ADDED later without the call is visible as a
    missing name here rather than as a silent hole. Each of the two is reachable
    on its own from the tutorial.
    """
    guarded = ("build_arm", "material_bound")
    for name in guarded:
        source = inspect.getsource(getattr(toy_run, name))
        assert "refuse_retired_hyper(hyper)" in source, name
    # And every module-level function whose signature takes a `hyper`
    # parameter is accounted for -- DERIVED from the module rather than listed,
    # so a function added later with a hyper dict and no guard turns this red
    # instead of being assumed absent. `refuse_retired_hyper` is the guard.
    takes_hyper = {
        name for name, value in vars(toy_run).items()
        if inspect.isfunction(value)
        and "hyper" in inspect.signature(value).parameters}
    # `material_bound` is a generator wrapped by `contextlib.contextmanager`,
    # so `vars()` holds the wrapper and not the function; its guard is asserted
    # in the loop above and its name is added here rather than derived.
    takes_hyper.add("material_bound")
    assert takes_hyper == set(guarded) | {"refuse_retired_hyper"}, (
        sorted(takes_hyper))


def _every_entry_taking_hyper(familia, tiny):
    """One callable per guarded entry, each taking just the hyper dict.

    Built so the four refusal tests above exercise the REAL call sites rather
    than `refuse_retired_hyper` alone: a guard that exists and is wired to
    nothing passes a direct test and protects nobody.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")

    def via_build_arm(hyper):
        with toy_run.material_bound(familia, tiny, "mlp"):
            return toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)

    def via_material_bound(hyper):
        with toy_run.material_bound(familia, tiny, "mlp", hyper):
            return None

    return (via_build_arm, via_material_bound)


# ------------------------------------ the decision per bag, proved by mutation
#
# Every test in this section proves its claim by MUTATING `toy_run.py`'s own
# source into a second module and showing that the claim fails there. Three
# habits make that a proof rather than a gesture, and all three were paid for in
# this repository before they were written down:
#
# - **Same length, so the byte count cannot be what fires.** Each mutation
#   replaces a span with another of identical length, and the helper asserts the
#   copy's size equals the original's. A test that detected a changed file size
#   would pass on a mutation that never reached the parser.
# - **md5, so «the mutation landed» is measured.** `git diff --stat` says
#   nothing about an untracked copy and a `str.replace` that matched nothing
#   returns the string unchanged, silently. The digest has to MOVE before
#   anything below it is believed.
# - **`__pycache__` cleared, because a same-length mutation reuses a stale
#   `.pyc`.** A cached bytecode file is validated against (size, mtime) by
#   default, and a same-size rewrite inside the same second has matched both
#   here before: the mutated source never ran and the live code read as dead.


def _md5(path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def _mutated_toy_run(tmp_path, before: str, after: str, name: str):
    """A second `toy_run`, loaded from a same-length mutation of its own source.

    Returns the loaded module. Refuses rather than returning something useless:
    a `before` that does not occur exactly once would make the mutation either a
    no-op or a shotgun, and either way the red below would not be about the line
    the test names.
    """
    import shutil
    from pathlib import Path

    original = Path(toy_run.__file__)
    source = original.read_text(encoding="utf-8")
    assert len(before) == len(after), (
        f"la mutación cambia el largo ({len(before)} -> {len(after)}): con otro "
        "largo el conteo de bytes alcanzaría para detectarla y la comprobación "
        "de md5 dejaría de ser la que mide que llegó")
    assert source.count(before) == 1, (
        f"{before!r} aparece {source.count(before)} veces en toy_run.py: una "
        "mutación que no cae en exactamente un lugar no prueba nada sobre la "
        "línea que este test nombra")

    destino = tmp_path / f"{name}.py"
    destino.write_text(source.replace(before, after), encoding="utf-8")
    assert destino.stat().st_size == original.stat().st_size, (
        destino.stat().st_size, original.stat().st_size,
        "la copia no tiene el mismo tamaño que el original")
    assert _md5(destino) != _md5(original), (
        "la mutación no movió el md5: el reemplazo no llegó al archivo y todo "
        "lo que sigue mediría el código sin mutar")
    # A same-length rewrite inside the same second matches a cached `.pyc` on
    # both of the fields the default validator reads.
    shutil.rmtree(tmp_path / "__pycache__", ignore_errors=True)
    return _load_module_from(destino, name)


def test_the_prediction_written_is_the_heads_own_decision_per_bag(
        notebook, familia, tiny):
    """`bag_positions` writes `argmax` of the arm's own scoring, bag by bag.

    Crossed against `Arm.score_target`, which is a DIFFERENT entry point: it
    re-encodes the bags itself instead of reading the embeddings the positions
    came from. Comparing the writer with a fixture this file wrote would verify
    both halves and never the join.
    """
    preparado = _prepared(notebook, familia, tiny)
    modelo, material = preparado["model"], preparado["material"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        leido = toy_run.bag_positions(modelo, material["target"], DEVICE,
                                      "target")
        modelo.eval()
        bolsas = material["target"]
        total = int(bolsas.members.shape[0])
        aparte = []
        with torch.no_grad():
            for inicio in range(0, total, config.BAGS_PER_STEP):
                trozo = torch.arange(inicio, min(inicio + config.BAGS_PER_STEP,
                                                 total))
                x = bolsas.images[bolsas.members[trozo]].to(DEVICE)
                aparte.append(modelo.score_target(x).argmax(dim=1).cpu())
        aparte = torch.cat(aparte).numpy()
    assert leido["predictions"].shape == (total,)
    assert np.array_equal(leido["predictions"].astype(int), aparte.astype(int))
    # The control: a decision channel that simply copied the labels would also
    # satisfy the line above only if the head were perfect, so the fixture has
    # to be one where it is not.
    assert not np.array_equal(leido["predictions"], leido["classes"]), (
        "el cabezal acierta TODAS las bolsas en esta fixtura, así que «la "
        "predicción» y «la clase» son el mismo arreglo y la aserción de arriba "
        "no distingue cuál de los dos se escribió")


def test_a_prediction_read_off_the_wrong_end_of_the_scores_is_caught(
        notebook, tmp_path, familia, tiny):
    """Take the argmin instead and the written decision stops being the head's.

    The red for the test above. Without it, a channel that wrote any
    class-shaped integer per bag would satisfy every shape and dtype assertion
    in this file.
    """
    mutado = _mutated_toy_run(tmp_path, ".argmax(dim=1).cpu())",
                              ".argmin(dim=1).cpu())", "toy_run_argmin")
    preparado = _prepared(notebook, familia, tiny)
    modelo, material = preparado["model"], preparado["material"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        vivo = toy_run.bag_positions(modelo, material["target"], DEVICE,
                                     "target")["predictions"]
        roto = mutado.bag_positions(modelo, material["target"], DEVICE,
                                    "target")["predictions"]
    assert not np.array_equal(vivo, roto), (
        "argmin y argmax dieron lo mismo sobre esta fixtura: los puntajes "
        "estarían empatados y el test de arriba no estaría distinguiendo nada")


# ===================================================================
#   el dispositivo del toyset: que lo resuelva quien lo midio
# ===================================================================
#
# El 2026-10-21 el toyset dejo de usar `harness.resolve_device` --- cuyo orden
# `cuda > mps > cpu` es el correcto para el banco de digitos y el equivocado
# aca, donde un paso empuja 90 puntos por un MLP de dimension 8 --- y resuelve
# por `dispositivo_del_toyset`, que devuelve `cpu`. Lo que estos dos tests
# cuidan no es la velocidad: el dispositivo MUEVE el `targetAccuracy` (el mismo
# brazo dio 0.5556 / 0.3333 / 0.6111 / 0.5000 en cuatro dispositivos), asi que
# un dispositivo cambiado por accidente no se lee como un error, se lee como
# otro resultado.


def _llamadas_a(fuente: str, punto: str, atributo: str) -> list[int]:
    """Las LINEAS donde `punto.atributo(...)` se llama de verdad.

    Con `ast` y no contando la cadena: el nombre viejo sigue --- y tiene que
    seguir --- nombrado en el docstring de `dispositivo_del_toyset`, que es
    donde se explica por que el banco de digitos lo quiere y el toyset no. Un
    conteo de subcadenas no distingue una explicacion de una llamada, asi que
    reprobaria el archivo por documentarse.
    """
    import ast
    arbol = ast.parse(fuente)
    return [n.lineno for n in ast.walk(arbol)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == atributo
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == punto]


def test_el_toyset_resuelve_su_dispositivo_por_el_suyo():
    """Ninguna funcion de `toy_run` llama `harness.resolve_device`."""
    fuente = pathlib.Path(toy_run.__file__).read_text(encoding="utf-8")
    llamadas = _llamadas_a(fuente, "harness", "resolve_device")
    assert llamadas == [], (
        f"`toy_run` llama `harness.resolve_device` en las lineas {llamadas}: "
        "ese orden pone el acelerador primero, que es correcto para el banco "
        "de digitos y 11.6x mas lento aca --- y cambia el resultado")
    assert toy_run.dispositivo_del_toyset() == torch.device("cpu")
    # Y que el nombre viejo siga EXPLICADO: si desaparece del archivo, la
    # proxima sesion no tiene donde leer por que no se usa.
    assert "harness.resolve_device" in fuente, (
        "se fue hasta la explicacion de por que el toyset no usa "
        "`harness.resolve_device`, y sin ella el cambio parece un capricho")


def test_el_cuaderno_tutorial_no_entrega_el_dispositivo_del_banco():
    """El cuaderno no puede pasarle a `toy_run` un dispositivo del banco.

    **Este es el defecto que nada pescaba.** Los seis sitios de `toy_run` que
    resuelven el dispositivo lo hacen cuando `device is None`; el cuaderno
    generador (retirado el 2026-11-02) lo pasaba EXPLICITO, desde
    `harness.resolve_device()`, asi que los seis quedaban inertes y la corrida
    entera habria ido por MPS --- 5.42 h se vuelven ~71 h, con otros numeros.
    Cambiar el default de un parametro no alcanza cuando el llamador lo pasa;
    lo que hay que medir es QUIEN llama. El tutorial de MIL-CREDA pasa
    `DISPOSITIVO` explicito a `toy_run`, asi que la misma guarda vale para el.
    """
    cuaderno = json.loads(
        (pathlib.Path(toy_run.__file__).resolve().parents[2]
         / "MIL-CREDA" / "Notebooks" / "Toyset.ipynb")
        .read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in cuaderno["cells"]
                       if c["cell_type"] == "code")

    assert _llamadas_a(codigo, "harness", "resolve_device") == [], (
        "el cuaderno tutorial resuelve el dispositivo con "
        "`harness.resolve_device`, y lo pasa explicito, asi que los seis "
        "sitios de `toy_run` no se alcanzan")
    assert _llamadas_a(codigo, "toy_run", "dispositivo_del_toyset"), (
        "el cuaderno no nombra `toy_run.dispositivo_del_toyset`, asi que no "
        "hay nada que diga en que dispositivo entrena")


# ===================================================================
#   la lectura de bolsas: g_j y w_j por bolsa, del dominio declarado
# ===================================================================


def test_la_lectura_de_bolsas_se_niega_ante_un_dominio_que_no_existe():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, toy_run.PILOT,
                                  hyper, torch.device("cpu"), 0)
        with pytest.raises(ValueError):
            toy_run.lectura_de_bolsas(brazo, material["target"],
                                      torch.device("cpu"), "el-de-al-lado")


def test_la_lectura_de_bolsas_devuelve_w_en_cero_uno_y_una_fila_por_bolsa():
    familia = toy_run.build_family("toroide", 0, toy_run.PILOT)
    hyper = toy_run.declared_hyper(toy_run.PILOT)
    material = toy_run.material_for(familia, 0, toy_run.PILOT, "mlp-nobn")
    with toy_run.material_bound(familia, toy_run.PILOT, "mlp-nobn", hyper):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, toy_run.PILOT,
                                  hyper, torch.device("cpu"), 0)
        leido = toy_run.lectura_de_bolsas(brazo, material["target"],
                                          torch.device("cpu"), "target")
    total = int(material["target"].members.shape[0])
    assert leido["scores"].shape == (total, toy_run.PILOT.classes)
    assert leido["w"].shape == (total,)
    assert np.all(leido["w"] >= 0.0) and np.all(leido["w"] <= 1.0)
    assert leido["scores"].sum(axis=1) == pytest.approx(
        np.ones(total), abs=1e-5)


# ===================================================================
#   la caminata de w: un destino por clase, y que no colapsen
# ===================================================================
#
# Hasta el 2026-10-21 la caminata iba a UN destino, `CLASE_DESTINO_W = 0`, y
# nada en la suite miraba su salida: los 101 tests que nombran «caminata»
# pasaban igual con la forma vieja y con la nueva, porque leen el codigo fuente
# de la funcion y nunca lo que devuelve. Estos la ejecutan.


def _caminata_chica(notebook, tiny):
    """`caminata_de_w` corrida de verdad, sobre un brazo entrenado al minimo."""
    familia = toy_run.build_family("lunas", 0, tiny)
    backbone = toy_run.NORMALIZATIONS["batchnorm"]
    material = toy_run.material_for(familia, 0, tiny, backbone)
    hyper = toy_run.declared_hyper(tiny)
    source_train, source_valid, _ = bags.roles(material["source"])
    with toy_run.material_bound(familia, tiny, backbone):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, tiny, hyper,
                                  DEVICE, 0)
        _train_phase_one(notebook, brazo, source_train, source_valid)
        # La caminata es geometria del cabezal YA entrenado --- no entrena
        # nada --- asi que la fase 1 de arriba es lo unico que hace falta.
        return toy_run.caminata_de_w(
            brazo, material["source"], material["target"], DEVICE)


def test_cada_camino_sale_de_SU_bolsa_del_blanco_y_va_a_SU_clase(notebook, tiny):
    """Tres caminos, no nueve: la declaracion de `CREDA.toy_plots`.

    Alla se escribe literal `arranques = [((0, 1), 0), ((1, 2), 2),
    (tuple(range(CLASES)), 1)]` --- la frontera de las clases nombradas camina
    al centro de UNA de ellas --- y aca es lo mismo un nivel arriba, con una
    BOLSA DEL BLANCO en lugar de una instancia, y la frontera como criterio
    para elegirla. Ruling del operador del 2026-10-22.

    Lo que se asierta es el EMPAREJAMIENTO, que es lo que se pidio: cada
    arranque trae exactamente el destino que `CAMINOS_W` le puso, y ninguno
    trae los otros dos. Con los tres destinos por arranque esto pasa a ser
    nueve caminos y la figura vuelve a ser la que no servia.
    """
    c = _caminata_chica(notebook, tiny)
    assert c["clasesDestino"] == [d for _, d in toy_run.CAMINOS_W], \
        c["clasesDestino"]
    assert set(c["caminos"]) == set(toy_run.ARRANQUES_W)
    for clases, suyo in toy_run.CAMINOS_W:
        arranque = toy_run._nombre_del_arranque(clases)
        destinos = c["caminos"][arranque]
        assert set(destinos) == {str(suyo)}, (arranque, sorted(destinos))
        camino = destinos[str(suyo)]
        assert len(camino["w"]) == c["barridos"], arranque
        assert camino["camino"].shape[0] == c["barridos"]


def test_la_bolsa_que_arranca_es_la_mas_indecisa_y_no_un_umbral(tiny):
    """La propiedad EXACTA que reemplazo a un umbral adivinado.

    Hasta el 2026-10-22 el arranque era el punto de frontera mismo y este test
    asertaba `w0 < 0.9`. Con el arranque en una bolsa REAL del blanco ese
    numero pasa a depender de donde cayeron las bolsas, asi que seria una
    adivinanza mia escrita como si fuera una cota de la formulacion. Lo que se
    puede afirmar exacto es la REGLA: la bolsa elegida es el argmin de
    indecision, y eso se prueba sobre `_bolsa_mas_indecisa` con un `g` armado a
    mano, donde la respuesta correcta se conoce sin correr nada.

    El caso que importa es el tercero: una bolsa que apuesta casi todo a la
    clase 2 y reparte una miga PAREJA entre 0 y 1 tiene la brecha `|g_0 - g_1|`
    mas chica de la tabla. Sin el filtro por las dos clases de arriba ganaria
    la eleccion de la frontera (0,1), y la figura arrancaria con una bolsa que
    no duda entre 0 y 1 --- duda nada, ya decidio la 2.
    """
    import torch
    g = torch.tensor([
        [0.45, 0.40, 0.15],   # 0: duda entre 0 y 1, brecha 0.05
        [0.48, 0.47, 0.05],   # 1: duda entre 0 y 1, brecha 0.01  <- la elegida
        [0.02, 0.015, 0.965],  # 2: decidio la 2, brecha 0.005 (la mas chica)
        [0.34, 0.33, 0.33],   # 3: casi uniforme   <- la del empate a tres
    ])
    j, relajado = toy_run._bolsa_mas_indecisa(g, (0, 1))
    assert j == 1, (j, "gano la que decidio la clase 2: falta el filtro por "
                       "las dos clases de arriba")
    assert relajado is False
    k, _ = toy_run._bolsa_mas_indecisa(g, (0, 1, 2))
    assert k == 3, (k, "el empate a TRES es el `w` mas bajo, no la brecha")


def test_un_par_que_ninguna_bolsa_duda_sale_marcado_como_relajado(tiny):
    """Un relajado silencioso convierte la figura en otra figura.

    Si ninguna bolsa del blanco tiene a `{a, b}` como sus dos clases mas
    probables, esa frontera no es una frontera que el blanco dude y la
    eleccion cae sobre todas. Eso puede pasar y no es un error --- pero tiene
    que VIAJAR, porque la leyenda sigue diciendo "mas indecisa entre a y b".
    """
    import torch
    g = torch.tensor([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
    ])
    _, relajado = toy_run._bolsa_mas_indecisa(g, (0, 2))
    assert relajado is True, (
        "ninguna bolsa tiene a {0, 2} arriba y la eleccion no se marco")


def test_los_tres_caminos_arrancan_en_bolsas_DISTINTAS(tiny):
    """El reparto, sobre el `g` que lo rompio de verdad.

    La bolsa 0 de abajo es casi uniforme: es a la vez la mas pareja entre 0 y
    1 Y la mas pareja entre las tres, asi que con la regla cruda gana las dos
    elecciones y la figura dibuja el mismo punto dos veces. Medido a escala
    completa el 2026-10-22: `rollo` dio la bolsa 41 a la frontera (1,2) y a la
    triple, y `toroide` dio la bolsa 0 a la (0,1) y a la triple.

    Lo que se asierta es el reparto, no un optimo: por el orden de
    `CAMINOS_W`, la (0,1) toma la casi uniforme y la TRIPLE --- que llega
    ultima --- paga el desempate con la que quede.
    """
    import torch
    g = torch.tensor([
        [0.34, 0.33, 0.33],   # casi uniforme: gana (0,1) Y la triple
        [0.45, 0.40, 0.15],   # duda entre 0 y 1, pero menos
        [0.10, 0.46, 0.44],   # duda entre 1 y 2
        [0.30, 0.36, 0.34],   # la segunda mas uniforme
    ])
    reparto = toy_run._bolsas_que_arrancan(g)
    indices = [reparto[toy_run._nombre_del_arranque(cl)][0]
               for cl, _ in toy_run.CAMINOS_W]
    assert len(set(indices)) == 3, (
        f"dos caminos arrancan en la misma bolsa {indices}: la figura dibuja "
        "el mismo punto dos veces y cuenta dos historias con una")
    assert indices[0] == 0, indices


def test_la_caminata_NO_mira_ninguna_etiqueta_del_blanco(notebook, tiny):
    """La mutacion es la unica prueba de que el oraculo no entra.

    La Ec. (24) define `w_j^t` sobre `g_j^t` y nada mas; la Ec. (22) deriva la
    pseudoetiqueta del mismo posterior. Ninguna de las dos conoce
    `target.labels` --- en el blanco no hay etiqueta observada, ese es el
    punto entero del problema. Un comentario que lo afirme no prueba nada:
    esto le da vuelta las etiquetas del blanco y exige que la caminata salga
    IDENTICA, byte por byte.

    Es lo que habria agarrado el defecto del 2026-10-22 en el otro sentido:
    la figura leia la geometria de la FUENTE, donde la etiqueta SI es
    observada, asi que una mutacion de las etiquetas del BLANCO no la habria
    movido tampoco. Por eso este test viene con el de arriba y no solo: aquel
    fija que el arranque es una bolsa del blanco, este que su eleccion no
    miro la respuesta.
    """
    import dataclasses
    import torch
    familia = toy_run.build_family("lunas", 0, tiny)
    backbone = toy_run.NORMALIZATIONS["batchnorm"]
    material = toy_run.material_for(familia, 0, tiny, backbone)
    hyper = toy_run.declared_hyper(tiny)
    source_train, source_valid, _ = bags.roles(material["source"])
    with toy_run.material_bound(familia, tiny, backbone):
        brazo = toy_run.build_arm(toy_run.TOY_ADAPTED, material, tiny, hyper,
                                  DEVICE, 0)
        _train_phase_one(notebook, brazo, source_train, source_valid)
        limpia = toy_run.caminata_de_w(
            brazo, material["source"], material["target"], DEVICE)
        # La mutacion, y la prueba de que la mutacion OCURRIO: una rotacion
        # que dejara las etiquetas donde estaban convertiria este test en uno
        # que no prueba nada y pasa igual.
        rotadas = torch.roll(material["target"].labels, 1)
        assert not torch.equal(rotadas, material["target"].labels), \
            "la rotacion no movio ninguna etiqueta: el test no probaria nada"
        sucio = dataclasses.replace(material["target"], labels=rotadas)
        mutada = toy_run.caminata_de_w(
            brazo, material["source"], sucio, DEVICE)
    for clases, suyo in toy_run.CAMINOS_W:
        arranque = toy_run._nombre_del_arranque(clases)
        assert (limpia["bolsaDelBlanco"][arranque]
                == mutada["bolsaDelBlanco"][arranque]), arranque
        a = limpia["caminos"][arranque][str(suyo)]["w"]
        b = mutada["caminos"][arranque][str(suyo)]["w"]
        assert (a == b).all(), (
            f"{arranque}: dar vuelta las etiquetas del blanco movio `w`, asi "
            "que la caminata las esta mirando y la Ec. (24) no las nombra")


def test_los_tres_caminos_no_son_el_mismo_camino_tres_veces(notebook, tiny):
    """Tres fronteras distintas y tres destinos distintos, asi que tres curvas.

    Con un destino unico para los tres arranques --- la forma anterior a
    2026-10-21 --- los tres terminaban en el MISMO `w`: medido sobre la
    corrida completa, lunas 0.999261, rollo 0.439340 y toroide 0.999966,
    identico para los tres. La figura mostraba tres caminos colapsando en un
    punto.
    """
    c = _caminata_chica(notebook, tiny)
    arranques = [c["caminos"][toy_run._nombre_del_arranque(cl)][str(d)]
                 for cl, d in toy_run.CAMINOS_W]
    iniciales = [k["w"][0] for k in arranques]
    assert len(set(iniciales)) > 1, (
        f"los tres arrancan en el mismo w = {iniciales}; si las fronteras no "
        "se distinguen, la figura compara tres copias de un camino")


def test_la_frontera_exacta_VIAJA_aunque_el_arranque_sea_una_bolsa(notebook, tiny):
    """El punto de empate se sigue calculando, y por que importa que viaje.

    Hasta el 2026-10-22 el arranque ERA ese punto, asi que `w` arrancaba en
    ~0 y este test lo acotaba en `< 1e-2`. Ahora arranca una bolsa real, que
    no esta sobre el empate: medido sobre `lunas` a escala completa, el triple
    arranca en 0.0618 contra los 0.0005 del punto fabricado. Esa cota ya no es
    una propiedad de la Ec. (24), es una propiedad de donde cayo una bolsa.

    Lo que SI se afirma: las tres fronteras exactas siguen viajando, y con
    ellas el residuo del empate triple. Son la referencia contra la que la
    figura dibuja cuanto le falto a la bolsa real --- la lectura que se perdia
    si el punto fabricado simplemente desaparecia.
    """
    c = _caminata_chica(notebook, tiny)
    assert set(c["fronteras"]) == set(toy_run.ARRANQUES_W), sorted(c["fronteras"])
    for nombre, punto in c["fronteras"].items():
        assert punto.ndim == 1 and punto.shape[0] > 0, (nombre, punto.shape)
    assert "residuoDelEmpateTriple" in c
    # Y que bolsa camino en cada arranque, en la numeracion del BAGSET.
    assert set(c["bolsaDelBlanco"]) == set(toy_run.ARRANQUES_W)
    assert set(c["eleccionRelajada"]) == set(toy_run.ARRANQUES_W)


def test_w_inicial_y_final_viajan_por_arranque_y_por_destino(notebook, tiny):
    c = _caminata_chica(notebook, tiny)
    suyo = {toy_run._nombre_del_arranque(cl): {str(d)}
            for cl, d in toy_run.CAMINOS_W}
    for mapa in ("wInicial", "wFinal"):
        assert set(c[mapa]) == set(toy_run.ARRANQUES_W), mapa
        for arranque, por_destino in c[mapa].items():
            assert set(por_destino) == suyo[arranque], (mapa, arranque)
            for clave, valor in por_destino.items():
                esperado = c["caminos"][arranque][clave]["w"][
                    0 if mapa == "wInicial" else -1]
                assert valor == float(esperado), (mapa, arranque, clave)
