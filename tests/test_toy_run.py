"""What `toy_run.py` claims, and the red each claim can still reach.

Every test here is written so that breaking the thing it names turns it red. The
two habits that make that true rather than hoped for:

- **Never an expression against itself.** A watchman test that compared
  `toy_run.watchman(...)` with `harness.target_accuracy(model, source_valid)`
  would be comparing one call with the same call. So each test first establishes
  that the two readings it is about to distinguish ACTUALLY DIFFER on the
  fixture, and only then asserts which one the code took. Without that
  precondition the assertion passes on a fixture where both answers are equal,
  and the test is green while measuring nothing.
- **Mutate, then prove the mutation landed.** Where a claim is about a branch,
  the test perturbs the thing the branch reads and asserts the perturbation
  moved what it was supposed to move, before believing the result.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import math

import numpy as np
import pytest
import torch

from MIL_CREDA_Benchmark import (bags, config, harness, toy_bags, toy_run,
                                 toy_tables, wiring)
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

    It is a `Scale` like any other, so it carries the same stamp machinery and
    is subject to the same `REPETITION_FIELDS` rule; it is a third reduction, not
    a special case.

    **It lowers `instances_per_bag`, which is a PROGRAMME field, and no pilot
    may.** `REPETITION_FIELDS` names what a reduction of the real run may touch,
    and the bag size is deliberately not among them: a bag holds 30 instances at
    `PILOT` and at `FULL` so an epoch processes the same material at both. This
    fixture is not a pilot of anything -- it exists so a test that needs a real
    run pays seconds -- and at 30 instances over 81 bags it would need
    `points_per_domain` at 2586 or above, which is the pilot's own material and
    minutes per test. So it declares 6 instances over 810 points.

    **810 and not 540, and the cut is what moved it.** 540 funded 6 instances
    over 36 bags with room to spare; over 81 bags it does not, and the refusal
    is measured rather than feared: at 540 the smallest quantile band over the
    thirteen seeds either scale draws comes back at 5 and
    `_instances_per_bag` refuses on 6 of the 39 (family, seed) pairs, `toroide`
    first. At 810 -- 81 bags at the 10 points per bag 540 gave 36 -- the
    smallest band is 8 over all 39.

    `test_the_pilot_and_the_full_scale_differ_in_repetition_counts_and_
    nothing_else` still holds `PILOT` against `FULL` on every field.

    **It does NOT lower `steps_2`, and that is the cost of the phase-2
    criterion being in passes.** A pass of this programme is
    `toy_run.STEPS_PER_EPOCH_DECLARED` steps at every scale -- the bag
    programme is not a repetition field -- so the six steps this fixture used
    to declare were less than one pass, and `adapt_phase_two` refuses a budget
    holding fewer than `MIN_EPOCHS_2` of them rather than running something
    that can never choose and never restores. The budget is the pilot's own,
    which is nine passes plus the full scale's own unread tail; what a run here
    costs instead is bounded by `instances_per_bag`, which this fixture does
    lower.
    """
    # `epochs_2` is MIN_EPOCHS_2 + 1, the same floor `PILOT` derives: enough
    # passes for the criterion to become legal and then choose. Below it
    # `adapt_phase_two` refuses, which is what the pilot's own paragraph says.
    return dataclasses.replace(toy_run.PILOT, name="test", epochs_1=2,
                               epochs_2=toy_run.MIN_EPOCHS_2 + 1, patience_1=1,
                               trials_1=1, trials_2=1, points_per_domain=810,
                               instances_per_bag=6)


@pytest.fixture(scope="module")
def familia(tiny) -> toy_bags.Familia:
    return toy_run.build_family("lunas", 0, tiny)


@pytest.fixture(scope="module")
def cell(tiny, familia) -> dict:
    return toy_run.run_cell(familia, "batchnorm", 0, tiny,
                            toy_run.declared_hyper(tiny), DEVICE)


def _prepared(familia, tiny, arm_id=ADAPTED, seed=0):
    """A phase-1-trained arm of `arm_id`, with theta_t copied and the source
    frozen: the state every phase-2 claim is about.
    """
    material = toy_run.material_for(familia, seed, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, source_valid, _ = bags.roles(material["source"])
        _, target_valid, _ = bags.roles(material["target"])
        model = toy_run.build_arm(arm_id, material, tiny, hyper, DEVICE, seed)
        generator = torch.Generator().manual_seed(seed + 9973)
        phase_one = toy_run.train_phase_one(model, source_train, source_valid,
                                           DEVICE, tiny, hyper, generator)
        toy_run.copy_source_to_target(model)
        yield_payload = {
            "model": model, "material": material, "hyper": hyper,
            "source_train": source_train, "source_valid": source_valid,
            "target_valid": target_valid, "phase_one": phase_one,
        }
    return yield_payload


def _curva_de_fase_dos(familia, scale, seed=0, backbone="mlp"):
    """Phase 2's per-step record, built the way `run_cell` builds it.

    The freeze is included deliberately: `run_cell` loads the shared phase-1
    state, freezes everything that is not `target_encoder.` and only then
    steps, so a curve read without it would be a different phase 2 from the one
    the record carries.
    """
    material = toy_run.material_for(familia, seed, scale, backbone)
    hyper = toy_run.declared_hyper(scale)
    with toy_run.material_bound(familia, scale, backbone, hyper):
        source_train, source_valid, _ = bags.roles(material["source"])
        model = toy_run.build_arm(ADAPTED, material, scale, hyper, DEVICE, seed)
        generator = torch.Generator().manual_seed(seed + 9973)
        toy_run.train_phase_one(model, source_train, source_valid, DEVICE,
                                scale, hyper, generator)
        toy_run.copy_source_to_target(model)
        for nombre, parametro in model.named_parameters():
            if not nombre.startswith("target_encoder."):
                parametro.requires_grad_(False)
        paso = torch.Generator().manual_seed(seed + 4242)
        return toy_run.adapt_phase_two(model, source_train, source_valid,
                                       DEVICE, scale, hyper, paso)["curve"]


# ------------------------------------------------------------------ the watchman

def test_the_watchman_reads_the_source_valid_split_through_the_target_encoder(
        familia, tiny):
    """The criterion is head o theta_t over the SOURCE valid role.

    Two halves, and each needs its own discrimination to be reachable:

    - THROUGH THETA_T: perturbing theta_t has to move the watchman while the
      source reading (`harness.accuracy`, which is `Arm.forward`) stays put. If
      the watchman went through theta_s instead, the first assertion fails.
    - OVER THE SOURCE SPLIT: the watchman has to equal the target-encoder
      reading of the SOURCE valid role and differ from the same reading of the
      TARGET's own valid role. The test refuses to run unless those two differ
      on this fixture, because where they agree the assertion distinguishes
      nothing.
    """
    # `toroide` and not `lunas`: the two valid splits have to READ DIFFERENTLY
    # for this test to distinguish them at all, and with one valid bag per class
    # every reading is a multiple of 1/3, so they collide often. Measured over
    # three families, two backbones and three seeds, `toroide` separates them on
    # all six of its cells and `lunas` on two of six -- so the fixture is chosen
    # by that measurement, and the assertion below replaces the skip it used to
    # take when they happened to agree.
    familia = toy_run.build_family("toroide", 0, tiny)
    payload = _prepared(familia, tiny)
    model = payload["model"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        before = toy_run.watchman(model, payload["source_valid"], DEVICE)
        source_side = harness.accuracy(model, payload["source_valid"], DEVICE)
        on_target_split = harness.target_accuracy(model, payload["target_valid"],
                                                  DEVICE)
        assert not math.isclose(before, on_target_split), (
            "the two splits read the same on this fixture, so the assertion "
            "below would distinguish nothing")
        assert math.isclose(before,
                            harness.target_accuracy(model,
                                                    payload["source_valid"],
                                                    DEVICE))
        assert not math.isclose(before, on_target_split)

        # Now move theta_t and nothing else, and prove the move landed.
        moved_by = 0.0
        with torch.no_grad():
            for parameter in model.target_encoder.parameters():
                delta = torch.randn_like(parameter) * 2.5
                parameter.add_(delta)
                moved_by += float(delta.norm())
        assert moved_by > 0.0, "the perturbation of theta_t did not land"
        toy_run.set_phase_two_modes(model)

        after = toy_run.watchman(model, payload["source_valid"], DEVICE)
        after_source_side = harness.accuracy(model, payload["source_valid"],
                                             DEVICE)
    assert after != before, "the watchman did not move when theta_t moved"
    assert math.isclose(after_source_side, source_side), (
        "the source reading moved when only theta_t was perturbed")


def test_the_call_site_hands_the_watchman_the_source_valid_role(familia, tiny,
                                                                monkeypatch):
    """The CALL SITE's split, read off what it actually passed.

    The watchman reads zero target labels, and the half of that claim the test
    above cannot see is WHICH dataset `run_cell` hands it: `watchman` is correct
    about whatever role it is given, so handing it the target's own valid role
    would read as a clean source-only criterion and be a criterion that saw the
    target.

    This used to be probed through the floor arm, where theta_t fell back to
    theta_s and the watchman therefore had to equal phase 1's restored reading.
    The floor is retired, so the call site is read directly: `watchman` is
    replaced by a spy that records the dataset and then calls the real one, and
    the material the spy was handed is compared -- tensor by tensor -- against
    the source valid role and against the target's. The precondition that the
    two roles actually hold different points is asserted first; where they
    agreed the comparison would distinguish nothing.
    """
    seen = []
    real = toy_run.watchman

    def spy(model, dataset, device):
        seen.append(dataset)
        return real(model, dataset, device)

    monkeypatch.setattr(toy_run, "watchman", spy)
    toy_run.run_cell(familia, "batchnorm", 0, tiny,
                     toy_run.declared_hyper(tiny), DEVICE)
    assert seen, "the watchman was never called; nothing below measures anything"

    material = toy_run.material_for(familia, 0, tiny, "mlp")
    with toy_run.material_bound(familia, tiny, "mlp"):
        _, source_valid, _ = bags.roles(material["source"])
        _, target_valid, _ = bags.roles(material["target"])

        def stacked(dataset):
            return torch.stack([dataset[i][0] for i in range(len(dataset))])

        on_source, on_target = stacked(source_valid), stacked(target_valid)
        handed = stacked(seen[0])

    assert not torch.equal(on_source, on_target), (
        "the source and target valid roles hold identical material on this "
        "fixture, so the assertions below could not tell them apart")
    assert torch.equal(handed, on_source), (
        "the call site handed the watchman material that is not the SOURCE "
        "valid role")
    assert not torch.equal(handed, on_target)


def test_cef_is_maximized_and_its_argmin_is_not_the_epoch_chosen(
        familia, tiny, monkeypatch):
    """Phase 2 keeps the PASS where `cef` is LARGEST.

    The fixture scripts `cef`'s own sequence so the argmax, the argmin and the
    last pass are three DIFFERENT passes -- asserted before the run, because a
    sequence whose argmax and argmin coincide could not tell a maximization
    from a minimization. Flip the comparison in `adapt_phase_two` and
    `bestEpoch` becomes the argmin: red.

    **It was the watchman's MINIMUM until 2026-10-19**, and this test is the
    inversion rather than a new one: the scripted sequence is the same, the
    guard is the same, and what changed is which extreme is kept and which
    function is scripted. The watchman is still read every epoch and is
    asserted NOT to have chosen.

    `epochs_2` is raised locally so the nine scripted passes all run: the
    fixture's own ceiling is `MIN_EPOCHS_2 + 1`, which is the floor at which
    the criterion can choose at all and leaves no room for three distinct
    extremes.
    """
    scale = dataclasses.replace(tiny, epochs_2=9)
    payload = _prepared(familia, scale)
    model = payload["model"]
    per_epoch = [0.9, 0.8, 0.2, 0.95, 0.6, 0.7, 0.75, 0.78, 0.79]
    # `MIN_EPOCHS_2` passes of guard, so the earlier entries are outside the
    # window by construction and the three extremes below all sit inside it.
    window = per_epoch[toy_run.MIN_EPOCHS_2 - 1:]
    assert len(per_epoch) >= scale.epochs_2
    argmin = per_epoch.index(min(window))
    argmax = per_epoch.index(max(window))
    assert argmin != argmax, "the scripted sequence cannot distinguish max from min"
    assert argmax != len(per_epoch) - 1, "the argmax is the last pass; useless"

    calls = {"n": 0}

    def scripted_cef(model_, device):
        value = per_epoch[min(calls["n"], len(per_epoch) - 1)]
        calls["n"] += 1
        return value

    monkeypatch.setattr(toy_run, "effective_classes", scripted_cef)
    with toy_run.material_bound(familia, scale, "mlp"):
        generator = torch.Generator().manual_seed(1)
        phase_two = toy_run.adapt_phase_two(model, payload["source_train"],
                                            payload["source_valid"], DEVICE,
                                            scale, payload["hyper"], generator)
    assert calls["n"] > 0, "the scripted cef was never called"
    assert calls["n"] == scale.epochs_2, (
        calls["n"], scale.epochs_2,
        "cef is read once per pass, so the call count is the epoch count")
    assert phase_two["bestCef"] == max(window)
    assert phase_two["bestEpoch"] == argmax + 1
    assert phase_two["bestStep"] == (argmax + 1) * phase_two["stepsPerEpoch"]
    assert phase_two["bestEpoch"] != argmin + 1
    # And the watchman chose NOTHING: it is read every epoch and recorded, so
    # the row of the chosen epoch carries it, but the epoch chosen is cef's.
    vigias = [row["watchman"] for row in phase_two["epochPass"]]
    assert len(vigias) == scale.epochs_2 and all(v is not None for v in vigias)
    assert phase_two["bestWatchman"] == vigias[argmax]


def test_phase_two_restores_the_weights_of_the_best_step_and_not_the_last(
        familia, tiny, monkeypatch):
    """The state at exit is the state at `cef`'s argmax, not the state at the end.

    The scripted watchman snapshots theta_t at every step, so the test can
    compare the exit state against the step it should have kept AND against the
    step it must not have kept. Both are needed: equality with the chosen step
    alone would pass on a run where theta_t never moved.

    **`cef` is scripted here too, and that is what makes the test a test.**
    Until 2026-10-19 the criterion was the watchman and scripting it fixed the
    chosen epoch; with `cef` deciding, a watchman-only script left the choice
    to the REAL `cef` over the fixture's `MIN_EPOCHS_2 + 1` passes -- so the
    only epochs it could choose were the guard's first and the LAST, and on the
    last one `kept` and `last_step` are the same snapshot and the control above
    dies with a message about theta_t not moving. It passed because `cef`
    happened to peak on the earlier of the two.

    So the sequence below has its argmax strictly inside the window and
    strictly before the end, asserted rather than assumed, and `epochs_2` is
    raised locally so all of it runs.
    """
    scale = dataclasses.replace(tiny, epochs_2=9)
    payload = _prepared(familia, scale)
    model = payload["model"]
    per_epoch = [0.9, 0.8, 0.2, 0.95, 0.6, 0.7, 0.75, 0.78, 0.79]
    cef_per_epoch = [0.1, 0.2, 0.3, 2.9, 0.4, 0.5, 0.6, 0.7, 0.8]
    _ventana = cef_per_epoch[toy_run.MIN_EPOCHS_2 - 1:]
    _argmax = cef_per_epoch.index(max(_ventana))
    assert _argmax >= toy_run.MIN_EPOCHS_2 - 1, _argmax
    assert _argmax != len(cef_per_epoch) - 1, (
        "the argmax is the last pass, so `kept` and `last_step` would be the "
        "same snapshot and the control below could not fail")
    snapshots: list[torch.Tensor] = []
    calls = {"n": 0}
    cef_calls = {"n": 0}

    def scripted_cef(model_, device):
        value = cef_per_epoch[min(cef_calls["n"], len(cef_per_epoch) - 1)]
        cef_calls["n"] += 1
        return value

    def scripted_watchman(model_, dataset, device):
        snapshots.append(toy_run._parameter_vector(model_.target_encoder).clone())
        epoch = min(calls["n"] // toy_run.STEPS_PER_EPOCH_DECLARED,
                    len(per_epoch) - 1)
        calls["n"] += 1
        return per_epoch[epoch]

    monkeypatch.setattr(toy_run, "watchman", scripted_watchman)
    monkeypatch.setattr(toy_run, "effective_classes", scripted_cef)
    with toy_run.material_bound(familia, scale, "mlp"):
        generator = torch.Generator().manual_seed(1)
        phase_two = toy_run.adapt_phase_two(model, payload["source_train"],
                                            payload["source_valid"], DEVICE,
                                            scale, payload["hyper"], generator)
        final = toy_run._parameter_vector(model.target_encoder).clone()

    best = phase_two["bestStep"]
    assert phase_two["bestEpoch"] == _argmax + 1, (
        phase_two["bestEpoch"], _argmax + 1)
    assert best >= toy_run.MIN_EPOCHS_2 * phase_two["stepsPerEpoch"]
    # `adapt_phase_two` reads the watchman once per step AND once more after the
    # restore, so the snapshot list is one longer than the run: the last STEP's
    # state is at `steps - 1` and the final entry is already the restored one.
    assert len(snapshots) == phase_two["steps"] + 1, (len(snapshots),
                                                      phase_two["steps"])
    kept = snapshots[best - 1]
    last_step = snapshots[phase_two["steps"] - 1]
    assert float((kept - last_step).norm()) > 0.0, (
        "theta_t never moved between the best step and the last; this fixture "
        "cannot tell a restore from a no-op")
    assert float((final - kept).norm()) == 0.0, "the best step was not restored"
    assert float((final - last_step).norm()) > 0.0, (
        "the LAST step was kept instead of the best one")


# ----------------------------------------------------------------- the freeze

def test_phase_one_freezes_theta_s_bit_identically_across_phase_two(cell):
    """Every arm's source parameters AND buffers are unmoved by phase 2.

    Both halves, because a BatchNorm encoder left in `train` re-estimates its
    running statistics in the forward: its weights can be bit-identical while it
    is not the same encoder any more. `sourceMoved`/`sourceBuffersMoved` are the
    norms `run_cell` measures across phase 2.
    """
    assert cell["rows"], "the cell produced no rows"
    for arm_id, row in cell["rows"].items():
        assert row["sourceMoved"] == 0.0, (arm_id, row["sourceMoved"])
        assert row["sourceBuffersMoved"] == 0.0, (arm_id,
                                                  row["sourceBuffersMoved"])


def test_phase_one_leaves_every_shared_parameter_unrequiring_grad(familia, tiny):
    """The freeze is on the parameters themselves, not only on which ones the
    optimizer was handed."""
    payload = _prepared(familia, tiny)
    model = payload["model"]
    frozen = {name: p.requires_grad for name, p in model.named_parameters()
              if not name.startswith("target_encoder.")}
    assert frozen, "there are no shared parameters to freeze"
    assert not any(frozen.values()), [n for n, v in frozen.items() if v]
    trainable = [p.requires_grad for p in model.target_encoder.parameters()]
    assert trainable and all(trainable), "theta_t is not trainable"


def test_the_eval_mode_in_phase_two_is_what_keeps_the_source_buffers_still(
        familia, tiny, monkeypatch):
    """`set_phase_two_modes`'s `eval()` is load-bearing, not decoration.

    Disable it and the source encoder's BatchNorm buffers move during phase 2 --
    which is precisely the failure `no_grad` alone would not catch. This test is
    the reachable red for the buffer half of the freeze above.
    """
    payload = _prepared(familia, tiny, arm_id=ADAPTED)
    model = payload["model"]
    before = toy_run._buffer_vector(model.encoder).clone()
    monkeypatch.setattr(toy_run, "set_phase_two_modes", lambda m: m.train())
    with toy_run.material_bound(familia, tiny, "mlp"):
        generator = torch.Generator().manual_seed(1)
        toy_run.adapt_phase_two(model, payload["source_train"],
                                payload["source_valid"], DEVICE, tiny,
                                payload["hyper"], generator)
    after = toy_run._buffer_vector(model.encoder)
    assert float((after - before).norm()) > 0.0, (
        "the source buffers stayed still even with the mode guard disabled; "
        "this fixture cannot show what the guard buys")


def test_theta_t_starts_bit_identical_to_theta_s(familia, tiny):
    """The one copy carries the buffers too.

    `Arm.sync_target_from_source` deliberately carries parameters only -- that is
    right for a repeated leash and wrong for CREDA's single
    `clone_for_target`, because a fresh BatchNorm starts at `running_mean = 0`
    and would not be the encoder phase 1 produced.
    """
    payload = _prepared(familia, tiny)
    model = payload["model"]
    assert float((toy_run._parameter_vector(model.encoder)
                  - toy_run._parameter_vector(model.target_encoder)).norm()) == 0.0
    assert float((toy_run._buffer_vector(model.encoder)
                  - toy_run._buffer_vector(model.target_encoder)).norm()) == 0.0


def test_the_buffer_copy_is_what_makes_the_start_identical(familia, tiny):
    """Without the buffer half of `copy_source_to_target`, theta_t does not
    start as theta_s -- the reachable red for the test above."""
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    hyper = toy_run.declared_hyper(tiny)
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, source_valid, _ = bags.roles(material["source"])
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
        generator = torch.Generator().manual_seed(9973)
        toy_run.train_phase_one(model, source_train, source_valid, DEVICE, tiny,
                                hyper, generator)
        model.sync_target_from_source()          # parameters only, on purpose
        gap = float((toy_run._buffer_vector(model.encoder)
                     - toy_run._buffer_vector(model.target_encoder)).norm())
    assert gap > 0.0, (
        "the buffers already agreed after a parameter-only sync; this backbone "
        "carries no buffers and the claim is untestable here")


# ------------------------------------------------------------------- the scales

def test_the_pilot_and_the_full_scale_differ_in_repetition_counts_and_nothing_else():
    """Two knobs separate a pilot from the run it rehearses, and
    `REPETITION_FIELDS` names them.

    Three assertions, because each catches a different way this goes wrong: a
    programme field drifting between the two, a repetition field left at the
    full value, and a name in `REPETITION_FIELDS` that is not a field at all --
    which would silently exempt nothing and look like a pass.
    """
    names = {f.name for f in dataclasses.fields(toy_run.Scale)} - {"name"}
    assert toy_run.REPETITION_FIELDS <= names, (
        toy_run.REPETITION_FIELDS - names)
    programme = names - toy_run.REPETITION_FIELDS
    assert programme, "every field is a repetition field; nothing is pinned"
    for field in sorted(programme):
        assert getattr(toy_run.PILOT, field) == getattr(toy_run.FULL, field), (
            f"{field} differs between the pilot and the full scale")
    for field in sorted(toy_run.REPETITION_FIELDS):
        full, pilot = getattr(toy_run.FULL, field), getattr(toy_run.PILOT, field)
        full_size = len(full) if isinstance(full, tuple) else full
        pilot_size = len(pilot) if isinstance(pilot, tuple) else pilot
        assert pilot_size < full_size, (
            f"{field} was not lowered by the pilot ({pilot_size} vs {full_size})")


def test_the_row_names_are_credas_own_spelling():
    """Los rótulos de fila se LEEN de `CREDA.toy_experiment`, no se escriben acá.

    El cuadro animado los imprime en el rótulo de cada fila y las tablas los
    imprimen en su columna de arquitectura, así que una diferencia de una letra
    entre los dos documentos se ve como dos arquitecturas distintas. Son
    minúsculas en «sin normalización» y mayúsculas internas en «BatchNorm» y
    «LayerNorm», que es la ortografía de CREDA y no una elección de este
    paquete: por eso se compara contra su diccionario entero, clave por clave,
    en vez de afirmarse en una frase.
    """
    from CREDA import toy_experiment as creda

    assert dict(toy_tables.NOMBRE_FILA) == dict(creda.NOMBRE_FILA), (
        toy_tables.NOMBRE_FILA, creda.NOMBRE_FILA)
    # Y el control de que la comparación mira algo: el rótulo del caso que una
    # mayúscula de más arruinaría está, y está en minúsculas.
    assert toy_tables.NOMBRE_FILA["ninguna"] == "sin normalización", (
        toy_tables.NOMBRE_FILA["ninguna"])


def test_the_full_scale_is_credas_own():
    """The declared scale is read off `CREDA.toy_experiment`, not chosen here.

    **One field of it is not, and the departure is asserted rather than
    dropped.** `points_per_domain` used to be `toy_bags.PUNTOS_POR_DOMINIO` --
    CREDA's own 1800 -- and 27 bags per class of 30 declared instances cannot
    be funded by it. Deleting the assertion would leave the departure
    invisible, so what is asserted instead is that the full scale draws MORE
    than CREDA's figure does, that CREDA's own number would be refused by
    `_instances_per_bag` at this cut, and that the figure's constant is
    untouched at 1800. The reachable red is a silent return to the old tie.
    """
    from CREDA import toy_experiment as creda

    assert toy_run.FULL.epochs_1 == creda.EPOCAS_1
    # THE TWO CEILINGS, IN THE SAME UNIT, which is what this pin could not
    # assert before 2026-10-19: it read `FULL.steps_2 == creda.PASOS_2`, two
    # STEP ceilings, and two equal step budgets buy DIFFERENT numbers of passes
    # whenever the two passes differ -- which they do, 16 steps here against
    # CREDA's 9, and in CREDA the lot is searched so its own three families
    # bought 22.2, 11.8 and 11.8. Equal epochs is the tie that means something.
    assert toy_run.FULL.epochs_2 == creda.EPOCAS_2
    assert toy_run.FULL.lr_1 == creda.LR_1
    assert toy_run.FULL.lr_2 == creda.LR_2
    assert toy_run.FULL.patience_1 == creda.PACIENCIA_1
    # `patience_2` and `min_steps_2` are GONE from the scale, with
    # `PACIENCIA_2` and `MIN_PASOS_2` on CREDA's side. Asserted as absences so
    # that reintroducing either on one side alone reddens here rather than
    # quietly restating a criterion the other side no longer has.
    for retirado in ("patience_2", "min_steps_2", "steps_2"):
        assert not hasattr(toy_run.FULL, retirado), retirado
    for retirado in ("PACIENCIA_2", "MIN_PASOS_2", "PASOS_2",
                     "PACIENCIA_2_EPOCAS"):
        assert not hasattr(creda, retirado), retirado
    # The guard is DECLARED on both sides and the two values DIFFER. What is
    # pinned is that each is declared in epochs and that both travel in the
    # record; equalizing them is a decision about the guard, not about the
    # unit, and this pin does not take it.
    assert toy_run.MIN_EPOCHS_2 == 3 and creda.MIN_EPOCAS_2 == 5
    assert (toy_run.PHASE_TWO_CRITERION["minEpochs2InCreda"]
            == creda.MIN_EPOCAS_2)
    assert list(toy_run.FULL.search_seeds) == list(creda.SEMILLAS_BUSQUEDA)
    assert list(toy_run.FULL.report_seeds) == list(creda.SEMILLAS_REPORTE)
    assert toy_run.FULL.trials_1 == creda.TRIALS_1
    assert toy_run.FULL.trials_2 == creda.TRIALS_2
    assert toy_run.FULL.frac_train == creda.FRAC_TRAIN
    assert toy_run.FULL.frac_valid == creda.FRAC_VALID

    # The one field that departed, and why. 1800 over 36 bags is the 50 points
    # per bag this scale has always drawn; 81 bags at that density is 4050.
    assert toy_bags.PUNTOS_POR_DOMINIO == 1800, (
        "the figure's own constant moved; it is CREDA's number and this cut "
        "does not touch it")
    assert toy_run.FULL.points_per_domain == 4050
    assert toy_run.FULL.points_per_domain > toy_bags.PUNTOS_POR_DOMINIO
    assert (toy_run.FULL.points_per_domain
            == toy_bags.PUNTOS_POR_DOMINIO // 36 * toy_run.FULL.bags_total)
    # And the pole that makes the departure a measurement rather than a
    # preference: CREDA's own number cannot fund this cut at all.
    starved = dataclasses.replace(toy_run.FULL,
                                  points_per_domain=toy_bags.PUNTOS_POR_DOMINIO)
    familia = toy_run.build_family("lunas", 0, starved)
    with pytest.raises(ValueError, match="raise `points_per_domain`"):
        toy_run._instances_per_bag(familia, starved)


def test_every_record_carries_the_scale_it_was_measured_at(cell, tiny):
    """A pilot number cannot travel as a result if every record says so."""
    stamp = tiny.stamp()
    assert stamp["isPilot"] is True
    assert cell["scale"] == stamp
    for arm_id, row in cell["rows"].items():
        assert row["scale"] == stamp, arm_id
        assert row["scale"]["name"] == "test"
        assert row["scale"]["isPilot"] is True


def test_the_full_scale_is_the_one_record_that_is_not_a_pilot():
    assert toy_run.FULL.stamp()["isPilot"] is False
    assert toy_run.PILOT.stamp()["isPilot"] is True


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
        familia, tiny):
    """What the corrected scope BUYS, measured on the mechanism and not on the
    bandwidth.

    Two readings, and the second is the one that matters:

    - **In band.** At the frozen bandwidth the within-bag kernel reads
      `exp(-1/(2m^2))` for the multiplier `m`, which lands inside `SIGMA_BAND`
      for every multiplier the grid offers. That is an identity rather than a
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
    measured. The search ranges gamma over `(0.0, 1.0)` and chose 0.6028 on the
    pilot; this test supplies a non-zero gamma and asserts it reached the arm,
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
    multiplier = toy_run.SIGMA_MULTIPLIERS[0]
    with toy_run.material_bound(familia, tiny, "mlp", hyper):
        source_train, source_valid, _ = bags.roles(source)
        model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
        assert model.attention_gamma > 0.0, (
            "the consensus term is the only path the bandwidth takes into "
            "Eq. (16); at gamma = 0 this test compares one number with itself")
        generator = torch.Generator().manual_seed(9973)
        toy_run.train_phase_one(model, source_train, source_valid, DEVICE,
                                tiny, hyper, generator)
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


def test_the_frozen_sigma_is_immune_to_anything_injected_into_the_target(
        familia, tiny):
    """Source only, which is the property that makes it worth deriving.

    CREDA's own reason, in `sigma_de_la_fuente`: nothing injected into the
    target -- intruders, relocation, contamination -- may move the unit the term
    is measured in, or damage entering through the class block stops being
    distinguishable from damage entering through the ruler.

    Measured as BIT-IDENTICAL and not as approximately equal: the statistic
    reads no target row at all, so any difference would be a leak rather than
    noise. The pole is in the same test -- the same perturbation DOES move the
    capped reading's target-side counterpart, so the fixture really was
    disturbed.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    source = material["source"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        model = toy_run.build_arm(ADAPTED, material, tiny,
                                  toy_run.declared_hyper(tiny), DEVICE, 0)
        before = toy_run.frozen_source_sigma(model, source, source.train_idx,
                                             DEVICE)

    # The injection: every target point moved, which is the strongest thing
    # that can be done to the target without touching the source.
    disturbed = dataclasses.replace(familia, Xt=familia.Xt + 7.0)
    assert not np.allclose(disturbed.Xt, familia.Xt), (
        "the fixture did not change; this test would pass measuring nothing")
    disturbed_material = toy_run.material_for(disturbed, 0, tiny, "mlp")
    with toy_run.material_bound(disturbed, tiny, "mlp"):
        disturbed_model = toy_run.build_arm(ADAPTED, disturbed_material, tiny,
                                            toy_run.declared_hyper(tiny),
                                            DEVICE, 0)
        after = toy_run.frozen_source_sigma(
            disturbed_model, disturbed_material["source"],
            disturbed_material["source"].train_idx, DEVICE)
    assert after["sigma"] == before["sigma"], (after["sigma"], before["sigma"])
    assert after["readsTargetRows"] is False

    # **And the same claim through the CALL SITE, which the two readings above
    # cannot reach.** They hand `frozen_source_sigma` the source bagset
    # themselves, so a `run_cell` that passed the TARGET bagset instead would
    # break the immunity and leave both assertions green -- the statistic would
    # still be source-only, read over the wrong bags. Measured end to end: the
    # value the RECORD carries is bit-identical with the target disturbed.
    hyper = toy_run.declared_hyper(tiny)
    clean = toy_run.run_cell(familia, "batchnorm", 0, tiny, hyper, DEVICE)
    dirty = toy_run.run_cell(disturbed, "batchnorm", 0, tiny, hyper, DEVICE)
    assert (dirty["sigma"]["frozenSigma"] == clean["sigma"]["frozenSigma"]), (
        dirty["sigma"]["frozenSigma"], clean["sigma"]["frozenSigma"])
    assert (dirty["sigma"]["phaseTwoSigma"]
            == clean["sigma"]["phaseTwoSigma"])
    # The control that the disturbance reached the run at all, so the equality
    # above is immunity and not an injection that never landed. It is read off
    # the LATENT displacement and the watchman and NOT off `targetAccuracy`:
    # measured, the accuracy comes back identical at 0.3333 on both runs --
    # with 18 evaluation bags and three classes a shift of this size lands both
    # at exactly chance, so the published column cannot register it. That is
    # the instrument's coarseness, which is what T1 narrowed and did not close,
    # and a control read there would pass by comparing one number with itself.
    clean_row, dirty_row = clean["rows"][ADAPTED], dirty["rows"][ADAPTED]
    assert (dirty_row["latentTargetMoved"]
            != clean_row["latentTargetMoved"]), (
        "the disturbed target did not move theta_t's own positions; the "
        "injection never reached the run")
    # The DIRECTION, and no longer a magnitude threshold. This read
    # `> 2 * clean` while phase 2 ran six steps on this fixture; with the
    # criterion in passes the budget is the pilot's own and theta_t travels far
    # further in BOTH arms, so the ratio compresses -- measured here, 9.2438
    # against 5.6698, a ratio of 1.63. The threshold was calibrated against a
    # budget that no longer exists, and keeping it would have been a claim about
    # the injection's size made at a scale nothing runs at; the claim that
    # survives the change is that the disturbance moves theta_t FURTHER, on a
    # deterministic run where the inequality above already settles that it moved
    # at all.
    assert dirty_row["latentTargetMoved"] > clean_row["latentTargetMoved"], (
        clean_row["latentTargetMoved"], dirty_row["latentTargetMoved"])
    # The CRITERION'S OWN DECISION, and no longer the watchman's value. This
    # read `watchmanBest != `, and measured at the pilot's budget that field
    # cannot distinguish the two runs: the watchman is MINIMIZED over the
    # source's fifteen selection bags, so its floor is 5/15 = 1/3, and over 152
    # steps both arms reach exactly 0.3333333333333333 -- `watchmanFinal` and
    # `watchmanRestored` too. It is the same instrument coarseness the comment
    # above records for `targetAccuracy`, one column over, and a control read
    # there would now pass by comparing one number with itself.
    #
    # What the disturbance does move is which PASS the criterion chose and
    # whether it stopped at all: measured, `phase2BestEpoch` 9 against 3 and
    # `phase2Steps` 152 against 128 -- the clean run reaches the ceiling and the
    # disturbed one is ended by the patience. That is a stronger control than
    # the value was, because it is the criterion itself answering differently.
    assert dirty_row["phase2BestEpoch"] != clean_row["phase2BestEpoch"], (
        clean_row["phase2BestEpoch"], dirty_row["phase2BestEpoch"])
    # And the other side of the same fact: the SUPERVISED term reads the source
    # alone, so it comes back bit-identical exactly as the bandwidth does.
    #
    # **Asserted PER STEP, because the cell's own field cannot carry it, and
    # that correction is what the role permutation exposed.** This assertion
    # used to read `dirty_row["supervised"] == clean_row["supervised"]`, and
    # `row["supervised"]` is the MEAN over the phase-2 steps that ran
    # (`run_cell`: `_mean([abs(p["supervised"]) for p in curve])`). How many ran
    # is decided by the watchman's early stop, and the watchman reads the
    # target -- which this test has just asserted MOVED. So the two runs stop at
    # different steps and the two means are averages of different numbers of
    # terms: measured here, `phase2Steps` comes back 5 against 6 and the means
    # differ in the third decimal. The equality held before only because both
    # runs happened to stop on the same step; it was never a reading of the
    # term's provenance.
    #
    # The per-step series is the claim itself: with the target moved by +7.0,
    # every step the two runs share reads the SAME supervised value to the last
    # bit, while the global contribution differs on every one of them -- which
    # is the control that the disturbance reached the step and the equality is
    # not two runs doing nothing.
    # **INVERTED on 2026-10-19, and the inversion is the finding.** It demanded
    # `phase2Steps` DIFFER between the two runs, and that was reading the
    # PATIENCE: the two runs stopped at different steps because the stop
    # depended on the target, which the disturbance moved. With the patience
    # gone both runs walk `epochs_2` whole, so equal step counts are now
    # guaranteed by construction and a difference would mean the ceiling is not
    # being honoured. What the disturbance moves instead is the epoch CHOSEN,
    # which the assertion above already reads.
    assert dirty_row["phase2Steps"] == clean_row["phase2Steps"], (
        dirty_row["phase2Steps"], clean_row["phase2Steps"],
        "phase 2 has no stop, so both runs must walk the same ceiling")
    assert (dirty_row["phase2Steps"]
            == tiny.epochs_2 * toy_run.STEPS_PER_EPOCH_DECLARED), (
        dirty_row["phase2Steps"], tiny.epochs_2)
    limpia = _curva_de_fase_dos(familia, tiny)
    sucia = _curva_de_fase_dos(disturbed, tiny)
    comunes = min(len(limpia), len(sucia))
    assert comunes > 0, (len(limpia), len(sucia))
    for paso in range(comunes):
        assert limpia[paso]["supervised"] == sucia[paso]["supervised"], (
            f"step {paso}: the supervised term moved with the target, "
            f"{limpia[paso]['supervised']!r} against "
            f"{sucia[paso]['supervised']!r}; Eq. (21) is not source-only here")
    movido = [paso for paso in range(comunes)
              if (limpia[paso]["contributionGlobal"]
                  != sucia[paso]["contributionGlobal"])]
    quieto = sorted(set(range(comunes)) - set(movido))
    # **`len(movido) == comunes` was the claim until 2026-10-18, and r33's
    # Eq. (37) made it false on purpose.** A class the target routing
    # abandoned contributes `l~_c = 1`, a literal: the disturbance cannot move
    # it, which is the whole reason the charge exists -- it makes the
    # abandonment visible in the VALUE rather than letting a shrinking sum
    # read as convergence. So a step whose routing lost classes carries a
    # constant the target cannot reach, and on a step that lost ALL of them the
    # term is exactly 1 whatever the target does.
    #
    # The control is therefore per step and paired, which is stronger than the
    # old blanket count: a step where the disturbance did NOT reach the global
    # term has to be a step where the routing abandoned a class, and the
    # message carries both readings so a different cause cannot hide behind
    # this explanation.
    assert movido, (
        "the disturbance reached the global term on no shared step at all, so "
        "the supervised equality above proves nothing anywhere")
    # **This was `len(movido) == comunes` until 2026-10-18, and it is weaker
    # now. Saying so rather than hiding it.** Measured on this fixture: 106 of
    # 128 shared steps move, 22 do not, and the 22 are three different things.
    #
    #   step 0, 27:  `activeClasses` 2 of 3, and the term reads about 0.336 --
    #                `1/3`, Eq. (37)'s charge for the one abandoned class. The
    #                charge is a literal and the target cannot move it, which
    #                is exactly why it exists.
    #   step 17:     all three active, the global half bit-identical, and
    #                `adaptation` DIFFERENT in its eighth digit -- so the
    #                disturbance reached the step through the local term and
    #                the global half rounded to the same float32.
    #   step 23:     all three active, and global AND adaptation identical.
    #                The disturbance did not reach this step at all: theta_t
    #                absorbed the +7.0 translation to within float32.
    #
    # That last kind is why the blanket count cannot come back: the target
    # encoder is free to absorb a translation, and no per-step claim can forbid
    # it. What survives is the pair below -- the supervised equality holds on
    # EVERY shared step (asserted above, unchanged), and the disturbance
    # demonstrably reached the term on the large majority of them, so the
    # equality is not two runs doing nothing.
    assert len(movido) > 0.75 * comunes, (
        f"the disturbance reached the global term on only {len(movido)} of "
        f"{comunes} shared steps, which is too few for the supervised "
        f"equality above to mean anything; the quiet steps were {quieto}")


def test_phase_two_runs_at_the_multiplier_times_the_frozen_sigma(cell, tiny):
    """What the searched multiplier multiplies is the frozen value, and the
    record says both.

    Three facts the record has to carry for a reader to be able to check this
    without re-running: the frozen value, the multiplier applied to it, and the
    product every arm's phase 2 actually ran at. The product is asserted to be
    the product, and the frozen value is asserted DIFFERENT from the declared
    constant -- `config.KERNEL_SIGMA` is 36.135 and was measured on digits under
    resnet18, so a run that still read it would be running the pre-T2 path.
    """
    sigma = cell["sigma"]
    assert sigma["frozenSigma"] > 0.0
    assert sigma["frozenSigmaMultiplier"] == pytest.approx(
        cell["rows"][ADAPTED]["hyper"].get("sigmaMultiplier",
                                           toy_run.SIGMA_MULTIPLIER_NEUTRAL))
    assert sigma["phaseTwoSigma"] == pytest.approx(
        sigma["frozenSigmaMultiplier"] * sigma["frozenSigma"])
    assert sigma["frozenSigma"] != config.KERNEL_SIGMA
    assert sigma["frozenStatistic"] == "CREDA.toy_experiment.sigma_de_la_fuente"
    # The bootstrap phase 1 pooled at travels beside it, with the ratio, so the
    # one thing this change cannot make identical to CREDA is visible: CREDA's
    # phase 1 needs no bandwidth at all and this one does, because the unit is
    # the bag.
    assert sigma["bootstrapSigma"] > 0.0
    assert sigma["frozenToBootstrapRatio"] == pytest.approx(
        sigma["frozenSigma"] / sigma["bootstrapSigma"])


def test_the_cell_record_carries_the_band_and_the_scope_it_was_corrected_from(
        cell):
    """The band is IN the record, per cell, with the scope this stretch left
    beside it -- so a reader can perform the subtraction instead of taking the
    correction on trust.

    Why the record carries both and not only the one that governs: a number
    that silently moved reads like a number that was always right. The pilot's
    own pre-correction record had 4 of 9 cells saturated at the smallest
    multiplier the grid offers, and nothing in it said which scope produced
    them. Now `frozenSigma` and `frozenPooledSigma` sit side by side with the
    ratio and with what the instance kernel reads at each.

    Each derived field is asserted against what it is derived FROM, never
    against a literal, so the record cannot drift from the reading it echoes.
    """
    sigma = cell["sigma"]
    frozen = sigma["frozenReading"]
    assert sigma["frozenScope"] == toy_run.FROZEN_SIGMA_SCOPE
    assert sigma["frozenSigma"] == frozen["sigma"]
    assert sigma["frozenPooledSigma"] == frozen["pooledSigma"]
    assert sigma["frozenPooledToWithinRatio"] == pytest.approx(
        sigma["frozenPooledSigma"] / sigma["frozenSigma"])

    # In band, and the value is the identity the scope makes hold. The record
    # says so in `bandIsAnIdentityNote` rather than letting a green here read as
    # evidence about the mechanism.
    low, high = toy_run.SIGMA_BAND
    assert low <= sigma["frozenKernelWithinBag"] <= high, sigma
    assert sigma["frozenKernelInBand"] is True
    assert sigma["frozenKernelWithinBag"] == pytest.approx(
        math.exp(-1.0 / (2.0 * sigma["frozenSigmaMultiplier"] ** 2)))
    # The note names the field that DOES carry a reachable red, by its key and
    # not by a word of prose: an identifier is a data contract and a sentence is
    # a reading, and a test that matched the sentence would redden on a rewording
    # and stay green on a note that pointed at nothing.
    assert "weightSpreadGainOverPooled" in frozen["bandIsAnIdentityNote"]
    assert frozen["weightSpreadGainOverPooled"] is not None

    # The multiplier the reading was taken at is the one the cell ran at: a
    # reading at some other bandwidth would describe a run that did not happen.
    assert frozen["multiplierRead"] == pytest.approx(
        sigma["frozenSigmaMultiplier"])
    assert sigma["phaseTwoSigma"] == pytest.approx(
        frozen["multiplierRead"] * frozen["sigma"])

    # The full role, on the trained encoder, and distinct from the pre-phase-1
    # bootstrap reading that sits in the same block under `scales`.
    assert frozen["bagsTruncated"] is False
    assert frozen["withinBagReading"]["bagsRead"] == frozen["bags"]
    assert frozen["withinBagReading"]["bagsRead"] > sigma["scales"]["bagsRead"], (
        "the frozen reading is capped to the same sample as the bootstrap one; "
        "the two answer different questions over different sets")
    assert frozen["sigma"] != sigma["scales"]["withinBag"], (
        "the frozen within-bag median equals the pre-phase-1 bootstrap one, so "
        "nothing here distinguishes the trained encoder from the fresh one")


# ------------------------------------- the supervised term leaves the phase-2 loss

def test_the_phase_two_objective_drops_eq_21_and_moves_no_step(familia, tiny):
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
        generator = torch.Generator().manual_seed(9973)
        toy_run.train_phase_one(model, source_train, source_valid, DEVICE,
                                tiny, hyper, generator)
        toy_run.copy_source_to_target(model)
        for name, parameter in model.named_parameters():
            if not name.startswith("target_encoder."):
                parameter.requires_grad_(False)
        toy_run.set_phase_two_modes(model)

        x, y = toy_run._batch(source_train, range(len(source_train)), DEVICE)
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
            gen = torch.Generator().manual_seed(9973)
            toy_run.train_phase_one(arm, train_i, valid_i, DEVICE, tiny, hyper,
                                    gen)
            toy_run.copy_source_to_target(arm)
            for name, parameter in arm.named_parameters():
                if not name.startswith("target_encoder."):
                    parameter.requires_grad_(False)
            toy_run.set_phase_two_modes(arm)
            optimizer = torch.optim.Adam(arm.target_encoder.parameters(),
                                         lr=tiny.lr_2)
            xi, yi = toy_run._batch(train_i, range(len(train_i)), DEVICE)
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


def test_the_phase_two_curve_records_eq_21_as_a_reading(cell):
    """Eq. (21) stays in the record and in the figure; what it stops being is a
    summand.

    Three fields make that readable without re-running: `supervised` is the
    reading, `phaseTwoObjective` is what was minimized, and the second equals
    the two declared contributions. A record that dropped the reading would
    have deleted the number rather than reclassified it, which is not what was
    asked and is not recoverable from a checkpoint.
    """
    row = cell["rows"][ADAPTED]
    assert row["supervised"] > 0.0, (
        "Eq. (21) comes back zero, so nothing distinguishes 'recorded as a "
        "reading' from 'deleted'")
    assert "phaseTwoObjective" in row, sorted(row)
    assert row["phaseTwoObjective"] == pytest.approx(
        row["contributionGlobal"] + row["contributionLocal"], rel=1e-6)
    # And the objective is NOT the three-term sum, which is the whole change.
    assert row["phaseTwoObjective"] != pytest.approx(
        row["supervised"] + row["contributionGlobal"] + row["contributionLocal"])
    assert toy_run.PHASE_TWO_OBJECTIVE["equation"].startswith("L = ")
    assert "21" in toy_run.PHASE_TWO_OBJECTIVE["readingOnly"]


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


def _tres_roles_de_la_fuente(familia, scale, seed=0, backbone="mlp"):
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
        generator = torch.Generator().manual_seed(seed + 9973)
        phase_one = toy_run.train_phase_one(model, source_train, source_valid,
                                            DEVICE, scale, hyper, generator)
        model.load_state_dict(phase_one["state"], strict=False)
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
        nombre, tiny):
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
    medido = _tres_roles_de_la_fuente(familia, tiny)

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
    alternativa = _tres_roles_de_la_fuente(contigua, tiny)
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
    assert "instances_per_bag" not in toy_run.REPETITION_FIELDS

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
            bags._raw_dataset("M", toy_run.destination(tiny))


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
#: instance that was found first. `run_cell` called `embedding_scales` -- a
#: measurement -- before `train_phase_one`, and that forward moved BatchNorm's
#: running statistics, so the training that followed started from a different
#: model than the one the seeds describe. Two more members were measured the
#: same way and are here for the same reason, not because either was suspected:
#: `frozen_source_sigma`, which was harmless only because `train_phase_one`
#: happens to leave the model in `eval`, and `weighting_cost`, whose docstring
#: claimed "it changes nothing it measures" while its weighted half ran on the
#: live model.
#:
#: Keyed by the attribute name so a member whose name changes reddens here
#: rather than silently dropping out of the sweep.
MEASUREMENTS = ("embedding_scales", "frozen_source_sigma", "weighting_cost")


def _measurement_calls(material, tiny):
    source_train, _, _ = bags.roles(material["source"])
    return {
        "embedding_scales": lambda m: toy_run.embedding_scales(
            m, material["source"], material["source"].train_idx, DEVICE),
        "frozen_source_sigma": lambda m: toy_run.frozen_source_sigma(
            m, material["source"], material["source"].train_idx, DEVICE),
        "weighting_cost": lambda m: toy_run.weighting_cost(
            m, source_train, DEVICE, 0),
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
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    assert sorted(_measurement_calls(material, tiny)) == sorted(MEASUREMENTS)


@pytest.mark.parametrize("measurement", MEASUREMENTS)
def test_a_measurement_leaves_every_buffer_of_the_model_bit_identical(
        measurement, familia, tiny):
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
    - and the mode is restored per module, because `train_phase_one` reads it
      and phase 2 runs the source in `eval` with `theta_t` in `train`.
    """
    preparado = _prepared(familia, tiny)
    model, material = preparado["model"], preparado["material"]
    call = _measurement_calls(material, tiny)[measurement]

    with toy_run.material_bound(familia, tiny, "mlp"):
        # The caller held it in `train()`: that is the condition the fix has to
        # hold under, and it is `run_cell`'s own condition -- a freshly built
        # `nn.Module` is in `train` mode and `embedding_scales` was handed one.
        model.train()
        before = _float_buffers(model)
        assert before, (
            "this backbone carries no floating-point buffer, so the equality "
            "below would hold over an empty set and say nothing")

        # The control: a training forward in this very mode moves them.
        control = copy.deepcopy(model)
        control.train()
        moving = _float_buffers(control)
        x, y = toy_run._batch(preparado["source_train"],
                              range(len(preparado["source_train"])), DEVICE)
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
    still `(0, 1)` while `train` mode normalizes by the batch. `sigma_grid` is
    centred on those, so forcing `eval` would move every grid the phase-1
    search has ever ranged over -- a decision about the experiment, and not
    this fix's to take.

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
        "caller holds -- which silently restates every bandwidth "
        "`sigma_grid` is centred on")
    assert readings["mlp-nobn"]["train"] == readings["mlp-nobn"]["eval"], (
        readings["mlp-nobn"],
        "the two modes disagree on a backbone with no mode-dependent layer, "
        "so the difference above is not the normalization")


def test_the_mechanism_is_batchnorms_running_statistics_and_nothing_else(
        tiny):
    """A forward pass taken in `train` mode before phase 1 moves the early stop
    under `mlp` and cannot move it under `mlp-nobn` or `mlp-layernorm`.

    This is the half that makes the attribution a MEASUREMENT and not a
    reading, and it keeps a reachable red after the fix because it builds the
    perturbation itself rather than depending on the code still carrying it: a
    bare `instance_embeddings` forward in `train` mode, which is what
    `embedding_scales` used to amount to. Under `mlp` that changes the
    validation column the early stop reads and so the epoch it chooses; under
    the other two backbones it changes nothing, because neither keeps a buffer
    a forward pass can move.

    The buffer inventory is counted beside it, so the two halves of the claim
    agree -- the backbone that moves is the one with running statistics, and
    the ones that cannot move carry none.
    """
    counts, histories = {}, {}
    hyper = toy_run.declared_hyper(tiny)
    familia = toy_run.build_family("toroide", 0, tiny)
    for backbone in sorted(set(toy_run.NORMALIZATIONS.values())):
        material = toy_run.material_for(familia, 0, tiny, backbone)
        readings = {}
        for perturb in (False, True):
            with toy_run.material_bound(familia, tiny, backbone, hyper):
                source_train, source_valid, _ = bags.roles(material["source"])
                model = toy_run.build_arm(ADAPTED, material, tiny, hyper,
                                          DEVICE, 0)
                counts[backbone] = len(_float_buffers(model))
                if perturb:
                    # The defect, by hand: a forward in `train` mode on the
                    # model about to be trained.
                    model.train()
                    with torch.no_grad():
                        latent_module.represent(
                            model, material["source"],
                            material["source"].train_idx, DEVICE,
                            domain="source", unit="instance")
                readings[perturb] = toy_run.train_phase_one(
                    model, source_train, source_valid, DEVICE, tiny, hyper,
                    torch.Generator().manual_seed(0 + 9973))
        histories[backbone] = (readings[False]["history"],
                               readings[True]["history"])

    # The pole: on the backbone that keeps running statistics the perturbation
    # DOES reach the early stop, so the two equalities below are a statement
    # about the mechanism and not about a perturbation that never landed.
    clean, dirty = histories["mlp"]
    assert ([row[1] for row in clean] == [row[1] for row in dirty]), (
        "the training-loss column moved too, so the perturbation is not "
        "confined to the eval-mode reading and this is not the mechanism "
        "named")
    assert clean != dirty, (
        clean, dirty,
        "a train-mode forward before phase 1 moved nothing under `mlp`, so "
        "the two controls below would be agreeing with a perturbation that "
        "does not perturb")

    for backbone in ("mlp-nobn", "mlp-layernorm"):
        assert histories[backbone][0] == histories[backbone][1], (
            backbone, histories[backbone])

    assert counts["mlp"] > 0, counts
    assert counts["mlp-nobn"] == 0, counts
    assert counts["mlp-layernorm"] == 0, counts


def test_the_recorded_best_epoch_of_phase_one_is_the_one_a_rerun_measures(
        tiny):
    """Same material, same seeds: the `phase1BestEpoch` the record carries is
    the one a rerun of phase 1 alone measures.

    The defect this names, measured before anything was written: recorded `0`
    against recomputed `1` under `mlp`, and `bestValidAccuracy` identical at
    0.9333 on both.

    **The channel is the early stop and not the training, measured rather than
    inferred -- which is sharper than the brief that commissioned this.** With
    `embedding_scales` called first and without it, on the same seeds:

    | epoch | trainLoss | validLoss (clean / dirty) | validAcc (clean / dirty) |
    | --- | --- | --- | --- |
    | 0 | 0.014930576613 both | 0.013483732939 / 0.015788229182 | 0.8667 / 0.9333 |
    | 1 | 0.001825315634 both | 0.004594453145 / 0.005847103894 | 0.9333 / 0.9333 |

    The training-loss column is BIT-IDENTICAL, because in `train` mode a
    BatchNorm normalizes by the BATCH statistics and the running ones never
    enter the training forward. What the diagnostic reaches is
    `harness.accuracy`, which reads in `eval` -- the one mode where the running
    statistics are used -- so the early stop chooses on a criterion a
    measurement contributed to. `bestEpoch` then goes 1 -> 0 and the weights
    phase 1 freezes and hands to phase 2 differ by `||dtheta|| = 2.548`.

    So the pole is the TRAINING-loss column, asserted first: identical there
    means the two runs are the same training, which is what makes a differing
    best epoch a statement about the criterion rather than about
    reproducibility in general. An earlier version of this test asserted the
    WHOLE history as its pole -- recorded here rather than quietly fixed,
    because that was a wrong claim of mine and not a defect in the code: the
    validation columns are exactly what the defect moves, so using them as the
    precondition made the test red on its own premise and hid the claim it was
    about.
    """
    familia = toy_run.build_family("toroide", 0, tiny)
    hyper = toy_run.declared_hyper(tiny)
    for normalization, backbone in sorted(toy_run.NORMALIZATIONS.items()):
        celda = toy_run.run_cell(familia, normalization, 0, tiny, hyper, DEVICE)
        material = toy_run.material_for(familia, 0, tiny, backbone)
        with toy_run.material_bound(familia, tiny, backbone, hyper):
            source_train, source_valid, _ = bags.roles(material["source"])
            model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
            rerun = toy_run.train_phase_one(
                model, source_train, source_valid, DEVICE, tiny, hyper,
                torch.Generator().manual_seed(0 + 9973))
        recorded = celda["phase1"]
        assert ([row[1] for row in recorded["history"]]
                == [row[1] for row in rerun["history"]]), (
            normalization,
            "the two phase-1 trainings disagree on their own TRAINING loss, so "
            "they are not the same training and what follows would not be "
            "about a measurement that reaches the early stop")
        assert recorded["bestEpoch"] == rerun["bestEpoch"], (
            normalization, recorded["bestEpoch"], rerun["bestEpoch"],
            "identical material, identical seeds and an identical training "
            "loss, and the recorded best epoch is not the one a rerun "
            "measures: something ran a forward pass between the build and the "
            "training")
        # And then the whole history, validation columns included: those are
        # what the defect moved, so their agreement is the claim and never the
        # premise.
        assert recorded["history"] == rerun["history"], (
            normalization,
            [(a, b) for a, b in zip(recorded["history"], rerun["history"])
             if a != b])
        assert (celda["rows"][ADAPTED]["phase1BestEpoch"]
                == recorded["bestEpoch"])


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


def test_the_sigma_grid_is_centred_on_a_measured_distance_and_lands_in_band(cell):
    """The within-bag kernel at the within-bag median depends only on the
    multiplier, and that is what makes the grid in band by construction.

    `exp(-d^2 / (2 (m d)^2))` is `exp(-1 / (2 m^2))`. Asserting the measured
    values against that identity compares two different computations; writing the
    four numbers down as literals would assert constants.
    """
    grid = cell["sigma"]
    within = grid["scales"]["withinBag"]
    assert within > 0.0
    assert grid["centredOn"] == "withinBag"
    assert len(grid["points"]) == len(toy_run.SIGMA_MULTIPLIERS)
    for point in grid["points"]:
        multiplier = point["multiplier"]
        assert math.isclose(point["sigma"], multiplier * within, rel_tol=1e-9)
        assert math.isclose(point["kernelWithinBag"],
                            math.exp(-1.0 / (2.0 * multiplier ** 2)),
                            rel_tol=1e-9)
        assert point["inBand"], point


def test_the_within_bag_scale_is_not_the_pooled_one(cell):
    """The measurement that corrected this module's first calibration: the
    distance the instance kernel evaluates is WITHIN a bag, and a bag is a band
    of one class, so it is much smaller than the pooled spread.

    **The between-bag comparison is deliberately NOT asserted here, and that is
    a correction.** This test used to also assert `withinBag < betweenBags`,
    which contradicts what the test below it states and what
    `embedding_scales`' own docstring says: which of those two is larger varies
    by family and by backbone, and no reader may assume a direction. At four
    bags per class this fixture happened to satisfy it; at twelve it does not
    (measured here: within 1.3581 against between 0.9395), and the assertion was
    pinning an accident of the cut. The claim about `pooled` is the one this test
    is named for and the one that holds by construction: a bag is a band of one
    class and the pool is every class.
    """
    scales = cell["sigma"]["scales"]
    assert scales["withinBag"] < scales["pooled"], scales
    # The two the ONE bandwidth has to serve differ, in whichever direction:
    # that they differ at all is Decision 1's consequence and is what the
    # searched range is widened to span.
    assert scales["withinBag"] != scales["betweenBags"], scales
    assert cell["sigma"]["scaleRatio"] != 1.0
    assert cell["sigma"]["scaleRatio"] == pytest.approx(
        scales["betweenBags"] / scales["withinBag"])


def test_one_bandwidth_cannot_be_in_band_for_both_jobs_and_the_range_spans_both(
        cell):
    """Decision 1 declares ONE bandwidth; the two kernels it feeds live on
    scales that differ. The grid reports both readings and the searched range
    covers both, so the trade-off is the search's and not a silent choice.

    **The direction is derived and never assumed.** Measured over this toyset's
    six cells, the between-bag median is the larger one in three and the smaller
    one in three, so a test that asserted `betweenBags > withinBag` would be
    asserting something the data contradicts on half the grid -- and the range
    that followed from that assumption would invert.
    """
    grid = cell["sigma"]
    within, between = grid["scales"]["withinBag"], grid["scales"]["betweenBags"]
    low, high = grid["range"]
    assert low < high, (low, high)
    assert low == pytest.approx(toy_run.SIGMA_MULTIPLIERS[0]
                                * min(within, between))
    assert high == pytest.approx(toy_run.SIGMA_MULTIPLIERS[-1]
                                 * max(within, between))
    at_within = [p for p in grid["points"] if p["multiplier"] == 1.0][0]
    assert at_within["kernelWithinBag"] != at_within["kernelBetweenBags"]
    # The larger distance reads the smaller kernel, whichever one it is.
    wider = ("kernelBetweenBags" if between > within else "kernelWithinBag")
    narrower = ("kernelWithinBag" if between > within else "kernelBetweenBags")
    assert at_within[wider] < at_within[narrower]


def test_the_range_stays_ordered_when_the_between_bag_scale_is_the_smaller_one():
    """The reachable red for the ordering above, on the half of the grid the
    fixture does not happen to land on: an inverted range is what `optuna`
    raises on, and it would have waited for the full scale to arrive."""
    narrow_between = {"withinBag": 1.28, "betweenBags": 0.30, "pooled": 1.5,
                      "bagsRead": 6, "instancesPerBag": 30,
                      "groupingCheckedAgainstBagUnit": 0.0,
                      "weightSpread": 0.001, "uniformWeight": 1 / 30,
                      "weightSpreadRelativeToUniform": 0.03}
    assert narrow_between["betweenBags"] < narrow_between["withinBag"]
    grid = toy_run.sigma_grid(narrow_between)
    low, high = grid["range"]
    assert low < high, (low, high)
    assert low == pytest.approx(toy_run.SIGMA_MULTIPLIERS[0] * 0.30)
    assert high == pytest.approx(toy_run.SIGMA_MULTIPLIERS[-1] * 1.28)


def test_the_attention_weights_are_reported_against_uniform(cell):
    """A weight spread of 1e-5 against a uniform 1/m means Eq. (16) is
    numerically mean pooling whatever the flags say, so the number travels with
    the record instead of being discovered later."""
    scales = cell["sigma"]["scales"]
    assert scales["uniformWeight"] == pytest.approx(
        1.0 / scales["instancesPerBag"])
    assert scales["weightSpread"] >= 0.0
    assert scales["weightSpreadRelativeToUniform"] == pytest.approx(
        scales["weightSpread"] / scales["uniformWeight"])


def test_the_declared_absolute_bandwidth_is_out_of_band_on_this_toyset(cell):
    """Why the grid had to be measured: `config.KERNEL_SIGMA` was calibrated on
    digits under resnet18, and at this toyset's scale it saturates the kernel."""
    grid = cell["sigma"]
    assert grid["declaredConstant"] == config.KERNEL_SIGMA
    assert grid["declaredKernelWithinBag"] > toy_run.SIGMA_BAND[1], (
        f"the declared sigma reads {grid['declaredKernelWithinBag']} inside the "
        "band; the measured grid would then be solving a problem this toyset "
        "does not have")


def test_the_eight_declared_ranges_split_into_four_searched_and_four_retired():
    """Four searched, four retired, and the two sets PARTITION `config`'s eight
    `*_RANGE` constants -- so a range cannot be quietly dropped from both.

    This is what replaced "eight entries, which is every declared range". The
    partition is the part worth asserting: a dimension that left the search and
    was not named as retired would be unsearched and unexplained, which is the
    state this module's own comment used to promise did not exist.
    """
    declared = {name for name in dir(config) if name.endswith("_RANGE")}
    assert len(declared) == 8, sorted(declared)

    searched = set(toy_run.PHASE_ONE_DIMENSIONS) | set(toy_run.PHASE_TWO_DIMENSIONS)
    retired = set(toy_run.RETIRED_RANGES)
    assert searched == set(toy_run.SEARCH_RANGES)
    assert len(searched) == 4, sorted(searched)
    assert len(retired) == 4, sorted(retired)
    assert not (searched & retired), sorted(searched & retired)
    assert not (set(toy_run.PHASE_ONE_DIMENSIONS)
                & set(toy_run.PHASE_TWO_DIMENSIONS))

    # The partition, against `config`'s own eight. `kernelSigma` is the one
    # searched dimension whose range is MEASURED rather than declared
    # (`sigma_grid`), so it is matched to `KERNEL_SIGMA_RANGE` by name here --
    # every other one carries `config`'s tuple itself.
    def range_of(name):
        return f"{_SNAKE[name]}_RANGE"

    accounted = {range_of(name) for name in searched | retired}
    assert accounted == declared, sorted(accounted ^ declared)

    # Each retired entry names the thing it parameterized and the reason, and
    # carries `config`'s own tuple -- not a copy that can drift.
    for name, entry in toy_run.RETIRED_RANGES.items():
        assert entry["range"] is getattr(config, range_of(name)), name
        assert entry["was"] and entry["why"], name
    # And the three searched ranges that are tuples are `config`'s own too.
    for name in ("attentionGamma", "attentionTemperature", "tauLocal"):
        assert toy_run.SEARCH_RANGES[name] is getattr(config, range_of(name))


#: `config`'s range name for each dimension this module searches or retired.
#: Spelled out because the mapping is not derivable: `kernelSigma` ->
#: `KERNEL_SIGMA_RANGE` is, but `ceiling` -> `CEILING_RANGE` and `rampDelta` ->
#: `RAMP_DELTA_RANGE` do not share a rule with `ceilingLocal` ->
#: `CEILING_LOCAL_RANGE`, whose suffix moves to the middle.
_SNAKE = {
    "kernelSigma": "KERNEL_SIGMA",
    "attentionGamma": "ATTENTION_GAMMA",
    "attentionTemperature": "ATTENTION_TEMPERATURE",
    "tauLocal": "TAU_LOCAL",
    "ceiling": "CEILING",
    "ceilingLocal": "CEILING_LOCAL",
    "rampDelta": "RAMP_DELTA",
    "rampDeltaLocal": "RAMP_DELTA_LOCAL",
}


def test_phase_one_reads_the_three_dimensions_the_split_assigns_to_it(familia,
                                                                      tiny):
    """The split is not a convention: phase 1's own loss MOVES when those three
    move, which is why they cannot be searched in phase 2.

    **And `kernelSigma` reaches it only while `attentionGamma` is non-zero**,
    which is measured here rather than assumed. Eq. (15) is
    `R_phi(h) + gamma * consensus(h, sigma)`: at `config.ATTENTION_GAMMA = 0.0`
    the consensus half -- the only place the instance kernel of Eq. (14) enters
    the attention -- is multiplied by zero and the bandwidth drops out of the
    forward entirely. So the first assertion below is that at the declared gamma
    the bandwidth is INERT, and the second is that at a gamma the search can
    reach it is not. Both have to hold or the phase split is wrong in one
    direction or the other.

    `tauLocal` is perturbed too and must NOT change the supervised loss -- the
    control that keeps this from passing for any hyperparameter at all.
    """
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    base = toy_run.declared_hyper(tiny)
    assert base["attentionGamma"] == config.ATTENTION_GAMMA == 0.0
    with toy_run.material_bound(familia, tiny, "mlp"):
        source_train, _, _ = bags.roles(material["source"])
        x, y = toy_run._batch(source_train, range(len(source_train)), DEVICE)

        def loss_under(hyper):
            model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
            model.eval()
            with torch.no_grad():
                return float(toy_run.supervised_loss(model, x, y))

        probe = toy_run.build_arm(ADAPTED, material, tiny, base, DEVICE, 0)
        scales = toy_run.embedding_scales(probe, material["source"],
                                         material["source"].train_idx, DEVICE)
        within = scales["withinBag"]

        reference = loss_under(base)
        assert loss_under({**base, "kernelSigma": within}) == reference, (
            "the bandwidth moved the loss at gamma = 0; Eq. (15)'s consensus "
            "half is supposed to be multiplied out there")

        # The gamma the search can reach, and the one the declared constant is
        # not, at the bandwidth the measurement centred the grid on. Every
        # assertion below is made against this base.
        live = {**base, "attentionGamma": 0.5, "kernelSigma": within}
        assert config.ATTENTION_GAMMA_RANGE[0] <= 0.5 <= config.ATTENTION_GAMMA_RANGE[1]
        live_reference = loss_under(live)
        assert live_reference != reference, (
            "turning the consensus half on changed nothing; gamma is inert")
        for name, value in (("kernelSigma", within * 0.5),
                            ("attentionGamma", 0.9),
                            ("attentionTemperature", 4.0)):
            assert live[name] != value, name
            assert loss_under({**live, name: value}) != live_reference, (
                f"{name} does not reach phase 1's loss; the split is wrong")
        assert live["tauLocal"] != 7.0
        assert loss_under({**live, "tauLocal": 7.0}) == live_reference, (
            "tauLocal reaches phase 1's loss; it belongs in the phase-1 search")


# --------------------------------------------- the dimensions CREDA searches

def test_phase_one_searches_the_output_dimension_creda_searches(familia, tiny):
    """`DIM` is a searched dimension here, over CREDA's own four choices.

    It used to be `config.MLP_OUTPUT_DIM`, fixed at 16, and
    `PHASE_ONE_NOT_SEARCHED` said so. `mlp.MLP` reads that global at
    CONSTRUCTION, so the only way to search it is to bend the global for the
    duration of a binding, which is what `material_bound` already does for the
    class count and the bag size.

    Both halves are asserted, because either alone passes while the dimension
    decides nothing: the CHOICES are CREDA's `buscar_hp1` categorical, and the
    bend REACHES the built encoder -- measured off the arm's own parameter
    shapes, never off the global it was set from.
    """
    assert toy_run.PHASE_ONE_DIM_CHOICES == (8, 16, 32, 64)
    assert "DIM" not in toy_run.PHASE_ONE_NOT_SEARCHED
    material = toy_run.material_for(familia, 0, tiny, "mlp")
    widths = {}
    for dim in toy_run.PHASE_ONE_DIM_CHOICES:
        hyper = {**toy_run.declared_hyper(tiny), "dim": dim}
        with toy_run.material_bound(familia, tiny, "mlp", hyper):
            assert config.MLP_OUTPUT_DIM == dim
            model = toy_run.build_arm(ADAPTED, material, tiny, hyper, DEVICE, 0)
            widths[dim] = int(model.head.in_features)
    assert widths == {dim: dim for dim in toy_run.PHASE_ONE_DIM_CHOICES}, widths
    # And the global is restored, so a bend cannot leak into the next binding.
    assert config.MLP_OUTPUT_DIM == toy_run._DECLARED_GLOBALS["MLP_OUTPUT_DIM"]


def test_phase_two_searches_its_own_learning_rate_and_weight_decay(familia,
                                                                   tiny):
    """`LR_2` and `WD_2`, CREDA's own two phase-2 optimizer scalars.

    Before this, `scale.lr_2` ran at its `1e-3` default and was never tuned,
    and the Adam carried NO weight decay at all -- not a searched zero, an
    absent argument.

    Read off the optimizer that is actually constructed and then off the
    WEIGHTS, because each catches a different failure: a value threaded into
    `Adam` that the loop never steps with would pass the first, and a value
    that changed the weights for some other reason would pass the second.
    """
    assert toy_run.PHASE_TWO_OPTIMIZER_RANGES == {"lr2": (1e-5, 1e-2),
                                                  "weightDecay2": (1e-7, 1e-2)}
    seen = []
    real_adam = torch.optim.Adam

    def spy(parameters, **kwargs):
        seen.append(dict(kwargs))
        return real_adam(parameters, **kwargs)

    payload = _prepared(familia, tiny, arm_id=ADAPTED)
    hyper = {**payload["hyper"], "lr2": 3.7e-3, "weightDecay2": 4.1e-4}
    with toy_run.material_bound(familia, tiny, "mlp", hyper):
        try:
            torch.optim.Adam = spy
            toy_run.adapt_phase_two(payload["model"], payload["source_train"],
                                    payload["source_valid"], DEVICE, tiny,
                                    hyper, torch.Generator().manual_seed(1))
        finally:
            torch.optim.Adam = real_adam
    assert seen and seen[-1]["lr"] == 3.7e-3, seen
    assert seen[-1]["weight_decay"] == 4.1e-4, seen

    # The behavioural pole: two decades apart on the rate must move theta_t.
    def final_theta(extra):
        payload = _prepared(familia, tiny, arm_id=ADAPTED)
        h = {**payload["hyper"], **extra}
        with toy_run.material_bound(familia, tiny, "mlp", h):
            toy_run.adapt_phase_two(payload["model"], payload["source_train"],
                                    payload["source_valid"], DEVICE, tiny, h,
                                    torch.Generator().manual_seed(1))
        return toy_run._parameter_vector(payload["model"].target_encoder).clone()

    slow = final_theta({"lr2": 1e-5, "weightDecay2": 1e-7})
    fast = final_theta({"lr2": 1e-2, "weightDecay2": 1e-7})
    decayed = final_theta({"lr2": 1e-2, "weightDecay2": 1e-2})
    assert not torch.allclose(slow, fast), "the searched rate steps nothing"
    assert not torch.allclose(fast, decayed), (
        "the searched weight decay steps nothing")


def test_the_bag_lot_is_not_searchable_here_and_the_record_says_so(tiny):
    """CREDA's `LOTE_1`/`LOTE_2` have no counterpart this path can search, and
    the reason is measured rather than asserted.

    `harness.balanced_batches` yields exactly ONE bag per class per step. It
    takes no lot argument and reads no `config.BAGS_PER_STEP`, so the training
    lot in bags is `classes` and nothing in this module can move it.
    `config.BAGS_PER_STEP` divides the role length to get the STEP COUNT of a
    phase-1 epoch, which is an epoch length and not a lot.

    Measured here so the claim in `PHASE_ONE_NOT_SEARCHED` is a reading: the
    batch sizes come back at `classes` whatever `BAGS_PER_STEP` is set to.
    """
    import inspect as _inspect

    parameters = _inspect.signature(harness.balanced_batches).parameters
    assert set(parameters) == {"targets", "steps", "generator"}, sorted(parameters)
    targets = [c for c in range(tiny.classes)
               for _ in range(tiny.realized_split()["bagsPerClass"]["train"])]
    before = config.BAGS_PER_STEP
    try:
        sizes = {}
        for forced in (before, tiny.classes, 24):
            config.BAGS_PER_STEP = forced
            generator = torch.Generator().manual_seed(0)
            sizes[forced] = [len(b) for b in
                             harness.balanced_batches(targets, 4, generator)]
        for forced, drawn in sizes.items():
            assert drawn == [tiny.classes] * 4, (forced, drawn)
    finally:
        config.BAGS_PER_STEP = before
    # And both declarations name it as unsearchable with that reason, so a
    # reader of the record is not left to rediscover this.
    assert "LOTE_1" in toy_run.PHASE_ONE_NOT_SEARCHED
    assert "LOTE_2" in toy_run.PHASE_TWO_NOT_SEARCHED
    for entry in (toy_run.PHASE_ONE_NOT_SEARCHED["LOTE_1"],
                  toy_run.PHASE_TWO_NOT_SEARCHED["LOTE_2"]):
        assert "balanced_batches" in entry, entry


def test_the_record_names_every_searched_dimension_with_its_range(tiny):
    """A record that named dimensions and not ranges would say a value was
    chosen and not from where.

    Asserted against the module's own declarations rather than against literals,
    and the COUNTS are asserted too: the brief for this change expected 7 and 4,
    and the bag lot is unreachable, so what landed is 6 and 3. A count written
    here is what makes that gap visible instead of silently absorbed.
    """
    one = toy_run.searched_ranges("phase1")
    two = toy_run.searched_ranges("phase2")
    assert set(one) == {"kernelSigma", "attentionGamma", "attentionTemperature",
                        "lr1", "weightDecay1", "dim"}, sorted(one)
    assert set(two) == {"tauLocal", "lr2", "weightDecay2"}, sorted(two)
    assert len(one) == 6 and len(two) == 3
    for ranges in (one, two):
        for name, span in ranges.items():
            assert span, name


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
    assert "sweep" in defined, sorted(defined)
    offenders = sorted(name for name in defined
                       if "FLOOR" in name or "floor" in name
                       or "LADDER" in name or "rung" in name)
    assert not offenders, offenders


def test_the_artefact_carries_no_rung_and_no_floor_block(tmp_path, cell, tiny,
                                                         familia):
    """The artefact's own shape: `rungs`/`floors` are ABSENT, not empty.

    An empty list reads as "it was computed and there was nothing", and what is
    true is that nothing can be computed. `guardar` is driven over a real
    `run_cell` result so the keys asserted are the keys that would travel.
    """
    barrido = _barrido_from(cell, tiny, familia)
    assert "rungs" not in barrido and "floors" not in barrido
    written = {path.name for path in toy_run.guardar(tmp_path, barrido)}
    assert "barrido.json" in written
    payload = json.loads((tmp_path / "barrido.json").read_text(encoding="utf-8"))
    assert "escalones" not in payload, sorted(payload)
    assert "pisos" not in payload, sorted(payload)
    # The control: the keys that DO travel are there, so the two absences above
    # are not an empty file.
    assert {"celdas", "filas", "hp1", "costo", "escala"} <= set(payload)
    assert "ladder" not in barrido["flags"] and "floorOf" not in barrido["flags"]
    assert "verdictRepetitionFloor" not in barrido["flags"]
    assert "one arm" in barrido["flags"]["armsNote"]
    assert "floor" in barrido["flags"]["syncRetired"]["riskNotMeasurableHere"]
    # And `cargar` does not reinstate them on the way back.
    read = toy_run.cargar(tmp_path)
    assert "rungs" not in read and "floors" not in read, sorted(read)


# ------------------------------------------------------------------ the artefact

def _barrido_from(cell: dict, tiny, familia) -> dict:
    """A `sweep`-shaped payload built from a REAL `run_cell` result.

    Shared by the round trip and by the absence test above so both are driven
    over the same thing `sweep` returns. What is NOT here is a `rungs` or a
    `floors` key: `sweep` no longer produces either, and a helper that supplied
    one would let the round trip pass over a shape the module cannot emit.
    """
    rows_by_arm = {arm: [row] for arm, row in cell["rows"].items()}
    return {
        "flags": toy_run.flags(tiny, DEVICE, (familia.nombre,),
                               [cell["rates"]], cell["elapsedSeconds"]),
        "cells": {toy_run.key(familia.nombre, "batchnorm", arm, 0): row
                  for arm, row in cell["rows"].items()},
        "rows": {toy_run.key(familia.nombre, "batchnorm", arm): {
            "runs": runs,
            "summary": {"targetAccuracy": harness.spread(
                [float(r["targetAccuracy"]) for r in runs])},
            "scale": tiny.stamp()} for arm, runs in rows_by_arm.items()},
        "hp1": {familia.nombre: {"lr1": 0.01}},
        "hp2": {familia.nombre: {"tauLocal": 0.5}},
        "records1": {familia.nombre: {"trials": 1, "perTrial": [{"value": 1.0}]}},
        "records2": {familia.nombre: {"trials": 1, "perTrial": [{"watchman": 0.3}]}},
        "perFamily": {familia.nombre: {
            "rows": {"batchnorm": rows_by_arm},
            "histories": {toy_run.key("batchnorm", 0): {
                "phase1": cell["phase1"], "sigma": cell["sigma"]}},
            # The same handle `sweep` builds -- normalization, arm, seed -- so
            # the round trip is driven over the key shape the real artefact
            # carries and not over one this file invented.
            "latent": {toy_run.key("batchnorm", arm, 0): members
                       for arm, members in cell["latent"].items()}}},
        "cost": toy_run.cost_report([cell["rates"]], tiny,
                                    cell["elapsedSeconds"], families=1),
        "scale": tiny.stamp(),
    }


def test_guardar_and_cargar_round_trip_through_the_real_files(tmp_path, cell,
                                                              tiny, familia):
    """The pair is tested across the join: `guardar` writes real files, `cargar`
    reads those files, and the keys compared are the ones that actually travelled.

    Writing a dict here and reading it back would verify neither half -- the same
    hand wrote it and reads it. What is written below comes out of a real
    `run_cell`, through the module's own `flags` and `cost_report`.
    """
    barrido = _barrido_from(cell, tiny, familia)

    written = toy_run.guardar(tmp_path, barrido)
    names = {path.name for path in written}
    assert names == {"banderas.json", "barrido.json", f"{familia.nombre}.npz"}
    for path in written:
        assert path.exists() and path.stat().st_size > 0, path

    read = toy_run.cargar(tmp_path)
    assert read["flags"]["scale"] == tiny.stamp()
    assert read["scale"] == tiny.stamp()
    assert set(read["cells"]) == set(barrido["cells"])
    assert set(read["rows"]) == set(barrido["rows"])
    assert read["hp1"] == barrido["hp1"]
    assert read["trials2"] == {familia.nombre: [{"watchman": 0.3}]}
    assert read["cost"]["secondsPerCell"] == barrido["cost"]["secondsPerCell"]

    family_back = read["families"][familia.nombre]
    assert family_back["info"], "no per-run record came back"
    for handle, info in family_back["info"].items():
        assert info["scale"]["name"] == "test", handle
        assert isinstance(info["targetAccuracy"], float)
    assert family_back["hist_s"], "no phase-1 history came back"
    one = next(iter(family_back["hist_s"].values()))
    assert len(one) == cell["phase1"]["epochs"]
    assert len(one[0]) == 4
    assert next(iter(family_back["sigma"].values())) == pytest.approx(
        [p["sigma"] for p in cell["sigma"]["points"]], rel=1e-6)


def test_the_round_trip_is_what_catches_a_record_that_cannot_travel(tmp_path,
                                                                    tiny):
    """`np.savez` stores a value it cannot serialize without pickle and raises on
    LOAD -- in the other notebook, with no clue which key it was. So the field is
    named here, on the way out."""
    with pytest.raises(TypeError, match="is not serializable"):
        toy_run._flat_scalars({"ok": 1.0, "bad": object()})


def test_a_bool_comes_back_a_bool_and_not_a_one(tmp_path):
    """`bool` IS a subclass of `int`: without the branch that tests it first,
    `True` returns as `1.0`, the write stays green, and the defect exists only on
    the way back."""
    flat = toy_run._flat_scalars({"flag": True, "count": 3})
    assert flat["flag"] is True
    assert flat["count"] == 3.0 and isinstance(flat["count"], float)


def test_the_destination_carries_the_scale_so_a_pilot_cannot_overwrite_a_run():
    assert toy_run.destination(toy_run.PILOT) != toy_run.destination(toy_run.FULL)
    assert toy_run.destination(toy_run.PILOT).name == "pilot"
    assert toy_run.destination(toy_run.FULL).name == "full"


# ------------------------------------------------------- the latent positions

def _perturbar_theta_t(model, paso: float = 0.05) -> float:
    """Move `theta_t` and return how far it actually moved.

    The return value is the point: a test that perturbs a model and then
    believes the result without checking the perturbation landed is reporting
    success while measuring nothing.
    """
    firma = toy_run._parameter_vector(model.target_encoder).clone()
    with torch.no_grad():
        for parametro in model.target_encoder.parameters():
            parametro.add_(torch.full_like(parametro, paso))
    return float((toy_run._parameter_vector(model.target_encoder) - firma).norm())


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


def test_the_cell_records_both_moments_and_measures_the_freeze(cell):
    """The three views, and the freeze the figure's frame rests on.

    `latentSourceMoved` being zero is only evidence because the same reading on
    the encoder phase 2 DOES move comes out non-zero: a function that returned
    zero whatever it was handed would satisfy the first half alone.
    """
    arm = next(iter(cell["latent"]))
    miembros = cell["latent"][arm]
    assert set(miembros) == set(toy_run.LATENT_MEMBERS)

    fila = cell["rows"][arm]
    assert fila["latentTargetMoved"] > 0, (
        "la fase 2 no movió ninguna posición del blanco: la figura no tiene "
        "nada que mostrar y este registro no dice por qué")
    assert fila["latentSourceMoved"] == 0.0, (
        f"la fuente se movió {fila['latentSourceMoved']} en el espacio que la "
        "figura dibuja: el panel usa esa nube como marco fijo y no lo es")

    antes, despues = miembros["targetBefore"], miembros["targetAfter"]
    assert not np.allclose(antes, despues), (
        "las dos vistas del blanco son la misma: una de las dos no se midió "
        "en su propio momento")
    # Igual cantidad de puntos por dominio, que es lo que el ojo lee como
    # cobertura sin que nadie se lo diga.
    assert miembros["source"].shape == despues.shape == antes.shape


def test_every_position_carries_its_own_bagsets_class_in_bag_order(cell, familia,
                                                                   tiny):
    """The class channel is the bagset's own labels, bag by bag, in order.

    **What this test can and cannot see, measured rather than assumed.** On this
    material the two domains' label vectors COINCIDE -- `bags.build` cuts twelve
    bags per class in class order, so both read `0`x12, `1`x12, `2`x12 -- and
    the test asserts that they do before claiming anything, because an
    assertion that the two differ would be green on a fixture where swapping
    them changes nothing. So what the equality below catches is a MISORDERED
    class channel (labels of a role, or of a permutation, instead of all bags in
    order); the crossing of the two DOMAINS is invisible here and is measured by
    the next test, on the positions, where the two are genuinely different.
    """
    arm = next(iter(cell["latent"]))
    miembros = cell["latent"][arm]
    material = toy_run.material_for(familia, 0, tiny, "mlp")

    for dominio, miembro in (("source", "classSource"), ("target", "classTarget")):
        esperado = material[dominio].labels.numpy()
        assert len(set(esperado.tolist())) > 1, (
            f"las etiquetas de {dominio} son constantes en este material: una "
            "igualdad elemento a elemento contra ellas no podría fallar")
        assert np.array_equal(miembros[miembro], esperado.astype(np.int16)), (
            f"el canal de clase de {dominio} no son las etiquetas de ese "
            f"dominio en orden de bolsa: dice {miembros[miembro].tolist()} y el "
            f"material dice {esperado.tolist()}")
        assert np.array_equal(miembros[f"bag{dominio.capitalize()}"],
                              np.arange(len(esperado), dtype=np.int16))

    assert np.array_equal(miembros["classSource"], miembros["classTarget"]), (
        "los dos vectores de clase dejaron de coincidir: el docstring de este "
        "test explica qué deja de poder ver si lo hacen, y hay que releerlo "
        "antes de cambiar la aserción de arriba")


def test_the_domain_channel_is_not_crossed(familia, tiny):
    """Each view comes from ITS bagset through ITS encoder, and the three
    crossings that would look identical in the artefact do not reproduce it.

    Four pairings of (material, encoder) exist and the artefact names one per
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


def test_the_artefact_round_trips_the_positions_through_real_files(
        tmp_path, cell, tiny, familia):
    """The positions cross the join: `guardar` writes `.npz` members, `cargar`
    reads those files, and the arrays compared are the ones that travelled.

    Exactly, not approximately. They are written as the dtype they were
    computed in, so a round trip that needed a tolerance would be a round trip
    that changed them.
    """
    barrido = _barrido_from(cell, tiny, familia)
    toy_run.guardar(tmp_path, barrido)

    with np.load(tmp_path / f"{familia.nombre}.npz", allow_pickle=False) as z:
        miembros = {m.split(toy_run.SEP)[-1] for m in z.files}
    assert set(toy_run.LATENT_MEMBERS) <= miembros, (
        f"el `.npz` no trae {sorted(set(toy_run.LATENT_MEMBERS) - miembros)}")

    leido = toy_run.cargar(tmp_path)
    latentes = leido["families"][familia.nombre]["latent"]
    assert set(latentes) == set(barrido["perFamily"][familia.nombre]["latent"]), (
        "la vuelta perdió o inventó un handle de posiciones")
    for handle, escrito in barrido["perFamily"][familia.nombre]["latent"].items():
        for nombre in toy_run.LATENT_MEMBERS:
            assert np.array_equal(latentes[handle][nombre], escrito[nombre]), (
                f"{handle}/{nombre} no volvió igual de disco")


def test_a_view_that_never_reached_disk_is_refused_on_the_way_out(
        tmp_path, cell, tiny, familia):
    """An absent view draws an EMPTY panel, and an empty panel reads exactly
    like a measurement that came out flat -- so it is refused where the write
    happens and not discovered in the other notebook."""
    barrido = _barrido_from(cell, tiny, familia)
    latentes = barrido["perFamily"][familia.nombre]["latent"]
    handle = next(iter(latentes))
    latentes[handle] = {k: v for k, v in latentes[handle].items()
                        if k != "targetAfter"}
    with pytest.raises(KeyError, match="does not carry"):
        toy_run.guardar(tmp_path, barrido)


#: The members that are neither a view nor the trajectory: one value per bag,
#: or a per-step trace of the loss. Named here so the cost test below partitions
#: `LATENT_MEMBERS` instead of assuming what is left over.
CANALES_POR_BOLSA = ("classSource", "classTarget", "bagSource", "bagTarget",
                     "predictionSource")
MIEMBROS_DE_LA_TRAYECTORIA = ("trajectorySteps", "trajectoryTarget",
                              "trajectoryPrediction")


def test_the_positions_cost_three_views_of_bags_and_not_of_instances(cell, tiny):
    """What the artefact pays, crossed against the declared shape, by category.

    The decision not to write the instance embeddings is in `LATENT_MEMBERS`'
    own docstring; this is what makes it checkable. A view is
    `bags x MLP_OUTPUT_DIM` float32, so a view silently widened to one row per
    instance -- thirty times the bytes, the same member name, the same dtype --
    fails here instead of being noticed when the full scale's artefact arrives.

    **Three categories and not two, which is a correction.** This used to assert
    that everything outside the three views was smaller than one view, because
    everything outside them was one integer per bag. The trajectory is not that
    and was never going to be: its whole point is one position per bag PER
    FRAME, so it is the bulk of the artefact by construction and an assertion
    that it stays below one view would be an assertion that it does not exist.
    What replaces it is the arithmetic the declaration already states -- frames
    times bags times (position + decision) -- so a trajectory widened to
    instances, or one that quietly stopped honouring the stride, still fails
    here.

    `LATENT_MEMBERS` is PARTITIONED rather than filtered, so a member added
    later belongs to a category somebody named or this test says so.
    """
    arm = next(iter(cell["latent"]))
    miembros = cell["latent"][arm]
    escala = cell["rows"][arm]["scale"]
    bolsas = escala["bags_per_class"] * escala["classes"]
    categorias = (set(toy_run.LATENT_VIEWS) | set(CANALES_POR_BOLSA)
                  | set(MIEMBROS_DE_LA_TRAYECTORIA) | {"phaseTwoHistory"})
    assert categorias == set(toy_run.LATENT_MEMBERS), (
        sorted(categorias ^ set(toy_run.LATENT_MEMBERS)),
        "hay un miembro del artefacto que no cae en ninguna categoría de costo")

    por_vista = bolsas * config.MLP_OUTPUT_DIM * 4
    for vista in toy_run.LATENT_VIEWS:
        assert miembros[vista].nbytes == por_vista, (
            f"la vista {vista} ocupa {miembros[vista].nbytes} bytes y "
            f"{bolsas} bolsas de {config.MLP_OUTPUT_DIM} dimensiones en "
            f"float32 son {por_vista}: no es una posición por bolsa")
    for canal in CANALES_POR_BOLSA:
        assert miembros[canal].nbytes == bolsas * 2, (
            f"el canal {canal} ocupa {miembros[canal].nbytes} bytes y "
            f"{bolsas} bolsas en int16 son {bolsas * 2}")

    cuadros = len(miembros["trajectorySteps"])
    assert cuadros == len(toy_run.frame_steps(tiny,
                                              cell["rows"][arm]["phase2Steps"]))
    por_cuadro = bolsas * config.MLP_OUTPUT_DIM * 4 + bolsas * 2
    trayectoria = sum(miembros[m].nbytes for m in MIEMBROS_DE_LA_TRAYECTORIA
                      if m != "trajectorySteps")
    assert trayectoria == cuadros * por_cuadro, (
        f"la trayectoria ocupa {trayectoria} bytes y {cuadros} cuadros de "
        f"{por_cuadro} son {cuadros * por_cuadro}: o una vista por cuadro "
        "dejó de ser una posición por bolsa, o los cuadros no son los que el "
        "cronograma declara")
    # The trajectory IS the bulk, and saying so is the point of the stride: at
    # the full scale it is twenty-one frames of this and the three views stay
    # three.
    assert trayectoria > 3 * por_vista


# --------------------------------------- the two named axes, and the conditional

def test_the_record_names_the_two_axes_and_the_value_this_run_used(tiny):
    """`weight_cross_block` and `detach_target_entropy` have names, and the
    record says which pole each run was measured under.

    **The value is DERIVED from the signature, never written down here.** Each
    axis is keyword-only with CREDA's declared value as its default, and
    `wiring.py`'s call sites pass neither -- so the effective value IS the
    default, and `flags()` reads it with `inspect.signature` rather than
    carrying a literal that could drift away from the function.

    **Which is only true while the call sites override nothing**, so that is
    asserted too, from the syntax tree of the file that holds them. A later
    session that threads a value through reddens this rather than leaving the
    record quietly wrong -- and the record would be wrong in the one direction
    that matters, claiming the published pole while the run used the other.

    **And each default is crossed against CREDA's own declaration**, read out
    of `toy_experiment.HP_2`, so neither side can move alone.
    """
    import ast
    import inspect

    from CREDA.toy_experiment import HP_2
    from MIL_CREDA.conditional import weighted_blocks
    from MIL_CREDA.global_term import balanced_dependency_score

    flags = toy_run.flags(tiny, DEVICE, ("lunas",),
                          [{"secondsPerPhase1Epoch": 1.0,
                            "secondsPerPhase2Step": 1.0}], 1.0)
    axes = flags["methodAxes"]
    assert set(axes) == {"weightCrossBlock", "detachTargetEntropy"}, sorted(axes)

    pairs = {
        "weightCrossBlock": (weighted_blocks, "weight_cross_block",
                             HP_2["CRUZADO_SOPESADO"]),
        "detachTargetEntropy": (balanced_dependency_score,
                                "detach_target_entropy",
                                HP_2["RHO_T_SIN_GRAD"]),
    }
    # The two poles must DIFFER, or "the record says which pole" is satisfied
    # by one constant printed twice.
    assert pairs["weightCrossBlock"][2] != pairs["detachTargetEntropy"][2], (
        "CREDA declares the same value for both axes, so this test cannot tell "
        "a record that reads each signature from one that prints a constant")

    for key, (function, parameter, credas) in pairs.items():
        entry = axes[key]
        default = inspect.signature(function).parameters[parameter].default
        assert entry["value"] is default, (key, entry["value"], default)
        assert entry["value"] is credas, (
            key, entry["value"], credas,
            "the default and CREDA's own declaration disagree: the pole this "
            "run was measured under is not the published one")
        assert entry["parameter"] == parameter
        assert entry["function"].endswith(f".{function.__name__}")
        assert entry["declaredBy"] == "CREDA.toy_experiment.HP_2"
        assert entry["carriedBy"] == "the function's own keyword-only default"

    # And the half that keeps the derivation true: the call sites pass neither.
    import MIL_CREDA_Benchmark.wiring as wiring_module

    tree = ast.parse(inspect.getsource(wiring_module))
    called = {"weighted_blocks": 0, "balanced_dependency_score": 0}
    overridden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.id if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", None))
        if name not in called:
            continue
        called[name] += 1
        for keyword in node.keywords:
            if keyword.arg in ("weight_cross_block", "detach_target_entropy"):
                overridden.append((name, keyword.arg))
    assert all(count > 0 for count in called.values()), (
        called, "the call sites this derivation is about are not in that file")
    assert not overridden, (
        overridden,
        "a call site now passes one of the two axes, so the effective value is "
        "no longer the signature default the record reports. Thread the value "
        "into `flags()` from the same place the call site reads it")
    assert flags["methodAxesNote"].endswith("wiring.py")


def test_the_record_says_the_term_is_conditional_by_construction(tiny):
    """A `1.0` in a table reads as a value somebody chose. This one is not
    chosen -- there is no knob.

    CREDA has `class_mixing`: `0.0` is the pure marginal, `1.0` the pure
    conditional, and `EPOCA_COND` switches between the two extremes at a declared
    step. A table that printed `1.0` for this package beside CREDA's would be
    saying "this run chose the conditional end", which is wrong in the way that
    matters: here the conditional character is not a setting at all, it comes
    from `conditional.py` building the Gram blocks class by class (Eq. 23).

    So the record states the construction and names the absence as an absence.
    The absence is MEASURED over `src/MIL_CREDA/` rather than asserted, because
    "there is no such knob" is a claim about a whole package and a reading that
    stopped at one file could not support it.
    """
    from pathlib import Path

    flags = toy_run.flags(tiny, DEVICE, ("lunas",),
                          [{"secondsPerPhase1Epoch": 1.0,
                            "secondsPerPhase2Step": 1.0}], 1.0)
    entry = flags["conditionalByConstruction"]
    assert entry["isConditional"] is True
    assert entry["byAKnob"] is False
    assert entry["knobHere"] is None
    assert entry["equation"] == "23"
    assert entry["builtBy"] == "MIL_CREDA.conditional.conditional_blocks"
    assert entry["credasKnob"] == "CREDA.models.CREDALoss.class_mixing"

    # The absence, measured where it would live and nowhere narrower.
    package = Path(toy_run.__file__).resolve().parent.parent / "MIL_CREDA"
    assert package.is_dir(), package
    modules = sorted(package.rglob("*.py"))
    assert len(modules) > 1, modules
    carrying = [str(m) for m in modules
                if "class_mixing" in m.read_text(encoding="utf-8")]
    assert not carrying, (
        carrying, "a module of this package names `class_mixing`, so the "
                  "record's claim that there is no such knob is false")
    assert entry["searchedUnder"] == "src/MIL_CREDA/"
    assert entry["modulesSearched"] == len(modules)

    # The pole: CREDA's own package DOES carry it, so the search above is
    # capable of finding the thing it reports absent.
    creda = Path(toy_run.__file__).resolve().parent.parent / "CREDA"
    found = [m.name for m in sorted(creda.rglob("*.py"))
             if "class_mixing" in m.read_text(encoding="utf-8")]
    assert found, (
        "`class_mixing` is nowhere in CREDA either, so the absence reported "
        "for this package distinguishes nothing")


# ------------------------------------------------------------------- the apparatus

def test_the_retired_transmission_is_declared_rather_than_silently_absent(cell,
                                                                          tiny):
    """`config.SOURCE_TO_TARGET_SYNC_STEPS` still exists and this path does not
    use it; the flags say so, with the risk that decision carries."""
    flags = toy_run.flags(tiny, DEVICE, ("lunas",),
                          [{"secondsPerPhase1Epoch": 1.0,
                            "secondsPerPhase2Step": 1.0}], 1.0)
    assert flags["syncRetired"]["constant"] == "config.SOURCE_TO_TARGET_SYNC_STEPS"
    assert flags["syncRetired"]["value"] == config.SOURCE_TO_TARGET_SYNC_STEPS
    assert "below the floor" in flags["syncRetired"]["declaredRisk"]
    assert flags["criteria"]["phase2Stop"].startswith("none:")
    assert flags["criteria"]["phase2Select"].startswith("max(cef)")
    assert flags["criteria"]["phase2Search"] == "max(cef)"


def test_every_normalization_creda_draws_has_a_backbone_here(cell):
    """CREDA draws three normalization rows and this benchmark now declares
    three backbones, one per row, so nothing is absent.

    **This test used to pin the opposite** -- that the third row was missing and
    the absence declared -- and the reason it gave was that `mlp.py` sat outside
    that stretch's authorized surface. That was a fact about a work boundary and
    never about the mathematics, and the row it left out is the CONTROL that
    separates «normalizing helps» from «re-estimating helps». The operator
    lifted the boundary, the backbone exists, and a test still pinning the gap
    would be a test that lies.

    `NORMALIZATION_ABSENT` stays as a declaration rather than being deleted, and
    what holds it honest is that it is asserted EQUAL to the difference against
    CREDA's own list -- in both directions. So it cannot drift into claiming a
    gap that is closed, and it cannot go silent if a row is dropped: the second
    half of this test drops one and watches the difference appear.
    """
    from CREDA import toy_experiment as creda
    from MIL_CREDA_Benchmark import mlp

    assert set(toy_run.NORMALIZATIONS.values()) == set(mlp.MLP_BACKBONES)
    assert list(toy_run.NORMALIZATIONS) == list(creda.POS_FILAS), (
        f"las filas de este paquete son {list(toy_run.NORMALIZATIONS)} y CREDA "
        f"dibuja {list(creda.POS_FILAS)}: el orden es la lectura de la "
        "escalera y de la figura, así que no es un detalle")
    missing = set(creda.POS_FILAS) - set(toy_run.NORMALIZATIONS)
    assert missing == set(toy_run.NORMALIZATION_ABSENT) == set(), (
        f"CREDA dibuja filas que este paquete no tiene: {sorted(missing)}")

    # El rojo alcanzable para la igualdad de arriba: sacada una fila, la
    # diferencia contra CREDA deja de estar vacía. Sin esto, «no falta ninguna»
    # sería cierto de una comparación que no mira nada.
    recortadas = {k: v for k, v in toy_run.NORMALIZATIONS.items()
                  if k != list(toy_run.NORMALIZATIONS)[-1]}
    assert set(creda.POS_FILAS) - set(recortadas) != set()


def test_the_declared_arm_commands_both_terms_and_the_shares_are_a_reading(
        cell, familia, tiny):
    """Each declared component's share of the objective, reported.

    A term that is correctly implemented and multiplied by something tiny
    produces a table of exact zeros that reads like a result; without the share,
    "it had no effect" and "it had no weight" are the same number.

    **The zero pole used to be the floor arm's row and now is a control built
    here.** With one arm every row in a cell carries both terms, so a test that
    only read this row could not tell a share from a default -- and `0.0` is
    exactly what a share would print if it were never computed.
    `_UNADAPTED_CONTROL_SPEC` is run through one phase-2-shaped step on the same
    material and reports zero for both, which is what makes the row's non-zeros
    a measurement.
    """
    row = cell["rows"][ADAPTED]
    assert row["contributionGlobal"] > 0.0
    assert row["contributionLocal"] > 0.0
    assert 0.0 < row["adaptationShare"] < 1.0
    assert row["phase2Steps"] > 0

    payload = _prepared(familia, tiny, arm_id=ADAPTED)
    with toy_run.material_bound(familia, tiny, "mlp"):
        control = _unadapted_control(payload["material"], tiny,
                                     payload["hyper"], DEVICE)
    record = _one_step(familia, tiny, control, payload)
    assert record["contributionGlobal"] == 0.0
    assert record["contributionLocal"] == 0.0
    assert record["contribution"] == 0.0


def test_phase_one_is_trained_once_and_loaded_into_the_arm_that_phase_two_moves(
        familia, tiny, monkeypatch):
    """Phase 1 is fitted ONCE per cell and its state loaded into the arm.

    This used to compare the phase-1 readings of a cell's two arms and assert
    they were identical to the bit. With one declared arm that comparison is a
    row against itself -- it passes whatever `run_cell` does, including an arm
    that trained its own phase 1 -- so what is read instead is the number of
    `train_phase_one` CALLS a cell makes, which is the fact the sharing is
    about and stays reachable at any arm count.

    The control: the call count is asserted against the arms the module declares
    rather than against `1`, so re-declaring a second arm does not quietly turn
    this green on a cell that fitted phase 1 twice.
    """
    calls = []
    real = toy_run.train_phase_one

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(toy_run, "train_phase_one", spy)
    cell = toy_run.run_cell(familia, "ninguna", 0, tiny,
                            toy_run.declared_hyper(tiny), DEVICE)
    assert len(calls) == 1, (
        f"phase 1 was fitted {len(calls)} times for one cell of "
        f"{len(toy_run.TOY_ARM_IDS)} arm(s); it is fitted once and shared")
    assert len(cell["rows"]) == len(toy_run.TOY_ARM_IDS)
    # And the state really reached the row: a cell that threw the fit away would
    # report a restored accuracy that no epoch of the fit ever read.
    assert cell["rows"][ADAPTED]["phase1ValidAccuracy"] == pytest.approx(
        cell["phase1"]["bestValidAccuracy"], abs=1e-9)


def test_phase_one_restores_the_epoch_its_stop_chose(cell):
    """A `state_dict` that is not restored leaves everything green: the reported
    accuracy would be the best epoch's and the weights the last one's."""
    phase_one = cell["phase1"]
    assert phase_one["restoredValidAccuracy"] == pytest.approx(
        phase_one["bestValidAccuracy"], abs=1e-9)
    assert phase_one["bestEpoch"] < phase_one["epochs"]


def test_the_cost_is_measured_per_epoch_and_per_step_and_the_bound_is_named(cell,
                                                                            tiny):
    """The forecast is built from measured RATES, not from the sweep's wall time
    multiplied by ratios -- see `work_counts` for why the ratios do not
    compose."""
    report = toy_run.cost_report([cell["rates"]], tiny, cell["elapsedSeconds"],
                                 families=1)
    assert report["secondsPerPhase1Epoch"] > 0.0
    assert report["secondsPerPhase2Step"] > 0.0
    assert report["measuredAt"] == "test"
    assert report["forecastFullSeconds"] > report["sweepSeconds"]
    assert "upper bound" in report["caveat"]
    assert report["instanceScale"] == pytest.approx(
        toy_run.FULL.instances_per_bag / tiny.instances_per_bag)


def test_the_work_counts_multiply_the_loops_they_belong_to(tiny):
    """The defect this replaced: multiplying every repetition ratio together
    treats the trial counts and the seed counts as if they multiplied the same
    loop. They multiply the searches and the cells respectively, and the counts
    below are read off those loops."""
    here = toy_run.work_counts(tiny, families=1)
    norms = len(toy_run.NORMALIZATIONS)
    expected_phase1 = (tiny.trials_1 * norms * len(tiny.search_seeds)
                       + norms * len(tiny.search_seeds)
                       + norms * len(tiny.report_seeds))
    assert here["phase1Trainings"] == expected_phase1
    # Every declared arm runs phase 2. It used to be `arms - 1`, because the
    # retired floor arm declared no adaptation term; at one arm that
    # subtraction forecasts ZERO phase-2 runs for work that happens.
    expected_phase2 = (tiny.trials_2 * norms * len(tiny.search_seeds)
                       + norms * len(tiny.report_seeds)
                       * len(toy_run.TOY_ARM_IDS))
    assert here["phase2Runs"] == expected_phase2
    assert here["arms"] == len(toy_run.TOY_ARM_IDS)
    # The reachable red for the subtraction: at one arm the two formulas differ
    # by the whole per-cell block, so this assertion is not vacuous.
    assert expected_phase2 != (tiny.trials_2 * norms * len(tiny.search_seeds)
                               + norms * len(tiny.report_seeds)
                               * (len(toy_run.TOY_ARM_IDS) - 1))
    assert here["phase1EpochCeiling"] == here["phase1Trainings"] * tiny.epochs_1
    full = toy_run.work_counts(toy_run.FULL, families=1)
    assert full["phase1Trainings"] > here["phase1Trainings"]


def test_the_full_scale_forecasts_to_its_own_work_at_the_measured_rates():
    """At the full scale nothing is scaled: the forecast is the measured rates
    times the work the full scale actually does."""
    rates = [{"secondsPerPhase1Epoch": 1.0, "secondsPerPhase2Step": 2.0}]
    report = toy_run.cost_report(rates, toy_run.FULL, 1.0, families=3)
    work = toy_run.work_counts(toy_run.FULL, families=3)
    assert report["instanceScale"] == pytest.approx(1.0)
    assert report["forecastFullSeconds"] == pytest.approx(
        work["phase1EpochCeiling"] * 1.0 + work["phase2StepCeiling"] * 2.0)


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
    was. Inside a function, `config.ARMS`/`ARMS_BY_ID` are legitimate: `flags`
    reports which declaration this run is NOT using, and `build_arm`'s refusal
    names the digit arms to say they are not reachable.

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


def test_the_adapted_arms_three_flags_equal_digit_gs_and_the_schedule_is_what_differs():
    """The separation's premise, after the confidence weighting came back on --
    and it is the OPPOSITE premise from the one this test used to assert.

    The test here before read: "the digit ladder declares no arm with an
    adaptation term, the local correspondence AND no confidence weighting", and
    the separation rested on that. With `weighting` back to `True` the adapted
    arm's three mechanism flags are EXACTLY digit `G`'s, so that premise is
    false and asserting it would pin a sentence the code contradicts.

    What is measured instead is both halves of the true statement: the flags
    match, AND the protocol does not. The second half is the one that carries
    the separation now, and it is measured on the thing that differs -- the
    coefficient. `harness.run_one` (the digit path) takes its coefficient from
    `harness.ramp`; `toy_run.adapt_phase_two` does not call `ramp` at all.
    """
    adapted = toy_run.TOY_ARMS_BY_ID[ADAPTED]
    flags_of = lambda arm: (arm["adaptation"], arm["weighting"], arm["local"],
                            arm["unit"])

    # The match, found by SEARCHING the digit declaration rather than naming
    # `"G"`: if the digit ladder is re-declared, this reports whichever arm
    # matches, or none.
    twins = [arm["id"] for arm in config.ARMS
             if flags_of(arm) == flags_of(adapted)]
    assert twins == ["G"], twins
    # The control that this predicate discriminates at all: a DIFFERENT shape
    # matches a DIFFERENT digit arm, and the two answers are not the same
    # answer. The shape used to be the retired floor arm's; it is now a local
    # spec, because reading the predicate off something this module declares
    # would make the control depend on the arm list it is controlling.
    assert [arm["id"] for arm in config.ARMS
            if flags_of(arm) == flags_of(_UNADAPTED_CONTROL_SPEC)] == ["B"]
    assert flags_of(_UNADAPTED_CONTROL_SPEC) != flags_of(adapted)

    # The adapted arm still carries the method.
    assert adapted["adaptation"] == "milcreda"
    assert adapted["weighting"] is True
    assert adapted["local"] is True

    # And the declaration is still this module's own, which the id disjointness
    # makes structural: no id declared here is an id declared there.
    assert not (set(toy_run.TOY_ARMS_BY_ID) & set(config.ARMS_BY_ID))

    # THE HALF THAT CARRIES THE SEPARATION: the schedule. `harness.ramp`'s
    # warmup at the declared delta is 0.0 on the first step and never reaches
    # its ceiling, so a fixed 1 is not this curve at any step -- which is why
    # `adapt_phase_two` dropped the call instead of passing `ceiling=1.0`.
    steps = 6
    curve = [harness.ramp(step, steps, "milcreda",
                          delta=float(config.RAMP_DELTA), ceiling=1.0)
             for step in range(steps)]
    assert curve[0] == 0.0, curve
    assert all(value != toy_run.FIXED_COEFFICIENT for value in curve), curve
    assert curve == sorted(curve) and curve[0] != curve[-1], curve
    # The digit path reads that curve; this path reads the constant.
    assert "ramp(" in inspect.getsource(harness.run_one)
    assert "ramp(" not in inspect.getsource(toy_run.adapt_phase_two)


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

def _one_step(familia, tiny, model, payload,
              coefficient=toy_run.FIXED_COEFFICIENT):
    """One phase-2 training step at Eq. (39)'s declared coefficient, so a
    contribution IS its term: `training_step` books `contributionGlobal =
    coefficient * global_term`.

    The default is read off `toy_run.FIXED_COEFFICIENT` and not written as
    `1.0` here: a test helper spelling the constant again would keep passing a
    one into a module that had moved off it.
    """
    with toy_run.material_bound(familia, tiny, "mlp"):
        x, y = toy_run._batch(payload["source_train"],
                              range(len(payload["source_train"])), DEVICE)
        generator = torch.Generator().manual_seed(4242)
        toy_run.set_phase_two_modes(model)
        return model.training_step(x, y, coefficient, generator,
                                   ramp_local=coefficient)


def test_the_adapted_arm_computes_a_nonzero_global_term_and_a_nonzero_local_one(
        familia, tiny):
    """BOTH phase-2 losses, and both measured away from zero.

    Two assertions and not one, in both directions: "the term did nothing" and
    "the term was never called" print the same `0.0`, so each term is read on
    its own. The step runs at coefficient one so the contribution the record
    carries IS the term, with no schedule between the assertion and the
    mathematics.
    """
    payload = _prepared(familia, tiny, arm_id=ADAPTED)
    record = _one_step(familia, tiny, payload["model"], payload)
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


def test_an_unadapted_spec_computes_neither_term(familia, tiny):
    """A spec with no adaptation term runs none, and the declared arm on the
    very same batch runs both -- which is what makes the zeros a reading and not
    a default.

    The spec is `_UNADAPTED_CONTROL_SPEC` and NOT an arm of this toyset: the
    floor arm this used to be written against was retired, and reintroducing it
    as a declared arm to keep a test's control would put a floor back in the
    run. What a control needs is to be built and stepped, which `wiring.Arm`
    offers on any spec dict; what it must not be is reachable as a row of the
    artefact, and `build_arm` refuses it by name.
    """
    payload = _prepared(familia, tiny, arm_id=ADAPTED)
    with toy_run.material_bound(familia, tiny, "mlp"):
        control = _unadapted_control(payload["material"], tiny,
                                     payload["hyper"], DEVICE)
        assert control.target_encoder is None, (
            "the control built a theta_t; its spec declares no adaptation term")
        with pytest.raises(ValueError, match="unknown toyset arm"):
            toy_run.build_arm(_UNADAPTED_CONTROL_SPEC["id"],
                              payload["material"], tiny, payload["hyper"],
                              DEVICE, 0)
    record = _one_step(familia, tiny, control, payload)
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
    adapted_payload = _prepared(familia, tiny, arm_id=ADAPTED)
    adapted = _one_step(familia, tiny, adapted_payload["model"],
                        adapted_payload)
    assert adapted["contributionGlobal"] > 0.0
    assert adapted["contributionLocal"] > 0.0
    assert not math.isnan(adapted["targetBagEntropy"])


def test_the_adapted_arm_declares_the_weighting_and_all_three_mechanisms_run(
        familia, tiny, monkeypatch):
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
    payload = _prepared(familia, tiny, arm_id=ADAPTED)
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
    assert float((toy_run._parameter_vector(model.encoder)
                  - toy_run._parameter_vector(twin.encoder)).norm()) == 0.0
    assert float((toy_run._parameter_vector(model.target_encoder)
                  - toy_run._parameter_vector(twin.target_encoder)).norm()) == 0.0

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

    on = _one_step(familia, tiny, model, payload)

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
    off = _one_step(familia, tiny, twin, payload)

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


def test_the_weighting_divides_the_global_term_and_the_run_measures_by_how_much(
        familia, tiny):
    """What turning the weighting on COSTS, measured by the module the run uses
    -- and the direction asserted, because this is the finding the operator
    asked to have travel with the tables.

    Two readings, and neither is a constant:

    - `weighting_cost` returns a ratio below one ON THIS FIXTURE, so the
      sub-convex `omega_c` really does divide the global term rather than
      merely re-weighting it. The CONTROL is the same function on a spec that
      declares no weighting (`_UNADAPTED_CONTROL_SPEC`, a test control and not
      an arm since the floor arm was retired): `None`, never a ratio.

      **`< 1` is this fixture's measurement and not a law**, and that is worth
      stating here because the field it describes is read in every table.
      `omega_c` sums to AT MOST one, so it divides the term where the routed
      confidences are low and leaves it essentially alone where they are not --
      measured on the pilot at twelve bags per class, the ratio runs 0.0398 to
      1.0411 over the six cells, and the cell it exceeds one on
      (`rollo`/`batchnorm`) is the one whose `Sum(omega)` came back at 0.9994.
      Nothing here may be read as "the weighting always divides".
    - that ratio equals the one the two arms of the test above compute
      end-to-end, which is what makes the field in every row the same number a
      reader could recompute from the two contributions.
    """
    payload = _prepared(familia, tiny, arm_id=ADAPTED)
    model = payload["model"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        cost = toy_run.weighting_cost(model, payload["source_train"], DEVICE, 0)
    assert cost is not None
    assert 0.0 < cost["ratio"] < 1.0, cost
    # `omegaSum` is `nan` since 2026-10-18: r32 took the confidences out of
    # Eq. (37)'s weights, so there is no `sum_c omega_c` to read. The key is
    # kept so a reader of `barrido.json` meets "unavailable" and not a gap.
    assert math.isnan(cost["omegaSum"]), cost
    assert cost["globalWeighted"] < cost["globalUnweighted"], cost
    assert cost["activeClasses"] >= 1, cost
    # The ratio IS the two terms divided -- not a second spelling of it.
    assert cost["ratio"] == pytest.approx(
        cost["globalWeighted"] / cost["globalUnweighted"])

    # THE CONTROL: a spec declaring no weighting has no cost to read, so the
    # answer is `None` rather than a number. The guard survives the floor arm's
    # retirement and is exercised here; what no longer exists is a path from
    # `run_cell` to it, which is why the `nan` fallbacks in `run_cell` are gone.
    with toy_run.material_bound(familia, tiny, "mlp"):
        control = _unadapted_control(payload["material"], tiny,
                                     payload["hyper"], DEVICE)
        assert toy_run.weighting_cost(
            control, payload["source_train"], DEVICE, 0) is None


def test_every_row_of_a_cell_carries_the_weighting_cost_as_a_real_number(cell):
    """Every row of a cell carries the three cost fields, and none of them is
    `nan`.

    The `nan` half of this test was the floor's row: it declared no weighting,
    so there was no cost to read and `0.0` would have been the reading of a term
    the weighting annihilated. With one arm declared every row carries the
    weighting, so `nan` is now the FAILURE and not the other pole -- a `nan`
    here means `weighting_cost` returned `None` for an arm that declares the
    flag, which is a defect and no longer a state.

    The other pole lives in
    `test_the_weighting_divides_the_global_term_and_the_run_measures_by_how_much`,
    on a spec that is not an arm.
    """
    # **`< 1.0` left this assertion on 2026-10-18, and `omegaSum` with it.**
    # The strict upper bound encoded r29's gate: Eq. (37)'s weights were
    # sub-convex, so turning the weighting ON could only DIVIDE the global
    # term and the ratio was below one by construction. r32 replaced those
    # weights with `n_c^s / N_s`, identical on both arms, so what the ratio
    # measures now is Eq. (26)'s `D` alone -- and that can leave the term
    # unchanged or raise it. Measured: exactly 1.0 on this cell, and 1.63 in
    # the configuration
    # `test_the_weighting_divides_the_global_term_and_the_run_measures_by_how_much`
    # records. So the claim is that the field is a real positive reading and
    # consistent with its own two components; `nan` is still the failure.
    adapted = cell["rows"][ADAPTED]
    ratio = adapted["weightingCostRatio"]
    assert ratio > 0.0 and math.isfinite(ratio), ratio
    assert adapted["weightingCostGlobalUnweighted"] > 0.0
    # `omegaSum` is `nan` on every row written from 2026-10-18: r32 took the
    # confidences out of Eq. (37)'s weights, so `sum_c omega_c` has no
    # counterpart. The key is kept so a reader of `barrido.json` meets
    # "unavailable" rather than a field that vanished between artefacts.
    assert math.isnan(adapted["weightingCostOmegaSum"])
    for row in cell["rows"].values():
        for field in ("weightingCostRatio", "weightingCostGlobalUnweighted"):
            assert not math.isnan(row[field]), (row["arm"], field)
        # Carried on every row, and carried as `nan` on every row.
        assert "weightingCostOmegaSum" in row, row["arm"]


# ----------------------------------- the two coefficients, fixed at one

def test_the_coefficient_reaching_training_step_is_exactly_one_at_every_step(
        familia, tiny, monkeypatch):
    """Eq. (39)'s two coefficients, read where they ARRIVE rather than where
    they are declared: `wiring.Arm.training_step`'s own `ramp` and `ramp_local`
    arguments, captured on every step of a real `adapt_phase_two`.

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
    payload = _prepared(familia, tiny, arm_id=ADAPTED)
    model = payload["model"]
    captured: list[tuple] = []
    real_step = wiring.Arm.training_step

    def spy(self, bags_, labels, ramp, generator, ramp_local=None):
        captured.append((ramp, ramp_local))
        return real_step(self, bags_, labels, ramp, generator,
                         ramp_local=ramp_local)

    monkeypatch.setattr(wiring.Arm, "training_step", spy)
    with toy_run.material_bound(familia, tiny, "mlp"):
        phase_two = toy_run.adapt_phase_two(
            model, payload["source_train"], payload["source_valid"], DEVICE,
            tiny, payload["hyper"], torch.Generator().manual_seed(7))

    assert len(captured) >= 2, captured
    assert len(captured) == len(phase_two["curve"]), (len(captured),
                                                      len(phase_two["curve"]))
    for coefficient, coefficient_local in captured:
        assert coefficient == toy_run.FIXED_COEFFICIENT, captured
        assert coefficient_local == toy_run.FIXED_COEFFICIENT, captured
        assert coefficient == coefficient_local
    assert toy_run.FIXED_COEFFICIENT == 1.0

    # And the curve the record publishes says the same, under names that do not
    # claim a schedule.
    assert all(point["coefficient"] == toy_run.FIXED_COEFFICIENT
               for point in phase_two["curve"])
    assert all(point["coefficientLocal"] == toy_run.FIXED_COEFFICIENT
               for point in phase_two["curve"])
    assert not (set(phase_two["curve"][0]) & {"ramp", "rampLocal"}), (
        "the curve still carries a key called ramp; 1.0 under that name reads "
        "as a schedule that saturated")


def test_no_ramp_is_evaluated_anywhere_on_this_path(familia, tiny, monkeypatch):
    """`harness.ramp` is not called by a whole cell -- and a fixed 1 is NOT that
    ramp evaluated at its ceiling, which is the half that makes the first
    assertion worth making.

    Three parts, because the first alone would pass against a broken patch and
    the first two alone would pass if the ramp happened to return one:

    1. a spy on `harness.ramp` records nothing across a full `run_cell`;
    2. THE CONTROL: the same spy, called directly, records -- so the patch is
       live and part 1 is a fact about `run_cell`;
    3. the curve that spy would have produced is measured: `0.0` on the first
       step and never `1.0` on any step, at the declared delta and a ceiling of
       one. So routing `FIXED_COEFFICIENT` through `ramp` would have trained
       under a schedule, not at a constant.
    """
    calls: list[tuple] = []
    real_ramp = harness.ramp

    def spy(epoch, epochs, family, delta=None, ceiling=None):
        calls.append((epoch, epochs, family, delta, ceiling))
        kwargs = {}
        if delta is not None:
            kwargs["delta"] = delta
        if ceiling is not None:
            kwargs["ceiling"] = ceiling
        return real_ramp(epoch, epochs, family, **kwargs)

    monkeypatch.setattr(harness, "ramp", spy)
    toy_run.run_cell(familia, "batchnorm", 0, tiny,
                     toy_run.declared_hyper(tiny), DEVICE)
    assert calls == [], calls

    # THE CONTROL: the patch is live.
    spy(0, 10, "milcreda", delta=20.0, ceiling=1.0)
    assert len(calls) == 1, calls

    # And what the ramp would have done, so "no ramp" is a difference.
    steps = tiny.epochs_2 * toy_run.STEPS_PER_EPOCH_DECLARED
    curve = [real_ramp(step, steps, "milcreda",
                       delta=float(config.RAMP_DELTA), ceiling=1.0)
             for step in range(steps)]
    assert curve[0] == 0.0, curve
    assert toy_run.FIXED_COEFFICIENT not in curve, curve
    assert curve[-1] > curve[0], curve


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
    missing name here rather than as a silent hole. A choke point is one early
    return away from being bypassed, and each of these four is reachable on its
    own from a notebook and from the two searches.
    """
    guarded = ("build_arm", "train_phase_one", "adapt_phase_two", "run_cell",
               "material_bound", "_celda_de_un_brazo")
    for name in guarded:
        source = inspect.getsource(getattr(toy_run, name))
        assert "refuse_retired_hyper(hyper)" in source, name
    # And every module-level function whose signature takes a `hyper`
    # parameter is accounted for -- DERIVED from the module rather than listed,
    # so a function added later with a hyper dict and no guard turns this red
    # instead of being assumed absent.
    #
    # `refuse_retired_hyper` is the guard itself. `_resolved_sigma` is private
    # and is the one exemption with a reason: it only ever receives a dict the
    # callers above already refused, and it calls `build_arm`, which refuses
    # again -- so it is covered twice over by the entries that reach it.
    #
    # `material_bound` joined `guarded` when `dim` became searchable: it is a
    # public entry, it now READS the hyper dict, and a guard it skipped would
    # be the one hole the derivation below exists to make visible.
    #
    # `_celda_de_un_brazo` joined it on 2026-10-20 with section 3. It is private
    # and it only forwards to `run_cell`, so it looks like `_resolved_sigma`'s
    # exemption and is NOT given one: the rule here is that every entry taking a
    # hyper dict refuses at its own door, and a forwarder that does not look
    # leaves the refusal to an implementation detail of the forwarder. It calls
    # the guard, so it belongs in the list above rather than in the exemptions.
    takes_hyper = {
        name for name, value in vars(toy_run).items()
        if inspect.isfunction(value)
        and "hyper" in inspect.signature(value).parameters}
    # `material_bound` is a generator wrapped by `contextlib.contextmanager`,
    # so `vars()` holds the wrapper and not the function; its guard is asserted
    # in the loop above and its name is added here rather than derived.
    takes_hyper.add("material_bound")
    assert takes_hyper == set(guarded) | {"refuse_retired_hyper",
                                          "_resolved_sigma"}, sorted(takes_hyper)
    assert "build_arm(" in inspect.getsource(toy_run._resolved_sigma)


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

    def via_train_phase_one(hyper):
        with toy_run.material_bound(familia, tiny, "mlp"):
            source_train, source_valid, _ = bags.roles(material["source"])
            model = toy_run.build_arm(ADAPTED, material, tiny,
                                      toy_run.declared_hyper(tiny), DEVICE, 0)
            return toy_run.train_phase_one(model, source_train, source_valid,
                                           DEVICE, tiny, hyper,
                                           torch.Generator().manual_seed(1))

    def via_adapt_phase_two(hyper):
        with toy_run.material_bound(familia, tiny, "mlp"):
            source_train, source_valid, _ = bags.roles(material["source"])
            model = toy_run.build_arm(ADAPTED, material, tiny,
                                      toy_run.declared_hyper(tiny), DEVICE, 0)
            toy_run.copy_source_to_target(model)
            return toy_run.adapt_phase_two(model, source_train, source_valid,
                                           DEVICE, tiny, hyper,
                                           torch.Generator().manual_seed(1))

    def via_run_cell(hyper):
        """`run_cell`'s OWN guard, not the one it would hit on the way down.

        Measured, not assumed: removing `refuse_retired_hyper` from `run_cell`
        leaves the refusal in place, because `run_cell` calls `build_arm` and
        `build_arm` refuses. So the `ValueError` alone cannot tell the two
        apart, and the test would be green over a `run_cell` that bends
        `config`'s globals and builds the material before anything says no.

        `build_arm` is counted. `run_cell` must refuse before reaching it.
        """
        calls = []
        real_build_arm = toy_run.build_arm

        def counting(*args, **kwargs):
            calls.append(args[0])
            return real_build_arm(*args, **kwargs)

        toy_run.build_arm = counting
        try:
            return toy_run.run_cell(familia, "batchnorm", 0, tiny, hyper, DEVICE)
        except ValueError:
            # The count matters only on the refusing path. A clean hyper dict
            # is SUPPOSED to reach `build_arm` -- that is the control these
            # tests run afterwards -- so asserting zero calls unconditionally
            # would redden the control instead of the defect.
            assert calls == [], (
                "run_cell reached build_arm before refusing, so the refusal "
                f"came from build_arm and not from run_cell: {calls}")
            raise
        finally:
            toy_run.build_arm = real_build_arm

    return (via_build_arm, via_train_phase_one, via_adapt_phase_two,
            via_run_cell)


def test_the_four_searched_dimensions_are_suggested_and_the_retired_four_are_not(
        familia, tiny):
    """What the two searches ACTUALLY suggest, read off Optuna's own trial
    parameters rather than off `SEARCH_RANGES`.

    The dict is a declaration and the trials are the behaviour: a dimension
    removed from the dict and left in the `objective` would still be searched,
    and a dimension listed in the dict and never suggested would be a promise.
    This reads the `params` of every completed trial of both searches.

    `kernelSigma` enters phase 1 as `sigmaMultiplier` -- the grid is relative to
    a measured distance (`sigma_grid`), so what the search chooses is the
    multiplier and what it resolves to travels in the record's own
    `kernelSigma`. Both are checked, so neither reads as absent.
    """
    scale = dataclasses.replace(tiny, trials_1=2, trials_2=2)
    material = {seed: toy_run.build_family(familia.nombre, seed, scale)
                for seed in scale.search_seeds}
    chosen1, record1 = toy_run.search_phase_one(material, scale, DEVICE,
                                                report=lambda *_: None)
    chosen2, record2 = toy_run.search_phase_two(material, chosen1, scale, DEVICE,
                                                report=lambda *_: None)

    suggested1 = {name for trial in record1["perTrial"] for name in trial["params"]}
    suggested2 = {name for trial in record2["perTrial"] for name in trial["params"]}
    assert record1["perTrial"] and record2["perTrial"]

    # The retired four: suggested by neither search.
    assert not (suggested1 & set(toy_run.RETIRED_RANGES)), sorted(
        suggested1 & set(toy_run.RETIRED_RANGES))
    assert not (suggested2 & set(toy_run.RETIRED_RANGES)), sorted(
        suggested2 & set(toy_run.RETIRED_RANGES))
    # Nor chosen, so neither reaches a hyper dict -- which the guard would
    # refuse anyway, and this says the search never offers it the chance.
    assert not (set(chosen1) & set(toy_run.RETIRED_RANGES))
    assert not (set(chosen2) & set(toy_run.RETIRED_RANGES))

    # The four that remain: every one suggested, by the phase the split assigns
    # it to. `kernelSigma` through its multiplier, and resolved in the record.
    assert "sigmaMultiplier" in suggested1
    assert "kernelSigma" in chosen1 and chosen1["kernelSigma"] > 0.0
    for name in ("attentionGamma", "attentionTemperature"):
        assert name in suggested1, (name, sorted(suggested1))
    assert "dim" in suggested1, sorted(suggested1)
    assert suggested2 == set(toy_run.searched_ranges("phase2")) == {
        "tauLocal", "lr2", "weightDecay2"}, sorted(suggested2)
    # And the phase-1 dimension of this split is not suggested by phase 2,
    # which is what keeps phase 1 trained once and frozen. `dim` is named
    # separately because it is a global the binding bends rather than one of
    # the config-range dimensions, and phase 2 inheriting it from `hyper1` --
    # never suggesting it -- is what lets the cached phase-1 state load.
    assert not (suggested2 & set(toy_run.PHASE_ONE_DIMENSIONS))
    assert "dim" not in suggested2, sorted(suggested2)

    # The record says which four left, so a reader of `barrido.json` holding an
    # eight-dimension search can tell the two runs apart.
    assert set(record2["retired"]) == set(toy_run.RETIRED_RANGES)
    assert record2["fixedCoefficient"] == toy_run.FIXED_COEFFICIENT


# ------------------------------------------- the animated grid, and its frames
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


def test_the_frame_schedule_is_the_stride_with_the_first_and_last_always_in():
    """`frame_steps` keeps every k-th update, plus zero and the last, and nothing
    else.

    Read over a grid of (declared steps, applied steps) rather than one case,
    because the two interesting shapes are a run that stopped ON a multiple of
    the stride and one that stopped between two. Both have to carry the last
    step; only the second has to ADD it.
    """
    # The grid is in EPOCHS now, and `declarados` below is the STEP count they
    # hold: `frame_steps` indexes frames in steps and always did -- a frame is
    # a state of the weights and the weights move once per step.
    for epocas in (1, 3, 13):
        escala = dataclasses.replace(toy_run.PILOT, epochs_2=epocas)
        declarados = epocas * toy_run.STEPS_PER_EPOCH_DECLARED
        zancada = toy_run.frame_stride(escala)
        for corridos in (0, 1, zancada, zancada + 1, declarados):
            pasos = toy_run.frame_steps(escala, corridos)
            assert pasos[0] == 0, (declarados, corridos, pasos)
            assert pasos[-1] == corridos, (declarados, corridos, pasos)
            assert list(pasos) == sorted(set(pasos)), pasos
            interiores = [p for p in pasos[1:-1]]
            assert all(p % zancada == 0 for p in interiores), (
                declarados, corridos, pasos, zancada)
            assert all(0 <= p <= corridos for p in pasos), (corridos, pasos)


def test_the_stride_follows_the_declared_frame_count_and_not_a_literal(tmp_path):
    """Double the frame count the module declares and the grid halves.

    The control is the whole test: the grid is the CEILING IN EPOCHS over the
    declared frame count, and a LITERAL would reproduce today's value while
    leaving any other ceiling animating at the wrong cadence. So the
    declaration is moved and the derived value has to move with it.

    Every expected value is DERIVED here and not written down, because both
    the ceiling and its UNIT moved: the ceiling was 200 steps, became
    `epochs_2` whole passes on 2026-10-19, and the grid itself was re-declared
    in epochs. A literal 10 would have had to be edited to 80 and then to 5,
    which is the same defect this test exists to catch, two scales along.

    The two frame counts at the end are the only literals, and they are the
    measured consequence the change was made for rather than an input.
    """
    # The grid is declared in EPOCHS and the step count is derived from it,
    # which is the direction the two assertions below read in that order.
    assert toy_run.frame_epochs(toy_run.FULL) == (
        toy_run.FULL.epochs_2 // toy_run.TRAJECTORY_FRAMES), (
        toy_run.frame_epochs(toy_run.FULL), toy_run.FULL.epochs_2,
        toy_run.TRAJECTORY_FRAMES)
    vivo = toy_run.frame_stride(toy_run.FULL)
    assert vivo == (toy_run.frame_epochs(toy_run.FULL)
                    * toy_run.STEPS_PER_EPOCH_DECLARED), (
        vivo, toy_run.frame_epochs(toy_run.FULL),
        toy_run.STEPS_PER_EPOCH_DECLARED)

    mutado = _mutated_toy_run(tmp_path, "TRAJECTORY_FRAMES = 20",
                              "TRAJECTORY_FRAMES = 40", "toy_run_frames_40")
    assert mutado.TRAJECTORY_FRAMES == 40
    assert mutado.frame_epochs(mutado.FULL) == mutado.FULL.epochs_2 // 40, (
        mutado.frame_epochs(mutado.FULL), mutado.FULL.epochs_2)
    assert mutado.frame_stride(mutado.FULL) != vivo

    # **Which unit is DECLARED, proven by moving the other one.** Change the
    # pass length and the stride in STEPS moves while the stride in EPOCHS does
    # not: that is the whole content of "the grid is declared in epochs", and
    # without this pole the two assertions above would read identically if the
    # derivation ran the other way round.
    otra_pasada = _mutated_toy_run(
        tmp_path,
        'STEPS_PER_EPOCH_DECLARED = FULL.realized_split()["bagsPerClass"]["train"]',
        'STEPS_PER_EPOCH_DECLARED = FULL.realized_split()["bagsPerClass"]["valid"]',
        "toy_run_otra_pasada")
    assert otra_pasada.STEPS_PER_EPOCH_DECLARED != (
        toy_run.STEPS_PER_EPOCH_DECLARED), (
        "la mutación no cambió la pasada: este test no estaría midiendo en qué "
        "unidad se declara la rejilla")
    assert otra_pasada.frame_epochs(otra_pasada.FULL) == (
        toy_run.frame_epochs(toy_run.FULL)), (
        otra_pasada.frame_epochs(otra_pasada.FULL),
        toy_run.frame_epochs(toy_run.FULL))
    assert otra_pasada.frame_stride(otra_pasada.FULL) != vivo, (
        otra_pasada.frame_stride(otra_pasada.FULL), vivo)

    # **The pilot sits ON the floor, and that is the declaration rather than a
    # shortfall.** Its ceiling is fewer epochs than `TRAJECTORY_FRAMES`, so no
    # whole-epoch stride reaches the declared count and the floor of one epoch
    # takes over. The previous version asserted the pilot's stride FELL when
    # the count doubled; at a floor it cannot, and asserting it would be
    # asserting the mechanism instead of the data.
    assert toy_run.PILOT.epochs_2 // toy_run.TRAJECTORY_FRAMES == 0, (
        "el piloto ya no está bajo el piso: este test mide que el piso es lo "
        "que manda ahí, y con otro techo habría que medir otra cosa")
    assert toy_run.frame_epochs(toy_run.PILOT) == 1
    assert toy_run.frame_stride(toy_run.PILOT) == (
        toy_run.STEPS_PER_EPOCH_DECLARED)
    assert mutado.frame_epochs(mutado.PILOT) == 1, (
        mutado.frame_epochs(mutado.PILOT))

    # And the floor is REACHABLE rather than frozen: a declared count BELOW the
    # pilot's own ceiling lifts it off the floor. Without this the two
    # assertions above would hold just as well on a pilot whose stride could
    # never move at all.
    # Two spaces: the mutator requires the same byte length, because that is
    # what makes its md5 check the thing that measures the mutation arrived.
    pocos = _mutated_toy_run(tmp_path, "TRAJECTORY_FRAMES = 20",
                             "TRAJECTORY_FRAMES =  2", "toy_run_frames_2")
    assert pocos.frame_epochs(pocos.PILOT) == pocos.PILOT.epochs_2 // 2
    assert pocos.frame_epochs(pocos.PILOT) > toy_run.frame_epochs(
        toy_run.PILOT), (pocos.frame_epochs(pocos.PILOT),
                         toy_run.frame_epochs(toy_run.PILOT))

    # What the change BOUGHT, which is why it was made: the full scale draws
    # the declared count and the pilot draws a fifth of it.
    assert len(toy_run.frame_steps(
        toy_run.FULL,
        toy_run.FULL.epochs_2 * toy_run.STEPS_PER_EPOCH_DECLARED)) == 21
    assert len(toy_run.frame_steps(
        toy_run.PILOT,
        toy_run.PILOT.epochs_2 * toy_run.STEPS_PER_EPOCH_DECLARED)) == 5


def test_a_schedule_that_dropped_the_first_frame_is_caught(tmp_path):
    """Frame zero is the step-0 checkpoint, and losing it is a red this reaches.

    Without it the animation opens on a target that has already moved and
    nothing in the picture says so -- and the first rung of the intervention
    table, which is read at that same checkpoint, would have no frame beside it.
    """
    mutado = _mutated_toy_run(tmp_path, "steps = {0, applied}",
                              "steps = {9, applied}", "toy_run_sin_cuadro_cero")
    escala = dataclasses.replace(toy_run.PILOT, epochs_2=1)
    assert toy_run.frame_steps(escala, 11)[0] == 0
    assert mutado.frame_steps(escala, 11)[0] != 0, (
        "la mutación no cambió el primer cuadro: este test no estaría midiendo "
        "que el cero es una decisión y no un accidente del rango")


def test_the_frames_a_cell_recorded_are_the_schedule_its_run_declares(cell, tiny):
    """What reached the artefact is exactly what `frame_steps` says, derived
    from two numbers the record already carries.

    The join crossed here is the one a fixture cannot: the schedule is computed
    in one place and followed in another, and asserting each half against
    something this file wrote would verify both and never the connection.
    """
    fila = cell["rows"][ADAPTED]
    miembros = cell["latent"][ADAPTED]
    pasos = [int(p) for p in miembros["trajectorySteps"].tolist()]
    assert pasos == list(toy_run.frame_steps(tiny, fila["phase2Steps"])), (
        pasos, fila["phase2Steps"], tiny.steps_2)
    assert fila["trajectoryFrames"] == len(pasos)
    assert fila["frameStride"] == toy_run.frame_stride(tiny)
    # Each member carries one row per frame, and the control is that there is
    # more than one frame: a single-frame trajectory would make every shape
    # assertion below true by having nothing to disagree about.
    assert len(pasos) > 1, pasos
    assert miembros["trajectoryTarget"].shape[0] == len(pasos)
    assert miembros["trajectoryPrediction"].shape[0] == len(pasos)
    assert miembros["trajectoryTarget"].shape[1:] == miembros["targetAfter"].shape


def test_frame_zero_is_the_step_zero_checkpoint_and_the_later_frames_are_not(
        cell):
    """Frame zero IS `targetBefore` -- same reading, same moment -- and the
    frames after it are the encoder having moved.

    Both halves. Bit-identity alone would hold for a trajectory that recorded
    the same snapshot at every frame, so the second assertion is what says the
    recording is live; and «the frames differ» alone would hold for a
    trajectory that started somewhere else, so the first is what ties it to the
    checkpoint the first rung of the table is read at.
    """
    miembros = cell["latent"][ADAPTED]
    cuadros = miembros["trajectoryTarget"]
    assert np.array_equal(cuadros[0], miembros["targetBefore"])
    movidos = [k for k in range(1, cuadros.shape[0])
               if not np.array_equal(cuadros[k], cuadros[k - 1])]
    assert movidos, (
        "ningún cuadro difiere del anterior: la trayectoria sería una foto "
        "repetida y la animación no mostraría ningún movimiento")
    assert cell["rows"][ADAPTED]["latentTargetMoved"] > 0.0


def test_a_trajectory_recorded_off_the_frozen_source_is_caught(tmp_path, familia,
                                                               tiny, cell):
    """Point the frame at the source and every frame comes out identical.

    This is the red for «the positions written are the ones each step
    produced». The source is frozen -- `latentSourceMoved` is measured at zero
    -- so a recorder reading it would still produce a well-shaped trajectory of
    the right length with the right schedule, and every other assertion in this
    file would stay green. What fires is that the frames stop differing from
    one another, and that they stop being the target's.
    """
    mutado = _mutated_toy_run(tmp_path, '"target").items()', '"source").items()',
                              "toy_run_traza_de_la_fuente")
    hiper = toy_run.declared_hyper(tiny)
    celda = mutado.run_cell(familia, "batchnorm", 0, tiny, hiper, DEVICE)
    cuadros = celda["latent"][ADAPTED]["trajectoryTarget"]
    assert cuadros.shape[0] > 1, cuadros.shape
    iguales = all(np.array_equal(cuadros[k], cuadros[0])
                  for k in range(1, cuadros.shape[0]))
    assert iguales, (
        "la mutación no congeló la trayectoria: o la fuente se movió, o la "
        "traza no es lo que este test cree, y entonces el verde del test "
        "vecino no prueba que los cuadros sean del blanco")
    # THE CONTROL, on the same scale, the same family and the same
    # architecture: the live module's frames DO differ. Without it «todos los
    # cuadros iguales» could be a property of this fixture rather than of the
    # mutation, and the red would be measuring nothing.
    vivos = cell["latent"][ADAPTED]["trajectoryTarget"]
    assert vivos.shape[0] > 1
    assert any(not np.array_equal(vivos[k], vivos[0])
               for k in range(1, vivos.shape[0])), (
        "la trayectoria sin mutar también sale congelada en esta fixtura, así "
        "que la mutación no distingue nada")
    # And frame zero is the same in both, which is what says the mutation moved
    # the RECORDING and not the starting point: at the step-0 checkpoint
    # `theta_t` is `theta_s`, so the two encoders read the target bags alike.
    assert np.array_equal(cuadros[0], celda["latent"][ADAPTED]["targetBefore"])


def test_a_frame_labelled_with_the_previous_step_is_caught(tmp_path, familia,
                                                           tiny):
    """Record after the update and label it with the one before, and the
    schedule stops matching.

    The label is not decoration: `_panel_fase2` truncates the loss curve at the
    step the frame stands at, so a frame labelled one step early draws a curve
    that stops before the state the embedding beside it shows.
    """
    mutado = _mutated_toy_run(tmp_path, "record_frame(step + 1)",
                              "record_frame(step + 0)", "toy_run_paso_corrido")
    hiper = toy_run.declared_hyper(tiny)
    celda = mutado.run_cell(familia, "batchnorm", 0, tiny, hiper, DEVICE)
    fila = celda["rows"][ADAPTED]
    pasos = [int(p) for p in celda["latent"][ADAPTED]["trajectorySteps"].tolist()]
    esperado = list(toy_run.frame_steps(tiny, fila["phase2Steps"]))
    assert pasos != esperado, (
        pasos, esperado,
        "la mutación no movió ninguna etiqueta: el test de la grilla vecina no "
        "estaría midiendo que el número del cuadro es el paso que lo produjo")


def test_the_prediction_written_is_the_heads_own_decision_per_bag(familia, tiny):
    """`bag_positions` writes `argmax` of the arm's own scoring, bag by bag.

    Crossed against `Arm.score_target`, which is a DIFFERENT entry point: it
    re-encodes the bags itself instead of reading the embeddings the positions
    came from. Comparing the writer with a fixture this file wrote would verify
    both halves and never the join.
    """
    preparado = _prepared(familia, tiny)
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
        tmp_path, familia, tiny):
    """Take the argmin instead and the written decision stops being the head's.

    The red for the test above. Without it, a channel that wrote any
    class-shaped integer per bag would satisfy every shape and dtype assertion
    in this file.
    """
    mutado = _mutated_toy_run(tmp_path, ".argmax(dim=1).cpu())",
                              ".argmin(dim=1).cpu())", "toy_run_argmin")
    preparado = _prepared(familia, tiny)
    modelo, material = preparado["model"], preparado["material"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        vivo = toy_run.bag_positions(modelo, material["target"], DEVICE,
                                     "target")["predictions"]
        roto = mutado.bag_positions(modelo, material["target"], DEVICE,
                                    "target")["predictions"]
    assert not np.array_equal(vivo, roto), (
        "argmin y argmax dieron lo mismo sobre esta fixtura: los puntajes "
        "estarían empatados y el test de arriba no estaría distinguiendo nada")


#: The cell these two tests run on, and `lunas` is deliberately not it. The
#: first rung and the method have to READ DIFFERENTLY for «read at step 0» to be
#: distinguishable from «read after phase 2» at all, and at nine evaluation bags
#: every reading is a multiple of 1/9, so they collide often. Measured over
#: three families, two backbones and three seeds at the `tiny` scale, the pair
#: separates on `toroide`/`batchnorm`/0 (0.3333 against 0.6667) and agrees on
#: every `lunas` cell. Named here rather than inside one test because both of
#: the tests below need the same separation.
FAMILIA_QUE_SEPARA_EL_ESCALON = "toroide"


def test_predict_is_the_step_zero_checkpoint_read_on_the_targets_eval_role(
        tiny):
    """The first rung is the source model applied to the target, before anything
    moved.

    Three assertions, and the last two are what make the first mean anything.
    At the step-0 checkpoint `theta_t` is `theta_s` bit for bit, so the
    TARGET-encoder reading and the SOURCE-encoder reading of the same bags must
    agree -- that is what «the source model applied to the target» means, and
    the two are different code paths (`score_target` against `forward`). Then
    the recorded rung has to equal THAT reading. And on this cell the method's
    own accuracy is a different number, so a rung read after phase 2 instead
    would fail here rather than passing on a coincidence.
    """
    familia = toy_run.build_family(FAMILIA_QUE_SEPARA_EL_ESCALON, 0, tiny)
    celda = toy_run.run_cell(familia, "batchnorm", 0, tiny,
                             toy_run.declared_hyper(tiny), DEVICE)
    preparado = _prepared(familia, tiny)
    modelo, material = preparado["model"], preparado["material"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        _, _, blanco_eval = bags.roles(material["target"])
        por_theta_t = harness.target_accuracy(modelo, blanco_eval, DEVICE)
        por_theta_s = harness.accuracy(modelo, blanco_eval, DEVICE)
    assert por_theta_t == por_theta_s, (por_theta_t, por_theta_s)
    fila = celda["rows"][ADAPTED]
    assert fila["predictTargetAccuracy"] == por_theta_t
    assert fila["predictTargetAccuracy"] != fila["targetAccuracy"], (
        fila["predictTargetAccuracy"], fila["targetAccuracy"],
        "el escalón cero y el método leen el mismo número en esta celda, así "
        "que la igualdad de arriba no distingue en qué momento se midió")
    assert fila["latentTargetMoved"] > 0.0


def test_predict_read_on_the_wrong_role_is_caught(tmp_path, tiny):
    """Read the first rung on the source's evaluation role and the number moves.

    The red for «on the TARGET's eval role». A rung read on the source answers
    how well the source model does at home, which is not what a floor for the
    target is -- and it would still be a plausible accuracy in the same range,
    so nothing about the shape of the record would say so.
    """
    mutado = _mutated_toy_run(
        tmp_path, "predict_accuracy = harness.target_accuracy(model, target_eval",
        "predict_accuracy = harness.target_accuracy(model, source_eval",
        "toy_run_predict_en_la_fuente")
    familia = toy_run.build_family(FAMILIA_QUE_SEPARA_EL_ESCALON, 0, tiny)
    hiper = toy_run.declared_hyper(tiny)
    roto = mutado.run_cell(familia, "batchnorm", 0, tiny, hiper, DEVICE)
    vivo = toy_run.run_cell(familia, "batchnorm", 0, tiny, hiper, DEVICE)
    assert (roto["rows"][ADAPTED]["predictTargetAccuracy"]
            != vivo["rows"][ADAPTED]["predictTargetAccuracy"]), (
        "los dos roles leen el mismo número en esta fixtura, así que el test de "
        "arriba no distingue sobre cuál se midió el escalón cero")
    # Everything else the mutation did NOT touch, so the red above is about the
    # role and not about having loaded a different module.
    assert (roto["rows"][ADAPTED]["targetAccuracy"]
            == vivo["rows"][ADAPTED]["targetAccuracy"])


def test_adabn_is_measured_where_there_are_statistics_and_absent_where_not(
        familia, tiny):
    """AdaBN's rung exists for the BatchNorm backbone and for no other.

    Measured off the backbone's own modules and never off the row's name: the
    row called `batchnorm` is reachable because `mlp` HOLDS `nn.BatchNorm1d`
    layers, and a map from a row name to a backbone name is a string.
    """
    hiper = toy_run.declared_hyper(tiny)
    con = toy_run.run_cell(familia, "batchnorm", 0, tiny, hiper,
                           DEVICE)["rows"][ADAPTED]
    sin = toy_run.run_cell(familia, "ninguna", 0, tiny, hiper,
                           DEVICE)["rows"][ADAPTED]
    assert con["adabnLayers"] > 0 and con["adabnTargetAccuracy"] is not None
    assert con["adabnBatches"] > 0
    assert con["adabnNote"] is None
    assert sin["adabnLayers"] == 0 and sin["adabnTargetAccuracy"] is None
    assert sin["adabnNote"] == toy_run.ADABN_ABSENT_NOTE


def test_adabn_leaves_the_model_it_was_given_bit_identical(familia, tiny):
    """It re-estimates a COPY: the arm phase 2 then moves is untouched.

    Without this the rung would be an intervention on the run it is supposed to
    be a floor for, and every number after it would be measured on a model whose
    normalization statistics somebody else had already replaced.
    """
    preparado = _prepared(familia, tiny)
    modelo, material = preparado["model"], preparado["material"]
    with toy_run.material_bound(familia, tiny, "mlp"):
        _, _, blanco_eval = bags.roles(material["target"])
        blanco_train, _, _ = bags.roles(material["target"])
        antes = {nombre: b.detach().clone()
                 for nombre, b in modelo.target_encoder.named_buffers()}
        assert antes, "este backbone no tiene buffers y el test no mide nada"
        lectura = toy_run.adabn_reading(modelo, blanco_train, blanco_eval, DEVICE)
        assert lectura["accuracy"] is not None
        for nombre, b in modelo.target_encoder.named_buffers():
            assert torch.equal(b, antes[nombre]), nombre
