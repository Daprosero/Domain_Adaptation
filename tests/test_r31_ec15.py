"""r31's Eq. (15): the l1 constraint on `v_R` becomes a max-norm projection.

r30 imposed `||v_R||_1 <= 1` by REPARAMETRIZATION, inside the forward pass:

    r30:  v_R_tilde = v_R / max(1, ||v_R||_1)   and the selector used v_R_tilde

r31 imposes the same constraint as a max-norm constraint on the parameter
itself: "La restriccion se impone como norma maxima: tras cada paso de
optimizacion, si ||v_R||_1 > 1, se reescala v_R <- v_R / ||v_R||_1."

Two things follow, and each is a separate test below.

1. **`v_R_tilde` stops existing as an object.** The selector reads `v_R`
   directly, so `R_phi(h) = v_R^T tanh(V_R h + b_R)`. The Hoelder bound
   `|R_phi(h)| <= ||v_R||_1 <= 1` still holds, but it now holds because the
   parameter is kept in the ball rather than because the forward pass divides.
2. **The constraint moved out of the graph.** r30's differentiability paragraph
   carried an exception for the `v_R_tilde` normalization ("lo es excepto en un
   conjunto de medida nula"); r31 drops it, because nothing in the forward pass
   normalizes any more. So the projection is NOT a function the loss is
   differentiated through -- it runs under `no_grad` after the step.

**This is a change to the training loop, not only to a function**, and the
third group of tests crosses that join: it runs a real optimizer step on a real
`Arm` and asks whether the parameter came back inside the ball. Testing the
projection function against a fixture would verify the function and never the
connection, and the connection is the whole of what r31 changed.

Every test here was observed RED against the r30 tree before the code moved.
"""

from __future__ import annotations

import math

import pytest
import torch

from MIL_CREDA.attention import (
    project_onto_l1_ball,
    relevance_component,
    relevance_logits,
)

DTYPE = torch.float64


def _selector(dim: int = 4, scale: float = 1.0, saturate: float = 1.0):
    """A selector with an EXPLICIT `v_R` whose l1 norm is `2.4 * scale`.

    Explicit rather than drawn: every test below needs to know whether the
    fixture is inside or outside the unit ball, and a norm that depends on a
    seed is a fixture whose premise has to be re-measured after any change to
    the generator. `saturate` scales `V_R` so `tanh` can be driven to its own
    bounds, which is what makes the Hoelder bound reachable from below.
    """
    generator = torch.Generator().manual_seed(777)
    V_R = torch.randn(3, dim, generator=generator, dtype=DTYPE) * saturate
    b_R = torch.randn(3, generator=generator, dtype=DTYPE)
    v_R = torch.tensor([0.9, -0.8, 0.7], dtype=DTYPE) * scale
    return V_R, b_R, v_R


# -------------------------------------------- the forward reads `v_R` raw


def test_the_selector_reads_v_R_raw_and_does_not_normalize_it() -> None:
    """`R_phi(h) = v_R^T tanh(V_R h + b_R)`, with no division anywhere.

    Reachable red: the r30 tree divides by `max(1, ||v_R||_1)`, so the fixture
    deliberately carries a `v_R` OUTSIDE the ball -- with `||v_R||_1 <= 1` the
    two revisions agree exactly and no test could separate them.
    """
    V_R, b_R, v_R = _selector()
    norm = float(torch.linalg.vector_norm(v_R, ord=1))
    assert norm > 1.0, f"fixture: ||v_R||_1 must exceed 1, got {norm}"

    H = torch.randn(5, 4, generator=torch.Generator().manual_seed(11), dtype=DTYPE)
    expected = torch.tanh(H @ V_R.transpose(0, 1) + b_R) @ v_R

    got = relevance_component(H, V_R, b_R, v_R)
    assert torch.allclose(got, expected, atol=1e-14, rtol=0.0)

    # And the r30 form differs here by a margin, so the agreement above is a
    # fact about the revision and not about the tolerance.
    r30 = torch.tanh(H @ V_R.transpose(0, 1) + b_R) @ (v_R / max(1.0, norm))
    assert float((expected - r30).abs().max()) > 1e-2, (
        "the fixture cannot distinguish r31 from r30")


def test_the_logit_carries_the_raw_selector_into_equation_fifteen() -> None:
    """The change reaches `relevance_logits`, not only its first component.

    `relevance_logits` is what every consumer calls, so a tree that fixed
    `relevance_component` and left the logit reading a normalized copy would
    pass the test above and still compute r30.
    """
    V_R, b_R, v_R = _selector()
    H = torch.randn(5, 4, generator=torch.Generator().manual_seed(12), dtype=DTYPE)
    gamma, sigma = 0.0, 2.0

    # gamma = 0 reduces Eq. (15) to R_phi alone, which is what isolates the
    # selector from the consensus term.
    got = relevance_logits(H, V_R, b_R, v_R, gamma, sigma)
    expected = relevance_component(H, V_R, b_R, v_R)
    assert torch.allclose(got, expected, atol=1e-14, rtol=0.0)

    norm = float(torch.linalg.vector_norm(v_R, ord=1))
    r30 = torch.tanh(H @ V_R.transpose(0, 1) + b_R) @ (v_R / max(1.0, norm))
    assert float((got - r30).abs().max()) > 1e-2


def test_the_reparametrized_helper_is_gone() -> None:
    """`v_R_tilde` stopped being an object, so no module may still expose it.

    r31 removes the symbol from the formulation rather than leaving it unused.
    A private helper left behind in `src/` would be a function implementing a
    superseded revision that nothing calls, and the next reader would not know
    which of the two the selector uses.
    """
    import MIL_CREDA.attention as attention

    assert not hasattr(attention, "_l1_ball_reparametrization"), (
        "the r30 reparametrization is still exposed by MIL_CREDA.attention")


# ------------------------------------------------------- the projection itself


def test_the_projection_fires_only_outside_the_ball() -> None:
    """`||v_R||_1 > 1` rescales to the edge; inside the ball nothing moves.

    This is the max-norm rule verbatim, and the two halves are separate
    claims: a projection that always divided by the norm would push an interior
    point OUT to the edge, which r31 does not ask for.
    """
    outside = torch.tensor([0.9, -0.8, 0.7], dtype=DTYPE)
    assert float(torch.linalg.vector_norm(outside, ord=1)) == pytest.approx(2.4)
    fired = project_onto_l1_ball(outside)
    assert fired is True
    assert float(torch.linalg.vector_norm(outside, ord=1)) == pytest.approx(1.0, abs=1e-14)

    inside = torch.tensor([0.2, -0.3, 0.1], dtype=DTYPE)
    before = inside.clone()
    fired = project_onto_l1_ball(inside)
    assert fired is False
    assert torch.equal(inside, before), "an interior point must not be moved"


def test_the_projection_preserves_direction() -> None:
    """A rescale, not a reprojection: only the magnitude changes."""
    v_R = torch.tensor([3.0, -1.0, 2.0], dtype=DTYPE)
    direction = v_R / torch.linalg.vector_norm(v_R, ord=2)
    project_onto_l1_ball(v_R)
    after = v_R / torch.linalg.vector_norm(v_R, ord=2)
    assert torch.allclose(direction, after, atol=1e-14, rtol=0.0)


def test_the_projection_is_idempotent_and_in_place() -> None:
    """Applying it twice fires once, and it writes through the parameter.

    In place matters: the optimizer holds a reference to the parameter, so a
    projection returning a new tensor would leave the trained object untouched
    and the constraint would silently never apply.
    """
    v_R = torch.nn.Parameter(torch.tensor([2.0, -2.0, 1.0], dtype=DTYPE))
    identity = id(v_R)
    assert project_onto_l1_ball(v_R) is True
    assert project_onto_l1_ball(v_R) is False
    assert id(v_R) == identity
    assert float(torch.linalg.vector_norm(v_R.detach(), ord=1)) == pytest.approx(
        1.0, abs=1e-14)


def test_the_projection_leaves_no_gradient_history() -> None:
    """r31 dropped the differentiability exception, so this is not in the graph.

    The projection runs after the step, on a leaf the optimizer owns. If it
    were recorded, the parameter would stop being a leaf and the next
    `backward` would either accumulate through the rescale or raise.
    """
    v_R = torch.nn.Parameter(torch.tensor([2.0, -2.0, 1.0], dtype=DTYPE))
    project_onto_l1_ball(v_R)
    assert v_R.is_leaf, "the projection made the parameter a non-leaf"
    assert v_R.grad_fn is None, "the projection was recorded on the graph"
    assert v_R.requires_grad, "the projection cleared requires_grad"


def test_the_hoelder_bound_holds_after_the_projection() -> None:
    """The bound Eq. (15) claims, measured where r31 now makes it hold.

    r30 got `|R_phi(h)| <= 1` from the forward division, for any `v_R`. r31
    gets it from the parameter being kept in the ball, so the claim is only
    true of a PROJECTED parameter -- and that is exactly why the projection has
    to be wired into the loop and not merely available.
    """
    V_R, b_R, v_R = _selector(scale=4.0, saturate=12.0)
    H = torch.randn(32, 4, generator=torch.Generator().manual_seed(13), dtype=DTYPE)

    unprojected = relevance_component(H, V_R, b_R, v_R).abs().max()
    assert float(unprojected) > 1.0, (
        "fixture: an unprojected v_R must be able to break the bound, "
        f"got {float(unprojected)}")

    project_onto_l1_ball(v_R)
    assert float(torch.linalg.vector_norm(v_R, ord=1)) <= 1.0 + 1e-14
    projected = relevance_component(H, V_R, b_R, v_R).abs().max()
    assert float(projected) <= 1.0 + 1e-14, float(projected)


# -------------------------------------------------- crossing the training join


class _SmallEncoder(torch.nn.Module):
    """A linear stand-in for `config.BACKBONE`, which is a resnet18.

    The same substitution `tests/test_arm_objectives.py` makes, and for the
    same reason: this file is about Eq. (15)'s constraint on `v_R`, and
    training a real resnet to observe a rescale of 128 numbers would measure
    the backbone instead. `wiring._resolve_encoder` reads the module-level
    `FeatureExtractor` name, so patching that name is what intercepts it.
    """

    output_dim = 8

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(x.shape[0], -1))


def _pool(seed: int):
    from MIL_CREDA_Benchmark import config, wiring

    generator = torch.Generator().manual_seed(seed)
    bags_count = config.BAGS_PER_STEP + 2
    images = torch.randn(bags_count * config.INSTANCES_PER_BAG, 3, 8, 8,
                         generator=generator)
    members = torch.arange(images.shape[0]).reshape(
        bags_count, config.INSTANCES_PER_BAG)
    labels = torch.arange(bags_count) % config.CLASSES
    return wiring.Pool(images, members, labels)


@pytest.fixture
def arm(monkeypatch):
    """Arm `G` -- the full method, and the one arm that owns a learned `v_R`."""
    from MIL_CREDA_Benchmark import config, wiring

    monkeypatch.setattr(wiring, "FeatureExtractor", _SmallEncoder)
    torch.manual_seed(11)
    built = wiring.build("G", config.CLASSES, _pool(1), _pool(2))
    assert hasattr(built, "v_R"), "fixture: arm G must own a learned selector"
    return built


def test_the_arm_starts_inside_the_ball(arm) -> None:
    """Construction leaves `v_R` FEASIBLE, which r31's prose does not say.

    r31 states `||v_R||_1 <= 1` as a constraint of Eq. (15) and then says how
    it is maintained -- "tras cada paso de optimizacion". Maintenance is not
    establishment: `nn.init.normal_(v_R, std=0.1)` over
    `config.ATTENTION_WIDTH = 128` coordinates draws a norm near 11, so a tree
    that projected only after steps would run its FIRST forward with
    `|R_phi| <= 11` and the bound Eq. (15) claims would be false there.

    Under r30 that first forward divided by the same 11, so `R_phi` sat at the
    ball's edge from the start. Projecting at construction is what keeps r31
    agreeing with r30 everywhere the two revisions should agree, leaving the
    intended difference -- what the optimizer sees -- as the only one.
    """
    norm = float(torch.linalg.vector_norm(arm.v_R.detach(), ord=1))
    assert norm <= 1.0 + 1e-12, (
        f"v_R left construction outside the ball at ||v_R||_1 = {norm:.4f}")
    # And the fixture's premise: an UNPROJECTED draw of this shape really is
    # outside, so the assertion above is enforcement and not a coincidence.
    torch.manual_seed(11)
    raw = torch.empty(arm.v_R.numel())
    torch.nn.init.normal_(raw, std=0.1)
    assert float(torch.linalg.vector_norm(raw, ord=1)) > 5.0


def test_the_arm_exposes_the_projection_for_its_loop(arm) -> None:
    """The harness can ask the method's modules to re-impose the constraint.

    The arm never reimplements the rescale -- it calls
    `MIL_CREDA.attention.project_onto_l1_ball`, which is where Eq. (15) lives.
    """
    with torch.no_grad():
        arm.v_R.mul_(0.0)
        arm.v_R[:3] = torch.tensor([2.0, -3.0, 1.0])
    assert float(torch.linalg.vector_norm(arm.v_R.detach(), ord=1)) > 1.0

    assert arm.project_constraints() is True
    assert float(torch.linalg.vector_norm(arm.v_R.detach(), ord=1)) == (
        pytest.approx(1.0, abs=1e-6))


def test_v_R_is_inside_the_ball_after_a_real_optimizer_step(arm) -> None:
    """The join itself: an actual step, then the actual constraint.

    Testing the projection against a fixture verifies the function and never
    the connection, and the connection is what r31 changed. So this runs the
    real `Arm`, a real backward, a real `optimizer.step()` and the real
    projection call, with a learning rate large enough that the step genuinely
    leaves the ball -- asserted, so a step that happened to stay inside cannot
    let this pass while testing nothing.
    """
    optimizer = torch.optim.SGD(arm.parameters(), lr=50.0)
    bags = arm.source.take(torch.arange(4))
    labels = arm.source.labels[:4]

    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(arm(bags), labels)
    loss.backward()
    optimizer.step()

    escaped = float(torch.linalg.vector_norm(arm.v_R.detach(), ord=1))
    fired = arm.project_constraints()
    settled = float(torch.linalg.vector_norm(arm.v_R.detach(), ord=1))

    assert escaped > 1.0, (
        f"fixture: the step must push v_R out of the ball, got {escaped}")
    assert fired is True
    assert settled <= 1.0 + 1e-6, settled


def test_the_projection_is_counted_so_a_no_op_change_is_visible(arm) -> None:
    """How often it fires is bookkeeping the harness keeps, not mathematics.

    If the projection never fired, r31's change would be a no-op in practice
    and that is worth knowing rather than assuming. The counters live on the
    `Arm` -- the harness implements no equation and is the right place for a
    tally -- and `MIL_CREDA.attention` stays a pure function returning a bool.
    """
    arm.projection_calls = 0
    arm.projection_firings = 0

    with torch.no_grad():
        arm.v_R.mul_(0.0)
        arm.v_R[0] = 4.0
    assert arm.project_constraints() is True
    assert (arm.projection_calls, arm.projection_firings) == (1, 1)

    with torch.no_grad():
        arm.v_R.mul_(0.0)
        arm.v_R[0] = 0.01
    assert arm.project_constraints() is False
    assert (arm.projection_calls, arm.projection_firings) == (2, 1), (
        "an interior point must be called and must not be counted as firing")
