"""The two-phase arrangement's six invariants, one test each.

`src/CREDA/two_phase.py` holds a training protocol rather than a term, so what
there is to assert is not a number but a shape: which module sees which domain,
which parameters can move, and when weights transfer. Each test below is named
after one id in that module's `__provenance__["invariants"]`, and
`test_every_declared_invariant_has_a_test` reddens if the two lists ever drift
apart -- a declared invariant with no test and a tested property with no
declaration look identical to a green run otherwise.

The term itself is not under test here; `test_creda_sample.py` covers it. Most
tests use a stand-in that depends on the target features and nothing else,
because a protocol test that needed a correctly shaped kernel would fail for
reasons that have nothing to do with the protocol. One test runs the real
`CREDALoss` to prove the two fit together.
"""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn as nn

from CREDA.models import CREDALoss
from CREDA.two_phase import (
    FREE,
    __provenance__,
    adapt,
    clone_for_target,
    parameter_distance,
    train_source,
)

CLASSES = 3
WIDTH = 6
OUT = 4


def _encoder(batch_norm: bool = False) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(WIDTH, OUT)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(OUT))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _fixture(batch_norm: bool = False, seed: int = 0):
    """A source cloud and a target cloud whose VALUES cannot be confused.

    Source features live near +10 and target features near -10, with no
    overlap, so a test can tell which domain a tensor came from by its sign
    alone. That is what makes the routing invariant checkable at all.
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    source_x = 10.0 + torch.rand(24, WIDTH, generator=generator)
    source_y = torch.arange(24) % CLASSES
    target_x = -10.0 - torch.rand(24, WIDTH, generator=generator)
    encoder = _encoder(batch_norm)
    head = nn.Linear(OUT, CLASSES)
    train_source(encoder, head, source_x, source_y,
                 epochs=2, batch_size=8, generator=generator)
    return encoder, head, source_x, source_y, target_x, generator


def _moves_the_target(f_s, f_t, y_s, g_t):
    """A stand-in term: depends on the target features, on nothing else."""
    return f_t.pow(2).mean()


def _ignores_the_target(f_s, f_t, y_s, g_t):
    """A stand-in with no path to the target encoder at all."""
    return f_s.pow(2).mean()


class _Recorder(nn.Module):
    """Wraps an encoder and keeps every input it was handed."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.detach().clone())
        return self.inner(x)


def test_the_first_phase_freezes_the_source_encoder_and_the_head():
    encoder, head, source_x, source_y, target_x, generator = _fixture()

    assert [p.requires_grad for p in encoder.parameters()] == [False] * len(
        list(encoder.parameters()))
    assert [p.requires_grad for p in head.parameters()] == [False] * len(
        list(head.parameters()))

    before = [p.detach().clone() for p in list(encoder.parameters()) + list(head.parameters())]
    target = clone_for_target(encoder)
    adapt(encoder, head, target, source_x, source_y, target_x,
          creda=_moves_the_target, steps=6, batch_size=8, generator=generator)
    after = [p.detach().clone() for p in list(encoder.parameters()) + list(head.parameters())]

    assert len(before) > 0
    for b, a in zip(before, after):
        assert torch.equal(b, a), "phase two moved a parameter phase one froze"


def test_the_target_encoder_starts_bit_identical_to_the_source():
    encoder, _, _, _, _, _ = _fixture()
    target = clone_for_target(encoder)

    assert parameter_distance(target, encoder) == 0.0
    pairs = list(zip(target.parameters(), encoder.parameters()))
    assert len(pairs) > 0
    for p_t, p_s in pairs:
        assert torch.equal(p_t, p_s)
    assert all(p.requires_grad for p in target.parameters())

    # Independent objects: writing to the copy must not reach the original.
    with torch.no_grad():
        next(target.parameters()).add_(1.0)
    assert parameter_distance(target, encoder) > 0.0


def test_the_source_never_passes_through_the_target_encoder():
    encoder, head, source_x, source_y, target_x, generator = _fixture()
    target = _Recorder(clone_for_target(encoder))

    adapt(encoder, head, target, source_x, source_y, target_x,
          creda=_moves_the_target, steps=5, batch_size=8,
          generator=generator, sync_steps=FREE)

    assert len(target.seen) == 5, "the target encoder was not called once per step"
    for seen in target.seen:
        # Source rows are all positive, target rows all negative. A single
        # source row reaching here would push the maximum above zero.
        assert float(seen.max()) < 0.0, "a source sample reached the target encoder"


def test_the_free_regime_copies_the_weights_exactly_once():
    encoder, head, source_x, source_y, target_x, generator = _fixture()
    target = clone_for_target(encoder)

    history = adapt(encoder, head, target, source_x, source_y, target_x,
                    creda=_moves_the_target, steps=8, batch_size=8,
                    generator=generator)

    assert FREE == 0
    assert not any(record["synced"] for record in history), "a free run transferred weights"
    # Having moved at all is what makes "never returned" meaningful.
    assert history[0]["drift"] > 0.0
    assert all(record["drift"] > 0.0 for record in history)


def test_the_leash_returns_the_weights_to_the_source_at_its_interval():
    encoder, head, source_x, source_y, target_x, generator = _fixture()
    target = clone_for_target(encoder)
    source_before = [p.detach().clone() for p in encoder.parameters()]

    history = adapt(encoder, head, target, source_x, source_y, target_x,
                    creda=_moves_the_target, steps=8, batch_size=8,
                    generator=generator, sync_steps=2)

    synced = [record for record in history if record["synced"]]
    loose = [record for record in history if not record["synced"]]
    assert [record["step"] for record in synced] == [1, 3, 5, 7]
    assert len(loose) == 4
    for record in synced:
        assert record["drift"] == 0.0, "a transfer left the weights apart"
    # And it copied in the right DIRECTION. Without this the same assertion
    # passes for a transfer that writes the target's weights into the source,
    # which would destroy the frozen-source invariant while reading as green.
    for p_s, p_before in zip(encoder.parameters(), source_before):
        assert torch.equal(p_s, p_before), "a transfer wrote into the source"
    # And the loose steps did move, or the interval would prove nothing.
    assert all(record["drift"] > 0.0 for record in loose)


def test_the_adaptation_term_is_the_only_force_on_the_target_encoder():
    encoder, head, source_x, source_y, target_x, generator = _fixture()

    # Stronger than "nothing else moves it": in this arrangement nothing else
    # CAN. The source encoder and the head are frozen and run under `no_grad`,
    # so a term that never reads the target features has no graph at all, and
    # the module says so with the reason rather than letting autograd name a
    # tensor.
    blind = clone_for_target(encoder)
    with pytest.raises(ValueError, match="no gradient path to the target encoder"):
        adapt(encoder, head, blind, source_x, source_y, target_x,
              creda=_ignores_the_target, steps=6, batch_size=8,
              generator=generator)
    assert parameter_distance(blind, encoder) == 0.0

    seeing = clone_for_target(encoder)
    history = adapt(encoder, head, seeing, source_x, source_y, target_x,
                    creda=_moves_the_target, steps=6, batch_size=8,
                    generator=generator)
    assert history[-1]["drift"] > 0.0, "the control arm never moved either"


def test_phase_two_refuses_an_unfrozen_source():
    encoder, head, source_x, source_y, target_x, generator = _fixture()
    target = clone_for_target(encoder)

    next(encoder.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen source encoder"):
        adapt(encoder, head, target, source_x, source_y, target_x,
              creda=_moves_the_target, steps=1, batch_size=8, generator=generator)

    next(encoder.parameters()).requires_grad_(False)
    next(head.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen head"):
        adapt(encoder, head, target, source_x, source_y, target_x,
              creda=_moves_the_target, steps=1, batch_size=8, generator=generator)


def test_the_arrangement_runs_with_the_real_adaptation_term():
    encoder, head, source_x, source_y, target_x, generator = _fixture()
    target = clone_for_target(encoder)

    history = adapt(encoder, head, target, source_x, source_y, target_x,
                    creda=CREDALoss(sigma="auto"), steps=6, batch_size=12,
                    generator=generator)

    assert len(history) == 6
    # `math.isnan` and not `x == x`. The self-comparison was a real check --
    # NaN is the one float that is not equal to itself, so it fired exactly
    # when a term went NaN -- but it is indistinguishable, to a reader and to
    # the forge's `validation.trivialAssertions`, from an assertion that
    # compares an expression with itself and therefore cannot fail. Written
    # this way it says what it means and stays reachable; `record["term"]` is
    # a float, `float(term.detach())` as `adapt` records it.
    nan_steps = [record["step"] for record in history if math.isnan(record["term"])]
    assert nan_steps == [], f"the term went NaN at steps {nan_steps}"
    assert history[-1]["drift"] > 0.0


def test_the_provenance_claims_no_equation():
    """A protocol is not a term, and the provenance must not pretend otherwise."""
    assert __provenance__["revision"] is None
    assert "equations" not in __provenance__
    assert "claims_no_equation" in __provenance__


def test_every_declared_invariant_has_a_test():
    module = sys.modules[__name__]
    missing = [name for name in __provenance__["invariants"]
               if not hasattr(module, f"test_{name}")]
    assert missing == [], f"declared invariants with no test: {missing}"
