"""CREDA's two-phase sample-level adaptation, with the free regime as default.

`sample_adaptation.py` trains the supervised and adaptation terms together in
one phase and requires a transfer interval of at least one. This module is the
other arrangement, and it is the one every measurement of the toy programme
used:

1. **Phase one** -- `train_source` fits the encoder and the head on source data
   alone, then freezes both in place. Nothing after this point writes to them.
2. **Phase two** -- `clone_for_target` copies those weights to a second encoder
   exactly once, and `adapt` lets the adaptation term alone move it. The source
   passes through the source encoder under `no_grad`; the target passes through
   the target encoder.

**The free regime is the default because it is what was measured.** Re-copying
the weights every seven steps keeps 6% of the gain without BatchNorm: `0.5943`
free against `0.2754` leashed over a `0.2572` floor on one toy transfer, and
`0.2906` against `0.1847` over `0.1798` on another. With BatchNorm the leashed
arm is indistinguishable from adapting the statistics and nothing else
(`0.8990` against `0.8932`), which is why a number measured under the leash
says nothing about the formulation. `sync_steps=FREE` is therefore zero, and a
leash is something a caller asks for rather than something it inherits.

**This arrangement is not a claim from the managed revision.** The revision
declares an objective, not a training protocol, and nothing in it fixes which
encoder sees which domain or when weights transfer. The provenance below names
the invariants this module holds and deliberately claims no equation.

**No ramp, by default, and that is a decision.** Every measurement in this
arrangement ran the term at coefficient one. Under Adam a constant coefficient
is very nearly a no-op -- the gradient, its first moment and the square root of
its second moment all scale together -- so only a coefficient that VARIES over
the run can change the outcome. `ramp=True` applies `creda_ramp`, whose time
variation is the part that could matter; leaving it off keeps the module
agreeing with the numbers that justified it.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import CREDA_CEILING, creda_ramp

__provenance__ = {
    "revision": None,
    "claims_no_equation": "a training protocol, not a term; see the module docstring",
    "invariants": [
        "the_first_phase_freezes_the_source_encoder_and_the_head",
        "the_target_encoder_starts_bit_identical_to_the_source",
        "the_source_never_passes_through_the_target_encoder",
        "the_free_regime_copies_the_weights_exactly_once",
        "the_leash_returns_the_weights_to_the_source_at_its_interval",
        "the_adaptation_term_is_the_only_force_on_the_target_encoder",
    ],
}

#: No transfer at all: one copy at the start and the target encoder is on its
#: own from there. The measured default -- see the module docstring.
FREE = 0


def _frozen(module: nn.Module) -> bool:
    return all(not p.requires_grad for p in module.parameters())


def parameter_distance(a: nn.Module, b: nn.Module) -> float:
    """The Euclidean distance between two modules' parameter vectors.

    Buffers are deliberately excluded. A BatchNorm buffer is estimated from
    data rather than learned, so including it would report a module as having
    drifted when all that happened is that it saw a different domain -- which
    is the one thing this number must not confuse.
    """
    with torch.no_grad():
        total = sum(((pa - pb) ** 2).sum() for pa, pb in zip(a.parameters(), b.parameters()))
        return float(torch.as_tensor(total).sqrt())


def train_source(encoder: nn.Module, head: nn.Module,
                 x: torch.Tensor, y: torch.Tensor, *,
                 epochs: int, batch_size: int,
                 generator: torch.Generator,
                 learning_rate: float = 1e-2) -> None:
    """Phase one: fit on the source alone, then freeze the encoder and the head.

    Freezing happens here rather than being left to the caller, because every
    guarantee phase two offers rests on it: if these parameters can still move,
    the source's own accuracy stops being an invariant and a later table cannot
    tell adaptation from the source drifting underneath it.

    A batch of one is skipped rather than passed through: a single sample makes
    BatchNorm's batch variance undefined, and the exception it raises names the
    tensor rather than the cause.
    """
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=learning_rate)
    for _ in range(epochs):
        encoder.train()
        head.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:
                continue
            loss = F.cross_entropy(head(encoder(x[index])), y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoder.eval()
    head.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)


def clone_for_target(encoder: nn.Module) -> nn.Module:
    """The one copy: a second encoder starting exactly where the source ended.

    It carries the source's buffers too, so a BatchNorm target encoder begins
    with the source's statistics and re-estimates them from target batches as
    it trains. That re-estimation is what a leashed run keeps even when it
    discards every weight update, which is why the two cannot be read as one
    mechanism.
    """
    target = copy.deepcopy(encoder)
    for parameter in target.parameters():
        parameter.requires_grad_(True)
    return target


def adapt(encoder_s: nn.Module, head: nn.Module, encoder_t: nn.Module,
          source_x: torch.Tensor, source_y: torch.Tensor,
          target_x: torch.Tensor, *,
          creda, steps: int, batch_size: int,
          generator: torch.Generator,
          learning_rate: float = 1e-3,
          sync_steps: int = FREE,
          ramp: bool = False,
          delta: float = 10.0,
          ceiling: float = CREDA_CEILING) -> list[dict]:
    """Phase two: the adaptation term alone moves the target encoder.

    `encoder_s` and `head` must already be frozen; this refuses otherwise rather
    than silently training them, because that failure is invisible in the loss
    and shows up only as a source accuracy that stopped being constant.

    The source's features are computed under `no_grad` through `encoder_s`, so
    no gradient path to the source exists even if it were left unfrozen. The
    pseudo-posteriors come from the frozen head reading the target encoder's
    output, also under `no_grad`: the discrete class decision inside the term is
    not differentiable, and routing a graph through it would only suggest
    otherwise.

    Returns one record per step carrying the term's value, the parameter
    distance to the source, and whether a transfer happened -- the three
    quantities a caller needs to tell a free run from a leashed one without
    re-deriving them.
    """
    if not _frozen(encoder_s):
        raise ValueError("phase two requires a frozen source encoder; run train_source first")
    if not _frozen(head):
        raise ValueError("phase two requires a frozen head; run train_source first")
    if sync_steps < 0:
        raise ValueError("a transfer interval cannot be negative")

    optimizer = torch.optim.Adam(encoder_t.parameters(), lr=learning_rate)
    history: list[dict] = []
    for step in range(steps):
        encoder_t.train()
        i = torch.randperm(len(source_x), generator=generator)[:batch_size]
        j = torch.randperm(len(target_x), generator=generator)[:batch_size]
        with torch.no_grad():
            f_s = encoder_s(source_x[i])
        f_t = encoder_t(target_x[j])
        with torch.no_grad():
            g_t = F.softmax(head(f_t), dim=1)
        term = creda(f_s, f_t, source_y[i], g_t)
        if term.grad_fn is None:
            # Autograd's own message here names a tensor and never the cause.
            # In this arrangement the source encoder and the head are frozen AND
            # run under `no_grad`, so the target encoder is the only
            # differentiable path into the loss: a term that does not read the
            # target features cannot move anything, and that is structural
            # rather than a matter of which parameters were left trainable.
            raise ValueError(
                "the adaptation term has no gradient path to the target encoder; "
                "in this arrangement it is the only path there is")
        coefficient = creda_ramp(step, steps, delta, ceiling) if ramp else 1.0
        optimizer.zero_grad()
        (coefficient * term).backward()
        optimizer.step()
        synced = bool(sync_steps) and (step + 1) % sync_steps == 0
        if synced:
            with torch.no_grad():
                for (name_t, p_t), (name_s, p_s) in zip(
                        encoder_t.named_parameters(), encoder_s.named_parameters()):
                    if name_t != name_s:
                        raise ValueError(f"parameter names do not align: {name_t} vs {name_s}")
                    p_t.copy_(p_s)
        history.append({
            "step": step,
            "term": float(term.detach()),
            "drift": parameter_distance(encoder_t, encoder_s),
            "synced": synced,
        })
    encoder_t.eval()
    return history
