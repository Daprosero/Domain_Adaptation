"""A sample-level CREDA trainer, and the two forks that live in the loop.

Three of the four things ported down from MIL-CREDA are properties of the
adaptation term itself and live on `CREDALoss` (`models.py`). The two that are
properties of the TRAINING LOOP live here, because no loss function can hold
them: `dual_encoder` is an architecture, and `normalize_supervised` is about
how the supervised term is composed with the adaptation one.

**Why a sample-level trainer at all.** Every measurement of the adaptation term
so far has run through the bag machinery, which means a notebook that wants to
ask one question about the term has to build bags, a relevance selector and a
pooling rule first. Here the unit is the sample, the data arrives as tensors,
and the whole step is short enough to read -- so each fork can be exercised in
seconds instead of in a campaign. This is not a replacement for
`CREDA.training_pipeline`, which runs the ResNet campaign and is untouched.

**Data as tensors, never as datasets.** The callers are notebooks and probes.
A `Dataset`/`DataLoader` pair would put the material behind an iteration
protocol with its own shuffling and its own random state, and the one thing
this module has to be able to promise is exactly which draws the generator
performed -- see `training_step`.

**The backbone is the caller's.** `encoder_factory` is a zero-argument callable
returning a module with an `output_dim` attribute. Nothing is imported for it:
this module may not reach into `MIL_CREDA_Benchmark`, and a baseline that
depends on the benchmark tree it is the baseline for is not independent.

**Every default leaves faithful CREDA as it is.** `dual_encoder=False` is one
encoder serving both domains; `normalize_supervised=False` is CREDA's own
per-sample cross-entropy. `True` selects ours, on each axis separately.

**Two things that are measured and are not up for redesign.** There is no
trainable bandwidth: with features frozen, a learned sigma grows about
seventeenfold and drives the adaptation term down two hundredfold by saturating
the kernel, which escapes the objective rather than minimizing it. And there is
no statistics re-estimation step after a transfer: in the two-encoder
architecture each encoder's normalization statistics already describe the domain
it serves, by construction, so a re-estimation pass would have nothing to
correct.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from CREDA.bounds import source_loss
from CREDA.models import CREDALoss
from CREDA.schedules import CREDA_CEILING, creda_ramp

__provenance__ = {
    "revision": "research-concept-r33.md",
    "sections": ["3", "5"],
    "equations": ["21", "39"],
    "invariants": [
        "the_generator_is_consumed_identically_whether_or_not_alignment_is_on",
        "the_floor_never_lets_a_target_sample_reach_an_encoder",
        "the_transfer_copies_parameters_and_leaves_buffers_alone",
        "the_transfer_happens_after_the_optimizer_step_at_the_declared_interval",
        "the_supervised_term_is_normalized_only_when_asked_and_carries_coefficient_one",
    ],
}

#: How many OPTIMIZER steps pass between two source-to-target transfers.
#: `config.SOURCE_TO_TARGET_SYNC_STEPS`' value, restated here because this
#: module may not import that tree. What was measured on the bag side is that
#: the outcome is flat from every step to every seventieth, so the mechanism's
#: only real degree of freedom is whether the re-anchoring happens at all --
#: what decides is that there is MORE THAN ONE transfer and that they occur
#: DURING training. An interval larger than the run's total step count
#: transfers NEVER, and nothing warns.
DEFAULT_SYNC_STEPS = 7


class SampleCREDA(nn.Module):
    """One encoder (or two), a linear head, and CREDA's adaptation term.

    `adaptation=False` is the floor: the supervised term alone. It is a
    separate flag from `dual_encoder` on purpose -- a floor has no second
    encoder to build, but an adapted arm may still choose to run on one.
    """

    def __init__(self, encoder_factory, classes, *, adaptation=True,
                 dual_encoder=False, normalize_supervised=False,
                 creda=None, epsilon=1e-8):
        super().__init__()
        self.classes = classes
        self.adaptation = adaptation
        self.dual_encoder = dual_encoder
        self.normalize_supervised = normalize_supervised
        self.epsilon = epsilon

        # The source encoder is built FIRST, always, so that two arms differing
        # only in `dual_encoder` get a bit-identical source encoder under the
        # same seed: whatever the second construction draws, it draws after.
        self.encoder = encoder_factory()
        self.target_encoder = (
            encoder_factory() if (adaptation and dual_encoder) else None
        )
        self.head = nn.Linear(self.encoder.output_dim, classes)

        # `lambda_creda` stays at one and the coefficient lives in the ramp's
        # ceiling: `CREDA.schedules` moved it there precisely so the effective
        # coefficient is one number in one place rather than a product
        # assembled across two files.
        self.creda = creda if creda is not None else CREDALoss(
            sigma="auto", lambda_creda=1.0, use_entropy_weighting=True)

    # ------------------------------------------------------- representation

    def source_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Encode source samples through `theta_s`."""
        return self.encoder(x)

    def target_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Encode target samples through `theta_t`, falling back to `theta_s`.

        The fallback is what `dual_encoder=False` means: one encoder, reached
        from both entry points, which is faithful CREDA. With the flag on, each
        domain goes through its own encoder -- `wiring.Arm`'s two-encoder shape
        on the bag side, where `instances` comes from `self.encoder` and `H_t`
        from `self.target_encoder`.
        """
        encoder = self.encoder if self.target_encoder is None else self.target_encoder
        return encoder(x)

    # ------------------------------------------------------------ transfer

    def sync_target_from_source(self) -> None:
        """Transmit `theta_s`'s parameters into `theta_t`. A no-op with one encoder.

        **Procedure, never a term.** Under `no_grad` and in place: the copy
        builds no graph, so the adaptation term's gradient never reaches
        `theta_s` through `theta_t`, and an optimizer keeps its own state
        because the tensors are the same objects it was built over. The two
        encoders remain distinct tensors sharing none, and they diverge again
        from the next step on.

        **Only parameters. The buffers stay the target's own.** Measured on the
        bag side, at the shortest interval: carrying the normalization
        statistics over as well was WORSE, 0.7222 against 1.0000. A running
        statistic is a reading of the material a domain actually saw, not a
        parameter of the map -- `named_parameters()` is what this walks, and
        that is the whole distinction.
        """
        if self.target_encoder is None:
            return
        source = dict(self.encoder.named_parameters())
        with torch.no_grad():
            for name, parameter in self.target_encoder.named_parameters():
                parameter.copy_(source[name])

    # ---------------------------------------------------------------- draw

    def draw_target(self, pool_size: int, batch_size: int,
                    generator: torch.Generator) -> torch.Tensor:
        """The step's unlabelled target indices. Plain: nothing is stratified by
        a label the method is not allowed to see."""
        order = torch.randperm(pool_size, generator=generator)
        return order[:batch_size]

    # ---------------------------------------------------------------- step

    def training_step(self, x_s: torch.Tensor, y_s: torch.Tensor,
                      target_pool: torch.Tensor, ramp: float,
                      generator: torch.Generator,
                      batch_size: int | None = None) -> dict:
        """One step's objective, composed as Eq. (39) composes it.

        The supervised term carries coefficient ONE and only the adaptation
        term carries the free coefficient. The global scale of an objective does
        not move its minimizer, so fixing the supervised term as the reference
        costs nothing and leaves exactly one coefficient to schedule.

        `normalize_supervised` chooses which supervised term that is. Off, it is
        CREDA's own per-sample cross-entropy, used exactly as it was written.
        On, it is Eq. (21): the averaged negative log-likelihood divided by its
        own exact supremum `B_src = ln(1 + 1/eps)`, which puts it in [0, 1)
        beside an adaptation term that is already bounded there -- the condition
        under which a coefficient of one actually weighs the two equally.
        `Y_source` is one-hot over the classes with one row per unit, and the
        unit here is the sample.

        **The generator is consumed identically whether or not alignment is
        on.** `draw_target` is called for EVERY arm, floor included, so two arms
        at the same seed see the same stream of draws and differ only in the
        term under test. What a floor never does with the indices it drew is
        index the pool or hand anything to an encoder. The reason is measured
        and it is not a stylistic one: a floor that forwards target samples
        through a BatchNorm encoder receives the target's statistics for free,
        which IS adaptation, and an adapted floor makes every rung above it look
        like it contributes nothing -- worth about +0.28 the one time it went
        wrong here.
        """
        if batch_size is None:
            batch_size = x_s.shape[0]

        f_s = self.source_embeddings(x_s)
        logits = self.head(f_s)
        if self.normalize_supervised:
            supervised = source_loss(                                 # Eq. (21)
                F.softmax(logits, dim=1),
                F.one_hot(y_s, self.classes).to(logits.dtype),
                self.epsilon,
            )
        else:
            supervised = F.cross_entropy(logits, y_s)

        target_indices = self.draw_target(
            target_pool.shape[0], batch_size, generator)

        adaptation = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.adaptation:
            x_t = target_pool[target_indices.to(target_pool.device)]
            f_t = self.target_embeddings(x_t)
            g_t = F.softmax(self.head(f_t), dim=1)
            adaptation = self.creda(f_s, f_t, y_s, g_t)

        return {                                                      # Eq. (39)
            "loss": supervised + ramp * adaptation,
            "supervised": supervised,
            "adaptation": adaptation,
            "ramp": ramp,
            "targetIndices": target_indices,
        }


def train(model: SampleCREDA, source_x: torch.Tensor, source_y: torch.Tensor,
          target_x: torch.Tensor, *, epochs: int, batch_size: int,
          generator: torch.Generator, delta: float = 10.0,
          ceiling: float = CREDA_CEILING, learning_rate: float = 1e-3,
          optimizer: torch.optim.Optimizer | None = None,
          sync_steps: int = DEFAULT_SYNC_STEPS) -> list[dict]:
    """Run the loop and hand back its curve, one record per step.

    The coefficient is `creda_ramp`, CREDA's published schedule, scaled to
    `ceiling`. It is ZERO at the first epoch, which is a property of the curve
    and not an initialization trick: a run of a single epoch therefore trains
    with no adaptation at all, and a short run is not a slower version of a long
    one but a run with almost no warm-up.

    The transfer happens AFTER `optimizer.step()` and every `sync_steps`
    optimizer steps, counted across the whole run. Both halves of that matter:
    transmitting before the update would re-anchor `theta_t` and the same step
    would immediately move it again, which measures an interval shifted one step
    from the one it names; and the count is of optimizer steps rather than of
    epochs because that is the unit the interval was measured in.

    `optimizer` is the caller's to supply when it wants one; the Adam built here
    is a default for a probe, not a declared part of any comparison.
    """
    if epochs < 1:
        raise ValueError("a run of fewer than one epoch is not a run")
    if batch_size < 1:
        raise ValueError("a step over fewer than one sample is not a step")
    if sync_steps < 1:
        raise ValueError(
            "an interval below one is not an interval; zero would mean never, "
            "and that is the absence of the mechanism rather than a value of it")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    curve: list[dict] = []
    optimizer_steps = 0
    for epoch in range(epochs):
        coefficient = creda_ramp(epoch, epochs, delta=delta, ceiling=ceiling)
        order = torch.randperm(source_x.shape[0], generator=generator)
        for start in range(0, order.shape[0], batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            step = model.training_step(
                source_x[batch], source_y[batch], target_x, coefficient,
                generator, batch_size=batch_size)
            step["loss"].backward()
            optimizer.step()
            optimizer_steps += 1
            if optimizer_steps % sync_steps == 0:
                model.sync_target_from_source()
            curve.append({
                "epoch": epoch,
                "step": optimizer_steps,
                "ramp": coefficient,
                "supervised": float(step["supervised"].detach()),
                "adaptation": float(step["adaptation"].detach()),
                "loss": float(step["loss"].detach()),
                "synced": optimizer_steps % sync_steps == 0,
            })
    return curve
