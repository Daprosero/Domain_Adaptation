# Per-Domain Encoders Specification

## Purpose

Eq. (13) of r25 declares two independent feature extractors, `F^s_{θ_s}` and
`F^t_{θ_t}`, sharing no parameter and sharing only the codomain `R^d`. This
capability states which arms carry a target encoder and which do not, which
encoder embeds which domain at every call site that touches instance
embeddings — training, evaluation, and post-hoc reading — and what the system
does when a caller does not say which domain it holds. It also states the
parameter consequence and the floor arm's untouched guarantee.

This capability describes WHAT must be true of the embedding boundary. It
does not prescribe HOW a caller states a domain — an explicit argument on a
shared method and two separately named methods (for example
`embed_source`/`embed_target`) both satisfy every requirement below, and that
choice is left to `sdd-design`.

## Requirements

### Requirement: Two independent encoders for every adapted arm, one for the floor

Every arm declaring an adaptation term (`E`, `F`, `G`) MUST hold two
`FeatureExtractor` instances after construction: a source encoder and a
target encoder. The two MUST share no parameter object — proven by identity
(`is`) over `named_parameters()`, never by equal shapes or equal values. The
floor arm (`B`), which carries no adaptation term, MUST hold exactly one
encoder.

A target encoder MUST exist only where an adaptation term trains it. This is
not an independent design choice: it follows from the requirement below that
adaptation's gradient reach both encoders, because a target encoder with no
adaptation term to train it would be an untrained, frozen-at-initialization
module carrying no information — and the floor has no adaptation term.

#### Scenario: E, F and G each carry two independently-parameterized encoders

- GIVEN an arm declared with an adaptation term (`E`, `F`, or `G`)
- WHEN the arm is constructed
- THEN it holds two `FeatureExtractor` instances
- AND no parameter object of one appears in `named_parameters()` of the other

#### Scenario: The floor carries exactly one encoder

- GIVEN the floor arm `B`, which declares no adaptation term
- WHEN the arm is constructed
- THEN it holds exactly one `FeatureExtractor` instance

### Requirement: Both encoders share the same codomain

The source encoder's and the target encoder's output dimensionality MUST be
identical. This is the one thing Eq. (13) declares the two encoders to share:
the codomain `R^d`, not the map.

#### Scenario: Source and target encoders produce vectors of the same width

- GIVEN an adapted arm's two encoders
- WHEN each encodes one instance
- THEN both outputs have the same `output_dim`

### Requirement: No embedding without a declared domain

There MUST be no reachable code path that embeds instances without stating
which domain — source or target — they belong to. The system MAY satisfy
this through an explicit domain argument on a shared embedding entry point,
or through two separately named entry points, one per domain; either
satisfies this requirement, and the choice is a design decision, not a spec
decision. A caller that cannot state which domain it holds MUST be refused
rather than defaulted to either encoder, because a silent default returns a
plausible number instead of an error — the same failure shape as a domain
mismatch that nobody notices.

This requirement covers every call site that embeds raw instances into
`R^d`: the training forward (source and target), the evaluation forward, and
every post-hoc reader in `latent.py` that embeds source or target material.

#### Scenario: A caller that names its domain is served by the matching encoder

- GIVEN a batch of instances belonging to a stated domain (source or target)
- WHEN the batch is embedded
- THEN the embedding is produced by that domain's encoder and no other

#### Scenario: A caller that cannot state a domain is refused

- GIVEN a code path that would embed instances without a domain being knowable
- WHEN that path is reachable
- THEN construction or invocation refuses (raises) rather than silently choosing an encoder

### Requirement: Source accuracy is measured through the source encoder, target accuracy through the target encoder

`sourceAccuracy` MUST be measured by routing source material through the
source encoder (`θ_s`). `targetAccuracy` MUST be measured by routing target
material through the target encoder (`θ_t`), for every arm that has a target
encoder (`E`, `F`, `G`). This is a settled product decision, made because
doing nothing — evaluating target material through `θ_s`, which is what the
domain-blind evaluation forward does today — would make the 1.99x parameter
count buy nothing at inference and would make the degeneracy mode this
proposal's diagnostic capability exists to catch invisible in exactly the one
dimension declared `HIGHER`.

The floor (`B`) has only one encoder and continues to predict the target
through that same source-trained encoder — this requirement changes nothing
about the floor.

#### Scenario: Target accuracy is measured through the target encoder

- GIVEN a trained arm with a target encoder (`E`, `F`, or `G`)
- WHEN `targetAccuracy` is measured
- THEN target material is embedded by the target encoder before the classification head is applied

#### Scenario: Source accuracy is measured through the source encoder

- GIVEN a trained arm with a target encoder (`E`, `F`, or `G`)
- WHEN `sourceAccuracy` is measured
- THEN source material is embedded by the source encoder before the classification head is applied

#### Scenario: A mixed-up domain is detectable, not silent

- GIVEN an evaluation call site
- WHEN target material is (incorrectly) routed through the source encoder, or source material through the target encoder
- THEN there exists a test that fails on that mutation — the measured accuracy must differ from the correctly-routed measurement in at least one scenario constructed so the two encoders are not identical at evaluation time

### Requirement: The floor's guarantee is unchanged

The floor arm (`B`) MUST continue to hold exactly one encoder, train it on
source only, and predict the target through that same source-trained
encoder. Its two existing guard tests MUST pass with no edit to either. An
edit to either guard test is itself the signal that the floor moved, which
this change must not do.

(Previously: floor semantics were never separately specified; this
requirement fixes them as an explicit boundary now that a second encoder
exists elsewhere in the system.)

#### Scenario: The floor's guard tests stay green, unedited

- GIVEN the floor arm `B`'s two existing guard tests (`tests/test_arm_objectives.py:371` and `:461`)
- WHEN the full suite runs after this change lands
- THEN both tests pass with no change to their source text

### Requirement: Adaptation's gradient reaches both encoders

For every adapted arm, the adaptation term MUST put a non-zero gradient on
the source encoder's parameters AND on the target encoder's parameters, from
one backward pass. The global term's mixed block is built from both domains
and neither is privileged by sign or coefficient; the local term measures
the target bag against the source prototype. Both facts mean gradient must
be observable reaching both sides — asserting it on one side alone would
pass while the other side silently received no gradient, which a single
combined assertion cannot distinguish.

#### Scenario: The adaptation term alone trains both encoders

- GIVEN an adapted arm (`E`, `F`, or `G`) with only the adaptation term's loss backpropagated (no supervised term)
- WHEN the backward pass completes
- THEN at least one parameter of the source encoder has a non-zero gradient
- AND at least one parameter of the target encoder has a non-zero gradient

#### Scenario: A break in either encoder's gradient path is caught

- GIVEN a mutation that detaches the target encoder's output from the adaptation loss
- WHEN the gradient-reaches-both-encoders test runs
- THEN it fails, because it separately asserts non-zero gradient on each encoder rather than only on their combination

### Requirement: BatchNorm running statistics no longer cross domains

Under two independent encoders, a target forward MUST update only the target
encoder's running statistics (`running_mean`, `running_var`), and a source
forward MUST update only the source encoder's. Consequently, after a full
adapted-arm training step, the adapted arm's SOURCE encoder's running
statistics MUST match its floor (`B`) twin's running statistics — because
both are updated only by source forwards under an identical draw — while the
TARGET encoder's running statistics move, since only the target encoder sees
target material.

(Previously: one shared encoder meant the adapted arm's single encoder's
running statistics necessarily DIFFERED from its `B` twin's after a step,
because the shared encoder's normalization layers learned from both domains
while `B`'s learned from source alone. That comparison is inverted by this
change: the requirement now distinguishes the two encoders instead of merely
proving that one of them moved, which is a strictly stronger guarantee.)

#### Scenario: The adapted arm's source encoder statistics match its floor twin's

- GIVEN an adapted arm (`E`, `F`, or `G`) and its floor (`B`) twin, both taking one identical training step under the same generator draw
- WHEN the step completes
- THEN the adapted arm's source encoder's `running_mean` and `running_var` equal the floor's encoder's `running_mean` and `running_var`

#### Scenario: The adapted arm's target encoder statistics move

- GIVEN an adapted arm (`E`, `F`, or `G`) taking one training step
- WHEN the step completes
- THEN the target encoder's `running_mean` and `running_var` differ from their values before the step

### Requirement: The target encoder is constructed through the module-level `FeatureExtractor` name

The target encoder MUST be constructed by calling the same module-level
`FeatureExtractor` name that the source encoder is constructed from, inside
the same `__init__`. It MUST NOT be constructed through a local alias, a
separate import, or a different module path.

This is a spec requirement rather than an implementation preference: six test
files patch this exact module-level name
(`monkeypatch.setattr(wiring, "FeatureExtractor", Stub)`) or read
`model.encoder` directly — `tests/test_arm_objectives.py`,
`tests/test_figure_surfaces.py` (three sites),
`tests/test_independent_adaptation_schedules.py`,
`tests/test_latent_sigma_readers.py`, and `tests/test_attention_mechanisms.py`.
A construction path that bypasses that name would take all six files red for
a reason unrelated to Eq. (13).

#### Scenario: Patching the module-level FeatureExtractor name stubs both encoders

- GIVEN a test that patches `wiring.FeatureExtractor` with a stub before constructing an adapted arm
- WHEN the arm is constructed
- THEN both the source encoder and the target encoder are instances of the stub

### Requirement: The parameter asymmetry is reported where the count is read

`parameters` MUST read 11,247,434 for the floor (`B`) and 22,423,946 for
every adapted arm (`E`, `F`, `G`), measured from `harness.py`'s own
`sum(p.numel() for p in model.parameters())` expression — never asserted
from a document. The report MUST state, wherever `parameters` is printed,
that the floor and the adapted arms are not parameter-matched: a target
encoder exists only where an adaptation term trains it, so the asymmetry is
not an imbalance to correct, but a reader who meets 1.99x without that
sentence reads the size of the apparatus as part of what the method gained.

This requirement carries `MIL-CREDA/AGREED.md:164`; it does not re-decide
it.

#### Scenario: Parameter counts are measured, not asserted

- GIVEN a constructed floor arm and a constructed adapted arm
- WHEN `parameters` is computed for each
- THEN the floor measures 11,247,434 and the adapted arm measures 22,423,946

#### Scenario: The asymmetry sentence accompanies the count

- GIVEN a report or table that prints `parameters` across arms
- WHEN the floor and an adapted arm are shown together
- THEN the report states that the two are not parameter-matched and why, beside the count

### Requirement: Every post-hoc reader in `latent.py` names its domain explicitly

Every `latent.py` function that embeds source or target material for
post-hoc analysis (RKHS distance readings, neighbour tables, attention
spread, and the projection pipeline they feed) MUST route source material
through the source encoder and target material through the target encoder,
by stating the domain at the call, never by inheriting a default. This
covers, at minimum: the paired-domain readers that embed both domains
through one call (`bag_pairs`, `correspondence`), and the single-pool readers
that embed one domain generically (`represent`, `attention_spread`).

This requirement exists because the phase-two geometry reading
(`AGREED.md:222`, the RKHS distance) is exactly what would be silently wrong
if a target-side reader defaulted to the source encoder: it would return a
plausible number instead of an error.

#### Scenario: A paired reader routes each side to its own encoder

- GIVEN `bag_pairs` or `correspondence`, called with a source `BagSet` and a target `BagSet`
- WHEN each side is embedded
- THEN the source side is embedded by the source encoder and the target side is embedded by the target encoder

#### Scenario: A single-pool reader is told which domain it holds

- GIVEN `represent` or `attention_spread`, called with one `BagSet`
- WHEN that `BagSet` is embedded
- THEN the caller has stated which domain it is (source or target), and the matching encoder is used
- AND a call that cannot state the domain refuses rather than defaulting to either encoder

#### Scenario: The bag-figure projection test's premise survives, resolved

- GIVEN `test_the_bag_figure_keeps_the_projection_and_never_becomes_a_bipartite_diagram` (`tests/test_figure_surfaces.py:963`), whose assertions concern `latent._embed` — the plotting projection called once per panel on the concatenation of `bag_pairs`' `sourceRows` and `targetRows` — and never concern `model.instance_embeddings` or which encoder produced those rows
- WHEN `bag_pairs` is updated to satisfy the paired-reader requirement above (source via the source encoder, target via the target encoder)
- THEN `sourceRows` and `targetRows` still land in the shared codomain `R^d`, `_embed` is still called exactly once per panel on their concatenation, and the test's existing assertions pass unedited
- Note: this scenario resolves what the proposal recorded as undetermined. Read from the code, `latent._embed` is a UMAP-style dimensionality-reduction stand-in for plotting, distinct from the domain encoders; the test's "ONE embedding call" premise refers to that projection call, which this change does not touch. This is a finding, not an assumption: `test_figure_surfaces.py:963` requires no re-pointing under two encoders, conditioned only on `bag_pairs` satisfying the paired-reader requirement above.

### Requirement: One optimizer trains both encoders

Both encoders' parameters MUST be registered under a single optimizer
instance (`torch.optim.Adam(model.parameters(), lr=config.LR)`) sharing one
learning rate. The system MUST NOT introduce a second optimizer or a
per-encoder learning rate; `config.LR` remains one fixed declared value.

#### Scenario: The target encoder's parameters are in the same optimizer as the source encoder's

- GIVEN a constructed adapted arm and its optimizer
- WHEN the optimizer's parameter groups are inspected
- THEN every parameter of the target encoder appears in the same optimizer instance as every parameter of the source encoder, under the same learning rate

### Requirement: Prose that asserted a shared encoder is corrected with it

`wiring.py`'s module docstring ("Every arm shares the same encoder…"),
`config.py`'s `normalization` axis rationale, and `__init__.py`'s
justification for pooling `parameters` ("bit-identical across every run")
MUST no longer assert a shared encoder or a parameter count identical across
every arm once this change lands. The `parameters` declaration itself (that
it is a pooled, descriptive dimension) is unaffected — only the justification
comment, which becomes false the moment `B` reads 11,247,434 and an adapted
arm reads 22,423,946, is corrected.

#### Scenario: No surviving prose claims one shared encoder

- GIVEN the corrected `wiring.py`, `config.py`, and `__init__.py` prose
- WHEN read after this change lands
- THEN none of it asserts that every arm shares one encoder or that `parameters` is bit-identical across arms
