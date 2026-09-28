# Target-Degeneracy Diagnostic Specification

## Purpose

r25 declares a failure mode it leaves "al terreno empírico": `θ_t` receives
gradient only through Eqs. 37 and 38, so a target encoder that concentrates
its instances in an arbitrarily small region can drive Eq. (24)'s confidence
weights to zero until Eq. 38 stops penalizing — a collapse that would be
invisible in `targetAccuracy` alone, and unmeasured today. r25's own final
paragraph (r25:1087) prescribes the diagnostic and, in the same sentence,
excludes a different one that looks similar and does not detect this case.
This capability states which reading is required, which readings are
excluded, what it must report at its degenerate boundary, and that the
confidence weights r25's mechanism vanishes through are recorded.

## Requirements

### Requirement: The diagnostic is quadratic entropy over the trace-normalized unconditional target bag Gram

The degeneracy reading MUST be `quadratic_entropy(trace_normalize(K_tt))`,
computed over the UNCONDITIONAL target bag Gram matrix `K_tt` — the same
matrix already built every training step
(`bag_kernel_matrix(bags_t, bags_t, sigma)`) — using `MIL_CREDA.renyi`'s
existing `trace_normalize` (Eq. 9) and `quadratic_entropy` (Eq. 10). This
requires no new mathematics: it is one reading of a matrix the training step
already has.

This is r25's own prescription, quoted directly: "la entropía cuadrática
(Ec. 10) sobre la matriz de Gram de bolsas objetivo normalizada por traza
decae hacia su mínimo justo cuando la representación objetivo degenera."

#### Scenario: The instrument is computed from the existing unconditional target Gram

- GIVEN a training step's already-built `K_tt` (the unconditional target bag kernel matrix)
- WHEN the degeneracy reading is computed
- THEN it equals `quadratic_entropy(trace_normalize(K_tt))`, with no per-class conditioning applied to `K_tt` before the reading

### Requirement: The excluded intra-bag instruments MUST NOT be used

`MIL_CREDA.attention.uniform_self_similarity` and `effective_bag_size` MUST
NOT be wired as the degeneracy diagnostic, and MUST NOT be presented as
equivalent or as a substitute for it. r25 explicitly excludes them: "distíngase
del colapso de atención intra-bolsa, fenómeno distinto que no detecta este
caso." Their own docstrings and module comments confirm they measure whether
attention weights within one bag are uniform — a different claim from whether
the cloud of target bags itself has collapsed.

#### Scenario: A collapsed target bag cloud is invisible to the excluded instruments

- GIVEN a target batch constructed so its bag representations collapse toward a single point (a degenerate target cloud), while individual bags' attention weights remain non-uniform
- WHEN `uniform_self_similarity` or `effective_bag_size` is read on that batch
- THEN neither reports a value indicating collapse, demonstrating they do not detect this failure mode

### Requirement: The per-class weighted `H_target` inside `dependency_score` MUST NOT substitute for the diagnostic

`global_term.dependency_score`'s per-class, confidence-weighted `H_target`
term MUST NOT be used or presented as covering the degeneracy diagnostic
requirement. It is conditional on a class and on confidence weighting, so it
cannot detect a collapse that is uniform across classes — precisely the case
r25 describes, where the hard pseudo-label (Eq. 22) becomes the same for
every target bag and `C_act` (Eq. 37) shrinks to one class, at which point
`dependency_score`'s per-class reading has at most one class left to read
and stops being informative about the collapse as a whole.

#### Scenario: A class-uniform collapse defeats the per-class reading but not the required instrument

- GIVEN a target batch constructed so every target bag's hard pseudo-label collapses to the same class, driving `C_act` toward a single active class, while the unconditional target bag cloud is also collapsing
- WHEN the per-class weighted `H_target` inside `dependency_score` is read against the required diagnostic (`quadratic_entropy(trace_normalize(K_tt))`) on the same batch
- THEN the required diagnostic falls toward its minimum while the per-class reading fails to characterize the collapse (having at most one active class left to read), demonstrating the per-class reading is not a substitute

### Requirement: The reading decays toward its minimum as the target representation degenerates

The diagnostic MUST decay toward its bound as the target representation
degenerates: a rank-one, trace-normalized Gram gives `‖A‖²_F = 1`, hence
`H₂ = -ln(1) = 0`. This bound MUST be pinned by a test, not merely asserted
in prose.

#### Scenario: A rank-one target Gram reads the diagnostic's minimum

- GIVEN a deliberately constructed target batch whose bag Gram matrix `K_tt` is (or trace-normalizes to) rank-one
- WHEN the diagnostic is computed
- THEN it equals (or is numerically indistinguishable from) `0`, its declared minimum

#### Scenario: The reading falls monotonically as collapse increases

- GIVEN a sequence of deliberately constructed target batches with increasing degrees of representational collapse (decreasing effective rank of `K_tt`)
- WHEN the diagnostic is computed on each
- THEN the reading is non-increasing across the sequence, moving toward its minimum

### Requirement: The diagnostic reports rather than raises at the fully degenerate boundary

`trace_normalize` raises `ValueError` on a non-positive trace, and
`quadratic_entropy` raises `ValueError` on a zero Frobenius norm. A fully
degenerate target batch could reach either boundary. The diagnostic MUST
handle both raise conditions at its own call site and report a value or a
declared "diagnostic unavailable" state instead of propagating the
exception — because a diagnostic that crashes exactly when the failure mode
it exists to catch is at its most severe is the opposite of what a
diagnostic is for.

#### Scenario: A non-positive-trace target Gram is reported, not raised

- GIVEN a target batch constructed so `K_tt`'s trace is non-positive (for example, an all-zero Gram)
- WHEN the diagnostic is computed
- THEN it returns a reported value or an explicit "unavailable" result, and no `ValueError` propagates out of the diagnostic's call site

#### Scenario: A zero-Frobenius-norm trace-normalized Gram is reported, not raised

- GIVEN a target batch constructed so `trace_normalize(K_tt)` has zero Frobenius norm
- WHEN the diagnostic is computed
- THEN it returns a reported value or an explicit "unavailable" result, and no `ValueError` propagates out of the diagnostic's call site

### Requirement: The confidence weights' magnitude is recorded

`w_t = confidences(G_t)` (Eq. 24) is already computed every training step and
reported nowhere. r25's mechanism is that these vanish as the target
representation collapses. The magnitude of `w_t` (for example, its mean or
another declared summary statistic) MUST reach a record or a trace that a
reader can inspect after training — it MUST NOT remain computed and
discarded.

#### Scenario: The confidence-weight magnitude is reachable after a step

- GIVEN a completed adapted-arm training step
- WHEN a reader inspects that step's outputs
- THEN a summary of `w_t`'s magnitude for that step is available, distinct from being silently discarded

### Requirement: The diagnostic is readable during training, not only after

Because nothing ties the two encoders' embedding clouds together once they
train independently (unlike under the former shared extractor), a run that
drifts toward degeneracy MUST be observable while it trains, not only
inferred after the fact from a final accuracy number. The diagnostic MUST be
computable at a cadence that lets a reader see it move over the course of
training. The exact recording location and cadence (a per-run field, a
per-epoch trace beside the loss curves, or another shape) is a design
decision, not fixed by this requirement — but silence (a diagnostic that
exists in code but is never read during or after any run) does not satisfy
it.

#### Scenario: The diagnostic is available at more than one point across a training run

- GIVEN an adapted arm training for multiple epochs
- WHEN the diagnostic is read at more than one point over the course of that training
- THEN distinct values are obtainable, showing the reading's trajectory rather than only its value at the end
