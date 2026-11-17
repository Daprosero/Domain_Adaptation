# Proposal: Two encoders, because each domain has its own space

## Intent

The paper stands at **r25**. The implementation is bound to **r21**
(`config.REVISION = "research-concept-r21.md"`, config.py:300). The net r21→r25 diff is
three regions and only one touches what is measured: **r22 declared two independent
per-domain encoders sharing no parameters**, and r25 carries it as Eq. (13) —

> `F^s_{θ_s} : X^s → R^d` and `F^t_{θ_t} : X^t → R^d` … "θ_s y θ_t agrupan los pesos de
> cada codificador, **sin compartir ninguno entre sí**. Lo único que ambos comparten es el
> codominio R^d, que es todo lo que la geometría posterior necesita."
> (r25:375-385)

The code has **one** shared encoder: `self.encoder = FeatureExtractor(...)`, built once per
arm at `wiring.py:95`, and reached through the single forward call at `wiring.py:134`. The
operator has confirmed that two encoders is the intent, so the mathematics is the authority
and the implementation is behind it.

**The motivation r25 gives is not the motivation that applies here, and the record says so.**
r25 derives the non-sharing from possibly-different input spaces (`X^s`, `X^t`). In this
bench all six transfers are resized to one common input, so `p_s = p_t` and that formal
premise does not hold in the experiment. The operator was asked exactly this and answered
that two encoders is what was sought **in itself** — "un segundo codificador para la fuente:
dos codificadores, uno por dominio, sin parámetros compartidos, es lo buscado y no un efecto
colateral de generalizar los espacios de entrada" (`position.jsonl` discuss, 2026-09-28T14:46:14Z).
Recorded here rather than left out, because a later reader who notices `p_s = p_t` would
otherwise re-litigate a decision that was already taken on other grounds.

**What this does not retract.** The shared extractor was agreed on 2026-09-17 after a
measurement, and that measurement stands. It measured **frozen** normalization for the target
forward, which normalized the target with accumulated statistics while the source used its own
batch statistics (M→U at step 0: median squared distance **1399 against 117**, off-diagonal bag
kernel 0.004 against 0.43). The shared extractor was the fix chosen for that artefact. Two
encoders that **both train** is a third option that measurement never examined. What the newer
formulation supersedes is the fix, never the finding. This is already ruled and written:
`AGREED.md` `## Reversed`, the 2026-09-28 paragraph (commit `8ed79de`), and it is cited here
rather than re-derived.

## Scope

### In Scope

1. **`config.REVISION` moves to r25, in its own commit.** One line that invalidates 67
   on-disk records; the history should show when that happened and nothing else should be in
   that diff.
2. **A target encoder for `E`, `F` and `G`, at the seam already named for it.**
   `_target_embeddings` (wiring.py:221-240) whose own docstring calls itself "where a freeze
   would have to be reintroduced" and notes that `tests/test_arm_objectives.py` points its
   guard at that exact name. Built through the same module-level `FeatureExtractor` name as
   the source encoder (see Approach: six test files monkeypatch that name).
3. **The floor `B` is untouched.** One encoder; it trains on source and predicts the target
   through that same source-trained encoder. Its two guard tests
   (`test_arm_objectives.py:371` and `:461`) stay green **unedited**.
4. **The four shared-encoder tests are re-pointed, not deleted** — and one of them inverts.
   See Approach.
5. **Every consumer that embeds target material chooses an encoder explicitly.** Five
   `model.instance_embeddings(...)` call sites in `latent.py` (:448, :601, :602, :649, :920,
   :921) take a `bagset` and are domain-blind by signature; `latent.analyse` (:657-659) drives
   `represent` once per domain through the same call. Under two encoders every one of those
   would silently put target material through the **source** encoder and return a plausible
   number. This is in scope because the phase-two geometry reading
   (`AGREED.md:222`, the RKHS distance) is exactly what would be wrong.
6. **The degeneracy instrument r25 asks for, in the form r25 prescribes.** See Approach —
   the correct instrument is not the one the handoff named.
7. **Prose that would outlive its mechanism is corrected with it**: `wiring.py:4` ("Every arm
   shares the same encoder…"), `config.py:886-892` and `:902-905` (the `normalization` axis
   rationale), and `__init__.py:477-480`, whose justification for putting `parameters` in
   `poolable` is "the replication measured `parameters` **bit-identical across every run**" —
   true while every arm had one encoder, false the moment `B` reads 11,247,434 and `G` reads
   22,423,946.
8. **The report states the parameter asymmetry where the count is read.** Already settled and
   already written: `AGREED.md:164` (commit `c55526a`). This change carries it, it does not
   re-decide it.
9. **The pilot walk and the nine-account rehearsal are re-run, under the existing launch
   authorization and not around it.** The revision bump un-ticks the six-step position and
   invalidates every notebook stamp, so the walk is owed. Spending metered quota remains the
   operator's own gate; this change delivers the launch precondition, never the launch.

### Out of Scope

- **A second optimizer, or a separate learning rate for the target encoder.** `harness.py:641`
  and `:818` build `torch.optim.Adam(model.parameters(), lr=config.LR)`; a second registered
  `nn.Module` is picked up automatically, and `parameters` (harness.py:737,
  `sum(p.numel() for p in model.parameters())`) recounts itself for free. Two optimizers would
  be a deliberate addition and are not wanted. Nothing in `config` today can express a
  different rate for one submodule — `LR` is one fixed declared value, explicitly not searched
  (config.py:864-870).
- **Re-anchoring the kernel bandwidth.** The measurement (below) reproduced
  `config.KERNEL_SIGMA = 36.135014304860874` byte-exactly for M. Nothing moves, so
  `harness.hyperparameter_drift` (:405-415) does **not** fire on this change and
  `KERNEL_SIGMA_RANGE` — derived as `(KERNEL_SIGMA / 10, KERNEL_SIGMA * 10)`, config.py:774 —
  needs no re-derivation.
- **`src/CREDA`.** `FeatureExtractor` (models.py:50) is **instantiated twice and edited zero
  times.** Prior work, out of scope by project rule.
- **The parent forge `papersmith-ai`.** Every edit stays inside
  `implementations/Domain_Adaptation`.
- **The `"creda"` adaptation branch.** `_creda_term` (wiring.py:312-318) also calls
  `_target_embeddings` and reads `self.encoder.output_dim`, but no declared arm sets
  `spec["adaptation"] == "creda"` (wiring.py:117-121). It is kept reachable-by-declaration and
  not extended.
- **Deciding the target-evaluation path.** Named under **Open Design Questions (1)** and
  deliberately not decided here. It is the one question the settled decisions do not answer.
- **The sibling change `a-transfer-that-never-came-back-is-not-one-nobody-searched`.** See
  Dependencies: not a dependency, not a blocker, and not touched.

## Capabilities

`openspec/specs/` does not exist in this repository, and `openspec/changes/archive/` is empty.
There are no established capability names to extend and nothing to delta.

### New Capabilities

- `per-domain-encoders`: which arms carry a target encoder and which do not; that `θ_s` and
  `θ_t` share no parameter and share only the codomain `R^d`; which encoder embeds which
  domain at **every** call site — training, evaluation and post-hoc reading — and what refuses
  rather than guesses when a caller does not say. Includes the parameter consequence
  (11,247,434 against 22,423,946) and the floor's untouched guarantee.
- `revision-bound-records`: the revision the benchmark declares, and what happens to every
  record, manifest and notebook stamped under another one. The refusal machinery already
  exists (`harness.py:393-403`, `latent.py:113`, `latent.py:393-397`); what this capability
  must state is the **disposition** of the 67 r21-stamped artifacts the bump invalidates.
- `target-degeneracy-diagnostic`: the failure mode r25 declares and leaves "al terreno
  empírico" — that `θ_t` receives gradient only through Eqs. 37 and 38, so a target encoder
  concentrating its instances in an arbitrarily small region can drive the confidence weights
  of Eq. (24) to zero until Eq. 38 stops penalizing — together with the reading that detects
  it and where that reading is recorded.

### Modified Capabilities

None.

## Approach

### The rung `B→E` carries ONE difference, not two (settled — recorded, not open)

The orchestrator initially read the target encoder as a second difference on that rung. The
operator corrected it, and the correction is load-bearing enough to write down as settled:

- The global term is `J_c = (H₂(s) + H₂(t))/2 − H₂(mix)` and its mixed block is built from both
  sides. `global_term.py:59-60`'s own docstring: "Both marginals carry the same sign and the
  same coefficient, so **neither is privileged**."
- The local term measures the target bag against the **source** prototype in the RKHS:
  `local_distance` (local_term.py:85-101) returns
  `self_similarity − 2·(π @ cross) + π @ K_ss @ π`, where `K_ss` is the source Gram.

So adaptation's gradient reaches **both** encoders, the target encoder is the side adaptation
trains, and the two are inseparable. A target encoder exists only where an adaptation term
trains it — which is why the floor has none **by construction** and not as an omission
(`AGREED.md:164`; `position.jsonl` discuss, 2026-09-28T20:36:13Z).

### The bandwidth, measured

Measured by mirroring `tools/measure_kernel_sigma.py` exactly — same material, `train_idx`
role, `model.train()` so batch statistics rather than accumulated, `torch.no_grad()`,
`BAGS_PER_STEP` chunks, `sqrt(median(off-diagonal squared) + 1e-6)`:

| | median d² | sigma |
| --- | --- | --- |
| within M | 1305.7393 | 36.1350 |
| within U | 1213.4900 | 34.8352 |
| within S | 1231.0259 | 35.0860 |
| across M–U | 1259.1362 | 35.4843 |
| across M–S | 1267.4597 | 35.6014 |
| across U–S | 1221.6569 | 34.9522 |

Every within-domain and cross-domain value sits within about **4%** of every other, and all of
them sit in the middle of the declared search range. The M row reproduces
`config.KERNEL_SIGMA = 36.135014304860874` byte-exactly, which validates the procedure rather
than the conclusion. **So one bandwidth is defensible and nothing needs to move.**

**The limit of that measurement is this change's central risk, and it is stated here rather
than buried.** Both encoders start from the same pretrained `resnet18`, so at step 0 they are
identical and this measures only that the **image** distributions are comparable. It says
nothing about what 20 epochs of independently trained encoders do, where — unlike under a
shared extractor — **nothing ties the two clouds together**. What limits the damage is that
`kernelSigma` is a searched dimension (`config.HYPER_DIMENSIONS`, config.py:1453; registered
log-scale at `harness.py:1687`), and its per-transfer winners under one encoder already spanned
30.93 to 262.92. The search adapts, within one decade of a source-measured anchor. Whether one
decade is still the right bracket under two encoders is a question only a run answers.

### The four shared-encoder tests

Re-pointed, not deleted. Each carries its own declared reachable-red, and one of those reds
**is this change**:

1. `test_arm_objectives.py:496` `test_an_adapted_arms_target_forward_carries_gradient_to_the_encoder`
   (E, F, G). Asserts the adaptation term alone puts a non-zero gradient on `arm.encoder`.
   Under two encoders that claim **splits**: adaptation must gradient both, and the existing
   assertion would keep passing on the source encoder alone while saying nothing about the
   target one. Its docstring's premise — "Eq. (39)'s whole point is one encoder both domains'
   inputs update" — is now false and goes with it.
2. `test_arm_objectives.py:557` `test_an_adapted_arms_target_forward_updates_running_statistics`.
   `assert all(module is arm.encoder for module in seen)` is the single assertion most directly
   contradicting Eq. (13). **Its own declared red mutation is "have it build and forward through
   a fresh `FeatureExtractor(...)` instead of `self.encoder`" — which is exactly this change.**
   So this is the red this work starts from: `strict_tdd: true` is satisfied without writing a
   new failing test to obtain a first red. Note it is parametrized over `_adapted_arm_ids()`
   (:545-550), derived from `config.ARMS`, so it covers E, F and G without a hand-written list.
3. `test_arm_objectives.py:617`
   `test_every_adapted_arms_full_training_step_updates_running_statistics_a_source_only_twin_does_not`.
   **This one inverts.** It asserts the adapted arm's `encoder.bn.running_mean` **differs** from
   its `B` twin's after one step. Under two encoders the target forward moves the **target**
   encoder's statistics, so the source encoder's will now **match** `B`'s and the current
   assertion fails for the right reason. Re-pointed at the target encoder it becomes
   *stronger*: it distinguishes the two encoders instead of merely proving one of them moved.
4. `test_figure_surfaces.py:963`
   `test_the_bag_figure_keeps_the_projection_and_never_becomes_a_bipartite_diagram`. Its stated
   premise is "both domains go through **ONE** embedding call"; it asserts
   `count == config.CLASSES * 2` per panel and that both clouds share an abscissa range
   (:989-1002). Under Eq. (13) the shared space is the **codomain** `R^d` and not the map, so
   the figure's claim survives in a weaker and still-true form. **Whether this test fails or
   merely stops meaning what it claims is not determinable by reading alone** — it depends on
   the encoder the projection ends up using, which is Open Design Question (1). Recorded as
   undetermined rather than guessed.

**Mechanically affected, for a reason unrelated to the claim.** Six test files patch the
module-level name — `monkeypatch.setattr(wiring, "FeatureExtractor", Stub)` — or read
`model.encoder`: `test_arm_objectives.py`, `test_figure_surfaces.py:651`, `:840`, `:917`,
`test_independent_adaptation_schedules.py:240`, `test_latent_sigma_readers.py:56`, and
**`test_attention_mechanisms.py:315`**, which the handoff's list omitted. Because
`wiring.py:41` binds `FeatureExtractor` as a module global, a target encoder constructed
through that same name inside `Arm.__init__` is stubbed by every one of those patches
automatically. This is therefore a **spec requirement, not a coincidence**: the target encoder
MUST be constructed through the module-level `FeatureExtractor` name, in the same
`__init__`. A local alias or a separate import would take six files red for a reason that has
nothing to do with Eq. (13).

### The degeneracy instrument — r25 names it, and it is not the one the plan named

r25's final paragraph (r25:1087) declares the failure mode and then prescribes its own
diagnostic:

> "El diagnóstico no exige maquinaria nueva: la **entropía cuadrática (Ec. 10)** sobre la
> **matriz de Gram de bolsas objetivo normalizada por traza** decae hacia su mínimo justo
> cuando la representación objetivo degenera; **distíngase del colapso de atención
> intra-bolsa, fenómeno distinto que no detecta este caso**."

The handoff proposed `MIL_CREDA.attention.uniform_self_similarity` (:166) and
`effective_bag_size` (:157) as the machinery to wire. **Those are the intra-bag quantities r25
explicitly rules out.** `uniform_self_similarity`'s own docstring — "equals 1 only if every
embedding of the bag coincides" — is a statement about instances *within* one bag; the
degeneracy r25 describes is the *bag cloud* collapsing. Their own module comment
(attention.py:149-154) agrees: they diagnose "that end-to-end training could degrade the
attention toward uniform weights," a different claim. Wiring them would produce an instrument
that reads green while the mode r25 warns about is happening. **This is a correction to the
approved plan's step 5, made on the paper's own text.**

What r25 prescribes is already on disk and already computed:

- `MIL_CREDA.renyi.quadratic_entropy` (renyi.py:77) *is* Eq. (10), `H₂(A) = −ln‖A‖²_F`.
- `MIL_CREDA.renyi.trace_normalize` (renyi.py:50) *is* Eq. (9)'s unit-trace normalization.
- `K_tt = bag_kernel_matrix(bags_t, bags_t, sigma)` is already built **every step** at
  `wiring.py:269`.

So the instrument is `quadratic_entropy(trace_normalize(K_tt))` over the **unconditional**
target bag Gram — zero new mathematics, one reading of a matrix the step already has. Note it
is *not* the `H_target` already inside `dependency_score` (global_term.py:63): that one is
per-class, conditional and confidence-weighted, so it cannot see a collapse that is uniform
across classes — which is precisely the case r25 describes, where the hard pseudo-label
(Eq. 22) becomes the same for every target bag and `C_act` in Eq. 37 shrinks to one class.

**And the confidence weights themselves are computed every step and reported nowhere**:
`w_t = confidences(G_t)` at `wiring.py:273`. r25's mechanism is that these vanish. Recording
their magnitude is the other half of the instrument, and today nothing reads it.

Two properties worth pinning in the spec rather than assuming:

- The reading decays toward its minimum as the target representation degenerates (a rank-one
  trace-normalized Gram gives `‖A‖²_F = 1`, hence `H₂ = 0`). The bound is derivable; it should
  be pinned by a test, not by this paragraph.
- `trace_normalize` **raises** on a non-positive trace (renyi.py:58-59), and `is_active`
  (wiring.py:289) already skips inactive classes. A fully degenerate target could therefore
  make the diagnostic crash rather than report — which is the opposite of what a diagnostic is
  for. The spec must say what it reports at the boundary.

### The revision bump, and what it actually costs

Measured on disk: **71 files carry the string `research-concept-r21`, 252 occurrences.** Four
of those are not records — `AGREED.md` (×2), `.implementation/position.jsonl` (×11),
`Notebooks/verification.ipynb` (×10), `Notebooks/Benchmark_Results.ipynb` (×1) — leaving
**67 records and 228 occurrences**, and **no file carries any other revision**. That splits 17
under `MIL-CREDA/Results/` and 50 model manifests under `MIL-CREDA/Models/`. `MIL-CREDA/Models`
is 2.1 GB and checkpoints roughly double it under two encoders.

Three guards fire on the bump, all by design:
`harness.py:393-403` (`StaleCheckpointRevision` on a record's own `revision` field),
`latent.py:393-397` (`available()` refusing a mismatched revision), and `latent.py:113`
(tagging each checkpoint `currentRevision: False`).

**One measured fact reduces this cost materially: every one of those 67 records is
pilot-scale.** Every `Models/` path is under `Models/Pilot/`; every `Results/` path is under
`Results/Pilot/` except `Results/Benchmark/ceilings.pilot.json`, which is itself a pilot
record. **There is no full-scale ceilings record on disk to lose.** `AGREED.md:208` and `:217`
already record that the trials search has not run at full scale and that "the current
declaration has no record yet," and `AGREED.md:187` already forbids quoting a pilot number as
a result. So the bump does not destroy a quotable result — it invalidates material that was
already declared non-quotable, and re-owes a walk that was already owed.

What the bump **does** cost is the walk itself: the six-step position goes un-ticked, and
because `report_digest.source_digest` covers every `*.py` under `src/`, the first byte of
`src/` this change touches invalidates every notebook stamp independently of the revision.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/MIL_CREDA_Benchmark/wiring.py` | Modified | Second `FeatureExtractor` for E/F/G at `__init__` (beside :95); `_target_embeddings` (:221-240) routes through it; module docstring :4 ("Every arm shares the same encoder…"); the degeneracy reading beside `K_tt` (:269) and `w_t` (:273) |
| `src/MIL_CREDA_Benchmark/config.py` | Modified | `REVISION` (:300) → r25, **its own commit**; the `normalization` rationale at :886-892 and :902-905 |
| `src/MIL_CREDA_Benchmark/latent.py` | Modified | `represent` (:448), `bag_pairs`/`:601-602`, `attention_spread` (:649), `:920-921` and `analyse` (:657-659): every target-side embedding names its encoder instead of inheriting `instance_embeddings` |
| `src/MIL_CREDA_Benchmark/harness.py` | Prose/record only | `parameters` (:737) recounts itself; the optimizer (:641, :818) picks the new module up unchanged. The revision guard (:393-403) is expected to fire — not weakened |
| `src/MIL_CREDA_Benchmark/__init__.py` | Modified | The `poolable` justification at :477-480 ("bit-identical across every run") becomes false across arms; the `parameters` declaration (:383, :484) itself stands |
| `src/MIL_CREDA/renyi.py`, `global_term.py`, `attention.py` | Unchanged | Eq. (9)/(10) reused as they are. The two `attention.py` diagnostics stay unwired — r25 rules them out for this mode |
| `tests/test_arm_objectives.py` | Modified | :496, :557, :617 re-pointed. **:371 and :461 (the floor's guards) stay green unedited** |
| `tests/test_figure_surfaces.py` | Modified | :963 premise re-read; :651, :840, :917 mechanically affected |
| `tests/test_independent_adaptation_schedules.py`, `tests/test_latent_sigma_readers.py`, `tests/test_attention_mechanisms.py` | Modified | `wiring.FeatureExtractor` patch sites (:240, :56, :315) |
| `MIL-CREDA/AGREED.md` | Already written | `## Reversed` 2026-09-28 (`8ed79de`) and `:164` (`c55526a`). Carried, not re-decided |
| `MIL-CREDA/Notebooks/*` | **Affected, and stamps invalidated** | Per `rules.proposal`: any `src/` byte invalidates every notebook stamp via `report_digest.source_digest`. `verification.ipynb` and `Benchmark_Results.ipynb` additionally carry the literal r21 string and must be re-executed under r25 |
| `MIL-CREDA/Results/` (17), `MIL-CREDA/Models/` (50) | Invalidated | 67 r21-stamped records; all pilot-scale. Disposition is `revision-bound-records`' to specify |
| `src/CREDA/` | **Untouched** | `FeatureExtractor` instantiated twice, edited zero times |
| Parent forge `papersmith-ai` | **Untouched** | Every edit inside `implementations/Domain_Adaptation` |

## Open Design Questions

Named rather than answered. `sdd-design` settles (2); **(1) may need the operator**, and the
orchestrator should treat it as an unresolved product decision rather than a design detail.

1. **Which encoder scores the target at evaluation, for E/F/G.** The settled decisions do not
   answer this, and the evidence points both ways.
   - `harness.accuracy(model, dataset, device)` (:423-435) calls `model(x)` with **no domain
     argument** and cannot tell which domain `dataset` holds. `Arm.forward` (:195-209) reaches
     `instance_embeddings` → the source encoder. So *doing nothing* silently evaluates target
     material through `θ_s`.
   - If it stays `θ_s`, the target encoder is a training-time-only regularizer whose learned
     representation is never read at test time: the 1.99× parameter count buys nothing at
     inference, and the degeneracy mode r25 warns about would be **invisible in
     `targetAccuracy`**, the one dimension declared `HIGHER`.
   - If it becomes `θ_t`, `forward` needs a domain and every eval caller changes; the head is
     trained on source embeddings only, and making that work on `θ_t` outputs is exactly what
     the alignment terms exist to do — so this reading is the one Eq. (13) implies, at a wider
     blast radius.
   - What **is** settled: `B` predicts the target through its one source-trained encoder
     ("B se queda con un codificador y predice el objetivo por el mismo, entrenado con
     fuente", `position.jsonl` discuss 2026-09-28T20:36:13Z). That decides the floor and only
     the floor.
   - This question also decides whether `test_figure_surfaces.py:963` fails or merely loses its
     premise, and what `latent.py`'s post-hoc readings should use.
2. **Where the degeneracy reading is recorded, and at what cadence.** It is computable every
   step at `wiring.py:269`, but `config.DIMENSIONS` HARD-requires every declared key on every
   run dict across the whole grid and every shard (config.py:965-968), and widening that set is
   a pooling and shard-agreement decision the `distribution` declaration owns — the precedent
   is already written at config.py:962-974, where `contributionGlobal`/`contributionLocal` were
   deliberately **not** declared for exactly that reason. The two live shapes: a per-run field
   outside `DIMENSIONS` (as those two already are, on `run_one`'s returned dict), or a
   per-epoch trace beside the loss curves.

## Cost

**Reach-class, and already ruled.** Two encoders change what the experiment **is**, so a new
protocol is obliged rather than an adjustment, and every result, checkpoint and dimension
already measured stops meaning anything. That ruling is `AGREED.md`'s Reversed entry of
2026-09-28, cited rather than re-derived.

What that costs concretely, measured:

- 67 r21-stamped records invalidated (17 `Results/`, 50 `Models/` manifests; 228 occurrences),
  **all pilot-scale**, so no quotable result is lost — see Approach.
- 2.1 GB of pilot checkpoints retired; checkpoints roughly double under two encoders.
- Every notebook stamp invalidated by the first `src/` byte, which forces the six-step pilot
  walk and the nine-account rehearsal to be repeated. Metered quota; the operator's gate.
- Paid **once** for all nine in-scope items. Landing them separately would pay the stamp cost
  repeatedly, which is why the revision bump is a separate *commit* and not a separate *change*.

**Delivery.** The authored-line forecast crosses the 400-line review budget once the four
re-pointed tests, the `latent.py` seam and the diagnostic's own tests are counted. `sdd-tasks`
owes the formal forecast and the guard lines; the natural slice boundaries are already visible:
(a) the revision bump alone, (b) the second encoder plus the four re-pointed tests, (c) the
`latent.py` seam, (d) the diagnostic. Strategy for this session is `ask-on-risk`, so the chain
shape is a decision to raise at tasks time, not here.

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| **The two embedding clouds drift apart over 20 epochs and one bandwidth stops describing both.** Under a shared extractor the two clouds were tied by construction; nothing ties them now | **High — and unmeasurable before a run** | The measurement above only establishes comparability **at step 0**, and says so. `kernelSigma` is searched log-scale over one decade around the anchor (harness.py:1687) and its winners already spanned 30.93–262.92, so the search absorbs drift within a decade. Spec must require the diagnostic (below) to be readable *during* training, so a run that drifts says so instead of returning a number |
| The degeneracy mode r25 declares happens and nothing notices | **High if unaddressed** — r25 leaves it "al terreno empírico" and nothing measures it today | `target-degeneracy-diagnostic`: `quadratic_entropy(trace_normalize(K_tt))` plus the magnitude of `w_t` (wiring.py:273), which is computed every step and reported nowhere |
| **The wrong instrument gets wired.** The approved plan named `uniform_self_similarity`/`effective_bag_size` | **Resolved here** | r25:1087 explicitly excludes intra-bag attention collapse as "fenómeno distinto que no detecta este caso." Spec MUST use the unconditional target bag Gram, and MUST NOT accept the per-class weighted `H_target` already inside `dependency_score` (global_term.py:63) as a substitute |
| Target material silently embedded through the source encoder by a post-hoc reader | **High if unaddressed** — six domain-blind call sites in `latent.py` | Every target-side embedding names its encoder. Spec should prefer a signature that **refuses** rather than defaults, since a default here returns a plausible number instead of an error — the same failure shape the sibling change documented at `ceiling_for` |
| `test_arm_objectives.py:617` inverts, and is "fixed" by weakening it instead of re-pointing it | Medium | It must be re-pointed at the **target** encoder and keep the twin comparison. Re-pointed correctly it is *stronger* than before; deleting the twin comparison would lose the only assertion that distinguishes the two encoders in a full `training_step` |
| The floor changes meaning while nobody is looking | Low — **by construction** | `B` has no adaptation term, so it has nothing to train a target encoder with. `test_arm_objectives.py:371` and `:461` must stay green **unedited**; an edit to either is the signal that the floor moved |
| Six test files go red for a reason unrelated to Eq. (13) | Medium | The target encoder is constructed through the module-level `wiring.FeatureExtractor` name (bound at :41), which every one of those six patches already replaces. Pin it as a requirement, not a hope |
| A degenerate target makes the diagnostic crash instead of report | Medium | `trace_normalize` raises on non-positive trace (renyi.py:58-59). Spec must state the boundary behaviour |
| `parameters` is pooled under a justification that is now false | Low impact, certain | `__init__.py:477-480` claims "bit-identical across every run." Still true within an arm (which is what `poolable` pools across — shards/seeds), false across arms. The declaration stands; the comment is corrected, and `AGREED.md:164` carries the report-side sentence |
| The r25 motivation (`X^s ≠ X^t`) does not hold in this bench, and a later reader re-opens the decision | Medium | Recorded in **Intent** with the operator's own answer and its timestamp, rather than omitted |
| The revision bump is read as destroying a result | Low, once stated | All 67 invalidated records are pilot-scale; `AGREED.md:187` already forbids quoting a pilot number, and `:208`/`:217` already record that no full-scale search record exists |
| Metered remote quota spent on the re-walk | High, accepted, **gated** | Inherent to a reach-class change. The existing launch-authorization mechanism owns it; this change does not launch |

## Rollback Plan

1. **Before any `src/` edit**, record the current `report_digest.source_digest` value. Reverting
   the `src/` diff restores that digest and every notebook stamp with it, so a revert does
   **not** re-owe the pilot walk.
2. **The revision bump is its own commit and its own revert.** Reverting `config.REVISION` to
   `"research-concept-r21.md"` makes all 67 existing records current again with no data
   migration: `harness.py:393-403` and `latent.py:393-397` compare a string, they do not rewrite
   anything.
3. **No on-disk record is mutated by this change or by undoing it.** The guards *refuse*
   stale records; they never rewrite them. New r25-stamped artifacts are additive files,
   deletable without touching an r21 one.
4. Revert order, each step independently safe: the diagnostic → the `latent.py` seam → the
   four re-pointed tests **together with** the second encoder (they are one atom: re-pointed
   tests without the encoder are red, and the encoder without them is red) → the prose → the
   revision bump.
5. **The safe partial state is "revision bumped, one encoder."** That is r21's implementation
   declared under r25, which every guard reports honestly as a mismatch between what the code
   does and what the paper says — visible, not silent. The unsafe partial state is the inverse:
   two encoders under an r21 stamp, where records would be written claiming a revision whose
   mathematics the code no longer implements. Spec and tasks must order the bump **first**.

## Dependencies

- **`MIL-CREDA/AGREED.md` must carry both entries before apply.** Verified present: the
  `## Reversed` paragraph dated 2026-09-28 (`8ed79de`) and `:164` (`c55526a`). Both were placed
  by `settle` after `discuss` recorded the questions and answers in
  `MIL-CREDA/.implementation/position.jsonl` (lines 477-480, verified read).
- **r25 itself**: `/Users/diego/Proyectos/papersmith-ai/proposals/research-concept-r25.md`,
  Eq. (13) at :375-385 and the failure-mode paragraph at :1087. It lives in the **parent** forge
  and is read, never edited.
- Test command `PYTHONPATH=src .venv/bin/python -m pytest -q`; a bare `python3.12` cannot import
  the package. `strict_tdd: true` (inherited; provenance at `openspec/config.yaml:17-24`) — and
  **the first red already exists**: `test_arm_objectives.py:557`'s own declared red mutation is
  this change.
- **The three current suite failures are not this change's business.** `main` measures
  `3 failed, 660 passed, 1 skipped, 129 subtests passed`. All three are deliberate committed red
  tests in `tests/test_search_distribution.py` belonging to the sibling change. They must not be
  counted as regressions here and must not be "fixed" here.
- **The sibling change `a-transfer-that-never-came-back-is-not-one-nobody-searched` survives this
  change untouched, and is not a dependency.** It is **machinery for splitting the search across
  machines, not science**: it has produced no full-scale result, so nothing it holds is
  invalidated by the revision bump. Its live work is on branch `search-by-transfer` with only its
  Phase 6 (a metered-quota pilot walk) outstanding; its `tasks.md` reads 0 of 84 ticked **on
  `main`**, because the completed 80 live on that branch (branch state per the orchestrator's
  handoff; not independently verifiable from `main`). This change touches none of its files:
  `shards.py`, `pooling.py`, `tools/distribute.py`, `tools/bridge.py` and
  `__init__.py`'s `distribution` groups are outside this scope except for the one `poolable`
  **comment** at `__init__.py:477-480`, which is a documented merge point to flag rather than a
  conflict to resolve here.
- **No forge change is required or permitted.** Every edit stays inside
  `implementations/Domain_Adaptation`.
- **One open product decision blocks part of the work, not the proposal**: Open Design Question
  (1), the target-evaluation path. Items 2, 3, 4, 6 and 7 of the scope proceed without it; item
  5 (`latent.py`) and `test_figure_surfaces.py:963` depend on it.

## Success Criteria

- [ ] `config.REVISION == "research-concept-r25.md"`, landed in a commit containing that change
      and its prose only
- [ ] `E`, `F` and `G` each carry two `FeatureExtractor` instances sharing **no** parameter
      object — proven by identity (`is`) over `named_parameters()`, not by equal shapes
- [ ] Both encoders' codomain is `R^d` with the same `output_dim`, and a test says so, because
      that is the one thing Eq. (13) declares them to share
- [ ] `B` carries exactly one encoder, and `tests/test_arm_objectives.py:371` and `:461` pass
      **with no edit to either**
- [ ] `parameters` reads 11,247,434 for `B` and 22,423,946 for E/F/G — measured from
      `harness.py:737`'s own expression, not asserted from this document
- [ ] The adaptation term alone puts a non-zero gradient on **both** encoders of E/F/G
      (`:496` re-pointed and split, not merely kept passing on the source side)
- [ ] `:557` passes by asserting the target forward ran through the **target** encoder, and a
      mutation that routes it back through `self.encoder` turns it red
- [ ] `:617` passes re-pointed at the target encoder, **and** additionally asserts what it now
      makes visible: the adapted arm's **source** encoder statistics MATCH its `B` twin's
- [ ] One optimizer, still built as `torch.optim.Adam(model.parameters(), lr=config.LR)` at
      `harness.py:641` and `:818`, and a test proves the target encoder's parameters are in it
- [ ] `config.KERNEL_SIGMA` is unchanged at `36.135014304860874`, and
      `harness.hyperparameter_drift` does **not** fire on this change
- [ ] The degeneracy reading is `quadratic_entropy(trace_normalize(K_tt))` over the
      **unconditional** target bag Gram; a test drives a deliberately collapsed target batch and
      shows the reading fall toward its minimum, and shows the per-class `H_target` inside
      `dependency_score` failing to detect the same collapse
- [ ] The magnitude of `w_t` (wiring.py:273) reaches a record or a trace instead of being
      computed and discarded
- [ ] The diagnostic **reports** rather than raises on a fully degenerate target batch
- [ ] No `latent.py` reader embeds target material through the source encoder by default; a
      caller that does not name a domain refuses rather than picks
- [ ] Open Design Question (1) is answered in `design.md` with the operator's decision recorded,
      and `test_figure_surfaces.py:963` is resolved against that answer — passing, or re-pointed
      with its premise rewritten
- [ ] `wiring.py:4`, `config.py:886-892`, `config.py:902-905` and `__init__.py:477-480` no longer
      assert a shared encoder or a parameter count identical across every run
- [ ] The disposition of the 67 r21-stamped records is executed as `revision-bound-records`
      specifies, and `latent.stale_revisions` reports exactly that set
- [ ] Full suite green under `PYTHONPATH=src .venv/bin/python -m pytest -q`, with the sibling
      change's three `tests/test_search_distribution.py` reds still red and still counted as
      its work, not as regressions here
- [ ] `src/CREDA/` is byte-identical, and nothing outside
      `implementations/Domain_Adaptation` is modified
