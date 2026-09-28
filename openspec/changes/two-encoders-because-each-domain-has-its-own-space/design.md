# Design: Two encoders, because each domain has its own space

## Technical Approach

Eq. (13) splits the **encoder** and nothing else. Everything downstream of the
codomain `R^d` stays single, and r25 says so in its own text one page later:
"Los mismos parámetros se emplean para fuente y objetivo" (r25:471, about
Eq. (15)'s `R_phi`), and "Un único ancho de banda `σ` gobierna los tres bloques,
ya que los tres derivan del mismo kernel de instancia" (r25:732). So `V_R`,
`b_R`, `v_R`, the head, `σ`, and the one optimizer are all untouched, and the
change is confined to three seams:

1. **A construction seam** — a second `FeatureExtractor` in `Arm.__init__`, for
   the arms that declare an adaptation term.
2. **A dispatch seam** — every place that turns raw instances into `R^d` names
   the domain it holds. There are nine such places in `src/` and one in
   `tools/`, and in every one of them the caller already knows.
3. **A reading seam** — four scalars and one count, read off matrices the
   training step already builds, so a run that degenerates says so while it
   runs instead of returning a number.

The revision bump is a fourth, separate, first commit. It is measurably larger
than the proposal's "one line" — see Decision 11.

This design settles Open Design Question (1)'s **shape** (Decision 1) and Open
Design Question (2) in full (Decision 8). Open Design Question (1)'s
**direction** was settled by the operator before this phase and is carried, not
re-decided: every domain is embedded by its own encoder at every site,
evaluation included.

Specs this design implements: `specs/per-domain-encoders/spec.md`,
`specs/target-degeneracy-diagnostic/spec.md`,
`specs/revision-bound-records/spec.md`.

---

## Architecture Decisions

### Decision 1: Two named entry points per domain, no domain argument, no default

**Choice.** `Arm` exposes exactly two embedding entry points, distinguished by
name and not by an argument:

| name | encoder | every arm? |
|---|---|---|
| `instance_embeddings(bags)` | `self.encoder` — `θ_s` | yes |
| `target_embeddings(bags)` | `self.target_encoder` — `θ_t`, falling back to `self.encoder` on the floor | yes |

and exactly two scoring entry points, built on those two:

| name | path | used by |
|---|---|---|
| `forward(bags)` | `_scores(instance_embeddings(bags))` | `harness.accuracy` → `sourceAccuracy` |
| `score_target(bags)` | `_scores(target_embeddings(bags))` | `harness.target_accuracy` → `targetAccuracy` |

There is no `domain=` parameter anywhere on this boundary, and no default.

**Alternatives considered and rejected.**

*Rejected — an explicit `domain` argument on one shared embedding call.* This is
the smaller diff and it is the shape **prior work in this very repository
already has**, which is why it can be judged on evidence rather than on taste.
`src/CREDA/models.py:189` is a two-encoder domain-adaptation model whose forward
reads:

```python
def forward(self, x, domain='source', mode='class'):
    if domain == 'source':
        f = self.Fs(x)
    elif domain == 'target':
        f = self.Ft(x)
```

Two things about that precedent decide this question.

- It carries `domain='source'` as a **default**, so a caller that does not think
  about the domain at all gets the source encoder and a plausible number. That
  is the exact failure shape this change exists to remove.
- Its own authors did not rely on it where it mattered. At the site where the
  domain decides the answer — evaluation — `training_pipeline.py:345-346` writes
  `eval_model(model.Fs, ...)` for the source and `eval_model(model.Ft, ...)` for
  the target, bypassing the defaulted argument and naming the encoder directly.

A shape whose own authors route around it at the decisive call site is not a
shape to adopt. (`src/CREDA/` is prior work, read here as evidence and never
edited; it stays byte-identical.)

*Rejected — a **required** positional `domain` on one shared call.* Stronger
than a default, and still the wrong shape at the sites that actually failed.
Measured: four of the domain-blind sites are **adjacent pairs** one line apart,
differing only in a variable name — `latent.py:601`/`:602`, `latent.py:920`/
`:921`, `analyse`'s `represent` calls at `latent.py:658`/`:659`, and
`harness.py:690`/`:691`. A string argument on two adjacent lines is precisely
what a copy-paste gets wrong, and both calls return a tensor of the right shape,
so nothing structural objects. Two different *method names* on those two lines
make the mistake visible to a reader and to `git diff`.

*Rejected — removing `instance_embeddings` in favour of a symmetric
`source_embeddings`/`target_embeddings` pair.* This is the shape I would choose
on a clean tree, and it is forbidden here by a constraint that outranks
symmetry. `tests/test_arm_objectives.py:429`, inside the floor guard at `:371`,
calls `floor.instance_embeddings(x)` to recompute the expected supervised loss.
S1 and the proposal's own success criterion require that test to pass **with no
edit**. Removing or renaming the method edits it. So the name stays, its
docstring is rewritten to say it is the **source** encoder's embedding and why it
is not called `source_embeddings`, and the asymmetry is recorded as ugly with its
reason rather than presented as a design.

**Rationale for the shape that was chosen.** The criterion is *make today's
error unreintroducible, not merely corrected*, and the repository has already
written down the rule for exactly this kind of quantity — a fact the caller knows
and the callee cannot infer. `config.py:482-484`, at the declaration of
`KERNEL_SIGMA`: "Every caller passes this explicitly, with **no default anywhere
in the call chain**". `Arm.weights_for`'s own docstring (`wiring.py:141-145`)
restates it: "passed explicitly and with no default, so a caller supplying a
different value is a visible, deliberate choice rather than something this method
could quietly default to."

The domain is the same kind of quantity and a stronger case than `σ`: a wrong
`σ` changes a number, a wrong domain changes **which model produced it**.

**Is a default admissible at all? No, and this is the load-bearing answer.** A
default on this boundary converts a program error into a plausible number. Every
refusal this repository has built on the same reasoning refuses instead:
`harness.py:393-403` (`StaleCheckpointRevision`), `latent.py:393-400`, and
`ceiling_for`'s pooled-fallback lesson the sibling change documented. `latent.py`
is where it matters most, because the phase-two RKHS reading (`AGREED.md:222`) is
what would be silently wrong.

**What B does under this shape.** `target_embeddings` exists on every arm,
including the floor, and on the floor it resolves to `self.encoder` — B's one
encoder, source-trained. That is not a fallback invented here; it is the settled
floor behaviour ("B se queda con un codificador y predice el objetivo por el
mismo, entrenado con fuente", `position.jsonl` discuss 2026-09-28T20:36:13Z) and
`AGREED.md`'s Reversed entry of 2026-09-28 says the floor's guarantee "must be
re-read under two encoders: it never sees the target, which under a per-domain
encoder becomes trivially true of a target encoder that never trains." B holds
**one** encoder object; it has **two** method names that both reach it. The spec's
"the floor MUST hold exactly one encoder" is about objects, and it holds.

**Can `forward` carry a domain? No, and it does not need to.** `nn.Module.__call__`
would pass one through, so this is not a language limit. It is rejected on two
grounds. A defaulted domain on `forward` reintroduces the default this decision
rejects, at the one method the whole framework calls implicitly. A *required*
domain makes `model(x)` a `TypeError`, which breaks `Arm` as a plain `nn.Module`,
breaks `MechanismArm` with it, and edits `tests/test_arm_objectives.py:443` —
inside the protected floor guard. So evaluation gets its **own entry point**:
`score_target`, and `harness.target_accuracy` beside `harness.accuracy`.
`harness.accuracy` keeps its three-argument signature untouched, which is what
keeps `:357` and `:443` green unedited.

`forward` and `score_target` share one private `_scores(embeddings)` holding the
bag/instance unit branch currently inlined at `wiring.py:205-209`. Duplicating
that branch into `harness` would fork Eq. (19) and the instance-averaging rule
across two modules, against `wiring.py`'s own stated boundary: "This file owns
the learnable parameters and the bookkeeping, and not one formula."

### Decision 2: `_target_embeddings` is promoted to the public `target_embeddings`

**Choice.** The private training-time seam at `wiring.py:221-240` becomes the
public `target_embeddings`. The private name is removed, not kept beside it.

**Alternatives considered.** Keep `_target_embeddings` private and add a public
`target_embeddings` that delegates to it — two names for one thing, and the
repository's own record is full of what that costs. Or have `latent.py` reach
into `model._target_embeddings` — a cross-module private, which is the
"domain-blind by convention" shape this change removes.

**Rationale.** Its docstring already says it exists "as a named place rather than
as a branch." That purpose survives and strengthens: it is now the place the
**encoder choice** lives, which is a better reason for a named place than the
freeze that was withdrawn. And this rename is one of the few pieces of prose
correction in this change that is actually **enforceable**: `verify`'s
`prose.unresolvedSymbols` resolves prose tokens against names defined under
`src/` (`implementation_engine.py:1259-1287`), so any docstring left saying
`_target_embeddings` is reported. Five docstrings mention it
(`tests/test_arm_objectives.py:575`, `:619`, `:628`, `:636`, `:638`) and move in
the same edit.

Both of its callers keep working unchanged in shape: `_milcreda_term:255` and
`_creda_term:315`. `_creda_term` needs **no edit at all** — it reads
`self.encoder.output_dim`, and under Eq. (13) `θ_s` and `θ_t` share exactly that
(the codomain is the one thing declared shared), so the `"creda"` branch stays
correct and reachable-by-declaration for free. That is a consequence of routing
at the seam rather than at each caller, and it is the argument for routing there.

### Decision 3: The target encoder is constructed LAST in `Arm.__init__`

**Choice.** `self.target_encoder` is constructed through the module-level
`FeatureExtractor` name, inside `Arm.__init__`, **after** the head and after the
`V_R`/`b_R`/`v_R` attention parameters — i.e. last.

**Alternatives considered and rejected.** Beside `self.encoder` at
`wiring.py:95-96`, which reads better. `copy.deepcopy(self.encoder)`, which
guarantees identical initial weights.

**Rationale.** Constructing a `FeatureExtractor` **consumes the global
generator**: `torchvision`'s resnet18 runs `nn.init.kaiming_normal_` over every
conv and norm layer before `load_state_dict` overwrites them with the pretrained
checkpoint, and the test stubs draw an `nn.Linear`. So a second construction
placed at `:96` shifts the generator state seen by the head and by the attention
parameters, and B's head would no longer be initialized identically to E's.

That would put a **second difference** on the rung `B→E` — the encoder count and
the head's draw — against the ladder's own attributability rule at
`config.py:944-946`: "A comparison is only attributable when its two arms differ
in one thing." S4 says the rung carries one difference; construction order is
what makes that literally true. Constructing last means every parameter B and E
share is drawn from identical generator state, and only the module B does not
have consumes state afterwards.

It also keeps a claim in `tools/measure_kernel_sigma.py:25-29` literally true —
"`Arm.__init__` constructs `self.encoder` first, before the head or any
arm-specific attention parameters" — which would otherwise become false prose in
the script that reproduces `config.KERNEL_SIGMA`.

`deepcopy` is rejected because it bypasses the module-level name, which is the
spec requirement (see Decision 4). Parameter *order* changes are harmless:
`Adam` keeps per-parameter state, `state_dict` is key-addressed, and
`sum(p.numel() ...)` is order-independent.

### Decision 4: The construction site is a spec requirement — how a second construction reaches six test files

**Choice.** `self.target_encoder = FeatureExtractor(backbone=config.BACKBONE,
pretrained=config.PRETRAINED)` — the same module-level name, the same call shape,
the same `__init__`. Not a local alias, not a second import, not a dotted
`models.FeatureExtractor`, not a `deepcopy`.

**How it reaches each of the six.** `wiring.py:41` binds `FeatureExtractor` as a
module global. A name looked up as a module global is resolved **at call time**,
so `monkeypatch.setattr(wiring, "FeatureExtractor", Stub)` replaces it for every
construction made through that name inside `Arm.__init__`, whether one or two:

| file | site | mechanism |
|---|---|---|
| `tests/test_arm_objectives.py:43` | `encoder` fixture → `_Encoder` | both encoders become `_Encoder` |
| `tests/test_arm_objectives.py:68` | `bn_encoder` fixture → `_BNEncoder` | both become `_BNEncoder`, each with its own `bn` |
| `tests/test_figure_surfaces.py:651` | `_TinyEncoder` | both |
| `tests/test_figure_surfaces.py:840` | `trained` fixture → `_Encoder` | both |
| `tests/test_figure_surfaces.py:917` | reads `model.encoder.output_dim` | still `θ_s`'s; unchanged, and gains a codomain assertion (Decision 6) |
| `tests/test_independent_adaptation_schedules.py:240` | patch site | both |
| `tests/test_latent_sigma_readers.py:56` | patch site | both |
| `tests/test_attention_mechanisms.py:315` | patch site | both |

**What that obliges.**

- Every stub must be constructible **twice** with the same keyword arguments. All
  five stubs already take `(backbone=None, pretrained=False)` and hold no
  class-level state, so all five satisfy this without edit — verified by reading
  `_Encoder`, `_BNEncoder`, `_TinyEncoder` and the two patch sites.
- The two stub instances are **not** identical: each draws its own `nn.Linear`.
  Production's two are identical at step 0 (both load the same ImageNet
  checkpoint). Any test whose assertion depends on the two encoders agreeing at
  initialization is asserting a stub property, not a model property, and must be
  written to survive both — this is the caveat behind Risk R3.
- `_BNEncoder`'s spy is patched on the **class** (`monkeypatch.setattr(_BNEncoder,
  "forward", spy)`), which is what lets `tests/test_arm_objectives.py:557` tell
  two encoder objects apart. That test's instrument was built for this change;
  see Decision 6.

### Decision 5: One optimizer, and what nothing can express

**Choice.** `torch.optim.Adam(model.parameters(), lr=config.LR)` at
`harness.py:641` and `:818` is unchanged. A second registered `nn.Module` is
picked up automatically, and `parameters` (`harness.py:737`) recounts itself.

**Rationale and the limit, stated rather than discovered later.** Nothing in
`config` today can express a different rate for one submodule, and it is not
merely absent — it is structurally unavailable: `harness.py:668-669` rewrites
**every** param group to one value every epoch (`for group in
optimizer.param_groups: group["lr"] = learning_rate(epoch, ...)`), so even a
hand-built second param group would be overwritten at the top of the next epoch.
`config.LR` is one fixed declared value, explicitly not searched
(`config.py:863-870`). Two optimizers are out of scope by the proposal.

Prior work's own ADDA uses two optimizers and freezes `Fs`
(`training_pipeline.py:276-279`). That is a different training regime, and this
design deliberately does not adopt it: both encoders train, under one rate, which
is what S3 settles and what makes S4's "adaptation trains the target side" true.

### Decision 6: How each shared-encoder test is re-pointed

Five tests, not four. The fifth was found by reading, and it is the one a
re-pointing pass would most easily miss because it is a *rebuild* rather than an
assertion about the encoder.

**6a. `tests/test_arm_objectives.py:496` — the claim SPLITS.**

`assert grads` / `assert any(torch.any(g != 0) ...)` over `arm.encoder.parameters()`
becomes the same two assertions over `arm.encoder.parameters()` **and**
separately over `arm.target_encoder.parameters()`. Renamed
`..._carries_gradient_to_both_encoders`.

Why the split is not cosmetic: the existing assertion keeps passing on the source
encoder alone while saying nothing about the target one. `H_s` enters
`_milcreda_term` with grad from `training_step`, so `K_ss`, `K_st` and the local
term's `π @ K_ss @ π` all carry gradient to `θ_s` regardless of what happens to
`θ_t`. So the old test cannot fail on the mutation this change is most exposed
to. **Two** reachable reds must be declared, one per side:
- `torch.no_grad()` around the target forward → the **target** assertion reds,
  the source one stays green. This is the red the old test could not see.
- `.detach()` on `H_s` before `_milcreda_term` → the **source** assertion reds.

The docstring's premise — "Eq. (39)'s whole point is one encoder both domains'
inputs update" — is now false and is replaced by S4 with its three citations:
`global_term.py:59-60` ("Both marginals carry the same sign and the same
coefficient, so neither is privileged"), `local_term.py:85-101` (the target bag
measured against the **source** prototype), and r25:1087's own sentence, which is
the same fact in the paper's words: "La discrepancia local (Ec. 31) no cede por
concentrarse: `μ_j^s` se construye con representaciones fuente."

**6b. `tests/test_arm_objectives.py:557` — the assertion that most directly
contradicts Eq. (13), and the first red.**

`assert all(module is arm.encoder for module in seen)` (line 604) becomes a
**partition**: `seen` must contain both encoders; the source forward must be
`arm.encoder`; the target forward must be `arm.target_encoder`; and
`arm.encoder is not arm.target_encoder`. The running-statistics half moves with
it: the warm-up at lines 583-585 and the `before_mean`/`before_var` snapshot move
to `arm.target_encoder.bn`, because it is the **target** encoder whose statistics
the target forward now moves.

Its class-level spy is exactly the right instrument and its own docstring says
why: "an instance-level patch would only ever see calls made through the one
object it is attached to, so it could not tell a shared extractor from a second
one quietly built and called instead." The test was written against this change.
Only the assertion's polarity and the object under observation move.

**The red-first sequence, written out because the tempting shortcut is
green-first with extra steps.** The proposal is right that this test's own
declared red mutation *is* this change, and that satisfies `strict_tdd` without
authoring a new failing test — but only in this order:

1. Run the suite on the current tree. Observe `:557` **green** under one encoder.
   Record the count.
2. Re-point `:557`'s assertions to demand two encoders. Run. Observe **RED**,
   with production still holding one encoder.
3. Build the second encoder and route `target_embeddings` through it. Run.
   Observe **GREEN**.

Doing (3) before (2) — change production, watch `:557` go red, then "fix" the
test — is how an assertion gets weakened instead of re-pointed, which is Risk R5.

**6c. `tests/test_arm_objectives.py:617` — this one INVERTS, and becomes
strictly stronger.**

Today: `assert not torch.equal(floor.encoder.bn.running_mean,
adapted.encoder.bn.running_mean)`. Re-pointed, it becomes a **pair**:

- **(a) inverts.** `assert torch.equal(floor.encoder.bn.running_mean,
  adapted.encoder.bn.running_mean)` — and the same for `running_var`. The adapted
  arm's **source** encoder now matches its B twin's, because both are updated
  only by source forwards under an identical batch and generator. This asserts
  something nothing asserted before: that the target forward does **not** touch
  `θ_s`.
- **(b) is new.** `assert not torch.equal(adapted.encoder.bn.running_mean,
  adapted.target_encoder.bn.running_mean)` — the arm's two encoders moved apart,
  so the target forward happened and went somewhere else.

Together these distinguish the two encoders instead of merely proving one of them
moved. The twin comparison is kept, not deleted — deleting it is Risk R5 and would
lose the only assertion that separates the two encoders inside a full
`training_step`.

(a) depends on Decision 3: `_BNEncoder`'s `bn` follows a `Linear`, so its running
statistics depend on that `Linear`'s weights, so B's and E's source encoders must
be initialized identically. Constructing the target encoder **last** is what
gives that. If the target encoder were constructed at `:96`, (a) would fail and
the failure would look like a wiring bug rather than a construction-order one.

**6d. `tests/test_figure_surfaces.py:963` — it PASSES, and its premise is
re-worded.**

Ruled from the code, not guessed. `count` is `rows.shape[0]` recorded inside the
`_embed` stub (`tests/test_figure_surfaces.py:856-861`), and `_embed` is
`latent.py`'s UMAP stand-in called once per panel on
`torch.cat([reading["sourceRows"], reading["targetRows"]])`
(`latent.py:1127`). It measures the **projection** call, never an encoder call.
`bag_pairs` still returns 10 source rows and 10 target rows under `_domains()`, so
`count == config.CLASSES * 2` still holds and the assertion passes **unedited**.
The parallel spec reached the same reading independently
(`specs/per-domain-encoders/spec.md`, scenario at :273-278).

What it loses is its docstring's wording, "both domains go through ONE embedding
call", which conflates the projection with the encoder. Obligations:

- Re-word the premise to *one projection call over one shared codomain `R^d`* —
  which is exactly what Eq. (13) declares shared, so the figure's claim survives
  in a weaker and still-true form.
- **Add** the assertion that makes the new claim measurable rather than assumed:
  the two sides came through **different** encoders
  (`model.encoder is not model.target_encoder`) and still landed in one projected
  space. Without it the test passes for a reason that no longer has anything to do
  with what it claims.

*Not determinable by reading, and recorded as such:* the abscissa-overlap
assertion at `:1002` now rests on **two independently initialized stub
encoders** (Decision 4), where it used to rest on one. Whether two random
`nn.Linear` maps put their outputs in overlapping `x` ranges is an empirical
question about that fixture's seed. It must be **observed**, not assumed. If it
fails, the correct repair is to replace the overlap heuristic with the structural
claim the test actually needs — one `_embed` call carrying both domains, already
asserted at `:986` and `:992` — and **not** to reseed the fixture until it
passes, which would make the assertion a description of a seed.

**6e. `tests/test_figure_surfaces.py:896-897` — the fifth test, absent from the
handoff's list.**

`test_the_nearest_source_bag_is_found_with_the_bag_kernel_in_the_representation_space`
rebuilds `K_st` "from the published pieces" using `model.instance_embeddings` for
**both** domains (`:896-897`) and asserts `torch.equal(reading["nearest"],
K_st.argmax(dim=0))` against `bag_pairs`' output. Once `bag_pairs` embeds the
target through `θ_t`, the rebuild and the function under test disagree and this
**fails**.

Re-pointed: `H_t = model.target_embeddings(...)` at `:897`. And `:917`'s
`H_s.shape[-1] == model.encoder.output_dim > 2` stays valid, extended with
`model.target_encoder.output_dim == model.encoder.output_dim` — the codomain
claim, asserted at the one place the two widths meet.

This test belongs to the `latent.py` slice, not the encoder slice: at the encoder
slice `bag_pairs` is still domain-blind, so the rebuild and the function still
agree and the test stays green. It reds only when `bag_pairs` splits.

**6f. The two floor guards stay green, unedited — and one sentence becomes
ambiguous.**

`:371` passes because its no-target-image spy is on `floor.encoder.forward` (an
encoder-level hook, chosen deliberately over an `instance_embeddings` hook by its
own docstring), because `instance_embeddings` survives with its signature for
line 429, and because `harness.accuracy` survives with its signature for line 443.
`:461` passes because it spies `wiring.Pool.take` and B still never calls it on
the target pool. Verified by reading every line of both.

Recorded honestly: clause (3)'s sentence, "it predicts on target through the
ordinary evaluation path every other arm uses", becomes **ambiguous** rather than
false — for B the two paths are one encoder and give one number, and every arm
still reaches `forward` for its source accuracy. Correcting the sentence would
edit a test S1 forbids editing. The disambiguation goes into the new target-side
test that cross-references it. A named, deliberate prose drift beats a silent one
or an unauthorized edit.

### Decision 7: `latent.py` takes a required `domain`, and the pairs take two method names

**Choice.**

| function | shape |
|---|---|
| `represent(model, bagset, positions, device, domain, unit=None)` | `domain` **required**, no default; refuses on anything but the two |
| `attention_spread(model, bagset, positions, device, domain)` | same |
| `correspondence` (`:601-602`) | `instance_embeddings` / `target_embeddings` on the two adjacent lines |
| `bag_pairs` (`:920-921`) | same |
| `analyse` (`:658-659`, `:671`) | names the domain at each call |
| `latent_grid` (`:868`, `:870-871`), `correspondence_grid` | names the domain at each call |

The two admissible values live in one module-level constant, `wiring.SIDES =
("source", "target")`, with the refusal written in the idiom
`mechanism_weights` already uses: `raise ValueError(f"unknown domain {domain!r};
known: {SIDES}")`. **Not** named `DOMAINS` — `config.DOMAINS` already means the
three datasets M/U/S (`harness.py:873`), and a second `DOMAINS` would be a
collision a reader has to work out.

**Why a string argument here, when Decision 1 rejected one on the embedding
call.** Because `represent` and `attention_spread` are **not** the embedding
call. They are chunking loops that happen to contain one, and they also carry the
`unit` override, the label bookkeeping and the `BAGS_PER_STEP` walk. Splitting
each into two functions would duplicate all of that, against the repository's own
written rule for exactly this choice — `_conditional_reading`'s docstring
(`latent.py:473-478`): "Una función y no dos copias: la lectura del kernel y la
euclidiana tienen que diferir **EXACTAMENTE** en la distancia. Escritas aparte,
cualquier otra cosa podría separarse también."

The pair readers get two method names because they hold both domains in one
function body, on two adjacent lines, which is the copy-paste shape (Decision 1).

**What closes the residual copy-paste hazard.** A required string argument can
still be passed wrong at four adjacent pairs. Two guards, and they answer
different halves:

- **Behavioural, one per consumer.** Perturb `θ_t`'s parameters on a loaded
  adapted arm; every target-side reading must **move** and every source-side
  reading must **not**. Then perturb `θ_s` and assert the mirror. This catches a
  wrong domain string, because a target reading taken through `θ_s` does not move
  when `θ_t` moves. One assertion per consumer, never inferred from a sibling —
  this repository has already paid for the inferred version.
- **Structural, derived rather than listed.** No module under
  `src/MIL_CREDA_Benchmark/` **other than `wiring.py`** may call
  `instance_embeddings` at all, asserted by an AST walk. After this change the
  only `src/` call sites are `wiring.py:204` (`forward`, source by construction)
  and `wiring.py:354` (`training_step`, source by context), and both are already
  guarded behaviourally. The condition is a derived sentence, not a roster, so it
  reddens on a **new** site in any other module — which is what the behavioural
  guard structurally cannot see.

  Precedent for the mechanism, twice: `tests/test_figure_surfaces.py:673-688`
  AST-reads `latent.analyse` to assert it never passes `unit=` to `represent` —
  the isomorphic problem, a default silently replacing a caller's intent — and
  `test_arm_objectives.py`'s `_unit_branch`/`_milcreda_calls` read
  `training_step`'s source the same way.

**`tools/measure_kernel_sigma.py` is deliberately NOT edited.** It calls
`instance_embeddings` at `:108` on a **freshly constructed** model, where both
encoders are the same pretrained resnet18, so it reproduces
`config.KERNEL_SIGMA = 36.135014304860874` byte-for-byte with no change. Editing
the script that produced the declared constant would change what produced it.
It sits outside `src/`, so the AST guard's scope (`src/`) does not reach it, and
that is stated here rather than left as a gap: its rule is that it measures the
**anchor at step 0**, which is domain-symmetric by construction, and the
post-training reading it cannot give is Decision 10's job.

### Decision 8: Where the degeneracy reading lives — Open Design Question (2), settled

**Choice.** **Not** a seventh `config.DIMENSIONS` entry. It is a per-step value on
`training_step`'s returned dict, aggregated into the **per-epoch** record
(`harness.py:688-692`), with exactly one per-run summary field. It is also **not**
declared in `__benchmark__["report"]["dimensions"]`.

**Rejected — a seventh declared dimension.** Three independent reasons, each
measured, and the first alone is decisive.

1. **The floor structurally cannot produce it.** `summarize()` (`harness.py:932`)
   is `{d: spread([float(r[d]) for r in runs]) for d in config.DIMENSIONS}` — a
   hard `float(r[d])` over **every** run of every arm, and `shards.partition`
   requires the same. B has no adaptation term, so it never reaches
   `_milcreda_term` and never builds `K_tt`; building one would mean encoding
   target images, which **Decision 2 forbids** ("a floor never lets a target
   image reach the encoder, ever, in training", `wiring.py:372-377`, guarded by
   `tests/test_arm_objectives.py:371` and `:461`). So B would have to report a
   fabricated number. And 0.0 is not available as that number: 0.0 is **exactly
   the collapsed value** (see Decision 9), so the floor would read as maximally
   degenerate on every run forever.
2. **`ladder_rows` would build rung rows nobody can read.** `ladder_rows`
   (`harness.py:947-962`) emits one row per rung per dimension, so a declared
   dimension produces `B->E targetBagEntropy` — a rung comparing B's fabricated
   value against E's real one. `adaptationShare` already had to special-case this
   ("An arm with no adaptation term reports zero rather than a ratio, because a
   floor has no share to command and a nan would propagate into the table",
   `harness.py:742-744`).
3. **`poolable` is a measured claim and there is no measurement.**
   `__init__.py:417-422` and `tests/test_benchmark_declarations.py:112-187` record
   that the six pooled dimensions "came back bit-identical on all three runs" of a
   named A/B/C replication, and `test_the_distribution_declares_exactly_what_was
   _approved` asserts `dist == {...}` against an **exact literal**. Adding a
   seventh dimension means editing that literal and classifying the new dimension
   into `poolable` with no replication behind it — a declaration claiming a
   measurement nobody made. `test_the_declared_dimensions_now_cover_every_dimension
   _the_harness_measures` (`:229`) makes the classification mandatory, so this is
   not avoidable by leaving it unclassified.

The precedent is already written for exactly this reason:
`config.py:962-974` keeps `contributionGlobal`/`contributionLocal` **out** of
`DIMENSIONS` while writing them on every run's dict, because "widening that set
is a decision about pooling and shard-agreement policy ... and not to this
constant alone."

**Rejected — a field on the latent record only** (`latent.analyse`,
`latent.py:661-674`). It would be computable there (`correspondence` already
builds a `bag_kernel_matrix`), it is cheap, and it would cover the floor too,
which is a real advantage. It is rejected because it answers the wrong question.
r25's failure mode is a training **dynamic**: `θ_t` receives gradient only through
Eqs. 37 and 38, the ramp climbs over the fraction of training elapsed, and the
collapse develops. A reading taken only on the final checkpoint cannot tell
"collapsed at epoch 3 and stayed" from "healthy throughout", and phase two rests
on three checkpoints per cell (`AGREED.md:224`), not on the 1800-run grid. The
diagnostic spec's own requirement — "readable during training, not only after" —
rules it out as the sole home. It is not foreclosed as a **later addition**;
this design simply does not owe it.

**Rejected — the per-step `curve`.** `curve` is written to `runs.jsonl` in full
(`harness.py:2310` writes the whole run dict), so persistence is not the issue.
Step resolution is: `steps = ceil(TRAIN_BAGS / BAGS_PER_STEP) = 7`, so
7 × 20 = 140 points per run against 20 for the epoch record, and r25's claim needs
nothing finer than an epoch. And the per-epoch record is where it **belongs**:
`epochs_record` already holds `sourceAccuracy` and `targetAccuracy` per epoch, so
placing the reading there puts it beside the accuracy it explains — which is the
pairing that makes "the representation collapsed and the accuracy did not fall"
readable at all.

**Chosen shape, concretely.**

- `training_step` returns the per-step readings alongside the existing
  `supervised`/`adaptation`/`contribution*` floats.
- `run_one` aggregates them per epoch into `epochs_record` (means over the
  epoch's 7 steps, and `min` where the minimum is the informative statistic),
  exactly as it already means `curve` into `mean_supervised`.
- One per-run field: **`targetBagEntropyMin`**, the minimum over epochs. Chosen
  because it is the one reading the epoch trace makes awkward (a `min` over 20
  points) and the one that answers r25's question — *did the representation ever
  degenerate during this run* — which `final = epochs_record[-1]`
  (`harness.py:724`) structurally cannot. Four per-run fields were considered and
  rejected as record weight: everything else reads off `epochs`.
- The floor reports `float("nan")` for these, **not** 0.0, and the initialization
  carries that reason in a comment, or a later reader will "fix" the inconsistency
  with the neighbouring `contribution*` zeros. NaN is this repository's
  established absent-reading sentinel: `harness.accuracy:435` and
  `latent.geometry:488-496` both already write it into records.

**`direction`: DESCRIPTIVE, and for a stated reason rather than by default.** It
is not that the reading has no direction — it has a **two-sided** one, and that is
different.

- Low is a failure: `H₂ → 0` is the collapse r25 declares.
- High is **also** not success. Maximal `H₂` means the target bags are mutually
  dissimilar under the one shared `σ`, which is the "two clouds drifted apart"
  failure Risk R1 names. The method aligns the target conditional to the source;
  a target cloud maximally spread in `R^d` is not the goal.

So `HIGHER` and `LOWER` are both wrong, and `DESCRIPTIVE` is right **because both
ends are failures and the middle is health** — which must be written beside the
field, since "descriptive" alone reads as "no direction" and a reader would take
higher for better. The nearest existing vocabulary word is
`domainSeparability`'s `"toward-chance"` (`__init__.py:387`), invented for exactly
this two-sided shape; it does not fit because it names a **target value** and no
healthy value of `H₂(trace_normalize(K_tt))` is derivable from r25 or from the
code. Only the degenerate end (0) is derivable.

**Named as undetermined, not defaulted: the healthy band is unknown.** Only a run
establishes it. This is why the reading is declared descriptive rather than given
a threshold, and why it must be read **paired** with the confidence weights and
the kernel triple, never alone.

**Not declared in `__benchmark__["report"]["dimensions"]`,** because membership in
that dict **is** the render declaration. Evidence:
`tests/test_latent_geometry.py:120-133` reads absence as "kept in the record and
never drawn" for `geometry.euclidean*`, and asserts presence for
`geometry.ratio` as "la lectura sí se dibuja". A run-time guard whose job is to
make a run say it degenerated is not a verdict quantity the report ranks; putting
it in that dict would enroll it in the render surface and oblige a notebook cell.
Its disposition is `geometry.euclidean`'s: in the record, not rendered.

### Decision 9: The instrument is four readings and one count, all free

**Choice.** Read off matrices `_milcreda_term` already builds at
`wiring.py:267-273`, with no new mathematics:

| field | expression | what it answers |
|---|---|---|
| `targetBagEntropy` | `quadratic_entropy(trace_normalize(K_tt))` | r25's prescribed instrument (Eq. 10 over the trace-normalized unconditional target bag Gram) |
| `targetConfidenceMean` | `w_t.mean()` | Eq. (24)'s magnitude — the vanishing half of r25's mechanism |
| `targetConfidenceMin` | `w_t.min()` | the **onset**; a mean over ten bags hides one vanished weight |
| `activeClasses` | `len(losses)` | `\|C_act\|` (Eq. 37) — r25's third named observable |
| `sourceKernelMean`, `crossKernelMean`, `targetKernelMean` | off-diagonal means of `K_ss`, `K_st`, `K_tt` | whether one `σ` still describes both clouds (Decision 10) |

**Why `w_t` is recorded — the "why" the brief asks for.** Not because it is
cheap, but because **neither half is diagnostic alone**, and that is r25's own
text. `bag_confidence` is 1 at a one-hot `g` and 0 at a uniform one
(`confidence.py:58-68`). So a collapsed target cloud can produce **either** a
vanished weight (uniform `g`) **or** a maximal weight (a confidently wrong single
class) — two different degenerate configurations with the **same** entropy
reading. And a low `w_t` with a healthy entropy is a different regime r25 names
separately: "el mismo comportamiento declarado para pseudoetiquetas no fiables."
Only the pair separates them. One field would be dishonest about which failure
the record is reporting.

`activeClasses` is the cheapest of all — one integer, already computed as the
length of the `losses` list — and it is the third observable r25 names in the same
sentence: "`C_act` en Ec. 37 se reduce a una sola clase, con lo que el término
global exige menos." It is what distinguishes *the global term is asking less*
from *the representation collapsed*.

**Why not the excluded instruments, restated from r25's own line numbers.**
r25:1087 excludes "el colapso de atención intra-bolsa, fenómeno distinto que no
detecta este caso." That excluded phenomenon has its own home in r25, at
**r25:516**, which defines `m_eff` and the uniform self-similarity and says
explicitly that near-uniform weights are the **expected** behaviour in a
dispersed bag. `MIL_CREDA.attention.effective_bag_size` and
`uniform_self_similarity` are that paragraph, implemented. They stay unwired.

**Why not `dependency_score`'s `H_target`.** `global_term.py:63` is
`quadratic_entropy(trace_normalize(K_t_weighted))` — per-class, conditional, and
confidence-weighted. r25:1087 confirms it is already in the objective with a
positive sign and coefficient ½, and that its fall raises `ℓ_glob,c`. But it is
conditional on the class routing, and r25's declared case is a collapse **uniform
across classes** driving `C_act` to one class — at which point the per-class
reading has one class left and characterizes nothing. It must not be accepted as a
substitute.

**The boundary: measured, and narrower than the spec assumes.**

- A rank-one trace-normalized Gram gives `‖A‖²_F = 1`, hence `H₂ = -ln 1 = 0`.
  It **reports its minimum and does not raise.** So the fully-collapsed case — the
  one the diagnostic exists for — is already safe.
- The range for `n` bags is `[0, ln n]`: at `K_tt = cI`, `A = I/n`,
  `‖A‖²_F = 1/n`, `H₂ = ln n`. With `BAGS_PER_STEP = 10` that is `[0, 2.302585]`.
  The floor 0 is **attained**; `ln n` is a supremum not attained, because a
  Gaussian bag kernel has strictly positive entries. Both ends are pinned by
  test, as the spec requires, not by this paragraph.
- The raises are therefore reachable only via a **non-positive trace**
  (`trace_normalize`, `renyi.py:58-59`) or a **zero Frobenius norm**
  (`quadratic_entropy`, `renyi.py:86-87`) — i.e. an all-zero Gram, or an **empty**
  target batch, neither of which real material produces. `_draw_target` always
  takes `BAGS_PER_STEP` bags. So the guard is required by spec and its tests must
  construct the boundary **synthetically**, which the spec's own scenarios already
  do by naming an all-zero Gram.
- The guard returns `float("nan")` and never propagates. Same sentinel as the
  floor's absent reading, same precedent.

### Decision 10: What makes the bandwidth divergence visible during a run

**What the measurement establishes.** Within M/U/S = 36.135 / 34.835 / 35.086;
across M–U / M–S / U–S = 35.484 / 35.601 / 34.952. All within ~4%, all mid-range
in `KERNEL_SIGMA_RANGE = (3.61, 361.35)`. So one bandwidth is defensible and
nothing moves: `config.KERNEL_SIGMA` is unchanged and
`harness.hyperparameter_drift` does not fire.

**What it does not establish, and one thing stronger than the brief credits.**
It was taken at initialization, where both encoders are the same pretrained
resnet18, so it says only that the **image** distributions are comparable. It says
nothing about 20 epochs of independent training, where — unlike under a shared
extractor — nothing ties the two clouds together. `kernelSigma` **is** searched
(`config.HYPER_DIMENSIONS`, log-scale at `harness.py:1687`) and its per-transfer
winners already spanned 30.93–262.92 under one encoder, so the search absorbs
drift within a decade; whether a decade is still the right bracket is a question
only a run answers.

Stronger than measurement, though: **r25 itself retains the one-`σ` claim under
two encoders.** r25:732, in the same document that declares Eq. (13), says "Un
único ancho de banda `σ` gobierna los tres bloques, ya que los tres derivan del
mismo kernel de instancia." The instance kernel is one kernel over the shared
codomain `R^d`. So "one bandwidth" is not only defensible by our measurement; it
is r25's own position *after* splitting the encoders. That moves the risk from
"the paper may not have meant this" to "the paper means it and the empirical
bracket is untested", which is a narrower and more honest risk.

**The instrument, and it costs nothing.** Divergence shows as the **cross block
collapsing relative to the within blocks** — which is exactly the statistic the
2026-09-17 measurement used to catch the frozen-normalization artefact:
"off-diagonal bag kernel **0.004 against 0.43**" (`AGREED.md`, Reversed,
2026-09-17). `K_ss`, `K_st` and `K_tt` are all built every step at
`wiring.py:267-269`. So the reading is the off-diagonal means of the three, per
epoch, plus their ratio.

The primitives are recorded **alongside** the ratio, not replaced by it, because
this repository has twice written down why: `latent.geometry` keeps
`crossDomainSameClass` and `betweenClasses` beside `ratio`, and `run_one` keeps
`supervised` beside `adaptationShare` because "without a denominator, 'the term
commanded nothing' and 'the term was scaled to nothing' print alike."

**So: yes, something makes divergence visible during a run.** The reading is the
measurement that already caught this class of artefact once, re-taken every epoch
at zero cost, on the matrices the objective is computed from — not a proxy and not
a post-hoc inference. What it **cannot** do is decide whether the search bracket
is still one decade; that stays a question a run answers, and it is named as such
rather than instrumented away.

### Decision 11: The revision bump is twelve source lines, not one

**Measured.** `research-concept-r21` appears **eleven** times under `src/`:

- `src/MIL_CREDA_Benchmark/config.py:300` — `REVISION`
- `src/MIL_CREDA/__init__.py:62` — `__implementation__["revision"]`, which
  `tests/test_benchmark_declarations.py:107` asserts equals `config.REVISION`
- nine `__provenance__["revision"]` entries: `attention.py:29`,
  `confidence.py:19`, `conditional.py:18`, `global_term.py:22`,
  `local_term.py:19`, `renyi.py:20`, `kernels.py:17`, `objective.py:21`,
  `bag_kernel.py:22`

and `tests/test_smoke.py:38` **hardcodes** `assert provenance["revision"] ==
"research-concept-r21.md"`. So the bump is 11 source lines + 1 test line, and the
suite refuses without the twelfth. The proposal's "one line" is measurably false
and the design records the correction; its *intent* — one commit, that change and
its prose only — stands.

**No `__provenance__["equations"]` list moves, verified.** Every equation number
this repository cites carries the identical `\tag{}` in r25: 7, 8, 9, 10, 11, 12,
13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 28, 29, 37, 38, 39 — checked directly
against `proposals/research-concept-r25.md`. So the revision string moves and the
mathematics does not, which is what makes this a bookkeeping commit.

**A consequence worth naming: the bump must be first, for a second reason.**
`latent.load` (`latent.py:415-424`) builds the model and then
`load_state_dict(...)` at strict default. Under two encoders the state dict gains
keys, so an r21 checkpoint would die on a key mismatch. It does not, because the
revision refusal at `:393-400` fires **first**. Ordered the other way — two
encoders under an r21 stamp — 67 checkpoints would fail with a confusing key error
instead of an honest revision refusal. This is the same conclusion the rollback
plan reaches from the record side, reached independently from the loader side.

### Decision 12: A citation names a revision and an anchor that does not move — never a line number

**Choice.** All **eleven** line-number citations lose the line number and gain a
stable anchor (the equation number, or the named quantity), and their revision
name moves `r21 → r25`. The numbers are **not** re-pointed to r25's 471/516/732.

**The sites — eleven, not ten.** The eleventh is why a search shape matters:

| site | cites | kind |
|---|---|---|
| `src/MIL_CREDA_Benchmark/tables.py:494` | `r21 l.501` | **report prose, in an f-string** |
| `src/MIL_CREDA_Benchmark/tables.py:1917` | `r21 l.501` | **report prose, in an f-string** |
| `src/MIL_CREDA_Benchmark/tables.py:132` | `r21 l.501` | comment |
| `src/MIL_CREDA_Benchmark/tables.py:1880` | `r21 l.501` | comment |
| `src/MIL_CREDA_Benchmark/config.py:453` | `r21 l.456` | comment |
| `src/MIL_CREDA_Benchmark/config.py:472` | `r21 l.715` | comment |
| `src/MIL_CREDA_Benchmark/config.py:474` | bare **`l.458`** | comment |
| `src/MIL_CREDA_Benchmark/wiring.py:260` | `r21 l.715` | comment |
| `tests/test_invariants.py:630` | `r21 l.456` | test docstring |
| `tests/test_report_tables.py:1302` | `r21 l.501` | test docstring |
| `tests/test_report_tables.py:1316` | `r21 l.501` | test comment |

`config.py:474` carries a bare `l.458` with no `r21` prefix, so a search keyed on
`r21 l\.` misses it. That is this repository's own named failure — a search shape
hiding evidence — and it is recorded so the pass is scoped by the derived
condition (*any* `l.NNN` in prose) rather than by the observed list.

**Load-bearing vs comment-only, because it decides slicing.**

- **Published, and under test.** `tables.py:494` and `:1917` are inside f-strings
  that reach the rendered report, and `tests/test_report_tables.py:1315` and
  `:1328` assert the literal substring `"l.501"` appears in the rendered
  objective and conclusion. These four are **one atom**: change the prose without
  the asserts and the suite reds; change the asserts without the prose and the
  test is asserting nothing.
- **Comment-only in `src/`.** `config.py:453`, `:472`, `:474`, `wiring.py:260`,
  `tables.py:132`, `:1880`. Six sites, no reader outside the source.
- **Test prose only.** `test_invariants.py:630`, `test_report_tables.py:1302`,
  `:1316`. No assertion.

**The sentences survive verbatim; only the numbers moved. Verified against r25.**

| cited | r25 line | the sentence |
|---|---|---|
| `l.715` | **732** | "Un único ancho de banda σ gobierna los tres bloques, ya que los tres derivan del mismo kernel de instancia." — verbatim |
| `l.501` | **516** | `m_eff` and the uniform self-similarity, read as a pair, with near-uniform weights "el comportamiento esperado" in a dispersed bag — verbatim |
| `l.456` / `l.458` | **471** | `R_phi`'s parameter grouping, `γ = 0`, the within-bag consensus, and "Los mismos parámetros se emplean para fuente y objetivo" |

So this is bookkeeping, and it must be described as such. It is not mathematics.

**Every one of the eleven already has a stable anchor named in its own
sentence:**

| cited | stable anchor | already named where |
|---|---|---|
| `l.715` | **Eq. (23)** — the three blocks of a class's mixed matrix | `config.py:478` names Eq. (23) in the same comment |
| `l.456`, `l.458` | **Eq. (15)** and its consensus component | `config.py:450` names Eq. (15) in the same comment |
| `l.501` | the **named quantities** — `m_eff` and the uniform self-similarity — plus §3.2 | `tables.py`'s own prose already names "autosimilitud" |

For `l.501` the anchor is the quantity name rather than an equation number,
because r25 gives that paragraph no `\tag{}`. A quantity name is **more** stable
than a section number and it is searchable in both the document and the code —
`effective_bag_size` and `uniform_self_similarity` are the names of the two
functions that implement it.

**Rejected — re-point the numbers to 471/516/732.** Smaller diff (eleven
substitutions plus two asserts) and honest about r25 today. Rejected because it
re-arms the identical trap for r26 at the same eleven sites, and the measured base
rate of that trap firing is **100%**: the one revision that has happened since
these were written invalidated every one of them, silently. Two of the eleven are
printed into the report, where a stale pointer is published with the authority of
a result.

**Rejected — leave them and add a checker that resolves line numbers against the
revision document.** This looks like the rigorous option and is the worst of the
three. The revision document lives in the **parent forge**
(`proposals/research-concept-r25.md`), outside this repository, which is its own
git repo nested inside and gitignored by that parent. A checker that cannot reach
its referent from a clone passes silently — "a check that always passes is not a
check", which this repository already wrote at `__init__.py:473`.

**Nothing will ever make these red, which is why the design must correct them
deliberately.** `verify`'s `prose.staleRevisions` builds its pattern as
`re.sub(r"\d+", r"\\d+", re.escape("research-concept-r25.md"))`
(`implementation_engine.py:1288-1291`), so it matches the **full filename** form
only. It cannot see `r21 l.501`, and it cannot see the bare `r21` either. It is
also report-only by design ("this is reported and never drifts a status"). So the
coordinator's read is right, and for a stronger reason than stated: the check is
blind to the revision *abbreviation*, not merely to the line number.

There is exactly one enforceable half available, and the design takes it:
`tests/test_report_tables.py:1315`/`:1328` pin the citation string into the
**rendered report**, so they must name whatever anchor the rewritten prose
adopts. That keeps the published citation under test, which is more than the
other nine sites can have.

**Slicing.** All eleven land in the **revision-bump commit**, not separately.

- Four of the eleven are in `src/tables.py`, so the atom cannot stand outside
  `src/`-touching sequencing — correcting the coordinator's hope on that point.
- But it costs **nothing incremental**: the bump commit already touches `src/` in
  eleven places, and `report_digest.source_digest` covers `src/**/*.py`, so the
  notebook-stamp cost is paid once by that commit whether or not the citations
  ride along.
- And the bump is the event that makes them wrong *to a reader*: after it, the
  report prints a claim attributed to `r21` under a `research-concept-r25.md`
  stamp.
- Splitting the three test-prose sites out (they carry no stamp cost) would leave
  the suite saying `l.501` while the report says something else — two spellings of
  one citation, which is the drift this repository has been bitten by twice
  (`test_benchmark_declarations.py:170-177` exists because of it).

---

## Data Flow

### Embedding dispatch — who calls which encoder

```
                          ┌─────────────────────────────────────┐
                          │ Arm                                 │
  training_step ──────────┤ instance_embeddings(bags) ──► θ_s   │
   (source batch)         │                                     │
                          │ target_embeddings(bags)  ──► θ_t    │
  _milcreda_term ─────────┤        (θ_s when target_encoder is  │
   (target batch)         │         None, i.e. the floor B)     │
                          │                                     │
  _creda_term ────────────┤ (unchanged; reaches θ_t through     │
   (undeclared branch)    │  target_embeddings for free)        │
                          │                                     │
  forward(bags) ──────────┤ _scores(instance_embeddings(bags))  │
  score_target(bags) ─────┤ _scores(target_embeddings(bags))    │
                          └─────────────────────────────────────┘
                                        ▲              ▲
                                        │              │
  harness.accuracy(m, ds, dev) ─────────┘              │
    → run_one "sourceAccuracy"                         │
                                                       │
  harness.target_accuracy(m, ds, dev) ─────────────────┘
    → run_one "targetAccuracy"

  latent.represent(..., domain)         ─┐
  latent.attention_spread(..., domain)   │  domain REQUIRED,
  latent.correspondence   (:601 / :602)  ├─ refuses on anything
  latent.bag_pairs        (:920 / :921)  │  but wiring.SIDES
  latent.analyse / latent_grid /        ─┘
         correspondence_grid
```

### The evaluation path, before and after

```
BEFORE                                AFTER
------                                -----
accuracy(model, judged_source) ─┐     accuracy(model, judged_source)
                                │        → forward → θ_s   ✓ sourceAccuracy
accuracy(model, judged_target) ─┘
     both → model(x) → forward         target_accuracy(model, judged_target)
          → instance_embeddings            → score_target → θ_t  ✓ targetAccuracy
          → θ_s                              (θ_s on B: one encoder, settled)
     ✗ target scored by θ_s
```

Four call sites move: `harness.py:691` and `:849` (`targetAccuracy`). Two stay:
`:690` and `:848` (`sourceAccuracy`). `model(x)` appears exactly **once** in
`src/MIL_CREDA_Benchmark` (`harness.py:432`), so the blast radius is closed.

### The diagnostic — a reading per step, aggregated per epoch, one field per run

```
_milcreda_term (wiring.py:255-310)
  H_t = target_embeddings(target_bags)                          ← θ_t
  K_ss, K_st, K_tt = bag_kernel_matrix(...)        :267-269
  w_t = confidences(G_t)                           :273  (Eq. 24)
  for class_id ...: losses.append(...)             :278-293
  │
  ├─ targetBagEntropy   = guarded H₂(trace_normalize(K_tt))   ← r25:1087, Eq. 10
  ├─ targetConfidence{Mean,Min} = w_t.mean() / w_t.min()      ← Eq. 24
  ├─ activeClasses      = len(losses)                         ← |C_act|, Eq. 37
  └─ {source,cross,target}KernelMean = off-diag means         ← the σ reading
                 │
training_step ───┤  returns them beside supervised/adaptation/contribution*
                 │  (floor B: float("nan"), never 0.0 — 0.0 is the collapsed value)
                 ▼
run_one ─────────►  epochs_record[epoch]  ── beside sourceAccuracy/targetAccuracy
                 │       (means over the epoch's 7 steps; min where informative)
                 │
                 ├──►  run["targetBagEntropyMin"]   (min over epochs)
                 │        answers "did it EVER degenerate" — which
                 │        `final = epochs_record[-1]` structurally cannot
                 ▼
campaign ───────►  json.dumps(run) → runs.jsonl        (harness.py:2310)
                   NOT config.DIMENSIONS, NOT report["dimensions"]
```

### Sequence: one epoch of an adapted arm, with the readings in place

```
run_one          Arm.training_step      Arm._milcreda_term        MIL_CREDA
   │                    │                       │                     │
   │─ ramp, ramp_local ►│                       │                     │
   │                    │─ instance_embeddings ─┼──► θ_s              │
   │                    │                       │                     │
   │                    │─ _draw_target ────────┤                     │
   │                    │─ target.take ────────►│                     │
   │                    │                       │─ target_embeddings ─┼─► θ_t
   │                    │                       │─ bag_kernel_matrix ►│ K_ss,K_st,K_tt
   │                    │                       │─ confidences ──────►│ w_t  (Eq. 24)
   │                    │                       │─ trace_normalize ──►│      (Eq. 9)
   │                    │                       │─ quadratic_entropy ►│      (Eq. 10)
   │                    │                       │  ── readings ──┐    │
   │                    │◄── terms + readings ──┤                │    │
   │◄── step dict ──────┤                       │                │    │
   │                                                             │    │
   │─ backward ──► ONE optimizer, both encoders (harness.py:641)  │    │
   │─ accuracy(judged_source)        → θ_s ──────────────┐        │    │
   │─ target_accuracy(judged_target) → θ_t ──────────────┤        │    │
   │                                                     ▼        ▼    │
   │─ epochs_record.append({epoch, sourceAccuracy, targetAccuracy,     │
   │                        targetBagEntropy, targetConfidence*,       │
   │                        activeClasses, *KernelMean})               │
```

---

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/MIL_CREDA_Benchmark/config.py` | Modify | `REVISION` (:300) → r25 **(own commit)**; the citations at :453, :472, :474; the `normalization` rationale at :886-892 and :902-905 |
| `src/MIL_CREDA/__init__.py` | Modify | `__implementation__["revision"]` (:62) → r25 **(own commit)** — enforced equal to `config.REVISION` by `test_benchmark_declarations.py:107` |
| `src/MIL_CREDA/{attention,confidence,conditional,global_term,local_term,renyi,kernels,objective,bag_kernel}.py` | Modify | `__provenance__["revision"]` → r25, nine files, string only **(own commit)**. Equation lists unchanged — verified against r25 |
| `src/MIL_CREDA_Benchmark/wiring.py` | Modify | `SIDES`; `self.target_encoder` built **last** in `__init__`; `instance_embeddings` docstring re-declared as the source encoder's; `_target_embeddings` → public `target_embeddings` routing to `θ_t`; `_scores` extracted from `forward`; `score_target`; the five readings beside `K_tt`/`w_t`/`losses`; module docstring :4; the citation at :260 |
| `src/MIL_CREDA_Benchmark/harness.py` | Modify | `target_accuracy` beside `accuracy` (`accuracy` itself **unchanged**); `:691` and `:849` move to it; the readings aggregated into `epochs_record`; `targetBagEntropyMin` on the run dict. Optimizer (:641, :818) and `parameters` (:737) unchanged; the revision guard (:393-403) unchanged and expected to fire |
| `src/MIL_CREDA_Benchmark/latent.py` | Modify | `represent` and `attention_spread` take a required `domain`; `correspondence` (:601-602) and `bag_pairs` (:920-921) name their two methods; `analyse` (:658-659, :671), `latent_grid` (:868, :870-871) and `correspondence_grid` pass the domain |
| `src/MIL_CREDA_Benchmark/tables.py` | Modify | Citations at :132, :494, :1880, :1917 — :494 and :1917 are **report prose under test** |
| `src/MIL_CREDA_Benchmark/__init__.py` | Modify | The `poolable` justification at :477-480 ("bit-identical across every run") — true within an arm, false across arms. The declaration itself (:383, :484) stands |
| `src/MIL_CREDA/{renyi,global_term,attention}.py` | Unchanged (beyond the revision string) | Eqs. (9)/(10) reused as they are. The two `attention.py` diagnostics stay unwired — r25:1087 excludes them, and r25:516 is the paragraph they implement |
| `src/CREDA/**` | **Untouched, byte-identical** | Read as **evidence** for Decision 1 (`models.py:185-193`, `training_pipeline.py:345-346`), never edited |
| `tools/measure_kernel_sigma.py` | **Unchanged** | Measures the step-0 anchor, domain-symmetric by construction; editing the script that produced the declared constant would change what produced it |
| `tests/test_smoke.py` | Modify | `:38`'s hardcoded `"research-concept-r21.md"` **(bump commit — the suite refuses without it)** |
| `tests/test_arm_objectives.py` | Modify | `:496` splits, `:557` inverts to a partition, `:617` inverts and gains the twin match. **`:371` and `:461` unedited**; `:219`, `:698`, `:984`, `:1027` unedited (all pass source material) |
| `tests/test_figure_surfaces.py` | Modify | `:897` re-pointed to `target_embeddings`; `:917` gains the codomain assertion; `:963`'s premise re-worded and an encoder-identity assertion added (assertions otherwise unedited) |
| `tests/test_report_tables.py` | Modify | `:1315`/`:1328` must name the anchor the rewritten report prose adopts; `:1302`/`:1316` docstring prose |
| `tests/test_invariants.py` | Modify | `:630` docstring citation |
| `tests/test_independent_adaptation_schedules.py`, `tests/test_latent_sigma_readers.py`, `tests/test_attention_mechanisms.py` | Observe | Patch sites (:240, :56, :315) reach both encoders automatically. Any red here is a stub-construction issue, not an Eq. (13) issue |
| New tests | Create | Both-encoders-gradient split; the running-statistics partition; codomain identity over `named_parameters()`; the optimizer membership; the per-consumer `latent.py` behavioural guards; the AST guard on `instance_embeddings`; the diagnostic's minimum, its `[0, ln n]` bound, its boundary report, the per-class-`H_target` contrast |
| `MIL-CREDA/AGREED.md` | **Already written** | Reversed 2026-09-28 (`8ed79de`) and `:164` (`c55526a`). Carried, not re-decided |
| `MIL-CREDA/Notebooks/*` | Stamps invalidated | Any `src/` byte invalidates every stamp via `report_digest.source_digest`; `verification.ipynb` and `Benchmark_Results.ipynb` additionally carry the literal r21 string |

---

## Interfaces / Contracts

```python
# src/MIL_CREDA_Benchmark/wiring.py

#: The two domains an embedding call can name. NOT `DOMAINS`: `config.DOMAINS`
#: already means the three datasets (M/U/S), and a second spelling of one word
#: is a second answer.
SIDES = ("source", "target")


class Arm(nn.Module):
    def __init__(self, spec, classes, source, target, hyper=None):
        ...
        self.encoder = FeatureExtractor(backbone=config.BACKBONE,
                                        pretrained=config.PRETRAINED)   # theta_s
        self.head = nn.Linear(self.encoder.output_dim, classes)
        # ... attention parameters, the "creda" branch ...

        # LAST, and that is load-bearing: constructing a FeatureExtractor
        # consumes the global generator (resnet18 runs nn.init before
        # load_state_dict; the test stubs draw an nn.Linear). Built here, every
        # parameter B and E share is drawn from identical generator state, so
        # the rung B->E keeps exactly ONE difference. Built beside
        # `self.encoder`, B's head and E's head would differ at init too.
        # Through the module-level `FeatureExtractor` name, so the six test
        # files that patch that name stub both encoders.
        self.target_encoder = (
            FeatureExtractor(backbone=config.BACKBONE,
                             pretrained=config.PRETRAINED)               # theta_t
            if spec["adaptation"] is not None else None)

    # -------------------------------------------------------- representation

    def instance_embeddings(self, bags: torch.Tensor) -> torch.Tensor:
        """Encode through this arm's SOURCE encoder (theta_s of Eq. 13).

        Named `instance_embeddings` and not `source_embeddings`, which would
        read better beside `target_embeddings`: `tests/test_arm_objectives.py`
        :429 -- inside the floor guard at :371, which this change may not edit
        -- calls this name. The asymmetry is recorded as ugly rather than
        presented as a design.
        """

    def target_embeddings(self, bags: torch.Tensor) -> torch.Tensor:
        """Encode through this arm's TARGET encoder (theta_t of Eq. 13).

        On the floor there is no target encoder, and this resolves to the one
        source-trained encoder it has: B predicts the target through the same
        encoder it trained on source, which is settled and not a fallback
        invented here.

        Promoted from the private `_target_embeddings`: this is now the place
        the ENCODER CHOICE lives, which is a better reason for a named place
        than the freeze that was withdrawn, and `latent.py` has to be able to
        call it without reaching into a private.
        """

    def _scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Eq. (19) + head for a bag-unit arm, the log of the averaged
        instance distribution for an instance-unit one. Extracted from
        `forward` so `forward` and `score_target` cannot drift: the unit rule
        is one formula, and duplicating it into `harness` would fork it across
        two modules."""

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """Bag-level scores through theta_s. Signature UNCHANGED -- `model(x)`
        stays the nn.Module contract, which is what keeps `Arm` usable as a
        plain module, keeps `MechanismArm` working, and keeps
        `tests/test_arm_objectives.py:443` green."""
        return self._scores(self.instance_embeddings(bags))

    def score_target(self, bags: torch.Tensor) -> torch.Tensor:
        """Bag-level scores through theta_t. Evaluation's own entry point,
        because `forward` cannot carry a domain without either a default (the
        thing this change removes) or a TypeError on `model(x)`."""
        return self._scores(self.target_embeddings(bags))
```

```python
# src/MIL_CREDA_Benchmark/harness.py

@torch.no_grad()
def accuracy(model, dataset, device) -> float:
    """UNCHANGED, signature included. Scores through `model(x)` -> theta_s.
    Correct for source material, and correct for the floor's target material
    (B has one encoder). Two tests call it with three arguments and one of
    them is a floor guard that may not be edited."""


@torch.no_grad()
def target_accuracy(model, dataset, device) -> float:
    """`accuracy`, through `model.score_target(x)` -> theta_t. The sibling
    rather than a fourth argument on `accuracy`: a required argument would
    edit `tests/test_arm_objectives.py:443`, and an optional one would be the
    default this design rejects."""
```

```python
# src/MIL_CREDA_Benchmark/latent.py

@torch.no_grad()
def represent(model, bagset, positions, device, domain, unit=None):
    """`domain` is REQUIRED and has no default: one of `wiring.SIDES`, and
    anything else raises. `config.py:482-484`'s own rule for `KERNEL_SIGMA`
    -- "Every caller passes this explicitly, with no default anywhere in the
    call chain" -- applied to a stronger case: a wrong sigma changes a number,
    a wrong domain changes which model produced it.

    A string argument here and two method names on `Arm`, deliberately:
    `represent` is a chunking loop that CONTAINS an embedding call, and
    splitting it would duplicate the chunking, the `unit` override and the
    label bookkeeping -- `_conditional_reading`'s own rule, one function
    differing in exactly one thing.
    """
```

```python
# the reading, guarded -- wiring.py, beside K_tt
def _target_bag_entropy(K_tt: torch.Tensor) -> float:
    """Eq. (10) over Eq. (9)'s normalization of the UNCONDITIONAL target bag
    Gram: r25:1087's own prescription, and no new mathematics.

    Reports and never raises. `trace_normalize` raises on a non-positive
    trace (renyi.py:58-59) and `quadratic_entropy` on a zero Frobenius norm
    (renyi.py:86-87); a diagnostic that crashes exactly when the mode it
    exists to catch is most severe is the opposite of a diagnostic.

    Measured range for n bags: [0, ln n]. The floor 0 is ATTAINED by a
    rank-one Gram (||A||_F^2 = 1, H2 = 0) -- so the fully collapsed case
    reports its minimum rather than raising. `ln n` is a supremum and not
    attained, because a Gaussian bag kernel has strictly positive entries.
    The raises are reachable only from an all-zero Gram or an empty batch,
    neither of which real material produces; their tests construct the
    boundary synthetically.

    The sentinel is `float("nan")` -- the same one `harness.accuracy:435` and
    `latent.geometry:488-496` already write into records -- and NEVER 0.0,
    because 0.0 is the collapsed reading itself.
    """
```

**Record shape.** Per epoch, in `epochs_record`, beside the two accuracies:

```json
{"epoch": 0, "sourceAccuracy": 0.42, "targetAccuracy": 0.31,
 "targetBagEntropy": 1.83, "targetConfidenceMean": 0.51,
 "targetConfidenceMin": 0.07, "activeClasses": 9,
 "sourceKernelMean": 0.44, "crossKernelMean": 0.39, "targetKernelMean": 0.46}
```

Per run, one field: `"targetBagEntropyMin": 1.62`. The floor writes `NaN` for
every one of the seven and `0` is never used in their place.

---

## Testing Strategy

`strict_tdd: true`. Every row below is RED before GREEN, **observed**, never
asserted. Test command: `PYTHONPATH=src .venv/bin/python -m pytest -q`.

| Layer | What to test | Approach |
|---|---|---|
| Unit — construction | E/F/G hold two `FeatureExtractor`s sharing **no parameter object** | identity (`is`) over `named_parameters()`, never equal shapes or values |
| Unit — construction | B holds exactly one | `arm.target_encoder is None` and one `FeatureExtractor` instance |
| Unit — construction | The codomain is shared | `target_encoder.output_dim == encoder.output_dim`, asserted where the widths meet (`test_figure_surfaces.py:917`) |
| Unit — construction | Construction order keeps the rung at one difference | B's and E's head weights are bit-identical after `_arm`; reds if the target encoder moves to `:96` |
| Unit — construction | Patching `wiring.FeatureExtractor` stubs **both** | both encoders are instances of the stub |
| Unit — parameters | 11,247,434 / 22,423,946 | measured from `sum(p.numel() for p in model.parameters())`, never from a document |
| Unit — optimizer | Both encoders in **one** optimizer at one rate | every `target_encoder` parameter appears in the same `Adam` instance; no second optimizer exists |
| Unit — gradient | Adaptation alone gradients **both** encoders | `:496` split; **two** declared reds, one per side (`no_grad` on the target forward; `detach` on `H_s`) |
| Unit — normalization | The target forward moves only `θ_t`'s running stats | `:557` partition over the class-level `_BNEncoder` spy |
| Unit — normalization | The adapted arm's **source** stats MATCH its B twin's, and its two encoders' differ | `:617` inverted pair — strictly stronger than the original |
| Unit — diagnostic | The reading is `quadratic_entropy(trace_normalize(K_tt))`, unconditional | recomputed from `MIL_CREDA.renyi` against the step's own `K_tt` |
| Unit — diagnostic | A rank-one Gram reads exactly 0; every real batch lands in `[0, ln BAGS_PER_STEP]` | both ends pinned by test, not by prose |
| Unit — diagnostic | Monotone fall across increasing collapse | a constructed sequence of decreasing effective rank |
| Unit — diagnostic | The **boundary reports** rather than raises | synthetic all-zero Gram and empty batch; `NaN`, no `ValueError` escapes |
| Unit — diagnostic | The per-class `H_target` **fails** to detect a class-uniform collapse the required reading catches | one batch, two readings, opposite verdicts |
| Unit — diagnostic | The excluded intra-bag instruments read healthy on a collapsed **cloud** | `uniform_self_similarity`/`effective_bag_size` on the same batch |
| Unit — diagnostic | `w_t`'s mean and min, and `activeClasses`, reach the record | present on `training_step`'s return and on `epochs_record` |
| Unit — floor | B writes `NaN`, never `0.0`, for the readings | a `0.0` mutation reds — the floor would otherwise read maximally degenerate |
| Integration — eval | `targetAccuracy` goes through `θ_t`, `sourceAccuracy` through `θ_s` | perturb `θ_t` only: `targetAccuracy` moves, `sourceAccuracy` does not; then the mirror |
| Integration — eval | The floor evaluates target through its one encoder | `accuracy` and `target_accuracy` agree on B, disagree on G |
| Integration — `latent.py` | **One assertion per consumer**: `represent`, `attention_spread`, `correspondence`, `bag_pairs` | perturb `θ_t`: the target-side reading moves, the source-side does not; then the mirror. Never inferred from a sibling |
| Integration — `latent.py` | A missing or unknown `domain` **refuses** | `TypeError` on omission (no default), `ValueError` on an unknown value |
| Structural | No module under `src/MIL_CREDA_Benchmark/` other than `wiring.py` calls `instance_embeddings` | AST walk, a **derived condition** not a roster; precedent at `test_figure_surfaces.py:673-688` |
| Integration — trace | The reading is obtainable at more than one point in a run | ≥2 distinct `epochs_record` entries carry it |
| Integration — record | The run dict survives `json.dumps` with `NaN` present | round-trip; precedent `latent.geometry`, `harness.accuracy` |
| Regression — bump | Every `__provenance__["revision"]` and `__implementation__["revision"]` equal `config.REVISION` | `test_smoke.py:38` re-pointed; `test_benchmark_declarations.py:107` unchanged |
| Regression — citations | The rendered report's citation string is under test | `test_report_tables.py:1315`/`:1328` name the adopted anchor |
| Regression — floor | `test_arm_objectives.py:371` and `:461` pass with **no edit** | read back the file's bytes for those two tests |
| Regression — suite | Full suite green, with the sibling change's three `tests/test_search_distribution.py` reds still red | counted as its work, not as regressions here, and not "fixed" |

**Observation obligations — things the design cannot settle by reading, and that
must be measured rather than assumed:**

1. `test_figure_surfaces.py:1002`'s abscissa overlap under two independently
   initialized stub encoders (Decision 6d). If it fails, replace the heuristic
   with the structural claim; do **not** reseed until it passes.
2. Any test asserting a specific accuracy value for E/F/G: `targetAccuracy` is now
   measured through `θ_t` and the numbers move.
3. `tests/test_latent_binding.py` and `tests/test_latent_sigma_readers.py`: a
   state dict saved under one encoder cannot load into a two-encoder model. For
   on-disk records the revision refusal fires first (Decision 11); for tests that
   build and load in-process it does not.
4. Whether `verify`'s `prose.unresolvedSymbols` already reports the eleven
   `l.NNN` tokens today (it is report-only either way).

---

## Threat Matrix

**N/A** — no routing, shell, subprocess, VCS/PR automation, executable-file
classification, or process-integration boundary. This change touches model
construction, an embedding-dispatch boundary inside one Python package, record
fields, and prose. `tools/measure_kernel_sigma.py` is a script and is
**unchanged**. No new file is made executable, no command is constructed, no
external process is invoked, and no path is taken from untrusted input.

---

## Migration / Rollout

**No data migration.** The guards refuse stale records; they never rewrite them
(`harness.py:393-403`, `latent.py:393-400`, `latent.py:113`). New r25-stamped
artifacts are additive files.

**Disposition of the 67 r21-stamped records** is `revision-bound-records`' to
specify; this design adds one loader-side fact: the revision refusal fires
**before** `load_state_dict`, so the two-encoder state-dict shape change never
surfaces as a key mismatch on an existing record — provided the bump lands first.

**Ordering, and it is not negotiable.** The safe partial state is *revision
bumped, one encoder*: r21's implementation declared under r25, which every guard
reports honestly as a mismatch between what the code does and what the paper says.
The unsafe inverse is *two encoders under an r21 stamp*, which would write records
claiming a revision whose mathematics the code no longer implements — and would
die on checkpoint keys instead of refusing on the revision.

**Before the first `src/` edit**, record the current `report_digest.source_digest`.
Reverting the `src/` diff restores it and every notebook stamp with it, so a
revert does **not** re-owe the pilot walk.

**Revert order** — each step independently safe, and it is the proposal's order
with the citations folded into the bump:

```
(d) the diagnostic
(c) the latent.py seam            + test_figure_surfaces.py:897/:963
(b) the second encoder            + test_arm_objectives.py:496/:557/:617
       ^^ ONE atom: re-pointed tests without the encoder are red,
          and the encoder without them is red
(a) the revision bump + the eleven citations + test_smoke.py:38
```

### Delivery slices and the 400-line budget

The forecast crosses the budget. `ask-on-risk` means the chain shape is a
tasks-time question; the natural slices, with the corrected grouping:

| slice | content | ~authored lines |
|---|---|---|
| **(a)** | The revision bump (11 `src/` lines + `test_smoke.py:38`) and the eleven citations. Own commit, first | ~30 |
| **(b)** | The second encoder, construction order, `target_embeddings`, `_scores`/`score_target`, `harness.target_accuracy`, the prose at `wiring.py:4` / `config.py:886-905` / `__init__.py:477-480`, and `test_arm_objectives.py:496`/`:557`/`:617` + the new construction/gradient/optimizer tests | ~200–260 |
| **(c)** | The `latent.py` seam, `test_figure_surfaces.py:897`/`:917`/`:963`, the per-consumer behavioural guards, the AST guard | ~150–200 |
| **(d)** | The five readings, their aggregation into `epochs_record`, `targetBagEntropyMin`, and the diagnostic's tests | ~180–220 |

**Correcting the proposal's grouping:** `test_figure_surfaces.py:897` and `:963`
belong to **(c)**, not to the encoder slice. At (b), `bag_pairs` is still
domain-blind, so `:897`'s rebuild and the function under test still agree and the
test stays green; it reds only when `bag_pairs` splits. So (b) carries **three**
re-pointed tests and (c) carries **two**.

`Decision needed before apply: Yes` — the chain shape.
`Chained PRs recommended: Yes`.
`400-line budget risk: High`.

**Not delivered by this change, and gated elsewhere.** The pilot walk and the
nine-account rehearsal are owed (the bump un-ticks the six-step position; the
first `src/` byte invalidates every notebook stamp), and spending metered quota
remains the operator's own gate. This change delivers the launch precondition,
never the launch.

---

## Open Questions

Both named open questions are settled above. What remains is named as
undetermined rather than defaulted:

- [ ] **The healthy band of `targetBagEntropy` is not derivable** from r25 or from
      the code — only its degenerate end (0) and its `[0, ln n]` range are. This is
      why it is declared `DESCRIPTIVE` with a written two-sided reading rather than
      given a threshold, and why it must be read paired with the confidence weights
      and the kernel triple. Only a run establishes it.
- [ ] **Whether one decade around the source-measured anchor is still the right
      `kernelSigma` bracket under two encoders.** The search absorbs drift within a
      decade and its winners already spanned 30.93–262.92; whether a decade is
      enough is a question only a run answers. Decision 10 makes the divergence
      *visible* per epoch; it does not decide the bracket.
- [ ] **`test_figure_surfaces.py:1002`'s abscissa overlap** under two independently
      initialized stub encoders. Must be observed. The repair, if it fails, is the
      structural claim — not a reseed.
- [ ] **Whether `activeClasses` should also reach a per-run field.** It is on the
      epoch record, which answers r25's claim; a per-run minimum was considered and
      left out as record weight. Reversible without touching anything else.
