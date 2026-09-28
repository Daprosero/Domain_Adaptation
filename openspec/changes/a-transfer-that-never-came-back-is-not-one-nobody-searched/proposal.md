# Proposal: A transfer that never came back is not one nobody searched

## Intent

The hyperparameter ceiling search runs whole on one machine. Measured: **5.6–6.2 h on a
Kaggle Tesla T4**, against a 12 h session cap — one interruption from losing the run that
governs every downstream campaign. The search already has an axis it could split on: its
six transfers are independent, and a rehearsal measured each at 0.9 min. Split six ways,
the projection is **~1.2 h per machine**.

The capability is half-built and the built half is unreachable. `search_ceilings` and
`search_ceilings_trials` have accepted `transfers=` for some time, and commit `7fcc150`
wired `run_search` to forward it through both hops. What is missing is above and below
them: nothing declares that the search may split at all, nothing translates a worker's
assigned units into transfers, and **nothing assembles the partial records six workers
would produce.**

The reason this matters more than the wall-clock saving is the failure mode. At
`harness.ceiling_for` (harness.py:502-504) a transfer absent from `ceilingsByTransfer`
resolves to the pooled winner, and the function's own docstring (:462-466) frames that as
the legitimate out-of-sample reading. `hyper_for` (:507-545) does the same for the seven
other searched dimensions. So a transfer missing **because its shard never came back** is
indistinguishable at the call site from a transfer nobody intended to search — and the
fallback is not a neutral, it is a plausible-looking winner. Splitting the search without
building the assembly first would produce a `ceilings.json` claiming six transfers while
holding one, with every existing check green.

Half the protection is already here and it is the half that blocks progress.
`harness.governs_the_ceilings_record(noise, transfers)` (:1527-1550) gates the only two
writes to the canonical record path (:1830, :2086), deriving the decision from the
writer's own arguments rather than a flag it must remember. Its consequence: **a
subset-transfer search writes nothing today.** There is no clobber risk, and equally no
per-shard artifact for any merge to read. This change must create that artifact, not only
the merge over it.

## Scope

### In Scope

Pieces 1, 3 and 4 of the four the statement of intent names. One change, not three,
because they share one indivisible cost (see **Cost** below).

1. **The declaration names the axes a shard may split on, with per-axis rules.**
   `MIL_CREDA_Benchmark.__benchmark__["distribution"]` gains an `axes` roster containing
   both `seed` and `transfer`, and the four grouping rules become resolvable per axis, so
   the instruction a search shard reads is the one that applies to search shards.
2. **A unit resolves in the axis the step declares.** A new `config.execution_units_for`
   taking an explicit `axis`, so `FORGE_RUN_UNITS` is never read by guessing which axis a
   value belongs to.
3. **Partial search records become a per-shard artifact and a refusing assembly.** A
   subset-transfer search persists a per-shard record; `harness.merge_search_records`
   assembles them into the canonical record and **refuses** when a transfer it expected is
   absent (Decision B).
4. **The scalar readers of `axis` keep working, and this is stated rather than left to be
   inferred.** Under the resolved shape (below) `axis: "seed"` survives untouched, so
   `tools/distribute.py:208`'s `.get("axis")` still resolves to a real string and the plan
   JSON still carries a real axis answer. The earlier worry — `None` into the plan with no
   error — is *avoided by the shape*, not repaired by an edit. What `tools/` does need is
   the per-axis selection wherever it reads the four groups, and those repairs are
   **sliceable**: `report_digest.source_digest` covers `src/**/*.py` only, so nothing under
   `tools/` carries notebook-stamp cost.
5. **Prose that would outlive its mechanism is corrected with it**: the
   `__init__.py:392-480` rationale block, the `config.execution_seed_units` docstring
   (config.py:237-239) and the `harness.run_campaign_shard` docstring (:2536-2538) all
   cite the singular axis as a justification. The field survives, but the *justification*
   does not: "there is no second axis a unit could name" becomes false the moment `axes`
   exists, and a comment that outlives its mechanism is this repository's most-repeated
   defect.
6. **One named source for the expected transfer set.** `expected` derives from
   **`config.SEARCH_TRANSFERS`** and nothing else — see Approach for why that one.

### Out of Scope

- **Piece 2 — `run_search` forwarding `transfers`.** Already done in `7fcc150`, together
  with the refusal at harness.py:2494-2518 that keeps a slice from being handed back as
  the record. Not re-proposed, not undone. Its two green tests
  (`test_run_search_forwards_the_transfers_it_is_given`,
  `test_a_partial_search_refuses_to_present_its_slice_as_the_record`) must stay green; the
  second one's message is the natural place for the assembly to be named once it exists.
- **Splitting the search by seed.** `config.SEARCH_ENGINE == "optuna"` cancels the draw by
  construction, and `test_the_search_still_has_no_seed_axis_to_split` pins `run_search`
  against acquiring a `seeds=` parameter. Splitting by transfer is not a door to that.
- **Splitting by arm.** The one split the declaration forbids
  (`tests/test_distribute.py:44-52`; the forge enforces it at
  `implementation_engine.py:1603`). Nothing here changes it, and see the Risks table for
  why this change must actively avoid *weakening* it.
- **`src/CREDA`** — prior work, not edited by this project.
- **Actually running the distributed search.** This change makes it askable and its result
  assemblable. Spending remote quota is a separate, authorized decision.
- **Reusing `shards.merge()`.** It is keyed on `(transfer, arm)` run cells and
  `runs.jsonl` lines (shards.py:339-343, 402-484), not on family-keyed ceiling entries.
  Its *pattern* is the precedent — refuse-on-disagree, refuse-on-incomplete, both
  `SystemExit` subclasses (shards.py:61-92) — the function is not reusable.
- **Teaching `disagreements()` dotted paths.** `__init__.py:448-462` records that
  `commit`/`codeDigest` cannot be checked because the vendored reader does a flat
  `stamp.get(field)`. Still true, still a forge-side change, still not this task's.

## Capabilities

`openspec/specs/` does not exist in this repository — this is its first SDD change, so
there are no existing capability names to extend and nothing to delta.

### New Capabilities

- `search-distribution-axis`: which axes this benchmark may split a shard on, and which
  agreement rule each of the four dimension groups carries **per axis** — including the
  inversion that motivates the whole capability (the searched ceiling fields must be
  identical across campaign shards and must differ across search shards).
- `run-unit-axis-resolution`: how one submission's opaque `FORGE_RUN_UNITS` identifiers
  become the units of a named axis, and what it refuses rather than guesses. Must preserve
  the existing three-way distinction (absent → `None`, `"[]"` → empty list, populated →
  validated).
- `partial-search-record-assembly`: the per-shard search record a subset-transfer search
  writes, the assembly of those records into the one record the campaign reads, and the
  refusal when an expected transfer is missing.

### Modified Capabilities

None.

## Approach

### The declaration (Decision A — settled: nest the four groups per axis)

One declaration must describe two shard classes whose rules about the same field names are
opposite. Today `identicalAcrossShards` holds `["epochs", "ceilings",
"ceilingsByTransfer", "hyperByTransfer", "labelNoise"]`. For **campaign** shards (split by
seed) those ceiling fields genuinely must be identical — two shards straddling a search
would merge into one table with adaptation inert on one half. For **search** shards (split
by transfer) each shard holds a *different slice* of exactly those fields; telling them to
agree is the opposite of their job.

The operator settled the shape: **nest the four groups per axis**, rather than adding a
plural `axes` beside one flat rules list (which leaves the wrong instruction standing for
search shards) or adding a second declaration block (two declarations that drift apart).

**The resolved shape (operator, 2026-09-28).** This does not reopen Decision A; it fixes
how the nesting lands, after a forge-side reader was measured requiring the flat keys:

1. **`axis: "seed"` stays exactly where it is and keeps that value.** Not moved, not
   renamed, not pluralized away. It is the one field the forge reads
   (`implementation_engine.py:1603`, `axis_is_comparison = dist.get("axis") == "arm"`), and
   it is what stops the forbidden arm split. Nothing here is permitted to cost that
   refusal its ability to fire.
2. **`axes` is added beside it**, and the four groups (`poolable`, `perEnvironment`,
   `perRun`, `identicalAcrossShards`) are nested per axis.
3. **The flat top-level keys the forge requires are bound to the same list objects as the
   seed axis — a projection, never a copy.** The per-axis nesting is the single source of
   truth; the flat keys are that same data seen from the other side. This is the whole
   reason the shape is safe: two structures that merely *hold equal lists* are two copies,
   and equality that happens to hold today is exactly how two copies begin to disagree.
   Identity is what makes drift unrepresentable rather than merely unlikely.
4. **The forge is not touched by this change**, and `shardsRoot` does not move. It sits
   beside the four groups at `__init__.py:498-506` and is read by forge-side commands
   (`gate`, `close`, `discuss`, `probe`) that do not exist in this repository.

`sdd-spec` MUST require a test that proves the flat keys and the seed axis block are **the
same objects** (`is`), not merely equal lists (`==`). A test written with `==` would pass
against a copy and would therefore measure nothing this decision is about.

One consequence worth stating because it removes a worry rather than answering it: with
`axis` surviving, every existing scalar reader of it keeps working unchanged —
`tools/distribute.py:208` included.

`shards.partition(dimensions, dist)` already takes `dist` as an argument rather than
calling `declaration()` itself (shards.py:316-336), and `tools/bridge.py:166` and
`shards.merge()` (:421) both accept a caller-supplied `dist` with a `declaration()`
default. That is the lever: callers select the axis sub-block and the generic partitioning
logic stays shape-stable. `harness.py:2353` and `pooling.py:32` read the flat keys
directly and are the two that must choose an axis explicitly.

### The unit reading (piece 3)

`config.execution_seed_units()` (config.py:233-295) decides "a unit IS a seed" and cites
the singular axis as its justification — a citation this change falsifies. The new
`execution_units_for(axis=...)` receives the axis and never infers it from content: `"0"`
is a valid seed and could equally be a transfer index, so inference here does not fail, it
runs a different experiment and returns numbers of the correct shape.

The three-way distinction at config.py:246-262 is deliberate and must survive whatever
wraps or replaces the existing function.

### The assembly (Decision B — settled: the merge refuses)

An incomplete search record is caught by the assembly function itself — not by a gate
above it and not by the consumer. The operator's two reasons: the assembler is the only
party already holding the list of transfers it asked for, and every existing refusal in
this machinery is a `SystemExit` raised by the producer or the merge, never by a separate
checker (`ShardsDisagree`, `ShardIncomplete`, `PlanConflict`, `ceilings_in_force`'s
stamp-drift exit, `run_search`'s no-record exit).

The red test requires `merge_search_records` to take `transfers=` or `expected=`, and
states why: counting what arrived cannot distinguish "all of them arrived" from "the ones
that were sent arrived."

**`expected` derives from `config.SEARCH_TRANSFERS`, and that choice is not arbitrary.**
Two constants name the same six transfers today — `SEARCH_TRANSFERS` (config.py:673) and
`VERDICT_TRANSFERS` (:844), both built from `TRANSFERS` (:306) — and the machinery already
reads them inconsistently: `governs_the_ceilings_record` compares against
`SEARCH_TRANSFERS` (harness.py:1549-1550) while `run_search`'s refusal message counts
`VERDICT_TRANSFERS` (:2509). Inheriting that ambiguity would let the assembly's
completeness test and the write guard disagree the day the two lists diverge — and the
disagreement would be silent, because the one deciding whether a record may be written and
the one deciding whether it is complete would be answering about different sets.
`SEARCH_TRANSFERS` is the correct source because completeness of a *search* record is a
question about what the search covers, and because it is what the write guard already
reads, so the two agree by construction rather than by coincidence.

Note for `sdd-apply`: aligning `run_search`'s message to the same constant keeps
`test_a_partial_search_refuses_to_present_its_slice_as_the_record` green, because that test
asserts `str(len(config.VERDICT_TRANSFERS))` appears in the message and both lists have
length six. That test must stay green on its own terms, not be edited to accommodate this.

Two structural constraints on where the writes may live, both measured:

- **`tests/test_label_noise.py:1253-1274` derives its roster from the module**: any
  `harness` function whose source contains both `ceilings_record_for` and `write_text`
  must also contain `governs_the_ceilings_record`. The assembly writes the canonical
  record, so it lands in that roster and must consult the guard. Assembling all six
  transfers makes the guard return `True` by construction, so this is satisfiable — but it
  is not optional, and a third writer added without it goes red.
- **The per-shard artifact needs a path of its own.** `config.ceilings_record_for(pilot)`
  is one path, not one per shard, and `search_record()` reads back from it. The per-shard
  destination is a new decision; `search_ceilings_trials`' existing scratch/`partial` file
  (harness.py:2091-2093) and `distribution.shardsRoot` are the two nearby precedents for
  `sdd-design` to weigh.

### On the statement of intent

`tests/test_search_distribution.py` is committed red and is the intent, **not the design**.
Its first test asserts only that a plural `axes` exists and contains both values — which
under-specifies Decision A entirely: a bare `axes: ["seed", "transfer"]` beside the
unchanged flat rules list would turn it green while leaving search shards reading the
campaign's rule. `sdd-spec` must grow the requirement past what that assertion measures,
and the growth must be red-first against the nesting, not against the roster.

## Accepted Limitations

Recorded as a decision with its reasoning, not as an open risk. This is known, priced, and
deliberately left standing.

**The forge vets only the axis named in `axis`. The transfer axis goes unvetted by it.**

The forge's single axis check is `axis_is_comparison = dist.get("axis") == "arm"`
(`implementation_engine.py:1603`) — it reads one scalar field and refuses one value. Under
the resolved shape that field still says `"seed"`, so the check still works and still
refuses what it always refused. But `axes` is invisible to it: the transfer axis enters the
declaration without anything on the forge side vetoing it.

**Harmless today**, because a transfer is provably not an arm: splitting by transfer keeps
every arm of every cell on one machine, which is the exact property the arm prohibition
exists to protect (`tests/test_distribute.py:44-52`). The hole is not that the transfer
axis is unsafe — it is measured safe. The hole is that **a future third axis would enter
the same way, unvetted**, and nothing would stop an unsafe one.

**The rejected alternative, and why it was rejected**: teaching the forge to vet every
declared axis rather than only the one in `axis`. Rejected on **cost** — it makes this two
repositories in one change, and the forge is outside this change's `allowedEditRoots` —
and **explicitly not on principle.** "A repository may divide by more than one thing" is
general to any paper and would not have been target leakage; it is the kind of thing the
forge legitimately could know. A later session can revisit this on cost alone, without
relitigating whether it belongs there.

## Open Design Questions

Named rather than answered. `sdd-design` settles these; this proposal deliberately does not.

1. **The transfer-axis unit vocabulary.** `FORGE_RUN_UNITS` carries JSON **strings**, while
   transfers are `(str, str)` tuples (`config.TRANSFERS:306`). Two live candidates: the
   canonical label `transfer_label()` already produces (`"M->U"`), or a positional index
   into the declared transfer list. The red test names the index as a real possibility
   (`"'0' is a valid seed and could equally be a transfer index"`), which is also precisely
   why the axis must be received and never inferred.
2. **The per-shard artifact destination.** `config.ceilings_record_for(pilot)` is one path,
   not one per shard, and `search_record()` reads back from it. Two nearby precedents to
   weigh: `search_ceilings_trials`' existing scratch/`partial` file (harness.py:2091-2093),
   and `distribution.shardsRoot`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/MIL_CREDA_Benchmark/__init__.py` | Modified | `distribution` block (481-507) gains `axes` + per-axis groups; the rationale block (392-480) is rewritten, since it currently justifies the singular axis and the flat `identicalAcrossShards` |
| `src/MIL_CREDA_Benchmark/config.py` | Modified | New `execution_units_for(axis=...)`; `execution_seed_units` (233-295) and its axis citation (237-239) |
| `src/MIL_CREDA_Benchmark/harness.py` | Modified | New `merge_search_records`; per-shard record write; `run_search` refusal message (2494-2518) names the assembly; flat-key read at 2353; axis citation at 2536-2538 |
| `src/MIL_CREDA_Benchmark/shards.py` | Modified | `declaration()` (95-98) and `partition()` (316-336) resolve an axis; `merge()`'s `dist` default (421) |
| `src/MIL_CREDA_Benchmark/pooling.py` | Modified | `declaration().get("perRun")` at line 32 — a flat read the handoff did not list |
| `tools/distribute.py` | Unchanged at :208 | `.get("axis")` keeps resolving to `"seed"` because the key survives — the resolved shape avoids this rather than repairing it. Any other flat-group read here selects an axis. **Outside the digest boundary: no stamp cost, sliceable** |
| `tools/bridge.py` | Modified | Flat reads at 167-168, 203, 216-217; `dist` default at 166. **Outside the digest boundary: no stamp cost, sliceable** |
| `tests/test_search_distribution.py` | Modified | Three red tests turn green; the first is grown past its current assertion |
| `tests/test_shards.py`, `test_benchmark_declarations.py`, `test_bridge.py`, `test_label_noise.py`, `test_report_tables.py`, `test_distribute.py` | Modified | Fixtures hardcoding the flat shape (`test_bridge.py:190-210`, `test_benchmark_declarations.py:160-227` and 1940-1994, `test_label_noise.py:375-401` and 1113, `test_report_tables.py:651,756`) |
| `MIL-CREDA/Notebooks/*` | Affected, not edited | Per `rules.proposal`: no notebook source changes, but every notebook **stamp** is invalidated — see Cost |
| `src/CREDA` | Untouched | Out of scope by project rule |

## Cost

**Stated plainly, because it is the reason these three pieces are one change.**

`report_digest.source_digest` (report_digest.py:28-56, verified) hashes every `*.py` under
`src/` and nothing else — deliberately excluding `tests/`, which is why the red tests could
be committed first at zero cost. **The first byte of `src/` this change touches invalidates
every notebook stamp**, which forces the whole six-step pilot walk to be re-run and the
eighteen remote rehearsals (nine accounts × two shapes) to be repeated.

That cost is paid once whether this change carries one piece or three. Landing pieces 1, 3
and 4 separately would pay it three times. Note the corollary for task slicing: the
`tools/distribute.py` and `tools/bridge.py` repairs are **outside** the digest boundary and
carry no stamp cost of their own.

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| **RESOLVED (operator, 2026-09-28).** The forge's declaration reader is outside `allowedEditRoots` and requires the flat shape: `implementation_engine.py:1326-1400` declares `axis` (**required, `str`**) plus the four groups (**required, `list`**). Nesting them away would make each read `missing`, the block read `incomplete`, `_distribution_list` return empty so every dimension reads `unpartitioned`, and `axis_is_comparison` (:1603) degrade to **permanently `False`** — a refusal that can no longer fire | Was **High — measured** | Closed by the resolved shape: `axis` survives with its value, the four flat keys stay as an identity-bound projection of the seed axis, and the forge is not touched. Retained in this table as the reason the shape is what it is — a later session that "simplifies" the flat keys away reintroduces exactly this. The residual is recorded under **Accepted Limitations**, not here |
| The flat keys drift from the per-axis nesting they project | Low — **by construction, not by care** | They are the same list objects, so drift is unrepresentable rather than discouraged. Spec requires an identity (`is`) test; an `==` test would pass against a copy and measure nothing |
| A transfer missing from a merged record still resolves to the pooled winner at `ceiling_for`/`hyper_for` | High if unaddressed | The refusing assembly (Decision B) is the mitigation: an incomplete record never reaches a consumer. Spec must state that the refusal is the *only* thing standing between a dead shard and a plausible number — the read-site fallback is deliberately kept |
| `merge_search_records` writes the canonical path without consulting `governs_the_ceilings_record` | Medium | Already caught by the derived roster at `test_label_noise.py:1253-1274`. Do not weaken that test to accommodate a new writer |
| Nesting silently drops `shardsRoot` or `currentWhen` | Medium | Both are forge-read optional keys with no in-repo consumer, so nothing here goes red if they are lost. Spec must pin them explicitly |
| The unit vocabulary for the transfer axis is undecided | Medium | Moved to **Open Design Questions** (1) — named for `sdd-design`, deliberately not decided here |
| `SEARCH_TRANSFERS` vs `VERDICT_TRANSFERS` as the expected set | **Resolved** | `expected` derives from **`config.SEARCH_TRANSFERS`** and nothing else, because that is what `governs_the_ceilings_record` already reads (harness.py:1549-1550), so the completeness test and the write guard agree by construction rather than by coincidence. See Approach |
| Re-running the pilot walk and eighteen rehearsals consumes metered remote quota | High, accepted | Inherent and priced above; paid once for all three pieces |
| The red tests pass without the intent being met | Medium | The first test under-specifies Decision A by construction. Grow the assertions red-first; a mutation that survives means the test measured the roster, not the rules |

## Rollback Plan

1. **Before any `src/` edit**, record the current `report_digest.source_digest` value and
   the notebook stamps that depend on it. That value is the rollback target: reverting the
   `src/` diff restores the digest and every stamp with it, so the pilot walk and the
   eighteen rehearsals do **not** have to be repeated on a revert.
2. The change is additive at every seam. Revert order, each independently safe:
   `harness.merge_search_records` and the per-shard write → `config.execution_units_for`
   → the `tools/` axis repairs → the declaration nesting.
3. The declaration is the only irreversible-feeling step and is not: restoring the flat
   `distribution` block restores every reader, in-repo and forge-side, with no data
   migration — the declaration describes rules, it stores no results.
4. **No on-disk record is rewritten by a revert.** `governs_the_ceilings_record` means a
   subset search never wrote the canonical path, so no existing `ceilings.json` /
   `ceilings.pilot.json` is at risk from this change or from undoing it. Per-shard
   artifacts written under the new path are additive files, deletable without touching the
   canonical record.
5. If only the assembly is reverted and the declaration kept, `run_search`'s
   `7fcc150` refusal (harness.py:2494-2518) is the backstop: a slice still cannot be handed
   back as the record. That is the safe partial state.

## Dependencies

- `7fcc150` (piece 2) must be present. Verified: `run_search(transfers=...)` exists at
  harness.py:2433 and the partial refusal at :2494-2518.
- Test command is `PYTHONPATH=src .venv/bin/python -m pytest -q`; a bare `python3.12`
  cannot import the package.
- `strict_tdd: true` (inherited, provenance in `openspec/config.yaml:17-24`). Every piece
  here is red-first, and three of the reds are already committed.
- **No blocking dependency remains.** The forge-side declaration reader was the one open
  question and the operator resolved it on 2026-09-28 (see Approach). Spec and design may
  proceed on all three pieces, including the declaration shape.
- **No forge change is required or permitted by this change.** Every edit stays inside
  `implementations/Domain_Adaptation`, which is exactly `allowedEditRoots`.

## Success Criteria

- [ ] `test_the_declaration_names_the_axis_the_search_can_split_on` passes, **and** a
      grown assertion proves a search shard resolves rules that differ from a campaign
      shard's for `ceilings`/`ceilingsByTransfer`/`hyperByTransfer`
- [ ] `distribution["axis"] == "seed"` still holds, and the forge's arm refusal is still
      reachable: `dist.get("axis") == "arm"` remains a question with a real answer rather
      than one that reads `None`
- [ ] A test proves the four flat top-level keys **are the same objects** (`is`) as the
      seed axis's nested groups — not merely equal to them. An `==` assertion here does not
      satisfy this criterion
- [ ] `expected` is derived from `config.SEARCH_TRANSFERS` at every site that needs it, and
      no site derives it from `VERDICT_TRANSFERS`
- [ ] `test_a_unit_resolves_to_the_axis_the_step_declares` passes, with the three-way
      absent/empty/populated distinction preserved and measured
- [ ] `test_partial_search_records_merge_instead_of_overwriting` passes, and the refusal is
      measured **in both directions**: an absent expected transfer refuses, a complete set
      assembles
- [ ] A subset-transfer `run_search` leaves a readable per-shard artifact where it
      previously left nothing
- [ ] Six per-shard artifacts assemble into a record byte-comparable in shape to a
      whole-search record (all fields listed for a real `ceilings.pilot.json` entry:
      `ceiling`, the seven other searched dimensions, `criterion`, `role`, `epochs`,
      `trials`, `search`, `decidedByFlatRule`, `plateau`, `noise`, `labelNoise`,
      `atRequiredScale`, `requiredScale`, `seconds`, `transfers`, `neutral`, `byTransfer`,
      `perTransfer`, `inheritanceRule`, `env`, `environment`, `revision`)
- [ ] `test_every_writer_of_that_record_is_guarded` still passes with the new writer in the
      module
- [ ] `test_the_search_still_has_no_seed_axis_to_split` and
      `tests/test_distribute.py:44-52` (arm split forbidden) still pass
- [ ] No `distribution` reader resolves a group to `None` or `[]` by accident —
      `tools/distribute.py:208` in particular produces a real axis answer in the plan JSON
- [ ] Full suite green: `PYTHONPATH=src .venv/bin/python -m pytest -q`, with the three
      formerly-red tests now counted as passing rather than expected-red
- [ ] The measured projection is recorded: whole-search 5.6–6.2 h vs ~1.2 h per machine
      across six, stated as a projection until a distributed run measures it
