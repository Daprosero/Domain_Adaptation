# Design: A transfer that never came back is not one nobody searched

## Technical Approach

Three capabilities, one `src/` diff, and the smallest possible blast radius at every
seam. The strategy is to add one new *reader* (`shards.rules_for_axis`) and one new
*writer/assembler* pair (`harness` slice write + `harness.merge_search_records`), and to
change existing call sites by **handing them a different mapping** rather than by editing
what they read from it.

The load-bearing observations behind the whole design, all measured:

1. `shards.partition(dimensions, dist)`, `shards.merge(..., dist=...)` and
   `tools/bridge.build_summary(found, dist=...)` already take the declaration *as an
   argument*. So an axis can be selected by passing a sub-block where a whole declaration
   used to go. Every flat read below that line stays byte-identical.
2. `shard_paths(shard, ...)` (harness.py:1087-1117) already routes a named shard to a home
   under `results_for(noise, kind, pilot)/shards/<id>` — which for the clean full run is
   exactly `distribution.shardsRoot`. The per-shard destination the proposal left open is
   therefore not a new concept, it is a fourth key in an existing mapping.
3. `search_ceilings_trials` — the live engine (`config.SEARCH_ENGINE == "optuna"`) —
   already *accepts* `shard` and **never uses it**. The slice write is the first use, which
   is why it costs nothing to place there.
4. `ceiling_record.choose(trials, noise)` is pure and already pools per-transfer picks into
   the family's pooled winner (harness.py:1752-1755). The assembly re-pools with the
   identical expression over the union, so no pooling rule is written twice.
5. `shard_io.disagreements(shards, fields)` compares `entry["stamp"].get(field)` with
   `json.dumps(..., sort_keys=True)`. Feeding it `{"shard": id, "stamp": family_entry}` is
   a real reuse — dict-valued fields (`search`, `requiredScale`) compare correctly — with
   no comparison logic duplicated.

Maps to the proposal's Approach: Decision A (nest per axis, flat keys as an identity
projection) lands in `__init__.py` plus `shards.rules_for_axis`; the unit reading lands in
`config.execution_units_for`; Decision B (the merge refuses) lands in
`harness.merge_search_records`. Requirements are in `specs/search-distribution-axis`,
`specs/run-unit-axis-resolution` and `specs/partial-search-record-assembly`.

---

## Architecture Decisions

### D1 — The transfer-axis unit vocabulary is the canonical label (`"M->U"`), never a positional index

**This settles Open Design Question 1.**

**Choice**: `FORGE_RUN_UNITS` for `axis="transfer"` carries the exact strings
`transfer_label()` produces — `"M->U"`, `"S->M"`, … — and `execution_units_for("transfer")`
resolves each against `{transfer_label(t): t for t in config.SEARCH_TRANSFERS}`, returning
`(str, str)` tuples and **refusing** any string not in that map.

**Alternatives considered**: a positional index into the declared transfer list (`"0"`…
`"5"`), which the committed red test names as a real possibility; an opaque id minted by a
planner.

**Rationale**, from the code:

- **What a wrong reading costs, and who can catch it.** A label is validated against a
  *closed set*: `"M->U"` either is in the map or it is not, and the refusal can name both
  what arrived and what was available. An index can only be range-checked, and `"0"` is in
  range for every non-empty list — so a stale index from a re-ordered plan passes every
  check that could exist and searches a different transfer. The number it returns has the
  correct shape (`harness.transfer_label`'s own docstring: "the failure that reads as
  success").
- **Stability across a reordering of the declared list.** `SEARCH_TRANSFERS` has already
  changed membership: `search_ceilings_trials`' own docstring records "la rejilla medía dos
  y las otras cuatro heredaban" (harness.py:1615-1617), and the constant is now
  `list(TRANSFERS)` (config.py:673). An index survives such a change by silently meaning a
  different transfer; a label either resolves or refuses. `governs_the_ceilings_record`
  already compares *ordered label lists* (harness.py:1549-1550), so labels are the
  vocabulary the existing guard already speaks.
- **A human reading a submission's units.** `FORGE_RUN_UNITS=["M->U"]` says what was sent.
  `["0"]` does not, and it is *indistinguishable on sight from a seed unit* — the exact
  hazard the red test names (`"'0' is a valid seed and could equally be a transfer index"`).
- **The decisive argument, and it is about an existing refusal.** Under the label
  vocabulary a transfer unit is **self-identifying across axes**: `int("M->U")` raises, so
  the axis-confusion case is caught by `execution_seed_units()`'s *already committed*
  refusal (config.py:287-294) without one new line. Under the index vocabulary it is
  caught by nothing. This is not hypothetical:
  `MIL-CREDA/Notebooks/Benchmark_Ceiling_Search.ipynb:300-302` calls
  `config.execution_seed_units()`, derives `SHARD = "seed-<n>"`, and calls
  `harness.run_search(pilot=…, shard=SHARD)` **without `transfers=`**. Feed that
  un-updated notebook six index units and all six machines read `"0".."5"` as seeds, name
  themselves `seed-0`…`seed-5`, and each run the **whole** six-transfer search —
  `governs_the_ceilings_record(0.0, None)` is `True` for all six, so all six write
  `config.ceilings_record_for(pilot)` and clobber each other, at ~6 h of metered quota
  apiece. Feed it six *labels* and it dies immediately on `int("M->U")`. The vocabulary
  choice is the difference between a loud refusal and six clobbering full searches.
- The axis is still **received and never inferred** (D3). D1 does not relax that; it makes
  a *misdeclared* axis detectable instead of silent.

**Rejected, and why**: the positional index buys shortness and avoids `>` in an
environment value. Shortness is worthless here, and the `>` concern is addressed in the
External Input Boundary section below — the value is JSON-serialised into the process
environment, and the one place it could become a path (a shard id) is deliberately *not*
derived from the label (see D5).

### D2 — The per-shard search artifact is a fourth key of `shard_paths`, named `ceilings.slice.json`

**This settles Open Design Question 2.**

**Choice**: `harness.SEARCH_SLICE_SUFFIX = ".slice.json"`, and `shard_paths` gains a
`"searchSlice"` key:

| `shard` | `searchSlice` path |
|---|---|
| `"s03"` | `results_for(noise, kind, pilot)/shards/s03/ceilings.slice.json` |
| `None` | `<canonical record>.with_name(stem + ".slice.json")` → `ceilings.slice.json` / `ceilings.pilot.slice.json` beside the canonical record |

**Alternatives considered**: (a) `search_ceilings`' existing scratch/`partial` file; (b) a
path composed from `distribution.shardsRoot`; (c) a new top-level results directory; (d) a
new `kind` coordinate in `results_for`.

**Rationale**:

- **(a) is the wrong artifact, and the proposal's citation of it needs a correction.** The
  proposal names "`search_ceilings_trials`' existing scratch/`partial` file
  (harness.py:2091-2093)". Measured: lines 2091-2093 are inside **`search_ceilings`'s grid
  engine body**, not `search_ceilings_trials`, and `search_ceilings_trials` — the live
  engine — has **no partial file at all** (no `shard_paths` call between :1598 and :1836).
  Beyond the misattribution, `ceilings.partial.json` is a *mid-measurement resume scratch*,
  written per cell by `config._write_partial` and **deleted the moment the answer exists**
  (`partial.unlink(missing_ok=True)`). A finished slice that the assembly must read later
  has the opposite lifecycle. Reusing that file would give one path two lifecycles, and the
  loser would be the slice.
- **(b) is the same place, reached the wrong way.** `shardsRoot` is
  `"MIL-CREDA/Results/Benchmark/shards"`, which is exactly
  `results_for(0.0, "campaign", False)/SHARDS_DIR` — the value `shard_paths` already
  computes. It is a *forge-read string* consumed by `gate`/`close`/`discuss`/`probe`, which
  do not exist in this repository (`__init__.py:498-506`), and the proposal forbids moving
  it. Composing a path from it would be a second spelling of a location `shard_paths`
  already owns, and it would lose the `pilot`/`noise` coordinates for free: `shard_paths`
  routes a pilot slice to `Results/Pilot/Benchmark/shards/…` and a contaminated one to the
  `Noise/rho…` tree, which a literal `shardsRoot` cannot.
- **(c)/(d) cost more than they buy.** `results_for`/`models_for` both validate
  `kind in ("campaign", "curve")` and raise otherwise; a third kind touches both functions
  and every fixture that names a tree.
- **The roster test is satisfied by construction.** `tests/test_label_noise.py:1253-1274`
  scans each `harness` function's own source for `ceilings_record_for` **and**
  `write_text`. The slice writer reaches its path through `shard_paths(...)`, so
  `ceilings_record_for` does not appear in its source — it is not a writer of the canonical
  record and correctly does not enter the roster. `merge_search_records` *does* write the
  canonical path, does enter the roster, and does consult the guard (D7).
- **Filename**: `.slice.json` rather than `.shard.json` because `home/shard.json` is
  already the environment stamp; two files one character apart in the same directory is a
  reading error waiting to happen. `.slice.json` also says what it is: a finished part, not
  a partial measurement.

### D3 — `execution_units_for(axis)` has no default axis, and validates the axis against the declaration

**Choice**:

```python
def execution_units_for(axis: str):
    """`RUN_UNITS_ENV` for THIS submission, in the vocabulary of the axis the
    STEP declares -- received, never inferred from the content."""
```

No default for `axis`. The axis is checked against
`__benchmark__["distribution"]["axes"]` and an unknown axis **refuses**.
`execution_seed_units()` survives as `return execution_units_for("seed")`.

**Alternatives considered**: `axis="seed"` as a default; inferring the axis from whether
every unit parses as an integer; a separate `execution_transfer_units()`.

**Rationale**: a default of `"seed"` recreates the exact guess the function exists to
remove — a caller that forgets the argument gets the old behaviour silently. Inference is
refused by the proposal and by the red test's own words. A separate function per axis puts
the three-way absent/empty/populated distinction in two places, and that distinction is
the part that must survive verbatim (config.py:246-262 and its three committed tests at
`tests/test_scale_readings.py:1098+`). Validating the axis against the *declaration* rather
than a local tuple is what makes "the declaration names the axes" load-bearing instead of
decorative, and it means a step that names an axis nobody declared cannot fall through to
seeds.

**Layering**: `config.py` imports nothing from the package (verified) and must stay that
way — `harness` imports `config`. So the canonical label formatter moves **down** to
`config.transfer_label`, and `harness.transfer_label` becomes a re-export
(`transfer_label = config.transfer_label`), the precedent `shards.disagreements =
_shard_io.disagreements` already sets ("re-exported as-is, with no wrapper and no logic
duplicated"). This keeps the "one spelling" rule
(`harness.transfer_label`'s docstring, :447-455) literally true: one definition, one
format string. All 23 in-repo uses are plain calls (verified), so an alias is safe.

### D4 — One axis reader, returning a declaration-shaped mapping

**Choice**: `shards.rules_for_axis(axis, dist=None) -> dict`, returning the sub-block for
`axis`. Callers pass its result where they used to pass the whole declaration.

```python
def rules_for_axis(axis: str, dist: dict | None = None) -> dict:
    dist = dist if dist is not None else declaration()
    axes = dist.get("axes")
    if axes is None:
        if axis == dist.get("axis"):
            return dist          # the flat keys ARE this axis's rules (D6)
        raise ShardsDisagree(...)  # a flat declaration has no second axis
    block = axes.get(axis)
    if block is None:
        raise ShardsDisagree(...)  # never a silent default to the flat keys
    missing = [g for g in GROUPS if g not in block]
    if missing:
        raise ShardsDisagree(...)  # a missing group is refused, never read as []
    return block
```

**Alternatives considered**: returning the four lists as a tuple; teaching `partition()`
an `axis=` parameter; a per-group accessor (`poolable_for(axis)`).

**Rationale**: returning a *declaration-shaped mapping* is what makes the blast radius one
line per call site. `partition()`'s body is untouched. `tools/bridge.build_summary`'s five
flat reads (:167-168, :203, :216-217) need **zero** edits — only its `dist` resolution at
:166 changes. The flat-`dist` fallback is exact rather than lenient: by D6 the flat keys
*are* the seed axis's groups, so reading them as `axis == dist["axis"]` is identity, not a
guess — and it is what keeps every hand-built flat fixture
(`test_bridge.py:190-210`, `test_benchmark_declarations.py:160-227`) green without an edit.
A missing group **refuses** rather than resolving to `[]`: a group read as empty would make
`partition()` report every dimension `unpartitioned` and a merge drop them silently.

### D5 — Search shard ids keep the existing `s{index:02d}` namespace and are never derived from a transfer label

**Choice**: a search shard's id is minted by whoever plans the split, in the same
`s00`-style namespace `tools/distribute.plan()` already mints, with a distinguishing prefix
(`search-s00`) so it cannot collide with a campaign shard's home. The slice itself declares
which transfers it covered; the assembly reads that field and never parses an id.

**Alternatives considered**: deriving the id from the transfer label (`shards/M->U/`).

**Rationale**: `PlanConflict`'s own docstring states the hazard — "`shard_paths()` keys a
shard's entire on-disk home on its id string alone, and a relaunch that recomputes ids
positionally over a failed subset lands back on `s00`". A label-derived id would also make
the transfer label a *filesystem path component*: `M->U` is legal on POSIX but `>` is
illegal in a Windows filename, so the one place the label could become a portability
problem is the one place we refuse to put it. Keeping ids opaque and coverage declared
inside the slice also means a re-plan that renumbers shards cannot silently change what a
slice claims to hold.

### D6 — The flat keys are projected by one `dict.update`, after the literal

**Choice**:

```python
# The flat top-level keys the forge's declaration reader requires
# (`implementation_engine.py:1326-1400`), bound to the SAME list objects as the
# seed axis's groups -- a projection, never a copy.
__benchmark__["distribution"].update(__benchmark__["distribution"]["axes"]["seed"])
```

**Alternatives considered**: module-level `_SEED_POOLABLE = [...]` helpers referenced from
both places; writing the lists twice; a `__getitem__`-overriding mapping class.

**Rationale**: `dict.update` inserts the *same list objects*, so
`distribution["poolable"] is distribution["axes"]["seed"]["poolable"]` holds — which is
exactly what the spec's identity requirement and the proposal's "identity, not equality"
demand. It introduces no new module-level names into `__init__.py` (a declaration module
whose contents are themselves asserted), it is one line, and it is self-maintaining: a
fifth group added to the seed axis projects automatically. A subclassed mapping would be
invisible to `json.dumps` and to the forge's reader. Two named constants would work but put
the projection in four places instead of one.

**Consequence to record**: the four flat keys are appended *after* `axis`, `axes` and
`shardsRoot` in iteration order. Nothing in-repo reads declaration key order; a golden that
does must be regenerated rather than the projection reshaped.

### D7 — The slice write is the `else` of `governs_the_ceilings_record`

**Choice**: in `search_ceilings_trials` (and symmetrically in `search_ceilings`' grid
body), the existing guarded write becomes an `if/else`:

```python
if governs_the_ceilings_record(noise, transfers):
    record = config.ceilings_record_for(pilot)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(sellar_techos(found, reduction), indent=2),
                      encoding="utf-8")
else:
    slice_path = shard_paths(shard, noise=reduction.labelNoise,
                             kind=reduction.kind,
                             pilot=reduction.pilot)["searchSlice"]
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    slice_path.write_text(json.dumps(
        sellar_rebanada(found, reduction, shard, transfers), indent=2),
        encoding="utf-8")
```

**Alternatives considered**: gating the slice write on `transfers is not None`; writing
both files always.

**Rationale**: the two gates are **behaviourally identical for every live caller** —
measured: `ceilings_in_force` (harness.py:1465) is the only caller of `search_ceilings`
anywhere in `src/`, and it never passes `noise=`, so `noise` is always `0.0` in-repo. The
`else` form is preferred because it states the invariant the design wants: **a search
writes exactly one record, and the same guard decides which one.** It also closes the noise
corner for free — a future contaminated subset search leaves a readable slice instead of
nothing, quarantined from the canonical path by the same guard it failed. Writing both would
create two sources of truth for one search.

The slice is routed with `noise=reduction.labelNoise, kind=reduction.kind`, exactly as
the grid engine already routes its partial (harness.py:1893-1894), so slices at different
noise levels or run kinds never share a path.

**Step declarations are unaffected**: `search-pilot` runs whole and clean, so
`governs_the_ceilings_record` is `True` and it writes only
`config.ceilings_record_for(True)` — the single output
`tests/test_steps.py:718` already declares. No new `produces` entry, no `foreign` file on
the pilot walk.

### D8 — The slice is a self-describing record; `sellar_rebanada` seals it and `sellar_techos` is not reused verbatim

**Choice**: the slice holds the same family→entry map a whole search produces, plus, per
family entry:

| field | value | why |
|---|---|---|
| `transfers` | the labels **this slice** covered | the assembly's only source for coverage; never an id parse (D5) |
| `env` / `environment` | this worker's handle and full stamp | honest: this machine measured these transfers |
| `revision` etc. | `ceiling_record.stamp(entry)` | so a drifted slice is refusable before it is merged |
| `shard` | the slice's own id | the short handle, the same shape `runs.jsonl` uses (harness.py:1165-1169) |

`sellar_techos(found, reduction)` is used **unchanged** for the slice, plus the two extra
fields — the slice genuinely was measured on one machine, which is the claim
`sellar_techos` makes and its docstring already anticipates ("las búsquedas de las dos
familias pueden repartirse entre workers distintos").

What `sellar_techos` **cannot** honestly seal is the *assembled* record: six slices carry
six `env` handles and a family entry has one. So the assembly seals differently (D9).

### D9 — The assembly re-pools rather than merging dicts, and records its own provenance under one new `assembly` key

**Choice**: `merge_search_records` does **not** dict-update slices together. For each
family it:

1. takes the **union** of the per-unit fields the declaration names
   (`transfers`, `byTransfer`, `perTransfer`), refusing on a duplicate label;
2. **recomputes** every pooled field with the identical expression the engine uses:

```python
agrupado = ceiling_record.choose(
    [{"ceiling": d["ceiling"], "value": d["best"],
      **{dim: d[dim] for dim in HYPER_DIMS}}
     for d in per_transfer.values()], ruido)
```

3. carries the `identicalAcrossShards` fields through after proving they agree;
4. sums the `perRun` field (`seconds`) as total compute;
5. sets `env`/`environment` to the assembling machine's — where the *pooling decision* was
   made — and adds one new key, `assembly`, holding
   `{"shards": [{"id", "env", "environment", "seconds", "transfers"}, …], "expected": [labels]}`.

**Alternatives considered**: copying one slice's pooled fields; taking the pooled winner
from whichever slice arrived first; sealing the assembled entry with `sellar_techos` alone;
duplicating each transfer's `env` inside `perTransfer[label]`.

**Rationale**:

- **Re-pooling is not optional.** `entry["ceiling"]` is the pooled winner over the family's
  per-transfer plateau. A slice that searched one transfer computed that "pooled" winner
  over **one** value. Copying it forward would put a one-transfer answer in the field
  `ceiling_for` falls back to, which is the exact class of defect this change is named
  after. Reusing `ceiling_record.choose` verbatim — including `d["best"]` as `value`, as
  the engine does — means the assembled pooled winner is computed by the same rule as an
  unsplit search's, not by a second implementation.
- **`env` must not silently name one machine.** `sellar_techos` exists so "un lector
  posterior no podría distinguir un registro buscado acá de uno buscado allá". An assembled
  record sealed with one handle would re-create precisely that confusion. The `assembly`
  key's **presence is the flag**: a reader holding the file alone can tell an assembled
  record from a searched one, and `assembly.shards` says where each transfer was measured.
  A test must assert the key is present, so it cannot be dropped later.
- **One place per fact.** `perTransfer[label]` gets no `env`; the join is
  `assembly.shards[*].transfers`. Writing the same origin twice is what
  `write_shard_stamp`'s own comment calls "un hecho escrito mil veces es uno que puede
  discrepar consigo mismo".
- **`seconds` is total compute, and that is what it is used for.** `search_ceilings_trials`'
  own docstring (:1627-1644) says `seconds` exists so cost can be projected from
  `epochs`/`trials`/`requiredScale`. Total compute is the right number for that arithmetic;
  the per-shard readings stay in `assembly.shards[*].seconds`, so a reader who wants wall
  clock has it and is never handed a sum disguised as one.
- **An entry carrying both `search` and `grid` is refused for free.**
  `ceiling_record.kind_of` already raises on that shape with the message "esto es una
  fusión". The assembly calls it per slice, so a grid-engine slice and a trials slice can
  never be fused.

### D10 — What the assembly refuses, and with which exception

**Choice**: `harness.merge_search_records(slices, expected=None, pilot=False, reduction=None)`
refuses, in this order, before writing anything:

| # | condition | exception | message names |
|---|---|---|---|
| 1 | a slice carries no `transfers` field, or an empty one | `shards.ShardIncomplete` | the slice id |
| 2 | `ceiling_record.stamp_drift(entry)` is non-empty for any slice | `shards.ShardsDisagree` | slice id, field, record vs current |
| 3 | `ceiling_record.kind_of(entry)` differs across slices | `shards.ShardsDisagree` | the two kinds |
| 4 | two slices claim the same transfer label | `shards.ShardsDisagree` | the label and both slice ids |
| 5 | slices disagree on an `identicalAcrossShards` field for the transfer axis | `shards.ShardsDisagree` | field and conflicting values (via `shard_io.disagreements`) |
| 6 | a label in `expected` has no slice | `shards.ShardIncomplete` | **every missing label by name**, not a count |
| 7 | a slice covers a label not in `expected` | `shards.ShardsDisagree` | the unexpected label |
| 8 | the families present differ across slices | `shards.ShardsDisagree` | the family sets |

`expected` defaults to `list(config.SEARCH_TRANSFERS)` and is **normalised into
`SEARCH_TRANSFERS` order** before the guard is consulted.

**Alternatives considered**: an `int` count like `shards.merge(expected=…)`; a new
exception class; a separate `validate_search_slices` checker above the assembly.

**Rationale**:

- **A count cannot answer the question.** `shards.merge` takes `expected: int | None` and
  computes `"missing": max(0, (expected or len(arrived)) - len(arrived))` (shards.py:470).
  That arithmetic cannot distinguish "all of them arrived" from "the ones that were sent
  arrived" — the red test's exact words. `expected` here is a **set of transfers**, and
  refusal #7 is why it must be a set in both directions: an *unexpected* transfer is as
  informative as a missing one.
- **Refusal #6 is the answer to the defect the change is named after** (see "The dead-shard
  distinction" below), and it must name the labels: "whoever reads the refusal knows what
  to relaunch".
- **No new exception class.** `ShardsDisagree`/`ShardIncomplete` already carry exactly
  these two meanings and are already `SystemExit` subclasses raised by the merge itself —
  Decision B's stated rationale. A third class would be a third spelling of the same two
  facts.
- **No separate checker.** Decision B settled that; the assembler is the only party holding
  `expected`.
- **`shard_io.disagreements` is reused, not reimplemented**, by shaping each family's
  slices as `[{"shard": id, "stamp": entry}, …]`. `shards.merge()` itself is **not**
  reused: it is keyed on `(transfer, arm)` run cells and `runs.jsonl` lines
  (shards.py:339-343, 402-484) and a search slice has neither.

**Reused / written fresh, stated plainly:**

| Reused verbatim | Written fresh |
|---|---|
| `ceiling_record.choose`, `plateau`, `stamp`, `stamp_drift`, `kind_of` | `merge_search_records` (the assembly) |
| `shard_io.disagreements` (via a `{"shard","stamp"}` adapter) | `read_search_slices` (the I/O half) |
| `shards.ShardsDisagree`, `shards.ShardIncomplete` | `sellar_rebanada` (slice sealing) |
| `harness.governs_the_ceilings_record`, `sellar_techos`, `shard_paths` | `shards.rules_for_axis`, `config.execution_units_for` |
| `config.SEARCH_TRANSFERS` as the one expected set | the transfer axis's declaration sub-block |

### D11 — The expected set is `config.SEARCH_TRANSFERS`, and the `run_search` message is aligned to it

**Choice**: every site that needs "expected" reads `config.SEARCH_TRANSFERS`. `run_search`'s
refusal message is repointed from `config.VERDICT_TRANSFERS` (harness.py:2509) to
`config.SEARCH_TRANSFERS`.

**Rationale**: `governs_the_ceilings_record` — the guard on the only write to the canonical
path — already compares against `SEARCH_TRANSFERS` (harness.py:1549-1550). Deriving
completeness from the same constant makes the write guard and the completeness test agree
**by construction rather than by coincidence**. Completeness of a *search* record is a
question about what the search covers, which is what `SEARCH_TRANSFERS` names.

The divergence is a live mechanism, not a hypothetical: `SEARCH_TRANSFERS = list(TRANSFERS)`
is a **copy** (config.py:673) while `VERDICT_TRANSFERS = TRANSFERS` is an **alias**
(config.py:844). Mutating `TRANSFERS` moves one and not the other, today.

`test_a_partial_search_refuses_to_present_its_slice_as_the_record` asserts
`str(len(config.VERDICT_TRANSFERS))` appears in the message; both lists have length six, so
the repoint keeps it green **without editing the test** — which is required.

### D12 — `run_search`'s subset branch asks the slice, not the canonical record, whether the work was already done

**Choice**: `run_search` gains an explicit subset branch, placed **before** the
`record is None` check, and `ceilings_in_force`'s "already answered" predicate becomes
slice-aware when `transfers is not None`.

**This fixes a defect the committed green test cannot see, and it is required for the
capability to work at all.** Traced:

- **Case A — no canonical record on disk** (the live state: `MIL-CREDA/Results/Benchmark/`
  holds only `ceilings.pilot.json`). `ceilings_in_force` sees no record, runs the subset
  search (~1.2 h of metered quota), the guard returns `False`, nothing is written; back in
  `run_search`, `search_record(pilot)` is `None`, so the **"the search ran and left no
  record"** `SystemExit` at :2487 fires — *not* the PARTIAL one. The slice's work
  evaporates and the message misdescribes what happened.
- **Case B — a canonical record exists.** `ceilings_in_force` short-circuits at :1463 and
  **never searches**. `run_search` then raises PARTIAL, whose text claims "the search ran
  for N of 6 transfers and left a PARTIAL record at `<canonical>`" — false in both clauses.
  Once the assembly starts writing `ceilings.json`, this is exactly the **relaunch of a
  dead shard**: it would write no new slice and report a partial that never ran.

So the PARTIAL refusal is reachable today only when no search ran, and the committed test
passes because it monkeypatches both `ceilings_in_force` and `search_record`.

**Shape**:

```python
ceilings_in_force(reduction, device, shard=shard, pilot=pilot, transfers=transfers)
if transfers is not None:
    slice_path = shard_paths(shard, pilot=pilot)["searchSlice"]
    raise SystemExit(
        f"the search ran for {len(transfers)} of "
        f"{len(config.SEARCH_TRANSFERS)} transfers and left a PARTIAL record, "
        "which is not the record the campaign runs at.\n"
        f"  This slice's own artifact: {slice_path} "
        f"({'written' if slice_path.exists() else 'ABSENT -- this slice produced nothing to assemble'})\n"
        f"  Asked for: {[transfer_label(t) for t in transfers]}\n"
        "  Assemble every slice with `harness.merge_search_records(...)`; it "
        "refuses until every transfer in `config.SEARCH_TRANSFERS` has one."
    )
record = search_record(pilot=pilot)
if record is None:
    raise SystemExit(...)   # unchanged
return record
```

and inside `ceilings_in_force`:

```python
already = (shard_paths(shard, pilot=pilot)["searchSlice"].exists()
           if transfers is not None else search_record(pilot=pilot) is not None)
if not already:
    search_ceilings(...)
```

**Alternatives considered**: calling `search_ceilings` directly from `run_search`'s subset
branch, bypassing `ceilings_in_force`.

**Rejected**: `test_a_partial_search_refuses_to_present_its_slice_as_the_record`
monkeypatches `ceilings_in_force`, `search_record` and `resolve_device` — and *not*
`search_ceilings`. A direct call would make that test run a real optuna search. The test
must stay green on its own terms, so the subset path stays behind
`ceilings_in_force`, and the branch below it is reordered so PARTIAL wins over "left no
record". The message keeps `"PARTIAL"` and `str(6)`, so the test passes unedited; the false
"at `<canonical>`" clause is dropped, which the proposal explicitly permits (:2494-2518 is
in scope).

**`ceilings_in_force`'s `transfers` parameter stays** and is now genuinely load-bearing
rather than merely forwarded.

### D13 — The dead-shard distinction lives in the assembly, at write time, and nowhere else

**Choice**: `ceiling_for` and `hyper_for` are **untouched** — no new parameter, no new
refusal, no change to the pooled-winner fallback. The distinction is made and recorded in
one place:

1. **Made** by `merge_search_records`: it is the only party holding both `expected` (from
   `config.SEARCH_TRANSFERS`) and the set of slices that arrived. `expected − arrived` is
   *exactly* "the shards that never came back", and refusal #6 names them. A record
   claiming six transfers while holding five therefore never comes into existence.
2. **Recorded** in the record it writes: `entry["transfers"]` names the covered set, and
   because the assembly writes only a record whose `transfers` equals both what arrived and
   what was expected, that field is a true statement about coverage. `ceiling_for`'s pooled
   branch is then reachable only for a transfer *outside* `transfers` — which is the
   legitimate out-of-sample case its docstring (:462-466) declares.

**Alternatives considered and rejected**:

- **A `strict=`/`expected=` parameter on `ceiling_for`.** Rejected on blast radius, which
  is the widest of any option: `ceiling_for` is called once per `run_one`, i.e. across
  5 arms × 6 transfers × 30 seeds for a campaign *and* once per optuna trial inside the
  search itself.
- **Raising inside `ceiling_for` when a transfer is absent.** Rejected because it would
  break the legitimate out-of-sample reading the function is documented to provide, and
  would fire inside the training loop where a refusal costs a truncated run.
- **Annotating coverage in `search_record()` and refusing in `ceilings_in_force()`.** This
  is technically the most attractive of the three: it reuses the *exact* existing mechanism
  (`search_record` already tags each entry `currentStamp`, and `ceilings_in_force` already
  refuses on that tag — harness.py:1341-1352, 1468-1480), costs nothing at the read site,
  fires once per campaign before any `runs.jsonl` is truncated, and would also catch a
  record that never went through the assembly at all — hand-edited, half-uploaded, or
  written by a future fourth writer. **Rejected as out of scope**, on two grounds. First,
  the proposal settles the question: "the refusal is the *only* thing standing between a
  dead shard and a plausible number — the read-site fallback is deliberately kept." Second,
  it has a measured cost this change has no reason to pay: the rule "a field absent IS
  drift" (`ceiling_record.stamp_drift`'s own precedent) would make every fixture that
  writes a record without a `transfers` key refuse — `tests/test_search_records.py:34-46`
  is one, and its own comment already says such a refusal would reject the fixture "por un
  motivo ajeno a lo que estos tests miden".

  **Recorded here so a later session can take it on cost alone**, the same way the
  proposal records the forge-side axis vetting: the implementation is
  `entry["coverageComplete"] = bool(claimed) and sorted(claimed) == sorted(entry.get("byTransfer") or {})`
  in `search_record`, a refusal on `False` in `ceilings_in_force`, and one added
  `"transfers"` line in each affected fixture. Verified harmless against the live
  `MIL-CREDA/Results/Benchmark/ceilings.pilot.json`, which claims six and holds six.

**The residual, stated**: a canonical record produced by something other than the assembly
is not covered by this change. Nothing in the repository produces one today, and
`governs_the_ceilings_record` gates every writer of that path.

---

## Data Flow

```
                        one worker per transfer
   FORGE_RUN_UNITS=["M->U"]
        |
        v
   config.execution_units_for("transfer")          <-- axis RECEIVED (D3)
        |  refuses a label not in SEARCH_TRANSFERS
        v  [("M","U")]
   harness.run_search(shard="search-s00", transfers=[("M","U")])
        |
        v
   ceilings_in_force ---> "has MY slice already answered?" (D12)
        |  no
        v
   search_ceilings -> search_ceilings_trials
        |
        |  governs_the_ceilings_record(0.0, [("M","U")]) -> False
        v
   shards/search-s00/ceilings.slice.json           <-- THE NEW ARTIFACT (D2, D7)
        |
        |  run_search then REFUSES: a slice is not the record (D12)
        v
   ......................  six workers, six slices  ......................
        |
        v
   harness.read_search_slices(pilot=False)         <-- I/O half
        |  [slice0 .. slice5]
        v
   harness.merge_search_records(slices, expected=SEARCH_TRANSFERS)
        |
        |  refusals 1-8 (D10)   <---- expected - arrived = the dead shards
        |  ceiling_record.choose over the UNION (D9)
        |
        v
   governs_the_ceilings_record(0.0, SEARCH_TRANSFERS) -> True
        |
        v
   config.ceilings_record_for(pilot)               <-- canonical, guarded (D7)
        |
        v
   ceilings_in_force -> with_ceilings_in_force -> campaign -> run_one
                                                        |
                                                        v
                                        ceiling_for / hyper_for  (UNCHANGED, D13)
```

### Sequence — the split search and its assembly

```
operator        worker_i            search_ceilings_trials      slice_i        assembly      canonical
   |               |                          |                    |              |             |
   |--plan split-->|                          |                    |              |             |
   |               |--execution_units_for---->|                    |              |             |
   |               |   ("transfer") -> [t_i]  |                    |              |             |
   |               |--run_search(transfers)-->|                    |              |             |
   |               |                          |--search t_i------->|              |             |
   |               |                          |  guard -> False    |              |             |
   |               |                          |--write slice------>|              |             |
   |               |<--SystemExit: PARTIAL ----|                    |              |             |
   |                                                               |              |             |
   |--read_search_slices()---------------------------------------->|              |             |
   |<--[slice_0 .. slice_5]----------------------------------------|              |             |
   |--merge_search_records(slices, expected)---------------------->|------------->|             |
   |                                                   refuse if expected-arrived |             |
   |                                                   re-pool via choose()       |             |
   |                                                   guard -> True              |             |
   |                                                                              |--write----->|
   |<--assembled record-----------------------------------------------------------|             |
```

### Sequence — a shard that never came back

```
operator                      assembly                          canonical
   |                             |                                  |
   |--merge(5 slices, expected=6)->|                                |
   |                             |  arrived = {M->U,U->M,M->S,S->M,U->S}
   |                             |  expected - arrived = {S->U}
   |<--ShardIncomplete: "no slice for S->U; relaunch it" -----------|  (nothing written)
   |                             |                                  |
```

---

## File Changes

| File | Action | Description |
|---|---|---|
| `src/MIL_CREDA_Benchmark/__init__.py` | Modify | `distribution` (481-507) gains `axes` with `seed` and `transfer` sub-blocks; the four flat keys become the one-line identity projection (D6); `shardsRoot` and `axis: "seed"` unchanged. The rationale block (392-480) is rewritten: the `axis`-is-singular justification is removed, the per-axis inversion and the projection's identity argument are written in its place |
| `src/MIL_CREDA_Benchmark/config.py` | Modify | New `transfer_label` (the single definition, moved down for layering, D3); new `execution_units_for(axis)`; `execution_seed_units()` becomes a one-line delegate and loses its singular-axis citation (237-239); `SEARCH_SLICE_SUFFIX` if the suffix constant lands here beside `PARTIAL_SUFFIX` (1080) |
| `src/MIL_CREDA_Benchmark/shards.py` | Modify | New `rules_for_axis` + `GROUPS` (D4); `merge()`'s `dist` default (421) resolves the seed axis; `partition()` body **unchanged** |
| `src/MIL_CREDA_Benchmark/harness.py` | Modify | `transfer_label` becomes a re-export of `config.transfer_label`; `shard_paths` gains `"searchSlice"` (D2); `search_ceilings_trials` and `search_ceilings`' grid body gain the `else` slice write (D7); new `sellar_rebanada`, `read_search_slices`, `merge_search_records` (D8-D10); `ceilings_in_force`'s slice-aware predicate and `run_search`'s reordered subset branch + repointed message (D11, D12); flat read at 2353 selects the seed axis; `run_campaign_shard` docstring's axis citation (2536-2538) corrected |
| `src/MIL_CREDA_Benchmark/pooling.py` | Modify | `per_run_dimensions()` at :32 selects the seed axis explicitly. Behaviourally identical today via the projection; the point is that the flat keys become forge-only |
| `tools/bridge.py` | Modify | **One line**: `dist` resolution at :166 becomes `shards.rules_for_axis("seed", dist)`. The five flat reads below (:167-168, :203, :216-217) are untouched. **Outside the digest boundary — no stamp cost, separately sliceable** |
| `tools/distribute.py` | **Unchanged** | `.get("axis")` at :208 still resolves to `"seed"`; `shard_seeds`/`plan` are campaign-only. Verified: no other flat-group read in this file |
| `tests/test_search_distribution.py` | Modify | Three reds turn green; test 1 grown past `axes` roster to the per-axis rule inversion; new identity (`is`) test; new tests for D1, D2, D7, D9, D10, D12 |
| `tests/test_benchmark_declarations.py`, `test_shards.py`, `test_bridge.py`, `test_label_noise.py`, `test_report_tables.py`, `test_distribute.py` | Modify | Only where a fixture asserts the `distribution` key roster. Flat-shaped `dist` fixtures (`test_bridge.py:190-210`, `test_benchmark_declarations.py:160-227`) stay green untouched via `rules_for_axis`' exact flat fallback (D4) — measure before editing |
| `MIL-CREDA/Notebooks/*` | Affected, not edited | No source change. Every notebook **stamp** is invalidated by the first `src/` byte. `Benchmark_Ceiling_Search.ipynb` still reads units as seeds and still calls `run_search` without `transfers=` — see Reachability below |

---

## Interfaces / Contracts

```python
# src/MIL_CREDA_Benchmark/__init__.py
"distribution": {
    "axis": "seed",            # untouched: the forge's one axis check reads this
    "axes": {
        "seed": {
            "poolable": ["sourceAccuracy", "targetAccuracy", "contribution",
                         "supervised", "adaptationShare", "parameters"],
            "perEnvironment": [],
            "perRun": [],
            "identicalAcrossShards": ["epochs", "ceilings", "ceilingsByTransfer",
                                      "hyperByTransfer", "labelNoise"],
        },
        "transfer": {
            # Re-derived by the assembly from the union of the slices' picks,
            # with `ceiling_record.choose` -- the engine's own pooling rule.
            "poolable": ["ceiling", "rampDelta", "rampDeltaLocal", "ceilingLocal",
                         "kernelSigma", "attentionGamma", "attentionTemperature",
                         "tauLocal"],
            # Declared and empty, the same convention and the same reason the seed
            # axis carries: no measurement has shown any part of a search's answer
            # to be a property of the machine that computed it.
            "perEnvironment": [],
            # One shard's own execution cost. Kept per slice in `assembly.shards`
            # and summed as total compute at the family level, never presented as
            # one wall clock.
            "perRun": ["seconds"],
            # THE INVERSION: `ceilings`/`ceilingsByTransfer`/`hyperByTransfer` are
            # absent here on purpose. Each search shard holds a DIFFERENT slice of
            # exactly those fields; telling them to agree is the opposite of their
            # job. What must agree is the backdrop the search ran under.
            "identicalAcrossShards": ["epochs", "labelNoise", "revision",
                                      "criterion", "role", "trials", "neutral",
                                      "requiredScale", "search", "arm",
                                      "flatRule", "atRequiredScale"],
            # Keyed by this axis's own unit, so the assembly takes the UNION and
            # refuses a duplicate. A fifth group: the four the forge requires are
            # the four above, and this one is invisible to it.
            "perUnit": ["transfers", "byTransfer", "perTransfer"],
        },
    },
    "shardsRoot": "MIL-CREDA/Results/Benchmark/shards",   # untouched, not moved
}
# The flat top-level keys the forge's reader requires, bound to the SAME list
# objects as the seed axis -- a projection, never a copy.
__benchmark__["distribution"].update(__benchmark__["distribution"]["axes"]["seed"])
```

```python
# src/MIL_CREDA_Benchmark/shards.py
GROUPS = ("poolable", "perEnvironment", "perRun", "identicalAcrossShards")

def rules_for_axis(axis: str, dist: dict | None = None) -> dict: ...
    # returns a declaration-SHAPED mapping, so every existing flat reader works
    # verbatim when handed it. Refuses an undeclared axis and a missing group.
```

```python
# src/MIL_CREDA_Benchmark/config.py
UNIT_AXES_FROM = "distribution.axes"   # the axis roster is READ, never listed here

def transfer_label(transfer: tuple[str, str]) -> str: ...   # the ONE spelling

def execution_units_for(axis: str) -> "list[int] | list[tuple[str, str]] | None":
    """Absent -> None. `"[]"` -> []. Populated -> validated in THIS axis's
    vocabulary. No default axis: a caller that does not name one refuses."""

def execution_seed_units() -> "list[int] | None":
    return execution_units_for("seed")   # three-way distinction preserved verbatim
```

```python
# src/MIL_CREDA_Benchmark/harness.py
SEARCH_SLICE_SUFFIX = ".slice.json"

def shard_paths(shard, pilot=False, noise=0.0, kind="campaign") -> dict:
    # ... + "searchSlice": home / f"ceilings{SEARCH_SLICE_SUFFIX}"
    #     (shard is None -> record.with_name(record.stem + SEARCH_SLICE_SUFFIX))

def sellar_rebanada(found: dict, reduction: Reduction, shard: str | None,
                    transfers: list) -> dict: ...

def read_search_slices(pilot: bool = False, noise: float = 0.0,
                       kind: str = "campaign",
                       root: "Path | None" = None) -> list[dict]: ...
    # the I/O half, the same split `shards.read_shards()` / `shards.merge()` has

def merge_search_records(slices: list[dict], expected: list | None = None,
                         pilot: bool = False,
                         reduction: Reduction | None = None) -> dict:
    """The one record six slices make, or a refusal naming what never came back.

    `expected` defaults to `config.SEARCH_TRANSFERS` and nothing else. Writes the
    canonical path only behind `governs_the_ceilings_record`, which returns True
    for a clean, complete assembly by construction -- and False for a deliberately
    smaller explicit `expected`, so a diagnostic assembly is quarantined from the
    campaign's record by the same guard that quarantines a diagnostic search.
    """
```

The assembled family entry, relative to a whole-search entry: every field of the roster in
`specs/partial-search-record-assembly` is present, **plus** one key:

```python
"assembly": {
    "expected": ["M->U", "U->M", "M->S", "S->M", "U->S", "S->U"],
    "shards": [
        {"id": "search-s00", "env": "<handle>", "environment": {...},
         "seconds": 1234.5, "transfers": ["M->U"]},
        ...
    ],
}
```

---

## Reachability, stated rather than assumed

`tools/kaggle/ceiling-search/run-config.json` names `run.notebook`, not a
`module.function`. The only worker-reachable route to the search is
`MIL-CREDA/Notebooks/Benchmark_Ceiling_Search.ipynb`, which reads
`config.execution_seed_units()` and calls `harness.run_search(pilot=…, shard=SHARD)`
**without `transfers=`**; `tests/test_scale_readings.py:1007` pins that call shape.

So this change makes the split **askable at the library level and assemblable**, exactly as
the proposal's Out of Scope says ("Actually running the distributed search" is a separate,
authorized decision). Wiring the notebook and its pinning test is a **follow-on change**,
and it is cheap to sequence because D1 makes the un-wired notebook refuse loudly
(`int("M->U")`) rather than run six clobbering full searches.

Likewise out of scope and named: no `tools/` launcher and no `__steps__` entry for the
assembly. `merge_search_records` is reached through the interpreter until one is added; a
`tools/` launcher would be free of stamp cost and separately sliceable.

---

## Testing Strategy

`strict_tdd: true`. Every row below is **red first, observed**, then green.
Command: `PYTHONPATH=src .venv/bin/python -m pytest -q`.

| Layer | What to test | Approach |
|---|---|---|
| Unit — declaration | `axes` holds both; `axis == "seed"` survives; every axis has all four groups | direct reads of `__benchmark__["distribution"]` |
| Unit — declaration | flat keys **are** the seed axis's lists | `assert dist["poolable"] is dist["axes"]["seed"]["poolable"]` — `is`, never `==`. Mutation that proves it: replace the `update` with a `copy.deepcopy`; an `==` test stays green, the `is` test reddens |
| Unit — declaration | the inversion: `"ceilingsByTransfer"` in the seed axis's `identicalAcrossShards` and **absent** from the transfer axis's | both directions asserted in one test |
| Unit — declaration | the forge's refusal is still able to fire | `dist.get("axis") == "arm"` is `False` on the real block **and** `True` on a hand-built `{"axis": "arm"}` |
| Unit — `rules_for_axis` | undeclared axis refuses; missing group refuses; flat `dist` resolves the seed axis and refuses `"transfer"` | three refusals, each naming what it was asked for |
| Unit — units | absent → `None`; `"[]"` → `[]`; populated → validated **per axis** | the three existing seed tests must still pass unchanged; new transfer cases beside them |
| Unit — units | **D1's cross-axis catch**: `execution_units_for("seed")` refuses `["M->U"]`, and `execution_units_for("transfer")` refuses `["0"]` | this is the test that measures the vocabulary decision rather than restating it |
| Unit — units | no default axis | `inspect.signature(...).parameters["axis"].default is inspect.Parameter.empty` |
| Unit — slice path | `shard_paths(id)["searchSlice"]` is under the shard home; `shard_paths(None)` lands beside the canonical record; pilot and noise move it | path assertions only, no search |
| Integration — slice write | a subset search leaves a readable slice where it previously left nothing, and the canonical path is untouched | drive `search_ceilings_trials` with a fake `run_one`, the shape `tests/test_search_declarations.py:50` already uses |
| Integration — assembly | six slices assemble; the entry carries every roster field **and** `assembly` | field-for-field against a whole-search entry built by the same fake engine |
| Integration — assembly | the pooled winner is **re-derived**, not copied | build slices whose individual pooled winners all differ from the union's winner; a dict-merge implementation reddens |
| Integration — assembly | five of six refuses **and names the sixth**; six of six does not raise | both directions, as the success criteria require |
| Integration — assembly | an unexpected transfer refuses; a duplicate label refuses; disagreeing `labelNoise` refuses and names both values; a drifted slice refuses | one test per refusal in D10 |
| Integration — assembly | `expected` order does not decide the guard | pass the six in reversed order; the canonical write still happens |
| Integration — assembly | an explicit smaller `expected` assembles **and writes nothing** | proves the guard quarantines a diagnostic assembly |
| Integration — `run_search` | subset: PARTIAL fires and names the slice path; whole: returns the record | `test_a_partial_search_refuses_to_present_its_slice_as_the_record` must pass **unedited** |
| Integration — `run_search` | D12's relaunch case: a canonical record on disk does **not** stop a subset search from running and writing its slice | the defect this design fixes; reddens against today's code |
| Regression | the roster check, the arm prohibition, the no-seed-axis pin | `test_every_writer_of_that_record_is_guarded`, `tests/test_distribute.py:44-52`, `test_the_search_still_has_no_seed_axis_to_split` — all unchanged |
| Regression | no site derives `expected` from `VERDICT_TRANSFERS` | source-level assertion; a length-only assertion would not catch it (both are six) |
| Regression | full suite, with the three formerly-red tests counted as passing | baseline `658 passed, 3 failed, 1 skipped, 129 subtests` → `661+ passed, 0 failed` |

**Mutation checks to run, not assume** (this repository's own recorded lesson: an anchor
that matched is not a mutation that ran):

1. `update(...)` → `deepcopy(...)`: the identity test must redden.
2. `ceiling_record.choose` over the union → take slice 0's `ceiling`: the re-pool test must
   redden.
3. `expected` default → `VERDICT_TRANSFERS`: the source-level test must redden (the
   count-based one will not).
4. `int(unidad)` removed from the seed reader: the cross-axis catch must redden.
5. The `else` branch deleted: the slice-write test must redden and `run_search`'s PARTIAL
   message must report `ABSENT`.

---

## Threat Matrix

| Boundary | Minimum adversarial cases | Applicability | Design response | Planned RED tests |
|---|---|---|---|---|
| Documentation-like paths | `requirements.txt`, executable Markdown, `README.sh` | **N/A** — this change classifies and executes no file; it reads one environment variable and writes/reads JSON records | — | — |
| Git repository selection | `git -C`, relative/absolute paths | **N/A** — no new VCS invocation. `harness._git_commit(config.REPOSITORY)` is pre-existing and untouched | — | — |
| Commit state | staged, `commit -a`, empty index | **N/A** — no commit is created or inspected by this change | — | — |
| Push state | tracking branch, first push, refspec | **N/A** — no push | — | — |
| PR commands | `--head`, environment prefix, composed commands | **N/A** — no PR automation; no subprocess and no shell command is composed anywhere in this diff | — | — |

### External input boundary (not a matrix row; recorded because it is the only untrusted seam)

`FORGE_RUN_UNITS` is a value this repository does not mint. It is the one place external
input crosses into the design, and it has two consequences that are handled rather than
assumed:

| Case | Behaviour | Test |
|---|---|---|
| Invalid JSON, non-array, non-string element | refuse, naming the offending value (existing messages preserved) | existing + per-axis |
| A label not in `SEARCH_TRANSFERS` | refuse, naming the label and the available set — **never** resolved, never coerced | new |
| A label used as a filesystem path component | **cannot happen**: shard ids are opaque and minted by the planner (D5); the label is a dict key and a JSON string only | path-shape test on `shard_paths` |
| A label interpolated into a shell command | **no shell command is composed** anywhere in this change; `FORGE_RUN_UNITS` is JSON in the process environment | — |

**What I could not determine** and therefore did not rely on: how the forge's
`runner_invoke.py` transports `units` into the child environment. It lives outside
`allowedEditRoots` and outside this repository, and `config.execution_seed_units`' docstring
is the only in-repo evidence ("`runner_invoke.py` siempre serializa `units` como una lista
de strings"). If that transport ever interpolates the value into a shell string, `>` in
`M->U` becomes a redirection character. The design's mitigation is structural rather than
hopeful: the label is never a path and never a command argument on this side, and the closed
-set validation means a mangled value refuses instead of resolving. `sdd-apply` should not
attempt to verify the forge's transport; the first remote rehearsal measures it.

---

## Migration / Rollout

No data migration. The declaration stores rules, not results.

1. **Before the first `src/` edit**, record `report_digest.source_digest`'s current value
   and the notebook stamps that depend on it — the rollback target from the proposal's
   Rollback Plan step 1.
2. Additive at every seam, and no existing on-disk record is rewritten:
   `governs_the_ceilings_record` means a subset search never wrote the canonical path, so
   no `ceilings.json` / `ceilings.pilot.json` is at risk. Slices are additive files,
   deletable without touching the canonical record.
3. **Revert order**, each independently safe: `merge_search_records` + the slice write →
   `run_search`/`ceilings_in_force` branches → `config.execution_units_for` → the `tools/`
   axis selection → `rules_for_axis` and the declaration nesting.
4. **The safe partial state**: with the declaration kept and the assembly reverted,
   `run_search`'s `7fcc150` refusal is the backstop — a slice still cannot be handed back
   as the record.

---

## Sequencing and slicing

`report_digest.source_digest` hashes `src/**/*.py` and nothing else. Consequences for
delivery:

| Slice | Contents | Stamp cost | Independence |
|---|---|---|---|
| **A** | `src/` in full: declaration + `rules_for_axis` + `execution_units_for` + slice write + assembly + `run_search`/`ceilings_in_force` + `pooling` + prose corrections + their tests | **Pays the whole cost once**: every notebook stamp invalidated → six-step pilot walk re-run + eighteen remote rehearsals (nine accounts × two shapes) repeated | Must be one unit — splitting `src/` across merged PRs pays the walk once per merge |
| **B** | `tools/bridge.py:166` axis selection | **None** | Order-independent: bridge keeps working via the projection whether B lands before or after A |

**400-line budget.** The forecast is High (roughly 300-350 authored `src/` lines plus
~350-450 test lines). This does **not** contradict the once-only cost: a Feature Branch
Chain whose slices all target the feature branch, with the pilot walk and the eighteen
rehearsals run **once after the last slice lands**, pays the cost once while keeping each
PR reviewable. `sdd-tasks` owns the forecast lines and the chain decision under the cached
`ask-on-risk` strategy.

Suggested reviewable units inside A, in dependency order:
A1 declaration + `rules_for_axis` + in-repo axis selection + prose · A2
`config.execution_units_for` (+ `config.transfer_label`, `harness` re-export) · A3
`shard_paths["searchSlice"]` + `sellar_rebanada` + the `else` slice write · A4
`read_search_slices` + `merge_search_records` · A5 `run_search`/`ceilings_in_force`
branches + the repointed message.

---

## Open Questions

Both questions the proposal named are **settled** above (D1, D2). What remains is recorded
rather than defaulted:

- [ ] **The forge's `runner_invoke.py` transport for `units` is unverified** from inside
      this repository. Named in the External Input Boundary section; the design does not
      depend on it, and the first remote rehearsal measures it. Not a blocker.
- [ ] **The notebook wiring is a follow-on change**, not this one. Until it lands, the
      transfer axis is reachable from the interpreter and not from a Kaggle submission.
      Deliberate, per the proposal's Out of Scope; the design makes the un-wired path
      refuse loudly rather than run wrong.
- [ ] **No `tools/` launcher or `__steps__` entry for the assembly.** Out of scope here;
      free of stamp cost when it is taken up.
- [ ] **The read-site coverage annotation** (`search_record` + `ceilings_in_force`) is
      rejected as out of scope, not as wrong. D13 records the implementation and the fixture
      cost so a later session can take it on cost alone — the same shape the proposal used
      for forge-side axis vetting.
