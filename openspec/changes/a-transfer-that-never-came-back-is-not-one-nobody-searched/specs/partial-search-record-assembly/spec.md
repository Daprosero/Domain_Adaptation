# Partial Search Record Assembly Specification

## Purpose

Declares the per-shard artifact a subset-transfer search leaves, the
assembly of those per-shard artifacts into the one record the campaign
reads, and the refusal that fires when an expected transfer is absent.

This is the capability the whole change is named after: today, a transfer
missing from a searched record because its shard never came back is
indistinguishable, at `harness.ceiling_for`/`harness.hyper_for`, from a
transfer nobody intended to search — both resolve through the same
pooled-winner fallback, and that fallback is a plausible-looking winner,
not a neutral value. This capability's refusing assembly is what keeps an
incomplete record from ever reaching those call sites in the first place.

## Requirements

### Requirement: A subset-transfer search leaves a per-shard artifact

A `run_search(transfers=...)` call that reaches the point where today's
code raises the "PARTIAL" refusal (`test_a_partial_search_refuses_to_present_its_slice_as_the_record`,
unchanged and still in scope) MUST first persist a per-shard record to a
path of its own — distinct from `config.ceilings_record_for(pilot)`, the
canonical path the campaign reads — so a subset search leaves something
readable where it previously left nothing. The existing refusal that keeps
the slice from being handed back as the canonical record is preserved
unchanged.

#### Scenario: a partial search writes a readable per-shard artifact

- GIVEN `run_search` is called with a `transfers` subset smaller than
  `config.SEARCH_TRANSFERS`
- WHEN the call completes its search and reaches the existing "PARTIAL"
  refusal
- THEN a per-shard record artifact exists on disk, readable back, at a
  path that is NOT `config.ceilings_record_for(pilot)`
- AND the existing `SystemExit` naming the record as partial still fires,
  unchanged in behavior and message content

### Requirement: The assembly knows one named expected transfer set

`harness.merge_search_records` (or an equivalently named assembly
function) MUST accept a parameter naming the expected transfer set
(`transfers=` or `expected=`). WHEN the caller omits it, the expected set
MUST derive from exactly one named source: `config.SEARCH_TRANSFERS`. No
call site of this assembly may derive its expectation from
`config.VERDICT_TRANSFERS` or from any value inferred from what the shards
being assembled happen to report about their own coverage.

#### Scenario: default expectation derives from the one named source

- GIVEN no explicit `transfers`/`expected` argument is passed to the
  assembly
- WHEN it computes completeness
- THEN the expected set it compares against equals `config.SEARCH_TRANSFERS`

#### Scenario: `VERDICT_TRANSFERS` is never the implicit source

- GIVEN the assembly's default-expectation code path
- WHEN inspected
- THEN it does not read `config.VERDICT_TRANSFERS` to determine
  completeness (both lists share length and content today, so a
  test relying on length alone would not catch a divergent source; the
  requirement is about which name is read, not merely the count)

### Requirement: A complete set of per-shard artifacts assembles into one record

WHEN a per-shard artifact exists for every transfer in the expected set,
the assembly MUST produce exactly one record containing every expected
transfer's searched values, shape-comparable to the record a whole,
unsplit search produces: `ceiling`, the seven other searched dimensions
(`rampDelta`, `rampDeltaLocal`, `ceilingLocal`, `kernelSigma`,
`attentionGamma`, `attentionTemperature`, `tauLocal`), `criterion`, `role`,
`epochs`, `trials`, `search`, `decidedByFlatRule`, `plateau`, `noise`,
`labelNoise`, `atRequiredScale`, `requiredScale`, `seconds`, `transfers`,
`neutral`, `byTransfer`, `perTransfer`, `inheritanceRule`, `env`,
`environment`, `revision`.

#### Scenario: six of six assembles to one complete record

- GIVEN six per-shard artifacts, one for each transfer in
  `config.SEARCH_TRANSFERS`
- WHEN the assembly runs
- THEN it returns one record naming all six transfers
- AND that record carries every field a whole-search
  `ceilings.pilot.json` entry carries, field-for-field

### Requirement: A missing expected transfer refuses, and names it

WHEN one or more transfers from the expected set have no corresponding
per-shard artifact, the assembly MUST refuse rather than assemble a record
that silently claims completeness. The refusal message MUST name the
specific missing transfer(s), not merely a count — so whoever reads the
refusal knows what to relaunch.

#### Scenario: five of six refuses and names the sixth

- GIVEN per-shard artifacts for five of the six transfers in
  `config.SEARCH_TRANSFERS`
- WHEN the assembly runs with the default expected set
- THEN it refuses (raises)
- AND the refusal message names the one transfer with no artifact

#### Scenario: a complete set never refuses on this ground

- GIVEN per-shard artifacts for every transfer in the expected set
- WHEN the assembly runs
- THEN it does not raise the missing-transfer refusal

### Requirement: Coverage completeness is judged against the expected set, not against what was handed in

The assembly MUST NOT treat "every shard it was handed is itself sealed
and complete" as sufficient grounds for declaring the assembled record
complete. Even when every per-shard artifact handed to the assembly is
individually well-formed, the assembly MUST still compare their combined
transfer coverage against the named expected set (the prior requirement's
`config.SEARCH_TRANSFERS`, or an explicit override) before declaring the
result complete.

#### Scenario: individually-sound shards covering fewer transfers still refuse

- GIVEN three well-formed, individually complete per-shard artifacts,
  covering three of the six transfers in `config.SEARCH_TRANSFERS`
- WHEN the assembly runs with the default expected set
- THEN it refuses citing the three missing transfers — even though none of
  the three artifacts handed to it is itself malformed or incomplete

#### Scenario: an explicit, deliberately smaller expected set is a distinct, named case

- GIVEN a caller explicitly passes `expected=` naming a set smaller than
  `config.SEARCH_TRANSFERS` (for example, a diagnostic re-search over one
  transfer)
- WHEN the shards handed in cover exactly that explicit set
- THEN the assembly succeeds against that explicitly named smaller
  expectation
- AND this outcome is reachable only through an explicit `expected=`
  override — omitting the argument never silently narrows the expected set
  to whatever the shards happen to cover

### Requirement: Shards that disagree on a field required to be identical refuse rather than average

WHEN two or more per-shard artifacts being assembled disagree on a field
the search declares must be identical across search shards (for example,
`labelNoise` or `epochs` — the values describing the search run itself,
not the per-transfer searched outputs), the assembly MUST refuse rather
than average, prefer one, or silently proceed. This mirrors the existing
`shards.merge()`/`disagreements()` precedent: refuse-on-disagree and
refuse-on-incomplete are both raised by the producer or the merge itself,
never left to a separate checker.

#### Scenario: disagreeing shards refuse and name the field

- GIVEN two per-shard artifacts, one stamped `labelNoise=0.0` and the
  other `labelNoise=0.4`
- WHEN the assembly runs
- THEN it refuses (raises)
- AND the refusal names the disagreeing field and the conflicting values

#### Scenario: agreeing shards proceed

- GIVEN every per-shard artifact being assembled agrees on every field
  required to be identical across search shards
- WHEN the assembly runs
- THEN it does not raise this disagreement refusal

### Requirement: The canonical write stays behind the existing write guard

The function that writes the assembled record to the canonical path
(`config.ceilings_record_for`) MUST consult
`harness.governs_the_ceilings_record` before writing, the same guard every
other writer of that path already consults. This capability MUST NOT
introduce a writer of that record that bypasses the guard the module-level
roster check (`test_every_writer_of_that_record_is_guarded`) already
enforces.

#### Scenario: a complete, clean assembly is guarded, not exempted

- GIVEN the assembly has produced a complete record covering exactly
  `config.SEARCH_TRANSFERS` with no contamination noise
- WHEN it writes that record to the canonical path
- THEN `harness.governs_the_ceilings_record` is consulted as part of that
  write, and it evaluates `True` for this case by construction

#### Scenario: an incomplete or noisy assembly never reaches this write path

- GIVEN the assembly has refused per the missing-transfer or
  disagreement requirements above
- WHEN that refusal fires
- THEN no write to the canonical path is attempted

### Requirement: A dead shard is distinguishable from a transfer nobody searched

Before `harness.ceiling_for`/`harness.hyper_for` ever resolve a value for a
transfer, any record they could read MUST have already passed through the
refusing assembly above. Consequently, a transfer absent from a record
those functions read is never absent *because a shard died silently* —
it is absent only because no search ever declared an intent to cover it
(the pre-existing, accepted out-of-sample case), since an assembly missing
a transfer it expected never produces a record those functions can reach
in the first place.

This requirement does not change `ceiling_for`/`hyper_for`'s own
pooled-winner fallback behavior for a genuinely out-of-sample transfer —
that fallback is a stated, accepted limitation of this change (see the
proposal's Accepted Limitations). What changes is which records are ever
allowed to reach those functions.

#### Scenario: a died shard's record never reaches the read site

- GIVEN a search was split across six shards and one shard died,
  leaving five per-shard artifacts and no artifact for the sixth transfer
- WHEN an operator attempts to promote that search's result to the
  canonical record the campaign reads
- THEN the assembly refuses (per the missing-transfer requirement above)
  before any record reaches `config.ceilings_record_for(pilot)`
- AND a subsequent `campaign()` run therefore cannot observe a record that
  silently claims six-transfer coverage while holding five

#### Scenario: a legitimately out-of-sample transfer is unaffected

- GIVEN a complete, assembled record whose expected set (by explicit,
  named declaration) never included some transfer `T`
- WHEN `ceiling_for`/`hyper_for` resolve a value for `T` against that
  record
- THEN they still fall back to the pooled winner for `T`, exactly as they
  do today — this capability does not alter that read-site behavior, only
  which records are permitted to exist for it to read
