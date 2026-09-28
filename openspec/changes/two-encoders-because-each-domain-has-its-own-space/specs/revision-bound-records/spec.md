# Revision-Bound Records Specification

## Purpose

The implementation is bound to `config.REVISION`, and every record, manifest,
and notebook stamps itself against that value. Moving `REVISION` to r25
invalidates every artifact stamped under r21. The refusal machinery that
compares a record's own revision against `config.REVISION` already exists
(`harness.py`, `latent.py`); this capability states the disposition of the
artifacts that bump invalidates, not new refusal machinery.

## Requirements

### Requirement: The revision bump is its own commit

`config.REVISION` MUST move from `"research-concept-r21.md"` to
`"research-concept-r25.md"` in a commit that contains that change and its
accompanying prose only. No source-code behavior change may land in the same
commit as the revision bump, so that reverting the bump alone (restoring
`"research-concept-r21.md"`) makes every existing r21-stamped record current
again with no data migration.

#### Scenario: The revision bump is isolated

- GIVEN the commit that changes `config.REVISION`
- WHEN that commit's diff is inspected
- THEN it contains only the `REVISION` value change and directly related prose, no other behavior change

#### Scenario: Reverting the bump restores every r21 record as current

- GIVEN `config.REVISION` reverted to `"research-concept-r21.md"`
- WHEN an r21-stamped record is loaded or checked for staleness
- THEN it is reported current, with no field rewritten

### Requirement: Every r21-stamped record is disposed of by refusal, never by mutation

The 67 records stamped under `research-concept-r21.md` (17 under
`MIL-CREDA/Results/`, 50 under `MIL-CREDA/Models/`) MUST NOT be deleted,
rewritten, or silently re-stamped once `config.REVISION` moves to r25. Every
existing refusal (`StaleCheckpointRevision` on a record's own `revision`
field, `available()` tagging a mismatched revision `currentRevision: False`,
and any analysis entry point that refuses a stale checkpoint) MUST continue
to refuse those exact 67 records under the new revision. No on-disk record
is mutated by the bump or by reverting it.

#### Scenario: A stale record is refused, not silently analysed

- GIVEN an r21-stamped checkpoint or manifest, and `config.REVISION` now set to r25
- WHEN that record is loaded for training-time reuse or post-hoc analysis
- THEN the load refuses with a message naming the record's own stamped revision and the current managed revision

#### Scenario: No on-disk record changes bytes

- GIVEN the full set of 67 r21-stamped records
- WHEN the revision bump lands, and independently, when it is reverted
- THEN none of the 67 records' file contents differ from their state before the bump

### Requirement: `latent.stale_revisions` reports exactly the invalidated set

`latent.stale_revisions()`, applied to the full set of records `available()`
discovers after the bump, MUST report exactly the 67 r21-stamped records (17
`Results/`, 50 `Models/` manifests) as stale under the new revision, and MUST
NOT report any r25-stamped record as stale.

#### Scenario: The stale count matches the measured 67

- GIVEN the full set of on-disk records after `config.REVISION` moves to r25
- WHEN `latent.stale_revisions()` is called
- THEN it reports exactly 67 records tagged under `"research-concept-r21.md"`, and none tagged under `"research-concept-r25.md"`

### Requirement: The bump is disclosed as re-owing a walk, never as destroying a result

Every one of the 67 invalidated records is pilot-scale: every `Models/` path
lives under `Models/Pilot/`, and every `Results/` path lives under
`Results/Pilot/` except `Results/Benchmark/ceilings.pilot.json`, itself a
pilot record. There is no full-scale ceilings record on disk for the bump to
lose. Reporting or prose describing the consequence of the bump MUST state
that no quotable result is lost — the bump invalidates material that was
already declared non-quotable (`AGREED.md:187` forbids quoting a pilot
number) — and MUST state that it re-owes the six-step pilot walk and the
notebook re-execution the first `src/` byte already invalidates via
`report_digest.source_digest`, rather than presenting the bump as if a
finished result were discarded.

#### Scenario: The disposition report names the walk, not a lost result

- GIVEN a report or commit message describing the consequence of the revision bump
- WHEN it states what the bump costs
- THEN it names the pilot walk and notebook re-execution owed, and does not claim a full-scale or quotable result was lost

### Requirement: Notebook stamps are invalidated by the same mechanism that already exists

`report_digest.source_digest` already covers every `*.py` under `src/`, so
the first `src/` byte this change touches invalidates every notebook stamp,
independently of the revision bump itself. `verification.ipynb` and
`Benchmark_Results.ipynb` additionally carry the literal r21 revision string
and MUST be re-executed under r25 before their stamps can be considered
current again. This requirement states that the existing digest mechanism is
sufficient and MUST NOT be bypassed, weakened, or duplicated by new
machinery for this change.

#### Scenario: A source edit invalidates every notebook stamp

- GIVEN any `src/`-tree edit that is part of this change
- WHEN a notebook's stamp is checked afterward
- THEN the stamp reports stale via the existing `report_digest.source_digest` comparison, with no new invalidation mechanism introduced
