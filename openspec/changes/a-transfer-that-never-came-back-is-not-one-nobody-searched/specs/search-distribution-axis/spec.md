# Search Distribution Axis Specification

## Purpose

Declares which axes a `MIL_CREDA_Benchmark` shard may split on, and the
per-axis rules for the four dimension groups (`poolable`, `perEnvironment`,
`perRun`, `identicalAcrossShards`) — so a search shard (split by transfer)
and a campaign shard (split by seed) each resolve the rule set that applies
to their own kind of split, including the inversion that motivates this
capability: the six ceiling-search fields (`ceilings`, `ceilingsByTransfer`,
`hyperByTransfer`, plus their siblings) must be identical across campaign
shards and must legitimately differ across search shards.

This spec covers the declaration shape only (piece 1 of the proposal). It
does not cover how a unit resolves to an axis (see
`run-unit-axis-resolution`) or how partial search records assemble (see
`partial-search-record-assembly`).

## Requirements

### Requirement: The declaration exposes a plural axis roster

`MIL_CREDA_Benchmark.__benchmark__["distribution"]` MUST expose a field
named `axes` naming every axis a shard of this benchmark may split on, in
addition to (not instead of) the existing scalar `axis` field.

#### Scenario: `axes` names both existing split kinds

- GIVEN `MIL_CREDA_Benchmark.__benchmark__["distribution"]`
- WHEN `axes` is read
- THEN it is not `None`
- AND it contains `"seed"` (the campaign's and the mechanism sweep's existing split)
- AND it contains `"transfer"` (the search's new split)

#### Scenario: the scalar `axis` field is untouched

- GIVEN the same `distribution` block
- WHEN `axis` is read
- THEN it still equals the exact string `"seed"` — not moved, not renamed,
  not pluralized away, not replaced by a list

### Requirement: Each declared axis resolves its own complete rule set

Every axis named in `axes` MUST resolve a nested sub-block containing all
four grouping keys — `poolable`, `perEnvironment`, `perRun`,
`identicalAcrossShards` — even when a group is empty for that axis, so a
generic reader can resolve any axis's rules without a missing-key error.
The seed axis's nested sub-block MUST equal today's flat top-level values
exactly, so seed-axis behavior is preserved byte-for-byte.

#### Scenario: the seed axis's nested rules match today's flat values

- GIVEN `distribution["axes"]["seed"]`
- WHEN its four groups are read
- THEN `identicalAcrossShards` equals `["epochs", "ceilings", "ceilingsByTransfer", "hyperByTransfer", "labelNoise"]`
- AND `poolable`, `perEnvironment`, `perRun` equal today's flat values

#### Scenario: every declared axis has all four keys, never a gap

- GIVEN any axis named in `axes` (`"seed"` or `"transfer"`)
- WHEN its nested sub-block is read
- THEN all four group keys (`poolable`, `perEnvironment`, `perRun`,
  `identicalAcrossShards`) are present, each resolving to a list (possibly
  empty) — never absent, never `None`

### Requirement: The transfer axis inverts the ceiling-field agreement rule

The transfer axis's `identicalAcrossShards` group MUST NOT require
agreement on the fields a search shard's own slice necessarily differs on —
`ceilings`, `ceilingsByTransfer`, `hyperByTransfer` — the opposite rule the
seed axis carries for those exact field names. Each search shard legitimately
holds a different transfer's own searched values, and telling those shards
to agree on them is the defect this capability exists to remove.

#### Scenario: the same field name carries opposite rules per axis

- GIVEN `distribution["axes"]["seed"]["identicalAcrossShards"]`
- AND `distribution["axes"]["transfer"]["identicalAcrossShards"]`
- WHEN both are read
- THEN `"ceilingsByTransfer"` is present in the seed axis's list
- AND `"ceilingsByTransfer"` is absent from the transfer axis's list (or is
  explicitly declared as a field search shards are permitted to differ on)
- AND the same inversion holds for `"ceilings"` and `"hyperByTransfer"`

### Requirement: The flat top-level keys project the seed axis by identity

The four flat top-level keys the forge's declaration reader requires
(`poolable`, `perEnvironment`, `perRun`, `identicalAcrossShards`) MUST be
bound to the exact same list objects as the seed axis's nested groups — a
projection, never a copy. This is the load-bearing requirement of the whole
capability: two structures that merely hold equal lists are two copies, and
equality that holds today is exactly how two copies begin to disagree.
Identity is what makes drift unrepresentable rather than merely unlikely.

A verifying test MUST assert object identity (`is`), explicitly not
equality (`==`). An `==` assertion passes against a copy and therefore
measures nothing this requirement is about.

#### Scenario: flat keys and seed-axis groups are the same objects

- GIVEN `distribution["identicalAcrossShards"]`
- AND `distribution["axes"]["seed"]["identicalAcrossShards"]`
- WHEN compared with `is`
- THEN they are the same object
- AND the same identity holds for `poolable`, `perEnvironment`, and `perRun`
- AND an `==` comparison alone (without `is`) does NOT satisfy this
  requirement, since it would pass against an independently-built copy

#### Scenario: a change to one is visible through the other

- GIVEN `distribution["axes"]["seed"]["identicalAcrossShards"]` is the same
  list object as the flat `distribution["identicalAcrossShards"]`
- WHEN either reference is read after the module has loaded
- THEN both references observe the identical contents, because they name
  one underlying list rather than two lists that happen to agree

### Requirement: The forge's arm-split refusal remains reachable

Nothing in this capability may cost `implementation_engine.py`'s
`axis_is_comparison = dist.get("axis") == "arm"` its ability to fire. The
check MUST keep reading a real string value from `axis`, capable of
evaluating both `True` and `False`, rather than degrading to a key that
reads `None`.

#### Scenario: the check evaluates False on the real declaration

- GIVEN the production `distribution` block, where `axis == "seed"`
- WHEN `dist.get("axis") == "arm"` is evaluated
- THEN it evaluates `False` — the refusal does not fire on a legitimate
  seed-split declaration

#### Scenario: the check is still capable of evaluating True

- GIVEN a `distribution`-shaped mapping whose `axis` field is set to
  `"arm"` (constructed for this test; not a value this benchmark ever
  declares)
- WHEN `dist.get("axis") == "arm"` is evaluated against it
- THEN it evaluates `True` — proving the comparison reads a live,
  real-valued field rather than one that would read `None` under the new
  nested shape and silently degrade to permanently `False`

### Requirement: Non-group declaration keys survive unchanged

Declaration keys outside `axis`/`axes` and the four groups — `shardsRoot`
today, and any other forge-read optional key that may be introduced later —
MUST survive the nesting change with their existing value, since nothing in
the current test suite would go red if one were silently dropped.

#### Scenario: `shardsRoot` is unchanged

- GIVEN `distribution["shardsRoot"]`
- WHEN read after `axes` is added
- THEN it still equals `"MIL-CREDA/Results/Benchmark/shards"`

### Requirement: Rationale that cites the singular axis is corrected

Comments and docstrings that justify current behavior by asserting "there
is no second axis a unit could name" (the `__init__.py` distribution
rationale block, `config.execution_seed_units`'s docstring, and
`harness.run_campaign_shard`'s docstring) MUST be corrected once `axes`
names two values, since that citation becomes false the moment this
capability lands. The field these docstrings describe survives; the
justification does not.

#### Scenario: no surviving docstring claims singularity

- GIVEN the docstrings of `config.execution_seed_units` and
  `harness.run_campaign_shard`, and the `__init__.py` rationale block
  beside `distribution`
- WHEN `axes` contains both `"seed"` and `"transfer"`
- THEN none of the three continues to assert that no second axis exists
