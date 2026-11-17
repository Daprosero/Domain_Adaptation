# Run Unit Axis Resolution Specification

## Purpose

Declares how one submission's opaque `FORGE_RUN_UNITS` identifiers become
the units of a named axis, and what this resolution refuses rather than
guesses. `config.execution_seed_units()` decides "a unit IS a seed" today
without asking which axis is in play; this capability makes the axis an
explicit input rather than an assumption, while preserving the existing
three-way distinction (absent / declared-empty / populated) exactly.

**Vocabulary-agnostic notice**: the transfer axis's own unit vocabulary
(the canonical label `transfer_label()` produces, e.g. `"M->U"`, versus a
positional index into the declared transfer list) is an open design
question left to `sdd-design`. Every requirement below is written to hold
under either choice; none of them names or assumes one.

## Requirements

### Requirement: A named entry point resolves units for an explicit axis

`config` MUST expose a callable — `execution_units_for` or an equivalent
name — that receives the axis to resolve as an explicit, required argument
and never infers it from the content of `FORGE_RUN_UNITS`.

#### Scenario: the axis is a required, explicit input

- GIVEN `config`'s unit-resolution entry point
- WHEN its signature is inspected
- THEN it declares a parameter naming the axis (e.g. `axis`)
- AND that parameter carries no default that would let a caller omit
  naming the axis

#### Scenario: the same raw value never resolves without an axis

- GIVEN the raw string `"0"` is a syntactically valid seed and could
  equally be a valid positional transfer index
- WHEN the entry point is called
- THEN it never resolves a result for `"0"` without the caller having
  named which axis's vocabulary governs the read — there is no
  content-based inference path that would produce a value of the correct
  shape while answering a different question

### Requirement: Absence of a declaration resolves to "the whole grid"

WHEN `FORGE_RUN_UNITS` is not set in the environment, resolving units for
any valid axis MUST return `None`, meaning "this submission never declared
units for this axis; run everything this axis would otherwise partition."

#### Scenario: absent env var, any axis

- GIVEN `FORGE_RUN_UNITS` is unset
- WHEN units are resolved for the seed axis
- THEN the result is `None`
- AND WHEN units are resolved for the transfer axis
- THEN the result is also `None`

### Requirement: An explicitly empty declaration resolves to zero units

WHEN `FORGE_RUN_UNITS` is set to the JSON array `"[]"`, resolving units for
any valid axis MUST return an empty list (`[]`), never `None`. Zero units
decided is a fact distinct from nobody having decided anything, and
collapsing the two would run the whole grid under an assignment that
explicitly says this machine received none of it.

#### Scenario: declared-empty is not treated as absent

- GIVEN `FORGE_RUN_UNITS="[]"`
- WHEN units are resolved for any valid axis
- THEN the result is `[]`
- AND the result is a real list, distinguishable by identity/type from the
  `None` the absent case returns

### Requirement: A populated declaration is validated per element, never guessed

WHEN `FORGE_RUN_UNITS` is set to a non-empty JSON array, every element MUST
be validated against the named axis's own vocabulary and returned resolved.
An element that fails validation for that axis MUST be refused outright —
never silently coerced or reinterpreted — because a misread unit does not
fail loudly, it runs a different experiment and returns a number that looks
exactly as correct as the right one.

#### Scenario: a fully valid populated declaration resolves every element

- GIVEN `FORGE_RUN_UNITS` names a JSON array of identifiers, every one
  valid under the axis being resolved
- WHEN units are resolved for that axis
- THEN every element is present in the result, resolved into that axis's
  own vocabulary (e.g., integers for the seed axis)

#### Scenario: invalid JSON refuses

- GIVEN `FORGE_RUN_UNITS` is not valid JSON
- WHEN units are resolved for any axis
- THEN resolution refuses (raises) rather than returning a guessed or
  partial result

#### Scenario: a non-list JSON value refuses

- GIVEN `FORGE_RUN_UNITS` is valid JSON but not a JSON array
- WHEN units are resolved
- THEN resolution refuses, naming that a list was expected

#### Scenario: a non-string element refuses

- GIVEN the declared array contains an element that is not a JSON string
- WHEN units are resolved
- THEN resolution refuses, naming the offending element

#### Scenario: an element invalid for the named axis refuses

- GIVEN the declared array contains a string element that does not satisfy
  the named axis's own validation (e.g., a non-integer-parseable string
  read for the seed axis)
- WHEN units are resolved for that axis
- THEN resolution refuses, naming the offending element and the axis it
  failed to validate against — it does not fall back to a default, an
  empty result, or a silently-coerced value

### Requirement: The seed axis keeps its exact existing behavior

Resolving units for `axis="seed"` MUST preserve today's
`config.execution_seed_units()` behavior byte-for-byte across all three
branches (absent, declared-empty, populated-with-validation), whether the
new entry point wraps, replaces, or delegates to the existing function.
Existing callers of the seed axis (`run_campaign_shard`,
`run_mechanism_sweep_shard`) MUST observe no behavior change.

#### Scenario: seed-axis resolution is unchanged

- GIVEN any `FORGE_RUN_UNITS` value that `execution_seed_units()` handles
  today (absent, `"[]"`, or a populated valid/invalid array)
- WHEN the same value is resolved through the new axis-aware entry point
  with `axis="seed"`
- THEN the result (return value or raised exception, including its
  message content) matches what `execution_seed_units()` produces today
