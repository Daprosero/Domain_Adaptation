# record-provenance-guard

## Objective

Close the record-provenance guard the target's own `AGREED.md` names, so the
item stops blocking every run — pilot, shard, notebook and search alike.

## The agreement, verbatim and currently unticked

> Nothing runs (pilot, shard, notebook or search) until the record-provenance
> guard is closed for every record the decided experiments keep: stale ceilings
> reaching a run through import-time defaults, `Reduction.from_record` ignoring
> the revision, and the noise diagnostic re-stamping an unchecked sweep.

Three clauses, and they are not the same kind of work. Measured (read-only)
before anything was written:

### 1. Stale ceilings through import-time defaults — live

`config.py:471/480` declares `CEILINGS`/`CEILINGS_BY_TRANSFER` as module dicts
filled once at import (`config.py:1405-1406`). `Reduction`'s own default
factories read them (`harness.py:303`, `:308-310`), and `ceiling_for`
(`:452-454`) and `hyper_for` (`:488-496`) read the fields off the instance —
never off disk. `with_ceilings_in_force` (`:1363`) is the one call that
re-reads both halves fresh.

`campaign()`'s guards (`:2016-2049`) catch an empty mapping and a family whose
per-transfer picks are ABSENT from the record. They do not catch a value that
CHANGED under an unchanged stamp. An existing test says so in its own words —
`tests/test_distribute.py:202-236` — and it documents the fix applied to
exactly one call site, `tools/distribute.py`'s `run_shard`. The general case
has no equivalent fix: anything building a bare `Reduction()` and calling
`ceiling_for`/`hyper_for`/`campaign()` directly inherits the import-time
snapshot and refuses nothing. `ceiling_for`'s own docstring admits it
(`:437-439`): a bare `Reduction` carries `ceilingSearch == {}`, and nothing to
check against means nothing refuses.

This is the class-sweep of a defect already fixed at one instance.

### 2. `from_record` ignoring the revision — real, and today unreachable

`harness.py:328-367` calls `_latent.hyperparameter_drift`, whose
`HYPERPARAMETER_FIELDS` (`latent.py:280-284`) holds `kernelSigma`,
`attentionGamma` and `attentionTemperature` — and not `revision`. Proven by
mutation, in process: a record rewritten to `revision: "r17-fake-old-revision"`
rebuilds into a `Reduction` carrying that revision, with no refusal.
`latent.load` (`latent.py:390-400`) raises `StaleCheckpointRevision` on exactly
that mismatch, and `from_record`'s own docstring (`:338-343`) claims the
equivalence it does not have — prose that outlived its mechanism.

Its only callers today are tests (`test_latent_binding.py:460,477,488`); no
live `src/` or `tools/` path calls it. So the defect is real and currently
unreached. It is closed anyway, because the docstring asserts a refusal that
does not exist and the next caller will believe it.

### 3. The noise diagnostic — retired, so this clause names nothing live

`config.py:254-261` states the removal outright: the diagnostic's config
entries, its two steps and its notebooks are gone. No `noise-diagnostic` key
exists in `__steps__`; `tests/test_steps.py` and
`tests/test_report_tables.py:1412-1414` assert the retirement; `AGREED.md`
itself records the decisions (2026-09-15, 2026-09-17). Nothing re-stamps a
sweep.

**So this clause is closed by correcting the agreement, not the code** — and
that is the operator's act, not this session's. The skill's own mechanism is
`settle --reverse`, which requires a written paragraph it deliberately refuses
to author. One loose end belongs with it: `src/MIL_CREDA_Benchmark/__init__.py:332`
still declares `Results/Noise/diagnostic.json`, a record nothing now produces.

## Scope

**Authorized here.** `src/MIL_CREDA_Benchmark/harness.py` and the target's own
tests. Target only — nothing under the forge's `skills/`, measured: all three
clauses resolve to target symbols and `rg` finds no match for any of them in
`skills/`.

**Not authorized here.** Any edit to `MIL-CREDA/AGREED.md`, and the
`diagnostic.json` declaration that travels with it. Ticking the agreement:
the mark is `settle --done` and it needs a witness test plus the operator.

## Constraints

- Red first, observed, in the target's own suite:
  `.venv/bin/python -m pytest` from the target root (`pythonpath=["src"]`,
  no install step).
- The target is its own git repository with its own remote.
- A green test proves nothing if red was not reachable — and reverting the
  implementation is not always the mutation that proves it.

## Tasks

- [ ] P1 — `ceiling_for`/`hyper_for` stop trusting the import-time snapshot:
      the value-level drift the one-call-site fix already exercises, swept to
      the class. (red first)
- [ ] P2 — `from_record` makes the revision check `latent.load` makes, and its
      docstring stops claiming a refusal it does not perform. (red first)
- [ ] P3 — Report clause 3 to the operator as an agreement correction, with the
      evidence. Not written here.

## Progress

Mapped read-only. Nothing written yet.
