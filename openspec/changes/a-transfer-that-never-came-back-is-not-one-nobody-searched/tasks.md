# Tasks: A transfer that never came back is not one nobody searched

`strict_tdd: true` (inherited). Every implementation task below names its RED step;
the RED step MUST be observed failing before its paired GREEN step is written.
Test command: `PYTHONPATH=src .venv/bin/python -m pytest -q`.

Already done, NOT re-tasked here:
- **Piece 2**, commit `7fcc150`: `run_search`/`ceilings_in_force` accept and forward `transfers`.
- **D12's guard repair**, commit `ee11b4d`: the subset refusal runs before the record
  read-back, counts `config.SEARCH_TRANSFERS`, tested against real `ceilings_in_force`/
  `search_record` with only `search_ceilings` stubbed, proven by mutation. The remaining
  open part of D12 — naming the slice artifact in the message, and the slice-aware
  "already answered" predicate — is Task 5.5/5.6 below, ordered after D2 (Work Unit A3).

Baseline: `660 passed, 3 failed, 1 skipped, 129 subtests passed`. The three failures are
`test_the_declaration_names_the_axis_the_search_can_split_on`,
`test_a_unit_resolves_to_the_axis_the_step_declares`,
`test_partial_search_records_merge_instead_of_overwriting` — this change's acceptance
signal. They under-specify the design; close tasks against the specs, not merely against
those three assertions as committed today.

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~300-350 authored `src/` lines + ~350-450 authored test lines (design's own estimate) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | Feature Branch Chain: PR1(A1) → PR2(A2) → PR3(A3) → PR4(A4) → PR5(A5), plus PR-B (Slice B, independent, bases on the tracker branch after A1 merges) |
| Delivery strategy | ask-on-risk |
| Chain strategy | feature-branch-chain (design's own determination — see "Sequencing and slicing" in design.md; orchestrator still confirms per ask-on-risk) |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High

**Why Feature Branch Chain, stated from design.md:** `report_digest.source_digest` hashes
all of `src/**/*.py`. The first `src/` byte this change touches invalidates every notebook
stamp, forcing the six-step pilot walk and eighteen remote rehearsals (nine accounts × two
shapes) to be repeated. That cost must be paid exactly once, after the *last* `src/` slice
lands — which is only true if every `src/`-touching slice targets one accumulating branch
(the tracker) rather than each merging independently to `main`. Slice A (all of A1-A5) is
therefore **one indivisible unit for the pilot-walk cost**, even though it is split into five
reviewable PRs for reviewer load. Slice B (`tools/bridge.py:166`) touches no `src/` file and
carries zero stamp cost — order-independent, and safe as a stacked-to-main PR if preferred.

### Suggested Work Units

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|------------------|--------------------|
| Pre-flight | Record digest/stamp baseline before any `src/` edit | n/a (precedes PR1) | `PYTHONPATH=src .venv/bin/python -c "from report_digest import source_digest; print(source_digest())"` | N/A — read-only measurement | N/A — no code change |
| A1 | Declaration gains `axes`, `rules_for_axis`, in-repo axis selection, prose corrections | PR1 (base: tracker branch) | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py tests/test_shards.py tests/test_benchmark_declarations.py` | N/A — library-level, no remote invocation added | Revert `__init__.py` distribution block + `shards.rules_for_axis`; flat declaration and all readers restored, no data at risk |
| A2 | `config.execution_units_for(axis)` + `config.transfer_label` + `harness.transfer_label` re-export | PR2 (base: PR1 branch) | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_scale_readings.py tests/test_search_distribution.py` | N/A | Revert `config.execution_units_for`/`transfer_label`; `execution_seed_units()` restored as a standalone function |
| A3 | `shard_paths["searchSlice"]` + `sellar_rebanada` + the `else` slice write (D2, D7, D8) | PR3 (base: PR2 branch) | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py tests/test_label_noise.py tests/test_search_declarations.py` | N/A — driven by a fake `run_one`, no live search | Revert the `else` branch in `search_ceilings_trials`/`search_ceilings`; canonical write path restored as the only write, per-shard files become dead but harmless |
| A4 | `read_search_slices` + `merge_search_records` (D9, D10, D11's default) | PR4 (base: PR3 branch) | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py tests/test_label_noise.py` | N/A | Revert `merge_search_records`/`read_search_slices`; no canonical record was ever written by them without the guard, so nothing to unwind on disk |
| A5 | `run_search`/`ceilings_in_force` branches + repointed message (D11, D12) | PR5 (base: PR4 branch, = tracker head) | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py tests/test_scale_readings.py -k "run_search or ceilings_in_force"` | Full suite regression: `PYTHONPATH=src .venv/bin/python -m pytest -q` | `run_search`'s `7fcc150` refusal is the backstop per design's Migration/Rollout step 4 — a slice still cannot be handed back as the record if this PR alone is reverted |
| Post-A5 | Six-step pilot walk + eighteen remote rehearsals, run once | after tracker merges | N/A — not a pytest command | **Requires separate explicit operator authorization**: spends metered Kaggle quota across nine accounts × two shapes. Do not run automatically from `sdd-apply` | N/A — read-only rehearsal, no repository state to roll back |
| B | `tools/bridge.py:166` selects the seed axis via `rules_for_axis` | PR-B (base: tracker branch, after A1 merges; independent of A2-A5) | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_bridge.py` | N/A | One-line revert; `dist` resolution falls back to the whole declaration, unchanged behavior via the D6 projection |

---

## Phase 0: Pre-flight (before any `src/` edit)

- [x] 0.1 Recorded (retroactively, via `git stash` of the uncommitted `src/` diff, before
      anything was committed — the measurement is honest, but it should have run strictly
      before the first `src/` edit rather than after; noted here as a process deviation).
      `source_digest(config.REPOSITORY, "MIL_CREDA_Benchmark")` at tracker HEAD (`2978a32`,
      the state before this apply batch's `src/` edits):
      `105756ca3d9a04c0305ea64efb7a89302819bc858a2beadd9bdc4fe99b3c8d4c` — this is the
      rollback target. All six notebooks under `MIL-CREDA/Notebooks/` (`Benchmark_
      Attention_Mechanisms`, `Benchmark_Campaign`, `Benchmark_Ceiling_Search`, `Benchmark_
      Noise_Sweep`, `Benchmark_Results`, `verification`) carry the `SOURCES-SHA256` stamp
      `3cbc5effcfc0742f7408261c7c1e9e63849764a44b54edb75586a8c8dc8b8072`, which does **not**
      match the value just measured at `2978a32` — the notebooks were already stale before
      this change touched anything (a pre-existing condition from earlier `src/`-touching
      commits, not caused by A1). Recorded as measured fact; not a blocker for this batch.

## Work Unit A1 — Declaration nesting, `rules_for_axis`, in-repo axis selection, prose (PR1)

Implements Decision A / D4 / D6. Spec: `search-distribution-axis`.

- [x] 1.1 RED — grew `test_the_declaration_names_the_axis_the_search_can_split_on` past the
      bare-roster assertion to prove the per-axis rule **inversion** (seed axis's
      `identicalAcrossShards` contains the three ceiling fields, transfer axis's does not)
      plus the four-groups-never-a-gap check — `tests/test_search_distribution.py`. Observed
      RED (`AssertionError: 'distribution' declara 'axis' en singular...`) before GREEN;
      observed GREEN after 1.8-1.9.
- [x] 1.2 RED — wrote `test_the_flat_keys_are_the_seed_axis_by_identity`, asserting `is` for
      all four groups, explicitly never `==` — `tests/test_search_distribution.py`. Observed
      RED (`KeyError: 'axes'`) before GREEN; observed GREEN after 1.8-1.9; confirmed
      load-bearing by the 1.15 mutation check.
- [x] 1.3 Wrote `test_the_forges_arm_split_refusal_remains_reachable` —
      `tests/test_search_distribution.py`. **Not RED**: both assertions (real declaration
      evaluates `False`, hand-built `{"axis": "arm"}` evaluates `True`) already hold on the
      unmodified `axis` scalar before any edit — this is an invariant-pinning test, not new
      functionality, so there was no red state to observe. Passed before and after 1.8-1.9,
      confirming the projection never touched `axis`.
- [x] 1.4 Wrote `test_shards_root_survives_the_axes_nesting_unchanged` —
      `tests/test_search_distribution.py`. **Not RED**, same reason as 1.3: `shardsRoot`'s
      value was never at risk from this task's own edits. Passed before and after.
- [x] 1.5 RED — wrote four `shards.rules_for_axis` unit tests: undeclared axis refuses;
      missing group refuses; flat `dist` resolves its own axis by identity (`is`, not just
      equality); flat `dist` refuses a different axis — `tests/test_shards.py`. Observed RED
      (`AttributeError: module has no attribute 'rules_for_axis'`) for all four before GREEN;
      observed GREEN after 1.10.
- [x] 1.6 RED — wrote `test_no_docstring_still_claims_the_axis_is_singular`, scanning
      `config.execution_seed_units.__doc__`, `harness.run_campaign_shard.__doc__`, and the
      `__init__.py` rationale block's source text — `tests/test_search_distribution.py`.
      Observed RED (all three sources flagged) before any GREEN. **After 1.13-1.14 this test
      is STILL RED** — `config.execution_seed_units`'s docstring ("no hay un segundo eje...")
      is deliberately not touched in A1 per this same task file's own parenthetical ("A2,
      where the function itself changes"); the `harness.run_campaign_shard` and `__init__.py`
      portions are fixed and no longer flagged. This is the deferred half, not a regression —
      full-suite run below confirms it is the only new (non-baseline) failure.
- [x] 1.7 Measured fixture churn by implementing 1.8-1.14 and then running the broader suite,
      not merely the two fixtures the design named. Result, recorded honestly because it
      diverges from the design's claim: `tests/test_bridge.py:182-213` and
      `tests/test_benchmark_declarations.py:218-232` (the two tests the design predicted)
      stayed green untouched, confirming `rules_for_axis`'s flat fallback is exact (D4). But
      **three fixtures the design did not name also broke**, and were genuinely required to
      change (never weakened): (a) `tests/test_benchmark_declarations.py`'s
      `test_the_distribution_declares_exactly_what_was_approved` (~line 160-187) does a full
      `dist == {...}` equality after popping only `shardsRoot`; the new `axes` key is an extra
      item the literal never expected. (b) `tests/test_report_tables.py`'s
      `test_render_refuses_a_dimension_the_declaration_calls_per_run` and
      `test_no_function_that_pools_a_dimension_accepts_one_the_declaration_forbids`, and (c)
      `tests/test_label_noise.py`'s
      `test_every_public_noise_entry_refuses_every_declared_per_run_dimension`, all three
      `monkeypatch.setitem` the **flat** top-level `perRun` key expecting
      `pooling.per_run_dimensions()` to see it — but 1.11 routes that reader through
      `rules_for_axis("seed")`, which (once `axes` exists) resolves the **nested**
      `axes["seed"]["perRun"]` block instead, bypassing a flat-only monkeypatch entirely. This
      is inherent to D4/D6, not a bug: the flat keys are now purely the forge's identity
      projection, and an in-repo reader routed through `rules_for_axis` correctly stops
      depending on them being independently mutable.
- [x] 1.8 GREEN — added the `axes` block with `seed` and `transfer` sub-blocks (all four
      groups each, transfer axis also carries `perUnit` per the Interfaces shape) to
      `MIL_CREDA_Benchmark.__benchmark__["distribution"]` in `src/MIL_CREDA_Benchmark/__init__.py`.
      `axis: "seed"` and `shardsRoot` stay exactly where they were.
- [x] 1.9 GREEN — added the one-line identity projection immediately after the `__benchmark__`
      literal closes: `__benchmark__["distribution"].update(__benchmark__["distribution"]["axes"]["seed"])`
      — `src/MIL_CREDA_Benchmark/__init__.py`.
- [x] 1.10 GREEN — implemented `shards.GROUPS` and `shards.rules_for_axis(axis, dist=None)` per
      the D4 shape — `src/MIL_CREDA_Benchmark/shards.py`.
- [x] 1.11 GREEN — updated the two in-repo flat-key readers to select the seed axis explicitly
      via `rules_for_axis("seed", ...)`: `harness.py:2353` (inside the bounded-summary report
      path) and `pooling.py:32` (`per_run_dimensions()`).
- [x] 1.12 GREEN — `shards.merge()`'s `dist` default (`shards.py:~473`) now resolves the seed
      axis via `rules_for_axis("seed")` instead of the flat `declaration()`. Verified no test
      or in-repo caller relies on `merge()`'s `dist=None` default (`tools/bridge.py` always
      passes `dist=` explicitly, and every `tests/test_shards.py`/`test_benchmark_declarations.py`
      call passes `dist=` too), so this is a safe, currently-unexercised-by-default change.
- [x] 1.13 GREEN — rewrote the `__init__.py` rationale block. **Finding, recorded honestly**:
      measured the block's actual text before editing and found no literal "no second axis"
      claim to remove (grepped for "only"/"solo"/"unico"/"singular"/"second"/"segund" —
      no hit inside the block); the proposal's citation of this block was imprecise on that
      specific point. Added the requested content regardless: the two-axis framing, the
      per-axis inversion argument, and the D6 identity-projection argument, placed right after
      the `axis:` paragraph and before "the four groups below" (which now explicitly reads as
      describing the seed axis).
- [x] 1.14 GREEN — corrected `harness.run_campaign_shard`'s docstring: replaced "the seed axis
      is what the declaration says may be split" (an implicit singularity claim via the scalar
      `axis` field) with prose naming both declared axes and stating this function reads only
      the seed one. `config.execution_seed_units`'s docstring is intentionally untouched, per
      this file's own note that it is corrected in A2.
- [x] 1.15 Mutation check performed and confirmed, not merely asserted: temporarily replaced
      the D6 `update(...)` with `copy.deepcopy(...)` in `__init__.py`, ran
      `test_the_flat_keys_are_the_seed_axis_by_identity` alone — **observed it redden**
      (`AssertionError: ... no es el MISMO objeto ...`) — then reverted via a pre-edit backup
      and re-ran the same test — **observed it green again**. Confirms the test measures
      identity, not equality.
- [x] 1.16 Applied all four genuinely-required fixture edits found in 1.7 (never a weakened
      assertion): `tests/test_benchmark_declarations.py` (pop `axes` before the equality
      check, with an explanatory comment), and the three `perRun` monkeypatch sites in
      `tests/test_report_tables.py` (two) and `tests/test_label_noise.py` (one) — each now
      also sets `distribution["axes"]["seed"]["perRun"]` alongside the flat key, so both the
      flat-key readback in the test and the `rules_for_axis`-routed reader under test see the
      synthetic dimension.
- [x] 1.17 Ran `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py
      tests/test_shards.py tests/test_benchmark_declarations.py tests/test_bridge.py`. Result:
      **175 passed, 3 failed** (`test_no_docstring_still_claims_the_axis_is_singular` — deferred
      to A2, see 1.6; `test_a_unit_resolves_to_the_axis_the_step_declares` — A2;
      `test_partial_search_records_merge_instead_of_overwriting` — A4).
      `test_the_declaration_names_the_axis_the_search_can_split_on` is green. No other test in
      this set regressed. Also ran the two fixtures affected outside this named set
      (`tests/test_report_tables.py`, `tests/test_label_noise.py`) after the 1.16 fixes: both
      fully green (110 passed, 1 skipped). Full repository suite (see summary): **668 passed,
      3 failed, 1 skipped, 129 subtests passed** — exactly the baseline's 660 passed plus the 8
      new tests this unit added (7 green, 1 deferred-red), with the acceptance-signal test now
      counted as passing instead of red.

## Work Unit A2 — `config.execution_units_for(axis)`, `transfer_label`, `harness` re-export (PR2)

Implements D1, D3. Spec: `run-unit-axis-resolution`.

- [x] 2.1 RED — write the no-default-axis test:
      `inspect.signature(config.execution_units_for).parameters["axis"].default is inspect.Parameter.empty`
      — `tests/test_scale_readings.py`. Observed RED (`AttributeError: module
      'MIL_CREDA_Benchmark.config' has no attribute 'execution_units_for'`) before GREEN;
      observed GREEN after 2.8-2.11.
- [x] 2.2 RED — write absent-env tests for both axes: `FORGE_RUN_UNITS` unset → `None` for
      `axis="seed"` and for `axis="transfer"` — `tests/test_scale_readings.py`. Same RED/GREEN
      as 2.1 (all 20 new tests in this unit shared one collective RED run before any GREEN
      task, per the batch below).
- [x] 2.3 RED — write declared-empty tests for both axes: `FORGE_RUN_UNITS="[]"` → `[]`
      (never `None`) for both axes — `tests/test_scale_readings.py`. RED/GREEN as above.
- [x] 2.4 RED — write the populated-transfer-axis test: a JSON array of valid transfer labels
      (`transfer_label()` output, e.g. `"M->U"`) resolves to `(str, str)` tuples against
      `config.SEARCH_TRANSFERS`; a label not in that set refuses, naming the label and the
      available set — `tests/test_scale_readings.py`. RED/GREEN as above.
- [x] 2.5 RED — write **D1's cross-axis catch**, the test that measures the vocabulary
      decision itself: `execution_units_for("seed")` refuses `["M->U"]` (via the existing
      `int()` parse refusal), and `execution_units_for("transfer")` refuses `["0"]` (not in the
      label map) — `tests/test_scale_readings.py`. RED/GREEN as above; also mutation-checked
      at 2.13.
- [x] 2.6 RED — write the invalid-JSON / non-list / non-string-element refusal tests,
      parametrized over both axes — `tests/test_scale_readings.py`. RED/GREEN as above.
- [x] 2.7 RED — write the seed-axis byte-identical regression test: every input
      `execution_seed_units()` handles today (absent, `"[]"`, populated valid/invalid) produces
      the identical return value or raised exception (including message content) through
      `execution_units_for("seed")` — `tests/test_scale_readings.py`. RED/GREEN as above.
      **Batch RED evidence for 2.1-2.7** (and D1's byte-identical class): ran
      `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_scale_readings.py -k
      "TestLasUnidadesSeLeenSegunElEjeQueElPasoDeclara or TestElEjeSemillaEsByteAByteIdenticoAAntes"`
      before any GREEN task — **20 failed**, every failure `AttributeError` on
      `config.execution_units_for` or `config.transfer_label` not yet existing. Observed
      GREEN on the same selector after 2.8-2.11: **20 passed**.
- [x] 2.8 GREEN — move `transfer_label` down to `config.py` as the one definition (layering:
      `config.py` imports nothing from the package) — `src/MIL_CREDA_Benchmark/config.py`.
- [x] 2.9 GREEN — implement `config.execution_units_for(axis)`: no default; validates `axis`
      against `__benchmark__["distribution"]["axes"]`; preserves the three-way
      absent/empty/populated distinction; validates each element in the named axis's
      vocabulary (seed: `int()`; transfer: `transfer_label()` closed-set map) —
      `src/MIL_CREDA_Benchmark/config.py`. Seed-axis populated/invalid branches extracted
      verbatim (byte-identical messages, Spanish, unedited) into `_seed_units_from`; the
      transfer-axis branch (`_transfer_units_from`) is new code with new English messages
      (config.py's own convention outside this one Spanish-authored function), since it has
      no legacy behaviour to match.
- [x] 2.10 GREEN — `execution_seed_units()` becomes `return execution_units_for("seed")`;
      remove its docstring's singular-axis citation (config.py:237-239) —
      `src/MIL_CREDA_Benchmark/config.py`. Confirmed via 5.13/full-suite run:
      `test_no_docstring_still_claims_the_axis_is_singular` is now fully green (its
      `config.execution_seed_units.__doc__` clause was the deferred-to-A2 part named in
      Task 1.6).
- [x] 2.11 GREEN — `harness.transfer_label = config.transfer_label` (re-export, no wrapper,
      the `shards.disagreements = _shard_io.disagreements` precedent) —
      `src/MIL_CREDA_Benchmark/harness.py`.
- [x] 2.12 Verify all in-repo call sites of `transfer_label` (23, per design's count) still
      resolve as plain calls — run the full suite once, confirm no `AttributeError`/import
      regression. Ran the full suite (see 5.13-equivalent summary below): **690 passed,
      1 failed, 1 skipped, 129 subtests** — zero `AttributeError`/import failures anywhere,
      confirming every call site resolves through the re-export.
- [x] 2.13 Mutation check: remove the `int(unit)` parse from the seed reader path and confirm
      the D1 cross-axis-catch test (2.5) reddens. Revert after confirming. Performed and
      confirmed, not merely asserted: replaced `semillas.append(int(unidad))` with
      `semillas.append(unidad)` in `_seed_units_from`, ran
      `test_la_captura_cruzada_de_ejes_de_D1` alone — **observed it redden**
      (`Failed: DID NOT RAISE ValueError`) — then reverted from a pre-edit backup and re-ran
      the same test — **observed it green again**.
- [x] 2.14 Run `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_scale_readings.py
      tests/test_search_distribution.py`. Confirm `test_a_unit_resolves_to_the_axis_the_step_declares`
      is green. Result: **70 passed, 1 failed** (the one failure is
      `test_partial_search_records_merge_instead_of_overwriting`, A4's, correctly still red).
      `test_a_unit_resolves_to_the_axis_the_step_declares` and
      `test_no_docstring_still_claims_the_axis_is_singular` both green.

**Work Unit A2 full-suite evidence**: `PYTHONPATH=src .venv/bin/python -m pytest -q` →
**690 passed, 1 failed, 1 skipped, 129 subtests passed** (baseline at A1's close was
`668 passed, 3 failed, 1 skipped, 129 subtests`). Delta: +20 new tests (all green) and 2
formerly-red tests now green (`test_no_docstring_still_claims_the_axis_is_singular`,
`test_a_unit_resolves_to_the_axis_the_step_declares`); the one remaining failure,
`test_partial_search_records_merge_instead_of_overwriting`, is A4's and unchanged by this
unit's edits. No other test regressed.

## Work Unit A3 — `shard_paths["searchSlice"]`, `sellar_rebanada`, the `else` slice write (PR3)

Implements D2, D7, D8. Depends on A1 (`axes` must exist for the slice-axis vocabulary
context, though the path logic itself is axis-agnostic). Must land **before** the rest of
D12 (Work Unit A5) — the slice-aware predicate in A5 reads `shard_paths(...)["searchSlice"]`,
which does not exist until this unit lands.

- [x] 3.1 RED — wrote `test_the_slice_artifact_lives_under_shard_paths`: a shard id lands
      under `results_for(...)/shards/<id>/ceilings.slice.json`; `shard_paths(None)` lands
      beside the canonical record (`<stem>.slice.json`); `pilot=True` and `noise=0.4` each
      route to a different path than the clean full-campaign shard —
      `tests/test_search_distribution.py`. Observed RED (`KeyError: 'searchSlice'`) before
      GREEN; observed GREEN after 3.4.
- [x] 3.2 RED — wrote `test_a_subset_search_leaves_a_readable_slice_and_the_canonical_path_
      untouched`: drives `search_ceilings_trials` with a fake `run_one` (the `estudios`
      fixture shape `tests/test_search_declarations.py:154-193` already uses, reproduced
      locally as the `_rebanada_engine` fixture), calls with a one-transfer subset, and
      asserts a readable slice file exists at `shard_paths(...)["searchSlice"]` afterward
      while `config.ceilings_record_for(False)` stays absent — `tests/test_search_
      distribution.py`. Observed RED (`KeyError: 'searchSlice'`) before GREEN; observed
      GREEN after 3.4-3.6.
- [x] 3.3 RED — wrote `test_the_slice_sealer_never_touches_the_governed_roster`: a direct
      assertion that `sellar_rebanada`'s own source contains neither `ceilings_record_for`
      nor `write_text` — the exact conjunction `tests/test_label_noise.py:1253-1274`'s roster
      derivation tests for, asserted directly rather than inferred from the roster's size
      staying constant — `tests/test_search_distribution.py`. Observed RED
      (`AttributeError: module 'MIL_CREDA_Benchmark.harness' has no attribute
      'sellar_rebanada'`) before GREEN; observed GREEN after 3.5. **Finding, recorded
      honestly**: the first draft of `sellar_rebanada`'s docstring prose named
      `config.ceilings_record_for` by its full dotted name, which made the test's own
      `"ceilings_record_for" not in fuente` assertion fail on the docstring text itself, not
      on any write path — the same substring check the real roster derivation applies to
      whole `inspect.getsource()` output, docstrings included. Fixed by rewording the
      docstring to describe the canonical path without spelling the function name, never by
      loosening the assertion.
- [x] 3.4 GREEN — added `SEARCH_SLICE_SUFFIX = ".slice.json"` and a `"searchSlice"` key to
      `shard_paths` per the D2 table (shard id → `shards/<id>/ceilings.slice.json`; `None` →
      beside the canonical record) — `src/MIL_CREDA_Benchmark/harness.py`.
- [x] 3.5 GREEN — implemented `sellar_rebanada(found, reduction, shard, transfers)` per D8:
      reuses `sellar_techos(found, reduction)` unchanged (same family→entry shape, `env`/
      `environment`/`revision` via `ceiling_record.stamp`), then adds `transfers` (this
      slice's covered labels, by name) and `shard` (this slice's own id) to every family
      entry — `src/MIL_CREDA_Benchmark/harness.py`.
- [x] 3.6 GREEN — converted the guarded write in `search_ceilings_trials` to the `if/else`
      shape (D7): `governs_the_ceilings_record(noise, transfers)` True → writes the canonical
      record as before; False → writes the slice via `shard_paths(shard,
      noise=reduction.labelNoise, kind=reduction.kind, pilot=reduction.pilot)["searchSlice"]`
      using `sellar_rebanada` — `src/MIL_CREDA_Benchmark/harness.py`.
- [x] 3.7 GREEN — applied the symmetric `else` branch to `search_ceilings`' grid-engine body,
      routed identically to how it already routes its `partial` file (same
      `shard_paths(shard, noise=reduction.labelNoise, kind=reduction.kind,
      pilot=reduction.pilot)` call) — `src/MIL_CREDA_Benchmark/harness.py`. Not independently
      RED/GREEN-tested by a dedicated new test (no task above named one for the grid engine
      specifically), but exercised passively and without failure by every existing
      `test_search_declarations.py`/`test_search_engine.py` fixture that already drives
      `search_ceilings` with a one-transfer subset (`_grid_search`'s default `transfers=1`) —
      confirmed by the full-suite run at 3.9/5.13-equivalent below, zero regressions.
- [x] 3.8 Mutation check performed and confirmed, not merely asserted: backed up
      `harness.py`, deleted the `else` branch added in 3.6 (`search_ceilings_trials`), ran
      `test_a_subset_search_leaves_a_readable_slice_and_the_canonical_path_untouched` alone —
      **observed it redden** (`AssertionError: ... ceilings.slice.json no existe`) — then
      restored the pre-mutation backup and re-ran the same test — **observed it green
      again**. (The companion half of this mutation check — that `run_search`'s PARTIAL
      message reports `ABSENT` when the slice is missing — is re-verified in Task 5.12, after
      A5's message exists, per this file's own note.)
- [x] 3.9 Ran `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py
      tests/test_label_noise.py tests/test_search_declarations.py`. Result: **83 passed,
      1 failed** (`test_partial_search_records_merge_instead_of_overwriting`, A4's, correctly
      still red — unchanged by this unit). `test_every_writer_of_that_record_is_guarded` is
      green (roster unaffected: `sellar_rebanada` carries neither `ceilings_record_for` nor
      `write_text` in its source, so it never enters the roster's conjunction; the two
      existing writers, `search_ceilings_trials`/`search_ceilings`, still carry
      `governs_the_ceilings_record` and stay correctly guarded). All three new slice-write
      tests (3.1-3.3) pass.

## Work Unit A4 — `read_search_slices` + `merge_search_records` (PR4)

Implements D9, D10, D11's default-source half. Spec: `partial-search-record-assembly`.
Depends on A3 (needs `shard_paths["searchSlice"]` and `sellar_rebanada`-shaped slices to read).

- [x] 4.1 RED — grew `test_partial_search_records_merge_instead_of_overwriting` (the
      committed red) into: six per-shard slices (one per `config.SEARCH_TRANSFERS` transfer,
      built by the real engine via the new `_seis_rebanadas` fixture, never hand-assembled)
      assemble into one record naming all six, field-for-field matching a real whole-search
      entry built by the SAME fake engine (not a hand-typed roster) **plus** the new
      `assembly` key — `tests/test_search_distribution.py`. Observed RED
      (`AttributeError: module 'MIL_CREDA_Benchmark.harness' has no attribute
      'merge_search_records'`) before GREEN; observed GREEN after 4.17-4.18. Also asserts
      `entrada["ceiling"] == registro_entero["ceiling"]` — dividing and reassembling produces
      the byte-identical number a whole search produces, since the per-(family,transfer) GP
      seed is deterministic (`crc32(familia|etiqueta)`).
- [x] 4.2 RED — wrote **D9's re-pooling mutation test**
      (`test_the_assembled_ceiling_is_repooled_over_the_union_not_copied_forward`): six
      single-transfer slices, ceilings `[0.05, 0.03, 0.01, 0.07, 0.02, 0.09]`, all `best`
      equal so the family-level plateau covers all six; asserts the assembled `ceiling`
      equals `0.01` (the union's smallest-ceiling winner) and is neither the first (`0.05`)
      nor the last (`0.09`) slice's own value — `tests/test_search_distribution.py`. Observed
      RED (`AttributeError`) before GREEN; observed GREEN after 4.17-4.18; re-confirmed
      reddening by mutation at 4.20.
- [x] 4.3 RED — wrote refusal #1 test `test_a_slice_with_no_declared_coverage_refuses`: a
      slice with `transfers: []` refuses via `shards.ShardIncomplete` naming the slice id —
      `tests/test_search_distribution.py`. Observed RED before GREEN; GREEN after 4.17-4.18.
- [x] 4.4 RED — wrote refusal #2 test
      `test_a_slice_stamped_under_a_different_revision_refuses`: a slice's `revision` field
      set to a stale value refuses via `shards.ShardsDisagree`, naming the slice id and
      `revision` — `tests/test_search_distribution.py`. Observed RED before GREEN; GREEN
      after 4.17-4.18.
- [x] 4.5 RED — wrote refusal #3 test `test_slices_of_different_search_kinds_refuse`: one
      slice's `search` key removed and a `grid` key added refuses via `shards.ShardsDisagree`,
      naming both kinds (`optuna`, `grid`) — `tests/test_search_distribution.py`. Observed RED
      before GREEN; GREEN after 4.17-4.18.
- [x] 4.6 RED — wrote refusal #4 test `test_two_slices_claiming_the_same_transfer_refuse`: two
      slices retargeted to the same transfer label refuse via `shards.ShardsDisagree`, naming
      the label and both slice ids — `tests/test_search_distribution.py`. Observed RED before
      GREEN; GREEN after 4.17-4.18.
- [x] 4.7 RED — wrote refusal #5 test
      `test_slices_disagreeing_on_labelnoise_refuse_and_name_both_values` (`labelNoise=0.0` vs
      `labelNoise=0.4`) refuses via `shards.ShardsDisagree` reusing `shard_io.disagreements`,
      naming `labelNoise` and `0.4` — `tests/test_search_distribution.py`. Also wrote the
      other direction, `test_agreeing_slices_do_not_raise_the_disagreement_refusal`. Observed
      RED before GREEN; GREEN after 4.17-4.18 (plus the message-formatting fix noted below).
- [x] 4.8 RED — wrote refusal #6 test `test_five_of_six_refuses_and_names_the_sixth`
      (**the defect this change is named after**): five of the six real slices refuse via
      `shards.ShardIncomplete`, naming the **specific** sixth missing transfer label, not a
      count — `tests/test_search_distribution.py`. Observed RED before GREEN; GREEN after
      4.17-4.18.
- [x] 4.9 RED — wrote refusal #7 test
      `test_a_slice_covering_a_label_outside_expected_refuses`: `expected=` narrowed to one
      transfer while a second slice covers another refuses via `shards.ShardsDisagree`, naming
      the unexpected label — `tests/test_search_distribution.py`. Observed RED before GREEN;
      GREEN after 4.17-4.18.
- [x] 4.10 RED — wrote refusal #8 test `test_slices_covering_different_family_sets_refuse`: a
      synthetic second family key (`"phantom"`) added to one slice refuses via
      `shards.ShardsDisagree`, naming both family sets — `tests/test_search_distribution.py`.
      Observed RED before GREEN; GREEN after 4.17-4.18.
- [x] 4.11 RED — wrote the "coverage judged against expected, not against what arrived" test
      `test_individually_complete_slices_still_refuse_on_expected_coverage`: three
      individually well-formed, individually complete real slices (covering three of six
      transfers) still refuse citing the three missing ones —
      `tests/test_search_distribution.py`. Observed RED before GREEN; GREEN after 4.17-4.18.
      Also wrote the direct complement, `test_a_complete_set_never_raises_the_missing_
      transfer_refusal` (six of six does not raise).
- [x] 4.12 RED — wrote the explicit-smaller-`expected` test
      `test_an_explicit_smaller_expected_set_assembles_without_writing_canonical`: one real
      slice, `expected=[config.SEARCH_TRANSFERS[0]]`, assembles successfully and leaves the
      canonical path unwritten (the guard quarantines the diagnostic assembly, D7's rule
      applied at assembly time) — `tests/test_search_distribution.py`. Observed RED before
      GREEN; GREEN after 4.17-4.18.
- [x] 4.13 RED — wrote the order-independence test `test_expected_order_does_not_decide_the_
      guard`: the six expected labels passed reversed; the canonical write still happens —
      `tests/test_search_distribution.py`. Observed RED before GREEN; GREEN after 4.17-4.18.
- [x] 4.14 RED — wrote the **source-level** test
      `test_the_assembly_never_reads_verdict_transfers_as_its_default`: inspects
      `inspect.getsource(harness.merge_search_records)` for the literal string
      `"VERDICT_TRANSFERS"` — `tests/test_search_distribution.py`. Observed RED
      (`AttributeError`, function did not exist) before GREEN; GREEN after 4.17-4.18; RE-RED
      confirmed by the 4.19 mutation. **Finding, recorded honestly, the same class 3.3 already
      hit**: the function's own docstring named `config.VERDICT_TRANSFERS` in prose (explaining
      it is never the source), which the substring scan cannot tell apart from a real read —
      reworded the docstring to describe "the campaign's own transfer roster" without spelling
      the constant, never loosened the assertion.
- [x] 4.15 RED — wrote `test_the_canonical_write_consults_the_guard` (spies on
      `harness.governs_the_ceilings_record` via monkeypatch, confirms it is actually CALLED
      for a complete assembly, not merely coincidentally satisfied) and
      `test_an_incomplete_assembly_never_attempts_the_canonical_write` (five of six refuses
      and leaves the canonical path unwritten) — `tests/test_search_distribution.py`. Observed
      RED before GREEN; GREEN after 4.17-4.18.
- [x] 4.16 RED — wrote `test_merge_search_records_enters_the_guarded_writer_roster`: asserts
      `merge_search_records`'s own source contains `ceilings_record_for`, `write_text` AND
      `governs_the_ceilings_record` — the explicit companion to Task 3.3's exclusion check —
      `tests/test_search_distribution.py`. Observed RED before GREEN; GREEN after 4.17-4.18.
      Confirmed `tests/test_label_noise.py::test_every_writer_of_that_record_is_guarded` stays
      green with the new writer present (roster grew by one, not weakened).
- [x] 4.17 GREEN — implemented `harness.read_search_slices(pilot=False, noise=0.0,
      kind="campaign", root=None)` — reads every `ceilings.slice.json` under the relevant
      `shards/` tree, the same enumerate-and-skip-missing shape `shard_io.read_shards` already
      uses — `src/MIL_CREDA_Benchmark/harness.py`, placed after `sellar_rebanada`, before
      `search_ceilings_trials`.
- [x] 4.18 GREEN — implemented `harness.merge_search_records(slices, expected=None,
      pilot=False, reduction=None)` per D9/D10/D11: refusals 1, 2, 3, 8 (in that order,
      global — checked before per-family grouping so a family-set mismatch cannot masquerade
      as a wrong per-family refusal), then per family 4, 5, 6, 7 (D10's literal order);
      `expected` defaults to `list(config.SEARCH_TRANSFERS)`, normalized into
      `SEARCH_TRANSFERS` order; re-pools via `ceiling_record.choose` over the union (reusing
      `d["best"]` as `value`, exactly as the engine does, over the seven hyper dims plus
      `ceiling`); carries `identicalAcrossShards` fields through from `entries[0]` after
      proving agreement (via a `{"shard": id, "stamp": entry}`-shaped call to
      `shard_io.disagreements`, using `shards.rules_for_axis("transfer")`'s own declared list);
      sums `seconds` as total compute; sets `env`/`environment` to the assembling machine's
      (`reduction.environment or environment()`); adds the `assembly` key
      (`{"expected": [...], "shards": [{"id","env","environment","seconds","transfers"}, ...]}`
      per family); writes the canonical path only behind `governs_the_ceilings_record` —
      `src/MIL_CREDA_Benchmark/harness.py`. **Deviation from the D10 table's literal numeric
      order, disclosed**: refusal #8 (family sets differ) is checked globally right after
      1-3, before #4-7's per-family loop, rather than textually last — necessary for
      soundness, since grouping by family before confirming the families agree would let a
      family-set mismatch surface as the WRONG refusal type (e.g. a phantom family's
      "missing transfers" instead of "family sets differ"). Each of the eight still fires
      correctly and independently, confirmed by its own isolated RED/GREEN test above.
      **Additional finding, not anticipated by design**: `tests/test_label_noise.py::
      TestEveryWriterFollowsTheRunAndNotTheTree::test_no_writer_takes_a_coordinate_the_bounds_
      already_carry` requires any `harness` function taking both `reduction` and a loose
      coordinate (`pilot`, here) to reconcile via `replace(reduction, ...)` in its own source
      — added `if reduction is not None: reduction = replace(reduction, pilot=pilot)`,
      the same pattern `search_ceilings_trials` already uses, never weakened the test.
- [x] 4.19 Mutation check performed and confirmed, not merely asserted: changed the `expected`
      default from `config.SEARCH_TRANSFERS` to `config.VERDICT_TRANSFERS`, ran
      `test_the_assembly_never_reads_verdict_transfers_as_its_default` alone — **observed it
      redden** (`AssertionError: 'VERDICT_TRANSFERS' is contained here...`) — and confirmed
      the count-based test (`test_partial_search_records_merge_instead_of_overwriting`)
      **stayed green** on the same mutated run, exactly as the design predicts (both lists
      share length six today). Reverted from the pre-mutation backup and re-ran both — both
      green again.
- [x] 4.20 Mutation check performed and confirmed, not merely asserted: replaced the
      `ceiling_record.choose(...)` re-pool step with a copy-forward of `entries[0]`'s own
      pooled fields, ran `test_the_assembled_ceiling_is_repooled_over_the_union_not_copied_
      forward` alone — **observed it redden** (`AssertionError: ... assert 0.05 == 0.01`,
      i.e. the mutated code returned slice 0's own ceiling instead of the union's winner) —
      then reverted from the pre-mutation backup (diffed byte-identical to confirm) and
      re-ran the same test — **observed it green again**.
- [x] 4.21 Ran `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py
      tests/test_label_noise.py`. Result: **92 passed**, 0 failed.
      `test_partial_search_records_merge_instead_of_overwriting` is green and
      `test_every_writer_of_that_record_is_guarded` is still green. Full repository suite
      (see closing note below): **712 passed, 0 failed, 1 skipped, 129 subtests passed** —
      baseline for this unit was `693 passed, 1 failed, 1 skipped, 129 subtests`; delta is
      +18 new tests (all green) and the one baseline failure
      (`test_partial_search_records_merge_instead_of_overwriting`, this change's acceptance
      signal) now passing. No other test regressed.

## Work Unit A5 — `run_search`/`ceilings_in_force` branches + repointed message (PR5)

Implements D11 (message repoint) and D12 (subset-branch reorder + slice-aware predicate).
Depends on A3 (`shard_paths["searchSlice"]`) and A4 (the assembly the message references).
Completes the still-open half of the already-committed D12 guard repair (`ee11b4d`).

- [ ] 5.1 RED — write **D12's relaunch-defect test**: with a canonical record already on
      disk, calling the subset search path still runs and writes its slice rather than
      short-circuiting (today's `ceilings_in_force` short-circuits at :1463 and never
      searches when a canonical record exists — this test reddens against current code) —
      `tests/test_search_distribution.py`.
- [ ] 5.2 RED — write the subset-PARTIAL-names-the-slice test: `run_search(transfers=...)`'s
      `SystemExit` message names the slice path and states whether it is `written` or
      `ABSENT -- this slice produced nothing to assemble` — `tests/test_search_distribution.py`.
      `test_a_partial_search_refuses_to_present_its_slice_as_the_record` MUST still pass
      **unedited** alongside this new test.
- [ ] 5.3 RED — write/confirm the whole-search regression: a `run_search` call with no
      `transfers` still returns the assembled record unchanged (existing coverage; add if
      missing) — `tests/test_search_distribution.py` or `tests/test_scale_readings.py`.
- [ ] 5.4 RED — write the harness-side source-level check mirroring 4.14: `run_search`'s
      refusal-message construction reads `config.SEARCH_TRANSFERS`'s length, not
      `config.VERDICT_TRANSFERS`'s, at the source level —
      `tests/test_search_distribution.py`.
- [ ] 5.5 GREEN — reorder `run_search`'s subset branch to run **before** the `record is None`
      check (harness.py ~2487-2518), per the D12 shape: call `ceilings_in_force(...,
      transfers=transfers)`, then if `transfers is not None` raise the slice-aware `SystemExit`
      naming `shard_paths(shard, pilot=pilot)["searchSlice"]` and its written/ABSENT state —
      `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 5.6 GREEN — make `ceilings_in_force`'s "already answered" predicate slice-aware when
      `transfers is not None`: check `shard_paths(shard, pilot=pilot)["searchSlice"].exists()`
      instead of `search_record(pilot=pilot) is not None` — `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 5.7 GREEN — repoint `run_search`'s refusal message (harness.py:2509) from
      `config.VERDICT_TRANSFERS` to `config.SEARCH_TRANSFERS` (D11) —
      `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 5.8 Verify `test_a_partial_search_refuses_to_present_its_slice_as_the_record` passes
      **unedited**: confirm `str(len(config.VERDICT_TRANSFERS))` still appears in the message
      (both lists have length six, so the repoint keeps it green without touching the test).
- [ ] 5.9 Regression: confirm `test_every_writer_of_that_record_is_guarded`,
      `tests/test_distribute.py:44-52` (arm split forbidden), and
      `test_the_search_still_has_no_seed_axis_to_split` all remain green, unchanged.
- [ ] 5.10 Regression: confirm no `distribution` reader resolves a group to `None` or `[]` by
      accident — specifically `tools/distribute.py:208`'s `.get("axis")` still produces a real
      axis answer (`"seed"`) in the plan JSON (this file is unedited by design; this task is a
      read-only confirmation, not an edit).
- [ ] 5.11 Apply the fixture edits (if any remained after 1.7/1.16) surfaced by running the
      full suite at this point — measure before editing, never weaken an assertion to make a
      fixture pass.
- [ ] 5.12 Re-run mutation check from 3.8 now that A5's message exists: delete the A3 `else`
      branch and confirm both (a) the A3 slice-write test reddens and (b) `run_search`'s
      PARTIAL message now correctly reports `ABSENT` for the missing slice. Revert after
      confirming.
- [ ] 5.13 Run the full suite: `PYTHONPATH=src .venv/bin/python -m pytest -q`. Confirm the
      baseline `660 passed, 3 failed, 1 skipped, 129 subtests` transitions to `663+ passed,
      0 failed, 129+ subtests` (all three named red tests now passing, nothing else newly red).
- [ ] 5.14 Confirm success-criterion checkbox: the measured projection (whole-search 5.6-6.2h
      vs ~1.2h/machine across six) is recorded in the change's closing notes as a projection,
      not yet a measurement — no remote run has occurred in this change.

## Phase 6 — Post-slice validation (requires separate operator authorization)

Gated on the tracker branch merging (i.e., after A5/PR5). Spends metered remote Kaggle
quota; do not run automatically from `sdd-apply`.

- [ ] 6.1 Obtain explicit operator authorization to spend Kaggle quota for the pilot walk and
      rehearsals (per this environment's remote-operation authorization contract — permission
      to implement locally does not authorize remote execution).
- [ ] 6.2 Once authorized: run the six-step pilot walk once, re-stamping every notebook
      invalidated by this change's `src/` diff.
- [ ] 6.3 Once authorized: run the eighteen remote rehearsals (nine accounts × two shapes) the
      pilot walk requires.
- [ ] 6.4 Record the outcome (pass/fail per step) in the change's closing notes. This task
      does not include "actually running the distributed search" — that remains a separate,
      later, separately-authorized decision per the proposal's Out of Scope.

## Work Unit B — `tools/bridge.py:166` axis selection (PR-B, independent)

Order-independent relative to A2-A5; functionally needs `shards.rules_for_axis` (A1). Zero
`src/` stamp cost — `tools/` is outside `report_digest.source_digest`'s boundary.
`tools/distribute.py` needs **no change** (verified in design: `.get("axis")` at :208 is its
only declaration read, and it stays a real string via the D6 projection).

- [ ] B.1 RED — write a test asserting `tools/bridge.py`'s `dist` resolution at line 166 calls
      `shards.rules_for_axis("seed", dist)` rather than defaulting to the whole declaration
      directly — `tests/test_bridge.py`.
- [ ] B.2 RED — confirm (do not assume) that the five flat reads below line 166 (:167-168,
      :203, :216-217) need **zero** edits, by running the existing `test_bridge.py:190-210`
      fixture against the new resolution and observing it pass unedited.
- [ ] B.3 GREEN — change the `dist` resolution at `tools/bridge.py:166` to
      `shards.rules_for_axis("seed", dist)`. Leave every flat read below it untouched.
- [ ] B.4 Run `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_bridge.py`. Confirm
      green, unedited elsewhere in the file.

---

## Notes for `sdd-apply`

- Every task above that references a mutation check is a **verification step to run**, not
  a claim to trust — this repository's own recorded lesson is that an anchor matching a
  string is not the same as a mutation having actually run and reddened the intended test.
- Never weaken, delete, or `==`-ify an `is`-based assertion, a roster-derived test, or the
  arm-refusal test to make a fixture pass. If a fixture genuinely needs to change (Tasks
  1.7/1.16/5.11), change the fixture's input data, not the assertion's meaning.
- `src/CREDA` is untouched throughout — no task above names a file under it.
- No task in this file authorizes spending remote Kaggle quota except Phase 6, which is
  explicitly gated on separate operator authorization.
