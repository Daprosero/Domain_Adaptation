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

- [ ] 0.1 Record the current `report_digest.source_digest()` value and list every notebook
      stamp under `MIL-CREDA/Notebooks/` that depends on it. This is the rollback target
      from the proposal's Rollback Plan step 1 and design's Migration/Rollout step 1 — reverting
      the `src/` diff restores this value and every stamp with it, so a revert does not repeat
      the pilot walk or the eighteen rehearsals.

## Work Unit A1 — Declaration nesting, `rules_for_axis`, in-repo axis selection, prose (PR1)

Implements Decision A / D4 / D6. Spec: `search-distribution-axis`.

- [ ] 1.1 RED — grow `test_the_declaration_names_the_axis_the_search_can_split_on` past the
      current bare-roster assertion to prove the per-axis rule **inversion**: the seed axis's
      `identicalAcrossShards` contains `"ceilings"`/`"ceilingsByTransfer"`/`"hyperByTransfer"`
      and the transfer axis's list does not — `tests/test_search_distribution.py`. Satisfies
      spec scenarios "the same field name carries opposite rules per axis" and "every declared
      axis has all four keys, never a gap" (`search-distribution-axis/spec.md`).
- [ ] 1.2 RED — write the identity test:
      `distribution["identicalAcrossShards"] is distribution["axes"]["seed"]["identicalAcrossShards"]`
      and the same for `poolable`, `perEnvironment`, `perRun` — using `is`, explicitly never
      `==` — `tests/test_search_distribution.py`. Satisfies "The flat top-level keys project
      the seed axis by identity" (load-bearing requirement; an `==` test passes against a copy
      and measures nothing).
- [ ] 1.3 RED — write the test proving the forge's arm-split refusal remains reachable:
      `dist.get("axis") == "arm"` evaluates `False` on the real `distribution` block **and**
      `True` on a hand-built `{"axis": "arm"}` mapping — `tests/test_search_distribution.py`.
      Satisfies "The forge's arm-split refusal remains reachable".
- [ ] 1.4 RED — write the test that `distribution["shardsRoot"]` still equals
      `"MIL-CREDA/Results/Benchmark/shards"` after `axes` is added — `tests/test_search_distribution.py`.
      Satisfies "Non-group declaration keys survive unchanged".
- [ ] 1.5 RED — write `shards.rules_for_axis` unit tests: an undeclared axis refuses; a
      declared axis missing a group refuses (never resolves to `[]`); a flat (non-nested)
      `dist` resolves the seed axis for `axis="seed"` and refuses `axis="transfer"` —
      `tests/test_shards.py`. Each refusal must name what it was asked for.
- [ ] 1.6 RED — write the docstring-singularity regression test: scan
      `config.execution_seed_units.__doc__`, `harness.run_campaign_shard.__doc__`, and the
      `__init__.py` rationale block's source text for a claim of "no second axis" — assert none
      remains once `axes` names two values — `tests/test_search_distribution.py`. Satisfies
      "Rationale that cites the singular axis is corrected".
- [ ] 1.7 Measure fixture churn before editing anything (do not assume the design's claim):
      run the current suite and confirm which fixtures hardcode the flat `distribution` shape.
      Confirm specifically whether `tests/test_bridge.py:190-210` and
      `tests/test_benchmark_declarations.py:160-227` stay green untouched once `rules_for_axis`'s
      flat fallback is exact (D4). Record the actual result; do not weaken any assertion to keep
      a fixture green — if a fixture must change, change the fixture's declaration, not the
      test's meaning.
- [ ] 1.8 GREEN — add the `axes` block with `seed` and `transfer` sub-blocks (all four groups
      each) to `MIL_CREDA_Benchmark.__benchmark__["distribution"]` in
      `src/MIL_CREDA_Benchmark/__init__.py` (block at 481-507), per the Interfaces/Contracts
      shape in design.md. `axis: "seed"` and `shardsRoot` stay exactly where they are.
- [ ] 1.9 GREEN — add the one-line identity projection immediately after the literal (D6):
      `__benchmark__["distribution"].update(__benchmark__["distribution"]["axes"]["seed"])` —
      `src/MIL_CREDA_Benchmark/__init__.py`.
- [ ] 1.10 GREEN — implement `shards.GROUPS` and `shards.rules_for_axis(axis, dist=None)` per
      the D4 shape (declaration-shaped return; refuses undeclared axis, missing group, never
      silently defaults) — `src/MIL_CREDA_Benchmark/shards.py`.
- [ ] 1.11 GREEN — update the two in-repo flat-key readers to select the seed axis explicitly
      via `rules_for_axis("seed", ...)`: `harness.py:2353` and `pooling.py:32`
      (`per_run_dimensions()`). Behaviorally identical today via the projection.
- [ ] 1.12 GREEN — `shards.merge()`'s `dist` default (`shards.py:421`) resolves the seed axis
      via `rules_for_axis`.
- [ ] 1.13 GREEN — rewrite the `__init__.py:392-480` rationale block: remove the "no second
      axis" justification, state the per-axis inversion and the identity-projection argument in
      its place.
- [ ] 1.14 GREEN — correct `harness.run_campaign_shard`'s docstring (2536-2538): remove its
      singular-axis citation. (`config.execution_seed_units`'s docstring is corrected in A2,
      where the function itself changes.)
- [ ] 1.15 Mutation check: replace the D6 `update(...)` with `copy.deepcopy(...)` and confirm
      the identity test from 1.2 reddens. Revert the mutation after confirming.
- [ ] 1.16 Apply the fixture edits identified in 1.7, if any were needed — one edit per
      genuinely-required fixture change, never a weakened assertion.
- [ ] 1.17 Run `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py
      tests/test_shards.py tests/test_benchmark_declarations.py tests/test_bridge.py`. Confirm
      `test_the_declaration_names_the_axis_the_search_can_split_on` is green and no other test
      in this set regressed.

## Work Unit A2 — `config.execution_units_for(axis)`, `transfer_label`, `harness` re-export (PR2)

Implements D1, D3. Spec: `run-unit-axis-resolution`.

- [ ] 2.1 RED — write the no-default-axis test:
      `inspect.signature(config.execution_units_for).parameters["axis"].default is inspect.Parameter.empty`
      — `tests/test_scale_readings.py`.
- [ ] 2.2 RED — write absent-env tests for both axes: `FORGE_RUN_UNITS` unset → `None` for
      `axis="seed"` and for `axis="transfer"` — `tests/test_scale_readings.py`.
- [ ] 2.3 RED — write declared-empty tests for both axes: `FORGE_RUN_UNITS="[]"` → `[]`
      (never `None`) for both axes — `tests/test_scale_readings.py`.
- [ ] 2.4 RED — write the populated-transfer-axis test: a JSON array of valid transfer labels
      (`transfer_label()` output, e.g. `"M->U"`) resolves to `(str, str)` tuples against
      `config.SEARCH_TRANSFERS`; a label not in that set refuses, naming the label and the
      available set — `tests/test_scale_readings.py`.
- [ ] 2.5 RED — write **D1's cross-axis catch**, the test that measures the vocabulary
      decision itself: `execution_units_for("seed")` refuses `["M->U"]` (via the existing
      `int()` parse refusal), and `execution_units_for("transfer")` refuses `["0"]` (not in the
      label map) — `tests/test_scale_readings.py`.
- [ ] 2.6 RED — write the invalid-JSON / non-list / non-string-element refusal tests,
      parametrized over both axes — `tests/test_scale_readings.py`.
- [ ] 2.7 RED — write the seed-axis byte-identical regression test: every input
      `execution_seed_units()` handles today (absent, `"[]"`, populated valid/invalid) produces
      the identical return value or raised exception (including message content) through
      `execution_units_for("seed")` — `tests/test_scale_readings.py`.
- [ ] 2.8 GREEN — move `transfer_label` down to `config.py` as the one definition (layering:
      `config.py` imports nothing from the package) — `src/MIL_CREDA_Benchmark/config.py`.
- [ ] 2.9 GREEN — implement `config.execution_units_for(axis)`: no default; validates `axis`
      against `__benchmark__["distribution"]["axes"]`; preserves the three-way
      absent/empty/populated distinction; validates each element in the named axis's
      vocabulary (seed: `int()`; transfer: `transfer_label()` closed-set map) —
      `src/MIL_CREDA_Benchmark/config.py`.
- [ ] 2.10 GREEN — `execution_seed_units()` becomes `return execution_units_for("seed")`;
      remove its docstring's singular-axis citation (config.py:237-239) —
      `src/MIL_CREDA_Benchmark/config.py`.
- [ ] 2.11 GREEN — `harness.transfer_label = config.transfer_label` (re-export, no wrapper,
      the `shards.disagreements = _shard_io.disagreements` precedent) —
      `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 2.12 Verify all in-repo call sites of `transfer_label` (23, per design's count) still
      resolve as plain calls — run the full suite once, confirm no `AttributeError`/import
      regression.
- [ ] 2.13 Mutation check: remove the `int(unit)` parse from the seed reader path and confirm
      the D1 cross-axis-catch test (2.5) reddens. Revert after confirming.
- [ ] 2.14 Run `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_scale_readings.py
      tests/test_search_distribution.py`. Confirm `test_a_unit_resolves_to_the_axis_the_step_declares`
      is green.

## Work Unit A3 — `shard_paths["searchSlice"]`, `sellar_rebanada`, the `else` slice write (PR3)

Implements D2, D7, D8. Depends on A1 (`axes` must exist for the slice-axis vocabulary
context, though the path logic itself is axis-agnostic). Must land **before** the rest of
D12 (Work Unit A5) — the slice-aware predicate in A5 reads `shard_paths(...)["searchSlice"]`,
which does not exist until this unit lands.

- [ ] 3.1 RED — write path tests: `shard_paths(shard_id)["searchSlice"]` is under
      `results_for(...)/shards/<id>/ceilings.slice.json`; `shard_paths(None)["searchSlice"]`
      lands beside the canonical record (`<stem>.slice.json`); `pilot=True` and a non-zero
      `noise` route to the pilot/noise trees respectively — `tests/test_search_distribution.py`.
- [ ] 3.2 RED — write the integration test: drive `search_ceilings_trials` with a fake
      `run_one` (the shape `tests/test_search_declarations.py:50` already uses), call with a
      `transfers` subset, and assert a readable slice file exists afterward at the
      `searchSlice` path while `config.ceilings_record_for(pilot)` is untouched —
      `tests/test_search_distribution.py`.
- [ ] 3.3 RED — write the roster-exclusion check: the slice-writing code path's source
      (reached via `shard_paths(...)`) does **not** contain `ceilings_record_for`, so it
      correctly stays **outside** the `tests/test_label_noise.py:1253-1274` derived roster —
      i.e. confirm the roster-derivation test does not pick up the new slice writer as a
      third unguarded writer. Add this as an explicit assertion, not an inference from the
      roster staying the same size.
- [ ] 3.4 GREEN — add `SEARCH_SLICE_SUFFIX = ".slice.json"` and a `"searchSlice"` key to
      `shard_paths` per the D2 table (shard id → `shards/<id>/ceilings.slice.json`; `None` →
      beside the canonical record) — `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 3.5 GREEN — implement `sellar_rebanada(found, reduction, shard, transfers)` per D8: same
      family→entry shape `sellar_techos` produces, plus `transfers` (this slice's covered
      labels), `env`/`environment` (this worker's stamp), `revision` (via
      `ceiling_record.stamp`), `shard` (this slice's own id) — `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 3.6 GREEN — convert the guarded write in `search_ceilings_trials` to the `if/else` shape
      (D7): `governs_the_ceilings_record(noise, transfers)` True → write the canonical record as
      today; False → write the slice via `shard_paths(shard, noise=reduction.labelNoise,
      kind=reduction.kind, pilot=reduction.pilot)["searchSlice"]` using `sellar_rebanada` —
      `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 3.7 GREEN — apply the symmetric `else` branch to `search_ceilings`' grid-engine body,
      routed identically to how it already routes its partial file (harness.py:1893-1894) —
      `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 3.8 Mutation check: delete the `else` branch and confirm the 3.2 slice-write test
      reddens. (The companion half of this mutation check — that `run_search`'s PARTIAL
      message reports `ABSENT` when the slice is missing — is re-verified in Task 5.12, after
      A5's message exists.) Revert the mutation after confirming.
- [ ] 3.9 Run `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py
      tests/test_label_noise.py tests/test_search_declarations.py`. Confirm
      `test_every_writer_of_that_record_is_guarded` is still green (roster unchanged in size)
      and the new slice-write tests pass.

## Work Unit A4 — `read_search_slices` + `merge_search_records` (PR4)

Implements D9, D10, D11's default-source half. Spec: `partial-search-record-assembly`.
Depends on A3 (needs `shard_paths["searchSlice"]` and `sellar_rebanada`-shaped slices to read).

- [ ] 4.1 RED — grow `test_partial_search_records_merge_instead_of_overwriting` (the
      committed red) into: six per-shard slices (one per `config.SEARCH_TRANSFERS` transfer)
      assemble into one record naming all six, field-for-field matching the full roster listed
      in `partial-search-record-assembly/spec.md` ("complete set assembles") **plus** the new
      `assembly` key (`{"expected": [...], "shards": [...]}` per D9) —
      `tests/test_search_distribution.py`.
- [ ] 4.2 RED — write **D9's re-pooling mutation test**: build slices whose individual pooled
      `ceiling` winners **all differ** from the union's pooled winner (i.e., no single slice's
      own value equals the assembled answer). Assert the assembled `ceiling` equals the value
      `ceiling_record.choose` produces over the **union** of per-transfer picks — a dict-merge
      or copy-forward implementation must fail this test — `tests/test_search_distribution.py`.
- [ ] 4.3 RED — write refusal #1 (D10 table): a slice carrying no `transfers` field, or an
      empty one, refuses via `shards.ShardIncomplete` naming the slice id —
      `tests/test_search_distribution.py`.
- [ ] 4.4 RED — write refusal #2: a slice whose `ceiling_record.stamp_drift(entry)` is
      non-empty refuses via `shards.ShardsDisagree`, naming the slice id, field, and
      record-vs-current values — `tests/test_search_distribution.py`.
- [ ] 4.5 RED — write refusal #3: slices whose `ceiling_record.kind_of(entry)` differ refuse
      via `shards.ShardsDisagree`, naming both kinds — `tests/test_search_distribution.py`.
- [ ] 4.6 RED — write refusal #4: two slices claiming the same transfer label refuse via
      `shards.ShardsDisagree`, naming the label and both slice ids —
      `tests/test_search_distribution.py`.
- [ ] 4.7 RED — write refusal #5: two slices disagreeing on a transfer-axis
      `identicalAcrossShards` field (e.g. `labelNoise=0.0` vs `labelNoise=0.4`, or `epochs`)
      refuse via `shard_io.disagreements`, naming the disagreeing field and both values —
      `tests/test_search_distribution.py`. Satisfies "Shards that disagree on a field required
      to be identical refuse rather than average".
- [ ] 4.8 RED — write refusal #6 (**the defect this change is named after**): five of six
      expected transfers present refuses via `shards.ShardIncomplete`, naming the **specific**
      sixth missing transfer by label, not a count — `tests/test_search_distribution.py`.
      Satisfies "A missing expected transfer refuses, and names it".
- [ ] 4.9 RED — write refusal #7: a slice covering a label **not** in `expected` refuses via
      `shards.ShardsDisagree`, naming the unexpected label — `tests/test_search_distribution.py`.
- [ ] 4.10 RED — write refusal #8: slices whose present family sets differ refuse via
      `shards.ShardsDisagree`, naming the two family sets — `tests/test_search_distribution.py`.
- [ ] 4.11 RED — write the "coverage judged against expected, not against what arrived" test:
      three individually well-formed, individually complete slices covering three of six
      transfers still refuse citing the three missing ones, even though none of the three
      handed-in slices is itself malformed — `tests/test_search_distribution.py`. Satisfies
      "Coverage completeness is judged against the expected set, not against what was handed
      in".
- [ ] 4.12 RED — write the explicit-smaller-`expected` test: a caller passes `expected=`
      naming fewer than six transfers (e.g. one, a diagnostic re-search); shards covering
      exactly that explicit set assemble successfully; this outcome is reachable **only**
      through the explicit override, never by omitting the argument —
      `tests/test_search_distribution.py`.
- [ ] 4.13 RED — write the order-independence test: pass the six expected labels in reversed
      order; the canonical write still happens (the guard does not depend on `expected`'s
      iteration order) — `tests/test_search_distribution.py`.
- [ ] 4.14 RED — write the **source-level** test that no default-expectation code path in
      `merge_search_records` reads `config.VERDICT_TRANSFERS` — inspect the function's source
      text or bytecode constants, not merely the resulting count (both lists have length six,
      so a length-only assertion would not catch a divergent source) —
      `tests/test_search_distribution.py`. Satisfies "`VERDICT_TRANSFERS` is never the implicit
      source".
- [ ] 4.15 RED — write the canonical-write-guard test: a complete, clean six-transfer assembly
      consults `harness.governs_the_ceilings_record` as part of its write, and that guard
      evaluates `True` for this case by construction; an incomplete or noisy assembly's
      refusal path never attempts the write — `tests/test_search_distribution.py`.
- [ ] 4.16 RED — write the roster-inclusion check: `merge_search_records`'s source contains
      both `ceilings_record_for` and `write_text`, so `tests/test_label_noise.py:1253-1274`'s
      derived roster **does** pick it up, and confirm it also contains
      `governs_the_ceilings_record` (i.e. the existing roster test passes with this new writer
      present, not because the roster was weakened) — this is the explicit companion to Task
      3.3's exclusion check.
- [ ] 4.17 GREEN — implement `harness.read_search_slices(pilot=False, noise=0.0,
      kind="campaign", root=None)`, the I/O half — reads every `*.slice.json` under the
      relevant `shards/` tree — `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 4.18 GREEN — implement `harness.merge_search_records(slices, expected=None, pilot=False,
      reduction=None)` per D9/D10/D11: refusals 1-8 in the D10 table's order, before any write;
      `expected` defaults to `list(config.SEARCH_TRANSFERS)`, normalized into
      `SEARCH_TRANSFERS` order; re-pools via `ceiling_record.choose` over the union (reusing
      `d["best"]` as `value`, exactly as the engine does); carries `identicalAcrossShards`
      fields through after proving agreement (via a `{"shard": id, "stamp": entry}`-shaped
      call to `shard_io.disagreements`); sums `seconds` as total compute; sets
      `env`/`environment` to the assembling machine's; adds the `assembly` key; writes the
      canonical path only behind `governs_the_ceilings_record` —
      `src/MIL_CREDA_Benchmark/harness.py`.
- [ ] 4.19 Mutation check: change the `expected` default from `config.SEARCH_TRANSFERS` to
      `config.VERDICT_TRANSFERS` and confirm the 4.14 **source-level** test reddens (the
      count-based tests will not, by design). Revert after confirming.
- [ ] 4.20 Mutation check: change the re-pool step from `ceiling_record.choose` over the union
      to copying slice 0's `ceiling` forward, and confirm the 4.2 re-pool test reddens. Revert
      after confirming.
- [ ] 4.21 Run `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_search_distribution.py
      tests/test_label_noise.py`. Confirm `test_partial_search_records_merge_instead_of_overwriting`
      is green and `test_every_writer_of_that_record_is_guarded` is still green.

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
