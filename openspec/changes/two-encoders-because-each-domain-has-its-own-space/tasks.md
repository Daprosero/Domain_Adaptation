# Tasks: Two encoders, because each domain has its own space

## Baseline (measured, not assumed)

- Branch `main` at `82ebeea`. Test command: `PYTHONPATH=src .venv/bin/python -m pytest -q`.
- Starting suite state: `3 failed, 660 passed, 1 skipped, 129 subtests passed`.
- **The 3 failures belong to a different change** (`a-transfer-that-never-came-back-is-not-one-nobody-searched`,
  `tests/test_search_distribution.py`, deliberate reds for its own unfinished Phase 6). They are
  **not this change's business** — do not fix them, do not count them as regressions caused by
  this change, and re-report them as still-red-and-still-theirs at every full-suite run below.
- `strict_tdd: true` (inherited). Every implementation task below names its RED step; it must be
  **observed**, never asserted.
- Repository: `/Users/diego/Proyectos/papersmith-ai/implementations/Domain_Adaptation` — its own
  nested git repo. All edits stay inside it. `src/CREDA/**` stays byte-identical throughout (read
  as evidence for Decision 1, never edited). No path outside this repository is an edit target;
  `/Users/diego/Proyectos/papersmith-ai/proposals/research-concept-r25.md` is cited as **(read-only)**
  wherever it appears below, and so is `MIL-CREDA/AGREED.md` (already written, carried not re-decided).

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~560–710 (design's own per-slice estimate: (a) ~30, (b) ~200–260, (c) ~150–200, (d) ~180–220) |
| 400-line budget risk | **High** |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 (Phase 1 / slice a) → PR 2 (Phase 2 / slice b) → PR 3 (Phase 3 / slice c) → PR 4 (Phase 4 / slice d) |
| Delivery strategy | ask-on-risk |
| Chain strategy | pending — **decision owed to the orchestrator before `sdd-apply` starts, not decided here** |

```text
Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: pending
400-line budget risk: High
```

### Suggested Work Units

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|-----------------|-------------------|
| 1 | Revision bump + 11 citations, isolated | PR 1 | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_smoke.py tests/test_benchmark_declarations.py tests/test_report_tables.py tests/test_invariants.py` | N/A — no runtime/GPU behavior changes, string/prose only | Revert `config.REVISION` to `"research-concept-r21.md"`; all 67 records read current again, no data migration |
| 2 | Second encoder + re-pointed gradient/BN tests + `target_accuracy` | PR 2 | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_arm_objectives.py` | N/A — CPU unit/integration tests only, no GPU launch | Revert together with unit 1's tests re-pointing: re-pointed tests without the encoder are red, encoder without them is red — one atom |
| 3 | `latent.py` domain-required seam + AST guard | PR 3 | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_figure_surfaces.py tests/test_latent_sigma_readers.py tests/test_independent_adaptation_schedules.py tests/test_attention_mechanisms.py` | N/A | Revert `latent.py` and its tests; depends on PR 2 (needs `target_embeddings` to exist) |
| 4 | Degeneracy diagnostic + record fields | PR 4 | `PYTHONPATH=src .venv/bin/python -m pytest -q -k "entropy or diagnostic or confidence or kernel_mean"` | N/A | Revert independently; reads matrices `_milcreda_term` already builds, no dependency on PR 3 |

Full suite (`PYTHONPATH=src .venv/bin/python -m pytest -q`, never filtered) is the closing gate for
every PR and for the change as a whole (Phase 5).

---

## Sequencing legend

- **[SEQ]** — must happen in this exact order relative to its siblings; do not reorder.
- **[P]** — independent of its siblings in the same phase; safe to parallelize across files.
- Phase order is load-bearing end to end: **Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5**.
  Phase 1 must land as its own first commit (design Decision 11, spec `revision-bound-records`
  Requirement "The revision bump is its own commit"). Phase 3 depends on Phase 2 (`target_embeddings`
  must exist before `latent.py` can route to it). Phase 4 depends only on Phase 2 (`_milcreda_term`
  already builds `K_tt`/`w_t` regardless of the encoder count) but is sequenced last per the design's
  stated build/revert order. Phase 6 is gated and outside ordinary apply scope (see below).

---

## Phase 0: Pre-flight (sequential, before any `src/` edit)

- [x] 0.1 **[SEQ]** Run `PYTHONPATH=src .venv/bin/python -m pytest -q` on current `main`. Confirm
      `3 failed, 660 passed, 1 skipped, 129 subtests passed` and that the 3 failures are exactly
      `tests/test_search_distribution.py`'s. Record the exact output. *(No spec — baseline evidence.)*
      **DONE**: measured `3 failed, 660 passed, 1 skipped, 129 subtests passed`, all 3 failures in
      `tests/test_search_distribution.py` (`test_the_declaration_names_the_axis...`,
      `test_a_unit_resolves_to_the_axis...`, `test_partial_search_records_merge...`).
- [x] 0.2 **[SEQ]** Record the current `report_digest.source_digest` value before any `src/` byte is
      touched (migration precondition: reverting the `src/` diff must restore this digest and every
      notebook stamp with it). *(Spec: `revision-bound-records` — Notebook stamps requirement;
      Design — Migration/Rollout.)*
      **DONE**: pre-Phase-1 `source_digest` =
      `105756ca3d9a04c0305ea64efb7a89302819bc858a2beadd9bdc4fe99b3c8d4c`.
- [x] 0.3 **[P]** Confirm on disk: 71 files carry `research-concept-r21` (252 occurrences); 67 are
      records (17 `Results/`, 50 `Models/` manifests, all under `.../Pilot/` except
      `Results/Benchmark/ceilings.pilot.json`); 4 are non-record (`AGREED.md` ×2,
      `.implementation/position.jsonl` ×11, `Notebooks/verification.ipynb` ×10,
      `Notebooks/Benchmark_Results.ipynb` ×1). *(Spec: `revision-bound-records` — disposition
      requirements; baseline for Phase 1's count assertions.)*
      **DONE**: measured exactly 71 files / 252 occurrences with this exact split; confirmed 50
      Models manifests (24 Benchmark + 1 curve/rho0 + 1 curve/rho0p2 + 24 Noise/rho0p2) and 17
      Results files, matching the design's count precisely.
- [x] 0.4 **[P]** Confirm `MIL-CREDA/AGREED.md` already carries both required entries: the
      `## Reversed` paragraph dated 2026-09-28 (`8ed79de`) and `:164` (`c55526a`). Read-only —
      `MIL-CREDA/AGREED.md` **(read-only for this change; carried, not re-decided)**.
      **DONE**: both entries present and attributed to the named commits; file left untouched.

---

## Phase 1 — Slice (a): The revision bump, isolated (own commit, FIRST)

Ordering is not negotiable (design Migration/Rollout, Decision 11): the revision refusal must fire
**before** any two-encoder state-dict shape exists, or 67 checkpoints would die on a confusing key
mismatch instead of an honest revision refusal. No behavior change may ride in this commit besides
the revision string and its citations (spec `revision-bound-records` — "The revision bump is its
own commit").

- [x] 1.1 **[SEQ]** Bump `src/MIL_CREDA_Benchmark/config.py:300` `REVISION` from
      `"research-concept-r21.md"` to `"research-concept-r25.md"`.
      *(Spec: `revision-bound-records` — "The revision bump is its own commit".)*
      **DONE**.
- [x] 1.2 **[P]** Bump `src/MIL_CREDA/__init__.py:62` `__implementation__["revision"]` to
      `"research-concept-r25.md"` (enforced `== config.REVISION` by
      `tests/test_benchmark_declarations.py:107`, unchanged).
      *(Spec: `revision-bound-records` — same requirement.)*
      **DONE**; `test_benchmark_declarations.py:107` stayed green throughout (both sides moved
      together).
- [x] 1.3 **[P]** Bump the nine `__provenance__["revision"]` entries to `"research-concept-r25.md"`,
      string only, no `__provenance__["equations"]` change (verified against r25: 7, 8, 9, 10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 28, 29, 37, 38, 39 all carry identical `\tag{}`):
      `src/MIL_CREDA/attention.py:29`, `confidence.py:19`, `conditional.py:18`, `global_term.py:22`,
      `local_term.py:19`, `renyi.py:20`, `kernels.py:17`, `objective.py:21`, `bag_kernel.py:22`.
      *(Spec: `revision-bound-records` — same requirement.)*
      **DONE**; RED observed at `tests/test_smoke.py::test_every_module_declares_provenance`
      (`AssertionError: 'research-concept-r25.md' == 'research-concept-r21.md'`) immediately after
      1.1–1.3, before 1.4 — this IS the declared red-first for the bump.
- [x] 1.4 **[SEQ]** Update `tests/test_smoke.py:38`'s hardcoded
      `assert provenance["revision"] == "research-concept-r21.md"` to `"research-concept-r25.md"`.
      The suite refuses without this line — it is the twelfth line the proposal's "one line" missed.
      *(Spec: `revision-bound-records` — same requirement.)*
      **DONE**; GREEN observed afterward (full suite: only the sibling's 3 reds remained).
- [x] 1.5 Correct the eleven `r21 l.NNN` citations to a stable anchor (equation number or named
      quantity) with the revision moved `r21 → r25`, **not** re-pointed to r25's own 471/516/732 line
      numbers (design Decision 12 rejects that as re-arming the identical trap for r26):
  - [x] 1.5.1 **[P]** `src/MIL_CREDA_Benchmark/tables.py:494` and `:1917` — report prose inside
        f-strings, reaches the rendered report. Reword to cite the named anchor (§3.2 /
        `effective_bag_size`/`uniform_self_similarity`).
        **DONE**; anchor adopted: `r25 §3.2`.
  - [x] 1.5.2 **[P]** `src/MIL_CREDA_Benchmark/tables.py:132` and `:1880` — comment-only, same anchor.
        **DONE**; also corrected the same-comment bare `r21` echo at `tables.py:134` for consistency
        (continuing description of the same `l.501` citation just corrected two lines above).
  - [x] 1.5.3 **[P]** `src/MIL_CREDA_Benchmark/config.py:453` and `:472` — comment-only, anchor Eq. (23)
        / Eq. (15).
        **DONE**; `:453` → `r25 Eq. (15)`, `:472` → `r25 Eq. (23)` (comment reflowed to keep line
        width consistent).
  - [x] 1.5.4 **[P]** `src/MIL_CREDA_Benchmark/config.py:474` — the **bare** `l.458` with no `r21`
        prefix (a `r21 l\.` search misses this one; scope the pass by the derived condition "any
        `l.NNN` in prose", not by the observed list). Anchor: Eq. (15).
        **DONE**; → `r25 Eq. (15)`. Verified with `rg 'l\.[0-9]{3}'` that no bare `l.NNN` remains in
        any touched file.
  - [x] 1.5.5 **[P]** `src/MIL_CREDA_Benchmark/wiring.py:260` — comment-only, anchor Eq. (23).
        **DONE**; → `r25 Eq. (23)`.
  - [x] 1.5.6 **[P]** `tests/test_invariants.py:630` — docstring only, anchor Eq. (15).
        **DONE**; → `r25 Eq. (15)`.
  - [x] 1.5.7 **[SEQ, depends on 1.5.1]** `tests/test_report_tables.py:1315` and `:1328` — these pin
        the literal substring the rendered report must contain. Update the pinned string to name
        whatever anchor 1.5.1 adopts (this pair and `tables.py:494`/`:1917` are **one atom**: prose
        without the updated assert reds the suite silently-wrong, assert without the prose reds it
        honestly).
        **DONE**; both asserts changed `"l.501" in ...` → `"§3.2" in ...`, landed with the prose in
        the same commit.
  - [x] 1.5.8 **[P]** `tests/test_report_tables.py:1302` and `:1316` — docstring/comment prose only,
        same anchor as 1.5.1.
        **DONE**; also corrected the same-comment bare `r21` echo at `:1317` for consistency (same
        citation, same paragraph).
      *(Spec: `revision-bound-records` — carried by "the revision bump is its own commit"; not a
      separate requirement, but the design places all eleven inside this commit because
      `report_digest.source_digest` cost is paid once regardless.)*
- [x] 1.6 **[SEQ]** Add or verify a regression test that `latent.stale_revisions()` reports exactly
      67 records tagged `"research-concept-r21.md"` (17 `Results/`, 50 `Models/`) and reports no
      `"research-concept-r25.md"`-tagged record as stale, once `config.REVISION` is bumped.
      *(Spec: `revision-bound-records` — "`latent.stale_revisions` reports exactly the invalidated
      set".)*
      **DONE with a documented deviation.** Added
      `tests/test_latent_binding.py::test_the_full_on_disk_pilot_tree_reports_exactly_67_r21_stamped_records_stale`,
      built from raw on-disk JSON (50 real `.manifest.json` via `rglob`, 17 real `Results/` files by
      exact path) rather than through `latent.available()`. **Discovered defect**: calling
      `latent.available(0.0, True)` against the REAL Pilot tree now raises an uncaught `SystemExit`
      — `latent._current_hyper`'s `except Exception:` does not catch `SystemExit` (a `BaseException`),
      and `config._refuse_on_stamp_drift` raises `SystemExit` reading `ceilings.pilot.json` (itself
      one of the 67 r21-stamped records) once `config.REVISION` no longer matches it. This is
      pre-existing and newly surfaced by this exact bump — no existing test calls `available()`
      against real on-disk data (all use `monkeypatch`ed `tmp_path`). Fixing it is an unauthorized
      behavior change for this isolated commit; reported as a risk, not fixed. GREEN observed for the
      new test; RED observed and reverted under the mutation "revert `config.REVISION` to r21".
- [x] 1.7 **[SEQ]** Verify no on-disk record's bytes changed (`git status`/`git diff --stat` on
      `MIL-CREDA/Results/` and `MIL-CREDA/Models/` shows nothing) and that the three existing guards
      (`harness.py:393-403` `StaleCheckpointRevision`, `latent.py:393-397` `available()`,
      `latent.py:113` `currentRevision: False`) fire unmodified on the 67 records.
      *(Spec: `revision-bound-records` — "disposed of by refusal, never by mutation".)*
      **DONE, with one guard's finding qualified.** `git diff --stat` empty for both paths — no byte
      changed. Guard 1 (`StaleCheckpointRevision`) fires correctly against a real Results/ record.
      Guard 3 (`latent.py:113`'s `currentRevision` expression) evaluates `False` correctly against a
      real r21 manifest. **Guard 2 (`available()`) does NOT fire gracefully against real data** — see
      1.6's discovered defect: it crashes with `SystemExit` before it can tag any entry, rather than
      completing and reporting `currentRevision: False` for the 67. The tagging expression itself
      (guard 3) is correct; `available()`'s own execution against real Pilot data is blocked by the
      unrelated ceilings-drift crash.
- [x] 1.8 **[SEQ]** Run `PYTHONPATH=src .venv/bin/python -m pytest -q` (full suite). Confirm only
      the sibling's 3 pre-existing reds remain red, and every test touched by 1.1–1.7 is green.
      Commit as an isolated commit containing only 1.1–1.5 (source + prose) and its accompanying test
      updates (1.4, 1.6, 1.5.7, 1.5.8) — **no other behavior change**.
      *(Spec: `revision-bound-records` — "The revision bump is its own commit"; "disclosed as
      re-owing a walk, never as destroying a result" — write the commit message so it states no
      quotable result is lost, per `AGREED.md:187`/`:208`/`:217`, and that it re-owes the pilot walk
      and notebook re-execution rather than presenting the bump as discarding a finished result.)*
      **DONE**; full suite: `3 failed, 661 passed, 1 skipped, 129 subtests passed` (the 3 failures
      are exactly the sibling's `tests/test_search_distribution.py` reds, unchanged; +1 passed vs
      baseline is the new 1.6 test). Committed as `b25da2f`
      `build(revision): bump the managed revision from r21 to r25, isolated first` — 17 files
      (13 `src/`, 4 `tests/`), 111 insertions / 33 deletions, exactly 1.1–1.5 + 1.4/1.6/1.5.7/1.5.8.

---

## Phase 2 — Slice (b): The second encoder (depends on Phase 1 landed)

`strict_tdd` red-first sequence, written out because the tempting shortcut (change production,
watch a test go red, then "fix" the test) is green-first with extra steps and is Risk R5 from the
design (an assertion gets weakened instead of re-pointed).

### 2.1 Red-first: re-point the three existing shared-encoder tests before any production change

- [ ] 2.1.1 **[SEQ]** Run the suite, confirm `tests/test_arm_objectives.py:557`
      (`test_an_adapted_arms_target_forward_updates_running_statistics`) is **green** under one
      encoder today. Record the count. *(Spec: `per-domain-encoders` — "BatchNorm running statistics
      no longer cross domains".)*
- [ ] 2.1.2 **[SEQ]** Re-point `:557`'s assertion from `assert all(module is arm.encoder for module
      in seen)` to a **partition**: `seen` contains both encoders; the source forward routed through
      `arm.encoder`; the target forward routed through `arm.target_encoder`; and
      `arm.encoder is not arm.target_encoder`. Move the warm-up (lines 583-585) and the
      `before_mean`/`before_var` snapshot to `arm.target_encoder.bn`. Run the suite. **Observe RED**
      (production still holds one encoder — this is the change's declared first red, satisfying
      `strict_tdd` without authoring a new failing test).
      *(Spec: `per-domain-encoders` — "BatchNorm running statistics no longer cross domains".)*
- [ ] 2.1.3 **[SEQ]** Re-point `tests/test_arm_objectives.py:496`
      (`test_an_adapted_arms_target_forward_carries_gradient_to_the_encoder`) into
      `..._carries_gradient_to_both_encoders`: split the single assertion over
      `arm.encoder.parameters()` into two separate assertions, one over
      `arm.encoder.parameters()` and one over `arm.target_encoder.parameters()`. Replace the
      docstring's now-false premise ("Eq. (39)'s whole point is one encoder both domains' inputs
      update") with the three citations from design Decision 6a
      (`global_term.py:59-60`, `local_term.py:85-101`, r25:1087 **(read-only, parent forge)**).
      Run the suite. **Observe RED**.
      *(Spec: `per-domain-encoders` — "Adaptation's gradient reaches both encoders".)*
- [ ] 2.1.4 **[SEQ]** Re-point `tests/test_arm_objectives.py:617`
      (`test_every_adapted_arms_full_training_step_updates_running_statistics_a_source_only_twin_does_not`)
      into a **pair**: (a) inverted — `assert torch.equal(floor.encoder.bn.running_mean,
      adapted.encoder.bn.running_mean)` and the same for `running_var`; (b) new — `assert not
      torch.equal(adapted.encoder.bn.running_mean, adapted.target_encoder.bn.running_mean)`. Keep
      the twin comparison — deleting it is Risk R5 and loses the only assertion distinguishing the
      two encoders inside a full `training_step`. Run the suite. **Observe RED**.
      *(Spec: `per-domain-encoders` — "BatchNorm running statistics no longer cross domains".)*

### 2.2 Red-first: write the new construction/gradient/optimizer tests before production exists

- [ ] 2.2.1 **[P]** Write a test asserting E/F/G each hold two `FeatureExtractor` instances after
      construction, and no parameter object of one appears in `named_parameters()` of the other —
      identity (`is`), never equal shapes or values. **Observe RED** (attribute does not exist yet).
      *(Spec: `per-domain-encoders` — "Two independent encoders for every adapted arm, one for the
      floor".)*
- [ ] 2.2.2 **[P]** Write a test asserting the floor `B` holds exactly one `FeatureExtractor`
      instance (`arm.target_encoder is None`, or equivalent). **Observe** current behavior (may
      already pass trivially if the attribute is absent — confirm it fails for the right reason once
      the attribute is introduced elsewhere, or add an explicit `hasattr`/`is None` check that reds
      until Decision 3/4 lands).
      *(Spec: `per-domain-encoders` — "The floor arm... MUST hold exactly one encoder".)*
- [ ] 2.2.3 **[P]** Write a test asserting `target_encoder.output_dim == encoder.output_dim` for an
      adapted arm. **Observe RED**.
      *(Spec: `per-domain-encoders` — "Both encoders share the same codomain".)*
- [ ] 2.2.4 **[P]** Write a construction-order test: B's and an adapted arm's head weights are
      bit-identical after `Arm.__init__` under the same seed; the test must **red** if the target
      encoder is constructed before the head (verifies Decision 3 — building the target encoder
      LAST keeps the `B→E` rung to exactly one difference). **Observe RED or absent-attribute
      failure** before Decision 3/4 lands.
      *(Spec: `per-domain-encoders` — carries the rung-attributability constraint from design
      Decision 3; no direct spec requirement text, but required by S4/`config.py:944-946`.)*
- [ ] 2.2.5 **[P]** Write/confirm a test that patching `wiring.FeatureExtractor` with a stub before
      constructing an adapted arm makes **both** the source and target encoder instances of the
      stub. **Observe RED** (target encoder does not exist).
      *(Spec: `per-domain-encoders` — "The target encoder is constructed through the module-level
      `FeatureExtractor` name".)*
- [ ] 2.2.6 **[P]** Write a test asserting `parameters` reads `11,247,434` for `B` and `22,423,946`
      for an adapted arm, measured from `harness.py`'s own `sum(p.numel() for p in
      model.parameters())` expression. **Observe RED** (current count is the one-encoder count).
      *(Spec: `per-domain-encoders` — "The parameter asymmetry is reported where the count is read".)*
- [ ] 2.2.7 **[P]** Write a test asserting every `target_encoder` parameter appears in the same
      `torch.optim.Adam` instance as `encoder`'s parameters, under one shared `lr`. **Observe RED**.
      *(Spec: `per-domain-encoders` — "One optimizer trains both encoders".)*

### 2.3 Production: build the second encoder and the seams around it

- [ ] 2.3.1 **[SEQ]** In `src/MIL_CREDA_Benchmark/wiring.py`, construct `self.target_encoder` through
      the **module-level** `FeatureExtractor` name, inside `Arm.__init__`, **after** the head and the
      `V_R`/`b_R`/`v_R` attention parameters (last) — not beside `self.encoder`, not via
      `copy.deepcopy`, not via a local alias or second import.
      `self.target_encoder = FeatureExtractor(backbone=config.BACKBONE,
      pretrained=config.PRETRAINED) if spec["adaptation"] is not None else None`.
      *(Spec: `per-domain-encoders` — "Two independent encoders...", "The target encoder is
      constructed through the module-level `FeatureExtractor` name". Design Decisions 3–4.)*
- [ ] 2.3.2 **[SEQ]** Promote the private `_target_embeddings` (`wiring.py:221-240`) to the public
      `target_embeddings`, routing through `self.target_encoder` (falling back to `self.encoder` on
      the floor, where `target_encoder is None`). Remove the private name — do not keep both. Update
      the five docstrings that still say `_target_embeddings`
      (`tests/test_arm_objectives.py:575`, `:619`, `:628`, `:636`, `:638`).
      *(Spec: `per-domain-encoders` — "No embedding without a declared domain"; Design Decision 2.)*
- [ ] 2.3.3 **[SEQ]** Extract `_scores(embeddings)` from `forward`'s existing bag/instance unit branch
      (`wiring.py:205-209`); `forward` becomes `_scores(instance_embeddings(bags))`, signature
      **unchanged**. Add `score_target(bags)` = `_scores(target_embeddings(bags))`.
      *(Spec: `per-domain-encoders` — "No embedding without a declared domain"; Design Decision 1
      interfaces.)*
- [ ] 2.3.4 **[SEQ]** Add `harness.target_accuracy(model, dataset, device)` beside `harness.accuracy`
      (which stays **unchanged**, signature included). Move `harness.py:691` and `:849` (the
      `targetAccuracy` call sites) to `target_accuracy`; leave `harness.py:690` and `:848`
      (`sourceAccuracy`) untouched.
      *(Spec: `per-domain-encoders` — "Source accuracy is measured through the source encoder,
      target accuracy through the target encoder".)*
- [ ] 2.3.5 **[P]** Correct prose that asserted a shared encoder: `wiring.py:4` module docstring,
      `config.py:886-892` and `:902-905` (`normalization` axis rationale), `__init__.py:477-480`
      (the `poolable` "bit-identical across every run" justification — now false across arms, true
      within an arm; the `parameters` declaration itself at `:383`/`:484` is unaffected).
      *(Spec: `per-domain-encoders` — "Prose that asserted a shared encoder is corrected with it".)*
- [ ] 2.3.6 **[SEQ]** Run the full suite. **Observe GREEN** for `:496`, `:557`, `:617`, and every test
      from 2.2.1–2.2.7. Confirm `tests/test_arm_objectives.py:371` and `:461` (the floor's guards)
      pass with **zero byte changed** — read back both tests' bytes to confirm.
      *(Spec: `per-domain-encoders` — "The floor's guarantee is unchanged".)*

### 2.4 Verify the mechanically-affected patch sites (no assertion change expected here)

- [ ] 2.4.1 **[P]** `tests/test_figure_surfaces.py:651` — confirm the `_TinyEncoder` patch now stubs
      both encoders.
- [ ] 2.4.2 **[P]** `tests/test_figure_surfaces.py:840` — confirm the `trained` fixture's `_Encoder`
      patch stubs both.
- [ ] 2.4.3 **[P]** `tests/test_figure_surfaces.py:917` — confirm `model.encoder.output_dim` is still
      `θ_s`'s and unedited; **add** the codomain assertion
      `model.target_encoder.output_dim == model.encoder.output_dim` at this exact site (the one
      place the two widths meet).
- [ ] 2.4.4 **[P]** `tests/test_independent_adaptation_schedules.py:240` — confirm the patch stubs
      both.
- [ ] 2.4.5 **[P]** `tests/test_latent_sigma_readers.py:56` — confirm the patch stubs both.
- [ ] 2.4.6 **[P]** `tests/test_attention_mechanisms.py:315` — confirm the patch stubs both. (This
      site was absent from the original handoff's list — do not skip it.)
      *(Spec: `per-domain-encoders` — "The target encoder is constructed through the module-level
      `FeatureExtractor` name" — this is that requirement's own stated proof obligation.)*

### 2.5 Integration test: evaluation routes through the right encoder

- [ ] 2.5.1 **[SEQ]** Write an integration test: perturb `θ_t`'s parameters on a loaded adapted arm
      — `targetAccuracy` (`target_accuracy`) moves, `sourceAccuracy` (`accuracy`) does not; then
      perturb `θ_s` and assert the mirror.
      *(Spec: `per-domain-encoders` — "A mixed-up domain is detectable, not silent".)*
- [ ] 2.5.2 **[P]** Write an integration test: on the floor `B`, `accuracy` and `target_accuracy`
      agree (one encoder); on an adapted arm (e.g. `G`), they disagree once `θ_t` has trained.
      *(Spec: `per-domain-encoders` — "Source accuracy is measured through the source encoder, target
      accuracy through the target encoder".)*

- [ ] 2.6 **[SEQ]** Run `PYTHONPATH=src .venv/bin/python -m pytest -q` (full suite). Confirm only the
      sibling's 3 reds remain and everything in Phase 2 is green. Commit as its own commit(s), atomic
      with the re-pointed tests (encoder without tests is red; tests without encoder are red).

---

## Phase 3 — Slice (c): `latent.py` names its domain (depends on Phase 2 landed)

- [ ] 3.1 **[SEQ]** Add `wiring.SIDES = ("source", "target")` (not `DOMAINS` — `config.DOMAINS`
      already means the three datasets M/U/S). Add the refusal idiom `raise ValueError(f"unknown
      domain {domain!r}; known: {SIDES}")`, matching `mechanism_weights`'s existing idiom.
      *(Spec: `per-domain-encoders` — "No embedding without a declared domain".)*
- [ ] 3.2 **[SEQ]** Write RED tests first for the domain-required contract: omitting `domain` on
      `represent`/`attention_spread` raises `TypeError` (no default anywhere in the signature); an
      unknown string raises `ValueError` naming `SIDES`. **Observe RED** (current signatures still
      lack the required parameter).
      *(Spec: `per-domain-encoders` — "A caller that cannot state a domain is refused".)*
- [ ] 3.3 **[SEQ]** Change `latent.represent(model, bagset, positions, device, domain, unit=None)` —
      `domain` required, no default, refuses on anything but `wiring.SIDES`.
- [ ] 3.4 **[SEQ]** Change `latent.attention_spread(model, bagset, positions, device, domain)` — same
      contract.
- [ ] 3.5 **[SEQ]** Update `correspondence` (`latent.py:601`/`:602`) — the two adjacent lines name
      `instance_embeddings`/`target_embeddings` explicitly (not a string argument — this is the
      copy-paste-adjacent-pair shape design Decision 1 flags).
- [ ] 3.6 **[SEQ]** Update `bag_pairs` (`latent.py:920`/`:921`) — same two-method-name treatment.
      This is what makes `tests/test_figure_surfaces.py:896-897`'s rebuild agree again once
      re-pointed (task 3.9).
- [ ] 3.7 **[SEQ]** Update `analyse` (`latent.py:658`/`:659`, `:671`), `latent_grid` (`:868`,
      `:870-871`), and `correspondence_grid` to pass the domain explicitly at each call.
      Run the suite. **Observe GREEN** for 3.2's new tests.
      *(3.3–3.7 spec: `per-domain-encoders` — "No embedding without a declared domain"; "Every
      post-hoc reader in `latent.py` names its domain explicitly".)*
- [ ] 3.8 **[SEQ]** Confirm `tools/measure_kernel_sigma.py` is **not edited** — it calls
      `instance_embeddings` on a freshly constructed model where both encoders are the same
      pretrained resnet18, reproducing `config.KERNEL_SIGMA = 36.135014304860874` byte-for-byte.
      Read-only verification task. *(No spec — explicitly out of scope, design Decision 7.)*
- [ ] 3.9 **[SEQ]** Re-point `tests/test_figure_surfaces.py:896-897`
      (`test_the_nearest_source_bag_is_found_...`) at `:897` from `model.instance_embeddings` to
      `model.target_embeddings` for the target side. Before 3.5/3.6 land, confirm it is still
      **green** (domain-blind `bag_pairs` still agrees with the domain-blind rebuild). After 3.5/3.6
      land, **observe it red** if left unedited, then re-point and **observe GREEN**.
      *(Spec: `per-domain-encoders` — "Every post-hoc reader in `latent.py` names its domain
      explicitly"; Design Decision 6e.)*
- [ ] 3.10 **[SEQ]** `tests/test_figure_surfaces.py:963` — reword the docstring premise from "both
      domains go through ONE embedding call" to "one projection call over one shared codomain
      `R^d`". **Add** the assertion `model.encoder is not model.target_encoder` so the new claim is
      measurable. Leave the existing assertions (`count == config.CLASSES * 2`, one `_embed` call
      per panel) **unedited** — they pass without modification (design Decision 6d, spec scenario
      "The bag-figure projection test's premise survives, resolved").
      *(Spec: `per-domain-encoders` — same scenario.)*
- [ ] 3.11 **[SEQ]** **Observe, do not assume**: run `tests/test_figure_surfaces.py:1002`'s
      abscissa-overlap assertion under two independently-initialized stub encoders. If it fails,
      replace the overlap heuristic with the structural claim already asserted at `:986`/`:992` (one
      `_embed` call carrying both domains) — **do not reseed the fixture** until it passes, which
      would make the assertion a description of a seed rather than a property.
      *(No spec — named as an observation obligation in design Testing Strategy and Open Questions.)*
- [ ] 3.12 Add the two-layer domain-mixup guard:
  - [ ] 3.12.1 **[SEQ]** Behavioral, one assertion per consumer: perturb `θ_t`'s parameters on a
        loaded adapted arm; assert every target-side reading (`represent`, `attention_spread`,
        `correspondence`, `bag_pairs`) **moves** and every source-side reading does **not**; then
        perturb `θ_s` and assert the mirror. Never infer one consumer's guard from a sibling's.
        *(Spec: `per-domain-encoders` — "A caller that names its domain is served by the matching
        encoder is served"; "A mixed-up domain is detectable, not silent".)*
  - [ ] 3.12.2 **[SEQ]** Structural, derived rather than listed: an AST walk asserting no module
        under `src/MIL_CREDA_Benchmark/` **other than `wiring.py`** calls `instance_embeddings` —
        precedent `tests/test_figure_surfaces.py:673-688`. The only expected `src/` call sites after
        this change are `wiring.py:204` (`forward`, source by construction) and `wiring.py:354`
        (`training_step`, source by context). This condition is derived (any call site, anywhere),
        not a roster, so it reddens on a **new** site the behavioral guard structurally cannot see.
        *(Spec: `per-domain-encoders` — "No embedding without a declared domain".)*
- [ ] 3.13 **[SEQ]** Run `PYTHONPATH=src .venv/bin/python -m pytest -q` (full suite). Confirm only the
      sibling's 3 reds remain. Commit.

---

## Phase 4 — Slice (d): The degeneracy diagnostic (depends on Phase 2; independent of Phase 3)

The diagnostic must be able to **fall** — a reading that cannot move measures nothing — and it has
**two** raise sites at the degenerate boundary, not one.

- [ ] 4.1 Write RED tests first for the diagnostic's math and boundary, before wiring any of it into
      `training_step`:
  - [ ] 4.1.1 **[P]** A rank-one, trace-normalized target bag Gram reads the diagnostic's minimum
        (`‖A‖²_F = 1` ⇒ `H₂ = 0`), exactly, not approximately.
        *(Spec: `target-degeneracy-diagnostic` — "The reading decays toward its minimum..." scenario
        "A rank-one target Gram reads the diagnostic's minimum".)*
  - [ ] 4.1.2 **[P]** A sequence of target batches with decreasing effective rank of `K_tt` produces
        a non-increasing sequence of readings.
        *(Spec: `target-degeneracy-diagnostic` — same requirement, scenario "falls monotonically".)*
  - [ ] 4.1.3 **[P]** A synthetic all-zero `K_tt` (non-positive trace, `renyi.trace_normalize`'s
        `ValueError`) is **reported** (a value or an explicit "unavailable"/NaN), never propagates a
        `ValueError` out of the diagnostic's call site.
        *(Spec: `target-degeneracy-diagnostic` — "The diagnostic reports rather than raises at the
        fully degenerate boundary", scenario "non-positive-trace...is reported, not raised".)*
  - [ ] 4.1.4 **[P]** A synthetic `trace_normalize(K_tt)` with zero Frobenius norm
        (`renyi.quadratic_entropy`'s `ValueError`) is likewise **reported**, never propagated.
        *(Spec: `target-degeneracy-diagnostic` — same requirement, scenario "zero-Frobenius-norm...
        is reported, not raised". Both boundaries MUST be tested — the two are distinct raise sites.)*
  - [ ] 4.1.5 **[P]** A target batch constructed so every bag's hard pseudo-label collapses to the
        same class (driving `C_act` toward one class) while the unconditional target cloud is also
        collapsing: the required diagnostic falls toward its minimum while the per-class weighted
        `H_target` inside `global_term.dependency_score` fails to characterize the collapse (at most
        one active class left).
        *(Spec: `target-degeneracy-diagnostic` — "The per-class weighted `H_target`...MUST NOT
        substitute for the diagnostic".)*
  - [ ] 4.1.6 **[P]** A target batch with a collapsing bag cloud but non-uniform intra-bag attention
        weights: `uniform_self_similarity`/`effective_bag_size` report no collapse on it — they must
        stay **unwired**, not presented as equivalent or substitute.
        *(Spec: `target-degeneracy-diagnostic` — "The excluded intra-bag instruments MUST NOT be
        used".)*
- [ ] 4.2 **[SEQ]** Implement `_target_bag_entropy(K_tt)` beside the existing `K_ss, K_st, K_tt =
      bag_kernel_matrix(...)` build in `wiring.py`'s `_milcreda_term` (`:267-269`):
      `quadratic_entropy(trace_normalize(K_tt))` over the **unconditional** target bag Gram, catching
      both `ValueError`s from `renyi.trace_normalize` (non-positive trace) and
      `renyi.quadratic_entropy` (zero Frobenius norm) at the call site and returning `float("nan")`
      instead of propagating. Run tests 4.1.1–4.1.4. **Observe GREEN**.
      *(Spec: `target-degeneracy-diagnostic` — "The diagnostic is quadratic entropy over the
      trace-normalized unconditional target bag Gram".)*
- [ ] 4.3 **[SEQ]** Add the remaining four readings and one count beside `w_t = confidences(G_t)`
      (`:273`) and the `losses` loop (`:278-293`): `targetConfidenceMean = w_t.mean()`,
      `targetConfidenceMin = w_t.min()` (the onset — a mean over ten bags hides one vanished
      weight), `activeClasses = len(losses)`, `sourceKernelMean`/`crossKernelMean`/`targetKernelMean`
      = the off-diagonal means of `K_ss`/`K_st`/`K_tt`. Run test 4.1.5. **Observe GREEN**.
      *(Spec: `target-degeneracy-diagnostic` — "The confidence weights' magnitude is recorded".)*
- [ ] 4.4 **[SEQ]** Have `training_step` return these six readings beside the existing
      `supervised`/`adaptation`/`contribution*` floats. On the floor `B`, all six report
      `float("nan")`, **never `0.0`** (0.0 is the collapsed value itself). Add a mutation-guard test:
      a `0.0` mutation on the floor's readings must **red**.
      *(Spec: `target-degeneracy-diagnostic` — boundary/reporting requirements, applied to the
      floor's structural absence of the reading.)*
- [ ] 4.5 **[SEQ]** Aggregate the six readings per epoch into `epochs_record` beside
      `sourceAccuracy`/`targetAccuracy` (means over the epoch's 7 steps; `min` where the minimum is
      informative, per the existing `curve` → `mean_supervised` pattern). Add
      `run["targetBagEntropyMin"]` (the minimum over epochs) as the one per-run field — **not** a
      new `config.DIMENSIONS` entry and **not** added to `__benchmark__["report"]["dimensions"]`.
      *(Spec: `target-degeneracy-diagnostic` — "The diagnostic is readable during training, not only
      after".)*
- [ ] 4.6 **[P]** Write a structural test confirming `targetBagEntropy`/`targetBagEntropyMin` and the
      other five readings are **absent** from `config.DIMENSIONS` and from
      `__benchmark__["report"]["dimensions"]` (design Decision 8's rejection, mirrored as a guard so
      a later PR does not silently re-add it as a declared dimension).
      *(No direct spec scenario — design-level guard for Decision 8.)*
- [ ] 4.7 **[P]** Write a trace test: reading the diagnostic at ≥2 distinct points over a multi-epoch
      training run yields distinct values (trajectory, not only a final number).
      *(Spec: `target-degeneracy-diagnostic` — "The diagnostic is available at more than one point
      across a training run".)*
- [ ] 4.8 **[P]** Write a round-trip test: the run dict (with `NaN` present on the floor) survives
      `json.dumps`/`runs.jsonl` writing, precedent `latent.geometry`/`harness.accuracy`.
      *(No direct spec scenario — regression precedent named in design Testing Strategy.)*
- [ ] 4.9 **[SEQ]** Run `PYTHONPATH=src .venv/bin/python -m pytest -q` (full suite). Confirm only the
      sibling's 3 reds remain. Commit.

---

## Phase 5 — Full regression and success-criteria audit (sequential, closes the change)

- [ ] 5.1 **[SEQ]** Run the **full, unfiltered** suite:
      `PYTHONPATH=src .venv/bin/python -m pytest -q`. Record the real numbers. Confirm the sibling
      change's 3 `tests/test_search_distribution.py` reds are still red and are reported as its work,
      not as regressions introduced here.
- [ ] 5.2 **[SEQ]** Walk `proposal.md`'s Success Criteria checklist item by item against the landed
      state; mark none as satisfied without having run the measurement it names (parameter counts,
      gradient-both-sides, `:557`/`:617` re-pointed assertions, `KERNEL_SIGMA` unchanged,
      `hyperparameter_drift` silent, diagnostic behavior, `latent.stale_revisions` count, full suite).
- [ ] 5.3 **[P]** Confirm `src/CREDA/**` is byte-identical (`git diff --stat` empty for that path) and
      that no file outside `implementations/Domain_Adaptation` was modified.
- [ ] 5.4 **[P]** Confirm the report states the parameter asymmetry wherever `parameters` is printed
      (carries `MIL-CREDA/AGREED.md:164`, already written — verify the sentence survives in
      `tables.py` after the Phase 1 citation edits, rather than being dropped by an unrelated diff).
      *(Spec: `per-domain-encoders` — "The parameter asymmetry is reported where the count is read".)*
- [ ] 5.5 **[P]** Re-confirm `MIL-CREDA/AGREED.md` **(read-only)** still carries both required entries
      unchanged (this change never edits it).

---

## Phase 6 — Gated: pilot walk and nine-account rehearsal (NOT ordinary apply scope)

**Do not execute without explicit, separate operator authorization.** This spends metered remote
quota and is scoped identically to the sibling change's own Phase 6 gate. The bump un-ticks the
six-step position and the first `src/` byte this change touches invalidates every notebook stamp
independently of the revision, via `report_digest.source_digest` — so the walk is **owed**, but this
change delivers the launch precondition, never the launch itself.

**The operator has asked to discuss results one by one BEFORE the pilot.** That discussion is a
precondition for 6.1–6.3, not a step inside them.

- [ ] 6.1 **[GATED — requires explicit operator authorization]** Re-run the six-step pilot walk under
      the existing launch-authorization mechanism (this change does not create or modify that
      mechanism).
- [ ] 6.2 **[GATED — requires explicit operator authorization]** Re-rehearse the nine-account
      benchmark under r25.
- [ ] 6.3 **[GATED — requires explicit operator authorization]** Re-execute `verification.ipynb` and
      `Benchmark_Results.ipynb` under r25 (both carry the literal r21 revision string and must be
      re-executed before their stamps can be current; every other notebook's stamp is already
      invalidated by the source-digest mechanism and needs no special-casing beyond re-execution).

---

## Cross-cutting discipline (applies to every phase above)

- Hold the sibling change's own bar: across its five slices, `tests/` gained 979 insertions against
  20 deletions and no assertion was weakened. No task above may satisfy a red by weakening an
  assertion instead of re-pointing it — `:617`'s inversion (2.1.4) is the named risk case (R5).
- `verify`'s `prose.unresolvedSymbols` reads `inspect.getsource()`, docstrings included — a source-
  substring roster test. Reword prose rather than merely mentioning a constant when touching
  `wiring.py:4`, `config.py:886-905`, `__init__.py:477-480` (2.3.5) and the eleven citations (1.5.*).
- Do not pin a constant belonging to a retired path (`SEARCH_ENGINE` is `optuna`; `CEILING_GRID` and
  `SEARCH_SEEDS` belong to the retired grid engine) anywhere in new test code.
- Run the **full** suite before reporting at every phase boundary, never a filtered subset.
