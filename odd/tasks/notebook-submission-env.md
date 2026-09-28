# notebook-submission-env

## Objective

Let a remote submission tell an executed notebook, per submission, what scale it
runs at and which shard of the split axis it owns — so the same pinned commit
can carry a rehearsal, a full-scale run, and one shard per worker, without a
file being edited between them.

## Problem

MIL-CREDA's four remote steps each own a notebook, and the operator's mission is
that the worker execute those notebooks rather than a library callable. Two
measured facts block that today:

- **Scale.** Every notebook opens with `ES_ENSAYO = config.is_pilot_scale()`,
  and `config.EPOCHS = 3` / `config.SEEDS = [0]` make that `True`. A
  notebook-shaped job at any current pin runs the pilot on the worker and writes
  to the pilot tree. The callable shape does not have this problem
  (`run_campaign_shard(pilot=False)` builds its `Reduction` from
  `FULL_EPOCHS`/`FULL_SEEDS`), which is why the conversion silently downgrades
  the run.
- **Shard.** No notebook passes `shard=`. The campaign notebook says so in its
  own prose: the seed split "se pide por el mismo callable, con su `shard`,
  desde un `run-config.json`" — i.e. only through the callable shape's
  `run.kwargs`. `generate-job` has no flag that sets a process environment
  variable for the run.

Consequence if converted as-is: the whole grid on one machine at pilot scale.
Projected from the measured pilot walk (2026-09-24, this Mac, epochs 3→20,
seeds 1→30; `Reduction` takes bag counts from `config` with no dial, so material
does not change between scales): campaign ≈ 22 h, mechanisms ≈ 32 h. Kaggle caps
a GPU session at 12 h (service knowledge, not measured here), so it would not
finish either.

## Why this shape

The target already carries the precedent, with its reasoning written down:
`REHEARSAL_ENV` is an environment variable and not a module constant, because a
rehearsal "no es un modo del REPOSITORIO sino de UNA ejecución: el mismo commit
clonado en el worker corre primero el ensayo y después la corrida real, y un
archivo editado entre las dos serían dos pines distintos". Scale and shard are
that same class of decision and today live in the file dial, which is a mode of
the repository. This extends the existing pattern rather than inventing one.

## Scope

**Authorized.** The forge's `remote-execution` skill (generate-job's per-run
environment, the runner that must export it, their tests), and the target's four
notebooks plus whatever in `config`/`steps` reads the new variables.

**Out of scope.** Launching anything. Every launch stays behind the operator's
own gate, one job at a time, with their confirmation between. Regenerating a job
folder is preparation and is not a launch.

## Constraints

- TDD is ON: `openspec/config.yaml` carries `strict_tdd: true`. Red before
  green, observed, never asserted.
- Two suites, and one alone is not the suite: `npm run test:all`
  (`node --test tests/**/*.mjs` and `.micromamba/envs/papersmith/bin/pytest`).
- The target's own suite runs under its own interpreter:
  `implementations/Domain_Adaptation/.venv/bin/python -m pytest`.
- No branches — commit on `main`, per the operator's standing instruction.
- A notebook is executed with the target venv's `bin` first on `PATH`; the
  kernelspec resolves a bare `python` off `PATH` at kernel start.
- `run-config.json` carries `runnerTemplate` digests; changing a runner asset
  obliges regenerating every job folder that pins the old digest.

## Ordering (forced by the code, not chosen)

`campaign()` refuses without a full-scale `ceilings.json`
(`harness.py:2018`, and `:2081` below scale), and that record is what the search
produces. So the first job that can be sent is `ceiling-search`; the campaign is
second, and its job folder must then be regenerated at the pin that carries the
record. `campaign` and `attention-mechanisms` also declare clone paths that
exclude `MIL-CREDA/Results/Benchmark`, so the record would not reach the worker
even once it exists.

## Delivery

Strategy: `ask-on-risk`. Forecast: pending the seam map.

## What the seam turned out to be

Measured, and it shrank the change. The per-submission channel already exists
end to end and is not a new thing to build:

- `submit --unit A --unit B ...` hands the list to the packer, which splits it
  across workers (`remote_cli.py:1247`); each worker's own slice is written as
  `job.run_config["units"]` (`:1261`), and `--smoke` sets
  `run_config["mode"] = "smoke"` the same way.
- The Kaggle adapter parses the job folder's `run-config.json` and **shallow
  updates it at the top level** with that mapping (`adapters/kaggle.py:962-965`),
  then injects a cell that writes the merged JSON to the worker's own
  `run-config.json`. So the units and the mode already arrive at the worker,
  per submission, today.
- `select_block()` reads `mode` to pick the smoke block
  (`runner_invoke.py:196`).

The gap is one hop, at the end: **nothing hands any of it to the kernel.**
`kernel_environment()` (`runner_invoke.py:120`) is the single composition point
for what an executed notebook receives, and it returns exactly three names —
`PYTHONPATH`, `FORGE_CLONE_ROOT`, `FORGE_CLONE_COMMIT`. Neither the mode nor the
units are among them, and `rg` finds zero reads of `units` in either runner
asset, in any adapter, or anywhere in the target.

**Consequence, and it is live.** A campaign submitted with `--unit` today
distributes the *ledger* and not the *work*: every worker receives its own slice
in a file nobody opens and runs the identical whole job. Nine accounts would
compute the same grid nine times.

So no new CLI flag, no new `run-config.json` key, no `jobfolder.py` change and
no schema change: `validate_run_config()` has no allowlist to widen
(`jobfolder.py:1489`). The forge publishes what the run-config already says, and
the target reads it.

The target's own vocabulary receives it cleanly: `shard` is a name (the
directory under `Results/Benchmark/shards/<shard>`, `harness.py:993`) and
`seeds` is the list that shard owns. A unit is a seed.

## Tasks

- [x] T1 — Map the per-submission environment seam (delegated, read-only).
- [x] T2 — `kernel_environment()` also composes the mode and this submission's
      units, threaded from `invoke()` through `execute_notebook()`, and the
      callable branch receives them too — `run.kwargs` is per-job, so the
      environment is the only per-submission channel either shape has.
      (red first, `tests/test_remote_execution.py`)
- [x] T3 — The four notebooks read them: full scale unless the mode is smoke,
      and the units as this shard's seeds, with the file dial as the default so
      the local pilot walk is byte-identical. (red first, target suite)
- [ ] T4 — Regenerate the `ceiling-search` job: notebook shape, clone paths
      covering `MIL-CREDA/Notebooks`, pinned to a pushed commit. Every job
      folder must be regenerated once the runner bytes change — the
      `runnerTemplate` sha256 is inert provenance and nothing re-verifies it
      (`jobfolder.py:1564`), so a stale folder keeps running the old cells in
      silence.
- [ ] T5 — Rehearse `ceiling-search` on the worker that will run it, and record
      the verdict from the returned artifact.
- [ ] T6 — Publish the launch: `propose` the campaign, `offer` to mint the
      token, and stop. The `gate` is the operator's.

Everything past T5 waits on the operator, one job at a time.

## Verification of record

- T2, commit `d747e23`. Red observed before green and proven reachable both
  ways: with the implementation reverted and the tests kept, 7 of the 8 new
  tests fail; the eighth asserts that NOTHING is exported when neither fact is
  declared, which cannot fail that way, so it was proven with the opposite
  mutation (export both names unconditionally) and fires.
- `pytest tests/test_remote_execution.py -q`: 772 passed, 76 subtests passed.
- `pytest -q` (full): 3 failed, 5157 passed, 3 skipped, 2688 subtests passed.
- `npm run test:node`: 653 pass, 0 fail.

**The three failures were pre-existing and are now resolved upstream.** They
were measured as pre-existing by stashing both changed files and re-running
them against the then-clean HEAD, where all three still failed (2026-09-26).
While this task was in flight the repository moved: releases `v0.3.0`, `v0.3.1`
and `v0.4.0` landed from another session, and at `v0.4.0` all three pass.

- The release guard — `shipped changes since the last release moved the
  version` — passes because the version did move. It names `skills` among its
  shipped roots, so this task's own commit is covered by it: any further change
  to a shipped file obliges the version to move again.
- `test_initialize_creates_the_workspace_contract` passes.
- The paper-writing D2 tripwire was **discharged, not re-pinned** (`64cc94a`).
  It now counts recorded `write` runs and judges once there are ten, and
  `paper_evidence.py:329` appends one line per real run, so the count grows on
  its own and the skip clears itself. That is the falsifier as `SKILL.md`
  states it, not a literal moved out of the way.

**Two unauthorized edits by the delegated writer were reverted** while this task
ran, both outside its named scope: `tests/test_paper_writing.py` and
`tests/test_papersmith_init.py`. The revert was right on the evidence available
then — an agent silencing a test it was not asked to touch — and the same two
concerns were afterwards addressed properly, in their own commits, by the
operator.

**Still in sync.** `d747e23` is an ancestor of `HEAD`; nothing touched
`runner_invoke.py` after it; the two new commits in this area — `9eb2010`
(the pin-published refusal now measures the push) and `aade456` — do not
conflict, and the first one helps T4 directly.

## Progress

T1, T2 and T3 are done. T4 is the next unit of work and has one precondition
this document did not previously name: **the pin must be a pushed commit.**

- 2026-09-28 — T3 measured green, having been left unticked while its code was
  already in. `config.execution_is_pilot_scale()` / `config.execution_seed_units()`
  are read by all four remote notebooks (`Benchmark_Ceiling_Search`,
  `Benchmark_Noise_Sweep`, `Benchmark_Campaign`,
  `Benchmark_Attention_Mechanisms`), and `tests/test_scale_readings.py` is 39
  passed. Ticked against that measurement and not against the intent written
  above it.
- 2026-09-28 — the declared pilot walk is COMPLETE, all six steps, each with its
  own commit (`faecebf` … `c11615c`). `probe` reports
  `pilotCompleteness.status = "complete"` and `position.status = "complete"`,
  six of six satisfied, no disagreements. `walk` reports every step `walked` at
  `rung: "none"` — pilot scale grants no rung, which is the point.
- 2026-09-28 — **T4's blocker, measured.** All four job folders are pinned at
  `bee6781` and `probe` reports every one of them `drift`, over six changed
  `src/MIL_CREDA_Benchmark` files; `smokeReady` is `false` for all four. Seven
  target commits are ahead of `origin/main`, so the pin a regenerated folder
  would want does not exist on the remote yet and `pin-published` refuses an
  unpushed commit. Pushing is therefore a precondition of T4, not a closing
  step.
- 2026-09-28 — **the rehearsal shapes, measured, and the target already decided
  them.** `steps.predecesores` returns empty for `campaign` and `mechanisms`, so
  `steps.ensayo_remoto` routes both to `harness.run_smoke()` — the wire — and
  not to their own notebook. That is written down and reasoned, not an
  oversight: no declared step produces the full `Results/Benchmark/ceilings.json`
  (the full search is `__records__`' single entry, run outside the step walk),
  so declaring it would chain nothing `predecesores` could derive, and a
  notebook rehearsal of the campaign would be a nine-and-a-half-hour rehearsal.
  Only `noise-sweep` has a predecessor (`search-pilot`), honours the input scale
  (`honra_la_escala_de_entrada` is `True`), and therefore rehearses its own
  notebook against the search's full-scale record — refusing today, correctly,
  because `Results/Benchmark/ceilings.json` does not exist yet.
- 2026-09-27 — a step's ledger `started` with no terminal partner does not mean
  it was killed; the `results` step's first process was still running when this
  session read it as dead, finished 1m43s later, and committed its own product
  as `e1d4742`. The duplicate run committed `c11615c`. Both are in history and
  the tree holds the second one's bytes.

- 2026-09-26 — all nine stored Kaggle accounts authenticate
  (`accounts_cli.py validate`: nine `ok`). Measured, so the distribution
  has nine workers to place shards on rather than an assumed number.
  Authentication is not capacity: how many concurrent sessions each one
  grants is discovered live by the packer at distribute time, and the
  weekly GPU allowance is the service's, not ours to read.

