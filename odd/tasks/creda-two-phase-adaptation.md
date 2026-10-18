# CREDA two-phase sample-level adaptation

## Objective

Fix the measured adaptation arrangement in `src/CREDA` as a module of its own,
so the per-sample gradient trace runs against committed code rather than against
a throwaway probe script.

## Problem

`src/CREDA/sample_adaptation.py` trains the supervised and adaptation terms
**together in one phase** and refuses `sync_steps < 1`. The arrangement every
measurement of the last session used is a different one and exists nowhere in
`src`:

1. **Phase one** trains the source encoder and the head on source data only, and
   freezes both.
2. **Phase two** copies the source encoder's weights to a target encoder **once**,
   and lets the adaptation term alone move it. Source through the source encoder,
   target through the target encoder.

## Why

The free regime is measured, not assumed. Without BatchNorm, re-copying the
weights every 7 steps keeps **6%** of the gain: rollo `0.5943` free against
`0.2754` leashed over a `0.2572` floor; toroides `0.2906` against `0.1847` over
`0.1798`. With BatchNorm the leashed arm is indistinguishable from pure
statistics (`0.8990` vs `0.8932`), so the term contributes nothing there.

## Scope

- `src/CREDA/two_phase.py` — the arrangement, with the free regime as its default
  and the leash available for comparison.
- `tests/test_creda_two_phase.py` — one test per declared invariant, red proven
  reachable by mutation.

Out of scope: any change to `sample_adaptation.py`, to `models.py`, or to the
managed revision. **This arrangement is not a claim from `research-concept-r25.md`
and its provenance says so.** The operator authorized placing it precisely
because it is not bound to a revision.

## Tasks

- [ ] T1 — `src/CREDA/two_phase.py` with six declared invariants.
- [ ] T2 — `tests/test_creda_two_phase.py`, one test per invariant, each proven
      red-reachable by a reverted mutation.
- [ ] T3 — the per-sample gradient trace, run through this module, aimed at what
      BatchNorm supplies that the term does not.

## Checks

- `python -m pytest tests/test_creda_two_phase.py -q`
- The whole suite stays green: `python -m pytest tests/ -q`
