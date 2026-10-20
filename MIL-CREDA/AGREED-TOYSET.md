# Agreed for the toyset — bound to research-concept-r34.md

Every item here was settled with the operator, not derived from a plan of work.
An item that never reaches the code shows up as an unticked box rather than as a
silence. Nothing here is re-decided while implementing: a collision with any of
these is raised, not resolved in passing.

**This is not `MIL-CREDA/AGREED.md`.** That document is bound to r21 and governs
the DIGIT benchmark report, which is being retired. Its content stays in git at
`7b9f4a0` and is not repaired. This file governs the toyset and nothing else.

## Scope — two notebooks, and the rest is gone

- [ ] The toyset is `Toyset_Generator.ipynb` and `Toyset_Plot.ipynb`, and no other
      notebook exists under `MIL-CREDA/Notebooks/`. The generator trains and writes
      the artifact; the plot reads it and draws. **No number is recomputed in the
      plot notebook.**
- [ ] The code the retired notebooks reach and the toyset does not goes with them.
      The surface is measured before anything is deleted, never estimated.

## The unit of time is the EPOCH

- [x] Everything a reader sees is in epochs: the axes, the table columns, the row
      labels and the frame titles. Settled 2026-10-19 and applied 2026-10-20 over
      thirteen sites in both packages.
- [x] The step survives only as the index of the history array, and never reaches
      a reader. Where a guard must compare steps and a label must print epochs,
      the two stay separate rather than being collapsed.
- [x] `fronteras_w` says **`barrido`**, not step and not epoch: its points are a
      CONSTRUCTED walk over the weight frontier of Eq. (24), not training.
- [x] A phase-1 epoch is 5 steps and a phase-2 epoch is 16, each one bag per class.
      The two phases count epochs differently and the figures say which they mean.

## Phase 1

- [x] Trains the source encoder and the head. Nothing of the target.
- [x] Its whole objective is Eq. (21), normalized by `B_src` so every bag lands in
      `[0, 1)` and the three terms of Eq. (39) are readable on one scale.
- [x] The criterion is **source VALID accuracy** and never the role the table
      publishes. **The patience stays**: the signal is supervised, so stopping on
      it risks nothing.
- [x] Eq. (15)'s `||v_R||_1 <= 1` is re-imposed AFTER each optimizer step, never
      before: the rule counts optimizer steps.
- [x] The search runs 25 trials over six dimensions, and each trial's score is the
      MEAN over 3 normalizations x 3 search seeds. Ties within `TIE_WIDTH = 0.01`
      are broken by the valid loss, which still discriminates once the accuracy is
      against its ceiling.
- [x] Search seeds (100, 101, 102) and report seeds (0..9) are DISJOINT. The
      hyperparameters never see the material their number is measured on.
- [x] The budget is reported rather than defended: 25 trials over 6 dimensions is
      4.17 per dimension, and a gap smaller than the variance between two searches
      of this budget is not a reading of the method.

## The bridge — four acts, in this order

- [x] The WEIGHTS of the best epoch are restored, not its number, and a second
      reading proves the load landed.
- [x] The frozen bandwidth is measured here, over the source training role ENTIRE,
      and a truncated reading is refused rather than tolerated.
- [x] Everything but the target encoder is frozen. **Every guarantee phase 2 offers
      rests on this act.**
- [x] `theta_s -> theta_t` is copied exactly once, WITH the buffers, or a BatchNorm
      `theta_t` would not be bit-identical to the encoder it claims to start as.

## Phase 2

- [x] Moves `theta_t` alone.
- [x] Minimizes `lambda_glob * L_glob (Eqs. 32-37) + lambda_loc * L_loc (Eqs. 30-31, 38)`
      and NOTHING else. Both coefficients are exactly 1, fixed, with no ramp.
- [x] Eq. (21) is computed, recorded as `supervised` and drawn, and **summed into
      nothing**. This is the objective, not a flag: there is no pole where it comes
      back. Measured parameter by parameter: the optimizer's step is bit-identical
      with and without it.
- [x] The confidence weighting is ON. `sopesado` OFF in Eq. (39) means the two
      COEFFICIENTS at 1, which is a different thing and was confused once.
- [x] The criterion is `max(cef)`, which reads no target label. **No early stop
      since 2026-10-19**: it runs its ceiling whole and chooses looking back.
- [x] The balance of `K_mix` is by TRACE and the bounds are the entropy-cosh pair,
      because that is what r34 states -- not because CREDA does it. Eq. (36)
      divides by the literal `ln 2` that r30 proved.
- [x] Eq. (37) aggregates over EVERY class with `n_c^s / N_s`, weights the source
      LABELS observed, and charges `1` to a class with no active pair. Excluding it
      rewarded collapse.
- [x] Eq. (38) is the uniform mean over the target bags, `(1/n_t) sum_j d_j^2 / 2`.
      The confidence left this aggregation in r32: `w_j^t` is differentiable, so a
      weighted mean could be lowered on the worst-aligned subjects instead of
      aligning them.

## The bandwidth

- [x] ONE bandwidth serves two jobs -- within-bag (Eqs. 14-15) and between-bags
      (Eqs. 32-37) -- and no value is in band for both. That is Decision 1's
      consequence: the range is widened to span both and the SEARCH chooses the
      trade-off. Which of the two is larger is not fixed across families.
- [x] The frozen bandwidth is source-only, so nothing injected into the target can
      move it. Damage entering through the class block stops being confused with
      damage entering through the unit of measurement.
- [ ] **Phase 1's bandwidth is measured on an UNTRAINED encoder and held fixed for
      the whole phase.** Measured over the 90 cells of the full run, the space then
      moves by a median of 2.46x, up to 31.21x, and in 27 of 90 cells it CONTRACTS.
      The operator proposes letting it move during phase 1 and ending with the
      frozen one. r34 declares `sigma in R+` and says nothing about its estimation
      or constancy -- it says "fijo durante el entrenamiento" of `gamma` and not of
      `sigma` -- so this contradicts nothing. **Open:** the cadence, and the
      consequence below.

## The sections of the toyset

**Only section 1 searches.** From section 2 on, every section reads the
hyperparameters section 1 chose and runs its own tests. Nothing re-searches, so
no later section can win by being tuned harder than the one before it.

**Every section from 2 on runs on the BEST ARCHITECTURE only**, read from the
record before it runs and never picked after seeing a result. Measured on the
full run: `lunas` none (0.4056), `toroide` BatchNorm (0.7944), `rollo` BatchNorm
(0.3944) -- it is not the same in the three families, so either one is fixed and
said out loud, or one runs per family.

- [x] **Section 1 -- the method against its rungs.** Three families, three
      architectures, ten seeds. One figure, one table and the declared tests per
      family block, which is CREDA's shape so the two documents can be read side
      by side. The only section that searches.

- [ ] **Section 2 -- three bags moved at the frontier.** The three target bags
      closest to the decision frontier are DISPLACED, and the section shows how
      `w` moves. This is not noise and nobody asked for noise: it is a directed
      geometric displacement, the bag-unit counterpart of CREDA's frontier. The
      bag is the right unit because `w_j` is defined per bag (Eq. 24); perturbing
      single points dilutes the damage across a weight that is not defined at
      that scale. Table structure mirrors CREDA's.

- [ ] **Section 3 -- is the aggregation the best one?** Two axes:
      - mechanism: `ours`, `abmil-published`, `abmil-gated`, `max`, `mean`.
        `wiring.MECHANISMS` already implements all five.
      - selection: all instances, `top-k`, `arbitrary-k`, `regular-k`.
      **`top-k` against `arbitrary-k` at the SAME `k` is the clean test**: if the
      attention knows which instances matter, keeping its best `k` has to beat
      keeping `k` at random. If they tie, the attention is scoring nothing even
      with the mechanism switched on, and `regular-k` is the third foot that
      separates "top-k wins" from "fewer instances already helps".
      **`k` is declared BEFORE the run: 5, 10 and 15 of 30, and all three are
      reported, never the best.** A `k` chosen after seeing the result chooses
      the number that gives the answer we wanted.
      Not the full cross: the five mechanisms over all instances, then the four
      selections over the winning mechanism.
      It runs the FULL two phases, not source only. The aggregation enters both
      -- phase 1 scores bags with it and phase 2 builds its representations from
      it -- so source-only would answer "which classifies source bags best" while
      the claim is about transfer. The two can disagree, and the interesting one
      is the second.

- [ ] **Section 4 -- the correspondence, moved and followed.** A target bag's
      correspondence mass `pi_ji` is read, the bag is DISPLACED toward another
      class's region, and the mass is read again. If the method does what it
      claims, **the mass moves with it**: toward source bags carrying the
      characteristics the bag now has. If it does not move, the correspondence is
      anchored in something that is not the bag's characteristics, and that reads
      without interpreting anybody.
      A static score against input-space proximity answers "did it hit"; this
      answers "does it follow", which is causal and stronger. It needs no new
      mathematics: `total_correspondence` already returns the mass per source bag
      and this reads it twice.
      **Sections 2 and 4 share the move-a-bag machinery**, built once in 2.

- [ ] **Sections 3 and 4 run CONSECUTIVELY, in that order, and that is not
      cosmetic.** They are MIL's claim split in two halves: "we know which
      INSTANCES matter inside the bag" and "we know which BAGS correspond across
      domains". Neither half holds alone -- a good attention over arbitrary
      correspondences aligns noise carefully.

- [ ] **Section 5 -- the ablation**: global only, local only, both, plus
      `.predict` as the floor. It goes LAST because it opens the fewest new
      questions and because read after 3 and 4 it interprets better: if the local
      term buys nothing, section 4 has already said whether that is because the
      correspondence fails or in spite of it working.
      `.predict` costs no second sweep -- `run_cell` measures it itself.
      **Local-only is the one that does not exist**: `spec["local"]` already gates
      the local term, and the symmetric gate on the global one has to be declared.

## Open -- nobody has decided these

- [ ] `MIN_EPOCAS_2 = 5` (CREDA) against `MIN_EPOCHS_2 = 3` (MIL): both declared,
      deliberately not equalized, still open.
- [ ] A moving `sigma` during phase 1 moves the scale of Eq. (15)'s consensus, and
      `gamma` -- which r34 declares FIXED during training -- would stop meaning the
      same thing across steps. **This one goes to the deliberator**: not "is it
      allowed", but "does fixed `gamma` with moving `sigma` still say what Eq. (15)
      says".
- [ ] The two-arm contamination design cannot tell "the weighting does not help"
      from "there was nothing to gain here", because there is no upper reference.
      Making the oracle arm survive changes what it computes, which is a claim.

## Ladder

- [x] `.predict` -- the same run read at its step-0 checkpoint. No adaptation.
- [x] `AdaBN` -- that checkpoint with the target's normalization statistics
      re-estimated. No optimizer, no label.
- [x] `MIL-CREDA` -- the full method, global and local, weighting on.

The first two cost no second sweep: `run_cell` measures them itself. A rung that
reads `y_t` is a CEILING and never a competing method, and a table that prints
one says so.
