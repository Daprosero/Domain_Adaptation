# Oracle screening of the phase-2 search spaces (registered before any result)

Written 2026-11-15, before any screening trial ran. Code: `src/MIL_CREDA_EEG/oracle_screen.py`.

## Question
Which phase-2 settings of each method matter, and where their good values lie, on synthetic data
where the target labels are known; and whether that answer stays the same when the data change.
Only what is irrelevant, or whose good region is the same, in every development cell is fixed or
narrowed. Anything whose importance or optimum moves with the data stays searched.

## Data
- The 12 synthetic development units: d in {1.5, 2.5} x shift in {0.5, 1.5, 3} x rotation in {0, 1}
  (6 cells). No other synthetic unit is touched, so the 86 held-out units stay clean.
- Seeds 1, 2, 3. Phase 1 is not refitted: each (unit, seed) loads its precomputed base
  (`MIL-CREDA-EEG/Manifests/oracle_bases/`, `xcompare.shared_base` at the unit's recorded phase-1
  settings), so all methods start phase 2 from the same file.
- Objective: balanced accuracy at the seed's cut on an evaluation pool of 120 new subjects of the
  unit's target domain (same device, shift and class effect, another seed, no source pair). It reads
  true target labels and decides nothing reported in the paper.

## Search
Every method (MIL-CREDA and the seven comparators): one TPE study per (unit, seed), 80 trials,
sampler seed 20261112 + rotation + 1000 x seed, over its full phase-2 space with epochs2 in
{10, 20, 40}. MIL-CREDA also searches wd2 [1e-7, 1e-2], gamma [0, 1] and tau_att [1, 10]; lr2 takes
the comparators' range [1e-5, 1e-2]. Nothing is fixed in advance.

## Decision rules (applied per method, reported whatever they give)
1. Importance: fANOVA (seed 0) per study over its 80 trials; per cell, the median over its 6
   studies (2 rotations x 3 seeds). Stability of the ranking: Kendall W over the 36 studies.
2. Irrelevant: median cell importance < 0.05 in at least 5 of the 6 cells. An irrelevant setting is
   fixed at the median of the top-decile trials over all studies (the mode for a categorical).
3. Good region: per study the top-decile trials; per cell the 10-90% quantiles of a setting's
   position in its range (log scale when log). Stable when the per-cell medians span < 0.2 of the
   range. A stable relevant setting is narrowed to the union of the per-cell 10-90% intervals.
   Otherwise its full range is kept.
4. epochs2: if 40 is the choice of more than half the top-decile trials in at least 4 of 6 cells,
   a second pass searches epochs2 in {40, 70, 100} with the rest of the space as reduced by 2-3.
5. Trial budget: per study, the first trial whose best-so-far BA is within 0.01 of its final best;
   the budget for the reduced space is the 90th percentile over the 36 studies, rounded up to 5
   (conservative, since the reduced space is smaller than the full one).
6. gamma: if its good region sits at 0, that is evidence against the attention consensus term and is
   reported as such.
7. "The space cannot be reduced" is an admissible outcome.

## Known limit
The nominal ADHD effect band (subject TBR d 0.3-1.0) is not covered: in those synthetic cells a
single TBR threshold reaches BA 0.60-0.71 on the source subjects, but the learned phase 1 stays at
chance (0.50-0.52), so there is no adapted model to screen. The closest cell by attainable
performance is d 1.5 / shift 1.5 (source-only BA 0.68; ADHD 0.63).
