# Agreed for the benchmark report — bound to research-concept-r16.md

Every item here was settled in deliberation, not derived from a plan of work. An
item that never reaches the code shows up as an unticked box rather than as a
silence. Nothing here is re-decided while implementing: a collision with any of
these is raised, not resolved in passing.

## Dispersion and scale

- [x] The `±` in every phase-1 table is the dispersion **across seeds**, not the
      batch-wise dispersion the reference paper reports. The seed dispersion is
      what the verdict rule consumes; a batch-wise `±` would stay plausible while
      the run grants no verdicts.
- [x] `LAMBDA_CONST` stays at **1.0**, declared and not calibrated. The realized
      share of the objective is reported instead, per arm, so a term scaled to
      irrelevance is visible rather than inferred.
- [x] Accuracy granularity is **1/36 = 2.78 points** with 36 evaluation bags, so
      tables print one decimal and say so. Two decimals would be false precision.

## Nomenclature — asterisks count what is absent

| id | display name | what it lacks |
| --- | --- | --- |
| `A` | `Baseline` | everything |
| `C` | `CREDA*` | the confidence weighting |
| `D` | `CREDA` | nothing |
| `B` | `MIL-Baseline` | everything |
| `E` | `MIL-CREDA**` | the local term and the weighting |
| `F` | `MIL-CREDA*` | the local term |
| `G` | `MIL-CREDA` | nothing |
| `SU` | `MIL-CREDA-U` | 20 of its 30 instances, regular selection |
| `SA` | `MIL-CREDA-A` | 20 of its 30 instances, arbitrary fixed selection |
| `SK` | `MIL-CREDA-K` | 20 of its 30 instances, top-K by learned attention |

- [x] `SU`, `SA`, `SK` **select 10 instances** and then apply the same learned
      attention over the selected ones, so the trio differs in one thing: the
      selection rule.
- [x] `SA`'s draw uses a **dedicated generator**. Consuming the training
      generator would shift every later draw and the rung would credit the
      selection with what the offset did.

## Ladder

- [x] The three attention rungs are replaced by `SU->SK`, `SA->SK` and `SK->G`.
      The first two hold the budget at 10 instances; the third reads what
      dropping from 30 to the top 10 costs.

## Tables (phase 1) — Table 6 of the reference, our names

- [x] Rows are arms by display name, columns are the six transfers, plus `Avg`.
- [x] **Two tables**: target accuracy as the headline, source accuracy as the
      complement. A method that wins on target by wrecking source is the
      degenerate case, and one table cannot distinguish it from a success.
- [x] A `max` column beside `mean ± stdev`, so the peak is documented where a
      peak is a number rather than a picture.
- [x] A `share` column: the realized contribution of the adaptation term.
- [x] The ladder table stays, directly below. Levels say who is ahead; only the
      rungs say which piece did the work.
- [x] Below the declared repetition floor, no verdicts are granted, the reason is
      stamped in the header, and the table is printed anyway.
- [x] **The wall-time table comes first**, in the same shape, times instead of
      accuracy. It replaces the per-run listing, which repeated in prose what the
      tables say in a grid.
- [x] **Reading order**: time, source, source rungs, target, target rungs, and the
      transversal reading last.
- [x] **One table per cell**, each preceded by three short lines — what is
      measured, why, and which direction is better — and followed by the most
      relevant conclusion.
- [x] That conclusion is **computed from the table**, never written by hand: a
      hand-written conclusion is a second source of truth and goes stale in
      silence.
- [x] A rung is named `Baseline → MIL-Baseline`, never `A->B`. An identifier is
      not a name.
- [x] Notebook prose and table headings in Spanish; identifiers, JSON keys and
      method names stay English, because they are a data contract rather than
      prose.
- [x] Progress during the run prints one line per transfer, not one per run. It is
      the only sign of life in a run measured in hours, so it is kept — but it is
      progress, not a report.

## Figures — phase 2

- [x] **One display seed for every comparative grid**, chosen as the seed whose
      across-arm mean target accuracy is the median. Panels drawn from each arm's
      own median seed would differ in the method *and* in the draw, and — because
      the seed fixes the partition — would not even share their bags.
- [x] Artefacts are **median, never best**. The best of N grows with the arm's
      dispersion, so best-vs-best flatters the noisiest arm by the most.
- [x] **Three transfers in every figure, not six.** Six rows at a legible panel
      size do not fit on a page and the tables already carry all six. Fixed in
      `FIGURE_TRANSFERS` by a rule that looks at no outcome: each domain appears
      once as source and once as target.
- [x] **Which three is computed, not fixed**: the transfers where the methods
      reach the highest mean target accuracy. It is a choice made by the outcome,
      so it is declared in every caption. What makes it defensible here is that
      the latent space of a transfer where everything sits near chance is a
      picture of a model that did not learn. It never touches *which draw* is
      shown — that stays the display seed.
- [x] Latent grid is the **shared original space** and then one column per method:
      both floors, both CREDA, all three MIL-CREDA. `Original` is the images
      themselves before any model — shared because preprocessing already brings
      both domains to one tensor shape — sampled representatively and stratified
      by class.
- [x] **Both floors stay, because the measurement says they are not redundant.**
      Drawn at the instance level on the pilot, `Baseline` and `MIL-Baseline`
      differ by up to 0.38 in distance ratio and 0.07 in domain separability.
      They train the same encoder through different objectives — per instance
      against per bag through the attention pooling — so their instance
      embeddings had no reason to agree. `latent.floors_agree` re-runs that check
      every campaign, and only it may retire one of the columns.
- [x] **Every panel of the grid is drawn at the instance level**, bag-unit arms
      included. Every arm encodes instances, so it is a space they all have and
      the only one where every panel carries the same number of points. One point
      per subject beside one point per instance made the instance-unit columns
      look like they covered the space and the bag-unit ones look sparse — the
      statistical unit drawn, not the alignment. The bag-level view stays in the
      phase-two tables, which measure each arm in its own unit.
- [x] Colour is the **class**; the marker is the domain — source circles, target
      triangles. Target markers larger with a dark edge, source smaller and
      semi-transparent, categorical 10-class palette.
- [x] Bag figure is **3 columns × 3 rows in one figure**, not one file per
      transfer: `MIL-Baseline` + `MIL-CREDA*` + `MIL-CREDA`, chosen by the
      mechanism rather than by the ranking — that is the rung the local
      correspondence lives on, so the figure can fail.
- [x] Bag figure highlights **the same bags in every panel**: the median bag of
      each class by correspondence mass, one colour each, everything else grey.
- [x] The nearest source bag is found with the **bag kernel in the representation
      space**, never Euclidean and never in the 2-D projection, and the pair is
      joined by a line so the correspondence survives UMAP's distortion.
- [x] The measured correspondence hit rate is printed with the figure, so it
      carries a number and not an impression.
- [x] **Every figure carries the same three lines as a table**: what is looked at,
      what is being sought, and the conclusion — and the conclusion is **computed
      from the results**. A sentence written by hand under a figure fixes itself:
      the figure is regenerated from other data and the sentence stays.
- [x] The bag figure keeps the projection rather than becoming a bipartite
      diagram. Three columns instead of six separate files already makes the
      panels large enough to read; if it is still a knot, the bipartite is next.

## Removed on purpose

- [x] The adaptation-term scale table (`min`/`max`/`width`). What it checked —
      that MIL-CREDA's terms stay in `[0, 1]` while the prior work's is bounded
      only by `ln n` — is guaranteed by the construction in Section 5 and already
      covered by the invariant suite. A table restating a proved bound is a third
      place for it to go stale.

## Figures — phase 1

- [x] Loss curves for `Baseline`, `CREDA`, `MIL-Baseline`, `MIL-CREDA`, as the
      median curve with an interquartile band across seeds.
- [x] A **contribution panel** beside them. The pilot already shows the realized
      share ranging from 0.03 to 0.99 across arms at a fixed coefficient, which
      is the thing the curves exist to make visible.

## Checkpoints

- [x] Three per arm per cell, for every arm, so the top 3 can be selected after
      the run instead of being guessed before it. Local, gitignored.
